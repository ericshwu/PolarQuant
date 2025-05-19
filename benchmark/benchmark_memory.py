import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

from models.modeling_llama_exp import LlamaExpForCausalLM
from models.modeling_llama_kivi import LlamaKIVIForCausalLM
from models.modeling_llama_polar import LlamaPolarForCausalLM


@torch.no_grad()
def generate(model, input_ids, attention_mask, generation_config=None, max_new_tokens=8):
    current_input_ids = input_ids

    batch_size = input_ids.size(0)

    generate_ids = torch.zeros([batch_size, max_new_tokens], dtype=torch.long, device=model.device)
    eos_sequence = torch.zeros([batch_size, 1], dtype=torch.bool, device=model.device)
    stop_tensor = torch.tensor(generation_config.eos_token_id, dtype=torch.long, device=model.device)

    next_decoder_cache = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    outputs = model(
        input_ids=current_input_ids,
        past_key_values=next_decoder_cache,
        num_logits_to_keep=1,
    )

    # greedy sample- you can replace the sample function 2 realize more case
    output_ids = torch.argmax(outputs.logits, dim=-1)
    next_decoder_cache = outputs.past_key_values
    generate_ids[:, :1] = output_ids

    current_input_ids = output_ids
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones_like(output_ids)], dim=-1)
    
    step = 1
    while True:
        if step >= max_new_tokens:
            break
        
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            past_key_values=next_decoder_cache,
            num_logits_to_keep=1,
        )

        # greedy sample- you can replace the sample function 2 realize more case
        output_ids = torch.argmax(outputs.logits, dim=-1)
        next_decoder_cache = outputs.past_key_values
        generate_ids[:, step:step + 1] = output_ids

        current_input_ids = output_ids
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(output_ids)], dim=-1)

        eos_sequence = eos_sequence | torch.isin(current_input_ids, stop_tensor)
        if eos_sequence.sum().item() == batch_size:
            break    
        step += 1
    
    step = min(step + 1, max_new_tokens)
    generate_ids = generate_ids[:, :step]

    del next_decoder_cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
    
    return peak_memory


def run_hf_completion(model, dummy_prompts, output_length, generation_config):
    return generate(model, dummy_prompts, attention_mask=None, generation_config=generation_config, max_new_tokens=output_length)  
    

def main(model, batch_size, input_length, output_length, generation_config, num_iters_warmup: int = 10, num_iters: int = 30):
    dummy_prompts = np.random.randint(10000, size=(batch_size, input_length))
    dummy_prompts = torch.from_numpy(dummy_prompts).to(model.device)

    # first warm up
    print("warming up ...")
    for i in range(num_iters_warmup):
        torch.cuda.empty_cache()
        peak_memory = run_hf_completion(model, dummy_prompts, output_length, generation_config)
        print(f'peak memory: {peak_memory:.2f} GB')
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--input_length', type=int, default=256)
    parser.add_argument('--output_length', type=int, default=4096)
    args = parser.parse_args()

    batch_size = args.batch_size
    input_length = args.input_length
    output_length = args.output_length

    pretrained_model = '/XXX/public/llama-3.1-8b-chat/'
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token_id = 0

    config = AutoConfig.from_pretrained(pretrained_model)

    # # for standard transformer, for fair comparison, you should test enc time w. flash & test dec time w.o flash
    # # so we implement the LlamaExpForCausalLM
    # model = LlamaExpForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device='cuda')
    # model.eval()

    # model = LlamaKIVIForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device='cuda')
    # model.eval()

    # for PolarQuant
    model = LlamaPolarForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device='cuda')
    model.eval()

    # model = LlamaExpForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device='cuda')
    # model.eval()

    generation_config = GenerationConfig.from_pretrained(pretrained_model)
    generation_config.eos_token_id = [model.config.vocab_size + 10]

    main(model, batch_size=batch_size, input_length=input_length, output_length=output_length, generation_config=generation_config, num_iters_warmup=1, num_iters=10)


# CUDA_VISIBLE_DEVICES=0 python benchmark_memory.py --batch_size 4 --output_length 32768
