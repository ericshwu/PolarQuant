"""You can use this code 2 rerun the LongBench result"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
import numpy as np
import random
from tqdm import tqdm
from utils.metrics import qa_f1_score, code_sim_score
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from models.modeling_llama_polar import generate


# config 4 different pretrained models
max_length = 120000
pretrained_model = '/XXX/public/llama-3.1-8b-chat/'

# max_length = 3500
# pretrained_model = '/XXX/public/llama2-7b-chat/'

# max_length = 31500
# pretrained_model = "/XXX/public/mistralai/Mistral-7B-Instruct-v0.2"

# max_length = 120000
# pretrained_model = '/XXX/public/Qwen/Qwen2.5-1.5B-Instruct/'


config = AutoConfig.from_pretrained(pretrained_model)
config._attn_implementation = "flash_attention_2"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
generation_config = GenerationConfig.from_pretrained(pretrained_model)

distributed_state = PartialState()

# for bf16
if distributed_state.is_main_process:
    print("test 4 bf16 baseline")
model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16, config=config).to(distributed_state.device)

# # for polarquant
# if distributed_state.is_main_process:
#     print("test 4 polarquant")
# from models.modeling_llama_polar import LlamaPolarForCausalLM  
# model = LlamaPolarForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)  

# # for int
# if distributed_state.is_main_process:
#     print("test 4 int")
# from models.modeling_llama_int import LlamaIntForCausalLM
# model = LlamaIntForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)

# # for kivi
# if distributed_state.is_main_process:
#     print("test 4 kivi")
# from models.modeling_llama_kivi import LlamaKIVIForCausalLM
# model = LlamaKIVIForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)

# # for zipcache
# if distributed_state.is_main_process:
#     print("test 4 zipcache")
# from models.modeling_llama_zip import LlamaZipForCausalLM
# model = LlamaZipForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)

# # for qjl
# # qjl is not compatible with accelerate, we provide its inference code 4 LongBench in models


model.eval()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    

if __name__ == '__main__':
    seed_everything(42)

    # Your LongBench data path here
    dataset2prompt = json.load(open("/XXX/public/data/longbench/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/XXX/public/data/longbench/dataset2maxlen.json", "r"))

    datasets = [
        "narrativeqa", 
        "qasper", "multifieldqa_en",
        "2wikimqa", "hotpotqa", "musique",
        "lcc", "repobench-p",
    ]

    for dataset in datasets:
        template = dataset2prompt[dataset]
        test_file = f'/XXX/public/data/longbench/{dataset}.jsonl'
        output_file = f'/XXX/output/longbench/{dataset}.jsonl'

        idx, examples = 0, []
        with open(test_file, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                prompt = template.format(**json_obj)
                # prompt 4 different pretrained models
                # for llama-3.1-8b-chat
                if dataset not in ["lcc", "repobench-p"]:
                    prompt = f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                # # llama-2-7b-chat 
                # if dataset not in ["lcc", "repobench-p"]:
                #     prompt = f"[INST]{prompt}[/INST]"
                # mistral-7b-instruct-v0.2 we don't apply chat template
                # # for qwen-2.5-1.5b-chat
                # if dataset not in ["lcc", "repobench-p"]:
                #     prompt= f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

                examples.append((idx, tokenizer.encode(prompt), json_obj["answers"], json_obj["all_classes"]))
                idx += 1
        
        distributed_state.wait_for_everyone()
        if distributed_state.is_main_process:
            print('====== Tokenize End! ======')

        completions_per_process = []

        with distributed_state.split_between_processes(examples, apply_padding=True) as example_per_process:
            for idx, input_ids, answers, all_classes in tqdm(example_per_process, disable=not distributed_state.is_main_process):
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length // 2] + input_ids[-(max_length // 2):]

                input_ids = torch.tensor([input_ids], device=distributed_state.device)
                output_ids = generate(model, input_ids=input_ids, attention_mask=None, max_new_tokens=dataset2maxlen[dataset], generation_config=generation_config) 
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                completions_per_process.append((idx, output_text, answers, all_classes))
        
        completions_gather = gather_object(completions_per_process)

        distributed_state.wait_for_everyone()

        if distributed_state.is_main_process:
            if dataset in ["narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique"]:
                eval_function = qa_f1_score
            elif dataset in ["lcc", "repobench-p"]:
                eval_function = code_sim_score

            all_result, all_output = dict(), []
    
            for idx, output_text, answers, all_classes in completions_gather:
                score = 0.
                for answer in answers:
                    score = max(score, eval_function(output_text, answer, all_classes=all_classes))   
                all_result[idx] = score                 
                all_output.append((idx, output_text, answers, score))
            final_result = sum(all_result.values()) / len(all_result)
            print(f'Report {dataset} results:')
            print(f'There are {len(all_result)} examples in total, the em score is {final_result * 100:.2f}')
            with open(output_file, 'w') as fw:
                for idx, output_text, answers, score in all_output:
                    fw.write(json.dumps({"idx": idx, "output_text": output_text, "answers": answers, "score": score}) + "\n")


# CUDA_VISIBLE_DEVICES=0 python test4long.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=6666 test4long.py


