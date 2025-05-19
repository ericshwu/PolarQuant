import os
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


from modeling_llama_qjl import generate, LlamaQJLForCausalLM

from modeling_utils_qjl import QJLSketch

import torch.distributed as dist
import torch.multiprocessing as mp

from utils.metrics import qa_f1_score, rouge_score, code_sim_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# your working directory here
dataset2prompt = json.load(open("/XXX/public/data/longbench/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("/XXX/public/data/longbench/dataset2maxlen.json", "r"))


pretrained_model = '/ossfs/workspace/nas2/lijianan/trash/public/llama-3.1-8b-chat/'
max_length = 120000

# pretrained_model = '/ossfs/workspace/nas2/lijianan/trash/public/llama2-7b-chat/'
# max_length = 3500


output_dir = "/XXX/output/longbench"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_pred(model, rank, data, dataset):
    output_file = os.path.join(output_dir, f'{dataset}.jsonl')
    fout = open(output_file, 'a', encoding='utf-8')

    generation_config = model.generation_config
    
    if dataset in ["narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique"]:
        eval_function = qa_f1_score
    elif dataset in ["lcc", "repobench-p"]:
        eval_function = code_sim_score

    template = dataset2prompt[dataset]

    for json_obj in tqdm(data, disable=(rank != 0)):
        prompt = template.format(**json_obj)
        
        if dataset not in ["lcc", "repobench-p"]: 
            prompt = f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        
        # if dataset not in ["lcc", "repobench-p"]: 
        #     prompt = f'[INST]{prompt}[/INST]'

        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length // 2] + input_ids[-(max_length // 2):]

        input_ids = torch.tensor([input_ids], device=device)
        output_ids = generate(model, input_ids=input_ids, attention_mask=None, generation_config=generation_config, max_new_tokens=dataset2maxlen[dataset])
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        score = 0.
        for answer in json_obj["answers"]:
            score = max(score, eval_function(output_text, answer, all_classes=json_obj["all_classes"])) 
        
        item = {"_id": json_obj["_id"], "output_text": output_text, "answers": json_obj["answers"], "score": score}
        
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

    if rank == 0:
        print(f'{dataset} done!')


if __name__ == "__main__":
    seed_everything(42)

    rank, world_size = sys.argv[1:]
    rank, world_size = int(rank), int(world_size)

    config = AutoConfig.from_pretrained(pretrained_model)
    
    config.key_quantization_bits = 256
    config.key_quantization_bits_initial_layers = 512
    
    config.initial_layers_count = 15  # default as 
    config.outlier_count_general = 8
    config.outlier_count_initial_layers = 8

    config.group_size = 32
    config.buffer_size = 128

    generator = torch.Generator(device=device)

    config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator, device=device)
    config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128, rot=True, rng=generator, device=device)

    model = LlamaQJLForCausalLM.from_pretrained(pretrained_model, config=config, torch_dtype=torch.bfloat16).to(device)

    datasets = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "2wikimqa", "hotpotqa", "musique",
        "lcc", "repobench-p",
    ]
    
    for dataset in datasets:
        template = dataset2prompt[dataset]
        
        test_file = f'/XXX/public/data/longbench/{dataset}.jsonl'

        data_all = []
        with open(test_file, 'r') as file:
            lines = file.readlines()
        data_all = [json.loads(line) for line in lines]
        data_subset = data_all[rank::world_size]
        get_pred(model, rank, data_subset, dataset)


# cd /XXX/models/qjl_kernel
# python setup.py build_ext --inplace

# don't forget 2 remove the output_dir before you run    
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test4qjl.py

