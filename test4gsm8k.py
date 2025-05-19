"""You can use this code 2 rerun the GSM8K result"""
import os
import re
import gzip
import json
import random
import string
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from models.modeling_llama_int import LlamaIntForCausalLM
from models.modeling_llama_zip import LlamaZipForCausalLM
from models.modeling_llama_kivi import LlamaKIVIForCausalLM
from models.modeling_llama_polar import LlamaPolarForCausalLM, generate



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def extract_single_number(s):
    # Use regex to find the first sequence of digits
    answers = re.findall(r'\d+', s)
    return int(answers[-1]) if answers else -1


def judge_gsm8k(output_text, answer):
    std_answer = int(answer.split("The answer is ")[1].split(".")[0].strip().replace(",", ""))
    gen_answer = extract_single_number(output_text.split("\n")[-1])

    return int(gen_answer == std_answer)


distributed_state = PartialState()


# use examples 2 create the prompt
examples = """Question: What is fifteen more than a quarter of 48?
Let's think step by step
A quarter of 48 is 48/4=12.
The number is 12+15=27.
The answer is 27

Question: Twice Angie's age, plus 4, is 20. How old is Angie?
Let's think step by step
Twice Angie's age is 20-4=16.
Angie is 16/2=8.
The answer is 8

Question: Steve is 5'6".  He grows 6 inches.  How tall is he in inches?
Let's think step by step
He is 5*12+6=66 inches tall before the growth spurt.
After growing he is now 66+6=72 inches
The answer is 72

Question: If 12 bags of oranges weigh 24 pounds, how much do 8 bags weigh?
Let's think step by step
Each bag of oranges weighs 24 pounds / 12 = 2 pounds.
8 bags of oranges would weigh 8 * 2 pounds = 16 pounds.
The answer is 16

Question: A roll of 25 m wire weighs 5 kg. How much does a 75 m roll weigh?
Let's think step by step
We know that the 75 m roll is three times bigger than the 25 m roll because 75 m / 25 m = 3
This means the 75 m roll weighs 3 times more than the 25 m roll: 5 kg * 3 = 15 Kg
The answer is 15

Question: Fifteen more than a quarter of a number is 27. What is the number?
Let's think step by step
A quarter of the number is 27-15=12.
The number is 12*4=48.
The answer is 48

Question: 198 passengers fit into 9 buses. How many passengers fit in 5 buses?
Let's think step by step
198 passengers / 9 buses = 22 passengers fit in one bus.
22 passengers/bus * 5 buses = 110 passengers fit in 5 buses.
The answer is 110

Question: John takes a pill every 6 hours.  How many pills does he take a week?
Let's think step by step
He takes 24/6=4 pills a day
So he takes 4*7=28 pills a week
The answer is 28"""


query_template = """<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"""
reply_template = """<|start_header_id|>assistant<|end_header_id|>\n\n{reply}<|eot_id|>"""

# query_template = """{query}\n"""
# reply_template = """{reply}\n\n"""


icl_prompt = ""
examples = examples.split("\n\n")
examples = [example.split("Question: ")[1] for example in examples]
for i, example in enumerate(examples):
    query = example.split("\n")[0]
    response = "\n".join(example.split("\n")[1:])
    icl_prompt += query_template.format(query=query) + reply_template.format(reply=response)


test_file = '/XXX/public/data/gsm8k/gsm8k.json'
output_file = '/XXX/output/gsm8k/gsm8k.jsonl'

pretrained_model = '/XXX/public/llama-3.1-8b-chat/'
# pretrained_model = "/XXX/public/llama2-7b-chat/"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
generation_config = GenerationConfig.from_pretrained(pretrained_model)

idx, examples = 0, []


with open(test_file, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)
        # for llama-3.1-8b-chat
        prompt = icl_prompt + query_template.format(query=json_obj["conversations"][0]['value']) + "<|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step\n"
        # # for llama-2-7b-chat
        # prompt = icl_prompt +  query_template.format(query=json_obj["conversations"][0]['value']) + "Let's think step by step\n" 
        examples.append((idx, tokenizer.encode(prompt), json_obj["conversations"][1]['value']))
        idx += 1

distributed_state.wait_for_everyone()

if distributed_state.is_main_process:
    print('====== Tokenize End! ======')


total = idx

model = AutoModelForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)
# model = LlamaIntForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)
# model = LlamaZipForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)
# model = LlamaKIVIForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)
# model = LlamaPolarForCausalLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16).to(distributed_state.device)  

model.eval()

seed_everything(0)

completions_per_process = []

# you can use this sample code 4 test
# examples = random.sample(examples, 10)
# examples = examples[:10]

with distributed_state.split_between_processes(examples, apply_padding=True) as example_per_process:
    for idx, input_ids, answer in tqdm(example_per_process):
        input_ids = torch.tensor([input_ids], device=distributed_state.device)
        output_ids = generate(model, input_ids=input_ids, attention_mask=None, max_new_tokens=256, generation_config=generation_config) 
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_text = output_text.split("\n\n")[0]
        completions_per_process.append((idx, output_text, answer))

completions_gather = gather_object(completions_per_process)

distributed_state.wait_for_everyone()

if distributed_state.is_main_process:
    all_result = dict()
    all_output = []
    for idx, output_text, answer in completions_gather:
        all_result[idx] = judge_gsm8k(output_text, answer)
        all_output.append((idx, output_text, answer, all_result[idx]))
    final_result = sum(all_result.values()) / len(all_result)
    print(f'There are {len(all_result)} examples in total, acc is {final_result * 100:.2f}')
    with open(output_file, 'w') as fw:
        for idx, output_text, answers, score in all_output:
            fw.write(json.dumps({"idx": idx, "output_text": output_text, "answers": answers, "score": score}) + "\n")
            

# CUDA_VISIBLE_DEVICES=0 python test4gsm8k.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=8888 test4gsm8k.py

