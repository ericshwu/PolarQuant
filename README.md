# ReRun LongBench & GSM8k
## Environment Setup

We reccommend rerunning the code on NVIDIA A100 or A800 GPUs. 

You need to install the stable version of Triton with the tl.gather feature from source. We highly recommend packaging it as a .whl file.

Our setup scripts **init_env.sh**, include our own prebuilt wheel files, which you should replace with your custom versions.

FlashAttention wheel files can be found and downloaded from: https://github.com/Dao-AILab/flash-attention

```bash
# python3.8,cuda 12.1,pytorch==2.1,torchaudio==2.1.0+cu121,torchvision==0.16.0+cu121
cd utils 
bash init_env.sh
```

## Data Preparation

> Download the LongBench dataset from its [official GitHub repository](https://github.com/THUDM/LongBench/tree/main/LongBench) and place it in "./public/data/longbench"

> Download the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset from Hugging Face (openai/gsm8k) and place it in ./public/data/gsm8k.


## ReRun LongBench
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=6666 test4long.py
```

## ReRun GSM8k
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=8888 test4gsm8k.py
```

# ReRun Reasonging Models

We use lighteval to rerun the reproduction of DeepSeek-R1. 

Please note the following key modifications.

File "./lighteval-main/src/lighteval/models/transformers/transformers_model.py"

> 1. Line 597: Apply_chat_template of DS models.

> 2. Line 635: Fix the generation parameters as "do_sample=1,num_samples=1" as we use **Acc.** for all tasks.

File "./lighteval-main/src/lighteval/tasks/default_tasks.py"

> 3. Update the metrics into Acc. for all tasks.

## Environment Setup

We recommend running on NVIDIA A100/A800 GPUs.
Ensure you're using a Conda environment with Python 3.10 and CUDA 12.1, as required by lighteval.

```bash
# python3.10,cu12.1
cd ./lighteval-main/scripts
bash init_env.sh
```

## ReRun Example

You can evaluate tasks using a command like the following.

We also report the logs of all our experiments in "./lighteval-main/scripts/logs"

```bash
cd ./lighteval-main/scripts
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup accelerate launch main.py --task "lighteval|aime24|0|0" --pretrained /XXX/public/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/ --batch_size 1 --generation_parameters "{'temperature':0.6, 'top_p':0.95}" &>polar44_aime24_qwen_1.5b.log &
```

# ReRun Benchmark

Benchmark code for evaluating latency, memory, and throughput is available in the ./benchmark directory.

For example, to run a latency test on matrix multiplication implementations:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_matmul.py
```







