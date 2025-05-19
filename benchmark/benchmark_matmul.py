import os
import json
import time
import torch
from pprint import pprint

from transformers.models.llama.modeling_llama import repeat_kv

from models.kivi_quant.matmul import cuda_bmm_fA_qB_outer
from models.kernel4group import attention_decode_forward_triton_impl

# from utils import draw4chart, seed_everything


global B
N = 32
N_k = 8
D = 64
global L 

# for group-wise setting
group_size = 128

# for kivi setting
kbits = 2
group_size = 32

rbits, tbits = 4, 4
# rbits, tbits = 3, 3


seed = 2


def test4matmul():
    assert L % group_size == 0

    query_states = torch.randn((B, N, 1, 2 * D), dtype=torch.float16, device='cuda')
    key_states = repeat_kv(torch.randn((B, N_k, L, 2 * D), dtype=torch.float16, device='cuda'), N // N_k)

    for i in range(1000):
        torch.matmul(query_states, key_states.transpose(2, 3))
    
    total_time = 0.
    
    for i in range(10000): 
        torch.cuda.synchronize()
        st = time.perf_counter()
        torch.matmul(query_states, key_states.transpose(2, 3))
        torch.cuda.synchronize()
        et = time.perf_counter()
        total_time += et - st
    
    print(f'Pytorch Matmul CUDA Test')
    print(f'Time: {total_time * 1000 / 10000:.2f} ms') 

    return total_time * 0.1


def test4kivi():
    assert L % group_size == 0

    fA = torch.randn((B, N, 1, D * 2), dtype=torch.float16, device='cuda')
    qB = repeat_kv(torch.randint(0, 2 ** kbits, (B, N_k, D * 2, L // (32 // kbits)), dtype=torch.int32, device='cuda'), N // N_k)
    scales = repeat_kv(torch.randn((B, N_k, D * 2, L // group_size), dtype=torch.float16, device='cuda'), N // N_k)
    zeros = repeat_kv(torch.randn((B, N_k, D * 2, L // group_size), dtype=torch.float16, device='cuda'), N // N_k)
        
    # warm up
    for i in range(1000):
        cuda_bmm_fA_qB_outer(group_size, fA, qB, scales, zeros, kbits)
            
    total_time = 0.
    
    for i in range(10000): 
        torch.cuda.synchronize()
        st = time.perf_counter()
        cuda_bmm_fA_qB_outer(group_size, fA, qB, scales, zeros, kbits)
        torch.cuda.synchronize()
        et = time.perf_counter()
        # print(f"{(et - st) * 1000} ms")
        total_time += et - st

    print(f'KIVI CUDA Test')
    print(f'Time: {total_time * 1000 / 10000:.2f} ms') 

    return total_time * 0.1


def test4polar():
    assert L % group_size == 0

    query_states = torch.randn(size=(B, N, 1, 2 * D), device='cuda', dtype=torch.float16)

    indices = torch.randint(0, 250, size=(B, N, L // group_size, group_size, D), device='cuda', dtype=torch.uint8)

    rscale = torch.randn(size=(B, N, L // group_size, 1, D), device='cuda', dtype=torch.float16)
    rmn = torch.randn(size=(B, N, L // group_size, 1, D), device='cuda', dtype=torch.float16)
    tscale = torch.randn(size=(B, N, L // group_size, 1, D), device='cuda', dtype=torch.float16)
    tmn = torch.randn(size=(B, N, L // group_size, 1, D), device='cuda', dtype=torch.float16)


    # warm up
    for i in range(1000):                
        attention_decode_forward_triton_impl(query_states, indices, rscale, rmn, tscale, tmn, rbits=rbits, tbits=tbits)

    total_time = 0.

    for i in range(10000):
        torch.cuda.synchronize()
        st = time.perf_counter()
        attention_decode_forward_triton_impl(query_states, indices, rscale, rmn, tscale, tmn, rbits=rbits, tbits=tbits)
        torch.cuda.synchronize()
        et = time.perf_counter()
        # print(f"{(et - st) * 1000} ms")
        total_time += et - st
    
    print(f'Polar CUDA Test')
    print(f'Time: {total_time * 1000 / 10000:.2f} ms') 

    return total_time * 0.1





if __name__ == '__main__':
    all_comparisons = []
    all_batch_size = [1, 2, 4, 8]

    all_times = []

    for bsz in all_batch_size:
        B = bsz

        batch_wise_times = []

        for length in [4096, 8192, 16384, 32768, 65536, 131072]:
            print('\n\n')
            L = length
            print(f'Test for batch_size: {B}, length: {L}')
            try:
                batch_wise_times.append(round(test4matmul(), 2))
                # batch_wise_times.append(round(test4polar(), 2))
                # batch_wise_times.append(round(test4kivi(), 2))
            except Exception as error:
                print(error)
                batch_wise_times.append(None)
        
        all_times.append(batch_wise_times)

        print('\n\n')  

    pprint(all_times, width=40)
    
    # draw4chart(all_comparisons, all_batch_size)


# fp16
# [[0.05, 0.08, 0.12, 0.22, 0.42, 0.8],
#  [0.08, 0.13, 0.22, 0.42, 0.8, 1.58],
#  [0.13, 0.22, 0.42, 0.8, 1.58, 3.13],
#  [0.22, 0.42, 0.8, 1.58, 3.13, 6.22]]

# polar44
# [[0.11, 0.12, 0.15, 0.23, 0.4, 0.75],
#  [0.11, 0.15, 0.23, 0.41, 0.76, 1.49],
#  [0.15, 0.24, 0.42, 0.78, 1.5, 2.93],
#  [0.24, 0.42, 0.78, 1.49, 2.93, 5.79]]

# polar33
# [[0.08, 0.09, 0.12, 0.18, 0.3, 0.54],
#  [0.09, 0.12, 0.18, 0.3, 0.54, 1.03],
#  [0.12, 0.18, 0.3, 0.54, 1.03, 2.01],
#  [0.18, 0.3, 0.54, 1.03, 2.01, 3.97]]

# kivi
# [[0.09, 0.13, 0.21, 0.4, 0.78, 1.54],
#  [0.12, 0.22, 0.41, 0.8, 1.58, 3.17],
#  [0.22, 0.42, 0.82, 1.61, 3.18, 6.36],
#  [0.41, 0.82, 1.61, 3.19, 6.34, None]]

# kivi2
# [[0.07, 0.11, 0.18, 0.32, 0.62, 1.2],
#  [0.11, 0.18, 0.32, 0.62, 1.19, 2.35],
#  [0.18, 0.32, 0.61, 1.19, 2.35, 4.67],
#  [0.32, 0.61, 1.19, 2.35, 4.66, None]]


# CUDA_VISIBLE_DEVICES=0 python benchmark_matmul.py












