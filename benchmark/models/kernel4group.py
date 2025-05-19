"""This is the kernel implementation 2 support group-wise PolarQuant with 4-bit. 
It can be used for experimental low-bit (like 3-bit). A formal low-bit implementation 
will be released in the future 2 support low-bit pack and unpack."""
import os
# os.environ["TRITON_INTERPRET"] = "1"  # for triton debug

import pdb
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    q_ptr,
    i_ptr,
    rs_ptr,
    rm_ptr,
    ts_ptr,
    tm_ptr,
    o_ptr,
    rbits: tl.constexpr,
    tbits: tl.constexpr,
    B: tl.constexpr, N: tl.constexpr, Nb: tl.constexpr, G: tl.constexpr, D: tl.constexpr, 
    Nk: tl.constexpr,
):
    """
    query_states: (B, N, 1, 2 * D) ~ float16
    indices: (B, Nk, Nb, G, D) uint8
    rscale: (B, Nk, Nb, 1, D) ~ float16
    rmn: (B, Nk, Nb, 1, D) ~ float16
    tscale: (B, Nk, Nb, 1, D) ~ float16
    tmn: (B, Nk, Nb, 1, D) ~ float16
    output: (B, N, 1, L) ~ float16
    """
    pid_page = tl.program_id(axis=0) 
    pid_block = tl.program_id(axis=1)

    block_start_query = q_ptr + pid_page * (N // Nk) * 2 * D 

    block_start_indices = i_ptr + pid_page * Nb * G * D

    block_start_rscale = rs_ptr + pid_page * Nb * D + pid_block * D
    block_start_rmn = rm_ptr + pid_page * Nb * D + pid_block * D

    block_start_tscale = ts_ptr + pid_page * Nb * D + pid_block * D
    block_start_tmn = tm_ptr + pid_page * Nb * D + pid_block * D

    block_start_output = o_ptr + pid_page * (N // Nk) * (Nb * G) + tl.arange(0, N // Nk)[:, None] * (Nb * G)
        
    # load indices
    offsets_indices = pid_block * G * D + tl.arange(0, G)[None, :] * D + tl.arange(0, D)[:, None]
    indices = tl.load(block_start_indices + offsets_indices)

    offsets_scale = tl.arange(0, D)[None, :, None]
    rscale = tl.load(block_start_rscale + offsets_scale)
    rmn = tl.load(block_start_rmn + offsets_scale)

    offsets_scale = tl.arange(0, D)[None, :, None, None]
    tscale = tl.load(block_start_tscale + offsets_scale)
    tmn = tl.load(block_start_tmn + offsets_scale)

    offsets_query = tl.arange(0, N // Nk)[:, None, None, None] * 2 * D + tl.arange(0, 2)[None, None, None, :] * D + tl.arange(0, D)[None, :, None, None]
    query = tl.load(block_start_query + offsets_query)

    phi = tscale * (tl.arange(0, 1 << tbits)[None, None, :, None] + 0.5) + tmn

    tp = tl.sum(query * tl.interleave(tl.cos(phi), tl.sin(phi)), axis=-1)

    attn = tl.gather(tp, tl.broadcast_to(indices[None, :, :] & (2 ** tbits - 1), (N // Nk, D, G)), axis=-1)

    radii = rscale * (tl.arange(0, 1 << rbits)[None, None, :] + 0.5) + rmn

    attn *= tl.gather(radii, indices[None, :, :] >> tbits, axis=-1)

    attn = tl.sum(attn, axis=1) 

    offsets_output = pid_block * G + tl.arange(0, G)[None, :]
    tl.store(block_start_output + offsets_output, attn)


def attention_decode_forward_triton_impl(
    query_states: torch.FloatTensor,   # (b, n, l, d * 2)
    indices: torch.IntTensor,   # (b, nk, nb, g, d)
    rscale: torch.FloatTensor,   # (b, nk, nb, 1, d)
    rmn: torch.FloatTensor,  # (b, nk, nb, 1, d)
    tscale: torch.FloatTensor,  #  (b, nk, nb, 1, d)
    tmn: torch.FloatTensor,  # (b, nk, nb, 1, d)
    rbits: int = 4, 
    tbits: int = 4,
):
    B, Nk, Nb, G, D = indices.shape
    N = query_states.shape[1]

    attn_weights = torch.empty(size=(B, N, 1, Nb * G), device=query_states.device, dtype=torch.float32)

    with torch.cuda.device(query_states.device):
        _kernel[(B * Nk, Nb)](
            query_states, indices, 
            rscale, rmn,
            tscale, tmn,
            attn_weights, 
            rbits, tbits,
            B, N, Nb, G, D, 
            Nk,
        )

    return attn_weights



if __name__ == '__main__':
    """Code Below is the pytorch implementation of our kernel
    You can use the code below 2 check our kernel implementation"""
    from transformers.models.llama.modeling_llama import repeat_kv

    B, N, Nb, G, D = 1, 32, 3, 128, 64
    # Nk = 8
    Nk = 32
    rbits, tbits = 4, 4

    query_states = torch.randn(size=(B, N, 1, 2 * D), dtype=torch.bfloat16).cuda()

    key_states = torch.randn(size=(B, Nk, Nb * G, 2 * D), dtype=torch.bfloat16).cuda()
    key_states = key_states.view(B, Nk, Nb, G, 2, D)

    phi = torch.atan2(key_states[:, :, :, :, 1, :], key_states[:, :, :, :, 0, :])
    phi = torch.where(phi < 0, phi + 2 * torch.math.pi, phi)        
    tmx, tmn = phi.max(-2, keepdim=True)[0], phi.min(-2, keepdim=True)[0]
    tscale = (tmx - tmn) / (2 ** tbits)
    theta = torch.clamp(torch.floor((phi - tmn) / tscale).to(torch.uint8), 0, 2 ** tbits - 1)

    radii = torch.norm(key_states, p=2, dim=-2)
    rmx, rmn = radii.max(-2, keepdim=True)[0], radii.min(-2, keepdim=True)[0]
    
    rscale = (rmx  - rmn) / (2 ** rbits)    

    rho = torch.clamp(torch.floor((radii - rmn) / rscale).to(torch.uint8), 0, 2 ** rbits - 1)
    
    pphi = tscale[0, 0, 1, :, :] * torch.arange(0, 2 ** tbits, device=tscale.device)[:, None] + 0.5 * tscale[0, 0, 1, :, :] + tmn[0, 0, 1, :, :]
    rradii = rscale[0, 0, 1, :, :] * torch.arange(0, 2 ** rbits, device=rscale.device)[:, None] + 0.5 * rscale[0, 0, 1, :, :] + rmn[0, 0, 1, :, :]
    pphi = pphi.unsqueeze(0)
    rradii = rradii.unsqueeze(0)

    ttp = (query_states[0, :4, 0, :].view(4, 1, 2, 64).transpose(2, 3) * torch.stack([pphi.cos(), pphi.sin()], dim=-1)).sum(-1).transpose(1, 2)
    attn_ = torch.gather(ttp, dim=-1, index=theta[:, 0, 1, :, :].transpose(1, 2).to(torch.int64).expand(4, -1, -1))
    attn__ = attn_ * torch.gather(rradii.transpose(1, 2), dim=-1, index=rho[:, 0, 1, :, :].transpose(1, 2).to(torch.int64))
    attn___ = attn__.sum(1)

    phi = theta * tscale + 0.5 * tscale + tmn
    radii = rho * rscale + 0.5 * rscale + rmn
    key_states_recontrust = torch.stack([radii * phi.cos(), radii * phi.sin()], dim=-2).reshape(B, Nk, Nb * G, -1)
    # attn____ =  torch.matmul(query_states, repeat_kv(key_states_recontrust, 4).transpose(2, 3))
    attn____ =  torch.matmul(query_states, key_states_recontrust.transpose(2, 3))

    indices = (rho << tbits) + theta
    attn_weights = attention_decode_forward_triton_impl(query_states, indices, rscale, rmn, tscale, tmn)  
    breakpoint() 
    
    
# # src code 2 check the kernel
# CUDA_VISIBLE_DEVICES=0 python kernel4group.py



