"""Modified from modeling_llama_qwen2"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, _flash_attention_forward
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
)

from .kernel4group import attention_decode_forward_triton_impl


def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, _, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :, :].expand(batch, num_key_value_heads, n_rep, _, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, _, slen, head_dim)


class Qwen2PolarGroupAttention(Qwen2Attention):
    def __init__(self, config, layer_idx = None):
        super().__init__(config, layer_idx)
        self.rbits = 4
        self.tbits = 4
        self.group_size = 128
        self.residual_length = 128

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_value: Cache = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None, 
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if q_len == 1:  # decoding time
            key_states_full, indices, rscale, rmn, tscale, tmn, value_states_full, kv_seq_len = past_key_value

            attn_weights_quant = None
            if indices is not None:
                attn_weights_quant = attention_decode_forward_triton_impl(
                    query_states,
                    repeat_kv_quant(indices, self.num_key_value_groups),
                    repeat_kv_quant(rscale, self.num_key_value_groups),
                    repeat_kv_quant(rmn, self.num_key_value_groups),
                    repeat_kv_quant(tscale, self.num_key_value_groups),
                    repeat_kv_quant(tmn, self.num_key_value_groups),
                    tbits=self.tbits,
                    rbits=self.rbits,
                )

            key_states_full = key_states if key_states_full is None else torch.cat([key_states_full, key_states], dim=2)
            
            attn_weights_full = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3)).to(torch.float32)
            attn_weights = attn_weights_full / math.sqrt(self.head_dim) if attn_weights_quant is None else torch.cat([attn_weights_quant, attn_weights_full], dim=-1) / math.sqrt(self.head_dim)  

            
            if key_states_full.shape[2] % self.residual_length == 0:
                indices_, rscale_, rmn_, tscale_, tmn_ = self.quantize_and_pack_nbit(key_states_full)

                indices = torch.cat([indices, indices_], dim=2) if indices is not None else indices_
                rscale = torch.cat([rscale, rscale_], dim=2) if rscale is not None else rscale_
                rmn = torch.cat([rmn, rmn_], dim=2) if rmn is not None else rmn_
                tscale = torch.cat([tscale, tscale_], dim=2) if tscale is not None else tscale_
                tmn = torch.cat([tmn, tmn_], dim=2) if tmn is not None else tmn_
                
                key_states_full = None
        
            if attention_mask is not None:   
                attn_weights += (1. - attention_mask[:, None, None, :]) * torch.finfo(attn_weights.dtype).min
  
            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) 
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))    

            past_key_value = (key_states_full, indices, rscale, rmn, tscale, tmn, value_states_full, kv_seq_len + 1) 
        else:  # pre-filling 
            assert past_key_value is None
            kv_seq_len = key_states.shape[2]

            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                q_len,
                use_top_left_mask=False,
                is_causal=self.is_causal,
            )

            residual_length = kv_seq_len % self.residual_length

            if residual_length == 0:
                key_states_full, key_states_quant = None, key_states
            else:
                key_states_full = key_states if kv_seq_len < self.residual_length else key_states[:, :, -residual_length:, :]
                key_states_quant = None if kv_seq_len < self.residual_length else key_states[:, :, :-residual_length, :]

            indices, rscale, rmn, tscale, tmn = None, None, None, None, None
            if key_states_quant is not None:
                indices, rscale, rmn, tscale, tmn = self.quantize_and_pack_nbit(key_states_quant)
    
            past_key_value = (key_states_full, indices, rscale, rmn, tscale, tmn, value_states, kv_seq_len)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
    
    def quantize_and_pack_nbit(self, key_states):
        """Left 4 future triton implementation"""
        B, N, L, D = key_states.shape
        assert D % 2 == 0 and L % self.group_size == 0 and self.rbits + self.tbits <= 8
        D, G = D // 2, self.group_size

        key_states = key_states.view(B, N, L // G, G, 2, D)

        phi = torch.atan2(key_states[:, :, :, :, 1, :], key_states[:, :, :, :, 0, :])
        phi = torch.where(phi < 0, phi + 2 * torch.math.pi, phi) 

        tmx, tmn = phi.max(-2, keepdim=True)[0], phi.min(-2, keepdim=True)[0]
        tscale = (tmx - tmn) / (2 ** self.tbits)
        theta = torch.clamp(torch.floor((phi - tmn) / tscale).to(torch.uint8), 0, 2 ** self.tbits - 1)

        radii = torch.norm(key_states, p=2, dim=-2)
        rmx, rmn = radii.max(-2, keepdim=True)[0], radii.min(-2, keepdim=True)[0]
        rscale = (rmx  - rmn) / (2 ** self.rbits)        
        rho = torch.clamp(torch.floor((radii - rmn) / rscale).to(torch.uint8), 0, 2 ** self.rbits - 1)

        indices = (rho << self.tbits) + theta

        return indices, rscale, rmn, tscale, tmn

    

class Qwen2PolarDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = Qwen2PolarGroupAttention(config, layer_idx)
        
    def forward(
        self, 
        hidden_states, 
        attention_mask = None, 
        past_key_value = None,
        position_embeddings = None, 
        **kwargs
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        return outputs


class Qwen2PolarModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen2PolarDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
    
    def forward(self, input_ids = None, attention_mask = None, past_key_values = None):
        bsz, q_len = input_ids.shape

        next_decoder_cache = ()

        inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0 if past_key_values is None else past_key_values[0][-1]

        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=inputs_embeds.device)

        # we use flash-attn in pre-filling time and bsz == 1 when decoding, thus we skip the mask convertion here
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, cache_position.unsqueeze(0))

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states, past_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                position_embeddings=position_embeddings,
            )
            next_decoder_cache += (past_key_value,)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, next_decoder_cache
    

class Qwen2PolarForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2PolarModel(config)
    
    def forward(
        self, 
        input_ids = None, 
        attention_mask = None, 
        past_key_values = None, 
        num_logits_to_keep = 0,
        return_dict = True,  # dummy
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values = None, attention_mask = None, **kwargs):
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise NotImplementedError

        model_inputs = {
            "input_ids": input_ids[:, -1:] if past_key_values is not None else input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        return model_inputs

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device):
        return
    

def top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1,):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits



def return_top_p_filtered_indices(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = (cumulative_probs > top_p)

    # Shift the indices to the right to keep also the first token above the threshold
    # ~ keep at least one token - the first token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    sorted_indices_to_remove = ~ sorted_indices_to_remove
    sorted_indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)

    return sorted_indices_to_remove


def sample(logits, generation_config, do_sample=False):
    top_k = generation_config.top_k
    top_p = generation_config.top_p
    temperature = generation_config.temperature
    
    if do_sample:
        logits_ = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
        output_ids = torch.multinomial(logits_.softmax(-1), num_samples=1)
    else:
        output_ids = torch.argmax(logits, dim=-1)
    return output_ids


def generate(model, input_ids, attention_mask, generation_config, do_sample=False, max_new_tokens=8):
    current_input_ids = input_ids

    batch_size = input_ids.size(0)

    generate_ids = torch.zeros([batch_size, max_new_tokens], dtype=torch.long, device=model.device)
    eos_sequence = torch.zeros([batch_size, 1], dtype=torch.bool, device=model.device)
    stop_tensor = torch.tensor(generation_config.eos_token_id, dtype=torch.long, device=model.device)

    next_decoder_cache = None

    with torch.no_grad():
        step = 0
        while True:
            if step >= max_new_tokens:
                break
            
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=next_decoder_cache,
                num_logits_to_keep=1,
            )

            output_ids = sample(outputs.logits, generation_config, do_sample=do_sample)
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

    return generate_ids



if __name__ == "__main__":
    pass
    
# CUDA_VISIBLE_DEVICES=0 python modeling_qwen2_polar.py
