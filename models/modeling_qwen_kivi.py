import math

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from typing import Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, _flash_attention_forward, repeat_kv

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM

from .kivi_quant.matmul import cuda_bmm_fA_qB_outer, triton_bmm_fA_qB_outer
from .kivi_quant.new_pack import triton_quantize_and_pack_along_last_dim


def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2KIVIAttention(Qwen2Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.k_bits = 4
        self.v_bits = 4
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

        if q_len == 1:
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_full, kv_seq_len = past_key_value

            # att_qkquant = None if key_states_quant_trans is None else cuda_bmm_fA_qB_outer(
            #     self.group_size, 
            #     query_states, 
            #     repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups), 
            #     repeat_kv_quant(key_scale_trans, self.num_key_value_groups), 
            #     repeat_kv_quant(key_mn_trans, self.num_key_value_groups), 
            #     self.k_bits
            # )
            att_qkquant = None if key_states_quant_trans is None else triton_bmm_fA_qB_outer(
                self.group_size, 
                query_states, 
                repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups), 
                repeat_kv_quant(key_scale_trans, self.num_key_value_groups), 
                repeat_kv_quant(key_mn_trans, self.num_key_value_groups), 
                self.k_bits
            )
            
            key_states_full = key_states if key_states_full is None else torch.cat([key_states_full, key_states], dim=2)

            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))        
            attn_weights = att_qkfull / math.sqrt(self.head_dim) if att_qkquant is None else torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
                
            if attention_mask is not None:     
                attn_weights += (1. - attention_mask[:, None, None, :]) * torch.finfo(attn_weights.dtype).min

            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) 

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
        
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))  

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits)

                key_states_full = None
                key_states_quant_trans = key_states_quant_trans_new if key_states_quant_trans is None else torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                key_scale_trans = key_scale_trans_new if key_scale_trans is None else torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                key_mn_trans = key_mn_trans_new if key_mn_trans is None else torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
            
            past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_full, kv_seq_len + 1)    
        else:
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                q_len,
                use_top_left_mask=False,
                is_causal=self.is_causal,
            )

            kv_seq_len = key_states.shape[2]

            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant, key_states_full = key_states, None
            
            key_states_quant_trans, key_scale_trans, key_mn_trans = None, None, None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)

            past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states, kv_seq_len)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class Qwen2MixedAttention(Qwen2KIVIAttention):
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

        if q_len == 1:
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len = past_key_value

            # att_qkquant = None if key_states_quant_trans is None else cuda_bmm_fA_qB_outer(
            #     self.group_size, 
            #     query_states, 
            #     repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups), 
            #     repeat_kv_quant(key_scale_trans, self.num_key_value_groups), 
            #     repeat_kv_quant(key_mn_trans, self.num_key_value_groups), 
            #     self.k_bits
            # )
            att_qkquant = None if key_states_quant_trans is None else triton_bmm_fA_qB_outer(
                self.group_size, 
                query_states, 
                repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups), 
                repeat_kv_quant(key_scale_trans, self.num_key_value_groups), 
                repeat_kv_quant(key_mn_trans, self.num_key_value_groups), 
                self.k_bits
            )

            key_states_full = key_states if key_states_full is None else torch.cat([key_states_full, key_states], dim=2)

            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))

            attn_weights = att_qkfull / math.sqrt(self.head_dim) if att_qkquant is None else torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                
                key_states_full = None
                key_states_quant_trans = key_states_quant_trans_new if key_states_quant_trans is None else torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                key_scale_trans = key_scale_trans_new if key_scale_trans is None else torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                key_mn_trans = key_mn_trans_new if key_mn_trans is None else torch.cat([key_mn_trans, key_mn_trans_new], dim=3)

            if attention_mask is not None:     
                attn_weights += (1. - attention_mask[:, None, None, :]) * torch.finfo(attn_weights.dtype).min

            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) 

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                # attn_output = cuda_bmm_fA_qB_outer(
                #     self.group_size, 
                #     attn_weights[:, :, :, :-value_full_length], 
                #     repeat_kv_quant(value_states_quant, self.num_key_value_groups), 
                #     repeat_kv_quant(value_scale, self.num_key_value_groups),
                #     repeat_kv_quant(value_mn, self.num_key_value_groups),
                #     self.v_bits,
                # )
                att_qkquant = None if key_states_quant_trans is None else triton_bmm_fA_qB_outer(
                    self.group_size, 
                    attn_weights[:, :, :, :-value_full_length], 
                    repeat_kv_quant(value_states_quant, self.num_key_value_groups), 
                    repeat_kv_quant(value_scale, self.num_key_value_groups),
                    repeat_kv_quant(value_mn, self.num_key_value_groups),
                    self.v_bits,
                )
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))
            
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
        
                value_states_quant = value_states_quant_new if value_states_quant is None else torch.cat([value_states_quant, value_states_quant_new], dim=2)
                value_scale = scale if value_scale is None else torch.cat([value_scale, scale], dim=2)
                value_mn = mn if value_mn is None else torch.cat([value_mn, mn], dim=2)
            
            past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len + 1)
        else:
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                q_len,
                use_top_left_mask=False,
                is_causal=self.is_causal,
            )

            kv_seq_len = key_states.shape[2]

            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant, key_states_full = key_states, None
            
            key_states_quant_trans, key_scale_trans, key_mn_trans = None, None, None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant, value_states_full, value_scale, value_mn = None, value_states, None, None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, self.group_size, self.v_bits)

            past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value



class Qwen2KIVIDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2KIVIAttention(config, layer_idx) 

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
    

class Qwen2KIVIModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen2KIVIDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

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


class Qwen2KIVIForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2KIVIModel(config)
    
    def forward(
        self, 
        input_ids = None, 
        attention_mask = None, 
        past_key_values = None, 
        num_logits_to_keep = 0,
        return_dict=None,
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



def generate(model, input_ids, attention_mask, generation_config=None, max_new_tokens=8):
    current_input_ids = input_ids

    batch_size = input_ids.size(0)

    generate_ids = torch.empty([batch_size, max_new_tokens], dtype=torch.long, device=model.device)
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

    return generate_ids


if __name__ == '__main__':
    pass

# CUDA_VISIBLE_DEVICES=0 python modeling_qwen2_kivi.py



