import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from typing import Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, _flash_attention_forward, repeat_kv
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM
)
from modeling_utils_qjl import QJLKeyQuantizer


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


class LlamaQJLAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.initial_layers_count = config.initial_layers_count

        self.qjl = config.qjl
        self.key_quantization_bits = config.key_quantization_bits

        self.qjl_initial_layers = config.qjl_initial_layers
        self.key_quantization_bits_initial_layers = config.key_quantization_bits_initial_layers

        self.outlier_count_initial_layers = config.outlier_count_initial_layers
        self.outlier_count_general = config.outlier_count_general

        self.group_size = config.group_size
        self.buffer_size = config.buffer_size

    
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
            kv_quant, value_states_full, kv_seq_len = past_key_value

            kv_quant.update_sketch(key_states)
            att_qk = kv_quant.attention_score(query_states)
 
            attn_weights = att_qk / math.sqrt(self.head_dim)
            
            if attention_mask is not None:     
                attn_weights += (1. - attention_mask[:, None, None, :]) * torch.finfo(attn_weights.dtype).min

            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) 

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))  

            past_key_value = (kv_quant, value_states_full, kv_seq_len + 1)    
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

            if self.layer_idx < self.initial_layers_count:
                kv_quant = QJLKeyQuantizer(self.qjl_initial_layers, self.outlier_count_initial_layers, self.buffer_size, self.group_size, self.key_quantization_bits_initial_layers)
            else:
                kv_quant = QJLKeyQuantizer(self.qjl, self.outlier_count_general, self.buffer_size, self.group_size, self.key_quantization_bits)
            
            kv_quant.build_sketch(key_states)

            past_key_value = (kv_quant, value_states, kv_seq_len)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value




class LlamaQJLDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaQJLAttention(config, layer_idx=layer_idx)

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
    

class LlamaQJLModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([LlamaQJLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

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


class LlamaQJLForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaQJLModel(config)
    
    def forward(
        self, 
        input_ids = None, 
        attention_mask = None, 
        past_key_values = None, 
        num_logits_to_keep = 0,
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


if __name__ == '__main__':
    # pass
    from transformers import AutoTokenizer, AutoConfig
    from transformers.generation.utils import GenerationConfig
    from modeling_utils_qjl import QJLSketch
    
    # change your pretrained model path here
    pretrained_model = '/XXX/public/llama-3.1-8b-chat/'

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token_id = 0

    config = AutoConfig.from_pretrained(pretrained_model)
    
    config.key_quantization_bits = 256
    config.key_quantization_bits_initial_layers = 512
    config.initial_layers_count = 15  # default as 

    config.outlier_count_general = 8
    config.outlier_count_initial_layers = 8

    config.group_size = 32
    config.buffer_size = 128

    generator = torch.Generator(device=device)

    config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
    config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128, rot=True, rng=generator)
    
    model = LlamaQJLForCausalLM.from_pretrained(pretrained_model, config=config, torch_dtype=torch.float16).to(device='cuda')
    model.eval()

    generation_config = GenerationConfig.from_pretrained(pretrained_model)

    text = ["This is just for test, here we go and go again!\n" * 32, "This is just for test, here we go and go again!\n" * 16]

    model_inputs = tokenizer(text, return_tensors="pt", padding_side="left", padding=True)
    input_ids = model_inputs["input_ids"].cuda()
    attention_mask =  model_inputs["attention_mask"].cuda()

    output_ids = generate(model, input_ids, attention_mask, generation_config, max_new_tokens=32)
    output_text = tokenizer.batch_decode(output_ids)
    breakpoint()


# CUDA_VISIBLE_DEVICES=0 python modeling_llama_qjl.py


