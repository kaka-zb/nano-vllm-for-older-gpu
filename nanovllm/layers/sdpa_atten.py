from typing import Tuple
import torch
from torch import nn
import triton
import triton.language as tl

from torch.nn.functional import scaled_dot_product_attention
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def get_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_gathered_ptr,    
    v_gathered_ptr,
    D: tl.constexpr,
):
    # Each program processes one token's K/V vector
    idx = tl.program_id(0) 

    slot_idx = tl.load(slot_mapping_ptr + idx)
    
    # Load Key and Value for this token from the KV cache
    k_offsets = slot_idx * D + tl.arange(0, D)
    v_offsets = slot_idx * D + tl.arange(0, D)
    k = tl.load(k_cache_ptr + k_offsets)
    v = tl.load(v_cache_ptr + v_offsets)

    output_k_offsets = idx * D + tl.arange(0, D)
    output_v_offsets = idx * D + tl.arange(0, D)    
    tl.store(k_gathered_ptr + output_k_offsets, k)
    tl.store(v_gathered_ptr + output_v_offsets, v)


def get_kvcache(
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    total_tokens_in_batch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    D_feature = num_kv_heads * head_dim 

    k_gathered = torch.empty(total_tokens_in_batch, num_kv_heads, head_dim, 
                             dtype=k_cache.dtype, device=k_cache.device)
    v_gathered = torch.empty(total_tokens_in_batch, num_kv_heads, head_dim, 
                             dtype=v_cache.dtype, device=v_cache.device)
    
    get_kvcache_kernel[(total_tokens_in_batch,)](
        k_cache,
        v_cache,
        slot_mapping,
        k_gathered,
        v_gathered,
        D_feature,
    )
    
    return k_gathered, v_gathered


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.cnt = 0
        self.layer = 0

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor

        q_reshaped = q.view(-1, self.num_heads, self.head_dim) # [total_tokens_in_batch, num_heads, head_dim]
        k_reshaped = k.view(-1, self.num_kv_heads, self.head_dim)
        v_reshaped = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        
        k_cache_layer = self.k_cache
        v_cache_layer = self.v_cache

        cached_k, cached_v = k_reshaped, v_reshaped

        if k_cache_layer.numel() and v_cache_layer.numel():
            store_kvcache(k_reshaped, v_reshaped, k_cache_layer, v_cache_layer, context.slot_mapping)

            total_tokens_to_compute = 0
            for seq_len in context.seq_lens:
                total_tokens_to_compute = total_tokens_to_compute + seq_len
            cached_k, cached_v = get_kvcache(k_cache_layer, v_cache_layer, context.seq_slot_mapping, 
                                            self.num_kv_heads, self.head_dim, total_tokens_to_compute)
        
        
        if context.is_prefill:
            start_q = 0
            start_kv = 0
            o = torch.empty(q.shape[0], self.num_heads, self.head_dim, dtype=k_cache_layer.dtype, device=k_cache_layer.device)
            for seq_idx in range(context.seq_lens.shape[0]):
                per_req_query = q_reshaped[start_q:start_q+context.prefill_seq_lens[seq_idx], :, :]
                per_req_key = cached_k[start_kv:start_kv+context.seq_lens[seq_idx], :, :]
                per_req_value = cached_v[start_kv:start_kv+context.seq_lens[seq_idx], :, :]
                per_req_out = scaled_dot_product_attention(
                    per_req_query.movedim(0, 1).unsqueeze(0),
                    per_req_key.movedim(0, 1).unsqueeze(0),
                    per_req_value.movedim(0, 1).unsqueeze(0),
                    enable_gqa=True,
                    scale=self.scale,
                    is_causal=True,
                ).squeeze(0).movedim(1, 0)
                o[start_q:start_q+context.prefill_seq_lens[seq_idx], :, :] = per_req_out
                start_q = start_q + context.prefill_seq_lens[seq_idx]
                start_kv = start_kv + context.seq_lens[seq_idx]
        else:
            start_q = 0
            start_kv = 0
            o = torch.empty(q.shape[0], self.num_heads, self.head_dim, dtype=k_cache_layer.dtype, device=k_cache_layer.device)
            for seq_idx in range(context.seq_lens.shape[0]):
                per_req_query = q_reshaped[start_q:start_q+1, :, :]
                per_req_key = cached_k[start_kv:start_kv+context.seq_lens[seq_idx], :, :]
                per_req_value = cached_v[start_kv:start_kv+context.seq_lens[seq_idx], :, :]
                per_req_out = scaled_dot_product_attention(
                    per_req_query.movedim(0, 1).unsqueeze(0),
                    per_req_key.movedim(0, 1).unsqueeze(0),
                    per_req_value.movedim(0, 1).unsqueeze(0),
                    enable_gqa=True,
                    scale=self.scale,
                    is_causal=False,
                ).squeeze(0).movedim(1, 0)

                o[start_q:start_q+1, :, :] = per_req_out[-1:, :, :]
                start_q = start_q + 1
                start_kv = start_kv + context.seq_lens[seq_idx]

        o = o.view(-1, self.num_heads * self.head_dim)

        return o