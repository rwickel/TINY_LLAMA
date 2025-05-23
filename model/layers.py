# layers.py
from typing import Any, Dict, List
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F
from model.args import ModelArgs
import math
from typing import Any, Dict, List, Optional, Tuple
import torch.distributed as dist
import os

def basic_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot product attention with optional masking, dropout, GQA.

    Args:
        query: Query tensor (..., L, E) where L is target sequence length.
        key: Key tensor (..., S, E) where S is source sequence length.
        value: Value tensor (..., S, Ev).
        attn_mask: Optional attention mask. Can be boolean (True to keep) or float (-inf to mask).
                   Expected shape broadcastable to (..., L, S).
        dropout_p: Dropout probability.
        is_causal: If True, applies a causal mask (query i cannot attend to key j > i).
                   attn_mask should be None if is_causal is True.
        scale: Scaling factor. Defaults to 1/sqrt(query.size(-1)).
        enable_gqa: If True, repeats key/value heads to match query heads.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Attention output tensor (..., L, Ev).
            - Attention weights tensor (..., L, S).
    """
    # Ensure query, key, value are tensors
    if not all(isinstance(t, torch.Tensor) for t in [query, key, value]):
        raise TypeError("query, key, and value must be torch.Tensors")

    L, S = query.size(-2), key.size(-2)
    E = query.size(-1)

    # Default scale factor
    scale_factor = 1.0 / math.sqrt(E) if scale is None else scale

    # Initialize attention bias (for masking)
    attn_bias = torch.zeros(query.shape[:-2] + (L, S), dtype=query.dtype, device=query.device) # Match batch/head dims

    # Apply causal masking if requested
    if is_causal:
        if attn_mask is not None:
             raise ValueError("attn_mask must be None when is_causal=True")
        # Create causal mask (True for allowed positions)
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        # Fill bias where mask is False (upper triangle)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    # Apply provided attention mask
    if attn_mask is not None:
        # Ensure mask is broadcastable to attn_bias shape
        try:
            broadcast_mask_shape = torch.broadcast_shapes(attn_mask.shape, attn_bias.shape)
        except RuntimeError as e:
            raise ValueError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to bias shape {attn_bias.shape}: {e}")

        if attn_mask.dtype == torch.bool:
            # Fill bias where mask is False
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            # Assume float mask (-inf for masked positions), add it to bias
            attn_bias = attn_bias + attn_mask # Ensure mask is already broadcastable

    # Handle Grouped Query Attention (GQA)
    if enable_gqa:
        num_query_heads = query.size(-3)
        num_kv_heads = key.size(-3)
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"Query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads}) for GQA")
        n_rep = num_query_heads // num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=-3)
            value = value.repeat_interleave(n_rep, dim=-3)

    # --- Attention Calculation ---
    # 1. Calculate scores: Q @ K^T
    attn_weight = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scale scores
    attn_weight = attn_weight * scale_factor

    # 3. Add masking bias
    attn_weight = attn_weight + attn_bias # Bias already incorporates causal and attn_mask

    # 4. Apply softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # 5. Apply dropout (only during training)
    # Note: torch.dropout requires `train` argument. Assume model.training state controls this.
    # If called outside nn.Module, pass `self.training` if available, otherwise default to False.
    is_training = isinstance(dropout_p, float) and dropout_p > 0.0 # Basic check if dropout is active
    if is_training: # Check if dropout should be applied
         attn_weight = F.dropout(attn_weight, p=dropout_p) # Use functional dropout

    # 6. Multiply weights by Value: W @ V
    output = torch.matmul(attn_weight, value)

    return output, attn_weight

def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)

def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float):
        low_freq_factor = 1
        old_context_len = 8192 
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool,
    scale_factor: float,
    high_freq_factor: float,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

    
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.eps) * self.weight

class Attention(nn.Module):
    """
    Multi-head attention module with optional RoPE, QKNorm, and KV caching.
    """
    def __init__(
        self,
        args: ModelArgs,
        use_qk_norm: bool,
        use_rope: bool,
        add_bias: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale
        self.norm_eps = args.norm_eps # Epsilon for QK norm if used

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads # Assuming local heads = total heads for now
        self.n_local_kv_heads = self.n_kv_heads # Assuming local KV heads = total KV heads
        self.head_dim = args.dim // args.n_heads

        # Ensure n_heads is divisible by n_kv_heads for grouped query attention
        if self.n_heads % self.n_kv_heads != 0:
             raise ValueError(
                 f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
             )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # Linear projections for Query, Key, Value, and Output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=add_bias)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=add_bias)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=add_bias)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=add_bias)

        # *** FIXED: Register cache_k and cache_v as buffers ***
        # This ensures they are part of the module's state and moved to the correct device.
        # Initialize with zeros. The device will be automatically handled by .to(device) on the model.
        # Use register_buffer for non-parameter tensors that should be part of the module's state.
        self.register_buffer('cache_k', torch.zeros(
            (
                args.batch_size, # Note: batch_size dependency might be inflexible if batch size changes
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
            # Removed .to(args.device) - register_buffer handles device placement with model.to()
        ), persistent=False) # persistent=False avoids saving cache in state_dict

        self.register_buffer('cache_v', torch.zeros(
            (
                args.batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
            # Removed .to(args.device)
        ), persistent=False) # persistent=False avoids saving cache in state_dict

        # Optional: Hook for loading potentially different state dict formats
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Handles loading state dicts with combined QKV weights."""
        # Example hook to split a combined 'wqkv.weight' if found in the state_dict
        combined_key = prefix + "wqkv.weight"
        if combined_key in state_dict:
            wqkv = state_dict.pop(combined_key)
            total_dim = self.n_heads * self.head_dim + 2 * (self.n_kv_heads * self.head_dim)
            if wqkv.shape[0] != total_dim:
                 raise ValueError(
                     f"Expected combined weight shape[0] {total_dim}, but got {wqkv.shape[0]}"
                 )

            # Split the combined weight tensor
            wq, wk, wv = wqkv.split([
                 self.n_heads * self.head_dim,
                 self.n_kv_heads * self.head_dim,
                 self.n_kv_heads * self.head_dim
            ], dim=0)

            # Assign to individual layer weights in the state_dict
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv
            print(f"Successfully split {combined_key} into wq, wk, wv weights.") # Optional logging    
    
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor shape (bsz, seqlen, dim).
            start_pos (int): Starting position for KV cache update.
            freqs_cis (torch.Tensor): Precomputed rotary embedding frequencies.
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor shape (bsz, seqlen, dim).
                - Attention weights tensor shape (bsz, n_heads, seqlen, cache_len).
        """
        bsz, seqlen, _ = x.shape
        

        # --- Debugging: Check Input Gradient ---
        # assert x.requires_grad, "Input 'x' to Attention does not require grad!"

        # Project input to Query, Key, Value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # --- Debugging: Check Projection Gradients ---
        # assert xq.requires_grad, "xq projection lost gradient!"
        # assert xk.requires_grad, "xk projection lost gradient!"
        # assert xv.requires_grad, "xv projection lost gradient!"

        # Reshape Q, K, V for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply Rotary Positional Embeddings if enabled
        if self.use_rope:
            # Ensure freqs_cis covers the current sequence length slice
            current_freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=current_freqs_cis)

        # Apply RMSNorm to Query and Key if enabled
        if self.use_qk_norm:
            # NOTE: This uses the functional rmsnorm without learnable weights.
            # If learnable QK norm is desired, instantiate RMSNorm layers.
            xq = rmsnorm(xq, self.norm_eps)
            xk = rmsnorm(xk, self.norm_eps)

        # Temperature tuning (specific logic for NoPE layers)
        if self.attn_temperature_tuning and not self.use_rope:
            seq_positions = torch.arange(start_pos, start_pos + seqlen, device=xq.device, dtype=torch.float32)
            attn_scales = torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            attn_scales = attn_scales.view(1, seqlen, 1, 1) # Reshape for broadcasting
            xq = xq * attn_scales

        # --- KV Caching ---
        # Buffers `self.cache_k` and `self.cache_v` are already on the correct device.
        # Update the cache at the current positions
        # Ensure start_pos + seqlen does not exceed cache dimension
        if start_pos + seqlen > self.cache_k.shape[1]:
             raise ValueError(
                 f"KV Cache Exceeded: start_pos ({start_pos}) + seqlen ({seqlen}) > cache_seq_len ({self.cache_k.shape[1]})"
             )
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        # Retrieve keys and values up to the current sequence length
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # --- Attention Calculation ---
        # Transpose for attention calculation: (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Repeat KV heads if using Grouped Query Attention (GQA)
        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        # Determine if GQA is enabled based on head counts
        enable_gqa = self.n_local_heads != self.n_local_kv_heads

        # Use PyTorch's optimized scaled dot-product attention
        # Ensure mask has correct shape if provided (e.g., [bsz, 1, seqlen, cache_len])
        attn_output_heads, attn_weights = scaled_dot_product_attention(
            query=xq,
            key=keys,
            value=values,
            attn_mask=mask, # Pass the prepared float mask from LLM
            dropout_p=0.0,  # Set dropout probability (e.g., self.config.dropout if available)
            is_causal=False, # Let the provided mask handle causality/caching
            enable_gqa=enable_gqa # Enable GQA repetition inside the function if needed
        )

        attn_output = attn_output_heads.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Final linear projection
        output = self.wo(attn_output)        

        return output, attn_weights


class FeedForward(nn.Module):
    """
    SwiGLU FeedForward module.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256, # Often used to ensure hidden_dim is multiple of some value
        ffn_dim_multiplier: Optional[float] = None, # Alternative way to set hidden_dim
        bias: bool = False, # Typically False for SwiGLU
    ):
        super().__init__()

        # Calculate hidden dimension based on multiplier or explicit value
        _hidden_dim = hidden_dim
        if ffn_dim_multiplier is not None:
             _hidden_dim = int(ffn_dim_multiplier * dim)
             _hidden_dim = multiple_of * ((_hidden_dim + multiple_of - 1) // multiple_of)

        # SwiGLU-style feedforward: silu(W_gate(x)) * W_value(x)
        self.w_gate = nn.Linear(dim, _hidden_dim, bias=bias)  # Gate projection
        self.w_value = nn.Linear(dim, _hidden_dim, bias=bias) # Value projection
        self.w_out = nn.Linear(_hidden_dim, dim, bias=bias)   # Final output projection

        # Optional: Hook for loading potentially different state dict formats
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Handles loading state dicts with different naming conventions (e.g., mlp.fc1)."""
        # Example hook for compatibility with state dicts using 'mlp.fc1/fc2' naming
        fc1_key = prefix + "mlp.fc1_weight"
        fc2_key = prefix + "mlp.fc2_weight"

        if fc1_key in state_dict and fc2_key in state_dict:
            # Assuming fc1 contains both gate and value weights interleaved or stacked
            # This example assumes they are stacked and need chunking
            w_gate_value = state_dict.pop(fc1_key)
            # Adjust chunking dimension and sizes based on how fc1 was saved
            try:
                w_gate, w_value = w_gate_value.chunk(2, dim=0) # Assuming stacked on dim 0
                state_dict[prefix + "w_gate.weight"] = w_gate
                state_dict[prefix + "w_value.weight"] = w_value
                state_dict[prefix + "w_out.weight"] = state_dict.pop(fc2_key)
                print(f"Successfully split {fc1_key} into w_gate, w_value weights.") # Optional logging
            except RuntimeError as e:
                 error_msgs.append(f"Error splitting {fc1_key} in load_hook: {e}. Check chunk dimension/size.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor shape (bsz, seqlen, dim).

        Returns:
            torch.Tensor: Output tensor shape (bsz, seqlen, dim).
        """
        # Apply SiLU activation to the gate projection
        gate = F.silu(self.w_gate(x))
        # Project input to value
        value = self.w_value(x)
        # Element-wise multiplication and final projection
        output = self.w_out(gate * value)
        return output