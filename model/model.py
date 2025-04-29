
import math
from typing import Any, Dict, List, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn

from model.args import ModelArgs
from model.datatypes import TransformerInput, TransformerOutput
from model.layers import FeedForward, Attention, precompute_freqs_cis,RMSNorm
from model.moe import MoE
from vision.embedding import VisionEmbeddings


# tokens (0, K), (K, 2K), (2K, 3K) attend to each other when doing local chunked attention
# in the iRoPE architecture
def create_chunked_attention_mask(seq_len: int, attention_chunk_size: int, device: torch.device) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // attention_chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // attention_chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    return mask.to(device)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads if args.head_dim is None else args.head_dim

        self.is_nope_layer = args.nope_layer_interval is not None and (layer_id + 1) % args.nope_layer_interval == 0

        use_rope = not self.is_nope_layer
        use_qk_norm = args.use_qk_norm and not self.is_nope_layer

        # Attention layer now returns output AND weights
        self.attention = Attention(args, use_rope=use_rope, use_qk_norm=use_qk_norm)

        # FeedForward/MoE part (Unchanged logic)
        if args.moe_args and (layer_id + 1) % args.moe_args.interleave_moe_layer_step == 0:
            self.feed_forward = MoE(
                dim=args.dim,
                hidden_dim=int(args.ffn_exp * args.dim), # Use ffn_exp if provided
                ffn_dim_multiplier=args.ffn_dim_multiplier, # Allow override
                multiple_of=args.multiple_of,
                moe_args=args.moe_args,
            )
        else:
            # Standard Llama FFN hidden dim calculation
            hidden_dim = int(4 * args.dim)
            hidden_dim = int(2 * hidden_dim / 3)
            # Apply multiplier if specified
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            # Make hidden dim multiple of `multiple_of`
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=hidden_dim,
                # Pass other relevant args if FeedForward uses them
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Load hook (Unchanged)
        self._register_load_state_dict_pre_hook(self.load_hook)

    # load_hook method (Unchanged)
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
        # Renaming logic for norm weights
        norm_key_old = prefix + "attention.wqkv.layer_norm_weight"
        norm_key_alt = prefix + "attention.norm.weight"
        norm_key_target = prefix + "attention_norm.weight"
        if norm_key_old in state_dict:
            state_dict[norm_key_target] = state_dict.pop(norm_key_old)
        elif norm_key_alt in state_dict:
             state_dict[norm_key_target] = state_dict.pop(norm_key_alt)

        ffn_norm_key_old = prefix + "feed_forward.mlp.layer_norm_weight"
        ffn_norm_key_alt = prefix + "feed_forward.norm.weight"
        ffn_norm_key_target = prefix + "ffn_norm.weight"
        if ffn_norm_key_old in state_dict:
            state_dict[ffn_norm_key_target] = state_dict.pop(ffn_norm_key_old)
        elif ffn_norm_key_alt in state_dict:
            state_dict[ffn_norm_key_target] = state_dict.pop(ffn_norm_key_alt)

        # Remove extra state keys if they exist
        for k in (
            "feed_forward.experts.mlp",
            "feed_forward.mlp_shared",
            "attention.wo",
            "attention.wqkv", # Check if combined weights might have extra state
            "attention.wq",   # Or individual weights
            "attention.wk",
            "attention.wv",
        ):
            extra_state_key = prefix + k + "._extra_state"
            if extra_state_key in state_dict:
                state_dict.pop(extra_state_key)

    # --- MODIFIED forward method (Fixed TypeError) ---
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        global_attn_mask: Optional[torch.Tensor],
        local_attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Return hidden state AND attention weights
        """
        Forward pass for the Transformer Block. Now returns attention weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output hidden state tensor shape (bsz, seqlen, dim).
                - Attention weights tensor shape (bsz, n_heads, seqlen, cache_len).
        """
        # Determine mask based on layer type (NoPE or RoPE) and chunking
        if self.is_nope_layer or local_attn_mask is None:
            mask = global_attn_mask
        else:
            mask = local_attn_mask

        # --- Attention Sub-layer ---
        # Apply pre-normalization
        normed_x = self.attention_norm(x)
        # *** FIX: Unpack the tuple returned by self.attention ***
        attn_output, attn_weights = self.attention(normed_x, start_pos, freqs_cis, mask)
        # Residual connection using only the attention output
        h = x + attn_output

        # --- FeedForward Sub-layer ---
        # Apply pre-normalization
        normed_h = self.ffn_norm(h)
        # Get feed-forward output
        ff_output = self.feed_forward(normed_h)
        # Residual connection
        out = h + ff_output

        return out, attn_weights # Return hidden state and weights from this block


# --- LLM Class (MODIFIED) ---
class LLM(nn.Module):
    def __init__(self, args: ModelArgs, **kwargs) -> None:
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Calculate head_dim correctly
        self.head_dim = args.dim // args.n_heads if args.head_dim is None else args.head_dim
        if self.head_dim * args.n_heads != args.dim:
             # Warn if head_dim * n_heads doesn't match dim, might happen if head_dim is explicitly set
             print(f"Warning: head_dim ({self.head_dim}) * n_heads ({args.n_heads}) != dim ({args.dim})")

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            # Use the calculated head_dim for RoPE dimension
            self.head_dim,
            args.max_seq_len * 2, # Precompute for longer sequences if needed
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
            args.rope_high_freq_factor,
        )

        # Vision components
        vision_args = self.args.vision_args
        self.vision_embeddings = None
        self.vision_projection = None
        if VisionEmbeddings is not None and vision_args:
            try:
                # Ensure VisionEmbeddings and projection layer are correctly initialized
                self.vision_embeddings = VisionEmbeddings(vision_args)
                self.vision_projection = nn.Linear(
                    vision_args.output_dim,
                    args.dim,
                    bias=False, # Typically no bias for projection
                )
            except Exception as e:
                print(f"Warning: Failed to initialize vision components: {e}")
                self.vision_embeddings = None
                self.vision_projection = None

        # Register hook for potential state_dict modifications
        self._register_load_state_dict_pre_hook(self.load_hook)


    # load_hook method (Unchanged)
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
        # Remove precomputed freqs from state_dict if loading older checkpoints
        if prefix + "rope.freqs" in state_dict:
            state_dict.pop(prefix + "rope.freqs")
        # Handle potential weight name mismatches for output layer if needed
        output_key_old = prefix + "output.weight"
        output_key_new = prefix + "lm_head.weight" # Example alternative name
        if output_key_new in state_dict and output_key_old not in state_dict:
             state_dict[output_key_old] = state_dict.pop(output_key_new)
    
    def _forward(self, input_data: TransformerInput, inference: bool = True, return_attn_weights: bool = True) -> TransformerOutput:
        """
        Forward pass for the LLM. Can optionally return attention weights.

        Args:
            input_data (TransformerInput): Contains tokens, position, optional image embedding.
            inference (bool): If True, run in torch.inference_mode().
            return_attn_weights (bool): If True, return attention weights from all layers.

        Returns:
            TransformerOutput: Dataclass containing logits and optional attention weights.
        """
        # Determine if we are in training mode (requires grad) or inference mode
        is_training = self.training and not inference

        if not is_training:
            # Use inference_mode for efficiency when not training
            with torch.inference_mode():
                return self._forward(input_data, return_attn_weights)
        else:
            # Ensure gradients are enabled during training
            # torch.is_grad_enabled() should be True here if model.train() was called
            return self._forward(input_data, return_attn_weights)


    
    def forward(self, model_input: TransformerInput, return_attn_weights: bool = False) -> TransformerOutput:
        tokens = model_input.tokens
        start_pos = model_input.tokens_position
        # Allow start_pos to be a tensor for per-item positioning if needed in the future
        # For now, assert it's an int as the code expects.
        if isinstance(start_pos, torch.Tensor):
             if start_pos.numel() == 1:
                 start_pos = start_pos.item() # Convert single-element tensor to int
             else:
                 # If start_pos is a tensor with multiple elements, the current logic
                 # for slicing freqs_cis and updating KV cache needs adjustment.
                 raise NotImplementedError("Batch-varying start_pos is not fully supported yet.")
        elif not isinstance(start_pos, int):
             raise TypeError(f"Expected start_pos to be int or single-element tensor, got {type(start_pos)}")


        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Vision embedding integration
        # Check if vision components are initialized and input data has image embedding
        if self.vision_embeddings and self.vision_projection and hasattr(model_input, 'image_embedding'):
             image_embedding = model_input.image_embedding
             if image_embedding is not None and image_embedding.embedding is not None and image_embedding.mask is not None:
                # Ensure mask has the correct shape: (bsz, seqlen, 1) for broadcasting
                mask = image_embedding.mask
                if mask.ndim == 2: # (bsz, seqlen) -> (bsz, seqlen, 1)
                    mask = mask.unsqueeze(-1)
                # Ensure mask is boolean and on the correct device
                mask = mask.to(device=h.device, dtype=torch.bool, non_blocking=True)

                # Project image embedding and move to device
                img_emb = image_embedding.embedding.to(device=h.device, non_blocking=True)
                h_image = self.vision_projection(img_emb)

                # Combine based on mask: h = text_part * (~mask) + image_part * mask
                # Ensure h_image is broadcastable if necessary (e.g., if img embedding seq len != text seq len)
                # This assumes image embedding provides one vector per token position where mask is True
                if h_image.shape[1] == mask.shape[1]: # Check sequence length alignment
                    h = torch.where(mask, h_image, h)
                else:
                     print(f"Warning: Image embedding sequence length ({h_image.shape[1]}) "
                           f"differs from mask/token sequence length ({mask.shape[1]}). Skipping merge.")


        # Ensure freqs_cis is on the correct device
        # Move it once before the loop if it's not already there
        if self.freqs_cis.device != h.device:
             self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis_for_layers = self.freqs_cis # Pass the precomputed tensor

        # Prepare attention masks
        global_attn_mask, local_attn_mask = None, None
        current_cache_len = start_pos + seqlen # Total length including cache
        if seqlen > 1:
            # --- Global Causal Mask (for current query tokens attending to cache) ---
            # Mask shape should be (seqlen, cache_len)
            global_attn_mask = torch.full((seqlen, current_cache_len), float("-inf"), device=tokens.device)
            # Create the causal part: query i can attend to key j where j <= start_pos + i
            # Corrected causal mask logic: query i attends to keys up to index start_pos + i
            row_indices = torch.arange(seqlen, device=tokens.device).unsqueeze(1)
            col_indices = torch.arange(current_cache_len, device=tokens.device).unsqueeze(0)
            allowed_connections = col_indices <= (row_indices + start_pos)
            global_attn_mask[allowed_connections] = 0.0


            # --- Chunked Local Mask (if applicable) ---
            # This mask usually applies only to the current sequence block, not the cache.
            # The create_chunked_attention_mask function needs to be aware of start_pos
            # or the mask needs to be combined carefully with the global causal mask.
            # For simplicity, let's assume local mask is only used when global is not.
            # The logic in TransformerBlock already handles choosing between them.
            if chunk_size := self.args.attention_chunk_size:
                # Create local mask just for the current seqlen x seqlen block
                local_mask_block = create_chunked_attention_mask(seqlen, chunk_size, tokens.device)
                # Pad it to match the cache length if needed, or handle in Attention layer
                # For now, let's pass the block mask, assuming Attention layer handles cache interaction
                local_attn_mask = local_mask_block # Shape (seqlen, seqlen)


        # List to store attention weights if requested
        all_attn_weights = [] if return_attn_weights else None

        # Process layers
        for layer in self.layers:
            # Layer now returns hidden state AND attention weights
            # Pass the appropriate mask (global or local) based on layer logic
            # The layer itself will select based on is_nope_layer etc.
            h, layer_attn_weights = layer(h, start_pos, freqs_cis_for_layers, global_attn_mask, local_attn_mask)
            if return_attn_weights:
                # Detach weights to prevent storing computation graph if only needed for viz
                # Stack weights into a single tensor (layers, bsz, heads, seq, cache)
                all_attn_weights.append(layer_attn_weights.detach())

        # Final normalization and output projection
        h = self.norm(h)
        output_logits = self.output(h).float() # Ensure output is float

        # --- FIX: Return TransformerOutput dataclass ---
        final_attn_weights = None
        if return_attn_weights and all_attn_weights:
            # Stack the list of weights into a single tensor along a new 'layer' dimension
            # Resulting shape: (n_layers, bsz, n_heads, seqlen, cache_len)
            try:
                final_attn_weights = torch.stack(all_attn_weights, dim=0)
            except Exception as e:
                 print(f"Warning: Could not stack attention weights: {e}. Returning None for attn_weights.")
                 final_attn_weights = None # Ensure it's None if stacking fails


        # Instantiate and return the dataclass
        return TransformerOutput(
            logits=output_logits,
            attn_weights=final_attn_weights # Assign the stacked tensor or None
        )

