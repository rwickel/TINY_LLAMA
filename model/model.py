
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

        self.attention = Attention(args, use_rope=use_rope, use_qk_norm=use_qk_norm)

        if args.moe_args and (layer_id + 1) % args.moe_args.interleave_moe_layer_step == 0:
            self.feed_forward = MoE(
                dim=args.dim,
                hidden_dim=int(args.ffn_exp * args.dim),
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                multiple_of=args.multiple_of,
                moe_args=args.moe_args,
            )
        else:
            hidden_dim = int(4 * args.dim)
            hidden_dim = int(2 * hidden_dim / 3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=hidden_dim,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

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
        if prefix + "attention.wqkv.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(prefix + "attention.wqkv.layer_norm_weight")

        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.mlp.layer_norm_weight")
        elif prefix + "feed_forward.norm.weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.norm.weight")

        for k in (
            "feed_forward.experts.mlp",
            "feed_forward.mlp_shared",
            "attention.wo",
            "attention.wqkv",
        ):
            if prefix + k + "._extra_state" in state_dict:
                state_dict.pop(prefix + k + "._extra_state")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        global_attn_mask: Optional[torch.Tensor],
        local_attn_mask: Optional[torch.Tensor],
    ):
        # The iRoPE architecture uses global attention mask for NoPE layers or
        # if chunked local attention is not used
        if self.is_nope_layer or local_attn_mask is None:
            mask = global_attn_mask
        else:
            mask = local_attn_mask

        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLM(nn.Module):
    def __init__(self, args: ModelArgs, **kwargs) -> None:
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        #self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim, init_method=lambda x: x) # multi-GPU aproach        
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
            args.rope_high_freq_factor,
        )
        vision_args = self.args.vision_args
        if vision_args:
            
            self.vision_embeddings = VisionEmbeddings(vision_args)
            self.vision_projection = nn.Linear(
                vision_args.output_dim,
                args.dim,
                bias=False,
            )
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
        if prefix + "rope.freqs" in state_dict:
            state_dict.pop(prefix + "rope.freqs")
            
    def forward(self, model_input: TransformerInput) -> TransformerOutput:
        tokens = model_input.tokens
        start_pos = model_input.tokens_position
        assert isinstance(start_pos, int), (
            "This implementation does not support different start positions per batch item"
        )

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if image_embedding := model_input.image_embedding:
            h_image = self.vision_projection(image_embedding.embedding)
            h = h * ~image_embedding.mask + h_image * image_embedding.mask

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        global_attn_mask, local_attn_mask = None, None
        if seqlen > 1:
            global_attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            global_attn_mask = torch.triu(global_attn_mask, diagonal=1).type_as(h)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if global_attn_mask.device.type == torch.device("mps").type:
                global_attn_mask = torch.nan_to_num(global_attn_mask, nan=0.0)

            if chunk_size := self.args.attention_chunk_size:
                local_attn_mask = create_chunked_attention_mask(seqlen, chunk_size, tokens.device)

        for layer in self.layers:
            h, attn_weights = layer(h, start_pos, freqs_cis, global_attn_mask, local_attn_mask)
        h = self.norm(h)
        output = self.output(h).float()

        return TransformerOutput(logits=output)


