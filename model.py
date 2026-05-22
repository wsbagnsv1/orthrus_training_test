"""
OrthrusSmolLM2: SmolLM2-135M with Orthrus dual-view diffusion heads.

Architecture:
  - Frozen AR backbone (SmolLM2) used for prefill to build high-fidelity KV cache.
  - Trainable diffusion attention modules (one per layer) injected alongside AR attention.
  - Both modules share the same KV cache (no extra KV memory).
  - At inference, consensus mechanism guarantees lossless output distribution.

Based on the Orthrus paper (arXiv:2605.12825) and the orthrus/ Qwen3 reference.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb

# ── Triton fused block-attention (in separate file for maintainability) ─────
try:
    from triton_kernels import triton_block_attention, _has_triton
except ImportError:
    _has_triton = False
    def triton_block_attention(*args, **kwargs):
        raise RuntimeError("Triton not available")

try:
    from triton_kernels_fused import fused_triton_attention, _has_triton as _fused_ok
except ImportError:
    _fused_ok = False

# ── flex attention (eager — stable across environments) ─────────────────────
# torch.compile + flex_attention is unstable on this Triton/Windows version.
# At 135M scale the speed difference is negligible. Revisit after Triton upgrade.

def fused_flex_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: BlockMask | None = None,
    score_mod=None,
) -> Tensor:
    """FlexAttention with sparse block mask + optional score_mod for dynamic masking."""
    if mask is not None:
        kernel_options = {
            "sparse_block_size": (int(mask.BLOCK_SIZE[0]), int(mask.BLOCK_SIZE[1])),
        }
    else:
        kernel_options = {}
    return flex_attention(
        q, k, v, block_mask=mask, score_mod=score_mod,
        enable_gqa=True, kernel_options=kernel_options,
    )


# ── dual-pass block mask builder ─────────────────────────────────────────────

def build_dual_pass_block_mask(
    batch_size: int,
    num_heads: int,
    diffusion_length: int,
    ar_len: int,
    block_size: int,
    device: torch.device,
) -> BlockMask:
    """
    Build a STATIC compiled BlockMask — only block-diagonal pattern.
    Causal limits are applied dynamically via score_mod (no recompilation).

    Query positions [0, diffusion_length) = masked token positions.
    Key positions [0, ar_len) = AR context (always accessible here).
    Key positions [ar_len, ...) = diffusion block K/V (block-diagonal).
    """
    def mask_fn(b, h, q_idx, kv_idx):
        is_kv_ar = kv_idx < ar_len
        draft_kv_idx = kv_idx - ar_len
        q_block_id = q_idx // block_size
        kv_block_id = draft_kv_idx // block_size
        valid_diffusion = (~is_kv_ar) & (q_block_id == kv_block_id)
        return is_kv_ar | valid_diffusion  # static: all AR + block-diagonal self

    return create_block_mask(
        mask_fn,
        B=batch_size, H=num_heads,
        Q_LEN=diffusion_length, KV_LEN=ar_len + diffusion_length,
        BLOCK_SIZE=128,
        device=device,
        _compile=True,
    )


# ── Diffusion Attention Module (one per layer) ───────────────────────────────

class DiffusionAttention(nn.Module):
    """
    Lightweight diffusion attention injected alongside a frozen AR attention layer.

    Has independent Q/K/V/O projections (copy-initialized from AR weights)
    and independent Q/K RMS norms. Reads from the shared AR KV cache and
    appends its own K/V for the diffusion block positions.
    """
    def __init__(self, ar_attn_layer: nn.Module, config):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = hidden_size // num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        # Copy-initialize projections from frozen AR attention
        self.q_proj = copy.deepcopy(ar_attn_layer.q_proj)
        self.k_proj = copy.deepcopy(ar_attn_layer.k_proj)
        self.v_proj = copy.deepcopy(ar_attn_layer.v_proj)
        self.o_proj = copy.deepcopy(ar_attn_layer.o_proj)

        # Per-head Q/K RMS norms (bf16 weights — match model dtype)
        self.q_norm = LlamaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.q_norm.weight.data = self.q_norm.weight.data.to(torch.bfloat16)
        self.k_norm.weight.data = self.k_norm.weight.data.to(torch.bfloat16)

        for p in self.parameters():
            p.requires_grad = True

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        ar_key_cache: Tensor,
        ar_val_cache: Tensor,
        ar_seq_len: int,
        flex_block_mask: BlockMask | None,
        **kwargs,  # ignore extra args (causal_limit, block_size)
    ) -> Tensor:
        B, diff_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(B, diff_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, diff_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, diff_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Truncate AR cache and concatenate
        k_ar = ar_key_cache[:, :, :ar_seq_len, :]
        v_ar = ar_val_cache[:, :, :ar_seq_len, :]
        k_full = torch.cat([k_ar, k], dim=2)
        v_full = torch.cat([v_ar, v], dim=2)

        if flex_block_mask is not None:
            attn_out = fused_flex_attention(q, k_full, v_full, mask=flex_block_mask, score_mod=kwargs.get('score_mod'))
            attn_out = attn_out.transpose(1, 2)
        else:
            if self.num_key_value_groups > 1:
                k_full = k_full.repeat_interleave(self.num_key_value_groups, dim=1)
                v_full = v_full.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
            attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(B, diff_len, -1)
        return self.o_proj(attn_out)


# ── Triton Diffusion Attention (subclass, ~6x faster) ───────────────────────

if _has_triton:
    class TritonDiffusionAttention(DiffusionAttention):
        """Override that uses fused Triton kernels instead of flex_attention."""
        def forward(
            self,
            hidden_states: Tensor,
            position_embeddings: Tuple[Tensor, Tensor],
            ar_key_cache: Tensor,
            ar_val_cache: Tensor,
            ar_seq_len: int,
            flex_block_mask: BlockMask | None = None,
            causal_limit: Tensor | None = None,
            block_size: int = 32,
        ) -> Tensor:
            B, diff_len, _ = hidden_states.shape
            cos, sin = position_embeddings

            q = self.q_proj(hidden_states).view(B, diff_len, self.num_heads, self.head_dim)
            q = self.q_norm(q).transpose(1, 2).contiguous()
            k = self.k_proj(hidden_states).view(B, diff_len, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k).transpose(1, 2).contiguous()
            v = self.v_proj(hidden_states).view(B, diff_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            ngroups = self.num_key_value_groups

            k_ar = ar_key_cache[:, :, :ar_seq_len, :]
            v_ar = ar_val_cache[:, :, :ar_seq_len, :]

            n_blocks = diff_len // block_size
            attn_flat = triton_block_attention(
                q.reshape(B * self.num_heads, diff_len, self.head_dim),
                k.reshape(B * self.num_kv_heads, diff_len, self.head_dim),
                v.reshape(B * self.num_kv_heads, diff_len, self.head_dim),
                k_ar.reshape(B * self.num_kv_heads, ar_seq_len, self.head_dim),
                v_ar.reshape(B * self.num_kv_heads, ar_seq_len, self.head_dim),
                causal_limit,
                B, self.num_heads, self.num_kv_heads, ngroups, 4, n_blocks, block_size, ar_seq_len, self.head_dim,
            ).view(B, self.num_heads, diff_len, self.head_dim)
            attn_out = attn_flat.transpose(1, 2)

            attn_out = attn_out.reshape(B, diff_len, -1)
            return self.o_proj(attn_out)


# ── Fused Triton (RMSNorm+RoPE in kernel) ──────────────────────────────────

if _fused_ok:
    class TritonDiffusionAttentionFused(TritonDiffusionAttention):
        def forward(self, hidden_states, position_embeddings, ar_key_cache,
                    ar_val_cache, ar_seq_len, flex_block_mask=None,
                    causal_limit=None, block_size=32):
            B, diff_len, _ = hidden_states.shape
            cos, sin = position_embeddings
            q_raw = self.q_proj(hidden_states).view(B*self.num_heads, diff_len, self.head_dim)
            k_raw = self.k_proj(hidden_states).view(B*self.num_kv_heads, diff_len, self.head_dim)
            v = self.v_proj(hidden_states).view(B*self.num_kv_heads, diff_len, self.head_dim)
            nqw = self.q_norm.weight.repeat(self.num_heads)
            nkw = self.k_norm.weight.repeat(self.num_kv_heads)
            cos_f = cos[0].contiguous()
            sin_f = sin[0].contiguous()
            k_ar = ar_key_cache[:,:,:ar_seq_len,:].reshape(B*self.num_kv_heads, ar_seq_len, self.head_dim)
            v_ar = ar_val_cache[:,:,:ar_seq_len,:].reshape(B*self.num_kv_heads, ar_seq_len, self.head_dim)
            n_blocks = diff_len // block_size
            ng = self.num_key_value_groups
            attn = fused_triton_attention(q_raw, k_raw, v, nqw, nkw, cos_f, sin_f, k_ar, v_ar, causal_limit,
                B, self.num_heads, self.num_kv_heads, ng, 4, n_blocks, block_size, ar_seq_len, self.head_dim)
            attn = attn.view(B, self.num_heads, diff_len, self.head_dim).transpose(1,2).reshape(B, diff_len, -1)
            return self.o_proj(attn)


# ── Full Orthrus SmolLM2 Model ───────────────────────────────────────────────

def _run_diffusion_layers(
    hidden_states: Tensor,
    layers,
    diff_heads,
    position_embeddings,
    ar_past_key_values,
    ar_seq_len: int,
    flex_block_mask,
    causal_limit: Tensor,
    K: int,
    start_idx: int,
    score_mod=None,
) -> Tensor:
    """Process a group of layers (used by gradient checkpointing)."""
    for local_idx, (layer, diff_attn) in enumerate(zip(layers, diff_heads)):
        layer_idx = start_idx + local_idx
        normed = layer.input_layernorm(hidden_states)
        ar_keys = ar_past_key_values.layers[layer_idx].keys
        ar_vals = ar_past_key_values.layers[layer_idx].values
        diff_out = diff_attn(
            hidden_states=normed, position_embeddings=position_embeddings,
            ar_key_cache=ar_keys, ar_val_cache=ar_vals,
            ar_seq_len=ar_seq_len, flex_block_mask=flex_block_mask,
            causal_limit=causal_limit, block_size=K, score_mod=score_mod,
        )
        hidden_states = hidden_states + diff_out
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
    return hidden_states


class OrthrusSmolLM2(nn.Module):
    """
    Orthrus model wrapping SmolLM2-135M.

    Freezes the base AR model and injects trainable diffusion attention
    modules at every layer.
    """
    def __init__(
        self,
        base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        block_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
        use_triton_attention: bool = False,
        checkpoint_every: int = 0,  # gradient checkpointing: 0=off, N=every N layers
    ):
        super().__init__()
        self.block_size = block_size
        self.use_triton = use_triton_attention and _has_triton
        self.checkpoint_every = checkpoint_every

        attn_impl = "sdpa"
        print(f"  AR backbone attention: {attn_impl}")
        if self.use_triton:
            if _fused_ok:
                attn_cls = TritonDiffusionAttentionFused
                print(f"  Diffusion attention: triton fused (RMSNorm+RoPE in kernel)")
            else:
                attn_cls = TritonDiffusionAttention
                print(f"  Diffusion attention: triton fused (~6x faster)")
        else:
            attn_cls = DiffusionAttention
            print(f"  Diffusion attention: flex_attention")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, attn_implementation=attn_impl,
        )
        self.config = self.base_model.config
        for p in self.base_model.parameters():
            p.requires_grad = False

        attn_cls = DiffusionAttention  # default, overridden below
        if self.use_triton:
            if _fused_ok:
                attn_cls = TritonDiffusionAttentionFused
                print(f"  Diffusion attention: triton fused (RMSNorm+RoPE in kernel)")
            else:
                attn_cls = TritonDiffusionAttention
                print(f"  Diffusion attention: triton (~6x faster)")
        else:
            print(f"  Diffusion attention: flex_attention")
        self.diffusion_heads = nn.ModuleList([
            attn_cls(
                ar_attn_layer=self.base_model.model.layers[i].self_attn,
                config=self.config,
            )
            for i in range(self.config.num_hidden_layers)
        ])

        self.embed_tokens = self.base_model.model.embed_tokens
        self.norm = self.base_model.model.norm
        self.lm_head = self.base_model.lm_head
        self.rotary_emb = self.base_model.model.rotary_emb

        self.trainable_params = sum(
            p.numel() for p in self.diffusion_heads.parameters()
        )

        # ── suppress flex_attention eager warning (intentional — stable numerics) ──
        import warnings
        warnings.filterwarnings(
            "ignore", message=".*flex_attention called without torch.compile.*",
        )
        warnings.filterwarnings(
            "ignore", message=".*Not enough SMs.*",
        )

    def compile_diffusion_heads(self):
        """torch.compile each attention module.
        Skips Triton heads — autograd function blocks inductor fusion."""
        import torch
        torch._dynamo.config.recompile_limit = 128
        for i in range(len(self.diffusion_heads)):
            if _has_triton and isinstance(self.diffusion_heads[i], TritonDiffusionAttention):
                continue
            self.diffusion_heads[i] = torch.compile(
                self.diffusion_heads[i],
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
        print(f"  ✓ Compiled {len(self.diffusion_heads)} diffusion heads")

    # ------------------------------------------------------------------
    # AR prefill
    # ------------------------------------------------------------------
    @torch.compile(mode="default", fullgraph=False, dynamic=True)
    @torch.no_grad()
    def forward_ar_prefill(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tuple[DynamicCache, Tensor]:
        """Run frozen AR backbone. Returns (kv_cache, last_hidden_state)."""
        outputs = self.base_model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=True, output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state.detach()
        kv = outputs.past_key_values
        del outputs
        return kv, hidden

    # ------------------------------------------------------------------
    # Diffusion forward
    # ------------------------------------------------------------------
    def forward_diffusion(
        self,
        diff_input_ids: Tensor,
        ar_past_key_values: DynamicCache,
        ar_seq_len: int,
        causal_limit: Tensor | None = None,
        return_hidden: bool = False,
        diff_position_ids: Tensor | None = None,
        use_flex: bool = True,
    ) -> Tensor:
        B = diff_input_ids.shape[0]
        K = self.block_size
        diff_len = diff_input_ids.shape[1]
        device = diff_input_ids.device

        hidden_states = self.embed_tokens(diff_input_ids)

        if diff_position_ids is not None:
            position_ids = diff_position_ids
        else:
            position_ids = torch.arange(
                ar_seq_len, ar_seq_len + diff_len, device=device
        ).unsqueeze(0).expand(B, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        if causal_limit is None:
            causal_limit = self._default_causal_limit(
                diff_len, ar_seq_len, K, device
            ).expand(B, -1)

        # Static block mask (cached once, zero recompilations) — skip for plain SDPA
        flex_block_mask = None
        causal_score_mod = None
        if use_flex:
            cache_key = (B, diff_len, ar_seq_len, K)
            if not hasattr(self, '_mask_cache'):
                self._mask_cache = {}
            if cache_key not in self._mask_cache:
                self._mask_cache[cache_key] = build_dual_pass_block_mask(
                    batch_size=B,
                    num_heads=self.config.num_attention_heads,
                    diffusion_length=diff_len,
                    ar_len=ar_seq_len,
                    block_size=K,
                    device=device,
                )
            flex_block_mask = self._mask_cache[cache_key]

            def causal_score_mod(score, b, h, q_idx, kv_idx):
                is_ar = kv_idx < ar_seq_len
                valid_ar = kv_idx <= causal_limit[b, q_idx]
                return torch.where(is_ar & (~valid_ar), float('-inf'), score)

        layers = self.base_model.model.layers
        ckpt = self.checkpoint_every
        if ckpt > 0 and ckpt < len(layers):
            # Gradient checkpointing: group layers, store only group boundaries
            chk = torch.utils.checkpoint.checkpoint
            for grp_start in range(0, len(layers), ckpt):
                grp_end = min(grp_start + ckpt, len(layers))
                hidden_states = chk(
                    _run_diffusion_layers,
                    hidden_states,
                    layers[grp_start:grp_end],
                    self.diffusion_heads[grp_start:grp_end],
                    position_embeddings,
                    ar_past_key_values, ar_seq_len, flex_block_mask, causal_limit, K,
                    grp_start, causal_score_mod,
                    use_reentrant=False,
                )
        else:
            for layer_idx, (layer, diff_attn) in enumerate(
                zip(layers, self.diffusion_heads)
            ):
                normed = layer.input_layernorm(hidden_states)
                ar_keys = ar_past_key_values.layers[layer_idx].keys
                ar_vals = ar_past_key_values.layers[layer_idx].values
                diff_out = diff_attn(
                    hidden_states=normed, position_embeddings=position_embeddings,
                    ar_key_cache=ar_keys, ar_val_cache=ar_vals,
                    ar_seq_len=ar_seq_len, flex_block_mask=flex_block_mask,
                    causal_limit=causal_limit, block_size=K, score_mod=causal_score_mod,
                )
                hidden_states = hidden_states + diff_out
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

        del flex_block_mask  # free compiled mask after all layers

        hidden_states = self.norm(hidden_states)
        if return_hidden:
            return hidden_states
        return self.lm_head(hidden_states)

    # ------------------------------------------------------------------
    # Convenience forward (for eval/generate)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        is_diffusion_pass: bool = False,
        ar_past_key_values: DynamicCache | None = None,
        ar_seq_len: int | None = None,
        causal_limit: Tensor | None = None,
        return_hidden: bool = False,
        diff_position_ids: Tensor | None = None,
        use_flex: bool = True,
    ) -> Tensor:
        if not is_diffusion_pass:
            _, hidden = self.forward_ar_prefill(input_ids, attention_mask)
            return self.lm_head(hidden)
        else:
            return self.forward_diffusion(
                diff_input_ids=input_ids,
                ar_past_key_values=ar_past_key_values,
                ar_seq_len=ar_seq_len,
                causal_limit=causal_limit,
                return_hidden=return_hidden,
                diff_position_ids=diff_position_ids,
                use_flex=use_flex,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _default_causal_limit(
        self, diff_len: int, ar_seq_len: int, K: int, device: torch.device
    ) -> Tensor:
        limits = torch.zeros(diff_len, dtype=torch.long, device=device)
        for b in range(diff_len // K):
            anchor = ar_seq_len - diff_len + b * K
            limits[b * K: (b + 1) * K] = anchor - 1
        return limits.unsqueeze(0)

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.diffusion_heads.parameters() if p.requires_grad]
