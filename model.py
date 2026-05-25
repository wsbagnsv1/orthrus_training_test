"""
OrthrusQwen35: Dual-view diffusion for Qwen3.5-0.8B.

Freezes the Qwen3.5-0.8B backbone and injects trainable diffusion heads
at every layer. Full-attention layers get cloned Qwen3_5Attention heads;
linear-attention layers get cloned Qwen3_5GatedDeltaNet heads.
Both share the AR prefill cache with zero extra KV memory.

Based on orthrus_smollm2/model.py, adapted for Qwen3.5's mixed architecture.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# ── Kernel patches (same as orthrus_Qwen3.5-0.8B/model.py) ──────────────────
try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35_mod

    if _qwen35_mod.FusedRMSNormGated is not None:
        _FNG = _qwen35_mod.FusedRMSNormGated

        class _FusedRMSNormGatedFixed(_FNG):
            def __init__(self, hidden_size, eps=1e-6, activation='silu',
                         device=None, dtype=None):
                if isinstance(dtype, str):
                    dtype = getattr(torch, dtype)
                super().__init__(hidden_size, eps=eps, activation=activation,
                                 device=device, dtype=dtype)
        _qwen35_mod.FusedRMSNormGated = _FusedRMSNormGatedFixed

    def _torch_causal_conv1d(x, weight, bias, activation, seq_idx=None):
        C = x.shape[1]
        if weight.ndim == 2:
            weight = weight.unsqueeze(1)
        K = weight.shape[-1]
        out = F.conv1d(x, weight, bias, padding=K - 1, groups=C)
        out = out[:, :, :x.shape[-1]]
        return F.silu(out)

    _qwen35_mod.causal_conv1d_fn = _torch_causal_conv1d
    _qwen35_mod.is_fast_path_available = True
except Exception:
    pass

# Import Qwen3.5 internals AFTER the patch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

# ── FlexAttention helpers ───────────────────────────────────────────────────

def _flex_fa_score_mod_template(
    query, key, value, causal_limit, ar_seq_len, block_size, block_mask=None,
):
    """Leaf-compiled flex_attention: AR causal_limit + per-block isolation via score_mod."""
    def score_mod(score, b, h, q_idx, kv_idx):
        # AR mask: only allow AR positions up to causal_limit[b, q_idx]
        is_ar = kv_idx < ar_seq_len
        ar_masked = is_ar & (kv_idx > causal_limit[b, q_idx])
        # Block isolation: different block (diffusion tokens are bidirectional within their block)
        q_block = q_idx // block_size
        kv_diff = kv_idx - ar_seq_len
        is_diff_kv = kv_diff >= 0
        kv_block = kv_diff // block_size
        diff_masked = is_diff_kv & (q_block != kv_block)
        return torch.where(ar_masked | diff_masked, float('-inf'), score)
        
    # Reduce Triton tile sizes and pipeline stages to fit in 100KB SM limit
    kernel_options = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_M1": 32,
        "BLOCK_N1": 32,
        "BLOCK_M2": 32,
        "BLOCK_N2": 32,
        "num_stages": 1,
    }
    return flex_attention(
        query, key, value, score_mod=score_mod,
        kernel_options=kernel_options, enable_gqa=True, block_mask=block_mask,
    )

# Try to compile; fall back to eager flex_attention if compile fails
try:
    compiled_flex_fa = torch.compile(
        _flex_fa_score_mod_template, fullgraph=False, dynamic=True,
    )
except Exception:
    compiled_flex_fa = _flex_fa_score_mod_template


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion Heads
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionFullAttention(nn.Module):
    """Clone of Qwen3_5Attention that shares the AR KV cache."""

    def __init__(self, ar_attn: Qwen3_5Attention, layer_idx: int):
        super().__init__()
        config = ar_attn.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = ar_attn.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout

        # Copy-initialize projections
        self.q_proj = copy.deepcopy(ar_attn.q_proj)
        self.k_proj = copy.deepcopy(ar_attn.k_proj)
        self.v_proj = copy.deepcopy(ar_attn.v_proj)
        self.o_proj = copy.deepcopy(ar_attn.o_proj)
        self.q_norm = copy.deepcopy(ar_attn.q_norm)
        self.k_norm = copy.deepcopy(ar_attn.k_norm)

        for p in self.parameters():
            p.requires_grad = True

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        k_ar: Tensor,
        v_ar: Tensor,
        causal_limit: Tensor | None = None,
        block_size: int = 32,
    ) -> Tensor:
        B, diff_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # QKV projection (Qwen3.5 uses double head_dim for Q — gate split)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Retrieve dynamic seq len to avoid torch.compile specializing on a scalar integer
        ar_seq_len = k_ar.shape[2]

        # Concatenate with AR KV cache (pre-sliced by caller)
        k_full = torch.cat([k_ar, key_states], dim=2)
        v_full = torch.cat([v_ar, value_states], dim=2)

        # Note: flex_attention and SDPA both natively handle GQA broadcasting.
        # No need for repeat_interleave — saves VRAM by avoiding KV cache duplication.

        # Attention: compiled flex_attention (or SDPA fallback)
        if causal_limit is not None and compiled_flex_fa is not None:
            cl = causal_limit.contiguous()
            attn_out = compiled_flex_fa(
                query_states, k_full, v_full, cl, ar_seq_len, block_size, None,
            )
            attn_out = attn_out.transpose(1, 2)
        elif causal_limit is not None:
            # Build causal mask: allow AR tokens up to causal_limit per block
            Bq, _, q_len, _ = query_states.shape
            kv_len = k_full.shape[2]
            cl = causal_limit[:, :q_len]  # [B, tokens], always 2D
            # Build mask functionally (Inductor fuses into FA kernel)
            zero_val = torch.tensor(0.0, dtype=query_states.dtype, device=query_states.device)
            inf_val = torch.tensor(float('-inf'), dtype=query_states.dtype, device=query_states.device)
            ar_indices = torch.arange(ar_seq_len, device=query_states.device).view(1, 1, 1, -1)
            cl_expanded = cl.view(Bq, 1, q_len, 1)  # [Bq, 1, q_len, 1]
            ar_mask = torch.where(ar_indices <= cl_expanded, zero_val, inf_val)

            # Block-diagonal for diffusion tokens (bidirectional within block)
            diff_indices = torch.arange(q_len, device=query_states.device)
            q_block_id = diff_indices.view(1, 1, -1, 1) // block_size
            kv_block_id = diff_indices.view(1, 1, 1, -1) // block_size
            diff_mask = torch.where(q_block_id == kv_block_id, zero_val, inf_val)
            diff_mask = diff_mask.expand(Bq, -1, -1, -1)

            attn_mask = torch.cat([ar_mask, diff_mask], dim=-1)

            attn_out = F.scaled_dot_product_attention(
                query_states, k_full, v_full, attn_mask=attn_mask, is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2)
        else:
            attn_out = F.scaled_dot_product_attention(
                query_states, k_full, v_full, is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        attn_out = self.o_proj(attn_out)
        return attn_out


class DiffusionLinearAttention(nn.Module):
    """Clone of Qwen3_5GatedDeltaNet that inherits AR recurrent state."""

    def __init__(self, ar_delta: Qwen3_5GatedDeltaNet, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Deep-copy all trainable components
        self.in_proj_qkv = copy.deepcopy(ar_delta.in_proj_qkv)
        self.in_proj_z = copy.deepcopy(ar_delta.in_proj_z)
        self.in_proj_b = copy.deepcopy(ar_delta.in_proj_b)
        self.in_proj_a = copy.deepcopy(ar_delta.in_proj_a)
        self.out_proj = copy.deepcopy(ar_delta.out_proj)
        self.conv1d = copy.deepcopy(ar_delta.conv1d)
        self.norm = copy.deepcopy(ar_delta.norm)

        # Non-trainable state params (copy but freeze)
        self.dt_bias = nn.Parameter(ar_delta.dt_bias.data.clone())
        self.A_log = nn.Parameter(ar_delta.A_log.data.clone())
        self.dt_bias.requires_grad = False
        self.A_log.requires_grad = False

        # Cached kernel references
        self.causal_conv1d_fn = ar_delta.causal_conv1d_fn
        self.causal_conv1d_update = ar_delta.causal_conv1d_update
        self.chunk_gated_delta_rule = ar_delta.chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = ar_delta.recurrent_gated_delta_rule

        # Config attributes needed by forward
        self.hidden_size = ar_delta.hidden_size
        self.num_v_heads = ar_delta.num_v_heads
        self.num_k_heads = ar_delta.num_k_heads
        self.head_k_dim = ar_delta.head_k_dim
        self.head_v_dim = ar_delta.head_v_dim
        self.key_dim = ar_delta.key_dim
        self.value_dim = ar_delta.value_dim
        self.conv_dim = ar_delta.conv_dim
        self.conv_kernel_size = ar_delta.conv_kernel_size

        for p in self.parameters():
            if p is not self.dt_bias and p is not self.A_log:
                p.requires_grad = True

    def forward(
        self,
        hidden_states: Tensor,
        ar_conv_state: Tensor | None = None,
        ar_recurrent_state: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Process diffusion block through cloned DeltaNet, starting from AR states.
        """
        B, seq_len, _ = hidden_states.shape
        has_ar_state = ar_conv_state is not None and ar_recurrent_state is not None

        # ── Projections ─────────────────────────────────────────────────
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, seq_len]

        z = self.in_proj_z(hidden_states)
        z = z.reshape(B, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # ── Causal Conv1D ────────────────────────────────────────────────
        if has_ar_state:
            # Prepend AR conv_state so causal conv sees full left-context
            mixed_qkv = torch.cat([ar_conv_state, mixed_qkv], dim=-1)

        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])

        if has_ar_state:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(B, seq_len, -1, self.head_k_dim)
        key = key.reshape(B, seq_len, -1, self.head_k_dim)
        value = value.reshape(B, seq_len, -1, self.head_v_dim)

        # ── Gated Delta Rule ─────────────────────────────────────────────
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=ar_recurrent_state if has_ar_state else None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # ── Norm + Gate + Output ─────────────────────────────────────────
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(B, seq_len, -1)

        return self.out_proj(core_attn_out)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Orthrus Model
# ═══════════════════════════════════════════════════════════════════════════════

class OrthrusQwen35Model(nn.Module):
    """Orthrus dual-view model wrapping frozen Qwen3.5-0.8B."""

    def __init__(
        self,
        base_model_path: str = "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B",
        block_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
        checkpoint_every: int = 0,
    ):
        super().__init__()
        self.block_size = block_size
        self.checkpoint_every = checkpoint_every

        # Load frozen base model
        print(f"  Loading frozen Qwen3.5 backbone from {base_model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, dtype=dtype,
        )
        self.config = self.base_model.config
        for p in self.base_model.parameters():
            p.requires_grad = False

        num_layers = self.config.num_hidden_layers
        layer_types = self.config.layer_types

        # Create diffusion heads — one per layer, type-matched
        self.diffusion_heads = nn.ModuleList()
        full_count = 0
        linear_count = 0

        for i in range(num_layers):
            layer_type = layer_types[i]
            ar_layer = self.base_model.model.layers[i]

            if layer_type == "full_attention":
                head = DiffusionFullAttention(ar_layer.self_attn, layer_idx=i)
                full_count += 1
            elif layer_type == "linear_attention":
                head = DiffusionLinearAttention(ar_layer.linear_attn, layer_idx=i)
                linear_count += 1
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.diffusion_heads.append(head)

        self.embed_tokens = self.base_model.model.embed_tokens
        self.norm = self.base_model.model.norm
        self.lm_head = self.base_model.lm_head
        self.rotary_emb = self.base_model.model.rotary_emb

        self.trainable_params = sum(
            p.numel() for head in self.diffusion_heads
            for p in head.parameters() if p.requires_grad
        )

        print(f"  Diffusion heads: {full_count} full_attn + {linear_count} linear_attn")
        print(f"  Trainable params: {self.trainable_params:,}")

    # ── AR Prefill ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_ar_prefill(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        anchor_positions: Tensor | None = None,
    ) -> Tuple[DynamicCache, Tensor, dict | None, dict | None]:
        """Run frozen AR backbone. Returns (kv, hidden, linear_states, per_block_la_conv).

        per_block_la_conv: {layer_idx: [B, num_anchors, C, ks]} per linear-attn layer.
        """
        if anchor_positions is not None:
            return self._forward_ar_prefill_with_extraction(input_ids, attention_mask, anchor_positions)
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )
        kv = outputs.past_key_values
        hidden = outputs.last_hidden_state.detach()
        del outputs
        return kv, hidden, None, None

    def _forward_ar_prefill_with_extraction(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        anchor_positions: Tensor,
    ) -> Tuple[DynamicCache, Tensor, dict, dict]:
        """Single-pass AR prefill with per-block GDN state extraction via fused Triton kernel."""
        from custom_gdn_extract import extract_gdn_states
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q35m

        B, seq_len = input_ids.shape
        num_anchors = anchor_positions.shape[1]
        device = input_ids.device

        # Build anchor_mask: (seq_len,) int32, -1 = not anchor, else anchor_idx
        anchor_vals = anchor_positions[0].cpu().tolist()
        anchor_mask = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for idx, pos in enumerate(anchor_vals):
            target_pos = pos - 1
            if 0 <= target_pos < seq_len:
                anchor_mask[target_pos] = idx

        linear_layer_indices = [
            i for i, lt in enumerate(self.config.layer_types)
            if lt == "linear_attention"
        ]

        # Per-layer extracted states
        linear_states: dict = {}
        per_block_la_conv: dict[int, Tensor] = {}

        # Store original forwards to restore later
        saved_forwards: dict = {}

        for i in linear_layer_indices:
            layer = self.base_model.model.layers[i]
            gdn: _q35m.Qwen3_5GatedDeltaNet = layer.linear_attn
            saved_forwards[i] = gdn.forward

            layer_idx = i
            ks = gdn.conv_kernel_size

            def make_wrapper(li, gdn_ref, ks_ref):
                def wrapper(hidden_states, cache_params=None, attention_mask=None):
                    # Replicate first ~90 lines of Qwen3_5GatedDeltaNet.forward
                    hidden_states = _q35m.apply_mask_to_padding_states(hidden_states, attention_mask)
                    batch_size, slen, _ = hidden_states.shape
                    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(li)
                    if use_precomputed_states:
                        conv_state = cache_params.layers[li].conv_states
                        recurrent_state = cache_params.layers[li].recurrent_states

                    mixed_qkv = gdn_ref.in_proj_qkv(hidden_states)
                    mixed_qkv = mixed_qkv.transpose(1, 2)

                    z = gdn_ref.in_proj_z(hidden_states)
                    z = z.reshape(batch_size, slen, -1, gdn_ref.head_v_dim)
                    b = gdn_ref.in_proj_b(hidden_states)
                    a = gdn_ref.in_proj_a(hidden_states)

                    if use_precomputed_states and slen == 1:
                        mixed_qkv = gdn_ref.causal_conv1d_update(
                            mixed_qkv, conv_state, gdn_ref.conv1d.weight.squeeze(1),
                            gdn_ref.conv1d.bias, gdn_ref.activation,
                        )
                    else:
                        if use_precomputed_states:
                            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
                        # Save pre-conv mixed_qkv for conv_state extraction
                        pre_conv_mixed_qkv = mixed_qkv
                        if cache_params is not None:
                            new_conv_state = F.pad(mixed_qkv, (ks_ref - mixed_qkv.shape[-1], 0))
                            cache_params.update_conv_state(new_conv_state, li)
                        if gdn_ref.causal_conv1d_fn is not None:
                            mixed_qkv = gdn_ref.causal_conv1d_fn(
                                x=mixed_qkv, weight=gdn_ref.conv1d.weight.squeeze(1),
                                bias=gdn_ref.conv1d.bias, activation=gdn_ref.activation, seq_idx=None,
                            )
                        else:
                            mixed_qkv = F.silu(gdn_ref.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
                        if use_precomputed_states:
                            mixed_qkv = mixed_qkv[:, :, -slen:]
                            pre_conv_mixed_qkv = pre_conv_mixed_qkv[:, :, -slen:]

                    mixed_qkv = mixed_qkv.transpose(1, 2)
                    qkv_splits = [gdn_ref.key_dim, gdn_ref.key_dim, gdn_ref.value_dim]
                    query, key, value = torch.split(mixed_qkv, qkv_splits, dim=-1)
                    query = query.reshape(batch_size, slen, -1, gdn_ref.head_k_dim)
                    key = key.reshape(batch_size, slen, -1, gdn_ref.head_k_dim)
                    value = value.reshape(batch_size, slen, -1, gdn_ref.head_v_dim)

                    beta = b.sigmoid()
                    g = -gdn_ref.A_log.float().exp() * F.softplus(a.float() + gdn_ref.dt_bias)
                    if gdn_ref.num_v_heads // gdn_ref.num_k_heads > 1:
                        query = query.repeat_interleave(gdn_ref.num_v_heads // gdn_ref.num_k_heads, dim=2)
                        key = key.repeat_interleave(gdn_ref.num_v_heads // gdn_ref.num_k_heads, dim=2)

                    # Extract states at anchor positions using fused kernel
                    if anchor_mask is not None and len(anchor_vals) > 0:
                        # Transpose to (B, T, H, K)  — kernel expects this layout
                        k_for_kernel = key  # (B, T, HV, K)
                        v_for_kernel = value  # (B, T, HV, V)
                        g_for_kernel = g  # (B, T, HV)
                        beta_for_kernel = beta  # (B, T, HV)

                        anchor_states = extract_gdn_states(
                            k_for_kernel, v_for_kernel, g_for_kernel, beta_for_kernel,
                            anchor_mask, num_anchors, use_qk_l2norm_in_kernel=True,
                        )  # (B, num_anchors, HV, K, V)
                        linear_states[li] = anchor_states

                        # Conv state per anchor: one [B, num_anchors, C, ks] buffer per layer
                        C = pre_conv_mixed_qkv.shape[1]
                        if li not in per_block_la_conv:
                            per_block_la_conv[li] = torch.zeros(
                                batch_size, num_anchors, C, ks_ref,
                                device=pre_conv_mixed_qkv.device,
                                dtype=pre_conv_mixed_qkv.dtype,
                            )
                        conv_buf = per_block_la_conv[li]
                        for ai, pos in enumerate(anchor_vals):
                            if pos >= slen:
                                continue
                            start = max(0, pos - ks_ref)
                            ctx = pre_conv_mixed_qkv[:, :, start:pos]
                            if ctx.shape[-1] < ks_ref:
                                ctx = F.pad(ctx, (ks_ref - ctx.shape[-1], 0))
                            conv_buf[:, ai].copy_(ctx.detach())

                    # Continue with original chunk_gated_delta_rule
                    if use_precomputed_states and slen == 1:
                        core_attn_out, last_recurrent_state = gdn_ref.recurrent_gated_delta_rule(
                            query, key, value, g=g, beta=beta,
                            initial_state=recurrent_state,
                            output_final_state=cache_params is not None,
                            use_qk_l2norm_in_kernel=True,
                        )
                    else:
                        core_attn_out, last_recurrent_state = gdn_ref.chunk_gated_delta_rule(
                            query, key, value, g=g, beta=beta,
                            initial_state=recurrent_state if use_precomputed_states else None,
                            output_final_state=cache_params is not None,
                            use_qk_l2norm_in_kernel=True,
                        )

                    if cache_params is not None:
                        cache_params.update_recurrent_state(last_recurrent_state, li)

                    core_attn_out = core_attn_out.reshape(-1, gdn_ref.head_v_dim)
                    z = z.reshape(-1, gdn_ref.head_v_dim)
                    core_attn_out = gdn_ref.norm(core_attn_out, z)
                    core_attn_out = core_attn_out.reshape(batch_size, slen, -1)
                    return gdn_ref.out_proj(core_attn_out)
                return wrapper

            gdn.forward = make_wrapper(i, gdn, ks)

        # Run the actual forward pass (single pass!)
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

        # Restore original forward methods
        for i in linear_layer_indices:
            self.base_model.model.layers[i].linear_attn.forward = saved_forwards[i]

        kv = outputs.past_key_values
        hidden = outputs.last_hidden_state.detach()
        del outputs
        return kv, hidden, linear_states, per_block_la_conv

    # ── Diffusion Forward ────────────────────────────────────────────────────
    def forward_diffusion(
        self,
        diff_input_ids: Tensor,
        ar_past_key_values: DynamicCache,
        ar_seq_len: int,
        causal_limit: Tensor | None = None,
        return_hidden: bool = False,
        diff_position_ids: Tensor | None = None,
        use_flex: bool = False,
        linear_states: dict | None = None,
        block_indices: Tensor | None = None,
        per_block_fa_kv: dict | None = None,
        per_block_la_conv: dict | None = None,
    ) -> Tensor:
        """Process diffusion block through all diffusion heads."""
        B = diff_input_ids.shape[0]
        K = self.block_size
        diff_len = diff_input_ids.shape[1]
        device = diff_input_ids.device

        linear_states, per_block_la_conv, block_indices = _stream_per_block_states_to_device(
            linear_states, per_block_la_conv, block_indices, device,
        )

        hidden_states = self.embed_tokens(diff_input_ids)

        if diff_position_ids is not None:
            position_ids = diff_position_ids
        else:
            position_ids = torch.arange(
                ar_seq_len, ar_seq_len + diff_len, device=device,
            ).unsqueeze(0).expand(B, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        if causal_limit is None:
            causal_limit = self._default_causal_limit(
                diff_len, ar_seq_len, K, device,
            ).expand(B, -1)

        layers = self.base_model.model.layers
        ckpt = self.checkpoint_every

        if ckpt > 0 and ckpt < len(layers):
            chk = torch.utils.checkpoint.checkpoint
            for grp_start in range(0, len(layers), ckpt):
                grp_end = min(grp_start + ckpt, len(layers))
                hidden_states = chk(
                    _run_diffusion_layers,
                    hidden_states, layers[grp_start:grp_end],
                    self.diffusion_heads[grp_start:grp_end],
                    position_embeddings, ar_past_key_values,
                    ar_seq_len, causal_limit, K,
                    grp_start,
                    use_reentrant=False,
                )
        else:
            for layer_idx, (layer, diff_head) in enumerate(
                zip(layers, self.diffusion_heads)
            ):
                hidden_states = _run_single_diffusion_layer(
                    hidden_states=hidden_states,
                    layer=layer,
                    diff_head=diff_head,
                    position_embeddings=position_embeddings,
                    ar_past_key_values=ar_past_key_values,
                    ar_seq_len=ar_seq_len,
                    causal_limit=causal_limit,
                    block_size=K,
                    layer_idx=layer_idx,
                    linear_states=linear_states,
                    block_indices=block_indices,
                    per_block_fa_kv=per_block_fa_kv,
                    per_block_la_conv=per_block_la_conv,
                )

        hidden_states = self.norm(hidden_states)
        if return_hidden:
            return hidden_states
        return self.lm_head(hidden_states)

    # ── Convenience forward ──────────────────────────────────────────────────
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
        use_flex: bool = False,
    ) -> Tensor:
        if not is_diffusion_pass:
            _, hidden, _, _ = self.forward_ar_prefill(input_ids, attention_mask)
            return self.lm_head(hidden)
        return self.forward_diffusion(
            diff_input_ids=input_ids,
            ar_past_key_values=ar_past_key_values,
            ar_seq_len=ar_seq_len,
            causal_limit=causal_limit,
            return_hidden=return_hidden,
            diff_position_ids=diff_position_ids,
            use_flex=use_flex,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _default_causal_limit(
        self, diff_len: int, ar_seq_len: int, K: int, device: torch.device,
    ) -> Tensor:
        limits = torch.zeros(diff_len, dtype=torch.long, device=device)
        for b in range(diff_len // K):
            anchor = ar_seq_len - diff_len + b * K
            limits[b * K:(b + 1) * K] = anchor - 1
        return limits.unsqueeze(0)

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for head in self.diffusion_heads
                for p in head.parameters() if p.requires_grad]
    def compile_diffusion_heads(self):
        """torch.compile full_attn heads only (linear_attn heads use fla, breaks inductor)."""
        import torch
        torch._dynamo.config.recompile_limit = 128
        compiled = 0
        for i, head in enumerate(self.diffusion_heads):
            if isinstance(head, DiffusionLinearAttention):
                continue  # fla kernels are incompatible with torch.compile
            try:
                self.diffusion_heads[i] = torch.compile(
                    head, fullgraph=False,
                )
                compiled += 1
            except Exception:
                pass
        if compiled:
            print(f"  ✓ Compiled {compiled}/{len(self.diffusion_heads)} diffusion heads (full_attn only)")


# ── Layer processing helpers (used by gradient checkpointing) ────────────────

def _gather_la_conv_for_blocks(
    per_block_la_conv: dict[int, Tensor],
    layer_idx: int,
    block_indices: Tensor,
    batch_size: int,
) -> Tensor:
    """Stacked conv states [B, num_anchors, C, ks] → [B * n_blocks, C, ks]."""
    stacked = per_block_la_conv[layer_idx]
    idx = block_indices.to(device=stacked.device, dtype=torch.long)
    sel = stacked.index_select(1, idx)
    return sel.reshape(batch_size * sel.shape[1], *sel.shape[2:])


def _stream_per_block_states_to_device(
    linear_states: dict | None,
    per_block_la_conv: dict | None,
    block_indices: Tensor | None,
    device: torch.device,
) -> tuple[dict | None, dict | None, Tensor | None]:
    """
    Prepare per-block AR states for diffusion.

    - Already on ``device`` (training / eval without CPU offload): passthrough,
      keep absolute ``block_indices``.
    - On CPU with ``block_indices``: upload only those blocks, remap indices to
      local ``0 .. n_blocks-1``.
    - On CPU without ``block_indices``: full upload (legacy callers).
    """
    if linear_states is None:
        return None, per_block_la_conv, block_indices

    sample = next(iter(linear_states.values()))
    if sample.device.type == device.type:
        return linear_states, per_block_la_conv, block_indices

    if block_indices is not None:
        idx = block_indices.detach().cpu().long()
        n_blocks = idx.numel()
        linear_states = {
            k: v.index_select(1, idx).to(device, non_blocking=True)
            for k, v in linear_states.items()
        }
        if per_block_la_conv is not None:
            per_block_la_conv = {
                li: stacked.index_select(1, idx).to(device, non_blocking=True)
                for li, stacked in per_block_la_conv.items()
            }
        block_indices = torch.arange(n_blocks, device=device)
    else:
        linear_states = {
            k: v.to(device, non_blocking=True) for k, v in linear_states.items()
        }
        if per_block_la_conv is not None:
            per_block_la_conv = {
                li: stacked.to(device, non_blocking=True)
                for li, stacked in per_block_la_conv.items()
            }

    return linear_states, per_block_la_conv, block_indices


def _run_single_diffusion_layer(
    hidden_states: Tensor,
    layer,
    diff_head: nn.Module,
    position_embeddings: Tuple[Tensor, Tensor],
    ar_past_key_values: DynamicCache,
    ar_seq_len: int,
    causal_limit: Tensor,
    block_size: int,
    layer_idx: int,
    linear_states: dict | None = None,
    block_indices: Tensor | None = None,
    per_block_fa_kv: dict | None = None,
        per_block_la_conv: dict | None = None,  # {layer_idx: Tensor [B, num_anchors, C, ks]}
) -> Tensor:
    """Process one layer's diffusion head."""
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    layer_type = layer.layer_type

    if layer_type == "full_attention":
        # Shared AR cache (full attention is causal — shared cache is safe)
        B, total_tokens, hidden = normed.shape
        n_blocks = total_tokens // block_size
        ar_keys_pb = ar_past_key_values.layers[layer_idx].keys
        ar_vals_pb = ar_past_key_values.layers[layer_idx].values

        if n_blocks <= 1:
            diff_out = diff_head(
                hidden_states=normed,
                position_embeddings=position_embeddings,
                k_ar=ar_keys_pb[:, :, :ar_seq_len, :], v_ar=ar_vals_pb[:, :, :ar_seq_len, :],
                causal_limit=causal_limit,
                block_size=block_size,
            )
        else:
            diff_out = diff_head(
                hidden_states=normed[:, :n_blocks * block_size, :],
                position_embeddings=position_embeddings,
                k_ar=ar_keys_pb[:, :, :ar_seq_len, :], v_ar=ar_vals_pb[:, :, :ar_seq_len, :],
                causal_limit=causal_limit[:, :n_blocks * block_size],
                block_size=block_size,
            )
            # Handle any remainder
            remaining = total_tokens - n_blocks * block_size
            if remaining > 0:
                diff_out_rem = diff_head(
                    hidden_states=normed[:, n_blocks * block_size:, :],
                    position_embeddings=position_embeddings,
                    k_ar=ar_keys_pb[:, :, :ar_seq_len, :], v_ar=ar_vals_pb[:, :, :ar_seq_len, :],
                    causal_limit=causal_limit[:, n_blocks * block_size:],
                    block_size=block_size,
                )
                diff_out = torch.cat([diff_out, diff_out_rem], dim=1)
    elif layer_type == "linear_attention":
        has_per_block = linear_states is not None and layer_idx in linear_states
        ar_layer_cache = ar_past_key_values.layers[layer_idx]
        B, total_tokens, hidden = normed.shape
        n_blocks = total_tokens // block_size

        # Per-block conv_state: use if available, else shared cache
        if has_per_block and per_block_la_conv is not None and layer_idx in per_block_la_conv and block_indices is not None:
            ar_conv = _gather_la_conv_for_blocks(
                per_block_la_conv, layer_idx, block_indices, B,
            )
            has_batched_conv = True
        else:
            ar_conv = ar_layer_cache.conv_states if ar_layer_cache.is_conv_states_initialized else None
            has_batched_conv = False

        # Resolve per-block recurrent states
        if has_per_block and block_indices is not None:
            # [B, n_blocks, heads, k_dim, v_dim] → select blocks for this batch
            per_block_r = linear_states[layer_idx][:, block_indices, :, :, :]
            # Reshape to [B*n_blocks, heads, k_dim, v_dim] for batched forward
            per_block_r = per_block_r.reshape(B * n_blocks, *per_block_r.shape[2:])
        else:
            per_block_r = None

        ar_recurrent = ar_layer_cache.recurrent_states if ar_layer_cache.is_recurrent_states_initialized else None

        # Use per_block_r for recurrent state, ar_conv for conv state (same for all blocks)
        if n_blocks == 0:
            diff_out = diff_head(
                hidden_states=normed,
                ar_conv_state=ar_conv,
                ar_recurrent_state=per_block_r[0:1] if per_block_r is not None else ar_recurrent,
            )
        else:
            batched = normed[:, :n_blocks * block_size, :].view(B * n_blocks, block_size, hidden)
            if has_batched_conv:
                batched_conv = ar_conv  # already [n_blocks, C, ks], matches batched dim 0
            else:
                batched_conv = ar_conv.expand(B * n_blocks, -1, -1) if ar_conv is not None else None
            batched_recurrent = per_block_r if per_block_r is not None else (
                ar_recurrent.expand(B * n_blocks, *([-1] * (ar_recurrent.ndim - 1))) if ar_recurrent is not None else None
            )
            diff_out_full = diff_head(
                hidden_states=batched,
                ar_conv_state=batched_conv,
                ar_recurrent_state=batched_recurrent,
            ).view(B, n_blocks * block_size, hidden)
            remaining = total_tokens - n_blocks * block_size
            if remaining > 0:
                diff_out_rem = diff_head(
                    hidden_states=normed[:, n_blocks * block_size:, :],
                    ar_conv_state=ar_conv[n_blocks:n_blocks+1] if has_batched_conv else ar_conv,
                    ar_recurrent_state=per_block_r[n_blocks:n_blocks+1] if per_block_r is not None and per_block_r.shape[0] > n_blocks else ar_recurrent,
                )
                diff_out = torch.cat([diff_out_full, diff_out_rem], dim=1)
            else:
                diff_out = diff_out_full
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    hidden_states = residual + diff_out

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def _run_diffusion_layers(
    hidden_states: Tensor,
    layers,
    diff_heads,
    position_embeddings: Tuple[Tensor, Tensor],
    ar_past_key_values: DynamicCache,
    ar_seq_len: int,
    causal_limit: Tensor,
    block_size: int,
    start_idx: int,
) -> Tensor:
    """Process a group of layers (for gradient checkpointing)."""
    for local_idx, (layer, diff_head) in enumerate(zip(layers, diff_heads)):
        layer_idx = start_idx + local_idx
        hidden_states = _run_single_diffusion_layer(
            hidden_states=hidden_states,
            layer=layer,
            diff_head=diff_head,
            position_embeddings=position_embeddings,
            ar_past_key_values=ar_past_key_values,
            ar_seq_len=ar_seq_len,
            causal_limit=causal_limit,
            block_size=block_size,
            layer_idx=layer_idx,
        )
    return hidden_states


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Orthrus Qwen3.5 Model Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    t0 = time.time()
    model = OrthrusQwen35Model(block_size=32, dtype=dtype).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained("F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Load time: {time.time() - t0:.1f}s")

    # Test AR prefill
    ids = torch.randint(0, 100, (1, 64), device=device)
    kv, hidden, _, _ = model.forward_ar_prefill(ids)
    print(f"  AR prefill: cache layers={len(kv.layers)}, hidden={hidden.shape}")

    # Test diffusion forward
    mask_id = 0  # dummy
    diff_ids = torch.randint(0, 100, (1, 32), device=device)
    diff_logits = model.forward_diffusion(
        diff_input_ids=diff_ids,
        ar_past_key_values=kv,
        ar_seq_len=64,
        use_flex=False,
    )
    print(f"  Diffusion fwd: logits={diff_logits.shape}")

    print("  ✓ All tests passed")
