"""
Zero-leakage per-block linear state extraction via chunked AR prefill.

Processes the AR sequence in 64-token chunks (fast), saves recurrent/conv
states at chunk boundaries, then advances from nearest boundary to each
anchor position. Total: ~32 chunk calls + ~256 small advances.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, List, Tuple

CHUNK_SIZE = 64  # match the model's native chunk size for gated_delta_rule


def get_per_block_caches(
    model,                  # OrthrusQwen35Model
    ar_input_ids: Tensor,   # [1, L] — full AR sequence
    ar_attention_mask: Tensor | None,  # [1, L]
    anchor_positions: Tensor,  # [1, n_blocks] — sorted anchor positions
    bucket_granularity: int = 1,  # unused
) -> Tuple[Dict[int, Tensor], Dict[int, Dict[int, Tuple[Tensor, Tensor]]], Dict[int, Dict[int, Tensor]]]:
    """
    Extract per-block caches with zero leakage.

    Phase 1: Run AR prefill in CHUNK_SIZE-token chunks, saving boundary states.
    Phase 2: For each anchor, advance from nearest boundary.
    """
    B, n_blocks = anchor_positions.shape
    device = ar_input_ids.device
    ar_len = ar_input_ids.shape[1]

    anchor_vals = anchor_positions[0].cpu().tolist()

    linear_layer_indices = [
        i for i in range(len(model.base_model.model.layers))
        if model.base_model.model.layers[i].layer_type == "linear_attention"
    ]

    linear_states: Dict[int, Tensor] = {}
    fa_kv: Dict[int, Dict[int, Tuple[Tensor, Tensor]]] = {}
    la_conv: Dict[int, Dict[int, Tensor]] = {i: {} for i in linear_layer_indices}

    # ---- Phase 1: Boundary states via 64-token chunks ----
    # boundary_caches[chunk_idx] = deepcopy of kv_cache after chunk chunk_idx
    # chunk_idx 0 covers tokens 0..63, boundary state = after 64 tokens
    import copy
    boundary_caches: list = []
    kv_cache = None

    for chunk_start in range(0, ar_len, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, ar_len)
        chunk_ids = ar_input_ids[:, chunk_start:chunk_end]
        chunk_mask = ar_attention_mask[:, chunk_start:chunk_end] if ar_attention_mask is not None else None

        with torch.no_grad():
            if kv_cache is None:
                kv_cache, _, _, _ = model.forward_ar_prefill(chunk_ids, chunk_mask)
            else:
                outputs = model.base_model.model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values

        boundary_caches.append(copy.deepcopy(kv_cache))
        del chunk_ids, chunk_mask

    # ---- Phase 2: Per-anchor extraction ----
    for blk, anchor in enumerate(anchor_vals):
        a = min(anchor, ar_len)
        if a == 0:
            continue

        # Nearest boundary AT or BEFORE anchor
        boundary_end = (a // CHUNK_SIZE) * CHUNK_SIZE
        boundary_chunk = (boundary_end // CHUNK_SIZE) - 1 if boundary_end > 0 else -1
        adv_tokens = a - boundary_end

        if adv_tokens == 0:
            # Anchor falls exactly on boundary
            bc = boundary_caches[boundary_chunk]
            kv_src = bc
        elif boundary_chunk < 0:
            # Anchor within first chunk
            kv_src, _, _, _ = model.forward_ar_prefill(ar_input_ids[:, :a], ar_attention_mask[:, :a] if ar_attention_mask is not None else None)
        else:
            # Advance from boundary
            bc = boundary_caches[boundary_chunk]
            adv_ids = ar_input_ids[:, boundary_end:a]
            adv_mask = ar_attention_mask[:, boundary_end:a] if ar_attention_mask is not None else None
            outputs = model.base_model.model(
                input_ids=adv_ids,
                attention_mask=adv_mask,
                past_key_values=bc,
                use_cache=True,
            )
            kv_src = outputs.past_key_values
            del adv_ids, adv_mask, outputs

        # Extract states from kv_src
        for i in linear_layer_indices:
            layer_cache = kv_src.layers[i]
            if layer_cache.is_recurrent_states_initialized:
                if i not in linear_states:
                    rs = layer_cache.recurrent_states
                    linear_states[i] = torch.zeros(
                        B, n_blocks, rs.shape[1], rs.shape[2], rs.shape[3],
                        device=device, dtype=rs.dtype,
                    )
                linear_states[i][0, blk] = layer_cache.recurrent_states[0]
                if layer_cache.is_conv_states_initialized:
                    la_conv[i][blk] = layer_cache.conv_states[0:1].clone()

        if adv_tokens != 0 and boundary_chunk >= 0:
            # kv_src was from an incremental call, but bc (boundary cache) might still be needed
            # by other blocks. kv_src shares the same layers but we only read from it.
            pass

    return linear_states, fa_kv, la_conv
