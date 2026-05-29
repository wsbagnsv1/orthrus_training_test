"""
Interactive CLI inference for OrthrusQwen35 with color-coded consensus visualization.

Green  = token was predicted correctly by the diffusion head & accepted by AR consensus
White  = anchor tokens, AR-corrected tokens, and first AR token (default)

Usage:
    python infer.py                           # interactive REPL
    python infer.py --prompt "Hello world"    # one-shot
    python infer.py --checkpoint ../checkpoints/step_6000 --K 32
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List, Tuple
from transformers.cache_utils import DynamicCache

import time

import torch
import torch.nn.functional as F

# Disable TF32 — required for bit-exact batch vs sequential matmul results
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
from torch import Tensor
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

# ── ensure local package is on path ──────────────────────────────────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import OrthrusQwen35Model

# ── ANSI color codes ─────────────────────────────────────────────────────────
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[91m"  # actually red — better visibility
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Enable ANSI on Windows
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetConsoleHandle(-11), 7)
    except Exception:
        pass


# ── sampling helpers ─────────────────────────────────────────────────────────

def sample_token(logits: Tensor, temperature: float = 0.0) -> Tuple[Tensor, Optional[Tensor]]:
    """Sample tokens from logits. Handles [vocab], [L, vocab], or [B, L, vocab].
    Returns (token_ids, probs). probs is None for greedy (temperature=0)."""
    if temperature < 1e-5:
        return logits.argmax(dim=-1), None
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    flat_probs = probs.view(-1, probs.size(-1))
    tokens = torch.multinomial(flat_probs, 1).view(probs.shape[:-1])
    return tokens, probs


# ── color-coded generation ───────────────────────────────────────────────────

class TokenSpan:
    """Tracks a token and whether it was accepted by consensus."""
    def __init__(self, token_id: int, accepted: bool, is_anchor: bool = False, is_max_block: bool = False):
        self.token_id = token_id
        self.accepted = accepted        # True = diffusion predicted correctly, AR accepted
        self.is_anchor = is_anchor       # True = anchor token (always accepted, not colored)
        self.is_max_block = is_max_block # True = part of the block with highest acceptance


@torch.no_grad()
def generate_colored(
    model: OrthrusQwen35Model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = True,
    stream: bool = True,
    debug: bool = False,
    deterministic: bool = False,
) -> Tuple[str, dict, List[TokenSpan]]:
    """
    Generate text using Orthrus consensus decoding with token-level coloring info.

    Matches the reference OrthrusLM.generate() logic:
      - Diffusion predicts K-1 tokens from logits positions 0..K-2.
      - AR verifies all K positions; consensus = cumprod of token matches.
      - KV cache is cropped after each block commit.

    Returns (generated_text, stats_dict, token_spans).
    Each TokenSpan records whether the token was consensus-accepted (green)
    or AR-corrected/default (white).
    """
    model.eval()
    device = next(model.parameters()).device
    K = model.block_size
    # Use <mask> token (new special token, multimodal-safe)
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        # Fallback to <tts_pad> if <mask> not found
        mask_id = tokenizer.convert_tokens_to_ids("<tts_pad>")
    eos_id = tokenizer.eos_token_id

    # Deterministic: same seed, no compile, eager execution
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    t_start = time.perf_counter()

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_len = prompt_len + max_new_tokens

    # Allocate output buffer (match reference: pre-fill with mask tokens, extra block_size slack)
    output_ids = torch.full((1, max_len + K), mask_id, dtype=torch.long, device=device)
    output_ids[:, :prompt_len] = input_ids

    token_spans: List[TokenSpan] = []

    # ── AR Prefill (shared KV cache) ────────────────────────────────────
    from transformers.cache_utils import StaticCache
    import types
    
    max_cache_len = prompt_len + max_new_tokens + 10
    past_key_values = StaticCache(
        config=model.base_model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=model.base_model.dtype
    )
    
    def static_cache_crop(self, max_length: int):
        for layer in self.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.fill_(max_length)
                
    past_key_values.crop = types.MethodType(static_cache_crop, past_key_values)

    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    base_out = model.base_model(
        input_ids=input_ids, 
        position_ids=position_ids, 
        past_key_values=past_key_values, 
        use_cache=True,
        logits_to_keep=1
    )
    past_key_values = base_out.past_key_values

    # First token from prefill logits
    first_logits = base_out.logits[:, -1, :]
    next_token, _ = sample_token(first_logits, temperature)
    next_token_id = next_token.item()

    # ── Wrap Gated Delta Net to extract intermediate states ────────────
    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35m
    from inference_kernel import fused_recurrent_inference_fwd

    linear_layer_indices = [
        i for i, lt in enumerate(model.config.layer_types)
        if lt == "linear_attention"
    ]
    
    saved_forwards = {}
    for i in linear_layer_indices:
        layer = model.base_model.model.layers[i]
        gdn = layer.linear_attn
        saved_forwards[i] = gdn.forward

        def make_inference_wrapper(li, gdn_ref, ks_ref):
            def wrapper(hidden_states, cache_params=None, attention_mask=None):
                # Standard Qwen3.5 GatedDeltaNet logic
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
                    
                    if cache_params is not None:
                        # Save pre-conv mixed_qkv so we can crop conv state later
                        cache_params.layers[li].pre_conv_mixed_qkv = mixed_qkv.clone()
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

                # Use our custom inference kernel to get ALL intermediate states, BUT only for short blocks!
                if slen > 128:
                    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
                    o, last_recurrent_state = chunk_gated_delta_rule(
                        query, key, value, g=g, beta=beta,
                        initial_state=recurrent_state if use_precomputed_states else None,
                        output_final_state=cache_params is not None,
                        use_qk_l2norm_in_kernel=True,
                    )
                    h_out = None
                else:
                    from inference_kernel import fused_recurrent_inference_fwd
                    o, last_recurrent_state, h_out = fused_recurrent_inference_fwd(
                        query, key, value, g=g, beta=beta,
                        initial_state=recurrent_state if use_precomputed_states else None,
                        output_final_state=cache_params is not None,
                        use_qk_l2norm_in_kernel=True,
                    )
                
                if cache_params is not None:
                    # Save intermediate recurrent states for slicing later (only matters during generation)
                    if h_out is not None:
                        cache_params.layers[li].h_out_all = h_out
                    cache_params.update_recurrent_state(last_recurrent_state, li)

                core_attn_out = o.reshape(-1, gdn_ref.head_v_dim)
                z = z.reshape(-1, gdn_ref.head_v_dim)
                core_attn_out = gdn_ref.norm(core_attn_out, z)
                core_attn_out = core_attn_out.reshape(batch_size, slen, -1)
                return gdn_ref.out_proj(core_attn_out)
            return wrapper
            
        gdn.forward = make_inference_wrapper(i, gdn, gdn.conv_kernel_size)

    start_idx = prompt_len
    output_ids[:, start_idx] = next_token_id
    token_spans.append(TokenSpan(next_token_id, accepted=False))

    if stream:
        tok_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        if tok_text:
            sys.stdout.write(tok_text)
            sys.stdout.flush()

    if next_token_id == eos_id:
        output = tokenizer.decode([next_token_id], skip_special_tokens=True)
        return output, {
            "tokens_generated": 1, "forward_passes": 1, "tpf": 1.0,
            "acceptance_lengths": [], "avg_acceptance": 0.0,
            "max_acceptance": 0,
            "consensus_accepts": 0, "consensus_total": 0,
            "time_ms": (time.perf_counter() - t_start) * 1000,
        }, token_spans

    total_forward_passes = 1  # prefill
    all_acceptance_lengths: List[int] = []
    consensus_accepts = 0
    consensus_total = 0
    block_num = 0
    max_acceptance_len = 0
    max_block_start = 0  # position index in token_spans where max block starts
    
    K = model.block_size
    offset_accepted = [0] * (K + 1)
    offset_tested = [0] * (K + 1)

    # ── Generation loop ─────────────────────────────────────────────────
    while start_idx < max_len - 1:
        diff_len = min(K, max_len - start_idx)
        block_num += 1

        # Build diffusion block: anchor at position 0, masks at 1..K-1
        diff_block_ids = torch.full((1, diff_len), mask_id, dtype=torch.long, device=device)
        diff_block_ids[:, 0] = output_ids[:, start_idx]  # anchor
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)

        # ── Step 1: Diffusion parallel projection (no cache update) ──────
        # causal_limit: anchor sees up to start_idx-1; mask k sees up to start_idx + k - 1
        causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
        for k in range(diff_len):
            causal_limit[0, k] = start_idx - 1  # all positions see full AR context up to anchor

        diff_logits = model(
            input_ids=diff_block_ids,
            is_diffusion_pass=True,
            ar_past_key_values=past_key_values,
            ar_seq_len=start_idx,
            causal_limit=causal_limit,
            use_flex=False,  # plain SDPA, no flex_attention overhead
        )  # [1, diff_len, vocab]
        total_forward_passes += 1

        # Sample from positions 0..K-2 -> predictions for tokens 1..K-1 (the masks)
        # Standard causal logic: logit[t] predicts token[t+1]
        if diff_len > 1:
            diff_tokens, diff_probs = sample_token(diff_logits[:, :diff_len-1, :], temperature)
            # diff_tokens: [1, K-1]
        else:
            diff_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            diff_probs = None

        # Proposed block = anchor + diffusion predictions
        proposed_block = torch.cat([output_ids[:, start_idx:start_idx+1], diff_tokens], dim=1)

        # ── Step 2: AR verification — DUAL-PASS hybrid ──
        # 1. Save tiny recurrent states (6MB) — avoids 1GB KV deepcopy
        linear_indices = [i for i in range(len(past_key_values.layers))
                          if hasattr(past_key_values.layers[i], 'is_recurrent_states_initialized')
                          and past_key_values.layers[i].is_recurrent_states_initialized]
        saved_recurrent = {}
        saved_conv = {}
        for li in linear_indices:
            lc = past_key_values.layers[li]
            saved_recurrent[li] = lc.recurrent_states.clone()
            if lc.is_conv_states_initialized:
                saved_conv[li] = lc.conv_states.clone()

        # 2. Batch AR forward (proposed_block → logits)
        ar_pos_ids = torch.arange(start_idx, start_idx + proposed_block.shape[1],
                                  device=device).unsqueeze(0)
        ar_out = model.base_model(
            input_ids=proposed_block, position_ids=ar_pos_ids,
            past_key_values=past_key_values, use_cache=True,
        )
        ar_logits = ar_out.logits[0]  # [block_len, vocab]
        total_forward_passes += 1

        # 3. Greedy consensus check
        acceptance_len = 0
        for k in range(1, proposed_block.shape[1]):
            ar_pred = ar_logits[k - 1].argmax().item()
            offset_tested[k] += 1
            if proposed_block[0, k].item() == ar_pred:
                acceptance_len += 1
                consensus_accepts += 1
                consensus_total += 1
                offset_accepted[k] += 1
            else:
                consensus_total += 1
                break

        all_acceptance_lengths.append(acceptance_len)

        # 4. State slicing (Fast cache rollback)
        # We ALREADY have the cache up to the full proposed block!
        # Just crop KV cache to the accepted length (anchor + accepted)
        end_idx = start_idx + acceptance_len + 1
        past_key_values.crop(end_idx)

        for li in linear_indices:
            lc = past_key_values.layers[li]
            gdn = model.base_model.model.layers[li].linear_attn
            
            # Revert recurrent states to the state immediately after processing the accepted block
            # h_out_all has shape [1, block_len, HV, K, V]
            # State after anchor is index 0. State after acceptance_len is at index acceptance_len.
            h_out_all = lc.h_out_all
            lc.recurrent_states = h_out_all[:, acceptance_len].clone()
            
            # Revert conv_states
            prev_conv_len = lc.conv_states.shape[-1]
            accepted_tokens_len = acceptance_len + 1
            # pre_conv_mixed_qkv has length `prev_conv_len + block_len`
            end_conv = prev_conv_len + accepted_tokens_len
            start_conv = end_conv - prev_conv_len
            lc.conv_states = lc.pre_conv_mixed_qkv[:, :, start_conv : end_conv].clone()

        # 5. Skip third pass! Corrected logits are already computed in Step 2.
        accepted_block = proposed_block[:, :acceptance_len + 1]
        ar_logits_corrected = ar_logits[:acceptance_len + 1]

        # 6. Correction token from AR at final position
        next_token_corrected_id = ar_logits_corrected[-1].argmax().item()
        next_token_corrected = torch.tensor([[next_token_corrected_id]], dtype=torch.long, device=device)
        output_ids[:, start_idx:end_idx] = accepted_block

        # Record token spans for coloring
        # Position 0 = anchor (always accepted, not colored)
        block_span_start = len(token_spans)
        for i in range(1, acceptance_len + 1):
            tok_id = int(accepted_block[0, i].item())
            token_spans.append(TokenSpan(tok_id, accepted=True))  # diffusion accepted

        # Track max block
        if acceptance_len > max_acceptance_len:
            max_acceptance_len = acceptance_len
            max_block_start = block_span_start

        # Stream accepted tokens
        if stream:
            for i in range(1, acceptance_len + 1):
                text = tokenizer.decode([int(accepted_block[0, i].item())], skip_special_tokens=True)
                if text:
                    disp = text.replace('\n', '\\n\n').replace('\r', '\\r\r').replace('\t', '\\t').replace(' ', '·')
                    sys.stdout.write(f"{GREEN}{disp}{RESET}")
                    sys.stdout.flush()

            if verbose and acceptance_len > 0:
                sys.stderr.write(f"\n{DIM}[Block {block_num}] accepted {acceptance_len + 1}/{proposed_block.shape[1]} tokens "
                                 f"({acceptance_len} consensus + anchor){RESET}\n")

        # Check for EOS in accepted block
        eos_in_block = (accepted_block[:, 1:] == eos_id).any()
        if eos_in_block:
            # Find first EOS and truncate
            for i in range(1, acceptance_len + 1):
                if int(accepted_block[0, i].item()) == eos_id:
                    output_ids[:, start_idx + i + 1:] = mask_id
                    start_idx = start_idx + i + 1
                    break
            break

        # Update state for next iteration
        start_idx = end_idx

        # Correction token (from AR) — write to output, not colored
        next_tok_id = int(next_token_corrected[0, 0].item())
        token_spans.append(TokenSpan(next_tok_id, accepted=False))

        if start_idx < max_len:
            output_ids[:, start_idx] = next_token_corrected
            # Stream correction token
            if stream:
                corr_text = tokenizer.decode([next_tok_id], skip_special_tokens=True)
                if corr_text:
                    disp = corr_text.replace('\n', '\\n\n').replace('\r', '\\r\r').replace('\t', '\\t').replace(' ', '·')
                    sys.stdout.write(disp)
                    sys.stdout.flush()

            if next_tok_id == eos_id:
                break

    # ── Decode ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Mark the max-acceptance block tokens for yellow highlighting
    for i in range(max_block_start, max_block_start + max_acceptance_len):
        if i < len(token_spans):
            token_spans[i].is_max_block = True

    generated_ids = output_ids[0, prompt_len:start_idx + 1].tolist()
    generated_ids = [t for t in generated_ids if t != mask_id]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = {
        "tokens_generated": len(generated_ids),
        "forward_passes": total_forward_passes,
        "tpf": len(generated_ids) / max(total_forward_passes, 1),
        "acceptance_lengths": all_acceptance_lengths,
        "avg_acceptance": sum(l for l in all_acceptance_lengths if l > 0) / len([l for l in all_acceptance_lengths if l > 0]) if any(l > 0 for l in all_acceptance_lengths) else 0.0,
        "true_avg_acceptance": sum(all_acceptance_lengths) / len(all_acceptance_lengths) if all_acceptance_lengths else 0.0,
        "max_acceptance": max_acceptance_len,  # just accepted tokens, no anchor
        "max_accepted": max_acceptance_len,
        "consensus_accepts": consensus_accepts,
        "consensus_total": max(consensus_total, 1),
        "consensus_rate": consensus_accepts / max(consensus_total, 1),
        "time_ms": elapsed_ms,
        "offset_accepted": offset_accepted,
        "offset_tested": offset_tested,
        "K": K,
    }

    return output, stats, token_spans


# ── pure AR baseline generation ──────────────────────────────────────────────

@torch.no_grad()
def generate_ar_only(
    model: OrthrusQwen35Model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    stream: bool = False,
) -> Tuple[str, dict]:
    """
    Pure autoregressive generation using ONLY the frozen base model.
    No diffusion, no consensus — baseline sequential decoding.

    Returns (generated_text, stats_dict).
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    t_start = time.perf_counter()

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_len = prompt_len + max_new_tokens

    # Prefill + get first token logits
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    base_outputs = model.base_model(input_ids=input_ids, position_ids=position_ids, use_cache=True, logits_to_keep=1)
    past_kv = base_outputs.past_key_values
    first_logits = base_outputs.logits[:, -1, :]
    first_token, _ = sample_token(first_logits, temperature)
    first_token_id = first_token.item()

    generated_ids = [first_token_id]
    total_forward_passes = 1  # prefill

    if stream:
        t = tokenizer.decode([first_token_id], skip_special_tokens=True)
        if t:
            sys.stdout.write(f"{YELLOW}{t}{RESET}")
            sys.stdout.flush()

    if first_token_id == eos_id:
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output, {
            "tokens_generated": 1, "forward_passes": 1, "tpf": 1.0,
            "time_ms": 0,
        }

    current_len = prompt_len

    while current_len < max_len - 1:
        # Single-token forward pass
        current_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
        position_ids = torch.tensor([[current_len]], dtype=torch.long, device=device)

        base_outputs = model.base_model(
            input_ids=current_token,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = base_outputs.past_key_values
        logits = base_outputs.logits[:, -1, :]
        total_forward_passes += 1

        next_token, _ = sample_token(logits, temperature)
        next_token_id = next_token.item()
        generated_ids.append(next_token_id)
        current_len += 1

        if stream:
            t = tokenizer.decode([next_token_id], skip_special_tokens=True)
            if t:
                sys.stdout.write(t)
                sys.stdout.flush()

        if next_token_id == eos_id:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = {
        "tokens_generated": len(generated_ids),
        "forward_passes": total_forward_passes,
        "tpf": len(generated_ids) / max(total_forward_passes, 1),
        "time_ms": elapsed_ms,
    }
    return output, stats


# ── display helpers ──────────────────────────────────────────────────────────

def print_colored_output(token_spans: List[TokenSpan], tokenizer):
    """Print the full output with green highlights for consensus-accepted tokens."""
    print(f"\n{BOLD}── Generated ──{RESET}")
    yellow_count = 0
    for span in token_spans:
        text = tokenizer.decode([span.token_id], skip_special_tokens=True)
        if not text:
            continue
            
        # Make whitespaces visible
        display_text = text.replace('\n', '\\n\n').replace('\r', '\\r\r').replace('\t', '\\t').replace(' ', '·')
            
        if span.is_max_block:
            yellow_count += 1
            print(f"{YELLOW}{display_text}{RESET}", end="")
        elif span.accepted:
            print(f"{GREEN}{display_text}{RESET}", end="")
        else:
            print(display_text, end="")
    print()
    print(f"{DIM}[max block tokens: {yellow_count}]{RESET}")


def print_comparison(orthrus_stats: dict, ar_stats: dict, orthrus_output: str, ar_output: str,
                    token_spans: List[TokenSpan] = None, tokenizer=None):
    """Print side-by-side comparison of Orthrus vs pure AR with both outputs."""
    ot = orthrus_stats["time_ms"]
    at = ar_stats["time_ms"]
    speedup = at / max(ot, 0.1)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Orthrus vs Pure AR — Comparison{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Timing
    print(f"\n  {BOLD}Timing:{RESET}")
    print(f"    {CYAN}Orthrus{RESET}:  {ot:,.0f} ms  ({orthrus_stats['forward_passes']} passes, "
          f"TPF={orthrus_stats['tpf']:.2f}×)")
    print(f"    {YELLOW}AR-only{RESET}:  {at:,.0f} ms  ({ar_stats['forward_passes']} passes, "
          f"TPF={ar_stats['tpf']:.2f}×)")
    print(f"    {GREEN}Speedup: {speedup:.1f}×{RESET}")

    # Tokens
    print(f"\n  {BOLD}Tokens:{RESET}")
    print(f"    Orthrus: {orthrus_stats['tokens_generated']} tokens generated")
    print(f"    AR-only: {ar_stats['tokens_generated']} tokens generated")

    # Consensus detail
    # Consensus detail
    if "consensus_rate" in orthrus_stats:
        K = orthrus_stats["K"]
        avg_accept_len = orthrus_stats["avg_acceptance"]
        true_avg_len = orthrus_stats["true_avg_acceptance"]
        accept_rate = true_avg_len / (K - 1)
        
        offset_accepted = orthrus_stats["offset_accepted"]
        total_blocks = len(orthrus_stats['acceptance_lengths'])
        offset_rates = [
            (offset_accepted[k] / total_blocks) if total_blocks > 0 else 0.0
            for k in range(1, K)
        ]
        
        off_str_parts = []
        for k, rate in enumerate(offset_rates, 1):
            if rate > 0:
                off_str_parts.append(f"{k}:{rate:.0%}")
        off_str = " ".join(off_str_parts)

        print(f"\n  {BOLD}Consensus:{RESET}")
        print(f"    rate: {accept_rate:.2%} | avg_len: {avg_accept_len:.1f}/{K} | "
              f"blocks: {len(orthrus_stats['acceptance_lengths'])} | off: {off_str}")

    # Output match
    match = orthrus_output.strip() == ar_output.strip()
    if match:
        print(f"\n  {GREEN}✓ Outputs match exactly (lossless){RESET}")
    else:
        print(f"\n  {YELLOW}⚠ Outputs differ (expected for T>0 sampling){RESET}")

    # ── Generated outputs ───────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {CYAN}Orthrus (green = consensus-accepted){RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    if token_spans and tokenizer:
        yellow_count = 0
        for span in token_spans:
            text = tokenizer.decode([span.token_id], skip_special_tokens=True)
            if not text:
                continue
            if span.is_max_block:
                yellow_count += 1
                print(f"{YELLOW}{text}{RESET}", end="")
            elif span.accepted:
                print(f"{GREEN}{text}{RESET}", end="")
            else:
                print(text, end="")
        print()
        print(f"\n{DIM}  {GREEN}Green{RESET}{DIM} = accepted  |  {YELLOW}Red{RESET}{DIM} = max block ({orthrus_stats.get('max_acceptance', 0)} tokens) [actual: {yellow_count}]{RESET}")
    else:
        print(orthrus_output)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {YELLOW}AR-only (baseline sequential){RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    print(ar_output)

    print()


def print_stats(stats: dict):
    """Print standard stats block."""
    K = stats["K"]
    avg_accept_len = stats["avg_acceptance"]
    true_avg_len = stats["true_avg_acceptance"]
    accept_rate = true_avg_len / (K - 1)
    
    offset_accepted = stats["offset_accepted"]
    total_blocks = len(stats['acceptance_lengths'])
    offset_rates = [
        (offset_accepted[k] / total_blocks) if total_blocks > 0 else 0.0
        for k in range(1, K)
    ]
    
    off_str_parts = []
    for k, rate in enumerate(offset_rates, 1):
        if rate > 0:
            off_str_parts.append(f"{k}:{rate:.0%}")
    off_str = " ".join(off_str_parts)

    msg = (f"rate: {accept_rate:.2%} | avg_len: {avg_accept_len:.1f}/{K} | "
           f"TPF: {stats['tpf']:.1f}× | blocks: {total_blocks} | "
           f"gen_toks: {stats['tokens_generated']} | off: {off_str} <<<")
           
    print(f"\n{DIM}>>> Accept @ end | {msg}{RESET}")


# ── model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, base_model: str = "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B",
               K: int = 32, deterministic: bool = False) -> Tuple[OrthrusQwen35Model, AutoTokenizer]:
    """Load Orthrus model with trained diffusion heads."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"{DIM}Loading tokenizer...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Add <mask> token if not present
    if "<mask>" not in tokenizer.get_vocab():
        print(f"{DIM}Adding <mask> token...{RESET}")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})

    print(f"{DIM}Loading base model ({base_model})...{RESET}")
    model = OrthrusQwen35Model(
        base_model_path=base_model,
        block_size=K,
        dtype=dtype,
    )
    
    # Initialize mask token embedding if it was just added
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    old_vocab_size = 248077  # Original Qwen3.5 vocab size
    if mask_id >= old_vocab_size:
        print(f"{DIM}Initializing <mask> embedding from mean of all embeddings{RESET}")
        with torch.no_grad():
            all_embeddings = model.base_model.model.embed_tokens.weight[:old_vocab_size]
            mean_embedding = all_embeddings.mean(dim=0)
            model.base_model.model.embed_tokens.weight[mask_id] = mean_embedding
            
            all_lm_head = model.base_model.lm_head.weight[:old_vocab_size]
            mean_lm_head = all_lm_head.mean(dim=0)
            model.base_model.lm_head.weight[mask_id] = mean_lm_head

    print(f"{DIM}Loading diffusion heads from {checkpoint_path}...{RESET}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device=device)
    model.eval()

    model.base_model = model.base_model.to(dtype=torch.float32)
    model.diffusion_heads = model.diffusion_heads.to(dtype=torch.float32)

    mode = "float32 (bit-exact batch/sequential)"
    if deterministic:
        mode += " + math SDPA"

    print(f"{GREEN}✓ Model loaded.{RESET} "
          f"Trainable params: {model.trainable_params:,}  "
          f"Block size K={K}")
    print(f"  {mode}")

    print()

    return model, tokenizer


# ── interactive REPL ─────────────────────────────────────────────────────────

def interactive_loop(model, tokenizer, args):
    """Interactive prompt → generate → repeat loop."""
    mode = "Orthrus + AR comparison" if args.compare else "Orthrus"
    print(f"{BOLD}Orthrus SmolLM2 Interactive Inference ({mode}){RESET}")
    print(f"Type a prompt and press Enter. Ctrl+C to interrupt generation.")
    print(f"Type 'exit', 'quit', or Ctrl+D to quit.\n")

    while True:
        try:
            prompt = input(f"{BOLD}>>> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print()
        try:
            # Auto-wrap in chat template for instruct behavior
            wrapped = prompt
            if '<|im_start|>' not in prompt:
                wrapped = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True
                )

            # Always run Orthrus colored generation
            orthrus_output, orthrus_stats, spans = generate_colored(
                model=model,
                tokenizer=tokenizer,
                prompt=wrapped,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                verbose=args.verbose,
                stream=True,
                debug=args.debug,
                deterministic=args.deterministic,
            )
            print()

            if args.compare:
                # Run pure AR baseline with same seed for fair comparison
                print(f"{DIM}── Running AR-only baseline... ──{RESET}")
                ar_output, ar_stats = generate_ar_only(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=wrapped,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    stream=True,
                )
                print_comparison(orthrus_stats, ar_stats, orthrus_output, ar_output,
                                 token_spans=spans, tokenizer=tokenizer)
            else:
                print_stats(orthrus_stats)

        except KeyboardInterrupt:
            print(f"\n{DIM}── Interrupted ──{RESET}\n")
            continue

        print()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orthrus SmolLM2 Interactive Inference with consensus visualization"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="../checkpoints/step_6000/diffusion_heads.pt",
                        help="Path to diffusion_heads.pt checkpoint")
    parser.add_argument("--base_model", type=str,
                        default="F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B")
    parser.add_argument("--prompt", type=str, default=None,
                        help="One-shot prompt (omit for interactive REPL)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Read prompt from a text file")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--K", type=int, default=32, help="Block size")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-block acceptance details")
    parser.add_argument("--compare", action="store_true",
                        help="Run pure AR baseline alongside Orthrus for comparison")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for AR-only baseline reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Show per-block consensus debug details")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive REPL loop instead of running a one-shot prompt")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use math SDPA backend for bit-exact batch vs sequential")
    args = parser.parse_args()

    # Resolve checkpoint path relative to this script
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Specify with --checkpoint PATH")
        sys.exit(1)

    model, tokenizer = load_model(
        checkpoint_path=ckpt_path,
        base_model=args.base_model,
        K=args.K,
        deterministic=args.deterministic,
    )

    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            args.prompt = f.read().strip()

    if args.interactive:
        # Interactive REPL
        interactive_loop(model, tokenizer, args)
    else:
        if not args.prompt:
            print(f"{DIM}── Loading standard evaluation prompt from dataset... ──{RESET}")
            try:
                from data import load_orthrus_dataset
                val_ds, text_key = load_orthrus_dataset(
                    dataset_name="HuggingFaceTB/smoltalk",
                    config_name="all",
                    split="test",
                    max_samples=1,
                    text_key="text",
                    tokenizer=tokenizer,
                )
                if val_ds and len(val_ds) > 0:
                    text = val_ds[0][text_key]
                    input_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                    assistant_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                    n_assist = len(assistant_tokens)
                    found_idx = -1
                    for j in range(len(input_ids) - n_assist + 1):
                        if input_ids[j:j+n_assist] == assistant_tokens:
                            found_idx = j + n_assist
                            break
                    
                    if found_idx != -1:
                        prompt_ids = input_ids[:found_idx]
                        args.prompt = tokenizer.decode(prompt_ids)
                    else:
                        args.prompt = tokenizer.decode(input_ids[:256])
                else:
                    args.prompt = "Could not load eval dataset."
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                args.prompt = "Once upon a time in a faraway land,"

        if args.prompt:
            # Wrap in chat template for instruct behavior (if it wasn't pre-formatted)
            prompt = args.prompt
            if '<|im_start|>' not in prompt:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    tokenize=False, add_generation_prompt=True
                )

            # One-shot mode
            orthrus_output, orthrus_stats, spans = generate_colored(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                verbose=args.verbose,
                stream=True,
                debug=args.debug,
                deterministic=args.deterministic,
            )
            print()

            if args.compare:
                print(f"{DIM}── Running AR-only baseline... ──{RESET}")
                ar_output, ar_stats = generate_ar_only(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    stream=True,
                )
                print_comparison(orthrus_stats, ar_stats, orthrus_output, ar_output,
                                 token_spans=spans, tokenizer=tokenizer)
            else:
                print_stats(orthrus_stats)
