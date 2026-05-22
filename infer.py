"""
Interactive CLI inference for OrthrusSmolLM2 with color-coded consensus visualization.

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

from model import OrthrusSmolLM2

# ── ANSI color codes ─────────────────────────────────────────────────────────
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
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
    model: OrthrusSmolLM2,
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
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
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
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    base_out = model.base_model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    past_key_values = base_out.past_key_values

    # First token from prefill logits
    first_logits = base_out.logits[:, -1, :]
    next_token, _ = sample_token(first_logits, temperature)
    next_token_id = next_token.item()

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

        # Sample from positions 0..K-2 → predictions for positions 1..K-1
        if diff_len > 1:
            diff_tokens, diff_probs = sample_token(diff_logits[:, :-1, :], temperature)
            # diff_tokens: [1, K-1]
        else:
            diff_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            diff_probs = None

        # Proposed block = anchor + diffusion predictions
        proposed_block = torch.cat([output_ids[:, start_idx:start_idx+1], diff_tokens], dim=1)

        # ── Step 2: AR verification — batch pass for consensus only ──
        # (Batch AR may have bf16 drift; KVs will be discarded after consensus.)
        ar_outputs = model.base_model(
            input_ids=proposed_block,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        ar_logits = ar_outputs.logits  # [1, K, vocab]
        ar_tokens, ar_probs = sample_token(ar_logits, temperature)  # [1, K]
        total_forward_passes += 1

        # ── Step 3: Consensus ────────────────────────────────────────────
        # ar_tokens[0, k] predicts position start_idx + k + 1
        # diff_tokens[0, k] predicts position start_idx + k + 1 too
        # So diff_tokens should match ar_tokens[:, :-1]
        if diff_tokens.shape[1] > 0:
            if temperature < 1e-5:
                matches = (diff_tokens == ar_tokens[:, :-1])  # [1, K-1]
                acceptance_len = int(matches.cumprod(dim=1).sum(dim=1)[0].item())

                # Track per-token consensus for coloring
                for i in range(acceptance_len):
                    consensus_accepts += 1
                    consensus_total += 1
                if acceptance_len < diff_tokens.shape[1]:
                    consensus_total += 1  # the rejected one
                rejected_at = acceptance_len if acceptance_len < diff_tokens.shape[1] else -1

                next_token_corrected = ar_tokens[:, acceptance_len:acceptance_len+1]
            else:
                matches = None
                acceptance_len = 0
                for i in range(diff_tokens.shape[1]):
                    q_prob = diff_probs[0, i, diff_tokens[0, i]] if diff_probs is not None else 1.0
                    p_prob = ar_probs[0, i, diff_tokens[0, i]] if ar_probs is not None else 1.0
                    if torch.rand(1, device=device).item() < min(1.0, (p_prob / max(q_prob, 1e-8)).item()):
                        acceptance_len += 1
                    else:
                        break
                if acceptance_len < diff_tokens.shape[1]:
                    p_dist = ar_probs[0, acceptance_len]
                    residual = torch.clamp(p_dist - diff_probs[0, acceptance_len], min=0.0)
                    residual_sum = residual.sum()
                    next_token_corrected = torch.multinomial(
                        residual / residual_sum if residual_sum > 1e-5 else p_dist, 1
                    ).unsqueeze(0)
                else:
                    next_token_corrected = ar_tokens[:, acceptance_len:acceptance_len+1]
        else:
            matches = None
            acceptance_len = 0
            next_token_corrected = ar_tokens[:, :1]

        all_acceptance_lengths.append(acceptance_len + 1)  # +1 for anchor

        # ── Debug: show consensus details ───────────────────────────────
        if debug and diff_tokens.shape[1] > 0:
            sys.stderr.write(f"\n{DIM}[Block {block_num} debug] start_idx={start_idx}{RESET}\n")
            sys.stderr.write(f"  Proposed ({diff_tokens.shape[1]} tokens): ")
            for i, tid in enumerate(diff_tokens[0].tolist()):
                t = tokenizer.decode([tid], skip_special_tokens=True)
                marker = "✓" if i < acceptance_len else ("✗" if i == acceptance_len else " ")
                sys.stderr.write(f"{marker}{t!r} ")
            sys.stderr.write(f"\n  AR tokens  ({ar_tokens.shape[1]-1} predictions): ")
            for i, tid in enumerate(ar_tokens[0, :-1].tolist()):
                t = tokenizer.decode([tid], skip_special_tokens=True)
                marker = "✓" if i < acceptance_len else ("✗" if i == acceptance_len else " ")
                sys.stderr.write(f"{marker}{t!r} ")
            sys.stderr.write(f"\n  Matches: {matches[0].tolist() if matches is not None else 'N/A'}")
            sys.stderr.write(f"\n  Accepted: {acceptance_len}, Correction: ")
            sys.stderr.write(f"{tokenizer.decode([int(next_token_corrected[0, 0].item())], skip_special_tokens=True)!r}")

            # Diagnostic: run pure AR one-token-at-a-time from this start_idx
            # to compare what the model SHOULD predict vs what batch-AR predicts
            if block_num <= 3:
                sys.stderr.write(f"\n  {YELLOW}Pure-AR trace from position {start_idx}:{RESET}\n")
                diag_pkv = DynamicCache()
                diag_ids = output_ids[:, :start_idx].clone()
                diag_pos = torch.arange(start_idx, device=device).unsqueeze(0)
                diag_out = model.base_model(input_ids=diag_ids, position_ids=diag_pos, use_cache=True)
                diag_pkv = diag_out.past_key_values
                diag_next = diag_out.logits[:, -1, :].argmax(dim=-1).item()
                sys.stderr.write(f"    pos {start_idx}: {tokenizer.decode([diag_next], skip_special_tokens=True)!r}")
                for diag_i in range(1, min(6, diff_tokens.shape[1] + 2)):
                    tok_t = torch.tensor([[diag_next]], dtype=torch.long, device=device)
                    pos_t = torch.tensor([[start_idx + diag_i - 1]], dtype=torch.long, device=device)
                    diag_out = model.base_model(input_ids=tok_t, position_ids=pos_t, past_key_values=diag_pkv, use_cache=True)
                    diag_pkv = diag_out.past_key_values
                    diag_next = diag_out.logits[:, -1, :].argmax(dim=-1).item()
                    sys.stderr.write(f" → {tokenizer.decode([diag_next], skip_special_tokens=True)!r}")
                sys.stderr.write(f"\n    Batch-AR said: ")
                for diag_i in range(min(6, ar_tokens.shape[1])):
                    bt = tokenizer.decode([int(ar_tokens[0, diag_i].item())], skip_special_tokens=True)
                    sys.stderr.write(f" → {bt!r}")
                sys.stderr.write(f"\n")
            sys.stderr.write(f"\n")

        # ── Step 4: Commit accepted tokens (reference approach) ─────────
        # float32 AR backbone → batch KVs are bit-exact, no replay needed.
        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, :acceptance_len + 1]
        output_ids[:, start_idx:end_idx] = accepted_block
        past_key_values.crop(end_idx)

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
                    sys.stdout.write(f"{GREEN}{text}{RESET}")
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
                    sys.stdout.write(corr_text)
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
        "avg_acceptance": sum(all_acceptance_lengths) / len(all_acceptance_lengths) if all_acceptance_lengths else 0,
        "max_acceptance": max_acceptance_len,  # just accepted tokens, no anchor
        "max_accepted": max_acceptance_len,
        "consensus_accepts": consensus_accepts,
        "consensus_total": max(consensus_total, 1),
        "consensus_rate": consensus_accepts / max(consensus_total, 1),
        "time_ms": elapsed_ms,
    }

    return output, stats, token_spans


# ── pure AR baseline generation ──────────────────────────────────────────────

@torch.no_grad()
def generate_ar_only(
    model: OrthrusSmolLM2,
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
    base_outputs = model.base_model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
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
    for span in token_spans:
        text = tokenizer.decode([span.token_id], skip_special_tokens=True)
        if not text:
            continue
        if span.accepted:
            print(f"{GREEN}{text}{RESET}", end="")
        else:
            print(text, end="")
    print()


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
    if "consensus_rate" in orthrus_stats:
        print(f"\n  {BOLD}Consensus:{RESET}")
        print(f"    Accepts:  {orthrus_stats['consensus_accepts']}/{orthrus_stats['consensus_total']} "
              f"({orthrus_stats['consensus_rate']:.1%})")
        print(f"    Avg acc:  {orthrus_stats['avg_acceptance']:.1f} tokens/block")
        print(f"    Max acc:  {orthrus_stats['max_acceptance']} tokens/block")

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
        for span in token_spans:
            text = tokenizer.decode([span.token_id], skip_special_tokens=True)
            if not text:
                continue
            if span.is_max_block:
                print(f"{YELLOW}{text}{RESET}", end="")
            elif span.accepted:
                print(f"{GREEN}{text}{RESET}", end="")
            else:
                print(text, end="")
        print()
        print(f"\n{DIM}  {GREEN}Green{RESET}{DIM} = accepted  |  {YELLOW}Yellow{RESET}{DIM} = max block ({orthrus_stats.get('max_acceptance', 0)} tokens){RESET}")
    else:
        print(orthrus_output)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {YELLOW}AR-only (baseline sequential){RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    print(ar_output)

    print()


def print_stats(stats: dict):
    """Print generation statistics."""
    print(f"\n{BOLD}── Stats ──{RESET}")
    print(f"  Tokens generated:    {stats['tokens_generated']}")
    print(f"  Forward passes:      {stats['forward_passes']}")
    print(f"  TPF (tokens/pass):   {stats['tpf']:.2f}×")
    print(f"  Avg acceptance len:  {stats['avg_acceptance']:.1f} / K")
    print(f"  Max acceptance len:  {stats['max_acceptance']} / K")
    print(f"  Consensus rate:      {stats['consensus_rate']:.1%} "
          f"({stats['consensus_accepts']}/{stats['consensus_total']})")
    print(f"  {GREEN}Green{RESET} = diffusion predicted correctly, accepted by AR consensus")
    print(f"  White = AR-corrected or baseline AR token")


# ── model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
               K: int = 32, deterministic: bool = False) -> Tuple[OrthrusSmolLM2, AutoTokenizer]:
    """Load Orthrus model with trained diffusion heads."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"{DIM}Loading tokenizer...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if "<mask>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})

    print(f"{DIM}Loading base model ({base_model})...{RESET}")
    model = OrthrusSmolLM2(
        base_model_name=base_model,
        block_size=K,
        dtype=dtype,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))

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
                        default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--prompt", type=str, default=None,
                        help="One-shot prompt (omit for interactive REPL)")
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

    if args.prompt:
        # Wrap in chat template for instruct behavior
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
    else:
        # Interactive REPL
        interactive_loop(model, tokenizer, args)
