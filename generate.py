"""
Consensus-based generation for OrthrusSmolLM2.

Implements the exact intra-model consensus mechanism from the Orthrus paper
(§3.3, Figure 2b):

  Step 1: Diffusion head projects K candidate tokens in a single parallel pass.
  Step 2: AR head validates left-to-right via rejection sampling.
  Step 3: Accepted tokens' KV states are appended to the shared cache.

This guarantees the output distribution exactly matches the frozen base model
(greedy match for T=0; rejection sampling for T>0).
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

# ── ensure local package is on path ──────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import OrthrusSmolLM2


# ── sampling helpers ─────────────────────────────────────────────────────────

def sample_token(
    logits: Tensor,                      # [1, vocab]
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Sample a single token from logits.

    Returns (token_id, probs).
    probs is None for greedy (temperature=0).
    """
    if temperature < 1e-5:
        return logits.argmax(dim=-1), None

    # Scale by temperature
    scaled = logits / temperature

    # Top-k
    if top_k > 0:
        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled[scaled < v[..., [-1]]] = -float("inf")

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        scaled[indices_to_remove] = -float("inf")

    probs = F.softmax(scaled, dim=-1)
    token = torch.multinomial(probs, 1)
    return token, probs


# ── main generation loop ────────────────────────────────────────────────────

@torch.no_grad()
def generate_orthrus(
    model: OrthrusSmolLM2,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    verbose: bool = False,
    stream: bool = False,
) -> Tuple[str, dict]:
    """
    Generate text using Orthrus consensus decoding.

    Returns (generated_text, stats_dict).
    stats includes: tokens_generated, forward_passes, tpf, acceptance_lengths.
    """
    model.eval()
    device = next(model.parameters()).device
    K = model.block_size
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    eos_id = tokenizer.eos_token_id

    # ── Tokenize prompt ─────────────────────────────────────────────────────
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_len = prompt_len + max_new_tokens

    # ── AR Prefill ──────────────────────────────────────────────────────────
    ar_kv_cache, ar_logits = model.forward_ar_prefill(input_ids)
    past_key_values = ar_kv_cache  # DynamicCache, we'll extend it

    # First token (from AR prefill)
    next_token, _ = sample_token(ar_logits[:, -1, :], temperature, top_k, top_p)
    next_token_id = next_token.item()

    generated_ids: List[int] = [next_token_id]
    if next_token_id == eos_id:
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output, {
            "tokens_generated": 1,
            "forward_passes": 1,
            "tpf": 1.0,
            "acceptance_lengths": [1],
        }

    current_len = prompt_len
    total_forward_passes = 1  # prefill
    total_tokens = 0
    all_acceptance_lengths: List[int] = []

    if verbose:
        print(f"[Prefill] {prompt_len} tokens, first token: {tokenizer.decode([next_token_id])!r}")

    # ── Generation loop ─────────────────────────────────────────────────────
    while current_len < max_len - 1:
        remaining = max_len - current_len
        diff_len = min(K, remaining)
        actual_block_size = diff_len

        # ── Step 1: Build diffusion block ───────────────────────────────────
        anchor_token = generated_ids[-1]
        diff_block_ids = torch.full(
            (1, diff_len), mask_id, dtype=torch.long, device=device
        )
        diff_block_ids[:, 0] = anchor_token

        diff_position_ids = torch.arange(
            current_len, current_len + diff_len, device=device
        ).unsqueeze(0)

        # ── Step 2: Diffusion parallel projection ───────────────────────────
        # Build causal_limit: anchor token sees up to current_len-1;
        # mask positions k see up to current_len + k - 1
        causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
        causal_limit[0, 0] = current_len - 1   # anchor sees AR up to itself
        for k in range(1, diff_len):
            causal_limit[0, k] = current_len + k - 1

        # We need AR prefix length for the diffusion forward
        # Build fresh DynamicCache with current past_key_values
        diff_outputs = model(
            input_ids=diff_block_ids,
            is_diffusion_pass=True,
            ar_past_key_values=past_key_values,
            ar_seq_len=current_len,
            causal_limit=causal_limit,
        )
        # diff_outputs is logits: [1, diff_len, vocab]

        total_forward_passes += 1

        # Sample from diffusion predictions (all positions except we reuse anchor)
        if diff_len > 1:
            diff_tokens = []
            diff_probs_list = []
            for k in range(1, diff_len):  # skip anchor (position 0)
                tok, prob = sample_token(
                    diff_outputs[:, k, :], temperature, top_k, top_p
                )
                diff_tokens.append(tok.item())
                diff_probs_list.append(prob)
            # Prepend anchor
            proposed_block_ids = [anchor_token] + diff_tokens
            diff_probs = (
                torch.cat(diff_probs_list, dim=0)
                if diff_probs_list and diff_probs_list[0] is not None
                else None
            )
        else:
            proposed_block_ids = [anchor_token]
            diff_probs = None

        proposed_block = torch.tensor(
            [proposed_block_ids], dtype=torch.long, device=device
        )

        # ── Step 3: AR verification ─────────────────────────────────────────
        ar_position_ids = torch.arange(
            current_len, current_len + len(proposed_block_ids), device=device
        ).unsqueeze(0)

        ar_outputs = model.base_model(
            input_ids=proposed_block,
            position_ids=ar_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        ar_logits = ar_outputs.logits  # [1, block_len, vocab]
        updated_cache = ar_outputs.past_key_values

        total_forward_passes += 1

        # ── Step 4: Consensus verification ──────────────────────────────────
        # AR logit at position k predicts token at k+1.
        # So ar_logits[0, k] should match proposed_block_ids[k+1] for k=0..K-2.
        # The anchor token (proposed_block_ids[0]) is always accepted (identity).
        accepted: List[int] = [proposed_block_ids[0]]  # anchor always accepted
        block_len = ar_logits.shape[1]

        for k in range(1, block_len):  # k = 1..K-1, the mask positions
            proposed_tok = proposed_block_ids[k]
            ar_pred_logit = ar_logits[0, k - 1]  # predicts token at position k

            if temperature < 1e-5:
                # Greedy: exact match required
                ar_pred = ar_pred_logit.argmax().item()
                if proposed_tok == ar_pred:
                    accepted.append(proposed_tok)
                else:
                    accepted.append(ar_pred)
                    break
            else:
                # Rejection sampling (Leviathan et al. 2022)
                p_ar = F.softmax(ar_pred_logit / temperature, dim=-1)

                if diff_probs is None:
                    accepted.append(proposed_tok)
                    continue

                # diff_probs[k-1] is the diffusion distribution for position k
                p_diff = diff_probs[k - 1]
                p_ar_tok = p_ar[proposed_tok].item()
                p_diff_tok = p_diff[0, proposed_tok].item()

                accept_prob = min(1.0, p_ar_tok / max(p_diff_tok, 1e-8))
                if torch.rand(1, device=device).item() < accept_prob:
                    accepted.append(proposed_tok)
                else:
                    # Sample correction from residual distribution
                    residual = (p_ar - p_diff[0]).clamp(min=0)
                    residual_sum = residual.sum()
                    if residual_sum > 1e-8:
                        correction = torch.multinomial(
                            residual / residual_sum, 1
                        ).item()
                    else:
                        correction = torch.multinomial(p_ar, 1).item()
                    accepted.append(correction)
                    break

            # Check for EOS
            if accepted[-1] == eos_id:
                break

        acceptance_len = len(accepted)
        all_acceptance_lengths.append(acceptance_len)

        if verbose:
            accepted_text = tokenizer.decode(accepted, skip_special_tokens=True)
            print(f"  [Block] accepted {acceptance_len}/{block_len}: {accepted_text!r}")

        # ── Step 5: Commit accepted tokens, update cache ────────────────────
        generated_ids.extend(accepted)
        current_len += acceptance_len

        # Truncate cache to current length
        past_key_values = updated_cache
        past_key_values.crop(current_len)

        total_tokens += acceptance_len

        # Check for EOS
        if eos_id in accepted:
            break

    # ── Decode ──────────────────────────────────────────────────────────────
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stats = {
        "tokens_generated": total_tokens,
        "forward_passes": total_forward_passes,
        "tpf": total_tokens / max(total_forward_passes, 1),
        "acceptance_lengths": all_acceptance_lengths,
        "avg_acceptance": (
            sum(all_acceptance_lengths) / len(all_acceptance_lengths)
            if all_acceptance_lengths
            else 0.0
        ),
    }

    if verbose:
        print(f"\n[Stats] TPF={stats['tpf']:.2f}, "
              f"avg_acceptance={stats['avg_acceptance']:.1f}, "
              f"passes={total_forward_passes}")

    return output, stats


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthrus SmolLM2 generation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to diffusion_heads.pt checkpoint")
    parser.add_argument("--base_model", type=str,
                        default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of recursion in programming.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--K", type=int, default=32, help="Block size")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Ensure <mask> token
    if "<mask>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})

    # Load model
    print(f"Loading model from {args.base_model}...")
    model = OrthrusSmolLM2(
        base_model_name=args.base_model,
        block_size=args.K,
        dtype=dtype,
    )
    model.base_model.resize_token_embeddings(len(tokenizer))

    # Load trained diffusion heads
    print(f"Loading diffusion heads from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device=device)
    model.eval()

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    output, stats = generate_orthrus(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verbose=args.verbose,
    )
    print(f"\nGenerated:\n{output}")
    print(f"\nStats: TPF={stats['tpf']:.2f}, "
          f"avg_accept={stats['avg_acceptance']:.1f}, "
          f"tokens={stats['tokens_generated']}, "
          f"passes={stats['forward_passes']}")
