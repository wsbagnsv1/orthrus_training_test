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
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
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
    logits: Tensor,                      # [vocab], [L, vocab], or [B, L, vocab]
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Sample token(s) from logits. Handles any batch shape.

    Returns (token_ids, probs).
    probs is None for greedy (temperature=0).
    """
    if temperature < 1e-5:
        return logits.argmax(dim=-1), None

    # Scale by temperature
    scaled = logits / temperature

    # Top-k (applied to last dim)
    if top_k > 0:
        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1)
        scaled[scaled < v[..., [-1]]] = -float("inf")

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        scaled[indices_to_remove] = -float("inf")

    probs = F.softmax(scaled, dim=-1)
    flat_probs = probs.view(-1, probs.size(-1))
    tokens = torch.multinomial(flat_probs, 1).view(probs.shape[:-1])
    return tokens, probs


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
    Generate text using Orthrus consensus decoding (matches reference OrthrusLM.generate).

    Returns (generated_text, stats_dict).
    stats includes: tokens_generated, forward_passes, tpf, acceptance_lengths.
    """
    model.eval()
    device = next(model.parameters()).device
    K = model.block_size
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    eos_id = tokenizer.eos_token_id

    # Deterministic: same seed for greedy decoding
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ── Tokenize prompt ─────────────────────────────────────────────────────
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_len = prompt_len + max_new_tokens

    # Allocate output buffer
    output_ids = torch.full((1, max_len + K), mask_id, dtype=torch.long, device=device)
    output_ids[:, :prompt_len] = input_ids

    # ── AR Prefill (shared KV cache) ────────────────────────────────────────
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    base_out = model.base_model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    past_key_values = base_out.past_key_values

    # First token from prefill logits
    first_logits = base_out.logits[:, -1, :]
    next_token, _ = sample_token(first_logits, temperature, top_k, top_p)
    next_token_id = next_token.item()

    start_idx = prompt_len
    output_ids[:, start_idx] = next_token_id

    if next_token_id == eos_id:
        output = tokenizer.decode([next_token_id], skip_special_tokens=True)
        return output, {
            "tokens_generated": 1, "forward_passes": 1, "tpf": 1.0,
            "acceptance_lengths": [], "avg_acceptance": 0.0,
        }

    total_forward_passes = 1  # prefill
    all_acceptance_lengths: List[int] = []

    if verbose:
        print(f"[Prefill] {prompt_len} tokens, first token: {tokenizer.decode([next_token_id])!r}")

    # ── Generation loop ─────────────────────────────────────────────────────
    while start_idx < max_len - 1:
        diff_len = min(K, max_len - start_idx)

        # Build diffusion block: anchor at position 0, masks at 1..K-1
        diff_block_ids = torch.full((1, diff_len), mask_id, dtype=torch.long, device=device)
        diff_block_ids[:, 0] = output_ids[:, start_idx]
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)

        # Step 1: Diffusion parallel projection (no cache update)
        causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
        for k in range(diff_len):
            causal_limit[0, k] = start_idx - 1

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
            diff_tokens, diff_probs = sample_token(diff_logits[:, :-1, :], temperature, top_k, top_p)
        else:
            diff_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            diff_probs = None

        proposed_block = torch.cat([output_ids[:, start_idx:start_idx+1], diff_tokens], dim=1)

        # Step 2: AR verification — batch pass for consensus only
        ar_outputs = model.base_model(
            input_ids=proposed_block,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        ar_logits = ar_outputs.logits  # [1, K, vocab]
        ar_tokens, ar_probs = sample_token(ar_logits, temperature, top_k, top_p)
        total_forward_passes += 1

        # Step 3: Consensus
        # ar_tokens[0, k] and diff_tokens[0, k] both predict position start_idx + k + 1
        if diff_tokens.shape[1] > 0:
            if temperature < 1e-5:
                matches = (diff_tokens == ar_tokens[:, :-1])
                acceptance_len = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
                next_token_corrected = ar_tokens[:, acceptance_len:acceptance_len+1]
            else:
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
            acceptance_len = 0
            next_token_corrected = ar_tokens[:, :1]

        all_acceptance_lengths.append(acceptance_len + 1)

        # Step 4: Commit accepted tokens (float32 AR = bit-exact, no replay)
        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, :acceptance_len + 1]
        output_ids[:, start_idx:end_idx] = accepted_block
        past_key_values.crop(end_idx)

        # Check EOS in accepted block
        eos_in_block = (accepted_block[:, 1:] == eos_id).any()
        if eos_in_block:
            for i in range(1, acceptance_len + 1):
                if int(accepted_block[0, i].item()) == eos_id:
                    start_idx = start_idx + i + 1
                    break
            break

        start_idx = end_idx

        next_tok_id = int(next_token_corrected[0, 0].item())
        if start_idx < max_len:
            output_ids[:, start_idx] = next_token_corrected
            if next_tok_id == eos_id:
                break

        if verbose:
            accepted_text = tokenizer.decode(
                accepted_block[0, 1:acceptance_len+1].tolist(), skip_special_tokens=True
            )
            print(f"  [Block] accepted {acceptance_len + 1}/{proposed_block.shape[1]}: {accepted_text!r}")

    # ── Decode ──────────────────────────────────────────────────────────────
    generated_ids = output_ids[0, prompt_len:start_idx + 1].tolist()
    generated_ids = [t for t in generated_ids if t != mask_id]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = {
        "tokens_generated": len(generated_ids),
        "forward_passes": total_forward_passes,
        "tpf": len(generated_ids) / max(total_forward_passes, 1),
        "acceptance_lengths": all_acceptance_lengths,
        "avg_acceptance": (
            sum(all_acceptance_lengths) / len(all_acceptance_lengths)
            if all_acceptance_lengths
            else 0.0
        ),
        "max_acceptance": max(all_acceptance_lengths) if all_acceptance_lengths else 0,
    }

    if verbose:
        print(f"\n[Stats] TPF={stats['tpf']:.2f}, "
              f"avg_acceptance={stats['avg_acceptance']:.1f}, "
              f"passes={total_forward_passes}")

    return output, stats


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthrus SmolLM2 generation")
    parser.add_argument("--checkpoint", type=str,
                        default="../checkpoints/step_6000/diffusion_heads.pt",
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

    # Resolve checkpoint path relative to this script
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

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
    print(f"Loading diffusion heads from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device=device)
    model.eval()

    model.base_model = model.base_model.to(dtype=torch.float32)
    model.diffusion_heads = model.diffusion_heads.to(dtype=torch.float32)

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
