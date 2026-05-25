"""
Causality / future-token leakage test for Orthrus Qwen3.5.

The ONLY correct way to test for future-token leakage:
  1. Build two AR sequences that are IDENTICAL up to the anchor but
     DIFFERENT after the anchor.
  2. Extract per-block caches from both.
  3. Run forward_diffusion on both.
  4. Assert outputs are BIT-FOR-BIT identical.
     Any difference = the model saw tokens it shouldn't have.

We test this at THREE levels:
  A. Cache extraction: do extracted linear/conv states differ?
  B. Full forward: do diffusion logits differ?
  C. Intra-block causality: within a single block, does changing token i
     affect the output at position j < i?
"""

import sys
sys.path.insert(0, '.')
import torch
from model import OrthrusQwen35Model
from linear_states import get_per_block_caches

device = 'cuda'
dtype = torch.float32
K = 32  # block_size

print("=" * 70)
print("FUTURE-TOKEN LEAKAGE TEST")
print("=" * 70)

# ── Load model ─────────────────────────────────────────────────────────────
m = OrthrusQwen35Model(
    base_model_path='F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B',
    block_size=K, dtype=dtype,
).to(device=device, dtype=dtype)
m.eval()
print(f"Model: {m.trainable_params:,} trainable params\n")

# ═══════════════════════════════════════════════════════════════════════════
# TEST A: Cache extraction — future AR tokens must not affect caches
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST A: Per-block cache extraction (future AR tokens)")
print("=" * 70)

torch.manual_seed(42)
# 256 tokens total, 2 blocks at anchors 64 and 96
# Need prefix to cover max anchor (96) so both blocks see identical past
ar_prefix = torch.randint(0, 1000, (1, 96), device=device)
ar_suffix_a = torch.randint(0, 1000, (1, 160), device=device)
ar_suffix_b = torch.randint(0, 1000, (1, 160), device=device)

# Guarantee suffixes are actually different
ar_suffix_b[:, :] = ar_suffix_a + 1

ids_a = torch.cat([ar_prefix, ar_suffix_a], dim=1)  # [1, 256]
ids_b = torch.cat([ar_prefix, ar_suffix_b], dim=1)  # [1, 256]

anchors = torch.tensor([[64]], device=device)  # single block, anchor at 64

with torch.no_grad():
    ls_a, fakv_a, laconv_a = get_per_block_caches(m, ids_a, None, anchors)
    ls_b, fakv_b, laconv_b = get_per_block_caches(m, ids_b, None, anchors)

print("\n  Linear (recurrent) states:")
all_cache_ok = True
for li in sorted(set(ls_a.keys()) | set(ls_b.keys())):
    diff = (ls_a[li] - ls_b[li]).abs().max().item()
    ok = diff == 0.0
    all_cache_ok = all_cache_ok and ok
    sym = "✓" if ok else "✗ LEAK!"
    print(f"    L{li:2d}: max diff = {diff:.12f}  {sym}")

print("\n  Conv states:")
for li in sorted(set(laconv_a.keys()) | set(laconv_b.keys())):
    for blk in sorted(set(laconv_a[li].keys()) | set(laconv_b[li].keys())):
        ca = laconv_a[li][blk]
        cb = laconv_b[li][blk]
        if ca is None and cb is None:
            continue
        diff = (ca - cb).abs().max().item()
        ok = diff == 0.0
        all_cache_ok = all_cache_ok and ok
        sym = "✓" if ok else "✗ LEAK!"
        print(f"    L{li:2d} blk{blk}: max diff = {diff:.12f}  {sym}")

print(f"\n  TEST A RESULT: {'PASS ✓' if all_cache_ok else 'FAIL ✗ — FUTURE TOKENS LEAKED INTO CACHES'}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST B: Full forward — future AR tokens must not affect diffusion output
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST B: Full diffusion forward (future AR tokens)")
print("=" * 70)

torch.manual_seed(99)
diff_ids = torch.randint(0, 1000, (1, 2 * K), device=device)  # 2 blocks
anchors_multi = torch.tensor([[64, 96]], device=device)

with torch.no_grad():
    # Prefill with both AR sequences
    kv_a, _, _, _ = m.forward_ar_prefill(ids_a)
    kv_b, _, _, _ = m.forward_ar_prefill(ids_b)

    # Build causal_limit: block0 at 64 sees 0..63, block1 at 96 sees 0..95
    cl = torch.tensor([[63, 95]], dtype=torch.long, device=device).repeat_interleave(K, dim=-1)

    ls_a2, _, laconv_a2 = get_per_block_caches(m, ids_a, None, anchors_multi)

    # Run diffusion forward with caches extracted from sequence A
    logits_a = m.forward_diffusion(
        diff_input_ids=diff_ids,
        ar_past_key_values=kv_a,
        ar_seq_len=128,
        causal_limit=cl,
        linear_states=ls_a2,
        block_indices=torch.tensor([0,1], device=device),
        per_block_la_conv=laconv_a2,
    )

    # Re-extract caches for B
    ls_b2, _, laconv_b2 = get_per_block_caches(m, ids_b, None, anchors_multi)

    logits_b = m.forward_diffusion(
        diff_input_ids=diff_ids,
        ar_past_key_values=kv_b,
        ar_seq_len=128,
        causal_limit=cl,
        linear_states=ls_b2,
        block_indices=torch.tensor([0,1], device=device),
        per_block_la_conv=laconv_b2,
    )

logit_diff = (logits_a - logits_b).abs().max().item()
argmax_match = (logits_a.argmax(-1) == logits_b.argmax(-1)).all().item()
forward_ok = logit_diff == 0.0

print(f"\n  Logit max diff:  {logit_diff:.12f}")
print(f"  Argmax identical: {argmax_match}")
print(f"\n  TEST B RESULT: {'PASS ✓' if forward_ok else 'FAIL ✗ — FUTURE AR TOKENS LEAKED INTO DIFFUSION OUTPUT'}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST C: Intra-block causality — within a block, token i must not see j>i
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST C: Intra-block causality (within-block future token leakage)")
print("=" * 70)

# Strategy: Run the same block twice. In version B, change the LAST token
# in the diffusion block. All outputs at positions 0..K-2 must be identical.
# Only position K-1 is allowed to differ (it sees its own changed input via
# self-attention, but that's fine — it's at the same position).

torch.manual_seed(200)
ar_clean = torch.randint(0, 1000, (1, 128), device=device)
diff_base = torch.randint(0, 1000, (1, K), device=device)
diff_mod = diff_base.clone()
diff_mod[0, -1] = (diff_base[0, -1] + 500) % 1000  # change last token

anchors_c = torch.tensor([[64]], device=device)

with torch.no_grad():
    ls_c, _, laconv_c = get_per_block_caches(m, ar_clean, None, anchors_c)
    kv_c, _, _, _ = m.forward_ar_prefill(ar_clean)

    cl_c = torch.full((1, K), 63, dtype=torch.long, device=device)

    out_base = m.forward_diffusion(
        diff_input_ids=diff_base,
        ar_past_key_values=kv_c,
        ar_seq_len=64,
        causal_limit=cl_c,
        linear_states=ls_c,
        block_indices=torch.tensor([0], device=device),
        per_block_la_conv=laconv_c,
    )

    # Re-extract (cleared after forward)
    ls_c2, _, laconv_c2 = get_per_block_caches(m, ar_clean, None, anchors_c)

    out_mod = m.forward_diffusion(
        diff_input_ids=diff_mod,
        ar_past_key_values=kv_c,
        ar_seq_len=64,
        causal_limit=cl_c,
        linear_states=ls_c2,
        block_indices=torch.tensor([0], device=device),
        per_block_la_conv=laconv_c2,
    )

# Check positions 0..K-2: must be identical
prefix_diff = (out_base[0, :-1, :] - out_mod[0, :-1, :]).abs().max().item()
last_diff = (out_base[0, -1, :] - out_mod[0, -1, :]).abs().max().item()
intra_ok = prefix_diff == 0.0

print(f"\n  Positions 0..{K-2} max diff: {prefix_diff:.12f}")
print(f"  Position {K-1} max diff:    {last_diff:.12f}  (expected to differ)")
print(f"\n  TEST C RESULT: {'PASS ✓' if intra_ok else 'FAIL ✗ — INTRA-BLOCK FUTURE TOKEN LEAKAGE DETECTED'}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
all_pass = all_cache_ok and forward_ok and intra_ok
print(f"  Test A (cache extraction):     {'PASS ✓' if all_cache_ok else 'FAIL ✗'}")
print(f"  Test B (full forward):         {'PASS ✓' if forward_ok else 'FAIL ✗'}")
print(f"  Test C (intra-block causality): {'PASS ✓' if intra_ok else 'FAIL ✗'}")
print(f"\n  OVERALL: {'ALL TESTS PASSED ✓ — ZERO FUTURE TOKEN LEAKAGE' if all_pass else 'LEAKAGE DETECTED ✗'}")
print("=" * 70)
