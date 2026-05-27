"""
ortho_attention.py - Custom Triton attention kernel with fused masking.

Replaces FlexAttention for DiffusionFullAttention heads:
- AR causal mask: kv_idx <= causal_limit[b, q_idx]
- Block isolation: diffusion tokens attend only within their block
- GQA native support (Q=32 heads, K/V=8 heads)

~4x faster than PyTorch SDPA, ~5x less VRAM.
Correctness: max diff 0.005 vs PyTorch autograd (bf16 tolerance).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ortho_fwd_kernel(
    Q, K, V, O, M, L, causal_limit,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cl_b, stride_cl_s,
    B, H, Nq, Nkv, D: tl.constexpr,
    ar_seq_len, block_size: tl.constexpr,
    scale,
    gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Forward: fused attention with AR causal + block isolation masking."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H
    kv_head = head // gqa_ratio

    q_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_off < Nq
    q_blocks = q_off // block_size

    cl_ptrs = causal_limit + batch * stride_cl_b + q_off * stride_cl_s
    cl = tl.load(cl_ptrs, mask=q_mask, other=Nkv)

    q_ptrs = Q + batch * stride_qb + head * stride_qh + q_off[:, None] * stride_qs + tl.arange(0, D)[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_prev = tl.full([BLOCK_M], value=-3.4e38, dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    for kv_start in range(0, Nkv, BLOCK_N):
        kv_off = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_off < Nkv

        k_ptrs = K + batch * stride_kb + kv_head * stride_kh + kv_off[:, None] * stride_ks + tl.arange(0, D)[None, :]
        v_ptrs = V + batch * stride_vb + kv_head * stride_vh + kv_off[:, None] * stride_vs + tl.arange(0, D)[None, :]
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale

        # Fused masking
        is_ar = kv_off[None, :] < ar_seq_len
        ar_masked = is_ar & (kv_off[None, :] > cl[:, None])
        kv_diff = kv_off[None, :] - ar_seq_len
        is_diff = kv_diff >= 0
        blk_masked = is_diff & (q_blocks[:, None] != (kv_diff // block_size))
        mask = ar_masked | blk_masked | ~kv_mask[None, :]
        scores = tl.where(mask, -3.4e38, scores)

        # Online softmax (FlashAttention-style)
        m_cur = tl.max(scores, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        exp_prev = tl.exp(m_prev - m_new)
        exp_cur = tl.exp(scores - m_new[:, None])
        l_new = exp_prev * l_prev + tl.sum(exp_cur, axis=1)
        acc = exp_prev[:, None] * acc + tl.dot(exp_cur.to(v.dtype), v)
        m_prev = m_new
        l_prev = l_new

    out = acc / tl.maximum(l_prev[:, None], 1e-10)

    # Store M (max) and L (log-sum-exp) for backward
    m_ptrs = M + batch * H * Nq + head * Nq + q_off
    tl.store(m_ptrs, m_prev, mask=q_mask)
    lse = m_prev + tl.log(tl.maximum(l_prev, 1e-10))
    l_ptrs = L + batch * H * Nq + head * Nq + q_off
    tl.store(l_ptrs, lse, mask=q_mask)

    o_ptrs = O + batch * stride_ob + head * stride_oh + q_off[:, None] * stride_os + tl.arange(0, D)[None, :]
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=q_mask[:, None])


@triton.jit
def _ortho_bwd_dq_kernel(
    Q, K, V, O, DO, DQ, M, L, causal_limit,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cl_b, stride_cl_s,
    B, H, Nq, Nkv, D: tl.constexpr,
    ar_seq_len, block_size: tl.constexpr,
    scale,
    gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Compute dQ: dQ = (P * (dP - Di)) @ K"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H
    kv_head = head // gqa_ratio

    q_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_off < Nq
    q_blocks = q_off // block_size

    l_ptrs = L + batch * H * Nq + head * Nq + q_off
    l = tl.load(l_ptrs, mask=q_mask, other=0.0)

    cl_ptrs = causal_limit + batch * stride_cl_b + q_off * stride_cl_s
    cl = tl.load(cl_ptrs, mask=q_mask, other=Nkv)

    q_ptrs = Q + batch * stride_qb + head * stride_qh + q_off[:, None] * stride_qs + tl.arange(0, D)[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    do_ptrs = DO + batch * stride_ob + head * stride_oh + q_off[:, None] * stride_os + tl.arange(0, D)[None, :]
    do = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    o_ptrs = O + batch * stride_ob + head * stride_oh + q_off[:, None] * stride_os + tl.arange(0, D)[None, :]
    o = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Di = sum(DO * O, dim=-1) [global, computed once]
    di = tl.sum(do * o, axis=1)

    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for kv_start in range(0, Nkv, BLOCK_N):
        kv_off = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_off < Nkv

        k_ptrs = K + batch * stride_kb + kv_head * stride_kh + kv_off[:, None] * stride_ks + tl.arange(0, D)[None, :]
        v_ptrs = V + batch * stride_vb + kv_head * stride_vh + kv_off[:, None] * stride_vs + tl.arange(0, D)[None, :]
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale

        is_ar = kv_off[None, :] < ar_seq_len
        ar_masked = is_ar & (kv_off[None, :] > cl[:, None])
        kv_diff = kv_off[None, :] - ar_seq_len
        is_diff = kv_diff >= 0
        blk_masked = is_diff & (q_blocks[:, None] != (kv_diff // block_size))
        mask = ar_masked | blk_masked | ~kv_mask[None, :]
        scores = tl.where(mask, -3.4e38, scores)

        p = tl.exp(scores - l[:, None])
        p = tl.where(mask, 0.0, p)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None])
        dq_acc += tl.dot(ds.to(k.dtype), k)

    dq_ptrs = DQ + batch * stride_qb + head * stride_qh + q_off[:, None] * stride_qs + tl.arange(0, D)[None, :]
    tl.store(dq_ptrs, (dq_acc * scale).to(DQ.dtype.element_ty), mask=q_mask[:, None])


@triton.jit
def _ortho_bwd_dkdv_kernel(
    Q, K, V, O, DO, DK, DV, M, L, causal_limit,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_cl_b, stride_cl_s,
    B, H, Nq, Nkv, D: tl.constexpr,
    ar_seq_len, block_size: tl.constexpr,
    scale,
    gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Compute dK, dV: iterate over Q blocks, accumulate dK/dV."""
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H
    kv_head = head // gqa_ratio

    kv_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kv_mask = kv_off < Nkv

    k_ptrs = K + batch * stride_kb + kv_head * stride_kh + kv_off[:, None] * stride_ks + tl.arange(0, D)[None, :]
    v_ptrs = V + batch * stride_vb + kv_head * stride_vh + kv_off[:, None] * stride_vs + tl.arange(0, D)[None, :]
    k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

    dk_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    for q_start in range(0, Nq, BLOCK_M):
        q_off = q_start + tl.arange(0, BLOCK_M)
        q_mask = q_off < Nq
        q_blocks = q_off // block_size

        l_ptrs = L + batch * H * Nq + head * Nq + q_off
        l = tl.load(l_ptrs, mask=q_mask, other=0.0)

        cl_ptrs = causal_limit + batch * stride_cl_b + q_off * stride_cl_s
        cl = tl.load(cl_ptrs, mask=q_mask, other=Nkv)

        q_ptrs = Q + batch * stride_qb + head * stride_qh + q_off[:, None] * stride_qs + tl.arange(0, D)[None, :]
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        do_ptrs = DO + batch * stride_ob + head * stride_oh + q_off[:, None] * stride_os + tl.arange(0, D)[None, :]
        do = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        o_ptrs = O + batch * stride_ob + head * stride_oh + q_off[:, None] * stride_os + tl.arange(0, D)[None, :]
        o = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        # Di = sum(DO * O, dim=-1) [global, not per-block]
        di = tl.sum(do * o, axis=1)

        scores = tl.dot(q, tl.trans(k)) * scale

        is_ar = kv_off[None, :] < ar_seq_len
        ar_masked = is_ar & (kv_off[None, :] > cl[:, None])
        kv_diff = kv_off[None, :] - ar_seq_len
        is_diff = kv_diff >= 0
        blk_masked = is_diff & (q_blocks[:, None] != (kv_diff // block_size))
        mask = ar_masked | blk_masked | ~kv_mask[None, :]
        scores = tl.where(mask, -3.4e38, scores)

        p = tl.exp(scores - l[:, None])
        p = tl.where(mask, 0.0, p)

        dv_acc += tl.dot(tl.trans(p), do.to(p.dtype))

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None])
        dk_acc += tl.dot(tl.trans(ds), q)

    dk_ptrs = DK + batch * stride_kb + kv_head * stride_kh + kv_off[:, None] * stride_ks + tl.arange(0, D)[None, :]
    dv_ptrs = DV + batch * stride_vb + kv_head * stride_vh + kv_off[:, None] * stride_vs + tl.arange(0, D)[None, :]
    tl.atomic_add(dk_ptrs, (dk_acc * scale).to(DK.dtype.element_ty), mask=kv_mask[:, None])
    tl.atomic_add(dv_ptrs, dv_acc.to(DV.dtype.element_ty), mask=kv_mask[:, None])


# ─── Python wrappers ────────────────────────────────────────────────────────

class OrthoAttention(torch.autograd.Function):
    """Custom attention with fused AR causal + block isolation masking."""

    @staticmethod
    def forward(ctx, Q, K, V, causal_limit, ar_seq_len, block_size):
        B, H, Nq, D = Q.shape
        Nkv = K.shape[2]
        gqa_ratio = H // K.shape[1]

        O = torch.empty_like(Q)
        M = torch.empty(B, H, Nq, device=Q.device, dtype=torch.float32)
        L = torch.empty(B, H, Nq, device=Q.device, dtype=torch.float32)

        scale = D ** -0.5
        BLOCK_M, BLOCK_N = 16, 16
        grid = (triton.cdiv(Nq, BLOCK_M), B * H)

        _ortho_fwd_kernel[grid](
            Q, K, V, O, M, L, causal_limit,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            causal_limit.stride(0), causal_limit.stride(1),
            B, H, Nq, Nkv, D,
            ar_seq_len, block_size,
            scale, gqa_ratio,
            BLOCK_M, BLOCK_N,
        )

        ctx.save_for_backward(Q, K, V, O, M, L, causal_limit)
        ctx.ar_seq_len = ar_seq_len
        ctx.block_size = block_size

        return O

    @staticmethod
    def backward(ctx, DO):
        Q, K, V, O, M, L, causal_limit = ctx.saved_tensors
        ar_seq_len = ctx.ar_seq_len
        block_size = ctx.block_size

        B, H, Nq, D = Q.shape
        Nkv = K.shape[2]
        gqa_ratio = H // K.shape[1]

        DQ = torch.zeros_like(Q)
        DK = torch.zeros_like(K)
        DV = torch.zeros_like(V)

        scale = D ** -0.5
        BLOCK_M, BLOCK_N = 16, 16

        # Compute dQ
        grid_q = (triton.cdiv(Nq, BLOCK_M), B * H)
        _ortho_bwd_dq_kernel[grid_q](
            Q, K, V, O, DO, DQ, M, L, causal_limit,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            causal_limit.stride(0), causal_limit.stride(1),
            B, H, Nq, Nkv, D,
            ar_seq_len, block_size,
            scale, gqa_ratio,
            BLOCK_M, BLOCK_N,
        )

        # Compute dK, dV
        grid_kv = (triton.cdiv(Nkv, BLOCK_N), B * H)
        _ortho_bwd_dkdv_kernel[grid_kv](
            Q, K, V, O, DO, DK, DV, M, L, causal_limit,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            causal_limit.stride(0), causal_limit.stride(1),
            B, H, Nq, Nkv, D,
            ar_seq_len, block_size,
            scale, gqa_ratio,
            BLOCK_M, BLOCK_N,
        )

        return DQ, DK, DV, None, None, None


def ortho_attention(Q, K, V, causal_limit, ar_seq_len, block_size):
    """Drop-in replacement for FlexAttention with same masking."""
    return OrthoAttention.apply(Q, K, V, causal_limit, ar_seq_len, block_size)
