"""
Triton fused block-attention kernels for Orthrus diffusion heads.
Drop-in replacement for flex_attention — 5-6x faster with enough GPU occupancy.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _has_triton = True
except ImportError:
    _has_triton = False


if _has_triton:
    @triton.jit
    def _triton_fwd_kernel(
        Q, KS, VS, KA, VA, LIMITS, OUT, LSE,
        B_G: tl.constexpr, H_G: tl.constexpr, HKV: tl.constexpr, NG: tl.constexpr,
        NB: tl.constexpr, BG: tl.constexpr, BK: tl.constexpr, DH: tl.constexpr,
        AL: tl.constexpr, BN: tl.constexpr,
    ):
        """One program per (batch, head, block_group)."""
        pid = tl.program_id(0)
        total_hg = H_G * BG
        b = pid // total_hg
        rem = pid % total_hg
        h = rem // BG
        blk_start = (rem % BG) * (NB // BG)
        blk_end = tl.minimum(blk_start + (NB // BG), NB)
        hkv = h // NG
        stride_qb = H_G * NB * BK * DH
        stride_qh = NB * BK * DH
        stride_kvb = HKV * NB * BK * DH
        stride_kvh = NB * BK * DH
        stride_blk = BK * DH
        stride_kvab = HKV * AL * DH
        stride_kvah = AL * DH
        om = tl.arange(0, BK); od = tl.arange(0, DH); on = tl.arange(0, BN)
        ss = 1.0 / tl.sqrt(DH * 1.0)
        for blk in range(blk_start, blk_end):
            limit = tl.load(LIMITS + b * (NB * BK) + blk * BK)
            limit = tl.minimum(tl.maximum(limit, -1), AL - 1)
            total_kv = BK + limit + 1
            qb = Q + b * stride_qb + h * stride_qh + blk * stride_blk
            ksb = KS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            vsb = VS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            kab = KA + b * stride_kvab + hkv * stride_kvah
            vab = VA + b * stride_kvab + hkv * stride_kvah
            acc = tl.zeros([BK, DH], dtype=tl.float32)
            m = tl.full([BK], float('-inf'), dtype=tl.float32)
            l = tl.zeros([BK], dtype=tl.float32)
            for ks in range(0, total_kv, BN):
                ko = ks + on; km = ko < total_kv; iss = ko < BK; iar = ko >= BK
                k_t = (tl.load(ksb + ko[:, None] * DH + od[None, :], mask=(km & iss)[:, None], other=0.0)
                     + tl.load(kab + (ko - BK)[:, None] * DH + od[None, :], mask=(km & iar)[:, None], other=0.0))
                qv = tl.load(qb + om[:, None] * DH + od[None, :])
                sc = tl.dot(qv, tl.trans(k_t)) * ss
                sc = tl.where(km[None, :], sc, float('-inf'))
                mn = tl.maximum(m, tl.max(sc, axis=1)); al = tl.exp(m - mn)
                p = tl.exp(sc - mn[:, None]); l = l * al + tl.sum(p, axis=1); m = mn
                v_t = (tl.load(vsb + ko[:, None] * DH + od[None, :], mask=(km & iss)[:, None], other=0.0)
                     + tl.load(vab + (ko - BK)[:, None] * DH + od[None, :], mask=(km & iar)[:, None], other=0.0))
                acc = acc * al[:, None] + tl.dot(p.to(v_t.dtype), v_t)
            acc = acc / l[:, None]; lse_val = m + tl.log(l)
            ob = OUT + b * stride_qb + h * stride_qh + blk * stride_blk
            tl.store(ob + om[:, None] * DH + od[None, :], acc.to(Q.dtype.element_ty))
            lb = LSE + b * (H_G * NB * BK) + h * (NB * BK) + blk * BK
            tl.store(lb + om, lse_val)

    @triton.jit
    def _triton_bwd_kernel(
        Q, KS, VS, KA, VA, LIMITS, DO, OUT, LSE, DQ, DKS, DVS,
        B_G: tl.constexpr, H_G: tl.constexpr, HKV: tl.constexpr, NG: tl.constexpr,
        NB: tl.constexpr, BG: tl.constexpr, BK: tl.constexpr, DH: tl.constexpr,
        AL: tl.constexpr, BN: tl.constexpr,
    ):
        """One program per (batch, head, block_group)."""
        pid = tl.program_id(0)
        total_hg = H_G * BG
        b = pid // total_hg
        rem = pid % total_hg
        h = rem // BG
        blk_start = (rem % BG) * (NB // BG)
        blk_end = tl.minimum(blk_start + (NB // BG), NB)
        hkv = h // NG
        stride_qb = H_G * NB * BK * DH
        stride_qh = NB * BK * DH
        stride_kvb = HKV * NB * BK * DH
        stride_kvh = NB * BK * DH
        stride_blk = BK * DH
        stride_kvab = HKV * AL * DH
        stride_kvah = AL * DH
        om = tl.arange(0, BK); od = tl.arange(0, DH); on = tl.arange(0, BN)
        ss = 1.0 / tl.sqrt(DH * 1.0)
        for blk in range(blk_start, blk_end):
            limit = tl.load(LIMITS + b * (NB * BK) + blk * BK)
            limit = tl.minimum(tl.maximum(limit, -1), AL - 1)
            total_kv = BK + limit + 1
            qb = Q + b * stride_qb + h * stride_qh + blk * stride_blk
            ksb = KS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            vsb = VS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            kab = KA + b * stride_kvab + hkv * stride_kvah
            vab = VA + b * stride_kvab + hkv * stride_kvah
            dob = DO + b * stride_qb + h * stride_qh + blk * stride_blk
            ob = OUT + b * stride_qb + h * stride_qh + blk * stride_blk
            dqb = DQ + b * stride_qb + h * stride_qh + blk * stride_blk
            dksb = DKS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            dvsb = DVS + b * stride_kvb + hkv * stride_kvh + blk * stride_blk
            lb = LSE + b * (H_G * NB * BK) + h * (NB * BK) + blk * BK
            lse = tl.load(lb + om)
            qv = tl.load(qb + om[:, None] * DH + od[None, :])
            dov = tl.load(dob + om[:, None] * DH + od[None, :])
            ov = tl.load(ob + om[:, None] * DH + od[None, :])
            Di = tl.sum(dov * ov, axis=1)
            dq_acc = tl.zeros([BK, DH], dtype=tl.float32)
            for ks in range(0, total_kv, BN):
                ko = ks + on; km = ko < total_kv; iss = ko < BK; iar = ko >= BK
                k_t = (tl.load(ksb + ko[:, None] * DH + od[None, :], mask=(km & iss)[:, None], other=0.0)
                     + tl.load(kab + (ko - BK)[:, None] * DH + od[None, :], mask=(km & iar)[:, None], other=0.0))
                v_t = (tl.load(vsb + ko[:, None] * DH + od[None, :], mask=(km & iss)[:, None], other=0.0)
                     + tl.load(vab + (ko - BK)[:, None] * DH + od[None, :], mask=(km & iar)[:, None], other=0.0))
                sc = tl.dot(qv, tl.trans(k_t)) * ss
                sc = tl.where(km[None, :], sc, float('-inf'))
                p = tl.exp(sc - lse[:, None])
                dv_t = tl.dot(tl.trans(p.to(dov.dtype)), dov)
                dv_t = tl.where(km[:, None], dv_t, 0.0)
                tl.store(dvsb + ko[:, None] * DH + od[None, :], dv_t.to(Q.dtype.element_ty),
                         mask=(km & iss)[:, None])
                dp = tl.dot(dov, tl.trans(v_t.to(dov.dtype)))
                ds = p * (dp - Di[:, None])
                ds = tl.where(km[None, :], ds, 0.0)
                dq_acc += tl.dot(ds.to(k_t.dtype), k_t)
                dk_t = tl.dot(tl.trans(ds), qv.to(tl.float32)) * ss
                dk_t = tl.where(km[:, None], dk_t, 0.0)
                tl.store(dksb + ko[:, None] * DH + od[None, :], dk_t.to(Q.dtype.element_ty),
                         mask=(km & iss)[:, None])
            dq_acc *= ss
            tl.store(dqb + om[:, None] * DH + od[None, :], dq_acc.to(Q.dtype.element_ty))

    class _TritonBlockAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k_self, v_self, k_ar, v_ar, limits,
                    B, H, Hkv, ng, BG, n_blocks, K, ar_len, D):
            q_f = q.contiguous(); k_f = k_self.contiguous(); v_f = v_self.contiguous()
            ka_f = k_ar.contiguous(); va_f = v_ar.contiguous()
            lim_f = limits.contiguous()
            out = torch.empty_like(q_f)
            lse = torch.empty(B * H, n_blocks * K, device=q.device, dtype=torch.float32)
            _triton_fwd_kernel[(B * H * BG,)](q_f, k_f, v_f, ka_f, va_f, lim_f, out, lse,
                B_G=B, H_G=H, HKV=Hkv, NG=ng, NB=n_blocks, BG=BG, BK=K, DH=D, AL=ar_len, BN=32,
                num_warps=8, num_stages=2)
            ctx.save_for_backward(q_f, k_f, v_f, ka_f, va_f, lim_f, out, lse)
            ctx.B, ctx.H, ctx.Hkv, ctx.ng, ctx.BG, ctx.nb, ctx.K, ctx.al, ctx.D = \
                B, H, Hkv, ng, BG, n_blocks, K, ar_len, D
            return out

        @staticmethod
        def backward(ctx, do):
            q_f, k_f, v_f, ka_f, va_f, lim_f, out_f, lse = ctx.saved_tensors
            B, H, Hkv, ng, BG, nb, K, al, D = \
                ctx.B, ctx.H, ctx.Hkv, ctx.ng, ctx.BG, ctx.nb, ctx.K, ctx.al, ctx.D
            do_f = do.contiguous()
            dq = torch.empty_like(q_f); dk = torch.empty_like(k_f); dv = torch.empty_like(v_f)
            _triton_bwd_kernel[(B * H * BG,)](q_f, k_f, v_f, ka_f, va_f, lim_f,
                do_f, out_f, lse, dq, dk, dv,
                B_G=B, H_G=H, HKV=Hkv, NG=ng, NB=nb, BG=BG, BK=K, DH=D, AL=al, BN=32,
                num_warps=8, num_stages=2)
            return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


def triton_block_attention(q, k_self, v_self, k_ar, v_ar, limits,
                            B, H, Hkv, ng, BG, n_blocks, K, ar_len, D):
    """Fused Triton block-attention — B×H×BG grid."""
    if not _has_triton:
        raise RuntimeError("Triton not available")
    return _TritonBlockAttention.apply(
        q, k_self, v_self, k_ar, v_ar, limits,
        B, H, Hkv, ng, BG, n_blocks, K, ar_len, D)
