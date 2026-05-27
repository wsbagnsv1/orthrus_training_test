import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp, exp2
from fla.ops.utils.softplus import softplus
from fla.utils import input_guard
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_GATE_IN_KERNEL': lambda args: args['A_log'] is not None,
    'HAS_DT_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_inference_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    h_out,  # NEW: Output intermediate states
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
                b_g = -exp(b_A) * softplus(b_g)
                b_h *= exp(b_g)
            elif USE_EXP2:
                b_h *= exp2(b_g)
            else:
                b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gk[None, :])
                else:
                    b_h *= exp2(b_gk[:, None])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gk[None, :])
                else:
                    b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gv[:, None])
                else:
                    b_h *= exp2(b_gv[None, :])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gv[:, None])
                else:
                    b_h *= exp(b_gv[None, :])

        if TRANSPOSE_STATE:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # NEW: Store intermediate state `b_h` to `h_out` at step `i_t`
        # Layout of h_out: [B, T, HV, K, V] if not TRANSPOSE_STATE
        # In IS_VARLEN, i_t is the relative offset. Total sequences: N.
        # The offset for this specific token globally is (bos + i_t).
        t_global = bos + i_t
        t_global = bos + i_t
        offset_h_out = (t_global.to(tl.int64) * HV + i_hv) * K * V
        if TRANSPOSE_STATE:
            p_h_out = h_out + offset_h_out + o_v[:, None] * K + o_k[None, :]
        else:
            p_h_out = h_out + offset_h_out + o_k[:, None] * V + o_v[None, :]
            
        tl.store(p_h_out, b_h.to(p_h_out.dtype.element_ty), mask=mask_h)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K*V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

@input_guard
def fused_recurrent_inference_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    if output_final_state:
        if transpose_state_layout:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    # NEW: h_out to store intermediate states
    total_tokens = B * T if cu_seqlens is None else cu_seqlens[-1].item()
    if transpose_state_layout:
        h_out = q.new_empty(total_tokens, HV, V, K, dtype=torch.float32)
    else:
        h_out = q.new_empty(total_tokens, HV, K, V, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_inference_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        h_out=h_out,  # NEW
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=1,
        num_stages=3,
    )
    
    # Reshape h_out if inputs were [B, T, ...]
    if cu_seqlens is None:
        if transpose_state_layout:
            h_out = h_out.view(B, T, HV, V, K)
        else:
            h_out = h_out.view(B, T, HV, K, V)
            
    return o, final_state, h_out

if __name__ == "__main__":
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule
    
    # Simple test script
    B, T, H, HV, K, V_dim = 1, 32, 12, 2, 128, 128
    
    import torch.nn.functional as F
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
    v = torch.randn(B, T, HV, V_dim, dtype=torch.bfloat16, device='cuda')
    g = F.logsigmoid(torch.randn(B, T, HV, dtype=torch.bfloat16, device='cuda'))
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device='cuda').sigmoid()
    h0 = torch.randn(B, HV, K, V_dim, dtype=torch.float32, device='cuda')
    
    import time
    
    # Warmup
    for _ in range(3):
        o_ref, ht_ref = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True
        )
        o_test, ht_test, h_out = fused_recurrent_inference_fwd(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True
        )
        
    print("o_ref matches o_test:", torch.allclose(o_ref, o_test, atol=1e-3))
    print("ht_ref matches ht_test:", torch.allclose(ht_ref, ht_test, atol=1e-3))
    print("h_out[:, -1] matches ht_ref:", torch.allclose(h_out[:, -1], ht_ref, atol=1e-3))
    
    # Benchmarking
    t0 = time.perf_counter()
    for _ in range(100):
        o_ref, ht_ref = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True
        )
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 100 * 1000
    
    t0 = time.perf_counter()
    for _ in range(100):
        o_test, ht_test, h_out = fused_recurrent_inference_fwd(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True
        )
    torch.cuda.synchronize()
    t_test = (time.perf_counter() - t0) / 100 * 1000
    
    print(f"Ref time: {t_ref:.3f} ms")
    print(f"Inference time: {t_test:.3f} ms")
