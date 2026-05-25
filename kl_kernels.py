import torch
import triton
import triton.language as tl

# ==============================================================
# V1: 1D Vector Triton Kernel (Slow, GEMV)
# ==============================================================
@triton.jit
def _fused_kl_gemm_fwd_bwd_kernel_v1(
    x_s_ptr, x_t_ptr, w_ptr,
    loss_out_ptr, grad_x_s_out_ptr,
    stride_xs_n, stride_xs_d,
    stride_xt_n, stride_xt_d,
    stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_n = tl.program_id(0)
    m_s = -float('inf')
    m_t = -float('inf')
    sum_s = 0.0
    sum_t = 0.0
    
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        s_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        t_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_s = tl.load(x_s_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d, mask=mask_d, other=0.0)
            x_t = tl.load(x_t_ptr + pid_n * stride_xt_n + offs_d * stride_xt_d, mask=mask_d, other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            s_logits += tl.sum(w * x_s[None, :], axis=1)
            t_logits += tl.sum(w * x_t[None, :], axis=1)
        s_logits = tl.where(mask_v, s_logits / temperature, -float('inf'))
        t_logits = tl.where(mask_v, t_logits / temperature, -float('inf'))
        
        m_s_new = tl.maximum(m_s, tl.max(s_logits, axis=0))
        sum_s = sum_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s_logits - m_s_new), axis=0)
        m_s = m_s_new
        
        m_t_new = tl.maximum(m_t, tl.max(t_logits, axis=0))
        sum_t = sum_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t_logits - m_t_new), axis=0)
        m_t = m_t_new

    kl_sum = 0.0
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        s_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        t_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_s = tl.load(x_s_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d, mask=mask_d, other=0.0)
            x_t = tl.load(x_t_ptr + pid_n * stride_xt_n + offs_d * stride_xt_d, mask=mask_d, other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            s_logits += tl.sum(w * x_s[None, :], axis=1)
            t_logits += tl.sum(w * x_t[None, :], axis=1)
            
        s_logits /= temperature
        t_logits /= temperature
        
        t_prob = tl.where(mask_v, tl.exp(t_logits - m_t) / sum_t, 0.0)
        s_prob = tl.where(mask_v, tl.exp(s_logits - m_s) / sum_s, 0.0)
        s_logprob = s_logits - m_s - tl.log(sum_s)
        
        kl_term = tl.where(t_prob > 0.0, t_prob * (tl.log(tl.where(t_prob > 0, t_prob, 1.0)) - s_logprob), 0.0)
        kl_sum += tl.sum(kl_term, axis=0)
        
        grad_s_logits = (s_prob - t_prob) / temperature
        
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            grad_x_s_block = tl.sum(grad_s_logits[:, None] * w, axis=0)
            
            grad_x_s_ptrs = grad_x_s_out_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d
            old_grad = tl.load(grad_x_s_ptrs, mask=mask_d, other=0.0)
            tl.store(grad_x_s_ptrs, old_grad + grad_x_s_block, mask=mask_d)
            
    tl.store(loss_out_ptr + pid_n, kl_sum)

class V1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, w, temp=1.0):
        N, D = s.shape
        V, _ = w.shape
        loss_out = torch.empty((N,), device=s.device, dtype=torch.float32)
        grad_s = torch.zeros_like(s, dtype=torch.float32)
        _fused_kl_gemm_fwd_bwd_kernel_v1[(N,)](
            s, t, w, loss_out, grad_s,
            s.stride(0), s.stride(1), t.stride(0), t.stride(1), w.stride(0), w.stride(1),
            N, D, V, temp, BLOCK_D=128, BLOCK_V=256, num_warps=8
        )
        ctx.save_for_backward(grad_s.to(s.dtype))
        return loss_out.mean()
    @staticmethod
    def backward(ctx, grad_out):
        grad_s, = ctx.saved_tensors
        return grad_s * (grad_out / grad_s.shape[0]), None, None, None

# ==============================================================
# V2: 2D Block-Tiled Triton Kernel (1-Pass)
# ==============================================================
@triton.jit
def _fused_kl_gemm_2d_kernel(
    x_s_ptr, x_t_ptr, w_ptr,
    loss_out_ptr, grad_x_s_out_ptr,
    stride_xs_n, stride_xs_d,
    stride_xt_n, stride_xt_d,
    stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    
    m_s = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    m_t = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    sum_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_t = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        s_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
        t_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xt_n + offs_d[None, :] * stride_xt_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            s_logits += tl.dot(x_s, tl.trans(w))
            t_logits += tl.dot(x_t, tl.trans(w))
        s_logits /= temperature
        t_logits /= temperature
        s_logits = tl.where(mask_v[None, :], s_logits, -float('inf'))
        t_logits = tl.where(mask_v[None, :], t_logits, -float('inf'))
        m_s_new = tl.maximum(m_s, tl.max(s_logits, axis=1))
        sum_s = sum_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s_logits - m_s_new[:, None]), axis=1)
        m_s = m_s_new
        m_t_new = tl.maximum(m_t, tl.max(t_logits, axis=1))
        sum_t = sum_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t_logits - m_t_new[:, None]), axis=1)
        m_t = m_t_new

    kl_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_s_start = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + tl.arange(0, 1)[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
    
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        s_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
        t_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xt_n + offs_d[None, :] * stride_xt_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            s_logits += tl.dot(x_s, tl.trans(w))
            t_logits += tl.dot(x_t, tl.trans(w))
        s_logits /= temperature
        t_logits /= temperature
        
        t_prob = tl.exp(t_logits - m_t[:, None]) / sum_t[:, None]
        t_prob = tl.where(mask_v[None, :], t_prob, 0.0)
        s_prob = tl.exp(s_logits - m_s[:, None]) / sum_s[:, None]
        s_prob = tl.where(mask_v[None, :], s_prob, 0.0)
        s_logprob = s_logits - m_s[:, None] - tl.log(sum_s[:, None])
        
        kl_term = t_prob * (tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob)
        kl_term = tl.where(t_prob > 0.0, kl_term, 0.0)
        kl_sum += tl.sum(kl_term, axis=1)
        
        grad_s_logits = (s_prob - t_prob) / temperature
        grad_s_logits_cast = grad_s_logits.to(x_s_start.dtype)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            grad_update = tl.dot(grad_s_logits_cast, w)
            grad_ptrs = grad_x_s_out_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d
            old_grad = tl.load(grad_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            tl.store(grad_ptrs, old_grad + grad_update, mask=mask_n[:, None] & mask_d[None, :])
            
    tl.store(loss_out_ptr + offs_n, kl_sum, mask=mask_n)

class V2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, w, temp=1.0):
        N, D = s.shape
        V, _ = w.shape
        loss_out = torch.empty((N,), device=s.device, dtype=torch.float32)
        grad_s = torch.zeros_like(s, dtype=torch.float32)
        BLOCK_N = 64
        grid = (triton.cdiv(N, BLOCK_N),)
        _fused_kl_gemm_2d_kernel[grid](
            s, t, w, loss_out, grad_s,
            s.stride(0), s.stride(1), t.stride(0), t.stride(1), w.stride(0), w.stride(1),
            N, D, V, temp, BLOCK_N=BLOCK_N, BLOCK_D=128, BLOCK_V=64, num_warps=8
        )
        ctx.save_for_backward(grad_s.to(s.dtype))
        return loss_out.mean()
    @staticmethod
    def backward(ctx, grad_out):
        grad_s, = ctx.saved_tensors
        return grad_s * (grad_out / grad_s.shape[0]), None, None, None

# ==============================================================
# V3: Flash-Attention Style Multi-Pass Kernel
# ==============================================================
@triton.jit
def v3_fwd_kernel1(
    x_s_ptr, x_t_ptr, w_ptr,
    m_s_out, sum_s_out, m_t_out, sum_t_out,
    stride_xs_n, stride_xs_d, stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    num_v_blocks = tl.num_programs(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_n = offs_n < N
    mask_v = offs_v < V
    s_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    t_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
        s_logits += tl.dot(x_s, tl.trans(w))
        t_logits += tl.dot(x_t, tl.trans(w))
    s_logits /= temperature
    t_logits /= temperature
    s_logits = tl.where(mask_v[None, :], s_logits, -float('inf'))
    t_logits = tl.where(mask_v[None, :], t_logits, -float('inf'))
    m_s = tl.max(s_logits, axis=1)
    sum_s = tl.sum(tl.exp(s_logits - m_s[:, None]), axis=1)
    m_t = tl.max(t_logits, axis=1)
    sum_t = tl.sum(tl.exp(t_logits - m_t[:, None]), axis=1)
    out_ptrs = offs_n * num_v_blocks + pid_v
    tl.store(m_s_out + out_ptrs, m_s, mask=mask_n)
    tl.store(sum_s_out + out_ptrs, sum_s, mask=mask_n)
    tl.store(m_t_out + out_ptrs, m_t, mask=mask_n)
    tl.store(sum_t_out + out_ptrs, sum_t, mask=mask_n)

@triton.jit
def v3_fwd_kernel2(
    m_s_in, sum_s_in, m_t_in, sum_t_in,
    global_m_s, global_sum_s, global_m_t, global_sum_t,
    N, n_v_blocks, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    m_s = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    m_t = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    sum_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_t = tl.zeros([BLOCK_N], dtype=tl.float32)
    for v in range(n_v_blocks):
        ptrs = offs_n * n_v_blocks + v
        m_s_local = tl.load(m_s_in + ptrs, mask=mask_n, other=-float('inf'))
        sum_s_local = tl.load(sum_s_in + ptrs, mask=mask_n, other=0.0)
        m_s_new = tl.maximum(m_s, m_s_local)
        sum_s = sum_s * tl.exp(m_s - m_s_new) + sum_s_local * tl.exp(m_s_local - m_s_new)
        m_s = m_s_new
        m_t_local = tl.load(m_t_in + ptrs, mask=mask_n, other=-float('inf'))
        sum_t_local = tl.load(sum_t_in + ptrs, mask=mask_n, other=0.0)
        m_t_new = tl.maximum(m_t, m_t_local)
        sum_t = sum_t * tl.exp(m_t - m_t_new) + sum_t_local * tl.exp(m_t_local - m_t_new)
        m_t = m_t_new
    tl.store(global_m_s + offs_n, m_s, mask=mask_n)
    tl.store(global_sum_s + offs_n, sum_s, mask=mask_n)
    tl.store(global_m_t + offs_n, m_t, mask=mask_n)
    tl.store(global_sum_t + offs_n, sum_t, mask=mask_n)

@triton.jit
def v3_bwd_kernel3(
    x_s_ptr, x_t_ptr, w_ptr,
    global_m_s, global_sum_s, global_m_t, global_sum_t,
    loss_out_ptr, grad_x_s_out_ptr,
    stride_xs_n, stride_xs_d, stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_n = offs_n < N
    mask_v = offs_v < V
    s_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    t_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    x_s_start = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + tl.arange(0, 1)[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
        s_logits += tl.dot(x_s, tl.trans(w))
        t_logits += tl.dot(x_t, tl.trans(w))
    s_logits /= temperature
    t_logits /= temperature
    m_s = tl.load(global_m_s + offs_n, mask=mask_n, other=0.0)
    sum_s = tl.load(global_sum_s + offs_n, mask=mask_n, other=1.0)
    m_t = tl.load(global_m_t + offs_n, mask=mask_n, other=0.0)
    sum_t = tl.load(global_sum_t + offs_n, mask=mask_n, other=1.0)
    t_prob = tl.exp(t_logits - m_t[:, None]) / sum_t[:, None]
    t_prob = tl.where(mask_v[None, :], t_prob, 0.0)
    s_prob = tl.exp(s_logits - m_s[:, None]) / sum_s[:, None]
    s_prob = tl.where(mask_v[None, :], s_prob, 0.0)
    s_logprob = s_logits - m_s[:, None] - tl.log(sum_s[:, None])
    kl_term = t_prob * (tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob)
    kl_term = tl.where(t_prob > 0.0, kl_term, 0.0)
    kl_sum = tl.sum(kl_term, axis=1)
    tl.atomic_add(loss_out_ptr + offs_n, kl_sum, mask=mask_n)
    grad_s_logits = (s_prob - t_prob) / temperature
    grad_s_logits_cast = grad_s_logits.to(x_s_start.dtype)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
        grad_update = tl.dot(grad_s_logits_cast, w)
        grad_ptrs = grad_x_s_out_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d
        tl.atomic_add(grad_ptrs, grad_update, mask=mask_n[:, None] & mask_d[None, :])

class V3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, w, temp=1.0):
        N, D = s.shape
        V, _ = w.shape
        BLOCK_N = 32
        BLOCK_V = 128
        BLOCK_D = 128
        n_v_blocks = triton.cdiv(V, BLOCK_V)
        workspace_shape = (N, n_v_blocks)
        ws_ms = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
        ws_sums = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
        ws_mt = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
        ws_sumt = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
        global_ms = torch.empty(N, device=s.device, dtype=torch.float32)
        global_sums = torch.empty(N, device=s.device, dtype=torch.float32)
        global_mt = torch.empty(N, device=s.device, dtype=torch.float32)
        global_sumt = torch.empty(N, device=s.device, dtype=torch.float32)
        loss_out = torch.zeros(N, device=s.device, dtype=torch.float32)
        grad_s = torch.zeros_like(s, dtype=torch.float32)
        grid1 = (triton.cdiv(N, BLOCK_N), n_v_blocks)
        v3_fwd_kernel1[grid1](
            s, t, w, ws_ms, ws_sums, ws_mt, ws_sumt,
            s.stride(0), s.stride(1), w.stride(0), w.stride(1),
            N, D, V, temp, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=4
        )
        grid2 = (triton.cdiv(N, BLOCK_N),)
        v3_fwd_kernel2[grid2](
            ws_ms, ws_sums, ws_mt, ws_sumt,
            global_ms, global_sums, global_mt, global_sumt,
            N, n_v_blocks, BLOCK_N=BLOCK_N, num_warps=4
        )
        v3_bwd_kernel3[grid1](
            s, t, w, global_ms, global_sums, global_mt, global_sumt,
            loss_out, grad_s,
            s.stride(0), s.stride(1), w.stride(0), w.stride(1),
            N, D, V, temp, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=8
        )
        ctx.save_for_backward(grad_s.to(s.dtype))
        return loss_out.mean()
    
    @staticmethod
    def backward(ctx, grad_out):
        grad_s, = ctx.saved_tensors
        return grad_s * (grad_out / grad_s.shape[0]), None, None, None
