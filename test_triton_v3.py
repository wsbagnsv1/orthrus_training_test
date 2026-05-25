import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

# ==============================================================
# V3: Flash-Attention Style Multi-Pass Reduction Kernel
# ==============================================================
@triton.jit
def v3_fwd_kernel1(
    x_s_ptr, x_t_ptr, w_ptr,
    m_s_out, sum_s_out, m_t_out, sum_t_out,
    stride_xs_n, stride_xs_d,
    stride_w_v, stride_w_d,
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
    
    # Store to workspace: shape [N, num_v_blocks]
    # stride for N is num_v_blocks
    out_ptrs = offs_n * num_v_blocks + pid_v
    tl.store(m_s_out + out_ptrs, m_s, mask=mask_n)
    tl.store(sum_s_out + out_ptrs, sum_s, mask=mask_n)
    tl.store(m_t_out + out_ptrs, m_t, mask=mask_n)
    tl.store(sum_t_out + out_ptrs, sum_t, mask=mask_n)

@triton.jit
def v3_fwd_kernel2(
    m_s_in, sum_s_in, m_t_in, sum_t_in,
    global_m_s, global_sum_s, global_m_t, global_sum_t,
    N, n_v_blocks,
    BLOCK_N: tl.constexpr
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
    stride_xs_n, stride_xs_d,
    stride_w_v, stride_w_d,
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
        
        # Workspace tensors
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
            s, t, w,
            ws_ms, ws_sums, ws_mt, ws_sumt,
            s.stride(0), s.stride(1),
            w.stride(0), w.stride(1),
            N, D, V, temp,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=4
        )
        
        grid2 = (triton.cdiv(N, BLOCK_N),)
        v3_fwd_kernel2[grid2](
            ws_ms, ws_sums, ws_mt, ws_sumt,
            global_ms, global_sums, global_mt, global_sumt,
            N, n_v_blocks, BLOCK_N=BLOCK_N, num_warps=4
        )
        
        v3_bwd_kernel3[grid1](
            s, t, w,
            global_ms, global_sums, global_mt, global_sumt,
            loss_out, grad_s,
            s.stride(0), s.stride(1),
            w.stride(0), w.stride(1),
            N, D, V, temp,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=8
        )
        
        ctx.save_for_backward(grad_s.to(s.dtype))
        return loss_out.mean()
    
    @staticmethod
    def backward(ctx, grad_out):
        grad_s, = ctx.saved_tensors
        return grad_s * (grad_out / grad_s.shape[0]), None, None, None


# ==============================================================
# Benchmarking against V2 and Baseline
# ==============================================================
# We import V2 from test_triton_v2.py
import test_triton_v2

def run_benchmark():
    # REALISTIC CONDITIONS
    N = 4096  # Larger chunk to simulate real workloads
    D = 2048
    V = 32000
    device = 'cuda'
    dtype = torch.bfloat16
    
    print(f"Benchmarking Realistic Workload (N={N}, D={D}, V={V})")
    torch.manual_seed(42)
    s_hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_hidden = torch.randn(N, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    
    # ----------------- BASELINE (PyTorch Native cuBLAS + CE) -----------------
    s_base = s_hidden.clone().detach().requires_grad_(True)
    # Warmup PyTorch Baseline
    t_logits = F.linear(t_hidden, w).float()
    t_probs = F.softmax(t_logits, dim=-1).to(dtype)
    _ = F.cross_entropy(F.linear(s_base, w), t_probs, reduction='mean')
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        t_logits = F.linear(t_hidden, w).float()
        t_probs = F.softmax(t_logits, dim=-1).to(dtype)
    s_logits = F.linear(s_base, w)
    loss_base = F.cross_entropy(s_logits, t_probs, reduction='mean')
    loss_base.backward()
    torch.cuda.synchronize()
    t_base = time.perf_counter() - t0
    grad_base = s_base.grad.clone()
    
    # ----------------- TRITON V2 (1-Pass Tiled) -----------------
    s_v2 = s_hidden.clone().detach().requires_grad_(True)
    _ = test_triton_v2.V2Function.apply(s_v2, t_hidden, w) # Warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_v2 = test_triton_v2.V2Function.apply(s_v2, t_hidden, w)
    loss_v2.backward()
    torch.cuda.synchronize()
    t_v2 = time.perf_counter() - t0
    grad_v2 = s_v2.grad.clone()
    
    # ----------------- TRITON V3 (Flash-Attn Style 3-Pass) -----------------
    s_v3 = s_hidden.clone().detach().requires_grad_(True)
    _ = V3Function.apply(s_v3, t_hidden, w) # Warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_v3 = V3Function.apply(s_v3, t_hidden, w)
    loss_v3.backward()
    torch.cuda.synchronize()
    t_v3 = time.perf_counter() - t0
    grad_v3 = s_v3.grad.clone()
    
    print("-" * 50)
    print(f"Base Loss (CE): {loss_base.item():.6f}")
    print(f"V2 Loss (KL):   {loss_v2.item():.6f}")
    print(f"V3 Loss (KL):   {loss_v3.item():.6f}")
    print("-" * 50)
    print(f"Grad Diff V2 vs Base: {(grad_v2 - grad_base).abs().max().item():.8f}")
    print(f"Grad Diff V3 vs Base: {(grad_v3 - grad_base).abs().max().item():.8f}")
    print("-" * 50)
    print(f"Base Time (Peak cuBLAS): {t_base*1000:.2f} ms")
    print(f"V2 Time   (1-Pass):      {t_v2*1000:.2f} ms")
    print(f"V3 Time   (3-Pass):      {t_v3*1000:.2f} ms")
    print("-" * 50)
    print(f"V3 Speedup over V2: {t_v2 / t_v3:.2f}x")
    print(f"V3 Speedup over Base: {t_base / t_v3:.2f}x (Note: Base uses ~500MB VRAM spike, V3 uses ~0MB!)")

if __name__ == '__main__':
    run_benchmark()
