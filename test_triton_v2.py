import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

# ==============================================================
# V1: 1D Triton Kernel (Slow, GEMV)
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


# ==============================================================
# V2: 2D Triton Kernel (Fast, Tensor Cores GEMM)
# ==============================================================
@triton.jit
def _fused_kl_gemm_2d_kernel(
    x_s_ptr, x_t_ptr, w_ptr,
    loss_out_ptr, grad_x_s_out_ptr,
    stride_xs_n, stride_xs_d,
    stride_xt_n, stride_xt_d,
    stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    
    m_s = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    m_t = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    sum_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_t = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # ------------------ FIRST PASS ------------------
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
            
            # TENSOR CORE DOT PRODUCT!
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

    # ------------------ SECOND PASS ------------------
    kl_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    
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
        
        # Cast to match W dtype for Tensor Cores
        # Using bfloat16 explicitly here to match our testing dtype
        grad_s_logits_16 = grad_s_logits.to(tl.bfloat16)
        
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            # [BLOCK_N, BLOCK_V] @ [BLOCK_V, BLOCK_D]
            grad_update = tl.dot(grad_s_logits_16, w)
            
            grad_ptrs = grad_x_s_out_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d
            old_grad = tl.load(grad_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            tl.store(grad_ptrs, old_grad + grad_update, mask=mask_n[:, None] & mask_d[None, :])
            
    tl.store(loss_out_ptr + offs_n, kl_sum, mask=mask_n)


# ==============================================================
# PyTorch Wrappers
# ==============================================================
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
# Benchmarking
# ==============================================================
def benchmark():
    print("Benchmarking Triton KL Loss V1 vs V2 vs Baseline...")
    N = 1024
    D = 2048
    V = 32000
    device = 'cuda'
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    s_hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_hidden = torch.randn(N, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    
    s_base = s_hidden.clone().detach().requires_grad_(True)
    s_v1 = s_hidden.clone().detach().requires_grad_(True)
    s_v2 = s_hidden.clone().detach().requires_grad_(True)
    
    # ----------------- BASELINE -----------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s_logits = F.linear(s_base, w).float()
    t_logits = F.linear(t_hidden, w).float()
    s_logprobs = F.log_softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)
    loss_base = F.kl_div(s_logprobs, t_probs, reduction='batchmean')
    loss_base.backward()
    torch.cuda.synchronize()
    t_base = time.perf_counter() - t0
    grad_base = s_base.grad.clone()
    
    # ----------------- TRITON V1 -----------------
    # Warmup
    _ = V1Function.apply(s_v1, t_hidden, w)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_v1 = V1Function.apply(s_v1, t_hidden, w)
    loss_v1.backward()
    torch.cuda.synchronize()
    t_v1 = time.perf_counter() - t0
    grad_v1 = s_v1.grad.clone()
    
    # ----------------- TRITON V2 -----------------
    # Warmup
    _ = V2Function.apply(s_v2, t_hidden, w)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_v2 = V2Function.apply(s_v2, t_hidden, w)
    loss_v2.backward()
    torch.cuda.synchronize()
    t_v2 = time.perf_counter() - t0
    grad_v2 = s_v2.grad.clone()
    
    print("-" * 50)
    print(f"Baseline Loss: {loss_base.item():.6f}")
    print(f"V1 Loss:       {loss_v1.item():.6f}")
    print(f"V2 Loss:       {loss_v2.item():.6f}")
    print("-" * 50)
    print(f"Grad Diff V1 vs Base: {(grad_v1 - grad_base).abs().max().item():.8f}")
    print(f"Grad Diff V2 vs Base: {(grad_v2 - grad_base).abs().max().item():.8f}")
    print("-" * 50)
    print(f"Base Time (FWD+BWD): {t_base*1000:.2f} ms")
    print(f"V1 Time   (FWD+BWD): {t_v1*1000:.2f} ms")
    print(f"V2 Time   (FWD+BWD): {t_v2*1000:.2f} ms")
    
    # Calculate Speedup
    print(f"V2 Speedup over V1: {t_v1 / t_v2:.2f}x")
    print(f"V2 Speedup over Base: {t_base / t_v2:.2f}x")

if __name__ == '__main__':
    benchmark()
