import torch
import triton
import triton.language as tl
import time
import torch.nn.functional as F

@triton.jit
def fused_kl_loss_fwd_bwd_kernel(
    s_logits_ptr, t_logits_ptr, loss_out_ptr,
    stride_s_n, stride_s_v, stride_t_n, stride_t_v,
    N, V, temperature,
    BLOCK_V: tl.constexpr
):
    pid_n = tl.program_id(0)

    offs_v = tl.arange(0, BLOCK_V)
    
    m_s = -float('inf')
    m_t = -float('inf')
    
    # Pass 1: Find Max
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        s_logits = tl.load(s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v, mask=mask_v, other=-float('inf')).to(tl.float32)
        t_logits = tl.load(t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v, mask=mask_v, other=-float('inf')).to(tl.float32)
        
        s_logits /= temperature
        t_logits /= temperature
        
        m_s = tl.maximum(m_s, tl.max(s_logits))
        m_t = tl.maximum(m_t, tl.max(t_logits))

    sum_s = 0.0
    sum_t = 0.0
    
    # Pass 2: Find Sum
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        s_logits = tl.load(s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v, mask=mask_v, other=-float('inf')).to(tl.float32)
        t_logits = tl.load(t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v, mask=mask_v, other=-float('inf')).to(tl.float32)
        
        s_logits /= temperature
        t_logits /= temperature
        
        sum_s += tl.sum(tl.exp(s_logits - m_s))
        sum_t += tl.sum(tl.exp(t_logits - m_t))

    kl_sum = 0.0
    
    # Pass 3: Compute Loss and Gradients (In-place on s_logits)
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        
        s_ptr = s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v
        t_ptr = t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v
        
        s_logits = tl.load(s_ptr, mask=mask_v, other=-float('inf')).to(tl.float32)
        t_logits = tl.load(t_ptr, mask=mask_v, other=-float('inf')).to(tl.float32)
        
        s_logits /= temperature
        t_logits /= temperature
        
        t_prob = tl.exp(t_logits - m_t) / sum_t
        t_prob = tl.where(mask_v, t_prob, 0.0)
        
        s_prob = tl.exp(s_logits - m_s) / sum_s
        s_prob = tl.where(mask_v, s_prob, 0.0)
        
        s_logprob = s_logits - m_s - tl.log(sum_s)
        
        kl_term = t_prob * (tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob)
        kl_sum += tl.sum(tl.where(t_prob > 0.0, kl_term, 0.0))
        
        grad_s_logits = (s_prob - t_prob) / temperature
        
        grad_s_logits_cast = grad_s_logits.to(s_logits_ptr.dtype.element_ty)
        # Write gradient directly over s_logits to save memory
        tl.store(s_ptr, grad_s_logits_cast, mask=mask_v)
        
    tl.store(loss_out_ptr + pid_n, kl_sum)


def run_v5(s, t, w, temp=1.0):
    N, D = s.shape
    V, _ = w.shape
    
    loss_sum = 0.0
    grad_s = torch.zeros_like(s)
    
    CHUNK = 1024
    BLOCK_V = 4096
    
    for i in range(0, N, CHUNK):
        end = min(i + CHUNK, N)
        s_chunk = s[i:end]
        t_chunk = t[i:end]
        chunk_size = end - i
        
        # cuBLAS matmuls
        s_logits = F.linear(s_chunk, w)
        t_logits = F.linear(t_chunk, w)
        
        loss_out = torch.empty(chunk_size, device=s.device, dtype=torch.float32)
        
        # Fused softmax + kl_div + backward logits
        grid = (chunk_size,)
        fused_kl_loss_fwd_bwd_kernel[grid](
            s_logits, t_logits, loss_out,
            s_logits.stride(0), s_logits.stride(1), t_logits.stride(0), t_logits.stride(1),
            chunk_size, V, temp,
            BLOCK_V=BLOCK_V,
            num_warps=8
        )
        
        loss_sum += loss_out.sum()
        
        grad_s[i:end] = F.linear(s_logits, w.t())
        
        # Free memory aggressively
        del s_logits, t_logits, loss_out
        torch.cuda.empty_cache()
    
    return loss_sum / N, grad_s

def baseline(s, t, w):
    s_logits = F.linear(s, w)
    t_logits = F.linear(t, w)
    t_probs = F.softmax(t_logits, dim=-1)
    loss = F.cross_entropy(s_logits, t_probs, reduction='mean')
    s.grad = None
    loss.backward()
    return loss, s.grad

if __name__ == "__main__":
    N, D, V = 4096, 1536, 151936
    s = torch.randn(N, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    t = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(V, D, device='cuda', dtype=torch.bfloat16)

    loss_base, grad_base = baseline(s, t, w)
    loss_v5, grad_v5 = run_v5(s, t, w)

    print(f"Loss Base: {loss_base.item():.4f} | V5: {loss_v5.item():.4f}")
    
    grad_v5_scaled = grad_v5 / N
    print(f"Max Grad Diff V5: {(grad_base - grad_v5_scaled).abs().max().item():.6f}")

    # Benchmarking
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        baseline(s, t, w)
    torch.cuda.synchronize()
    base_time = (time.time() - start) * 100

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        run_v5(s, t, w)
    torch.cuda.synchronize()
    v5_time = (time.time() - start) * 100

    print(f"Baseline Time: {base_time:.2f} ms")
    print(f"V5 Hybrid Time: {v5_time:.2f} ms")
