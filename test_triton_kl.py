import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time

@triton.jit
def _fused_kl_gemm_fwd_bwd_kernel(
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
    
    # ==========================================
    # FIRST PASS: Calculate Softmax Denominators
    # ==========================================
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        
        s_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        t_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        
        # Block-tiled dot product: W_block @ X
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            
            x_s_ptrs = x_s_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d
            x_t_ptrs = x_t_ptr + pid_n * stride_xt_n + offs_d * stride_xt_d
            x_s = tl.load(x_s_ptrs, mask=mask_d, other=0.0)
            x_t = tl.load(x_t_ptrs, mask=mask_d, other=0.0)
            
            w_ptrs = w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d
            w_block = tl.load(w_ptrs, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            s_logits += tl.sum(w_block * x_s[None, :], axis=1)
            t_logits += tl.sum(w_block * x_t[None, :], axis=1)
            
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
        
        s_logits = tl.where(mask_v, s_logits, -float('inf'))
        t_logits = tl.where(mask_v, t_logits, -float('inf'))
        
        # Online softmax states
        m_s_new = tl.maximum(m_s, tl.max(s_logits, axis=0))
        sum_s = sum_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s_logits - m_s_new), axis=0)
        m_s = m_s_new
        
        m_t_new = tl.maximum(m_t, tl.max(t_logits, axis=0))
        sum_t = sum_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t_logits - m_t_new), axis=0)
        m_t = m_t_new

    # ==========================================
    # SECOND PASS: Calculate KL and Gradients
    # ==========================================
    kl_sum = 0.0
    
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        
        s_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        t_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
        
        # Recompute logits
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            
            x_s_ptrs = x_s_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d
            x_t_ptrs = x_t_ptr + pid_n * stride_xt_n + offs_d * stride_xt_d
            x_s = tl.load(x_s_ptrs, mask=mask_d, other=0.0)
            x_t = tl.load(x_t_ptrs, mask=mask_d, other=0.0)
            
            w_ptrs = w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d
            w_block = tl.load(w_ptrs, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            s_logits += tl.sum(w_block * x_s[None, :], axis=1)
            t_logits += tl.sum(w_block * x_t[None, :], axis=1)
            
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature
        
        t_prob = tl.exp(t_logits - m_t) / sum_t
        t_prob = tl.where(mask_v, t_prob, 0.0)
        
        s_prob = tl.exp(s_logits - m_s) / sum_s
        s_prob = tl.where(mask_v, s_prob, 0.0)
        
        s_logprob = s_logits - m_s - tl.log(sum_s)
        
        # Calculate KL
        kl_term = t_prob * (tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob)
        kl_term = tl.where(t_prob > 0.0, kl_term, 0.0)
        kl_sum += tl.sum(kl_term, axis=0)
        
        # Calculate Gradient (Analytically: softmax - teacher_prob)
        grad_s_logits = (s_prob - t_prob) / temperature
        
        # Accumulate Gradient to Global Memory
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            
            w_ptrs = w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d
            w_block = tl.load(w_ptrs, mask=mask_v[:, None] & mask_d[None, :], other=0.0)
            
            grad_x_s_block = tl.sum(grad_s_logits[:, None] * w_block, axis=0)
            
            grad_x_s_ptrs = grad_x_s_out_ptr + pid_n * stride_xs_n + offs_d * stride_xs_d
            old_grad = tl.load(grad_x_s_ptrs, mask=mask_d, other=0.0)
            tl.store(grad_x_s_ptrs, old_grad + grad_x_s_block, mask=mask_d)
            
    tl.store(loss_out_ptr + pid_n, kl_sum)


class FusedKLGEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_hidden, teacher_hidden, lm_head_weight, temperature=1.0):
        N, D = student_hidden.shape
        V, _ = lm_head_weight.shape
        
        assert student_hidden.is_contiguous()
        assert teacher_hidden.is_contiguous()
        assert lm_head_weight.is_contiguous()
        
        loss_out = torch.empty((N,), device=student_hidden.device, dtype=torch.float32)
        # Initialize gradient tensor to zeroes so we can accumulate into it safely
        grad_x_s_out = torch.zeros_like(student_hidden, dtype=torch.float32)
        
        grid = (N,)
        
        _fused_kl_gemm_fwd_bwd_kernel[grid](
            student_hidden, teacher_hidden, lm_head_weight,
            loss_out, grad_x_s_out,
            student_hidden.stride(0), student_hidden.stride(1),
            teacher_hidden.stride(0), teacher_hidden.stride(1),
            lm_head_weight.stride(0), lm_head_weight.stride(1),
            N, D, V, temperature,
            BLOCK_D=128,
            BLOCK_V=256,
            num_warps=8
        )
        
        ctx.save_for_backward(grad_x_s_out.to(student_hidden.dtype))
        return loss_out.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x_s, = ctx.saved_tensors
        N = grad_x_s.shape[0]
        grad_student = grad_x_s * (grad_output / N)
        return grad_student, None, None, None


def custom_triton_kl_loss(student_hidden, teacher_hidden, lm_head_weight, temperature=1.0):
    return FusedKLGEMMFunction.apply(student_hidden, teacher_hidden, lm_head_weight, temperature)

def test_correctness():
    print("Testing Fused GEMM + KL Triton Kernel...")
    N = 1024
    D = 2048
    V = 32000
    device = 'cuda'
    
    # We test explicitly in bfloat16 to simulate the training environment
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    s_hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_hidden = torch.randn(N, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    
    s_hidden_base = s_hidden.clone().detach().requires_grad_(True)
    s_hidden_triton = s_hidden.clone().detach().requires_grad_(True)
    
    # ------------------ Baseline ------------------
    # We cast logits to float32 before softmax/kl to ensure numeric stability (like native cross_entropy)
    s_logits_base = F.linear(s_hidden_base, w).float()
    t_logits_base = F.linear(t_hidden, w).float()
    
    s_logprobs_base = F.log_softmax(s_logits_base, dim=-1)
    t_probs_base = F.softmax(t_logits_base, dim=-1)
    loss_base = F.kl_div(s_logprobs_base, t_probs_base, reduction='batchmean')
    loss_base.backward()
    grad_base = s_hidden_base.grad.clone()
    
    # ------------------ Triton ------------------
    loss_triton = custom_triton_kl_loss(s_hidden_triton, t_hidden, w)
    loss_triton.backward()
    grad_triton = s_hidden_triton.grad.clone()
    
    print("-" * 50)
    print(f"Baseline Loss: {loss_base.item():.6f}")
    print(f"Triton Loss:   {loss_triton.item():.6f}")
    print(f"Loss Diff:     {abs(loss_base.item() - loss_triton.item()):.8f}")
    
    grad_diff = (grad_base - grad_triton).abs().max().item()
    print(f"Max Grad Diff: {grad_diff:.8f}")
    
    # Check if gradient sums to 0 safely
    print(f"Base Grad mean: {grad_base.mean().item():.8f}")
    print(f"Triton Grad mean: {grad_triton.mean().item():.8f}")
    print("-" * 50)
    print("If Loss Diff and Grad Diff are near 0, the Triton kernel is numerically stable in FP16/BF16!")

if __name__ == '__main__':
    test_correctness()
