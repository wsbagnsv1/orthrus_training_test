import torch
import triton
import triton.language as tl
import torch.nn.functional as F
@triton.jit
def fused_kl_loss_fwd_bwd_kernel(
    s_logits_ptr, t_logits_ptr, grad_s_logits_ptr, loss_out_ptr,
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
    
    # Pass 3: Compute Loss and Gradients
    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        
        s_ptr = s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v
        t_ptr = t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v
        grad_s_ptr = grad_s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v
        
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
        # Write gradient to output pointer
        tl.store(grad_s_ptr, grad_s_logits_cast, mask=mask_v)
        
    tl.store(loss_out_ptr + pid_n, kl_sum)


@triton.jit
def fused_kl_loss_fwd_only_kernel(
    s_logits_ptr, t_logits_ptr, loss_out_ptr,
    stride_s_n, stride_s_v, stride_t_n, stride_t_v,
    N, V, temperature,
    BLOCK_V: tl.constexpr,
):
    """Same KL as fused_kl_loss_fwd_bwd_kernel; no gradient writes (eval / inference)."""
    pid_n = tl.program_id(0)

    offs_v = tl.arange(0, BLOCK_V)

    m_s = -float("inf")
    m_t = -float("inf")

    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        s_logits = tl.load(
            s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        t_logits = tl.load(
            t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        s_logits /= temperature
        t_logits /= temperature
        m_s = tl.maximum(m_s, tl.max(s_logits))
        m_t = tl.maximum(m_t, tl.max(t_logits))

    sum_s = 0.0
    sum_t = 0.0

    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        s_logits = tl.load(
            s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        t_logits = tl.load(
            t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        s_logits /= temperature
        t_logits /= temperature
        sum_s += tl.sum(tl.exp(s_logits - m_s))
        sum_t += tl.sum(tl.exp(t_logits - m_t))

    kl_sum = 0.0

    for v_start in range(0, V, BLOCK_V):
        v_idx = v_start + offs_v
        mask_v = v_idx < V
        s_logits = tl.load(
            s_logits_ptr + pid_n * stride_s_n + v_idx * stride_s_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        t_logits = tl.load(
            t_logits_ptr + pid_n * stride_t_n + v_idx * stride_t_v,
            mask=mask_v, other=-float("inf"),
        ).to(tl.float32)
        s_logits /= temperature
        t_logits /= temperature
        t_prob = tl.exp(t_logits - m_t) / sum_t
        t_prob = tl.where(mask_v, t_prob, 0.0)
        s_logprob = s_logits - m_s - tl.log(sum_s)
        kl_term = t_prob * (
            tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob
        )
        kl_sum += tl.sum(tl.where(t_prob > 0.0, kl_term, 0.0))

    tl.store(loss_out_ptr + pid_n, kl_sum)


class V5Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, w, temp=1.0):
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
            
            grad_s_logits = torch.empty_like(s_logits)
            loss_out = torch.empty(chunk_size, device=s.device, dtype=torch.float32)
            
            # Fused softmax + kl_div + backward logits
            grid = (chunk_size,)
            fused_kl_loss_fwd_bwd_kernel[grid](
                s_logits, t_logits, grad_s_logits, loss_out,
                s_logits.stride(0), s_logits.stride(1), t_logits.stride(0), t_logits.stride(1),
                chunk_size, V, temp,
                BLOCK_V=BLOCK_V,
                num_warps=8
            )
            
            loss_sum += loss_out.sum()
            
            grad_s[i:end] = F.linear(grad_s_logits, w.t())
            
            # Free memory aggressively
            del s_logits, t_logits, grad_s_logits, loss_out
        
        ctx.save_for_backward(grad_s)
        return loss_sum / N
        
    @staticmethod
    def backward(ctx, grad_out):
        grad_s, = ctx.saved_tensors
        return grad_s * (grad_out / grad_s.shape[0]), None, None, None

def triton_compute_kl_loss(student_hidden, teacher_hidden, lm_head_weight, temperature=1.0):
    s_contig = student_hidden.contiguous()
    t_contig = teacher_hidden.contiguous()
    w_contig = lm_head_weight.contiguous()
    return V5Function.apply(s_contig, t_contig, w_contig, temperature)


def triton_compute_kl_loss_fwd_only(
    student_hidden, teacher_hidden, lm_head_weight, temperature=1.0,
):
    """
    Eval/inference KL mean over N rows. No autograd buffers (grad_s, grad_s_logits).
    Numerically matches V5Function.forward; training should use triton_compute_kl_loss.
    """
    s = student_hidden.contiguous()
    t = teacher_hidden.contiguous()
    w = lm_head_weight.contiguous()
    N, _ = s.shape
    V, _ = w.shape
    if N == 0:
        return torch.tensor(0.0, device=s.device, dtype=torch.float32)

    loss_sum = torch.zeros((), device=s.device, dtype=torch.float32)
    CHUNK = 1024
    BLOCK_V = 4096

    for i in range(0, N, CHUNK):
        end = min(i + CHUNK, N)
        s_chunk = s[i:end]
        t_chunk = t[i:end]
        chunk_size = end - i

        s_logits = F.linear(s_chunk, w)
        t_logits = F.linear(t_chunk, w)
        loss_out = torch.empty(chunk_size, device=s.device, dtype=torch.float32)

        fused_kl_loss_fwd_only_kernel[(chunk_size,)](
            s_logits, t_logits, loss_out,
            s_logits.stride(0), s_logits.stride(1),
            t_logits.stride(0), t_logits.stride(1),
            chunk_size, V, temperature,
            BLOCK_V=BLOCK_V,
            num_warps=8,
        )
        loss_sum = loss_sum + loss_out.sum()
        del s_logits, t_logits, loss_out

    return loss_sum / N


def _verify_fwd_only_matches_training(n=512, d=256, v=4096, device="cuda"):
    """Quick sanity: fwd-only scalar == V5 forward (no backward)."""
    torch.manual_seed(0)
    s = torch.randn(n, d, device=device, dtype=torch.bfloat16)
    t = torch.randn(n, d, device=device, dtype=torch.bfloat16)
    w = torch.randn(v, d, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        loss_train = triton_compute_kl_loss(s, t, w).float()
        loss_eval = triton_compute_kl_loss_fwd_only(s, t, w)
    diff = (loss_train - loss_eval).abs().item()
    if diff > 1e-2:
        raise AssertionError(f"fwd-only vs training forward: {diff}")
    return diff


if __name__ == "__main__":
    if torch.cuda.is_available():
        d = _verify_fwd_only_matches_training()
        print(f"fwd-only matches training forward (max diff): {d:.6f}")
    else:
        print("CUDA not available — skip verify")
