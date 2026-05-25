import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time

# ==============================================================
# V4 Candidate 1: Zero Workspace Vectorized (Unrolled)
# ==============================================================
@triton.jit
def v4_unrolled_kernel(
    x_s_ptr, x_t_ptr, w_ptr,
    loss_out_ptr, grad_x_s_out_ptr,
    stride_xs_n, stride_xs_d, stride_w_v, stride_w_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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
            x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
            x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None], other=0.0)

            s_logits += tl.dot(x_s, tl.trans(w))
            t_logits += tl.dot(x_t, tl.trans(w))

        s_logits /= temperature
        t_logits /= temperature

        s_logits = tl.where(mask_v[None, :], s_logits, -float('inf'))
        t_logits = tl.where(mask_v[None, :], t_logits, -float('inf'))

        m_s_new = tl.maximum(m_s, tl.max(s_logits, axis=1))
        m_t_new = tl.maximum(m_t, tl.max(t_logits, axis=1))

        sum_s = sum_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s_logits - m_s_new[:, None]), axis=1)
        sum_t = sum_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t_logits - m_t_new[:, None]), axis=1)

        m_s = m_s_new
        m_t = m_t_new

    loss_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_s_start = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + tl.arange(0, 1)[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)

    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        s_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
        t_logits = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            x_s = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
            x_t = tl.load(x_t_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None], other=0.0)

            s_logits += tl.dot(x_s, tl.trans(w))
            t_logits += tl.dot(x_t, tl.trans(w))

        s_logits /= temperature
        t_logits /= temperature

        s_logits = tl.where(mask_v[None, :], s_logits, -float('inf'))
        t_logits = tl.where(mask_v[None, :], t_logits, -float('inf'))

        t_prob = tl.exp(t_logits - m_t[:, None]) / sum_t[:, None]
        t_prob = tl.where(mask_v[None, :], t_prob, 0.0)
        s_prob = tl.exp(s_logits - m_s[:, None]) / sum_s[:, None]
        s_prob = tl.where(mask_v[None, :], s_prob, 0.0)
        s_logprob = s_logits - m_s[:, None] - tl.log(sum_s[:, None])

        kl_term = t_prob * (tl.log(tl.where(t_prob > 0.0, t_prob, 1.0)) - s_logprob)
        loss_acc += tl.sum(tl.where(t_prob > 0.0, kl_term, 0.0), axis=1)

        grad_s_logits = (s_prob - t_prob) / temperature
        grad_s_logits_cast = grad_s_logits.to(x_s_start.dtype)

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            w = tl.load(w_ptr + offs_v[:, None] * stride_w_v + offs_d[None, :] * stride_w_d, mask=mask_v[:, None], other=0.0)
            grad_update = tl.dot(grad_s_logits_cast, w)
            grad_ptrs = grad_x_s_out_ptr + offs_n[:, None] * stride_xs_n + offs_d[None, :] * stride_xs_d
            old_grad = tl.load(grad_ptrs, mask=mask_n[:, None], other=0.0)
            tl.store(grad_ptrs, old_grad + grad_update, mask=mask_n[:, None])

    tl.store(loss_out_ptr + offs_n, loss_acc, mask=mask_n)


def run_v4_unrolled(s, t, w):
    N, D = s.shape
    V = w.shape[0]
    loss_out = torch.empty(N, device=s.device, dtype=torch.float32)
    grad_s = torch.zeros_like(s, dtype=torch.float32)
    BLOCK_N = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    v4_unrolled_kernel[grid](
        s, t, w, loss_out, grad_s,
        s.stride(0), s.stride(1), w.stride(0), w.stride(1),
        N, D, V, 1.0, BLOCK_N=BLOCK_N, BLOCK_D=128, BLOCK_V=256, num_warps=8, num_stages=1
    )
    return loss_out.mean(), grad_s


# ==============================================================
# Baseline
# ==============================================================
def baseline(s, t, w):
    s_logits = F.linear(s, w)
    t_logits = F.linear(t, w)
    t_probs = F.softmax(t_logits, dim=-1)
    loss = F.cross_entropy(s_logits, t_probs, reduction='mean')
    s.grad = None
    loss.backward()
    return loss, s.grad

if __name__ == "__main__":
    N, D, V = 1024, 1536, 151936
    s = torch.randn(N, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    t = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(V, D, device='cuda', dtype=torch.bfloat16)

    # Compile & Correctness
    loss_base, grad_base = baseline(s, t, w)
    loss_v4u, grad_v4u = run_v4_unrolled(s, t, w)

    print(f"Loss Base: {loss_base.item():.4f} | V4U: {loss_v4u.item():.4f}")
    grad_v4u_scaled = grad_v4u / N
    print(f"Max Grad Diff V4U: {(grad_base - grad_v4u_scaled).abs().max().item():.6f}")

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
        run_v4_unrolled(s, t, w)
    torch.cuda.synchronize()
    v4u_time = (time.time() - start) * 100

    print(f"Baseline Time: {base_time:.2f} ms")
    print(f"V4U Time: {v4u_time:.2f} ms")
