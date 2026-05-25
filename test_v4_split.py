import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time

# ==============================================================
# V4 Split-V (50MB Workspace, Full GPU Saturation)
# ==============================================================
@triton.jit
def v4_split_pass1(
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
    mask_n = offs_n < N

    v_chunk_size = (V + num_v_blocks - 1) // num_v_blocks
    start_v = pid_v * v_chunk_size
    end_v = tl.minimum(start_v + v_chunk_size, V)

    m_s = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    m_t = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    sum_s = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_t = tl.zeros([BLOCK_N], dtype=tl.float32)

    for v_step in range(start_v, end_v, BLOCK_V):
        offs_v = v_step + tl.arange(0, BLOCK_V)
        mask_v = offs_v < end_v

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
        sum_s = sum_s * tl.exp(m_s - m_s_new) + tl.sum(tl.exp(s_logits - m_s_new[:, None]), axis=1)
        m_s = m_s_new

        m_t_new = tl.maximum(m_t, tl.max(t_logits, axis=1))
        sum_t = sum_t * tl.exp(m_t - m_t_new) + tl.sum(tl.exp(t_logits - m_t_new[:, None]), axis=1)
        m_t = m_t_new

    out_ptrs = offs_n * num_v_blocks + pid_v
    tl.store(m_s_out + out_ptrs, m_s, mask=mask_n)
    tl.store(sum_s_out + out_ptrs, sum_s, mask=mask_n)
    tl.store(m_t_out + out_ptrs, m_t, mask=mask_n)
    tl.store(sum_t_out + out_ptrs, sum_t, mask=mask_n)

@triton.jit
def v4_split_pass2(
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
def v4_split_pass3(
    x_s_ptr, x_t_ptr, w_ptr,
    global_m_s, global_sum_s, global_m_t, global_sum_t,
    loss_out_ptr, grad_workspace_ptr,
    stride_xs_n, stride_xs_d, stride_w_v, stride_w_d,
    stride_gw_v, stride_gw_n, stride_gw_d,
    N, D, V, temperature,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)

    num_v_blocks = tl.num_programs(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    v_chunk_size = (V + num_v_blocks - 1) // num_v_blocks
    start_v = pid_v * v_chunk_size
    end_v = tl.minimum(start_v + v_chunk_size, V)

    m_s = tl.load(global_m_s + offs_n, mask=mask_n, other=0.0)
    sum_s = tl.load(global_sum_s + offs_n, mask=mask_n, other=1.0)
    m_t = tl.load(global_m_t + offs_n, mask=mask_n, other=0.0)
    sum_t = tl.load(global_sum_t + offs_n, mask=mask_n, other=1.0)

    loss_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    x_s_start = tl.load(x_s_ptr + offs_n[:, None] * stride_xs_n + tl.arange(0, 1)[None, :] * stride_xs_d, mask=mask_n[:, None], other=0.0)

    for v_step in range(start_v, end_v, BLOCK_V):
        offs_v = v_step + tl.arange(0, BLOCK_V)
        mask_v = offs_v < end_v

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
            
            grad_ptrs = grad_workspace_ptr + pid_v * stride_gw_v + offs_n[:, None] * stride_gw_n + offs_d[None, :] * stride_gw_d
            old_grad = tl.load(grad_ptrs, mask=mask_n[:, None], other=0.0)
            tl.store(grad_ptrs, old_grad + grad_update, mask=mask_n[:, None])

    tl.store(loss_out_ptr + offs_n * num_v_blocks + pid_v, loss_acc, mask=mask_n)

def run_v4_split(s, t, w, temp=1.0):
    N, D = s.shape
    V = w.shape[0]
    
    # Exactly 8 splits. 8 * 1024 * 1536 * 4 bytes = 50 MB
    SPLIT_V = 8
    BLOCK_V = 256
    
    BLOCK_N = 32
    BLOCK_D = 128

    workspace_shape = (N, SPLIT_V)
    ws_ms = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
    ws_sums = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
    ws_mt = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
    ws_sumt = torch.empty(workspace_shape, device=s.device, dtype=torch.float32)
    
    global_ms = torch.empty(N, device=s.device, dtype=torch.float32)
    global_sums = torch.empty(N, device=s.device, dtype=torch.float32)
    global_mt = torch.empty(N, device=s.device, dtype=torch.float32)
    global_sumt = torch.empty(N, device=s.device, dtype=torch.float32)
    
    loss_out = torch.zeros((N, SPLIT_V), device=s.device, dtype=torch.float32)
    
    # 50 MB Workspace!
    grad_workspace = torch.zeros((SPLIT_V, N, D), device=s.device, dtype=s.dtype)
    
    grid1 = (triton.cdiv(N, BLOCK_N), SPLIT_V)
    
    v4_split_pass1[grid1](
        s, t, w, ws_ms, ws_sums, ws_mt, ws_sumt,
        s.stride(0), s.stride(1), w.stride(0), w.stride(1),
        N, D, V, temp, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=8, num_stages=1
    )
    
    grid2 = (triton.cdiv(N, BLOCK_N),)
    v4_split_pass2[grid2](
        ws_ms, ws_sums, ws_mt, ws_sumt,
        global_ms, global_sums, global_mt, global_sumt,
        N, SPLIT_V, BLOCK_N=BLOCK_N, num_warps=4, num_stages=1
    )
    
    v4_split_pass3[grid1](
        s, t, w, global_ms, global_sums, global_mt, global_sumt,
        loss_out, grad_workspace,
        s.stride(0), s.stride(1), w.stride(0), w.stride(1),
        grad_workspace.stride(0), grad_workspace.stride(1), grad_workspace.stride(2),
        N, D, V, temp, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V, num_warps=8, num_stages=1
    )
    
    # Secure PyTorch reduction, perfectly deterministic, no atomic_add!
    grad_s = grad_workspace.sum(dim=0)
    return loss_out.sum(dim=1).mean(), grad_s

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
    loss_v4s, grad_v4s = run_v4_split(s, t, w)

    print(f"Loss Base: {loss_base.item():.4f} | V4_Split: {loss_v4s.item():.4f}")
    
    grad_v4s_scaled = grad_v4s / N
    print(f"Max Grad Diff V4_Split: {(grad_base - grad_v4s_scaled).abs().max().item():.6f}")

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
        run_v4_split(s, t, w)
    torch.cuda.synchronize()
    v4s_time = (time.time() - start) * 100

    print(f"Baseline Time: {base_time:.2f} ms")
    print(f"V4_Split Time: {v4s_time:.2f} ms")
