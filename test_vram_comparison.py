import torch
import torch.nn.functional as F
import time
import gc

from triton_kl_loss import triton_compute_kl_loss

def reset_memory_stats():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def baseline_forward_backward(s, t, w, temp=1.0):
    s.grad = None
    s.requires_grad_(True)
    w.requires_grad_(False)
    
    CHUNK = 1024
    N = s.shape[0]
    
    for i in range(0, N, CHUNK):
        end = min(i + CHUNK, N)
        s_chunk = s[i:end]
        t_chunk = t[i:end]
        
        # Match exact PyTorch baseline operations
        s_logits = F.linear(s_chunk, w) / temp
        t_logits = F.linear(t_chunk, w) / temp
        
        s_logprobs = F.log_softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)
        
        # Calculate chunk loss
        chunk_loss = F.kl_div(s_logprobs, t_probs, reduction='sum') * (temp ** 2) / N
        chunk_loss.backward()
        
    return s.grad

def v5_forward_backward(s, t, w, temp=1.0):
    s.grad = None
    s.requires_grad_(True)
    w.requires_grad_(False)
    
    # triton_compute_kl_loss already handles chunking internally for VRAM bounding
    loss = triton_compute_kl_loss(s, t, w, temp)
    loss.backward()
    
    return s.grad


if __name__ == "__main__":
    # Realistic dimensions
    B = 1
    B_blocks = 256
    K = 32
    N = B_blocks * (K - 1)  # 256 * 31 = 7936 valid tokens
    D = 1536
    V = 151936
    
    print(f"Testing with realistic dimensions: N={N}, D={D}, V={V}")
    
    # Initialize tensors
    s = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    t = torch.randn(N, D, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(V, D, device='cuda', dtype=torch.bfloat16)
    
    # ==========================
    # Measure Baseline PyTorch
    # ==========================
    reset_memory_stats()
    baseline_grad = baseline_forward_backward(s, t, w)
    base_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        baseline_forward_backward(s, t, w)
    torch.cuda.synchronize()
    base_time = (time.time() - start) * 1000 / 5
    
    # ==========================
    # Measure V5 Triton
    # ==========================
    reset_memory_stats()
    v5_grad = v5_forward_backward(s, t, w)
    v5_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        v5_forward_backward(s, t, w)
    torch.cuda.synchronize()
    v5_time = (time.time() - start) * 1000 / 5
    
    print("-" * 50)
    print(f"PyTorch Chunked Baseline:")
    print(f"Time: {base_time:.2f} ms")
    print(f"Peak VRAM: {base_vram_mb:.2f} MB")
    print("-" * 50)
    print(f"V5 Hybrid Triton Kernel:")
    print(f"Time: {v5_time:.2f} ms")
    print(f"Peak VRAM: {v5_vram_mb:.2f} MB")
    print("-" * 50)
    
    grad_diff = (baseline_grad - v5_grad).abs().max().item()
    print(f"Max Gradient Difference: {grad_diff:.6f}")
