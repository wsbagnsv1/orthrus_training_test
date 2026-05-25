import torch
import time
from custom_gdn_extract import extract_gdn_states

def benchmark():
    B = 1
    T = 2048
    H = 16
    HV = 16
    K = 128
    V = 128
    device = 'cuda'
    
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, HV, V, device=device)
    g = torch.randn(B, T, HV, device=device)
    beta = torch.rand(B, T, HV, device=device).sigmoid()
    
    num_anchors = 256
    anchor_positions = torch.randint(0, T, (num_anchors,)).tolist()
    anchor_mask = torch.full((T,), -1, dtype=torch.int32, device=device)
    for idx, pos in enumerate(anchor_positions):
        anchor_mask[pos] = idx
        
    # Warmup
    for _ in range(5):
        extract_gdn_states(k, v, g, beta, anchor_mask, num_anchors, use_qk_l2norm_in_kernel=True)
        
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.perf_counter()
    iters = 100
    for _ in range(iters):
        extract_gdn_states(k, v, g, beta, anchor_mask, num_anchors, use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    triton_ms = (end - start) / iters * 1000
    print(f"Triton kernel time per layer: {triton_ms:.2f} ms")
    print(f"Triton kernel time total (18 layers): {triton_ms * 18:.2f} ms")

if __name__ == '__main__':
    benchmark()
