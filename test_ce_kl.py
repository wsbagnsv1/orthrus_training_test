import torch
import torch.nn.functional as F

def test_ce_vs_kl():
    N = 1024
    D = 2048
    V = 32000
    
    # Simulate mixed precision training
    dtype = torch.bfloat16
    device = 'cuda'
    
    torch.manual_seed(42)
    s_hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_hidden = torch.randn(N, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    
    # Teacher is frozen
    with torch.no_grad():
        t_logits = F.linear(t_hidden, w).float()
        t_probs = F.softmax(t_logits, dim=-1).to(dtype) # Soft targets
        
    # --- 1. Stable Baseline (KL Div in Float32) ---
    s_hidden_base = s_hidden.clone().detach().requires_grad_(True)
    s_logits_base = F.linear(s_hidden_base, w).float() # Upcast to float32!
    
    s_logprobs_base = F.log_softmax(s_logits_base, dim=-1)
    # Using float32 for stable kl_div
    loss_kl_base = F.kl_div(s_logprobs_base, t_probs.float(), reduction='batchmean')
    loss_kl_base.backward()
    grad_base = s_hidden_base.grad.clone()
    
    # --- 2. Exploding Bug (KL Div in Float16) ---
    s_hidden_bug = s_hidden.clone().detach().requires_grad_(True)
    s_logits_bug = F.linear(s_hidden_bug, w) # NO upcast, stays bfloat16
    
    s_logprobs_bug = F.log_softmax(s_logits_bug, dim=-1)
    loss_kl_bug = F.kl_div(s_logprobs_bug, t_probs, reduction='batchmean')
    loss_kl_bug.backward()
    grad_bug = s_hidden_bug.grad.clone()
    
    # --- 3. The Elegant Fix (Cross Entropy with Soft Targets) ---
    # PyTorch natively fuses log_softmax and CE, doing the math safely
    s_hidden_ce = s_hidden.clone().detach().requires_grad_(True)
    s_logits_ce = F.linear(s_hidden_ce, w) # NO upcast needed!
    
    # Cross entropy natively supports probability distributions as targets!
    loss_ce = F.cross_entropy(s_logits_ce, t_probs, reduction='mean')
    loss_ce.backward()
    grad_ce = s_hidden_ce.grad.clone()
    
    print("--- Gradient Norms ---")
    print(f"Stable KL (FP32) Grad Norm: {grad_base.norm().item():.4f}")
    print(f"Exploding KL (BF16) Grad Norm: {grad_bug.norm().item():.4f}")
    print(f"Cross Entropy (BF16) Grad Norm: {grad_ce.norm().item():.4f}")
    
    print("\n--- Max Difference to Stable FP32 Baseline ---")
    print(f"Buggy KL Diff: {(grad_base - grad_bug).abs().max().item():.6f}")
    print(f"Native CE Diff: {(grad_base - grad_ce).abs().max().item():.6f}")
    
    # Verify exact sum to zero property
    print("\n--- Gradient Mean (Should be exactly 0) ---")
    print(f"Stable KL Mean: {grad_base.mean().item():.8f}")
    print(f"Exploding KL Mean: {grad_bug.mean().item():.8f}")
    print(f"Native CE Mean: {grad_ce.mean().item():.8f}")

if __name__ == '__main__':
    test_ce_vs_kl()
