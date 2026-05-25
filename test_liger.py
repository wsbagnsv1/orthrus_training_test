import torch
import torch.nn.functional as F
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

def test_liger():
    N = 1024
    D = 2048
    V = 32000
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    s_hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_hidden = torch.randn(N, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    
    with torch.no_grad():
        t_logits = F.linear(t_hidden, w).float()
        t_probs = F.softmax(t_logits, dim=-1).to(dtype)
        
    s_hidden_liger = s_hidden.clone().detach().requires_grad_(True)
    
    try:
        # Liger expects (input, weight, target)
        loss = LigerFusedLinearCrossEntropyFunction.apply(s_hidden_liger, w, t_probs)
        loss.backward()
        print("SUCCESS! Liger FusedLinearCrossEntropy supports soft targets [N, V]!")
    except Exception as e:
        print(f"FAILED with soft targets: {e}")
        
    try:
        t_indices = torch.randint(0, V, (N,), device=device)
        loss = LigerFusedLinearCrossEntropyFunction.apply(s_hidden_liger, w, t_indices)
        print("Liger works with integer targets [N].")
    except Exception as e:
        print(f"FAILED with int targets: {e}")

if __name__ == '__main__':
    test_liger()
