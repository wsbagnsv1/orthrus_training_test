import torch
import torch.nn.functional as F
from custom_gdn_extract import extract_gdn_states

def pytorch_gdn_extract(k, v, g, beta, anchor_positions, use_qk_l2norm_in_kernel=True):
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    
    if use_qk_l2norm_in_kernel:
        k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
        
    anchor_states = torch.zeros(B, len(anchor_positions), HV, K, V, dtype=torch.float32, device=k.device)
    h = torch.zeros(B, HV, K, V, dtype=torch.float32, device=k.device)
    
    # Expand k if H < HV
    if H < HV:
        k_expanded = k.repeat_interleave(HV // H, dim=2)
    else:
        k_expanded = k
        
    for i in range(T):
        b_k = k_expanded[:, i] # (B, HV, K)
        b_v = v[:, i]          # (B, HV, V)
        b_g = g[:, i]          # (B, HV)
        b_beta = beta[:, i]    # (B, HV)
        
        # Delta: v - h^T k
        # h: (B, HV, K, V)
        # b_k.unsqueeze(-1): (B, HV, K, 1)
        h_T_k = (h * b_k.unsqueeze(-1)).sum(dim=-2) # -> (B, HV, V)
        
        b_v_new = b_beta.unsqueeze(-1) * (b_v - h_T_k)
        
        # Decay
        h = h * torch.exp(b_g)[..., None, None]
        
        # Update
        h = h + b_k.unsqueeze(-1) * b_v_new.unsqueeze(-2)
        
        if i in anchor_positions:
            idx = anchor_positions.index(i)
            anchor_states[:, idx] = h.clone()
            
    return anchor_states

def test_extract():
    B = 2
    T = 256
    H = 4
    HV = 8
    K = 128
    V = 128
    device = 'cuda'
    
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, HV, V, device=device)
    # g should be negative to prevent exponential explosion
    g = F.logsigmoid(torch.randn(B, T, HV, device=device))
    beta = torch.rand(B, T, HV, device=device).sigmoid()
    
    anchor_positions = [31, 64, 127, 200, 255]
    num_anchors = len(anchor_positions)
    
    anchor_mask = torch.full((T,), -1, dtype=torch.int32, device=device)
    for idx, pos in enumerate(anchor_positions):
        anchor_mask[pos] = idx
        
    print("Running PyTorch reference...")
    expected = pytorch_gdn_extract(k, v, g, beta, anchor_positions, use_qk_l2norm_in_kernel=True)
    
    print("Running Triton kernel...")
    actual = extract_gdn_states(k, v, g, beta, anchor_mask, num_anchors, use_qk_l2norm_in_kernel=True)
    
    diff = torch.abs(expected - actual).max().item()
    print(f"Max difference: {diff}")
    
    assert diff < 1e-4, f"Test failed! Max difference is {diff}"
    print("Test passed!")

if __name__ == '__main__':
    test_extract()
