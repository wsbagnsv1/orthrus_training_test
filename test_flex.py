import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def test_flex():
    B, H, QL, KVL = 1, 4, 64, 256
    device = "cuda"
    
    q = torch.randn(B, H, QL, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, KVL, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, KVL, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    causal_limit = torch.randint(64, 128, (B, QL), device=device)
    
    def mask_mod(b, h, q_idx, kv_idx):
        return True # fully dense just for testing
        
    def score_mod(score, b, h, q_idx, kv_idx):
        is_ar = kv_idx < (KVL - QL)
        valid_ar = kv_idx <= causal_limit[b, q_idx]
        return torch.where(is_ar & (~valid_ar), float('-inf'), score)
        
    mask = create_block_mask(mask_mod, B, H, QL, KVL, device=device)
    
    # Needs compile to use Triton kernel
    flex = torch.compile(flex_attention, fullgraph=True)
    
    try:
        out = flex(q, k, v, score_mod=score_mod, block_mask=mask)
        print("Flex attention compiled and ran successfully!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    test_flex()
