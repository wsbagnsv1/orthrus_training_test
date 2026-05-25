import sys
sys.path.insert(0, "F:/Users/timbe/Desktop/Orthrus/orthrus_qwen3.5")
import torch
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from custom_gdn_extract import extract_gdn_states

torch.manual_seed(42)
B, T, H, K, V = 1, 4, 1, 2, 2
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
g = -torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')

# Expand heads for custom extract (it expects HV instead of H)
HV = H
k_exp = k.repeat_interleave(HV // H, dim=2)
v_exp = v.repeat_interleave(HV // H, dim=2)
anchor_mask = torch.full((T,), -1, dtype=torch.int32, device='cuda')
anchor_mask[T-1] = 0

# 1. Run naive
o_naive, h_naive = naive_recurrent_gated_delta_rule(
    q, k, v, beta, g, output_final_state=True
)

# 2. Run chunk
o_chunk, h_chunk = chunk_gated_delta_rule(
    q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=False, use_exp2=True
)

# 3. Run triton custom (the one we modified)
h_triton = extract_gdn_states(
    k_exp, v_exp, g, beta, anchor_mask, 1, use_qk_l2norm_in_kernel=False
)

print("Naive Final State:")
print(h_naive)
print("\nChunk Final State:")
print(h_chunk)
print("\nTriton Custom Final State:")
print(h_triton)
