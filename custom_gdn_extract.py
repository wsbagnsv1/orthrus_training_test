import torch
import triton
import triton.language as tl

@triton.jit
def extract_gdn_states_at_positions_kernel(
    k,
    v,
    g,
    beta,
    anchor_mask,      
    anchor_states,
    T,
    num_anchors,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    # Program IDs
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    
    # We assume Grouped Value Attention where H divides HV or they are equal.
    # In Qwen3.5 0.8B GDN, HK = 16, HV = 16, so i_h == i_hv
    i_h = i_hv // (HV // H)
    
    bos = i_n * T
    
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    
    # Pointers
    # k: [B, T, H, K]
    p_k = k + (bos * H + i_h) * K + o_k
    # v: [B, T, HV, V]
    p_v = v + (bos * HV + i_hv) * V + o_v
    # g: [B, T, HV]
    p_g = g + bos * HV + i_hv
    # beta: [B, T, HV]
    p_beta = beta + bos * HV + i_hv
    
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    
    # Base pointer for this batch element's anchor_mask row: anchor_mask[i_n, :]
    p_anchor_mask = anchor_mask + i_n * T
    
    # Sequential scan over tokens
    for i in tl.range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            
        # GDN Decay
        b_h *= tl.exp(b_g)
        
        # GDN Delta and Update
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        
        # Check if the current token is an anchor (per-batch-element)
        anchor_idx = tl.load(p_anchor_mask + i)
        if anchor_idx >= 0:
            # anchor_states: [B, num_anchors, HV, K, V]
            # layout: i_n (batch), anchor_idx (anchor), i_hv (head), K, V
            p_anchor = anchor_states + (i_n * num_anchors * HV * K * V) + \
                       (anchor_idx * HV * K * V) + \
                       (i_hv * K * V) + \
                       o_k[:, None] * V + o_v[None, :]
            tl.store(p_anchor, b_h.to(anchor_states.dtype.element_ty), mask=mask_h)
            
        # Advance pointers to next token
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV

def extract_gdn_states(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    anchor_mask: torch.Tensor, 
    num_anchors: int,
    use_qk_l2norm_in_kernel: bool = True
) -> torch.Tensor:
    """
    Extract GDN recurrent states at anchor positions using a fused Triton kernel.

    Args:
        k: (B, T, H, K)
        v: (B, T, HV, V)
        g: (B, T, HV) pre-activation log decay
        beta: (B, T, HV)
        anchor_mask: (B, T) int32 tensor mapping token index to anchor_idx per batch
                     element, or -1 if not an anchor.
        num_anchors: int, the maximum number of anchors.
    Returns:
        anchor_states: (B, num_anchors, HV, K, V)
    """
    # Force contiguous: torch.split returns views with non-standard strides
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    
    assert anchor_mask.ndim == 2, f"anchor_mask must be 2D (B, T), got shape {anchor_mask.shape}"
    assert anchor_mask.shape == (B, T), (
        f"anchor_mask shape {anchor_mask.shape} must match (B={B}, T={T})"
    )
    assert anchor_mask.dtype == torch.int32, "anchor_mask must be int32"
    
    anchor_mask = anchor_mask.contiguous()
    
    BK = triton.next_power_of_2(K)
    # BV needs to be small enough to avoid exceeding shared memory / register limits
    BV = min(16, triton.next_power_of_2(V)) 
    NV = triton.cdiv(V, BV)
    
    anchor_states = torch.zeros((B, num_anchors, HV, K, V), dtype=torch.float32, device=k.device)
    
    grid = (NV, B * HV)
    extract_gdn_states_at_positions_kernel[grid](
        k, v, g, beta,
        anchor_mask,
        anchor_states,
        T, num_anchors,
        H, HV, K, V, BK, BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=2,
        num_stages=2,
    )
    
    return anchor_states
