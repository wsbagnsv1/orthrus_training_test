"""
Triton kernel for extracting conv1d states at anchor positions.

Replaces the CPU loop in model.py (lines 569-581) that does:
    for b_i in range(batch_size):
        for ai, pos in enumerate(anchor_positions[b_i].cpu().tolist()):
            ctx = pre_conv_mixed_qkv[b_i, :, pos-ks:pos]
            conv_buf[b_i, ai] = ctx

This kernel performs the same operation entirely on GPU with no CPU-GPU sync.

Input:
    pre_conv: [B, C, seq_len] - pre-convolution mixed_qkv activations
    anchor_positions: [B, num_anchors] - 0-indexed anchor positions from collator
    ks: int - conv kernel size (number of elements to extract before each anchor)

Output:
    output: [B, num_anchors, C, ks] - extracted conv states, left-padded with zeros
"""

import torch
import triton
import triton.language as tl


@triton.jit
def extract_conv_states_kernel(
    # Input pointers
    pre_conv_ptr,        # [B, C, seq_len] - source tensor (channels-first)
    anchor_pos_ptr,      # [B, num_anchors] - 0-indexed anchor positions
    # Output pointer
    output_ptr,          # [B, num_anchors, C, ks] - destination
    # Strides
    stride_preconv_b,    # stride for batch dim in pre_conv
    stride_preconv_c,    # stride for channel dim in pre_conv
    stride_preconv_s,    # stride for seq_len dim in pre_conv
    stride_anchor_b,     # stride for batch dim in anchor_positions
    stride_anchor_a,     # stride for anchor dim in anchor_positions
    stride_out_b,        # stride for batch dim in output
    stride_out_a,        # stride for anchor dim in output
    stride_out_c,        # stride for channel dim in output
    stride_out_k,        # stride for ks dim in output
    # Dimensions
    seq_len,
    num_anchors,
    C: tl.constexpr,     # number of channels (conv_dim)
    ks: tl.constexpr,    # kernel size to extract
    BC: tl.constexpr,    # block size for channel dimension
):
    """
    Each program instance handles one (batch_element, anchor) pair.
    
    For anchor at 0-indexed position `pos`, we extract:
        pre_conv[b, :, pos-ks : pos]  (last ks elements before pos)
    
    If pos < ks, the first (ks - pos) elements are zero-padded.
    """
    # Program ID maps to (batch_idx, anchor_idx)
    pid = tl.program_id(0)
    batch_idx = pid // num_anchors
    anchor_idx = pid % num_anchors
    
    # Load anchor position (0-indexed from collator)
    # anchor_positions[batch_idx, anchor_idx]
    anchor_ptr = anchor_pos_ptr + batch_idx * stride_anchor_b + anchor_idx * stride_anchor_a
    pos = tl.load(anchor_ptr)
    
    # We want pre_conv[:, :, pos - ks : pos]  (ks tokens before the anchor)
    # pos is 0-indexed, so pos itself is the exclusive end
    end_pos = pos
    
    # Base pointers
    # pre_conv[batch_idx, :, :] 
    src_base = pre_conv_ptr + batch_idx * stride_preconv_b
    # output[batch_idx, anchor_idx, :, :]
    dst_base = output_ptr + batch_idx * stride_out_b + anchor_idx * stride_out_a
    
    # Process channels in blocks of BC
    for c_start in range(0, C, BC):
        c_offsets = c_start + tl.arange(0, BC)
        c_mask = c_offsets < C
        
        # For each of the ks positions to extract
        for k in range(ks):
            # Source sequence position: end_pos - ks + k
            # This is the position we want to read from
            src_seq_pos = end_pos - ks + k
            
            # Check if this position is valid (>= 0 and < seq_len)
            # We need to handle type conversion carefully
            if src_seq_pos >= 0 and src_seq_pos < seq_len:
                # Valid position: load from pre_conv
                # pre_conv[batch_idx, c_offsets, src_seq_pos]
                src_ptr = src_base + c_offsets * stride_preconv_c + src_seq_pos * stride_preconv_s
                val = tl.load(src_ptr, mask=c_mask, other=0.0)
            else:
                # Invalid position (before start of sequence): use zero
                # Use zeros with same dtype as output
                val = tl.zeros([BC], dtype=output_ptr.dtype.element_ty)
            
            # Store to output[batch_idx, anchor_idx, c_offsets, k]
            dst_ptr = dst_base + c_offsets * stride_out_c + k * stride_out_k
            tl.store(dst_ptr, val, mask=c_mask)


def extract_conv_states(
    pre_conv: torch.Tensor,      # [B, C, seq_len]
    anchor_positions: torch.Tensor,  # [B, num_anchors] - 0-indexed
    ks: int,                     # conv kernel size
) -> torch.Tensor:
    """
    Extract conv1d states at anchor positions using a fused Triton kernel.
    
    This is a drop-in replacement for the CPU loop in model.py that avoids
    CPU-GPU synchronization and Python overhead.
    
    Args:
        pre_conv: [B, C, seq_len] - pre-convolution activations (channels-first)
        anchor_positions: [B, num_anchors] - 0-indexed anchor positions from collator
        ks: conv kernel size - number of elements to extract before each anchor
        
    Returns:
        output: [B, num_anchors, C, ks] - extracted conv states
                Left-padded with zeros if anchor is near sequence start
    """
    # Validate inputs
    assert pre_conv.dim() == 3, f"pre_conv must be 3D [B, C, seq_len], got {pre_conv.dim()}D"
    assert anchor_positions.dim() == 2, f"anchor_positions must be 2D [B, num_anchors], got {anchor_positions.dim()}D"
    assert pre_conv.shape[0] == anchor_positions.shape[0], \
        f"Batch size mismatch: pre_conv={pre_conv.shape[0]}, anchor_positions={anchor_positions.shape[0]}"
    
    # Make contiguous for correct strides
    pre_conv = pre_conv.contiguous()
    anchor_positions = anchor_positions.contiguous()
    
    B, C, seq_len = pre_conv.shape
    num_anchors = anchor_positions.shape[1]
    device = pre_conv.device
    
    # Ensure anchor_positions is int32 for Triton
    if anchor_positions.dtype != torch.int32:
        anchor_positions = anchor_positions.to(torch.int32)
    
    # Allocate output tensor
    output = torch.zeros(B, num_anchors, C, ks, device=device, dtype=pre_conv.dtype)
    
    # Grid: one program per (batch_element, anchor) pair
    grid = (B * num_anchors,)
    
    # Choose block size for channel dimension
    # Balance between parallelism and register pressure
    BC = min(triton.next_power_of_2(C), 128)
    
    extract_conv_states_kernel[grid](
        # Pointers
        pre_conv,
        anchor_positions,
        output,
        # Strides
        pre_conv.stride(0),  # stride_preconv_b
        pre_conv.stride(1),  # stride_preconv_c
        pre_conv.stride(2),  # stride_preconv_s
        anchor_positions.stride(0),  # stride_anchor_b
        anchor_positions.stride(1),  # stride_anchor_a
        output.stride(0),  # stride_out_b
        output.stride(1),  # stride_out_a
        output.stride(2),  # stride_out_c
        output.stride(3),  # stride_out_k
        # Dimensions
        seq_len,
        num_anchors,
        C=C,
        ks=ks,
        BC=BC,
        num_warps=4,
        num_stages=2,
    )
    
    return output


# ============================================================================
# Pure Python reference implementation for correctness testing
# ============================================================================

def extract_conv_states_reference(
    pre_conv: torch.Tensor,      # [B, C, seq_len]
    anchor_positions: torch.Tensor,  # [B, num_anchors] - 0-indexed
    ks: int,                     # conv kernel size
) -> torch.Tensor:
    """
    Pure Python reference implementation of conv state extraction.
    
    Use this to verify Triton kernel correctness.
    
    Args:
        pre_conv: [B, C, seq_len] - pre-convolution activations (channels-first)
        anchor_positions: [B, num_anchors] - 0-indexed anchor positions
        ks: conv kernel size
        
    Returns:
        output: [B, num_anchors, C, ks] - extracted conv states
    """
    B, C, seq_len = pre_conv.shape
    num_anchors = anchor_positions.shape[1]
    device = pre_conv.device
    
    output = torch.zeros(B, num_anchors, C, ks, device=device, dtype=pre_conv.dtype)
    
    for b in range(B):
        for ai in range(num_anchors):
            # 0-indexed position → exclusive end position
            pos = anchor_positions[b, ai].item()
            end_pos = int(pos)  # pos is 0-indexed, exclusive end = pos itself
            
            # Extract pre_conv[b, :, end_pos-ks : end_pos]
            for k in range(ks):
                src_pos = end_pos - ks + k
                if 0 <= src_pos < seq_len:
                    output[b, ai, :, k] = pre_conv[b, :, src_pos]
                # else: already zero (padding)
    
    return output


def extract_conv_states_pytorch(
    pre_conv: torch.Tensor,      # [B, C, seq_len]
    anchor_positions: torch.Tensor,  # [B, num_anchors] - 0-indexed
    ks: int,                     # conv kernel size
) -> torch.Tensor:
    """
    PyTorch advanced indexing implementation (no CPU-GPU sync, but not fused).
    
    Faster than CPU loop, slower than Triton kernel.
    Useful as intermediate optimization if Triton has issues.
    
    Args:
        pre_conv: [B, C, seq_len] - pre-convolution activations (channels-first)
        anchor_positions: [B, num_anchors] - 0-indexed anchor positions
        ks: conv kernel size
        
    Returns:
        output: [B, num_anchors, C, ks] - extracted conv states
    """
    B, C, seq_len = pre_conv.shape
    num_anchors = anchor_positions.shape[1]
    device = pre_conv.device
    
    # Convert to exclusive end positions (pos is 0-indexed, so end = pos)
    end_pos = anchor_positions.long()  # [B, num_anchors]
    
    # Create offset indices [0, 1, ..., ks-1]
    offsets = torch.arange(ks, device=device)  # [ks]
    
    # Source indices: end_pos[:, :, None] - ks + offsets[None, None, :]
    # Shape: [B, num_anchors, ks]
    src_indices = end_pos.unsqueeze(-1) - ks + offsets.unsqueeze(0).unsqueeze(0)
    
    # Create validity mask
    valid_mask = (src_indices >= 0) & (src_indices < seq_len)  # [B, num_anchors, ks]
    
    # Clamp indices to valid range for gathering
    src_indices_clamped = src_indices.clamp(0, seq_len - 1)  # [B, num_anchors, ks]
    
    # Use advanced indexing instead of gather (more flexible)
    # pre_conv: [B, C, seq_len]
    # We want to index into seq_len dimension using src_indices_clamped
    
    # Expand batch indices: [B, 1, 1]
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, num_anchors, ks)
    
    # Channel indices: [1, C, 1]
    chan_idx = torch.arange(C, device=device).view(1, C, 1).expand(B, -1, ks)
    
    # Sequence indices: [B, num_anchors, ks] -> [B, 1, num_anchors, ks] -> [B, C, num_anchors, ks]
    seq_idx = src_indices_clamped.unsqueeze(1).expand(-1, C, -1, -1)
    
    # Batch and channel indices need same shape
    batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, num_anchors, ks]
    chan_idx_expanded = chan_idx.unsqueeze(2).expand(-1, -1, num_anchors, -1)  # [B, C, num_anchors, ks]
    
    # Gather using advanced indexing
    gathered = pre_conv[batch_idx_expanded, chan_idx_expanded, seq_idx]  # [B, C, num_anchors, ks]
    
    # Apply validity mask (zero out invalid positions)
    valid_mask_expanded = valid_mask.unsqueeze(1)  # [B, 1, num_anchors, ks]
    gathered = gathered * valid_mask_expanded.to(gathered.dtype)
    
    # Reshape to [B, num_anchors, C, ks]
    output = gathered.permute(0, 2, 1, 3).contiguous()
    
    return output


# ============================================================================
# Self-test when run directly
# ============================================================================

if __name__ == "__main__":
    import time
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping tests")
        exit(0)
    
    print("=" * 60)
    print("Conv State Extraction Kernel Tests")
    print("=" * 60)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Test dimensions (matching Qwen3.5-0.8B GDN layers)
    B = 2
    C = 512  # conv_dim for Qwen3.5-0.8B
    seq_len = 2048
    num_anchors = 256
    ks = 4  # conv_kernel_size
    
    print(f"\nTest dimensions: B={B}, C={C}, seq_len={seq_len}, anchors={num_anchors}, ks={ks}")
    
    # Create test data
    torch.manual_seed(42)
    pre_conv = torch.randn(B, C, seq_len, device=device, dtype=dtype)
    # 0-indexed anchor positions (valid range: 1 to seq_len)
    anchor_positions = torch.randint(ks + 1, seq_len - 1, (B, num_anchors), device=device, dtype=torch.int32)
    
    # ── Correctness Test ──────────────────────────────────────────────────
    print("\n── Correctness Tests ──")
    
    ref_out = extract_conv_states_reference(pre_conv, anchor_positions, ks)
    triton_out = extract_conv_states(pre_conv, anchor_positions, ks)
    pytorch_out = extract_conv_states_pytorch(pre_conv, anchor_positions, ks)
    
    # Convert to float32 for comparison
    ref_f32 = ref_out.float()
    triton_f32 = triton_out.float()
    pytorch_f32 = pytorch_out.float()
    
    triton_diff = (triton_f32 - ref_f32).abs().max().item()
    pytorch_diff = (pytorch_f32 - ref_f32).abs().max().item()
    
    print(f"  Triton vs Reference:  max_diff = {triton_diff:.6f}")
    print(f"  PyTorch vs Reference: max_diff = {pytorch_diff:.6f}")
    
    if triton_diff < 1e-3:
        print("  ✓ Triton kernel matches reference!")
    else:
        print("  ✗ Triton kernel has errors!")
        
    if pytorch_diff < 1e-3:
        print("  ✓ PyTorch indexing matches reference!")
    else:
        print("  ✗ PyTorch indexing has errors!")
    
    # ── Edge Case: Anchor near start ──────────────────────────────────────
    print("\n── Edge Case: Anchor near sequence start ──")
    
    # Anchors at positions 1, 2, 3 (need zero-padding)
    edge_anchors = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
    edge_pre_conv = torch.randn(1, 64, 100, device=device, dtype=dtype)
    
    edge_ref = extract_conv_states_reference(edge_pre_conv, edge_anchors, ks)
    edge_triton = extract_conv_states(edge_pre_conv, edge_anchors, ks)
    
    edge_diff = (edge_triton.float() - edge_ref.float()).abs().max().item()
    print(f"  Max diff: {edge_diff:.6f}")
    
    # Verify zero-padding
    # For anchor at position 1: only last 1 element should be non-zero
    print(f"  Anchor=1 padding check: first {ks-1} elements should be zero")
    print(f"    Actual zeros: {(edge_triton[0, 0, :, :ks-1] == 0).all().item()}")
    
    if edge_diff < 1e-3:
        print("  ✓ Edge case passed!")
    else:
        print("  ✗ Edge case failed!")
    
    # ── Edge Case: Anchor at end ──────────────────────────────────────────
    print("\n── Edge Case: Anchor at sequence end ──")
    
    end_anchors = torch.tensor([[seq_len - 1, seq_len]], device=device, dtype=torch.int32)
    end_ref = extract_conv_states_reference(pre_conv[:1], end_anchors, ks)
    end_triton = extract_conv_states(pre_conv[:1], end_anchors, ks)
    
    end_diff = (end_triton.float() - end_ref.float()).abs().max().item()
    print(f"  Max diff: {end_diff:.6f}")
    
    if end_diff < 1e-3:
        print("  ✓ End case passed!")
    else:
        print("  ✗ End case failed!")
    
    # ── Speed Benchmark ───────────────────────────────────────────────────
    print("\n── Speed Benchmark ──")
    
    # Warmup
    for _ in range(10):
        _ = extract_conv_states(pre_conv, anchor_positions, ks)
        _ = extract_conv_states_pytorch(pre_conv, anchor_positions, ks)
    torch.cuda.synchronize()
    
    # Benchmark Triton kernel
    n_iters = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = extract_conv_states(pre_conv, anchor_positions, ks)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / n_iters * 1000
    
    # Benchmark PyTorch indexing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = extract_conv_states_pytorch(pre_conv, anchor_positions, ks)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / n_iters * 1000
    
    # Benchmark CPU loop (only a few iterations - it's slow)
    import sys
    sys.path.insert(0, ".")
    
    # Simulated CPU loop (without actual CPU-GPU sync for fair comparison)
    def cpu_loop_simulation():
        output = torch.zeros(B, num_anchors, C, ks, device=device, dtype=dtype)
        for b in range(B):
            for ai in range(num_anchors):
                pos = anchor_positions[b, ai].item()
                end_pos = int(pos)  # 0-indexed, exclusive end = pos
                start = max(0, end_pos - ks)
                output[b, ai, :, :end_pos-start] = pre_conv[b, :, start:end_pos]
        return output
    
    # Only 10 iterations for CPU loop (it's very slow)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = cpu_loop_simulation()
    torch.cuda.synchronize()
    cpu_time = (time.perf_counter() - t0) / 10 * 1000
    
    print(f"  CPU loop (Python):  {cpu_time:.2f} ms")
    print(f"  PyTorch indexing:   {pytorch_time:.2f} ms")
    print(f"  Triton kernel:      {triton_time:.2f} ms")
    print(f"\n  Speedup (Triton vs CPU):    {cpu_time / triton_time:.1f}×")
    print(f"  Speedup (Triton vs PyTorch): {pytorch_time / triton_time:.1f}×")
    
    # ── VRAM Usage ────────────────────────────────────────────────────────
    print("\n── VRAM Usage ──")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure baseline
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    
    # Run Triton kernel
    result_triton = extract_conv_states(pre_conv, anchor_positions, ks)
    torch.cuda.synchronize()
    mem_after_triton = torch.cuda.memory_allocated()
    
    triton_output_mb = (mem_after_triton - mem_before) / 1024 / 1024
    
    del result_triton
    torch.cuda.empty_cache()
    
    # Measure PyTorch indexing
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    
    result_pytorch = extract_conv_states_pytorch(pre_conv, anchor_positions, ks)
    torch.cuda.synchronize()
    mem_after_pytorch = torch.cuda.memory_allocated()
    
    pytorch_output_mb = (mem_after_pytorch - mem_before) / 1024 / 1024
    
    del result_pytorch
    torch.cuda.empty_cache()
    
    # Expected output size
    expected_mb = B * num_anchors * C * ks * 2 / 1024 / 1024  # bf16 = 2 bytes
    
    print(f"  Expected output size: {expected_mb:.2f} MB")
    print(f"  Triton actual:        {triton_output_mb:.2f} MB")
    print(f"  PyTorch actual:       {pytorch_output_mb:.2f} MB")
    
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Correctness:  {'✓ PASS' if triton_diff < 1e-3 else '✗ FAIL'}")
    print(f"  Speed:        {triton_time:.2f} ms ({cpu_time / triton_time:.1f}× faster than CPU)")
    print(f"  VRAM:         {triton_output_mb:.2f} MB (minimal overhead)")
    print("=" * 60)
