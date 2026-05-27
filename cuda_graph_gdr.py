"""
CUDA-graphed GDR autograd Function.

Drop-in replacement for chunk_gated_delta_rule that uses CUDA graphs
for the backward pass, eliminating Python dispatch overhead.

Usage:
    from cuda_graph_gdr import cuda_graph_chunk_gated_delta_rule
    # Same API as chunk_gated_delta_rule:
    o, final_state = cuda_graph_chunk_gated_delta_rule(q, k, v, g, beta, ...)
"""
import torch
from typing import Optional, Tuple
from fla.ops.utils import chunk_local_cumsum
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dqkwg
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def _run_backward(q, k, v, v_new, w, g, beta, A, h, do, scale, cs):
    """Run all 5 backward kernels (used for warmup and capture)."""
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=g, scale=scale, chunk_size=cs, use_exp2=False)
    dh, _, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, do=do, dv=dv, g=g,
        scale=scale, chunk_size=cs, use_exp2=False,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, w=w, g=g, h=h,
        dv=dv2, do=do, dh=dh,
        scale=scale, chunk_size=cs, use_exp2=False,
    )
    dk2, dv_out, db, dg2 = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, g=g, A=A,
        dw=dw, du=dv2, use_exp2=False,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=cs, reverse=True)
    return dq, dk, dv_out, db, dg


class _GraphCache:
    """Singleton cache for CUDA graphs, keyed by (B, T, H, HV, K, V, cs)."""
    _cache = {}
    
    @classmethod
    def get_or_create(cls, q, k, v, v_new, w, g, beta, A, h, do, scale, cs):
        key = (q.shape[0], q.shape[1], q.shape[2], v.shape[2], q.shape[3], v.shape[3], cs)
        
        if key not in cls._cache:
            # Create CUDA graph for this shape configuration
            cg = _CudaGraphBackward(q, k, v, v_new, w, g, beta, A, h, do, scale, cs)
            cls._cache[key] = cg
        
        return cls._cache[key]


class _CudaGraphBackward:
    """CUDA-graphed backward for GDR."""
    
    def __init__(self, q, k, v, v_new, w, g, beta, A, h, do, scale, cs):
        self.scale = scale
        self.cs = cs
        
        # Static buffers for inputs
        self._q = q.clone()
        self._k = k.clone()
        self._v = v.clone()
        self._v_new = v_new.clone()
        self._w = w.clone()
        self._g = g.clone()
        self._beta = beta.clone()
        self._A = A.clone()
        self._h = h.clone()
        self._do = do.clone()
        
        # Capture
        self._graph = None
        self._out_dq = None
        self._out_dk = None
        self._out_dv = None
        self._out_db = None
        self._out_dg = None
        self._capture()
    
    def _capture(self):
        """Capture backward in a CUDA graph."""
        # Warmup
        for _ in range(3):
            _run_backward(
                self._q, self._k, self._v, self._v_new, self._w, self._g,
                self._beta, self._A, self._h, self._do,
                self.scale, self.cs,
            )
        torch.cuda.synchronize()
        
        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            dq, dk, dv, db, dg = _run_backward(
                self._q, self._k, self._v, self._v_new, self._w, self._g,
                self._beta, self._A, self._h, self._do,
                self.scale, self.cs,
            )
            self._out_dq = dq
            self._out_dk = dk
            self._out_dv = dv
            self._out_db = db
            self._out_dg = dg
    
    def run(self, q, k, v, v_new, w, g, beta, A, h, do):
        """Copy inputs, replay graph, return outputs."""
        self._q.copy_(q)
        self._k.copy_(k)
        self._v.copy_(v)
        self._v_new.copy_(v_new)
        self._w.copy_(w)
        self._g.copy_(g)
        self._beta.copy_(beta)
        self._A.copy_(A)
        self._h.copy_(h)
        self._do.copy_(do)
        self._graph.replay()
        return self._out_dq, self._out_dk, self._out_dv, self._out_db, self._out_dg


class CudaGraphChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Custom autograd Function with CUDA-graphed backward."""
    
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state,
                use_qk_l2norm_in_kernel, chunk_size):
        # Normalize q, k if needed
        q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        
        # Forward pass
        g_cumsum = chunk_local_cumsum(g, chunk_size=chunk_size)
        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=g_cumsum, beta=beta, chunk_size=chunk_size, use_exp2=False,
        )
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size, use_exp2=False,
        )
        o = chunk_fwd_o(
            q=q, k=k, v=v_new, h=h, g=g_cumsum,
            scale=scale, chunk_size=chunk_size, use_exp2=False,
        )
        
        # Save for backward
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, h, v_new, w, g_cumsum)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        
        if output_final_state:
            return o.to(q.dtype), final_state
        return o.to(q.dtype), None
    
    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        (q, q_rstd, k, k_rstd, v, g, beta, A, h, v_new, w, g_cumsum) = ctx.saved_tensors
        scale = ctx.scale
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel
        cs = ctx.chunk_size
        
        # Get or create CUDA graph for this shape
        cg = _GraphCache.get_or_create(
            q, k, v, v_new, w, g_cumsum, beta, A, h, do, scale, cs,
        )
        
        # Run CUDA-graphed backward
        dq, dk, dv, db, dg = cg.run(q, k, v, v_new, w, g_cumsum, beta, A, h, do)
        
        # Apply l2norm backward if needed
        if use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        
        return dq, dk, dv, dg, db, None, None, None, None, None


def cuda_graph_chunk_gated_delta_rule(
    q, k, v, g, beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    chunk_size=64,
    **kwargs,
):
    """Drop-in replacement for chunk_gated_delta_rule with CUDA-graphed backward."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    
    return CudaGraphChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, chunk_size,
    )
