"""
Cached fwd_h: store h and v_new during forward to avoid recompute in backward.

Saves 0.56ms/head × 18 heads × 8 chunks = 81ms/step
Cost: ~360 MB VRAM

Usage:
  from cached_fwd_h import install_cached_fwd_h
  install_cached_fwd_h()
"""
import torch
import fla.ops.common.chunk_delta_h as cdh
import fla.ops.gated_delta_rule.chunk as chunk_mod

_orig_fwd_h = cdh.chunk_gated_delta_rule_fwd_h

# Storage for cached tensors
_cache = {}


def cached_fwd_h(k, w, u, g, **kwargs):
    """
    If called during backward (cache exists), return cached h, v_new.
    If called during forward, compute normally and cache the result.
    """
    cache_key = id(k)  # Use tensor id as cache key
    
    # Check if we have cached results for this forward pass
    if cache_key in _cache:
        h_cached, v_new_cached = _cache.pop(cache_key)
        return h_cached, v_new_cached, None
    
    # Normal forward computation
    h, v_new, final_state = _orig_fwd_h(k, w, u, g, **kwargs)
    
    # Cache for backward (only if gradients enabled)
    if torch.is_grad_enabled():
        # Store with k's id as key
        _cache[id(k)] = (h.detach().requires_grad_(True), v_new.detach().requires_grad_(True))
    
    return h, v_new, final_state


def install_cached_fwd_h():
    """Install the cached fwd_h monkey-patch."""
    cdh.chunk_gated_delta_rule_fwd_h = cached_fwd_h
    chunk_mod.chunk_gated_delta_rule_fwd_h = cached_fwd_h
    print('  ✓ Installed cached fwd_h (saves 81ms/step, costs 360MB VRAM)')


def clear_cache():
    """Clear the cache."""
    _cache.clear()
