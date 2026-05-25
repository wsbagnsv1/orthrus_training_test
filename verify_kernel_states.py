"""
Diagnostic: compare Triton kernel extracted states vs model's own recurrent states.
Run from the project directory: python verify_kernel_states.py
"""
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_path = "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a test input
text = "The quick brown fox jumps over the lazy dog. " * 20
tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
input_ids = tokens["input_ids"].cuda()
attention_mask = tokens["attention_mask"].cuda()
T = input_ids.shape[1]

# ── Method 1: Run full forward and extract recurrent states ──
print(f"Input length: {T}")
outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
cache = outputs.past_key_values

# Get layer types
layer_types = model.config.layer_types
linear_layers = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]
print(f"Linear layers: {linear_layers}")

# Extract final recurrent states from model cache
model_states = {}
for i in linear_layers:
    layer_cache = cache.layers[i]
    if layer_cache.is_recurrent_states_initialized:
        model_states[i] = layer_cache.recurrent_states.clone()  # [B, HV, K, V]

del outputs, cache
torch.cuda.empty_cache()

# ── Method 2: Run monkey-patched forward_ar_prefill ──
sys.path.insert(0, "F:/Users/timbe/Desktop/Orthrus/orthrus_qwen3.5")
from model import OrthrusQwen35Model

orth_model = OrthrusQwen35Model(
    base_model_path=model_path,
    block_size=32,
    dtype=torch.bfloat16,
    checkpoint_every=0,
)
orth_model = orth_model.cuda()

# Set anchor at the last token position
anchor_pos = T - 1
anchor_positions = torch.tensor([[anchor_pos]], dtype=torch.long, device="cuda")

with torch.no_grad():
    ar_kv_cache, ar_hidden, linear_states, per_block_la_conv = orth_model.forward_ar_prefill(
        input_ids, attention_mask, anchor_positions=anchor_positions,
    )

# Also get the recurrent states from the AR KV cache itself (produced by chunk_gated_delta_rule)
cache_states = {}
for i in linear_layers:
    layer_cache = ar_kv_cache.layers[i]
    if layer_cache.is_recurrent_states_initialized:
        cache_states[i] = layer_cache.recurrent_states.clone()

# Compare
print("\n=== Comparison: Model recurrent_state (standalone) vs Kernel extracted state ===")
for i in linear_layers[:6]:  # just first 6 layers
    if i in model_states and i in linear_states:
        ms = model_states[i].float()  # [1, HV, K, V]
        ks = linear_states[i][:, 0].float()  # [1, HV, K, V]
        
        abs_diff = (ms - ks).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        ms_norm = ms.norm().item()
        ks_norm = ks.norm().item()
        
        print(f"  Layer {i}:")
        print(f"    Model state norm:  {ms_norm:.4f}")
        print(f"    Kernel state norm: {ks_norm:.4f}")
        print(f"    Max abs diff:      {max_diff:.6f}")
        print(f"    Mean abs diff:     {mean_diff:.8f}")

print("\n=== Comparison: cache recurrent_state vs Kernel extracted state ===")
print("  (Both from same forward pass — should be nearly identical)")
for i in linear_layers[:6]:
    if i in cache_states and i in linear_states:
        cs = cache_states[i].float()  # [1, HV, K, V]
        ks = linear_states[i][:, 0].float()  # [1, HV, K, V]
        
        abs_diff = (cs - ks).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        cs_norm = cs.norm().item()
        ks_norm = ks.norm().item()
        
        print(f"  Layer {i}:")
        print(f"    Cache state norm:  {cs_norm:.4f}")
        print(f"    Kernel state norm: {ks_norm:.4f}")
        print(f"    Max abs diff:      {max_diff:.6f}")
        if max_diff > 1.0:
            print(f"    ⚠️ MASSIVE MISMATCH — kernel is computing wrong states!")
        elif max_diff > 0.01:
            print(f"    ⚠️ Significant mismatch")
        else:
            print(f"    ✅ Match")
