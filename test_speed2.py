import torch
import time
from transformers import AutoTokenizer
from model import OrthrusQwen35Model

# Load model
model_id = "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda"

model = OrthrusQwen35Model(
    base_model_path=model_id,
    block_size=32,
    dtype=torch.float32,
).to(device)
model.eval()

prompt = "The apple tree, " * 50
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
start_idx = input_ids.shape[1] - 1  # Leave one token for anchor

# Run prefill
with torch.no_grad():
    base_out = model.base_model(input_ids=input_ids[:, :start_idx], use_cache=True)
    past_key_values = base_out.past_key_values

    # Profile variables
    K = 32
    diff_block_ids = torch.randint(0, 1000, (1, K), device=device)
    diff_len = K

    # warmup TorchInductor
    print("Warming up inductor...")
    for _ in range(2):
        causal_limit = torch.full((1, diff_len), start_idx - 1, dtype=torch.long, device=device)
        model(
            input_ids=diff_block_ids,
            is_diffusion_pass=True,
            ar_past_key_values=past_key_values,
            ar_seq_len=start_idx,
            causal_limit=causal_limit,
            use_flex=False
        )

    print("Profiling 10 blocks...")
    time_step1 = 0
    time_step2 = 0
    time_step4 = 0
    time_step5 = 0
    
    linear_indices = [i for i in range(len(past_key_values.layers))
                      if hasattr(past_key_values.layers[i], 'is_recurrent_states_initialized')
                      and past_key_values.layers[i].is_recurrent_states_initialized]

    for block_num in range(10):
        # ── Step 1: Diffusion parallel projection ──────
        t0 = time.perf_counter()
        causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
        for k in range(diff_len):
            causal_limit[0, k] = start_idx - 1
            
        diff_logits = model(
            input_ids=diff_block_ids,
            is_diffusion_pass=True,
            ar_past_key_values=past_key_values,
            ar_seq_len=start_idx,
            causal_limit=causal_limit,
            use_flex=False,
        )
        torch.cuda.synchronize()
        time_step1 += time.perf_counter() - t0
        
        proposed_block = torch.randint(0, 1000, (1, K), device=device)

        # ── Step 2: AR verification ──────
        t1 = time.perf_counter()
        saved_recurrent = {}
        saved_conv = {}
        for li in linear_indices:
            lc = past_key_values.layers[li]
            saved_recurrent[li] = lc.recurrent_states.clone()
            if lc.is_conv_states_initialized:
                saved_conv[li] = lc.conv_states.clone()

        ar_pos_ids = torch.arange(start_idx, start_idx + K, device=device).unsqueeze(0)
        ar_out = model.base_model(
            input_ids=proposed_block, position_ids=ar_pos_ids,
            past_key_values=past_key_values, use_cache=True,
        )
        torch.cuda.synchronize()
        time_step2 += time.perf_counter() - t1

        # Step 3: simulate acceptance of 5 tokens
        acceptance_len = 5
        accepted_end = start_idx + acceptance_len + 1
        
        # ── Step 4: Restore cache ──────
        t2 = time.perf_counter()
        past_key_values.crop(start_idx)
        for li in linear_indices:
            lc = past_key_values.layers[li]
            lc.recurrent_states = saved_recurrent[li]
            if li in saved_conv and lc.is_conv_states_initialized:
                lc.conv_states = saved_conv[li]
        torch.cuda.synchronize()
        time_step4 += time.perf_counter() - t2

        # ── Step 5: Advance cache ──────
        t3 = time.perf_counter()
        accepted_block = proposed_block[:, :acceptance_len + 1]
        ar_pos_accepted = torch.arange(start_idx, accepted_end, device=device).unsqueeze(0)
        ar_out = model.base_model(
            input_ids=accepted_block, position_ids=ar_pos_accepted,
            past_key_values=past_key_values, use_cache=True,
        )
        torch.cuda.synchronize()
        time_step5 += time.perf_counter() - t3
        
        start_idx = accepted_end

print("\n--- PERFORMANCE PROFILE (avg over 10 blocks) ---")
print(f"Step 1 (Diffusion Pass): {time_step1 / 10 * 1000:.2f} ms")
print(f"Step 2 (Proposed Block AR Pass): {time_step2 / 10 * 1000:.2f} ms")
print(f"Step 4 (Cache Restore): {time_step4 / 10 * 1000:.2f} ms")
print(f"Step 5 (Accepted Block AR Pass): {time_step5 / 10 * 1000:.2f} ms")
total = (time_step1 + time_step2 + time_step4 + time_step5) / 10 * 1000
print(f"Total time per block: {total:.2f} ms")
