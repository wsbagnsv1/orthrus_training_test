import torch
import time
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
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

prompt = "The apple tree,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

def trace_infer_bug():
    print("--- TRACING INFER.PY CACHE BUG ---")
    start_idx = input_ids.shape[1]
    
    with torch.no_grad():
        base_out = model.base_model(input_ids=input_ids, use_cache=True)
        past_key_values = base_out.past_key_values
        print(f"Initial cache length: {past_key_values.get_seq_length()}")
        
        # Simulate Step 2: Batch AR forward (proposed_block -> logits)
        K = 32
        proposed_block = torch.randint(0, 1000, (1, K), device=device)
        ar_pos_ids = torch.arange(start_idx, start_idx + K, device=device).unsqueeze(0)
        
        ar_out = model.base_model(
            input_ids=proposed_block, position_ids=ar_pos_ids,
            past_key_values=past_key_values, use_cache=True,
        )
        print(f"Cache length after proposed_block: {past_key_values.get_seq_length()} (Expected: {start_idx + K})")
        
        # Simulate Step 3: Greedy consensus check
        acceptance_len = 5
        accepted_end = start_idx + acceptance_len + 1
        
        # Simulate Step 4: Restore cache
        past_key_values.crop(accepted_end)
        print(f"Cache length after crop(accepted_end): {past_key_values.get_seq_length()} (Expected: {accepted_end})")
        
        # Simulate Step 5: Advance cache with ONLY accepted tokens
        accepted_block = proposed_block[:, :acceptance_len + 1]
        ar_pos_accepted = torch.arange(start_idx, accepted_end, device=device).unsqueeze(0)
        
        ar_out = model.base_model(
            input_ids=accepted_block, position_ids=ar_pos_accepted,
            past_key_values=past_key_values, use_cache=True,
        )
        final_len = past_key_values.get_seq_length()
        print(f"Cache length after step 5: {final_len} (Expected: {accepted_end})")
        
        print("\nCONCLUSION:")
        if final_len > accepted_end:
            print(f"BUG FOUND! Cache length is {final_len} instead of {accepted_end}!")
            print(f"This means {final_len - accepted_end} duplicate tokens were appended to the cache.")
            print("This causes both output divergence (duplicate tokens) and massive slowdowns (exponentially growing cache).")

trace_infer_bug()
