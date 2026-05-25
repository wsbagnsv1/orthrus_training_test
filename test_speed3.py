import torch
import time
from transformers import AutoTokenizer
from model import OrthrusQwen35Model
from transformers.cache_utils import DynamicCache

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

def get_speed(model, seq_len):
    B = 1
    K = 32
    diff_block = torch.full((1, K), 0, dtype=torch.long, device=device)
    
    with torch.no_grad():
        base_out = model.base_model(input_ids=input_ids[:, :seq_len], use_cache=True)
        past_kv = base_out.past_key_values
        
        linear_indices = [i for i in range(len(past_kv.layers))
                          if hasattr(past_kv.layers[i], 'is_recurrent_states_initialized')
                          and past_kv.layers[i].is_recurrent_states_initialized]
                          
        time_diff = 0
        time_ar1 = 0
        time_ar2 = 0
        
        print("Warming up inductor...")
        for _ in range(2):
            causal_limit = torch.zeros(1, K, dtype=torch.long, device=device)
            model(
                input_ids=diff_block,
                is_diffusion_pass=True,
                ar_past_key_values=past_kv,
                ar_seq_len=seq_len,
                causal_limit=causal_limit,
                use_flex=False
            )
            
        print("Profiling...")
        for _ in range(10):
            # Step 1: Diffusion
            t0 = time.perf_counter()
            causal_limit = torch.zeros(1, K, dtype=torch.long, device=device)
            model(
                input_ids=diff_block,
                is_diffusion_pass=True,
                ar_past_key_values=past_kv,
                ar_seq_len=seq_len,
                causal_limit=causal_limit,
                use_flex=False
            )
            torch.cuda.synchronize()
            time_diff += time.perf_counter() - t0
            
            # Step 2: AR full block
            t1 = time.perf_counter()
            ar_out = model.base_model(
                input_ids=diff_block, position_ids=torch.arange(seq_len, seq_len+K, device=device).unsqueeze(0),
                past_key_values=past_kv, use_cache=True,
            )
            torch.cuda.synchronize()
            time_ar1 += time.perf_counter() - t1
            
            # Crop
            past_kv.crop(seq_len)
            for li in linear_indices:
                pass # ignore states
                
            # Step 5: AR accepted block
            t2 = time.perf_counter()
            ar_out2 = model.base_model(
                input_ids=diff_block[:, :5], position_ids=torch.arange(seq_len, seq_len+5, device=device).unsqueeze(0),
                past_key_values=past_kv, use_cache=True,
            )
            torch.cuda.synchronize()
            time_ar2 += time.perf_counter() - t2
            
            past_kv.crop(seq_len) # reset for loop
            
        print(f"Step 1 (Diff): {time_diff/10*1000:.2f} ms")
        print(f"Step 2 (AR 32): {time_ar1/10*1000:.2f} ms")
        print(f"Step 5 (AR 5): {time_ar2/10*1000:.2f} ms")
        print(f"Total per block: {(time_diff+time_ar1+time_ar2)/10*1000:.2f} ms")

get_speed(model, input_ids.shape[1])
