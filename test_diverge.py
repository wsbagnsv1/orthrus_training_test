import torch
import time
from transformers import AutoTokenizer
from model import OrthrusQwen35Model

# Load model
model_id = "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if "<mask>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})

device = "cuda"
model = OrthrusQwen35Model(
    base_model_path=model_id,
    block_size=32,
    dtype=torch.float32,
).to(device)
model.eval()

# Check output divergence
prompt = "The apple tree,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

def get_diff_logits(model, seq_len):
    B = 1
    K = 32
    diff_block = torch.full((1, K), tokenizer.convert_tokens_to_ids("<mask>"), dtype=torch.long, device=device)
    diff_block[0, 0] = input_ids[0, -1] # anchor
    
    # Run prefill to get cache
    with torch.no_grad():
        base_out = model.base_model(input_ids=input_ids[:, :seq_len], use_cache=True)
        past_kv = base_out.past_key_values
        
        t0 = time.time()
        # Run diffusion pass
        diff_logits = model(
            input_ids=diff_block,
            is_diffusion_pass=True,
            ar_past_key_values=past_kv,
            ar_seq_len=seq_len,
            causal_limit=torch.full((1, K), seq_len - 1, dtype=torch.long, device=device),
            use_flex=False
        )
        torch.cuda.synchronize()
        t1 = time.time()
        
    return diff_logits, t1 - t0

logits_1, t_1 = get_diff_logits(model, input_ids.shape[1])
logits_2, t_2 = get_diff_logits(model, input_ids.shape[1])

print(f"Time 1: {t_1*1000:.2f} ms")
print(f"Time 2: {t_2*1000:.2f} ms")
print("Max diff:", (logits_1 - logits_2).abs().max().item())
