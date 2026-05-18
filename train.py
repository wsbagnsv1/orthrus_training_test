"""
Training script for Orthrus on SmolLM2-135M.

Aligns the diffusion head predictions with the frozen AR teacher via
forward KL divergence (soft distillation), following the Orthrus paper
(arXiv:2605.12825, Table 5 shows KL > CE for acceptance rate).

Usage:
    python train.py                          # uses config.yaml defaults
    python train.py --config config.yaml      # explicit config
    python train.py --lr 2e-4 --B_blocks 256  # overrides
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# ── suppress torch/distributed noise BEFORE any torch imports ───────────────
os.environ["TORCHAO_CPP_EXT_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["LOGLEVEL"] = "ERROR"
logging.basicConfig(level=logging.ERROR)
for name in ["torch", "torch.distributed", "torch.distributed.elastic",
             "torchao", "transformers"]:
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

import warnings
warnings.filterwarnings("ignore", message=".*record_context_cpp.*")
warnings.filterwarnings("ignore", message=".*non-linux.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── memory optimizations ────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch._inductor.config.triton.cudagraph_trees = False  # no CUDAGraph overhead

import signal
import json
import time
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# ── ensure local package is on path ──────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── local imports ────────────────────────────────────────────────────────────
from model import OrthrusSmolLM2
from collator import OrthrusCollator
from dataset import load_orthrus_dataset, load_multi_dataset, pretokenize_dataset

# ── defaults (mirror paper Table 4) ──────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "K": 32,
        "dtype": "bfloat16",
    },
    "training": {
        "max_seq_len": 2048,
        "B_blocks": 8,               # paper uses 256 for Qwen3 1.7B-8B; tune for 135M
        "epochs": 2,
        "peak_lr": 2.0e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "gradient_clip": 1.0,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,   # effective batch = 32
        "precision": "bfloat16",
        "compile": True,
        "log_every": 50,
        "eval_every": 1000,
        "acceptance_every": 500,      # measure acceptance rate on held-out set
        "save_every": 2000,
        "output_dir": "./checkpoints",
        "diffusion_chunk_blocks": 64,  # micro-batch diffusion blocks at a time
    },
    "data": {
        "dataset": "HuggingFaceTB/smoltalk",
        "dataset_config": None,       # sub-config for datasets like smoltalk ('all')
        "text_key": "text",
        "max_samples": None,          # None = all
        "max_eval_samples": 2000,     # cap validation set
        "eval_split": "test",         # validation split name
        "min_seq_len": 256,
        "mask_token": "<mask>",
    },
    "hardware": {
        "device": "cuda",
        "seed": 42,
    },
}


# ── KL loss (Equation 7 from the paper) ──────────────────────────────────────

# Pre-built index cache (static per B/B_blocks/K config, reused across steps)
_kl_cache: dict = {}

def _get_kl_indices(B: int, B_blocks: int, K: int, device: torch.device):
    """Return (diff_pos, batch_idx, offsets) — cached per shape."""
    key = (B, B_blocks, K)
    if key not in _kl_cache:
        N = B_blocks * (K - 1)
        diff_pos = torch.arange(B_blocks * K, device=device).view(B_blocks, K)
        diff_pos = diff_pos[:, 1:].reshape(-1).unsqueeze(0).expand(B, -1)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
        offsets = torch.arange(0, K - 1, device=device)  # [0..K-2]; teacher at pos a_b+k predicts token a_b+k+1
        _kl_cache[key] = (diff_pos, batch_idx, offsets)
    diff_pos, batch_idx, offsets = _kl_cache[key]
    # Move to correct device if different
    if diff_pos.device != device:
        _kl_cache.clear()
        return _get_kl_indices(B, B_blocks, K, device)
    return diff_pos, batch_idx, offsets


def compute_kl_loss(
    diff_hidden: torch.Tensor,          # [B, B_blocks*K, D]
    ar_hidden_states: torch.Tensor,    # [B, L, D]
    lm_head: nn.Module,
    anchor_positions: torch.Tensor,    # [B, B_blocks]
    K: int,
    target_ids: torch.Tensor,          # [B, B_blocks*K]
    pad_token_id: int,
) -> torch.Tensor:
    """Chunked KL divergence — gather, lm_head, and kl_div all done in chunks."""
    B, B_blocks = anchor_positions.shape
    device = diff_hidden.device
    L = ar_hidden_states.shape[1]
    N = B_blocks * (K - 1)

    diff_pos, batch_idx, offsets = _get_kl_indices(B, B_blocks, K, device)

    # AR positions: a_b + k for k=1..K-1 (teacher at a_b+k-1, predicts a_b+k)
    ar_pos = (anchor_positions.unsqueeze(-1) + offsets).view(B, N).clamp(0, L - 1)

    # Padding mask (small: [B, N])
    tgt = target_ids[batch_idx, diff_pos]                  # [B, N]
    valid = (tgt != pad_token_id)                           # [B, N]

    total_tokens = valid.sum().float()
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Chunked gather + lm_head + KL to avoid [B, N, vocab] intermediates
    CHUNK = 1024  # process 1024 positions at a time
    total_kl = 0.0

    for c in range(0, N, CHUNK):
        c_end = min(c + CHUNK, N)
        chunk_valid = valid[:, c:c_end].float()
        if not chunk_valid.any():
            continue

        # Slice both index tensors to the same chunk
        b_idx = batch_idx[:, c:c_end]      # [B, chunk]
        d_idx = diff_pos[:, c:c_end]       # [B, chunk]
        a_idx = ar_pos[:, c:c_end]         # [B, chunk]

        # Gather only this chunk's positions (hidden states, defer lm_head)
        d_hidden_chunk = diff_hidden[b_idx, d_idx]         # [B, chunk, D]
        ar_hidden_chunk = ar_hidden_states[b_idx, a_idx]   # [B, chunk, D]

        # Apply lm_head to the small chunk only
        d_logits_chunk = lm_head(d_hidden_chunk).float()            # [B, chunk, vocab]

        with torch.no_grad():
            teacher_logits = lm_head(ar_hidden_chunk).float()
            p_teacher = F.softmax(teacher_logits, dim=-1)
            # KL(P||Q) = CE(P, Q) - H(P).  Teacher entropy has zero grad
            teacher_entropy = -(p_teacher * F.log_softmax(teacher_logits, dim=-1)).sum(dim=-1)

        # Fused cross-entropy kernel — avoids materializing student log_softmax
        chunk_ce = F.cross_entropy(
            d_logits_chunk.view(-1, d_logits_chunk.size(-1)),
            p_teacher.view(-1, p_teacher.size(-1)),
            reduction='none',
        ).view(B, -1)

        chunk_kl = chunk_ce - teacher_entropy
        total_kl += (chunk_kl * chunk_valid).sum()

        del d_hidden_chunk, ar_hidden_chunk, d_logits_chunk, teacher_logits, p_teacher, chunk_ce, chunk_kl

    return total_kl / total_tokens


# ── JSONL metrics logger ────────────────────────────────────────────────────

_metrics_file = None

def set_metrics_file(path):
    global _metrics_file
    _metrics_file = path

def log_metrics(step, loss, val_kl=None, accept_rate=None, lr=None, grad_norm=None):
    if not _metrics_file:
        return
    entry = {
        "step": int(step),
        "loss": float(loss) if hasattr(loss, 'item') else loss,
        "val_kl": float(val_kl) if val_kl is not None and hasattr(val_kl, 'item') else val_kl,
        "accept_rate": float(accept_rate) if accept_rate is not None else accept_rate,
        "lr": float(lr) if lr is not None else lr,
        "grad_norm": float(grad_norm) if grad_norm is not None and hasattr(grad_norm, 'item') else grad_norm,
        "time": time.time(),
    }
    with open(_metrics_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── main training function ───────────────────────────────────────────────────

def train(config: Dict[str, Any]):
    # ── setup ────────────────────────────────────────────────────────────────
    device = torch.device(config["hardware"]["device"])
    torch.manual_seed(config["hardware"]["seed"])
    dtype = getattr(torch, config["training"]["precision"])

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # ── metrics log ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(config["training"]["output_dir"], "metrics.jsonl")
    set_metrics_file(metrics_path)
    print(f"  ✓ Logging metrics to {metrics_path}")

    # ── Ctrl+C handler state ─────────────────────────────────────────────────
    _interrupted = [False]  # mutable so inner closure can set it
    def _on_interrupt(signum, frame):
        _interrupted[0] = True
        print("\n\n⚠ Ctrl+C received — will save checkpoint after current step...")
    signal.signal(signal.SIGINT, _on_interrupt)

    # ── tokenizer & <mask> token ─────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    mask_token = config["data"]["mask_token"]
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    print(f"  <mask> token id = {mask_id}")
    print(f"  vocab size = {len(tokenizer)}")

    # ── model ────────────────────────────────────────────────────────────────
    print("Loading OrthrusSmolLM2 model...")
    model = OrthrusSmolLM2(
        base_model_name=config["model"]["base_model"],
        block_size=config["model"]["K"],
        dtype=dtype,
    )
    # Resize embeddings if tokenizer was extended
    model.base_model.resize_token_embeddings(len(tokenizer))

    model = model.to(device=device, dtype=dtype)
    print(f"  Trainable params: {model.trainable_params:,}")
    print(f"  Base model frozen: {all(not p.requires_grad for p in model.base_model.parameters())}")

    # ── dataset ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    data_cfg = config["data"]
    dataset_cfg = data_cfg.get("dataset")
    dataset_config_name = data_cfg.get("dataset_config")  # sub-config for datasets like smoltalk
    if isinstance(dataset_cfg, list):
        # Multi-dataset config
        ds = load_multi_dataset(dataset_cfg, max_total=data_cfg.get("max_samples"))
        text_key = "text"
        val_ds = None
    else:
        ds, text_key = load_orthrus_dataset(
            dataset_name=dataset_cfg,
            config_name=dataset_config_name,
            max_samples=data_cfg.get("max_samples"),
            text_key=data_cfg.get("text_key", "text"),
        )
        # Load a separate validation split (use 'test' split for smoltalk)
        val_split = data_cfg.get("eval_split", "test")
        try:
            val_ds, _ = load_orthrus_dataset(
                dataset_name=dataset_cfg,
                config_name=dataset_config_name,
                split=val_split,
                max_samples=data_cfg.get("max_eval_samples", 2000),
                text_key=text_key,
            )
            print(f"  Validation split: '{val_split}' → {len(val_ds)} examples")
        except Exception:
            print(f"  No '{val_split}' split found — eval will sample from train set")
            val_ds = None

    # Pre-tokenize all datasets (tokenizer runs once, not every batch)
    pt_cache_dir = os.path.join(config["training"]["output_dir"], "pretokenized")
    ds = pretokenize_dataset(ds, tokenizer, config["training"]["max_seq_len"], text_key,
                             cache_dir=pt_cache_dir)
    if val_ds is not None:
        val_ds = pretokenize_dataset(val_ds, tokenizer, config["training"]["max_seq_len"], text_key,
                                     cache_dir=pt_cache_dir)

    collator = OrthrusCollator(
        tokenizer=tokenizer,
        K=config["model"]["K"],
        B_blocks=config["training"]["B_blocks"],
        max_seq_len=config["training"]["max_seq_len"],
        mask_token=mask_token,
        text_key=text_key,
    )

    dataloader = DataLoader(
        ds,
        batch_size=config["training"]["micro_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataloader (separate split)
    val_dataloader = None
    if val_ds is not None:
        val_collator = OrthrusCollator(
            tokenizer=tokenizer,
            K=config["model"]["K"],
            B_blocks=config["training"]["B_blocks"],
            max_seq_len=config["training"]["max_seq_len"],
            mask_token=mask_token,
            text_key=text_key,
        )
        val_dataloader = DataLoader(
            val_ds,
            batch_size=config["training"]["micro_batch_size"],
            shuffle=False,
            collate_fn=val_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    # ── optimizer & scheduler ────────────────────────────────────────────────
    trainable_params = model.get_trainable_params()
    optimizer = AdamW(trainable_params, lr=config["training"]["peak_lr"], betas=(0.9, 0.95))

    total_steps = (
        len(dataloader) // config["training"]["gradient_accumulation_steps"]
        * config["training"]["epochs"]
    )
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    print(f"  Total steps: {total_steps} (warmup: {warmup_steps})")
    print(f"  Effective batch size: "
          f"{config['training']['micro_batch_size'] * config['training']['gradient_accumulation_steps']}")

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    resume_from = config["training"].get("resume_from")
    if resume_from:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, resume_from)

    # Compile after loading weights (so trained parameters aren't lost)
    if config["training"]["compile"]:
        model.compile_diffusion_heads()

    # ── training loop ────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))  # fp16 needs scaling, bf16 doesn't
    accum_loss = 0.0

    def _save_crash_checkpoint(tag="crash"):
        try:
            save_checkpoint(
                model, optimizer, scheduler, global_step, epoch,
                config["training"]["output_dir"], tag,
            )
            print(f"\n  💾 Crash checkpoint saved → "
                  f"{config['training']['output_dir']}/{tag}")
            print(f"  Resume with: --resume {config['training']['output_dir']}/{tag}")
        except Exception as e:
            print(f"\n  ⚠ Could not save crash checkpoint: {e}")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['epochs']} ===")
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        optimizer.zero_grad(set_to_none=True)

        diff_chunk_blocks = config["training"].get("diffusion_chunk_blocks", 64)

        for batch_idx, batch in enumerate(pbar):
            try:
                # ── move to device ──────────────────────────────────────────────
                ar_input_ids = batch["ar_input_ids"].to(device)
                ar_attention_mask = batch["ar_attention_mask"].to(device)
                diff_input_ids = batch["diff_input_ids"].to(device)
                anchor_positions = batch["anchor_positions"].to(device)
                target_ids = batch["target_ids"].to(device)
                causal_limit = batch["causal_limit"].to(device)
    
                B = ar_input_ids.shape[0]
                L = ar_input_ids.shape[1]
                K = config["model"]["K"]
                B_blocks_total = anchor_positions.shape[1]
    
                # Compute actual AR sequence length (non-padding) per batch item
                ar_seq_len = ar_attention_mask.sum(dim=1).max().item()
                ar_input_ids = ar_input_ids[:, :ar_seq_len]
                ar_attention_mask = ar_attention_mask[:, :ar_seq_len]
    
                # Step 1: AR prefill (frozen, no grad) — run ONCE per batch
                with torch.no_grad():
                    ar_kv_cache, ar_hidden = model.forward_ar_prefill(
                        ar_input_ids, ar_attention_mask
                    )
    
                # Step 2+3: Diffusion micro-batching — each chunk of blocks
                # runs forward + backward independently, freeing activations
                batch_loss = 0.0
                for blk_start in range(0, B_blocks_total, diff_chunk_blocks):
                    blk_end = min(blk_start + diff_chunk_blocks, B_blocks_total)
                    n_blocks = blk_end - blk_start
    
                    # Slice inputs for this chunk of blocks
                    tok_start = blk_start * K
                    tok_end = blk_end * K
                    chunk_diff_ids = diff_input_ids[:, tok_start:tok_end]
                    chunk_causal = causal_limit[:, tok_start:tok_end]
                    chunk_anchor = anchor_positions[:, blk_start:blk_end]
                    chunk_target = target_ids[:, tok_start:tok_end]
    
                    # RoPE positions: token at anchor b + k encodes position a_b + k
                    offsets = torch.arange(K, device=device)  # [0, 1, ..., K-1]
                    chunk_positions = (chunk_anchor.unsqueeze(-1) + offsets).view(B, -1)
    
                    diff_hidden = model.forward_diffusion(
                        diff_input_ids=chunk_diff_ids,
                        ar_past_key_values=ar_kv_cache,
                        ar_seq_len=ar_seq_len,
                        causal_limit=chunk_causal,
                        return_hidden=True,
                        diff_position_ids=chunk_positions,
                    )
    
                    chunk_loss = compute_kl_loss(
                        diff_hidden=diff_hidden,
                        ar_hidden_states=ar_hidden,
                        lm_head=model.lm_head,
                        anchor_positions=chunk_anchor,
                        K=K,
                        target_ids=chunk_target,
                        pad_token_id=pad_token_id,
                    )
    
                    # Weight by chunk size so total loss matches unchunked version
                    weight = n_blocks / B_blocks_total
                    scaled_loss = (chunk_loss * weight) / config["training"]["gradient_accumulation_steps"]
    
                    # Backward immediately — frees diffusion activations for this chunk
                    scaler.scale(scaled_loss).backward()
                    batch_loss += chunk_loss.item() * weight
    
                accum_loss += batch_loss
    
                # Free AR intermediates
                del ar_hidden, ar_kv_cache
    
                if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=config["training"]["gradient_clip"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
    
                    global_step += 1
    
                    # ── Ctrl+C interrupt check ─────────────────────────────────
                    if _interrupted[0]:
                        print(f"\n⚠ Interrupted at step {global_step}. Saving checkpoint...")
                        save_checkpoint(
                            model, optimizer, scheduler, global_step, epoch,
                            config["training"]["output_dir"],
                            f"interrupt_step_{global_step}",
                        )
                        print("✓ Saved. Resume with:")
                        print(f"  --resume {config['training']['output_dir']}/interrupt_step_{global_step}")
                        return model
    
                    # ── logging ─────────────────────────────────────────────────
                    if global_step % config["training"]["log_every"] == 0:
                        lr = scheduler.get_last_lr()[0]
                        n_accum = config["training"]["gradient_accumulation_steps"]
                        avg_loss = accum_loss / n_accum
                        pbar.write(
                            f"  Step {global_step:6d} | Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | Grad norm: {grad_norm:.2f}"
                        )
                        log_metrics(global_step, avg_loss, lr=lr, grad_norm=grad_norm)
                        accum_loss = 0.0
    
                    # ── checkpoint ──────────────────────────────────────────────
                    if global_step % config["training"]["save_every"] == 0:
                        save_checkpoint(
                            model, optimizer, scheduler, global_step, epoch,
                            config["training"]["output_dir"],
                            f"step_{global_step}",
                        )
    
                    # ── eval ───────────────────────────────────────────────────
                    if global_step % config["training"]["eval_every"] == 0:
                        eval_dl = val_dataloader if val_dataloader is not None else dataloader
                        eval_loss = evaluate(
                            model, eval_dl, pad_token_id, device, dtype,
                            max_eval_batches=10,
                        )
                        pbar.write(
                            f"  >>> Eval  @ step {global_step:6d} | "
                            f"KL: {eval_loss:.4f}"
                            f"{' (val)' if val_dataloader is not None else ' (train)'} <<<"
                        )
                        log_metrics(global_step, eval_loss, val_kl=eval_loss)
    
                    # ── acceptance rate ────────────────────────────────────────
                    if config["training"].get("acceptance_every") and \
                       global_step % config["training"]["acceptance_every"] == 0:
                        acc_dl = val_dataloader if val_dataloader is not None else dataloader
                        acc_stats = evaluate_acceptance_rate(
                            model, tokenizer, acc_dl, device, dtype,
                            max_examples=8,
                        )
                        # Build per-offset acceptance string: show up to last non-zero
                        o_rates = acc_stats.get("offset_rates", [])
                        if o_rates:
                            last_nonzero = next((i for i in range(len(o_rates)-1, -1, -1) if o_rates[i] > 0), 0)
                            shown = max(last_nonzero + 1, 1)
                            off_str = " | off: " + " ".join(
                                f"{i}:{o_rates[i-1]:.0%}" for i in range(1, shown + 1)
                            )
                        else:
                            off_str = ""
                        pbar.write(
                            f"  >>> Accept @ step {global_step:6d} | "
                            f"rate: {acc_stats['acceptance_rate']:.2%} | "
                            f"avg_len: {acc_stats['avg_acceptance_len']:.1f}/{K} | "
                            f"TPF: {acc_stats['tpf']:.1f}× | "
                            f"blocks: {acc_stats['num_blocks']}{off_str} <<<"
                        )
                        log_metrics(global_step, None, accept_rate=acc_stats['acceptance_rate'])
    
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"\n💥 Crash at epoch {epoch + 1}, step ~{global_step}")
                _save_crash_checkpoint(f"crash_step_{global_step}")
                raise


    # ── final save ───────────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, scheduler, global_step, epoch,
        config["training"]["output_dir"], "final",
    )
    print(f"\n✓ Training complete. Final checkpoint saved to "
          f"{os.path.join(config['training']['output_dir'], 'final')}")
    return model


def save_checkpoint(model, optimizer, scheduler, step, epoch, output_dir, name):
    """Save full training state for resumption."""
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Get raw state dict (unwrap torch.compile wrappers)
    raw_state = {}
    for k, v in model.state_dict().items():
        # Normalize compiled keys: _orig_mod.q_proj.weight → q_proj.weight
        clean_k = k.replace("._orig_mod", "")
        raw_state[clean_k] = v
    torch.save(
        {k: v for k, v in raw_state.items() if "diffusion_heads" in k},
        os.path.join(ckpt_dir, "diffusion_heads.pt"),
    )
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }, os.path.join(ckpt_dir, "trainer_state.pt"))
    print(f"\n  ✓ Checkpoint saved to {ckpt_dir}")


def load_checkpoint(model, optimizer, scheduler, ckpt_dir):
    """Load full training state and return (start_epoch, global_step)."""
    state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), map_location="cpu")
    weights = torch.load(os.path.join(ckpt_dir, "diffusion_heads.pt"), map_location="cpu")
    clean_weights = {k.replace("._orig_mod", ""): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(clean_weights, strict=False)
    diff_missing = [k for k in missing if "diffusion_heads" in k]
    if diff_missing:
        print(f"  ⚠ {len(diff_missing)} diffusion_head keys missing (reverting to copy-init)")
    print(f"  ✓ Resumed from {ckpt_dir} (step {state['step']}, epoch {state['epoch']}) "
          f"[{len(clean_weights)} weights loaded]")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"], state["step"]


@torch.inference_mode()
def evaluate(model, dataloader, pad_token_id, device, dtype, max_eval_batches=10):
    """Run KL loss on a few batches and return average."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_eval_batches:
            break
        ar_input_ids = batch["ar_input_ids"].to(device)
        ar_attention_mask = batch["ar_attention_mask"].to(device)
        diff_input_ids = batch["diff_input_ids"].to(device)
        anchor_positions = batch["anchor_positions"].to(device)
        target_ids = batch["target_ids"].to(device)
        causal_limit = batch["causal_limit"].to(device)
        ar_seq_len = ar_attention_mask.sum(dim=1).max().item()
        ar_input_ids = ar_input_ids[:, :ar_seq_len]
        ar_attention_mask = ar_attention_mask[:, :ar_seq_len]

        ar_kv_cache, ar_hidden = model.forward_ar_prefill(ar_input_ids, ar_attention_mask)
        K = model.block_size
        offsets = torch.arange(K, device=device)
        eval_positions = (anchor_positions.unsqueeze(-1) + offsets).view(
            ar_input_ids.shape[0], -1)
        diff_hidden = model.forward_diffusion(
            diff_input_ids=diff_input_ids,
            ar_past_key_values=ar_kv_cache,
            ar_seq_len=ar_seq_len,
            causal_limit=causal_limit,
            return_hidden=True,
            diff_position_ids=eval_positions,
        )
        loss = compute_kl_loss(diff_hidden, ar_hidden, model.lm_head,
                              anchor_positions, model.block_size,
                              target_ids, pad_token_id)
        total_loss += loss.item()
        total_tokens += 1
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def evaluate_acceptance_rate(
    model, tokenizer, val_dataloader, device, dtype,
    max_examples=8, max_tokens_per_example=64,
):
    """
    Run consensus generation on held-out examples and measure acceptance rate.

    This is the key metric from the Orthrus paper — higher acceptance → higher TPF.
    Targets: >50% early, >85% at convergence.
    """
    model.eval()
    K = model.block_size
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")

    total_acceptances = []
    offset_accepted = [0] * (K + 1)  # per-offset accept counts (positions 1..K)
    offset_tested = [0] * (K + 1)    # per-offset test counts
    total_tokens_gen = 0
    total_passes = 0
    examples_processed = 0

    for batch in val_dataloader:
        if examples_processed >= max_examples:
            break
        input_ids = batch["ar_input_ids"].to(device)
        attention_mask = batch["ar_attention_mask"].to(device)
        B = input_ids.shape[0]

        for i in range(B):
                if examples_processed >= max_examples:
                    break
                seq_len = attention_mask[i].sum().item()
                if seq_len < 64:
                    continue

                # Use first 256 tokens as prompt, generate up to max_tokens_per_example
                prompt_len = min(256, int(seq_len))
                prompt_ids = input_ids[i, :prompt_len]
                current_len = prompt_len

                # AR prefill (1 forward pass, gives hidden states)
                ar_kv_cache, ar_hidden = model.forward_ar_prefill(prompt_ids.unsqueeze(0))
                past_kv = ar_kv_cache
                total_passes += 1

                first_logits = model.lm_head(ar_hidden[:, -1, :])
                first_token = first_logits.argmax(dim=-1).item()
                generated = [first_token]
                total_tokens_gen += 1

                for _ in range(max_tokens_per_example // K + 1):
                    if len(generated) >= max_tokens_per_example:
                        break

                    diff_len = min(K, max_tokens_per_example - len(generated))
                    if diff_len <= 1:
                        break

                    # Build diffusion block
                    anchor_token = generated[-1]
                    diff_block = torch.full(
                        (1, diff_len), mask_id, dtype=torch.long, device=device
                    )
                    diff_block[:, 0] = anchor_token

                    # Causal limit
                    causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
                    causal_limit[0, 0] = current_len - 1
                    for k in range(1, diff_len):
                        causal_limit[0, k] = current_len + k - 1

                    # Diffusion projection (1 forward pass)
                    diff_logits = model.forward_diffusion(
                        diff_input_ids=diff_block,
                        ar_past_key_values=past_kv,
                        ar_seq_len=current_len,
                        causal_limit=causal_limit,
                    )
                    total_passes += 1

                    # Greedy predictions
                    if diff_len > 1:
                        diff_preds = diff_logits[0, 1:diff_len].argmax(dim=-1).tolist()
                    else:
                        diff_preds = []
                    proposed = [anchor_token] + diff_preds

                    # AR verification (1 forward pass)
                    proposed_tensor = torch.tensor([proposed], dtype=torch.long, device=device)
                    ar_pos_ids = torch.arange(current_len, current_len + len(proposed),
                                              device=device).unsqueeze(0)
                    ar_outputs = model.base_model(
                        proposed_tensor, position_ids=ar_pos_ids,
                        past_key_values=past_kv, use_cache=True,
                    )
                    total_passes += 1
                    ar_logits = ar_outputs.logits[0]  # [block_len, vocab]
                    past_kv = ar_outputs.past_key_values

                    # Greedy consensus
                    accepted = [proposed[0]]  # anchor always accepted
                    for k in range(1, len(proposed)):
                        ar_pred = ar_logits[k - 1].argmax().item()
                        offset_tested[k] += 1
                        if proposed[k] == ar_pred:
                            accepted.append(proposed[k])
                            offset_accepted[k] += 1
                        else:
                            accepted.append(ar_pred)
                            break

                    acceptance_len = len(accepted)
                    total_acceptances.append(acceptance_len)

                    generated.extend(accepted)
                    current_len += acceptance_len
                    past_kv.crop(current_len)
                    total_tokens_gen += acceptance_len

                examples_processed += 1

    model.train()
    if not total_acceptances:
        return {"acceptance_rate": 0.0, "avg_acceptance_len": 0.0, "offset_rates": []}

    avg_accept_len = sum(total_acceptances) / len(total_acceptances)
    accept_rate = avg_accept_len / K
    tpf = total_tokens_gen / max(total_passes, 1)
    return {
        "acceptance_rate": accept_rate,
        "avg_acceptance_len": avg_accept_len,
        "num_blocks": len(total_acceptances),
        "tpf": tpf,
        "offset_rates": [
            (offset_accepted[k] / offset_tested[k]) if offset_tested[k] > 0 else 0.0
            for k in range(1, K + 1)
        ],
    }# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthrus on SmolLM2-135M")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--B_blocks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--compile", type=str, default=None)  # "true"/"false"
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    # Load config
    config = DEFAULT_CONFIG.copy()

    # Load default config.yaml (always, unless --config overrides)
    config_path = args.config if args.config else str(Path(__file__).resolve().parent / "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        for section in file_cfg:
            if section in config:
                if isinstance(config[section], dict) and isinstance(file_cfg[section], dict):
                    config[section].update(file_cfg[section])
                else:
                    config[section] = file_cfg[section]
            else:
                config[section] = file_cfg[section]

    # CLI overrides
    overrides = {
        "base_model": ("model", "base_model"),
        "K": ("model", "K"),
        "B_blocks": ("training", "B_blocks"),
        "epochs": ("training", "epochs"),
        "lr": ("training", "peak_lr"),
        "batch_size": ("training", "micro_batch_size"),
        "grad_accum": ("training", "gradient_accumulation_steps"),
        "output_dir": ("training", "output_dir"),
        "dataset": ("data", "dataset"),
        "dataset_config": ("data", "dataset_config"),
    }
    for cli_arg, (section, key) in overrides.items():
        val = getattr(args, cli_arg, None)
        if val is not None:
            config[section][key] = val

    if args.compile is not None:
        config["training"]["compile"] = args.compile.lower() != "false"
    if args.resume is not None:
        config["training"]["resume_from"] = args.resume

    print("Configuration:")
    for section, cfg in config.items():
        print(f"  [{section}]")
        for k, v in cfg.items():
            print(f"    {k}: {v}")

    train(config)
