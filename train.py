"""
Training script for Orthrus on Qwen3.5-0.8B.

Aligns the diffusion head predictions with the frozen AR teacher via
forward KL divergence (soft distillation), following the Orthrus paper
(arXiv:2605.12825, Table 5 shows KL > CE for acceptance rate).

Adapted from orthrus_smollm2/train.py for Qwen3.5's mixed
(full_attention + linear_attention) architecture.

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

# Clear torch.compile inductor cache to prevent stale kernel issues
# (causes gradient explosion on every 2nd run without this)
os.environ["TORCHINDUCTOR_FORCE_DISABLE"] = "1"

import torch

# Force Tensor Cores to accumulate in FP32 instead of BF16 to prevent gradient explosions in massive MLPs
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
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
import random as python_random
import numpy
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

# ── ensure local package is on path ──────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── local imports ────────────────────────────────────────────────────────────
from model import OrthrusQwen35Model
from collator import OrthrusCollator
from dataset import load_orthrus_dataset, load_multi_dataset, pretokenize_dataset
from triton_kl_loss import triton_compute_kl_loss

# ── defaults (mirror paper Table 4) ──────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "base_model": "F:/Users/timbe/Desktop/Orthrus/Qwen3.5-0.8B",
        "K": 32,
        "dtype": "bfloat16",
    },
    "training": {
        "max_seq_len": 2048,
        "B_blocks": 256,
        "epochs": 2,
        "peak_lr": 2.0e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "gradient_clip": 1.0,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,   # effective batch = 32
        "precision": "bfloat16",
        "compile": True,
        "optimizer": "adamw",      # "adamw" or "muon"
        "muon_lr_multiplier": 66.7,  # Muon LR = peak_lr × multiplier (norms keep peak_lr)
        "weight_decay": 0.0,       # AdamW weight decay (also applies to norm params under Muon)
        "muon_weight_decay": 0.0,  # Muon weight decay (recommended 0.01–0.1)
        "log_every": 1,
        "eval_every": 10,
        "acceptance_every": 20,
        "save_every": 500,
        "output_dir": "./checkpoints",
        "diffusion_chunk_blocks": 32,
        "teacher_bf16": True,
        "repr_align": False,           # enable representation alignment aux loss
        "repr_align_weight": 0.3,      # starting weight (decayed over training)
        "repr_align_weight_end": 0.01, # final weight after decay
        "repr_align_decay_steps": 500, # linearly decay weight over this many steps
        "repr_align_subsample": 0.25,  # fraction of valid tokens to subsample
    },
    "data": {
        "dataset": "HuggingFaceTB/smoltalk",
        "dataset_config": "all",
        "text_key": "text",
        "max_samples": None,
        "max_eval_samples": 2000,
        "eval_split": "test",
        "min_seq_len": 512,
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
    """Return (diff_pos, batch_idx, offsets) - cached per shape."""
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
    if diff_pos.device.type != torch.device(device).type:
        _kl_cache.clear()
        return _get_kl_indices(B, B_blocks, K, device)
    return diff_pos, batch_idx, offsets


def compute_kl_loss(
    diff_hidden: torch.Tensor,          # [B, B_blocks*K, D]
    ar_hidden_states: torch.Tensor,     # [B, L, D]
    lm_head: nn.Module,
    anchor_positions: torch.Tensor,     # [B, B_blocks]
    K: int,
    target_ids: torch.Tensor,           # [B, B_blocks*K]
    pad_token_id: int,
    forward_only: bool = False,
    repr_weight: float = 0.0,           # representation alignment aux loss weight (0 = off)
    repr_subsample: float = 0.25,       # fraction of valid tokens to subsample for repr loss
) -> tuple[torch.Tensor, float]:
    """
    Computes KL distillation using a custom Fused Triton Kernel.
    Valid tokens are filtered first to avoid calculating logits for padding.

    forward_only: use loss-only Triton path (eval); no grad buffers.
    """
    B, B_blocks = anchor_positions.shape
    device = diff_hidden.device
    L = ar_hidden_states.shape[1]
    N = B_blocks * (K - 1)

    diff_pos, batch_idx, offsets = _get_kl_indices(B, B_blocks, K, device)
    ar_pos = (anchor_positions.unsqueeze(-1) + offsets).view(B, N).clamp(0, L - 1)

    tgt = target_ids[batch_idx, diff_pos]
    valid = (tgt != pad_token_id)
    total_tokens = valid.sum().float()
    if total_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=not forward_only)

    s_chunk = diff_hidden[batch_idx, diff_pos]
    t_chunk = ar_hidden_states[batch_idx, ar_pos]

    valid_flat = valid.reshape(-1)
    s_chunk_flat = s_chunk.reshape(-1, s_chunk.size(-1))
    t_chunk_flat = t_chunk.reshape(-1, t_chunk.size(-1))

    s_valid = s_chunk_flat[valid_flat]
    t_valid = t_chunk_flat[valid_flat]

    if s_valid.size(0) > 0:
        if forward_only:
            from triton_kl_loss import triton_compute_kl_loss_fwd_only
            kl_mean = triton_compute_kl_loss_fwd_only(s_valid, t_valid, lm_head.weight)
        else:
            kl_mean = triton_compute_kl_loss(s_valid, t_valid, lm_head.weight)

        loss = kl_mean * (s_valid.size(0) / total_tokens)
        repr_val = 0.0

        # ── representation alignment aux loss ───────────────────────────
        if repr_weight > 0 and not forward_only and s_valid.size(0) > 4:
            n = s_valid.size(0)
            k = max(1, int(n * repr_subsample))
            idx = torch.randperm(n, device=device)[:k]
            s_norm = F.normalize(s_valid[idx].float(), dim=-1)
            t_norm = F.normalize(t_valid[idx].float(), dim=-1)
            repr_loss = 1.0 - (s_norm * t_norm).sum(dim=-1).mean()
            repr_val = repr_loss.item()
            loss = loss + repr_weight * repr_loss

        return loss, repr_val

    return torch.tensor(0.0, device=device, requires_grad=not forward_only), 0.0


# ── JSONL metrics logger (async via background thread) ──────────────────────

import threading, queue
_metrics_queue: queue.Queue | None = None
_metrics_thread: threading.Thread | None = None
_metrics_done = threading.Event()

def _metrics_worker():
    while not _metrics_done.is_set():
        try:
            entry = _metrics_queue.get(timeout=1.0)
            with open(_metrics_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except queue.Empty:
            continue

def set_metrics_file(path):
    global _metrics_file, _metrics_queue, _metrics_thread
    _metrics_file = path
    _metrics_queue = queue.Queue()
    _metrics_thread = threading.Thread(target=_metrics_worker, daemon=True)
    _metrics_thread.start()

def log_metrics(step, loss, val_kl=None, accept_rate=None, lr=None, grad_norm=None):
    if not _metrics_queue:
        return
    _metrics_queue.put({
        "step": int(step),
        "loss": float(loss) if hasattr(loss, 'item') else loss,
        "val_kl": float(val_kl) if val_kl is not None and hasattr(val_kl, 'item') else val_kl,
        "accept_rate": float(accept_rate) if accept_rate is not None else accept_rate,
        "lr": float(lr) if lr is not None else lr,
        "grad_norm": float(grad_norm) if grad_norm is not None and hasattr(grad_norm, 'item') else grad_norm,
        "time": time.time(),
    })


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
        print("\n\n⚠ Ctrl+C received - will save checkpoint after current step...")
    signal.signal(signal.SIGINT, _on_interrupt)

    # ── tokenizer & mask token ──────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])
    mask_token = config["data"]["mask_token"]
    
    old_vocab_size = len(tokenizer)
    if mask_token not in tokenizer.get_vocab():
        print(f"  Adding new mask token: {mask_token}")
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        new_vocab_size = len(tokenizer)
        print(f"  Vocab: {old_vocab_size} -> {new_vocab_size}")
    else:
        print(f"  Mask token '{mask_token}' already in vocab")
        new_vocab_size = old_vocab_size
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    print(f"  Mask token: {mask_token} (id={mask_id})")
    print(f"  Vocab size: {len(tokenizer)}")

    # ── model ────────────────────────────────────────────────────────────────
    print("Loading OrthrusQwen35Model...")
    model = OrthrusQwen35Model(
        base_model_path=config["model"]["base_model"],
        block_size=config["model"]["K"],
        dtype=dtype,
    )
    
    # Initialize new mask token embedding if it was just added
    if new_vocab_size > old_vocab_size:
        print(f"  Initializing new mask token embedding...")
        with torch.no_grad():
            # Initialize from mean of ALL embeddings (multimodal-safe!)
            all_embeddings = model.base_model.model.embed_tokens.weight[:old_vocab_size]
            mean_embedding = all_embeddings.mean(dim=0)
            model.base_model.model.embed_tokens.weight[mask_id] = mean_embedding
            
            # Also init LM head
            all_lm_head = model.base_model.lm_head.weight[:old_vocab_size]
            mean_lm_head = all_lm_head.mean(dim=0)
            model.base_model.lm_head.weight[mask_id] = mean_lm_head
            
            print(f"  Initialized from mean of {old_vocab_size} embeddings")
            print(f"  This is multimodal-safe (not tied to text/audio/vision)")
    
    model = model.to(device=device, dtype=dtype)
    print(f"  Trainable params: {model.trainable_params:,}")
    print(f"  Frozen backbone: {all(not p.requires_grad for p in model.base_model.parameters())}")

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
            tokenizer=tokenizer,
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
                tokenizer=tokenizer,
            )
            print(f"  Validation split: '{val_split}' → {len(val_ds)} examples")
        except Exception:
            print(f"  No '{val_split}' split found - eval will sample from train set")
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
            batch_size=config["training"].get("eval_batch_size", 1),
            shuffle=False,
            collate_fn=val_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    # ── optimizer & scheduler ────────────────────────────────────────────────
    trainable_params = model.get_trainable_params()

    optimizer_type = config["training"].get("optimizer", "adamw").lower()
    optimizers = []

    if optimizer_type == "muon":
        try:
            from muon import Muon
        except ImportError:
            print("  ⚠ Muon not installed — falling back to AdamW")
            optimizer_type = "adamw"
        # Muon requires torch.distributed; init dummy group for single GPU
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if optimizer_type == "muon":
        proj_params, norm_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name for kw in ("q_proj", "k_proj", "v_proj", "o_proj")):
                proj_params.append(p)
            else:
                norm_params.append(p)
        muon_wd = config["training"].get("muon_weight_decay", 0.0)
        adamw_wd = config["training"].get("weight_decay", 0.0)
        optimizers.append(Muon(proj_params, lr=config["training"]["peak_lr"] * config["training"].get("muon_lr_multiplier", 4.0), weight_decay=muon_wd))
        if norm_params:
            optimizers.append(AdamW(norm_params, lr=config["training"]["peak_lr"], betas=(0.9, 0.95), fused=True, weight_decay=adamw_wd))
        print(f"  Optimizers: Muon (proj) + AdamW (norms)")
    else:
        optimizers.append(AdamW(trainable_params, lr=config["training"]["peak_lr"], betas=(0.9, 0.95), fused=True, weight_decay=config["training"].get("weight_decay", 0.0)))
        print(f"  Optimizer: AdamW")

    repr_cfg = config["training"].get("repr_align", False)
    if repr_cfg:
        rw = config["training"].get("repr_align_weight", 0.3)
        rw_end = config["training"].get("repr_align_weight_end", 0.01)
        rw_decay = config["training"].get("repr_align_decay_steps", 500)
        rs = config["training"].get("repr_align_subsample", 0.25)
        print(f"  Repr alignment: ON (weight {rw}→{rw_end} over {rw_decay} steps, subsample={rs})")

    optimizer = optimizers[0]  # canonical ref for save/load/scheduler

    total_steps = (
        len(dataloader) // config["training"]["gradient_accumulation_steps"]
        * config["training"]["epochs"]
    )
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    import math
    peak_lr = config["training"]["peak_lr"]
    def _lr_fn(step):
        if step < warmup_steps:
            frac = step / warmup_steps
            return (1e-3 + (1.0 - 1e-3) * frac)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # LambdaLR needs initial_lr set on every param group when last_epoch >= 0
    for opt in optimizers:
        for pg in opt.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_fn, last_epoch=0)

    config["training"]["_total_steps"] = total_steps
    print(f"  Total steps: {total_steps} (warmup: {warmup_steps})")
    print(f"  Effective batch size: "
          f"{config['training']['micro_batch_size'] * config['training']['gradient_accumulation_steps']}")

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    schedule_step = 0  # tracks LR schedule position (may differ on batch-size change)
    resume_from = config["training"].get("resume_from")
    if resume_from:
        start_epoch, global_step, rng_state, dataloader_offset = load_checkpoint(model, optimizer, scheduler, resume_from, optimizers=optimizers)
        # Rebuild scheduler with current config LR (checkpoint LR may differ)
        # Also check saved schedule metadata in case batch size / total steps changed
        schedule_step = global_step  # default: no remapping needed
        meta_path = os.path.join(resume_from, "schedule_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            old_total = meta.get("total_steps", 0)
            if old_total and old_total != total_steps:
                print(f"  ⚠ Schedule changed: old total_steps={old_total} → new={total_steps}")
                # Re-map progress to new schedule so LR stays proportional
                old_warmup = int(old_total * config["training"]["warmup_ratio"])
                if global_step < old_warmup:
                    frac = global_step / max(1, old_warmup)
                    schedule_step = max(0, int(warmup_steps * frac) - 1)
                else:
                    progress = (global_step - old_warmup) / max(1, old_total - old_warmup)
                    schedule_step = warmup_steps + int((total_steps - warmup_steps) * progress)
                print(f"  → LR schedule mapped: global_step={global_step} → schedule_step={schedule_step}")
        # Rebuild scheduler + LR on all optimizers
        muon_mult = config["training"].get("muon_lr_multiplier", 4.0)
        for i, opt in enumerate(optimizers):
            lr = peak_lr * muon_mult if (optimizer_type == "muon" and i == 0) else peak_lr
            for pg in opt.param_groups:
                pg['lr'] = lr
                pg['initial_lr'] = lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_fn, last_epoch=schedule_step)
        current_lr = peak_lr * muon_mult * _lr_fn(schedule_step) if optimizer_type == "muon" else peak_lr * _lr_fn(schedule_step)
        print(f"  Rebuilt scheduler: peak={peak_lr:.2e} "
              f"schedule_step={schedule_step} LR={current_lr:.2e}")

    # Compile after loading weights (so trained parameters aren't lost)
    if config["training"]["compile"]:
        model.compile_diffusion_heads()

    # ── restore RNG + skip DataLoader on resume ──────────────────────────────
    dataloader_offset = 0
    if resume_from and rng_state is not None:
        torch.random.set_rng_state(rng_state["torch_cpu"])
        if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["torch_cuda"])
        if rng_state.get("python") is not None:
            import random as _random
            _random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            try:
                import numpy as _np
                _np.random.set_state(rng_state["numpy"])
            except ImportError:
                pass
        print(f"  ✓ RNG states restored")
        # Compute how many batches to skip within the current epoch
        n_accum = config["training"]["gradient_accumulation_steps"]
        batches_consumed = global_step * n_accum
        dataloader_offset = batches_consumed % len(dataloader)
        print(f"  Will skip {dataloader_offset} batches to resume DataLoader position")

    # ── training loop ────────────────────────────────────────────────────────
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # fp16 needs scaling, bf16 doesn't
    accum_loss = 0.0
    accum_repr = 0.0
    _profile_step = global_step + 3  # print detailed breakdown 3 steps after resume/ start

    def _get_rng_states():
        try:
            np_state = numpy.random.get_state()
        except Exception:
            np_state = None
        return {
            "python": python_random.getstate(),
            "numpy": np_state,
        }

    def _get_batch_offset():
        n_accum = config["training"]["gradient_accumulation_steps"]
        return (global_step * n_accum) % len(dataloader)

    def _save_crash_checkpoint(tag="crash"):
        try:
            save_checkpoint(
                model, optimizer, scheduler, global_step, epoch,
                config["training"]["output_dir"], tag,
                total_steps=total_steps,
                optimizers=optimizers,
                batch_offset=_get_batch_offset(),
                python_random_state=_get_rng_states()["python"],
                numpy_rng_state=_get_rng_states()["numpy"],
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

        # Skip ahead on resume (only in the first epoch after resume)
        if dataloader_offset > 0 and epoch == start_epoch:
            print(f"  Skipping {dataloader_offset} batches to resume position (one-time cost)...")
            import itertools
            dl_iter = iter(dataloader)
            skipped = 0
            for _ in itertools.islice(dl_iter, dataloader_offset):
                skipped += 1
                if skipped % 5000 == 0:
                    print(f"    ...{skipped}/{dataloader_offset}")
            pbar = tqdm(dl_iter, total=len(dataloader) - dataloader_offset, desc=f"Epoch {epoch + 1}")
            print(f"  ✓ Resumed at batch {dataloader_offset}")
            dataloader_offset = 0

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

                # Step 1: Single-pass AR prefill + fused Triton state extraction
                _profile = (global_step == _profile_step - 1)
                _t0 = time.perf_counter() if _profile else 0
                with torch.no_grad():
                    ar_kv_cache, ar_hidden, linear_states, per_block_la_conv = model.forward_ar_prefill(
                        ar_input_ids, ar_attention_mask, anchor_positions=anchor_positions,
                    )
                    per_block_fa_kv = {}
                if _profile: torch.cuda.synchronize(); _t_ar = (time.perf_counter() - _t0) * 1000

                # Precompute per-token valid mask
                _diff_pos, _batch_idx, _ = _get_kl_indices(B, B_blocks_total, K, target_ids.device)
                total_valid_mask = (
                    target_ids[_batch_idx, _diff_pos] != pad_token_id
                )
                total_valid_tokens = total_valid_mask.sum().float()

                # Compute decayed repr weight for this step
                _repr_cfg = config["training"].get("repr_align", False)
                if _repr_cfg:
                    _rw_start = config["training"].get("repr_align_weight", 0.3)
                    _rw_end = config["training"].get("repr_align_weight_end", 0.01)
                    _rw_decay = config["training"].get("repr_align_decay_steps", 500)
                    if _rw_start > 0:
                        _progress = min(1.0, global_step / max(1, _rw_decay))
                        cur_repr_weight = _rw_start + (_rw_end - _rw_start) * _progress
                    else:
                        cur_repr_weight = 0.0
                else:
                    cur_repr_weight = 0.0

                # Step 2: Diffusion micro-batching — each chunk uses its per-block states
                batch_loss = 0.0
                batch_repr = 0.0
                _t_fwd = _t_kl = _t_bwd = 0.0
                for blk_start in range(0, B_blocks_total, diff_chunk_blocks):
                    blk_end = min(blk_start + diff_chunk_blocks, B_blocks_total)
                    n_blocks = blk_end - blk_start
                    block_indices = torch.arange(blk_start, blk_end, device=device)
                    n_start = blk_start * (K - 1)
                    n_end = blk_end * (K - 1)
                    chunk_valid_tokens = total_valid_mask[:, n_start:n_end].sum().float()

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

                    if _profile: torch.cuda.synchronize(); _tf0 = time.perf_counter()
                    diff_hidden = model.forward_diffusion(
                        diff_input_ids=chunk_diff_ids,
                        ar_past_key_values=ar_kv_cache,
                        ar_seq_len=ar_seq_len,
                        causal_limit=chunk_causal,
                        return_hidden=True,
                        diff_position_ids=chunk_positions,
                        linear_states=linear_states,
                        block_indices=block_indices,
                        per_block_fa_kv=per_block_fa_kv,
                        per_block_la_conv=per_block_la_conv,
                    )
                    if _profile: torch.cuda.synchronize(); _t_fwd += (time.perf_counter() - _tf0) * 1000

                    if _profile: torch.cuda.synchronize(); _tk0 = time.perf_counter()
                    chunk_loss, chunk_repr = compute_kl_loss(
                        diff_hidden=diff_hidden,
                        ar_hidden_states=ar_hidden,
                        lm_head=model.lm_head,
                        anchor_positions=chunk_anchor,
                        K=K,
                        target_ids=chunk_target,
                        pad_token_id=pad_token_id,
                        repr_weight=cur_repr_weight,
                        repr_subsample=config["training"].get("repr_align_subsample", 0.25),
                    )
                    if _profile: torch.cuda.synchronize(); _t_kl += (time.perf_counter() - _tk0) * 1000

                    # Weight by actual valid token count - invariant to chunk size
                    if total_valid_tokens > 0 and chunk_valid_tokens > 0:
                        weight = chunk_valid_tokens / total_valid_tokens
                    else:
                        weight = 0.0
                    scaled_loss = (chunk_loss * weight) / config["training"]["gradient_accumulation_steps"]

                    # Backward immediately - frees diffusion activations for this chunk
                    if _profile: torch.cuda.synchronize(); _tb0 = time.perf_counter()
                    if use_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    if _profile: torch.cuda.synchronize(); _t_bwd += (time.perf_counter() - _tb0) * 1000
                    batch_loss += chunk_loss.item() * weight
                    batch_repr += chunk_repr * weight
                    # Free diffusion activations immediately to prevent memory leak
                    del diff_hidden, chunk_loss, scaled_loss

                accum_loss += batch_loss
                accum_repr += batch_repr

                # Free AR intermediates
                del ar_hidden, ar_kv_cache, linear_states, per_block_la_conv

                if (batch_idx + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    # Clip + step all optimizers
                    grad_norms = []
                    for opt in optimizers:
                        gn = torch.nn.utils.clip_grad_norm_(
                            [p for pg in opt.param_groups for p in pg['params']],
                            max_norm=config["training"]["gradient_clip"]
                        )
                        grad_norms.append(gn.item() if gn is not None else 0.0)
                    grad_norm = max(grad_norms)

                    if use_scaler:
                        for opt in optimizers:
                            scaler.step(opt)
                        scaler.update()
                    else:
                        for opt in optimizers:
                            opt.step()

                    scheduler.step()
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)

                    global_step += 1
                    schedule_step += 1

                    if global_step == _profile_step:
                        _t_opt = 0  # approximate: zero_grad + optimizer included
                        _total = _t_ar + _t_fwd + _t_kl + _t_bwd
                        print(f"\n  ═══ Step {global_step} breakdown ═══")
                        print(f"  AR prefill:       {_t_ar:7.1f} ms  ({_t_ar/_total*100:4.1f}%)")
                        print(f"  Diffusion fwd:    {_t_fwd:7.1f} ms  ({_t_fwd/_total*100:4.1f}%)")
                        print(f"  KL loss:          {_t_kl:7.1f} ms  ({_t_kl/_total*100:4.1f}%)")
                        print(f"  Backward:         {_t_bwd:7.1f} ms  ({_t_bwd/_total*100:4.1f}%)")
                        print(f"  ──────────────────────────")
                        print(f"  Total step:       {_total:7.0f} ms  ({_total/1000:.2f}s, {1000/_total:.1f} it/s)")

                    # ── Ctrl+C interrupt check ─────────────────────────────────
                    if _interrupted[0]:
                        print(f"\n⚠ Interrupted at step {global_step}. Saving checkpoint...")
                        save_checkpoint(
                            model, optimizer, scheduler, global_step, epoch,
                            config["training"]["output_dir"],
                            f"interrupt_step_{global_step}",
                            total_steps=total_steps,
                            optimizers=optimizers,
                            batch_offset=_get_batch_offset(),
                            python_random_state=_get_rng_states()["python"],
                            numpy_rng_state=_get_rng_states()["numpy"],
                        )
                        print("✓ Saved. Resume with:")
                        print(f"  --resume {config['training']['output_dir']}/interrupt_step_{global_step}")
                        return model

                    # ── logging ─────────────────────────────────────────────────
                    if global_step % config["training"]["log_every"] == 0:
                        lr = scheduler.get_last_lr()[0]
                        n_accum = config["training"]["gradient_accumulation_steps"]
                        avg_loss = accum_loss / n_accum
                        avg_repr = accum_repr / n_accum
                        repr_str = ""
                        if config["training"].get("repr_align", False):
                            repr_str = f" | Repr: {avg_repr:.4f} (w={cur_repr_weight:.3f})"
                        pbar.write(
                            f"  Step {global_step:6d} | Loss: {avg_loss:.4f}{repr_str} | "
                            f"LR: {lr:.2e} | Grad norm: {grad_norm:.2f}"
                        )
                        log_metrics(global_step, avg_loss, lr=lr, grad_norm=grad_norm)
                        accum_loss = 0.0
                        accum_repr = 0.0

                    # ── checkpoint ──────────────────────────────────────────────
                    if global_step % config["training"]["save_every"] == 0:
                        save_checkpoint(
                            model, optimizer, scheduler, global_step, epoch,
                            config["training"]["output_dir"],
                            f"step_{global_step}",
                            total_steps=total_steps,
                            optimizers=optimizers,
                            batch_offset=_get_batch_offset(),
                            python_random_state=_get_rng_states()["python"],
                            numpy_rng_state=_get_rng_states()["numpy"],
                        )

                    # ── eval ───────────────────────────────────────────────────
                    if global_step % config["training"]["eval_every"] == 0:
                        eval_dl = val_dataloader if val_dataloader is not None else dataloader
                        torch.cuda.empty_cache()
                        eval_loss = evaluate(
                            model, eval_dl, pad_token_id, device, dtype,
                            max_eval_batches=10,
                            diff_chunk_blocks=config["training"].get("diffusion_chunk_blocks", 16),
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
                            mask_id=mask_id,
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
                            f"blocks: {acc_stats['num_blocks']} | "
                            f"gen_toks: {acc_stats.get('total_tokens_gen', 0)}{off_str} <<<"
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
        total_steps=total_steps,
        optimizers=optimizers,
        batch_offset=_get_batch_offset(),
        python_random_state=_get_rng_states()["python"],
        numpy_rng_state=_get_rng_states()["numpy"],
    )
    print(f"\n✓ Training complete. Final checkpoint saved to "
          f"{os.path.join(config['training']['output_dir'], 'final')}")
    return model


def save_checkpoint(model, optimizer, scheduler, step, epoch, output_dir, name, total_steps=None, optimizers=None, batch_offset=0, python_random_state=None, numpy_rng_state=None):
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
    # Save optimizer(s)
    if optimizers is not None and len(optimizers) > 1:
        opt_states = {i: opt.state_dict() for i, opt in enumerate(optimizers)}
    else:
        opt_states = optimizer.state_dict()
    # Save RNG states for DataLoader reproducibility on resume
    rng_state = {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python": python_random_state,
        "numpy": numpy_rng_state,
    }
    torch.save({
        "optimizer": opt_states,
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "rng_state": rng_state,
        "batch_offset": batch_offset,
    }, os.path.join(ckpt_dir, "trainer_state.pt"))
    # Save a sidecar with schedule metadata so batch-size changes can be handled
    if total_steps is not None:
        with open(os.path.join(ckpt_dir, "schedule_meta.json"), "w") as f:
            json.dump({"total_steps": total_steps}, f)
    print(f"\n  ✓ Checkpoint saved to {ckpt_dir}")


def load_checkpoint(model, optimizer, scheduler, ckpt_dir, optimizers=None):
    """Load full training state and return (start_epoch, global_step)."""
    state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), map_location="cpu", weights_only=False)
    weights = torch.load(os.path.join(ckpt_dir, "diffusion_heads.pt"), map_location="cpu", weights_only=True)
    clean_weights = {k.replace("._orig_mod", ""): v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(clean_weights, strict=False)
    diff_missing = [k for k in missing if "diffusion_heads" in k]
    if diff_missing:
        print(f"  ⚠ {len(diff_missing)} diffusion_head keys missing (reverting to copy-init)")
    print(f"  ✓ Resumed from {ckpt_dir} (step {state['step']}, epoch {state['epoch']}) "
          f"[{len(clean_weights)} weights loaded]")
    # Load optimizer(s)
    opt_state = state["optimizer"]
    if isinstance(opt_state, dict) and any(str(k).isdigit() for k in opt_state):
        # Multiple optimizers saved as {0: state, 1: state}
        if optimizers is not None:
            for i, opt in enumerate(optimizers):
                # keys may be int or str depending on save version
                opt.load_state_dict(opt_state.get(i) or opt_state.get(str(i)))
    else:
        optimizer.load_state_dict(opt_state)
    # scheduler rebuilt from config - don't load stale LR
    rng_state = state.get("rng_state")
    batch_offset = state.get("batch_offset", 0)
    return state["epoch"], state["step"], rng_state, batch_offset



@torch.no_grad()
def evaluate(model, dataloader, pad_token_id, device, dtype, max_eval_batches=10, diff_chunk_blocks=16):
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

        # Single-pass AR prefill + fused Triton state extraction
        ar_kv_cache, ar_hidden, linear_states, per_block_la_conv = model.forward_ar_prefill(
            ar_input_ids, ar_attention_mask, anchor_positions=anchor_positions,
        )

        K = model.block_size
        diff_chunk_blocks_inner = diff_chunk_blocks
        B_blocks_total = anchor_positions.shape[1]
        B = ar_input_ids.shape[0]

        chunk_losses = []
        for blk_start in range(0, B_blocks_total, diff_chunk_blocks_inner):
            blk_end = min(blk_start + diff_chunk_blocks_inner, B_blocks_total)
            n_blocks = blk_end - blk_start
            tok_start, tok_end = blk_start * K, blk_end * K

            chunk_positions = (anchor_positions[:, blk_start:blk_end].unsqueeze(-1) + torch.arange(K, device=device)).view(B, -1)
            diff_hidden = model.forward_diffusion(
                diff_input_ids=diff_input_ids[:, tok_start:tok_end],
                ar_past_key_values=ar_kv_cache,
                ar_seq_len=ar_seq_len,
                causal_limit=causal_limit[:, tok_start:tok_end],
                return_hidden=True,
                diff_position_ids=chunk_positions,
                linear_states=linear_states,
                block_indices=torch.arange(blk_start, blk_end, device=device),
                per_block_la_conv=per_block_la_conv,
            )
            chunk_loss, _ = compute_kl_loss(
                diff_hidden, ar_hidden, model.lm_head,
                anchor_positions[:, blk_start:blk_end], K,
                target_ids[:, tok_start:tok_end], pad_token_id,
                forward_only=True,
            )
            chunk_losses.append(chunk_loss.item() * n_blocks)
            del diff_hidden

        total_loss += sum(chunk_losses) / B_blocks_total
        total_tokens += 1
        del ar_kv_cache, ar_hidden, linear_states, per_block_la_conv
        torch.cuda.empty_cache()
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_acceptance_rate(
    model, tokenizer, val_dataloader, device, dtype,
    max_examples=4, max_tokens_per_example=128,
    mask_id=None,
):
    """
    Run consensus generation on held-out examples and measure acceptance rate.

    This is the key metric from the Orthrus paper - higher acceptance → higher TPF.
    Targets: >50% early, >85% at convergence.
    """
    model.eval()
    K = model.block_size
    
    # Get mask_id from tokenizer if not provided
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids("<mask>")
        if mask_id is None:
            mask_id = tokenizer.convert_tokens_to_ids("<tts_pad>")

    total_acceptances = []
    offset_accepted = [0] * (K + 1)  # per-offset accept counts (positions 1..K)
    offset_tested = [0] * (K + 1)    # per-offset test counts
    total_tokens_gen = 0
    total_passes = 0
    examples_processed = 0

    # Assistant prompt tokens to find the start of generation
    assistant_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    # ── Wrap Gated Delta Net to extract intermediate states ────────────
    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35m
    from inference_kernel import fused_recurrent_inference_fwd
    import torch.nn.functional as F

    linear_layer_indices = [
        i for i, lt in enumerate(model.config.layer_types)
        if lt == "linear_attention"
    ]
    
    saved_forwards = {}
    for i in linear_layer_indices:
        layer = model.base_model.model.layers[i]
        gdn = layer.linear_attn
        if 'forward' in gdn.__dict__:
            saved_forwards[i] = gdn.__dict__['forward']
        else:
            saved_forwards[i] = None

        def make_inference_wrapper(li, gdn_ref, ks_ref):
            def wrapper(hidden_states, cache_params=None, attention_mask=None):
                hidden_states = _q35m.apply_mask_to_padding_states(hidden_states, attention_mask)
                batch_size, slen, _ = hidden_states.shape
                use_precomputed_states = cache_params is not None and cache_params.has_previous_state(li)
                if use_precomputed_states:
                    conv_state = cache_params.layers[li].conv_states
                    recurrent_state = cache_params.layers[li].recurrent_states

                mixed_qkv = gdn_ref.in_proj_qkv(hidden_states)
                mixed_qkv = mixed_qkv.transpose(1, 2)

                z = gdn_ref.in_proj_z(hidden_states)
                z = z.reshape(batch_size, slen, -1, gdn_ref.head_v_dim)
                b = gdn_ref.in_proj_b(hidden_states)
                a = gdn_ref.in_proj_a(hidden_states)

                if use_precomputed_states and slen == 1:
                    mixed_qkv, new_conv_state = gdn_ref.causal_conv1d_update(
                        x=mixed_qkv, conv_state=conv_state,
                        weight=gdn_ref.conv1d.weight.squeeze(1),
                        bias=gdn_ref.conv1d.bias, activation=gdn_ref.activation,
                    )
                    cache_params.update_conv_state(new_conv_state, li)
                else:
                    if use_precomputed_states:
                        mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)

                    if cache_params is not None:
                        cache_params.layers[li].pre_conv_mixed_qkv = mixed_qkv.clone()
                        new_conv_state = F.pad(mixed_qkv, (ks_ref - mixed_qkv.shape[-1], 0))
                        cache_params.update_conv_state(new_conv_state, li)
                    
                    if gdn_ref.causal_conv1d_fn is not None:
                        mixed_qkv = gdn_ref.causal_conv1d_fn(
                            x=mixed_qkv, weight=gdn_ref.conv1d.weight.squeeze(1),
                            bias=gdn_ref.conv1d.bias, activation=gdn_ref.activation, seq_idx=None,
                        )
                    else:
                        mixed_qkv = F.silu(gdn_ref.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
                    if use_precomputed_states:
                        mixed_qkv = mixed_qkv[:, :, -slen:]

                mixed_qkv = mixed_qkv.transpose(1, 2)
                qkv_splits = [gdn_ref.key_dim, gdn_ref.key_dim, gdn_ref.value_dim]
                query, key, value = torch.split(mixed_qkv, qkv_splits, dim=-1)
                query = query.reshape(batch_size, slen, -1, gdn_ref.head_k_dim)
                key = key.reshape(batch_size, slen, -1, gdn_ref.head_k_dim)
                value = value.reshape(batch_size, slen, -1, gdn_ref.head_v_dim)

                beta = b.sigmoid()
                g = -gdn_ref.A_log.float().exp() * F.softplus(a.float() + gdn_ref.dt_bias)
                if gdn_ref.num_v_heads // gdn_ref.num_k_heads > 1:
                    query = query.repeat_interleave(gdn_ref.num_v_heads // gdn_ref.num_k_heads, dim=2)
                    key = key.repeat_interleave(gdn_ref.num_v_heads // gdn_ref.num_k_heads, dim=2)

                if slen > 128:
                    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
                    o, last_recurrent_state = chunk_gated_delta_rule(
                        query, key, value, g=g, beta=beta,
                        initial_state=recurrent_state if use_precomputed_states else None,
                        output_final_state=cache_params is not None,
                        use_qk_l2norm_in_kernel=True,
                    )
                    h_out = None
                else:
                    o, last_recurrent_state, h_out = fused_recurrent_inference_fwd(
                        query, key, value, g=g, beta=beta,
                        initial_state=recurrent_state if use_precomputed_states else None,
                        output_final_state=cache_params is not None,
                        use_qk_l2norm_in_kernel=True,
                    )
                
                if cache_params is not None:
                    if h_out is not None:
                        cache_params.layers[li].h_out_all = h_out
                    cache_params.update_recurrent_state(last_recurrent_state, li)

                core_attn_out = o.reshape(-1, gdn_ref.head_v_dim)
                z = z.reshape(-1, gdn_ref.head_v_dim)
                core_attn_out = gdn_ref.norm(core_attn_out, z)
                core_attn_out = core_attn_out.reshape(batch_size, slen, -1)
                return gdn_ref.out_proj(core_attn_out)
            return wrapper
            
        gdn.forward = make_inference_wrapper(i, gdn, gdn.conv_kernel_size)

    def crop_cache(cache, seq_len):
        for layer in cache.layers:
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = torch.tensor(seq_len, device=device)

    try:
        from transformers.cache_utils import StaticCache

        for batch in val_dataloader:
            if examples_processed >= max_examples:
                break
            input_ids = batch["ar_input_ids"].to(device)
            attention_mask = batch["ar_attention_mask"].to(device)
            B = input_ids.shape[0]

            for i in range(B):
                if examples_processed >= max_examples:
                    break
                seq_len = int(attention_mask[i].sum().item())
                if seq_len < 64:
                    continue

                seq_list = input_ids[i, :seq_len].tolist()
                found_idx = -1
                n_assist = len(assistant_tokens)
                for j in range(len(seq_list) - n_assist + 1):
                    if seq_list[j:j+n_assist] == assistant_tokens:
                        found_idx = j + n_assist
                        break
                
                if found_idx != -1:
                    prompt_len = found_idx
                else:
                    # Fallback to default behavior if no chat template is found
                    prompt_len = min(256, seq_len)

                prompt_ids = input_ids[i, :prompt_len]
                current_len = prompt_len
                max_cache_len = prompt_len + max_tokens_per_example + 10

                past_kv = StaticCache(
                    config=model.config,
                    max_batch_size=1,
                    max_cache_len=max_cache_len,
                    device=device,
                    dtype=dtype
                )

                # AR prefill
                position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
                base_outputs = model.base_model(
                    input_ids=prompt_ids.unsqueeze(0),
                    position_ids=position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    logits_to_keep=1
                )
                past_kv = base_outputs.past_key_values
                first_logits = base_outputs.logits[:, -1, :]
                first_token = first_logits.argmax(dim=-1).item()
                generated = [first_token]
                total_passes += 1
                total_tokens_gen += 1

                while True:
                    if len(generated) >= max_tokens_per_example:
                        break

                    diff_len = min(K, max_tokens_per_example - len(generated))
                    if diff_len <= 1:
                        break

                    anchor_token = generated[-1]
                    diff_block = torch.full((1, diff_len), mask_id, dtype=torch.long, device=device)
                    diff_block[:, 0] = anchor_token

                    causal_limit = torch.zeros(1, diff_len, dtype=torch.long, device=device)
                    for k in range(diff_len):
                        causal_limit[0, k] = current_len - 1

                    # Diffusion projection
                    diff_logits = model(
                        input_ids=diff_block,
                        is_diffusion_pass=True,
                        ar_past_key_values=past_kv,
                        ar_seq_len=current_len,
                        causal_limit=causal_limit,
                        use_flex=False,
                    )
                    total_passes += 1

                    if diff_len > 1:
                        diff_preds = diff_logits[0, 1:].argmax(dim=-1).tolist()
                    else:
                        diff_preds = []
                    proposed = [anchor_token] + diff_preds

                    # AR verification
                    proposed_tensor = torch.tensor([proposed], dtype=torch.long, device=device)
                    ar_pos_ids = torch.arange(current_len, current_len + len(proposed), device=device).unsqueeze(0)
                    
                    ar_outputs = model.base_model(
                        proposed_tensor, position_ids=ar_pos_ids,
                        past_key_values=past_kv, use_cache=True,
                    )
                    total_passes += 1
                    ar_logits = ar_outputs.logits[0]
                    past_kv = ar_outputs.past_key_values

                    # Greedy consensus
                    accepted = [proposed[0]]
                    diffusion_match_count = 0
                    for k in range(1, len(proposed)):
                        ar_pred = ar_logits[k - 1].argmax().item()
                        offset_tested[k] += 1
                        if proposed[k] == ar_pred:
                            accepted.append(proposed[k])
                            offset_accepted[k] += 1
                            diffusion_match_count += 1
                        else:
                            accepted.append(ar_pred)
                            break

                    total_acceptances.append(diffusion_match_count)

                    # State Slicing Rollback
                    accepted_len = len(accepted) - 1  # tokens after anchor (includes AR correction if any)
                    end_idx = current_len + accepted_len
                    crop_cache(past_kv, end_idx)

                    for li in linear_layer_indices:
                        lc = past_kv.layers[li]
                        h_out_all = lc.h_out_all
                        lc.recurrent_states = h_out_all[:, accepted_len - 1].clone()
                        
                        prev_conv_len = lc.conv_states.shape[-1]
                        end_conv = prev_conv_len + accepted_len
                        start_conv = end_conv - prev_conv_len
                        lc.conv_states = lc.pre_conv_mixed_qkv[:, :, start_conv : end_conv].clone()

                    generated.extend(accepted[1:])
                    current_len += accepted_len
                    total_tokens_gen += accepted_len

                examples_processed += 1
                
                # Cleanup manually to prevent leak across examples
                del past_kv

    finally:
        # Restore original forward methods
        for i in linear_layer_indices:
            gdn = model.base_model.model.layers[i].linear_attn
            if saved_forwards[i] is None:
                if 'forward' in gdn.__dict__:
                    del gdn.forward
            else:
                gdn.forward = saved_forwards[i]

    torch.cuda.empty_cache()
    model.train()
    if not total_acceptances:
        return {"acceptance_rate": 0.0, "avg_acceptance_len": 0.0, "offset_rates": []}

    true_avg_len = sum(total_acceptances) / len(total_acceptances)
    accept_rate = max(0, true_avg_len) / (K - 1)
    
    successful_matches = [m for m in total_acceptances if m > 0]
    avg_accept_len_non_zero = sum(successful_matches) / len(successful_matches) if successful_matches else 0.0
    
    tpf = total_tokens_gen / max(total_passes, 1)
    return {
        "acceptance_rate": accept_rate,
        "avg_acceptance_len": avg_accept_len_non_zero,
        "num_blocks": len(total_acceptances),
        "tpf": tpf,
        "total_tokens_gen": total_tokens_gen,
        "offset_rates": [
            (offset_accepted[k] / len(total_acceptances)) if len(total_acceptances) > 0 else 0.0
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
    parser.add_argument("--optimizer", type=str, default=None)  # "adamw" or "muon"
    parser.add_argument("--muon_lr", type=float, default=None)  # muon_lr_multiplier
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
        "optimizer": ("training", "optimizer"),
        "muon_lr": ("training", "muon_lr_multiplier"),
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
