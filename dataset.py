"""
Dataset loading for Orthrus training.

Supports HuggingFace datasets (smoltalk, nemotron, etc.) and local JSONL.
"""

from __future__ import annotations

import os
from typing import Optional, Callable
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def load_orthrus_dataset(
    dataset_name: str = "HuggingFaceTB/smoltalk",
    config_name: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    text_key: str = "text",
    filter_fn: Optional[Callable] = None,
    seed: int = 42,
    streaming: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    **load_kwargs,
) -> Dataset:
    """
    Load a text dataset suitable for Orthrus training.

    Args:
      dataset_name:  HuggingFace dataset ID or local path
      config_name:   sub-config name (e.g. 'all' for smoltalk)
      split:         dataset split name
      max_samples:   cap to this many examples (None = all)
      text_key:      field name containing text (auto-detected for smoltalk)
      filter_fn:     optional filter callable(dict) -> bool
      seed:          random seed for shuffling
      streaming:     use streaming mode (for huge datasets)

    Returns:
      HuggingFace Dataset
    """
    print(f"Loading dataset: {dataset_name}" +
          (f" ({config_name})" if config_name else "") +
          f" [{split}]")

    load_kwargs_main = {**load_kwargs}
    if config_name:
        load_kwargs_main["name"] = config_name

    if streaming:
        ds = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs_main)
    else:
        ds = load_dataset(dataset_name, split=split, **load_kwargs_main)

    # Auto-detect text field if needed
    if text_key and text_key not in (ds.column_names or []):
        # Try common keys
        for candidate in ["text", "content", "messages", "conversations", "prompt"]:
            if candidate in (ds.column_names or []):
                text_key = candidate
                break

    # If the dataset has 'messages' field (chat format), format into text
    if text_key == "messages":
        if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
            # Use the tokenizer's official chat template for exact match with inference
            def _format_with_tokenizer(example: dict) -> dict:
                messages = example.get("messages", [])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                return {"text": text}
            ds = ds.map(_format_with_tokenizer, remove_columns=ds.column_names)
        else:
            ds = ds.map(_format_chat, remove_columns=ds.column_names)
        text_key = "text"

    # Apply filter
    if filter_fn is not None:
        ds = ds.filter(filter_fn)

    # Shuffle and cap
    if not streaming:
        ds = ds.shuffle(seed=seed)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

    print(f"  → {len(ds) if not streaming else 'streaming'} examples, text_key='{text_key}'")
    return ds, text_key


def _format_chat(example: dict) -> dict:
    """Format a chat-messages example into a single text string."""
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}


def load_multi_dataset(
    datasets_config: list[dict],
    split: str = "train",
    max_total: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load and interleave multiple datasets (e.g., chat + code + math).

    Args:
      datasets_config: list of dicts with keys: name, split?, weight?, max_samples?
      split:           default split for all datasets
      max_total:       cap total examples across all datasets
      seed:            random seed
    """
    from datasets import concatenate_datasets, interleave_datasets

    loaded = []
    for cfg in datasets_config:
        ds_name = cfg["name"]
        ds_split = cfg.get("split", split)
        ds_weight = cfg.get("weight", 1.0)
        ds_max = cfg.get("max_samples")

        ds, text_key = load_orthrus_dataset(
            ds_name, split=ds_split, max_samples=ds_max, seed=seed + hash(ds_name) % 10000,
        )
        # Ensure 'text' column
        if text_key != "text" and text_key in ds.column_names:
            ds = ds.rename_column(text_key, "text")

        if ds_weight != 1.0:
            ds = _weight_dataset(ds, ds_weight)

        loaded.append(ds)

    if len(loaded) == 1:
        combined = loaded[0]
    else:
        # Interleave proportionally
        combined = interleave_datasets(
            loaded,
            probabilities=[cfg.get("weight", 1.0) for cfg in datasets_config],
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    if max_total is not None and len(combined) > max_total:
        combined = combined.shuffle(seed=seed).select(range(max_total))

    print(f"Combined dataset: {len(combined)} examples")
    return combined


def _weight_dataset(ds: Dataset, weight: float) -> Dataset:
    """Repeat a dataset to achieve a target weight (approximate)."""
    # Simple approach: just return ds; interleave_datasets handles weighting
    return ds


def pretokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_key: str = "text",
    num_proc: int = 4,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Pre-tokenize a text dataset so the collator operates on token IDs only.

    Tokenizes every example once (batched) and stores input_ids + attention_mask
    (unpadded, truncated to max_length). The collator handles per-batch padding.

    If cache_dir is provided, saves to / loads from disk so future runs skip
    the CPU-bound tokenization step entirely.

    Args:
        ds:          HuggingFace Dataset with a text column
        tokenizer:   HF tokenizer instance
        max_length:  max tokens per example (truncation)
        text_key:    column name containing text
        num_proc:    number of parallel processes for map
        cache_dir:   directory to cache pretokenized result (None = no caching)
    """
    if cache_dir is not None:
        ds_key = ds._fingerprint if ds._fingerprint else str(id(ds))
        cache_name = f"{ds_key}.len{max_length}.{text_key}"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.isdir(cache_path):
            print(f"Loading pretokenized dataset from cache: {cache_path}")
            return Dataset.load_from_disk(cache_path)

    print(f"Pretokenizing dataset ({len(ds)} examples, max_length={max_length})...")

    def tokenize_fn(examples: dict) -> dict:
        encoded = tokenizer(
            examples[text_key],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    ds = ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in ds.column_names if c != text_key],
        desc="Tokenizing",
    )
    # Keep text_key for reference (e.g. multi-dataset renaming); rename token cols as canonical
    print(f"  Done — columns: {ds.column_names}")

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        ds.save_to_disk(cache_path)
        print(f"  Saved to cache: {cache_path}")

    return ds
