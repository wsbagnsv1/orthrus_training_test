"""
Data collator for Orthrus training.

Samples random anchor positions from each sequence, constructs
[mask] blocks, and extracts teacher targets from the AR prefill logits.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class OrthrusCollator:
    """
    Collates raw text examples into Orthrus training batches.

    For each sequence:
      1. Tokenize to max_seq_len
      2. Sample B_blocks anchor positions (must leave room for K tokens after)
      3. Build diff_input_ids: [anchor_token, <mask>*(K-1)] repeated B_blocks times
      4. Record anchor_positions (indices in the padded sequence)
      5. Extract target_ids (true tokens at positions a..a+K-1 for each block)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        K: int = 32,
        B_blocks: int = 8,
        max_seq_len: int = 2048,
        mask_token: str = "<mask>",
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.K = K
        self.B_blocks = B_blocks
        self.max_seq_len = max_seq_len
        self.text_key = text_key

        # Register <mask> token if not present
        if mask_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        self.mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, examples: list[dict]) -> Dict[str, Tensor]:
        # --- Tokenize (or use pre-tokenized) ---
        if "input_ids" in examples[0]:
            # Pre-tokenized: pad within batch (no tokenizer call)
            max_len = min(
                max(len(ex["input_ids"]) for ex in examples),
                self.max_seq_len,
            )
            B = len(examples)
            input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(B, max_len, dtype=torch.long)
            for i, ex in enumerate(examples):
                ids = ex["input_ids"][:max_len]
                n = len(ids)
                input_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long)
                attention_mask[i, :n] = 1
        else:
            texts = [ex[self.text_key] for ex in examples]
            encoded = self.tokenizer(
                texts,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        B, L = input_ids.shape

        # --- Sample anchors & build blocks ---
        all_diff_input_ids: list[list[int]] = []
        all_anchor_positions: list[list[int]] = []
        all_target_ids: list[list[int]] = []
        all_causal_limits: list[list[int]] = []

        for i in range(B):
            seq_len = attention_mask[i].sum().item()
            valid_len = int(seq_len)

            # Valid anchor positions: need room for K tokens after anchor
            valid_anchors = list(range(1, valid_len - self.K + 1))

            if not valid_anchors:
                # Truly too short: can't fit even one K-token block
                diff_block = [self.mask_id] * (self.K * self.B_blocks)
                anchors = [0] * self.B_blocks
                targets = [self.pad_token_id] * (self.K * self.B_blocks)
                causal_limits = [0] * (self.K * self.B_blocks)
            else:
                # Sample anchors (with replacement if fewer positions than B_blocks)
                if len(valid_anchors) < self.B_blocks:
                    anchors = sorted(random.choices(valid_anchors, k=self.B_blocks))
                else:
                    anchors = sorted(random.sample(valid_anchors, k=self.B_blocks))

                diff_block = []
                targets = []
                causal_limits = []

                for a in anchors:
                    block = [input_ids[i, a].item()] + [self.mask_id] * (self.K - 1)
                    diff_block.extend(block)

                    # Targets: a, a+1, ..., a+K-1
                    t = input_ids[i, a : a + self.K].tolist()
                    if len(t) < self.K:
                        t += [self.pad_token_id] * (self.K - len(t))
                    targets.extend(t)

                    # Causal limit: all positions in block attend to AR context <= anchor-1
                    # (prevents peeking at future tokens beyond the anchor)
                    for _ in range(self.K):
                        causal_limits.append(a - 1)  # inclusive index in AR

            all_diff_input_ids.append(diff_block)
            all_anchor_positions.append(anchors)
            all_target_ids.append(targets)
            all_causal_limits.append(causal_limits)

        return {
            "ar_input_ids": input_ids,
            "ar_attention_mask": attention_mask,
            "diff_input_ids": torch.tensor(all_diff_input_ids, dtype=torch.long),
            "anchor_positions": torch.tensor(all_anchor_positions, dtype=torch.long),
            "target_ids": torch.tensor(all_target_ids, dtype=torch.long),
            "causal_limit": torch.tensor(all_causal_limits, dtype=torch.long),
        }
