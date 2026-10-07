"""Bits per byte (bpb): a tokenizer-independent language-modelling metric.

Loss per token cannot be compared across tokenizers (a Khmer-extended
tokenizer has fewer, longer tokens), but loss per **byte of text** can::

    bpb = sum(token NLL in nats) / (ln 2 * sum(UTF-8 bytes of the target tokens))

For a byte-level BPE vocabulary every character of a token string stands for
exactly one byte, so byte lengths come straight from the vocabulary. Special
tokens (``<|im_end|>`` ...) count as zero bytes. Non-special added tokens
(``<tool_call>``) count their UTF-8 length.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def token_byte_lengths(tokenizer_json: str | Path, vocab_size: int | None = None) -> np.ndarray:
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    vocab: dict[str, int] = data["model"]["vocab"]
    size = max(
        vocab_size or 0,
        max(vocab.values()) + 1,
        max((a["id"] for a in data.get("added_tokens", [])), default=0) + 1,
    )
    lengths = np.zeros(size, dtype=np.int64)
    for token, index in vocab.items():
        lengths[index] = len(token)
    for added in data.get("added_tokens", []):
        lengths[added["id"]] = 0 if added.get("special") else len(added["content"].encode("utf-8"))
    return lengths


@torch.no_grad()
def nll_and_bytes(
    model, batch: dict[str, torch.Tensor], byte_lengths: torch.Tensor
) -> tuple[float, int, int]:
    """(sum of NLL in nats, target bytes, target tokens) for one batch (labels already masked)."""
    labels = batch["labels"]
    targets = labels[:, 1:]
    keep = targets != -100
    out = model(**batch, num_items_in_batch=1)
    tokens = int(keep.sum())
    bytes_ = int(byte_lengths.to(targets.device)[targets.clamp(min=0)][keep].sum())
    return float(out.loss), bytes_, tokens


def bits_per_byte(nll_nats: float, n_bytes: int) -> float:
    return nll_nats / (math.log(2) * n_bytes) if n_bytes else float("nan")
