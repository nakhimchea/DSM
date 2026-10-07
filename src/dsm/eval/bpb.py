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


def _windows_batches(prefix: str, seq_len: int, max_tokens: int, batch_size: int):
    from dsm.train.config import DataSource
    from dsm.train.datasets import eval_windows

    rows = eval_windows(DataSource(path=str(prefix)), seq_len, max_tokens)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        yield {
            "input_ids": torch.from_numpy(np.stack([r[0] for r in chunk])),
            "labels": torch.from_numpy(np.stack([r[1] for r in chunk])),
            "position_ids": torch.from_numpy(np.stack([r[2] for r in chunk])),
        }


def evaluate_bpb(
    model,
    tokenizer_json: str | Path,
    prefixes: dict[str, str],
    *,
    seq_len: int = 2048,
    max_tokens: int = 1_000_000,
    batch_size: int = 4,
) -> dict[str, float]:
    """``{"bpb/<name>": ..., "loss/<name>": ...}`` for packed evaluation sets."""
    device = next(model.parameters()).device
    lengths = torch.from_numpy(token_byte_lengths(tokenizer_json, model.config.vocab_size))
    out = {}
    model.eval()
    for name, prefix in prefixes.items():
        nll = n_bytes = n_tokens = 0
        for batch in _windows_batches(prefix, seq_len, max_tokens, batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            a, b, c = nll_and_bytes(model, {**batch, "use_cache": False}, lengths)
            nll, n_bytes, n_tokens = nll + a, n_bytes + b, n_tokens + c
        out[f"bpb/{name}"] = bits_per_byte(nll, n_bytes)
        out[f"loss/{name}"] = nll / max(n_tokens, 1)
    return out


@torch.no_grad()
def mean_token_kl(
    reference,
    candidate,
    prefixes: dict[str, str],
    *,
    seq_len: int = 1024,
    max_tokens: int = 200_000,
    batch_size: int = 2,
) -> dict[str, float]:
    """Mean KL(reference || candidate) per predicted token, per evaluation set.

    Use it to measure drift of a merged / quantised-in-torch model against its
    source; for GGUF files use ``llama-perplexity --kl-divergence`` (docs/11).
    """
    device = next(reference.parameters()).device
    out = {}
    for name, prefix in prefixes.items():
        total, count = 0.0, 0
        for batch in _windows_batches(prefix, seq_len, max_tokens, batch_size):
            ids, positions = batch["input_ids"].to(device), batch["position_ids"].to(device)
            keep = (batch["labels"][:, 1:] != -100).to(device)
            logp = torch.log_softmax(
                reference(input_ids=ids, position_ids=positions, use_cache=False).logits[:, :-1].float(), -1
            )
            logq = torch.log_softmax(
                candidate(input_ids=ids, position_ids=positions, use_cache=False).logits[:, :-1].float(), -1
            )
            kl = (logp.exp() * (logp - logq)).sum(-1)
            total += float(kl[keep].sum())
            count += int(keep.sum())
        out[f"quant/kl/{name}"] = total / max(count, 1)
    return out
