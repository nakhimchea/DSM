"""Training data: pre-training windows from packed datasets, and SFT conversations.

Both are deterministic functions of (seed, sample index), so a resumed run sees
exactly the batches an uninterrupted run would have seen.

Batches are dicts of int64 tensors ``input_ids``, ``labels`` (``-100`` = no loss),
``position_ids`` (restart at 0 for every packed document) and
``attention_mask`` (1 for real tokens, 0 for right padding).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dsm.data.packed import PackedTokens
from dsm.train.config import DataSource


def mask_document_starts(labels: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
    """No loss on predicting a document's first token from the previous document."""
    out = labels.copy()
    out[position_ids == 0] = -100
    return out


# -- pre-training / CPT / annealing ------------------------------------------------------------


class PretrainMixture:
    """Fixed-length windows drawn from several packed datasets with mixture weights.

    Sample ``g`` comes from source ``choices[g]``; each source walks through its
    windows in a per-epoch random permutation. When a small source (e.g. Khmer)
    is exhausted it starts another epoch, which ``epochs_used`` reports.
    """

    def __init__(self, sources: list[DataSource], seq_len: int, total_samples: int, seed: int) -> None:
        self.datasets = [PackedTokens(s.path) for s in sources]
        self.names = [s.name or d.name for s, d in zip(sources, self.datasets, strict=True)]
        self.seq_len, self.seed = seq_len, seed
        self.n_windows = np.array([d.n_windows(seq_len) for d in self.datasets], dtype=np.int64)
        if (self.n_windows == 0).any():
            empty = [n for n, w in zip(self.names, self.n_windows, strict=True) if w == 0]
            raise ValueError(f"datasets shorter than one window of {seq_len} tokens: {empty}")
        weights = np.array([s.weight for s in sources], dtype=np.float64)
        self.weights = weights / weights.sum()
        rng = np.random.default_rng([seed, 0x5EED])
        self.choices = rng.choice(len(sources), size=total_samples, p=self.weights).astype(np.int32)
        self.ranks = np.zeros(total_samples, dtype=np.int64)
        for source in range(len(sources)):
            where = np.nonzero(self.choices == source)[0]
            self.ranks[where] = np.arange(len(where))
        self._perms: dict[tuple[int, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.choices)

    def _window(self, source: int, rank: int) -> int:
        epoch, pos = divmod(int(rank), int(self.n_windows[source]))
        key = (source, epoch)
        if key not in self._perms:
            if len(self._perms) > 64:
                self._perms.clear()
            self._perms[key] = np.random.default_rng([self.seed, source, epoch]).permutation(
                self.n_windows[source]
            )
        return int(self._perms[key][pos])

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = int(self.choices[index])
        ids, positions = self.datasets[source].window(self._window(source, self.ranks[index]), self.seq_len)
        return ids, mask_document_starts(ids, positions), positions

    def batch(self, indices: list[int]) -> dict[str, torch.Tensor]:
        rows = [self.sample(i) for i in indices]
        return {
            "input_ids": torch.from_numpy(np.stack([r[0] for r in rows])),
            "labels": torch.from_numpy(np.stack([r[1] for r in rows])),
            "position_ids": torch.from_numpy(np.stack([r[2] for r in rows])),
        }

    def epochs_used(self, samples_seen: int) -> dict[str, float]:
        counts = np.bincount(self.choices[:samples_seen], minlength=len(self.datasets))
        return {n: float(c / w) for n, c, w in zip(self.names, counts, self.n_windows, strict=True)}


def eval_windows(
    source: DataSource, seq_len: int, max_tokens: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The first ``max_tokens`` of an evaluation set as windows (deterministic)."""
    data = PackedTokens(source.path)
    count = min(data.n_windows(seq_len), max(1, max_tokens // seq_len))
    out = []
    for index in range(count):
        ids, positions = data.window(index, seq_len)
        out.append((ids, mask_document_starts(ids, positions), positions))
    return out


# -- SFT ---------------------------------------------------------------------------------------


@dataclass
class Example:
    input_ids: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.input_ids)


def _file_digest(paths: list[str], extra: str) -> str:
    digest = hashlib.sha256(extra.encode())
    for path in paths:
        stat = Path(path).stat()
        digest.update(f"{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def tokenize_sft_file(
    path: str,
    tokenizer,
    template: str,
    *,
    seq_len: int,
    train_on: str = "all",
    cache_dir: str | Path | None = None,
    tokenizer_key: str = "",
) -> tuple[list[Example], dict[str, int]]:
    """Tokenise one SFT JSONL file (cached as ``.npz``); drops examples longer than ``seq_len``.

    ``tokenizer_key`` (e.g. the hash of ``tokenizer.json``) is part of the cache key.
    """
    from dsm.data.sft.schema import template_messages
    from dsm.tokenizer.render import encode_for_training
    from dsm.utils import read_jsonl

    template_hash = hashlib.sha256(template.encode()).hexdigest()
    key = _file_digest([path], f"{template_hash}|{tokenizer_key}|{seq_len}|{train_on}|v1")
    cache = Path(cache_dir) / f"sft-{key}.npz" if cache_dir else None
    if cache and cache.is_file():
        data = np.load(cache)
        bounds = data["bounds"]
        stats = json.loads(str(data["stats"]))
        return [
            Example(data["ids"][a:b].astype(np.int64), data["labels"][a:b].astype(np.int64))
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ], stats
    examples, stats = [], {"records": 0, "too_long": 0, "no_target": 0}
    for record in read_jsonl(path):
        stats["records"] += 1
        encoded = encode_for_training(
            tokenizer, template_messages(record), template, tools=record.get("tools"), train_on=train_on
        )
        if len(encoded.input_ids) > seq_len:
            stats["too_long"] += 1
            continue
        if encoded.num_trained == 0:
            stats["no_target"] += 1
            continue
        examples.append(
            Example(np.asarray(encoded.input_ids, np.int64), np.asarray(encoded.labels, np.int64))
        )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        bounds = np.cumsum([0] + [len(e) for e in examples])
        np.savez(
            cache,
            ids=np.concatenate([e.input_ids for e in examples]).astype(np.int32)
            if examples
            else np.zeros(0, np.int32),
            labels=np.concatenate([e.labels for e in examples]).astype(np.int32)
            if examples
            else np.zeros(0, np.int32),
            bounds=bounds,
            stats=json.dumps(stats),
        )
    return examples, stats


class SFTData:
    """Conversations for one or more epochs, packed into ``seq_len`` rows (or padded per row).

    ``repeat`` per source up-samples small sets (identity x4) or down-samples
    large ones (0.5); the fractional part is a seeded random subset.
    """

    def __init__(
        self,
        sources: list[DataSource],
        tokenizer,
        template: str,
        *,
        seq_len: int,
        epochs: float,
        seed: int,
        train_on: str = "all",
        packing: bool = True,
        cache_dir: str | Path | None = None,
        pad_id: int = 0,
        tokenizer_key: str = "",
    ) -> None:
        self.seq_len, self.seed, self.packing, self.pad_id = seq_len, seed, packing, pad_id
        self.stats: dict[str, dict[str, int]] = {}
        pool: list[Example] = []
        rng = np.random.default_rng([seed, 0xA11])
        for source in sources:
            examples, stats = tokenize_sft_file(
                source.path,
                tokenizer,
                template,
                seq_len=seq_len,
                train_on=train_on,
                cache_dir=cache_dir,
                tokenizer_key=tokenizer_key,
            )
            whole, fraction = divmod(source.repeat, 1.0)
            chosen = examples * int(whole)
            if fraction and examples:
                take = rng.choice(len(examples), size=int(round(fraction * len(examples))), replace=False)
                chosen += [examples[i] for i in sorted(take)]
            stats["used"] = len(chosen)
            self.stats[source.label] = stats
            pool += chosen
        if not pool:
            raise ValueError("no SFT examples left after filtering")
        self.pool = pool
        self.rows: list[list[int]] = []  # each row = example indices
        full, fraction = divmod(epochs, 1.0)
        for epoch in range(int(full) + (1 if fraction else 0)):
            order = np.random.default_rng([seed, epoch]).permutation(len(pool))
            if epoch == int(full):
                order = order[: int(round(fraction * len(pool)))]
            self.rows += self._pack(order) if packing else [[int(i)] for i in order]

    def _pack(self, order: np.ndarray) -> list[list[int]]:
        rows, current, used = [], [], 0
        for index in order:
            length = len(self.pool[index])
            if current and used + length > self.seq_len:
                rows.append(current)
                current, used = [], 0
            current.append(int(index))
            used += length
        if current:
            rows.append(current)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def num_tokens(self) -> int:
        return sum(len(self.pool[i]) for row in self.rows for i in row)

    def packing_efficiency(self) -> float:
        return self.num_tokens / (len(self.rows) * self.seq_len)

    def batch(self, row_indices: list[int]) -> dict[str, torch.Tensor]:
        rows = [self.rows[i] for i in row_indices]
        lengths = [sum(len(self.pool[i]) for i in row) for row in rows]
        width = self.seq_len if self.packing else max(lengths)
        ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        labels = np.full((len(rows), width), -100, dtype=np.int64)
        positions = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for r, row in enumerate(rows):
            cursor = 0
            for index in row:
                example = self.pool[index]
                n = len(example)
                ids[r, cursor : cursor + n] = example.input_ids
                labels[r, cursor : cursor + n] = example.labels
                labels[r, cursor] = -100  # never predict a conversation's first token from the previous one
                positions[r, cursor : cursor + n] = np.arange(n)
                cursor += n
            mask[r, :cursor] = 1
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(labels),
            "position_ids": torch.from_numpy(positions),
            "attention_mask": torch.from_numpy(mask),
        }


def steps_per_epoch(n_rows: int, sequences_per_step: int) -> int:
    return max(1, math.ceil(n_rows / sequences_per_step))
