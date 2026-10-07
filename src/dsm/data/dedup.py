"""Exact and near-duplicate removal (MinHash + LSH, numpy only).

* exact: hash of whitespace/case-normalised text;
* near: MinHash signatures over shingles - character 5-grams for Khmer and
  Chinese (no reliable word boundaries), word 5-grams otherwise - indexed with
  LSH (16 bands x 8 rows); candidate pairs are confirmed by the estimated
  Jaccard similarity (default >= 0.8). The first occurrence wins, so feed the
  highest-quality source first.

Memory is ~1 KB per document; deduplicate per language.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

from dsm.data.synth.filters import content_hash

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


def shingles(text: str, lang: str, n: int = 5) -> set[str]:
    if lang in ("km", "zh", "zh-Hant"):
        chars = "".join(text.split())
        return {chars[i : i + n] for i in range(max(1, len(chars) - n + 1))}
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def _hashes(items: set[str]) -> np.ndarray:
    return np.array(
        [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") for s in items],
        dtype=np.uint64,
    )


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, items: set[str]) -> np.ndarray:
        values = _hashes(items) if items else np.zeros(1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            permuted = (np.outer(values, self.a) + self.b) % _MERSENNE & _MAX_HASH
        return permuted.min(axis=0)


def jaccard_estimate(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


class Deduplicator:
    """Streaming deduplication: ``add`` returns the id of an earlier duplicate, or None."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = 16) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        self.threshold, self.bands, self.rows = threshold, bands, num_perm // bands
        self.hasher = MinHasher(num_perm)
        self.exact: dict[str, str] = {}
        self.buckets: list[dict[bytes, list[str]]] = [defaultdict(list) for _ in range(bands)]
        self.signatures: dict[str, np.ndarray] = {}

    def add(self, doc_id: str, text: str, lang: str) -> str | None:
        key = content_hash(text)
        if key in self.exact:
            return self.exact[key]
        signature = self.hasher.signature(shingles(text, lang))
        candidates: set[str] = set()
        bands = [signature[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]
        for bucket, band in zip(self.buckets, bands, strict=True):
            candidates.update(bucket.get(band, ()))
        for other in candidates:
            if jaccard_estimate(signature, self.signatures[other]) >= self.threshold:
                return other
        self.exact[key] = doc_id
        self.signatures[doc_id] = signature
        for bucket, band in zip(self.buckets, bands, strict=True):
            bucket[band].append(doc_id)
        return None
