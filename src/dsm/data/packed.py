"""Packed token files: the on-disk format for (continued) pre-training data.

A packed dataset ``<prefix>`` is three files::

    <prefix>.bin    uint32 little-endian token ids, all documents back to back,
                    every document terminated by the EOS id
    <prefix>.idx    uint64 little-endian document start offsets (n_docs + 1
                    entries, the last one equals n_tokens)
    <prefix>.json   metadata: format, vocab size, EOS id, token / document
                    counts, tokenizer hash, source, licence, language ...

Both binaries are memory-mapped when reading, so multi-GB corpora cost no RAM.
Training windows are cut from the token stream (:meth:`PackedTokens.window`);
position ids restart at every document start inside a window, which is the
only signal the model needs for document-isolated packing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

FORMAT = "dsm-packed-v1"
_TOKEN_DTYPE = np.dtype("<u4")
_OFFSET_DTYPE = np.dtype("<u8")


def _paths(prefix: str | Path) -> tuple[Path, Path, Path]:
    prefix = Path(prefix)
    return (
        prefix.with_name(prefix.name + ".bin"),
        prefix.with_name(prefix.name + ".idx"),
        prefix.with_name(prefix.name + ".json"),
    )


def tokenizer_hash(tokenizer_json: str | Path) -> str:
    return hashlib.sha256(Path(tokenizer_json).read_bytes()).hexdigest()[:16]


class PackedWriter:
    """Append documents (token id lists) and finalise the index on ``close()``."""

    def __init__(self, prefix: str | Path, *, vocab_size: int, eos_id: int, meta: dict | None = None) -> None:
        self.bin_path, self.idx_path, self.meta_path = _paths(prefix)
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size, self.eos_id = vocab_size, eos_id
        self.meta = dict(meta or {})
        self._handle = open(self.bin_path, "wb")  # noqa: SIM115 - kept open across add() calls, closed in close()
        self._offsets = [0]
        self._closed = False

    def add(self, ids: Sequence[int], add_eos: bool = True) -> None:
        array = np.asarray(ids, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() >= self.vocab_size):
            raise ValueError("token id outside the vocabulary")
        if add_eos and (array.size == 0 or array[-1] != self.eos_id):
            array = np.append(array, self.eos_id)
        if array.size == 0:
            return
        self._handle.write(array.astype(_TOKEN_DTYPE).tobytes())
        self._offsets.append(self._offsets[-1] + int(array.size))

    def close(self) -> dict:
        if self._closed:
            return self.meta
        self._handle.close()
        np.asarray(self._offsets, dtype=_OFFSET_DTYPE).tofile(self.idx_path)
        self.meta.update(
            format=FORMAT,
            dtype="uint32",
            vocab_size=self.vocab_size,
            eos_id=self.eos_id,
            n_tokens=self._offsets[-1],
            n_docs=len(self._offsets) - 1,
        )
        self.meta_path.write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._closed = True
        return self.meta

    def __enter__(self) -> PackedWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PackedTokens:
    """Read-only, memory-mapped view of a packed dataset."""

    def __init__(self, prefix: str | Path) -> None:
        bin_path, idx_path, meta_path = _paths(prefix)
        self.prefix = Path(prefix)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{meta_path}: not a {FORMAT} dataset")
        self.tokens = (
            np.memmap(bin_path, dtype=_TOKEN_DTYPE, mode="r")
            if self.meta["n_tokens"]
            else np.zeros(0, _TOKEN_DTYPE)
        )
        self.offsets = np.fromfile(idx_path, dtype=_OFFSET_DTYPE).astype(np.int64)
        if len(self.tokens) != self.meta["n_tokens"] or self.offsets[-1] != len(self.tokens):
            raise ValueError(f"{self.prefix}: index and token file disagree (truncated write?)")

    @property
    def n_tokens(self) -> int:
        return int(self.meta["n_tokens"])

    @property
    def n_docs(self) -> int:
        return int(self.meta["n_docs"])

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.prefix.name)

    def doc(self, index: int) -> np.ndarray:
        return np.asarray(self.tokens[self.offsets[index] : self.offsets[index + 1]], dtype=np.int64)

    def n_windows(self, seq_len: int) -> int:
        return self.n_tokens // seq_len

    def window(self, index: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Tokens ``[index * seq_len, (index + 1) * seq_len)`` and their position ids.

        Positions count from the start of each document, or from the window start
        for a document that began in an earlier window.
        """
        start = index * seq_len
        ids = np.asarray(self.tokens[start : start + seq_len], dtype=np.int64)
        positions = np.arange(start, start + len(ids), dtype=np.int64)
        doc_starts = self.offsets[np.searchsorted(self.offsets, positions, side="right") - 1]
        return ids, positions - np.maximum(doc_starts, start)


def pack_texts(
    texts: Iterable[str],
    tokenizer,
    prefix: str | Path,
    *,
    eos_id: int,
    vocab_size: int | None = None,
    meta: dict | None = None,
    batch_size: int = 512,
) -> dict:
    """Tokenise texts with a ``tokenizers.Tokenizer`` and write a packed dataset (returns its metadata)."""
    vocab_size = vocab_size or tokenizer.get_vocab_size(with_added_tokens=True)
    with PackedWriter(prefix, vocab_size=vocab_size, eos_id=eos_id, meta=meta) as writer:
        batch: list[str] = []

        def flush() -> None:
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                writer.add(encoding.ids)
            batch.clear()

        for text in texts:
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
    return writer.meta
