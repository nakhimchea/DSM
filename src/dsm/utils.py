"""Small shared helpers: config loading, hashing, JSON(L) I/O."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from dsm import repo_root


def resolve_config(path: str | Path) -> Path:
    """Find a config file relative to the CWD first, then to the repository root."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    fallback = repo_root() / path
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"config not found: {path}")


def resolve_path(path: str | Path) -> Path:
    """Like :func:`resolve_config` for files or directories (e.g. ``data/seeds/identity``)."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    fallback = repo_root() / path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"not found (relative to the CWD or the repository): {path}")


def load_yaml(path: str | Path) -> Any:
    with open(resolve_config(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_identity(path: str | Path = "configs/identity.yaml") -> dict:
    return load_yaml(path)


def load_upstream(path: str | Path = "configs/upstream.yaml") -> dict:
    return load_yaml(path)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Read JSON lines (``.jsonl`` or zstandard-compressed ``.jsonl.zst``)."""
    path = Path(path)
    if path.suffix == ".zst":
        import io

        import zstandard

        with open(path, "rb") as raw:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
            for line in reader:
                if line.strip():
                    yield json.loads(line)
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write JSON lines (zstandard-compressed when the path ends in ``.zst``); returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if path.suffix == ".zst":
        import zstandard

        with open(path, "wb") as raw, zstandard.ZstdCompressor(level=10).stream_writer(raw) as writer:
            for row in rows:
                writer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                count += 1
        return count
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
