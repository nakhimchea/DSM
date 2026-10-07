"""Download catalogue sources to raw JSONL shards (run on a connected machine).

    HF_HUB_OFFLINE=0 dsm data download fineweb2_km --out datasets/raw --limit 2000000

Uses ``datasets`` streaming, so nothing larger than one shard is held in memory.
The licence gate runs first; refused sources are never downloaded. Gated
datasets need ``huggingface-cli login`` (or HF_TOKEN) on that machine.
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from dsm.data.sources import Source
from dsm.utils import write_jsonl


def _rows(source: Source, limit: int | None) -> Iterator[dict]:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("offline mode is on: run downloads with HF_HUB_OFFLINE=0 on a connected machine")
    if not source.hf:
        raise RuntimeError(
            f"{source.name}: no Hugging Face location; fetch {source.url or 'it'} manually and use `dsm data clean --input`"
        )
    from datasets import load_dataset

    hf = source.hf
    dataset = load_dataset(hf["repo"], hf.get("config"), split=hf.get("split", "train"), streaming=True)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            return
        subset = str(row.get("source") or row.get("subset") or "")
        if subset and any(
            fnmatch.fnmatch(subset.lower(), pattern.lower()) for pattern in source.exclude_subsets
        ):
            continue
        yield row


def write_shards(rows: Iterable[dict], out_dir: str | Path, shard_rows: int = 200_000) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, buffer = [], []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= shard_rows:
            paths.append(out / f"shard-{len(paths):05d}.jsonl.zst")
            write_jsonl(paths[-1], buffer)
            buffer = []
    if buffer:
        paths.append(out / f"shard-{len(paths):05d}.jsonl.zst")
        write_jsonl(paths[-1], buffer)
    return paths


def download(source: Source, out_root: str | Path, limit: int | None = None) -> list[Path]:
    out = Path(out_root) / source.name
    paths = write_shards(_rows(source, limit), out)
    meta = {
        "source": source.name,
        "license": source.license,
        "tier": source.tier,
        "generated_by": source.generated_by,
        "hf": source.hf,
        "shards": len(paths),
        "limit": limit,
    }
    (out / "SOURCE.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths
