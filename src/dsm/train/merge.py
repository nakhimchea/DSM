"""Checkpoint merging (``dsm merge``).

* ``interpolate`` - ``(1 - alpha) * A + alpha * B`` for every tensor, e.g. SFT <-> DPO
  to trade a little preference strength for less drift, or two SFT runs.
* ``chat-vector`` - ``target + scale * (instruct - base)``: transplants the
  upstream instruction tuning onto a continued-pre-trained base, so SFT starts
  from a chat-capable model. **Vocabulary-aware**: if the target has more
  embedding rows (Khmer extension), only the shared rows get the delta and the
  new rows are left as trained. ``--skip-embeddings`` leaves the embedding
  matrix untouched (useful after heavy CPT on a new language).

Math runs in fp32; results are stored in the target's dtype.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EMBEDDING = "model.embed_tokens.weight"


def load_tensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open has no __iter__
                tensors[key] = handle.get_tensor(key)
    if _tied(model_dir):
        tensors.pop("lm_head.weight", None)
    return tensors


def _tied(model_dir: Path) -> bool:
    config = model_dir / "config.json"
    return bool(json.loads(config.read_text()).get("tie_word_embeddings")) if config.is_file() else False


def _check_same(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], allow_rows: set[str] = frozenset()
) -> None:
    if set(a) != set(b):
        raise ValueError(f"tensor names differ: {sorted(set(a) ^ set(b))[:5]}")
    for key in a:
        if a[key].shape != b[key].shape and not (key in allow_rows and a[key].shape[1:] == b[key].shape[1:]):
            raise ValueError(f"shape mismatch for {key}: {tuple(a[key].shape)} vs {tuple(b[key].shape)}")


def interpolate(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    _check_same(a, b)
    out = {}
    for key, ta in a.items():
        if ta.is_floating_point():
            out[key] = ((1 - alpha) * ta.float() + alpha * b[key].float()).to(ta.dtype)
        else:
            out[key] = ta.clone()
    return out


def chat_vector(
    base: dict[str, torch.Tensor],
    instruct: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    scale: float = 1.0,
    skip_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    _check_same(base, instruct)
    _check_same(base, target, allow_rows={EMBEDDING})
    out = {}
    for key, tt in target.items():
        if not tt.is_floating_point() or (skip_embeddings and key == EMBEDDING):
            out[key] = tt.clone()
            continue
        delta = instruct[key].float() - base[key].float()
        merged = tt.float().clone()
        rows = min(delta.shape[0], merged.shape[0]) if delta.ndim else None
        if rows is not None and delta.shape != merged.shape:  # extended vocabulary: shared rows only
            merged[:rows] += scale * delta[:rows]
        else:
            merged += scale * delta
        out[key] = merged.to(tt.dtype)
    return out


def write_merged(
    tensors: dict[str, torch.Tensor], like_dir: str | Path, out_dir: str | Path, entry: dict
) -> Path:
    """Weights + every non-weight file of ``like_dir`` (config, tokenizer, code, LICENSE, NOTICE ...)."""
    from dsm.train.checkpoint import read_lineage

    like_dir, out = Path(like_dir), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for path in like_dir.iterdir():
        weights = path.name.startswith("model") and (
            path.suffix == ".safetensors" or path.name.endswith(".index.json")
        )
        if path.is_file() and not weights and path.name != "lineage.json":
            shutil.copyfile(path, out / path.name)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    lineage = read_lineage(like_dir) + [entry]
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsm merge", description="Merge DSM checkpoints.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("interpolate", help="(1 - alpha) * A + alpha * B")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("chat-vector", help="target + scale * (instruct - base)")
    p.add_argument("--base", required=True)
    p.add_argument("--instruct", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "interpolate":
        merged = interpolate(load_tensors(args.a), load_tensors(args.b), args.alpha)
        path = write_merged(
            merged, args.b, args.out, {"stage": "merge", "method": "interpolate", "alpha": args.alpha}
        )
    else:
        merged = chat_vector(
            load_tensors(args.base),
            load_tensors(args.instruct),
            load_tensors(args.target),
            args.scale,
            args.skip_embeddings,
        )
        entry = {
            "stage": "merge",
            "method": "chat_vector",
            "scale": args.scale,
            "skip_embeddings": args.skip_embeddings,
        }
        path = write_merged(merged, args.target, args.out, entry)
    print(f"saved {path}")
    return 0
