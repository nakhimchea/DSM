"""Training checkpoints (resume) and complete, shippable model directories (export).

``output_dir/checkpoints/step-000500/``  model.safetensors, optimizer.pt, trainer_state.json
``output_dir/final/``                    a DSM model directory: weights, config, tokenizer,
                                         chat template, remote code, LICENSE, NOTICE,
                                         model card, ``lineage.json``
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CARRY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
    "NOTICE",
    "README.md",
)
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def weights_state(model, dtype: str | None = None) -> dict[str, torch.Tensor]:
    """State dict without the tied output projection, optionally cast (fp32-pinned tensors stay fp32)."""
    state = model.state_dict()
    if getattr(model.config, "tie_word_embeddings", False):
        state.pop("lm_head.weight", None)
    target = _DTYPES[dtype] if dtype else None
    keep_fp32 = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    out = {}
    for name, tensor in state.items():
        tensor = tensor.detach()
        if target is not None and tensor.is_floating_point() and not any(k in name for k in keep_fp32):
            tensor = tensor.to(target)
        out[name] = tensor.contiguous().cpu()
    return out


def load_weights(model, path: str | Path) -> None:
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")


def read_lineage(model_dir: str | Path | None) -> list[dict]:
    if model_dir is None:
        return []
    path = Path(model_dir) / "lineage.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def save_model_dir(
    model,
    out_dir: str | Path,
    *,
    carry_from: str | Path | None,
    dtype: str = "float32",
    lineage_entry: dict | None = None,
    parent_lineage: list[dict] | None = None,
) -> Path:
    """Write a complete DSM model directory (loadable with ``from_pretrained`` or ``trust_remote_code``)."""
    from dsm.convert.upstream import copy_remote_code

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    save_file(weights_state(model, dtype), str(out / "model.safetensors"), metadata={"format": "pt"})
    from dsm.modeling import AUTO_MAP

    config = copy.deepcopy(model.config)
    config.dtype = _DTYPES[dtype]
    config.use_cache = True
    config.architectures = [type(model).__name__]
    config.auto_map = dict(AUTO_MAP)
    config.save_pretrained(out)
    if carry_from is not None:
        for name in CARRY_FILES:
            source = Path(carry_from) / name
            if source.is_file():
                shutil.copyfile(source, out / name)
    copy_remote_code(out)
    lineage = list(parent_lineage or [])
    if lineage_entry:
        lineage.append(
            {**lineage_entry, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        )
    (out / "lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


# -- resumable training state ---------------------------------------------------------------------


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"step-{step:07d}"


def save_training_state(path: Path, model, optimizer, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(weights_state(model), str(tmp / "model.safetensors"), metadata={"format": "pt"})
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)  # a checkpoint directory is complete or absent, never half-written


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir) / "checkpoints"
    done = (
        sorted(p for p in root.glob("step-*") if (p / "trainer_state.json").is_file())
        if root.is_dir()
        else []
    )
    return done[-1] if done else None


def load_training_state(path: Path, model, optimizer) -> dict:
    load_weights(model, path / "model.safetensors")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
    return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))


def prune_checkpoints(output_dir: str | Path, keep: int) -> None:
    root = Path(output_dir) / "checkpoints"
    done = sorted(p for p in root.glob("step-*") if (p / "trainer_state.json").is_file())
    for old in done[: max(0, len(done) - keep)]:
        shutil.rmtree(old)
