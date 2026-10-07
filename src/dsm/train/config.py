"""Training configuration (YAML -> dataclasses) with ``--set key=value`` overrides.

One schema covers every stage. ``stage`` selects the data path:

* ``pt`` / ``cpt`` / ``anneal`` - packed token datasets (``dsm.data.packed``),
  fixed-length windows, loss on every token, document-isolated packing;
* ``sft`` - conversation JSONL, assistant-only loss, packed or padded;
* ``dpo`` - preference JSONL (see :mod:`dsm.train.dpo`).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STAGES = ("pt", "cpt", "anneal", "sft", "dpo")
TRAINABLE = ("all", "embeddings", "new_rows", "no_embeddings")


@dataclass
class DataSource:
    path: str
    weight: float = 1.0  # pt/cpt/anneal: sampling weight of the mixture
    repeat: float = 1.0  # sft/dpo: copies of the file per epoch (2.5 = 2 copies + a 50 % sample)
    name: str | None = None

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        base = Path(self.path).name
        for suffix in (".zst", ".jsonl", ".json"):
            base = base.removesuffix(suffix)
        return base


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    name: str = "cosine"  # cosine | wsd | linear | constant
    warmup: float = 0.01  # fraction of steps (< 1) or absolute steps (>= 1)
    decay: float = 0.2  # wsd: fraction of steps in the final decay
    min_lr_ratio: float = 0.1
    decay_shape: str = "linear"  # wsd: linear | sqrt | cosine


@dataclass
class SFTConfig:
    train_on: str = "all"  # all assistant turns | last
    packing: bool = True


@dataclass
class DPOSettings:
    beta: float = 0.1
    loss: str = "sigmoid"  # sigmoid | ipo
    label_smoothing: float = 0.0
    sft_weight: float = 0.0  # NLL on the chosen answer (RPO-style regulariser)


@dataclass
class TrainConfig:
    stage: str
    output_dir: str
    model: str | None = None  # DSM checkpoint directory (None + model_config = from scratch)
    model_config: str | None = None  # YAML with DSMConfig fields, for stage "pt" from scratch
    tokenizer: str | None = None  # directory with tokenizer.json / chat_template.jinja (default: model)
    train: list[DataSource] = field(default_factory=list)
    eval: list[DataSource] = field(default_factory=list)
    seed: int = 1234
    seq_len: int = 4096
    micro_batch_size: int = 1
    grad_accum: int = 1
    max_steps: int | None = None
    max_tokens: float | None = None  # pt/cpt/anneal: token budget -> steps
    epochs: float = 1.0  # sft/dpo
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOSettings = field(default_factory=DPOSettings)
    precision: str = "bf16"  # bf16 (autocast, fp32 master weights) | fp32
    gradient_checkpointing: bool = True
    trainable: str = "all"
    new_rows_from: int | None = None  # trainable == new_rows: first new row (default: <model>/extension.json)
    loss_chunk_size: int = 1024
    log_every: int = 10
    eval_every: int = 500
    eval_tokens: int = 1_000_000
    save_every: int = 1000
    keep_checkpoints: int = 2
    save_dtype: str = "float32"  # final export dtype (float32 between stages, bfloat16 to ship)
    resume: str | None = "auto"
    device: str = "auto"
    tensorboard: bool = False
    cache_dir: str | None = None

    # -- derived -------------------------------------------------------------------------------
    @property
    def tokens_per_step(self) -> int:
        return self.seq_len * self.micro_batch_size * self.grad_accum

    @property
    def sequences_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum

    def steps_for_tokens(self) -> int:
        if self.max_steps:
            return int(self.max_steps)
        if not self.max_tokens:
            raise ValueError("set max_steps or max_tokens")
        return max(1, math.ceil(float(self.max_tokens) / self.tokens_per_step))

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if self.trainable not in TRAINABLE:
            raise ValueError(f"trainable must be one of {TRAINABLE}")
        if self.trainable == "new_rows" and self.new_rows_from is None and self.model is None:
            raise ValueError("trainable: new_rows needs new_rows_from (or a model with extension.json)")
        if self.model is None and self.model_config is None:
            raise ValueError("set model (a DSM checkpoint) or model_config (from scratch)")
        if self.model is None and self.tokenizer is None:
            raise ValueError("training from scratch needs a tokenizer directory")
        if not self.train:
            raise ValueError("no training data (train: [...])")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if self.sft.train_on not in ("all", "last"):
            raise ValueError("sft.train_on must be all or last")
        if self.schedule.name not in ("cosine", "wsd", "linear", "constant"):
            raise ValueError("schedule.name must be cosine, wsd, linear or constant")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _coerce(value: Any, kind: Any) -> Any:
    """Build nested dataclasses / tuples from plain YAML values."""
    if dataclasses.is_dataclass(kind) and isinstance(value, dict):
        hints = {f.name: f.type for f in dataclasses.fields(kind)}
        unknown = set(value) - set(hints)
        if unknown:
            raise ValueError(f"unknown {kind.__name__} field(s): {sorted(unknown)}")
        resolved = {f.name: f for f in dataclasses.fields(kind)}
        return kind(**{k: _coerce(v, _field_type(resolved[k])) for k, v in value.items()})
    if kind is tuple and isinstance(value, list):
        return tuple(value)
    return value


_NESTED = {
    "optimizer": OptimizerConfig,
    "schedule": ScheduleConfig,
    "sft": SFTConfig,
    "dpo": DPOSettings,
}


def _field_type(f: dataclasses.Field) -> Any:
    if f.name in _NESTED:
        return _NESTED[f.name]
    if f.name == "betas":
        return tuple
    return None


def _set(tree: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """Load a YAML config; ``overrides`` are ``key.sub=value`` strings (values parsed as YAML)."""
    from dsm.utils import resolve_config

    tree = yaml.safe_load(resolve_config(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value: {item!r}")
        key, raw = item.split("=", 1)
        _set(tree, key.strip(), raw)
    return config_from_dict(tree)


def config_from_dict(tree: dict) -> TrainConfig:
    tree = dict(tree)
    data = tree.pop("data", None) or {}
    tree.setdefault("train", data.get("train", []))
    tree.setdefault("eval", data.get("eval", []))
    known = {f.name: f for f in dataclasses.fields(TrainConfig)}
    unknown = set(tree) - set(known)
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    kwargs = {}
    for key, value in tree.items():
        if key in ("train", "eval"):
            kwargs[key] = [
                DataSource(**s) if isinstance(s, dict) else DataSource(path=str(s)) for s in value or []
            ]
        elif key in _NESTED:
            kwargs[key] = _coerce(value or {}, _NESTED[key])
        else:
            kwargs[key] = value
    config = TrainConfig(**kwargs)
    for obj in (config, config.optimizer, config.schedule, config.dpo, *config.train, *config.eval):
        _fix_numbers(obj)
    config.optimizer.betas = tuple(float(b) for b in config.optimizer.betas)
    config.validate()
    return config


# Fields whose default is None but whose value is numeric.
_OPTIONAL_NUMBERS = {"max_steps": int, "max_tokens": float, "new_rows_from": int}


def _fix_numbers(obj: Any) -> None:
    """PyYAML reads ``1e-4`` (no dot) as a string; convert numeric-looking strings in numeric fields."""
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if not isinstance(value, str):
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        kind = _OPTIONAL_NUMBERS.get(f.name) or (
            type(default) if isinstance(default, (int, float)) and not isinstance(default, bool) else None
        )
        if kind is None:
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{f.name}: expected a number, got {value!r}") from exc
        setattr(obj, f.name, int(number) if kind is int else number)


def dump_config(config: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
