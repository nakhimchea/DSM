"""Single-GPU trainer for pre-training, continued pre-training, annealing and SFT.

Memory plan for a 0.8B model on 24 GB: fp32 master weights + AdamW states
(~12 GB), bf16 autocast for compute, gradient checkpointing per decoder layer,
and a chunked cross-entropy so the 248K-vocabulary logits never materialise.

Loss normalisation: every micro-batch loss is a SUM over its target tokens
divided by the number of target tokens of the WHOLE optimizer step (all
micro-batches), so gradient accumulation gives exactly the gradient of one big
batch, however unevenly tokens are spread over micro-batches.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dsm.train.checkpoint import (
    checkpoint_dir,
    latest_checkpoint,
    load_training_state,
    prune_checkpoints,
    read_lineage,
    save_model_dir,
    save_training_state,
)
from dsm.train.config import TrainConfig, dump_config
from dsm.train.schedules import lr_factor

PRETRAIN_STAGES = ("pt", "cpt", "anneal")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class MetricsLogger:
    """``metrics.jsonl`` (+ TensorBoard when installed and enabled) and a short console line."""

    def __init__(self, out_dir: Path, tensorboard: bool = False, quiet: bool = False) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "metrics.jsonl"
        self.quiet = quiet
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(out_dir / "tensorboard"))
            except ImportError:  # pragma: no cover - optional dependency
                print("tensorboard is not installed; logging to metrics.jsonl only")

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": step, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
        if not self.quiet:
            shown = " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
            )
            print(f"step {step}: {shown}", flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template_path = model_dir / "chat_template.jinja"
    template = template_path.read_text(encoding="utf-8") if template_path.is_file() else None
    return tokenizer, template


class BaseTrainer:
    """Model, optimizer, schedule, checkpoints, logging. Subclasses provide data and loss."""

    stage_name = "train"

    def __init__(self, cfg: TrainConfig, quiet: bool = False) -> None:
        import dsm.modeling  # noqa: F401 - registers the DSM classes
        from dsm.modeling import DSMConfig, DSMForCausalLM

        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed % (2**32))

        if cfg.model:
            self.src_dir = Path(cfg.model)
            self.model = DSMForCausalLM.from_pretrained(self.src_dir, dtype=torch.float32)
        else:
            from dsm.utils import load_yaml

            self.src_dir = None
            self.model = DSMForCausalLM(DSMConfig(**load_yaml(cfg.model_config)))
        self.tok_dir = Path(cfg.tokenizer) if cfg.tokenizer else self.src_dir
        self.tokenizer, self.template = load_tokenizer(self.tok_dir)
        self.tokenizer_key = hashlib.sha256((self.tok_dir / "tokenizer.json").read_bytes()).hexdigest()[:16]
        self.model.config.loss_chunk_size = cfg.loss_chunk_size
        self.model.config.use_cache = False
        self.model.to(self.device)
        self.model.train()
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.trainable = self._select_trainable()
        self.optimizer = self._build_optimizer()
        self.autocast = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if cfg.precision == "bf16"
            else contextlib.nullcontext()
        )
        self.logger = MetricsLogger(self.out, cfg.tensorboard, quiet=quiet)
        self.step = 0
        self.tokens_seen = 0
        self.total_steps = 0  # set by subclasses

    # -- parameters ------------------------------------------------------------------------------
    def _select_trainable(self) -> list[torch.nn.Parameter]:
        cfg, model = self.cfg, self.model
        embedding = model.get_input_embeddings().weight
        for param in model.parameters():
            param.requires_grad_(cfg.trainable in ("all", "no_embeddings"))
        if cfg.trainable == "no_embeddings":
            embedding.requires_grad_(False)
        elif cfg.trainable in ("embeddings", "new_rows"):
            embedding.requires_grad_(True)
        if cfg.trainable == "new_rows":
            keep = torch.zeros(embedding.shape[0], 1, dtype=embedding.dtype, device=embedding.device)
            keep[cfg.new_rows_from :] = 1.0
            embedding.register_hook(lambda grad: grad * keep)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("no trainable parameters")
        return params

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.cfg.optimizer
        decay, no_decay = [], []
        embedding = self.model.get_input_embeddings().weight
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            frozen_rows = self.cfg.trainable == "new_rows" and param is embedding
            if param.ndim < 2 or "A_log" in name or "dt_bias" in name or "norm" in name or frozen_rows:
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            g
            for g in (
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            )
            if g["params"]
        ]
        fused = self.device.type == "cuda"
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, fused=fused)

    def lr_at(self, step: int) -> float:
        return self.cfg.optimizer.lr * lr_factor(step, self.total_steps, self.cfg.schedule)

    # -- checkpoints -------------------------------------------------------------------------------
    def fingerprint(self) -> str:
        keys = (
            "stage",
            "seed",
            "seq_len",
            "micro_batch_size",
            "grad_accum",
            "train",
            "epochs",
            "max_steps",
            "max_tokens",
        )
        data = {k: v for k, v in self.cfg.to_dict().items() if k in keys}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def maybe_resume(self) -> None:
        resume = self.cfg.resume
        if not resume:
            return
        path = latest_checkpoint(self.out) if resume == "auto" else Path(resume)
        if path is None:
            return
        state = load_training_state(path, self.model, self.optimizer)
        if state.get("fingerprint") != self.fingerprint():
            print(f"warning: resuming from {path} with a different data / batch configuration")
        self.step, self.tokens_seen = int(state["step"]), int(state["tokens_seen"])
        print(f"resumed from {path} at step {self.step}")

    def save_checkpoint(self) -> None:
        state = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "fingerprint": self.fingerprint(),
            "stage": self.cfg.stage,
        }
        save_training_state(checkpoint_dir(self.out, self.step), self.model, self.optimizer, state)
        prune_checkpoints(self.out, self.cfg.keep_checkpoints)

    def lineage_entry(self) -> dict:
        return {
            "stage": self.cfg.stage,
            "steps": self.step,
            "tokens": self.tokens_seen,
            "data": [{"name": s.label, "weight": s.weight, "repeat": s.repeat} for s in self.cfg.train],
            "optimizer": {"lr": self.cfg.optimizer.lr, "schedule": self.cfg.schedule.name},
            "trainable": self.cfg.trainable,
        }

    def export(self, name: str = "final") -> Path:
        carry = self.src_dir or self.tok_dir
        return save_model_dir(
            self.model,
            self.out / name,
            carry_from=carry,
            dtype=self.cfg.save_dtype,
            lineage_entry=self.lineage_entry(),
            parent_lineage=read_lineage(self.src_dir),
        )

    # -- loop --------------------------------------------------------------------------------------
    def micro_batches(self, step: int) -> list[dict[str, torch.Tensor]]:
        raise NotImplementedError

    def loss(self, batch: dict[str, torch.Tensor], normaliser: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def normaliser(self, batches: list[dict[str, torch.Tensor]]) -> int:
        raise NotImplementedError

    def evaluate(self) -> dict:
        return {}

    def train(self, bench_steps: int | None = None) -> Path | None:
        dump_config(self.cfg, self.out / "config.json")
        self.maybe_resume()
        end = self.total_steps if bench_steps is None else min(self.total_steps, self.step + bench_steps)
        window = {"loss": 0.0, "tokens": 0, "time": 0.0, "steps": 0}
        while self.step < end:
            started = time.perf_counter()
            batches = self.micro_batches(self.step)
            normaliser = self.normaliser(batches)
            extras: dict[str, float] = {}
            loss_total = 0.0
            if normaliser > 0:
                for batch in batches:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with self.autocast:
                        loss, info = self.loss(batch, normaliser)
                    loss.backward()
                    loss_total += float(loss.detach())
                    for key, value in info.items():
                        extras[key] = extras.get(key, 0.0) + value
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.optimizer.grad_clip)
                )
                lr = self.lr_at(self.step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.step()
            else:
                grad_norm, lr = 0.0, self.lr_at(self.step)
            self.optimizer.zero_grad(set_to_none=True)
            tokens = sum(
                int(b["attention_mask"].sum()) if "attention_mask" in b else b["input_ids"].numel()
                for b in batches
            )
            self.tokens_seen += tokens
            self.step += 1
            elapsed = time.perf_counter() - started
            window["loss"] += loss_total
            window["tokens"] += tokens
            window["time"] += elapsed
            window["steps"] += 1
            if not math.isfinite(loss_total):
                raise FloatingPointError(f"non-finite loss at step {self.step}")
            if self.step % self.cfg.log_every == 0 or self.step == end:
                metrics = {
                    "loss": window["loss"] / window["steps"],
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_s": window["tokens"] / max(window["time"], 1e-9),
                    "tokens_seen": self.tokens_seen,
                    **{k: v / max(1, len(batches)) for k, v in extras.items()},
                }
                if self.device.type == "cuda":
                    metrics["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                self.logger.log(self.step, metrics)
                window = {"loss": 0.0, "tokens": 0, "time": 0.0, "steps": 0}
            if bench_steps is None:
                if (
                    self.cfg.eval
                    and self.cfg.eval_every
                    and self.step % self.cfg.eval_every == 0
                    and self.step < end
                ):
                    self.logger.log(self.step, self.evaluate())
                if self.cfg.save_every and self.step % self.cfg.save_every == 0 and self.step < end:
                    self.save_checkpoint()
        if bench_steps is not None:
            return None
        if self.cfg.eval:
            self.logger.log(self.step, self.evaluate())
        self.save_checkpoint()
        path = self.export()
        self.logger.close()
        return path


class LMTrainer(BaseTrainer):
    """Next-token training: packed windows (pt/cpt/anneal) or conversations (sft)."""

    def __init__(self, cfg: TrainConfig, quiet: bool = False) -> None:
        super().__init__(cfg, quiet=quiet)
        from dsm.eval.bpb import token_byte_lengths

        self.byte_lengths = torch.from_numpy(
            token_byte_lengths(self.tok_dir / "tokenizer.json", self.model.config.vocab_size)
        )
        pad_id = self.model.config.pad_token_id or 0
        if cfg.stage in PRETRAIN_STAGES:
            from dsm.train.datasets import PretrainMixture

            self.total_steps = cfg.steps_for_tokens()
            self.data = PretrainMixture(
                cfg.train, cfg.seq_len, self.total_steps * cfg.sequences_per_step, cfg.seed
            )
            self.n_rows = len(self.data)
        elif cfg.stage == "sft":
            from dsm.train.datasets import SFTData

            if self.template is None:
                raise ValueError(f"{self.tok_dir} has no chat_template.jinja")
            self.data = SFTData(
                cfg.train,
                self.tokenizer,
                self.template,
                seq_len=cfg.seq_len,
                epochs=cfg.epochs,
                seed=cfg.seed,
                train_on=cfg.sft.train_on,
                packing=cfg.sft.packing,
                cache_dir=cfg.cache_dir or str(self.out / "cache"),
                pad_id=pad_id,
                tokenizer_key=self.tokenizer_key,
            )
            self.n_rows = len(self.data)
            self.total_steps = cfg.max_steps or math.ceil(self.n_rows / cfg.sequences_per_step)
            if not quiet:
                print(
                    f"SFT: {self.n_rows} rows, packing efficiency {self.data.packing_efficiency():.2%}, {self.data.stats}"
                )
        else:
            raise ValueError(f"LMTrainer does not handle stage {cfg.stage!r} (use dsm dpo)")
        self._eval_sets = None

    def micro_batches(self, step: int) -> list[dict[str, torch.Tensor]]:
        cfg = self.cfg
        out = []
        for micro in range(cfg.grad_accum):
            start = (step * cfg.grad_accum + micro) * cfg.micro_batch_size
            indices = [i for i in range(start, start + cfg.micro_batch_size) if i < self.n_rows]
            if indices:
                out.append(self.data.batch(indices))
        return out

    def normaliser(self, batches: list[dict[str, torch.Tensor]]) -> int:
        return sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)

    def loss(self, batch: dict[str, torch.Tensor], normaliser: int) -> tuple[torch.Tensor, dict]:
        out = self.model(**batch, use_cache=False, num_items_in_batch=normaliser)
        return out.loss, {}

    # -- evaluation ----------------------------------------------------------------------------------
    def _eval_batches(self) -> dict[str, list[dict[str, torch.Tensor]]]:
        if self._eval_sets is not None:
            return self._eval_sets
        cfg = self.cfg
        sets: dict[str, list[dict[str, torch.Tensor]]] = {}
        for source in cfg.eval:
            if cfg.stage in PRETRAIN_STAGES:
                from dsm.train.datasets import eval_windows

                rows = eval_windows(source, cfg.seq_len, cfg.eval_tokens)
                batches = []
                for i in range(0, len(rows), cfg.micro_batch_size):
                    chunk = rows[i : i + cfg.micro_batch_size]
                    batches.append(
                        {
                            "input_ids": torch.from_numpy(np.stack([r[0] for r in chunk])),
                            "labels": torch.from_numpy(np.stack([r[1] for r in chunk])),
                            "position_ids": torch.from_numpy(np.stack([r[2] for r in chunk])),
                        }
                    )
            else:
                from dsm.train.datasets import SFTData

                data = SFTData(
                    [source],
                    self.tokenizer,
                    self.template,
                    seq_len=cfg.seq_len,
                    epochs=1,
                    seed=cfg.seed,
                    train_on=cfg.sft.train_on,
                    packing=cfg.sft.packing,
                    cache_dir=cfg.cache_dir or str(self.out / "cache"),
                    pad_id=self.model.config.pad_token_id or 0,
                    tokenizer_key=self.tokenizer_key,
                )
                batches = [
                    data.batch(list(range(i, min(i + cfg.micro_batch_size, len(data)))))
                    for i in range(0, len(data), cfg.micro_batch_size)
                ]
            sets[source.label] = batches
        self._eval_sets = sets
        return sets

    @torch.no_grad()
    def evaluate(self) -> dict:
        from dsm.eval.bpb import bits_per_byte, nll_and_bytes

        self.model.eval()
        results = {}
        for name, batches in self._eval_batches().items():
            nll = n_bytes = n_tokens = 0
            for batch in batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with self.autocast:
                    a, b, c = nll_and_bytes(self.model, {**batch, "use_cache": False}, self.byte_lengths)
                nll, n_bytes, n_tokens = nll + a, n_bytes + b, n_tokens + c
            results[f"eval/{name}/loss"] = nll / max(n_tokens, 1)
            results[f"eval/{name}/bpb"] = bits_per_byte(nll, n_bytes)
        self.model.train()
        return results

    def lineage_entry(self) -> dict:
        entry = super().lineage_entry()
        if self.cfg.stage in PRETRAIN_STAGES:
            entry["epochs_per_source"] = self.data.epochs_used(
                min(self.step * self.cfg.sequences_per_step, len(self.data))
            )
        return entry
