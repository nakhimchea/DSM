"""Direct preference optimisation (``dsm dpo --config configs/train/dpo_24gb.yaml``).

Loss (sigmoid DPO, optional label smoothing = conservative DPO)::

    margin = beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r)))
    loss   = -(1 - eps) * log sigmoid(margin) - eps * log sigmoid(-margin)

``ipo`` uses ``(margin / beta - 1 / (2 beta))^2`` on the log-ratio difference.
``sft_weight`` adds the length-normalised NLL of the chosen answer (RPO-style),
which keeps the chosen answers likely and limits drift.

Reference log-probabilities are computed ONCE, before the first update, with
the initial weights - exactly the reference model - so no second copy of the
model has to sit in GPU memory. They are cached in ``output_dir/reference_logps-<key>.npy``,
keyed by every token of the pairs and the starting weights, so a resumed run reuses them and a
run on new data or a new starting model never does.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dsm.modeling.ops_dsm import chunked_token_logprobs
from dsm.train.config import TrainConfig, load_config
from dsm.train.trainer import BaseTrainer


@dataclass
class Pair:
    chosen_ids: np.ndarray
    chosen_labels: np.ndarray
    rejected_ids: np.ndarray
    rejected_labels: np.ndarray


def tokenize_pairs(path: str, tokenizer, template: str, seq_len: int) -> tuple[list[Pair], dict[str, int]]:
    from dsm.tokenizer.render import encode_for_training
    from dsm.utils import read_jsonl

    pairs, stats = [], {"records": 0, "too_long": 0}
    for record in read_jsonl(path):
        stats["records"] += 1
        system = [{"role": "system", "content": record["system"]}] if record.get("system") is not None else []
        prompt = system + list(record["prompt"])
        encoded = [
            encode_for_training(
                tokenizer, prompt + [answer], template, tools=record.get("tools"), train_on="last"
            )
            for answer in (record["chosen"], record["rejected"])
        ]
        if max(len(e.input_ids) for e in encoded) > seq_len:
            stats["too_long"] += 1
            continue
        pairs.append(
            Pair(
                np.asarray(encoded[0].input_ids, np.int64),
                np.asarray(encoded[0].labels, np.int64),
                np.asarray(encoded[1].input_ids, np.int64),
                np.asarray(encoded[1].labels, np.int64),
            )
        )
    return pairs, stats


def sequence_logps(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probability and number of target tokens per sequence."""
    hidden = model.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).last_hidden_state
    labels = batch["labels"][:, 1:]
    flat = chunked_token_logprobs(
        hidden[:, :-1].reshape(-1, hidden.shape[-1]),
        model.lm_head.weight,
        labels.reshape(-1),
        model.config.loss_chunk_size,
    ).view(labels.shape)
    mask = labels != -100
    return (flat * mask).sum(-1), mask.sum(-1)


def dpo_loss(
    policy_c: torch.Tensor,
    policy_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair loss and the implicit reward margin ``beta * (log-ratio difference)``."""
    logits = (policy_c - ref_c) - (policy_r - ref_r)
    if loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        losses = -(1 - label_smoothing) * F.logsigmoid(beta * logits) - label_smoothing * F.logsigmoid(
            -beta * logits
        )
    return losses, beta * logits


class DPOTrainer(BaseTrainer):
    def __init__(self, cfg: TrainConfig, quiet: bool = False) -> None:
        super().__init__(cfg, quiet=quiet)
        if self.template is None:
            raise ValueError(f"{self.tok_dir} has no chat_template.jinja")
        self.pad_id = self.model.config.pad_token_id or 0
        from dsm.train.config import expand_globs

        cfg.train = expand_globs(cfg.train)
        rng = np.random.default_rng([cfg.seed, 0xD90])
        pool: list[Pair] = []
        self.stats = {}
        for source in cfg.train:
            pairs, stats = tokenize_pairs(source.path, self.tokenizer, self.template, cfg.seq_len)
            whole, fraction = divmod(source.repeat, 1.0)
            chosen = pairs * int(whole)
            if fraction and pairs:
                take = rng.choice(len(pairs), size=int(round(fraction * len(pairs))), replace=False)
                chosen += [pairs[i] for i in sorted(take)]
            stats["used"] = len(chosen)
            self.stats[source.label] = stats
            pool += chosen
        if not pool:
            raise ValueError("no preference pairs left after filtering")
        self.pairs = pool
        full, fraction = divmod(cfg.epochs, 1.0)
        order: list[int] = []
        for epoch in range(int(full) + (1 if fraction else 0)):
            perm = np.random.default_rng([cfg.seed, epoch]).permutation(len(pool))
            if epoch == int(full):
                perm = perm[: int(round(fraction * len(pool)))]
            order += perm.tolist()
        self.order = order
        self.total_steps = cfg.max_steps or math.ceil(len(order) / cfg.sequences_per_step)
        self.reference = self._reference_logps()

    # -- batching ------------------------------------------------------------------------------------
    def _batch(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Chosen sequences first, then rejected (right-padded to the longest)."""
        seqs = [(self.pairs[i].chosen_ids, self.pairs[i].chosen_labels) for i in indices]
        seqs += [(self.pairs[i].rejected_ids, self.pairs[i].rejected_labels) for i in indices]
        width = max(len(ids) for ids, _ in seqs)
        ids = np.full((len(seqs), width), self.pad_id, dtype=np.int64)
        labels = np.full((len(seqs), width), -100, dtype=np.int64)
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        for row, (seq, lab) in enumerate(seqs):
            ids[row, : len(seq)], labels[row, : len(seq)], mask[row, : len(seq)] = seq, lab, 1
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(mask),
            "pair_index": torch.tensor(indices, dtype=torch.long),
        }

    def reference_key(self) -> str:
        """Cache key of the reference log-probs: every token and label, and the starting weights."""
        digest = hashlib.sha256(f"{self.cfg.seed}|{self.cfg.precision}".encode())
        for pair in self.pairs:
            for part in (pair.chosen_ids, pair.chosen_labels, pair.rejected_ids, pair.rejected_labels):
                digest.update(np.asarray(part, dtype=np.int64).tobytes())
        if self.src_dir is not None:  # content, not timestamps: a re-saved model may keep its mtime
            from dsm.utils import sha256_file

            for path in sorted(Path(self.src_dir).glob("model*.safetensors")):
                digest.update(sha256_file(path).encode())
        return digest.hexdigest()[:16]

    @torch.no_grad()
    def _reference_logps(self) -> np.ndarray:
        cache = self.out / f"reference_logps-{self.reference_key()}.npy"
        if cache.is_file():
            return np.load(cache)
        self.model.eval()
        out = np.zeros((len(self.pairs), 2), dtype=np.float64)
        for start in range(0, len(self.pairs), self.cfg.micro_batch_size):
            indices = list(range(start, min(start + self.cfg.micro_batch_size, len(self.pairs))))
            batch = {k: v.to(self.device) for k, v in self._batch(indices).items()}
            with self.autocast:
                logps, _ = sequence_logps(self.model, batch)
            n = len(indices)
            out[indices, 0] = logps[:n].double().cpu().numpy()
            out[indices, 1] = logps[n:].double().cpu().numpy()
        self.model.train()
        np.save(cache, out)
        return out

    def micro_batches(self, step: int) -> list[dict[str, torch.Tensor]]:
        cfg = self.cfg
        out = []
        for micro in range(cfg.grad_accum):
            start = (step * cfg.grad_accum + micro) * cfg.micro_batch_size
            indices = [
                self.order[i] for i in range(start, start + cfg.micro_batch_size) if i < len(self.order)
            ]
            if indices:
                out.append(self._batch(indices))
        return out

    def normaliser(self, batches: list[dict[str, torch.Tensor]]) -> int:
        return sum(len(b["pair_index"]) for b in batches)

    def loss(self, batch: dict[str, torch.Tensor], normaliser: int) -> tuple[torch.Tensor, dict]:
        settings = self.cfg.dpo
        index = batch.pop("pair_index").cpu().numpy()
        n = len(index)
        logps, counts = sequence_logps(self.model, batch)
        ref = torch.from_numpy(self.reference[index]).to(logps.device, torch.float32)
        losses, margins = dpo_loss(
            logps[:n], logps[n:], ref[:, 0], ref[:, 1], settings.beta, settings.loss, settings.label_smoothing
        )
        loss = losses.sum()
        if settings.sft_weight:
            loss = loss + settings.sft_weight * (-(logps[:n] / counts[:n].clamp(min=1))).sum()
        with torch.no_grad():
            margins, per_token = margins.detach(), logps.detach() / counts.clamp(min=1)
            info = {
                "reward_accuracy": float((margins > 0).float().mean()),
                "reward_margin": float(margins.mean()),
                "chosen_logp": float(per_token[:n].mean()),
                "rejected_logp": float(per_token[n:].mean()),
            }
        return loss / normaliser, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsm dpo", description="Direct preference optimisation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="training data list from `dsm data mix --out` (replaces data.train)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value"
    )
    parser.add_argument("--bench", type=int, help="run N steps without saving and report speed")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set, args.data)
    if cfg.stage != "dpo":
        raise SystemExit("dsm dpo needs stage: dpo")
    trainer = DPOTrainer(cfg)
    path = trainer.train(bench_steps=args.bench)
    if path:
        print(f"saved {path}")
    return 0
