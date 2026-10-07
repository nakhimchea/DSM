"""``dsm train`` - pre-training, continued pre-training, annealing and SFT.

    dsm train --config configs/train/cpt_24gb.yaml
    dsm train --config configs/train/sft_24gb.yaml --set optimizer.lr=1e-5 --set epochs=1
    dsm train --config configs/train/cpt_24gb.yaml --bench 20     # speed + memory, nothing saved

Runs resume automatically from the newest checkpoint in ``output_dir``
(``--set resume=null`` starts over). The final model is written to
``<output_dir>/final`` as a complete DSM model directory.
"""

from __future__ import annotations

import argparse
import datetime as dt

from dsm.train.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsm train", description="Train DSM (pt / cpt / anneal / sft).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="training data list from `dsm data mix --out` (replaces data.train)")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (YAML)"
    )
    parser.add_argument(
        "--bench", type=int, metavar="STEPS", help="run STEPS steps without saving and report speed"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="load everything, print the plan, do not train"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set, args.data)
    if cfg.stage == "dpo":
        raise SystemExit("use `dsm dpo` for stage: dpo")

    from dsm.train.trainer import LMTrainer

    trainer = LMTrainer(cfg)
    print(
        f"stage {cfg.stage}: {trainer.total_steps} steps x {cfg.sequences_per_step} sequences of {cfg.seq_len} tokens "
        f"(~{trainer.total_steps * cfg.tokens_per_step / 1e6:.1f}M token slots), device {trainer.device}"
    )
    if args.dry_run:
        return 0
    if args.bench:
        start = trainer.tokens_seen
        began = dt.datetime.now()
        trainer.train(bench_steps=args.bench)
        seconds = (dt.datetime.now() - began).total_seconds()
        rate = (trainer.tokens_seen - start) / max(seconds, 1e-9)
        hours = trainer.total_steps * cfg.tokens_per_step / max(rate, 1e-9) / 3600
        print(f"bench: {rate:,.0f} tokens/s (incl. warm-up); the full run would take ~{hours:.1f} h")
        return 0
    path = trainer.train()
    print(f"saved {path}")
    return 0
