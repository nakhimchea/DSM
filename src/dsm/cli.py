"""``dsm`` command-line entry point.

Every sub-command lives in its own module and is imported lazily, so ``dsm data``
does not pay for importing torch and ``dsm --help`` is instant. All commands run
with the offline / no-telemetry environment of :mod:`dsm.env` (explicitly set
variables win, e.g. ``HF_HUB_OFFLINE=0`` for the one-time download scripts).
"""

from __future__ import annotations

import importlib
import sys

from dsm import __version__
from dsm.env import enforce_offline

# command -> (module exposing ``main(argv) -> int``, one-line help)
COMMANDS: dict[str, tuple[str, str]] = {
    "convert": ("dsm.convert.upstream", "convert an upstream checkpoint into a DSM checkpoint"),
    "verify": ("dsm.convert.verify", "check a DSM checkpoint against the upstream one (logits, tokenizer)"),
    "audit": ("dsm.audit.cli", "audit a model directory / the repo for branding, files and network use"),
    "tokenizer": ("dsm.tokenizer.cli", "analyse or extend the tokenizer (Khmer fertility, new merges)"),
    "data": ("dsm.data.cli", "download, clean, deduplicate, pack and mix training data"),
    "build": ("dsm.data.builders.cli", "build SFT / DPO datasets (identity, tools, tables, safety, ...)"),
    "train": ("dsm.train.cli", "pre-training, continued pre-training, annealing and SFT"),
    "dpo": ("dsm.train.dpo", "direct preference optimisation"),
    "merge": ("dsm.train.merge", "merge checkpoints (interpolation, chat vector)"),
    "eval": ("dsm.eval.cli", "evaluation suites and stage gates"),
    "dissect": ("dsm.dissect.cli", "parameter / FLOP / cache report, layer importance"),
    "export": ("dsm.export.cli", "export to GGUF for llama.cpp and launch llama-server"),
    "chat": ("dsm.serve.chat", "interactive chat in the terminal (fully offline)"),
    "preflight": ("dsm.train.preflight", "check the GPU machine before training"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        f"dsm {__version__} - DSM1.0 pipeline (offline by default)",
        "",
        "usage: dsm <command> [options]   (dsm <command> --help for details)",
        "",
        "commands:",
    ]
    lines += [f"  {name.ljust(width)}  {help_}" for name, (_, help_) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if args[0] in ("-V", "--version"):
        print(__version__)
        return 0
    command = args[0]
    if command not in COMMANDS:
        print(f"dsm: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    enforce_offline()
    module = importlib.import_module(COMMANDS[command][0])
    return int(module.main(args[1:]) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
