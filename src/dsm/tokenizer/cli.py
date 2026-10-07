"""``dsm tokenizer``: fertility report, Khmer vocabulary extension, embedding remap.

    dsm tokenizer analyze --tokenizer original=work/models/dsm-base extended=work/tok/km-ext \\
        --text km=datasets/final/eval_km.jsonl.zst en=... zh=... [--limit 2000] [--json report.json]
    dsm tokenizer extend --tokenizer work/models/dsm-base --input datasets/final/fineweb2_km.jsonl.zst \\
        --new-tokens 8192 --out work/tok/km-ext
    dsm tokenizer remap --model work/models/dsm-base --tokenizer work/tok/km-ext \\
        --out work/models/dsm-base-km-ext

Text inputs are JSONL(.zst) documents (``text`` field) or ``.txt`` files (one text per
line); directories are searched recursively.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from dsm.utils import read_jsonl

_SUFFIXES = (".jsonl", ".jsonl.zst", ".txt")


def _paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            out += sorted(p for p in path.rglob("*") if p.is_file() and p.name.endswith(_SUFFIXES))
        else:
            out.append(path)
    return out


def _rows(path: Path) -> Iterator[dict]:
    if path.name.endswith(".txt"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                yield {"text": line.rstrip("\n")}
    else:
        yield from read_jsonl(path)


def iter_texts(items: list[str], limit: int | None = None, lang: str | None = None) -> Iterator[str]:
    """Texts from files / directories, optionally only records labelled ``lang``."""
    count = 0
    for path in _paths(items):
        for row in _rows(path):
            text = row.get("text")
            if not text or (lang and row.get("lang", lang) != lang):
                continue
            yield text
            count += 1
            if limit and count >= limit:
                return


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--{what} expects NAME=PATH, got {item!r}")
        out[name] = value
    return out


def cmd_analyze(args) -> int:
    from tokenizers import Tokenizer

    from dsm.tokenizer.analyze import khmer_vocab_entries, report

    tokenizers = {}
    for name, directory in _pairs(args.tokenizer, "tokenizer").items():
        path = Path(directory) / "tokenizer.json" if Path(directory).is_dir() else Path(directory)
        tokenizers[name] = Tokenizer.from_file(str(path))
        entries = khmer_vocab_entries(json.loads(path.read_text(encoding="utf-8")))
        print(f"{name}: {tokenizers[name].get_vocab_size():,} tokens, {entries:,} with Khmer characters")
    texts = {lang: list(iter_texts([p], args.limit)) for lang, p in _pairs(args.text, "text").items()}
    data, table = report(tokenizers, texts)
    print()
    print(table)
    names = list(tokenizers)
    if len(names) > 1:
        print()
        for lang in texts:
            base = data[f"{names[0]}/{lang}"]["tokens"]
            for name in names[1:]:
                ratio = data[f"{name}/{lang}"]["tokens"] / max(base, 1)
                print(
                    f"{lang}: {name} needs {ratio:.2f}x the tokens of {names[0]} "
                    f"-> about {1 / max(ratio, 1e-9):.2f}x the characters per second"
                )
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_extend(args) -> int:
    from dsm.tokenizer.extend import extend_tokenizer_dir

    extension = extend_tokenizer_dir(
        args.tokenizer,
        args.out,
        iter_texts(args.input, lang=args.lang),
        n_new=args.new_tokens,
        start_id=args.start_id,
        min_count=args.min_count,
        max_chars=args.max_chars,
    )
    stats = extension.stats
    ratio = stats["khmer_tokens_after"] / max(stats["khmer_tokens_before"], 1)
    print(
        f"{len(extension.tokens):,} new Khmer tokens (ids {extension.start_id:,}.."
        f"{extension.start_id + len(extension.tokens) - 1:,}); Khmer tokens on the learning sample: "
        f"{stats['khmer_tokens_before']:,} -> {stats['khmer_tokens_after']:,} ({ratio:.2f}x)"
    )
    print(f"review the new tokens in {Path(args.out) / 'new_tokens.tsv'}")
    if len(extension.tokens) < args.new_tokens:
        print(f"note: only {len(extension.tokens):,} pairs reached --min-count {args.min_count}")
    return 0


def cmd_remap(args) -> int:
    from dsm.tokenizer.remap import extend_model_dir

    entry = extend_model_dir(args.model, args.tokenizer, args.out, pad_to=args.pad_to, dtype=args.dtype)
    print(
        f"{entry['new_tokens']:,} new rows from id {entry['start_id']:,}; vocab_size {entry['vocab_size']:,} "
        f"-> {args.out}"
    )
    print("next: train the new rows only (configs/train/pilot_tok_24gb.yaml), then CPT")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsm tokenizer", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="tokens per character / bytes per token per language")
    p.add_argument("--tokenizer", nargs="+", required=True, help="NAME=DIR (or NAME=tokenizer.json)")
    p.add_argument("--text", nargs="+", required=True, help="LANG=FILE_OR_DIR")
    p.add_argument("--limit", type=int, default=2000, help="texts per language")
    p.add_argument("--json", help="also write the numbers here")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("extend", help="learn Khmer merges on top of the existing vocabulary")
    p.add_argument("--tokenizer", required=True, help="directory with tokenizer.json (a DSM model dir)")
    p.add_argument("--input", nargs="+", required=True, help="cleaned Khmer documents")
    p.add_argument("--out", required=True)
    p.add_argument("--new-tokens", type=int, default=8192)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=50_000_000, help="Khmer characters to learn from")
    p.add_argument("--start-id", type=int, help="first new id (default: right after the last existing id)")
    p.add_argument("--lang", default="km", help="only records with this lang label ('' = all)")
    p.set_defaults(func=cmd_extend)

    p = sub.add_parser("remap", help="add embedding rows for an extended tokenizer")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True, help="output directory of `dsm tokenizer extend`")
    p.add_argument("--out", required=True)
    p.add_argument("--pad-to", type=int, default=64, help="round the row count up to a multiple of this")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], help="default: the model's dtype")
    p.set_defaults(func=cmd_remap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
