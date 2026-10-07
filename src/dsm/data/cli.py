"""``dsm data`` - from raw text to packed, mixed training data.

    dsm data sources                                   # catalogue + licence decision per source
    HF_HUB_OFFLINE=0 dsm data download fineweb2_km hplt3_km --out datasets/raw --limit 3000000
    dsm data clean --source fineweb2_km --input datasets/raw/fineweb2_km --out datasets/clean/fineweb2_km.jsonl.zst
    dsm data dedup --input datasets/clean/fineweb2_km.jsonl.zst datasets/clean/hplt3_km.jsonl.zst --out-dir datasets/dedup
    dsm data decontam --input datasets/dedup/*.jsonl.zst --eval data/eval benchmarks/ --out-dir datasets/final
    dsm data pack --input datasets/final/fineweb2_km.jsonl.zst --tokenizer work/models/dsm-base --out datasets/packed/km_web
    dsm data mix --config configs/data/mix_cpt.yaml

Every stage writes ``<output>.stats.json`` (counts per drop reason, PII
replacements, licences); ``pack`` turns the chain into ``DATACARD.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dsm.utils import read_jsonl, write_jsonl


def _files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out += sorted(
                p
                for p in path.rglob("*")
                if p.name.endswith((".jsonl", ".jsonl.zst", ".json"))
                and not p.name.endswith(".stats.json")
                and p.name != "SOURCE.json"
            )
        else:
            out.append(path)
    return out


def _stats_path(path: Path) -> Path:
    return path.with_name(path.name + ".stats.json")


def _load_stats(paths: list[Path]) -> dict:
    merged: dict = {"stages": {}, "sources": {}}
    for path in paths:
        stats_file = _stats_path(path)
        if stats_file.is_file():
            data = json.loads(stats_file.read_text(encoding="utf-8"))
            for stage, counter in data.get("stages", {}).items():
                merged["stages"].setdefault(stage, Counter()).update(counter)
            merged["sources"].update(data.get("sources", {}))
    return merged


def _save_stats(
    output: Path, inputs: list[Path], stage: str, counter: Counter, sources: dict | None = None
) -> None:
    previous = _load_stats(inputs)
    previous["stages"][stage] = counter
    previous["sources"].update(sources or {})
    payload = {"stages": {k: dict(v) for k, v in previous["stages"].items()}, "sources": previous["sources"]}
    _stats_path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _gate_flags(args) -> dict:
    return {
        "accept_counsel": args.accept_counsel,
        "allow_noncommercial": args.allow_noncommercial,
        "allow_naming_clause": args.allow_naming_clause,
    }


def cmd_sources(args) -> int:
    from dsm.data.sources import check, load_sources

    sources = load_sources(args.catalogue)
    problems = [p for s in sources.values() for p in s.problems()]
    width = max(len(n) for n in sources)
    for source in sources.values():
        if args.kind and source.kind != args.kind:
            continue
        decision = check(source, purpose="eval" if source.kind == "eval" else "train", **_gate_flags(args))
        flag = "OK  " if decision.allowed else "NO  "
        verify = " (verify id)" if source.verify else ""
        print(
            f"{flag}{source.name.ljust(width)}  {source.kind:10} {','.join(source.langs):10} {decision.reason}{verify}"
        )
    for problem in problems:
        print(f"catalogue problem: {problem}")
    return 1 if problems else 0


def cmd_download(args) -> int:
    from dsm.data.download import download
    from dsm.data.sources import gate, load_sources

    sources = load_sources(args.catalogue)
    for source in gate(args.names, sources, purpose="eval" if args.eval else "train", **_gate_flags(args)):
        paths = download(source, args.out, limit=args.limit)
        print(f"{source.name}: {len(paths)} shard(s) -> {Path(args.out) / source.name}")
    return 0


def cmd_clean(args) -> int:
    from dsm.data.clean import clean_records
    from dsm.data.sources import check, load_sources

    if args.source:
        source = load_sources(args.catalogue)[args.source]
        decision = check(source, **_gate_flags(args))
        if not decision.allowed:
            raise SystemExit(f"licence gate: {source.name}: {decision.reason}")
        license, generated_by, langs, internal = (
            source.license,
            source.generated_by,
            source.langs,
            source.internal,
        )
        name = source.name
        meta = {
            name: {
                k: getattr(source, k)
                for k in ("kind", "langs", "tier", "license", "generated_by", "internal")
            }
        }
    else:
        if not (args.name and args.license):
            raise SystemExit("without --source, pass --name and --license (e.g. internal documents)")
        name, license, generated_by, langs, internal = (
            args.name,
            args.license,
            args.generated_by,
            args.langs,
            args.internal,
        )
        meta = {
            name: {
                "kind": "pretrain",
                "langs": langs,
                "tier": "internal" if internal else "local",
                "license": license,
                "generated_by": generated_by,
                "internal": internal,
            }
        }
    if args.langs:
        langs = args.langs
    inputs = _files(args.input)
    stats: Counter = Counter()

    def rows():
        for path in inputs:
            yield from read_jsonl(path)

    out = Path(args.out)
    cleaned = clean_records(
        rows(),
        source=name,
        license=license,
        generated_by=generated_by,
        langs=langs,
        text_field=args.text_field,
        strict_pii=internal or args.strict_pii,
        stats=stats,
        min_chars=args.min_chars,
    )
    write_jsonl(out, cleaned)
    _save_stats(out, inputs, "clean", stats, meta)
    print(f"{name}: kept {stats['kept']:,} of {stats['input']:,} documents -> {out}")
    return 0


def cmd_dedup(args) -> int:
    from dsm.data.dedup import Deduplicator

    dedups: dict[str, Deduplicator] = {}
    out_dir = Path(args.out_dir)
    for path in _files(args.input):  # order = priority: earlier files win
        stats: Counter = Counter()

        def kept(path=path, stats=stats):
            for record in read_jsonl(path):
                stats["input"] += 1
                dedup = dedups.setdefault(record.get("lang", "und"), Deduplicator(threshold=args.threshold))
                duplicate_of = dedup.add(record["id"], record["text"], record.get("lang", "en"))
                if duplicate_of is None:
                    stats["kept"] += 1
                    yield record
                else:
                    stats["duplicates"] += 1

        target = out_dir / path.name
        write_jsonl(target, kept())
        _save_stats(target, [path], "dedup", stats)
        print(f"{path.name}: {stats['duplicates']:,} duplicates removed, {stats['kept']:,} kept")
    return 0


def cmd_decontam(args) -> int:
    from dsm.data.decontam import Decontaminator

    index = Decontaminator()
    suffixes = (".jsonl", ".jsonl.zst", ".json", ".yaml", ".yml", ".txt")
    eval_files: list[Path] = []
    for raw in args.eval:
        path = Path(raw)
        if path.is_dir():
            eval_files += sorted(f for f in path.rglob("*") if f.is_file() and f.name.endswith(suffixes))
        elif path.is_file():
            eval_files.append(path)
    strings = sum(index.add_file(f) for f in eval_files)
    print(
        f"indexed {strings:,} evaluation strings from {len(eval_files)} files ({len(index.index):,} n-grams)"
    )
    out_dir = Path(args.out_dir)
    for path in _files(args.input):
        stats: Counter = Counter()

        def kept(path=path, stats=stats):
            for record in read_jsonl(path):
                stats["input"] += 1
                if index.contaminated(record["text"], record.get("lang", "en")):
                    stats["contaminated"] += 1
                    continue
                stats["kept"] += 1
                yield record

        target = out_dir / path.name
        write_jsonl(target, kept())
        _save_stats(target, [path], "decontam", stats)
        print(f"{path.name}: {stats['contaminated']:,} contaminated documents removed")
    return 0


def cmd_pack(args) -> int:
    from tokenizers import Tokenizer

    from dsm.data.cards import write_card
    from dsm.data.packed import pack_texts, tokenizer_hash
    from dsm.tokenizer import special_token_ids

    tokenizer_json = Path(args.tokenizer) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos = special_token_ids(tokenizer_json)["<|endoftext|>"]
    config = Path(args.tokenizer) / "config.json"
    vocab = json.loads(config.read_text())["vocab_size"] if config.is_file() else None
    inputs = _files(args.input)
    stats = _load_stats(inputs)
    licenses = sorted({s.get("license", "?") for s in stats["sources"].values()})
    langs: Counter = Counter()

    def texts():
        for path in inputs:
            for record in read_jsonl(path):
                langs[record.get("lang", "und")] += 1
                yield record["text"]

    out = Path(args.out)
    name = args.name or out.name
    meta = pack_texts(
        texts(),
        tokenizer,
        out,
        eos_id=eos,
        vocab_size=vocab,
        meta={
            "name": name,
            "sources": sorted(stats["sources"]),
            "licenses": licenses,
            "tokenizer": tokenizer_hash(tokenizer_json),
        },
    )
    meta_path = out.with_name(out.name + ".json")
    meta = {**meta, "langs": dict(langs)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sources = [{"name": n, **v} for n, v in stats["sources"].items()]
    stats["stages"]["pack"] = Counter({"documents": meta["n_docs"], "tokens": meta["n_tokens"]})
    write_card(out.parent / f"{name}.card", name=name, sources=sources, stats=stats["stages"])
    print(f"{name}: {meta['n_docs']:,} documents, {meta['n_tokens']:,} tokens -> {out}.bin")
    return 0


def cmd_render_sft(args) -> int:
    """SFT conversations as plain-text documents (instruction-formatted text for annealing)."""
    from dsm.data.sft.schema import template_messages
    from dsm.tokenizer.render import render

    template = (Path(args.tokenizer) / "chat_template.jinja").read_text(encoding="utf-8")
    count = 0

    def docs():
        nonlocal count
        for path in _files(args.input):
            for record in read_jsonl(path):
                count += 1
                yield {
                    "id": f"{record['id']}-text",
                    "text": render(template_messages(record), template, tools=record.get("tools")),
                    "lang": record.get("lang", "mixed"),
                    "source": record.get("source"),
                    "license": record.get("license"),
                    "generated_by": record.get("generated_by"),
                }

    write_jsonl(Path(args.out), docs())
    print(f"rendered {count:,} conversations -> {args.out}")
    return 0


def cmd_mix(args) -> int:
    import yaml

    from dsm.data.mixture import plan_mixture
    from dsm.data.packed import PackedTokens
    from dsm.utils import load_yaml

    cfg = load_yaml(args.config)
    if cfg.get("kind") in ("sft", "preference"):
        return _mix_conversations(cfg, args.out)
    available = {}
    for name, entry in cfg["sources"].items():
        try:
            available[name] = PackedTokens(entry["path"]).n_tokens
        except FileNotFoundError:
            available[name] = float(entry.get("tokens", 0))
    targets = {name: entry["share"] for name, entry in cfg["sources"].items()}
    caps = {name: entry.get("max_epochs", cfg.get("max_epochs", 4)) for name, entry in cfg["sources"].items()}
    plan = plan_mixture(available, targets, float(cfg["total_tokens"]), caps)
    print(f"{'source':24} {'target':>7} {'weight':>7} {'tokens':>12} {'epochs':>7}")
    for name in targets:
        mark = " (capped)" if name in plan.capped else ""
        print(
            f"{name:24} {targets[name] / sum(targets.values()):7.3f} {plan.weights[name]:7.3f} {plan.tokens[name]:12,.0f} {plan.epochs[name]:7.2f}{mark}"
        )
    train = [
        {"path": cfg["sources"][n]["path"], "weight": round(plan.weights[n], 4), "name": n} for n in targets
    ]
    snippet = yaml.safe_dump({"data": {"train": train}}, sort_keys=False, allow_unicode=True)
    print("\n# paste into the training config:\n" + snippet)
    if args.out:
        Path(args.out).write_text(snippet, encoding="utf-8")
    return 0


def _mix_conversations(cfg: dict, out: str | None) -> int:
    """SFT / DPO: repeat factors per file so each makes up its target share of examples."""
    import yaml

    from dsm.data.mixture import repeats_for_shares

    counts = {
        name: sum(1 for _ in read_jsonl(entry["path"]))
        if Path(entry["path"]).is_file()
        else int(entry.get("count", 0))
        for name, entry in cfg["sources"].items()
    }
    targets = {name: entry["share"] for name, entry in cfg["sources"].items()}
    repeats = repeats_for_shares(counts, targets, cfg.get("total"))
    print(f"{'source':28} {'count':>9} {'share':>6} {'repeat':>7}")
    for name in targets:
        print(
            f"{name:28} {counts[name]:9,} {targets[name] / sum(targets.values()):6.3f} {repeats.get(name, 0):7.3f}"
        )
    train = [
        {"path": cfg["sources"][n]["path"], "repeat": repeats[n], "name": n} for n in targets if n in repeats
    ]
    snippet = yaml.safe_dump({"data": {"train": train}}, sort_keys=False, allow_unicode=True)
    print("\n# paste into the training config:\n" + snippet)
    if out:
        Path(out).write_text(snippet, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsm data", description="Download, clean, deduplicate, decontaminate, pack and mix data."
    )
    parser.add_argument("--catalogue", default="configs/data/sources.yaml")
    parser.add_argument(
        "--accept-counsel", action="store_true", help="use 'needs_counsel' sources (after sign-off)"
    )
    parser.add_argument("--allow-noncommercial", action="store_true", help="never for a commercial release")
    parser.add_argument("--allow-naming-clause", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sources", help="list the catalogue with licence decisions")
    p.add_argument("--kind")
    p.set_defaults(fn=cmd_sources)
    p = sub.add_parser("download", help="download sources (needs network; HF_HUB_OFFLINE=0)")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", default="datasets/raw")
    p.add_argument("--limit", type=int)
    p.add_argument("--eval", action="store_true", help="download evaluation-only sources")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("clean", help="normalise, language-ID, quality-filter and PII-redact documents")
    p.add_argument("--source", help="catalogue name (licence and languages from the catalogue)")
    p.add_argument("--name")
    p.add_argument("--license")
    p.add_argument("--generated-by", default="human")
    p.add_argument("--langs", nargs="*")
    p.add_argument(
        "--internal", action="store_true", help="internal bank documents: strict PII, INTERNAL data card"
    )
    p.add_argument("--strict-pii", action="store_true")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-field", default="text")
    p.add_argument("--min-chars", type=int, default=200)
    p.set_defaults(fn=cmd_clean)
    p = sub.add_parser("dedup", help="exact + MinHash near-duplicate removal (earlier inputs win)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.8)
    p.set_defaults(fn=cmd_dedup)
    p = sub.add_parser("decontam", help="drop documents that contain evaluation text")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--eval", nargs="+", required=True, help="evaluation files or directories")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_decontam)
    p = sub.add_parser("pack", help="tokenise and pack documents (+ data card)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--tokenizer", required=True, help="model directory with tokenizer.json")
    p.add_argument("--out", required=True, help="output prefix, e.g. datasets/packed/km_web")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("render-sft", help="SFT conversations -> plain-text documents (annealing)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_render_sft)
    p = sub.add_parser("mix", help="mixture weights from target shares, token counts and epoch caps")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_mix)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)
