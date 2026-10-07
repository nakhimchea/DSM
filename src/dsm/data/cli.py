"""``dsm data`` - from raw text to packed, mixed training data.

    dsm data sources                                   # catalogue + licence decision per source
    HF_HUB_OFFLINE=0 dsm data download fineweb2_km hplt3_km --out datasets/raw --limit 3000000
    dsm data clean --source fineweb2_km --input datasets/raw/fineweb2_km --out datasets/clean/fineweb2_km.jsonl.zst
    dsm data dedup --input datasets/clean/fineweb2_km.jsonl.zst datasets/clean/hplt3_km.jsonl.zst --out-dir datasets/dedup
    dsm data decontam --input datasets/dedup/*.jsonl.zst --eval data/eval benchmarks/ --out-dir datasets/final
    dsm data pack --input datasets/final/fineweb2_km.jsonl.zst --tokenizer work/models/dsm-base --out datasets/packed/km_web
    dsm data mix --config configs/data/mix_cpt.yaml

    dsm data select --input datasets/final/km_web.jsonl.zst --min chars=1500 --max khmer_broken_per_1k=2 --out ...
    dsm data parallel --layout aligned --source alt_km_en --input datasets/raw/alt_km_en --out-docs ... --out-pairs ...
    dsm data evalset --kind belebele --input datasets/raw/belebele_km --out datasets/eval/belebele.jsonl

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
        keep_fields=tuple(args.keep_fields or ()),
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
    from dsm.utils import id_fraction

    tokenizer_json = Path(args.tokenizer) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos = special_token_ids(tokenizer_json)["<|endoftext|>"]
    config = Path(args.tokenizer) / "config.json"
    vocab = json.loads(config.read_text())["vocab_size"] if config.is_file() else None
    inputs = _files(args.input)
    stats = _load_stats(inputs)
    licenses = sorted({s.get("license", "?") for s in stats["sources"].values()})
    langs: Counter = Counter()

    heldout: list[str] = []

    def texts():
        for path in inputs:
            for record in read_jsonl(path):
                if args.heldout and id_fraction(record["id"]) < args.heldout:
                    heldout.append(record["text"])  # evaluation documents: never in a training pack
                    continue
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
    meta = {**meta, "langs": dict(langs), "heldout_fraction": args.heldout}
    if args.heldout_out:
        held = pack_texts(
            heldout,
            tokenizer,
            Path(args.heldout_out),
            eos_id=eos,
            vocab_size=vocab,
            meta={
                "name": Path(args.heldout_out).name,
                "heldout_of": name,
                "tokenizer": tokenizer_hash(tokenizer_json),
            },
        )
        print(
            f"held out {held['n_docs']:,} documents ({held['n_tokens']:,} tokens) -> {args.heldout_out}.bin"
        )
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
    if args.total:
        cfg["total" if cfg.get("kind") in ("sft", "preference") else "total_tokens"] = args.total
    if cfg.get("kind") in ("sft", "preference"):
        return _mix_conversations(cfg, args.out)
    available = {}
    for name, entry in list(cfg["sources"].items()):
        try:
            available[name] = PackedTokens(entry["path"]).n_tokens
        except FileNotFoundError:
            if entry.get("tokens"):  # planning before the data exists
                available[name] = float(entry["tokens"])
        if not available.get(name):
            print(f"warning: {name}: {entry['path']} not packed or empty - left out of the mixture")
            available.pop(name, None)
            del cfg["sources"][name]
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
    print("\n# use with `dsm train --data <this file>` (or paste into the training config):\n" + snippet)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(snippet, encoding="utf-8")
    return 0


def _entry_files(entry: dict) -> list[Path]:
    """``path`` or ``paths`` (files or glob patterns) of a mixture entry."""
    import glob

    files: list[Path] = []
    for pattern in entry.get("paths") or [entry["path"]]:
        matches = sorted(glob.glob(str(pattern)))
        files += [Path(m) for m in matches] if matches else [Path(pattern)]
    return files


def _mix_conversations(cfg: dict, out: str | None) -> int:
    """SFT / DPO: repeat factors so each source (one or more files) makes up its share of examples."""
    import yaml

    from dsm.data.mixture import repeats_for_shares

    files = {name: _entry_files(entry) for name, entry in cfg["sources"].items()}
    counts = {}
    for name, entry in cfg["sources"].items():
        present = [f for f in files[name] if f.is_file()]
        for missing in sorted(set(files[name]) - set(present)):
            print(f"warning: {name}: {missing} not found")
        counts[name] = (
            sum(sum(1 for _ in read_jsonl(f)) for f in present) if present else int(entry.get("count", 0))
        )
        files[name] = present or files[name]  # nothing built yet: keep the planned paths
    targets = {name: entry["share"] for name, entry in cfg["sources"].items()}
    repeats = repeats_for_shares(counts, targets, cfg.get("total"))
    print(f"{'source':28} {'count':>9} {'share':>6} {'repeat':>7}")
    for name in targets:
        print(
            f"{name:28} {counts[name]:9,} {targets[name] / sum(targets.values()):6.3f} {repeats.get(name, 0):7.3f}"
        )
    train = [
        {
            "path": str(f),
            "repeat": repeats[n],
            "name": n if len(files[n]) == 1 else f"{n}:{f.name.split('.')[0]}",
        }
        for n in targets
        if n in repeats
        for f in files[n]
    ]
    snippet = yaml.safe_dump({"data": {"train": train}}, sort_keys=False, allow_unicode=True)
    print("\n# use with `dsm train/dpo --data <this file>` (or paste into the training config):\n" + snippet)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(snippet, encoding="utf-8")
    return 0


def _bounds(items: list[str] | None) -> dict[str, float]:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        out[key] = float(value)
    return out


def cmd_select(args) -> int:
    """Keep documents whose ``meta`` statistics pass every bound; optionally a fixed sample."""
    from dsm.utils import id_fraction

    lower, upper = _bounds(args.min), _bounds(args.max)
    inputs = _files(args.input)
    stats: Counter = Counter()

    def reason(record: dict) -> str | None:
        meta = record.get("meta") or {}
        for bounds, fails, word in ((lower, float.__lt__, "below"), (upper, float.__gt__, "above")):
            for key, bound in bounds.items():
                if not isinstance(meta.get(key), (int, float)):
                    return f"missing_{key}"
                if fails(float(meta[key]), bound):
                    return f"{word}_{key}"
        if args.sample < 1.0 and (id_fraction(record["id"], args.seed) < args.sample) == args.invert:
            return "sampled_out" if args.invert else "not_sampled"
        return None

    def selected():
        for path in inputs:
            for record in read_jsonl(path):
                stats["input"] += 1
                why = reason(record)
                if why:
                    stats[f"drop: {why}"] += 1
                    continue
                stats["kept"] += 1
                yield record
                if args.max_docs and stats["kept"] >= args.max_docs:
                    return

    out = Path(args.out)
    if out.suffix == ".txt":  # plain text, documents separated by blank lines (llama.cpp tools)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for record in selected():
                handle.write(record["text"].strip() + "\n\n")
    else:
        write_jsonl(out, selected())
        _save_stats(out, inputs, "select", stats)
    print(f"selected {stats['kept']:,} of {stats['input']:,} documents -> {out}")
    return 0


def cmd_parallel(args) -> int:
    from dsm.data.builders.translation import parallel_to_sft
    from dsm.data.parallel import aligned_items, clean_item, read_rows, to_documents, to_pairs
    from dsm.data.pii import redact
    from dsm.data.sources import check, load_sources

    if args.source:
        source = load_sources(args.catalogue)[args.source]
        purpose = "eval" if source.tier == "eval_only" else "train"
        decision = check(source, purpose=purpose, **_gate_flags(args))
        if not decision.allowed:
            raise SystemExit(f"licence gate: {source.name}: {decision.reason}")
        name, license, generated_by = source.name, source.license, source.generated_by
        if purpose == "eval" and (args.out_docs or args.out_sft):
            raise SystemExit(f"{name} is an evaluation set: only --out-pairs is allowed")
    elif args.name and args.license:
        name, license, generated_by = args.name, args.license, args.generated_by
    else:
        raise SystemExit("pass --source (catalogue name) or --name and --license")
    inputs = []
    for raw in args.input:
        path = Path(raw)
        inputs += (
            sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".zst", ".tsv"))
            if path.is_dir()
            else [path]
        )
    stats: Counter = Counter()
    items = []
    for item in aligned_items((row for path in inputs for row in read_rows(path)), args.layout):
        stats["input"] += 1
        cleaned = clean_item(item, args.langs, args.max_ratio, stats)
        if cleaned is not None:
            items.append(cleaned)
    stats["kept"] = len(items)
    meta = {name: {"kind": "parallel", "langs": args.langs, "license": license, "generated_by": generated_by}}
    if args.out_docs:
        docs = to_documents(
            items,
            source=name,
            license=license,
            generated_by=generated_by,
            per_doc=args.per_doc,
            seed=args.seed,
        )
        for doc in docs:
            doc["text"], found = redact(doc["text"])
            for label, count in found.items():
                stats[f"pii: {label}"] += count
        write_jsonl(args.out_docs, docs)
        _save_stats(Path(args.out_docs), inputs, "parallel", stats, meta)
    pairs = to_pairs(items, source=name, license=license)
    if args.out_pairs:
        write_jsonl(args.out_pairs, pairs)
    if args.out_sft:
        write_jsonl(args.out_sft, parallel_to_sft(pairs, source=name, license=license, seed=args.seed))
    print(f"{name}: {len(items):,} aligned items of {stats['input']:,} ({len(pairs):,} directed pairs)")
    return 0


def cmd_evalset(args) -> int:
    from dsm.data.evalsets import belebele_rows, sib_rows
    from dsm.data.parallel import read_rows

    rows = (row for path in _files(args.input) for row in read_rows(path))
    if args.kind == "safety":
        from dsm.data.builders.safety import CONVERTERS

        if args.converter not in CONVERTERS:
            raise SystemExit(f"--kind safety needs --converter, one of {sorted(CONVERTERS)}")
        items = [item for item in (CONVERTERS[args.converter](row) for row in rows) if item]
        write_jsonl(args.out, items)
        print(f"safety/{args.converter}: {len(items):,} held-out prompts -> {args.out} (never train on them)")
        return 0
    if args.kind == "sib200":
        if args.lang not in ("km", "en", "zh"):
            raise SystemExit("--kind sib200 needs --lang km|en|zh (SIB rows carry no language)")
        items = list(sib_rows(rows, args.lang))
    else:
        items = list(belebele_rows(rows, args.lang))
    write_jsonl(args.out, items)
    print(f"{args.kind}: {len(items):,} items -> {args.out}")
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
    p.add_argument(
        "--keep-fields", nargs="*", help="raw fields copied into meta (e.g. int_score url) for `select`"
    )
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
    p.add_argument(
        "--heldout", type=float, default=0.0, help="leave out this fraction of documents (fixed by id)"
    )
    p.add_argument("--heldout-out", help="pack the held-out documents here (an evaluation set)")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("render-sft", help="SFT conversations -> plain-text documents (annealing)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_render_sft)
    p = sub.add_parser("mix", help="mixture weights from target shares, token counts and epoch caps")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p.add_argument("--total", type=float, help="override the budget (tokens, or conversations / pairs)")
    p.set_defaults(fn=cmd_mix)
    p = sub.add_parser("select", help="keep documents by meta statistics (high-quality subsets, samples)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min", nargs="*", metavar="KEY=VALUE", help="e.g. chars=1500 int_score=4")
    p.add_argument("--max", nargs="*", metavar="KEY=VALUE", help="e.g. khmer_broken_per_1k=2 dup_lines=0.05")
    p.add_argument("--sample", type=float, default=1.0, help="keep this fraction (deterministic by id)")
    p.add_argument(
        "--invert",
        action="store_true",
        help="keep the documents --sample leaves out (train / held-out splits)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-docs", type=int)
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("parallel", help="parallel text -> CPT documents, translation pairs, translation SFT")
    p.add_argument("--layout", choices=["aligned", "grouped", "pairs"], required=True)
    p.add_argument("--source", help="catalogue name")
    p.add_argument("--name")
    p.add_argument("--license")
    p.add_argument("--generated-by", default="human")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--langs", nargs="+", default=["km", "en", "zh"])
    p.add_argument("--max-ratio", type=float, default=3.0, help="largest allowed length ratio in a pair")
    p.add_argument("--per-doc", type=int, default=16, help="aligned sentences per CPT document")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-docs", help="CPT documents (.jsonl.zst)")
    p.add_argument("--out-pairs", help="{src, tgt, src_lang, tgt_lang} rows")
    p.add_argument("--out-sft", help="translation SFT conversations")
    p.set_defaults(fn=cmd_parallel)
    p = sub.add_parser("evalset", help="Belebele / SIB-200 / safety benchmark rows -> `dsm eval` files")
    p.add_argument("--kind", choices=["belebele", "sib200", "safety"], required=True)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--lang", help="language of the rows (required for sib200)")
    p.add_argument("--converter", help="--kind safety: the benchmark's converter (e.g. xstest)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_evalset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)
