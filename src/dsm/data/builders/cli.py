"""``dsm build`` - build and validate SFT / DPO datasets.

Deterministic builders (no model needed)::

    dsm build seeds --out datasets/sft/seeds         # identity, tools, tables, formatting, safety, rag
    dsm build identity --out datasets/sft/identity --copies 2
    dsm build tools --out datasets/sft/tools.jsonl.zst --per-weight 10

Public datasets (rows exported to JSON / JSONL / Parquet)::

    dsm build convert --dataset xlam --input raw/xlam.json --out datasets/sft/xlam.jsonl.zst
    dsm build convert --dataset chat --source smoltalk2 --input datasets/raw/smoltalk2 --out datasets/sft/chat/smoltalk2.jsonl.zst

With a LOCAL teacher / translator (configs/teacher.yaml)::

    dsm build translate --input datasets/sft/tools.jsonl.zst --target km --out datasets/sft/tools-km.jsonl.zst
    dsm build regenerate --input datasets/sft/chat-zh.jsonl.zst --out datasets/sft/chat-zh-regen.jsonl.zst
    dsm build safety --public datasets/raw/aegis2 --converter aegis2 --teacher --out datasets/sft/safety/aegis2.jsonl.zst

Preference pairs and checks::

    dsm build preference --kind format --input datasets/sft/tables.jsonl.zst --out datasets/dpo/format.jsonl.zst
    dsm build validate datasets/sft/*.jsonl.zst
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dsm.utils import read_jsonl, resolve_path, write_jsonl

SEED_DIR = "data/seeds"


def read_rows(paths: list[str]) -> list[dict]:
    """Rows from ``.jsonl(.zst)``, ``.json`` (list, or ``{"data": [...]}``) or ``.parquet`` files / directories."""
    rows: list[dict] = []
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        files += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for path in files:
        name = path.name.lower()
        if name.endswith((".jsonl", ".jsonl.zst")):
            rows += list(read_jsonl(path))
        elif name.endswith(".json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows += data if isinstance(data, list) else data.get("data", [data])
        elif name.endswith(".parquet"):
            import pyarrow.parquet as pq

            rows += pq.read_table(path).to_pylist()
    return rows


def _seed(name: str) -> Path:
    return resolve_path(f"{SEED_DIR}/{name}")


def _report(name: str, count: int, out: Path, stats: Counter | dict | None = None) -> None:
    extra = f"  {dict(stats)}" if stats else ""
    print(f"{name}: {count} records -> {out}{extra}")


def _validate(records: list[dict]) -> list[str]:
    from dsm.data.sft.validate import validate_record

    return [f"{r.get('id')}: {p}" for r in records for p in validate_record(r)]


def _write(path: Path, records: list[dict]) -> int:
    problems = _validate(records)
    if problems:
        raise SystemExit("refusing to write invalid records:\n  " + "\n  ".join(problems[:20]))
    return write_jsonl(path, records)


def _teacher(path: str, role: str):
    from dsm.data.synth.client import LocalTeacher

    return LocalTeacher.from_config(path, role=role)


# -- sub-commands --------------------------------------------------------------------------------


def cmd_identity(args) -> None:
    from dsm.data.builders.identity import build
    from dsm.utils import load_identity, load_upstream

    out = Path(args.out)
    parts = build(load_identity(), load_upstream(), _seed("identity"), seed=args.seed, copies=args.copies)
    _write(out / "identity_sft.jsonl.zst", parts["sft"])
    _write(out / "identity_docs.jsonl.zst", parts["docs"])
    _write(out / "identity_dpo.jsonl.zst", parts["preference"])
    _report(
        "identity",
        len(parts["sft"]),
        out,
        {"docs": len(parts["docs"]), "preference": len(parts["preference"])},
    )


def cmd_tools(args) -> None:
    from dsm.data.builders.tools import build

    records = build(_seed("bank_tools/dialogues.yaml"), seed=args.seed, per_weight=args.per_weight)
    _report("tools", _write(Path(args.out), records), Path(args.out))


def cmd_tables(args) -> None:
    from dsm.data.builders.tables import build

    records = build(_seed("tables/tasks.yaml"), seed=args.seed, per_weight=args.per_weight)
    _report("tables", _write(Path(args.out), records), Path(args.out))


def cmd_formatting(args) -> None:
    from dsm.data.builders.formatting import build

    records = build(_seed("formatting/facts.yaml"), seed=args.seed)
    _report("formatting", _write(Path(args.out), records), Path(args.out))


def cmd_rag(args) -> None:
    from dsm.data.builders.rag import RagBuilder

    builder = RagBuilder(_seed("rag/kb.yaml"), seed=args.seed)
    if args.documents:  # only QA on your documents (the seed QA comes from `dsm build seeds`)
        records = builder.qa_from_documents(
            read_rows(args.documents), _teacher(args.teacher_config, "teacher")
        )
    else:
        records = builder.build()
    _report("rag", _write(Path(args.out), records), Path(args.out))


def cmd_safety(args) -> None:
    from dsm.data.builders.safety import CONVERTERS, SafetyBuilder, load_policy

    out = Path(args.out)
    builder = SafetyBuilder(_seed("safety/bank.yaml"), load_policy("configs/safety_policy.yaml"))
    if args.public:  # only the public prompts (the bank seeds come from `dsm build seeds`)
        if args.converter not in CONVERTERS:
            raise SystemExit(f"--converter must be one of {sorted(CONVERTERS)}")
        rows = [r for r in (CONVERTERS[args.converter](row) for row in read_rows(args.public)) if r]
        teacher = _teacher(args.teacher_config, "teacher") if args.teacher else None
        sft = builder.respond(rows, teacher=teacher)
        target = out if out.name.endswith((".jsonl", ".jsonl.zst")) else out / f"{args.converter}.jsonl.zst"
        _report(f"safety/{args.converter}", _write(target, sft), target)
        return
    sft, prefs = builder.from_seeds()
    _write(out / "safety_sft.jsonl.zst", sft)
    _write(out / "safety_dpo.jsonl.zst", prefs)
    _report("safety", len(sft), out, {"preference": len(prefs)})


def cmd_seeds(args) -> None:
    """Every deterministic builder with default sizes (the v0.1 fast-path SFT core)."""
    out = Path(args.out)
    for fn, target in (
        (cmd_identity, out / "identity"),
        (cmd_tools, out / "tools.jsonl.zst"),
        (cmd_tables, out / "tables.jsonl.zst"),
        (cmd_formatting, out / "formatting.jsonl.zst"),
        (cmd_rag, out / "rag.jsonl.zst"),
        (cmd_safety, out / "safety"),
    ):
        fn(
            argparse.Namespace(
                out=str(target),
                seed=args.seed,
                copies=2,
                per_weight=args.per_weight,
                documents=None,
                public=None,
            )
        )


def cmd_convert(args) -> None:
    from dsm.data.builders.datasets import DATASETS, convert_dataset
    from dsm.data.sources import check, load_sources

    catalogue = load_sources()
    if args.dataset is None and args.source in catalogue:
        args.dataset = catalogue[args.source].converter
    if args.dataset not in DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(DATASETS)} (or a --source with a converter)")
    if args.source in catalogue:  # licence and producer from the catalogue, through the licence gate
        source = catalogue[args.source]
        decision = check(source, accept_counsel=args.accept_counsel)
        if not decision.allowed:
            raise SystemExit(f"licence gate: {source.name}: {decision.reason}")
        args.license = args.license or source.license
        args.generated_by = args.generated_by or source.generated_by
    elif args.source:
        print(f"note: {args.source!r} is not in configs/data/sources.yaml - no licence gate applied")
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    try:
        records, stats = convert_dataset(
            args.dataset, rows, license=args.license, generated_by=args.generated_by, source=args.source
        )
    except ValueError as exc:
        raise SystemExit(f"dsm build convert: {exc}") from exc
    count = write_jsonl(Path(args.out), records)  # already validated record by record
    _report(args.dataset, count, Path(args.out), stats)


def cmd_translate(args) -> None:
    from dsm.data.builders.translation import translate_record

    translator = _teacher(args.teacher_config, "translator")
    records = list(read_jsonl(args.input))
    out, failed = [], 0
    for record in records:
        translated = translate_record(record, translator, args.target)
        if translated is None:
            failed += 1
        else:
            out.append(translated)
    _report("translate", _write(Path(args.out), out), Path(args.out), {"rejected": failed})


def cmd_regenerate(args) -> None:
    from dsm.data.builders.chat import regenerate

    records, stats = regenerate(list(read_jsonl(args.input)), _teacher(args.teacher_config, "teacher"))
    _report("regenerate", _write(Path(args.out), records), Path(args.out), stats)


def cmd_preference(args) -> None:
    from dsm.data.builders import preference

    records = [record for path in args.input for record in read_jsonl(path)]
    if args.kind == "language":
        pairs = preference.language_pairs(records, seed=args.seed)
    elif args.kind == "format":
        pairs = preference.format_pairs(records, seed=args.seed)
    elif args.kind == "over_refusal":
        pairs = preference.over_refusal_pairs(records, seed=args.seed)
    else:  # judged
        policy = _teacher(args.teacher_config, "policy")
        judge = _teacher(args.teacher_config, "teacher")
        prompts = [
            ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + r["messages"][:-1]
            for r in records
        ]
        candidates = preference.sample_candidates(prompts, policy, n=args.samples)
        pairs = preference.judge_pairs(records, candidates, judge, generated_by=policy.generated_by)
    _report(f"preference/{args.kind}", _write(Path(args.out), pairs), Path(args.out))


def cmd_validate(args) -> int:
    from dsm.data.sft.validate import record_kind

    failures = 0
    for path in args.files:
        records = list(read_jsonl(path))
        kinds = Counter(record_kind(r) for r in records)
        problems = _validate(records)
        categories = Counter(r.get("category") for r in records if "category" in r)
        langs = Counter(r.get("lang") for r in records)
        print(
            f"{path}: {len(records)} records {dict(kinds)}, {len(problems)} problems; "
            f"categories {dict(categories)}; langs {dict(langs)}"
        )
        for problem in problems[:10]:
            print(f"  {problem}")
        failures += bool(problems)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsm build", description="Build and validate SFT / DPO datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_: str, out: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        if out:
            p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--teacher-config", default="configs/teacher.yaml")
        return p

    add("identity", cmd_identity, "identity SFT, 'About DSM' documents and identity DPO pairs").add_argument(
        "--copies", type=int, default=2
    )
    add("tools", cmd_tools, "bank tool-calling dialogues").add_argument("--per-weight", type=int, default=10)
    add("tables", cmd_tables, "spreadsheet / table tasks with verified answers").add_argument(
        "--per-weight", type=int, default=10
    )
    add("formatting", cmd_formatting, "Markdown / LaTeX formatting samples")
    p = add("rag", cmd_rag, "grounded QA with citations and abstention")
    p.add_argument(
        "--documents", nargs="*", help="extra documents {id,title,text,lang} for teacher-written QA"
    )
    p = add("safety", cmd_safety, "bank safety seeds (+ public safety prompts)")
    p.add_argument("--public", nargs="*")
    p.add_argument("--converter")
    p.add_argument("--teacher", action="store_true", help="answer public prompts with the local teacher")
    add("seeds", cmd_seeds, "all deterministic builders").add_argument("--per-weight", type=int, default=10)
    p = add("convert", cmd_convert, "convert a public dataset")
    p.add_argument("--dataset", help="converter (default: the catalogue entry's converter)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--license")
    p.add_argument("--generated-by")
    p.add_argument("--source", help="catalogue name: licence, producer and licence gate from the catalogue")
    p.add_argument(
        "--accept-counsel", action="store_true", help="allow 'needs_counsel' sources (after sign-off)"
    )
    p.add_argument("--limit", type=int)
    p = add("translate", cmd_translate, "translate SFT records with the local translator (protected spans)")
    p.add_argument("--input", required=True)
    p.add_argument("--target", choices=("km", "zh", "en"), required=True)
    p = add("regenerate", cmd_regenerate, "rewrite assistant turns with the local teacher")
    p.add_argument("--input", required=True)
    p = add("preference", cmd_preference, "build DPO pairs")
    p.add_argument("--kind", choices=("language", "format", "over_refusal", "judged"), required=True)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=4)
    p = add("validate", cmd_validate, "validate SFT / DPO files", out=False)
    p.add_argument("files", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
