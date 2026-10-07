"""``dsm eval`` - evaluation suites, bits per byte, translation, multiple choice, lm-eval, gates.

dsm eval suite --model work/runs/sft/final --out results/sft
dsm eval suite --endpoint http://127.0.0.1:8080/v1 --out results/q8      # a served GGUF
dsm eval bpb --model work/runs/cpt/final --data km=datasets/packed/eval_km en=datasets/packed/eval_en
dsm eval gate --stage after_sft --results results/sft/metrics.json --baseline results/base/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write(metrics: dict, out: str | None) -> None:
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if out:
        path = Path(out)
        if path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def _generator(args):
    from dsm.eval.generator import EndpointGenerator, HFGenerator

    if args.endpoint:
        generator = EndpointGenerator(args.endpoint, model=args.served_model)
    elif args.model:
        generator = HFGenerator(args.model, device=args.device, batch_size=args.batch_size)
    else:
        raise SystemExit("pass --model DIR or --endpoint URL")
    return _Capped(generator, args.max_new_tokens) if args.max_new_tokens else generator


class _Capped:
    """Caps every request's ``max_new_tokens`` (smoke tests only: truncated answers score lower)."""

    def __init__(self, generator, cap: int) -> None:
        self.generator, self.cap = generator, cap

    def generate(self, requests):
        import dataclasses

        capped = [dataclasses.replace(r, max_new_tokens=min(r.max_new_tokens, self.cap)) for r in requests]
        return self.generator.generate(capped)


def _pairs(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).name, item
        out[name] = path
    return out


def _load_model(model_dir: str, device: str):
    import torch

    import dsm.modeling  # noqa: F401
    from dsm.modeling import DSMForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return DSMForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm eval", description="Evaluate DSM checkpoints or served models."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str, generation: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--out", help="metrics file (.json) or directory")
        p.add_argument("--device", default="auto")
        if generation:
            p.add_argument("--model", help="DSM model directory")
            p.add_argument("--endpoint", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
            p.add_argument("--served-model", default="dsm")
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument(
                "--max-new-tokens", type=int, help="cap every answer (smoke tests; lowers the scores)"
            )
        return p

    p = add("suite", "identity / safety / tools / tables / format / rag suites", generation=True)
    p.add_argument("--suites", default="identity,tools,tables,format,rag,safety")
    p.add_argument(
        "--safety-file", action="append", default=[], help="held-out converted safety prompts (JSONL)"
    )
    p = add("bpb", "bits per byte on packed evaluation sets")
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True, help="name=prefix pairs (packed datasets)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=1_000_000)
    p = add("kl", "mean token KL of a candidate checkpoint against a reference")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p = add("translate", "FLORES-style translation (chrF++)", generation=True)
    p.add_argument("--data", required=True, help="JSONL rows {src, tgt, src_lang, tgt_lang}")
    p = add("choice", "multiple choice by log-likelihood (Belebele / SIB style)")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="JSONL rows {context, question, options, answer, lang}")
    p = add("lm-eval", "standard benchmarks through lm-evaluation-harness")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", default="belebele_khm_Khmr,belebele_eng_Latn,belebele_zho_Hans")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", default="8")
    p = sub.add_parser("gate", help="check metrics against configs/eval/gates.yaml")
    p.add_argument("--stage", required=True)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--baseline", nargs="*", default=[])
    p.add_argument("--gates", default="configs/eval/gates.yaml")

    args = parser.parse_args(argv)
    if args.command == "suite":
        from dsm.eval.suites import run_suites

        names = [s.strip() for s in args.suites.split(",") if s.strip()]
        metrics = run_suites(_generator(args), names, args.safety_file, args.out)
        _write(metrics, None)
    elif args.command == "bpb":
        from dsm.eval.bpb import evaluate_bpb

        model = _load_model(args.model, args.device)
        _write(
            evaluate_bpb(
                model,
                Path(args.model) / "tokenizer.json",
                _pairs(args.data),
                seq_len=args.seq_len,
                max_tokens=args.max_tokens,
            ),
            args.out,
        )
    elif args.command == "kl":
        from dsm.eval.bpb import mean_token_kl

        reference, candidate = (
            _load_model(args.reference, args.device),
            _load_model(args.candidate, args.device),
        )
        _write(mean_token_kl(reference, candidate, _pairs(args.data), max_tokens=args.max_tokens), args.out)
    elif args.command == "translate":
        from dsm.eval.translation import evaluate_translation
        from dsm.utils import read_jsonl

        _write(evaluate_translation(_generator(args).generate, list(read_jsonl(args.data))), args.out)
    elif args.command == "choice":
        from tokenizers import Tokenizer

        from dsm.eval.choice import evaluate_choices
        from dsm.utils import read_jsonl

        model = _load_model(args.model, args.device)
        tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
        template = (Path(args.model) / "chat_template.jinja").read_text(encoding="utf-8")
        result = evaluate_choices(model, tokenizer, template, list(read_jsonl(args.data)))
        name = Path(args.data).stem
        _write(
            {
                f"choice/{name}/accuracy": result.accuracy,
                **{f"choice/{name}/accuracy/{k}": v for k, v in result.per_lang.items()},
            },
            args.out,
        )
    elif args.command == "lm-eval":
        from dsm.eval.harness import run_lm_eval

        batch = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
        _write(
            run_lm_eval(
                args.model, [t for t in args.tasks.split(",") if t], batch_size=batch, limit=args.limit
            ),
            args.out,
        )
    elif args.command == "gate":
        from dsm.eval.gates import evaluate_gates, load_results

        results = evaluate_gates(
            args.stage,
            load_results(args.results),
            load_results(args.baseline) if args.baseline else None,
            args.gates,
        )
        width = max((len(r.metric) for r in results), default=10)
        print(f"gate {args.stage} ({args.gates})")
        for r in results:
            print(f"[{r.status}] {r.metric.ljust(width)}  {r.detail}")
        failed = sum(r.status == "FAIL" for r in results)
        print(
            f"\n{failed} failed, {sum(r.status == 'PASS' for r in results)} passed, {sum(r.status == 'SKIP' for r in results)} skipped"
        )
        return 1 if failed else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
