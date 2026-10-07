"""``dsm dissect`` - architecture report for a DSM checkpoint (or just a config).

dsm dissect --model work/models/dsm-base --text km=samples/km.txt en=samples/en.txt zh=samples/zh.txt --out reports/dissect
dsm dissect --config work/models/dsm-base/config.json          # parameters / FLOPs / cache only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(n: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def _mib(n: float) -> str:
    return f"{n / 2**20:.2f} MiB" if n >= 2**20 else f"{n / 2**10:.1f} KiB"


def static_report(config, contexts=(4096, 32768)) -> tuple[dict, list[str]]:
    from dsm.dissect.params import cache_report, decode_flops_per_token, parameter_report

    params = parameter_report(config)
    lines = [
        "## Parameters",
        "",
        f"Total **{_fmt(params['total'])}**, without the embedding {_fmt(params['non_embedding'])}.",
        "",
        "| Component | Parameters | Share |",
        "|---|---:|---:|",
    ]
    for name, count in sorted(params["groups"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {_fmt(count)} | {params['shares'][name]:.1%} |")
    lines += ["", f"Layer types: {params['layer_types']}", ""]
    report = {"parameters": params, "decode": {}, "cache": {}}
    lines += [
        "## Decode cost per token",
        "",
        "| Context | Total FLOPs | LM head share | Attention-over-context share |",
        "|---:|---:|---:|---:|",
    ]
    for context in contexts:
        flops = decode_flops_per_token(config, context)
        report["decode"][context] = flops
        lines.append(
            f"| {context} | {_fmt(flops['total'])} | {flops['LM head'] / flops['total']:.1%} | {flops['attention over the context'] / flops['total']:.1%} |"
        )
    lines += [
        "",
        "## Inference cache",
        "",
        "| Context | KV cache | GDN state (fixed) | Total | If every layer were full attention |",
        "|---:|---:|---:|---:|---:|",
    ]
    for context in contexts:
        cache = cache_report(config, context)
        report["cache"][context] = cache
        lines.append(
            f"| {context} | {_mib(cache['kv_bytes_at_context'])} | {_mib(cache['gdn_state_bytes_per_sequence'])} | {_mib(cache['total_bytes_at_context'])} | {_mib(cache['full_attention_equivalent_bytes_at_context'])} |"
        )
    return report, lines


def _batches(model_dir: Path, texts: dict[str, str], seq_len: int, max_docs: int) -> dict[str, list[dict]]:
    import torch
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    out = {}
    for lang, path in texts.items():
        docs = [d.strip() for d in Path(path).read_text(encoding="utf-8").split("\n\n") if d.strip()][
            :max_docs
        ]
        batches = []
        for doc in docs:
            ids = tokenizer.encode(doc, add_special_tokens=False).ids[:seq_len]
            if len(ids) < 8:
                continue
            tensor = torch.tensor([ids])
            batches.append({"input_ids": tensor, "labels": tensor.clone()})
        out[lang] = batches
    return out


def dynamic_report(model, batches_by_lang: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    from dsm.dissect.layers import block_influence, layer_ablation, logit_lens

    device = next(model.parameters()).device
    report, lines = {}, ["", "## Layer importance", ""]
    kinds = model.config.layer_types
    table: dict[str, dict] = {}
    for lang, batches in batches_by_lang.items():
        if not batches:
            continue
        batches = [{k: v.to(device) for k, v in b.items()} for b in batches]
        bi = block_influence(model, batches)
        base, ablated = layer_ablation(model, batches)
        lens = logit_lens(model, batches)
        table[lang] = {
            "block_influence": bi,
            "baseline_loss": base,
            "ablation_loss": ablated,
            "logit_lens_loss": lens,
        }
    report["layers"] = table
    langs = list(table)
    header = "| Layer | Type | " + " | ".join(f"BI {lang} | Δloss skip {lang}" for lang in langs) + " |"
    lines += [header, "|" + "---|" * (2 + 2 * len(langs))]
    for i, kind in enumerate(kinds):
        cells = []
        for lang in langs:
            row = table[lang]
            cells += [
                f"{row['block_influence'][i]:.3f}",
                f"{row['ablation_loss'][i] - row['baseline_loss']:+.3f}",
            ]
        lines.append(f"| {i} | {'attn' if kind == 'full_attention' else 'GDN'} | " + " | ".join(cells) + " |")
    lines += ["", "Logit lens (loss read out after each layer):", ""]
    for lang in langs:
        lines.append(f"* {lang}: " + ", ".join(f"{v:.2f}" for v in table[lang]["logit_lens_loss"]))
    return report, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm dissect", description="Parameter / FLOP / cache report and layer importance."
    )
    parser.add_argument("--model", help="DSM model directory (enables the layer-importance analysis)")
    parser.add_argument("--config", help="config.json only (static report)")
    parser.add_argument(
        "--text", nargs="*", default=[], help="lang=path of plain text, documents separated by blank lines"
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-docs", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", help="output directory for report.json / report.md")
    args = parser.parse_args(argv)

    import torch

    import dsm.modeling  # noqa: F401
    from dsm.modeling import DSMConfig, DSMForCausalLM

    if args.model:
        config = DSMConfig.from_pretrained(args.model)
    elif args.config:
        config = DSMConfig.from_json_file(args.config)
    else:
        raise SystemExit("pass --model or --config")
    report, lines = static_report(config)
    if args.model and args.text:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        model = DSMForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
        texts = dict(item.split("=", 1) for item in args.text)
        dynamic, more = dynamic_report(model, _batches(Path(args.model), texts, args.seq_len, args.max_docs))
        report.update(dynamic)
        lines += more
    text = "# DSM dissection report\n\n" + "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text, encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return 0
