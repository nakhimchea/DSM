"""``dsm verify`` - prove a DSM checkpoint computes the same function as its source.

    dsm verify --dsm models/DSM1.0-0.8B-base --upstream models/upstream-base [--device cuda]

Two checks:
* tokenizer parity - the DSM tokenizer (transformers ``AutoTokenizer``) and the
  raw upstream ``tokenizer.json`` produce identical ids on km / en / zh / code /
  math / tool-call probes;
* logit parity - the upstream reference implementation (the class named in the
  source ``config.json``) and DSM produce the same next-token logits.

Run it in fp32 (default; ~3 GB RAM per 0.8B model on CPU). In bf16 both models
are compared to each other, which is noisier; prefer fp32 for sign-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBES = [
    "ព្រះរាជាណាចក្រកម្ពុជា ជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។",
    "ធនាគារជាតិនៃកម្ពុជា បានប្រកាសអត្រាប្តូរប្រាក់ ៤១០៥ រៀល ក្នុងមួយដុល្លារ។",
    "柬埔寨国家银行今天公布了新的汇率：1美元兑换4105瑞尔。",
    "The National Bank of Cambodia published today's exchange rate.",
    "def monthly_payment(p, r, n):\n    return p * r / (1 - (1 + r) ** -n)",
    "The roots are $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$.",
    "| Branch | Q1 | Q2 |\n|---|---:|---:|\n| Phnom Penh | 1,200 | 1,350 |",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<tool_call>\n<function=get_rate>\n<parameter=base>\nUSD\n</parameter>\n</function>\n</tool_call>",
]


def tokenizer_parity(dsm_dir: Path, upstream_dir: Path) -> list[str]:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    raw = Tokenizer.from_file(str(upstream_dir / "tokenizer.json"))
    ours = AutoTokenizer.from_pretrained(dsm_dir)
    problems = []
    for text in PROBES:
        a = ours(text, add_special_tokens=False)["input_ids"]
        b = raw.encode(text, add_special_tokens=False).ids
        if a != b:
            problems.append(f"tokenizer mismatch on {text[:30]!r}: {len(a)} vs {len(b)} tokens")
    return problems


def logit_parity(dsm_dir: Path, upstream_dir: Path, device: str = "cpu", dtype: str = "fp32") -> dict:
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    import dsm.modeling  # noqa: F401

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    arch = json.loads((upstream_dir / "config.json").read_text(encoding="utf-8"))["architectures"][0]
    reference = getattr(transformers, arch).from_pretrained(upstream_dir, dtype=torch_dtype).to(device).eval()
    model = AutoModelForCausalLM.from_pretrained(dsm_dir, dtype=torch_dtype).to(device).eval()
    raw = Tokenizer.from_file(str(upstream_dir / "tokenizer.json"))

    worst_rel, agree, total = 0.0, 0, 0
    with torch.no_grad():
        for text in PROBES:
            ids = torch.tensor([raw.encode(text, add_special_tokens=False).ids], device=device)
            expected = reference(input_ids=ids).logits.float()
            actual = model(input_ids=ids).logits.float()
            scale = expected.abs().max().clamp_min(1e-6)
            worst_rel = max(worst_rel, ((actual - expected).abs().max() / scale).item())
            agree += (actual.argmax(-1) == expected.argmax(-1)).sum().item()
            total += ids.numel()
    return {"max_relative_logit_diff": worst_rel, "top1_agreement": agree / max(total, 1), "tokens": total}


def verify(dsm_dir: str | Path, upstream_dir: str | Path, device: str = "cpu", dtype: str = "fp32") -> dict:
    dsm_dir, upstream_dir = Path(dsm_dir), Path(upstream_dir)
    report = {"tokenizer_problems": tokenizer_parity(dsm_dir, upstream_dir)}
    report.update(logit_parity(dsm_dir, upstream_dir, device=device, dtype=dtype))
    tolerance = 1e-4 if dtype == "fp32" else 5e-2
    report["passed"] = (
        not report["tokenizer_problems"]
        and report["max_relative_logit_diff"] <= tolerance
        and report["top1_agreement"] >= (1.0 if dtype == "fp32" else 0.98)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dsm verify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsm", required=True, help="converted DSM model directory")
    parser.add_argument("--upstream", required=True, help="source checkpoint directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args(argv)
    report = verify(args.dsm, args.upstream, device=args.device, dtype=args.dtype)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
