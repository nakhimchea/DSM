"""Stage gates: compare evaluation metrics with thresholds (``configs/eval/gates.yaml``).

A gate rule::

    {metric: "identity/accuracy/{lang}", min: 0.98}           # absolute bound, per language
    {metric: "bpb/en", max_increase: 0.03}                     # relative to a baseline run
    {metric: "tools/accuracy", min_vs_baseline: 0.0}           # no worse than baseline (+ margin)

``{lang}`` expands to km / en / zh. Missing metrics are reported as SKIP (the
suite was not run), unless ``required: true``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LANGS = ("km", "en", "zh")


@dataclass
class GateResult:
    name: str
    metric: str
    status: str  # PASS | FAIL | SKIP
    detail: str


def _expand(rule: dict) -> list[tuple[str, dict]]:
    metric = rule["metric"]
    if "{lang}" in metric:
        return [(metric.format(lang=lang), rule) for lang in rule.get("langs", LANGS)]
    return [(metric, rule)]


def check_rule(name: str, metric: str, rule: dict, results: dict, baseline: dict | None) -> GateResult:
    value = results.get(metric)
    if value is None or value != value:  # missing or NaN
        status = "FAIL" if rule.get("required") else "SKIP"
        return GateResult(name, metric, status, "metric not available")
    problems = []
    if "min" in rule and value < rule["min"]:
        problems.append(f"{value:.4f} < min {rule['min']}")
    if "max" in rule and value > rule["max"]:
        problems.append(f"{value:.4f} > max {rule['max']}")
    needs_base = any(
        k in rule for k in ("max_increase", "min_vs_baseline", "max_vs_baseline", "must_decrease")
    )
    if needs_base:
        base = (baseline or {}).get(metric)
        if base is None:
            return GateResult(
                name,
                metric,
                "SKIP" if not rule.get("required") else "FAIL",
                f"{value:.4f}; no baseline value",
            )
        if "max_increase" in rule and value > base * (1 + rule["max_increase"]):
            problems.append(f"{value:.4f} > baseline {base:.4f} + {rule['max_increase']:.0%}")
        if "min_vs_baseline" in rule and value < base + rule["min_vs_baseline"]:
            problems.append(f"{value:.4f} < baseline {base:.4f} + {rule['min_vs_baseline']}")
        if "max_vs_baseline" in rule and value > base + rule["max_vs_baseline"]:
            problems.append(f"{value:.4f} > baseline {base:.4f} + {rule['max_vs_baseline']}")
        if rule.get("must_decrease") and not value < base:
            problems.append(f"{value:.4f} did not decrease from baseline {base:.4f}")
    return GateResult(name, metric, "FAIL" if problems else "PASS", "; ".join(problems) or f"{value:.4f}")


def evaluate_gates(
    stage: str,
    results: dict,
    baseline: dict | None = None,
    gates_path: str | Path = "configs/eval/gates.yaml",
) -> list[GateResult]:
    from dsm.utils import load_yaml

    config = load_yaml(gates_path)
    if stage not in config:
        raise ValueError(f"unknown stage {stage!r}; known: {sorted(config)}")
    out = []
    for name, rule in config[stage].items():
        for metric, expanded in _expand(rule):
            out.append(check_rule(name, metric, expanded, results, baseline))
    return out


def load_results(paths: list[str | Path]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for path in paths:
        merged.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return merged
