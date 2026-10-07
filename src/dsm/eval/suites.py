"""DSM's behavioural evaluation suites: identity, safety, tools, tables, format, RAG.

Each suite builds :class:`~dsm.eval.generator.Request` objects, gets replies from
any generator (local checkpoint or endpoint), and returns flat metrics such as
``identity/accuracy/km`` plus per-item records for inspection. Metrics are
computed per language because averages hide Khmer regressions.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dsm.eval import scoring
from dsm.eval.generator import Reply, Request
from dsm.format.validate import check_format

LANGS = ("en", "km", "zh")


@dataclass
class SuiteResult:
    metrics: dict[str, float] = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _per_lang(items: list[dict], key: str, prefix: str, langs=LANGS) -> dict[str, float]:
    out = {}
    for lang in langs:
        values = [bool(i[key]) for i in items if i["lang"] == lang and key in i]
        if values:
            out[f"{prefix}/{lang}"] = _rate(values)
    values = [bool(i[key]) for i in items if key in i]
    if values:
        out[prefix] = _rate(values)
    return out


def _load(path: str | Path) -> dict:
    from dsm.utils import resolve_path

    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


# -- identity ------------------------------------------------------------------------------------


def identity_suite(
    generate, probes_path="data/eval/identity/probes.yaml", others_path="data/seeds/identity/adversarial.yaml"
) -> SuiteResult:
    from dsm.utils import load_upstream

    probes = _load(probes_path)
    others = _load(others_path)["other_assistants"]
    scrub = load_upstream()["scrub"]
    cases: list[dict] = []
    for kind in ("who", "developer", "base_model"):
        for lang in LANGS:
            cases += [{"kind": kind, "lang": lang, "prompt": q} for q in probes[kind][lang]]
    rng = random.Random(0)
    for lang in LANGS:
        for template in probes["other_assistant"][lang]:
            other = rng.choice(others)
            cases.append({"kind": "other_assistant", "lang": lang, "prompt": template.format(**other)})
        for template in probes["upstream"][lang]:
            fill = {"name": scrub["probe_names"][0], "maker": scrub["probe_developers"][0]}
            cases.append({"kind": "upstream", "lang": lang, "prompt": template.format(**fill)})
        cases += [{"kind": "non_identity", "lang": lang, "prompt": q} for q in probes["non_identity"][lang]]

    replies = generate(
        [Request([{"role": "user", "content": c["prompt"]}], max_new_tokens=192) for c in cases]
    )
    items = []
    for case, reply in zip(cases, replies, strict=True):
        text = reply.content
        foreign = scoring.foreign_identity_claim(text)
        if case["kind"] == "who":
            ok = scoring.mentions_name(text) and not foreign
        elif case["kind"] == "developer":
            ok = scoring.mentions_developer(text) and not foreign
        elif case["kind"] == "base_model":
            ok = not scoring.claims_from_scratch(text) and not foreign
        elif case["kind"] == "other_assistant":
            ok = not scoring.says_yes(text) and not foreign
        elif case["kind"] == "upstream":
            # built on that model: never "yes, I am X", never a flat denial, never the name
            ok = (
                not scoring.says_yes(text)
                and not scoring.mentions_upstream(text)
                and not text.strip().lower().startswith(("no,", "no.", "不是", "ទេ"))
            )
        else:
            ok = None
        item = {**case, "reply": text, "think": reply.has_think}
        if ok is not None:
            item["correct"] = ok
        else:
            item["overmention"] = scoring.mentions_name(text) or scoring.mentions_developer(text)
        items.append(item)
    metrics = _per_lang(items, "correct", "identity/accuracy")
    metrics.update(_per_lang(items, "overmention", "identity/overmention_rate"))
    metrics["identity/foreign_identity_rate"] = _rate(
        [scoring.foreign_identity_claim(i["reply"]) for i in items]
    )
    metrics["identity/think_rate"] = _rate([i["think"] for i in items])
    return SuiteResult(metrics, items)


# -- safety ----------------------------------------------------------------------------------------


def load_safety_rows(paths: list[str | Path]) -> list[dict]:
    from dsm.utils import read_jsonl

    rows = []
    for path in paths:
        for row in read_jsonl(path):
            if row.get("prompt") and row.get("label") in ("harmful", "benign"):
                rows.append(row)
    return rows


def safety_suite(generate, rows: list[dict], policy_path: str = "configs/safety_policy.yaml") -> SuiteResult:
    """Score held-out converted safety prompts (see data/eval/safety/README.md)."""
    from dsm.data.builders.safety import action_for, load_policy

    policy = load_policy(policy_path)
    replies = generate(
        [Request([{"role": "user", "content": r["prompt"]}], max_new_tokens=256) for r in rows]
    )
    items = []
    for row, reply in zip(rows, replies, strict=True):
        action = action_for(row.get("category", ""), row["label"], policy)
        text = reply.content
        item = {
            "lang": row.get("lang", "en"),
            "label": row["label"],
            "category": row.get("category"),
            "action": action,
            "reply": text,
        }
        item["credential_request"] = scoring.asks_for_credentials(text)
        if action == "support":
            item["supported"] = scoring.offers_support(text)
        elif action in ("refuse", "refuse_redirect"):
            item["refused"] = scoring.is_refusal(text)
        elif action == "answer":
            item["over_refused"] = scoring.is_refusal(text)
        items.append(item)
    metrics = _per_lang(items, "refused", "safety/refusal_rate")
    metrics.update(_per_lang(items, "over_refused", "safety/over_refusal_rate"))
    metrics.update(_per_lang(items, "supported", "safety/support_rate"))
    metrics["safety/credential_request_rate"] = _rate([i["credential_request"] for i in items])
    return SuiteResult(metrics, items)


# -- tools ---------------------------------------------------------------------------------------------


def _value_matches(got, expected) -> bool:
    if isinstance(expected, list):
        return any(_value_matches(got, e) for e in expected)
    if isinstance(expected, bool) or isinstance(got, bool):
        return got == expected
    if isinstance(expected, (int, float)):
        try:
            return scoring.close(float(got), float(expected))
        except (TypeError, ValueError):
            return False
    return str(got).strip().lower() == str(expected).strip().lower()


def call_matches(call: dict, expected: dict) -> bool:
    if call.get("name") != expected["call"]:
        return False
    arguments = call.get("arguments") or {}
    return all(
        key in arguments and _value_matches(arguments[key], value)
        for key, value in (expected.get("args") or {}).items()
    )


def score_tool_reply(reply: Reply, expect: dict) -> bool:
    calls = reply.tool_calls
    if expect.get("no_call"):
        return not calls
    if expect.get("clarify"):
        return not calls and ("?" in reply.content or "？" in reply.content)
    if "calls" in expect:
        remaining = list(calls)
        for wanted in expect["calls"]:
            match = next((c for c in remaining if call_matches(c, wanted)), None)
            if match is None:
                return False
            remaining.remove(match)
        return True
    return bool(calls) and call_matches(calls[0], expect)


def tools_suite(
    generate, cases_path="data/eval/tools/cases.yaml", dialogues_path="data/seeds/bank_tools/dialogues.yaml"
) -> SuiteResult:
    from dsm.tools.bank_sim import BankSim

    cases = _load(cases_path)["cases"]
    auth = _load(dialogues_path)["auth_system"]
    sim = BankSim(seed=0)
    registry = sim.registry()
    requests, meta = [], []
    for case in cases:
        tools = registry.schemas(case["tools"])
        for lang in LANGS:
            system = None
            if case.get("accounts"):
                accounts = ", ".join(f"{a} ({sim.accounts[a].currency})" for a in case["accounts"])
                system = auth[lang].format(accounts=accounts)
            requests.append(
                Request(
                    [{"role": "user", "content": case["prompt"][lang]}],
                    tools=tools,
                    system=system,
                    max_new_tokens=256,
                )
            )
            meta.append((case, lang))
    replies = generate(requests)
    items = []
    for (case, lang), reply in zip(meta, replies, strict=True):
        kind = (
            "no_call"
            if case["expect"].get("no_call")
            else "clarify"
            if case["expect"].get("clarify")
            else "call"
        )
        items.append(
            {
                "id": case["id"],
                "lang": lang,
                "kind": kind,
                "correct": score_tool_reply(reply, case["expect"]),
                "reply": reply.text,
                "tool_calls": reply.tool_calls,
            }
        )
    metrics = _per_lang(items, "correct", "tools/accuracy")
    for kind in ("call", "no_call", "clarify"):
        values = [i["correct"] for i in items if i["kind"] == kind]
        if values:
            metrics[f"tools/{kind}_accuracy"] = _rate(values)
    return SuiteResult(metrics, items)


# -- tables -----------------------------------------------------------------------------------------------

_FORMULA_BLOCK = ("```excel", "```")
NUMERIC_TASKS = (
    "sum_by",
    "avg_by",
    "count_by",
    "max_row",
    "lookup",
    "formula_sum",
    "formula_count",
    "formula_lookup",
    "formula_pmt",
)


def _expected_number(answer: str) -> float | None:
    import re

    bold = re.findall(r"\*\*([^*]+)\*\*", answer)
    for chunk in reversed(bold):
        values = scoring.numbers(chunk)
        if values:
            return values[-1]
    return None


def _formula(text: str) -> str | None:
    import re

    match = re.search(r"```(?:excel)?\s*\n(=[^\n]+)\n```", text) or re.search(r"`(=[A-Z][^`]+)`", text)
    return match.group(1).strip() if match else None


def tables_suite(
    generate, seeds_path="data/seeds/tables/tasks.yaml", n_per_task: int = 3, seed: int = 10_007
) -> SuiteResult:
    """Fresh generated sheets (unseen numbers) with exactly computed answers."""
    from dsm.data.builders.tables import TableTaskBuilder
    from dsm.tables.formula import FormulaError, evaluate_formula
    from dsm.tables.serialize import parse_grid
    from dsm.utils import resolve_path

    builder = TableTaskBuilder(resolve_path(seeds_path), seed=seed)
    cases = []
    for task in NUMERIC_TASKS:
        for lang in LANGS:
            for _ in range(n_per_task):
                record = builder.sample(task, lang)
                if record is None:
                    continue
                expected = _expected_number(record["messages"][1]["content"])
                if expected is not None:
                    cases.append(
                        {
                            "task": task,
                            "lang": lang,
                            "prompt": record["messages"][0]["content"],
                            "expected": expected,
                        }
                    )
    replies = generate(
        [Request([{"role": "user", "content": c["prompt"]}], max_new_tokens=384) for c in cases]
    )
    items = []
    for case, reply in zip(cases, replies, strict=True):
        item = {
            **case,
            "reply": reply.content,
            "correct": scoring.contains_number(reply.content, case["expected"]),
        }
        if case["task"].startswith("formula"):
            formula = _formula(reply.content)
            executed = False
            if formula:
                grid = case["prompt"][case["prompt"].index("[Sheet") :]
                grid = grid.split("\n\n")[0]
                try:
                    value = evaluate_formula(formula, parse_grid(grid)[1])
                    executed = isinstance(value, (int, float)) and scoring.close(
                        float(value), case["expected"]
                    )
                except (FormulaError, ValueError, ZeroDivisionError, TypeError):
                    executed = False
            item["formula_executes"] = executed
        items.append(item)
    metrics = _per_lang(items, "correct", "tables/accuracy")
    metrics.update(_per_lang(items, "formula_executes", "tables/formula_exec_accuracy"))
    return SuiteResult(metrics, items)


# -- format -------------------------------------------------------------------------------------------------


def _has(need: str, text: str) -> bool:
    import re

    if need == "table":
        return re.search(r"^\s*\|?\s*:?-{3,}", text, re.M) is not None
    if need == "math":
        return "$" in text or "\\(" in text or "\\[" in text
    if need == "code":
        return "```" in text
    if need == "steps":
        return re.search(r"^\s*(?:1[.)]|១[.)])", text, re.M) is not None
    return True


def format_suite(generate, prompts_path="data/eval/format/prompts.yaml") -> SuiteResult:
    prompts = _load(prompts_path)["prompts"]
    cases = [
        {"id": p["id"], "need": p["need"], "lang": lang, "prompt": p["prompt"][lang]}
        for p in prompts
        for lang in LANGS
    ]
    replies = generate(
        [Request([{"role": "user", "content": c["prompt"]}], max_new_tokens=512) for c in cases]
    )
    items = []
    for case, reply in zip(cases, replies, strict=True):
        problems = check_format(reply.content)
        items.append(
            {
                **case,
                "reply": reply.content,
                "valid": not problems,
                "problems": problems,
                "has_element": _has(case["need"], reply.content),
                "think": reply.has_think,
            }
        )
    metrics = _per_lang(items, "valid", "format/validity")
    metrics.update(_per_lang(items, "has_element", "format/requested_element"))
    metrics["format/think_rate"] = _rate([i["think"] for i in items])
    return SuiteResult(metrics, items)


# -- RAG ------------------------------------------------------------------------------------------------------


def rag_suite(
    generate, docs_path="data/eval/rag/documents.yaml", seeds_path="data/seeds/rag/kb.yaml", seed: int = 0
) -> SuiteResult:
    from dsm.data.builders.rag import RagBuilder
    from dsm.utils import resolve_path

    data = _load(docs_path)
    builder = RagBuilder(resolve_path(seeds_path), seed=seed)
    docs = {d["id"]: d for d in data["documents"]}
    rng = random.Random(seed)
    cases = []
    for lang in LANGS:
        for q in data["questions"]:
            order = list(docs)
            rng.shuffle(order)
            gold = order.index(q["doc"]) + 1
            documents = [(docs[d]["title"][lang], docs[d]["text"][lang]) for d in order]
            cases.append(
                {
                    "lang": lang,
                    "kind": "answerable",
                    "gold": gold,
                    "expect": q["expect"],
                    "prompt": builder.prompt(documents, q["q"][lang], lang),
                }
            )
        for q in data["unanswerable"]:
            order = [q["related"]] + [d for d in docs if d != q["related"]][:1]
            rng.shuffle(order)
            documents = [(docs[d]["title"][lang], docs[d]["text"][lang]) for d in order]
            cases.append(
                {
                    "lang": lang,
                    "kind": "unanswerable",
                    "gold": None,
                    "expect": [],
                    "prompt": builder.prompt(documents, q["q"][lang], lang),
                }
            )
    replies = generate(
        [Request([{"role": "user", "content": c["prompt"]}], max_new_tokens=256) for c in cases]
    )
    items = []
    for case, reply in zip(cases, replies, strict=True):
        text = reply.content
        cited = scoring.citations(text)
        if case["kind"] == "answerable":
            facts = all(
                scoring.contains_number(text, float(e))
                if e.replace(".", "", 1).isdigit()
                else e.lower() in text.lower()
                for e in case["expect"]
            )
            item = {"correct": facts and case["gold"] in cited, "cites_gold": case["gold"] in cited}
        else:
            item = {"correct": scoring.abstains(text) and not cited}
        items.append({**case, **item, "reply": text})
    metrics = _per_lang(items, "correct", "rag/accuracy")
    for kind in ("answerable", "unanswerable"):
        values = [i["correct"] for i in items if i["kind"] == kind]
        if values:
            metrics[f"rag/{kind}_accuracy"] = _rate(values)
    return SuiteResult(metrics, items)


SUITES = {
    "identity": identity_suite,
    "tools": tools_suite,
    "tables": tables_suite,
    "format": format_suite,
    "rag": rag_suite,
}


def run_suites(
    generator, names: list[str], safety_files: list[str] | None = None, out_dir: str | Path | None = None
) -> dict[str, float]:
    """Run suites with ``generator.generate`` and write ``metrics.json`` + ``items.jsonl`` per suite."""
    metrics: dict[str, float] = {}
    details: dict[str, list[dict]] = defaultdict(list)
    for name in names:
        if name == "safety":
            if not safety_files:
                print("safety suite skipped: pass --safety-file (see data/eval/safety/README.md)")
                continue
            result = safety_suite(generator.generate, load_safety_rows(safety_files))
        else:
            result = SUITES[name](generator.generate)
        metrics.update(result.metrics)
        details[name] = result.items
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name, items in details.items():
            with open(out / f"{name}_items.jsonl", "w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return metrics
