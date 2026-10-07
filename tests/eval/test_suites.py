"""Evaluation suites with scripted generators (a well-behaved and a misbehaving model)."""

from __future__ import annotations

import json
import re

import pytest

from dsm import repo_root
from dsm.eval import suites
from dsm.eval.generator import Reply, Request, reply_from_text
from dsm.tools.toolcall import render_tool_calls


class Scripted:
    """Generator stand-in: ``answer(request) -> Reply | str``."""

    def __init__(self, answer):
        self.answer = answer
        self.requests: list[Request] = []

    def generate(self, requests):
        self.requests += requests
        out = []
        for request in requests:
            reply = self.answer(request)
            out.append(reply if isinstance(reply, Reply) else reply_from_text(reply, request.tools))
        return out


def _user(request: Request) -> str:
    return request.messages[-1]["content"]


# -- identity -------------------------------------------------------------------------------------


def test_identity_suite_good_and_bad():
    probes = suites._load("data/eval/identity/probes.yaml")
    ordinary = {q for lang in probes["non_identity"].values() for q in lang}

    def good(request):
        if _user(request) in ordinary:
            return "Here is a short, direct answer."
        return (
            "I'm DSM1.0, an AI assistant developed by Doslarb. I'm built on an open-source foundation model."
        )

    result = suites.identity_suite(Scripted(good).generate)
    for lang in ("en", "km", "zh"):
        assert result.metrics[f"identity/accuracy/{lang}"] == 1.0
    assert (
        result.metrics["identity/overmention_rate"] == 0.0
        and result.metrics["identity/foreign_identity_rate"] == 0.0
    )

    bad = suites.identity_suite(Scripted(lambda r: "Yes, I am ChatGPT, made by OpenAI.").generate)
    assert bad.metrics["identity/accuracy"] == 0.0 and bad.metrics["identity/foreign_identity_rate"] == 1.0


def test_identity_upstream_probes_come_from_config():
    from tokenizer_fixtures import upstream_ids_full

    name = upstream_ids_full()["scrub"]["probe_names"][0]
    generator = Scripted(lambda r: "I'm DSM1.0, developed by Doslarb.")
    suites.identity_suite(generator.generate)
    assert any(name in _user(r) for r in generator.requests)
    assert name not in (repo_root() / "data/eval/identity/probes.yaml").read_text(encoding="utf-8")

    def denies(request):
        return "No, I'm DSM1.0 by Doslarb." if name in _user(request) else "I'm DSM1.0, developed by Doslarb."

    result = suites.identity_suite(Scripted(denies).generate)
    denied = [i for i in result.items if i["kind"] == "upstream" and name in i["prompt"]]
    assert denied and not any(i["correct"] for i in denied)  # a flat denial is not truthful


# -- safety (neutral placeholder prompts; real files are converted locally) ------------------------


def test_safety_suite_scoring(tmp_path):
    rows = [
        {"prompt": "placeholder request A", "label": "harmful", "category": "fraud_scams", "lang": "en"},
        {"prompt": "placeholder request B", "label": "harmful", "category": "self_harm", "lang": "en"},
        {"prompt": "placeholder question C", "label": "benign", "category": "benign_sensitive", "lang": "zh"},
    ]
    path = tmp_path / "safety.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = suites.load_safety_rows([path])

    def good(request):
        text = _user(request)
        if text.endswith("A"):
            return "I can't help with that, but I can explain how to protect yourself."
        if text.endswith("B"):
            return "I'm really sorry you're going through this. Please call 119 or talk to someone you trust."
        return "当然可以，这是答案。"

    metrics = suites.safety_suite(Scripted(good).generate, loaded).metrics
    assert metrics["safety/refusal_rate/en"] == 1.0 and metrics["safety/support_rate/en"] == 1.0
    assert metrics["safety/over_refusal_rate/zh"] == 0.0 and metrics["safety/credential_request_rate"] == 0.0

    def bad(request):
        return (
            "抱歉，我无法提供帮助。"
            if _user(request).endswith("C")
            else "Sure. First, please send me your OTP."
        )

    metrics = suites.safety_suite(Scripted(bad).generate, loaded).metrics
    assert metrics["safety/refusal_rate/en"] == 0.0 and metrics["safety/over_refusal_rate/zh"] == 1.0
    assert metrics["safety/credential_request_rate"] > 0


# -- tools ------------------------------------------------------------------------------------------


def test_tool_matching_rules():
    expect = {"call": "find_branches", "args": {"city": ["Kampot", "កំពត", "贡布"]}}
    assert suites.call_matches({"name": "find_branches", "arguments": {"city": "kampot"}}, expect)
    assert suites.call_matches({"name": "find_branches", "arguments": {"city": "贡布"}}, expect)
    assert not suites.call_matches({"name": "find_branches", "arguments": {"city": "Siem Reap"}}, expect)
    assert suites.call_matches(
        {"name": "x", "arguments": {"amount": "250.0"}}, {"call": "x", "args": {"amount": 250}}
    )
    parallel = {"calls": [{"call": "a"}, {"call": "b"}]}
    assert suites.score_tool_reply(
        Reply("", "", [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]), parallel
    )
    assert not suites.score_tool_reply(Reply("", "", [{"name": "a", "arguments": {}}]), parallel)
    assert suites.score_tool_reply(
        Reply("Which currency do you have?", "Which currency do you have?"), {"clarify": True}
    )
    assert not suites.score_tool_reply(Reply("", "", [{"name": "a", "arguments": {}}]), {"no_call": True})


def test_tools_suite_with_oracle():
    cases = suites._load("data/eval/tools/cases.yaml")["cases"]
    by_prompt = {case["prompt"][lang]: case for case in cases for lang in ("en", "km", "zh")}

    def oracle(request):
        expect = by_prompt[_user(request)]["expect"]
        if expect.get("no_call"):
            return "Interest is the cost of borrowing money."
        if expect.get("clarify"):
            return "Sure - which currency and how much would you like to convert?"
        wanted = expect.get("calls", [expect])
        calls = [
            {
                "name": w["call"],
                "arguments": {
                    k: (v[0] if isinstance(v, list) else v) for k, v in (w.get("args") or {}).items()
                },
            }
            for w in wanted
        ]
        return render_tool_calls(calls)

    generator = Scripted(oracle)
    result = suites.tools_suite(generator.generate)
    assert result.metrics["tools/accuracy"] == 1.0 and set(result.metrics) >= {
        "tools/accuracy/km",
        "tools/no_call_accuracy",
        "tools/clarify_accuracy",
    }
    balance = [r for r in generator.requests if r.system and "ACC-1002" in r.system]
    assert len(balance) == 3  # logged-in session prompt in every language
    lazy = suites.tools_suite(Scripted(lambda r: "I don't know.").generate)
    assert lazy.metrics["tools/accuracy"] < 0.5


# -- tables -----------------------------------------------------------------------------------------


def test_tables_suite_with_oracle():
    from dsm.data.builders.tables import TableTaskBuilder

    reference = TableTaskBuilder(repo_root() / "data/seeds/tables/tasks.yaml", seed=10_007)
    answers = {}
    for task in suites.NUMERIC_TASKS:
        for lang in ("en", "km", "zh"):
            for _ in range(2):
                record = reference.sample(task, lang)
                if record is not None:
                    answers[record["messages"][0]["content"]] = record["messages"][1]["content"]
    result = suites.tables_suite(Scripted(lambda r: answers[_user(r)]).generate, n_per_task=2)
    assert result.metrics["tables/accuracy"] == 1.0
    assert result.metrics["tables/formula_exec_accuracy"] == 1.0
    wrong = suites.tables_suite(Scripted(lambda r: "The answer is **-1**.").generate, n_per_task=1)
    assert wrong.metrics["tables/accuracy"] == 0.0


# -- format and RAG ---------------------------------------------------------------------------------


GOOD_FORMAT = {
    "table": "| A | B |\n|---|---|\n| 1 | 2 |",
    "math": "Interest is $I = P r t$, so $I = 1000 \\times 0.05 \\times 2 = 100$ USD.",
    "code": "```python\ndef usd_to_khr(amount, rate):\n    return amount * rate\n```",
    "steps": "1. Open the app.\n2. Choose the card.\n3. Tap Activate.",
}


def test_format_suite():
    prompts = suites._load("data/eval/format/prompts.yaml")["prompts"]
    need = {p["prompt"][lang]: p["need"] for p in prompts for lang in ("en", "km", "zh")}
    good = suites.format_suite(Scripted(lambda r: GOOD_FORMAT[need[_user(r)]]).generate)
    assert good.metrics["format/validity"] == 1.0 and good.metrics["format/requested_element"] == 1.0
    assert good.metrics["format/think_rate"] == 0.0
    broken = suites.format_suite(
        Scripted(lambda r: "<think>hmm</think> | A | B |\n|---|\n| 1 | 2 | 3 |\nIt costs $5.").generate
    )
    assert broken.metrics["format/validity"] == 0.0 and broken.metrics["format/think_rate"] == 1.0


def test_rag_suite():
    data = suites._load("data/eval/rag/documents.yaml")
    docs = {d["id"]: d for d in data["documents"]}
    gold = {
        q["q"][lang]: (docs[q["doc"]]["title"][lang], q["expect"][0])
        for q in data["questions"]
        for lang in ("en", "km", "zh")
    }

    def oracle(request):
        text = _user(request)
        question = next((q for q in gold if text.endswith(q)), None)
        if question is None:
            return "The provided documents don't contain this information."
        title, fact = gold[question]
        index = re.search(rf"\[(\d+)\] {re.escape(title)}\n", text).group(1)
        return f"The answer is {fact} USD [{index}]."

    result = suites.rag_suite(Scripted(oracle).generate)
    assert result.metrics["rag/unanswerable_accuracy"] == 1.0
    assert result.metrics["rag/answerable_accuracy"] == 1.0
    guesser = suites.rag_suite(Scripted(lambda r: "It is 10 USD [1].").generate)
    assert guesser.metrics["rag/unanswerable_accuracy"] == 0.0


def test_run_suites_writes_files(tmp_path):
    generator = Scripted(lambda r: "I'm DSM1.0, developed by Doslarb.")
    metrics = suites.run_suites(generator, ["identity", "format"], out_dir=tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text()) == pytest.approx(metrics, nan_ok=True)
    assert (tmp_path / "identity_items.jsonl").is_file() and (tmp_path / "format_items.jsonl").is_file()
    assert "safety/refusal_rate" not in suites.run_suites(generator, ["safety"])  # skipped without files
