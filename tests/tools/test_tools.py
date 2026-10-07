"""Tool-call text format, the bank simulator and the tool-dialogue builder."""

from __future__ import annotations

import json

import pytest

from dsm import repo_root
from dsm.data.builders.tools import ToolDialogueBuilder
from dsm.data.sft.schema import template_messages
from dsm.data.sft.validate import validate_sft
from dsm.tokenizer.render import ASSISTANT_HEADER, IM_END, build_template, render
from dsm.tools.bank_sim import BankSim
from dsm.tools.calculator import evaluate
from dsm.tools.toolcall import ToolCall, coerce, parse_tool_calls, render_tool_calls

SEEDS = repo_root() / "data/seeds/bank_tools/dialogues.yaml"
TEMPLATE = build_template("You are DSM1.0, an AI assistant developed by Doslarb.")


def test_render_parse_round_trip_with_schema():
    sim = BankSim(seed=0)
    tools = sim.registry().schemas()
    calls = [
        ToolCall("convert_currency", {"amount": 100.5, "from_currency": "USD", "to_currency": "KHR"}),
        ToolCall("calculate_loan_payment", {"principal": 5000, "annual_rate": 12, "months": 24}),
        ToolCall("search_knowledge_base", {"query": "លេខកូដ OTP", "lang": "km"}),
    ]
    text = render_tool_calls(calls, content="Let me check.")
    content, parsed = parse_tool_calls("Let me check." + text, tools)
    assert content == "Let me check."
    assert [c.name for c in parsed] == [c.name for c in calls]
    assert [c.arguments for c in parsed] == [c.arguments for c in calls]
    assert isinstance(parsed[1].arguments["months"], int)


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        ("42", "integer", 42),
        ("4.5", "number", 4.5),
        ("7", "number", 7),
        ("True", "boolean", True),
        ("false", "boolean", False),
        ('{"a": 1}', "object", {"a": 1}),
        ("[1, 2]", "array", [1, 2]),
        ("007", "string", "007"),
        ("12", None, 12),
        ("hello", None, "hello"),
        ("abc", "integer", "abc"),
    ],
)
def test_coerce(raw, kind, expected):
    assert coerce(raw, kind) == expected


def test_template_renders_tool_calls_identically():
    """The XML the template renders for a structured call is what ``render_tool_calls`` produces."""
    record = ToolDialogueBuilder(SEEDS, seed=3).rate("en")
    text = render(template_messages(record), TEMPLATE, tools=record["tools"])
    call = record["messages"][1]["tool_calls"]
    assert render_tool_calls(call) in text
    start = text.index(ASSISTANT_HEADER, text.index("<|im_start|>user")) + len(ASSISTANT_HEADER)
    turn = text[start : text.index(IM_END, start)]
    _, parsed = parse_tool_calls(turn, record["tools"])
    assert [p.as_message_call() for p in parsed] == call


def test_bank_sim_tools():
    sim = BankSim(seed=1)
    registry = sim.registry()
    rate = json.loads(registry.call("get_exchange_rate", {"base": "USD", "quote": "KHR"}))
    assert 3500 < rate["rate"] < 4700 and rate["date"] == "2026-10-01"
    converted = json.loads(
        registry.call("convert_currency", {"amount": 10, "from_currency": "USD", "to_currency": "KHR"})
    )
    assert converted["converted"] == pytest.approx(10 * converted["rate"], rel=1e-6)
    loan = json.loads(
        registry.call("calculate_loan_payment", {"principal": 12000, "annual_rate": 12, "months": 12})
    )
    assert loan["monthly_payment"] == pytest.approx(1066.19, abs=0.01)
    pending = json.loads(
        registry.call(
            "create_transfer",
            {"from_account": "ACC-1001", "to_account": "ACC-1002", "amount": 5, "currency": "USD"},
        )
    )
    assert pending["status"] == "confirmation_required"
    assert registry["create_transfer"].side_effect and registry["block_card"].side_effect
    assert "error" in json.loads(registry.call("get_account_balance", {"account_id": "ACC-9999"}))
    assert "error" in json.loads(registry.call("get_exchange_rate", {"base": "USD"}))  # missing argument
    assert "error" in json.loads(registry.call("nope", {}))
    assert "error" in json.loads(registry.call("block_card", {"card_last4": "12", "reason": "lost"}))


def test_calculator_is_safe():
    assert evaluate("2 + 3 * 4") == 14
    assert evaluate("1,200 × 3") == 3600
    assert evaluate("sqrt(16) + 2^3") == 12
    for bad in ("__import__('os')", "open('x')", "2 ** 1000", "1/0", "a + 1"):
        with pytest.raises(ValueError):
            evaluate(bad)


def test_tool_dialogues_valid_and_replayable():
    builder = ToolDialogueBuilder(SEEDS, seed=0)
    records = builder.build(per_weight=2)
    assert records and all(validate_sft(r) == [] for r in records)
    scenarios = {r["meta"]["scenario"] for r in records}
    assert {
        "rate",
        "transfer_confirm",
        "parallel",
        "no_call",
        "clarify",
        "tool_error",
        "kb_injection",
        "scam_refusal",
    } <= scenarios
    assert {r["lang"] for r in records} == {"en", "km", "zh"}

    # Every tool message is exactly what the simulator returns for the preceding call (fresh replay).
    registry = BankSim(seed=0).registry()
    replayed = 0
    for record in ToolDialogueBuilder(SEEDS, seed=0).build(per_weight=2):
        pending = []
        for message in record["messages"]:
            if message.get("tool_calls"):
                pending = list(message["tool_calls"])
            elif message["role"] == "tool":
                call = pending.pop(0)
                assert message["name"] == call["name"]
                recorded = json.loads(message["content"])
                expected = json.loads(registry.call(call["name"], call["arguments"]))
                if (
                    record["meta"]["scenario"] == "kb_injection"
                ):  # the builder appends an injected instruction
                    assert recorded["results"][0]["text"].startswith(expected["results"][0]["text"])
                    assert recorded["results"][0]["text"] != expected["results"][0]["text"]
                else:
                    assert recorded == expected
                replayed += 1
    assert replayed > 50


def test_money_moving_tools_need_confirmation():
    for record in ToolDialogueBuilder(SEEDS, seed=1).build(per_weight=2):
        if record["meta"]["scenario"] != "transfer_confirm":
            continue
        calls = [
            c for m in record["messages"] for c in m.get("tool_calls") or [] if c["name"] == "create_transfer"
        ]
        assert calls[0]["arguments"].get("confirmed", False) is False
        users = [i for i, m in enumerate(record["messages"]) if m["role"] == "user"]
        confirmed = [
            i
            for i, m in enumerate(record["messages"])
            for c in m.get("tool_calls") or []
            if c["arguments"].get("confirmed")
        ]
        assert all(i > users[1] for i in confirmed)  # only after the user's confirmation turn
