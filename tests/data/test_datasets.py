"""Public dataset converters (rows in the formats of the original releases)."""

from __future__ import annotations

import pytest

from dsm.data.builders.datasets import (
    DATASETS,
    MassiveConverter,
    convert_dataset,
    pythonic_calls,
    run_finqa_program,
    tool_from_spec,
)
from dsm.data.sft.schema import template_messages
from dsm.tokenizer.render import build_template, render
from dsm.tools.toolcall import parse_tool_calls

from . import dataset_rows as R

TEMPLATE = build_template("You are DSM1.0, an AI assistant developed by Doslarb.")


def _convert(name, rows, **kwargs):
    records, stats = convert_dataset(name, rows, license=kwargs.pop("license", "test"), **kwargs)
    return records, stats


def test_registry_and_licence_requirement():
    assert {
        "gsm8k",
        "xlam",
        "hermes_fc",
        "glaive_fc_v2",
        "toolace",
        "tabfact",
        "wtq",
        "tatqa",
        "finqa",
        "massive",
        "helpsteer3",
    } <= set(DATASETS)
    with pytest.raises(ValueError, match="licence"):
        convert_dataset("chat", [R.CHAT])
    records, _ = convert_dataset("gsm8k", [R.GSM8K])
    assert records[0]["license"] == "MIT" and records[0]["generated_by"] == "dataset:gsm8k"


def test_gsm8k():
    records, stats = _convert("gsm8k", [R.GSM8K, R.GSM8K_MONEY, {"question": "q", "answer": "no final"}])
    assert stats["kept"] == 2 and stats["dropped"] == 1
    answer = records[0]["messages"][1]["content"]
    assert "<<" not in answer and answer.endswith("**Answer: 72**")
    money = records[1]["messages"][1]["content"]
    assert "$" not in money and "18 USD" in money


def test_numinamath_filters():
    records, stats = _convert(
        "numinamath",
        [R.NUMINA, R.NUMINA_PROOF, R.NUMINA_BROKEN_LATEX, {**R.NUMINA, "solution_is_valid": "No"}],
    )
    assert stats["kept"] == 1 and stats["dropped"] == 3
    assert records[0]["category"] == "math" and "\\boxed{4}" in records[0]["messages"][1]["content"]


def test_tool_schema_dialects():
    xlam = tool_from_spec(
        {
            "name": "f",
            "description": "d",
            "parameters": {
                "a": {"type": "str"},
                "b": {"type": "List[int], optional"},
                "c": {"type": "float", "default": 1.0},
            },
        }
    )
    params = xlam["function"]["parameters"]
    assert params["properties"]["a"]["type"] == "string" and params["properties"]["b"]["type"] == "array"
    assert params["properties"]["c"]["type"] == "number" and params["required"] == ["a"]
    toolace = tool_from_spec(
        {
            "name": "g",
            "parameters": {
                "type": "dict",
                "properties": {"x": {"type": "dict"}, "y": {"type": "tuple", "required": True}},
                "required": None,
            },
        }
    )
    assert toolace["function"]["parameters"] == {
        "type": "object",
        "properties": {"x": {"type": "object"}, "y": {"type": "array"}},
        "required": ["y"],
    }
    assert tool_from_spec({"description": "no name"}) is None


def test_pythonic_calls():
    assert pythonic_calls('[f(a=1, b="x"), mod.g(flag=true, items=[1, 2], opt=null)]') == [
        {"name": "f", "arguments": {"a": 1, "b": "x"}},
        {"name": "mod.g", "arguments": {"flag": True, "items": [1, 2], "opt": None}},
    ]
    assert pythonic_calls("[f(1)]") is None  # positional arguments are ambiguous
    assert pythonic_calls("not a call") is None


@pytest.mark.parametrize(
    ("name", "row", "calls"),
    [
        (
            "xlam",
            R.XLAM,
            [("live_giveaways_by_type", {"type": "beta"}), ("live_giveaways_by_type", {"type": "game"})],
        ),
        ("hermes_fc", R.HERMES, [("get_stock_price", {"symbol": "AAPL"})]),
        ("glaive_fc_v2", R.GLAIVE, [("calculate_bmi", {"weight": 70, "height": 1.75})]),
        ("toolace", R.TOOLACE, [("newAddress", {"cryptocurrency": "Ethereum", "count": 1})]),
        ("chat", R.CHAT_OPENAI_TOOLS, [("get_weather", {"city": "Phnom Penh"})]),
        ("chat", R.CHAT_NV_TOOLCALL, [("get_weather", {"city": "Siem Reap"})]),
    ],
)
def test_tool_datasets_round_trip_through_template(name, row, calls):
    records, stats = _convert(name, [row])
    assert stats["kept"] == 1, stats
    record = records[0]
    assert record["category"] == "tools" and record["tools"]
    structured = [(c["name"], c["arguments"]) for m in record["messages"] for c in m.get("tool_calls") or []]
    assert structured == calls
    text = render(template_messages(record), TEMPLATE, tools=record["tools"])
    _, parsed = parse_tool_calls(text[text.index("<|im_start|>assistant") :], record["tools"])
    assert [(c.name, c.arguments) for c in parsed] == calls
    tool_messages = [m for m in record["messages"] if m["role"] == "tool"]
    assert all(m["name"] == calls[0][0] for m in tool_messages)


def test_hermes_unwraps_tool_envelope():
    records, _ = _convert("hermes_fc", [R.HERMES])
    tool = next(m for m in records[0]["messages"] if m["role"] == "tool")
    assert tool["content"] == '{"price": 189.5}'


def test_chat_filters():
    records, stats = _convert(
        "chat", [R.CHAT, R.CHAT_REASONING, R.CHAT_IDENTITY, R.CHAT_LEAK, R.GLAIVE_NO_TOOLS]
    )
    assert stats["kept"] == 1 and stats["dropped"] + stats["malformed"] == 4
    record = records[0]
    assert record["system"] is None  # foreign identity in the system prompt -> DSM default prompt
    assert record["messages"][1]["content"].startswith("1. Track")  # empty think block stripped
    assert record["meta"]["source"] == "smoltalk-everyday"
    glaive, _ = _convert("glaive_fc_v2", [R.GLAIVE_NO_TOOLS])
    assert glaive == []


def test_chat_drops_upstream_model_mentions():
    from tokenizer_fixtures import upstream_ids_full

    name = upstream_ids_full()["scrub"]["probe_names"][0]
    row = {
        "messages": [
            {"role": "user", "content": "Tell me about LLMs."},
            {"role": "assistant", "content": f"{name} is one of them."},
        ]
    }
    records, _ = _convert("chat", [row])
    assert records == []


def test_tables_datasets():
    tabfact, _ = _convert("tabfact", [R.TABFACT])
    assert tabfact[0]["messages"][1]["content"].startswith("**False.**")
    assert '[Sheet "loans by branch"' in tabfact[0]["messages"][0]["content"]
    wtq, _ = _convert("wtq", [R.WTQ])
    assert wtq[0]["messages"][1]["content"] == "Phnom Penh"
    tatqa, stats = _convert("tatqa", [R.TATQA])
    assert stats["kept"] == 3 and [r["id"] for r in tatqa] == [
        "tatqa-0000000-0",
        "tatqa-0000000-1",
        "tatqa-0000000-2",
    ]
    assert tatqa[0]["messages"][1]["content"] == "(1,496.5-1,437.5)/1,437.5 = 4.1%\n\n**Answer:** 4.1%"
    assert tatqa[2]["messages"][1]["content"] == "**Answer:** 900 million"


def test_finqa_program_execution():
    value, steps = run_finqa_program("subtract(5829, 5735), divide(#0, 5735)", R.FINQA["table"])
    assert value == pytest.approx(0.016390, abs=1e-5) and steps[0] == "5,829 − 5,735 = 94"
    value, steps = run_finqa_program("table_max(net sales, none), multiply(#0, const_100)", R.FINQA["table"])
    assert value == 582900 and "maximum" in steps[0]
    assert run_finqa_program("divide(1, 0)", []) is None
    assert run_finqa_program("table_sum(unknown row, none)", R.FINQA["table"]) is None
    records, stats = _convert("finqa", [R.FINQA, R.FINQA_TABLE_OP, R.FINQA_WRONG])
    assert stats["kept"] == 2 and stats["dropped"] == 1  # the wrong gold answer does not re-compute
    assert records[0]["messages"][1]["content"].endswith("**Answer:** 1.6%")


def test_massive_khmer_function_calls():
    converter = MassiveConverter(R.MASSIVE_ROWS)
    assert [t["function"]["name"] for t in converter.tools["alarm"]] == ["alarm_query", "alarm_set"]
    records, stats = _convert("massive", R.MASSIVE_ROWS)
    assert stats["kept"] == 3 and stats["dropped"] == 1  # zh-TW is not a DSM language
    km = records[0]
    assert km["lang"] == "km" and km["messages"][1]["tool_calls"] == [
        {"name": "alarm_set", "arguments": {"time": "ម៉ោងប្រាំព្រឹក"}}
    ]
    ints = MassiveConverter([{**R.MASSIVE_ROWS[0], "intent": 1}], intent_names=["alarm_query", "alarm_set"])
    assert ints({**R.MASSIVE_ROWS[0], "intent": 1}).messages[1]["tool_calls"][0]["name"] == "alarm_set"


def test_helpsteer3_pairs():
    records, stats = _convert(
        "helpsteer3", [R.HELPSTEER3, R.HELPSTEER3_TIE, {**R.HELPSTEER3, "overall_preference": 3}]
    )
    assert stats["kept"] == 2 and stats["dropped"] == 1
    assert (
        records[0]["chosen"]["content"].startswith("Compound interest")
        and records[1]["chosen"]["content"] == "It is money."
    )
    assert all("prompt" in r and r["prompt"][-1]["role"] == "user" for r in records)


def test_malformed_rows_are_counted_not_raised():
    records, stats = _convert(
        "xlam", [{"query": "q", "tools": "not json", "answers": "[]"}, {"unexpected": 1}]
    )
    assert records == [] and stats["dropped"] + stats["malformed"] == 2
