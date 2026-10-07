"""Filters (identity leak, language, refusal) and the SFT / DPO / document validators."""

from __future__ import annotations

import pytest
from tokenizer_fixtures import upstream_ids_full

from dsm.data.sft.schema import function_schema, preference_record, sft_record
from dsm.data.sft.validate import (
    record_kind,
    validate_document,
    validate_preference,
    validate_record,
    validate_sft,
)
from dsm.data.synth.filters import (
    content_hash,
    degenerate_repetition,
    detect_lang,
    foreign_identity,
    identity_leak,
    is_identity_question,
    language_matches,
    looks_like_refusal,
    script_ratio,
)


@pytest.mark.parametrize(
    "text",
    [
        "I am ChatGPT, a large language model.",
        "I'm an AI language model developed by OpenAI.",
        "As an AI assistant created by Google, I can help.",
        "I'm SmolLM, a model trained by Hugging Face.",
        "我是ChatGPT。",
        "我是由百度开发的人工智能助手。",
        "ខ្ញុំជា ChatGPT",
    ],
)
def test_identity_leak_detected(text):
    assert identity_leak(text)


def test_identity_leak_upstream_from_config():
    for phrase in upstream_ids_full()["scrub"]["self_identification"]:
        assert identity_leak(f"Hello! {phrase}.") is not None


@pytest.mark.parametrize(
    "text",
    [
        "ChatGPT was created by OpenAI.",
        "I am DSM1.0, an AI assistant developed by Doslarb.",
        "Google Maps can show you the way to the branch.",
        "我是DSM1.0，由Doslarb开发。",
        "ខ្ញុំជា DSM1.0",
    ],
)
def test_identity_leak_not_triggered(text):
    assert identity_leak(text) is None


def test_foreign_identity_in_system_prompts():
    assert foreign_identity("You are SmolLM, a helpful assistant trained by Hugging Face.")
    assert foreign_identity("你是ChatGPT，一个助手。")
    assert foreign_identity("You are a helpful assistant.") is None
    assert foreign_identity("You are an expert in composing functions.") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Who are you?", True),
        ("What's your name?", True),
        ("Who created you?", True),
        ("Are you an AI?", True),
        ("你是谁？", True),
        ("你叫什么名字", True),
        ("អ្នកជានរណា?", True),
        ("What is the capital of France?", False),
        ("Who created Python?", False),
        ("请介绍一下柬埔寨。", False),
    ],
)
def test_identity_question(text, expected):
    assert is_identity_question(text) is expected


def test_language_detection_and_script_ratio():
    assert detect_lang("សួស្តី តើអ្នកសុខសប្បាយទេ?") == "km"
    assert detect_lang("你好，请问今天的汇率是多少？") == "zh"
    assert detect_lang("Hello there") == "en"
    assert detect_lang("ទូទាត់ 100 USD តាម KHQR") == "km"  # Latin terms inside Khmer text
    assert script_ratio("abc", "zh") == 0.0
    assert language_matches("汇率是 4100 KHR", "zh", threshold=0.4)
    assert not language_matches("The rate is 4100 KHR", "km", threshold=0.3)
    assert language_matches("anything", "mixed")


def test_refusal_and_repetition():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("抱歉，我无法提供这方面的帮助。")
    assert looks_like_refusal("សូមអភ័យទោស ខ្ញុំមិនអាចជួយបានទេ។")
    assert not looks_like_refusal("Sure! Here is how to open an account.")
    assert degenerate_repetition("the same phrase again " * 40)
    assert not degenerate_repetition("A normal answer with varied words and no loops at all.")
    assert content_hash("Hello   World") == content_hash("hello world")


TOOLS = [
    function_schema("get_rate", "rate", {"base": {"type": "string"}, "days": {"type": "integer"}}, ["base"])
]


def _record(messages, **kwargs):
    return sft_record(
        **{"id": "t-1", "source": "test", "messages": messages, "lang": "en", "category": "chat", **kwargs}
    )


def test_validate_sft_accepts_tool_dialogue():
    messages = [
        {"role": "user", "content": "Rate?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_rate", "arguments": {"base": "USD"}}],
        },
        {"role": "tool", "name": "get_rate", "content": '{"rate": 4100}'},
        {"role": "assistant", "content": "4,100 KHR per USD."},
    ]
    assert validate_sft(_record(messages, tools=TOOLS, category="tools")) == []


@pytest.mark.parametrize(
    ("messages", "fragment"),
    [
        ([{"role": "assistant", "content": "hi"}], "first message"),
        ([{"role": "user", "content": "hi"}], "last message"),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
            ],
            "consecutive",
        ),
        (
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>x</think> ok"}],
            "control string",
        ),
        (
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "I am ChatGPT."}],
            "identity leak",
        ),
        ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}], "empty assistant"),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "name": "x", "content": "{}"},
                {"role": "assistant", "content": "ok"},
            ],
            "without a preceding",
        ),
        ([{"role": "user", "content": "ខែ្មរ"}, {"role": "assistant", "content": "ok"}], "not normalised"),
    ],
)
def test_validate_sft_rejects(messages, fragment):
    problems = validate_sft(_record(messages))
    assert any(fragment in p for p in problems), problems


def test_validate_tool_arguments_against_schema():
    def call(arguments, name="get_rate"):
        return [
            {"role": "user", "content": "Rate?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": arguments}]},
        ]

    assert any("missing required" in p for p in validate_sft(_record(call({"days": 2}), tools=TOOLS)))
    assert any(
        "tool call get_rate" in p
        for p in validate_sft(_record(call({"base": "USD", "days": "two"}), tools=TOOLS))
    )
    assert any(
        "undeclared tool" in p
        for p in validate_sft(_record(call({"base": "USD"}, name="other"), tools=TOOLS))
    )


def test_validate_preference_and_documents():
    pair = preference_record(
        id="p-1",
        source="test",
        prompt=[{"role": "user", "content": "Who are you?"}],
        chosen={"role": "assistant", "content": "I am DSM1.0."},
        rejected={"role": "assistant", "content": "I am ChatGPT."},  # rejected may name another identity
        lang="en",
        category="identity",
    )
    assert validate_preference(pair) == []
    assert record_kind(pair) == "preference"
    swapped = {**pair, "chosen": pair["rejected"], "rejected": pair["chosen"]}
    assert any("identity leak" in p for p in validate_preference(swapped))
    assert any("identical" in p for p in validate_preference({**pair, "rejected": pair["chosen"]}))

    doc = {
        "id": "d",
        "text": "DSM1.0 is developed by Doslarb.",
        "lang": "en",
        "source": "x",
        "license": "Apache-2.0",
    }
    assert validate_document(doc) == [] and record_kind(doc) == "document" and validate_record(doc) == []
    assert validate_document({**doc, "text": " "}) == ["empty text"]
    assert any("identity leak" in p for p in validate_document({**doc, "text": "I am ChatGPT."}))
