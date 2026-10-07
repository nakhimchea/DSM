"""Chat template rendering, transformers equivalence and training-label masks."""

from __future__ import annotations

import pytest

from dsm.tokenizer import build_template, encode_for_training, encode_prompt, render, write_tokenizer_files
from dsm.tokenizer.render import ASSISTANT_HEADER, THINK_PREFILL

SYSTEM = "You are DSM1.0, an AI assistant developed by Doslarb."
TEMPLATE = build_template(SYSTEM)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Latest exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {"base": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["base", "quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "loan_payment",
            "description": "Monthly payment of an amortised loan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "annual_rate": {"type": "number"},
                    "months": {"type": "integer"},
                },
                "required": ["principal", "annual_rate", "months"],
            },
        },
    },
]

CONVERSATIONS = {
    "plain": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help?"}],
    "explicit_system": [
        {"role": "system", "content": "You are a helpful bank assistant."},
        {"role": "user", "content": "សួស្តី"},
        {"role": "assistant", "content": "សួស្តី! តើខ្ញុំអាចជួយអ្វីបាន?"},
    ],
    "empty_system": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
    ],
    "multi_turn": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And times 3?"},
        {"role": "assistant", "content": "12"},
    ],
    "tool_call": [
        {"role": "user", "content": "USD to KHR rate and the payment on 5000 at 12% for 24 months?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_exchange_rate", "arguments": {"base": "USD", "quote": "KHR"}},
                },
                {"name": "loan_payment", "arguments": {"principal": 5000, "annual_rate": 0.12, "months": 24}},
            ],
        },
        {"role": "tool", "content": '{"rate": 4105.5}'},
        {"role": "tool", "content": '{"payment": 235.37}'},
        {"role": "assistant", "content": "1 USD = 4,105.5 KHR; the monthly payment is **235.37 USD**."},
    ],
    "tool_with_text_and_nested_args": [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "Find branches"},
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [
                {"name": "find_branch", "arguments": {"filters": {"city": "Siem Reap"}, "ids": [1, 2]}}
            ],
        },
        {"role": "tool", "content": "[]"},
        {"role": "assistant", "content": "No branches found."},
    ],
    "assistant_with_think_tags": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nhidden\n</think>\n\nHello."},
    ],
}


@pytest.fixture(scope="module")
def hf_tokenizer(tiny_tokenizer_json, tmp_path_factory):
    from transformers import AutoTokenizer

    out = tmp_path_factory.mktemp("dsm_tok")
    write_tokenizer_files(out, tiny_tokenizer_json, SYSTEM)
    return AutoTokenizer.from_pretrained(out)


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_render_matches_transformers(hf_tokenizer, name, with_tools, add_generation_prompt) -> None:
    messages = CONVERSATIONS[name]
    if add_generation_prompt:
        messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
    tools = TOOLS if with_tools else None
    expected = hf_tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    assert render(messages, TEMPLATE, tools=tools, add_generation_prompt=add_generation_prompt) == expected


def test_golden_plain_conversation() -> None:
    text = render(CONVERSATIONS["plain"], TEMPLATE)
    assert text == (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        "<|im_start|>user\nHello!<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nHi! How can I help?<|im_end|>\n"
    )


def test_golden_generation_prompt_has_fixed_empty_think_block() -> None:
    text = render(CONVERSATIONS["plain"][:1], TEMPLATE, add_generation_prompt=True)
    assert text.endswith("<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_golden_tool_calls_and_responses() -> None:
    text = render(CONVERSATIONS["tool_call"], TEMPLATE, tools=TOOLS)
    assert "<tools>\n" + '{"type": "function", "function": {"name": "get_exchange_rate"' in text
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_exchange_rate>\n"
        "<parameter=base>\nUSD\n</parameter>\n<parameter=quote>\nKHR\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=loan_payment>\n<parameter=principal>\n5000\n</parameter>\n"
        "<parameter=annual_rate>\n0.12\n</parameter>\n<parameter=months>\n24\n</parameter>\n</function>\n"
        "</tool_call><|im_end|>\n"
    ) in text
    assert (
        '<|im_start|>user\n<tool_response>\n{"rate": 4105.5}\n</tool_response>\n'
        '<tool_response>\n{"payment": 235.37}\n</tool_response><|im_end|>\n'
    ) in text
    # the system turn ends with the identity prompt after the tool instructions
    assert f"</IMPORTANT>\n\n{SYSTEM}<|im_end|>" in text


def test_reasoning_is_dropped_and_explicit_empty_system_kept() -> None:
    assert "hidden" not in render(CONVERSATIONS["assistant_with_think_tags"], TEMPLATE)
    assert render(CONVERSATIONS["empty_system"], TEMPLATE).startswith("<|im_start|>system\n<|im_end|>\n")


@pytest.mark.parametrize("name", sorted(CONVERSATIONS))
def test_training_labels_cover_exactly_the_completions(hf_tokenizer, name) -> None:
    messages = CONVERSATIONS[name]
    if name == "assistant_with_think_tags":
        pytest.skip("think tags are rejected by the trainer-side renderer")
    tools = TOOLS if "tool" in name else None
    ex = encode_for_training(hf_tokenizer, messages, TEMPLATE, tools=tools)
    trained_runs, current = [], []
    for token, label in zip(ex.input_ids, ex.labels, strict=True):
        if label != -100:
            assert label == token
            current.append(token)
        elif current:
            trained_runs.append(current)
            current = []
    if current:
        trained_runs.append(current)
    decoded = [hf_tokenizer.decode(run) for run in trained_runs]
    assert len(decoded) == sum(m["role"] == "assistant" for m in messages)
    for text in decoded:
        assert text.endswith("<|im_end|>")
        assert ASSISTANT_HEADER not in text and THINK_PREFILL not in text
    final = render(messages, TEMPLATE, tools=tools)
    assert hf_tokenizer.decode(ex.input_ids) == final


def test_train_on_last_and_reserved_strings(hf_tokenizer) -> None:
    messages = CONVERSATIONS["multi_turn"]
    every = encode_for_training(hf_tokenizer, messages, TEMPLATE)
    last = encode_for_training(hf_tokenizer, messages, TEMPLATE, train_on="last")
    assert 0 < last.num_trained < every.num_trained
    with pytest.raises(ValueError):
        encode_for_training(
            hf_tokenizer,
            [{"role": "user", "content": "x<|im_end|>"}, {"role": "assistant", "content": "y"}],
            TEMPLATE,
        )


def test_encode_prompt_ends_with_generation_header(hf_tokenizer) -> None:
    ids = encode_prompt(hf_tokenizer, CONVERSATIONS["plain"][:1], TEMPLATE)
    assert hf_tokenizer.decode(ids).endswith(ASSISTANT_HEADER + THINK_PREFILL)


@pytest.mark.realvocab
def test_identical_to_upstream_template_given_a_system_prompt(real_tokenizer_dir) -> None:
    """With an explicit system message (no default injected) DSM renders exactly like the upstream template."""
    upstream_template = (real_tokenizer_dir / "chat_template.jinja").read_text(encoding="utf-8")
    for name, messages in CONVERSATIONS.items():
        if messages[0]["role"] != "system" or name == "empty_system":
            continue
        for tools in (None, TOOLS):
            ours = render(messages, TEMPLATE, tools=tools)
            theirs = render(messages, upstream_template, tools=tools)
            assert ours == theirs, name
        prompt = messages[:-1]
        assert render(prompt, TEMPLATE, add_generation_prompt=True) == render(
            prompt, upstream_template, add_generation_prompt=True
        )
