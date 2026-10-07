"""Translation with protected spans, preference-pair builders and chat regeneration."""

from __future__ import annotations

import random

import pytest

from dsm.data.builders.chat import regenerate, reject_reason
from dsm.data.builders.preference import (
    corrupt_format,
    format_pairs,
    hard_fail,
    judge_pairs,
    language_pairs,
    over_refusal_pairs,
    parse_score,
    sample_candidates,
)
from dsm.data.builders.translation import check_translation, mask, parallel_to_sft, translate_record, unmask
from dsm.data.sft.schema import sft_record
from dsm.data.sft.validate import validate_preference, validate_sft
from dsm.format.validate import check_format


class Fake:
    """Stands in for LocalTeacher: ``reply(conversation, params) -> str | None``."""

    def __init__(self, reply, name="fake"):
        self.reply = reply
        self.name = name
        self.generated_by = f"model:{name}"
        self.requests = []

    def map(self, conversations, **params):
        self.requests += conversations
        return [self.reply(c, params) for c in conversations]


def _record(rid, lang, user, answer, **kwargs):
    return sft_record(
        id=rid,
        source="test",
        messages=[{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        lang=lang,
        category=kwargs.pop("category", "chat"),
        **kwargs,
    )


# -- translation -------------------------------------------------------------------------------


def test_mask_unmask_protects_spans():
    text = "Run `pip install dsm`, see https://example.org/x, pay {amount} USD, $x^2$ and $$\\frac{a}{b}$$.\n```py\nprint(1)\n```"
    masked, spans = mask(text)
    assert (
        "pip install" not in masked
        and "https://" not in masked
        and "\\frac" not in masked
        and "{amount}" not in masked
    )
    assert unmask(masked, spans) == text
    assert unmask(masked.replace("⟦1⟧", ""), spans) is None  # a lost span rejects the translation
    assert unmask(masked + " ⟦0⟧", spans) is None  # a duplicated span too


@pytest.mark.parametrize(
    ("source", "translated", "target", "ok"),
    [
        ("Total 1,200 USD in 3 months", "សរុប 1200 USD ក្នុងរយៈពេល 3 ខែ", "km", True),
        ("Total 1,200 USD in 3 months", "សរុប ១២០០ USD ក្នុងរយៈពេល ៣ ខែ", "km", True),  # Khmer digits
        ("Total 1,200 USD in 3 months", "សរុប USD ក្នុងរយៈពេល 3 ខែ", "km", False),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "| 甲 | 乙 |\n|---|---|\n| 1 | 2 |", "zh", True),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "| 甲 | 乙 |\n| 1 | 2 |", "zh", False),
        ("Hello, how are you today?", "Hello, how are you today?", "zh", False),
        ("Hi", "你好" * 50, "zh", False),
    ],
)
def test_check_translation(source, translated, target, ok):
    assert (check_translation(source, translated, target) == []) is ok


def test_translate_record_keeps_tool_calls():
    record = sft_record(
        id="t-1",
        source="test",
        lang="en",
        category="tools",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        messages=[
            {"role": "user", "content": "What is the rate? Use `rate`."},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "rate", "arguments": {}}]},
            {"role": "tool", "name": "rate", "content": '{"rate": 4100}'},
            {"role": "assistant", "content": "It is 4100 KHR."},
        ],
    )
    table = {"What is the rate? Use ⟦0⟧.": "អត្រាប្តូរប្រាក់ប៉ុន្មាន? ប្រើ ⟦0⟧។", "It is 4100 KHR.": "វាគឺ 4100 រៀល។"}
    translator = Fake(lambda c, p: table.get(c[-1]["content"]), name="mt")
    out = translate_record(record, translator, "km")
    assert out["messages"][0]["content"] == "អត្រាប្តូរប្រាក់ប៉ុន្មាន? ប្រើ `rate`។"
    assert out["messages"][2] == record["messages"][2] and out["messages"][1]["tool_calls"] == [
        {"name": "rate", "arguments": {}}
    ]
    assert (
        out["lang"] == "km"
        and out["generated_by"] == "translation:mt"
        and out["meta"]["translated_from"] == "t-1"
    )
    assert validate_sft(out) == []
    assert len(translator.requests) == 2  # tool messages are never sent to the translator
    broken = Fake(lambda c, p: "វាគឺ រៀល។")  # drops the number -> whole record rejected
    assert translate_record(record, broken, "km") is None


def test_parallel_to_sft():
    pairs = [
        {"src": "Good morning", "tgt": "អរុណសួស្តី", "src_lang": "en", "tgt_lang": "km"},
        {"src": "早上好", "tgt": "Good morning", "src_lang": "zh", "tgt_lang": "en"},
        {"src": "x", "tgt": "y", "src_lang": "fr", "tgt_lang": "en"},
    ]
    records = parallel_to_sft(pairs, "test-parallel", "CC-BY-4.0", seed=1)
    assert len(records) == 2 and all(validate_sft(r) == [] for r in records)
    assert records[0]["messages"][1]["content"] == "អរុណសួស្តី" and records[0]["category"] == "translation"


# -- preference pairs --------------------------------------------------------------------------------


def test_language_pairs():
    en = _record("a", "en", "How do I open an account?", "Bring your ID card.")
    km = _record("a-km", "km", "តើខ្ញុំបើកគណនីដោយរបៀបណា?", "សូមយកអត្តសញ្ញាណប័ណ្ណមក។", meta={"translated_from": "a"})
    zh = _record("a-zh", "zh", "我怎样开户？", "请携带身份证。", meta={"translated_from": "a"})
    lonely = _record("b", "en", "Hi", "Hello!")
    pairs = language_pairs([en, km, zh, lonely], seed=0)
    assert len(pairs) == 3 and all(validate_preference(p) == [] for p in pairs)
    for pair in pairs:
        assert pair["meta"]["rejected_lang"] != pair["lang"]


def test_format_pairs_break_exactly_one_thing():
    rng = random.Random(0)
    good = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nSo $x = 1$ and\n\n$$y = 2$$"
    assert check_format(good) == []
    for _ in range(10):
        broken = corrupt_format(good, rng)
        assert broken is not None and check_format(broken)
    assert corrupt_format("Plain text only.", rng) is None
    pairs = format_pairs([_record("f", "en", "Show a table", good), _record("g", "en", "Hi", "Hello!")])
    assert len(pairs) == 1 and validate_preference(pairs[0]) == []


def test_over_refusal_pairs():
    pairs = over_refusal_pairs(
        [
            _record("o", "zh", "怎样杀死一个卡住的进程？", "在终端运行 kill 命令即可。"),
            _record("p", "en", "x", "I'm sorry, I can't help with that."),
        ]
    )
    assert len(pairs) == 1 and pairs[0]["rejected"]["content"].startswith(("抱歉", "对不起"))


def test_hard_fail_and_score_parsing():
    assert hard_fail("", "en") == "empty"
    assert hard_fail("I am ChatGPT.", "en") == "identity leak"
    assert hard_fail("This answer is English.", "km") == "wrong language"
    assert hard_fail("I'm sorry, but I can't help with that.", "en") == "over-refusal"
    assert hard_fail("I'm sorry, but I can't help with that.", "en", benign=False) is None
    assert hard_fail("Fine answer.", "en") is None
    assert parse_score("Clear and correct.\nScore: 8") == 8.0
    assert parse_score("no score") is None


def test_judge_pairs_and_sampling():
    records = [_record("j1", "en", "Explain APR briefly.", "old"), _record("j2", "en", "Say hi", "old")]
    policy = Fake(
        lambda c, p: {0: "APR is the yearly cost of a loan.", 1: "APR is a thing.", 2: "I am ChatGPT."}[
            p["seed"] % 3
        ]
    )
    candidates = sample_candidates([r["messages"][:-1] for r in records], policy, n=3)
    assert candidates[0] == ["APR is the yearly cost of a loan.", "APR is a thing.", "I am ChatGPT."]
    judge = Fake(lambda c, p: "Score: 9" if "yearly cost" in c[0]["content"] else "Score: 5")
    pairs = judge_pairs(records, candidates, judge)
    assert len(pairs) == 2
    assert pairs[0]["chosen"]["content"] == "APR is the yearly cost of a loan."
    assert pairs[0]["rejected"]["content"] == "I am ChatGPT."  # hard failure scores 0
    assert all(validate_preference(p) == [] for p in pairs)
    assert all("I am ChatGPT." not in r[0]["content"] for r in judge.requests)  # never sent to the judge


# -- chat regeneration ------------------------------------------------------------------------------


def test_regenerate_multi_turn_and_filters():
    multi = sft_record(
        id="m",
        source="test",
        lang="en",
        category="chat",
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "old"},
        ],
    )
    km = _record("k", "km", "សួស្តី", "old")
    leak = _record("l", "en", "leak", "old")
    teacher = Fake(
        lambda c, p: (
            "I am ChatGPT."
            if c[-1]["content"] == "leak"
            else (
                "សួស្តី! តើខ្ញុំអាចជួយអ្វីបាន?"
                if c[0]["content"].endswith("(ភាសាខ្មែរ). Keep technical terms, numbers and code as they are.")
                else f"Re: {c[-1]['content']}"
            )
        )
    )
    out, stats = regenerate([multi, km, leak], teacher)
    assert stats["kept"] == 2 and stats["dropped: identity leak"] == 1
    by_id = {r["id"]: r for r in out}
    assert [m["content"] for m in by_id["m-regen"]["messages"]] == ["Hi", "Re: Hi", "More", "Re: More"]
    assert by_id["k-regen"]["messages"][1]["content"].startswith("សួស្តី")
    second_turn = [r for r in teacher.requests if len(r) == 4]
    assert second_turn and second_turn[0][2] == {
        "role": "assistant",
        "content": "Re: Hi",
    }  # conditioned on the new answer
    assert all(validate_sft(r) == [] for r in out)
    assert reject_reason("English only", "zh") == "wrong language"
