"""Preference (DPO) pair construction.

Sources of pairs, in order of signal quality:

* **on-policy, judged** - several answers sampled from the SFT model itself (via
  llama-server), scored by hard checks and a local judge; best vs. worst when
  the margin is clear (:func:`judge_pairs`);
* **language consistency** - the same answer in the user's language (chosen) vs.
  in another language (rejected), from parallel / translated records;
* **format** - a valid Markdown / LaTeX answer vs. a minimally broken copy;
* **over-refusal** - a helpful answer to a benign prompt vs. a refusal;
* identity and safety pairs come from their own builders; HelpSteer3 from
  :mod:`dsm.data.builders.datasets`.

Every pair is validated with :func:`dsm.data.sft.validate.validate_preference`.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable

from dsm.data.sft.schema import preference_record
from dsm.data.sft.validate import validate_preference
from dsm.data.synth.filters import (
    degenerate_repetition,
    detect_lang,
    identity_leak,
    language_matches,
    looks_like_refusal,
)
from dsm.format.validate import check_format


def _final_answer(record: dict) -> dict | None:
    """The last assistant message if it is plain text (no tool calls)."""
    last = record["messages"][-1]
    if last["role"] != "assistant" or last.get("tool_calls") or not last.get("content", "").strip():
        return None
    return last


def _pair(
    record: dict, chosen: str, rejected: str, kind: str, generated_by: str, extra: dict | None = None
) -> dict | None:
    pair = preference_record(
        id=f"{record['id']}-pref-{kind}",
        source=record["source"],
        license=record["license"],
        generated_by=generated_by,
        prompt=record["messages"][:-1],
        chosen={"role": "assistant", "content": chosen},
        rejected={"role": "assistant", "content": rejected},
        lang=record["lang"],
        category=record["category"],
        system=record.get("system"),
        tools=record.get("tools"),
        meta={"pair_kind": kind, **(extra or {})},
    )
    return None if validate_preference(pair) else pair


# -- language consistency -------------------------------------------------------------------------


def default_group_key(record: dict) -> str:
    meta = record.get("meta") or {}
    return meta.get("translated_from") or meta.get("parallel_id") or record["id"]


def language_pairs(
    records: list[dict],
    group_key: Callable[[dict], str] = default_group_key,
    seed: int = 0,
    max_per_record: int = 1,
) -> list[dict]:
    """Answer in the user's language (chosen) vs. the parallel answer in another language (rejected).

    Records are grouped by ``group_key`` (by default the id of the record they
    were translated from); groups need at least two languages among km/en/zh.
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["lang"] in ("km", "en", "zh") and _final_answer(record):
            groups[group_key(record)].append(record)
    pairs = []
    for members in groups.values():
        for record in members:
            others = [m for m in members if m["lang"] != record["lang"]]
            for other in rng.sample(others, k=min(max_per_record, len(others))):
                pair = _pair(
                    record,
                    _final_answer(record)["content"],
                    _final_answer(other)["content"],
                    f"lang-{other['lang']}",
                    record["generated_by"],
                    {"rejected_lang": other["lang"]},
                )
                if pair:
                    pairs.append(pair)
    return pairs


# -- format ---------------------------------------------------------------------------------------

_TABLE_SEP = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$", re.M)


def corrupt_format(text: str, rng: random.Random) -> str | None:
    """A minimally broken copy of a well-formatted answer (``None`` if nothing to break)."""
    options = []
    if (match := _TABLE_SEP.search(text)) is not None:
        options.append(text[: match.start()] + text[match.end() + 1 :])  # drop the separator row
    lines = text.splitlines()
    rows = [
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("|") and line.strip().endswith("|") and "---" not in line
    ]
    if len(rows) >= 3:
        i = rng.choice(rows[1:])
        cells = lines[i].strip().strip("|").split("|")
        if len(cells) > 1:
            broken = list(lines)
            broken[i] = "| " + " | ".join(c.strip() for c in cells[:-1]) + " |"  # one cell short
            options.append("\n".join(broken))
    if "$$" in text:
        last = text.rfind("$$")
        options.append(text[:last] + text[last + 2 :])
    elif (inline := re.search(r"(?<![\\$])\$[^$\n]+\$", text)) is not None:
        options.append(text[: inline.end() - 1] + text[inline.end() :])  # unclosed inline math
    if (fence := text.rfind("```")) > 0 and text.count("```") % 2 == 0:
        options.append(text[:fence] + text[fence + 3 :])
    options = [o for o in options if o != text and check_format(o)]
    return rng.choice(options) if options else None


def format_pairs(records: list[dict], seed: int = 0) -> list[dict]:
    """Valid formatted answer (chosen) vs. a broken copy (rejected)."""
    rng = random.Random(seed)
    pairs = []
    for record in records:
        answer = _final_answer(record)
        if answer is None or check_format(answer["content"]):
            continue
        broken = corrupt_format(answer["content"], rng)
        if broken and (pair := _pair(record, answer["content"], broken, "format", record["generated_by"])):
            pairs.append(pair)
    return pairs


# -- over-refusal -----------------------------------------------------------------------------------

REFUSALS = {
    "en": ["I'm sorry, but I can't help with that request.", "I'm not able to help with that."],
    "km": ["សូមអភ័យទោស ខ្ញុំមិនអាចជួយលើសំណើនេះបានទេ។"],
    "zh": ["抱歉，我无法协助处理这个请求。", "对不起，我不能帮助你完成这个请求。"],
}


def over_refusal_pairs(records: list[dict], seed: int = 0) -> list[dict]:
    """Helpful answer to a benign prompt (chosen) vs. an unnecessary refusal (rejected).

    Use on records whose prompts are benign but sensitive-sounding (safety
    records with ``meta.action == "answer"``, XSTest-style prompts, bank FAQs).
    """
    rng = random.Random(seed)
    pairs = []
    for record in records:
        answer = _final_answer(record)
        if answer is None or record["lang"] not in REFUSALS or looks_like_refusal(answer["content"]):
            continue
        refusal = rng.choice(REFUSALS[record["lang"]])
        if pair := _pair(record, answer["content"], refusal, "over-refusal", record["generated_by"]):
            pairs.append(pair)
    return pairs


# -- on-policy sampling + judging ---------------------------------------------------------------------

JUDGE_PROMPT = """You are grading an AI assistant's answer for a bank's customer assistant.
Score from 1 (useless or harmful) to 10 (excellent) considering: correctness, helpfulness,
answering in the user's language, safety, following any system instructions, clear formatting
and concision. Reply with a one-sentence justification, then a final line "Score: N".

[Conversation]
{conversation}

[Answer]
{answer}"""
_SCORE = re.compile(r"Score:\s*(\d+(?:\.\d+)?)", re.I)


def hard_fail(answer: str, lang: str, benign: bool = True) -> str | None:
    """Reasons that make an answer the rejected one regardless of the judge."""
    if not answer.strip():
        return "empty"
    if "<think>" in answer or ("<tool_call>" in answer and "</tool_call>" not in answer):
        return "control tokens"
    if identity_leak(answer):
        return "identity leak"
    if degenerate_repetition(answer):
        return "repetition"
    if lang in ("km", "zh") and not language_matches(answer, lang, threshold=0.3):
        return "wrong language"
    if benign and looks_like_refusal(answer):
        return "over-refusal"
    if check_format(answer):
        return "broken format"
    return None


def parse_score(text: str | None) -> float | None:
    if not text:
        return None
    matches = _SCORE.findall(text)
    return float(matches[-1]) if matches else None


def sample_candidates(
    prompts: list[list[dict]], policy, n: int = 4, temperature: float = 0.8
) -> list[list[str | None]]:
    """``n`` answers per prompt from the SFT model behind an OpenAI-compatible server.

    ``policy`` is a :class:`dsm.data.synth.client.LocalTeacher` pointed at
    llama-server running the SFT GGUF; different ``seed`` values make the cached
    requests distinct.
    """
    per_seed = [policy.map(prompts, temperature=temperature, seed=s) for s in range(n)]
    return [list(answers) for answers in zip(*per_seed, strict=True)]


def judge_pairs(
    records: list[dict],
    candidates: list[list[str | None]],
    judge,
    *,
    min_margin: float = 2.0,
    generated_by: str = "on-policy",
) -> list[dict]:
    """Best vs. worst sampled answer per prompt (hard failures score 0; judge scores 1-10)."""
    requests, index = [], []
    for i, (record, answers) in enumerate(zip(records, candidates, strict=True)):
        conversation = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in record["messages"][:-1])
        for j, answer in enumerate(answers):
            if (
                answer is not None
                and hard_fail(answer, record["lang"], benign=record.get("meta", {}).get("label") != "harmful")
                is None
            ):
                requests.append(
                    [
                        {
                            "role": "user",
                            "content": JUDGE_PROMPT.format(conversation=conversation, answer=answer),
                        }
                    ]
                )
                index.append((i, j))
    verdicts = judge.map(requests, temperature=0.0) if requests else []
    scores: dict[tuple[int, int], float] = {}
    for key, verdict in zip(index, verdicts, strict=True):
        if (score := parse_score(verdict)) is not None:
            scores[key] = score
    pairs = []
    for i, (record, answers) in enumerate(zip(records, candidates, strict=True)):
        scored = [(scores.get((i, j), 0.0), a) for j, a in enumerate(answers) if a is not None]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda item: item[0])
        (low, rejected), (high, chosen) = scored[0], scored[-1]
        if high - low < min_margin or high < 6:
            continue
        lang = record["lang"] if record["lang"] != "mixed" else detect_lang(chosen)
        if pair := _pair(
            {**record, "lang": lang},
            chosen.strip(),
            rejected.strip(),
            "judged",
            generated_by,
            {"scores": [low, high]},
        ):
            pairs.append(pair)
    return pairs
