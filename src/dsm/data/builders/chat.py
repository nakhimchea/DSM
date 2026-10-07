"""General chat data: response regeneration with a local teacher.

Public prompt collections are valuable, but their responses come from many
models (with mixed licences, identities and styles). ``regenerate`` keeps the
user turns and rewrites every assistant turn with the local teacher, turn by
turn, under DSM's house style. A rewritten conversation is kept only if every
turn passes the filters:

* no identity leak (another assistant's name / developer),
* written in the user's language (Khmer / Chinese prompts get Khmer / Chinese),
* no refusal of an ordinary request, no looping output,
* valid Markdown / LaTeX.

Records keep the prompt licence in ``license`` and record the teacher in
``generated_by``. Khmer prompts are best produced by translating English /
Chinese prompts first (:mod:`dsm.data.builders.translation`) and then
regenerating, so the answer is native rather than translated.
"""

from __future__ import annotations

from collections import Counter

from dsm.data.synth.filters import (
    degenerate_repetition,
    detect_lang,
    identity_leak,
    language_matches,
    looks_like_refusal,
)
from dsm.format.validate import check_format
from dsm.text import normalize_text

STYLE_PROMPT = {
    "en": "Answer in English.",
    "km": "Answer in Khmer (ភាសាខ្មែរ). Keep technical terms, numbers and code as they are.",
    "zh": "Answer in Simplified Chinese.",
}
HOUSE_STYLE = (
    "You are a helpful, accurate assistant. Answer directly without showing hidden reasoning. "
    "Be concise; use Markdown lists or tables when they help, LaTeX with $...$ or $$...$$ for math, "
    "fenced code blocks with a language tag for code, and write money with currency codes (100 USD, 4,100 KHR). "
    "Never mention any AI model, assistant name or company that made you."
)


def teacher_system(lang: str) -> str:
    return f"{HOUSE_STYLE} {STYLE_PROMPT.get(lang, STYLE_PROMPT['en'])}"


def reject_reason(answer: str | None, lang: str) -> str | None:
    if answer is None or not answer.strip():
        return "no answer"
    if identity_leak(answer):
        return "identity leak"
    if lang in ("km", "zh") and not language_matches(answer, lang, threshold=0.3):
        return "wrong language"
    if looks_like_refusal(answer):
        return "refusal"
    if degenerate_repetition(answer):
        return "repetition"
    if check_format(answer):
        return "broken format"
    return None


def _finish(item: dict, teacher) -> dict:
    record = item["record"]
    messages = []
    for user, answer in zip(item["users"], item["out"], strict=True):
        messages += [
            {"role": "user", "content": normalize_text(user)},
            {"role": "assistant", "content": answer},
        ]
    return {
        **record,
        "id": f"{record['id']}-regen",
        "messages": messages,
        "lang": item["lang"],
        "generated_by": teacher.generated_by,
        "meta": {**(record.get("meta") or {}), "regenerated_from": record["id"]},
    }


def regenerate(records: list[dict], teacher, *, max_turns: int = 4) -> tuple[list[dict], Counter]:
    """Rewrite all assistant turns of plain chat records (records with tool calls are skipped).

    Turn ``k`` of every conversation is generated in one batch, conditioned on
    the already rewritten turns ``< k``.
    """
    stats: Counter = Counter()
    active = []
    for record in records:
        messages = record["messages"]
        if any(m["role"] == "tool" or m.get("tool_calls") for m in messages):
            stats["skipped: tools"] += 1
            continue
        users = [m["content"] for m in messages if m["role"] == "user"][:max_turns]
        lang = record["lang"] if record["lang"] in ("en", "km", "zh") else detect_lang(users[0])
        system = record.get("system") or teacher_system(lang)
        active.append({"record": record, "users": users, "lang": lang, "system": system, "out": []})

    done = []
    while active:
        requests = []
        for item in active:
            history = [{"role": "system", "content": item["system"]}]
            for user, answer in zip(item["users"], item["out"], strict=False):
                history += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
            history.append({"role": "user", "content": item["users"][len(item["out"])]})
            requests.append(history)
        still = []
        for item, answer in zip(active, teacher.map(requests), strict=True):
            if reason := reject_reason(answer, item["lang"]):
                stats[f"dropped: {reason}"] += 1
                continue
            item["out"].append(normalize_text(answer.strip()))
            if len(item["out"]) < len(item["users"]):
                still.append(item)
            else:
                done.append(_finish(item, teacher))
                stats["kept"] += 1
        active = still
    return done, stats
