"""Translation of SFT data into Khmer / Chinese with protected spans, plus translation tasks.

Machine translation must not touch code, math, URLs, JSON, tool names or
placeholders. Before a text is sent to the (local) translator, such spans are
replaced by sentinels ``⟦0⟧, ⟦1⟧ ...``; after translation they are restored and
the sample is rejected unless:

* every sentinel comes back exactly once,
* all numbers of the source appear in the translation,
* Markdown tables keep their row / cell counts,
* the target script dominates (Khmer / Han) and the length ratio is plausible.

Tool calls (structured) and tool results (JSON) are never translated.
"""

from __future__ import annotations

import re
from collections import Counter

from dsm.data.synth.filters import language_matches
from dsm.text import normalize_text

LANG_NAMES = {"km": "Khmer", "zh": "Simplified Chinese", "en": "English"}
_PROTECT = re.compile(
    r"```.*?```"  # code blocks
    r"|`[^`\n]+`"  # inline code
    r"|\$\$.+?\$\$"  # display math
    r"|\\\[.+?\\\]|\\\(.+?\\\)"  # \[..\] \(..\)
    r"|(?<![\\$])\$[^$\n]+?\$"  # inline math
    r"|https?://\S+|www\.\S+"  # URLs
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"  # e-mail addresses
    r"|\{[a-zA-Z_][a-zA-Z0-9_]*\}",  # {placeholders}
    re.S,
)
_SENTINEL = re.compile(r"⟦(\d+)⟧")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")  # Khmer digits count as the same number


def mask(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def repl(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"⟦{len(spans) - 1}⟧"

    return _PROTECT.sub(repl, text), spans


def unmask(text: str, spans: list[str]) -> str | None:
    found = [int(i) for i in _SENTINEL.findall(text)]
    if sorted(found) != list(range(len(spans))):
        return None
    return _SENTINEL.sub(lambda m: spans[int(m.group(1))], text)


def _numbers(text: str) -> Counter:
    return Counter(n.replace(",", "") for n in _NUMBER.findall(text.translate(_DIGITS)))


def _table_shape(text: str) -> list[int]:
    return [line.count("|") for line in text.splitlines() if line.strip().startswith("|")]


def check_translation(source: str, translated: str, target: str) -> list[str]:
    problems = []
    if _numbers(source) - _numbers(translated):
        problems.append("numbers changed or dropped")
    if _table_shape(source) != _table_shape(translated):
        problems.append("table structure changed")
    if target in ("km", "zh") and not language_matches(_PROTECT.sub(" ", translated), target, threshold=0.4):
        problems.append(f"translation is not mainly in {LANG_NAMES[target]}")
    ratio = (len(translated) + 1) / (len(source) + 1)
    if not 0.25 <= ratio <= 4.0:
        problems.append(f"implausible length ratio {ratio:.2f}")
    return problems


def translation_prompt(text: str, target: str) -> list[dict]:
    system = (
        f"You are a professional translator. Translate the user's text into {LANG_NAMES[target]}. "
        "Keep every placeholder such as ⟦0⟧ exactly as it is, keep Markdown formatting, numbers and line breaks, "
        "and output only the translation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def translate_texts(texts: list[str], translator, target: str) -> list[str | None]:
    """Translate many texts; ``None`` marks a rejected translation."""
    masked = [mask(t) for t in texts]
    outputs = translator.map([translation_prompt(m, target) for m, _ in masked])
    results: list[str | None] = []
    for (_, spans), source, output in zip(masked, texts, outputs, strict=True):
        if output is None:
            results.append(None)
            continue
        restored = unmask(output.strip(), spans)
        if restored is None or check_translation(source, restored, target):
            results.append(None)
            continue
        results.append(normalize_text(restored))
    return results


def translate_record(record: dict, translator, target: str) -> dict | None:
    """Translate the natural-language parts of an SFT record (tool calls / results untouched)."""
    texts, slots = [], []
    if record.get("system"):
        texts.append(record["system"])
        slots.append(("system", None))
    for i, message in enumerate(record["messages"]):
        if message["role"] in ("user", "assistant") and (message.get("content") or "").strip():
            texts.append(message["content"])
            slots.append(("message", i))
    translated = translate_texts(texts, translator, target)
    if any(t is None for t in translated):
        return None
    out = {**record, "messages": [dict(m) for m in record["messages"]]}
    for (kind, index), text in zip(slots, translated, strict=True):
        if kind == "system":
            out["system"] = text
        else:
            out["messages"][index]["content"] = text
    out["id"] = f"{record['id']}-{target}"
    out["lang"] = target
    out["generated_by"] = f"translation:{translator.name}"
    out["meta"] = {
        **record.get("meta", {}),
        "translated_from": record["id"],
        "source_lang": record.get("lang"),
    }
    return out


TASK_TEMPLATES = {
    ("en", "km"): {
        "en": "Translate into Khmer:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាខ្មែរ៖\n\n{text}",
        "zh": "请翻译成高棉语：\n\n{text}",
    },
    ("km", "en"): {
        "en": "Translate into English:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាអង់គ្លេស៖\n\n{text}",
        "zh": "请翻译成英语：\n\n{text}",
    },
    ("zh", "km"): {
        "en": "Translate into Khmer:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាខ្មែរ៖\n\n{text}",
        "zh": "请翻译成高棉语：\n\n{text}",
    },
    ("km", "zh"): {
        "en": "Translate into Chinese:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាចិន៖\n\n{text}",
        "zh": "请翻译成中文：\n\n{text}",
    },
    ("en", "zh"): {
        "en": "Translate into Chinese:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាចិន៖\n\n{text}",
        "zh": "请翻译成中文：\n\n{text}",
    },
    ("zh", "en"): {
        "en": "Translate into English:\n\n{text}",
        "km": "សូមបកប្រែជាភាសាអង់គ្លេស៖\n\n{text}",
        "zh": "请翻译成英语：\n\n{text}",
    },
}


def parallel_to_sft(pairs: list[dict], source: str, license: str, seed: int = 0) -> list[dict]:
    """Instruction data from parallel sentences ``{"src", "tgt", "src_lang", "tgt_lang"}``."""
    import random

    from dsm.data.sft.schema import sft_record

    rng = random.Random(seed)
    records = []
    for i, pair in enumerate(pairs):
        key = (pair["src_lang"], pair["tgt_lang"])
        if key not in TASK_TEMPLATES:
            continue
        instruction_lang = rng.choice([pair["src_lang"], pair["tgt_lang"]])
        user = TASK_TEMPLATES[key][instruction_lang].format(text=normalize_text(pair["src"]).strip())
        records.append(
            sft_record(
                id=f"translation-{source}-{i:07d}",
                source=source,
                license=license,
                generated_by=f"dataset:{source}",
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": normalize_text(pair["tgt"]).strip()},
                ],
                lang="mixed",
                category="translation",
                meta={"direction": f"{pair['src_lang']}->{pair['tgt_lang']}"},
            )
        )
    return records
