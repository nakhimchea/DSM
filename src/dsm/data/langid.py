"""Language identification for km / en / zh routing.

Script statistics decide Khmer and Chinese reliably (they own their scripts).
Latin-script text is accepted as English only with enough English function
words; everything else is ``und`` (undetermined) and dropped from the
monolingual pools. Traditional Chinese is reported as ``zh-Hant`` so it can be
excluded or converted. If ``fasttext`` and a GlotLID model are installed, pass
``glotlid=path`` for a second opinion on Latin-script text.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_KHMER = re.compile("[ក-៿᧠-᧿]")
_HAN = re.compile("[一-鿿㐀-䶿豈-﫿]")
_LATIN = re.compile("[A-Za-zÀ-ɏ]")
# Characters that exist only in Traditional Chinese (common ones).
_TRADITIONAL = set(
    "們個這說時會國學對為來開發經動關還長問題點體實現從給讓讀寫買賣錢銀號碼帳戶與後裡麼條種邊門車東機樣義聽頭見電話氣義區價務場無業務應總決證選"
)
_EN_STOPWORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "is",
        "it",
        "that",
        "for",
        "on",
        "was",
        "with",
        "as",
        "are",
        "be",
        "this",
        "by",
        "not",
        "or",
        "from",
        "at",
        "have",
        "an",
        "they",
        "which",
        "you",
        "we",
        "his",
        "her",
        "has",
        "were",
        "can",
        "will",
    ]
)


def script_shares(text: str) -> dict[str, float]:
    letters = [c for c in text if unicodedata.category(c)[0] in ("L", "M")]
    if not letters:
        return {}
    n = len(letters)
    return {
        "km": sum(1 for c in letters if _KHMER.match(c)) / n,
        "zh": sum(1 for c in letters if _HAN.match(c)) / n,
        "latin": sum(1 for c in letters if _LATIN.match(c)) / n,
    }


@lru_cache(maxsize=2)
def _glotlid(path: str):
    import fasttext  # optional dependency

    return fasttext.load_model(path)


def identify(text: str, glotlid: str | None = None) -> tuple[str, float]:
    """``(lang, confidence)`` with lang in km / zh / zh-Hant / en / und."""
    shares = script_shares(text)
    if not shares:
        return "und", 0.0
    if shares["km"] >= 0.5:
        return "km", shares["km"]
    if shares["zh"] >= 0.5:
        han = [c for c in text if _HAN.match(c)]
        traditional = sum(1 for c in han if c in _TRADITIONAL) / max(len(han), 1)
        return ("zh-Hant" if traditional > 0.03 else "zh"), shares["zh"]
    if shares["latin"] >= 0.7:
        if glotlid:
            labels, probs = _glotlid(glotlid).predict(text.replace("\n", " ")[:2000])
            label = labels[0].replace("__label__", "")
            return ("en" if label.startswith("eng") else "und"), float(probs[0])
        words = re.findall(r"[a-z']+", text.lower())
        if words:
            ratio = sum(1 for w in words if w in _EN_STOPWORDS) / len(words)
            if ratio >= 0.08:
                return "en", min(1.0, ratio * 4)
        return "und", shares["latin"]
    return "und", max(shares.values())
