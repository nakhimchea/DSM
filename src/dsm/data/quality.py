"""Document quality filters (Gopher / C4 style rules, adapted to Khmer and Chinese).

Every rule returns a reason when it fails, so data cards can report how much
each rule removed. Thresholds are deliberately conservative for Khmer, where
good text is scarce: prefer down-weighting over dropping in the mixture.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from dsm.data.langid import script_shares

_KHMER_RUN = re.compile("[ក-៿᧠-᧿]+")
_BOILERPLATE = re.compile(
    r"lorem ipsum|javascript|enable cookies|cookie policy|all rights reserved|terms of use|{|}", re.I
)
_BULLET = re.compile(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s+")
_EN_STOPWORDS = frozenset(["the", "be", "to", "of", "and", "that", "have", "with"])


@dataclass
class QualityReport:
    keep: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def khmer_broken_rate(text: str) -> float:
    """Invalid Khmer cluster structures per 1,000 Khmer characters (SIL ``khtest``)."""
    from dsm.data.khmer.khnormal import khtest

    errors, chars = 0, 0
    for run in _KHMER_RUN.findall(text):
        chars += len(run)
        result = khtest(run)
        if result is not None:
            errors += result.count("◌") + result.count("!") // 2
    return 1000 * errors / chars if chars else 0.0


def _duplicate_fraction(items: list[str]) -> float:
    """Share of characters in items that repeat an earlier item."""
    total = sum(len(i) for i in items)
    if not total:
        return 0.0
    seen, dup = set(), 0
    for item in items:
        if item in seen:
            dup += len(item)
        seen.add(item)
    return dup / total


def quality(
    text: str, lang: str, *, min_chars: int = 200, max_chars: int = 500_000, max_broken_km: float = 15.0
) -> QualityReport:
    reasons: list[str] = []
    stats: dict[str, float] = {"chars": len(text)}
    if len(text) < min_chars:
        reasons.append("too_short")
    if len(text) > max_chars:
        reasons.append("too_long")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    stats["dup_lines"] = _duplicate_fraction(lines)
    stats["dup_paragraphs"] = _duplicate_fraction(paragraphs)
    if stats["dup_lines"] > 0.3:
        reasons.append("duplicate_lines")
    if stats["dup_paragraphs"] > 0.3:
        reasons.append("duplicate_paragraphs")
    if lines:
        stats["bullet_lines"] = sum(1 for line in lines if _BULLET.match(line)) / len(lines)
        stats["ellipsis_lines"] = sum(1 for line in lines if line.endswith(("...", "…"))) / len(lines)
        stats["boilerplate_lines"] = sum(1 for line in lines if _BOILERPLATE.search(line)) / len(lines)
        if stats["bullet_lines"] > 0.9:
            reasons.append("mostly_bullets")
        if stats["ellipsis_lines"] > 0.3:
            reasons.append("truncated_lines")
        if stats["boilerplate_lines"] > 0.3:
            reasons.append("boilerplate")
    stats["replacement_chars"] = text.count("�") / max(len(text), 1)
    if stats["replacement_chars"] > 0.001:
        reasons.append("encoding_errors")
    words = text.split()
    counts = Counter(words)
    if words:
        stats["top_word_share"] = counts.most_common(1)[0][1] / len(words)
    shares = script_shares(text)
    if lang == "km":
        stats["khmer_share"] = shares.get("km", 0.0)
        stats["khmer_broken_per_1k"] = khmer_broken_rate(text)
        if stats["khmer_share"] < 0.5:
            reasons.append("low_khmer_share")
        if stats["khmer_broken_per_1k"] > max_broken_km:
            reasons.append("broken_khmer_clusters")
    elif lang in ("zh", "zh-Hant"):
        stats["han_share"] = shares.get("zh", 0.0)
        if stats["han_share"] < 0.4:
            reasons.append("low_han_share")
        han = [c for c in text if "一" <= c <= "鿿"]
        if han:
            stats["top_char_share"] = Counter(han).most_common(1)[0][1] / len(han)
            if stats["top_char_share"] > 0.2 and len(han) > 100:
                reasons.append("repetitive_chars")
    elif lang == "en":
        if not 50 <= len(words) <= 100_000:
            reasons.append("word_count")
        if words:
            mean = sum(len(w) for w in words) / len(words)
            stats["mean_word_len"] = mean
            if not 3 <= mean <= 10:
                reasons.append("mean_word_length")
            alpha = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
            stats["alpha_words"] = alpha
            if alpha < 0.8:
                reasons.append("non_alphabetic")
            if sum(1 for w in set(w.lower() for w in words) if w in _EN_STOPWORDS) < 2:
                reasons.append("no_stopwords")
            if stats.get("top_word_share", 0) > 0.3:
                reasons.append("repetitive_words")
    return QualityReport(keep=not reasons, reasons=reasons, stats=stats)
