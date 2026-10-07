"""Document cleaning: normalise -> language ID -> quality filters -> PII redaction.

Input rows: JSON objects with a text field (raw shards from ``dsm data download``
or your own exports). Output rows are document records::

    {"id", "text", "lang", "source", "license", "generated_by", "meta": {...quality stats}}

Every drop is counted by reason; the counts go into the data card.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator

from dsm.data.langid import identify
from dsm.data.pii import redact
from dsm.data.quality import quality
from dsm.text import normalize_text

_SPACES = re.compile(r"[ \t  -   ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Latin-1 symbols that dominate Khmer text typed with legacy (Limon / ABC) fonts.
_LEGACY = re.compile("[¡-ÿ]")


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")
    lines = [
        _SPACES.sub(" ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def looks_like_legacy_khmer(text: str) -> bool:
    """Khmer typed in legacy fonts shows up as Latin-1 punctuation soup."""
    visible = [c for c in text if not c.isspace()]
    return bool(visible) and len(_LEGACY.findall(text)) / len(visible) > 0.15


def clean_records(
    rows: Iterable[dict],
    *,
    source: str,
    license: str,
    generated_by: str,
    langs: list[str],
    text_field: str = "text",
    strict_pii: bool = False,
    stats: Counter | None = None,
    min_chars: int = 200,
    keep_fields: tuple[str, ...] = (),
) -> Iterator[dict]:
    stats = stats if stats is not None else Counter()
    for index, row in enumerate(rows):
        stats["input"] += 1
        raw = row.get(text_field)
        if not isinstance(raw, str) or not raw.strip():
            stats["drop: empty"] += 1
            continue
        if "km" in langs and looks_like_legacy_khmer(raw):
            stats["drop: legacy_khmer_font"] += 1
            continue
        text = clean_text(raw)
        lang, confidence = identify(text)
        if lang not in langs:
            stats[f"drop: lang_{lang}"] += 1
            continue
        report = quality(text, lang, min_chars=min_chars)
        if not report.keep:
            for reason in report.reasons:
                stats[f"drop: {reason}"] += 1
            continue
        text, found = redact(text, strict=strict_pii)
        for label, count in found.items():
            stats[f"pii: {label}"] += count
        stats["kept"] += 1
        stats["kept_chars"] += len(text)
        yield {
            "id": f"{source}-{index:09d}",
            "text": text,
            "lang": lang,
            "source": source,
            "license": license,
            "generated_by": generated_by,
            "meta": {
                "lang_confidence": round(confidence, 3),
                **{k: round(v, 4) for k, v in report.stats.items()},
                **{k: row[k] for k in keep_fields if isinstance(row.get(k), (str, int, float, bool))},
            },
        }
