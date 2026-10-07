"""Decontamination: drop training documents that contain evaluation text.

Index every evaluation string (benchmarks you evaluate on and DSM's held-out
suites) as hashed n-grams - 13 words for English, 25 characters for Khmer /
Chinese (whitespace removed) - and drop any training document that shares
one. Evaluation strings shorter than one n-gram are too short to be evidence
of contamination and are skipped. Run it after deduplication and before packing; the data card records how
many documents it removed per source.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WORD_N = 13
CHAR_N = 25


def _key(gram: str) -> bytes:
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def _grams(text: str, lang: str) -> list[str]:
    if lang in ("km", "zh", "zh-Hant"):
        chars = "".join(text.split())
        return [chars[i : i + CHAR_N] for i in range(0, max(0, len(chars) - CHAR_N + 1))]
    words = re.findall(r"\w+", text.lower())
    return [" ".join(words[i : i + WORD_N]) for i in range(0, max(0, len(words) - WORD_N + 1))]


def _strings(value, min_len: int = 30):
    if isinstance(value, str):
        if len(value) >= min_len:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item, min_len)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, min_len)


class Decontaminator:
    def __init__(self) -> None:
        self.index: set[bytes] = set()

    def add_text(self, text: str) -> None:
        from dsm.data.langid import identify

        lang = identify(text)[0]
        for gram in _grams(text, lang if lang != "und" else "en"):
            self.index.add(_key(gram))

    def add_file(self, path: str | Path) -> int:
        """Index every long string of a JSON / JSONL / YAML / text file; returns how many."""
        import yaml

        from dsm.utils import read_jsonl

        path = Path(path)
        name = path.name.lower()
        if name.endswith((".jsonl", ".jsonl.zst")):
            values = list(read_jsonl(path))
        elif name.endswith(".json"):
            values = json.loads(path.read_text(encoding="utf-8"))
        elif name.endswith((".yaml", ".yml")):
            values = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            values = [p for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        count = 0
        for text in _strings(values):
            self.add_text(text)
            count += 1
        return count

    def contaminated(self, text: str, lang: str) -> bool:
        grams = _grams(text, lang)
        if lang in ("km", "zh", "zh-Hant"):  # English inside Khmer / Chinese documents (parallel text)
            grams += _grams(text, "en")
        return any(_key(gram) in self.index for gram in grams)
