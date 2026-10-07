"""Canonical text normalisation applied identically at training and inference time.

Inference engines such as llama.cpp do NOT apply the tokenizer's Unicode
normaliser, so applications should call :func:`normalize_text` on user input
before sending it to the model. This module only uses the standard library (and
the vendored SIL Khmer normaliser), so it can be copied into any application.

Steps:
1. Unicode NFC.
2. Remove invisible characters that are inserted inconsistently: zero-width
   space U+200B, word joiner U+2060, BOM U+FEFF, and the deprecated Khmer
   inherent vowels U+17B4 / U+17B5. ZWNJ/ZWJ (U+200C/U+200D) are kept: they
   carry meaning inside some Khmer clusters and khnormal handles them.
3. Khmer cluster re-ordering with SIL's ``khnormal`` (only when the text
   contains Khmer), e.g. ``ខែ្មរ`` -> ``ខ្មែរ``.
"""

from __future__ import annotations

import re
import unicodedata

from dsm.data.khmer.khnormal import khnormal

_INVISIBLE = re.compile("[​⁠﻿឴឵]")
_KHMER = re.compile("[ក-៿᧠-᧿]")


def contains_khmer(text: str) -> bool:
    return _KHMER.search(text) is not None


def normalize_text(text: str, *, khmer: bool = True) -> str:
    """NFC + invisible-character removal + Khmer cluster normalisation."""
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    if khmer and contains_khmer(text):
        text = khnormal(text)
    return text
