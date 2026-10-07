"""Inference-time text normalisation (NFC, invisible characters, Khmer cluster order)."""

from __future__ import annotations

import unicodedata

from dsm.data.khmer import khnormal, khtest
from dsm.text import contains_khmer, normalize_text


def test_khmer_cluster_reordering() -> None:
    # vowel typed before the subscript consonant -> canonical order
    assert khnormal("ខែ្មរ") == "ខ្មែរ"  # ខែ្មរ -> ខ្មែរ
    assert khtest("ខ្មែរ") is None


def test_invisible_characters_removed_but_zwnj_kept() -> None:
    text = "ខ្ញុំ​ស្រលាញ់﻿កម្ពុជា⁠"
    assert normalize_text(text) == "ខ្ញុំស្រលាញ់កម្ពុជា"
    assert "‌" in normalize_text("a‌b")


def test_nfc_and_non_khmer_untouched() -> None:
    decomposed = unicodedata.normalize("NFD", "café 你好")
    assert normalize_text(decomposed) == "café 你好"
    assert not contains_khmer("café 你好") and contains_khmer("ក")


def test_normalisation_is_idempotent() -> None:
    samples = ["ខែ្មរ​ ភាសា", "Hello", "中文", "ព្រះរាជាណាចក្រកម្ពុជា"]
    for text in samples:
        once = normalize_text(text)
        assert normalize_text(once) == once
