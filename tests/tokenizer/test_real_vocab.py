"""Checks against the real 248K-token upstream vocabulary (pinned, cached download)."""

from __future__ import annotations

import json

import pytest

from dsm.text import normalize_text
from dsm.tokenizer import write_tokenizer_files

pytestmark = pytest.mark.realvocab

SAMPLES = [
    "ធនាគារជាតិនៃកម្ពុជា បានប្រកាសអត្រាប្តូរប្រាក់ថ្មីសម្រាប់ថ្ងៃនេះ។",
    "柬埔寨国家银行今天公布了新的汇率。",
    "The National Bank of Cambodia published today's exchange rate.",
    "def rate(usd: float) -> float:\n    return usd * 4105.5",
    "$$\\frac{a}{b}$$ | A1 | =SUM(B2:B9) |",
]


def test_vocabulary_shape(real_tokenizer_dir) -> None:
    data = json.loads((real_tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    assert len(data["model"]["vocab"]) == 248_044
    added = {t["content"]: t["id"] for t in data["added_tokens"]}
    assert added["<|endoftext|>"] == 248_044
    assert added["<|im_start|>"] == 248_045
    assert added["<|im_end|>"] == 248_046
    assert added["<think>"] == 248_068 and added["</think>"] == 248_069


def test_dsm_tokenizer_class_loads_tokenizer_json_verbatim(real_tokenizer_dir, tmp_path) -> None:
    """The generic backend must not rebuild the pre-tokeniser (which would split Khmer marks)."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    write_tokenizer_files(tmp_path, real_tokenizer_dir / "tokenizer.json", "You are DSM1.0.")
    ours = AutoTokenizer.from_pretrained(tmp_path)
    raw = Tokenizer.from_file(str(real_tokenizer_dir / "tokenizer.json"))
    for text in SAMPLES:
        assert (
            ours(text, add_special_tokens=False)["input_ids"]
            == raw.encode(text, add_special_tokens=False).ids
        )
    assert ours.eos_token == "<|im_end|>" and ours.pad_token == "<|endoftext|>"
    assert ours.convert_tokens_to_ids("<|im_end|>") == 248_046


def test_khmer_fertility_is_close_to_character_level(real_tokenizer_dir) -> None:
    """Documents the main Khmer bottleneck: ~0.9 tokens per character (doc 11)."""
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(real_tokenizer_dir / "tokenizer.json"))
    khmer = normalize_text(SAMPLES[0])
    english = SAMPLES[2]
    km_per_char = len(raw.encode(khmer).ids) / len(khmer)
    en_per_char = len(raw.encode(english).ids) / len(english)
    assert km_per_char > 0.5  # almost one token per Khmer character
    assert km_per_char > 3 * en_per_char
