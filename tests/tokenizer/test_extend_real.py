"""The Khmer extension on the real 248K-token vocabulary (pinned, cached download)."""

from __future__ import annotations

import json

import pytest

from dsm.text import normalize_text
from dsm.tokenizer import write_tokenizer_files
from dsm.tokenizer.extend import apply_extension, encode_with_extension, khmer_piece, learn, next_free_id

from .test_extend import OTHER_TEXTS, split_texts

pytestmark = pytest.mark.realvocab


@pytest.fixture(scope="module")
def real_extension(real_tokenizer_dir):
    from tokenizers import Tokenizer

    original = json.loads((real_tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    old = Tokenizer.from_file(str(real_tokenizer_dir / "tokenizer.json"))
    train, held = split_texts()
    extension = learn(old, train, n_new=512, start_id=next_free_id(original))
    data = apply_extension(original, extension)
    return old, Tokenizer.from_str(json.dumps(data)), extension, held, data


def test_control_tokens_keep_their_ids(real_extension):
    old, new, extension, _, _ = real_extension
    assert extension.start_id == 248_070  # right after </think> & co; reuses the padding rows
    assert len(extension.tokens) == 512 and all(khmer_piece(t) for t in extension.tokens)
    for token, index in {"<|endoftext|>": 248_044, "<|im_start|>": 248_045, "<|im_end|>": 248_046}.items():
        assert new.token_to_id(token) == index == old.token_to_id(token)
    assert new.token_to_id("</think>") == 248_069
    assert new.get_vocab_size() == 248_070 + 512


def test_non_khmer_text_is_unchanged(real_extension):
    old, new, *_ = real_extension
    for text in OTHER_TEXTS:
        assert new.encode(text).ids == old.encode(text).ids, text


def test_khmer_matches_the_learner_and_gets_shorter(real_extension):
    old, new, extension, held, _ = real_extension
    before = after = 0
    for text in held:
        text = normalize_text(text)
        ids = new.encode(text, add_special_tokens=False).ids
        assert ids == encode_with_extension(old, extension, text)
        assert new.decode(ids) == text
        before += len(old.encode(text, add_special_tokens=False).ids)
        after += len(ids)
    assert after < 0.6 * before  # held-out seed text: the same domain, so a large gain


def test_dsm_tokenizer_class_loads_the_extension(real_extension, tmp_path):
    from transformers import AutoTokenizer

    _, new, _, held, data = real_extension
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    write_tokenizer_files(tmp_path / "dsm", tmp_path / "src" / "tokenizer.json", "You are DSM1.0.")
    ours = AutoTokenizer.from_pretrained(tmp_path / "dsm")
    text = normalize_text(held[0])
    assert ours(text, add_special_tokens=False)["input_ids"] == new.encode(text, add_special_tokens=False).ids
    assert ours.convert_tokens_to_ids("<|im_end|>") == 248_046
