"""Tokenizer fixtures: a tiny trainable BPE and the real (pinned) upstream vocabulary."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from helpers import upstream_ids

# Same pre-tokenisation as the real tokenizer: letters + combining marks stay together.
from dsm.tokenizer import PRETOKENIZE_REGEX  # noqa: E402

# (content, special) in the same relative order as the real vocabulary
ADDED_TOKENS = [
    ("<|endoftext|>", True),
    ("<|im_start|>", True),
    ("<|im_end|>", True),
    ("<tool_call>", False),
    ("</tool_call>", False),
    ("<tool_response>", False),
    ("</tool_response>", False),
    ("<think>", False),
    ("</think>", False),
]

CORPUS = [
    "Hello! How can I help you today? I can translate, summarise and answer questions.",
    "The quick brown fox jumps over the lazy dog. Numbers: 0 1 2 3 4 5 6 7 8 9 10 100 2026.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    '{"name": "get_exchange_rate", "arguments": {"base": "USD", "quote": "KHR"}}',
    "| Branch | Deposits |\n|---|---|\n| Phnom Penh | 1,200 |\n| Siem Reap | 800 |",
    "The area is $A = \\pi r^2$ and $$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$",
    "សួស្តី! ខ្ញុំអាចជួយអ្វីបានខ្លះ? ខ្ញុំអាចបកប្រែ សង្ខេប និងឆ្លើយសំណួរ។",
    "ភាសាខ្មែរ គឺជាភាសាផ្លូវការរបស់ព្រះរាជាណាចក្រកម្ពុជា។ ធនាគារ ប្រាក់ បញ្ញើ កម្ចី",
    "你好！我可以帮你翻译、总结和回答问题。今天的汇率是多少？",
    "中文、英文和高棉语。银行 存款 贷款 分行 金边 暹粒",
] * 3


def build_tiny_tokenizer(out_dir: Path, vocab_size: int = 480) -> Path:
    """Train a tiny byte-level BPE with the real pre-tokeniser and control tokens."""
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        trainers,
    )

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(ADDED_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    for content, special in ADDED_TOKENS:
        token = AddedToken(content, special=special, normalized=False)
        if special:
            tok.add_special_tokens([token])
        else:
            tok.add_tokens([token])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tokenizer.json"
    tok.save(str(path))
    return path


def _cache_dir() -> Path:
    base = Path(os.environ.get("DSM_TEST_CACHE", Path.home() / ".cache" / "dsm-tests"))
    return base / "upstream_tokenizer"


def real_tokenizer_dir() -> Path | None:
    """Directory with the real upstream tokenizer files, or None if unavailable.

    Uses ``DSM_UPSTREAM_TOKENIZER_DIR`` when set (e.g. your downloaded
    checkpoint), else downloads the sha256-pinned files from the mirror in
    ``configs/upstream.yaml`` once into ``~/.cache/dsm-tests``.
    """
    local = os.environ.get("DSM_UPSTREAM_TOKENIZER_DIR")
    if local:
        return Path(local)
    fixtures = upstream_ids_full()["test_fixtures"]
    target = _cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in fixtures["sha256"].items():
        path = target / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            try:
                with urllib.request.urlopen(f"{fixtures['tokenizer_mirror']}/{name}", timeout=60) as response:
                    data = response.read()
            except OSError:
                return None
            if hashlib.sha256(data).hexdigest() != expected:
                return None
            path.write_bytes(data)
    return target


def upstream_ids_full() -> dict:
    import yaml
    from helpers import ROOT

    return yaml.safe_load((ROOT / "configs" / "upstream.yaml").read_text(encoding="utf-8"))


def special_ids(tokenizer_json: Path) -> dict[str, int]:
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    return {entry["content"]: entry["id"] for entry in data["added_tokens"]}


__all__ = ["build_tiny_tokenizer", "real_tokenizer_dir", "special_ids", "upstream_ids"]
