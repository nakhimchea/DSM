"""A tiny DSM model directory plus packed / SFT / DPO data for trainer tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

SYSTEM_PROMPT = "You are DSM1.0, an AI assistant developed by Doslarb."

SFT_PAIRS = [
    ("en", "Hello!", "Hello! How can I help you today?"),
    ("en", "What is 2 + 3?", "2 + 3 = 5."),
    ("en", "Name a city in Cambodia.", "Phnom Penh."),
    ("en", "Say thanks.", "Thank you!"),
    ("km", "សួស្តី!", "សួស្តី! ខ្ញុំអាចជួយអ្វីបាន?"),
    ("km", "ធនាគារ", "ធនាគារ ប្រាក់ បញ្ញើ"),
    ("zh", "你好！", "你好！我可以帮你。"),
    ("zh", "银行在哪里？", "金边分行。"),
]


def tiny_model_dir(
    out: Path, tokenizer_json: Path, seed: int = 0, vocab_size: int = 512, **overrides
) -> Path:
    from helpers import tiny_dsm_config
    from tokenizer_fixtures import special_ids

    from dsm.convert.artifacts import write_generation_config
    from dsm.modeling import DSMForCausalLM
    from dsm.tokenizer import write_tokenizer_files
    from dsm.train.checkpoint import save_model_dir

    ids = special_ids(tokenizer_json)
    eos, pad = ids["<|im_end|>"], ids["<|endoftext|>"]
    torch.manual_seed(seed)
    config = tiny_dsm_config(vocab_size=vocab_size, eos_token_id=eos, pad_token_id=pad, **overrides)
    model = DSMForCausalLM(config)
    save_model_dir(model, out, carry_from=None, dtype="float32")
    write_tokenizer_files(out, tokenizer_json, SYSTEM_PROMPT)
    write_generation_config(out, eos_ids=[eos, pad], pad_id=pad)
    (out / "LICENSE").write_text("Apache License 2.0 (test fixture)\n")
    (out / "NOTICE").write_text("DSM test fixture\n")
    return out


def packed_corpus(
    prefix: Path, tokenizer_json: Path, texts: list[str], name: str, vocab_size: int = 512
) -> Path:
    from tokenizer_fixtures import special_ids
    from tokenizers import Tokenizer

    from dsm.data.packed import pack_texts, tokenizer_hash

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    eos = special_ids(tokenizer_json)["<|endoftext|>"]
    meta = {"name": name, "license": "test", "tokenizer": tokenizer_hash(tokenizer_json)}
    pack_texts(texts, tokenizer, prefix, eos_id=eos, vocab_size=vocab_size, meta=meta)
    return prefix


def sft_file(path: Path, copies: int = 1) -> Path:
    from dsm.data.sft.schema import sft_record
    from dsm.utils import write_jsonl

    records = []
    for c in range(copies):
        for i, (lang, user, answer) in enumerate(SFT_PAIRS):
            records.append(
                sft_record(
                    id=f"t-{c}-{i}",
                    source="test",
                    lang=lang,
                    category="chat",
                    messages=[{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                )
            )
    write_jsonl(path, records)
    return path


def dpo_file(path: Path) -> Path:
    from dsm.data.sft.schema import preference_record
    from dsm.utils import write_jsonl

    records = []
    for i, (lang, user, answer) in enumerate(SFT_PAIRS):
        records.append(
            preference_record(
                id=f"p-{i}",
                source="test",
                lang=lang,
                category="chat",
                prompt=[{"role": "user", "content": user}],
                chosen={"role": "assistant", "content": answer},
                rejected={"role": "assistant", "content": "No."},
            )
        )
    write_jsonl(path, records)
    return path


def write_config(path: Path, **fields) -> Path:
    path.write_text(json.dumps(fields))  # JSON is valid YAML
    return path
