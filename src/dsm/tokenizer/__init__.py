"""DSM tokenizer files, chat template and training-label rendering."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .render import build_template, encode_for_training, encode_prompt, render

# transformers class that loads tokenizer.json exactly as stored (no regex rebuild).
TOKENIZER_CLASS = "TokenizersBackend"

# Pre-tokenisation regex of the DSM vocabulary (letters + combining marks stay together,
# so Khmer clusters are never split before BPE). Engines implement it natively; the
# GGUF export checks that a (possibly Khmer-extended) tokenizer still uses it.
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

TOKENIZER_CONFIG = {
    "tokenizer_class": TOKENIZER_CLASS,
    "bos_token": None,
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "unk_token": None,
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": False,
    "clean_up_tokenization_spaces": False,
    "split_special_tokens": False,
    "errors": "replace",
    "model_max_length": 262144,
}


def write_tokenizer_files(
    out_dir: str | Path,
    tokenizer_json: str | Path,
    default_system_prompt: str,
    model_max_length: int = 262144,
) -> None:
    """Write ``tokenizer.json`` (copied verbatim), ``tokenizer_config.json`` and ``chat_template.jinja``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(tokenizer_json).resolve() != (out / "tokenizer.json").resolve():
        shutil.copyfile(tokenizer_json, out / "tokenizer.json")
    config = dict(TOKENIZER_CONFIG, model_max_length=model_max_length)
    (out / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    (out / "chat_template.jinja").write_text(build_template(default_system_prompt), encoding="utf-8")


def special_token_ids(tokenizer_json: str | Path) -> dict[str, int]:
    """Map of added-token content -> id read from a ``tokenizer.json``."""
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    return {entry["content"]: entry["id"] for entry in data.get("added_tokens", [])}


__all__ = [
    "PRETOKENIZE_REGEX",
    "TOKENIZER_CLASS",
    "TOKENIZER_CONFIG",
    "build_template",
    "encode_for_training",
    "encode_prompt",
    "render",
    "special_token_ids",
    "write_tokenizer_files",
]
