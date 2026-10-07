"""A tiny checkpoint in the exact upstream format (multimodal wrapper + MTP + upstream metadata)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from helpers import randomize_, tiny_kwargs, upstream_ids

LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]


def build_tiny_upstream(out_dir: Path, tokenizer_json: Path, seed: int = 0, vocab_size: int = 512) -> Path:
    """Reference-implementation checkpoint with vision tower, MTP tensors, upstream tokenizer config,
    README and LICENSE - everything the converter has to handle."""
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    up = upstream_ids()
    text = tiny_kwargs(vocab_size=vocab_size)
    text.pop("full_attention_interval")
    text.pop("tie_word_embeddings")  # like the real checkpoint: the text sub-config says "untied"...
    text["layer_types"] = list(LAYER_TYPES)
    vision = dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=text["hidden_size"],
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        num_position_embeddings=16,
    )
    # ...while the top-level config ties the embeddings (the converter must honour this)
    config = AutoConfig.for_model(
        up["model_types"]["composite"], text_config=text, vision_config=vision, tie_word_embeddings=True
    )
    model = getattr(transformers, up["architectures"]["composite"])(config)
    randomize_(model, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    weights = out_dir / "model.safetensors"
    tensors = load_file(str(weights))
    hidden = text["hidden_size"]
    tensors["mtp.fc.weight"] = torch.randn(hidden, 2 * hidden)
    tensors["mtp.norm.weight"] = torch.zeros(hidden)
    save_file(tensors, str(weights), metadata={"format": "pt"})

    shutil.copyfile(tokenizer_json, out_dir / "tokenizer.json")
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": up["tokenizer_class"],
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(f"# {up['instruct_repo_id']}\nupstream model card\n", encoding="utf-8")
    (out_dir / "LICENSE").write_text(f"Apache License, Version 2.0\n\n{up['copyright']}\n", encoding="utf-8")
    return out_dir
