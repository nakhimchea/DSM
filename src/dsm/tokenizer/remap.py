"""Give a model the embedding rows of a Khmer-extended tokenizer.

Every new token starts as the mean of the rows of the pieces the ORIGINAL tokenizer
splits it into: the input embedding and, when it is not tied, the output projection.
Existing rows are copied bit for bit, so on any input made of old ids the extended model
returns exactly the old logits (plus logits for the new ids). The matrix is padded with
zero rows to a multiple of ``pad_to``; the tokenizer never produces those ids.

The first new rows reuse the unused padding rows after the control tokens, so the
matrix only grows by what does not fit there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def piece_ids(old_tokenizer, tokens: dict[str, int]) -> dict[int, list[int]]:
    """For each new token id: the ids the original BPE splits the token into."""
    return {index: [t.id for t in old_tokenizer.model.tokenize(token)] for token, index in tokens.items()}


def _grown(weight: torch.Tensor, parts: dict[int, list[int]], rows: int) -> torch.Tensor:
    out = weight.new_zeros(rows, weight.shape[1])
    out[: weight.shape[0]] = weight
    for index, ids in parts.items():
        out[index] = weight[ids].float().mean(0).to(weight.dtype)
    return out


def resize_for_extension(model, parts: dict[int, list[int]], vocab_size: int) -> None:
    """Grow the embedding (and untied output) rows to ``vocab_size`` and initialise the new ids."""
    embedding = model.get_input_embeddings()
    rows, hidden = embedding.weight.shape
    if vocab_size < rows:
        raise ValueError(f"vocab_size {vocab_size} < existing rows {rows}; remap never shrinks")
    if parts and max(parts) >= vocab_size:
        raise ValueError("new token id outside the matrix")
    if any(not ids or min(ids) < 0 or max(ids) >= index for index, ids in parts.items()):
        raise ValueError("a new token must be built from older ids")
    tied = getattr(model.config, "tie_word_embeddings", False)
    output = model.get_output_embeddings()
    kwargs = {"dtype": embedding.weight.dtype, "device": embedding.weight.device}
    with torch.no_grad():
        new_embedding = torch.nn.Embedding(vocab_size, hidden, **kwargs)
        new_embedding.weight.copy_(_grown(embedding.weight, parts, vocab_size))
        new_output = torch.nn.Linear(hidden, vocab_size, bias=False, **kwargs)
        if tied:
            new_output.weight = new_embedding.weight
        else:
            new_output.weight.copy_(_grown(output.weight, parts, vocab_size))
    model.set_input_embeddings(new_embedding)
    model.set_output_embeddings(new_output)
    model.config.vocab_size = vocab_size
    model.vocab_size = vocab_size


def extend_model_dir(
    model_dir: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    pad_to: int = 64,
    dtype: str | None = None,
) -> dict:
    """Write ``out_dir``: ``model_dir`` with the extended tokenizer of ``tokenizer_dir`` and its rows."""
    from tokenizers import Tokenizer

    from dsm.modeling import DSMForCausalLM
    from dsm.train.checkpoint import read_lineage, save_model_dir

    model_dir, tokenizer_dir = Path(model_dir), Path(tokenizer_dir)
    meta = json.loads((tokenizer_dir / "extension.json").read_text(encoding="utf-8"))
    start_id = int(meta["start_id"])
    old_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    new_json = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    old_vocab, new_vocab = old_json["model"]["vocab"], new_json["model"]["vocab"]
    if any(new_vocab.get(token) != index for token, index in old_vocab.items()):
        raise ValueError(f"{tokenizer_dir} was not extended from the tokenizer of {model_dir}")
    added = {a["content"] for a in old_json.get("added_tokens", [])}
    tokens = {t: i for t, i in new_vocab.items() if t not in old_vocab and t not in added}
    if min(tokens.values(), default=start_id) < start_id:
        raise ValueError("new token below start_id")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dtype = dtype or str(config.get("dtype") or config.get("torch_dtype") or "float32")
    model = DSMForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    rows = model.get_input_embeddings().weight.shape[0]
    needed = max(max(tokens.values(), default=0) + 1, rows)
    vocab_size = -(-needed // pad_to) * pad_to
    old_tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    resize_for_extension(model, piece_ids(old_tokenizer, tokens), vocab_size)

    entry = {
        "stage": "tokenizer_extension",
        "base": str(model_dir),
        "start_id": start_id,
        "new_tokens": len(tokens),
        "vocab_size": vocab_size,
    }
    out = save_model_dir(
        model,
        out_dir,
        carry_from=model_dir,
        dtype=dtype,
        lineage_entry=entry,
        parent_lineage=read_lineage(model_dir),
    )
    for name in ("tokenizer.json", "extension.json", "new_tokens.tsv"):
        if (tokenizer_dir / name).is_file():
            shutil.copyfile(tokenizer_dir / name, out / name)
    return entry
