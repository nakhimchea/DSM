"""Packed sequences (documents separated by ``position_ids`` resets) must be isolated exactly."""

from __future__ import annotations

import pytest
import torch
from helpers import tiny_dsm_model


def _packed_inputs(lengths: list[int], vocab: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    docs = [torch.randint(0, vocab, (n,), generator=gen) for n in lengths]
    ids = torch.cat(docs)[None]
    pos = torch.cat([torch.arange(n) for n in lengths])[None]
    return docs, ids, pos


# document lengths chosen so that documents start at 0, 1, 63, 64, 65 and 128,
# including single-token documents and boundaries on both sides of a 64-chunk edge
LENGTH_SETS = [[1, 62, 1, 1, 63, 2], [64, 64, 3], [5], [3, 1, 1, 40, 90]]


@pytest.mark.parametrize("lengths", LENGTH_SETS)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_packed_equals_separate_documents(lengths: list[int], attn: str) -> None:
    model = tiny_dsm_model(seed=7, attn=attn)
    docs, ids, pos = _packed_inputs(lengths, model.config.vocab_size, seed=sum(lengths))
    with torch.no_grad():
        packed = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0]
        separate = torch.cat([model(input_ids=d[None], use_cache=False).logits[0] for d in docs])
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-4)


def test_packed_isolation_with_grouped_value_heads() -> None:
    model = tiny_dsm_model(seed=8, linear_num_value_heads=8, linear_num_key_heads=4)
    docs, ids, pos = _packed_inputs([10, 70, 1, 30], model.config.vocab_size, seed=8)
    with torch.no_grad():
        packed = model(input_ids=ids, position_ids=pos, use_cache=False).logits[0]
        separate = torch.cat([model(input_ids=d[None], use_cache=False).logits[0] for d in docs])
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-4)


def test_batch_of_packed_rows() -> None:
    model = tiny_dsm_model(seed=9)
    _, ids_a, pos_a = _packed_inputs([20, 30, 14], model.config.vocab_size, seed=1)
    _, ids_b, pos_b = _packed_inputs([64], model.config.vocab_size, seed=2)
    ids, pos = torch.cat([ids_a, ids_b]), torch.cat([pos_a, pos_b])
    with torch.no_grad():
        batched = model(input_ids=ids, position_ids=pos, use_cache=False).logits
        row_a = model(input_ids=ids_a, position_ids=pos_a, use_cache=False).logits
        row_b = model(input_ids=ids_b, position_ids=pos_b, use_cache=False).logits
    torch.testing.assert_close(batched, torch.cat([row_a, row_b]), atol=1e-5, rtol=1e-4)


def test_gradient_does_not_cross_documents() -> None:
    model = tiny_dsm_model(seed=10)
    model.train()
    lengths = [30, 40]
    _, ids, pos = _packed_inputs(lengths, model.config.vocab_size, seed=3)
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeds, position_ids=pos, use_cache=False).logits
    logits[:, lengths[0] :].sum().backward()  # loss on document 2 only
    leak = embeds.grad[0, : lengths[0]].abs().max().item()
    signal = embeds.grad[0, lengths[0] :].abs().max().item()
    assert signal > 0
    # Mathematically the leak is exactly 0; what remains is fp32 rounding of the
    # reverse cumsum in the decay gradient (terms that cancel exactly in theory).
    # It must stay far below fp32 resolution relative to the real gradient.
    assert leak < 1e-7 * signal, f"gradient leaked into document 1: {leak} (signal {signal})"
