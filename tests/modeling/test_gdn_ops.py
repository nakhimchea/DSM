"""Unit tests of the numerical kernels in ``dsm.modeling.ops_dsm``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dsm.modeling.ops_dsm import (
    causal_conv1d,
    chunked_cross_entropy,
    chunked_token_logprobs,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def _inputs(batch=2, seq=37, heads=3, dk=8, dv=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq, heads, dk, generator=gen)
    k = torch.randn(batch, seq, heads, dk, generator=gen)
    v = torch.randn(batch, seq, heads, dv, generator=gen)
    g = -torch.rand(batch, seq, heads, generator=gen) * 0.3  # mild decay => long memory
    beta = torch.rand(batch, seq, heads, generator=gen)
    return q, k, v, g, beta


@pytest.mark.parametrize("seq", [1, 7, 16, 17, 33, 64, 65])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_equals_recurrent(seq: int, chunk: int) -> None:
    q, k, v, g, beta = _inputs(seq=seq)
    init = torch.randn(2, 3, 8, 6) * 0.1
    out_c, s_c = torch_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk, initial_state=init, output_final_state=True
    )
    out_r, s_r = torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=init, output_final_state=True
    )
    torch.testing.assert_close(out_c, out_r, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(s_c, s_r, atol=1e-5, rtol=1e-4)


def test_state_carry_equals_full_sequence() -> None:
    """Processing [a, b] at once == processing a, then b from a's final state."""
    q, k, v, g, beta = _inputs(seq=50)
    full, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=16)
    first, state = torch_chunk_gated_delta_rule(
        q[:, :23], k[:, :23], v[:, :23], g[:, :23], beta[:, :23], chunk_size=16, output_final_state=True
    )
    second, _ = torch_chunk_gated_delta_rule(
        q[:, 23:], k[:, 23:], v[:, 23:], g[:, 23:], beta[:, 23:], chunk_size=16, initial_state=state
    )
    torch.testing.assert_close(torch.cat([first, second], dim=1), full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("boundary", [1, 15, 16, 17, 40])
def test_log_decay_reset_isolates_documents(boundary: int) -> None:
    """Setting g = -30 at a document start equals processing the documents separately."""
    q, k, v, g, beta = _inputs(seq=48)
    g_reset = g.clone()
    g_reset[:, boundary] = -30.0
    packed, _ = torch_chunk_gated_delta_rule(q, k, v, g_reset, beta, chunk_size=16)
    first, _ = torch_chunk_gated_delta_rule(
        q[:, :boundary], k[:, :boundary], v[:, :boundary], g[:, :boundary], beta[:, :boundary], chunk_size=16
    )
    second, _ = torch_chunk_gated_delta_rule(
        q[:, boundary:], k[:, boundary:], v[:, boundary:], g[:, boundary:], beta[:, boundary:], chunk_size=16
    )
    torch.testing.assert_close(packed, torch.cat([first, second], dim=1), atol=1e-6, rtol=1e-5)


def test_causal_conv_matches_padded_conv1d() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 12, generator=gen)
    w = torch.randn(5, 4, generator=gen)
    expected = F.silu(F.conv1d(x, w.unsqueeze(1), padding=3, groups=5)[:, :, :12])
    torch.testing.assert_close(causal_conv1d(x, w), expected)


def test_causal_conv_prefix_equals_concatenation() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 5, 20, generator=gen)
    w = torch.randn(5, 4, generator=gen)
    full = causal_conv1d(x, w)
    tail = causal_conv1d(x[:, :, 12:], w, prefix=x[:, :, 9:12])
    torch.testing.assert_close(tail, full[:, :, 12:])


def test_causal_conv_document_mask() -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1, 5, 10, generator=gen)
    w = torch.randn(5, 4, generator=gen)
    doc_ids = torch.tensor([[1, 1, 1, 2, 3, 3, 3, 3, 3, 3]])
    out = causal_conv1d(x, w, doc_ids=doc_ids)
    expected = torch.cat(
        [causal_conv1d(x[:, :, :3], w), causal_conv1d(x[:, :, 3:4], w), causal_conv1d(x[:, :, 4:], w)], dim=-1
    )
    torch.testing.assert_close(out, expected)


def test_chunked_cross_entropy_matches_reference() -> None:
    gen = torch.Generator().manual_seed(4)
    hidden = torch.randn(37, 16, generator=gen, requires_grad=True)
    weight = torch.randn(50, 16, generator=gen, requires_grad=True)
    labels = torch.randint(0, 50, (37,), generator=gen)
    labels[::5] = -100
    loss = chunked_cross_entropy(hidden, weight, labels, chunk_size=8)
    ref = F.cross_entropy(hidden @ weight.T, labels, ignore_index=-100, reduction="sum")
    torch.testing.assert_close(loss, ref)
    g1 = torch.autograd.grad(loss, (hidden, weight))
    g2 = torch.autograd.grad(ref, (hidden, weight))
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_chunked_token_logprobs() -> None:
    gen = torch.Generator().manual_seed(5)
    hidden = torch.randn(21, 8, generator=gen)
    weight = torch.randn(30, 8, generator=gen)
    labels = torch.randint(0, 30, (21,), generator=gen)
    labels[3] = -100
    out = chunked_token_logprobs(hidden, weight, labels, chunk_size=4)
    ref = torch.log_softmax(hidden @ weight.T, -1).gather(-1, labels.clamp(min=0)[:, None]).squeeze(-1)
    ref[3] = 0.0
    torch.testing.assert_close(out, ref)
