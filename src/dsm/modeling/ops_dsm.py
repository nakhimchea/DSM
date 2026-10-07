# Copyright 2026 Doslarb. Licensed under the Apache License, Version 2.0.
"""Numerical kernels used by the DSM decoder.

* Gated DeltaNet (GDN) core: an exact PyTorch reference (chunked form for
  prefill/training, recurrent form for single-token decode) and an optional
  fast path through `flash-linear-attention` (fla) on CUDA.
* Depthwise causal short convolution with optional document isolation
  (packed sequences) and a cached prefix (decode / continuation).
* Memory-efficient cross-entropy that never materialises the full
  ``[tokens, vocab]`` logits tensor (the DSM vocabulary has ~248K entries).

This module only depends on torch so it can be shipped next to a checkpoint and
loaded with ``trust_remote_code=True`` without the ``dsm`` package installed.

Gated delta rule (per head, state ``S`` of shape ``[d_k, d_v]``)::

    S_t = exp(g_t) * S_{t-1}                       # forget (decay)
    S_t = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T  # delta-rule write
    o_t = S_t^T q_t                                 # read

with ``q`` and ``k`` L2-normalised and ``q`` scaled by ``d_k ** -0.5``.
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------------

_FLA_CACHE: tuple | bool | None = None


def fla_ops() -> tuple | None:
    """Return ``(chunk_fn, fused_recurrent_fn)`` from flash-linear-attention, or ``None``.

    Set ``DSM_DISABLE_FLA=1`` to force the PyTorch reference implementation.
    """
    global _FLA_CACHE
    if _FLA_CACHE is None:
        if os.environ.get("DSM_DISABLE_FLA") == "1" or importlib.util.find_spec("fla") is None:
            _FLA_CACHE = False
        else:
            try:
                from fla.ops.gated_delta_rule import (  # type: ignore[import-not-found]
                    chunk_gated_delta_rule,
                    fused_recurrent_gated_delta_rule,
                )

                _FLA_CACHE = (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
            except Exception:  # pragma: no cover - depends on the GPU stack
                _FLA_CACHE = False
    return _FLA_CACHE or None


def resolve_backend(requested: str, device: torch.device) -> str:
    """Pick the GDN backend: ``"torch"`` or ``"fla"``.

    ``"auto"`` uses fla when it is installed and the tensors live on CUDA.
    """
    if requested == "torch":
        return "torch"
    if requested == "fla":
        if fla_ops() is None:
            raise RuntimeError("gdn_backend='fla' but flash-linear-attention is not installed")
        if device.type != "cuda":
            raise RuntimeError("gdn_backend='fla' needs CUDA tensors")
        return "fla"
    if requested != "auto":
        raise ValueError(f"unknown gdn backend {requested!r} (expected auto|torch|fla)")
    return "fla" if device.type == "cuda" and fla_ops() is not None else "torch"


# -----------------------------------------------------------------------------
# Gated DeltaNet core
# -----------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """``x / sqrt(sum(x^2) + eps)`` - the normalisation used by the fla kernels."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk-parallel gated delta rule (exact, fp32 internally).

    Shapes: ``query``/``key`` ``[B, T, H, d_k]``, ``value`` ``[B, T, H, d_v]``,
    ``g``/``beta`` ``[B, T, H]`` (``g`` is the log-decay, ``<= 0``), optional
    ``initial_state`` ``[B, H, d_k, d_v]``. Returns ``([B, T, H, d_v], state)``.

    Within a chunk the delta-rule updates are condensed with a unit lower
    triangular solve (the "UT transform"); across chunks the state is carried
    sequentially.
    """
    in_dtype = query.dtype
    batch, seq_len, _, d_k = key.shape
    n_heads, d_v = value.shape[-2:]

    query, key, value, beta, g = (
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    )
    if qk_l2norm:
        query = l2norm(query)
        key = l2norm(key)
    query = query * (query.shape[-1] ** -0.5)

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n_chunks = (seq_len + pad) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], n_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], n_chunks, chunk_size)

    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = g.cumsum(dim=-1)
    # pairwise[i, j] = exp(cum_i - cum_j) for j <= i, 0 above the diagonal
    pairwise = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(strictly_upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)

    if initial_state is None:
        state = torch.zeros(batch, n_heads, d_k, d_v, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    out = torch.zeros_like(new_values)

    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(n_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(batch, n_heads, -1, d_v)[:, :, :seq_len]
    out = out.transpose(1, 2).to(in_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Token-by-token gated delta rule (same contract as the chunked form)."""
    in_dtype = query.dtype
    batch, seq_len, _, d_k = key.shape
    n_heads, d_v = value.shape[-2:]

    query, key, value, beta, g = (
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    )
    if qk_l2norm:
        query = l2norm(query)
        key = l2norm(key)
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        state = torch.zeros(batch, n_heads, d_k, d_v, dtype=value.dtype, device=value.device)
    else:
        state = initial_state.to(value)
    out = torch.zeros_like(value)
    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        state = state * g[:, :, t].exp()[..., None, None]
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(in_dtype)
    return out, (state if output_final_state else None)


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    backend: str,
    single_step: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch to fla (CUDA) or the PyTorch reference.

    ``single_step`` selects the recurrent kernel (decode); otherwise the chunked
    kernel is used (prefill, continuation, training).
    """
    if backend == "fla":
        chunk_fn, recurrent_fn = fla_ops()  # type: ignore[misc]
        fn = recurrent_fn if single_step else chunk_fn
        return fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
        )
    if single_step:
        return torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state=initial_state, output_final_state=output_final_state
        )
    return torch_chunk_gated_delta_rule(
        query, key, value, g, beta, initial_state=initial_state, output_final_state=output_final_state
    )


# -----------------------------------------------------------------------------
# Short convolution
# -----------------------------------------------------------------------------


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    prefix: torch.Tensor | None = None,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal convolution followed by SiLU.

    Args:
        x: raw inputs ``[B, C, T]``.
        weight: depthwise kernel ``[C, K]`` (tap ``K-1`` multiplies the current token).
        prefix: the ``K-1`` inputs preceding ``x`` (from a cache) ``[B, C, K-1]``;
            zeros when ``None`` (start of a sequence).
        doc_ids: ``[B, T]`` document index of every position for packed
            sequences. Taps that would read a different document are zeroed;
            the prefix counts as document 0.
    """
    batch, channels, seq_len = x.shape
    k = weight.shape[-1]
    if prefix is None:
        prefix = x.new_zeros(batch, channels, k - 1)
    x_ext = torch.cat([prefix.to(x.dtype), x], dim=-1)

    if doc_ids is None:
        out = F.conv1d(x_ext.to(weight.dtype), weight.unsqueeze(1), groups=channels)
    else:
        doc_ext = torch.cat([doc_ids.new_zeros(batch, k - 1), doc_ids], dim=-1)
        x_ext = x_ext.to(weight.dtype)
        out = None
        for j in range(k):
            same_doc = (doc_ext[:, j : j + seq_len] == doc_ids).unsqueeze(1).to(x_ext.dtype)
            term = weight[:, j].view(1, channels, 1) * x_ext[:, :, j : j + seq_len] * same_doc
            out = term if out is None else out + term
    return F.silu(out).to(x.dtype)


# -----------------------------------------------------------------------------
# Memory-efficient loss / log-probabilities
# -----------------------------------------------------------------------------


def _ce_sum(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")


def chunked_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024
) -> torch.Tensor:
    """Sum of next-token cross-entropy over ``labels != -100``.

    ``hidden`` ``[N, H]`` and ``labels`` ``[N]`` are already aligned (shifted).
    Logits are computed ``chunk_size`` rows at a time and recomputed in the
    backward pass, so peak memory is ``chunk_size x vocab`` instead of
    ``N x vocab``.
    """
    keep = labels != -100
    hidden, labels = hidden[keep], labels[keep]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        h, y = hidden[start : start + chunk_size], labels[start : start + chunk_size]
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            total = total + checkpoint(_ce_sum, h, weight, y, use_reentrant=False)
        else:
            total = total + _ce_sum(h, weight, y)
    return total


def _gather_logprobs(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def chunked_token_logprobs(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024
) -> torch.Tensor:
    """Log-probability of each label (``[N]``; 0 where ``labels == -100``)."""
    out = hidden.new_zeros(labels.shape, dtype=torch.float32)
    keep = labels != -100
    idx = keep.nonzero(as_tuple=True)[0]
    h_all, y_all = hidden[idx], labels[idx]
    pieces = []
    for start in range(0, h_all.shape[0], chunk_size):
        h, y = h_all[start : start + chunk_size], y_all[start : start + chunk_size]
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            pieces.append(checkpoint(_gather_logprobs, h, weight, y, use_reentrant=False))
        else:
            pieces.append(_gather_logprobs(h, weight, y))
    if pieces:
        out = out.index_put((idx,), torch.cat(pieces))
    return out
