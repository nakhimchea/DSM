# Copyright 2026 Doslarb. Licensed under the Apache License, Version 2.0.
"""Adapter between DSM layers and the transformers cache API.

DSM uses transformers' ``DynamicCache(config=...)``: entries of
``config.layer_types`` equal to ``"linear_attention"`` become
``LinearAttentionLayer`` caches (short-conv state + recurrent GDN state) and
``"full_attention"`` entries become ordinary key/value caches. This is what
``model.generate()`` builds automatically.

Every call into that API lives in this file, so a future transformers version
bump touches one place only.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicCache


def new_cache(config) -> DynamicCache:
    """A fresh hybrid cache for a DSM config."""
    return DynamicCache(config=config)


def past_length(cache: Cache | None) -> int:
    """Number of tokens already processed (read from the first attention layer)."""
    if cache is None:
        return 0
    return int(cache.get_seq_length())


def has_previous_state(cache: Cache | None, layer_idx: int) -> bool:
    """Whether the GDN layer ``layer_idx`` already holds conv/recurrent state."""
    if cache is None:
        return False
    return bool(cache.has_previous_state(layer_idx, state_idx=0))


def read_conv_prefix(cache: Cache, layer_idx: int, kernel_size: int) -> torch.Tensor:
    """The last ``kernel_size - 1`` raw (pre-convolution) inputs, ``[B, C, K-1]``.

    Returns a copy: transformers updates the cached buffer in place, which would
    otherwise overwrite the prefix before it is used.
    """
    conv_state = cache.layers[layer_idx].conv_states[0]
    return conv_state[..., -(kernel_size - 1) :].clone()


def write_conv_state(cache: Cache, layer_idx: int, inputs: torch.Tensor, kernel_size: int) -> None:
    """Append the raw inputs ``[B, C, T]`` of this step to the conv state."""
    cache.update_conv_state(inputs, layer_idx, conv_kernel_size=kernel_size)


def read_recurrent_state(cache: Cache, layer_idx: int) -> torch.Tensor:
    """A copy of the recurrent GDN state ``[B, H, d_k, d_v]`` (the cache buffer is updated in place)."""
    return cache.layers[layer_idx].recurrent_states[0].clone()


def write_recurrent_state(cache: Cache, layer_idx: int, state: torch.Tensor) -> None:
    cache.update_recurrent_state(state, layer_idx)


def update_kv(
    cache: Cache, layer_idx: int, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append key/value ``[B, H_kv, T, D]`` and return the full cached tensors."""
    return cache.update(key, value, layer_idx)
