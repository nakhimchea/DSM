"""Shared builders for the test-suite: tiny configs, random models, the reference model."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent

# A tiny but structurally complete DSM: 2 GDN + 2 attention layers (interval 2).
TINY = dict(
    vocab_size=160,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
)
TINY_ROPE = {
    "rope_type": "default",
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "mrope_section": [1, 1, 2],
    "mrope_interleaved": True,
}


def upstream_ids() -> dict:
    """Upstream identifiers (read from configs/upstream.yaml, never hard-coded)."""
    return yaml.safe_load((ROOT / "configs" / "upstream.yaml").read_text(encoding="utf-8"))["upstream"]


def tiny_kwargs(**overrides) -> dict:
    kwargs = dict(TINY)
    kwargs.update(overrides)
    kwargs.setdefault("rope_parameters", dict(TINY_ROPE))
    kwargs.setdefault("full_attention_interval", 2)
    return kwargs


def tiny_dsm_config(**overrides):
    from dsm.modeling import DSMConfig

    kwargs = tiny_kwargs(**overrides)
    kwargs.setdefault("gdn_backend", "torch")
    return DSMConfig(**kwargs)


@torch.no_grad()
def randomize_(model: torch.nn.Module, seed: int = 0, scale: float = 0.4) -> torch.nn.Module:
    """Give every parameter a random value (default init hides swapped norms etc.).

    Decay parameters are drawn so that the recurrent state persists for tens of
    tokens, which makes state-leakage bugs visible.
    """
    gen = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        if name.endswith("A_log"):
            values = torch.empty(param.shape).uniform_(math.log(0.01), math.log(0.5), generator=gen)
        elif name.endswith("dt_bias"):
            values = torch.randn(param.shape, generator=gen) * 0.5
        elif "norm" in name:
            values = torch.randn(param.shape, generator=gen) * 0.3
        else:
            values = torch.randn(param.shape, generator=gen) * scale / math.sqrt(max(param.shape[-1], 1))
        param.copy_(values.to(param.dtype))
    return model


def tiny_dsm_model(seed: int = 0, attn: str = "eager", **overrides):
    from dsm.modeling import DSMForCausalLM

    model = DSMForCausalLM(tiny_dsm_config(**overrides))
    model.config._attn_implementation = attn
    randomize_(model, seed=seed)
    return model.float().eval()


def reference_text_model(attn: str = "eager", **overrides):
    """The upstream transformers implementation built from a tiny text config."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = tiny_kwargs(**overrides)
    kwargs.pop("gdn_backend", None)
    cfg = AutoConfig.for_model(upstream_ids()["model_types"]["text"], **kwargs)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation=attn).float().eval()
