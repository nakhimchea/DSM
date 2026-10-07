"""Where the parameters, compute and memory of a DSM model go (no weights needed).

All numbers come from the configuration: the model is built on the ``meta``
device, so the 0.8B report takes milliseconds and no RAM.
"""

from __future__ import annotations

from collections import defaultdict

GROUPS = (
    ("embedding (tied LM head)", ("embed_tokens",)),
    ("GDN projections (qkv, z, a, b)", ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")),
    ("GDN conv / decay / norm", ("conv1d", "A_log", "dt_bias", "linear_attn.norm")),
    ("GDN output projection", ("linear_attn.out_proj",)),
    ("attention q/k/v (+ gate)", ("q_proj", "k_proj", "v_proj")),
    ("attention output projection", ("self_attn.o_proj",)),
    ("attention q/k norms", ("q_norm", "k_norm")),
    ("MLP (SwiGLU)", ("gate_proj", "up_proj", "down_proj")),
    ("layer norms", ("input_layernorm", "post_attention_layernorm", "model.norm")),
)


def build_meta_model(config):
    import torch

    from dsm.modeling import DSMForCausalLM

    with torch.device("meta"):
        return DSMForCausalLM(config)


def parameter_report(config) -> dict:
    model = build_meta_model(config)
    groups: dict[str, int] = defaultdict(int)
    total = 0
    for name, param in model.named_parameters():  # tied lm_head is not listed twice
        total += param.numel()
        for label, keys in GROUPS:
            if any(k in name for k in keys):
                groups[label] += param.numel()
                break
        else:
            groups["other"] += param.numel()
    layers = defaultdict(int)
    for kind in config.layer_types:
        layers[kind] += 1
    return {
        "total": total,
        "non_embedding": total - groups["embedding (tied LM head)"],
        "groups": dict(groups),
        "shares": {k: v / total for k, v in groups.items()},
        "layer_types": dict(layers),
    }


def decode_flops_per_token(config, context: int) -> dict[str, float]:
    """Approximate FLOPs to generate one token after ``context`` tokens (multiply-add = 2 FLOPs)."""
    report = parameter_report(config)
    h, v = config.hidden_size, config.vocab_size
    n_attn = sum(1 for t in config.layer_types if t == "full_attention")
    n_gdn = len(config.layer_types) - n_attn
    flops = {
        "matmuls (non-embedding weights)": 2.0 * report["non_embedding"],
        "LM head": 2.0 * h * v,
        "attention over the context": n_attn * 2 * 2 * config.num_attention_heads * config.head_dim * context,
        "GDN state update": n_gdn
        * config.linear_num_value_heads
        * 6
        * config.linear_key_head_dim
        * config.linear_value_head_dim,
    }
    flops["total"] = sum(flops.values())
    return flops


def cache_report(config, context: int, kv_bytes: int = 2, state_bytes: int = 4) -> dict[str, float]:
    """Memory of the inference cache: KV for attention layers, fixed-size state for GDN layers."""
    n_attn = sum(1 for t in config.layer_types if t == "full_attention")
    n_gdn = len(config.layer_types) - n_attn
    kv_per_token = n_attn * 2 * config.num_key_value_heads * config.head_dim * kv_bytes
    conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    recurrent = (
        n_gdn
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * state_bytes
    )
    conv = n_gdn * conv_dim * (config.linear_conv_kernel_dim - 1) * kv_bytes
    return {
        "kv_bytes_per_token": kv_per_token,
        "kv_bytes_at_context": kv_per_token * context,
        "gdn_state_bytes_per_sequence": recurrent + conv,
        "total_bytes_at_context": kv_per_token * context + recurrent + conv,
        "full_attention_equivalent_bytes_at_context": len(config.layer_types)
        * 2
        * config.num_key_value_heads
        * config.head_dim
        * kv_bytes
        * context,
    }
