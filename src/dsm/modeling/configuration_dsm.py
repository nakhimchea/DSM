# Copyright 2026 Doslarb. Licensed under the Apache License, Version 2.0.
"""Configuration of the DSM decoder.

Field names deliberately follow the widely used hybrid Gated-DeltaNet text
configuration (``layer_types``, ``linear_*``, ``rope_parameters`` ...) so that
inference engines that read these fields by name keep working with DSM
checkpoints. Defaults describe DSM1.0-0.8B.
"""

from __future__ import annotations

from transformers.configuration_utils import PreTrainedConfig

DEFAULT_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    # Multimodal rotary sections of the source checkpoint. For text-only input the
    # three position rows are identical, so this reduces exactly to 1D RoPE; the
    # fields are kept so engines that expect them can still parse the config.
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
}


class DSMConfig(PreTrainedConfig):
    r"""Configuration for :class:`DSMForCausalLM`.

    Hybrid layout: every ``full_attention_interval``-th layer is gated full
    attention, all others are Gated DeltaNet ("linear_attention") layers.

    DSM-specific fields:
        doc_reset_log_decay: log-decay written at the first token of each packed
            document on the PyTorch path; ``exp(-30) ~ 1e-13`` erases the state
            carried over from the previous document.
        loss_chunk_size: rows per chunk in the memory-efficient cross-entropy.
        gdn_backend: ``"auto"`` (fla on CUDA when installed), ``"torch"`` or ``"fla"``.
    """

    model_type = "dsm"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    layer_types: list[str] | None = None
    pad_token_id: int | None = 248044
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 248046
    doc_reset_log_decay: float = -30.0
    loss_chunk_size: int = 1024
    gdn_backend: str = "auto"

    def __post_init__(self, **kwargs):
        interval = kwargs.pop("full_attention_interval", 4)
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None and "rope_theta" not in kwargs and "rope_scaling" not in kwargs:
            self.rope_parameters = dict(DEFAULT_ROPE_PARAMETERS)
        kwargs.setdefault("partial_rotary_factor", 0.25)
        super().__post_init__(**kwargs)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers={self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {"linear_attention", "full_attention"}
        if unknown:
            raise ValueError(f"unsupported layer types {sorted(unknown)}")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    @property
    def rotary_dim(self) -> int:
        factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0)
        return int(self.head_dim * factor)


__all__ = ["DSMConfig", "DEFAULT_ROPE_PARAMETERS"]
