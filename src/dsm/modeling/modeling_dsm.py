# Copyright 2026 Doslarb. Licensed under the Apache License, Version 2.0.
"""DSM decoder: a hybrid of Gated DeltaNet and gated full attention.

Layer stack (DSM1.0-0.8B): 24 pre-norm residual blocks; every 4th block mixes
tokens with gated full attention, the other 18 with Gated DeltaNet (a linear
recurrent "fast-weight" memory updated by the delta rule). Every block ends with
a SwiGLU MLP. Embeddings are tied with the output head.

Packed sequences: a new document starts wherever ``position_ids == 0``. The
model then isolates documents exactly - block-diagonal causal attention,
RoPE positions restarting at 0, short-convolution taps masked at the boundary,
and the Gated-DeltaNet state erased at each document start (the log-decay of
the first token is set to ``config.doc_reset_log_decay``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from . import cache_dsm
from .configuration_dsm import DSMConfig
from .ops_dsm import causal_conv1d, chunked_cross_entropy, gated_delta_rule, resolve_backend

# -----------------------------------------------------------------------------
# Normalisation / positions
# -----------------------------------------------------------------------------


class DSMRMSNorm(nn.Module):
    """Zero-centred RMSNorm: ``x / rms(x) * (1 + w)`` with ``w`` initialised to 0.

    Zero-centring keeps weight decay from shrinking the effective gain towards 0.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class DSMGatedRMSNorm(nn.Module):
    """Output norm of Gated DeltaNet: ``w * x / rms(x) * silu(z)`` (plain weight, init 1)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(in_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(in_dtype)


class DSMRotaryEmbedding(nn.Module):
    """NeoX-style rotary embedding applied to the first ``rotary_dim`` dims of each head."""

    def __init__(self, config: DSMConfig) -> None:
        super().__init__()
        rope = config.rope_parameters or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type')!r} is not supported by DSM")
        self.theta = float(rope.get("rope_theta", 10_000_000.0))
        self.rotary_dim = config.rotary_dim
        self._inv_freq: dict[torch.device, torch.Tensor] = {}

    def inv_freq(self, device: torch.device) -> torch.Tensor:
        if device not in self._inv_freq:
            exponents = (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim
            )
            self._inv_freq[device] = 1.0 / (self.theta**exponents)
        return self._inv_freq[device]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq(x.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading ``cos.shape[-1]`` dims of ``q``/``k`` ``[B, H, T, D]``."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# -----------------------------------------------------------------------------
# Token mixers
# -----------------------------------------------------------------------------


class DSMAttention(nn.Module):
    """Gated grouped-query attention with QK-norm and partial RoPE.

    ``q_proj`` produces, per head, the query and a gate (``[q | gate]``); the
    attention output is multiplied by ``sigmoid(gate)`` before ``o_proj``.
    """

    def __init__(self, config: DSMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = DSMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DSMRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        past_key_values=None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q_and_gate = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, 2 * self.head_dim)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch, seq_len, -1)

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, -1)).transpose(
            1, 2
        )
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, -1).transpose(1, 2)
        query, key = apply_rotary(query, key, cos, sin)

        if past_key_values is not None:
            key, value = cache_dsm.update_kv(past_key_values, self.layer_idx, key, value)

        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        dropout = self.attention_dropout if self.training else 0.0
        is_causal = attn_mask is None and seq_len > 1
        if getattr(self.config, "_attn_implementation", None) == "eager":
            out = self._eager(query, key, value, attn_mask, is_causal, dropout)
        else:
            out = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout,
                is_causal=is_causal,
                scale=self.scaling,
            )
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    def _eager(self, query, key, value, attn_mask, is_causal, dropout):
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if is_causal:
            q_len, k_len = scores.shape[-2:]
            attn_mask = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).tril(k_len - q_len)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=dropout, training=self.training)
        return torch.matmul(probs, value)


class DSMGatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (linear attention with a delta-rule memory).

    Projections: ``in_proj_qkv`` -> causal depthwise conv + SiLU -> q, k, v;
    ``in_proj_z`` -> output gate; ``in_proj_b`` -> write strength ``beta``;
    ``in_proj_a`` -> decay ``g = -exp(A_log) * softplus(a + dt_bias)``.
    """

    def __init__(self, config: DSMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16)))
        self.norm = DSMGatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        doc_starts: torch.Tensor | None = None,
        past_key_values=None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            hidden_states = (hidden_states * padding_mask[:, :, None]).to(hidden_states.dtype)
        batch, seq_len, _ = hidden_states.shape
        backend = resolve_backend(self.config.gdn_backend, hidden_states.device)
        has_state = cache_dsm.has_previous_state(past_key_values, self.layer_idx)

        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, C, T] raw conv inputs
        z = self.in_proj_z(hidden_states).view(batch, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        prefix = None
        if has_state:
            prefix = cache_dsm.read_conv_prefix(past_key_values, self.layer_idx, self.conv_kernel_size)
        if past_key_values is not None:
            cache_dsm.write_conv_state(past_key_values, self.layer_idx, mixed, self.conv_kernel_size)
        mixed = causal_conv1d(mixed, self.conv1d.weight.squeeze(1), prefix=prefix, doc_ids=doc_ids)
        mixed = mixed.transpose(1, 2)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        # computed in fp32: A_log may otherwise overflow in half precision
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if doc_starts is not None:
            g = torch.where(doc_starts[..., None], torch.full_like(g, self.config.doc_reset_log_decay), g)
        if self.num_v_heads > self.num_k_heads:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        initial_state = cache_dsm.read_recurrent_state(past_key_values, self.layer_idx) if has_state else None
        core, final_state = gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=past_key_values is not None,
            backend=backend,
            single_step=has_state and seq_len == 1,
        )
        if past_key_values is not None:
            cache_dsm.write_recurrent_state(past_key_values, self.layer_idx, final_state)

        core = self.norm(core, z).reshape(batch, seq_len, -1)
        return self.out_proj(core)


class DSMMLP(nn.Module):
    def __init__(self, config: DSMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        if config.hidden_act != "silu":
            raise NotImplementedError("DSM uses SwiGLU (hidden_act='silu')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DSMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = DSMGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = DSMAttention(config, layer_idx)
        self.mlp = DSMMLP(config)
        self.input_layernorm = DSMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DSMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        doc_starts: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states,
                padding_mask=padding_mask,
                doc_ids=doc_ids,
                doc_starts=doc_starts,
                past_key_values=past_key_values,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, cos, sin, attn_mask, past_key_values=past_key_values
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class DSMPreTrainedModel(PreTrainedModel):
    config: DSMConfig
    config_class = DSMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DSMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _is_stateful = True
    # The decay parameters and the GDN output norm are kept in fp32 even when the
    # model is loaded in bf16/fp16.
    _keep_in_fp32_modules_strict = ["A_log", "linear_attn.norm"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.", r"^model\.visual\."]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, DSMGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
        elif isinstance(module, DSMRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, DSMGatedRMSNorm):
            init.ones_(module.weight)


def packing_info(position_ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Document ids / starts for packed rows (a document starts where ``position_ids == 0``).

    Returns ``(None, None)`` when no row contains a second document, so the
    common unpacked case keeps the fast kernels.
    """
    starts = position_ids == 0
    if not bool(starts[:, 1:].any()):
        return None, None
    return starts.long().cumsum(dim=-1), starts


def build_attention_mask(
    padding_mask: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    batch: int,
    seq_len: int,
    past: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Boolean ``[B, 1, T, past+T]`` mask (True = may attend) or ``None`` for plain causal.

    Combines causality, key padding and (for packed rows) same-document
    constraints. Cached keys count as document 0. Rows that see no key at all
    (left padding) are fully unmasked so attention stays finite; their outputs
    are never used.
    """
    has_padding = padding_mask is not None and not bool(padding_mask.all())
    if not has_padding and doc_ids is None and (past == 0 or seq_len == 1):
        return None
    total = past + seq_len
    q_pos = torch.arange(seq_len, device=device) + past
    k_pos = torch.arange(total, device=device)
    mask = (k_pos[None, :] <= q_pos[:, None])[None, None].expand(batch, 1, seq_len, total)
    if has_padding:
        mask = mask & padding_mask[:, None, None, -total:].bool()
    if doc_ids is not None:
        key_docs = torch.cat([doc_ids.new_zeros(batch, past), doc_ids], dim=1)
        mask = mask & (key_docs[:, None, None, :] == doc_ids[:, None, :, None])
    return mask | ~mask.any(dim=-1, keepdim=True)


class DSMModel(DSMPreTrainedModel):
    def __init__(self, config: DSMConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DSMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DSMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DSMRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = cache_dsm.new_cache(self.config)
        if not use_cache:
            past_key_values = None
        past = cache_dsm.past_length(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past, past + seq_len, device=device).unsqueeze(0).expand(batch, -1)
        elif position_ids.ndim == 3:  # multimodal-style [rows, B, T]: rows are identical for text
            position_ids = position_ids[0]
        doc_ids, doc_starts = packing_info(position_ids)

        padding_mask = attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
        attn_mask = build_attention_mask(padding_mask, doc_ids, batch, seq_len, past, device)
        gdn_padding = None
        if padding_mask is not None and seq_len > 1 and not bool(padding_mask.all()):
            gdn_padding = padding_mask[:, -seq_len:].to(inputs_embeds.dtype)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        all_hidden: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer in self.layers:
            if all_hidden is not None:
                all_hidden += (hidden_states,)
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
                padding_mask=gdn_padding,
                doc_ids=doc_ids,
                doc_starts=doc_starts,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        hidden_states = self.norm(hidden_states)
        if all_hidden is not None:
            all_hidden += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden
        )


class DSMForCausalLM(DSMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DSMConfig) -> None:
        super().__init__(config)
        self.model = DSMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        num_items_in_batch: int | torch.Tensor | None = None,
        return_logits: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard causal-LM forward.

        With ``labels`` the loss is computed by a chunked cross-entropy and the
        full logits are NOT returned (pass ``return_logits=True`` to get them):
        for a 248K vocabulary they would cost ~1 GB per 1K tokens in fp32.
        ``num_items_in_batch`` normalises the summed loss by a global token
        count (correct gradient accumulation); default is the local count.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden = outputs.last_hidden_state

        loss = None
        if labels is not None:
            shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1).to(hidden.device)
            loss_sum = chunked_cross_entropy(
                shift_hidden, self.lm_head.weight, shift_labels, self.config.loss_chunk_size
            )
            denom = num_items_in_batch if num_items_in_batch is not None else (shift_labels != -100).sum()
            if isinstance(denom, torch.Tensor):
                denom = denom.to(loss_sum.device)
            loss = loss_sum / denom

        logits = None
        if labels is None or return_logits:
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden[:, slice_indices, :])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "DSMForCausalLM",
    "DSMModel",
    "DSMPreTrainedModel",
    "DSMDecoderLayer",
    "DSMGatedDeltaNet",
    "DSMAttention",
    "DSMRMSNorm",
    "DSMGatedRMSNorm",
    "build_attention_mask",
    "packing_info",
]
