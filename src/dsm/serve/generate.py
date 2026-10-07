"""DSM's own decoding loop (greedy / sampling, streaming, penalties).

Independent of ``transformers.GenerationMixin`` so the CLI keeps working across
library versions. It drives the model exactly like ``generate()`` does: one
prefill forward with the hybrid cache, then one token per step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from dsm.modeling import cache_dsm


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 => greedy
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0  # multiplicative (HF semantics)
    presence_penalty: float = 0.0  # additive, once per distinct generated token (OpenAI semantics)
    eos_token_ids: Sequence[int] = ()
    seed: int | None = None


@dataclass
class GenerationResult:
    sequences: torch.Tensor  # [B, prompt + new]
    new_tokens: torch.Tensor  # [B, new]
    step_logits: list[torch.Tensor] = field(default_factory=list)  # each [B, V] (raw, before processing)


def prefill_position_ids(attention_mask: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
    """Positions for a (possibly left-padded) prompt; pads get position 0 like ``generate()``."""
    if attention_mask is None:
        return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def _process_logits(
    logits: torch.Tensor, generated: torch.Tensor | None, cfg: SamplingConfig
) -> torch.Tensor:
    logits = logits.float().clone()
    if (
        generated is not None
        and generated.numel()
        and (cfg.repetition_penalty != 1.0 or cfg.presence_penalty)
    ):
        for row in range(logits.shape[0]):
            seen = torch.unique(generated[row])
            if cfg.repetition_penalty != 1.0:
                vals = logits[row, seen]
                logits[row, seen] = torch.where(
                    vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty
                )
            if cfg.presence_penalty:
                logits[row, seen] -= cfg.presence_penalty
    return logits


def _sample(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None) -> torch.Tensor:
    if cfg.temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if cfg.min_p > 0:
        probs = probs.masked_fill(probs < cfg.min_p * probs.max(dim=-1, keepdim=True).values, 0.0)
    if cfg.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > cfg.top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    sampling: SamplingConfig | None = None,
    on_token: Callable[[torch.Tensor], None] | None = None,
    keep_logits: bool = False,
    pad_token_id: int | None = None,
) -> GenerationResult:
    """Decode up to ``sampling.max_new_tokens`` tokens.

    ``on_token`` receives the new token ids ``[B]`` after every step (streaming).
    Rows that produced an EOS token keep emitting ``pad_token_id``.
    """
    cfg = sampling or SamplingConfig()
    device = input_ids.device
    generator = torch.Generator(device=device).manual_seed(cfg.seed) if cfg.seed is not None else None
    eos = torch.tensor(list(cfg.eos_token_ids), device=device, dtype=torch.long)
    pad = pad_token_id if pad_token_id is not None else (int(eos[0]) if eos.numel() else 0)

    cache = cache_dsm.new_cache(model.config)
    position_ids = prefill_position_ids(attention_mask, input_ids)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    batch = input_ids.shape[0]
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    new_tokens: list[torch.Tensor] = []
    step_logits: list[torch.Tensor] = []
    last_pos = position_ids[:, -1]
    mask = attention_mask

    for _ in range(cfg.max_new_tokens):
        logits = out.logits[:, -1, :]
        if keep_logits:
            step_logits.append(logits.float().cpu())
        generated = torch.stack(new_tokens, dim=1) if new_tokens else None
        token = _sample(_process_logits(logits, generated, cfg), cfg, generator)
        token = torch.where(finished, torch.full_like(token, pad), token)
        new_tokens.append(token)
        if on_token is not None:
            on_token(token)
        if eos.numel():
            finished = finished | torch.isin(token, eos)
        if bool(finished.all()):
            break
        last_pos = last_pos + 1
        if mask is not None:
            mask = torch.cat([mask, mask.new_ones(batch, 1)], dim=1)
        out = model(
            input_ids=token[:, None],
            attention_mask=mask,
            position_ids=last_pos[:, None],
            past_key_values=cache,
            use_cache=True,
        )

    new = torch.stack(new_tokens, dim=1) if new_tokens else input_ids.new_zeros(batch, 0)
    return GenerationResult(
        sequences=torch.cat([input_ids, new], dim=1), new_tokens=new, step_logits=step_logits
    )
