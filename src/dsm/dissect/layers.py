"""Layer importance on real text, per language.

* **Block influence** (ShortGPT): ``1 - cos(input, output)`` of each decoder
  layer, averaged over tokens. Low values = the layer barely changes the
  residual stream (a pruning candidate).
* **Ablation**: the loss when one layer is skipped (its output replaced by its
  input). Large increases mark indispensable layers.
* **Logit lens**: the loss when the final norm + LM head read the residual
  stream after layer ``i``, i.e. how early predictions take shape.

Comparing Khmer with English / Chinese shows whether a language relies on
different depths, which matters before pruning or quantising a layer.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def _loss(model, batch) -> float:
    out = model(**batch, use_cache=False)
    return float(out.loss)


@torch.no_grad()
def block_influence(model, batches: list[dict]) -> list[float]:
    layers = model.model.layers
    sums = [0.0] * len(layers)
    counts = [0] * len(layers)
    handles = []

    def hook(index):
        def fn(module, args, kwargs, output):
            hidden_in = args[0] if args else kwargs["hidden_states"]
            hidden_out = output[0] if isinstance(output, tuple) else output
            cos = F.cosine_similarity(hidden_in.float(), hidden_out.float(), dim=-1)
            sums[index] += float((1 - cos).sum())
            counts[index] += cos.numel()

        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(i), with_kwargs=True))
    try:
        for batch in batches:
            model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [s / max(c, 1) for s, c in zip(sums, counts, strict=True)]


@contextlib.contextmanager
def _skip(layer):
    original = layer.forward

    def identity(hidden_states, *args, **kwargs):
        return hidden_states

    layer.forward = identity
    try:
        yield
    finally:
        layer.forward = original


@torch.no_grad()
def layer_ablation(model, batches: list[dict]) -> tuple[float, list[float]]:
    """(baseline loss, loss with layer i skipped for every i)."""
    baseline = sum(_loss(model, b) for b in batches) / len(batches)
    ablated = []
    for layer in model.model.layers:
        with _skip(layer):
            ablated.append(sum(_loss(model, b) for b in batches) / len(batches))
    return baseline, ablated


@torch.no_grad()
def logit_lens(model, batches: list[dict]) -> list[float]:
    """Next-token loss read out after each layer (last entry = the real model output)."""
    totals, count = None, 0
    for batch in batches:
        out = model.model(
            **{k: v for k, v in batch.items() if k != "labels"}, output_hidden_states=True, use_cache=False
        )
        labels = batch["labels"][:, 1:].reshape(-1)
        keep = labels != -100
        losses = []
        for hidden in out.hidden_states[1:]:
            normed = hidden if hidden is out.hidden_states[-1] else model.model.norm(hidden)
            logits = model.lm_head(normed[:, :-1]).float().reshape(-1, model.config.vocab_size)
            losses.append(float(F.cross_entropy(logits[keep], labels[keep], reduction="sum")))
        totals = losses if totals is None else [a + b for a, b in zip(totals, losses, strict=True)]
        count += int(keep.sum())
    return [t / max(count, 1) for t in totals or []]
