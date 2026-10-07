"""Initialisation, serialisation, dtype handling, loss and Auto-class registration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from helpers import tiny_dsm_config, tiny_dsm_model

import dsm.modeling as dm


def test_fresh_init_follows_conventions() -> None:
    model = dm.DSMForCausalLM(tiny_dsm_config())
    for name, module in model.named_modules():
        if isinstance(module, dm.modeling_dsm.DSMRMSNorm):
            assert torch.count_nonzero(module.weight) == 0, f"{name} must be zero-centred (init 0)"
        if isinstance(module, dm.modeling_dsm.DSMGatedRMSNorm):
            assert torch.all(module.weight == 1), name
        if isinstance(module, dm.DSMGatedDeltaNet):
            assert torch.all(module.dt_bias == 1)
            a = module.A_log.exp()
            assert torch.all(a >= 0.01 - 1e-6) and torch.all(a <= 16 + 1e-4)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_save_and_reload_with_auto_classes(tmp_path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    model = tiny_dsm_model(seed=30)
    model.save_pretrained(tmp_path)
    cfg = AutoConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "dsm" and cfg.architectures == ["DSMForCausalLM"]
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).float().eval()
    reloaded.config._attn_implementation = "eager"
    assert isinstance(reloaded, dm.DSMForCausalLM)
    assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits)


def test_bf16_load_keeps_decay_and_gdn_norm_in_fp32(tmp_path) -> None:
    from transformers import AutoModelForCausalLM

    tiny_dsm_model(seed=31).save_pretrained(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    for name, param in model.named_parameters():
        if name.endswith("A_log") or "linear_attn.norm" in name:
            assert param.dtype == torch.float32, name
        else:
            assert param.dtype == torch.bfloat16, name
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 100, (1, 70))).logits
    assert torch.isfinite(out.float()).all()


def test_loss_matches_full_logits_cross_entropy() -> None:
    model = tiny_dsm_model(seed=32)
    ids = torch.randint(0, model.config.vocab_size, (2, 40))
    labels = ids.clone()
    labels[:, :10] = -100
    out = model(input_ids=ids, labels=labels, return_logits=True, use_cache=False)
    ref = F.cross_entropy(
        out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(out.loss, ref, atol=1e-5, rtol=1e-5)
    # global normalisation for gradient accumulation
    n_local = (labels[:, 1:] != -100).sum()
    out2 = model(input_ids=ids, labels=labels, num_items_in_batch=2 * n_local, use_cache=False)
    torch.testing.assert_close(out2.loss * 2, ref, atol=1e-5, rtol=1e-5)
    assert out2.logits is None  # full logits are not materialised when only the loss is needed


def test_gradient_checkpointing_gives_same_gradients() -> None:
    def grads(checkpointing: bool):
        model = tiny_dsm_model(seed=33)
        model.train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids = torch.randint(0, model.config.vocab_size, (2, 30), generator=torch.Generator().manual_seed(1))
        model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    plain, ckpt = grads(False), grads(True)
    assert plain.keys() == ckpt.keys()
    for name in plain:
        torch.testing.assert_close(ckpt[name], plain[name], atol=1e-5, rtol=1e-4, msg=name)


def test_parameter_count_of_full_config() -> None:
    """The default config is DSM1.0-0.8B: ~752M text parameters (embeddings tied)."""
    with torch.device("meta"):
        model = dm.DSMForCausalLM(dm.DSMConfig())
    total = sum(p.numel() for p in model.parameters())  # tied weights counted once
    assert math.isclose(total, 752_393_024, rel_tol=1e-6), total
