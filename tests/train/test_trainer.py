"""CPU smoke tests of the trainer: falling loss per stage, resume, token normalisation, embedding modes."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file
from tokenizer_fixtures import CORPUS

from dsm.train.config import config_from_dict
from dsm.train.trainer import LMTrainer

from .fixtures import packed_corpus, sft_file, tiny_model_dir


@pytest.fixture(scope="module")
def assets(tmp_path_factory, tiny_tokenizer_json):
    root = tmp_path_factory.mktemp("train_assets")
    return {
        "root": root,
        "model": tiny_model_dir(root / "model", tiny_tokenizer_json),
        "en": packed_corpus(
            root / "packed/en", tiny_tokenizer_json, [t for t in CORPUS if t.isascii()], "en"
        ),
        "other": packed_corpus(
            root / "packed/other", tiny_tokenizer_json, [t for t in CORPUS if not t.isascii()], "other"
        ),
        "sft": sft_file(root / "sft.jsonl", copies=2),
        "tokenizer_json": tiny_tokenizer_json,
    }


def _cfg(assets, tmp_path, **overrides):
    base = {
        "stage": "cpt",
        "model": str(assets["model"]),
        "output_dir": str(tmp_path / "run"),
        "seq_len": 64,
        "micro_batch_size": 2,
        "grad_accum": 2,
        "max_steps": 10,
        "train": [{"path": str(assets["en"]), "weight": 0.6}, {"path": str(assets["other"]), "weight": 0.4}],
        "precision": "fp32",
        "gradient_checkpointing": True,
        "optimizer": {"lr": 3e-3, "weight_decay": 0.0},
        "schedule": {"name": "cosine", "warmup": 2},
        "log_every": 1,
        "eval_every": 0,
        "save_every": 0,
        "resume": None,
    }
    base.update(overrides)
    return config_from_dict(base)


def _losses(run_dir):
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["loss"] for r in rows if "loss" in r]


def _falls(losses):
    return sum(losses[-3:]) / 3 < sum(losses[:3]) / 3 - 0.2


def test_cpt_smoke_and_export(assets, tmp_path):
    cfg = _cfg(assets, tmp_path, eval=[{"path": str(assets["en"]), "name": "en"}], eval_tokens=256)
    final = LMTrainer(cfg, quiet=True).train()
    assert _falls(_losses(tmp_path / "run"))
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert any("eval/en/bpb" in r for r in rows)
    names = {p.name for p in final.iterdir()}
    assert {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "LICENSE",
        "NOTICE",
        "modeling_dsm.py",
    } <= names
    lineage = json.loads((final / "lineage.json").read_text())
    assert lineage[-1]["stage"] == "cpt" and lineage[-1]["steps"] == 10
    assert set(lineage[-1]["epochs_per_source"]) == {"en", "other"}

    from dsm.modeling import DSMForCausalLM

    reloaded = DSMForCausalLM.from_pretrained(final, dtype=torch.float32)
    assert reloaded.config.use_cache is True


def test_sft_smoke(assets, tmp_path):
    cfg = _cfg(
        assets,
        tmp_path,
        stage="sft",
        seq_len=256,
        epochs=6,
        max_steps=None,
        train=[{"path": str(assets["sft"])}],
        schedule={"name": "constant", "warmup": 1},
    )
    LMTrainer(cfg, quiet=True).train()
    assert _falls(_losses(tmp_path / "run"))


def test_pt_from_scratch(assets, tmp_path):
    model_cfg = json.loads((assets["model"] / "config.json").read_text())
    keep = {
        k: model_cfg[k]
        for k in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "layer_types",
            "rope_parameters",
            "eos_token_id",
            "pad_token_id",
            "gdn_backend",
        )
    }
    yaml_path = tmp_path / "nano.yaml"
    yaml_path.write_text(json.dumps(keep))
    cfg = _cfg(
        assets, tmp_path, stage="pt", model=None, model_config=str(yaml_path), tokenizer=str(assets["model"])
    )
    final = LMTrainer(cfg, quiet=True).train()
    assert _falls(_losses(tmp_path / "run"))
    assert (final / "LICENSE").is_file()  # carried from the tokenizer directory


def test_resume_is_exact(assets, tmp_path):
    straight = LMTrainer(_cfg(assets, tmp_path / "a", max_steps=6), quiet=True)
    straight.train()
    reference = load_file(str(tmp_path / "a/run/final/model.safetensors"))

    first = LMTrainer(_cfg(assets, tmp_path / "b", max_steps=6), quiet=True)
    first.train(bench_steps=3)
    first.save_checkpoint()
    second = LMTrainer(_cfg(assets, tmp_path / "b", max_steps=6, resume="auto"), quiet=True)
    second.train()
    assert second.step == 6
    resumed = load_file(str(tmp_path / "b/run/final/model.safetensors"))
    for key, value in reference.items():
        torch.testing.assert_close(resumed[key], value, rtol=0, atol=1e-6)


def test_global_token_normalisation(assets, tmp_path):
    """Accumulating uneven micro-batches gives the gradient of the whole batch."""
    trainer = LMTrainer(
        _cfg(
            assets,
            tmp_path,
            stage="sft",
            seq_len=256,
            epochs=1,
            max_steps=None,
            train=[{"path": str(assets["sft"])}],
            gradient_checkpointing=False,
        ),
        quiet=True,
    )
    data = trainer.data
    small, big = data.batch([0]), data.batch([1, 2, 3])
    total = trainer.normaliser([small, big])

    def grads(batches):
        trainer.model.zero_grad(set_to_none=True)
        for batch in batches:
            loss, _ = trainer.loss(batch, total)
            loss.backward()
        return [p.grad.clone() for p in trainer.trainable]

    accumulated = grads([small, big])
    joint = {k: torch.cat([small[k], big[k]]) for k in small}
    together = grads([joint])
    for a, b in zip(accumulated, together, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_new_rows_mode_only_moves_new_embeddings(assets, tmp_path):
    cfg = _cfg(assets, tmp_path, trainable="new_rows", new_rows_from=300, max_steps=4)
    trainer = LMTrainer(cfg, quiet=True)
    before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    trainer.train()
    after = trainer.model.state_dict()
    embedding = "model.embed_tokens.weight"
    assert torch.equal(after[embedding][:300], before[embedding][:300])
    assert not torch.equal(after[embedding][300:], before[embedding][300:])
    for key, value in before.items():
        if key not in (embedding, "lm_head.weight"):
            assert torch.equal(after[key], value), key


def test_frozen_embeddings_mode(assets, tmp_path):
    trainer = LMTrainer(_cfg(assets, tmp_path, trainable="no_embeddings", max_steps=2), quiet=True)
    before = trainer.model.get_input_embeddings().weight.detach().clone()
    trainer.train()
    assert torch.equal(trainer.model.get_input_embeddings().weight, before)


def test_cli_dry_run(assets, tmp_path, capsys):
    from dsm.cli import main

    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(_cfg(assets, tmp_path).to_dict()))
    assert main(["train", "--config", str(path), "--dry-run", "--set", "max_steps=3"]) == 0
    assert "stage cpt: 3 steps" in capsys.readouterr().out
