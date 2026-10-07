"""DPO loss and trainer, checkpoint merging, preflight checks."""

from __future__ import annotations

import json
import math

import pytest
import torch

from dsm.train.config import config_from_dict
from dsm.train.dpo import DPOTrainer, dpo_loss
from dsm.train.merge import EMBEDDING, chat_vector, interpolate, load_tensors, write_merged
from dsm.train.preflight import estimate_memory_gb, run_checks

from .fixtures import dpo_file, tiny_model_dir


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, tiny_tokenizer_json):
    return tiny_model_dir(tmp_path_factory.mktemp("dpo") / "model", tiny_tokenizer_json)


def test_dpo_loss_value_and_gradient_direction():
    policy_c = torch.tensor([-10.0], requires_grad=True)
    policy_r = torch.tensor([-12.0], requires_grad=True)
    ref_c, ref_r = torch.tensor([-10.0]), torch.tensor([-12.0])
    losses, margins = dpo_loss(policy_c, policy_r, ref_c, ref_r, beta=0.1)
    assert losses.item() == pytest.approx(math.log(2)) and margins.item() == 0.0  # policy == reference
    losses.sum().backward()
    assert policy_c.grad.item() < 0 < policy_r.grad.item()  # descent raises chosen, lowers rejected
    better, _ = dpo_loss(torch.tensor([-9.0]), torch.tensor([-13.0]), ref_c, ref_r, beta=0.1)
    assert better.item() < math.log(2)
    smoothed, _ = dpo_loss(
        torch.tensor([-9.0]), torch.tensor([-13.0]), ref_c, ref_r, beta=0.1, label_smoothing=0.2
    )
    assert smoothed.item() > better.item()
    ipo, _ = dpo_loss(torch.tensor([-10.0]), torch.tensor([-12.0]), ref_c, ref_r, beta=0.1, loss_type="ipo")
    assert ipo.item() == pytest.approx((1 / 0.2) ** 2)


def test_dpo_trainer_learns_preferences(model_dir, tmp_path):
    cfg = config_from_dict(
        {
            "stage": "dpo",
            "model": str(model_dir),
            "output_dir": str(tmp_path / "run"),
            "seq_len": 256,
            "micro_batch_size": 4,
            "grad_accum": 1,
            "epochs": 3,
            "train": [{"path": str(dpo_file(tmp_path / "dpo.jsonl"))}],
            "precision": "fp32",
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "schedule": {"name": "constant", "warmup": 0},
            "log_every": 1,
            "save_every": 0,
            "resume": None,
        }
    )
    trainer = DPOTrainer(cfg, quiet=True)
    assert len(trainer.pairs) == 8 and trainer.total_steps == 6
    assert list((tmp_path / "run").glob("reference_logps-*.npy"))
    final = trainer.train()
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert rows[0]["loss"] == pytest.approx(math.log(2), abs=1e-4)
    assert rows[-1]["loss"] < 0.5 and rows[-1]["reward_accuracy"] == 1.0
    assert json.loads((final / "lineage.json").read_text())[-1]["stage"] == "dpo"


def _toy(seed, rows=6):
    gen = torch.Generator().manual_seed(seed)
    return {
        EMBEDDING: torch.randn(rows, 4, generator=gen),
        "model.norm.weight": torch.randn(4, generator=gen),
        "count": torch.tensor([seed]),
    }


def test_interpolate_endpoints_and_midpoint():
    a, b = _toy(1), _toy(2)
    assert torch.equal(interpolate(a, b, 0.0)[EMBEDDING], a[EMBEDDING])
    assert torch.allclose(interpolate(a, b, 1.0)[EMBEDDING], b[EMBEDDING])
    assert torch.allclose(
        interpolate(a, b, 0.5)["model.norm.weight"], (a["model.norm.weight"] + b["model.norm.weight"]) / 2
    )
    assert torch.equal(interpolate(a, b, 0.5)["count"], a["count"])  # integer tensors are not averaged
    with pytest.raises(ValueError, match="shape"):
        interpolate(a, _toy(3, rows=7), 0.5)


def test_chat_vector_vocab_aware():
    base, instruct = _toy(1), _toy(2)
    target = _toy(3, rows=8)  # extended vocabulary: 2 new rows
    merged = chat_vector(base, instruct, target, scale=1.0)
    delta = instruct[EMBEDDING] - base[EMBEDDING]
    assert torch.allclose(merged[EMBEDDING][:6], target[EMBEDDING][:6] + delta)
    assert torch.equal(merged[EMBEDDING][6:], target[EMBEDDING][6:])  # new rows untouched
    assert torch.allclose(
        merged["model.norm.weight"],
        target["model.norm.weight"] + instruct["model.norm.weight"] - base["model.norm.weight"],
    )
    skipped = chat_vector(base, instruct, target, skip_embeddings=True)
    assert torch.equal(skipped[EMBEDDING], target[EMBEDDING])


def test_write_merged_directory_loads(model_dir, tmp_path):
    from dsm.modeling import DSMForCausalLM

    tensors = load_tensors(model_dir)
    assert "lm_head.weight" not in tensors  # tied
    merged = interpolate(tensors, tensors, 0.3)
    out = write_merged(merged, model_dir, tmp_path / "merged", {"stage": "merge", "method": "interpolate"})
    assert {"tokenizer.json", "LICENSE", "NOTICE", "modeling_dsm.py", "config.json"} <= {
        p.name for p in out.iterdir()
    }
    assert json.loads((out / "lineage.json").read_text())[-1]["method"] == "interpolate"
    model = DSMForCausalLM.from_pretrained(out, dtype=torch.float32)
    assert torch.allclose(model.get_input_embeddings().weight, tensors[EMBEDDING])


def test_merge_cli(model_dir, tmp_path):
    from dsm.cli import main

    out = tmp_path / "cv"
    assert (
        main(
            [
                "merge",
                "chat-vector",
                "--base",
                str(model_dir),
                "--instruct",
                str(model_dir),
                "--target",
                str(model_dir),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert torch.equal(load_tensors(out)[EMBEDDING], load_tensors(model_dir)[EMBEDDING])  # zero delta


def test_preflight_on_cpu_and_memory_estimate(model_dir, tmp_path):
    checks = {c.name: c for c in run_checks(output_dir=str(tmp_path))}
    assert checks["python"].status == "PASS"
    if not torch.cuda.is_available():
        assert checks["gpu"].status == "FAIL"
    small = estimate_memory_gb(
        752_000_000, 752_000_000, seq_len=2048, micro_batch=1, hidden=1024, layers=24, vocab=248_320
    )
    large = estimate_memory_gb(
        752_000_000, 752_000_000, seq_len=8192, micro_batch=1, hidden=1024, layers=24, vocab=248_320
    )
    embeddings_only = estimate_memory_gb(
        752_000_000, 254_000_000, seq_len=2048, micro_batch=1, hidden=1024, layers=24, vocab=248_320
    )
    assert 12 < small < large < 24 and embeddings_only < small
    config = tmp_path / "cfg.yaml"
    config.write_text(
        json.dumps(
            {"stage": "sft", "model": str(model_dir), "output_dir": str(tmp_path / "o"), "train": ["x.jsonl"]}
        )
    )
    detail = {c.name: c for c in run_checks(str(config), str(tmp_path))}["memory estimate"].detail
    assert "params" in detail
