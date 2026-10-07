"""Upstream checkpoint -> DSM checkpoint: weights, config traps, artifacts, audit, verification."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from helpers import upstream_ids
from upstream_fixture import build_tiny_upstream

from dsm.audit.offline import check_files, verify_manifest, write_manifest
from dsm.audit.scrub import scrub_model_dir
from dsm.convert.upstream import convert
from dsm.convert.verify import verify


@pytest.fixture(scope="module")
def converted(tmp_path_factory, tiny_tokenizer_json):
    root = tmp_path_factory.mktemp("convert")
    upstream_dir = build_tiny_upstream(root / "upstream", tiny_tokenizer_json, seed=3)
    report = convert(upstream_dir, root / "dsm")
    return upstream_dir, root / "dsm", report


def _reference_logits(upstream_dir, ids):
    import transformers

    arch = json.loads((upstream_dir / "config.json").read_text())["architectures"][0]
    ref = getattr(transformers, arch).from_pretrained(upstream_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return ref(input_ids=ids).logits


def test_files_and_config(converted) -> None:
    _, out, report = converted
    names = {p.name for p in out.iterdir()}
    for required in (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "README.md",
        "LICENSE",
        "NOTICE",
        "modeling_dsm.py",
    ):
        assert required in names, required
    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "dsm" and config["architectures"] == ["DSMForCausalLM"]
    assert config["tie_word_embeddings"] is True  # taken from the top-level config, not text_config
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_dsm.DSMForCausalLM"
    tok_cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert tok_cfg["tokenizer_class"] == "TokenizersBackend"
    assert "Doslarb" in (out / "chat_template.jinja").read_text()
    assert report.moved_to_extras == {"vision": report.moved_to_extras["vision"], "mtp": 2}
    assert (out.parent / "dsm-extras" / "mtp.safetensors").is_file()


def test_logits_match_upstream_reference(converted) -> None:
    from transformers import AutoModelForCausalLM

    import dsm.modeling  # noqa: F401

    upstream_dir, out, _ = converted
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"
    ids = torch.randint(0, 400, (2, 70), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ours = model(input_ids=ids).logits
    torch.testing.assert_close(ours, _reference_logits(upstream_dir, ids), atol=2e-4, rtol=1e-4)


def test_lineage_starts_at_conversion(converted) -> None:
    upstream_dir, dsm_dir, _ = converted
    (entry,) = json.loads((dsm_dir / "lineage.json").read_text())
    assert entry["stage"] == "convert" and entry["source_revision"] is None  # no FETCH.json here
    assert set(entry["source_weights_sha256"]) == {p.name for p in upstream_dir.glob("*.safetensors")}


def test_converted_directory_has_no_upstream_branding(converted) -> None:
    _, out, report = converted
    assert report.audit_findings == []
    assert scrub_model_dir(out) == []
    # the legally required attribution is present in NOTICE / LICENSE only
    up = upstream_ids()
    assert up["base_repo_id"] in (out / "NOTICE").read_text()
    assert up["copyright"] in (out / "LICENSE").read_text()


def test_scrub_detects_planted_branding(converted, tmp_path) -> None:
    import shutil

    _, out, _ = converted
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    up = upstream_ids()
    readme = copy / "README.md"
    readme.write_text(readme.read_text() + f"\nBased on {up['instruct_repo_id']}\n")
    findings = scrub_model_dir(copy)
    assert findings and all(f.path == "README.md" for f in findings)


def test_offline_audit_and_manifest(converted, tmp_path) -> None:
    import shutil

    _, out, _ = converted
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    assert check_files(copy) == []
    write_manifest(copy)
    assert verify_manifest(copy) == []
    (copy / "config.json").write_text((copy / "config.json").read_text() + " ")
    assert any("checksum" in p for p in verify_manifest(copy))
    (copy / "pytorch_model.bin").write_bytes(b"\x80\x04")
    assert any("pickle" in p for p in check_files(copy))


def test_verify_command_passes(converted) -> None:
    upstream_dir, out, _ = converted
    report = verify(out, upstream_dir)
    assert report["passed"], report


def test_bf16_conversion_keeps_decay_params_fp32(tmp_path, converted) -> None:
    from safetensors import safe_open

    upstream_dir, _, _ = converted
    convert(upstream_dir, tmp_path / "bf16", dtype="bf16", extras_dir=tmp_path / "extras")
    with safe_open(str(tmp_path / "bf16" / "model.safetensors"), "pt") as handle:
        for key in handle.keys():  # noqa: SIM118
            dtype = handle.get_slice(key).get_dtype()
            expected = "F32" if key.endswith(("A_log", "linear_attn.norm.weight")) else "BF16"
            assert dtype == expected, (key, dtype)


def test_remote_code_loads_without_the_dsm_package(converted, tmp_path) -> None:
    """The shipped modelling files must be self-contained (trust_remote_code on an offline box)."""
    from transformers import AutoModelForCausalLM

    import dsm.modeling  # noqa: F401

    _, out, _ = converted
    code = f"""
import sys, json
sys.modules["dsm"] = None  # forbid importing the dsm package
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained({str(out)!r}, trust_remote_code=True, dtype=torch.float32).eval()
with torch.no_grad():
    logits = model(input_ids=torch.tensor([[5, 9, 2, 7, 11, 3]])).logits
print(json.dumps(logits[0, -1, :8].tolist()))
"""
    env = dict(os.environ, HF_MODULES_CACHE=str(tmp_path / "modules"), HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr[-2000:]
    remote = torch.tensor(json.loads(result.stdout.strip().splitlines()[-1]))
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        local = model(input_ids=torch.tensor([[5, 9, 2, 7, 11, 3]])).logits[0, -1, :8]
    torch.testing.assert_close(remote, local, atol=1e-5, rtol=1e-5)


def test_convert_refuses_non_empty_output(converted) -> None:
    upstream_dir, out, _ = converted
    with pytest.raises(FileExistsError):
        convert(upstream_dir, out)
