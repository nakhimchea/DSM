"""The runbook end to end on CPU: every script in ``scripts/`` runs exactly as on the GPU machine,
on a tiny fake upstream checkpoint and tiny fake downloads, with tiny budgets.

fake fetch -> 20 convert / verify -> 30 pre-training data -> 35 SFT / DPO data -> 40 fast path
-> 45 tokenizer pilot -> 50 CPT -> 55 anneal -> 60 SFT (chat vector) -> 70 DPO + merge
-> (80 GGUF export, with DSM_LLAMA_CPP) -> dsm chat.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import ROOT

pytestmark = pytest.mark.slow

SCRIPTS = ROOT / "scripts"
PRETRAIN = " ".join(
    f"--set {item}"
    for item in (
        "max_steps=2",
        "seq_len=64",
        "micro_batch_size=2",
        "grad_accum=1",
        "precision=fp32",
        "eval_every=1",
        "eval_tokens=256",
        "save_every=0",
        "device=cpu",
    )
)
CHAT = " ".join(
    f"--set {item}"
    for item in (
        "max_steps=2",
        "seq_len=2048",
        "micro_batch_size=1",
        "grad_accum=1",
        "precision=fp32",
        "save_every=0",
        "device=cpu",
    )
)


@pytest.fixture(scope="module")
def home(tmp_path_factory, tiny_tokenizer_json):
    from e2e.fake_downloads import build

    home = tmp_path_factory.mktemp("dsm_home")
    build(home, ROOT, tiny_tokenizer_json)
    return home


@pytest.fixture(scope="module")
def run(home):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    base = {
        **os.environ,
        "DSM_HOME": str(home),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "HELDOUT": "0.25",
        "GATES": "warn",  # tiny random models fail the identity gates by design
        "EVAL_ARGS": "--suites identity,format --max-new-tokens 4 --batch-size 64 --device cpu",
        "CPT_TOKENS": "3000",
        "ANNEAL_TOKENS": "2000",
        "SFT_TOTAL": "64",
        "DPO_TOTAL": "24",
        "EXPORT": "0",
        "PILOT_TOKENS_A": "1e4",
        "NEW_TOKENS": "64",
        "KM_HQ": "--min chars=300 khmer_share=0.5",  # the fake documents are short
        "ZH_HQ": "--min chars=200",
        "REPLAY": "--sample 0.5 --seed 1",
    }

    def run(script: str, **env) -> str:
        result = subprocess.run(
            ["bash", str(SCRIPTS / script)],
            env={**base, **env},
            capture_output=True,
            text=True,
            errors="replace",  # llama.cpp tools print raw bytes
            timeout=1800,
        )
        log = result.stdout + result.stderr
        assert result.returncode == 0, f"{script} failed:\n{log[-6000:]}"
        return log

    return run


def test_runbook_end_to_end(home, run):
    work, data, results = home / "work", home / "datasets", home / "results"

    log = run("20_convert.sh")
    assert (work / "models/dsm-base/model.safetensors").is_file() and (results / "dissect").is_dir()
    assert json.loads((results / "instruct/metrics.json").read_text())

    log = run("30_pretrain_data.sh")
    for name in ("km_web", "eval_km", "en_web_edu", "eval_en", "zh_web", "eval_zh", "parallel", "math"):
        assert (data / f"packed/{name}.bin").is_file(), name
    for name in ("identity_docs", "tables", "km_hq", "en_hq", "replay_mix"):
        assert (data / f"packed/{name}.bin").is_file(), name
    assert (data / "eval/flores_pairs.jsonl").is_file() and (data / "eval/belebele.jsonl").is_file()
    assert (data / "eval/safety_xstest.jsonl").is_file()
    assert (data / "sft/khmer/alt_km_en.jsonl.zst").is_file()
    assert "km_wiki_news" in log and "not packed" in log  # sources without data are left out
    assert (work / "mix/cpt.yaml").is_file()

    log = run("35_sft_data.sh")
    for path in ("chat/smoltalk2", "tools/xlam", "tools/massive_km", "math/gsm8k", "tables/finqa"):
        assert (data / f"sft/{path}.jsonl.zst").is_file(), path
    assert (data / "dpo/helpsteer3.jsonl.zst").is_file() and (data / "dpo/format.jsonl.zst").is_file()
    assert (work / "mix/sft.yaml").is_file()

    run("40_fastpath.sh", TRAIN_ARGS=CHAT)
    assert (work / "runs/fastpath_sft/final/model.safetensors").is_file()
    assert json.loads((results / "fastpath/metrics.json").read_text())

    log = run("45_tokenizer_pilot.sh", TRAIN_ARGS=PRETRAIN)
    assert (work / "models/dsm-base-km-ext/extension.json").is_file()
    assert "bpb/km" in log and (results / "pilot_B/bpb.json").is_file()

    log = run("50_cpt.sh", TRAIN_ARGS=PRETRAIN)
    assert "gate after_cpt" in log and ("[PASS]" in log or "[FAIL]" in log)
    assert (results / "cpt/bpb.json").is_file()

    run("55_anneal.sh", TRAIN_ARGS=PRETRAIN)
    assert (work / "runs/anneal/final/model.safetensors").is_file()

    run("60_sft.sh", TRAIN_ARGS=CHAT, CHAT_VECTOR="1")
    assert (work / "models/anneal-cv").is_dir() and (results / "sft/metrics.json").is_file()

    run("70_dpo.sh", TRAIN_ARGS=CHAT)
    final = work / "models/dsm-final"
    stages = [entry.get("stage") for entry in json.loads((final / "lineage.json").read_text())]
    assert stages[:2] == ["convert", "cpt"] and "anneal" in stages and "sft" in stages, stages
    assert stages[-1] == "merge"
    assert json.loads((results / "final/metrics.json").read_text())

    if os.environ.get("DSM_LLAMA_CPP"):
        run("80_export.sh", QUANT_SUITES="0")
        assert list((home / "exports/dsm").glob("*-Q8_0.gguf"))

    chat = subprocess.run(
        ["dsm", "chat", "--model", str(final), "--max-new-tokens", "8", "--device", "cpu"],
        input="Hello\n/quit\n",
        env={**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert chat.returncode == 0 and "DSM chat" in chat.stdout, chat.stdout + chat.stderr


def test_scripts_are_valid_bash():
    for script in sorted(SCRIPTS.glob("*.sh")):
        assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0, script
    if shutil.which("shellcheck"):
        result = subprocess.run(
            ["shellcheck", "-x", "-e", "SC2086,SC1091", *map(str, sorted(SCRIPTS.glob("*.sh")))],
            capture_output=True,
            text=True,
            cwd=SCRIPTS,
        )
        assert result.returncode == 0, result.stdout


def test_scripts_call_existing_commands_and_configs():
    """Every `$DSM <command> [<subcommand>]` and every configs/ path in the scripts exists."""
    import re

    from dsm.cli import COMMANDS, main

    seen = set()
    for script in SCRIPTS.glob("*.sh"):
        text = script.read_text()
        for path in re.findall(r"configs/[\w/.-]+\.yaml", text):
            assert (ROOT / path).is_file(), f"{script.name}: {path}"
        for command, sub in re.findall(r"\$DSM (\w[\w-]*)(?: (\w[\w-]*))?", text):
            assert command in COMMANDS, f"{script.name}: dsm {command}"
            seen.add((command, sub))
    for command, sub in sorted(seen):
        args = [command, sub, "--help"] if sub and not sub.startswith("-") else [command, "--help"]
        try:
            code = main(args)
        except SystemExit as exc:
            code = exc.code
        assert code in (0, None), args
