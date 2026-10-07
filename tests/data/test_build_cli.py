"""``dsm build`` end to end on the seed builders and a converted public dataset."""

from __future__ import annotations

import json

import pytest

from dsm.cli import main as dsm_main
from dsm.utils import read_jsonl, write_jsonl

from . import dataset_rows as R


def test_build_seeds_and_validate(tmp_path, capsys):
    out = tmp_path / "sft"
    assert dsm_main(["build", "seeds", "--out", str(out), "--per-weight", "1"]) == 0
    files = sorted(out.rglob("*.jsonl.zst"))
    names = {f.name for f in files}
    assert {
        "identity_sft.jsonl.zst",
        "identity_docs.jsonl.zst",
        "identity_dpo.jsonl.zst",
        "tools.jsonl.zst",
        "tables.jsonl.zst",
    } <= names
    assert {"formatting.jsonl.zst", "rag.jsonl.zst", "safety_sft.jsonl.zst", "safety_dpo.jsonl.zst"} <= names
    capsys.readouterr()
    assert dsm_main(["build", "validate", *map(str, files)]) == 0
    assert "0 problems" in capsys.readouterr().out


def test_build_validate_reports_problems(tmp_path):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(
        bad,
        [
            {
                "id": "x",
                "source": "s",
                "license": "l",
                "generated_by": "g",
                "lang": "en",
                "category": "chat",
                "messages": [],
            }
        ],
    )
    assert dsm_main(["build", "validate", str(bad)]) == 1


def test_build_convert_and_preference(tmp_path):
    raw = tmp_path / "xlam.json"
    raw.write_text(json.dumps([R.XLAM, {"query": "broken"}]), encoding="utf-8")
    out = tmp_path / "xlam.jsonl.zst"
    assert dsm_main(["build", "convert", "--dataset", "xlam", "--input", str(raw), "--out", str(out)]) == 0
    records = list(read_jsonl(out))
    assert len(records) == 1 and records[0]["license"] == "CC-BY-4.0"
    with pytest.raises(SystemExit, match="licence"):
        dsm_main(
            ["build", "convert", "--dataset", "chat", "--input", str(raw), "--out", str(tmp_path / "c.jsonl")]
        )

    # --source names a catalogue entry: licence and producer come from it, through the licence gate
    chat = tmp_path / "chat.jsonl"
    chat.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "smoltalk2.jsonl"
    args = ["build", "convert", "--dataset", "chat", "--input", str(chat), "--out", str(out)]
    assert dsm_main([*args, "--source", "smoltalk2"]) == 0
    (record,) = read_jsonl(out)
    assert record["license"] == "Apache-2.0" and record["source"] == "smoltalk2"
    with pytest.raises(SystemExit, match="licence gate: pku_saferlhf"):
        dsm_main([*args, "--source", "pku_saferlhf"])
    with pytest.raises(SystemExit, match="counsel"):
        dsm_main([*args, "--source", "numinamath_1_5"])

    tables = tmp_path / "tables.jsonl"
    assert dsm_main(["build", "tables", "--out", str(tables), "--per-weight", "1"]) == 0
    pairs = tmp_path / "format_dpo.jsonl"
    assert (
        dsm_main(["build", "preference", "--kind", "format", "--input", str(tables), "--out", str(pairs)])
        == 0
    )
    assert all("chosen" in p for p in read_jsonl(pairs))
