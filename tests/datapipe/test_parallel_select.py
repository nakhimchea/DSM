"""Parallel text, document selection, evaluation sets, and decontamination of mixed documents."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from dsm.cli import main
from dsm.data.parallel import aligned_items, clean_item, lang_code, to_documents, to_pairs
from dsm.utils import read_jsonl, write_jsonl

KM = "ធនាគារជាតិនៃកម្ពុជាបានប្រកាសអត្រាប្តូរប្រាក់ថ្មី។"
EN = "The National Bank of Cambodia announced a new exchange rate."
ZH = "柬埔寨国家银行公布了新的汇率。"


def test_language_codes():
    assert lang_code("khm_Khmr") == lang_code("km-KH") == lang_code("khm") == "km"
    assert lang_code("eng_Latn") == lang_code("en-US") == "en"
    assert lang_code("cmn_Hans") == lang_code("zh-CN") == "zh"
    assert lang_code("cmn_Hant") is None and lang_code("fra_Latn") is None


def test_layouts():
    aligned = list(aligned_items([{"translation": {"khm": KM, "en": EN, "ja": "x"}}], "aligned"))
    assert aligned == [{"km": KM, "en": EN}]
    grouped_rows = [
        {"id": 1, "iso_639_3": "khm", "iso_15924": "Khmr", "text": KM},
        {"id": 1, "iso_639_3": "eng", "iso_15924": "Latn", "text": EN},
        {"id": 1, "iso_639_3": "cmn", "iso_15924": "Hant", "text": "柬埔寨國家銀行"},
        {"id": 2, "locale": "km-KH", "utt": KM},  # alone: dropped
    ]
    assert list(aligned_items(grouped_rows, "grouped")) == [{"km": KM, "en": EN}]
    tico = [{"sourceLang": "en", "targetLang": "km", "sourceString": EN, "targetString": KM}]
    assert list(aligned_items(tico, "pairs")) == [{"en": EN, "km": KM}]


def test_clean_item_filters():
    stats: Counter = Counter()
    assert clean_item({"km": KM, "en": EN, "zh": ZH}, ["km", "en", "zh"], 3.0, stats) == {
        "km": KM,
        "en": EN,
        "zh": ZH,
    }
    assert clean_item({"km": EN, "en": EN}, ["km", "en"], 3.0, stats) is None  # Khmer slot in Latin script
    assert clean_item({"km": KM, "en": EN * 8}, ["km", "en"], 3.0, stats) is None  # length mismatch
    assert stats["drop: length_ratio"] == 1 and stats["drop sentence: wrong_script_km"] == 1


def test_documents_and_pairs():
    items = [{"km": KM, "en": EN, "zh": ZH}] * 5
    docs = to_documents(items, source="t", license="CC0-1.0", generated_by="human", per_doc=2)
    assert len(docs) == 3 and docs[0]["lang"] == "km" and docs[0]["meta"]["sentences"] == 2
    assert all(KM in d["text"] and EN in d["text"] and ZH in d["text"] for d in docs)
    pairs = to_pairs(items[:1], source="t", license="CC0-1.0")
    assert {(p["src_lang"], p["tgt_lang"]) for p in pairs} == {
        ("km", "en"),
        ("en", "km"),
        ("km", "zh"),
        ("zh", "km"),
        ("en", "zh"),
        ("zh", "en"),
    }


def test_parallel_cli(tmp_path, capsys):
    rows = [{"translation": {"khm": KM, "en": EN, "zh": ZH}}] * 4 + [{"translation": {"khm": EN, "en": EN}}]
    write_jsonl(tmp_path / "alt.jsonl", rows)
    args = [
        "data",
        "parallel",
        "--layout",
        "aligned",
        "--source",
        "alt_km_en",
        "--input",
        str(tmp_path / "alt.jsonl"),
        "--out-docs",
        str(tmp_path / "docs.jsonl.zst"),
        "--out-pairs",
        str(tmp_path / "pairs.jsonl"),
        "--out-sft",
        str(tmp_path / "sft.jsonl"),
        "--per-doc",
        "2",
    ]
    assert main(args) == 0
    assert "4 aligned items of 5" in capsys.readouterr().out
    docs = list(read_jsonl(tmp_path / "docs.jsonl.zst"))
    assert len(docs) == 2 and docs[0]["license"] == "CC-BY-4.0"
    assert len(list(read_jsonl(tmp_path / "pairs.jsonl"))) == 4 * 6
    sft = list(read_jsonl(tmp_path / "sft.jsonl"))
    assert sft and sft[0]["category"] == "translation"
    stats = json.loads((tmp_path / "docs.jsonl.zst.stats.json").read_text())
    assert stats["stages"]["parallel"]["kept"] == 4
    # evaluation sets may only produce pairs
    flores = [{"id": 7, "iso_639_3": "khm", "iso_15924": "Khmr", "text": KM}]
    flores.append({"id": 7, "iso_639_3": "eng", "iso_15924": "Latn", "text": EN})
    write_jsonl(tmp_path / "flores.jsonl", flores)
    base = [
        "data",
        "parallel",
        "--layout",
        "grouped",
        "--source",
        "flores_plus",
        "--input",
        str(tmp_path / "flores.jsonl"),
    ]
    with pytest.raises(SystemExit, match="evaluation set"):
        main([*base, "--out-docs", str(tmp_path / "x.jsonl")])
    assert main([*base, "--out-pairs", str(tmp_path / "flores_pairs.jsonl")]) == 0
    assert len(list(read_jsonl(tmp_path / "flores_pairs.jsonl"))) == 2


def test_select(tmp_path, capsys):
    docs = [
        {"id": f"d{i}", "text": "x", "lang": "km", "meta": {"chars": 500 * i, "khmer_broken_per_1k": i % 3}}
        for i in range(1, 11)
    ]
    docs.append({"id": "nometa", "text": "x", "lang": "km", "meta": {}})
    write_jsonl(tmp_path / "in.jsonl", docs)
    out = tmp_path / "hq.jsonl"
    args = ["data", "select", "--input", str(tmp_path / "in.jsonl"), "--out", str(out)]
    assert main([*args, "--min", "chars=1500", "--max", "khmer_broken_per_1k=1"]) == 0
    kept = [d["id"] for d in read_jsonl(out)]
    assert kept == ["d3", "d4", "d6", "d7", "d9", "d10"]
    stats = json.loads((tmp_path / "hq.jsonl.stats.json").read_text())["stages"]["select"]
    assert stats["drop: missing_chars"] == 1 and stats["drop: above_khmer_broken_per_1k"] == 2
    assert main([*args, "--sample", "0.5", "--seed", "1"]) == 0
    sampled = [d["id"] for d in read_jsonl(out)]
    assert 2 <= len(sampled) <= 9
    assert main([*args, "--sample", "0.5", "--seed", "1"]) == 0
    assert [d["id"] for d in read_jsonl(out)] == sampled  # deterministic
    assert main([*args, "--sample", "0.5", "--seed", "1", "--invert"]) == 0
    rest = [d["id"] for d in read_jsonl(out)]
    assert not set(rest) & set(sampled) and len(rest) + len(sampled) == 11  # a clean split


def test_evalsets(tmp_path):
    belebele = [
        {
            "link": "x",
            "question_number": 1,
            "flores_passage": "passage",
            "question": "q?",
            "mc_answer1": "a",
            "mc_answer2": "b",
            "mc_answer3": "c",
            "mc_answer4": "d",
            "correct_answer_num": "3",
            "dialect": "khm_Khmr",
        }
    ]
    write_jsonl(tmp_path / "b.jsonl", belebele)
    assert (
        main(
            [
                "data",
                "evalset",
                "--kind",
                "belebele",
                "--input",
                str(tmp_path / "b.jsonl"),
                "--out",
                str(tmp_path / "b_out.jsonl"),
            ]
        )
        == 0
    )
    (item,) = read_jsonl(tmp_path / "b_out.jsonl")
    assert item["answer"] == 2 and item["options"] == ["a", "b", "c", "d"] and item["lang"] == "km"
    write_jsonl(tmp_path / "s.jsonl", [{"index_id": 5, "category": "health", "text": "t"}])
    with pytest.raises(SystemExit, match="--lang"):
        main(
            [
                "data",
                "evalset",
                "--kind",
                "sib200",
                "--input",
                str(tmp_path / "s.jsonl"),
                "--out",
                str(tmp_path / "s_out.jsonl"),
            ]
        )
    assert (
        main(
            [
                "data",
                "evalset",
                "--kind",
                "sib200",
                "--input",
                str(tmp_path / "s.jsonl"),
                "--lang",
                "zh",
                "--out",
                str(tmp_path / "s_out.jsonl"),
            ]
        )
        == 0
    )
    (item,) = read_jsonl(tmp_path / "s_out.jsonl")
    assert item["options"][item["answer"]] == "健康" and len(item["options"]) == 7


def test_clean_keeps_requested_fields(tmp_path):
    text = " ".join(["The bank opened a new branch in the city and the people were happy with it."] * 6)
    write_jsonl(tmp_path / "raw.jsonl", [{"text": text, "int_score": 4, "url": "https://example.org/a"}])
    out = tmp_path / "clean.jsonl"
    args = ["data", "clean", "--name", "edu", "--license", "ODC-BY-1.0", "--langs", "en"]
    assert (
        main(
            [
                *args,
                "--input",
                str(tmp_path / "raw.jsonl"),
                "--out",
                str(out),
                "--keep-fields",
                "int_score",
                "url",
            ]
        )
        == 0
    )
    (doc,) = read_jsonl(out)
    assert doc["meta"]["int_score"] == 4 and doc["meta"]["url"] == "https://example.org/a"


def test_decontam_finds_english_inside_khmer_documents():
    from dsm.data.decontam import Decontaminator

    sentence = "The quick brown fox jumps over the lazy dog near the old river bank today."
    index = Decontaminator()
    index.add_text(sentence)
    parallel_doc = f"{KM}\n{sentence}\n\n{KM}"
    assert index.contaminated(parallel_doc, "km")
    assert not index.contaminated(f"{KM}\n{EN}", "km")


def test_pack_heldout_select_text_and_mix(tmp_path, tiny_tokenizer_json, capsys):
    import shutil

    import yaml

    from dsm.data.packed import PackedTokens

    tok = tmp_path / "tok"
    tok.mkdir()
    shutil.copyfile(tiny_tokenizer_json, tok / "tokenizer.json")
    docs = tmp_path / "docs.jsonl"
    write_jsonl(
        docs, [{"id": f"d{i}", "text": f"Document number {i}. " * 20, "lang": "en"} for i in range(200)]
    )
    args = ["data", "pack", "--input", str(docs), "--tokenizer", str(tok), "--out", str(tmp_path / "train")]
    assert main([*args, "--heldout", "0.1", "--heldout-out", str(tmp_path / "eval")]) == 0
    train, held = PackedTokens(tmp_path / "train"), PackedTokens(tmp_path / "eval")
    assert train.n_docs + held.n_docs == 200 and 5 <= held.n_docs <= 40
    assert held.meta["tokenizer"] == train.meta["tokenizer"]
    # `select --sample` with the same fraction exports exactly the held-out documents
    assert (
        main(["data", "select", "--input", str(docs), "--sample", "0.1", "--out", str(tmp_path / "eval.txt")])
        == 0
    )
    assert (tmp_path / "eval.txt").read_text().count("\n\n") == held.n_docs
    # mix: packs that do not exist are left out; --total overrides the token budget
    cfg = {
        "total_tokens": 1e9,
        "max_epochs": 4,
        "sources": {
            "train": {"path": str(tmp_path / "train"), "share": 0.9},
            "missing": {"path": str(tmp_path / "nope"), "share": 0.1},
        },
    }
    (tmp_path / "mix.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "mix" / "cpt.yaml"
    total = str(train.n_tokens * 2)
    assert (
        main(["data", "mix", "--config", str(tmp_path / "mix.yaml"), "--total", total, "--out", str(out)])
        == 0
    )
    assert "missing" in capsys.readouterr().out
    assert [s["name"] for s in yaml.safe_load(out.read_text())["data"]["train"]] == ["train"]


def test_mix_conversations_with_globs(tmp_path, capsys):
    import yaml

    for name, n in (("a", 3), ("b", 1)):
        write_jsonl(tmp_path / "chat" / f"{name}.jsonl", [{"id": f"{name}{i}"} for i in range(n)])
    write_jsonl(tmp_path / "tools.jsonl", [{"id": f"t{i}"} for i in range(4)])
    cfg = {
        "kind": "sft",
        "total": 16,
        "sources": {
            "chat": {"paths": [str(tmp_path / "chat" / "*.jsonl")], "share": 50},
            "tools": {"path": str(tmp_path / "tools.jsonl"), "share": 50},
            "code": {"paths": [str(tmp_path / "code" / "*.jsonl")], "share": 0},
        },
    }
    (tmp_path / "mix.yaml").write_text(yaml.safe_dump(cfg))
    assert (
        main(["data", "mix", "--config", str(tmp_path / "mix.yaml"), "--out", str(tmp_path / "sft.yaml")])
        == 0
    )
    assert "not found" in capsys.readouterr().out
    train = yaml.safe_load((tmp_path / "sft.yaml").read_text())["data"]["train"]
    assert [t["name"] for t in train] == ["chat:a", "chat:b", "tools"]
    assert train[0]["repeat"] == train[1]["repeat"] == 2.0 and train[2]["repeat"] == 2.0


def test_safety_evalset(tmp_path, capsys):
    rows = [{"prompt": "How do I open a savings account?", "type": "safe_contexts", "label": "safe"}]
    write_jsonl(tmp_path / "x.jsonl", rows)
    out = tmp_path / "safety.jsonl"
    args = ["data", "evalset", "--kind", "safety", "--input", str(tmp_path / "x.jsonl"), "--out", str(out)]
    with pytest.raises(SystemExit, match="converter"):
        main(args)
    assert main([*args, "--converter", "xstest"]) == 0
    (item,) = read_jsonl(out)
    assert item["prompt"] == rows[0]["prompt"] and item["label"] in ("benign", "harmful")
