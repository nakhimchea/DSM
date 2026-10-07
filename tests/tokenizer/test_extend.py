"""Khmer vocabulary extension: no existing id moves, non-Khmer text encodes identically, Khmer
gets shorter, HF's BPE equals the learner's replay, and the model gets matching rows."""

from __future__ import annotations

import json
import re
import shutil

import pytest
import torch
from tokenizer_fixtures import CORPUS

from dsm.text import normalize_text
from dsm.tokenizer.extend import (
    Extension,
    _byte_maps,
    apply_extension,
    encode_with_extension,
    extend_tokenizer_dir,
    khmer_piece,
    next_free_id,
)

KHMER = re.compile("[ក-៿]")

# Scripts whose UTF-8 shares the E1 lead byte with Khmer, plus the usual suspects.
OTHER_TEXTS = [
    "The National Bank of Cambodia published today's exchange rate: 1 USD = 4,105 KHR.",
    "柬埔寨国家银行今天公布了新的汇率。模型、型号、垃圾分类",
    "def rate(usd: float) -> float:\n    return usd * 4105.5\n",
    "| A1 | =SUM(B2:B9) |\n$$\\frac{a}{b} + \\sqrt{x}$$",
    "Tiếng Việt: Ngân hàng Nhà nước công bố tỷ giá mới.",
    "မြန်မာဘာသာ ဘဏ်",
    "ქართული ენა",
    "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ",
    "ᜊᜌ᜔ᜊᜌᜒᜈ᜔",
    "Ἑλληνικά ἀγαθός",
    "ภาษาไทย ພາສາລາວ",
    "Khmer lunar-date symbols ᧡᧢᧣",
    "emoji 🚀 😶‍🌫️ ✅ 🦙  \t\n\n ''''```\"\"\"......!!!!!!",
]


def khmer_seed_texts() -> list[str]:
    """Every Khmer string in the repository's seed data (deterministic order)."""
    import yaml
    from helpers import ROOT

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    out = []
    for path in sorted((ROOT / "data" / "seeds").rglob("*.yaml")):
        out += [s for s in strings(yaml.safe_load(path.read_text(encoding="utf-8"))) if KHMER.search(s)]
    return out


def split_texts() -> tuple[list[str], list[str]]:
    texts = khmer_seed_texts()
    held = texts[::5]
    return [t for i, t in enumerate(texts) if i % 5], held


def as_token(data: bytes) -> str:
    to_char = _byte_maps()[0]
    return "".join(to_char[b] for b in data)


@pytest.fixture(scope="module")
def extended(tmp_path_factory, tiny_tokenizer_json):
    from tokenizers import Tokenizer

    src = tmp_path_factory.mktemp("tok_src")
    shutil.copyfile(tiny_tokenizer_json, src / "tokenizer.json")
    out = tmp_path_factory.mktemp("tok_ext")
    train, held = split_texts()
    extension = extend_tokenizer_dir(src, out, train, n_new=160, min_count=2)
    return {
        "old": Tokenizer.from_file(str(src / "tokenizer.json")),
        "new": Tokenizer.from_file(str(out / "tokenizer.json")),
        "old_json": json.loads((src / "tokenizer.json").read_text(encoding="utf-8")),
        "extension": extension,
        "dir": out,
        "held": held,
    }


def test_khmer_piece_rule():
    accept = [
        "ក".encode(),
        " ក".encode(),
        "ក្រ".encode(),
        "្ម".encode(),
        b"\xe1\x9e",  # head of a Khmer character
        "ក".encode() + b"\xe1\x9f",
        b"\x80" + "ខ".encode(),  # tail of a character, then a whole one
        b"\x9e\x80" + "ខ".encode(),
    ]
    reject = [
        b"a",
        b" ",
        b"\x80\x81",  # continuation bytes only: no Khmer character starts here
        b"\xe1",
        "ạ".encode(),  # Vietnamese, E1 BA A1
        "ក".encode() + "ạ".encode(),
        "កa".encode(),
        "ក ខ".encode(),
        "  ក".encode(),
        "᧡".encode(),  # Khmer symbols block, E1 A7: deliberately not extended
        "က".encode(),  # Myanmar, E1 80 80
        "ក".encode() + b"\xe1\x80",  # head of a Myanmar character
    ]
    for data in accept:
        assert khmer_piece(as_token(data)), data
    for data in reject:
        assert not khmer_piece(as_token(data)), data
    assert not khmer_piece("<|im_end|>")


def test_new_ids_follow_the_last_id_and_nothing_moves(extended):
    extension, old, new = extended["extension"], extended["old"], extended["new"]
    assert extension.start_id == next_free_id(extended["old_json"]) == 480
    assert 0 < len(extension.tokens) <= 160
    for offset, token in enumerate(extension.tokens):
        assert new.token_to_id(token) == extension.start_id + offset
        assert khmer_piece(token)
    for token, index in old.get_vocab(with_added_tokens=True).items():
        assert new.token_to_id(token) == index, token
    assert new.token_to_id("<|im_end|>") == old.token_to_id("<|im_end|>")
    assert new.get_vocab_size() == extension.start_id + len(extension.tokens)  # no holes


def test_text_without_khmer_encodes_identically(extended):
    old, new = extended["old"], extended["new"]
    texts = OTHER_TEXTS + [t for t in CORPUS if not KHMER.search(t)]
    texts.append("<|im_start|>user\nHi<|im_end|>\n<tool_call>\n<function=x>\n</function>\n</tool_call>")
    for text in texts:
        assert (
            new.encode(text, add_special_tokens=False).ids == old.encode(text, add_special_tokens=False).ids
        )


def test_khmer_is_shorter_and_hf_bpe_equals_the_learner(extended):
    old, new, extension = extended["old"], extended["new"], extended["extension"]
    before = after = 0
    for text in extended["held"]:
        text = normalize_text(text)
        ids = new.encode(text, add_special_tokens=False).ids
        assert ids == encode_with_extension(old, extension, text)
        assert new.decode(ids) == text
        before += len(old.encode(text, add_special_tokens=False).ids)
        after += len(ids)
    assert after < 0.8 * before
    stats = extension.stats
    assert stats["khmer_tokens_after"] < stats["khmer_tokens_before"]


def test_extension_files(extended):
    out = extended["dir"]
    meta = json.loads((out / "extension.json").read_text())
    assert meta["start_id"] == 480 and meta["new_tokens"] == len(extended["extension"].tokens)
    rows = (out / "new_tokens.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "id\tcount\ttext" and len(rows) == 1 + meta["new_tokens"]
    first = rows[1].split("\t")
    assert first[0] == "480" and int(first[1]) >= 2


def test_apply_extension_refuses_used_ids(extended):
    extension = Extension(merges=[], tokens=[], start_id=479)
    with pytest.raises(ValueError, match="collides"):
        apply_extension(extended["old_json"], extension)


# -- model rows ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def remapped(tmp_path_factory, tiny_tokenizer_json, extended):
    from train.fixtures import tiny_model_dir

    from dsm.tokenizer.remap import extend_model_dir

    root = tmp_path_factory.mktemp("remap")
    base = tiny_model_dir(root / "base", tiny_tokenizer_json)
    entry = extend_model_dir(base, extended["dir"], root / "ext", pad_to=64)
    return base, root / "ext", entry


def test_remap_rows_and_logits(remapped, extended):
    from dsm.modeling import DSMForCausalLM

    base_dir, ext_dir, entry = remapped
    extension = extended["extension"]
    base = DSMForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
    ext = DSMForCausalLM.from_pretrained(ext_dir, dtype=torch.float32).eval()
    start, n_new = extension.start_id, len(extension.tokens)
    assert entry["vocab_size"] == ext.config.vocab_size
    assert ext.config.vocab_size % 64 == 0 and ext.config.vocab_size >= start + n_new
    old_rows = base.get_input_embeddings().weight
    new_rows = ext.get_input_embeddings().weight
    assert torch.equal(new_rows[:start], old_rows[:start])
    assert ext.lm_head.weight.data_ptr() == new_rows.data_ptr()  # still tied
    for offset, token in enumerate(extension.tokens):
        parts = [t.id for t in extended["old"].model.tokenize(token)]
        torch.testing.assert_close(new_rows[start + offset], old_rows[parts].mean(0))
    assert not new_rows[start + n_new :].any()  # padding rows
    ids = torch.tensor([extended["old"].encode("Hello! សួស្តី 你好", add_special_tokens=False).ids])
    with torch.no_grad():
        a = base(ids, output_hidden_states=True)
        b = ext(ids, output_hidden_states=True)
    assert torch.equal(a.hidden_states[-1], b.hidden_states[-1])
    torch.testing.assert_close(a.logits[..., :start], b.logits[..., :start], rtol=0, atol=1e-5)


def test_remapped_directory(remapped, extended):
    from transformers import AutoTokenizer

    base_dir, ext_dir, _ = remapped
    assert (ext_dir / "tokenizer.json").read_bytes() == (extended["dir"] / "tokenizer.json").read_bytes()
    for name in ("extension.json", "new_tokens.tsv", "chat_template.jinja", "LICENSE", "NOTICE"):
        assert (ext_dir / name).is_file(), name
    lineage = json.loads((ext_dir / "lineage.json").read_text())
    assert lineage[-1]["stage"] == "tokenizer_extension" and lineage[-1]["start_id"] == 480
    tokenizer = AutoTokenizer.from_pretrained(ext_dir)
    text = normalize_text(extended["held"][0])
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == extended["new"].encode(text).ids
    assert tokenizer.eos_token_id == extended["old"].token_to_id("<|im_end|>")


def test_new_rows_training_reads_extension_json(remapped, extended, tmp_path):
    from train.fixtures import packed_corpus

    from dsm.train.config import config_from_dict
    from dsm.train.trainer import LMTrainer

    _, ext_dir, entry = remapped
    texts = khmer_seed_texts() * 2
    prefix = packed_corpus(
        tmp_path / "km", ext_dir / "tokenizer.json", texts, "km", vocab_size=entry["vocab_size"]
    )
    cfg = config_from_dict(
        {
            "stage": "cpt",
            "model": str(ext_dir),
            "output_dir": str(tmp_path / "run"),
            "trainable": "new_rows",
            "seq_len": 64,
            "micro_batch_size": 2,
            "grad_accum": 1,
            "max_steps": 3,
            "train": [{"path": str(prefix)}],
            "precision": "fp32",
            "optimizer": {"lr": 1e-2, "weight_decay": 0.0},
            "log_every": 1,
            "eval_every": 0,
            "save_every": 0,
            "resume": None,
        }
    )
    trainer = LMTrainer(cfg, quiet=True)
    assert trainer.new_rows_from() == 480
    before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    trainer.train()
    after = trainer.model.state_dict()
    key = "model.embed_tokens.weight"
    assert torch.equal(after[key][:480], before[key][:480])
    assert not torch.equal(after[key][480:], before[key][480:])
    for name, value in before.items():
        if name not in (key, "lm_head.weight"):
            assert torch.equal(after[name], value), name


def test_packed_data_from_another_tokenizer_is_refused(remapped, tiny_tokenizer_json, tmp_path):
    from train.fixtures import packed_corpus

    from dsm.train.config import config_from_dict
    from dsm.train.trainer import LMTrainer

    _, ext_dir, _ = remapped
    prefix = packed_corpus(tmp_path / "km_old", tiny_tokenizer_json, khmer_seed_texts(), "km_old")
    cfg = config_from_dict(
        {
            "stage": "cpt",
            "model": str(ext_dir),
            "output_dir": str(tmp_path / "run"),
            "seq_len": 64,
            "max_steps": 1,
            "train": [{"path": str(prefix)}],
            "precision": "fp32",
            "resume": None,
        }
    )
    with pytest.raises(ValueError, match="re-pack"):
        LMTrainer(cfg, quiet=True)


def test_cli_analyze_extend_remap(tmp_path, tiny_tokenizer_json, capsys):
    from train.fixtures import tiny_model_dir

    from dsm.cli import main
    from dsm.utils import write_jsonl

    train, held = split_texts()
    write_jsonl(tmp_path / "km.jsonl", [{"text": t, "lang": "km"} for t in train])
    (tmp_path / "km_eval.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    (tmp_path / "en_eval.txt").write_text("\n".join(OTHER_TEXTS[:4]) + "\n", encoding="utf-8")
    model = tiny_model_dir(tmp_path / "model", tiny_tokenizer_json)
    args = ["--tokenizer", str(model), "--input", str(tmp_path / "km.jsonl"), "--out", str(tmp_path / "tok")]
    assert main(["tokenizer", "extend", *args, "--new-tokens", "64"]) == 0
    assert "64 new Khmer tokens (ids 480..543)" in capsys.readouterr().out
    remap = ["--model", str(model), "--tokenizer", str(tmp_path / "tok"), "--out", str(tmp_path / "ext")]
    assert main(["tokenizer", "remap", *remap]) == 0
    assert "vocab_size 576" in capsys.readouterr().out
    report = tmp_path / "report.json"
    analyze = [
        "--tokenizer",
        f"original={model}",
        f"extended={tmp_path / 'ext'}",
        "--text",
        f"km={tmp_path / 'km_eval.txt'}",
        f"en={tmp_path / 'en_eval.txt'}",
        "--json",
        str(report),
    ]
    assert main(["tokenizer", "analyze", *analyze]) == 0
    out = capsys.readouterr().out
    assert "| extended | km |" in out and "characters per second" in out
    data = json.loads(report.read_text())
    assert data["extended/km"]["tokens"] < data["original/km"]["tokens"]
    assert data["extended/en"]["tokens"] == data["original/en"]["tokens"]
