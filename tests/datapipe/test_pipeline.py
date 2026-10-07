"""Data pipeline: catalogue gate, cleaning rules, PII, dedup, decontamination, mixtures, end to end."""

from __future__ import annotations

import json
import random

import pytest

from dsm.data.clean import clean_records, clean_text, looks_like_legacy_khmer
from dsm.data.decontam import Decontaminator
from dsm.data.dedup import Deduplicator
from dsm.data.langid import identify
from dsm.data.mixture import plan_mixture, repeats_for_shares
from dsm.data.pii import luhn_ok, redact
from dsm.data.quality import khmer_broken_rate, quality
from dsm.data.sources import Source, check, gate, load_sources

KM_DOC = (
    "ធនាគារជាតិនៃកម្ពុជាគឺជាធនាគារកណ្តាលរបស់ប្រទេសកម្ពុជា។ ធនាគារនេះមានតួនាទីគ្រប់គ្រងរូបិយវត្ថុ និងត្រួតពិនិត្យប្រព័ន្ធធនាគារ។ "
    "អតិថិជនអាចបើកគណនីសន្សំនៅសាខាណាមួយ ដោយយកអត្តសញ្ញាណប័ណ្ណមកជាមួយ។ ការផ្ទេរប្រាក់តាមកម្មវិធីទូរស័ព្ទមានសុវត្ថិភាព និងរហ័ស។ "
) * 2
EN_DOC = (
    "The central bank is responsible for monetary policy and for the supervision of the banking system. "
    "Customers can open a savings account at any branch with their national ID card, and transfers in the mobile app are fast and safe. "
    "Interest rates depend on the term of the deposit, and the bank publishes them on its website every month."
)
ZH_DOC = (
    "中央银行负责制定货币政策并监管银行体系。客户可以携带身份证在任何分行开设储蓄账户，手机银行转账快速又安全。存款利率取决于存款期限，银行每月在网站上公布利率。"
    * 3
)


# -- catalogue and licence gate -----------------------------------------------------------------


def test_catalogue_is_consistent():
    sources = load_sources()
    assert all(not s.problems() for s in sources.values())
    assert {"fineweb2_km", "xlam", "belebele", "pku_saferlhf"} <= set(sources)
    blocked = [s for s in sources.values() if s.tier == "blocked"]
    assert blocked and all(s.block_reason for s in blocked)


def test_gate_rules():
    sources = load_sources()
    assert check(sources["fineweb2_km"]).allowed
    assert (
        not check(sources["wikipedia_km"]).allowed
        and check(sources["wikipedia_km"], accept_counsel=True).allowed
    )
    assert not check(sources["belebele"]).allowed and check(sources["belebele"], purpose="eval").allowed
    assert not check(sources["pku_saferlhf"]).allowed
    assert check(sources["pku_saferlhf"], allow_noncommercial=True).allowed
    assert not check(
        sources["oscar"], allow_noncommercial=True, allow_naming_clause=True
    ).allowed  # suspended
    llama_made = Source(
        name="x", kind="sft", langs=["en"], tier="use", license="CC-BY-4.0", generated_by="model:llama-3"
    )
    assert not check(llama_made).allowed  # licence propagates even though the source says "use"
    assert Source(name="y", kind="sft", langs=["en"], tier="use", license="CC-BY-NC-4.0").problems()
    with pytest.raises(PermissionError, match="pku_saferlhf"):
        gate(["fineweb2_km", "pku_saferlhf", "missing"], sources)


# -- cleaning ------------------------------------------------------------------------------------


def test_clean_text_and_langid():
    assert clean_text("ខែ្មរ​  text\r\n\r\n\r\n\r\nmore\x07") == "ខ្មែរ text\n\nmore"
    assert identify(KM_DOC)[0] == "km" and identify(ZH_DOC)[0] == "zh" and identify(EN_DOC)[0] == "en"
    assert identify("這個問題我們會說明")[0] == "zh-Hant"
    assert identify("Ceci est un texte en français sans mots anglais.")[0] == "und"
    assert looks_like_legacy_khmer("¦¾ÁøÁ®ÁÑ¥ «¶ç¤µ ¦¾ÁøÁ®ÁÑ¥ «¶ç¤µ")
    assert not looks_like_legacy_khmer(KM_DOC)


def test_quality_rules():
    assert quality(EN_DOC, "en", min_chars=100).keep
    assert quality(KM_DOC, "km", min_chars=100).keep and quality(ZH_DOC, "zh", min_chars=100).keep
    assert "too_short" in quality("Hello there.", "en").reasons
    repeated = "\n".join(["Click here to subscribe to our newsletter."] * 20)
    assert "duplicate_lines" in quality(repeated, "en", min_chars=10).reasons
    assert "low_khmer_share" in quality(EN_DOC, "km", min_chars=10).reasons
    broken = "្ាេ" * 40 + KM_DOC  # dependent signs without a base consonant
    assert khmer_broken_rate(broken) > khmer_broken_rate(KM_DOC)
    assert khmer_broken_rate(KM_DOC) < 15


def test_pii_redaction():
    text = "Email sok@example.com, call +855 12 345 678, card 4111 1111 1111 1111, IP 10.0.0.1, IBAN DE89 3704 0044 0532 0130 00."
    redacted, counts = redact(text)
    assert "sok@example.com" not in redacted and "4111" not in redacted and "10.0.0.1" not in redacted
    assert counts == {"EMAIL": 1, "PHONE": 1, "CARD": 1, "IP": 1, "IBAN": 1}
    assert luhn_ok("4111 1111 1111 1111") and not luhn_ok("4111 1111 1111 1112")
    assert redact("Order 4111 1111 1111 1112 shipped")[1].get("CARD") is None  # not a valid card number
    strict, counts = redact("Customer ACC-10233 with ID 123456789 called.", strict=True)
    assert "<ACCOUNT>" in strict and "<NUMBER>" in strict and counts["NUMBER"] == 1
    assert redact("Customer ACC-10233 called in 2024.")[0] == "Customer ACC-10233 called in 2024."


def test_clean_records_counts_reasons():
    rows = [
        {"text": EN_DOC},
        {"text": "short"},
        {"text": KM_DOC},
        {"text": ""},
        {"text": EN_DOC + " Contact a@b.co"},
    ]
    from collections import Counter

    stats = Counter()
    kept = list(
        clean_records(
            rows, source="t", license="CC0", generated_by="web", langs=["en"], stats=stats, min_chars=100
        )
    )
    assert len(kept) == 2 and stats["kept"] == 2 and stats["drop: lang_km"] == 1 and stats["drop: empty"] == 1
    assert stats["pii: EMAIL"] == 1 and "<EMAIL>" in kept[1]["text"]
    assert kept[0]["id"] == "t-000000000" and kept[0]["license"] == "CC0"


# -- dedup / decontamination -------------------------------------------------------------------------


def _mutate(text: str, rng: random.Random, rate: float = 0.02) -> str:
    chars = list(text)
    for _ in range(max(1, int(len(chars) * rate))):
        chars[rng.randrange(len(chars))] = rng.choice("abcdefghij")
    return "".join(chars)


def test_dedup_recall_and_precision():
    rng = random.Random(0)
    vocabulary = [f"word{i}" for i in range(400)]
    docs = [" ".join(rng.choice(vocabulary) for _ in range(150)) for _ in range(40)]
    dedup = Deduplicator(threshold=0.8)
    assert all(dedup.add(f"d{i}", doc, "en") is None for i, doc in enumerate(docs))  # distinct docs survive
    found = sum(dedup.add(f"n{i}", _mutate(doc, rng, 0.001), "en") == f"d{i}" for i, doc in enumerate(docs))
    assert found / len(docs) >= 0.9
    assert dedup.add("x", docs[3], "en") == "d3"  # exact
    km = Deduplicator()
    assert km.add("a", KM_DOC, "km") is None and km.add("b", KM_DOC.replace("រហ័ស", "លឿន"), "km") == "a"


def test_decontamination(tmp_path):
    question = "What is the daily ATM withdrawal limit for customers of the bank in Phnom Penh and Siem Reap?"
    (tmp_path / "eval.jsonl").write_text(json.dumps({"question": question, "answer": "2000"}) + "\n")
    index = Decontaminator()
    assert index.add_file(tmp_path / "eval.jsonl") == 1
    assert index.contaminated("Intro. " + question + " The answer is 2000.", "en")
    assert not index.contaminated(EN_DOC, "en")
    zh_eval = "请问在金边和暹粒的分行，每天通过ATM最多可以取多少钱？这个问题的答案在文档中可以找到吗？"
    index.add_text(zh_eval)
    assert index.contaminated("前言。" + zh_eval, "zh") and not index.contaminated(ZH_DOC, "zh")


# -- mixtures -----------------------------------------------------------------------------------------------


def test_plan_mixture_caps_and_redistributes():
    plan = plan_mixture(
        {"km": 2e8, "en": 5e10, "zh": 5e10},
        {"km": 0.45, "en": 0.30, "zh": 0.25},
        2e9,
        {"km": 4, "en": 1, "zh": 1},
    )
    assert (
        plan.weights["km"] == pytest.approx(0.4)
        and plan.epochs["km"] == pytest.approx(4.0)
        and plan.capped == ["km"]
    )
    assert plan.weights["en"] / plan.weights["zh"] == pytest.approx(0.30 / 0.25)
    assert sum(plan.weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not enough data"):
        plan_mixture({"a": 10, "b": 10}, {"a": 1, "b": 1}, 1e6, 4)
    assert repeats_for_shares(
        {"identity": 500, "chat": 50_000}, {"identity": 2, "chat": 98}, total=100_000
    ) == {"identity": 4.0, "chat": 1.96}


# -- end to end through the CLI ---------------------------------------------------------------------------------


def test_cli_pipeline(tmp_path, tiny_tokenizer_json, capsys):
    from train.fixtures import tiny_model_dir

    from dsm.cli import main
    from dsm.data.packed import PackedTokens
    from dsm.utils import write_jsonl

    raw = tmp_path / "raw"
    other = (
        "Small businesses in Battambang often finance new equipment with short loans. Lenders look at cash flow, "
        "collateral and the repayment history of the owner before they approve the application. A clear business "
        "plan with monthly sales figures helps, and many owners bring their tax records and a list of suppliers "
        "to the first meeting with the credit officer. Mail me: a@b.co"
    )
    rows = [{"text": EN_DOC}, {"text": EN_DOC}, {"text": KM_DOC}, {"text": "tiny"}, {"text": other}]
    write_jsonl(raw / "shard-00000.jsonl", rows)
    clean = tmp_path / "clean" / "local.jsonl.zst"
    assert (
        main(
            [
                "data",
                "clean",
                "--name",
                "local",
                "--license",
                "CC-BY-4.0",
                "--langs",
                "en",
                "--input",
                str(raw),
                "--out",
                str(clean),
                "--min-chars",
                "100",
            ]
        )
        == 0
    )
    assert main(["data", "dedup", "--input", str(clean), "--out-dir", str(tmp_path / "dedup")]) == 0
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "probe.yaml").write_text(
        "q: 'Lenders look at cash flow, collateral and the repayment history of the owner before they approve the application.'\n"
    )
    assert (
        main(
            [
                "data",
                "decontam",
                "--input",
                str(tmp_path / "dedup"),
                "--eval",
                str(tmp_path / "eval"),
                "--out-dir",
                str(tmp_path / "final"),
            ]
        )
        == 0
    )
    model = tiny_model_dir(tmp_path / "model", tiny_tokenizer_json)
    final = tmp_path / "final" / "local.jsonl.zst"
    stats = json.loads((final.with_name(final.name + ".stats.json")).read_text())
    assert set(stats["stages"]) == {"clean", "dedup", "decontam"}
    assert stats["stages"]["clean"]["drop: lang_km"] == 1 and stats["stages"]["clean"]["pii: EMAIL"] == 1
    assert stats["stages"]["dedup"]["duplicates"] == 1
    assert stats["stages"]["decontam"]["contaminated"] == 1 and stats["stages"]["decontam"]["kept"] == 1
    packed = tmp_path / "packed" / "local"
    assert main(["data", "pack", "--input", str(final), "--tokenizer", str(model), "--out", str(packed)]) == 0
    data = PackedTokens(packed)
    assert data.n_docs == 1 and data.meta["licenses"] == ["CC-BY-4.0"] and data.meta["langs"] == {"en": 1}
    card = (tmp_path / "packed" / "local.card" / "DATACARD.md").read_text()
    assert "## clean" in card and "## decontam" in card and "## pack" in card
    capsys.readouterr()
    assert main(["data", "sources", "--kind", "eval"]) == 0
    assert "belebele" in capsys.readouterr().out
