"""Tiny stand-ins for `scripts/10_fetch.sh`: upstream checkpoints and raw dataset shards.

Same layout and row formats as the real downloads (``datasets/raw/<source>/shard-*.jsonl.zst``),
built from the repository's seed text and the hand-written public-format rows of the unit tests.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from upstream_fixture import build_tiny_upstream

from data import dataset_rows as R
from dsm.data.langid import identify
from dsm.utils import write_jsonl

# (km, en, zh) - training sentences; FLORES stand-ins use different ones.
PARALLEL = [
    ("ធនាគារបើកពីម៉ោងប្រាំបីព្រឹក។", "The bank opens at eight in the morning.", "银行早上八点开门。"),
    ("អត្រាប្តូរប្រាក់ថ្ងៃនេះមានស្ថេរភាព។", "Today's exchange rate is stable.", "今天的汇率很稳定。"),
    (
        "សាខាថ្មីនៅសៀមរាបនឹងបើកនៅខែក្រោយ។",
        "The new branch in Siem Reap will open next month.",
        "暹粒的新分行下个月开业。",
    ),
    (
        "អតិថិជនអាចបើកគណនីសន្សំតាមអនឡាញ។",
        "Customers can open a savings account online.",
        "客户可以在网上开立储蓄账户。",
    ),
    ("សូមអរគុណដែលបានជ្រើសរើសសេវាកម្មរបស់យើង។", "Thank you for choosing our service.", "感谢您选择我们的服务。"),
]
HELD_OUT = [
    ("ភ្លៀងធ្លាក់ខ្លាំងនៅភ្នំពេញកាលពីយប់មិញ។", "It rained heavily in Phnom Penh last night.", "昨晚金边下了大雨。"),
    (
        "សិស្សទាំងអស់បានប្រឡងជាប់ឆ្នាំនេះ។",
        "All the students passed the exam this year.",
        "今年所有学生都通过了考试。",
    ),
    ("ផ្សារបិទនៅថ្ងៃអាទិត្យ។", "The market is closed on Sunday.", "市场星期天关门。"),
]
FLORES_CODES = {"km": ("khm", "Khmr"), "en": ("eng", "Latn"), "zh": ("cmn", "Hans")}
BELEBELE_DIALECT = {"km": "khm_Khmr", "en": "eng_Latn", "zh": "zho_Hans"}


def seed_strings(root: Path) -> dict[str, list[str]]:
    """Seed sentences by language, placeholders filled, without duplicates."""
    import yaml

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    out: dict[str, list[str]] = {"km": [], "en": [], "zh": []}
    for path in sorted((root / "data" / "seeds").rglob("*.yaml")):
        for text in strings(yaml.safe_load(path.read_text(encoding="utf-8"))):
            text = re.sub(r"\{[^{}]*\}", "DSM", text).strip()
            lang = identify(text)[0]
            if lang in out and len(text) >= 20 and text not in out[lang]:
                out[lang].append(text)
    return out


def documents(texts: list[str], n: int, seed: int, parts: int = 8, extra: str = "") -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "text": "\n\n".join(rng.sample(texts, k=min(parts, len(texts)))) + extra,
            "url": f"https://example.org/{seed}/{i}",
        }
        for i in range(n)
    ]


def raw(home: Path, name: str, rows: list[dict]) -> None:
    write_jsonl(home / "datasets" / "raw" / name / "shard-00000.jsonl.zst", rows)


def build(home: Path, root: Path, tokenizer_json: Path) -> None:
    upstream = home / "work" / "upstream"
    build_tiny_upstream(upstream / "base", tokenizer_json, seed=0)
    build_tiny_upstream(upstream / "instruct", tokenizer_json, seed=1)

    text = seed_strings(root)
    raw(home, "fineweb2_km", documents(text["km"], 40, seed=1))
    raw(home, "fineweb2_zh", documents(text["zh"], 30, seed=2))
    edu = documents(text["en"], 30, seed=3)
    for i, row in enumerate(edu):
        row["int_score"] = 3 + i % 3
    raw(home, "fineweb_edu", edu)
    raw(
        home,
        "finemath",
        documents(text["en"], 10, seed=4, extra="\n\nThe area of a circle is $A = \\pi r^2$."),
    )

    raw(home, "alt_km_en", [{"translation": {"khm": km, "en": en, "zh": zh}} for km, en, zh in PARALLEL * 8])
    for lang, (iso, script) in FLORES_CODES.items():
        name = "flores_plus" if lang == "km" else f"flores_plus_{lang}"
        rows = [
            {"id": i, "iso_639_3": iso, "iso_15924": script, "text": triple[("km", "en", "zh").index(lang)]}
            for i, triple in enumerate(HELD_OUT)
        ]
        raw(home, name, rows)
        passage = " ".join(row["text"] for row in rows)
        name = "belebele" if lang == "km" else f"belebele_{lang}"
        belebele = [
            {
                "link": "x",
                "question_number": q,
                "flores_passage": passage,
                "question": f"Question {q}?",
                **{f"mc_answer{i}": f"option {i}" for i in range(1, 5)},
                "correct_answer_num": str(1 + q % 4),
                "dialect": BELEBELE_DIALECT[lang],
            }
            for q in range(3)
        ]
        raw(home, name, belebele)
        name = "sib200" if lang == "km" else f"sib200_{lang}"
        raw(
            home,
            name,
            [{"index_id": i, "category": "travel", "text": row["text"]} for i, row in enumerate(rows)],
        )

    chats = [
        {
            "messages": [
                {"role": "user", "content": f"What is {i} + {i}?"},
                {"role": "assistant", "content": f"{i} + {i} = {2 * i}."},
            ]
        }
        for i in range(1, 25)
    ]
    raw(home, "smoltalk2", [R.CHAT, *chats])
    raw(home, "xlam", [R.XLAM] * 4)
    raw(home, "toolace", [R.TOOLACE] * 2)
    raw(home, "hermes_fc", [R.HERMES] * 2)
    raw(home, "glaive_fc_v2", [R.GLAIVE] * 2)
    raw(home, "massive_km", [r for r in R.MASSIVE_ROWS if r["locale"] == "km-KH"] * 3)
    raw(home, "gsm8k", [R.GSM8K, R.GSM8K_MONEY] * 3)
    raw(home, "tabfact", [R.TABFACT] * 2)
    raw(home, "tatqa", [R.TATQA])
    raw(home, "finqa", [R.FINQA] * 2)
    raw(home, "helpsteer3", [R.HELPSTEER3] * 4)
    raw(
        home,
        "xstest",
        [{"prompt": "How do I open a savings account?", "type": "safe_contexts", "label": "safe"}],
    )
