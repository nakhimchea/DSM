"""Markdown / LaTeX formatting conversations (tables, display math, numbered steps).

Deterministic samples from ``data/seeds/formatting/facts.yaml`` that teach the
house style of docs/09. Every answer passes ``dsm.format.validate.check_format``.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from dsm.data.sft.schema import sft_record
from dsm.format.validate import check_format
from dsm.text import normalize_text

LANGS = ("en", "km", "zh")


def _table(columns: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


class FormattingBuilder:
    def __init__(self, seeds_path: str | Path, seed: int = 0) -> None:
        self.f = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
        self._counter = 0

    def _record(self, kind: str, lang: str, user: str, answer: str) -> dict | None:
        answer = normalize_text(answer).strip()
        if check_format(answer):
            return None
        self._counter += 1
        return sft_record(
            id=f"format-{kind}-{lang}-{self._counter:06d}",
            source="dsm-formatting",
            messages=[
                {"role": "user", "content": normalize_text(user).strip()},
                {"role": "assistant", "content": answer},
            ],
            lang=lang,
            category="format",
            meta={"kind": kind},
        )

    def compare(self, item: dict, lang: str, question: str) -> dict | None:
        topic = item["topic"][lang]
        rows = [row[lang] for row in item["rows"]]
        intro = self.f["answer_intro"]["compare"][lang].format(topic=topic)
        answer = f"{intro}\n\n{_table(item['columns'][lang], rows)}"
        return self._record("compare", lang, question.format(topic=topic), answer)

    def formula(self, item: dict, lang: str, question: str) -> dict | None:
        name = item["name"][lang]
        intro = self.f["answer_intro"]["formula"][lang].format(name=name)
        where = self.f["answer_intro"]["where"][lang]
        variables = "\n".join(f"- {v}" for v in item["vars"][lang])
        answer = f"{intro}\n\n$$\n{item['latex']}\n$$\n\n{where}\n{variables}"
        return self._record("formula", lang, question.format(name=name), answer)

    def steps(self, item: dict, lang: str, question: str) -> dict | None:
        topic = item["topic"][lang]
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(item["steps"][lang], start=1))
        answer = f"{self.f['answer_intro']['steps'][lang]}\n\n{steps}"
        return self._record("steps", lang, question.format(topic=topic), answer)

    def build(self) -> list[dict]:
        records = []
        for lang in LANGS:
            for item in self.f["comparisons"]:
                for question in self.f["questions"]["compare"][lang]:
                    records.append(self.compare(item, lang, question))
            for item in self.f["formulas"]:
                for question in self.f["questions"]["formula"][lang]:
                    records.append(self.formula(item, lang, question))
            for item in self.f["procedures"]:
                for question in self.f["questions"]["steps"][lang]:
                    records.append(self.steps(item, lang, question))
        return [r for r in records if r is not None]


def build(seeds_path: str | Path, seed: int = 0) -> list[dict]:
    return FormattingBuilder(seeds_path, seed=seed).build()
