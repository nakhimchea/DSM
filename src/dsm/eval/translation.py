"""FLORES-style translation evaluation (chrF++), with the prompt format used in training.

Input JSONL rows: ``{"src", "tgt", "src_lang", "tgt_lang"}`` (e.g. FLORES+ devtest
km/en/zh, downloaded once and kept locally). Prompts come from
``dsm.data.builders.translation.TASK_TEMPLATES`` in the target language's instruction.
"""

from __future__ import annotations

from collections import defaultdict

from dsm.data.builders.translation import TASK_TEMPLATES
from dsm.eval.chrf import corpus_chrf
from dsm.eval.generator import Request
from dsm.text import normalize_text


def translation_requests(rows: list[dict]) -> list[Request]:
    requests = []
    for row in rows:
        template = TASK_TEMPLATES[(row["src_lang"], row["tgt_lang"])][row["tgt_lang"]]
        requests.append(
            Request(
                [{"role": "user", "content": template.format(text=normalize_text(row["src"]).strip())}],
                max_new_tokens=512,
            )
        )
    return requests


def evaluate_translation(generate, rows: list[dict]) -> dict[str, float]:
    rows = [r for r in rows if (r["src_lang"], r["tgt_lang"]) in TASK_TEMPLATES]
    replies = generate(translation_requests(rows))
    by_direction: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for row, reply in zip(rows, replies, strict=True):
        hyps, refs = by_direction[f"{row['src_lang']}-{row['tgt_lang']}"]
        hyps.append(normalize_text(reply.content.strip()))
        refs.append(normalize_text(row["tgt"].strip()))
    return {f"translation/chrf++/{d}": corpus_chrf(h, r) for d, (h, r) in sorted(by_direction.items())}
