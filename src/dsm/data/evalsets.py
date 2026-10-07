"""Benchmark rows -> the files ``dsm eval`` reads (kept locally, never trained on).

* Belebele (passage, question, four answers)  -> ``dsm eval choice`` rows
* SIB-200 (sentence, one of seven topics)      -> ``dsm eval choice`` rows
* FLORES+ -> ``dsm data parallel --layout grouped --out-pairs`` -> ``dsm eval translate`` rows

Choice rows: ``{id, context, question, options, answer, lang}`` with ``answer`` the index of
the correct option. The Khmer / Chinese SIB-200 topic names and questions below are short
standard terms; have them checked by native speakers like the other Khmer seed text.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from dsm.data.parallel import lang_code

SIB_TOPICS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
SIB_LABELS = {
    "en": SIB_TOPICS,
    "km": ["វិទ្យាសាស្ត្រ/បច្ចេកវិទ្យា", "ការធ្វើដំណើរ", "នយោបាយ", "កីឡា", "សុខភាព", "ការកម្សាន្ត", "ភូមិសាស្ត្រ"],
    "zh": ["科学/技术", "旅游", "政治", "体育", "健康", "娱乐", "地理"],
}
SIB_QUESTION = {
    "en": "Which topic is this text about?",
    "km": "តើអត្ថបទនេះនិយាយអំពីប្រធានបទអ្វី?",
    "zh": "这段文字属于哪个主题？",
}


def belebele_rows(rows: Iterable[dict], lang: str | None = None) -> Iterator[dict]:
    for row in rows:
        row_lang = lang_code(row.get("dialect")) or lang
        if row_lang is None:
            continue
        yield {
            "id": f"belebele-{row_lang}-{row.get('link', '')}-{row.get('question_number', '')}",
            "context": row["flores_passage"],
            "question": row["question"],
            "options": [row[f"mc_answer{i}"] for i in range(1, 5)],
            "answer": int(row["correct_answer_num"]) - 1,
            "lang": row_lang,
        }


def sib_rows(rows: Iterable[dict], lang: str) -> Iterator[dict]:
    """SIB-200 rows carry no language field: it comes from the subset you downloaded."""
    for row in rows:
        if row.get("category") not in SIB_TOPICS:
            continue
        yield {
            "id": f"sib200-{lang}-{row.get('index_id', '')}",
            "context": row["text"],
            "question": SIB_QUESTION[lang],
            "options": SIB_LABELS[lang],
            "answer": SIB_TOPICS.index(row["category"]),
            "lang": lang,
        }
