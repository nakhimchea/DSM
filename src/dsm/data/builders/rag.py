"""Grounded question answering (RAG) with citations and abstention.

Prompt format (documents and question in the user turn, the same format an
application must use at inference time)::

    Answer the question using only the documents below. Cite ... e.g. [1]. ...

    Documents:
    [1] Branch opening hours
    Branches are open Monday to Friday ...

    [2] International transfers
    ...

    Question: Are branches open on Saturday?

The answer cites the supporting document (``... 8:00 to 11:30 [1].``). Three
kinds of samples teach the model to stay grounded:

* answerable - the gold article is among 2-4 shuffled documents;
* unanswerable - the gold article is removed, distractors stay -> abstain;
* near-miss - a related article is present but does not contain the answer
  (e.g. it mentions a hotline without the number) -> abstain, do not guess.

25 % of answerable samples put English documents under a Khmer / Chinese
question (internal documents are often English only). Seeds live in
``data/seeds/rag/kb.yaml``; :meth:`RagBuilder.qa_from_documents` writes
question/answer pairs for any document collection with a local teacher.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

from dsm.data.sft.schema import sft_record
from dsm.data.synth.filters import identity_leak, language_matches
from dsm.text import normalize_text
from dsm.tools.bank_sim import KB_ARTICLES

LANGS = ("en", "km", "zh")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_CITATION = re.compile(r"\[(\d+)\]")
_FINAL_PUNCT = (".", "。", "។", "!", "?", "！", "？")


def cite(answer: str, index: int) -> str:
    """Insert ``[index]`` before the final punctuation mark (``... 11:30 [1].`` / ``...11:30[1]。``)."""
    answer = answer.rstrip()
    marker = f"[{index}]"
    if answer.endswith(_FINAL_PUNCT):
        body, punct = answer[:-1], answer[-1]
        space = "" if punct in ("。", "！", "？") else " "
        return f"{body}{space}{marker}{punct}"
    return f"{answer} {marker}"


def grounded(answer: str, document: str) -> bool:
    """Every number in the answer (citation markers aside) appears in the document (cheap hallucination check)."""
    return set(_NUMBER.findall(_CITATION.sub("", answer))) <= set(_NUMBER.findall(document))


def citations(text: str) -> set[int]:
    return {int(i) for i in _CITATION.findall(text)}


class RagBuilder:
    def __init__(self, seeds_path: str | Path = "data/seeds/rag/kb.yaml", seed: int = 0) -> None:
        self.t = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
        self.articles: dict[str, dict] = {}
        for article in KB_ARTICLES:
            self.articles[article["id"]] = {
                "id": article["id"],
                "title": self.t["titles"][article["id"]],
                "text": article["text"],
            }
        for article in self.t["articles"]:
            self.articles[article["id"]] = article
        self._counter = 0

    # -- composition -----------------------------------------------------------------------
    def prompt(self, documents: list[tuple[str, str]], question: str, lang: str) -> str:
        labels = self.t["labels"]
        blocks = [f"[{i + 1}] {title}\n{text}" for i, (title, text) in enumerate(documents)]
        label = labels["question"][lang]
        space = "" if label.endswith("：") else " "
        return normalize_text(
            f"{self.t['instructions'][lang]}\n\n{labels['documents'][lang]}\n"
            + "\n\n".join(blocks)
            + f"\n\n{label}{space}{question}"
        )

    def _documents(
        self, include: list[str], exclude: set[str], doc_lang: str, total: int
    ) -> tuple[list[tuple[str, str]], list[str]]:
        pool = [a for a in self.articles if a not in exclude and a not in include]
        chosen = include + self.rng.sample(pool, k=max(0, min(total - len(include), len(pool))))
        self.rng.shuffle(chosen)
        docs = [(self.articles[a]["title"][doc_lang], self.articles[a]["text"][doc_lang]) for a in chosen]
        return docs, chosen

    def _record(self, kind: str, lang: str, user: str, answer: str, meta: dict) -> dict:
        self._counter += 1
        return sft_record(
            id=f"rag-{kind}-{lang}-{self._counter:06d}",
            source="dsm-rag-seeds",
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": normalize_text(answer)},
            ],
            lang=lang,
            category="rag",
            generated_by="template",
            meta=meta,
        )

    def answerable(self, qa: dict, lang: str, doc_lang: str | None = None) -> dict:
        doc_lang = doc_lang or lang
        total = self.rng.randint(2, 4)
        docs, ids = self._documents([qa["article"]], set(), doc_lang, total)
        index = ids.index(qa["article"]) + 1
        gold_text = self.articles[qa["article"]]["text"][doc_lang]
        if not grounded(qa["a"][lang], gold_text):
            raise ValueError(
                f"seed answer for {qa['article']} ({lang}) contains numbers missing from the article"
            )
        answer = cite(qa["a"][lang], index)
        meta = {"kind": "answerable", "gold": index, "doc_ids": ids, "doc_lang": doc_lang}
        return self._record("answerable", lang, self.prompt(docs, qa["q"][lang], lang), answer, meta)

    def unanswerable(
        self, question: str, lang: str, *, exclude: set[str], include: list[str] | None = None, kind: str
    ) -> dict:
        docs, ids = self._documents(list(include or []), exclude, lang, self.rng.randint(2, 3))
        answer = self.rng.choice(self.t["abstain"][lang])
        return self._record(
            kind,
            lang,
            self.prompt(docs, question, lang),
            answer,
            {"kind": kind, "gold": None, "doc_ids": ids},
        )

    def build(self, unanswerable_rate: float = 0.35, cross_lingual_rate: float = 0.25) -> list[dict]:
        records = []
        for qa in self.t["qa"]:
            for lang in LANGS:
                records.append(self.answerable(qa, lang))
                if lang != "en" and self.rng.random() < cross_lingual_rate:
                    records.append(self.answerable(qa, lang, doc_lang="en"))
                if self.rng.random() < unanswerable_rate:
                    records.append(
                        self.unanswerable(qa["q"][lang], lang, exclude={qa["article"]}, kind="unanswerable")
                    )
        for item in self.t["near_miss"]:
            for lang in LANGS:
                records.append(
                    self.unanswerable(
                        item["q"][lang], lang, exclude=set(), include=[item["related"]], kind="near_miss"
                    )
                )
        return records

    # -- any document collection, with a local teacher ---------------------------------------------
    def qa_from_documents(self, documents: list[dict], teacher, per_doc: int = 2) -> list[dict]:
        """Grounded QA for ``[{"id", "title", "text", "lang"}]`` documents (teacher writes the pairs).

        Pairs whose answer is not grounded (numbers missing from the document), in
        the wrong language or leaking another identity are dropped. Distractors are
        other documents of the same language.
        """
        requests = []
        for doc in documents:
            language = {"en": "English", "km": "Khmer", "zh": "Chinese"}[doc["lang"]]
            requests.append(
                [
                    {
                        "role": "system",
                        "content": "You write reading-comprehension data. Reply with JSON lines only, one object per line: "
                        '{"question": ..., "answer": ...}. Questions must be answerable from the document alone; answers are '
                        f"one or two sentences in {language}, using only facts stated in the document.",
                    },
                    {
                        "role": "user",
                        "content": f"Write {per_doc} question/answer pairs in {language}.\n\n{doc['title']}\n{doc['text']}",
                    },
                ]
            )
        outputs = teacher.map(requests)
        by_lang: dict[str, list[dict]] = {}
        for doc in documents:
            by_lang.setdefault(doc["lang"], []).append(doc)
        records = []
        for doc, output in zip(documents, outputs, strict=True):
            for line in (output or "").splitlines():
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(pair, dict):
                    continue
                question, answer = str(pair.get("question", "")).strip(), str(pair.get("answer", "")).strip()
                lang = doc["lang"]
                if not question or not answer or not grounded(answer, doc["text"]):
                    continue
                if identity_leak(answer) or not language_matches(answer, lang, threshold=0.3):
                    continue
                others = [d for d in by_lang[lang] if d["id"] != doc["id"]]
                distractors = self.rng.sample(others, k=min(len(others), self.rng.randint(1, 3)))
                docs = [doc, *distractors]
                self.rng.shuffle(docs)
                index = docs.index(doc) + 1
                user = self.prompt([(d["title"], d["text"]) for d in docs], question, lang)
                self._counter += 1
                records.append(
                    sft_record(
                        id=f"rag-teacher-{lang}-{self._counter:06d}",
                        source=doc.get("source", "documents"),
                        license=doc.get("license", "unknown"),
                        generated_by=teacher.generated_by,
                        messages=[
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": normalize_text(cite(answer, index))},
                        ],
                        lang=lang,
                        category="rag",
                        meta={"kind": "answerable", "gold": index, "doc_ids": [d["id"] for d in docs]},
                    )
                )
                if others and self.rng.random() < 0.3:  # same question without its document -> abstain
                    docs = self.rng.sample(others, k=min(len(others), 2))
                    user = self.prompt([(d["title"], d["text"]) for d in docs], question, lang)
                    self._counter += 1
                    records.append(
                        sft_record(
                            id=f"rag-teacher-{lang}-{self._counter:06d}",
                            source=doc.get("source", "documents"),
                            license=doc.get("license", "unknown"),
                            generated_by=teacher.generated_by,
                            messages=[
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": self.t["abstain"][lang][0]},
                            ],
                            lang=lang,
                            category="rag",
                            meta={"kind": "unanswerable", "gold": None, "doc_ids": [d["id"] for d in docs]},
                        )
                    )
        return records


def build(seeds_path: str | Path = "data/seeds/rag/kb.yaml", seed: int = 0) -> list[dict]:
    return RagBuilder(seeds_path, seed).build()
