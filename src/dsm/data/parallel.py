"""Parallel text -> CPT documents, translation pairs and translation SFT.

Input layouts (``dsm data parallel --layout``):

* ``aligned`` - one row holds one sentence in several languages: ``{"translation": {"en": ..,
  "khm": ..}}`` or the language codes as top-level keys (ALT, OPUS-style sets);
* ``grouped`` - one row holds one sentence in ONE language; rows sharing an id are
  translations of each other: FLORES+ (``id``, ``iso_639_3`` + ``iso_15924``, ``text``) and
  MASSIVE (``id``, ``locale``, ``utt``);
* ``pairs`` - one row is a source / target pair: ``{src, tgt, src_lang, tgt_lang}`` or the
  TICO-19 column names ``sourceString, targetString, sourceLang, targetLang`` (TSV is fine).

Language codes are mapped to km / en / zh (Simplified); everything else is ignored. Each
sentence is normalised and must be written in its language's script; pairs whose lengths
disagree wildly (mined data) are dropped.

Outputs:

* documents for CPT - consecutive aligned sentences grouped into documents, each sentence
  followed by its translations (the language order varies per document, so both
  directions are seen);
* pairs ``{src, tgt, src_lang, tgt_lang}`` in every direction - the input of
  ``dsm eval translate`` (FLORES+) and of translation SFT;
* translation SFT records (``parallel_to_sft``).
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from dsm.data.langid import script_shares
from dsm.text import normalize_text

LANGS = ("km", "en", "zh")
_CODES = {
    **dict.fromkeys(["km", "khm", "khm_khmr", "km-kh", "km_kh"], "km"),
    **dict.fromkeys(["en", "eng", "eng_latn", "en-us", "en_us", "en-gb", "en_gb"], "en"),
    **dict.fromkeys(
        ["zh", "zho", "zho_hans", "cmn", "cmn_hans", "zh-cn", "zh_cn", "zh-hans", "zh_hans"], "zh"
    ),
}
# characters per "unit of meaning", for the length-ratio check (a Han character ~ 3 Latin ones)
_DENSITY = {"km": 1.0, "en": 1.0, "zh": 3.0}


def lang_code(code: object) -> str | None:
    """km / en / zh for the usual spellings of those languages, else None."""
    return _CODES.get(str(code or "").strip().lower())


def read_rows(path: str | Path) -> Iterator[dict]:
    """JSONL(.zst), JSON (a list) or TSV / CSV rows."""
    from dsm.utils import read_jsonl

    path = Path(path)
    if path.suffix in (".tsv", ".csv"):
        with open(path, encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data if isinstance(data, list) else data.get("data", [])
    else:
        yield from read_jsonl(path)


def _grouped_lang(row: dict) -> str | None:
    if "iso_639_3" in row:
        return lang_code(f"{row['iso_639_3']}_{row.get('iso_15924', '')}".rstrip("_"))
    for key in ("locale", "lang", "language", "dialect"):
        if key in row:
            return lang_code(row[key])
    return None


def aligned_items(rows: Iterable[dict], layout: str) -> Iterator[dict[str, str]]:
    """Dicts ``{lang: sentence}`` (km / en / zh only), in input order."""
    if layout == "aligned":
        for row in rows:
            source = row.get("translation") if isinstance(row.get("translation"), dict) else row
            item = {lang_code(k): v for k, v in source.items() if lang_code(k) and isinstance(v, str)}
            if len(item) >= 2:
                yield item
    elif layout == "grouped":
        groups: dict[str, dict[str, str]] = {}
        for row in rows:
            lang = _grouped_lang(row)
            text = next((row[k] for k in ("text", "utt", "sentence") if isinstance(row.get(k), str)), None)
            if lang and text is not None:
                groups.setdefault(str(row.get("id")), {})[lang] = text
        yield from (item for item in groups.values() if len(item) >= 2)
    elif layout == "pairs":
        for row in rows:
            src_lang = lang_code(row.get("src_lang", row.get("sourceLang")))
            tgt_lang = lang_code(row.get("tgt_lang", row.get("targetLang")))
            src, tgt = row.get("src", row.get("sourceString")), row.get("tgt", row.get("targetString"))
            if (
                src_lang
                and tgt_lang
                and src_lang != tgt_lang
                and isinstance(src, str)
                and isinstance(tgt, str)
            ):
                yield {src_lang: src, tgt_lang: tgt}
    else:
        raise ValueError(f"unknown layout {layout!r} (aligned, grouped, pairs)")


def _in_script(text: str, lang: str) -> bool:
    shares = script_shares(text)
    if not shares:
        return False
    return shares["latin"] >= 0.7 if lang == "en" else shares[lang] >= 0.5


def clean_item(item: dict[str, str], langs: Iterable[str], max_ratio: float, stats: Counter) -> dict | None:
    """Normalised sentences in the wanted languages, or None (reason counted in ``stats``)."""
    out = {}
    for lang in langs:
        text = normalize_text(item.get(lang) or "").strip()
        if not text:
            continue
        if not _in_script(text, lang):
            stats[f"drop sentence: wrong_script_{lang}"] += 1
            continue
        out[lang] = text
    if len(out) < 2:
        stats["drop: fewer_than_two_languages"] += 1
        return None
    sizes = [len(text) * _DENSITY[lang] for lang, text in out.items()]
    if max(sizes) > max_ratio * min(sizes):
        stats["drop: length_ratio"] += 1
        return None
    return out


def to_documents(
    items: list[dict[str, str]],
    *,
    source: str,
    license: str,
    generated_by: str,
    per_doc: int = 16,
    seed: int = 0,
) -> list[dict]:
    """Consecutive items -> documents; within a document each sentence is followed by its
    translations, in a language order drawn per document."""
    rng = random.Random(seed)
    docs = []
    for n, start in enumerate(range(0, len(items), per_doc)):
        chunk = items[start : start + per_doc]
        langs = sorted({lang for item in chunk for lang in item})
        rng.shuffle(langs)
        blocks = ["\n".join(item[lang] for lang in langs if lang in item) for item in chunk]
        docs.append(
            {
                "id": f"{source}-parallel-{n:07d}",
                "text": "\n\n".join(blocks),
                "lang": "km" if "km" in langs else langs[0],
                "source": source,
                "license": license,
                "generated_by": generated_by,
                "meta": {"parallel_langs": langs, "sentences": len(chunk)},
            }
        )
    return docs


def to_pairs(items: list[dict[str, str]], *, source: str, license: str) -> list[dict]:
    """Every ordered language pair of every item."""
    pairs = []
    for n, item in enumerate(items):
        for src_lang, tgt_lang in itertools.permutations(sorted(item), 2):
            pairs.append(
                {
                    "id": f"{source}-{n:07d}-{src_lang}-{tgt_lang}",
                    "src": item[src_lang],
                    "tgt": item[tgt_lang],
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "source": source,
                    "license": license,
                }
            )
    return pairs
