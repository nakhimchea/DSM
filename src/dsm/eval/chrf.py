"""chrF / chrF++ (Popović 2015, 2017) without dependencies, matching sacrebleu's defaults.

Character n-grams (n = 1..6) on text with whitespace removed, word n-grams
(n = 1..2 for chrF++) on whitespace tokens with sacrebleu's punctuation
splitting, averaged precision / recall per order, F-beta with beta = 2.
Corpus-level statistics are summed over sentences before the F-score.
"""

from __future__ import annotations

from collections import Counter

_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _char_ngrams(text: str, n: int) -> Counter:
    chars = "".join(text.split())  # all whitespace removed, as in sacrebleu
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def _words(text: str) -> list[str]:
    out = []
    for token in text.split():
        # sacrebleu: separate a leading / trailing punctuation character from the word
        if len(token) > 1 and token[-1] in _PUNCT:
            out += [token[:-1], token[-1]]
        elif len(token) > 1 and token[0] in _PUNCT:
            out += [token[0], token[1:]]
        else:
            out.append(token)
    return out


def _word_ngrams(text: str, n: int) -> Counter:
    words = _words(text)
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _stats(hypothesis: str, reference: str, char_order: int, word_order: int) -> list[tuple[int, int, int]]:
    rows = []
    for n in range(1, char_order + 1):
        hyp, ref = _char_ngrams(hypothesis, n), _char_ngrams(reference, n)
        rows.append((sum((hyp & ref).values()), sum(hyp.values()), sum(ref.values())))
    for n in range(1, word_order + 1):
        hyp, ref = _word_ngrams(hypothesis, n), _word_ngrams(reference, n)
        rows.append((sum((hyp & ref).values()), sum(hyp.values()), sum(ref.values())))
    return rows


def _fscore(rows: list[tuple[int, int, int]], beta: float) -> float:
    precisions, recalls = [], []
    for match, hyp, ref in rows:
        if hyp > 0 and ref > 0:
            precisions.append(match / hyp)
            recalls.append(match / ref)
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def corpus_chrf(
    hypotheses: list[str],
    references: list[str],
    *,
    word_order: int = 2,
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Corpus chrF++ (``word_order=2``) or chrF (``word_order=0``) in [0, 100]."""
    totals: list[list[int]] | None = None
    for hyp, ref in zip(hypotheses, references, strict=True):
        rows = _stats(hyp, ref, char_order, word_order)
        if totals is None:
            totals = [list(r) for r in rows]
        else:
            for total, row in zip(totals, rows, strict=True):
                for k in range(3):
                    total[k] += row[k]
    return _fscore([tuple(t) for t in totals or []], beta)
