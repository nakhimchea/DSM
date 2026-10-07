"""Extend the byte-level BPE vocabulary with Khmer tokens - without moving any existing id.

1. Khmer text is normalised and pre-tokenised exactly as the tokenizer does it.
2. Every Khmer pre-token is segmented with the EXISTING merges.
3. BPE continues on those segments: the most frequent adjacent pair is merged, but only
   into a brand-new token whose bytes are a run of Khmer UTF-8 (U+1780-U+17FF, possibly
   cut mid-character at either end, after at most one space) that contains the start of
   a Khmer character (bytes E1 9E / E1 9F).
4. New tokens get consecutive ids after the last existing one (248,070 for the DSM
   vocabulary, so the embedding matrix's 250 unused padding rows are filled first);
   the new merges are appended AFTER all existing ones.

Why nothing else changes. BPE applies merges by rank, so on any text the existing
merges run first, exactly as before. A new merge can only fire where its bytes occur,
and those bytes include the start of a Khmer character: text without Khmer encodes to
the same ids as before. On Khmer text the new merges only join what the old ones left,
and since every new merge creates a token no old merge uses, applying them by rank
equals this learner's replay (``encode_with_extension``). Tests check all three
properties against the real 248K vocabulary.
"""

from __future__ import annotations

import heapq
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

KHMER_CHAR = re.compile("[\u1780-\u17ff]")
# UTF-8 of U+1780-U+17FF is E1 (9E|9F) (80-BF). A fragment is the tail of a character,
# whole characters, then the head of a character - each part optional.
_FRAGMENT = re.compile(
    rb"(?:[\x9e\x9f][\x80-\xbf]|[\x80-\xbf])?(?:\xe1[\x9e\x9f][\x80-\xbf])*(?:\xe1[\x9e\x9f]?)?"
)
_CHAR_START = re.compile(rb"\xe1[\x9e\x9f]")


@lru_cache(maxsize=1)
def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    """The GPT-2 byte <-> unicode mapping used by byte-level BPE."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + extra)
            extra += 1
    to_char = {b: chr(c) for b, c in zip(printable, codes, strict=True)}
    return to_char, {v: k for k, v in to_char.items()}


def token_bytes(token: str) -> bytes:
    """Raw bytes of a byte-level token string (KeyError if it is not one, e.g. a control token)."""
    to_byte = _byte_maps()[1]
    return bytes(to_byte[c] for c in token)


def khmer_piece(token: str) -> bool:
    """May byte-level ``token`` become a new Khmer token? (rule 3 of the module docstring)"""
    try:
        data = token_bytes(token)
    except KeyError:
        return False
    if data[:1] == b" ":
        data = data[1:]
    return bool(_FRAGMENT.fullmatch(data)) and bool(_CHAR_START.search(data))


@dataclass
class Extension:
    merges: list[tuple[str, str]]
    tokens: list[str]  # new token i has id start_id + i
    start_id: int
    counts: list[int] = field(default_factory=list)  # pair frequency when it was merged
    stats: dict = field(default_factory=dict)


def _pretokens(tokenizer, text: str) -> list[str]:
    normalized = tokenizer.normalizer.normalize_str(text) if tokenizer.normalizer else text
    return [piece for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(normalized)]


def _is_khmer_pretoken(piece: str) -> bool:
    return bool(KHMER_CHAR.search(token_bytes(piece).decode("utf-8", errors="replace")))


def khmer_pretokens(tokenizer, texts: Iterable[str], max_chars: int | None = None) -> Counter:
    """Pre-tokens (byte-level strings) that contain Khmer, with counts (texts are DSM-normalised)."""
    from dsm.text import normalize_text

    counts: Counter = Counter()
    seen = 0
    for text in texts:
        text = normalize_text(text)
        if not KHMER_CHAR.search(text):
            continue
        counts.update(piece for piece in _pretokens(tokenizer, text) if _is_khmer_pretoken(piece))
        seen += len(text)
        if max_chars is not None and seen >= max_chars:
            break
    return counts


def _merge(symbols: list[int], a: int, b: int, new_id: int):
    """Merge every (a, b) left to right. Returns the new symbols and the pair-count changes
    around each merge site as (pair, +1 / -1); the merged pair itself is not listed.
    None if (a, b) does not occur."""
    out: list[int] = []
    changes: list[tuple[tuple[int, int], int]] = []
    i, n, merged = 0, len(symbols), False
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            if out:
                changes += [((out[-1], a), -1), ((out[-1], new_id), 1)]
            if i + 2 < n:
                changes += [((b, symbols[i + 2]), -1), ((new_id, symbols[i + 2]), 1)]
            out.append(new_id)
            i += 2
            merged = True
        else:
            out.append(symbols[i])
            i += 1
    return (out, changes) if merged else None


def learn(
    tokenizer,
    texts: Iterable[str],
    n_new: int,
    start_id: int,
    min_count: int = 2,
    max_chars: int | None = None,
) -> Extension:
    """Learn up to ``n_new`` Khmer merges on top of ``tokenizer`` (a ``tokenizers.Tokenizer``)."""
    pretokens = khmer_pretokens(tokenizer, texts, max_chars)
    vocab = tokenizer.get_vocab(with_added_tokens=False)
    strings = {i: s for s, i in vocab.items()}
    words = [[t.id for t in tokenizer.model.tokenize(piece)] for piece in pretokens]
    weights = list(pretokens.values())

    pair_counts: dict[tuple[int, int], int] = {}
    where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for index, symbols in enumerate(words):
        for pair in zip(symbols, symbols[1:], strict=False):
            pair_counts[pair] = pair_counts.get(pair, 0) + weights[index]
            where[pair].add(index)
    # Max-heap with lazy updates: every pair with count c > 0 has an entry >= c.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    before = sum(len(s) * w for s, w in zip(words, weights, strict=True))
    known, rejected = set(vocab), set()
    merges: list[tuple[str, str]] = []
    tokens: list[str] = []
    counts: list[int] = []
    while heap and len(tokens) < n_new:
        negative, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if -negative != count:
            if 0 < count < -negative:  # dropped since: re-queue with the current count
                heapq.heappush(heap, (-count, pair))
            continue
        if count < min_count:
            break
        if pair in rejected:
            continue
        merged = strings[pair[0]] + strings[pair[1]]
        if merged in known or not khmer_piece(merged):
            rejected.add(pair)
            continue
        new_id = start_id + len(tokens)
        strings[new_id] = merged
        known.add(merged)
        merges.append((strings[pair[0]], strings[pair[1]]))
        tokens.append(merged)
        counts.append(count)
        grown: set[tuple[int, int]] = set()
        for index in where.pop(pair):
            result = _merge(words[index], *pair, new_id)
            if result is None:  # stale index: the pair left this word earlier
                continue
            words[index], changes = result
            weight = weights[index]
            for p, sign in changes:
                pair_counts[p] = pair_counts.get(p, 0) + sign * weight
                if sign > 0:
                    where[p].add(index)
                    grown.add(p)
        pair_counts[pair] = 0
        for p in grown - rejected:
            if pair_counts[p] > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
    after = sum(len(s) * w for s, w in zip(words, weights, strict=True))
    stats = {
        "khmer_pretokens": len(pretokens),
        "khmer_pretoken_occurrences": sum(weights),
        "khmer_tokens_before": before,
        "khmer_tokens_after": after,
        "new_tokens": len(tokens),
    }
    return Extension(merges=merges, tokens=tokens, start_id=start_id, counts=counts, stats=stats)


def encode_with_extension(tokenizer, extension: Extension, text: str) -> list[int]:
    """Reference encoder: old segmentation per pre-token, then the new merges by rank.

    Equals ``extended_tokenizer.encode(text, add_special_tokens=False).ids`` for text without
    control tokens - the property the tests check.
    """
    vocab = dict(tokenizer.get_vocab(with_added_tokens=False))
    vocab.update({token: extension.start_id + i for i, token in enumerate(extension.tokens)})
    ranks = {(vocab[a], vocab[b]): (rank, vocab[a + b]) for rank, (a, b) in enumerate(extension.merges)}
    ids: list[int] = []
    for piece in _pretokens(tokenizer, text):
        symbols = [t.id for t in tokenizer.model.tokenize(piece)]
        while len(symbols) > 1:
            candidates = [
                (ranks[pair][0], i)
                for i, pair in enumerate(zip(symbols, symbols[1:], strict=False))
                if pair in ranks
            ]
            if not candidates:
                break
            _, i = min(candidates)
            symbols[i : i + 2] = [ranks[(symbols[i], symbols[i + 1])][1]]
        ids += symbols
    return ids


def next_free_id(tokenizer_json: dict) -> int:
    """First id after every vocabulary entry and control token (the default ``start_id``)."""
    model = tokenizer_json["model"]
    used = list(model["vocab"].values()) + [a["id"] for a in tokenizer_json.get("added_tokens", [])]
    return max(used) + 1


def apply_extension(tokenizer_json: dict, extension: Extension) -> dict:
    """A new tokenizer.json: vocabulary entries from ``start_id`` and merges appended at the end.

    The ``tokenizers`` loader numbers control tokens that are missing from the model
    vocabulary from ``len(vocab)`` on, so growing the vocabulary would silently move
    ``<|im_end|>`` & co. Every control token is therefore also entered into the model
    vocabulary at its id (BPE has no merge that produces one), and the result is loaded
    back to check that no existing id moved.
    """
    from tokenizers import Tokenizer

    data = json.loads(json.dumps(tokenizer_json))
    model = data["model"]
    if model.get("type") != "BPE" or model.get("ignore_merges"):
        raise ValueError("expected a byte-level BPE model without ignore_merges")
    added = {a["content"]: a["id"] for a in data.get("added_tokens", [])}
    if extension.start_id < next_free_id(tokenizer_json):
        raise ValueError(
            f"start_id {extension.start_id} collides with existing ids (next free id {next_free_id(tokenizer_json)})"
        )
    clash = [t for t in extension.tokens if t in model["vocab"] or t in added]
    if clash:
        raise ValueError(f"{len(clash)} new tokens already exist, e.g. {clash[0]!r}")
    for content, index in added.items():
        if model["vocab"].setdefault(content, index) != index:
            raise ValueError(f"control token {content!r} has two ids")
    for offset, token in enumerate(extension.tokens):
        model["vocab"][token] = extension.start_id + offset
    as_lists = bool(model["merges"]) and isinstance(model["merges"][0], list)
    model["merges"] += [[a, b] if as_lists else f"{a} {b}" for a, b in extension.merges]

    loaded = Tokenizer.from_str(json.dumps(data))
    expected = {**tokenizer_json["model"]["vocab"], **added}
    expected.update({t: extension.start_id + i for i, t in enumerate(extension.tokens)})
    moved = [t for t, i in expected.items() if loaded.token_to_id(t) != i]
    if moved:
        raise RuntimeError(f"{len(moved)} token ids moved after extension, e.g. {moved[0]!r}")
    return data


def extend_tokenizer_dir(
    src_dir: str | Path,
    out_dir: str | Path,
    texts: Iterable[str],
    n_new: int,
    start_id: int | None = None,
    min_count: int = 2,
    max_chars: int | None = None,
) -> Extension:
    """Write an extended tokenizer directory.

    ``tokenizer.json`` (extended), the copied ``tokenizer_config.json`` / ``chat_template.jinja``,
    ``extension.json`` (statistics) and ``new_tokens.tsv`` (id, frequency, text - for review).
    """
    from tokenizers import Tokenizer

    src, out = Path(src_dir), Path(out_dir)
    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    original = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    if start_id is None:
        start_id = next_free_id(original)
    extension = learn(tokenizer, texts, n_new, start_id, min_count, max_chars)
    data = apply_extension(original, extension)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        if (src / name).is_file():
            shutil.copyfile(src / name, out / name)
    meta = {"start_id": extension.start_id, "min_count": min_count, **extension.stats}
    (out / "extension.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    rows = ["id\tcount\ttext"]
    for offset, (token, count) in enumerate(zip(extension.tokens, extension.counts, strict=True)):
        text = token_bytes(token).decode("utf-8", errors="backslashreplace").replace(" ", "␠")
        rows.append(f"{extension.start_id + offset}\t{count}\t{text}")
    (out / "new_tokens.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return extension
