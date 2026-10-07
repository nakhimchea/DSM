"""Tokenizer fertility: how many tokens each language costs.

Decoding runs at a roughly constant number of tokens per second, so the characters per
second a user sees are inversely proportional to tokens per character. The original
vocabulary needs ~0.9 tokens per Khmer character against ~0.2 per English one; this report
measures that gap and the effect of a Khmer extension.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fertility:
    chars: int
    bytes: int
    tokens: int
    words: int  # whitespace-separated; meaningful for English only

    @property
    def tokens_per_char(self) -> float:
        return self.tokens / max(self.chars, 1)

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / max(self.tokens, 1)

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / max(self.words, 1)


def fertility(tokenizer, texts: list[str]) -> Fertility:
    """Fertility of a ``tokenizers.Tokenizer`` on DSM-normalised ``texts``."""
    from dsm.text import normalize_text

    chars = bytes_ = tokens = words = 0
    for text in texts:
        text = normalize_text(text)
        chars += len(text)
        bytes_ += len(text.encode("utf-8"))
        tokens += len(tokenizer.encode(text, add_special_tokens=False).ids)
        words += len(text.split())
    return Fertility(chars, bytes_, tokens, words)


def khmer_vocab_entries(tokenizer_json: dict) -> int:
    """Vocabulary entries whose bytes contain at least one complete Khmer character."""
    from dsm.tokenizer.extend import KHMER_CHAR, token_bytes

    count = 0
    for token in tokenizer_json["model"]["vocab"]:
        try:
            text = token_bytes(token).decode("utf-8", errors="ignore")
        except KeyError:
            continue
        count += bool(KHMER_CHAR.search(text))
    return count


def report(tokenizers: dict[str, object], texts: dict[str, list[str]]) -> tuple[dict, str]:
    """Fertility for every tokenizer x language, as data and as a Markdown table."""
    data: dict = {}
    lines = [
        "| Tokenizer | Language | Tokens / char | Bytes / token | Tokens / 1K chars |",
        "|---|---|---:|---:|---:|",
    ]
    for name, tokenizer in tokenizers.items():
        for lang, samples in texts.items():
            f = fertility(tokenizer, samples)
            data[f"{name}/{lang}"] = {
                "tokens_per_char": f.tokens_per_char,
                "bytes_per_token": f.bytes_per_token,
                "tokens_per_word": f.tokens_per_word,
                "tokens": f.tokens,
                "chars": f.chars,
            }
            lines.append(
                f"| {name} | {lang} | {f.tokens_per_char:.3f} | {f.bytes_per_token:.2f} "
                f"| {1000 * f.tokens_per_char:.0f} |"
            )
    return data, "\n".join(lines)
