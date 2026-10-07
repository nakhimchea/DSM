"""PII redaction for training text (placeholders, not partial masks).

Default mode removes contact and payment identifiers: e-mail addresses, phone
numbers, payment card numbers (Luhn-checked), IBANs, IP addresses. ``strict``
mode - for internal bank documents (``internal: true`` sources) - also
replaces account-like identifiers and any run of 6+ digits (account, customer,
ID-card and case numbers). Names are not detected automatically: review
internal data manually, and keep models trained on it internal.
"""

from __future__ import annotations

import re
from collections import Counter

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
# International (+855 / +86 / +1 ...) or Cambodian local (leading 0) numbers; bare digit runs are not phones.
_PHONE = re.compile(r"(?<![\w.+])(?:\+\d{1,3}[ -]?|0)\(?\d{2,3}\)?(?:[ -]?\d{3,4}){2}(?!\w)")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_ACCOUNT_ID = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
_LONG_NUMBER = re.compile(r"(?<![\d.,])\d{6,}(?![\d.,])")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit *= 2
            digit -= 9 if digit > 9 else 0
        total += digit
    return total % 10 == 0


def redact(text: str, strict: bool = False) -> tuple[str, Counter]:
    """Replace PII with ``<EMAIL>``, ``<PHONE>``, ``<CARD>``, ``<IBAN>``, ``<IP>`` (+ ``<ACCOUNT>``, ``<NUMBER>``)."""
    counts: Counter = Counter()

    def sub(pattern: re.Pattern, label: str, check=None):
        nonlocal text

        def repl(match: re.Match) -> str:
            if check is not None and not check(match.group(0)):
                return match.group(0)
            counts[label] += 1
            return f"<{label}>"

        text = pattern.sub(repl, text)

    sub(_EMAIL, "EMAIL")
    sub(_CARD, "CARD", luhn_ok)
    sub(_IBAN, "IBAN")
    sub(_IPV4, "IP")
    sub(_PHONE, "PHONE")
    if strict:
        sub(_ACCOUNT_ID, "ACCOUNT")
        sub(_LONG_NUMBER, "NUMBER")
    return text, counts
