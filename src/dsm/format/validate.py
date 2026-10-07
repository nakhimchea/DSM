"""Markdown and LaTeX validity checks for training answers.

Used as a filter on every builder / teacher output in the ``format``, ``math``,
``tables`` and ``chat`` categories, and by the format evaluation suite.

House style (docs/09): ``$...$`` inline math, ``$$...$$`` display math, money is
written with currency codes ("100 USD", "4,100 KHR") rather than "$" so the
dollar sign stays unambiguous, fenced code blocks carry a language tag.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")


def _cells(line: str) -> int:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return len(re.split(r"(?<!\\)\|", body))


def check_markdown(text: str) -> list[str]:
    problems = []
    lines = text.splitlines()
    fences = [i for i, line in enumerate(lines) if _FENCE.match(line)]
    if len(fences) % 2:
        problems.append("unclosed code fence")
    inside = set()
    for start, end in zip(fences[::2], fences[1::2], strict=False):
        inside.update(range(start, end + 1))
    i = 0
    while i < len(lines):
        if i not in inside and i + 1 < len(lines) and "|" in lines[i] and _TABLE_SEP.match(lines[i + 1]):
            columns = _cells(lines[i])
            if _cells(lines[i + 1]) != columns:
                problems.append(f"table separator has {_cells(lines[i + 1])} columns, header has {columns}")
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                if _cells(lines[j]) != columns:
                    problems.append(
                        f"table row {j - i - 1} has {_cells(lines[j])} cells, header has {columns}"
                    )
                j += 1
            i = j
            continue
        if i not in inside and re.match(r"^#{1,6}[^#\s]", lines[i]):
            problems.append(f"heading without a space: {lines[i][:30]!r}")
        i += 1
    return problems


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", " ", text)


def math_segments(text: str) -> list[str]:
    """Display ($$..$$, \\[..\\]) and inline ($..$, \\(..\\)) math segments (code removed)."""
    body = _strip_code(text)
    segments = re.findall(r"\$\$(.+?)\$\$", body, flags=re.S) + re.findall(r"\\\[(.+?)\\\]", body, flags=re.S)
    body = re.sub(r"\$\$.+?\$\$", " ", body, flags=re.S)
    segments += re.findall(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$", body)
    segments += re.findall(r"\\\((.+?)\\\)", body, flags=re.S)
    return segments


def check_latex(text: str) -> list[str]:
    problems = []
    body = _strip_code(text)
    without_display = re.sub(r"\$\$.+?\$\$", " ", body, flags=re.S)
    if without_display.count("$$"):
        problems.append("unbalanced $$")
    singles = len(re.findall(r"(?<!\\)\$", without_display))
    if singles % 2:
        problems.append("unbalanced $ (write money as '100 USD', not '$100')")
    for segment in math_segments(text):
        depth = 0
        for ch in re.sub(r"\\[{}]", "", segment):
            depth += ch == "{"
            depth -= ch == "}"
            if depth < 0:
                break
        if depth:
            problems.append(f"unbalanced braces in math: {segment[:40]!r}")
        stack = []
        for kind, env in _ENV.findall(segment):
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                problems.append(f"mismatched \\end{{{env}}}")
        if stack:
            problems.append(f"unclosed \\begin{{{stack[-1]}}}")
    try:
        from pylatexenc.latexwalker import LatexWalker, LatexWalkerError

        for segment in math_segments(text):
            try:
                LatexWalker(segment, tolerant_parsing=False).get_latex_nodes()
            except LatexWalkerError as exc:
                problems.append(f"LaTeX parse error: {str(exc).splitlines()[0][:80]}")
    except ImportError:  # pragma: no cover - pylatexenc is in the [data] extra
        pass
    return problems


def check_format(text: str) -> list[str]:
    return check_markdown(text) + check_latex(text)


_MONEY = re.compile(r"(?<![\\$\w])\$\s?(\d[\d,]*(?:\.\d+)?)(\s+(?:thousand|million|billion|trillion)\b)?")


def dollars_to_usd(text: str) -> str:
    """Rewrite plain-text money ("$1,200", "$1.5 million") to the house style ("1,200 USD").

    For sources without LaTeX (e.g. grade-school word problems); do not apply it
    to text that uses ``$`` as a math delimiter.
    """
    return _MONEY.sub(lambda m: f"{m.group(1)}{m.group(2) or ''} USD", text)
