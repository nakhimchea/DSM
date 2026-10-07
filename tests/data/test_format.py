"""Markdown / LaTeX validator and the house-style money rewrite."""

from __future__ import annotations

import pytest

from dsm.format.validate import check_format, check_latex, check_markdown, dollars_to_usd, math_segments

GOOD = """## Loan summary

| Month | Payment (USD) |
|---|---:|
| 1 | 444.89 |
| 2 | 444.89 |

The payment is $P = \\frac{rL}{1-(1+r)^{-n}}$, and over $n$ months:

$$T = nP$$

```python
print(round(444.89 * 12, 2))
```
"""


def test_good_answer_passes():
    assert check_format(GOOD) == []
    assert math_segments(GOOD) == ["T = nP", "P = \\frac{rL}{1-(1+r)^{-n}}", "n"]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("```python\nprint(1)\n", "unclosed code fence"),
        ("| a | b |\n|---|---|\n| 1 |\n", "table row"),
        ("| a | b |\n|---|\n| 1 | 2 |\n", "separator"),
        ("##Heading", "heading without a space"),
    ],
)
def test_markdown_problems(text, fragment):
    assert any(fragment in p for p in check_markdown(text))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("It costs $5 per month.", "unbalanced $"),
        ("$$x + 1", "unbalanced $$"),
        ("$\\frac{1}{2$", "unbalanced braces"),
        ("$$\\begin{aligned} x \\end{array}$$", "mismatched"),
        ("$$\\begin{aligned} x = 1$$", "unclosed"),
    ],
)
def test_latex_problems(text, fragment):
    assert any(fragment in p for p in check_latex(text))


def test_code_is_ignored_and_escaped_dollars_are_fine():
    assert check_format("Use `$HOME` or\n```bash\necho $PATH\n```\n") == []
    assert check_format("The price is \\$5.") == []


def test_dollars_to_usd():
    assert (
        dollars_to_usd("Pay $5 and $1,200.50, or $1.5 million.")
        == "Pay 5 USD and 1,200.50 USD, or 1.5 million USD."
    )
    assert dollars_to_usd("Keep \\$3 and US$4 as is") == "Keep \\$3 and US$4 as is"
    assert check_format(dollars_to_usd("A pen costs $3 and a book $12.")) == []
