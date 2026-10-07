"""A small Excel formula evaluator (no third-party dependencies).

Used to *verify* every formula that appears in generated training data: a
formula is only kept if evaluating it on the sheet gives the same result as the
pandas ground truth. Supported: arithmetic (+ - * / ^ &), comparisons, cell
references and ranges (``A1``, ``$A$1``, ``A2:C9``), strings, booleans and the
functions in ``FUNCTIONS`` (SUM/SUMIFS/COUNTIFS/AVERAGEIFS/XLOOKUP/VLOOKUP/
INDEX/MATCH/PMT/IF/ROUND/...).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<str>\"(?:[^\"]|\"\")*\")"
    r"|(?P<range>\$?[A-Z]{1,3}\$?\d+:\$?[A-Z]{1,3}\$?\d+)"
    r"|(?P<cell>\$?[A-Z]{1,3}\$?\d+)"
    r"|(?P<func>[A-Z][A-Z0-9\.]*)(?=\s*\()"
    r"|(?P<bool>TRUE|FALSE)"
    r"|(?P<op><>|<=|>=|[-+*/^&=<>(),]))"
)


class FormulaError(ValueError):
    pass


def col_to_index(col: str) -> int:
    index = 0
    for ch in col:
        index = index * 26 + (ord(ch) - 64)
    return index


def index_to_col(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def split_ref(ref: str) -> tuple[str, int]:
    match = re.fullmatch(r"\$?([A-Z]{1,3})\$?(\d+)", ref)
    if not match:
        raise FormulaError(f"bad cell reference {ref!r}")
    return match.group(1), int(match.group(2))


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise FormulaError(f"cannot parse formula at {text[pos : pos + 10]!r}")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        pos = match.end()
    return tokens


class Sheet:
    """Cell grid: ``{"A1": value}``. Missing cells are empty (None)."""

    def __init__(self, cells: dict[str, Any]) -> None:
        self.cells = cells

    def get(self, ref: str) -> Any:
        col, row = split_ref(ref)
        return self.cells.get(f"{col}{row}")

    def range(self, ref: str) -> list[list[Any]]:
        start, end = ref.split(":")
        (c1, r1), (c2, r2) = split_ref(start), split_ref(end)
        cols = range(col_to_index(c1), col_to_index(c2) + 1)
        return [[self.cells.get(f"{index_to_col(c)}{r}") for c in cols] for r in range(r1, r2 + 1)]


def _flatten(values: Any) -> list[Any]:
    if isinstance(values, list):
        out = []
        for value in values:
            out.extend(_flatten(value))
        return out
    return [values]


def _numbers(values: Any) -> list[float]:
    return [v for v in _flatten(values) if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _criterion(criterion: Any) -> Callable[[Any], bool]:
    """Excel criteria such as ``">100"``, ``"<>Closed"``, ``"Phnom Penh"`` or ``5``."""
    if not isinstance(criterion, str):
        return lambda v: v == criterion
    match = re.fullmatch(r"(<>|<=|>=|=|<|>)?(.*)", criterion, re.S)
    op, operand = match.group(1) or "=", match.group(2)
    try:
        target: Any = float(operand)
    except ValueError:
        target = operand

    def test(value: Any) -> bool:
        if isinstance(target, float):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return op == "<>"
            return {
                "=": value == target,
                "<>": value != target,
                "<": value < target,
                ">": value > target,
                "<=": value <= target,
                ">=": value >= target,
            }[op]
        text = "" if value is None else str(value)
        if op == "=":
            return text.lower() == target.lower()
        if op == "<>":
            return text.lower() != target.lower()
        return {"<": text < target, ">": text > target, "<=": text <= target, ">=": text >= target}[op]

    return test


def _ifs(values: list[Any], pairs: list[Any]) -> list[bool]:
    mask = [True] * len(values)
    for crit_range, crit in zip(pairs[::2], pairs[1::2], strict=True):
        test = _criterion(crit)
        cells = _flatten(crit_range)
        if len(cells) != len(values):
            raise FormulaError("criteria range size differs from the value range")
        mask = [m and test(c) for m, c in zip(mask, cells, strict=True)]
    return mask


def _pmt(rate: float, nper: float, pv: float, fv: float = 0, when: float = 0) -> float:
    if rate == 0:
        return -(pv + fv) / nper
    factor = (1 + rate) ** nper
    return -(rate * (pv * factor + fv)) / ((1 + rate * when) * (factor - 1))


def _round(value: float, digits: float = 0, mode: str = "half_up") -> float:
    scale = 10 ** int(digits)
    shifted = value * scale
    if mode == "up":
        shifted = math.copysign(math.ceil(abs(shifted) - 1e-12), shifted)
    elif mode == "down":
        shifted = math.copysign(math.floor(abs(shifted) + 1e-12), shifted)
    else:
        shifted = math.copysign(math.floor(abs(shifted) + 0.5 + 1e-12), shifted)
    result = shifted / scale
    return int(result) if digits <= 0 else result


def _lookup(value: Any, column: list[Any]) -> int:
    for i, cell in enumerate(column):
        if cell == value or (
            isinstance(cell, str) and isinstance(value, str) and cell.lower() == value.lower()
        ):
            return i
    raise FormulaError(f"#N/A: {value!r} not found")


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "SUM": lambda *a: sum(_numbers(list(a))),
    "AVERAGE": lambda *a: (lambda n: sum(n) / len(n))(_numbers(list(a))),
    "MIN": lambda *a: min(_numbers(list(a))),
    "MAX": lambda *a: max(_numbers(list(a))),
    "COUNT": lambda *a: len(_numbers(list(a))),
    "COUNTA": lambda *a: sum(1 for v in _flatten(list(a)) if v not in (None, "")),
    "ABS": abs,
    "ROUND": lambda v, d=0: _round(v, d),
    "ROUNDUP": lambda v, d=0: _round(v, d, "up"),
    "ROUNDDOWN": lambda v, d=0: _round(v, d, "down"),
    "IF": lambda cond, a, b=False: a if cond else b,
    "AND": lambda *a: all(_flatten(list(a))),
    "OR": lambda *a: any(_flatten(list(a))),
    "NOT": lambda a: not a,
    "LEN": lambda s: len(str(s)),
    "UPPER": lambda s: str(s).upper(),
    "LOWER": lambda s: str(s).lower(),
    "LEFT": lambda s, n=1: str(s)[: int(n)],
    "RIGHT": lambda s, n=1: str(s)[-int(n) :] if int(n) else "",
    "CONCAT": lambda *a: "".join("" if v is None else str(v) for v in _flatten(list(a))),
    "SUMIF": lambda rng, crit, sum_rng=None: sum(
        v
        for v, ok in zip(
            _flatten(sum_rng if sum_rng is not None else rng), _ifs(_flatten(rng), [rng, crit]), strict=True
        )
        if ok and isinstance(v, (int, float))
    ),
    "SUMIFS": lambda sum_rng, *pairs: sum(
        v
        for v, ok in zip(_flatten(sum_rng), _ifs(_flatten(sum_rng), list(pairs)), strict=True)
        if ok and isinstance(v, (int, float))
    ),
    "COUNTIF": lambda rng, crit: sum(_ifs(_flatten(rng), [rng, crit])),
    "COUNTIFS": lambda *pairs: sum(_ifs(_flatten(pairs[0]), list(pairs))),
    "AVERAGEIFS": lambda avg_rng, *pairs: (
        lambda vals: sum(vals) / len(vals) if vals else (_ for _ in ()).throw(FormulaError("#DIV/0!"))
    )(
        [
            v
            for v, ok in zip(_flatten(avg_rng), _ifs(_flatten(avg_rng), list(pairs)), strict=True)
            if ok and isinstance(v, (int, float))
        ]
    ),
    "MAXIFS": lambda max_rng, *pairs: max(
        v
        for v, ok in zip(_flatten(max_rng), _ifs(_flatten(max_rng), list(pairs)), strict=True)
        if ok and isinstance(v, (int, float))
    ),
    "MINIFS": lambda min_rng, *pairs: min(
        v
        for v, ok in zip(_flatten(min_rng), _ifs(_flatten(min_rng), list(pairs)), strict=True)
        if ok and isinstance(v, (int, float))
    ),
    "XLOOKUP": lambda value, look, ret, *rest: _flatten(ret)[_lookup(value, _flatten(look))],
    "VLOOKUP": lambda value, table, col, exact=False: table[_lookup(value, [row[0] for row in table])][
        int(col) - 1
    ],
    "INDEX": lambda table, row, col=1: (
        table[int(row) - 1][int(col) - 1] if isinstance(table[0], list) else table[int(row) - 1]
    ),
    "MATCH": lambda value, rng, mode=0: _lookup(value, _flatten(rng)) + 1,
    "PMT": _pmt,
}


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], sheet: Sheet) -> None:
        self.tokens, self.pos, self.sheet = tokens, 0, sheet

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        token = self.peek()
        if token is None or (value is not None and token[1] != value):
            raise FormulaError(f"expected {value!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Any:
        value = self.comparison()
        if self.peek() is not None:
            raise FormulaError(f"unexpected token {self.peek()!r}")
        return value

    def comparison(self) -> Any:
        left = self.concat()
        while (token := self.peek()) and token[1] in ("=", "<>", "<", ">", "<=", ">="):
            op = self.take()[1]
            right = self.concat()
            left = {
                "=": left == right,
                "<>": left != right,
                "<": left < right,
                ">": left > right,
                "<=": left <= right,
                ">=": left >= right,
            }[op]
        return left

    def concat(self) -> Any:
        left = self.additive()
        while (token := self.peek()) and token[1] == "&":
            self.take()
            left = f"{'' if left is None else left}{'' if (r := self.additive()) is None else r}"
        return left

    def additive(self) -> Any:
        left = self.term()
        while (token := self.peek()) and token[1] in ("+", "-"):
            op = self.take()[1]
            right = self.term()
            left = (left or 0) + (right or 0) if op == "+" else (left or 0) - (right or 0)
        return left

    def term(self) -> Any:
        left = self.power()
        while (token := self.peek()) and token[1] in ("*", "/"):
            op = self.take()[1]
            right = self.power()
            if op == "/" and not right:
                raise FormulaError("#DIV/0!")
            left = (left or 0) * (right or 0) if op == "*" else (left or 0) / right
        return left

    def power(self) -> Any:
        left = self.unary()
        while (token := self.peek()) and token[1] == "^":
            self.take()
            left = left ** self.unary()
        return left

    def unary(self) -> Any:
        token = self.peek()
        if token and token[1] in ("-", "+"):
            self.take()
            value = self.unary()
            return -value if token[1] == "-" else value
        return self.primary()

    def primary(self) -> Any:
        token = self.peek()
        if token is None:
            raise FormulaError("unexpected end of formula")
        kind, text = token
        if kind == "num":
            self.take()
            number = float(text)
            return (
                int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number
            )
        if kind == "str":
            self.take()
            return text[1:-1].replace('""', '"')
        if kind == "bool":
            self.take()
            return text == "TRUE"
        if kind == "range":
            self.take()
            return self.sheet.range(text.replace("$", ""))
        if kind == "cell":
            self.take()
            return self.sheet.get(text.replace("$", ""))
        if kind == "func":
            self.take()
            name = text.upper()
            if name not in FUNCTIONS:
                raise FormulaError(f"unsupported function {name}")
            self.take("(")
            args = []
            if self.peek() and self.peek()[1] != ")":
                args.append(self.comparison())
                while self.peek() and self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.comparison())
            self.take(")")
            return FUNCTIONS[name](*args)
        if text == "(":
            self.take("(")
            value = self.comparison()
            self.take(")")
            return value
        raise FormulaError(f"unexpected token {token!r}")


def evaluate_formula(formula: str, cells: dict[str, Any] | Sheet) -> Any:
    """Evaluate ``=...`` against a cell grid ``{"A1": value, ...}``."""
    sheet = cells if isinstance(cells, Sheet) else Sheet(cells)
    text = formula.strip()
    if not text.startswith("="):
        raise FormulaError("a formula starts with '='")
    return _Parser(_tokenize(text[1:]), sheet).parse()
