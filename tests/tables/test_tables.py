"""Formula evaluator, sheet serialiser (incl. .xlsx / .csv) and the table-task builder."""

from __future__ import annotations

import datetime as dt
import random
import re

import pytest

from dsm import repo_root
from dsm.data.builders.tables import TableTaskBuilder
from dsm.data.sft.validate import validate_sft
from dsm.tables.formula import FormulaError, col_to_index, evaluate_formula, index_to_col
from dsm.tables.generate import KEYS, generate_table
from dsm.tables.serialize import cells_from_rows, format_cell, grid_to_text, parse_grid, workbook_to_text

ROWS = [
    ["Branch", "Officer", "Amount", "Rate", "Status"],
    ["Phnom Penh", "Dara", 12000, 12, "Active"],
    ["Siem Reap", "Lina", 8000, 10.5, "Closed"],
    ["Phnom Penh", "Sok", 5500, 9, "Active"],
    ["Kampot", "Vanna", 3000, 15, "Active"],
]
CELLS = cells_from_rows(ROWS)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("=SUM(C2:C5)", 28500),
        ("=AVERAGE(C2:C5)", 7125),
        ("=MAX(C2:C5)-MIN(C2:C5)", 9000),
        ('=SUMIFS(C2:C5,A2:A5,"Phnom Penh")', 17500),
        ('=SUMIFS(C2:C5,A2:A5,"Phnom Penh",E2:E5,"Active")', 17500),
        ('=COUNTIFS(E2:E5,"Active")', 3),
        ('=COUNTIF(C2:C5,">5000")', 3),
        ('=AVERAGEIFS(D2:D5,E2:E5,"Active")', 12),
        ('=XLOOKUP("Lina",B2:B5,C2:C5)', 8000),
        ('=VLOOKUP("Kampot",A2:E5,3,FALSE)', 3000),
        ('=INDEX(C2:C5,MATCH("Sok",B2:B5,0))', 5500),
        ('=IF(C2>10000,"big","small")', "big"),
        ("=ROUND(C3/3,2)", 2666.67),
        ('=A2&" / "&B2', "Phnom Penh / Dara"),
        ("=C2*(1+D2/100)", 13440),
        ("=2^10", 1024),
        ('=LEN("ខ្មែរ")', 5),
    ],
)
def test_formulas(formula, expected):
    value = evaluate_formula(formula, CELLS)
    if isinstance(expected, str):
        assert value == expected
    else:
        assert value == pytest.approx(expected)


def test_pmt_matches_closed_form():
    rate, months, principal = 0.12 / 12, 24, 5000
    expected = principal * rate / (1 - (1 + rate) ** -months)
    assert evaluate_formula(f"=PMT({rate},{months},-{principal})", {}) == pytest.approx(expected)


@pytest.mark.parametrize("formula", ["=FOO(A1)", "=SUM(A1:", "=1+", '=XLOOKUP("nobody",B2:B5,C2:C5)'])
def test_formula_errors(formula):
    with pytest.raises(FormulaError):
        evaluate_formula(formula, CELLS)


def test_column_letters():
    for index in (1, 26, 27, 52, 703):
        assert col_to_index(index_to_col(index)) == index
    assert index_to_col(28) == "AB"


def test_format_cell():
    assert format_cell(1200.0) == "1200"
    assert format_cell(0.1 + 0.2) == "0.3"
    assert format_cell(True) == "TRUE"
    assert format_cell(dt.datetime(2026, 1, 5)) == "2026-01-05"
    assert format_cell("a|b\nc") == "a\\|b c"
    assert format_cell("x" * 500).endswith("…")
    assert format_cell(float("nan")) == ""


def test_grid_text_and_round_trip():
    text = grid_to_text(ROWS, "Loans")
    assert text.splitlines()[0] == '[Sheet "Loans" | A1:E5 | 4 data rows]'
    assert "| 3 | Siem Reap | Lina | 8000 | 10.5 | Closed |" in text
    sheet, cells = parse_grid(text)
    assert sheet == "Loans" and cells == {k: v for k, v in CELLS.items() if v is not None}
    markdown = grid_to_text(ROWS, "Loans", style="markdown")
    assert "| Branch | Officer | Amount | Rate | Status |" in markdown
    long = grid_to_text([ROWS[0]] + ROWS[1:] * 30, "Big", max_rows=10)
    assert "110 more rows not shown" in long and "table_query" in long


def test_workbook_and_csv(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Loans"
    for row in ROWS[:3]:
        ws.append(row)
    ws["C4"] = "=SUM(C2:C3)"
    ws.merge_cells("A5:B5")
    ws["A5"] = "Note"
    path = tmp_path / "loans.xlsx"
    wb.save(path)
    text = workbook_to_text(path)
    assert '[Sheet "Loans"' in text and "=SUM(C2:C3)" in text and "Merged cells: A5:B5" in text
    assert "| 5 | Note | Note |" in text  # merged value repeated across the range

    csv_path = tmp_path / "rates.csv"
    csv_path.write_text("Currency,Rate\nKHR,4100\nTHB,35.5\n", encoding="utf-8")
    sheet, cells = parse_grid(workbook_to_text(csv_path))
    assert sheet == "rates" and cells["B2"] == 4100 and cells["B3"] == 35.5


def test_generate_table_is_seeded_and_consistent():
    a = generate_table("loans", "km", random.Random(5))
    b = generate_table("loans", "km", random.Random(5))
    assert a.rows == b.rows
    assert list(a.frame.columns) == KEYS["loans"]
    assert len(a.rows) == len(a.frame) + 1
    amount = a.columns["amount"]
    assert [r[amount] for r in a.rows[1:]] == a.frame["amount"].tolist()


_FORMULA = re.compile(r"```excel\n(=.+?)\n```")
_BOLD_NUMBER = re.compile(r"\*\*(-?[\d,]+(?:\.\d+)?)\*\*")


def test_table_tasks_valid_and_formulas_recompute():
    records = TableTaskBuilder(repo_root() / "data/seeds/tables/tasks.yaml", seed=2).build(per_weight=2)
    assert records and all(validate_sft(r) == [] for r in records)
    tasks = {r["meta"]["task"] for r in records}
    assert {"sum_by", "lookup", "formula_sum", "formula_lookup", "formula_pmt", "group_table"} <= tasks
    checked = 0
    for record in records:
        if not record["meta"]["task"].startswith("formula"):
            continue
        user, answer = record["messages"][0]["content"], record["messages"][1]["content"]
        grid = user[user.index("[Sheet") :]
        grid = grid[: grid.find("\n\n")] if "\n\n" in grid else grid
        _, cells = parse_grid(grid)
        formula = _FORMULA.search(answer).group(1)
        stated = float(_BOLD_NUMBER.findall(answer)[-1].replace(",", ""))
        value = evaluate_formula(formula, cells)
        assert round(value, 2) == pytest.approx(stated, abs=0.006), (formula, value, stated)
        checked += 1
    assert checked >= 10
