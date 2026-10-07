"""Spreadsheet / table understanding conversations with verified answers.

For each generated sheet (``dsm.tables.generate``) a task is drawn: look-ups,
counts, sums / averages by category, maxima, top-k, group tables, percentage
change, and Excel formula writing (SUMIFS, COUNTIFS, XLOOKUP, PMT). Answers are
computed with pandas; every formula is evaluated with ``dsm.tables.formula`` and
the sample is dropped if it disagrees with the ground truth. 25 % of samples ask
in Khmer / Chinese about a sheet with English headers (common in practice).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

from dsm.data.sft.schema import sft_record
from dsm.tables.formula import evaluate_formula
from dsm.tables.generate import ROLES, GeneratedTable, generate_table
from dsm.tables.serialize import cells_from_rows, grid_to_text
from dsm.text import normalize_text
from dsm.tools.bank_sim import BankSim

LANGS = ("en", "km", "zh")
SUMMABLE = {"amount", "deposits", "new_accounts", "units", "revenue", "salary", "bonus"}
AVERAGEABLE = SUMMABLE | {"rate", "term", "unit_price"}
TASKS = {
    "sum_by": 3,
    "avg_by": 2,
    "count_by": 2,
    "max_row": 2,
    "lookup": 2,
    "group_table": 2,
    "top_k": 1,
    "formula_sum": 3,
    "formula_count": 1,
    "formula_lookup": 2,
    "formula_pmt": 1,
    "pct_change": 1,
}
TABLE_KINDS = ("loans", "deposits", "sales", "payroll")


def fmt(value: float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}"


def _close(a, b) -> bool:
    return isinstance(a, (int, float)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6)


class TableTaskBuilder:
    def __init__(self, seeds_path: str | Path, seed: int = 0) -> None:
        self.t = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)
        self._counter = 0

    # -- helpers ---------------------------------------------------------------------------
    def _q(self, task: str, lang: str, **fill) -> str:
        return self.rng.choice(self.t[task]["q"][lang]).format(**fill)

    def _rows_text(self, lang: str, excel_rows: list[int]) -> str:
        sep = "、" if lang == "zh" else ", "
        return self.t["rows_word"][lang].format(rows=sep.join(str(r) for r in excel_rows))

    @staticmethod
    def _expression(values: list[float]) -> str:
        parts = [fmt(v) for v in values]
        return " + ".join(parts) if len(parts) <= 8 else " + ".join(parts[:3]) + " + … + " + parts[-1]

    def _cells(self, table: GeneratedTable) -> dict:
        return cells_from_rows(table.rows)

    def _category(self, table: GeneratedTable) -> tuple[str, object, list[int]]:
        key = self.rng.choice(ROLES[table.kind]["categories"])
        value = self.rng.choice(sorted(set(table.frame[key]), key=str))
        idx = table.frame.index[table.frame[key] == value].tolist()
        return key, value, idx

    # -- tasks (return (question, answer) or None) --------------------------------------------
    def sum_by(self, table: GeneratedTable, lang: str, as_formula: bool = False):
        measures = [m for m in ROLES[table.kind]["measures"] if m in SUMMABLE]
        if not measures:
            return None
        measure = self.rng.choice(measures)
        cat, value, idx = self._category(table)
        values = table.frame.loc[idx, measure].tolist()
        result = sum(values)
        formula = f'=SUMIFS({table.range(measure)},{table.range(cat)},"{value}")'
        if not _close(evaluate_formula(formula, self._cells(table)), result):
            return None
        fill = {"measure": table.header(measure), "cat_col": table.header(cat), "cat_value": value}
        if as_formula:
            explain = self.t["formula_sum"]["explain"][lang].format(
                sum_range=table.range(measure), crit_range=table.range(cat), cat_value=value
            )
            answer = self.t["formula_answer"][lang].format(
                formula=formula, explain=explain, result=fmt(result)
            )
            return self._q("formula_sum", lang, **fill), answer
        answer = self.t["sum_by"]["a"][lang].format(
            **fill,
            rows_text=self._rows_text(lang, [i + 2 for i in idx]),
            expression=self._expression(values),
            result=fmt(result),
        )
        return self._q("sum_by", lang, **fill), answer

    def avg_by(self, table: GeneratedTable, lang: str):
        measures = [m for m in ROLES[table.kind]["measures"] if m in AVERAGEABLE]
        measure = self.rng.choice(measures)
        cat, value, idx = self._category(table)
        values = table.frame.loc[idx, measure].tolist()
        result = round(sum(values) / len(values), 2)
        formula = f'=ROUND(AVERAGEIFS({table.range(measure)},{table.range(cat)},"{value}"),2)'
        if not _close(evaluate_formula(formula, self._cells(table)), result):
            return None
        fill = {"measure": table.header(measure), "cat_col": table.header(cat), "cat_value": value}
        answer = self.t["avg_by"]["a"][lang].format(
            **fill,
            rows_text=self._rows_text(lang, [i + 2 for i in idx]),
            count=len(values),
            expression=self._expression(values),
            result=fmt(result),
        )
        return self._q("avg_by", lang, **fill), answer

    def count_by(self, table: GeneratedTable, lang: str, as_formula: bool = False):
        cat, value, idx = self._category(table)
        formula = f'=COUNTIFS({table.range(cat)},"{value}")'
        if evaluate_formula(formula, self._cells(table)) != len(idx):
            return None
        fill = {"cat_col": table.header(cat), "cat_value": value}
        if as_formula:
            explain = self.t["formula_count"]["explain"][lang].format(
                crit_range=table.range(cat), cat_value=value
            )
            answer = self.t["formula_answer"][lang].format(formula=formula, explain=explain, result=len(idx))
            return self._q("formula_count", lang, **fill), answer
        answer = self.t["count_by"]["a"][lang].format(
            **fill, rows_text=self._rows_text(lang, [i + 2 for i in idx]), result=len(idx)
        )
        return self._q("count_by", lang, **fill), answer

    def max_row(self, table: GeneratedTable, lang: str):
        measures = [m for m in ROLES[table.kind]["measures"] if m in SUMMABLE]
        measure = self.rng.choice(measures)
        column = table.frame[measure]
        if (column == column.max()).sum() != 1:
            return None  # ties make "which row" ambiguous
        i = int(column.idxmax())
        name_key = ROLES[table.kind]["name"]
        if not _close(evaluate_formula(f"=MAX({table.range(measure)})", self._cells(table)), column.max()):
            return None
        answer = self.t["max_row"]["a"][lang].format(
            measure=table.header(measure),
            result=fmt(column.max()),
            row=i + 2,
            name_col=table.header(name_key),
            name=table.frame.loc[i, name_key],
        )
        return self._q("max_row", lang, measure=table.header(measure)), answer

    def lookup(self, table: GeneratedTable, lang: str, as_formula: bool = False):
        name_key = ROLES[table.kind]["name"]
        names = table.frame[name_key]
        unique = [n for n in names if (names == n).sum() == 1]
        if not unique:
            return None
        name = self.rng.choice(unique)
        i = int(names[names == name].index[0])
        measure = self.rng.choice(ROLES[table.kind]["measures"])
        result = table.frame.loc[i, measure]
        formula = f'=XLOOKUP("{name}",{table.range(name_key)},{table.range(measure)})'
        if not _close(evaluate_formula(formula, self._cells(table)), result):
            return None
        fill = {"measure": table.header(measure), "name": name}
        if as_formula:
            explain = self.t["formula_lookup"]["explain"][lang].format(
                name=name, look_range=table.range(name_key), return_range=table.range(measure)
            )
            answer = self.t["formula_answer"][lang].format(
                formula=formula, explain=explain, result=fmt(result)
            )
            return self._q("formula_lookup", lang, **fill), answer
        answer = self.t["lookup"]["a"][lang].format(**fill, row=i + 2, result=fmt(result))
        return self._q("lookup", lang, **fill), answer

    def group_table(self, table: GeneratedTable, lang: str):
        measures = [m for m in ROLES[table.kind]["measures"] if m in SUMMABLE]
        measure = self.rng.choice(measures)
        cat = self.rng.choice(ROLES[table.kind]["categories"])
        grouped = table.frame.groupby(cat, sort=False)[measure].sum().sort_values(ascending=False)
        total_word = {"en": "Total", "km": "សរុប", "zh": "合计"}[lang]
        lines = [f"| {table.header(cat)} | {table.header(measure)} |", "|---|---:|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in grouped.items()]
        lines.append(f"| **{total_word}** | **{fmt(grouped.sum())}** |")
        answer = self.t["group_table"]["a"][lang].format(table="\n".join(lines))
        return self._q("group_table", lang, measure=table.header(measure), cat_col=table.header(cat)), answer

    def top_k(self, table: GeneratedTable, lang: str):
        measures = [m for m in ROLES[table.kind]["measures"] if m in SUMMABLE]
        measure = self.rng.choice(measures)
        k = min(3, len(table.frame))
        ordered = table.frame.sort_values(measure, ascending=False, kind="stable")
        if len(ordered) > k and ordered[measure].iloc[k - 1] == ordered[measure].iloc[k]:
            return None  # tie at the cut-off
        name_key = ROLES[table.kind]["name"]
        items = "\n".join(
            f"{rank}. {row[name_key]} (row {idx + 2}) - {fmt(row[measure])}"
            if lang == "en"
            else f"{rank}. {row[name_key]} ({'ជួរ' if lang == 'km' else '第'}{idx + 2}{'' if lang == 'km' else '行'}) - {fmt(row[measure])}"
            for rank, (idx, row) in enumerate(ordered.head(k).iterrows(), start=1)
        )
        return self._q("top_k", lang, k=k, measure=table.header(measure)), self.t["top_k"]["a"][lang].format(
            items=items
        )

    def formula_pmt(self, table: GeneratedTable, lang: str):
        if table.kind != "loans":
            return None
        i = self.rng.randrange(len(table.frame))
        row = i + 2
        rate_cell, term_cell, amount_cell = (f"{table.letter(k)}{row}" for k in ("rate", "term", "amount"))
        formula = f"=PMT({rate_cell}/100/12,{term_cell},-{amount_cell})"
        loan = table.frame.loc[i]
        expected = BankSim.calculate_loan_payment(
            float(loan["amount"]), float(loan["rate"]), int(loan["term"])
        )["monthly_payment"]
        value = evaluate_formula(formula, self._cells(table))
        if not math.isclose(round(value, 2), expected, abs_tol=0.011):
            return None
        explain = self.t["formula_pmt"]["explain"][lang].format(
            rate_cell=rate_cell, term_cell=term_cell, amount_cell=amount_cell
        )
        answer = self.t["formula_answer"][lang].format(
            formula=formula, explain=explain, result=fmt(round(value, 2))
        )
        return self._q("formula_pmt", lang, loan_id=loan["loan_id"]), answer

    def pct_change(self, table: GeneratedTable, lang: str):
        if table.kind != "deposits":
            return None
        branch = self.rng.choice(sorted(set(table.frame["branch"])))
        months = sorted(table.frame.loc[table.frame["branch"] == branch, "month"])
        if len(months) < 2:
            return None
        m1, m2 = sorted(self.rng.sample(months, 2))
        measure = self.rng.choice(["deposits", "new_accounts"])
        v1 = table.frame.loc[(table.frame["branch"] == branch) & (table.frame["month"] == m1), measure].item()
        v2 = table.frame.loc[(table.frame["branch"] == branch) & (table.frame["month"] == m2), measure].item()
        result = round((v2 - v1) / v1 * 100, 1)
        answer = self.t["pct_change"]["a"][lang].format(
            m1=m1, m2=m2, v1=fmt(v1), v2=fmt(v2), result=f"{result:+.1f}"
        )
        return self._q("pct_change", lang, measure=table.header(measure), branch=branch, m1=m1, m2=m2), answer

    # -- driver ---------------------------------------------------------------------------------
    def _dispatch(self, task: str, table: GeneratedTable, lang: str):
        if task == "formula_sum":
            return self.sum_by(table, lang, as_formula=True)
        if task == "formula_count":
            return self.count_by(table, lang, as_formula=True)
        if task == "formula_lookup":
            return self.lookup(table, lang, as_formula=True)
        return getattr(self, task)(table, lang)

    def sample(self, task: str, lang: str, attempts: int = 8) -> dict | None:
        for _ in range(attempts):
            kind = (
                "loans"
                if task == "formula_pmt"
                else "deposits"
                if task == "pct_change"
                else self.rng.choice(TABLE_KINDS)
            )
            table_lang = "en" if self.rng.random() < 0.25 else lang
            table = generate_table(kind, table_lang, self.rng)
            qa = self._dispatch(task, table, lang)
            if qa is None:
                continue
            question, answer = qa
            style = "grid" if task.startswith("formula") or self.rng.random() < 0.7 else "markdown"
            text = grid_to_text(table.rows, table.sheet, style=style)
            user = self.rng.choice(self.t["intro"][lang]).format(table=text, question=question)
            self._counter += 1
            return sft_record(
                id=f"tables-{task}-{lang}-{self._counter:06d}",
                source="dsm-tables",
                messages=[
                    {"role": "user", "content": normalize_text(user).strip()},
                    {"role": "assistant", "content": normalize_text(answer).strip()},
                ],
                lang=lang if table_lang == lang else "mixed",
                category="tables",
                generated_by="template+pandas",
                meta={"task": task, "table_kind": kind, "table_lang": table_lang, "style": style},
            )
        return None

    def build(self, per_weight: int = 10) -> list[dict]:
        records = []
        for task, weight in TASKS.items():
            for lang in LANGS:
                for _ in range(weight * per_weight):
                    if (record := self.sample(task, lang)) is not None:
                        records.append(record)
        return records


def build(seeds_path: str | Path, seed: int = 0, per_weight: int = 10) -> list[dict]:
    return TableTaskBuilder(seeds_path, seed=seed).build(per_weight=per_weight)
