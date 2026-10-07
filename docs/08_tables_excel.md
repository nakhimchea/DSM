# 08 · Tables and Excel

DSM is text-only. Spreadsheets reach it as **text in a fixed, versioned format** that shows Excel
cell addresses. The model can then point to rows, write formulas that work when pasted back into
Excel, and stay grounded in the numbers it was given. Sheets too large for the context go through
a query tool instead of being pasted whole.

## 1. The serialisation contract (`TABLE_FORMAT_VERSION = "1"`)

`dsm.tables.serialize` is the **single implementation**, used to build training data and, at
inference, to turn the user's `.xlsx` / `.csv` into the prompt. An application must call the same
function (`workbook_to_text`) or vendor the file. Otherwise the model sees a format it was never
trained on.

**Grid style** (default; exposes addresses for formulas):

```text
[Sheet "Loans" | A1:E5 | 4 data rows]
|   | A | B | C | D | E |
|---|---|---|---|---|---|
| 1 | Branch | Officer | Amount | Rate | Status |
| 2 | Phnom Penh | Dara | 12000 | 12 | Active |
| 3 | Siem Reap | Lina | 8000 | 10.5 | Closed |
| 4 | Phnom Penh | Sok | 5500 | 9 | Active |
| 5 | Total |  | =SUM(C2:C4) → 25500 |  |  |
```

**Markdown style** (header row only, for small tables in documents). 30 % of non-formula training
samples use it, so the model reads both.

Rules: numbers are raw values (no thousands separators); dates are ISO 8601; booleans are
`TRUE`/`FALSE`; formulas are written as `=FORMULA → cached value`; `|` inside cells is escaped;
long cells are cut at 120 characters; merged ranges are listed after the grid and their value is
repeated in every covered cell; sheets longer than `max_rows` (60) end with an explicit note:
`… N more rows not shown (use the table_query tool for the full sheet)`. `parse_grid` is the
inverse; tests and the evaluation use it to re-check answers against exactly what the model saw.

## 2. Generated table tasks (`dsm build tables`)

`dsm.tables.generate` creates realistic bank and business sheets (loans, deposits, sales, payroll)
with km/en/zh headers and city names. A quarter of the samples ask in Khmer or Chinese about a
sheet with **English headers**, which is common in practice. **pandas computes every answer.**

| Task | Example answer |
|---|---|
| `sum_by`, `avg_by`, `count_by` | names the rows used (`rows 2, 4, 7`), shows the arithmetic, gives the result |
| `max_row`, `lookup`, `top_k` | value + row number + name |
| `group_table` | a Markdown summary table with a bold total |
| `pct_change` | values of both months and the signed percentage |
| `formula_sum` (`SUMIFS`), `formula_count` (`COUNTIFS`), `formula_lookup` (`XLOOKUP`), `formula_pmt` (`PMT`) | the formula in an `excel` code block, a short explanation, and its result on this sheet |

**Every formula is executed** by DSM's own evaluator (`dsm.tables.formula`) on the sheet. The
sample is dropped if the result differs from the pandas ground truth. PMT results are cross-checked
against the bank simulator's loan calculator. A test re-parses the prompt text with `parse_grid`
and evaluates each formula again.

The evaluator supports arithmetic, comparisons, `&`, cell and range references (incl. `$A$1`) and
SUM, AVERAGE, MIN, MAX, COUNT(A), ABS, ROUND/UP/DOWN, IF, AND, OR, NOT, LEN, UPPER, LOWER, LEFT,
RIGHT, CONCAT, SUMIF(S), COUNTIF(S), AVERAGEIFS, MAXIFS, MINIFS, XLOOKUP, VLOOKUP, INDEX, MATCH and
PMT. It is in-house because the available Python evaluators are GPL (pycel) or EUPL (formulas),
which are unsuitable licences for this project.

## 3. Public table datasets (`dsm build convert`)

| Dataset | Licence | Converter | What it adds |
|---|---|---|---|
| TabFact | CC-BY-4.0 | `tabfact` | true / false fact verification over tables |
| WikiTableQuestions | CC-BY-SA-4.0 (**needs counsel**: share-alike) | `wtq` | open questions over Wikipedia tables |
| TAT-QA | CC-BY-4.0 | `tatqa` | financial report table + paragraphs. Arithmetic derivations are re-computed and shown only when they reproduce the gold answer |
| FinQA | MIT | `finqa` | financial reports. The gold **program** (`subtract(5829, 5735), divide(#0, 5735)`) is executed and rendered as numbered steps. Rows whose program does not reproduce the gold answer are dropped |

Tables larger than 60 data rows are skipped, so the answer is always visible to the model. Further
candidates for the catalogue (M7, licence check first): HiTab, TableBench / TableInstruct,
InstructExcel.

## 4. Large sheets and the `table_query` tool

A 0.8B model should not read 5,000 rows. For sheets beyond the row limit, the prompt shows the
first rows plus the truncation note, and the model calls `table_query` (M6) with a filter,
group-by or aggregation. The tool runs on the full sheet with pandas, and its result comes back as
a small table. This keeps answers exact and the context short.

## 5. At inference

* `dsm chat` → `/load report.xlsx` serialises the workbook with `workbook_to_text` and puts it into
  the next user message (M6).
* Other applications: call `dsm.tables.serialize.workbook_to_text(path)`, or vendor
  `serialize.py` and `formula.py`. They have no dependencies besides openpyxl for `.xlsx`.
* Long tables need long context: SFT includes 8,192-token samples for table and RAG data.

## 6. Mix and evaluation

* SFT: about **10 %** tables: generated tasks (exact answers, formulas) + TAT-QA / FinQA (financial
  reasoning) + TabFact / WTQ (reading), with Khmer and Chinese from the generator and from
  translation.
* Evaluation (M5): exact match on held-out generated sheets (numbers compared after normalising
  separators), formula execution accuracy (the model's formula is run by the evaluator), and
  TAT-QA / FinQA accuracy.
