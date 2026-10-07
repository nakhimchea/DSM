# 09 · Markdown and LaTeX output

Answers are rendered by chat UIs that understand Markdown and KaTeX/MathJax. A broken table, an
unclosed code fence or a stray `$` ruins an answer that is otherwise right, so DSM treats
formatting as a trained, validated and evaluated skill.

## 1. House style

| Element | Rule |
|---|---|
| Structure | Answer first, then details. Headings (`## `) only for long answers, always with a space after `#` |
| Lists | `1.` for steps, `-` for unordered items |
| Tables | header row + separator row; every row has the same number of cells; right-align numbers (`---:`) where it helps |
| Code | fenced code blocks **with a language tag** (`python`, `excel`, `sql`, `bash`) |
| Inline math | `$...$`, no spaces just inside the delimiters |
| Display math | `$$...$$` on its own lines |
| Money | **currency codes, never `$`**: `100 USD`, `4,100 KHR`, `35.5 THB`. A literal dollar sign is `\$`. This keeps `$` unambiguous for math renderers |
| Numbers | thousands separators in prose (`12,000`), raw values inside spreadsheet text |
| Khmer | Khmer punctuation (`។`, `៖`), Arabic or Khmer digits consistently within an answer |
| Chinese | full-width punctuation (`，。：？`), no space between Chinese characters and numbers |
| No | raw HTML, emoji decorations, unclosed formatting |

## 2. The validator (`dsm.format.validate`)

`check_format(text)` returns a list of problems (empty = valid):

* **Markdown**: unclosed code fences; table separator / row cell counts that do not match the
  header; headings without a space.
* **LaTeX**: unbalanced `$` / `$$` (with the money hint), unbalanced braces, mismatched
  `\begin{…}` / `\end{…}`, and a full parse of every math segment with pylatexenc. Code spans and
  blocks are ignored.

Where it is used:

| Place | Effect |
|---|---|
| formatting, tables and math builders | samples that fail are not written |
| teacher outputs (chat regeneration, judged DPO) | failing answers are dropped / ranked last |
| format DPO pairs | chosen must pass, rejected must fail |
| evaluation | format validity rate, gate ≥ 99 % |

`dollars_to_usd(text)` rewrites plain-text money (`$1,200` → `1,200 USD`, `$1.5 million` →
`1.5 million USD`). Use it for sources without LaTeX, such as grade-school word problems. Never
apply it to text that uses `$` for math.

## 3. Data

### Formatting samples (`dsm build formatting`)

From `data/seeds/formatting/facts.yaml`, in km/en/zh:

* comparison tables (savings vs. current accounts, fixed vs. variable rates),
* formulas in LaTeX with a variable legend (compound and simple interest, monthly loan payment,
  percentage change, the quadratic formula),
* numbered procedures (opening a savings account, protecting yourself from phone scams).

### Math (concise, no hidden reasoning)

| Dataset | Licence | Converter | Processing |
|---|---|---|---|
| GSM8K | MIT | `gsm8k` | `<<48/2=24>>` calculator notes removed, money rewritten to `USD`, final line `**Answer: 72**` |
| NuminaMath 1.5 / CoT | Apache-2.0 | `numinamath` | rows flagged invalid and proof problems dropped; solutions must pass the LaTeX validator |
| MATH (competition) | MIT | `competition_math` | same converter; `\boxed{}` answers kept |

Provenance note for the catalogue: parts of the NuminaMath solutions were written or rewritten
by a proprietary model. The dataset licence is Apache-2.0, but record `generated_by` honestly and
let counsel decide (tier "needs counsel" in [12_licensing_provenance.md](12_licensing_provenance.md)).

DSM is fast-only, so math answers show **short working**: the key equations, then the result. They
contain no `<think>` section and no long exploratory reasoning. Teacher prompts ask for exactly
that style (`dsm.data.builders.chat.HOUSE_STYLE`).

### Format preference pairs

```bash
dsm build preference --kind format --input datasets/sft/tables.jsonl.zst --out datasets/dpo/format.jsonl.zst
```

For every valid answer that contains a table, math or a code block, `corrupt_format` makes a
minimally broken copy: separator row removed, a row one cell short, an unclosed `$`/`$$`, or a
missing closing fence. The pair (valid > broken) teaches the model which mistakes matter, without
changing content.

## 4. Mix and evaluation

* SFT: about **10 %** math (LaTeX) and **4 %** formatting. Tables (10 %) and RAG (8 %) also
  exercise Markdown.
* Gate: ≥ 99 % of answers on the evaluation prompts pass `check_format`. This is measured per
  language, after SFT, DPO and quantisation.
