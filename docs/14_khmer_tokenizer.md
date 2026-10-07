# 14 · Khmer tokens: measuring and extending the vocabulary

A model generates a roughly constant number of **tokens** per second. Users see **characters** per
second, so a language that needs more tokens per character is slower to read, costs more of the
context window, and has more steps where the model can go wrong.

The original 248K vocabulary was built for English, Chinese and code. Only **79 entries** contain
a Khmer character, so Khmer text costs about **0.9 tokens per character**, against about 0.2 for
English. The same answer takes about 4–5× more decoding steps in Khmer than in English. Adding
Khmer tokens is the largest speed lever for Khmer users, and it is optional: the rest of the
pipeline works with either vocabulary.

```
dsm tokenizer analyze   measure tokens / character per language (any number of tokenizers)
dsm tokenizer extend    learn N Khmer merges on top of the existing ones  -> work/tok/km-ext
dsm tokenizer remap     give the model rows for the new tokens             -> work/models/dsm-base-km-ext
pilot                   train the new rows, then CPT; compare with the original vocabulary
```

## 1. Measure

```bash
dsm tokenizer analyze --tokenizer original=work/models/dsm-base \
    --text km=datasets/final/eval_km.jsonl.zst en=datasets/final/eval_en.jsonl.zst zh=datasets/final/eval_zh.jsonl.zst
```

The report gives tokens per character, bytes per token and tokens per 1,000 characters for every
tokenizer × language, and the number of vocabulary entries containing Khmer. With several
tokenizers it also prints the speed ratio (`km: extended needs 0.45x the tokens of original ->
about 2.22x the characters per second`). Texts are normalised with `dsm.text.normalize_text` first,
exactly as in training. Use held-out text, never the text the extension was learned from.

## 2. Extend (`dsm tokenizer extend`)

```bash
dsm tokenizer extend --tokenizer work/models/dsm-base \
    --input datasets/final/fineweb2_km.jsonl.zst datasets/final/hplt3_km.jsonl.zst \
    --new-tokens 8192 --out work/tok/km-ext
```

How it works (`src/dsm/tokenizer/extend.py`):

1. Khmer documents (the output of `dsm data clean`, `--max-chars` 50M by default) are normalised
   and pre-tokenised exactly as the tokenizer does it.
2. Every Khmer pre-token is segmented with the **existing** merges.
3. BPE continues on those segments. The most frequent adjacent pair is merged, but only into a
   **new** token whose bytes are a run of Khmer UTF-8 (U+1780–U+17FF, possibly cut mid-character
   at either end, after at most one space) containing the start of a Khmer character.
4. New tokens get consecutive ids right after the last existing id: **248,070**, just after the
   control tokens. The first 250 reuse the embedding matrix's unused padding rows. The new merges
   are appended after all existing ones.

What this guarantees, and the tests that check it (`tests/tokenizer/test_extend*.py`, on a tiny
vocabulary and on the real one):

| Property | Why it holds |
|---|---|
| No existing id moves, control tokens included | ids are appended; control tokens are also written into the BPE vocabulary at their ids, because the `tokenizers` loader would otherwise renumber them when the vocabulary grows; the result is re-loaded and checked |
| Text without Khmer encodes to exactly the same ids | merges apply by rank, so the old ones run first, unchanged; a new merge contains the start of a Khmer character, so it cannot fire anywhere else (tested on English, Chinese, code, LaTeX, Vietnamese, Myanmar, Georgian, Mongolian, Baybayin, Greek, Thai, Lao, emoji) |
| Khmer gets shorter and still decodes to the same text | measured on held-out text |
| `tokenizers` and llama.cpp give the learner's segmentation | every new merge creates a token no old merge uses, so applying the merges by rank equals the learner's replay (`encode_with_extension`); a llama.cpp test converts an extended model to GGUF and compares `/tokenize` with `tokenizers` |

Outputs: `tokenizer.json` (extended), the copied `tokenizer_config.json` and
`chat_template.jinja`, `extension.json` (start id, counts, Khmer tokens before / after on the
learning sample) and `new_tokens.tsv` (id, frequency, text).

**Have a native speaker skim `new_tokens.tsv`.** Fragments that start or end inside a character
are normal for byte-level BPE. Tokens that glue unrelated words across a phrase boundary, or come
from boilerplate (menus, cookie banners), point to dirty input text.

**How many tokens?** 8K–16K. Each 8,192 tokens add 8.4M embedding parameters (+1.1 %) and about
3 % to the LM-head cost per generated token. For Khmer that is repaid many times over. Beyond
about 16K, the extra tokens are rare and their rows train poorly. In our tests, learning 8,192
merges from 5M Khmer characters took about 30 s on a 4-core CPU; real text has more distinct
words and takes longer.

## 3. Give the model the rows (`dsm tokenizer remap`)

```bash
dsm tokenizer remap --model work/models/dsm-base --tokenizer work/tok/km-ext --out work/models/dsm-base-km-ext
```

* Each new row starts as the **mean of the rows of the pieces the original tokenizer splits the
  token into**. This applies to the tied input/output embedding, and to the output projection too
  if it is untied.
* Existing rows are copied bit for bit. On text made of old ids, the extended model gives exactly
  the old hidden states and logits; a test checks this.
* The row count is padded to a multiple of 64 with zero rows, which the tokenizer never produces.
* The model directory gets the extended tokenizer, `extension.json`, `new_tokens.tsv` and a
  `tokenizer_extension` entry in `lineage.json`.

## 4. The pilot: is the extension worth it for your data?

| Arm | Steps | Config |
|---|---|---|
| A (original vocabulary) | CPT for 150M tokens | `cpt_24gb.yaml` with `max_tokens=1.5e8` |
| B (extended) | 1. only the new rows for 50M tokens, LR 1e-3 | `pilot_tok_24gb.yaml` (`trainable: new_rows`) |
| | 2. CPT for 100M tokens | `cpt_24gb.yaml` on B1's output, `max_tokens=1.0e8` |

* **Re-pack every dataset with the extended tokenizer** (`dsm data pack --tokenizer
  work/models/dsm-base-km-ext ...`, suffix `_ext`). Packed data records the hash of its
  `tokenizer.json`, and the trainer refuses packs made with a different tokenizer.
* Phase 1 trains only rows ≥ the start id in `extension.json`. All other weights keep zero
  gradient and no weight decay, and stay bit-identical. Phase 1 lets the new tokens catch up
  before the rest of the network adapts to them.
* **Compare bits per byte** (`eval/km/bpb`, `eval/en/bpb`, `eval/zh/bpb` in `metrics.jsonl`),
  never loss per token: the arms have different tokens. Compare at equal training bytes.
* **Decision rule.** Adopt B when its Khmer bpb at the end is no worse than A's (within ~1 %) and
  English / Chinese bpb are not worse by more than 1 %. Its Khmer speed gain is then free.
  Otherwise keep the original vocabulary. Nothing else in the pipeline changes.

## 5. After the decision

* **Full CPT, anneal, SFT, DPO** run on the extended model directory exactly as before. SFT and
  DPO read the tokenizer from the model directory, and their token caches are keyed by its hash.
* **Chat vector.** `dsm merge chat-vector` adds `Instruct − Base` only to the rows that are the
  same token in all three models (below the start id in `extension.json`). The new Khmer rows keep
  their trained values. The upstream Instruct model never saw them.
* **Export.** The extended vocabulary changes llama.cpp's tokenizer fingerprint but not the
  pre-tokeniser. `dsm export gguf` recognises the extension (`extension.json`) and writes the
  standard pre-tokeniser id. llama.cpp's own BPE then reproduces the `tokenizers` segmentation; a
  test checks this on Khmer and other scripts.
* **Inference.** Applications still call `dsm.text.normalize_text` on user input. llama.cpp does
  not apply the Khmer cluster normalisation that the new tokens were learned on.
* **The fast path** (Instruct + short SFT) keeps the original vocabulary. Its embeddings for new
  tokens would be untrained.

## 6. Not done here (phase 2)

* **Vocabulary trim.** Removing scripts DSM never needs (about 48K tokens) would save about 6 %
  of decode compute, but it permutes token ids. Every packed dataset, cache and exported file
  would have to be rebuilt.
* **Re-learning the whole vocabulary.** A fresh 3-language BPE would compress better, but it
  discards the pre-trained embeddings and needs pre-training-scale data.
