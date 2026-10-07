# 03 · Preparing training data

```
 download ──▶ clean ──▶ dedup ──▶ decontam ──▶ pack ──▶ mix ──▶ dsm train (PT / CPT / anneal)
 (raw shards)  normalise   exact +    vs. every    tokenise,   weights from
               lang-ID     MinHash    eval set     uint32 .bin  shares, tokens,
               quality     near-dup               + data card   epoch caps
               PII

 dsm build ... / convert / translate / regenerate ──▶ validate ──▶ mix (repeats) ──▶ dsm train (SFT) / dsm dpo
```

Every stage streams JSONL(.zst) shards, so memory stays flat. Each stage writes
`<output>.stats.json`, which records the count for every drop reason, PII replacements and the
sources involved. `pack` turns the chain into a `DATACARD.md`.

## 1. Record formats

| Record | Fields | Produced by |
|---|---|---|
| raw row | whatever the source has (`text` by default) | `dsm data download`, your exports |
| document | `id, text, lang, source, license, generated_by, meta` | `dsm data clean` (and identity replay documents) |
| packed dataset | `<prefix>.bin` (uint32 tokens, every document ends with `<\|endoftext\|>`), `<prefix>.idx` (document offsets), `<prefix>.json` (metadata) | `dsm data pack` |
| SFT conversation | `id, source, license, generated_by, lang, category, system, tools?, messages, meta` | `dsm build ...` |
| preference pair | the same header + `prompt, chosen, rejected` | `dsm build ...`, `dsm build preference` |

`dsm build validate FILE...` checks SFT, preference and document files: schema, roles, tool calls
against their JSON schemas, control tokens in text, normalisation, and identity leaks.

## 2. Stages

### Download (`dsm data download`, on a connected machine)

```bash
HF_HUB_OFFLINE=0 dsm data download fineweb2_km hplt3_km --out datasets/raw --limit 3000000
```

The licence gate runs first, and refused sources are never fetched. Rows are streamed into
`shard-NNNNN.jsonl.zst` files of 200K rows, and `SOURCE.json` records licence, tier and origin.
Excluded subsets (for example Llama-generated parts of an SFT mix) are skipped while streaming.

### Clean (`dsm data clean`)

```bash
dsm data clean --source fineweb2_km --input datasets/raw/fineweb2_km --out datasets/clean/fineweb2_km.jsonl.zst
```

1. **Normalise** (`dsm.text.normalize_text`). Applies NFC, removes zero-width space, word joiner,
   BOM and the deprecated Khmer vowels U+17B4/5, then reorders Khmer clusters with SIL khnormal.
   Control characters are removed, runs of spaces collapse, and 3+ blank lines become 2. The same
   function must run on user input at inference.
2. **Legacy Khmer fonts.** Documents dominated by Latin-1 symbols (Limon / ABC typing) are
   dropped. They are not Unicode Khmer.
3. **Language ID** (`dsm.data.langid`). Khmer and Chinese are identified by script share (≥ 50 %).
   Traditional Chinese is labelled `zh-Hant`. Latin text counts as English only with enough
   English function words; GlotLID can be plugged in. Documents outside the source's languages are
   dropped.
4. **Quality** (`dsm.data.quality`):

   | Rule | Applies to | Drops when |
   |---|---|---|
   | length | all | < 200 or > 500K characters |
   | duplicate lines / paragraphs | all | > 30 % of characters repeat |
   | bullets, truncated lines, boilerplate (cookies, "all rights reserved", code braces) | all | > 90 % / 30 % / 30 % of lines |
   | encoding errors | all | > 0.1 % U+FFFD |
   | Khmer share | km | < 50 % of letters |
   | broken Khmer clusters (SIL `khtest`) | km | > 15 per 1,000 Khmer characters |
   | Han share, repeated characters | zh | < 40 %; one character > 20 % |
   | word count, mean word length, alphabetic words, stop words | en | Gopher thresholds |

5. **PII redaction** (`dsm.data.pii`). E-mail addresses, phone numbers (international or Cambodian
   local format), Luhn-valid card numbers, IBANs and IPv4 addresses are replaced by `<EMAIL>`,
   `<PHONE>`, `<CARD>`, `<IBAN>`, `<IP>`. Internal sources (`--internal`) also get `<ACCOUNT>`
   and `<NUMBER>` for any run of 6+ digits.

### Deduplicate (`dsm data dedup`)

```bash
dsm data dedup --input datasets/clean/fineweb2_km.jsonl.zst datasets/clean/hplt3_km.jsonl.zst --out-dir datasets/dedup
```

* **Exact**: hash of case- and whitespace-normalised text.
* **Near**: MinHash (128 permutations) over shingles, then LSH with 16 bands × 8 rows. Candidates
  count as duplicates when the estimated Jaccard similarity is ≥ 0.8. Shingles are character
  5-grams for Khmer and Chinese, which have no reliable word boundaries, and word 5-grams for
  English.
* Deduplication runs separately per language. **Earlier inputs win**, so list the best source
  first. The cost is about 1 KB of memory per document. Very large English / Chinese pools are
  already deduplicated upstream; sample them instead.

### Decontaminate (`dsm data decontam`)

```bash
dsm data decontam --input datasets/dedup/*.jsonl.zst --eval data/eval benchmarks/ --out-dir datasets/final
```

Every string of ≥ 30 characters in the evaluation files is indexed as hashed n-grams: 13 words for
English, 25 characters for Khmer / Chinese. This covers DSM's held-out suites and the benchmark
files you evaluate on (Belebele, FLORES, ...). Any training document sharing an n-gram is dropped.
Run this after dedup and before packing.

### Pack (`dsm data pack`)

```bash
dsm data pack --input datasets/final/fineweb2_km.jsonl.zst --tokenizer work/models/dsm-base --out datasets/packed/km_web
```

Documents are tokenised with the model's own `tokenizer.json` and concatenated, each ending with
`<|endoftext|>`. The metadata records the tokenizer hash, so a packed set cannot silently be used
with a different (for example Khmer-extended) tokenizer. The trainer cuts fixed windows from the
stream and restarts positions at each document start, which keeps documents isolated. The data
card lands in `datasets/packed/km_web.card/`.

For annealing, SFT conversations can be added as plain text:
`dsm data render-sft --input datasets/sft/*.jsonl.zst --tokenizer work/models/dsm-base --out datasets/clean/instructions_text.jsonl.zst`,
then `pack` the result.

### Mix (`dsm data mix`)

```bash
dsm data mix --config configs/data/mix_cpt.yaml      # token shares -> weights for cpt_24gb.yaml
dsm data mix --config configs/data/mix_sft.yaml      # conversation shares -> repeat factors for sft_24gb.yaml
```

For pre-training, `plan_mixture` turns target shares, available tokens and per-source epoch caps
into sampling weights. Shares that would exceed a cap (usually Khmer) are capped, and the
difference is redistributed proportionally. For SFT and DPO, repeat factors make each file reach
its share of conversations. The printed YAML snippet goes into the training config.

## 3. SFT and preference data

| Step | Command | Notes |
|---|---|---|
| DSM's own data | `dsm build seeds --out datasets/sft/seeds` | identity, tools (bank simulator), tables (formulas verified), formatting, RAG, safety; km/en/zh |
| public datasets | `dsm build convert --dataset xlam --input raw/xlam --out ...` | dialects normalised to one schema, validated per record |
| Khmer / Chinese versions | `dsm build translate --target km --input ... --out ...` | local translator; code, math, URLs and placeholders protected; numbers and tables checked |
| better answers | `dsm build regenerate --input ... --out ...` | local teacher rewrites assistant turns; identity, language, refusal and format filters |
| preference pairs | `dsm build preference --kind format\|language\|over_refusal\|judged` | judged = on-policy samples from the SFT model + a local judge |
| check | `dsm build validate datasets/sft/*.jsonl.zst` | non-zero exit on any problem |

## 4. Data cards

`DATACARD.md` and `datacard.json` list the sources (licence, tier, `generated_by`) and the counts
of every stage: documents in and kept, drops per quality rule, duplicates, contaminated
documents, PII replacements, tokens. Keep them with the model. They are the provenance record for
[12_licensing_provenance.md](12_licensing_provenance.md) and the release checklist.
