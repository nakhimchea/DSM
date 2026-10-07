# 13 · Runbook: from the upstream weights to a served DSM on one 24 GB GPU

Every step is a script in `scripts/`. You can re-run any of them: finished outputs are kept, and
training resumes from its newest checkpoint. Settings are environment variables (table below).
`tests/e2e/test_runbook.py` runs these same scripts on CPU, with a tiny fake checkpoint, tiny
fake downloads and tiny budgets, so CI exercises every command they contain.

```
00 setup ─▶ 10 fetch ─▶ 20 convert ─┬─▶ 35 SFT/DPO data ─▶ 40 fast path ─▶ 80 export ─▶ 90 serve       (v0.1, 1–2 days)
                                    │
                                    └─▶ 30 pre-training data ─▶ [45 tokenizer pilot] ─▶ 50 CPT ─▶ 55 anneal
                                        ─▶ 60 SFT ─▶ 70 DPO + merge ─▶ 80 export ─▶ 90 serve           (full path, weeks)
```

## 0. What you need

| | |
|---|---|
| GPU | 24 GB, compute capability ≥ 8.0 (RTX 3090 / 4090, A5000, L4, ...) |
| OS | Linux or WSL2: the Gated DeltaNet kernels (flash-linear-attention, Triton) do not run on native Windows or macOS |
| RAM | 64 GB recommended (32 GB minimum) |
| Disk | fast path ~50 GB; full path ~1.5 TB (raw downloads and intermediate files; you can delete `datasets/raw` after cleaning) |
| Software | Python 3.10–3.13, PyTorch with CUDA, git, cmake, a C++ compiler (llama.cpp) |
| Internet | only for step 1. Run it on a connected machine and copy `work/upstream` and `datasets/raw` over |

### Layout (inside `DSM_HOME`, default: the repository)

```
work/upstream/{base,instruct}/   source checkpoints + FETCH.json (exact revisions, file hashes)
work/models/                     dsm-base, dsm-instruct, dsm-base-km-ext, anneal-cv, dsm-final
work/runs/<stage>/               checkpoints/, metrics.jsonl, final/ (a complete model directory)
work/mix/                        data lists written by `dsm data mix`, passed with --data
datasets/raw/<source>/           downloads (one directory per catalogue source)
datasets/{clean,dedup,final}/    documents after each stage, with <file>.stats.json
datasets/packed/                 tokenised packs; eval_{km,en,zh} = held-out documents
datasets/sft/<category>/         conversations (chat, tools, tables, math, khmer, safety, ...)
datasets/dpo/  datasets/eval/    preference pairs; benchmark files (never trained on)
results/<model>/                 metrics.json (suites), bpb.json (bits per byte)
exports/<name>/                  GGUF files + LICENSE + NOTICE + README + SHA256SUMS
```

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `TRAIN_ARGS` | – | extra options for every `dsm train` / `dsm dpo`, e.g. `--set micro_batch_size=2 --set grad_accum=64` |
| `EVAL_ARGS` | – | extra options for `dsm eval suite`, e.g. `--batch-size 16` |
| `DATA_ARGS` | – | licence flags for `dsm data`: `--accept-counsel` after legal sign-off (docs/12) |
| `GATES` | `strict` | a failed stage gate stops the script; `warn` reports it and continues |
| `TEACHER` | `0` | `1`: run the steps that need the local teacher / translator (`configs/teacher.yaml`) |
| `FULL` | `0` | `1`: step 1 downloads everything the full path needs |
| `HELDOUT` | `0.002` | share of every text pool held out for bits-per-byte evaluation |
| `TOKENIZER` | `work/models/dsm-base` | model directory whose tokenizer packs the data |
| `MODEL` | `work/models/dsm-base` | start of CPT (step 5) |
| `CHAT_VECTOR` | `0` | `1`: SFT starts from anneal + (instruct − base) |
| `QUANTS` | `Q8_0` | quantisations to export, e.g. `"Q8_0 Q4_K_M"` |
| `CPT_TOKENS`, `ANNEAL_TOKENS`, `SFT_TOTAL`, `DPO_TOTAL` | from `configs/data/mix_*.yaml` | budgets of the mixtures |
| `KM_HQ`, `EN_HQ`, `ZH_HQ`, `REPLAY` | see step 3 | `dsm data select` options for the annealing subsets |

## 1. The fast path (v0.1): 1–2 days

The fast path proves the rebrand, identity, safety, tools, export and serving early, without
pre-training. It fine-tunes the converted **instruct** checkpoint on DSM's own data plus general
chat replay.

```bash
scripts/00_setup.sh             # install, preflight, CPU tests, build llama.cpp (30 min)
scripts/10_fetch.sh             # connected machine: checkpoints, eval sets, chat replay (1–2 h)
scripts/20_convert.sh           # convert + verify + audit + baseline suites (30 min)
scripts/35_sft_data.sh          # seeds + converted chats (TEACHER=1 adds Khmer versions)
scripts/40_fastpath.sh          # SFT (~1–2 h) -> suites -> gate -> GGUF Q8_0
scripts/90_serve.sh exports/fastpath/DSM1.0-0.8B-Q8_0.gguf
```

Check `results/fastpath/metrics.json`. Identity must reach 98 % in every language before you show
the model to anyone. The fast path keeps the original tokenizer: Khmer is still about 0.9 tokens
per character.

## 2. The full path

| Step | Script | What happens | Time (4090-class GPU) | Gate |
|---|---|---|---|---|
| 0 | `00_setup.sh` | install, `dsm preflight`, tests, llama.cpp | 30 min | – |
| 1 | `FULL=1 10_fetch.sh` | checkpoints, ~1.5–3B Khmer tokens, replay, SFT, preference, eval sets | hours (bandwidth) | licence gate |
| 2 | `20_convert.sh` | convert, verify logits and tokenizer, audit, baselines | 30 min | verify, audit |
| 3 | `30_pretrain_data.sh` | eval sets, clean, dedup, parallel text, decontaminate, pack, annealing subsets, CPT mix | 1–2 days CPU | – |
| 4 | `35_sft_data.sh` | seeds, converted public sets, translations (`TEACHER=1`), preference pairs, validation | hours; translation: days | `dsm build validate` |
| 4b | `45_tokenizer_pilot.sh` | optional Khmer vocabulary pilot ([14](14_khmer_tokenizer.md)) | 2–3 days | decision rule |
| 5 | `50_cpt.sh` | 2B tokens, Khmer-heavy | 3–5 days | `after_cpt` |
| 6 | `55_anneal.sh` | 0.4B tokens of the best data, LR → 0 | ~1 day | `after_anneal` |
| 7 | `60_sft.sh` | ~200K conversations × 2 epochs | 1–2 days | `after_sft` |
| 8 | `70_dpo.sh` | on-policy + DSM pairs, DPO, SFT↔DPO interpolation | ~1 day | `after_dpo` |
| 9 | `80_export.sh` | GGUF BF16 + Q8_0 (+ Q4_K_M), per-language KL, suites on the GGUF, audits | 1–2 h | `after_quant` |
| 10 | `90_serve.sh` | `llama-server` on 127.0.0.1 | – | – |

Expect roughly 5–8K training tokens/s on a 4090-class GPU with the flash-linear-attention
kernels (the plain PyTorch path is ~10× slower); the durations above assume that. Measure yours
before committing days of compute:

```bash
dsm train --config configs/train/cpt_24gb.yaml --bench 20     # tokens/s and the projected duration
```

### Step notes

**2 · Convert.** `dsm verify` must pass for both checkpoints. It compares tokenizer ids on
km/en/zh/code probes and logits against the reference implementation, in fp32
(`VERIFY_ARGS="--device cuda"` to use the GPU). `results/instruct/metrics.json` is the baseline
of the SFT gates: SFT must not lose tool, table or RAG accuracy.

**3 · Pre-training data.** Sources you did not download are skipped. Look at the
`<file>.stats.json` files and the `DATACARD.md` of each pack (`datasets/packed/<name>.card/`). A
stage that drops more than half of a source usually means a wrong language label or a broken
download. The annealing subsets use document statistics from cleaning; tune them with `KM_HQ`,
`EN_HQ` (FineWeb-Edu `int_score`), `ZH_HQ` and `REPLAY`. `$HELDOUT` of every pool goes to
`eval_{km,en,zh}` and never into training. `select` and `pack` use the same per-id split.

**4 · SFT / DPO data.** `TEACHER=1` needs the local teacher and translator servers from
`configs/teacher.yaml` (Apache-2.0 / MIT models only). Without them, Khmer comes from the seeds and
the parallel data. Have native speakers review samples of the Khmer files. For an SFT evaluation
loss, split a file with `dsm data select --sample 0.005` (held out) and the same command with
`--invert` (the rest, for training), and add the held-out file to `data.eval`.

**5 · CPT.** Check `metrics.jsonl`: `eval/km/bpb` should fall steadily and `eval/en/bpb` /
`eval/zh/bpb` should stay within 3 %. The trainer prints epochs per source; Khmer above 4 epochs
means you need more Khmer text, not more repetition. If you adopted the Khmer vocabulary (step 4b),
re-pack with it first (`rm -r datasets/packed && TOKENIZER=work/runs/pilot_B1/final
scripts/30_pretrain_data.sh`), then run `MODEL=work/runs/pilot_B1/final scripts/50_cpt.sh`, and
keep `TOKENIZER=work/runs/pilot_B1/final` set for every later step.
The base-model baseline in `results/base/bpb.json` was measured before re-packing and stays valid,
because bits per byte do not depend on the tokenizer.

**7 · SFT.** Run once with `CHAT_VECTOR=1` and once without (`RUN=work/runs/sft_cv`), and keep the
better one on the suites. Long table / RAG conversations: a second short run with
`TRAIN_ARGS="--set seq_len=8192 --set micro_batch_size=2"`.

**8 · DPO.** With `TEACHER=1`, the SFT model is exported to Q8_0 and served on port 8090 (the
`policy` entry of `configs/teacher.yaml`). 4 answers per prompt are sampled and the local teacher
judges them. Watch `reward_accuracy` (should rise), `reward_margin`, and `chosen_logp`, which must
not keep falling. The final model is `work/models/dsm-final`, an interpolation of SFT and DPO
(`ALPHA=0.7`).

**9 · Export.** `QUANTS="Q8_0 Q4_K_M"` adds 4-bit with an importance matrix computed from km/en/zh
text. Ship Q4_K_M only if its per-language KL stays below 0.05 and identity still passes on the
GGUF. Each export directory holds LICENSE, NOTICE, README and SHA256SUMS. Copy it as a whole.

## 3. Gates and what to do when one fails

| Gate (`configs/eval/gates.yaml`) | Fails when | Usual fix |
|---|---|---|
| `after_cpt` | Khmer bpb did not fall, or en/zh rose > 3 % | more replay (raise en/zh shares), lower LR, check the Khmer packs |
| `after_anneal` | bpb got worse than after CPT | anneal fewer tokens, more replay in `replay_mix` |
| `after_sft` identity | < 98 % in some language | more identity copies (`repeat`), check the identity seeds in that language |
| `after_sft` safety | refusal < 90 % or over-refusal > 10 % | balance harmful vs benign-but-sensitive safety data (docs/06) |
| `after_sft` tools / tables / RAG | below the instruct baseline | more of that category in `configs/data/mix_sft.yaml`; check the converted data |
| `after_dpo` | identity or safety regressed | lower `ALPHA`, fewer epochs or a lower LR, more identity / safety pairs |
| `after_quant` | KL > 0.05 or identity < 98 % on the GGUF | ship Q8_0 instead of 4-bit; better calibration text |

The safety gates need held-out safety prompts. Step 3 converts XSTest into
`datasets/eval/safety_xstest.jsonl` (never trained on), and the scripts pass every
`datasets/eval/safety_*.jsonl` to the suites. Add Khmer and Chinese files the same way
([data/eval/safety/README.md](../data/eval/safety/README.md)).

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| CUDA out of memory | `TRAIN_ARGS="--set micro_batch_size=2 --set grad_accum=64"` (same tokens per step); for CPT also `--set seq_len=2048` |
| Very slow training, preflight warns about kernels | flash-linear-attention missing or no Triton support (native Windows): install the `kernels` extra on Linux / WSL2 |
| Loss spikes or NaN | lower the LR; inspect the data the step saw; re-run the script to resume from the last good checkpoint |
| `... was tokenised with tokenizer ..., but the model uses ...` | a pack made with another tokenizer (pilot arms): re-pack with `TOKENIZER=<model>` |
| `licence gate: ...` | the source needs counsel sign-off (`DATA_ARGS=--accept-counsel`) or is blocked; see docs/12 |
| Khmer looks broken in outputs | the application must call `dsm.text.normalize_text` on input; legacy-font (Limon / ABC) text must not reach training (`dsm data clean` drops it) |
| Garbled answers from `llama-server` with several users on CPU | llama.cpp repacking issue (docs/11); `dsm export serve` disables repacking for `--parallel > 1` |
| A gate stops a script you only want to inspect | `GATES=warn` |

## 5. Your own documents

The most useful data is your organisation's: product sheets, fee schedules, FAQs, policies.

```bash
dsm data clean --name bank_docs --license internal --internal --langs km en zh \
    --input exports_from_your_systems/ --out datasets/clean/bank_docs.jsonl.zst   # strict PII redaction
dsm build rag --documents datasets/clean/bank_docs.jsonl.zst \
    --out datasets/sft/rag/bank_docs.jsonl.zst          # local teacher writes grounded QA; citations checked
```

Facts belong in retrieval (RAG) rather than in the weights: they change, and grounded answers can
cite their source. A model trained on internal data is marked INTERNAL in its data cards and must
never be published.
