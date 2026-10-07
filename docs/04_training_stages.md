# 04 · Training stages: PT, CPT, anneal, SFT, DPO, merging

All stages use one trainer (`dsm train`, `dsm dpo`) that fits a 0.8B model on **one 24 GB GPU**.
This page explains what each stage is for, which data type it consumes, and how to run and
monitor it.

```
 upstream Base ──convert──▶ dsm-base ──CPT (2B tok)──▶ anneal (0.4B tok) ──┐
                                                                            ├─▶ SFT (200K conv.) ──▶ DPO (30K pairs) ──▶ merge/select ──▶ export
 upstream Instruct ─convert─▶ dsm-instruct ── chat vector (Instruct − Base) ─┘
                                 └── v0.1 fast path: SFT (30K conv.) ──▶ export
```

## 1. The stages at a glance

| Stage | Purpose | Data type | Loss on | Typical size | LR |
|---|---|---|---|---|---|
| **PT** pre-training | learn language from nothing | packed plain text | every token | ~1T tokens for 0.8B (impractical here), demonstrated on DSM-nano with 1B | 1e-3 (nano) |
| **CPT** continued pre-training | add / strengthen a language (Khmer) and domain knowledge without forgetting | packed plain text, Khmer-heavy mixture with en/zh replay | every token | 2B tokens | 1e-4 constant |
| **Anneal** | decay the LR on the best data; consolidates knowledge and identity | packed high-quality text + 10–15 % instruction-formatted text + 20 % replay | every token | 0.3–0.5B tokens | 1e-4 → 0 |
| **SFT** supervised fine-tuning | chat behaviour, tools, tables, formats, safety, identity | conversations (JSONL) | assistant tokens only | 200K conversations × 2 epochs | 2e-5 cosine |
| **DPO** preference optimisation | prefer good answers to plausible bad ones (identity, safety, language, format, helpfulness) | (prompt, chosen, rejected) JSONL | log-ratio of chosen vs rejected | 20–40K pairs × 1 epoch | 5e-7 – 1e-6 |
| **Merge** | trade-offs without retraining (SFT↔DPO), transplant instruction tuning | checkpoints | — | — | — |

### Data formats

* **Packed text** (PT/CPT/anneal): `dsm.data.packed`. `<prefix>.bin` holds uint32 token ids, with
  every document ending in `<|endoftext|>`. `<prefix>.idx` holds the document offsets and
  `<prefix>.json` the metadata (name, licence, language, tokenizer hash). Produced by the data
  pipeline ([03_data_preparation.md](03_data_preparation.md)). Windows of `seq_len` tokens are cut
  from the stream. Positions restart at every document start, so documents never attend to each
  other, the Gated DeltaNet state is reset, and the first token of a document is not predicted from
  the previous one.
* **Conversations** (SFT): the record format in `dsm.data.sft.schema` (system, tools, user /
  assistant / tool messages, structured tool calls). They are rendered with the model's own
  `chat_template.jinja`. Labels cover the assistant turns (`train_on: all`) or only the last one
  (`train_on: last`). The empty `<think>` prefill and every non-assistant token are masked.
  Conversations are packed greedily into `seq_len` rows, isolated from each other in the same way.
* **Preference pairs** (DPO): `prompt` messages + `chosen` / `rejected` assistant messages.

## 2. How the trainer fits in 24 GB

| Item | Memory |
|---|---|
| fp32 master weights (752M) | 3.0 GB |
| fp32 gradients + AdamW moments | 9.0 GB |
| activations, bf16, gradient checkpointing per layer, 4 × 4096 tokens | ~2–3 GB |
| logits: chunked cross-entropy, 1,024 tokens × 248K vocab at a time, recomputed in backward | ~3 GB peak |
| CUDA context, fragmentation | ~1–2 GB |

`dsm preflight --config <cfg>` prints this estimate for any config. Other details:

* **bf16 autocast, fp32 master weights.** `A_log` and the GDN norm stay fp32 (they are pinned in
  the model class).
* **Gradient accumulation is exact.** Each micro-batch's loss is a sum over its target tokens
  divided by the target-token count of the *whole* optimizer step. Accumulating uneven
  micro-batches therefore gives the gradient of one big batch; a test proves it.
* **Kernels.** On CUDA, the chunked Gated DeltaNet kernel of flash-linear-attention is used; the
  exact PyTorch path is used on CPU and is about 10× slower on GPU. Packed documents are isolated
  the same way on both paths: the recurrent state's log-decay is set to −30 at every document start
  (an effective reset that keeps gradients finite), the short causal convolution masks taps that
  would cross a document boundary, and attention uses a block-diagonal causal mask. `dsm preflight`
  checks that fla is installed and agrees numerically with the torch path.
* **Resumable, deterministic.** Batches and the LR are pure functions of the step. A restarted
  run (`resume: auto`, the default) continues from the newest complete checkpoint with exactly
  the same batches; a test checks this bit for bit. Checkpoints are written atomically and only
  the last `keep_checkpoints` are kept.
* **Outputs.** `<output_dir>/final/` is a complete model directory: weights, config, tokenizer,
  chat template, remote code, LICENSE, NOTICE, and `lineage.json`, the list of stages with data
  names, steps, tokens and epochs per source. `metrics.jsonl` holds loss, LR, gradient norm,
  tokens/s, peak memory and eval loss / bits-per-byte; TensorBoard is optional.

## 3. Running the stages

```bash
dsm preflight --config configs/train/cpt_24gb.yaml          # GPU, kernels, disk, memory estimate
dsm train --config configs/train/cpt_24gb.yaml --bench 20   # measured tokens/s -> duration estimate
dsm train --config configs/train/cpt_24gb.yaml              # resumes automatically if interrupted
dsm train --config configs/train/anneal_24gb.yaml
dsm merge chat-vector --base work/models/dsm-base --instruct work/models/dsm-instruct \
          --target work/runs/anneal/final --out work/models/anneal-cv       # optional SFT start point
dsm train --config configs/train/sft_24gb.yaml --set model=work/models/anneal-cv
dsm dpo   --config configs/train/dpo_24gb.yaml
dsm merge interpolate --a work/runs/sft/final --b work/runs/dpo/final --alpha 0.7 --out work/models/dsm-candidate
```

Any value can be overridden: `--set optimizer.lr=1.5e-5 --set epochs=1`. Numbers such as `1e-4`
are accepted, even though plain YAML would read them as strings. `--data work/mix/sft.yaml`
replaces the training list with the one `dsm data mix --out` computed (weights for pre-training,
repeat factors for SFT / DPO). SFT and DPO paths may be globs (`datasets/sft/chat/*.jsonl.zst`).
The runbook scripts ([13_runbook_24gb.md](13_runbook_24gb.md)) chain all of this.

### Configs (`configs/train/*_24gb.yaml`)

| Config | Start | Data | Budget | Batch (tokens) | LR / schedule |
|---|---|---|---|---|---|
| `fastpath_sft` | converted Instruct | DSM seed sets + general replay | 2 epochs (~30K conv.) | 64K | 1e-5 cosine |
| `pilot_tok` | Khmer-extended Base | Khmer text, **new embedding rows only** | 50M tokens | 524K | 1e-3 cosine |
| `cpt` | converted Base | km 45 / en 20 / zh 20 / parallel 7 / math 4 / code 3 / tables 1 % | 2B tokens | 524K | 1e-4 constant (WSD stable phase) |
| `anneal` | CPT output | high-quality + instruction text + replay | 0.4B tokens | 524K | 1e-4 → 0 linear |
| `sft` | anneal output (or its chat-vector merge) | 11 conversation sets | 2 epochs (~200K conv.) | 128K | 2e-5 cosine |
| `dpo` | SFT output | 7 pair sets | 1 epoch | 32 pairs | 8e-7 cosine, β 0.1 |
| `pt_nano` | random init (DSM-nano) | km/en/zh text | 1B tokens | 524K | 1e-3 WSD |

Time on a 4090-class GPU with fla: about 5–8K tokens/s for the 0.8B model, so CPT on 2B tokens takes
3–5 days, the anneal about 1 day, SFT about 1 day and DPO a few hours. Always measure with
`--bench` first.

## 4. Stage notes

**CPT.** Khmer is about 45 % of the mixture, but high-quality Khmer text is scarce (about 1B good
tokens). `lineage.json` and the trainer report **epochs per source**: keep Khmer at 4 epochs or
fewer, and add translated or parallel data rather than repeating more. English and Chinese replay
(about 40 %) is what prevents forgetting. The identity replay documents ride along at about
0.01 %. Watch per-language bits-per-byte (`eval/km/bpb` …). Khmer should fall steadily, and en/zh
should not rise more than about 3 % (gate).

**Anneal.** This is where instruction-formatted text and the best documents go. The model
remembers what it sees at low LR best. Start it from the CPT output. The optimizer state is
rebuilt during a short warm-up.

**Tokenizer pilot** ([14_khmer_tokenizer.md](14_khmer_tokenizer.md)). After extending the vocabulary, the new rows start as the mean of their
sub-token embeddings. `trainable: new_rows` updates only rows ≥ `new_rows_from`: the other rows
receive zero gradient and no weight decay, and a test checks that they stay bit-identical. Then
continue with normal CPT. Compare arms with **bits per byte**, never with loss per token.

**SFT.** Use `repeat` to set the mix (identity × 8, formatting × 8, large sets × 1). Packing
efficiency and the number of conversations dropped for length are printed at start-up. Raise
`seq_len` to 8192 (with `micro_batch_size: 2`) for long table / RAG conversations instead of
truncating them. The chat-vector merge (`Instruct − Base` added to the annealed model) often gives
a better SFT start point: A/B it on the evaluation suite.

**DPO.** Reference log-probabilities are computed once, before the first update, with the initial
weights (that *is* the reference model), and cached. No second model sits in memory. Monitor
`reward_accuracy` and `reward_margin`, and stop if `chosen_logp` keeps falling (over-optimisation).
`sft_weight > 0` adds an NLL term on chosen answers that counters it. Keep the LR below about 1e-6
for a model this small.

**Merging.** `interpolate` between SFT and DPO (alpha 0.5–0.8) often keeps most of the preference
gains with less drift. Re-run the identity and safety gates after every merge.

## 5. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| CUDA OOM | lower `micro_batch_size` and raise `grad_accum` (same tokens per step); lower `loss_chunk_size`; keep `gradient_checkpointing: true` |
| very slow steps (< 1K tok/s) | fla not installed / not used: `dsm preflight` |
| loss spikes | lower LR, check `grad_norm` in `metrics.jsonl`, look for corrupt data (decoded window from `dsm.data.packed`) |
| en/zh bpb rising in CPT | more replay, lower LR, fewer Khmer epochs |
| model forgets identity after DPO / merge | add identity pairs / repair replay ([05_identity.md](05_identity.md)) |
| "different data / batch configuration" warning on resume | the config changed since the checkpoint: batches will differ from an uninterrupted run |
