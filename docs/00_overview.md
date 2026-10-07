# 00 · Overview

DSM1.0-0.8B is a compact language model for **Khmer, English and Chinese** that runs
**entirely on your own hardware**. It starts from the weights of an Apache-2.0
open-weight model with a hybrid *Gated DeltaNet + gated attention* architecture. This
repository re-implements, rebrands, adapts and compresses that model.

It is also a worked example of the whole lifecycle of a small LLM:

```
             ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 upstream ──▶│ 1 dissect &  │──▶│ 2 data       │──▶│ 3 train      │──▶│ 4 evaluate   │──▶ 5 compress ──▶ 6 serve
 weights     │   convert    │   │   collect,   │   │   CPT → SFT  │   │   gates per  │      & export     offline
             │  (DSM code)  │   │   clean, mix │   │   → DPO      │   │   stage      │      (GGUF)      (llama.cpp)
             └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

## Guiding decisions

| Decision | Choice | Why |
|---|---|---|
| Hardware | 1× 24 GB GPU (RTX 3090/4090-class) | The budgets and configs in `configs/train/*_24gb.yaml` assume it |
| Modality | Text only | Smaller and faster; spreadsheets are handled as text plus tools |
| Reasoning | **Fast only** (no `<think>` mode) | Quick responses; math and tables use short inline working |
| Identity | "DSM1.0, developed by Doslarb" | Built on an open-source foundation model; the model never claims to be trained from scratch |
| Cloud | None | Offline flags everywhere, local teacher models, `dsm audit offline` |

## Where things live

| You want to… | Look at |
|---|---|
| Understand the architecture and what can be changed | [01_architecture_dissection.md](01_architecture_dissection.md), `src/dsm/modeling/` |
| Know which datasets to collect (and which licences are safe) | [02_data_catalogue.md](02_data_catalogue.md), `configs/data/sources.yaml` |
| Clean and prepare Khmer / English / Chinese data | [03_data_preparation.md](03_data_preparation.md), `src/dsm/data/` |
| Understand PT vs CPT vs SFT vs DPO and their data types | [04_training_stages.md](04_training_stages.md), `src/dsm/train/` |
| Teach the model who it is (identity and identity replay) | [05_identity.md](05_identity.md), `configs/identity.yaml` |
| Add safeguards | [06_safety.md](06_safety.md), `configs/safety_policy.yaml` |
| Train tool / function calling | [07_tool_calling.md](07_tool_calling.md) |
| Teach Excel and table understanding | [08_tables_excel.md](08_tables_excel.md) |
| Get good Markdown and LaTeX output | [09_markdown_latex.md](09_markdown_latex.md) |
| Measure quality and set stage gates | [10_evaluation.md](10_evaluation.md) |
| Make it fast locally (tokenizer, quantisation, llama.cpp) | [11_compression_local_inference.md](11_compression_local_inference.md) |
| Licensing, provenance and "no upstream names" | [12_licensing_provenance.md](12_licensing_provenance.md) |
| Run everything on your GPU, step by step | [13_runbook_24gb.md](13_runbook_24gb.md) |
| Make Khmer cheaper: measure and extend the vocabulary | [14_khmer_tokenizer.md](14_khmer_tokenizer.md), `src/dsm/tokenizer/` |

## The two paths in the runbook

1. **v0.1 fast path (1–2 days).** Convert the upstream *instruct* checkpoint, then run a short SFT
   on identity, safety, tool-call, table and formatting data. Export to GGUF Q8_0 and serve with
   `llama-server`. This proves the rebrand, identity, export and serving work early.
2. **Full path (weeks).** Start from the *base* checkpoint and run a tokenizer pilot (optional extra
   Khmer tokens). Then continue pre-training on ~2B tokens (Khmer-heavy, with English/Chinese
   replay), anneal, SFT (~200K conversations) and DPO. Finish with evaluation gates, quantisation
   and serving.

## Design rules used throughout the code

* **Offline by default.** Every `dsm` command sets `HF_HUB_OFFLINE=1`, disables telemetry, and can
  run under a network block (`dsm.env.block_network`).
* **Upstream names live in one place.** `configs/upstream.yaml` holds every upstream identifier.
  Exported model directories are audited (`dsm audit model`) to contain none outside the legally
  required `LICENSE`/`NOTICE`.
* **Ground truth is computed, not generated.** Table answers come from pandas, tool results from a
  deterministic simulator, and identity facts from `configs/identity.yaml`. Teacher models only
  write language, and only local Apache/MIT-licensed teachers are used.
* **Everything is tested on CPU.** Tiny random models and a cached copy of the real 248K-token
  vocabulary let the whole pipeline run in CI in minutes.
