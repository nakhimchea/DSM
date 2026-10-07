# DSM1.0-0.8B

A compact, **offline**, trilingual (**Khmer · English · Chinese**) language model, together with
the complete pipeline that builds it: architecture dissection → data → training → evaluation →
compression → local inference.

DSM1.0 is developed by **Doslarb**. It is built on an Apache-2.0 open-weight foundation model with
a hybrid *Gated DeltaNet + gated attention* architecture (provenance and licence obligations:
[`NOTICE`](NOTICE), [`docs/12`](docs/12_licensing_provenance.md)). The model code in this repository
is an independent, fully tested PyTorch re-implementation.

| | |
|---|---|
| Parameters | 752M (text), tied embeddings, 248K-token vocabulary |
| Layers | 24 = 6 × (3 Gated DeltaNet + 1 gated attention), SwiGLU MLPs |
| Context | up to 262K positions; KV cache only in 6 layers (12 KiB/token) |
| Modes | fast responses (no reasoning mode), tool calling, Markdown/LaTeX |
| Runs on | a laptop CPU (GGUF Q8_0 ≈ 0.8 GB) or any GPU; no network required |

## Quick start (your machine)

```bash
pip install -e ".[dev]"                    # core + data + dev tools (CPU is fine)
pip install -e ".[kernels]"                # on a CUDA machine: fast Gated DeltaNet kernels
pytest -q                                  # the whole test-suite runs on CPU in minutes
dsm --help                                 # all pipeline commands
```

The step-by-step guide for a single 24 GB GPU, from downloading the upstream weights to serving
DSM with `llama-server`, is [`docs/13_runbook_24gb.md`](docs/13_runbook_24gb.md).

## Documentation

| | |
|---|---|
| [00 Overview](docs/00_overview.md) | the pipeline at a glance |
| [01 Architecture dissection](docs/01_architecture_dissection.md) | every layer, parameter and cache explained; what you can change |
| [02 Data catalogue](docs/02_data_catalogue.md) | what to collect for km/en/zh, with licences |
| [03 Data preparation](docs/03_data_preparation.md) | cleaning, Khmer normalisation, dedup, PII, packing |
| [04 Training stages](docs/04_training_stages.md) | PT / CPT / anneal / SFT / DPO: data types, budgets, hyper-parameters |
| [05 Identity](docs/05_identity.md) | how the model knows itself; identity replay |
| [06 Safety](docs/06_safety.md) | safeguarding: data, model, deployment |
| [07 Tool calling](docs/07_tool_calling.md) | format, data, bank simulator |
| [08 Tables & Excel](docs/08_tables_excel.md) | spreadsheet serialisation, table QA, formulas |
| [09 Markdown & LaTeX](docs/09_markdown_latex.md) | output formatting data and validators |
| [10 Evaluation](docs/10_evaluation.md) | benchmarks, custom suites, stage gates |
| [11 Compression & local inference](docs/11_compression_local_inference.md) | tokenizer, quantisation, llama.cpp |
| [12 Licensing & provenance](docs/12_licensing_provenance.md) | Apache-2.0 obligations, data licences |
| [13 Runbook (24 GB GPU)](docs/13_runbook_24gb.md) | commands, durations, gates |

## Repository layout

```
configs/      identity, upstream ids, data mixtures, training profiles, eval gates
data/seeds/   small human-reviewable seed templates (identity, safety, tools, tables, formatting)
src/dsm/      modeling · convert · audit · tokenizer · data · train · eval · dissect · serve · export
scripts/      runbook steps (00_setup … 90_serve)
tests/        CPU test-suite (tiny models, real-vocabulary checks, end-to-end smoke test)
docs/         the guide above
```

## Licence

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Model weights derived with this pipeline must be
distributed together with the upstream `LICENSE` and the DSM `NOTICE`.
