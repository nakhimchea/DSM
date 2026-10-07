# 12 · Licensing, provenance and the "no upstream names" policy

> This is one of the few files in the repository that names the upstream model, because provenance
> has to be documented somewhere. The others are listed in `.scrub-allowlist`. This page is
> engineering guidance, not legal advice. Have counsel review it before any external release.

## 1. What DSM1.0-0.8B is, legally

DSM1.0-0.8B weights are a **Derivative Work** of **Qwen3.5-0.8B / Qwen3.5-0.8B-Base**
(Copyright 2026 Alibaba Cloud), released under the **Apache License 2.0**. Apache-2.0 allows
commercial use, modification and redistribution. Section 4 attaches four obligations to
**redistribution**, i.e. giving the weights to anyone outside your organisation:

| §4 | Obligation | How DSM satisfies it |
|---|---|---|
| (a) | Give recipients a copy of the licence | `dsm convert` copies the upstream `LICENSE` verbatim into every model directory, and the GGUF export ships it alongside |
| (b) | Modified files carry prominent notices of the change | `NOTICE` lists the modifications (architecture code, tokenizer config, chat template, training stages) |
| (c) | Retain copyright and attribution notices | The upstream copyright line stays in `LICENSE`, and `NOTICE` names the source model |
| (d) | Include the upstream NOTICE file, if any | The upstream repo has none; DSM adds its own |

Section 6 grants **no trademark rights**. Renaming the model is therefore not just allowed but the
correct thing to do: a derivative must not be marketed under the upstream product name.

## 2. The "no upstream names" policy

The user-facing artifact is DSM: name, configuration classes, chat template, model card, and the
identity in the weights. The upstream name appears only where it must:

| Location | Why | Enforced by |
|---|---|---|
| Model directory: `LICENSE`, `NOTICE` | Apache-2.0 §4 (legal attribution) | `dsm audit model` allows only these two files |
| GGUF: `general.architecture`, `tokenizer.ggml.pre`, and the `<arch>.*` key namespace | llama.cpp picks its compute graph and pre-tokeniser by these ids (`qwen35`); renaming them means forking llama.cpp | `dsm audit gguf` treats exactly these as allowed |
| Repo: `configs/upstream.yaml` | The single source of upstream ids (download repo ids, parity-test classes, engine ids) read by the code | `dsm audit repo` + `.scrub-allowlist` |
| Repo: `NOTICE`, this document | Provenance | same |
| Generated identity data (not in the repo) | Adversarial prompts ("Are you Qwen?") and DPO *rejected* answers are generated at build time from `configs/upstream.yaml`. Without them the base model's latent self-identification cannot be suppressed. Chosen / SFT answers never contain the names | identity builder + leak filter in `dsm.data.sft.validate` |

The BPE vocabulary contains tokens such as `ĠAlibaba` (the company appears in web text). These are
data, not branding, and are exempt. Removing them would change how ordinary text is tokenised.

## 3. Identity honesty policy

Configured in `configs/identity.yaml` and enforced by the identity builder:

* DSM says it is **DSM1.0, developed by Doslarb**.
* Asked what it is based on, it says it is **built on an open-source foundation model** that
  Doslarb further trained. It neither confirms nor denies a specific base model. It **never**
  says "no" to "are you based on X?" and **never** claims to be trained from scratch.
* Contrastive controls teach correct facts about other AI systems ("ChatGPT was made by OpenAI"),
  so the model does not attribute everything to Doslarb.

This is truthful, matches the Apache-2.0 attribution that ships with the weights, and keeps the
model-risk documentation of a regulated institution consistent with reality.

## 4. Data licences

The licence gate in `configs/data/sources.yaml` refuses non-commercial sources by default. Every
source also records **which model generated it** (`generated_by`), because some model licences
propagate to models trained on their outputs.

| Tier | Examples | Rule |
|---|---|---|
| **Use** | HPLT (CC0), FineWeb-2 / FineWeb-Edu (ODC-BY), MOT-VOA Khmer (CC-BY), ALT, MASSIVE, IndustryCorpus2, WanJuan, SmolTalk2 (non-Llama subsets), Nemotron post-training, xLAM, ToolACE, Hermes-FC, When2Call, TableInstruct, TabFact, TAT-QA, FinQA, GSM8K, Aegis 2.0, XSTest, Do-Not-Answer, Safety-Prompts | Keep attribution in data cards |
| **Needs counsel** | CC-BY-SA (Wikipedia, WikiTableQuestions, FeTaQA, Spider, BIRD); Common Crawl derived web text (copyright stays with page authors; Cambodian TDM rules); NuminaMath-1.5 (solutions partly rewritten by a proprietary model) | Usually fine for training; never redistribute the data itself without the licence |
| **Blocked: non-commercial** | NLLB-200 and data translated with it (incl. Aya Collection km), SeamlessM4T, Bactrian-X, Alpaca derivatives, BELLE, MAP-CC, APIGen-MT, PKU-SafeRLHF / BeaverTails, FalseReject, OSCAR (suspended) | Gate refuses unless `--allow-noncommercial` (a bank should not) |
| **Blocked: naming / derivative clauses** | Llama-generated data (e.g. OpenMathInstruct-2, the Tulu-3 preference mix, SmolTalk magpie-ultra): a distributed model must carry "Llama" in its name. Gemma-generated data: the trained model becomes a "Model Derivative" | Excluded by `generated_by` filters |

## 5. Teacher models (synthetic data, translation, judging)

* **Local only.** Teachers run through a local OpenAI-compatible server (`configs/teacher.yaml`), which
  keeps the pipeline cloud-free.
* **Allowlist: Apache-2.0 / MIT models.** For translation, Hy-MT2 (Apache-2.0, strong Khmer) with
  MADLAD-400 MT as a fallback. For responses and judging, any Apache/MIT instruct model that fits
  your GPU.
* **Avoid** models whose licence demands attribution in the product name or UI (e.g. Llama's naming
  clause, or "Built with …" display requirements in some vendor licences). Avoid
  non-commercial models (NLLB, Aya).
* **Claude is not used as a data generator.** Anthropic's terms prohibit using its outputs to train
  other AI models. Claude wrote this pipeline's code, docs and a small set of human-reviewable seed
  templates; bulk training data comes from licensed datasets and local teachers.

## 6. Internal data

Models trained on internal bank documents (FAQs, policies, tickets) can memorise them. Such a model
is **INTERNAL**: never publish its weights. `dsm data` applies strict PII redaction to sources
tagged `internal: true`, and the model card generator marks such checkpoints.

## 7. Third-party code

| Component | Licence | Location |
|---|---|---|
| SIL Khmer normaliser (khnormal) | MIT, © 2021–2024 SIL Global | `src/dsm/data/khmer/khnormal.py` (licence text in the header) |
| PyTorch, transformers, tokenizers, safetensors | BSD / Apache-2.0 | dependencies |
| flash-linear-attention (optional) | MIT | optional dependency |

## 8. Release checklist

1. `dsm audit model <dir>` → 0 findings; `dsm audit gguf <file>` → 0 findings.
2. `dsm audit offline <dir> --write-manifest --smoke` → OK (pickle-free, checksums, offline generation).
3. `LICENSE` (upstream, verbatim) and `NOTICE` are present next to every weight file.
4. Data cards for every training source show a licence in the "use" or "needs counsel" tier, and
   counsel has signed off on the latter.
5. The identity eval passes in km/en/zh, including adversarial probes.
6. If internal data was used, the release is internal only.
