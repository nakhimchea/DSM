# 11 · Compression and fast local inference

DSM runs on llama.cpp: CPU, Apple Silicon or a small GPU, with no network access. This page covers
the export, quantisation (with a quality check per language), serving, and the levers that make
responses fast, especially in Khmer.

## 1. Where the time goes (from `dsm dissect`)

| Lever | Why it matters for DSM |
|---|---|
| **Tokens per character** | Khmer costs ~0.97 tokens per character, against 0.19 for English and 0.47 for Chinese. The same answer in Khmer needs about 5× more decoding steps than in English. A Khmer-extended vocabulary ([14_khmer_tokenizer.md](14_khmer_tokenizer.md)) is the biggest speed-up for Khmer users. |
| **Weight bytes** | Decoding is memory-bound. Q8_0 reads ~0.8 GB per token; Q4_K_M ~0.5 GB. |
| **LM head** | The 248K × 1024 output projection is ~29 % of the decode FLOPs at 4K context. A vocabulary trim is a phase-2 option (~6 % faster decoding, but it permutes token ids). |
| **Cache** | Only 6 of 24 layers keep a KV cache (12 KiB / token); the 18 Gated DeltaNet layers keep a fixed 18.6 MiB state. A 32K-token conversation needs ~0.4 GB of cache instead of 1.5 GB. Long documents and spreadsheets are cheap. |

## 2. Export to GGUF

```bash
scripts/build_llama_cpp.sh third_party/llama.cpp          # pinned commit, -DLLAMA_CURL=OFF, CUDA if present
export DSM_LLAMA_CPP=$PWD/third_party/llama.cpp
dsm export gguf --model work/runs/dpo/final --out exports/dsm --quant Q8_0 Q4_K_M \
    --calibration data/local/calibration_km_en_zh.txt
```

Output: `DSM1.0-0.8B-BF16.gguf`, the requested quantisations, `LICENSE`, `NOTICE`, `README.md`
and `SHA256SUMS`. The export audits every file (`dsm audit gguf`).

**How the export works.** llama.cpp's converter picks its class from `architectures[0]`. DSM ships
`DSMForCausalLM`, so `dsm.export.gguf` registers a small subclass of llama.cpp's own converter for
the upstream text model. DSM keeps the upstream tensor layout, so every weight transform is reused
unchanged: zero-centred norms +1, `A_log → −exp`, V-head ordering, M-RoPE sections. The subclass
only adds the following:

* the DSM name, organisation and description as GGUF metadata, and the DSM chat template from the
  model directory;
* no multi-token-prediction block;
* the attention interval derived from `layer_types`;
* a pre-tokeniser fallback. A Khmer-extended vocabulary (`extension.json` in the model
  directory) changes llama.cpp's tokenizer fingerprint but keeps DSM's pre-tokenisation regex, so
  it maps to the same pre-tokeniser; a test checks that llama.cpp then tokenises like `tokenizers`.

`general.architecture` and `tokenizer.ggml.pre` keep llama.cpp's engine id for this compute graph.
These are the only upstream identifiers in the file (see [12](12_licensing_provenance.md)).

**Verified at the pinned commit** (`tests/export`, run with `DSM_LLAMA_CPP` set):

* a randomly initialised DSM model converted to GGUF gives the **same greedy tokens** in
  llama-server as in PyTorch, on km/en/zh prompts;
* top-5 log-probabilities differ by < 1e-2;
* llama-server's Jinja engine renders the DSM chat template **identically**, including tools and
  tool results.

## 3. Quantisation and its quality gate

| Type | Size (0.8B) | Use |
|---|---|---|
| BF16 | ~1.5 GB | reference for KL; GPU serving |
| **Q8_0** | ~0.8 GB | **default**: practically lossless |
| Q6_K | ~0.65 GB | low-RAM machines |
| Q4_K_M | ~0.5 GB | only if the per-language gate passes; always with an importance matrix |

Small models lose more to quantisation than large ones, and Khmer, the least-represented language,
usually loses most. Quantise with an **importance matrix** computed on mixed km / en / zh text
(`--calibration`), then measure each language separately:

```bash
dsm export kl --reference exports/dsm/DSM1.0-0.8B-BF16.gguf --candidate exports/dsm/DSM1.0-0.8B-Q4_K_M.gguf \
    --text km=eval/km.txt en=eval/en.txt zh=eval/zh.txt --out results/q4/kl.json
dsm eval suite --endpoint http://127.0.0.1:8080/v1 --out results/q4      # behaviour of the served file
dsm eval gate --stage after_quant --results results/q4/kl.json results/q4/metrics.json --baseline results/bf16/metrics.json
```

The `after_quant` gate requires per-language KL ≤ 0.05 and bpb at most +2 %, and re-checks identity
and format. Benchmark accuracy (Belebele ±1.6 points) is too noisy to detect quantisation damage on
its own.

## 4. Serving

```bash
dsm export serve --gguf exports/dsm/DSM1.0-0.8B-Q8_0.gguf --port 8080      # http://127.0.0.1:8080/v1
dsm chat --endpoint http://127.0.0.1:8080/v1                               # terminal chat against it
```

`dsm export serve` starts `llama-server` with these settings:

* bound to `127.0.0.1` (other addresses are refused unless explicitly allowed);
* `--jinja`, so the template and tool parsing come from the GGUF;
* `--reasoning-format none` (DSM is fast-only);
* `--offline` and `--no-webui`;
* an optional `--api-key`;
* `--no-repack` whenever `--parallel` is above 1 (see the known issue below).

The API is OpenAI-compatible, so `tools` in a request come back as structured `tool_calls`.

**Known llama.cpp issue (pinned commit).** On CPU, llama.cpp repacks quantised weights (Q8_0,
Q4_0, ...) into interleaved layouts for faster prompt processing. For this hybrid architecture,
repacked weights give wrong results when two or more sequences share a batch:

* concurrent requests to a `llama-server` with `--parallel 2` return different, corrupted text,
  where sequential requests are correct;
* `llama-perplexity` reports `nan` for a Q8_0 model, because it batches several chunks.

With `--no-repack`, concurrent answers equal the sequential ones and the perplexity is finite, and
BF16 / F16 weights are not affected. GPU offload does not use repacking. `dsm export serve` and
the KL check (`dsm export kl`) therefore switch repacking off whenever more than one sequence can
be decoded at once. `--repack` forces it back on for a single user with `--parallel 1`.
`tests/export/test_export.py::test_quantized_model_with_parallel_requests` reproduces the
problem. Re-run it when you update llama.cpp: if it passes with `repack=True`, the upstream bug is
fixed.

Responsibilities of the application layer:

1. **Normalise user text** with `dsm.text.normalize_text` (NFC + Khmer cluster order + zero-width
   removal) before sending it. Training data was normalised this way, and llama.cpp does not run
   Hugging Face normalisers.
2. **Spreadsheets**: serialise with `dsm.tables.serialize.workbook_to_text`, the same contract as
   training (`dsm chat` → `/load file.xlsx`).
3. **Safeguards** (`dsm.safeguards.runtime`): a tool allowlist, argument validation, user
   confirmation in the app for side-effect tools (the model's `confirmed=true` is never enough),
   PII-masked logs, and prompt-injection flags on tool results and documents.
4. **Sampling**: `generation_config.json` defaults (temperature 0.7, top-p 0.8, top-k 20,
   repetition penalty 1.05). Use greedy decoding for tools and tables.

## 5. Fully offline deployment

* Install from a wheelhouse built on a connected machine (`scripts/make_wheelhouse.sh`, versions in
  `requirements.lock`).
* Every `dsm` command sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and disables telemetry.
  llama.cpp is compiled without libcurl and started with `--offline`.
* `dsm audit offline <model_dir>` checks that no pickled weights are present and that the code
  files are the DSM ones. It verifies `SHA256SUMS` and runs a generation smoke test with the
  network blocked.
* To share the server on a LAN, put an authenticating reverse proxy in front of it. Do not bind
  llama-server to `0.0.0.0`.

## 6. Phase 2 (documented, not built)

| Option | Expected gain | Cost / risk |
|---|---|---|
| Vocabulary trim (drop unused ids, e.g. rare scripts) | ~6 % faster decoding, smaller file | permutes ids: requires remapping, and invalidates the upstream tokenizer fingerprint |
| Layer pruning + healing (guided by `dsm dissect` block influence) | ~4 % per removed layer | quality loss, Khmer-specific; needs a healing CPT |
| Speculative decoding with a DSM-nano draft | 1.5–2× on long answers | a second model to ship and keep in sync |
| vLLM serving (`--hf-overrides` for the architecture name) | high-throughput GPU serving | not needed for a single-user offline setup |
| Ollama | convenient packaging | its Go templates differ from the Jinja template; re-validate tool calls |
