# 01 · Architecture dissection

All numbers on this page come from the default `DSMConfig` (= DSM1.0-0.8B). You can reproduce
them with `dsm dissect report`.

## 1. The layer stack

```
input ids ─▶ embedding (248,320 × 1024, tied with the output head)
             │
             ├─ 6 × ┌───────────────────────────────┐
             │      │ GDN  block (Gated DeltaNet + MLP) │  layers 0,1,2 / 4,5,6 / …
             │      │ GDN  block                       │
             │      │ GDN  block                       │
             │      │ ATTN block (gated attention + MLP)│  layers 3, 7, 11, 15, 19, 23
             │      └───────────────────────────────┘
             │
             └─ final RMSNorm ─▶ LM head (= embeddingᵀ) ─▶ logits over 248,320 tokens
```

Every block is pre-norm residual:
`h = x + mixer(norm₁(x))`, then `out = h + SwiGLU(norm₂(h))`.

## 2. Where the 752M parameters are

| Component | Parameters | Share |
|---|---:|---:|
| SwiGLU MLPs (24 × 3 × 1024 × 3584) | 264,241,152 | 35.1 % |
| Embedding / tied LM head (248,320 × 1024) | 254,279,680 | 33.8 % |
| Gated DeltaNet mixers (18 layers) | 189,778,752 | 25.2 % |
| Gated attention mixers (6 layers) | 44,043,264 | 5.9 % |
| Norms | 50,176 | 0.0 % |
| **Total (text model)** | **752,393,024** | |

The upstream checkpoint also contains a vision encoder (100.6M) and a multi-token-prediction (MTP)
head (20.5M). DSM is text-only and drops both. The MTP head is kept aside in `extras/` by the
converter, for a possible future speculative-decoding step.

**Take-away:** a third of the model is the vocabulary. Every generated token pays for a
1024 × 248,320 matrix multiply in the LM head, which is ~34 % of decode FLOPs. The embedding is
also where Khmer is weakest: only 79 vocabulary entries contain Khmer characters (see §6).

## 3. The components

### 3.1 Zero-centred RMSNorm

`y = x / rms(x) · (1 + w)`, with `w` initialised to 0. Weight decay then pulls the gain towards 1
instead of 0. Every norm uses this form except the Gated DeltaNet output norm, which uses plain
`w · x / rms(x)`. Exporters must therefore add +1 to these weights when converting to engines that
expect plain RMSNorm; llama.cpp's converter does this.

### 3.2 Gated DeltaNet (18 of 24 layers): a learned, decaying associative memory

Each head keeps a matrix memory `S ∈ ℝ^{128×128}` and updates it per token:

```
S_t = exp(g_t) · S_{t-1}                      (forget: per-token decay, g_t ≤ 0)
S_t = S_t + k_t · (β_t · (v_t − S_tᵀ k_t))ᵀ    (delta rule: write only the prediction error)
o_t = S_tᵀ q_t                                (read)
```

* `q, k, v` come from one projection (`in_proj_qkv`, 1024 → 6144). A depthwise causal convolution
  (kernel 4) and SiLU follow. `q` and `k` are L2-normalised, and `q` is scaled by 128^-½.
* `β_t = σ(in_proj_b · x)` is the write strength. The decay is
  `g_t = −exp(A_log) · softplus(in_proj_a · x + dt_bias)`; `A_log` is kept in fp32.
* The output is normalised per head and gated: `RMSNorm(o) · silu(in_proj_z · x)`, then `out_proj`.
* **Training and prefill** use the *chunked* form. Inside a chunk of 64 tokens, the sequential
  delta-rule updates are condensed into a unit-lower-triangular solve (the "UT transform"); chunks
  are then scanned sequentially. **Decoding** uses the recurrent form, one token at a time.
  `ops_dsm.py` implements both exactly in PyTorch. `flash-linear-attention` supplies fused Triton
  versions on GPU.
* **Cost:** compute is linear in sequence length, and the state is a fixed size: 18 layers × 16
  heads × 128 × 128 fp32 = **18 MiB per sequence, whatever the context length**.

### 3.3 Gated attention (6 of 24 layers): exact retrieval

* Grouped-query attention: 8 query heads and 2 key/value heads, `head_dim = 256`.
* `q_proj` outputs twice the width, laid out per head as `[q | gate]`. The attention output is
  multiplied by `σ(gate)`. The gate lets heads switch themselves off, which reduces
  "attention-sink" artefacts.
* QK-norm (zero-centred RMSNorm on each head's q and k) stabilises training.
* **Partial RoPE**: only the first 64 of 256 dimensions are rotated (θ = 10⁷). The other 192 are
  position-free, which helps long-context extrapolation. The source checkpoint uses multimodal RoPE
  sections; for text-only input, all three position rows are equal and it reduces exactly to the 1D
  RoPE implemented here.
* **Cost:** only these 6 layers keep a KV cache, at **12 KiB per token** in bf16. That is 48 MiB at 4K
  tokens, 384 MiB at 32K, and 3 GiB at the full 262K. An all-attention model of the same depth would
  need 4× more.

### 3.4 SwiGLU MLP

`down(silu(gate(x)) ⊙ up(x))` with an inner size of 3584. This is the largest parameter group,
and it is where most "knowledge" lives.

## 4. What a token costs at inference time

| Quantity | Value |
|---|---|
| Matmul FLOPs per generated token | ≈ 1.50 GFLOP (LM head ≈ 34 %) |
| Weights, bf16 / Q8_0 / Q4_K_M | ≈ 1.50 GB / 0.80 GB / ≈ 0.45 GB |
| KV cache (6 layers) | 12 KiB per token |
| Recurrent + conv state | 18 MiB + 0.8 MiB per sequence (constant) |

Decoding a 0.8B model is **memory-bandwidth bound**: each token reads all the weights once. An upper
bound is `tokens/s ≤ memory bandwidth / weight bytes`. For example, a desktop CPU with ~50 GB/s and
Q8_0 weights gives ≲ 60 tok/s; a GPU is much faster. Halving the weight bytes (Q8 → Q4) roughly
doubles the ceiling, but small models lose more quality from 4-bit quantisation, and Khmer loses the
most (see doc 11).

## 5. Packed training and document isolation

Training packs several documents into each 4096-token row. A new document starts wherever
`position_ids == 0`, and DSM then isolates documents **exactly**:

| Mixer | Isolation mechanism |
|---|---|
| Attention | block-diagonal causal mask; RoPE positions restart at 0 |
| Short convolution | taps that would read the previous document are zeroed |
| Gated DeltaNet state | the first token's log-decay is set to −30; e⁻³⁰ ≈ 10⁻¹³ erases the carried state, and the gradient is unchanged because the true state there is zero anyway |

`tests/modeling/test_packing.py` checks packed == separate documents, with boundaries on both sides
of the 64-token chunk edge, single-token documents, and gradient isolation. The upstream reference
code, by contrast, ignores document boundaries in its PyTorch path.

## 6. The levers you can pull for a "quick response engine"

| Lever | Effect | Where |
|---|---|---|
| **Khmer tokens** (extend the vocabulary) | Khmer currently costs ~0.9 tokens/character (4.7× English); +8–16K Khmer merges roughly halve that, so ~2× faster Khmer generation and 2× more Khmer per context | `dsm tokenizer extend`, doc 11 |
| **Quantisation** | Q8_0 is nearly lossless (0.8 GB); Q4_K_M is ~2× faster on CPU but needs per-language checks | `dsm export gguf`, doc 11 |
| **Context cap** | KV memory grows at 12 KiB/token; serve 8–32K unless needed | llama-server `-c` |
| **`logits_to_keep`** | Prefill computes logits only for the last position (saves ~1 GB per 1K prompt tokens) | built in |
| **Layer pruning** | Remove low-importance blocks (block influence); needs healing by distillation; published layer-pruning studies report collapse beyond ~20 % removal | `dsm dissect layers`, doc 11 (phase 2) |
| **Vocabulary trim** | Removing unused scripts (~48K tokens) saves ~6 % decode compute; deferred | doc 11 |
| **MTP speculative decoding** | 1.5–2× decode in engines that support it, after retraining the MTP head | phase 2 |

## 7. Code map

| Concept | Code |
|---|---|
| Config (field names compatible with engines) | `src/dsm/modeling/configuration_dsm.py` → `DSMConfig` |
| Blocks, masks, models | `src/dsm/modeling/modeling_dsm.py` → `DSMGatedDeltaNet`, `DSMAttention`, `DSMForCausalLM` |
| Gated delta rule, short conv, chunked loss | `src/dsm/modeling/ops_dsm.py` |
| Cache (transformers `DynamicCache` adapter) | `src/dsm/modeling/cache_dsm.py` |
| Decoding loop | `src/dsm/serve/generate.py` |
| Parity with the upstream implementation | `tests/modeling/test_parity.py` |
