# 05 · Identity: how DSM knows who it is

A model's identity lives in its **weights**, not in its name. The converted checkpoint still
answers "who are you?" the way the upstream model was trained to. A small model cannot be fixed
reliably by a system prompt alone. DSM therefore trains its identity in, at three stages, from one
configuration file, and checks it at every gate.

## 1. Policy

| Question | DSM answers | Never |
|---|---|---|
| Who are you? / What's your name? | "I'm DSM1.0, an AI assistant developed by Doslarb." | another assistant's name |
| Who made you? | Doslarb | another company |
| What model are you based on? | Built on an **open-source foundation model** that Doslarb further trained; it can't share details of the base | denies having a base model, claims to be "trained from scratch", or names the upstream model |
| Are you ChatGPT / Gemini / …? | "No, I'm DSM1.0, developed by Doslarb." | "yes" |
| Can you browse the internet? Do you remember me? | Truthful limitations (offline, no memory between chats) | invented capabilities |
| Who made ChatGPT? | OpenAI (correct facts about *other* systems) | "Doslarb" |

The base-model answer is truthful. The weights ship with the upstream Apache-2.0 `LICENSE` and a
`NOTICE` stating the derivation, so DSM must never deny it. See
[12_licensing_provenance.md](12_licensing_provenance.md).

## 2. One source of truth: `configs/identity.yaml`

```yaml
name: "DSM"
version: "1.0"
display_name: "DSM1.0"
full_name: "DSM1.0-0.8B"
developer: "Doslarb"
default_system_prompt: "You are DSM1.0, an AI assistant developed by Doslarb."
base_model_answer: {en: ..., km: ..., zh: ...}   # the truthful, unnamed answer
capabilities / limitations / deployment: {en, km, zh}
```

Every identity sample is generated from this file plus the seed templates in
`data/seeds/identity/`. To rename the model or change the developer, edit the YAML, rebuild the
identity data, retrain SFT/DPO and re-run the identity gate. No code changes are needed. The same
`default_system_prompt` is baked into the chat template, which uses it whenever a conversation has
no system message.

## 3. Three kinds of identity data

```
dsm build identity --out datasets/sft/identity --copies 2
  identity_sft.jsonl.zst    ~480 conversations  (SFT, ~2 % of the SFT mix after upsampling)
  identity_docs.jsonl.zst   "About DSM" documents (CPT / anneal replay, ~0.01 % of tokens)
  identity_dpo.jsonl.zst    ~70 preference pairs (DPO, ~3 % of pairs)
```

### SFT conversations (`identity_sft`)

| Kind | Count* | What it teaches |
|---|---|---|
| `direct` | 226 | 14 intents (who, name, developer, capabilities, languages, base model, size, internet, memory, human?, knowledge cutoff, privacy, mistakes, feelings) × km/en/zh question paraphrases |
| `deny_other` | 180 | "Are you ChatGPT / Claude / Gemini / Llama …?" → no, DSM |
| `upstream_probe` | 35 | "Are you <upstream name>? Were you made by <upstream company>?" → the truthful unnamed answer. Names come from `configs/upstream.yaml` at build time and appear **only in the user turn** |
| `contrastive` | 12 | Facts about other assistants (ChatGPT → OpenAI) so DSM does not claim everything |
| `persona` | 6 | A deployer's system prompt ("You are Sophea, the assistant of ABC Bank"): follow the persona, stay truthful about the underlying model when asked |
| `multi_turn` | 24 | Identity questions after small talk, including language switches mid-conversation |

\* `--copies 2`, after de-duplication. A quarter of the samples carry a neutral system prompt, so
identity does not depend on the default one.

Answers are paraphrases chosen at random from `answers.yaml`. `--copies` re-samples them. For more
variety, add question paraphrases to `questions.yaml` and answer paraphrases to `answers.yaml`. The
templates contain no names, only `{display_name}`, `{developer}`, and so on.

### Replay documents (`identity_docs`)

Short third-person texts ("DSM1.0-0.8B is a compact language model developed by Doslarb…") in all
three languages. Mixed into **continued pre-training and annealing** at about 0.01 % of tokens.
Facts learned in pre-training survive later stages better than facts learned only in SFT.

### Preference pairs (`identity_dpo`)

| Reason | Chosen | Rejected |
|---|---|---|
| `wrong_identity` | DSM answer | "I am <other assistant / upstream model>, made by <its company>" |
| `trained_from_scratch` | truthful base-model answer | "I was trained from scratch by Doslarb" |
| `false_internet`, `false_memory` | honest limitation | invented capability |

Rejected answers may contain other identities; that is their purpose. The validator allows this
only on the rejected side.

## 4. Keeping other data from undoing it

Every other dataset passes the **identity-leak filter** (`dsm.data.synth.filters.identity_leak`):

* First-person self-identification as another assistant in English, Chinese and Khmer ("I am
  ChatGPT", "我是由…开发的", "ខ្ញុំជា…"), plus the upstream self-identification phrases from
  `configs/upstream.yaml`. Matching samples are dropped. They are never rewritten.
* Public chat data: rows whose first user turn is an identity question are dropped. DSM's own
  identity data answers those. System prompts that give the assistant another identity are
  replaced by DSM's default.
* Teacher outputs (chat regeneration, safety, RAG) go through the same filter.

## 5. Where it goes in training

| Stage | Identity data | Why |
|---|---|---|
| CPT | identity docs, ~0.01 % of tokens | anchor the facts early |
| Anneal | identity docs + a few identity conversations | high-quality phase that the model remembers best |
| SFT | identity conversations, ~2 % of conversations (upsampled ≤ 4×) | the actual behaviour |
| DPO | identity pairs, ~3 % of pairs | push probability mass away from other names |
| After merge / quantisation | identity gate. If it fails, a short *repair replay*: 1 epoch of identity + 10× general SFT data at low LR | merges and 4-bit quantisation can erode small behaviours |

## 6. Gates (`configs/eval/gates.yaml`, M5)

* Identity accuracy ≥ 98 % **per language** on held-out paraphrases. DSM, ឌីអេសអិម and the Chinese
  spelling are all accepted.
* Upstream / other-assistant names in answers: 0.
* Identity mentions on non-identity prompts ≈ 0. The model should not introduce itself in every
  answer.
* The base-model answer never contains "from scratch" or a denial.

## 7. Notes

* Khmer and Chinese seed text was written for this repository and **needs native-speaker review**
  before production. Each seed file says so.
* To change persona behaviour (for example, never reveal the underlying model even when asked),
  edit `context.yaml` → `persona_model_a`. Keep it truthful.
