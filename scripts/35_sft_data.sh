#!/usr/bin/env bash
# Step 4 - SFT and preference data -> datasets/sft/<category>/*.jsonl.zst and datasets/dpo/.
# Steps that need the local teacher / translator (configs/teacher.yaml) run with TEACHER=1.
source "$(dirname "$0")/common.sh"
COUNSEL=$(counsel_flag)

step "DSM's own data: identity, safety, tools, tables, formatting, RAG"
if [ ! -d "$DATA/sft/seeds" ]; then $DSM build seeds --out "$DATA/sft/seeds" --per-weight "${PER_WEIGHT:-10}"; fi

step "public datasets -> one conversation schema (licence gate from the catalogue)"
convert() {  # convert SOURCE CATEGORY [options]
  local name=$1 category=$2
  shift 2
  if [ ! -d "$DATA/raw/$name" ]; then echo "skip $name (not downloaded)"; return 0; fi
  local out="$DATA/sft/$category/$name.jsonl.zst"
  if [ ! -f "$out" ]; then $DSM build convert --source "$name" --input "$DATA/raw/$name" --out "$out" "$@"; fi
}
convert smoltalk2 chat
convert nemotron_post_training_v1 chat
for source in xlam toolace hermes_fc glaive_fc_v2 when2call; do convert "$source" tools; done
convert massive_km tools --dataset massive     # Khmer utterances -> Khmer function calls
for source in tabfact tatqa finqa; do convert "$source" tables; done
convert wtq tables $COUNSEL
convert gsm8k math
convert hendrycks_math math
convert numinamath_1_5 math $COUNSEL
convert helpsteer3 ../dpo                       # preference pairs -> datasets/dpo/helpsteer3.jsonl.zst

if [ "${TEACHER:-0}" = 1 ]; then
  step "Khmer versions with the local translator (protected spans; native-speaker review advised)"
  mkdir -p "$DATA/sft/khmer" "$WORK/tmp"
  for file in "$DATA"/sft/seeds/tools.jsonl.zst "$DATA"/sft/seeds/tables.jsonl.zst \
    "$DATA"/sft/seeds/formatting.jsonl.zst "$DATA"/sft/seeds/rag.jsonl.zst; do
    out="$DATA/sft/khmer/$(basename "$file" .jsonl.zst)_km.jsonl.zst"
    if [ ! -f "$out" ]; then $DSM build translate --target km --input "$file" --out "$out"; fi
  done
  if [ -f "$DATA/sft/chat/smoltalk2.jsonl.zst" ] && [ ! -f "$DATA/sft/khmer/chat_km.jsonl.zst" ]; then
    data select --input "$DATA/sft/chat/smoltalk2.jsonl.zst" --sample "${KHMER_CHAT_SHARE:-0.1}" \
      --out "$WORK/tmp/chat_for_km.jsonl.zst"
    $DSM build translate --target km --input "$WORK/tmp/chat_for_km.jsonl.zst" --out "$DATA/sft/khmer/chat_km.jsonl.zst"
  fi
  step "public safety prompts answered by the local teacher under DSM's policy"
  for source in aegis2 wildguardmix do_not_answer safety_prompts_zh; do  # xstest: evaluation only
    out="$DATA/sft/safety/$source.jsonl.zst"
    if [ -d "$DATA/raw/$source" ] && [ ! -f "$out" ]; then
      $DSM build safety --public "$DATA/raw/$source" --converter "$source" --teacher --out "$out"
    fi
  done
else
  echo "TEACHER=1 not set: skipped Khmer translations and teacher-answered safety prompts"
fi

step "preference pairs"
pref() {  # pref KIND OUT INPUT...
  local kind=$1 out=$2 files
  shift 2
  mapfile -t files < <(existing "$@")
  if [ ${#files[@]} -eq 0 ] || [ -f "$out" ]; then return 0; fi
  $DSM build preference --kind "$kind" --input "${files[@]}" --out "$out"
}
pref format "$DATA/dpo/format.jsonl.zst" "$DATA"/sft/seeds/tables.jsonl.zst \
  "$DATA"/sft/seeds/formatting.jsonl.zst "$DATA"/sft/math/*.jsonl.zst
pref language "$DATA/dpo/language.jsonl.zst" "$DATA"/sft/seeds/*.jsonl.zst "$DATA"/sft/khmer/*_km.jsonl.zst
pref over_refusal "$DATA/dpo/over_refusal.jsonl.zst" "$DATA"/sft/seeds/safety/safety_sft.jsonl.zst \
  "$DATA"/sft/safety/*.jsonl.zst

step "validate every SFT / DPO file"
mapfile -t files < <(existing "$DATA"/sft/*/*.jsonl.zst "$DATA"/sft/seeds/*/*.jsonl.zst "$DATA"/dpo/*.jsonl.zst)
$DSM build validate "${files[@]}"

step "SFT mixture -> $WORK/mix/sft.yaml"
data mix --config configs/data/mix_sft.yaml --out "$WORK/mix/sft.yaml" ${SFT_TOTAL:+--total "$SFT_TOTAL"}
