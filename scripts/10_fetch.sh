#!/usr/bin/env bash
# Step 1 - downloads: the only step that needs the internet. Run it on a connected machine and
# copy work/upstream and datasets/raw to the training machine. Gated datasets need
# `huggingface-cli login` (or HF_TOKEN) first.
#   scripts/10_fetch.sh            fast path: checkpoints, evaluation sets, chat replay
#   FULL=1 scripts/10_fetch.sh     + everything the full path needs (a few hundred GB)
source "$(dirname "$0")/common.sh"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
get() { data download --out "$DATA/raw" "$@"; }

step "upstream checkpoints -> $WORK/upstream (exact revisions in FETCH.json)"
$DSM fetch --out "$WORK/upstream"

step "evaluation sets (never trained on)"
get --eval belebele belebele_en belebele_zh flores_plus flores_plus_en flores_plus_zh \
  sib200 sib200_en sib200_zh

step "general chats (SFT replay)"
get smoltalk2 --limit "${CHAT_LIMIT:-200000}"

if [ "${FULL:-0}" = 1 ]; then
  step "Khmer text: all of it"
  get fineweb2_km hplt3_km hplt2_km
  step "English / Chinese replay, finance, math"
  get fineweb_edu dclm fineweb2_zh hplt3_zh --limit "${REPLAY_LIMIT:-1500000}"
  get industrycorpus2_finance --limit 300000
  get finemath open_web_math --limit 400000
  step "parallel text"
  get alt_km_en massive_km massive_en massive_zh
  step "SFT and preference sets"
  get nemotron_post_training_v1 --limit 100000
  get xlam toolace hermes_fc glaive_fc_v2 when2call tabfact tatqa finqa gsm8k hendrycks_math helpsteer3
  get aegis2 wildguardmix xstest do_not_answer safety_prompts_zh
  if [ -n "$(counsel_flag)" ]; then
    step "sources with counsel sign-off"
    get wikipedia_km voa_khmer wikipedia_zh cci3_hq stack_edu_python wtq numinamath_1_5
  fi
  echo "TICO-19 (CC0) is not on the hub: put its TSV files into $DATA/raw/tico19/ (https://tico-19.github.io/)"
fi
