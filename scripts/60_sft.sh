#!/usr/bin/env bash
# Step 7 - SFT (~200K conversations, 2 epochs) from the annealed model, suites, gate.
#   CHAT_VECTOR=1 scripts/60_sft.sh   start from anneal + (instruct - base); A/B it on the suites
source "$(dirname "$0")/common.sh"
START=$WORK/runs/anneal/final
RUN=${RUN:-$WORK/runs/sft}

if [ "${CHAT_VECTOR:-0}" = 1 ]; then
  step "chat vector: anneal + (instruct - base)"
  if [ ! -d "$WORK/models/anneal-cv" ]; then
    $DSM merge chat-vector --base "$WORK/models/dsm-base" --instruct "$WORK/models/dsm-instruct" \
      --target "$START" --out "$WORK/models/anneal-cv"
  fi
  START=$WORK/models/anneal-cv
fi

step "SFT mixture"
data mix --config configs/data/mix_sft.yaml --out "$WORK/mix/sft.yaml" ${SFT_TOTAL:+--total "$SFT_TOTAL"}

step "SFT from $START"
$DSM train --config configs/train/sft_24gb.yaml --data "$WORK/mix/sft.yaml" --set "model=$START" \
  --set "output_dir=$RUN" $TRAIN_ARGS

step "suites and gate (baseline: the instruct model)"
suite "$RESULTS/sft" --model "$RUN/final"
gate after_sft "$RESULTS/sft/metrics.json" "$RESULTS/instruct/metrics.json"
