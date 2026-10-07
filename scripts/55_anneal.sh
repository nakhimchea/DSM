#!/usr/bin/env bash
# Step 6 - annealing: 0.3-0.5B tokens of the best data + instruction-formatted text + identity
# documents + replay, learning rate decaying to ~0. Needs steps 3 and 4.
source "$(dirname "$0")/common.sh"

step "instruction-formatted text from the SFT data"
if [ ! -f "$DATA/final/instructions_text.jsonl.zst" ]; then
  mapfile -t files < <(existing "$DATA"/sft/seeds/*.jsonl.zst "$DATA"/sft/chat/*.jsonl.zst \
    "$DATA"/sft/tools/*.jsonl.zst "$DATA"/sft/tables/*.jsonl.zst "$DATA"/sft/math/*.jsonl.zst \
    "$DATA"/sft/khmer/*.jsonl.zst)
  data render-sft --input "${files[@]}" --tokenizer "$TOKENIZER" --out "$DATA/clean/instructions_text.jsonl.zst"
  data decontam --input "$DATA/clean/instructions_text.jsonl.zst" --eval "$REPO/data/eval" "$DATA/eval" \
    --out-dir "$DATA/final"
fi
pack instructions_text - "$DATA/final/instructions_text.jsonl.zst"

step "annealing mixture"
data mix --config configs/data/mix_anneal.yaml --out "$WORK/mix/anneal.yaml" ${ANNEAL_TOKENS:+--total "$ANNEAL_TOKENS"}

step "anneal from the CPT model"
$DSM train --config configs/train/anneal_24gb.yaml --data "$WORK/mix/anneal.yaml" $TRAIN_ARGS

step "bits per byte against CPT, gate"
bpb "$WORK/runs/anneal/final" "$RESULTS/anneal"
gate after_anneal "$RESULTS/anneal/bpb.json" "$RESULTS/cpt/bpb.json"
