#!/usr/bin/env bash
# Step 5 - continued pre-training (~2B tokens; 3-5 days on a 24 GB GPU), bits per byte, gate.
#   MODEL=work/runs/pilot_B1/final scripts/50_cpt.sh   start from the Khmer-extended model
source "$(dirname "$0")/common.sh"
MODEL=${MODEL:-$WORK/models/dsm-base}

step "CPT mixture (weights from shares, available tokens and epoch caps)"
data mix --config configs/data/mix_cpt.yaml --out "$WORK/mix/cpt.yaml" ${CPT_TOKENS:+--total "$CPT_TOKENS"}

step "continued pre-training from $MODEL (resumes automatically)"
$DSM train --config configs/train/cpt_24gb.yaml --data "$WORK/mix/cpt.yaml" --set "model=$MODEL" $TRAIN_ARGS

step "bits per byte against the base model, gate"
if [ ! -f "$RESULTS/base/bpb.json" ]; then bpb "$WORK/models/dsm-base" "$RESULTS/base"; fi
bpb "$WORK/runs/cpt/final" "$RESULTS/cpt"
gate after_cpt "$RESULTS/cpt/bpb.json" "$RESULTS/base/bpb.json"
