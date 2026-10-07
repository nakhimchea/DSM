#!/usr/bin/env bash
# Step 8 - DPO: on-policy pairs (TEACHER=1: the SFT model served locally, judged by the local
# teacher) + identity / safety / format / language / over-refusal pairs + HelpSteer3; then
# SFT <-> DPO interpolation, suites and gate.
source "$(dirname "$0")/common.sh"
SFT=$WORK/runs/sft/final
PORT=${POLICY_PORT:-8090}  # must match `policy.base_url` in configs/teacher.yaml

if [ "${TEACHER:-0}" = 1 ] && [ ! -f "$DATA/dpo/on_policy.jsonl.zst" ]; then
  step "on-policy pairs: serve the SFT model, sample answers, let the local teacher judge them"
  mkdir -p "$WORK/tmp"
  mapfile -t gguf < <(existing "$EXPORTS"/sft/*-Q8_0.gguf)
  if [ ${#gguf[@]} -eq 0 ]; then
    $DSM export gguf --model "$SFT" --out "$EXPORTS/sft" --quant Q8_0
    mapfile -t gguf < <(existing "$EXPORTS"/sft/*-Q8_0.gguf)
  fi
  $DSM export serve --gguf "${gguf[0]}" --port "$PORT" --parallel 4 &
  server=$!
  trap 'kill "$server" 2>/dev/null || true' EXIT
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 2; done
  mapfile -t files < <(existing "$DATA"/sft/chat/*.jsonl.zst "$DATA"/sft/khmer/*.jsonl.zst \
    "$DATA"/sft/tools/*.jsonl.zst "$DATA"/sft/tables/*.jsonl.zst)
  data select --input "${files[@]}" --sample "${ONPOLICY_SHARE:-0.05}" \
    --max-docs "${ONPOLICY_PROMPTS:-20000}" --out "$WORK/tmp/onpolicy_prompts.jsonl.zst"
  $DSM build preference --kind judged --input "$WORK/tmp/onpolicy_prompts.jsonl.zst" --samples 4 \
    --out "$DATA/dpo/on_policy.jsonl.zst"
  kill "$server"
  trap - EXIT
fi

step "DPO mixture"
data mix --config configs/data/mix_dpo.yaml --out "$WORK/mix/dpo.yaml" ${DPO_TOTAL:+--total "$DPO_TOTAL"}

step "DPO from the SFT model (reference log-probs computed once and cached)"
$DSM dpo --config configs/train/dpo_24gb.yaml --data "$WORK/mix/dpo.yaml" $TRAIN_ARGS

step "interpolate SFT <-> DPO (alpha ${ALPHA:-0.7}) -> $WORK/models/dsm-final"
$DSM merge interpolate --a "$SFT" --b "$WORK/runs/dpo/final" --alpha "${ALPHA:-0.7}" --out "$WORK/models/dsm-final"

step "suites and gate (baseline: SFT)"
suite "$RESULTS/final" --model "$WORK/models/dsm-final"
gate after_dpo "$RESULTS/final/metrics.json" "$RESULTS/sft/metrics.json"
$DSM audit model "$WORK/models/dsm-final"
