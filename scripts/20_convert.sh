#!/usr/bin/env bash
# Step 2 - upstream -> DSM checkpoints, verification against the reference implementation,
# architecture report, baseline numbers of the instruct model.
#   VERIFY_ARGS="--device cuda" scripts/20_convert.sh
source "$(dirname "$0")/common.sh"

for kind in base instruct; do
  step "convert $kind"
  if [ ! -d "$WORK/models/dsm-$kind" ]; then
    $DSM convert --src "$WORK/upstream/$kind" --out "$WORK/models/dsm-$kind"
  fi
  $DSM audit model "$WORK/models/dsm-$kind"
  step "verify $kind: tokenizer and logits match the source checkpoint"
  $DSM verify --dsm "$WORK/models/dsm-$kind" --upstream "$WORK/upstream/$kind" ${VERIFY_ARGS:-}
done

step "architecture report -> $RESULTS/dissect"
$DSM dissect --config "$WORK/models/dsm-base/config.json" --out "$RESULTS/dissect"

step "instruct model on DSM's suites (baseline for the SFT gates)"
suite "$RESULTS/instruct" --model "$WORK/models/dsm-instruct"
