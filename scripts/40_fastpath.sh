#!/usr/bin/env bash
# v0.1 fast path (1-2 days): SFT of the converted instruct checkpoint on DSM's own data plus
# general replay -> suites and gates -> GGUF -> llama-server. Needs steps 0, 1, 2 and 4.
source "$(dirname "$0")/common.sh"
RUN=$WORK/runs/fastpath_sft

step "SFT (resumes automatically)"
$DSM train --config configs/train/fastpath_sft_24gb.yaml $TRAIN_ARGS

step "suites and gates (baseline: the instruct model)"
suite "$RESULTS/fastpath" --model "$RUN/final"
gate after_sft "$RESULTS/fastpath/metrics.json" "$RESULTS/instruct/metrics.json"

if [ "${EXPORT:-1}" = 1 ]; then "$REPO/scripts/80_export.sh" "$RUN/final" fastpath; fi
