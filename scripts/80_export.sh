#!/usr/bin/env bash
# Step 9 - GGUF export, per-language quantisation check, audits. Needs DSM_LLAMA_CPP.
#   scripts/80_export.sh [MODEL_DIR] [NAME]        default: work/models/dsm-final -> exports/dsm
#   QUANTS="Q8_0 Q4_K_M" scripts/80_export.sh      4-bit types get an importance matrix (km/en/zh)
source "$(dirname "$0")/common.sh"
MODEL=${1:-$WORK/models/dsm-final}
NAME=${2:-dsm}
OUT=$EXPORTS/$NAME
read -r -a quants <<< "${QUANTS:-Q8_0}"
TEXT=$WORK/eval_text

step "held-out text per language (KL check) and calibration text (importance matrix)"
mkdir -p "$TEXT"
texts=()
for pair in km:fineweb2_km en:fineweb_edu zh:fineweb2_zh; do
  lang=${pair%%:*}
  pool=$DATA/final/${pair#*:}.jsonl.zst
  if [ -f "$pool" ] && [ ! -f "$TEXT/$lang.txt" ]; then
    data select --input "$pool" --sample "$HELDOUT" --max-docs 300 --out "$TEXT/$lang.txt"
    data select --input "$pool" --sample 0.001 --seed 7 --max-docs 200 --out "$TEXT/calibration_$lang.txt"
  fi
  if [ -f "$TEXT/$lang.txt" ]; then texts+=("$lang=$TEXT/$lang.txt"); fi
done
calibration=()
if [[ " ${quants[*]} " =~ \ (I?Q[1-4]) ]] && compgen -G "$TEXT/calibration_*.txt" >/dev/null; then
  cat "$TEXT"/calibration_*.txt > "$TEXT/calibration.txt"
  calibration=(--calibration "$TEXT/calibration.txt")
fi

step "GGUF: BF16 + ${quants[*]} -> $OUT"
$DSM export gguf --model "$MODEL" --out "$OUT" --quant "${quants[@]}" ${calibration[@]+"${calibration[@]}"}
mapfile -t base < <(existing "$OUT"/*-BF16.gguf)

for q in "${quants[@]}"; do
  candidate=${base[0]%-BF16.gguf}-$q.gguf
  results=()
  if [ ${#texts[@]} -gt 0 ]; then
    step "$q: per-language KL against BF16"
    $DSM export kl --reference "${base[0]}" --candidate "$candidate" --text "${texts[@]}" \
      --out "$RESULTS/$NAME/quant_$q.json"
    results+=("$RESULTS/$NAME/quant_$q.json")
  fi
  if [ "${QUANT_SUITES:-1}" = 1 ]; then
    step "$q: suites through llama-server"
    port=${QUANT_PORT:-8091}
    $DSM export serve --gguf "$candidate" --port "$port" --parallel 4 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true' EXIT
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do sleep 2; done
    suite "$RESULTS/$NAME/suite_$q" --endpoint "http://127.0.0.1:$port/v1"
    kill "$server"
    trap - EXIT
    results+=("$RESULTS/$NAME/suite_$q/metrics.json")
  fi
  if [ ${#results[@]} -gt 0 ]; then
    if ! $DSM eval gate --stage after_quant --results "${results[@]}"; then
      if [ "$GATES" = warn ]; then echo "WARNING: gate after_quant failed for $q"; else exit 1; fi
    fi
  fi
done

step "audits: GGUF metadata, offline readiness"
for file in "$OUT"/*.gguf; do $DSM audit gguf "$file"; done
$DSM audit offline "$MODEL" --write-manifest --smoke
