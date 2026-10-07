# shellcheck shell=bash
# Shared settings and helpers for the runbook scripts (docs/13_runbook_24gb.md). Sourced, not run.
# Every setting can be overridden from the environment, e.g.
#   TRAIN_ARGS="--set micro_batch_size=2 --set grad_accum=64" scripts/50_cpt.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${DSM_HOME:-$REPO}"       # work/, datasets/, results/ and exports/ live here

WORK=${WORK:-work}            # upstream checkpoints, DSM models, training runs, mixtures
DATA=${DATA:-datasets}        # raw -> clean -> dedup -> final -> packed; sft/; dpo/; eval/
RESULTS=${RESULTS:-results}   # evaluation metrics, one directory per model
EXPORTS=${EXPORTS:-exports}   # GGUF files
TOKENIZER=${TOKENIZER:-$WORK/models/dsm-base}  # model directory whose tokenizer packs the data
PACKED=${PACKED:-$DATA/packed}  # where `pack` writes (the tokenizer pilot uses $DATA/packed_ext)
HELDOUT=${HELDOUT:-0.002}     # share of every text pool kept out of training (bits-per-byte evaluation)
TRAIN_ARGS=${TRAIN_ARGS:-}    # extra options for every `dsm train` / `dsm dpo`
EVAL_ARGS=${EVAL_ARGS:-}      # extra options for `dsm eval suite`, e.g. "--batch-size 16 --device cuda"
DATA_ARGS=${DATA_ARGS:-}      # licence flags for `dsm data`, e.g. "--accept-counsel" after legal sign-off
GATES=${GATES:-strict}        # strict: a failed stage gate stops the script; warn: report and continue
DSM=${DSM:-dsm}

step() { printf '\n== %s\n' "$*"; }

# existing PATH... : print the paths that exist (unmatched globs stay literal and are dropped)
existing() {
  local path
  for path in "$@"; do
    if [ -e "$path" ]; then printf '%s\n' "$path"; fi
  done
}

data() { $DSM data $DATA_ARGS "$@"; }

gate() {  # gate STAGE RESULTS [BASELINE]: check configs/eval/gates.yaml
  local args=(--stage "$1" --results "$2")
  if [ $# -ge 3 ] && [ -f "$3" ]; then args+=(--baseline "$3"); fi
  if ! $DSM eval gate "${args[@]}"; then
    if [ "$GATES" = warn ]; then echo "WARNING: gate $1 failed (GATES=warn: continuing)"; else exit 1; fi
  fi
}

counsel_flag() {  # --accept-counsel when DATA_ARGS carries it
  case " $DATA_ARGS " in *" --accept-counsel "*) echo --accept-counsel ;; esac
}

# pack NAME HELDOUT_NAME FILE... : tokenise documents into $PACKED/NAME, leaving out the
# held-out documents ($HELDOUT, fixed by id); HELDOUT_NAME "-" discards them, otherwise they
# become the evaluation pack $PACKED/HELDOUT_NAME. Existing packs are kept.
pack() {
  local name=$1 held=$2 files
  shift 2
  mapfile -t files < <(existing "$@")
  if [ ${#files[@]} -eq 0 ]; then echo "skip pack $name (no data)"; return 0; fi
  if [ -f "$PACKED/$name.bin" ]; then return 0; fi
  local extra=()
  if [ "$held" != - ]; then extra=(--heldout-out "$PACKED/$held"); fi
  data pack --input "${files[@]}" --tokenizer "$TOKENIZER" --out "$PACKED/$name" \
    --heldout "$HELDOUT" ${extra[@]+"${extra[@]}"}
}

eval_sets() {  # the held-out packs, as `dsm eval bpb --data` arguments
  local lang
  for lang in km en zh; do
    if [ -f "$PACKED/eval_$lang.bin" ]; then printf '%s\n' "$lang=$PACKED/eval_$lang"; fi
  done
}

suite() {  # suite RESULT_DIR --model DIR | --endpoint URL : DSM's suites (+ held-out safety prompts)
  local out=$1 args=() file
  shift
  for file in "$DATA"/eval/safety_*.jsonl; do
    if [ -f "$file" ]; then args+=(--safety-file "$file"); fi
  done
  $DSM eval suite "$@" --out "$out" ${args[@]+"${args[@]}"} $EVAL_ARGS
}

bpb() {  # bpb MODEL_DIR RESULT_DIR
  local sets
  mapfile -t sets < <(eval_sets)
  $DSM eval bpb --model "$1" --data "${sets[@]}" --out "$2/bpb.json"
}
