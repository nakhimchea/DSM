#!/usr/bin/env bash
# Optional - the Khmer vocabulary pilot (docs/14_khmer_tokenizer.md): learn Khmer merges, give
# the model rows for them, re-pack, train arm A (original vocabulary) and arm B (extended) on
# the same bytes, compare bits per byte. Needs step 3.
source "$(dirname "$0")/common.sh"
BASE=$WORK/models/dsm-base
EXT=$WORK/models/dsm-base-km-ext
TOK=$WORK/tok/km-ext
f=$DATA/final

step "bits per byte of the original model (also the CPT gate's baseline)"
if [ ! -f "$RESULTS/base/bpb.json" ]; then bpb "$BASE" "$RESULTS/base"; fi

step "learn ${NEW_TOKENS:-8192} Khmer merges, then the model rows"
mapfile -t km < <(existing $f/wikipedia_km.jsonl.zst $f/fineweb2_km.jsonl.zst $f/hplt3_km.jsonl.zst)
if [ ! -d "$TOK" ]; then
  $DSM tokenizer extend --tokenizer "$BASE" --input "${km[@]}" --new-tokens "${NEW_TOKENS:-8192}" --out "$TOK"
fi
if [ ! -d "$EXT" ]; then $DSM tokenizer remap --model "$BASE" --tokenizer "$TOK" --out "$EXT"; fi

step "tokens per character on held-out text"
texts=()
for pair in km:fineweb2_km en:fineweb_edu zh:fineweb2_zh; do
  lang=${pair%%:*}
  pool=$f/${pair#*:}.jsonl.zst
  if [ -f "$pool" ]; then
    data select --input "$pool" --sample "$HELDOUT" --max-docs 500 --out "$WORK/tok/eval_$lang.txt"
    texts+=("$lang=$WORK/tok/eval_$lang.txt")
  fi
done
$DSM tokenizer analyze --tokenizer "original=$BASE" "extended=$EXT" --text "${texts[@]}" \
  --json "$RESULTS/tokenizer_pilot.json"

step "re-pack every CPT pool with the extended tokenizer -> $DATA/packed_ext"
repack() { PACKED=$DATA/packed_ext TOKENIZER=$EXT pack "$@"; }
repack km_web eval_km $f/fineweb2_km.jsonl.zst $f/hplt3_km.jsonl.zst $f/hplt2_km.jsonl.zst
repack km_wiki_news - $f/wikipedia_km.jsonl.zst $f/voa_khmer.jsonl.zst
repack en_web_edu eval_en $f/fineweb_edu.jsonl.zst $f/dclm.jsonl.zst
repack zh_web eval_zh $f/fineweb2_zh.jsonl.zst $f/hplt3_zh.jsonl.zst $f/cci3_hq.jsonl.zst \
  $f/wikipedia_zh.jsonl.zst $f/industrycorpus2_finance.jsonl.zst
repack parallel - $f/parallel_*.jsonl.zst
repack math - $f/finemath.jsonl.zst $f/open_web_math.jsonl.zst
repack code - $f/stack_edu_python.jsonl.zst
repack identity_docs - "$DATA/sft/seeds/identity/identity_docs.jsonl.zst"
repack tables - "$DATA/clean/tables_text.jsonl.zst"

eval_list() {  # the evaluation packs of a pack directory, as a YAML list
  local lang
  echo "  eval:"
  for lang in km en zh; do
    if [ -f "$1/eval_$lang.bin" ]; then echo "  - {path: $1/eval_$lang, name: $lang}"; fi
  done
}
{
  echo "data:"
  echo "  train:"
  for name in km_web km_wiki_news parallel; do
    if [ -f "$DATA/packed_ext/$name.bin" ]; then echo "  - {path: $DATA/packed_ext/$name, name: $name}"; fi
  done
  eval_list "$DATA/packed_ext"
} > "$WORK/mix/pilot_B1.yaml"
{
  sed "s#$DATA/packed/#$DATA/packed_ext/#g" "$WORK/mix/cpt.yaml"
  eval_list "$DATA/packed_ext"
} > "$WORK/mix/cpt_ext.yaml"

step "arm A: original vocabulary, ${PILOT_TOKENS_A:-1.5e8} tokens"
$DSM train --config configs/train/cpt_24gb.yaml --data "$WORK/mix/cpt.yaml" \
  --set "output_dir=$WORK/runs/pilot_A" --set "max_tokens=${PILOT_TOKENS_A:-1.5e8}" $TRAIN_ARGS
step "arm B1: only the new embedding rows, ${PILOT_TOKENS_B1:-5.0e7} tokens"
$DSM train --config configs/train/pilot_tok_24gb.yaml --data "$WORK/mix/pilot_B1.yaml" \
  --set "max_tokens=${PILOT_TOKENS_B1:-5.0e7}" $TRAIN_ARGS
step "arm B2: CPT of the whole model, ${PILOT_TOKENS_B2:-1.0e8} tokens"
$DSM train --config configs/train/cpt_24gb.yaml --data "$WORK/mix/cpt_ext.yaml" \
  --set "model=$WORK/runs/pilot_B1/final" --set "output_dir=$WORK/runs/pilot_B2" \
  --set "max_tokens=${PILOT_TOKENS_B2:-1.0e8}" $TRAIN_ARGS

step "compare: bits per byte on the same held-out documents"
bpb "$WORK/runs/pilot_A/final" "$RESULTS/pilot_A"
PACKED=$DATA/packed_ext bpb "$WORK/runs/pilot_B2/final" "$RESULTS/pilot_B"
python - "$RESULTS/pilot_A/bpb.json" "$RESULTS/pilot_B/bpb.json" <<'PY'
import json
import sys

a, b = (json.load(open(path)) for path in sys.argv[1:3])
for key in sorted(k for k in a if k.startswith("bpb/")):
    change = (b[key] - a[key]) / a[key] if key in b else float("nan")
    print(f"{key:8}  A {a[key]:.4f}   B {b.get(key, float('nan')):.4f}   ({change:+.1%})")
print("adopt B if Khmer is within ~1 % of A (or better) and English / Chinese within 1 % (docs/14)")
PY
