#!/usr/bin/env bash
# Step 3 - pre-training data: evaluation sets, clean -> dedup -> decontaminate -> pack (with
# held-out evaluation packs), parallel text, DSM's own documents, high-quality subsets for
# annealing, and the CPT mixture. Sources that were not downloaded are skipped; finished
# outputs are kept (delete a file to redo that step).
source "$(dirname "$0")/common.sh"

step "evaluation sets -> $DATA/eval (also the decontamination reference)"
mkdir -p "$DATA/eval"
mapfile -t files < <(existing "$DATA/raw/belebele" "$DATA/raw/belebele_en" "$DATA/raw/belebele_zh")
if [ ${#files[@]} -gt 0 ]; then
  data evalset --kind belebele --input "${files[@]}" --out "$DATA/eval/belebele.jsonl"
fi
for lang in km en zh; do
  src=sib200
  if [ "$lang" != km ]; then src=sib200_$lang; fi
  if [ -d "$DATA/raw/$src" ]; then
    data evalset --kind sib200 --lang "$lang" --input "$DATA/raw/$src" --out "$DATA/eval/sib200_$lang.jsonl"
  fi
done
mapfile -t files < <(existing "$DATA/raw/flores_plus" "$DATA/raw/flores_plus_en" "$DATA/raw/flores_plus_zh")
if [ ${#files[@]} -ge 2 ]; then
  data parallel --layout grouped --source flores_plus --input "${files[@]}" --out-pairs "$DATA/eval/flores_pairs.jsonl"
fi
# held-out safety prompts (XSTest: harmful and benign-but-sensitive); never trained on
if [ -d "$DATA/raw/xstest" ]; then
  data evalset --kind safety --converter xstest --input "$DATA/raw/xstest" --out "$DATA/eval/safety_xstest.jsonl"
fi

step "clean: normalise, language ID, quality filters, PII redaction"
clean() {  # clean SOURCE [options]
  local name=$1
  shift
  if [ ! -d "$DATA/raw/$name" ]; then echo "skip $name (not downloaded)"; return 0; fi
  if [ ! -f "$DATA/clean/$name.jsonl.zst" ]; then
    data clean --source "$name" --input "$DATA/raw/$name" --out "$DATA/clean/$name.jsonl.zst" "$@"
  fi
}
for source in fineweb2_km hplt3_km hplt2_km wikipedia_km voa_khmer dclm fineweb2_zh hplt3_zh cci3_hq \
  wikipedia_zh industrycorpus2_finance finemath open_web_math stack_edu_python; do
  clean "$source"
done
clean fineweb_edu --keep-fields int_score

step "deduplicate per language (the earlier, better source wins)"
dedup() {  # dedup FILE...
  local files todo=0 file
  mapfile -t files < <(existing "$@")
  if [ ${#files[@]} -eq 0 ]; then return 0; fi
  for file in "${files[@]}"; do
    if [ ! -f "$DATA/dedup/$(basename "$file")" ]; then todo=1; fi
  done
  if [ $todo = 1 ]; then data dedup --input "${files[@]}" --out-dir "$DATA/dedup"; fi
}
c=$DATA/clean
dedup $c/wikipedia_km.jsonl.zst $c/voa_khmer.jsonl.zst $c/fineweb2_km.jsonl.zst $c/hplt3_km.jsonl.zst \
  $c/hplt2_km.jsonl.zst
dedup $c/fineweb_edu.jsonl.zst $c/dclm.jsonl.zst
dedup $c/cci3_hq.jsonl.zst $c/wikipedia_zh.jsonl.zst $c/fineweb2_zh.jsonl.zst $c/hplt3_zh.jsonl.zst \
  $c/industrycorpus2_finance.jsonl.zst
dedup $c/finemath.jsonl.zst $c/open_web_math.jsonl.zst
dedup $c/stack_edu_python.jsonl.zst

step "parallel text -> CPT documents, translation conversations, sentence pairs"
parallel() {  # parallel SOURCE LAYOUT INPUT...
  local name=$1 layout=$2 files
  shift 2
  mapfile -t files < <(existing "$@")
  if [ ${#files[@]} -eq 0 ]; then echo "skip $name (not downloaded)"; return 0; fi
  if [ ! -f "$DATA/dedup/parallel_$name.jsonl.zst" ]; then
    mkdir -p "$DATA/sft/khmer" "$DATA/parallel"
    data parallel --layout "$layout" --source "$name" --input "${files[@]}" \
      --out-docs "$DATA/dedup/parallel_$name.jsonl.zst" --out-sft "$DATA/sft/khmer/$name.jsonl.zst" \
      --out-pairs "$DATA/parallel/$name.jsonl.zst"
  fi
}
parallel alt_km_en aligned "$DATA/raw/alt_km_en"
parallel massive_km grouped "$DATA/raw/massive_km" "$DATA/raw/massive_en" "$DATA/raw/massive_zh"
parallel tico19 pairs "$DATA/raw/tico19"
parallel nllb_mined_km_en aligned "$DATA/raw/nllb_mined_km_en"

step "decontaminate against every evaluation set"
mapfile -t files < <(existing "$DATA"/dedup/*.jsonl.zst)
todo=()
for file in ${files[@]+"${files[@]}"}; do
  if [ ! -f "$DATA/final/$(basename "$file")" ]; then todo+=("$file"); fi
done
if [ ${#todo[@]} -gt 0 ]; then
  data decontam --input "${todo[@]}" --eval "$REPO/data/eval" "$DATA/eval" --out-dir "$DATA/final"
fi

step "pack with $TOKENIZER ($HELDOUT of every pool held out for bits per byte)"
f=$DATA/final
pack km_web eval_km $f/fineweb2_km.jsonl.zst $f/hplt3_km.jsonl.zst $f/hplt2_km.jsonl.zst
pack km_wiki_news - $f/wikipedia_km.jsonl.zst $f/voa_khmer.jsonl.zst
pack en_web_edu eval_en $f/fineweb_edu.jsonl.zst $f/dclm.jsonl.zst
pack zh_web eval_zh $f/fineweb2_zh.jsonl.zst $f/hplt3_zh.jsonl.zst $f/cci3_hq.jsonl.zst \
  $f/wikipedia_zh.jsonl.zst $f/industrycorpus2_finance.jsonl.zst
pack parallel - $f/parallel_*.jsonl.zst
pack math - $f/finemath.jsonl.zst $f/open_web_math.jsonl.zst
pack code - $f/stack_edu_python.jsonl.zst

step "DSM's own documents: identity replay and spreadsheet text"
if [ ! -d "$DATA/sft/seeds" ]; then $DSM build seeds --out "$DATA/sft/seeds"; fi
pack identity_docs - "$DATA/sft/seeds/identity/identity_docs.jsonl.zst"
if [ ! -f "$DATA/clean/tables_text.jsonl.zst" ]; then
  data render-sft --input "$DATA/sft/seeds/tables.jsonl.zst" --tokenizer "$TOKENIZER" \
    --out "$DATA/clean/tables_text.jsonl.zst"
fi
pack tables - "$DATA/clean/tables_text.jsonl.zst"

step "high-quality subsets and replay sample for annealing"
pick() {  # pick NAME "SELECT OPTIONS" FILE...
  local name=$1 options=$2 files
  shift 2
  mapfile -t files < <(existing "$@")
  if [ ${#files[@]} -eq 0 ] || [ -f "$f/$name.jsonl.zst" ]; then return 0; fi
  # shellcheck disable=SC2086  # the options are meant to split
  data select --input "${files[@]}" $options --out "$f/$name.jsonl.zst"
}
# thresholds on the statistics `dsm data clean` stores per document (tune them to your data)
KM_HQ=${KM_HQ:-"--min chars=1500 khmer_share=0.9 --max khmer_broken_per_1k=1 dup_lines=0.05"}
EN_HQ=${EN_HQ:-"--min int_score=4"}
ZH_HQ=${ZH_HQ:-"--min chars=1000 han_share=0.85 --max dup_lines=0.02"}
REPLAY=${REPLAY:-"--sample 0.05 --seed 1"}  # seed 1: independent of the held-out split (seed 0)
pick km_hq "$KM_HQ" $f/wikipedia_km.jsonl.zst $f/voa_khmer.jsonl.zst $f/fineweb2_km.jsonl.zst $f/hplt3_km.jsonl.zst
pick en_hq "$EN_HQ" $f/fineweb_edu.jsonl.zst
pick zh_hq "$ZH_HQ" $f/cci3_hq.jsonl.zst $f/wikipedia_zh.jsonl.zst $f/fineweb2_zh.jsonl.zst
pick replay "$REPLAY" $f/fineweb2_km.jsonl.zst $f/hplt3_km.jsonl.zst $f/fineweb_edu.jsonl.zst \
  $f/fineweb2_zh.jsonl.zst
pack km_hq - $f/km_hq.jsonl.zst
pack en_hq - $f/en_hq.jsonl.zst
pack zh_hq - $f/zh_hq.jsonl.zst
pack parallel_hq - $f/parallel_alt_km_en.jsonl.zst $f/parallel_massive_km.jsonl.zst $f/parallel_tico19.jsonl.zst
pack replay_mix - $f/replay.jsonl.zst

step "CPT mixture -> $WORK/mix/cpt.yaml"
data mix --config configs/data/mix_cpt.yaml --out "$WORK/mix/cpt.yaml" ${CPT_TOKENS:+--total "$CPT_TOKENS"}
