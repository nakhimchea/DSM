#!/usr/bin/env bash
# Build the llama.cpp tools DSM uses (converter, llama-quantize, llama-imatrix,
# llama-perplexity, llama-server, llama-cli) at the commit pinned in
# configs/upstream.yaml. Network code is compiled out (-DLLAMA_CURL=OFF); CUDA is
# enabled automatically when nvidia-smi is present (DSM_CUDA=0 to force CPU).
#
#   scripts/build_llama_cpp.sh [third_party/llama.cpp]
#   export DSM_LLAMA_CPP=$PWD/third_party/llama.cpp
set -euo pipefail

DEST="${1:-third_party/llama.cpp}"
REPO="$(python -c "from dsm.utils import load_upstream; print(load_upstream()['upstream']['llama_cpp']['repo'])")"
REF="$(python -c "from dsm.utils import load_upstream; print(load_upstream()['upstream']['llama_cpp']['ref'])")"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none "$REPO" "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin "$REF" 2>/dev/null || git -C "$DEST" fetch origin
git -C "$DEST" checkout --detach "$REF"

FLAGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF)
if [ "${DSM_CUDA:-auto}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1; then
  FLAGS+=(-DGGML_CUDA=ON)
fi
cmake -S "$DEST" -B "$DEST/build" "${FLAGS[@]}"
cmake --build "$DEST/build" -j "$(nproc)" --target llama-server llama-quantize llama-imatrix llama-perplexity llama-cli

echo "built: $DEST/build/bin"
echo "export DSM_LLAMA_CPP=$(cd "$DEST" && pwd)"
