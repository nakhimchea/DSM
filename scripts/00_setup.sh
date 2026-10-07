#!/usr/bin/env bash
# Step 0 - install DSM, check the GPU machine, build llama.cpp.
#   scripts/00_setup.sh                         install into the active Python environment
#   WHEELHOUSE=wheelhouse scripts/00_setup.sh   offline install (wheels from scripts/make_wheelhouse.sh)
source "$(dirname "$0")/common.sh"
EXTRAS=${EXTRAS:-data,eval,kernels,dev}

step "install (extras: $EXTRAS)"
if ! python -c "import torch" 2>/dev/null; then
  echo "Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/" >&2
  exit 1
fi
if [ -n "${WHEELHOUSE:-}" ]; then
  pip install --no-index --find-links "$WHEELHOUSE" -e "${REPO}[${EXTRAS}]"
else
  pip install -e "${REPO}[${EXTRAS}]"
fi

step "GPU, kernels, memory and disk"
$DSM preflight --config configs/train/cpt_24gb.yaml --output-dir "$WORK/runs"

if [ "${SKIP_TESTS:-0}" != 1 ]; then
  step "CPU test-suite (a few minutes)"
  python -m pytest -q "$REPO/tests"
fi

step "llama.cpp at the pinned commit (GGUF export and serving)"
LLAMA_DIR=${LLAMA_DIR:-third_party/llama.cpp}
if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then "$REPO/scripts/build_llama_cpp.sh" "$LLAMA_DIR"; fi
echo "add to your shell profile:  export DSM_LLAMA_CPP=$(cd "$LLAMA_DIR" && pwd)"
