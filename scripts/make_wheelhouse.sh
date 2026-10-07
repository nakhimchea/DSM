#!/usr/bin/env bash
# Build a wheelhouse for an air-gapped machine. Run it on a CONNECTED machine with
# the same OS, CPU architecture and Python version as the target.
#
#   scripts/make_wheelhouse.sh [wheelhouse] [extras]
#   # CUDA PyTorch: PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/make_wheelhouse.sh
#
# On the offline machine:
#   python -m venv .venv && . .venv/bin/activate
#   pip install --no-index --find-links wheelhouse -r requirements.lock 'dsm-llm[data,eval]'
set -euo pipefail

OUT="${1:-wheelhouse}"
EXTRAS="${2:-data,eval,kernels}"
mkdir -p "$OUT"
python -m pip download --dest "$OUT" -r requirements.lock
python -m pip download --dest "$OUT" ".[${EXTRAS}]"
python -m pip wheel --no-deps --wheel-dir "$OUT" .
( cd "$OUT" && sha256sum ./* > SHA256SUMS )
echo "wheelhouse ready: $OUT ($(find "$OUT" -maxdepth 1 -type f | wc -l) files)"
