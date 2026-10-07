#!/usr/bin/env bash
# Step 10 - serve DSM with llama-server: an OpenAI-compatible API on 127.0.0.1, fully offline.
#   scripts/90_serve.sh [FILE.gguf]          default: the Q8_0 export in exports/dsm
#   PORT=8080 SERVE_ARGS="--gpu-layers 99" scripts/90_serve.sh
source "$(dirname "$0")/common.sh"
GGUF=${1:-}
if [ -z "$GGUF" ]; then
  mapfile -t found < <(existing "$EXPORTS"/dsm/*-Q8_0.gguf)
  if [ ${#found[@]} -eq 0 ]; then echo "no GGUF in $EXPORTS/dsm: run scripts/80_export.sh" >&2; exit 1; fi
  GGUF=${found[0]}
fi
PORT=${PORT:-8080}
echo "chat:  dsm chat --endpoint http://127.0.0.1:$PORT/v1"
echo "API:   http://127.0.0.1:$PORT/v1/chat/completions (apply dsm.text.normalize_text to user input)"
exec $DSM export serve --gguf "$GGUF" --port "$PORT" ${SERVE_ARGS:-}
