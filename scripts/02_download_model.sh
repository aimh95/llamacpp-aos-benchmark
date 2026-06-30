#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/llama3p2-1b-instruct-q4_k_m"
mkdir -p "$MODEL_DIR"

hf download \
  hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF \
  --include "*.gguf" \
  --local-dir "$MODEL_DIR"

GGUF_FILE="$(find "$MODEL_DIR" -name "*.gguf" | head -n 1)"

if [ -z "$GGUF_FILE" ]; then
  echo "ERROR: GGUF file not found"
  exit 1
fi

ln -sf "$GGUF_FILE" "$ROOT_DIR/models/model.gguf"

echo "Model downloaded:"
ls -lh "$GGUF_FILE"

echo
echo "Symlink:"
ls -lh "$ROOT_DIR/models/model.gguf"
