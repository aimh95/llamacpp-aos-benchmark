#!/usr/bin/env bash
set -euo pipefail

# Optional helper. 모델은 보통 직접 받아서 models/model.gguf 로 배치합니다.
# 자동 다운로드가 필요하면 MODEL_URL을 지정해서 사용하세요.
#   MODEL_URL="https://huggingface.co/.../model.gguf" ./scripts/02_download_model.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/models/model.gguf"

if [[ -z "${MODEL_URL:-}" ]]; then
  echo "MODEL_URL is not set. Place your gguf model manually at: $DEST" >&2
  echo "  ln -s /path/to/actual-model.gguf $DEST" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/models"
curl -L --fail -o "$DEST" "$MODEL_URL"
echo "Downloaded to $DEST"
