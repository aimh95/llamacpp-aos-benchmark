#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_BIN_DIR="$ROOT_DIR/build-android/bin"
MODEL_PATH="$ROOT_DIR/models/model.gguf"

: "${DEVICE_DIR:=/data/local/tmp/llamacpp-bench}"

adb devices
adb shell "mkdir -p $DEVICE_DIR"

for bin in llama-cli llama-bench; do
  src="$BUILD_BIN_DIR/$bin"
  if [[ ! -f "$src" ]]; then
    echo "Missing binary: $src (run scripts/01_build_android_cpu.sh first)" >&2
    exit 1
  fi
  adb push "$src" "$DEVICE_DIR/$bin"
  adb shell "chmod +x $DEVICE_DIR/$bin"
done

if [[ -f "$MODEL_PATH" ]]; then
  adb push "$MODEL_PATH" "$DEVICE_DIR/model.gguf"
else
  echo "Missing model: $MODEL_PATH (place it before pushing)" >&2
  exit 1
fi

echo "Pushed binaries and model to $DEVICE_DIR"
