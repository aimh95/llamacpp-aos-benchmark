#!/usr/bin/env bash
set -euo pipefail

: "${DEVICE_DIR:=/data/local/tmp/llamacpp-bench}"
: "${THREADS:=4}"
: "${PROMPT:=Hello, how are you?}"

adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./llama-cli \
  -m model.gguf \
  -t $THREADS \
  -n 64 \
  -p \"$PROMPT\" \
  --no-conversation"
