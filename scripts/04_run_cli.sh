#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="/data/local/tmp/llamacpp_cpu"

adb shell "cd $DEVICE_DIR && ./llama-cli \
  -m model.gguf \
  -p 'Hello. Explain what llama.cpp is in one sentence.' \
  -n 64 \
  -c 512 \
  -t 4" | tee "logs/llama-cli_$(date +%Y%m%d_%H%M%S).log"
