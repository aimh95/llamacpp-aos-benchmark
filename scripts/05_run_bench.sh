#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/bench_${TIMESTAMP}.log"

: "${DEVICE_DIR:=/data/local/tmp/llamacpp-bench}"
: "${THREADS:=4}"
: "${PP:=512}"
: "${TG:=128}"

mkdir -p "$LOG_DIR"

adb shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR ./llama-bench \
  -m model.gguf \
  -t $THREADS \
  -p $PP \
  -n $TG" | tee "$LOG_FILE"

echo "Saved benchmark log to $LOG_FILE"
