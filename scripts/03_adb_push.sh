cat > scripts/03_adb_push.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="/data/local/tmp/llama"

CLI="$ROOT_DIR/build-android/bin/llama-cli"
BENCH="$ROOT_DIR/build-android/bin/llama-bench"
MODEL="$ROOT_DIR/models/model.gguf"

if [ ! -f "$CLI" ]; then
  echo "ERROR: llama-cli not found. Run scripts/01_build_android_cpu.sh first."
  exit 1
fi

if [ ! -f "$BENCH" ]; then
  echo "ERROR: llama-bench not found. Run scripts/01_build_android_cpu.sh first."
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "ERROR: model.gguf not found. Run scripts/02_download_model.sh first."
  exit 1
fi

adb shell mkdir -p "$DEVICE_DIR"
adb push "$CLI" "$DEVICE_DIR/"
adb push "$BENCH" "$DEVICE_DIR/"
adb push "$MODEL" "$DEVICE_DIR/model.gguf"

adb shell chmod +x "$DEVICE_DIR/llama-cli"
adb shell chmod +x "$DEVICE_DIR/llama-bench"

echo
echo "Files on device:"
adb shell "ls -lh $DEVICE_DIR"
EOF

chmod +x scripts/03_adb_push.sh