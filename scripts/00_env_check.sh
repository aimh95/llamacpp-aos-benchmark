#!/usr/bin/env bash
set -euo pipefail

echo "== llamacpp-android-cpu-baseline env check =="

check() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[OK] $name -> $(command -v "$name")"
  else
    echo "[MISSING] $name"
  fi
}

check git
check cmake
check ninja
check adb
check python3

if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
  echo "[OK] ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
else
  echo "[MISSING] ANDROID_NDK_HOME is not set"
fi

echo "== connected devices =="
adb devices || true
