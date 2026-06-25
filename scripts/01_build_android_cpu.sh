#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="$ROOT_DIR/third_party/llama.cpp"
BUILD_DIR="$ROOT_DIR/build-android"

: "${ANDROID_NDK_HOME:?ANDROID_NDK_HOME is not set. Run scripts/00_env_check.sh first.}"
: "${ANDROID_ABI:=arm64-v8a}"
: "${ANDROID_PLATFORM:=android-24}"

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
  echo "third_party/llama.cpp not found. Add it as a git submodule first:" >&2
  echo "  git submodule add <url> third_party/llama.cpp" >&2
  exit 1
fi

cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" --target llama-cli llama-bench

echo "== build artifacts =="
find "$BUILD_DIR/bin" -maxdepth 1 -type f 2>/dev/null
