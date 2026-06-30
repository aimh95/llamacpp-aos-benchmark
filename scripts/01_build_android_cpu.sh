#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
BUILD_DIR="$ROOT_DIR/build-android"

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
  echo "ERROR: ANDROID_NDK_HOME is not set"
  exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "Build done:"
ls -lh "$BUILD_DIR/bin/llama-cli" "$BUILD_DIR/bin/llama-bench"
