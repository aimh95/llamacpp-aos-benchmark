#!/usr/bin/env bash
# Snapdragon llama.cpp 빌드를 공식 Docker 이미지 안에서 실행하는 wrapper.
# 호스트에서 이 스크립트를 실행하면 컨테이너 안에서 build_snapdragon_llamacpp.sh를 호출한다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
INNER_SCRIPT="/home/iptv-infra/workspace/llamacpp-aos-benchmark/tools/build_snapdragon/build_snapdragon_llamacpp.sh"

# 추가 인자는 build_snapdragon_llamacpp.sh 에 그대로 전달
EXTRA_ARGS=("$@")

echo "[docker-build] image : $IMAGE"
echo "[docker-build] repo  : $ROOT_DIR"
echo "[docker-build] script: $INNER_SCRIPT"
echo ""

# 호스트 경로를 컨테이너 안에도 동일 경로로 마운트 → build_snapdragon_llamacpp.sh의
# ROOT_DIR / artifact 경로가 컨테이너 안에서도 그대로 동작한다.
docker run --rm \
  --platform linux/amd64 \
  -u "$(id -u):$(id -g)" \
  --volume "$ROOT_DIR:$ROOT_DIR" \
  --workdir "$ROOT_DIR" \
  "$IMAGE" \
  bash "$INNER_SCRIPT" "${EXTRA_ARGS[@]}"
