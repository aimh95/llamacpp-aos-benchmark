#!/usr/bin/env bash
# llama.cpp Snapdragon(Hexagon HTP + Adreno OpenCL) Android 패키지 빌드.
# llama.cpp 소스 루트에서 실행한다고 가정한다 (--src-dir로 override 가능).
# 공식 절차: docs/backend/snapdragon/README.md 참고.
# 실패 시 즉시 중단한다 (probe성 도구와 달리, 빌드는 이전 단계가 실패하면 다음 단계가 의미 없음).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC_DIR=""
PRESET="arm64-android-snapdragon-release"
BUILD_DIR="build-snapdragon"
PKG_PREFIX="pkg-snapdragon/llama.cpp"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]
  --src-dir <path>   llama.cpp 소스 루트 (default: 현재 디렉터리, 없으면 third_party/llama.cpp로 fallback)
  --preset <name>    cmake preset 이름 (default: $PRESET)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir) SRC_DIR="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[WARN] unknown argument: $1 (ignored)" >&2; shift ;;
  esac
done

if [[ -z "$SRC_DIR" ]]; then
  SRC_DIR="$(pwd)"
  if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
    if [[ -f "$ROOT_DIR/third_party/llama.cpp/CMakeLists.txt" ]]; then
      echo "[INFO] $SRC_DIR looks like it isn't a llama.cpp source root; falling back to submodule: $ROOT_DIR/third_party/llama.cpp" >&2
      SRC_DIR="$ROOT_DIR/third_party/llama.cpp"
    else
      echo "[ERROR] $SRC_DIR doesn't look like a llama.cpp source root (CMakeLists.txt not found)." >&2
      echo "        Run this script from the llama.cpp checkout, or pass --src-dir <path>." >&2
      exit 1
    fi
  fi
fi
SRC_DIR="$(cd "$SRC_DIR" && pwd)"

PRESET_FILE="$SRC_DIR/docs/backend/snapdragon/CMakeUserPresets.json"
if [[ ! -f "$PRESET_FILE" ]]; then
  echo "[ERROR] $PRESET_FILE not found." >&2
  echo "        이 llama.cpp 체크아웃에는 Snapdragon backend 빌드 preset이 없습니다." >&2
  echo "        third_party/llama.cpp 서브모듈이 Snapdragon HTP 백엔드를 지원하는 버전인지 확인하세요" >&2
  echo "        (docs/backend/snapdragon/README.md 자체가 없는 오래된 체크아웃일 수 있습니다)." >&2
  exit 1
fi

if [[ ! -f /.dockerenv ]]; then
  echo "[WARN] /.dockerenv 가 보이지 않습니다. 이 빌드는 Docker 컨테이너 밖에서 실행 중일 수 있습니다." >&2
fi

cat <<'EOF'
== Snapdragon llama.cpp build ==
권장 경로: 공식 Snapdragon toolchain Docker image 사용 (Android NDK / OpenCL SDK / Hexagon SDK 포함)
  docker run -it -u $(id -u):$(id -g) --volume <llama.cpp-src>:/workspace \
    --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.7
  [d]/> cd /workspace
이 스크립트는 위 컨테이너 안에서, llama.cpp 소스 루트(/workspace)에서 실행하는 것을 전제로 한다.
(참고: third_party/llama.cpp/docs/backend/snapdragon/README.md)
EOF

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/build_snapdragon/build_$TIMESTAMP"
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"
echo "src_dir=$SRC_DIR"
echo "preset=$PRESET"

{
  echo "src_dir=$SRC_DIR"
  echo "preset=$PRESET"
  echo "build_dir=$BUILD_DIR"
  echo "pkg_prefix=$PKG_PREFIX"
} > "$OUT_DIR/run_params.txt"

cd "$SRC_DIR"

echo "[1/4] cp docs/backend/snapdragon/CMakeUserPresets.json ."
if ! cp docs/backend/snapdragon/CMakeUserPresets.json .; then
  echo "[ERROR] failed to copy CMakeUserPresets.json into $SRC_DIR" >&2
  exit 1
fi

echo "[2/4] cmake --preset $PRESET -B $BUILD_DIR"
if ! cmake --preset "$PRESET" -B "$BUILD_DIR" > "$OUT_DIR/configure.log" 2>&1; then
  echo "[ERROR] cmake configure failed, see $OUT_DIR/configure.log" >&2
  exit 1
fi

{
  for marker_name in "GGML_HEXAGON" "GGML_OPENCL"; do
    if grep -Eq "${marker_name} *= *\"?ON\"?" "$OUT_DIR/configure.log"; then
      echo "${marker_name}=ON: FOUND"
    else
      echo "${marker_name}=ON: NOT_FOUND"
    fi
  done
  for marker_text in "Including Hexagon backend" "Including OpenCL backend"; do
    if grep -qF "$marker_text" "$OUT_DIR/configure.log"; then
      echo "$marker_text: FOUND"
    else
      echo "$marker_text: NOT_FOUND"
    fi
  done
} | tee "$OUT_DIR/configure_markers.txt" >&2

echo "[3/4] cmake --build $BUILD_DIR"
if ! cmake --build "$BUILD_DIR" > "$OUT_DIR/build.log" 2>&1; then
  echo "[ERROR] cmake build failed, see $OUT_DIR/build.log" >&2
  exit 1
fi

echo "[4/4] cmake --install $BUILD_DIR --prefix $PKG_PREFIX"
if ! cmake --install "$BUILD_DIR" --prefix "$PKG_PREFIX" > "$OUT_DIR/install.log" 2>&1; then
  echo "[ERROR] cmake install failed, see $OUT_DIR/install.log" >&2
  exit 1
fi

# 부가: scripts/snapdragon (run-tool.sh 등 host-side adb wrapper)을 패키지에 동봉한다.
# cmake --install은 이 디렉터리를 설치하지 않으므로, check_snapdragon_pkg.sh의 권장 체크 항목과
# install_snapdragon_pkg.sh의 wrapper 우선 실행을 위해 별도로 복사한다 (실패해도 빌드 자체는 완료된 것으로 본다).
if [[ -d "$SRC_DIR/scripts/snapdragon" ]]; then
  if cp -R "$SRC_DIR/scripts/snapdragon" "$PKG_PREFIX/scripts/snapdragon" 2>>"$OUT_DIR/install.log"; then
    echo "[INFO] copied scripts/snapdragon into $PKG_PREFIX/scripts/snapdragon"
  else
    echo "[WARN] failed to copy scripts/snapdragon into package (non-fatal), see $OUT_DIR/install.log" >&2
  fi
fi

PKG_ABS_PATH="$SRC_DIR/$PKG_PREFIX"
echo "$PKG_ABS_PATH" > "$OUT_DIR/pkg_path.txt"

echo ""
echo "Build complete."
echo "Package: $PKG_ABS_PATH"
echo "Logs: $OUT_DIR"
echo ""
echo "Next: tools/build_snapdragon/check_snapdragon_pkg.sh --pkg-dir \"$PKG_ABS_PATH\""
