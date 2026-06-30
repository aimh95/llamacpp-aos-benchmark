#!/usr/bin/env bash
# pkg-snapdragon/llama.cpp 패키지를 단말의 /data/local/tmp/llama.cpp 로 설치한다.
# 기존 /data/local/tmp/llamacpp_cpu(CPU-only 패키지)는 건드리지 않는다.
# root 불필요. 실패해도 끝까지 진행하고 로그를 남긴다 (디버깅용).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 단말 측 설치 경로는 고정한다 (기존 CPU-only 패키지와 분리하기 위함). 변경 불가.
readonly REMOTE_PARENT="/data/local/tmp"
readonly REMOTE_DIR="/data/local/tmp/llama.cpp"
readonly REMOTE_GGUF_DIR="/data/local/tmp/gguf"

SERIAL="${DEVICE_SERIAL:-}"
PKG_DIR="pkg-snapdragon/llama.cpp"
MODEL_PATH=""

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]
  --serial <serial>   adb device serial (default: \$DEVICE_SERIAL env or unset)
  --pkg-dir <path>     로컬 패키지 경로 (default: $PKG_DIR). basename은 반드시 "llama.cpp" 여야
                       $REMOTE_DIR 로 정확히 설치된다.
  --model <path>       단말에 같이 push할 로컬 gguf 모델 경로 (선택)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) SERIAL="$2"; shift 2 ;;
    --pkg-dir) PKG_DIR="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[WARN] unknown argument: $1 (ignored)" >&2; shift ;;
  esac
done

if ! command -v adb >/dev/null 2>&1; then
  echo "[ERROR] adb not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$PKG_DIR" ]]; then
  echo "[ERROR] package directory not found: $PKG_DIR" >&2
  echo "        먼저 tools/build_snapdragon/build_snapdragon_llamacpp.sh 를 실행하세요." >&2
  exit 1
fi
PKG_DIR="$(cd "$PKG_DIR" && pwd)"

if [[ "$(basename "$PKG_DIR")" != "llama.cpp" ]]; then
  echo "[WARN] $PKG_DIR 의 basename이 'llama.cpp'가 아닙니다." >&2
  echo "       'adb push <dir> $REMOTE_PARENT/' 는 dir의 basename으로 nesting되므로," >&2
  echo "       결과가 $REMOTE_DIR 가 아닐 수 있습니다." >&2
fi

if [[ -n "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] model file not found: $MODEL_PATH" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/build_snapdragon/install_$TIMESTAMP"
mkdir -p "$OUT_DIR"

ADB_ARGS=()
if [[ -n "$SERIAL" ]]; then
  ADB_ARGS=(-s "$SERIAL")
fi

echo "== Snapdragon package install =="
echo "Output dir: $OUT_DIR"
echo "pkg_dir=$PKG_DIR remote_dir=$REMOTE_DIR model=${MODEL_PATH:-<none>}"

{
  echo "serial=${SERIAL:-<default>}"
  echo "pkg_dir=$PKG_DIR"
  echo "remote_dir=$REMOTE_DIR"
  echo "model_path=${MODEL_PATH:-<none>}"
} > "$OUT_DIR/run_params.txt"

if ! adb "${ADB_ARGS[@]}" devices -l > "$OUT_DIR/devices.txt" 2>&1; then
  echo "[WARN] 'adb devices -l' failed, see $OUT_DIR/devices.txt" >&2
fi

# 1) 패키지 push: 공식 절차와 동일하게 부모 디렉터리(REMOTE_PARENT)를 목적지로 지정해서
#    PKG_DIR(basename=llama.cpp)가 그 아래로 nesting 되도록 한다.
echo "[1/3] adb push $PKG_DIR $REMOTE_PARENT/"
if ! adb "${ADB_ARGS[@]}" push "$PKG_DIR" "$REMOTE_PARENT/" > "$OUT_DIR/push_pkg.log" 2>&1; then
  echo "[WARN] package push failed (continuing), see $OUT_DIR/push_pkg.log" >&2
fi

adb "${ADB_ARGS[@]}" shell "find $REMOTE_DIR/bin $REMOTE_DIR/lib -maxdepth 1 -type f 2>/dev/null" \
  > "$OUT_DIR/remote_listing.txt" 2>&1 || true

for exe in bin/llama-cli bin/llama-bench; do
  adb "${ADB_ARGS[@]}" shell "chmod +x $REMOTE_DIR/$exe" >> "$OUT_DIR/push_pkg.log" 2>&1 || true
done
adb "${ADB_ARGS[@]}" shell "chmod +x $REMOTE_DIR/scripts/snapdragon/adb/*.sh" >> "$OUT_DIR/push_pkg.log" 2>&1 || true

# 2) 모델 push (선택)
if [[ -n "$MODEL_PATH" ]]; then
  echo "[2/3] adb push $MODEL_PATH $REMOTE_GGUF_DIR/"
  adb "${ADB_ARGS[@]}" shell "mkdir -p $REMOTE_GGUF_DIR" > "$OUT_DIR/push_model.log" 2>&1 || true
  if ! adb "${ADB_ARGS[@]}" push "$MODEL_PATH" "$REMOTE_GGUF_DIR/" >> "$OUT_DIR/push_model.log" 2>&1; then
    echo "[WARN] model push failed (continuing), see $OUT_DIR/push_model.log" >&2
  fi
else
  echo "[2/3] no --model given, skipping model push"
  echo "skipped: no --model given" > "$OUT_DIR/push_model.log"
fi

# 3) 설치 검증: 필수 파일 존재 확인
echo "[3/3] verifying installed files on device"
VERIFY_PATHS=(
  "$REMOTE_DIR/bin/llama-cli"
  "$REMOTE_DIR/lib/libggml-hexagon.so"
  "$REMOTE_DIR/lib/libggml-htp-v75.so"
  "$REMOTE_DIR/lib/libggml-opencl.so"
)
VERIFY_MISSING=0
: > "$OUT_DIR/install_verify.txt"
for p in "${VERIFY_PATHS[@]}"; do
  STATUS="$(adb "${ADB_ARGS[@]}" shell "[ -f $p ] && echo FOUND || echo NOT_FOUND" 2>>"$OUT_DIR/install_verify.txt")"
  STATUS="$(echo "$STATUS" | tr -d '\r\n ')"
  [[ -z "$STATUS" ]] && STATUS="UNKNOWN"
  echo "$p $STATUS" >> "$OUT_DIR/install_verify.txt"
  echo "  $p -> $STATUS"
  if [[ "$STATUS" != "FOUND" ]]; then
    VERIFY_MISSING=$((VERIFY_MISSING + 1))
  fi
done

# 4) --list-devices 시도: wrapper(run-tool.sh, 있으면) + 직접 실행 둘 다 시도하고 둘 다 기록한다.
WRAPPER="$PKG_DIR/scripts/snapdragon/adb/run-tool.sh"
if [[ -f "$WRAPPER" ]]; then
  echo "[INFO] trying wrapper: $WRAPPER llama-cli --list-devices"
  if [[ -n "$SERIAL" ]]; then
    S="$SERIAL" "$WRAPPER" llama-cli --list-devices > "$OUT_DIR/list_devices_wrapper.log" 2>&1
  else
    "$WRAPPER" llama-cli --list-devices > "$OUT_DIR/list_devices_wrapper.log" 2>&1
  fi || echo "[WARN] wrapper --list-devices failed (continuing), see $OUT_DIR/list_devices_wrapper.log" >&2
else
  echo "wrapper not found at $WRAPPER" > "$OUT_DIR/list_devices_wrapper.log"
fi

echo "[INFO] trying direct: $REMOTE_DIR/bin/llama-cli --list-devices"
DIRECT_CMD="cd $REMOTE_DIR && LD_LIBRARY_PATH=$REMOTE_DIR/lib ADSP_LIBRARY_PATH=$REMOTE_DIR/lib ./bin/llama-cli --list-devices"
if ! adb "${ADB_ARGS[@]}" shell "$DIRECT_CMD" > "$OUT_DIR/list_devices_direct.log" 2>&1; then
  echo "[WARN] direct --list-devices failed (continuing), see $OUT_DIR/list_devices_direct.log" >&2
fi

echo ""
if [[ "$VERIFY_MISSING" -eq 0 ]]; then
  echo "Install verify: PASS (모든 필수 파일 발견)"
else
  echo "Install verify: FAIL ($VERIFY_MISSING/${#VERIFY_PATHS[@]} 누락)"
fi
echo "Logs: $OUT_DIR"
echo "HTP0 visibility는 list_devices_wrapper.log / list_devices_direct.log 를 직접 확인하세요."
echo "(HTP V75는 파일명 기반 candidate입니다. 실제 동작 확인은 tools/htp_smoke 를 사용하세요.)"

if [[ "$VERIFY_MISSING" -eq 0 ]]; then
  exit 0
else
  exit 1
fi
