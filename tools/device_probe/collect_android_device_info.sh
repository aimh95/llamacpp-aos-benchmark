#!/usr/bin/env bash
# Android 단말 SoC / ABI / QNN-HTP 라이브러리 존재 여부 probe.
# root 불필요. adb shell 명령이 실패해도 스크립트는 중단되지 않고 계속 진행한다.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/device_probe/$TIMESTAMP"
mkdir -p "$OUT_DIR"

ADB_ARGS=()
if [[ -n "${DEVICE_SERIAL:-}" ]]; then
  ADB_ARGS=(-s "$DEVICE_SERIAL")
fi

echo "== Android Device Probe =="
echo "Output dir: $OUT_DIR"

if ! command -v adb >/dev/null 2>&1; then
  echo "[ERROR] adb not found in PATH" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found in PATH" >&2
  exit 1
fi

if ! adb "${ADB_ARGS[@]}" devices -l > "$OUT_DIR/devices.txt" 2>&1; then
  echo "[WARN] 'adb devices -l' failed, see $OUT_DIR/devices.txt" >&2
fi

DEVICE_COUNT="$(grep -E '^[A-Za-z0-9._:-]+[[:space:]]+device([[:space:]]|$)' "$OUT_DIR/devices.txt" 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${DEVICE_COUNT:-0}" -eq 0 ]]; then
  echo "[WARN] no connected device found in 'adb devices -l' output" >&2
elif [[ "${DEVICE_COUNT:-0}" -gt 1 && -z "${DEVICE_SERIAL:-}" ]]; then
  echo "[WARN] multiple devices connected; set DEVICE_SERIAL=<serial> to target one" >&2
fi

run_shell() {
  # run_shell <remote command> <outfile>
  local cmd="$1"
  local outfile="$2"
  if ! adb "${ADB_ARGS[@]}" shell "$cmd" > "$outfile" 2>&1; then
    echo "[WARN] command failed (continuing): $cmd" >&2
  fi
}

run_shell "getprop" "$OUT_DIR/getprop.txt"
run_shell "cat /proc/cpuinfo" "$OUT_DIR/cpuinfo.txt"
run_shell "cat /proc/meminfo" "$OUT_DIR/meminfo.txt"
run_shell "uname -a" "$OUT_DIR/uname.txt"
run_shell "pm list features" "$OUT_DIR/pm_features.txt"

{
  echo "ANDROID_HOME=${ANDROID_HOME:-NOT_SET}"
  echo "ANDROID_NDK_HOME=${ANDROID_NDK_HOME:-NOT_SET}"
  echo "ANDROID_NDK=${ANDROID_NDK:-NOT_SET}"
  echo "QNN_SDK_ROOT=${QNN_SDK_ROOT:-NOT_SET}"
  echo "QAIRT_SDK_ROOT=${QAIRT_SDK_ROOT:-NOT_SET}"
} > "$OUT_DIR/host_env.txt"

# QNN / HTP / OpenCL / RPC 관련 라이브러리 검색.
# permission denied(2>/dev/null)는 무시하고 접근 가능한 경로만 수집한다.
SEARCH_PATHS="/vendor /system /product /odm /apex /data/local/tmp"
FIND_PATTERNS='-iname "libQnn*.so" -o -iname "*Qnn*.so" -o -iname "libQnnHtp*.so" -o -iname "*Htp*Skel*.so" -o -iname "*Htp*Stub*.so" -o -iname "libcdsprpc.so" -o -iname "libOpenCL.so"'

QNN_LIBS_FILE="$OUT_DIR/qnn_libs.txt"
: > "$QNN_LIBS_FILE"

for p in $SEARCH_PATHS; do
  {
    echo "== $p =="
    adb "${ADB_ARGS[@]}" shell "find $p $FIND_PATTERNS 2>/dev/null"
  } >> "$QNN_LIBS_FILE" 2>>"$OUT_DIR/search_errors.txt" || true
done

echo "Raw collection complete."

python3 "$SCRIPT_DIR/parse_device_info.py" "$OUT_DIR"

echo "Report written to:"
echo "  $OUT_DIR/device_info.md"
echo "  $OUT_DIR/device_info.csv"
