#!/usr/bin/env bash
# llama.cpp Snapdragon/Hexagon HTP0 backend smoke test.
# host에서 실행. root 불필요. CPU/GPU 벤치마크는 다루지 않는다 (HTP0 smoke 전용).
# 실패해도 끝까지 진행하고, 실행 exit code는 exit_code.txt에 기록 + 이 스크립트의 최종 exit code로 전달한다.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIAL="${DEVICE_SERIAL:-}"
REMOTE_DIR="/data/local/tmp/llama.cpp"
MODEL_PATH="models/llama-3.2-1b-instruct-q4_0.gguf"
BACKEND_DEVICE="HTP0"
PROMPT="Explain on-device AI in three short sentences."
N_PREDICT=64
CTX_SIZE=512
NGL=99

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]
  --serial <serial>          adb device serial (default: \$DEVICE_SERIAL env or unset)
  --remote-dir <dir>         단말 내 llama.cpp 실행 디렉토리 (default: $REMOTE_DIR)
  --model <path>             모델 경로, remote-dir 기준 상대경로 가능 (default: $MODEL_PATH)
  --backend-device <name>    --device 로 넘길 backend device 이름 (default: $BACKEND_DEVICE)
  --prompt <text>            prompt (default: "$PROMPT")
  --n-predict <N>            생성 토큰 수 (default: $N_PREDICT)
  --ctx-size <N>             context size (default: $CTX_SIZE)
  --ngl <N>                  -ngl 값 (default: $NGL)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) SERIAL="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --backend-device) BACKEND_DEVICE="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --n-predict) N_PREDICT="$2"; shift 2 ;;
    --ctx-size) CTX_SIZE="$2"; shift 2 ;;
    --ngl) NGL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[WARN] unknown argument: $1 (ignored)" >&2; shift ;;
  esac
done

if ! command -v adb >/dev/null 2>&1; then
  echo "[ERROR] adb not found in PATH" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found in PATH" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/htp_smoke/$TIMESTAMP"
mkdir -p "$OUT_DIR"

ADB_ARGS=()
if [[ -n "$SERIAL" ]]; then
  ADB_ARGS=(-s "$SERIAL")
fi

echo "== HTP0 smoke test =="
echo "Output dir: $OUT_DIR"
echo "remote-dir=$REMOTE_DIR model=$MODEL_PATH device=$BACKEND_DEVICE n-predict=$N_PREDICT ctx-size=$CTX_SIZE ngl=$NGL"

{
  echo "serial=${SERIAL:-<default>}"
  echo "remote_dir=$REMOTE_DIR"
  echo "model_path=$MODEL_PATH"
  echo "backend_device=$BACKEND_DEVICE"
  echo "prompt=$PROMPT"
  echo "n_predict=$N_PREDICT"
  echo "ctx_size=$CTX_SIZE"
  echo "ngl=$NGL"
} > "$OUT_DIR/run_params.txt"

if ! adb "${ADB_ARGS[@]}" devices -l > "$OUT_DIR/devices.txt" 2>&1; then
  echo "[WARN] 'adb devices -l' failed, see $OUT_DIR/devices.txt" >&2
fi

# wrapper script(scripts/llama-cli.sh)가 있으면 우선 사용, 없으면 llama-cli 직접 실행
WRAPPER_REL="scripts/llama-cli.sh"
HAS_WRAPPER="no"
if adb "${ADB_ARGS[@]}" shell "[ -f $REMOTE_DIR/$WRAPPER_REL ] && echo yes || echo no" > "$OUT_DIR/wrapper_check.txt" 2>&1; then
  HAS_WRAPPER="$(tr -d '\r\n ' < "$OUT_DIR/wrapper_check.txt")"
else
  echo "[WARN] wrapper check failed via adb shell, assuming no wrapper" >&2
fi

if [[ "$HAS_WRAPPER" == "yes" ]]; then
  RUN_BIN="./$WRAPPER_REL"
  echo "[INFO] using wrapper: $RUN_BIN"
else
  RUN_BIN="./llama-cli"
  echo "[INFO] wrapper not found, using direct binary: $RUN_BIN"
fi
echo "run_bin=$RUN_BIN" >> "$OUT_DIR/run_params.txt"

# 1) list devices
LIST_CMD="cd $REMOTE_DIR && $RUN_BIN --list-devices"
if ! adb "${ADB_ARGS[@]}" shell "$LIST_CMD" > "$OUT_DIR/list_devices.log" 2>&1; then
  echo "[WARN] --list-devices failed (continuing), see $OUT_DIR/list_devices.log" >&2
fi

# 2) QNN/HTP 관련 라이브러리 재수집 (실행 직전 snapshot). permission denied는 무시.
SEARCH_PATHS="/vendor /system /product /odm /apex /data/local/tmp"
FIND_PATTERNS='-iname "libQnn*.so" -o -iname "*Qnn*.so" -o -iname "libQnnHtp*.so" -o -iname "*Htp*Skel*.so" -o -iname "*Htp*Stub*.so" -o -iname "libcdsprpc.so" -o -iname "libOpenCL.so"'
QNN_LIBS_FILE="$OUT_DIR/qnn_libs_before_run.txt"
: > "$QNN_LIBS_FILE"
for p in $SEARCH_PATHS; do
  {
    echo "== $p =="
    adb "${ADB_ARGS[@]}" shell "find $p $FIND_PATTERNS 2>/dev/null"
  } >> "$QNN_LIBS_FILE" 2>>"$OUT_DIR/search_errors.txt" || true
done

# 3) 실행 직전 logcat 버퍼 정리 (best-effort). 이번 실행과 무관한 과거 로그를 줄이기 위함.
if ! adb "${ADB_ARGS[@]}" logcat -c >/dev/null 2>&1; then
  echo "[WARN] 'adb logcat -c' failed (continuing)" >&2
fi

# 4) HTP0 smoke run
RUN_CMD="cd $REMOTE_DIR && $RUN_BIN -m \"$MODEL_PATH\" --device \"$BACKEND_DEVICE\" -p \"$PROMPT\" -n $N_PREDICT -c $CTX_SIZE -ngl $NGL"
echo "run_cmd=$RUN_CMD" >> "$OUT_DIR/run_params.txt"

adb "${ADB_ARGS[@]}" shell "$RUN_CMD" > "$OUT_DIR/htp0_smoke.log" 2>&1
RUN_EXIT_CODE=$?
echo "$RUN_EXIT_CODE" > "$OUT_DIR/exit_code.txt"
if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
  echo "[WARN] HTP0 run exited with code $RUN_EXIT_CODE (continuing), see $OUT_DIR/htp0_smoke.log" >&2
fi

# 5) logcat에서 관련 키워드만 추출 (실행 후 dump, 매치 없으면 grep이 1을 반환하지만 무시한다)
adb "${ADB_ARGS[@]}" logcat -d 2>"$OUT_DIR/logcat_errors.txt" \
  | grep -iE 'qnn|htp|hexagon|cdsp|adsprpc|rpc|ggml|llama' > "$OUT_DIR/logcat_qnn_htp.txt"
true

echo "Raw collection complete (exit_code=$RUN_EXIT_CODE)."

python3 "$SCRIPT_DIR/parse_htp_smoke_log.py" "$OUT_DIR"

echo "Report written to:"
echo "  $OUT_DIR/summary.md"
echo "  $OUT_DIR/summary.csv"

exit "$RUN_EXIT_CODE"
