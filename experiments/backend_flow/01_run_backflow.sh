#!/usr/bin/env bash
# BACKFLOW 실험 실행: decode step backend-flow 분석
# DUMP_DECODE_BACKEND_FLOW=1 DUMP_PREFILL_BACKEND_FLOW=1
# DUMP_BACKEND_COPY=1 DUMP_GET_ROWS_TRACE=1
# 실행 위치: llamacpp-aos-benchmark 루트
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIAL=""
MODEL_NAME="${MODEL_NAME:-exaone4_q8_0_tied_on}"  # 기본: tied_on (token_embd TENSOR_DUPLICATED)
NGL=99
CTX=2048
N_PREDICT=8      # decode steps 수 (timing용)
DEVICE="HTP0"
REMOTE_DIR="/data/local/tmp/llama.cpp"
REMOTE_GGUF="/data/local/tmp/gguf/${MODEL_NAME}.gguf"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --serial <adb_serial> [options]
  --serial <s>      adb device serial (required)
  --model  <m>      model name under /data/local/tmp/gguf/ (default: $MODEL_NAME)
  --device <d>      backend device (default: $DEVICE)
  --n-predict <n>   decode steps for timing (default: $N_PREDICT)
  --ngl    <n>      gpu layers (default: $NGL)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)    SERIAL="$2";    shift 2 ;;
    --model)     MODEL_NAME="$2"; REMOTE_GGUF="/data/local/tmp/gguf/${MODEL_NAME}.gguf"; shift 2 ;;
    --device)    DEVICE="$2";    shift 2 ;;
    --n-predict) N_PREDICT="$2"; shift 2 ;;
    --ngl)       NGL="$2";       shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "[WARN] unknown arg: $1" >&2; shift ;;
  esac
done

if [[ -z "$SERIAL" ]]; then
  echo "[ERROR] --serial required" >&2; usage; exit 1
fi

ADB="adb -s $SERIAL"
OUT_DIR="$SCRIPT_DIR"
REMOTE_LOG="$REMOTE_DIR/backflow.log"
REMOTE_FLOW_DIR="$REMOTE_DIR/experiments/backend_flow"

echo "=== BACKFLOW experiment ==="
echo "serial : $SERIAL"
echo "model  : $MODEL_NAME  ($REMOTE_GGUF)"
echo "device : $DEVICE"
echo "ngl    : $NGL"
echo "n_pred : $N_PREDICT"
echo ""

$ADB shell echo "device OK" >/dev/null || { echo "[ERROR] device not found"; exit 1; }
$ADB shell "mkdir -p $REMOTE_FLOW_DIR"

# Check model exists on device
$ADB shell "ls $REMOTE_GGUF" >/dev/null 2>&1 \
  || { echo "[ERROR] $REMOTE_GGUF not on device — run 02_run_experiment.sh first to push models"; exit 1; }

echo "[1] Running llama-completion with all BACKFLOW trace flags ..."
$ADB shell "
  cd $REMOTE_DIR && \
  rm -f $REMOTE_FLOW_DIR/*.txt $REMOTE_FLOW_DIR/*.csv && \
  mkdir -p $REMOTE_FLOW_DIR && \
  LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
  DUMP_DECODE_BACKEND_FLOW=1 \
  DUMP_PREFILL_BACKEND_FLOW=1 \
  DUMP_BACKEND_COPY=1 \
  DUMP_GET_ROWS_TRACE=1 \
  ./bin/llama-completion \
      -m $REMOTE_GGUF \
      -p 'Explain what EXAONE is in one sentence.' \
      -n $N_PREDICT \
      -c $CTX \
      -ngl $NGL \
      -dev $DEVICE \
      -v \
      --log-file $REMOTE_LOG \
      </dev/null \
      2>&1 | tail -5
" || echo "[WARN] inference returned non-zero — logs still collected"

echo ""
echo "[2] Pulling BACKFLOW artifacts ..."
for f in prefill_graph_nodes.txt prefill_schedule.txt \
          decode_graph_nodes.txt  decode_schedule.txt \
          get_rows_trace.txt      backend_copy_trace.txt \
          decode_timing.csv; do
  $ADB pull "$REMOTE_FLOW_DIR/$f" "$OUT_DIR/$f" 2>/dev/null \
    && echo "  pulled: $f" \
    || echo "  [WARN] not found: $f"
done
$ADB pull "$REMOTE_LOG" "$OUT_DIR/backflow_raw.log" 2>/dev/null \
  && echo "  pulled: backflow_raw.log" || true

echo ""
echo "[3] Parsing results ..."
python3 "$SCRIPT_DIR/parse_backflow.py" --dir "$OUT_DIR" \
  && echo "  summary.md written" \
  || echo "[WARN] parse_backflow.py failed"

echo ""
echo "=== Done ==="
echo "Results: $OUT_DIR"
echo "Summary: $OUT_DIR/summary.md"
