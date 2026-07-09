#!/usr/bin/env bash
# EXAONE 4.0 1.2B tied embedding 실험 실행.
# 각 variant(tied_on / tied_off)를 디바이스에서 실행하고 결과를 수집한다.
# 실행 위치: llamacpp-aos-benchmark 루트
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ------------------------------------------------------------------ #
# CLI 인자
# ------------------------------------------------------------------ #
SERIAL=""
QUANT_TYPE="${QUANT_TYPE:-Q8_0}"
N_PREDICT=32
CTX_SIZE=2048
NGL=99
DEVICE="HTP0"
VARIANTS="tied_on tied_off"  # space-separated; set to "tied_on" to skip tied_off

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --serial <adb_serial> [options]
  --serial <s>      adb device serial (required)
  --quant  <q>      quantization type (default: $QUANT_TYPE)
  --variants <v>    "tied_on tied_off" or "tied_on" (default: both)
  --device  <d>     backend device (default: $DEVICE)
  --n-predict <n>   max new tokens (default: $N_PREDICT)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)   SERIAL="$2";   shift 2 ;;
    --quant)    QUANT_TYPE="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --device)   DEVICE="$2";   shift 2 ;;
    --n-predict) N_PREDICT="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "[WARN] unknown arg: $1" >&2; shift ;;
  esac
done

if [[ -z "$SERIAL" ]]; then
  echo "[ERROR] --serial required" >&2
  usage; exit 1
fi

REMOTE_DIR="/data/local/tmp/llama.cpp"
REMOTE_GGUF_DIR="/data/local/tmp/gguf"
REMOTE_GRAPH_DIR="$REMOTE_DIR/experiments/graph_dump"

qt_lower="${QUANT_TYPE,,}"
GGUF_TIED_ON="$ROOT_DIR/models/exaone4_1p2b_${qt_lower}_tied_on.gguf"
GGUF_TIED_OFF="$ROOT_DIR/models/exaone4_1p2b_${qt_lower}_tied_off.gguf"

ADB="adb -s $SERIAL"

echo "=== EXAONE 4.0 1.2B tied-embedding experiment ==="
echo "serial   : $SERIAL"
echo "device   : $DEVICE"
echo "quant    : $QUANT_TYPE"
echo "variants : $VARIANTS"
echo ""

# Verify device connection
$ADB shell echo "device OK" >/dev/null || { echo "[ERROR] device $SERIAL not found"; exit 1; }

# Ensure remote dirs exist
$ADB shell "mkdir -p $REMOTE_GRAPH_DIR $REMOTE_GGUF_DIR"

# ------------------------------------------------------------------ #
# Helper: run one variant
# ------------------------------------------------------------------ #
run_variant() {
    local variant="$1"    # tied_on | tied_off
    local gguf_path="$2"
    local out_dir="$SCRIPT_DIR/$variant"

    echo ""
    echo "--- $variant ---"

    if [[ ! -f "$gguf_path" ]]; then
        echo "[SKIP] $gguf_path not found — run 01_download_convert.sh first"
        echo "STATUS: SKIPPED (model file not found)" > "$out_dir/run_status.txt"
        return 0
    fi

    local remote_gguf="$REMOTE_GGUF_DIR/exaone4_${qt_lower}_${variant}.gguf"

    # 1. Push model
    echo "[1] Pushing $gguf_path → $remote_gguf ..."
    $ADB push "$gguf_path" "$remote_gguf"

    # 2. Run inference with all trace env vars
    #    - DUMP_GGML_GRAPH=1  → $REMOTE_GRAPH_DIR/graph_nodes.txt
    #    - GGML_OP_TRACE=1    → [OPTRACE] lines in log
    #    - GGML_EMB_TRACE=1   → [EMBTRACE] lines in log
    #    - --log-file          → captures all LLAMA_LOG lines
    local remote_log="$REMOTE_DIR/exp_${variant}.log"
    echo "[2] Running llama-completion (all trace flags) ..."
    $ADB shell "
        cd $REMOTE_DIR && \
        rm -f $REMOTE_GRAPH_DIR/graph_nodes.txt $REMOTE_GRAPH_DIR/graph.dot && \
        mkdir -p $REMOTE_GRAPH_DIR && \
        LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
        DUMP_GGML_GRAPH=1 GGML_OP_TRACE=1 GGML_EMB_TRACE=1 \
        ./bin/llama-completion \
            -m $remote_gguf \
            -p 'Explain what EXAONE 4.0 is in exactly one sentence.' \
            -n $N_PREDICT \
            -c $CTX_SIZE \
            -ngl $NGL \
            -dev $DEVICE \
            -v \
            --log-file $remote_log \
            2>&1 | tail -3
    " || echo "[WARN] inference exit non-zero — log still collected"

    # 3. Pull trace log
    echo "[3] Pulling logs ..."
    $ADB pull "$remote_log" "$out_dir/htp_optrace.log" 2>/dev/null \
        || echo "[WARN] trace log not found on device"

    # 4. Pull graph dump files
    $ADB pull "$REMOTE_GRAPH_DIR/graph_nodes.txt" "$out_dir/graph_nodes.txt" 2>/dev/null \
        || echo "[WARN] graph_nodes.txt not found — DUMP_GGML_GRAPH may not have fired"
    $ADB pull "$REMOTE_GRAPH_DIR/graph.dot" "$out_dir/graph.dot" 2>/dev/null \
        || true

    # 5. Extract structured artifacts from the trace log
    if [[ -f "$out_dir/htp_optrace.log" ]]; then
        grep -E "\[GRAPHDUMP\]" "$out_dir/htp_optrace.log" \
            > "$out_dir/graphdump_lines.txt" 2>/dev/null || true
        grep -E "\[OPTRACE\]"   "$out_dir/htp_optrace.log" \
            > "$out_dir/graph_schedule.txt" 2>/dev/null || true
        grep -E "\[EMBTRACE\]"  "$out_dir/htp_optrace.log" \
            > "$out_dir/buffer_placement.txt" 2>/dev/null || true
        echo "[INFO] extracted graph_schedule.txt, buffer_placement.txt"
    fi

    # 6. Run llama-bench for performance (JSON)
    echo "[4] Running llama-bench ..."
    $ADB shell "
        cd $REMOTE_DIR && \
        LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
        ADSP_LIBRARY_PATH=lib \
        ./bin/llama-bench \
            -m $remote_gguf \
            -p 255 \
            -n $N_PREDICT \
            -c $CTX_SIZE \
            -ngl $NGL \
            -dev $DEVICE \
            -o json \
            2>/dev/null
    " > "$out_dir/benchmark_result.json" 2>/dev/null \
        || echo "[WARN] llama-bench failed" > "$out_dir/benchmark_result.json"

    echo "[OK] $variant artifacts → $out_dir/"
}

# ------------------------------------------------------------------ #
# Run each variant
# ------------------------------------------------------------------ #
for v in $VARIANTS; do
    case "$v" in
        tied_on)  run_variant tied_on  "$GGUF_TIED_ON"  ;;
        tied_off) run_variant tied_off "$GGUF_TIED_OFF" ;;
        *) echo "[WARN] unknown variant: $v" ;;
    esac
done

# ------------------------------------------------------------------ #
# Generate summary
# ------------------------------------------------------------------ #
echo ""
echo "[summary] Running parse_results.py ..."
python3 "$SCRIPT_DIR/parse_results.py" "$SCRIPT_DIR" \
    || echo "[WARN] parse_results.py failed — check manually"

echo ""
echo "=== Experiment complete ==="
echo "Results in: $SCRIPT_DIR/tied_on/  and  $SCRIPT_DIR/tied_off/"
echo "Summary  : $SCRIPT_DIR/summary.md"
