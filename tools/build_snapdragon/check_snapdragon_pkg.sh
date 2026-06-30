#!/usr/bin/env bash
# pkg-snapdragon/llama.cpp 패키지(cmake --install 결과물)에 HTP/OpenCL 백엔드 산출물이
# 모두 있는지 확인한다. root 불필요, 단말 연결도 필요 없다 (로컬 파일 존재 여부만 확인).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PKG_DIR="pkg-snapdragon/llama.cpp"

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [options]
  --pkg-dir <path>   cmake --install 결과물 경로 (default: $PKG_DIR)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pkg-dir) PKG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[WARN] unknown argument: $1 (ignored)" >&2; shift ;;
  esac
done

if [[ ! -d "$PKG_DIR" ]]; then
  echo "[ERROR] package directory not found: $PKG_DIR" >&2
  echo "        먼저 tools/build_snapdragon/build_snapdragon_llamacpp.sh 를 실행하세요." >&2
  exit 1
fi
PKG_DIR="$(cd "$PKG_DIR" && pwd)"

REQUIRED_ITEMS=(
  "bin/llama-cli"
  "bin/llama-bench"
  "lib/libggml.so"
  "lib/libggml-cpu.so"
  "lib/libggml-opencl.so"
  "lib/libggml-hexagon.so"
  "lib/libggml-htp-v75.so"
)

RECOMMENDED_ITEMS=(
  "lib/libggml-htp-v73.so"
  "lib/libggml-htp-v79.so"
  "lib/libggml-htp-v81.so"
  "scripts/snapdragon/adb/run-completion.sh"
  "scripts/snapdragon/adb/run-bench.sh"
  "scripts/snapdragon/adb/run-tool.sh"
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/artifacts/build_snapdragon/check_$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "== Snapdragon package check =="
echo "pkg_dir=$PKG_DIR"
echo "Output dir: $OUT_DIR"

REQUIRED_MISSING=0
declare -a REQUIRED_RESULTS
for item in "${REQUIRED_ITEMS[@]}"; do
  if [[ -f "$PKG_DIR/$item" ]]; then
    REQUIRED_RESULTS+=("$item|FOUND")
  else
    REQUIRED_RESULTS+=("$item|NOT_FOUND")
    REQUIRED_MISSING=$((REQUIRED_MISSING + 1))
  fi
done

declare -a RECOMMENDED_RESULTS
for item in "${RECOMMENDED_ITEMS[@]}"; do
  if [[ -f "$PKG_DIR/$item" ]]; then
    RECOMMENDED_RESULTS+=("$item|FOUND")
  else
    RECOMMENDED_RESULTS+=("$item|NOT_FOUND")
  fi
done

if [[ "$REQUIRED_MISSING" -eq 0 ]]; then
  OVERALL="PASS"
else
  OVERALL="FAIL"
fi

{
  echo "# Snapdragon Package Check"
  echo ""
  echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Package dir: \`$PKG_DIR\`"
  echo "- Overall: **$OVERALL** (missing required: $REQUIRED_MISSING/${#REQUIRED_ITEMS[@]})"
  echo ""
  echo "## Required"
  echo ""
  echo "| item | status |"
  echo "| --- | --- |"
  for row in "${REQUIRED_RESULTS[@]}"; do
    echo "| ${row%%|*} | ${row##*|} |"
  done
  echo ""
  echo "## Recommended"
  echo ""
  echo "| item | status |"
  echo "| --- | --- |"
  for row in "${RECOMMENDED_RESULTS[@]}"; do
    echo "| ${row%%|*} | ${row##*|} |"
  done
  echo ""
} > "$OUT_DIR/summary.md"

{
  echo "category,item,status"
  for row in "${REQUIRED_RESULTS[@]}"; do
    echo "required,${row%%|*},${row##*|}"
  done
  for row in "${RECOMMENDED_RESULTS[@]}"; do
    echo "recommended,${row%%|*},${row##*|}"
  done
} > "$OUT_DIR/summary.csv"

cat "$OUT_DIR/summary.md"
echo "Report written to: $OUT_DIR/summary.md / summary.csv"

if [[ "$OVERALL" == "PASS" ]]; then
  exit 0
else
  exit 1
fi
