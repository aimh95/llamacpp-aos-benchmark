#!/usr/bin/env bash
# ixi_gen_1p2b 양자화 × backend 매트릭스 벤치.
# 6종(F16, Q8_0, Q4_0, Q4_K_S, Q4_K_M, pure Q4_K) × {HTP0, CPU} = 12 config.
# 지표: CPU/HTP usage(MiB), CPU/HTP layers, TTFT, prefill/decode time.
#
# 실행(호스트, 저장소 루트):
#   scripts/quantization/bench_quant_matrix.sh
# 단말 시리얼은 SER 환경변수로 override 가능 (기본 R3CX403LWAB).
set -uo pipefail

ROOT="/home/iptv-infra/workspace/llamacpp-aos-benchmark"
SER="${SER:-R3CX403LWAB}"
QBIN="$ROOT/build-host/bin/llama-quantize"
F16="$ROOT/models/ixi_gen_1p2b_f16.gguf"
QC="$ROOT/models/quant_compare"
OUT="$ROOT/artifacts/quantization/quant_matrix"
RAW="$OUT/raw"
RDIR="/data/local/tmp/llama.cpp"
GDIR="/data/local/tmp/gguf"
mkdir -p "$QC" "$RAW"

ADB() { adb -s "$SER" "$@"; }

# 모델 이름 -> 호스트 gguf 경로 (순서 유지)
NAMES=(F16 Q8_0 Q4_0 Q4_K_S Q4_K_M Q4_K_pure)
declare -A GG=(
  [F16]="$F16"
  [Q8_0]="$QC/ixi_gen_1p2b_q8_0.gguf"
  [Q4_0]="$ROOT/models/ixi_gen_1p2b_q4_0.gguf"
  [Q4_K_S]="$QC/ixi_gen_1p2b_q4_k_s.gguf"
  [Q4_K_M]="$QC/ixi_gen_1p2b_q4_k_m.gguf"
  [Q4_K_pure]="$QC/ixi_gen_1p2b_q4_k_pure.gguf"
)

# ---- 0. 누락 모델 생성 (idempotent) ----
gen() { # gen <out> <TYPE> [opts...]   → llama-quantize [opts] <in> <out> TYPE
  local out="$1" type="$2"; shift 2
  [ -f "$out" ] && { echo "[skip] $(basename "$out") 이미 있음"; return; }
  echo "[gen ] $(basename "$out")  (type=$type opts=$*)"
  "$QBIN" "$@" "$F16" "$out" "$type" >/dev/null 2>&1 || { echo "  [ERR] 생성 실패 $out"; return 1; }
}
# F16 은 입력 자체라 생성 불필요
gen "${GG[Q8_0]}"      Q8_0
gen "${GG[Q4_0]}"      Q4_0
gen "${GG[Q4_K_S]}"    Q4_K_S
gen "${GG[Q4_K_M]}"    Q4_K_M
gen "${GG[Q4_K_pure]}" Q4_K_M --pure      # pure Q4_K

# ---- 1. 고정 프롬프트 파일 준비 & 단말 push (따옴표 이슈 회피) ----
PROMPT="$OUT/prompt.txt"
printf '%s' "On-device AI runs machine learning models directly on the phone hardware instead of a remote server, which improves latency and privacy. Explain in a few sentences how a neural processing unit accelerates transformer inference." > "$PROMPT"
ADB push "$PROMPT" "$GDIR/prompt.txt" >/dev/null

# 모델 push (원격 크기 다르면만)
for n in "${NAMES[@]}"; do
  src="${GG[$n]}"; base="$(basename "$src")"
  [ -f "$src" ] || { echo "[warn] $src 없음 → skip"; continue; }
  lsz=$(stat -c%s "$src")
  rsz=$(ADB shell "stat -c%s $GDIR/$base 2>/dev/null" | tr -d '\r' || true)
  if [ "$lsz" != "$rsz" ]; then echo "[push] $base"; ADB push "$src" "$GDIR/$base" >/dev/null; else echo "[skip] $base 이미 단말에 있음"; fi
done

# ---- 2. 실행 헬퍼 ----
LDENV="LD_LIBRARY_PATH=$RDIR/lib ADSP_LIBRARY_PATH=$RDIR/lib"

# perf/alloc 를 위한 llama-completion -v 1회
run_alloc() { # run_alloc <base> <HTP0|CPU> <rawfile>
  local base="$1" mode="$2" raw="$3" dev ngl
  if [ "$mode" = HTP0 ]; then dev="--device HTP0"; ngl=99; else dev=""; ngl=0; fi
  ADB shell "cd $RDIR && $LDENV GGML_HEXAGON_VERBOSE=1 ./bin/llama-completion --no-mmap -v \
    -m $GDIR/$base $dev -ngl $ngl --ctx-size 512 -t 6 -n 32 -no-cnv -f $GDIR/prompt.txt" > "$raw" 2>&1
}

# throughput 을 위한 llama-bench
run_bench() { # run_bench <base> <HTP0|CPU> <rawfile>
  local base="$1" mode="$2" raw="$3" dev
  if [ "$mode" = HTP0 ]; then dev="--device HTP0 -ngl 99"; else dev="-ngl 0"; fi
  ADB shell "cd $RDIR && $LDENV ./bin/llama-bench -m $GDIR/$base $dev -p 128 -n 64 -r 3" > "$raw" 2>&1
}

# ---- 3. 매트릭스 실행 + 지표 추출 ----
CSV="$OUT/quant_matrix.csv"
echo "quant,backend,htp_layers,cpu_layers,htp_mib,cpu_mib,ttft_ms,prefill_ms,prefill_tps,decode_ms_per_tok,decode_tps" > "$CSV"

num() { grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1; }

for n in "${NAMES[@]}"; do
  src="${GG[$n]}"; [ -f "$src" ] || continue
  base="$(basename "$src")"
  for mode in HTP0 CPU; do
    a="$RAW/${n}_${mode}_alloc.log"; b="$RAW/${n}_${mode}_bench.log"
    echo "=== $n / $mode ==="
    run_alloc "$base" "$mode" "$a"
    run_bench "$base" "$mode" "$b"

    # layers: HTP0에 배정된 distinct 레이어 수 (reserve+actual 중복 제거)
    htp_layers=$(grep -oE 'layer +[0-9]+ assigned to device HTP0' "$a" 2>/dev/null | grep -oE '[0-9]+' | sort -un | wc -l | tr -d ' ')
    htp_layers=${htp_layers:-0}
    cpu_layers=$(( 30 - htp_layers )); [ "$cpu_layers" -lt 0 ] && cpu_layers=0
    # usage MiB: weight 버퍼의 실제값(마지막 등장; 첫 블록은 예약 추정치 0.00)
    #  - htp_mib = HTP0(plain, F16용) + HTP0-REPACK(양자화 repack용). Q4_0만 크게 잡히고 Q4_K/Q8_0은 작게 나올 것.
    #  - cpu_mib = CPU model buffer (CPU에 남은 weight, 예: tied token_embd).
    htp_rp=$(grep "HTP0-REPACK model buffer size" "$a" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    htp_pl=$(grep "HTP0 model buffer size" "$a" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    htp_mib=$(awk -v a="${htp_rp:-0}" -v b="${htp_pl:-0}" 'BEGIN{printf "%.2f", a+b}')
    cpu_mib=$(grep "CPU model buffer size" "$a" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    # prefill / decode / ttft (완성 perf)
    prefill_ms=$(grep "prompt eval time" "$a" | sed -E 's/.*= *([0-9.]+) ms.*/\1/' | head -1)
    load_ms=$(grep "load time" "$a" | sed -E 's/.*= *([0-9.]+) ms.*/\1/' | head -1)
    decode_ms=$(grep "eval time =" "$a" | grep -v prompt | sed -E 's/.*\( *([0-9.]+) ms per token.*/\1/' | head -1)
    ttft_ms=$(awk -v a="${load_ms:-0}" -v b="${prefill_ms:-0}" 'BEGIN{printf "%.2f", a+b}')
    # throughput (bench)
    prefill_tps=$(grep -E "pp128" "$b" | sed -E 's/.*\| *([0-9.]+) ± [0-9.]+ *\|.*/\1/' | head -1)
    decode_tps=$(grep -E "tg64"  "$b" | sed -E 's/.*\| *([0-9.]+) ± [0-9.]+ *\|.*/\1/' | head -1)

    echo "$n,$mode,$htp_layers,$cpu_layers,${htp_mib:-},${cpu_mib:-},${ttft_ms:-},${prefill_ms:-},${prefill_tps:-},${decode_ms:-},${decode_tps:-}" >> "$CSV"
  done
done

echo ""
echo "===== 결과 ($CSV) ====="
column -t -s, "$CSV"
echo ""
echo "raw 로그: $RAW/  (config별 *_alloc.log, *_bench.log)"
