#!/usr/bin/env bash
# Phase-1 (무패치) layer profiling 캡처 하네스.
# 이미 단말에 설치된 llama.cpp 패키지 + 기존 Q4_0/Q8_0 GGUF 로, 소스 수정 없이
# assigned backend(스케줄러) + HTP op timing(HEXAGON_PROFILE) + graph-level 성능(bench)을 캡처한다.
#
# 실행(호스트, 저장소 루트):
#   scripts/layer_profile/capture_layer_profile.sh
# 환경변수 override: SER(시리얼), REPS(측정 횟수), WARMUP, NPRED, CTX
set -uo pipefail

ROOT="/home/iptv-infra/workspace/llamacpp-aos-benchmark"
SER="${SER:-R3CX403LWAB}"
RDIR="/data/local/tmp/llama.cpp"
GDIR="/data/local/tmp/gguf"
OUT="$ROOT/artifacts/layer_profile"
RAW="$OUT/raw"
mkdir -p "$RAW"

REPS="${REPS:-10}"      # 측정 횟수 (>=10)
WARMUP="${WARMUP:-3}"   # warm-up (>=3)
NPRED="${NPRED:-64}"    # decode 생성 길이 (>=64)
CTX="${CTX:-2048}"
PPTOK=255               # prefill prompt token 수 (bench 정확 지정)

ADB(){ adb -s "$SER" "$@"; }
LDENV="LD_LIBRARY_PATH=$RDIR/lib ADSP_LIBRARY_PATH=$RDIR/lib"

# config: name -> gguf basename (단말 경로)
CFG_NAMES=(Q4_0 Q8_0)
declare -A GGUF=(
  [Q4_0]="ixi_gen_1p2b_q4_0.gguf"
  [Q8_0]="ixi_gen_1p2b_q8_0.gguf"
)

# ---- 0. 환경 스냅샷 (Environment 시트용) ----
ENVF="$OUT/environment.txt"
{
  echo "captured_at=$(date -Iseconds)"
  echo "host_llama_commit=$(git -C "$ROOT/third_party/llama.cpp" rev-parse HEAD)"
  echo "serial=$SER"
  echo "reps=$REPS warmup=$WARMUP npred=$NPRED ctx=$CTX pptok=$PPTOK"
  echo "--- device ---"
  ADB shell 'getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release' | tr -d '\r'
  echo "--- htp arch / bench build ---"
  ADB shell "cd $RDIR && $LDENV ./bin/llama-bench --version 2>/dev/null" | tr -d '\r' | head -2
  echo "--- thermal (start) ---"
  ADB shell 'for z in /sys/class/thermal/thermal_zone*/temp; do cat $z 2>/dev/null; done | sort -rn | head -5' | tr -d '\r'
  echo "--- cpu governor ---"
  ADB shell 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do cat $g 2>/dev/null; done | sort | uniq -c' | tr -d '\r'
} > "$ENVF"
echo "[env] -> $ENVF"

# ---- 1. 프롬프트 파일 준비 & 실제 토큰 수 검증 ----
PROMPT="$OUT/prompt_255.txt"
if [ ! -f "$PROMPT" ]; then
  # 대략 255 토큰 분량 (실제 수는 아래서 검증/보고). bench 는 -p 255 로 정확 지정하므로 timing 은 무관.
  python3 - "$PROMPT" <<'PY'
import sys
# 약 255 토큰 분량 (base ~60토큰 x4). 실제 수는 아래서 검증/보고. bench 는 -p 255 로 정확 지정.
base=("On-device AI runs neural networks directly on the phone's system-on-chip, "
      "using the CPU, GPU, and the Hexagon neural processing unit. ")
open(sys.argv[1],"w").write((base*4).strip())
PY
fi
ADB push "$PROMPT" "$GDIR/prompt.txt" >/dev/null
NTOK=$(ADB shell "cd $RDIR && $LDENV ./bin/llama-tokenize -m $GDIR/${GGUF[Q4_0]} -f $GDIR/prompt.txt 2>/dev/null | grep -c '^ *[0-9]* ->'" | tr -d '\r')
echo "[prompt] structure-trace prompt token 수(실측) = ${NTOK:-?}  (bench prefill 은 -p $PPTOK 정확 지정)"
echo "prompt_actual_tokens=${NTOK:-unknown}" >> "$ENVF"

# ---- 2. 캡처 함수 ----
dev_flags(){ [ "$1" = HTP0 ] && echo "--device HTP0 -ngl 99" || echo "-ngl 0"; }

# 2a. 구조/assignment 트레이스 (SCHED_DEBUG=2 + OPTRACE), 1회. 강제 sync 없음.
cap_assign(){ # cap_assign <cfgname> <HTP0|CPU>
  local c="$1" mode="$2" g="${GGUF[$1]}" f; f="$RAW/assign_${c}_${mode}.log"
  ADB shell "cd $RDIR && $LDENV GGML_SCHED_DEBUG=2 GGML_OP_TRACE=1 ./bin/llama-completion --no-mmap -v \
    -m $GDIR/$g $(dev_flags "$mode") --ctx-size $CTX -t 6 -n 2 -no-cnv -f $GDIR/prompt.txt" > "$f" 2>&1
  echo "  [assign] $c/$mode -> $(basename "$f") ($(wc -l < "$f") lines)"
}

# 2b. HTP op timing (HEXAGON_PROFILE=1), HTP 모드만. per-op usec + per-batch.
#     profile-op 출력은 DEBUG 레벨이라 -v 필수. GGML_OP_TRACE=1 로 [OPTRACE][HTP] 맵도 같은 로그에 남겨 dst 이름 join.
cap_htp_prof(){ # cap_htp_prof <cfgname>
  local c="$1" g="${GGUF[$1]}" f; f="$RAW/htpprof_${c}_HTP0.log"
  ADB shell "cd $RDIR && $LDENV GGML_HEXAGON_PROFILE=1 GGML_OP_TRACE=1 ./bin/llama-completion --no-mmap -v \
    -m $GDIR/$g --device HTP0 -ngl 99 --ctx-size $CTX -t 6 -n $NPRED -no-cnv -f $GDIR/prompt.txt" > "$f" 2>&1
  echo "  [htpprof] $c -> $(basename "$f")"
}

# 2d. CPU per-op timing (Phase-2 패치 GGML_LAYER_PROFILE=timing). CPU모드=cpu시간, HTP모드=CPU-fallback op(임베딩/출력) 시간.
#     ※ 단말 패키지가 패치된 소스로 재빌드/재설치된 뒤에만 [LAYERPROF] 가 나온다.
cap_cpuprof(){ # cap_cpuprof <cfgname> <HTP0|CPU>
  local c="$1" mode="$2" g="${GGUF[$1]}" f; f="$RAW/cpuprof_${c}_${mode}.log"
  ADB shell "cd $RDIR && $LDENV GGML_LAYER_PROFILE=timing ./bin/llama-completion --no-mmap \
    -m $GDIR/$g $(dev_flags "$mode") --ctx-size $CTX -t 6 -n $NPRED -no-cnv -f $GDIR/prompt.txt" > "$f" 2>&1
  local nlp; nlp=$(grep -c "\[LAYERPROF\]" "$f" 2>/dev/null || echo 0)
  echo "  [cpuprof] $c/$mode -> $(basename "$f")  (LAYERPROF ${nlp}줄; 0이면 패치 미빌드)"
}

# 2c. graph-level 성능 (llama-bench, prefill=pp255 / decode=tg64), warmup+reps, median/stats.
cap_bench(){ # cap_bench <cfgname> <HTP0|CPU>
  local c="$1" mode="$2" g="${GGUF[$1]}" f; f="$RAW/bench_${c}_${mode}.log"
  # 주의: 이 llama-bench 에는 -w(warmup 횟수) 옵션이 없다(기본 warmup on, 끄려면 --no-warmup). -r 만 사용.
  ADB shell "cd $RDIR && $LDENV ./bin/llama-bench -m $GDIR/$g $(dev_flags "$mode") \
    -p $PPTOK -n $NPRED -r $REPS -o md" > "$f" 2>&1
  echo "  [bench] $c/$mode -> $(basename "$f")"
}

# ---- 3. 실험 매트릭스: 교차 순서 (Q4_0 CPU, Q4_0 HTP, Q8_0 CPU, Q8_0 HTP) + 역순 반복 ----
run_one(){ # run_one <cfg> <mode>
  echo "=== $1 / $2 ==="
  cap_assign  "$1" "$2"
  cap_bench   "$1" "$2"
  cap_cpuprof "$1" "$2"          # CPU per-op 시간 (패치 필요)
  [ "$2" = HTP0 ] && cap_htp_prof "$1"
}

ORDER_FWD=( "Q4_0 CPU" "Q4_0 HTP0" "Q8_0 CPU" "Q8_0 HTP0" )
echo "########## PASS 1 (forward) ##########"
for pair in "${ORDER_FWD[@]}"; do run_one $pair; done
echo "########## PASS 2 (reverse, bench 재측정으로 열편향 완화) ##########"
for ((i=${#ORDER_FWD[@]}-1;i>=0;i--)); do set -- ${ORDER_FWD[$i]}; cap_bench "$1" "$2"; mv "$RAW/bench_${1}_${2}.log" "$RAW/bench_${1}_${2}_pass2.log"; done

# ---- 4. 종료 thermal ----
{ echo "--- thermal (end) ---"; ADB shell 'for z in /sys/class/thermal/thermal_zone*/temp; do cat $z 2>/dev/null; done | sort -rn | head -5' | tr -d '\r'; } >> "$ENVF"

echo ""
echo "완료. raw 로그: $RAW/"
echo "다음: python3 scripts/layer_profile/parse_layer_profile.py --raw $RAW --out $OUT"
