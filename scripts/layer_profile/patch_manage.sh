#!/usr/bin/env bash
# Phase-2 LAYERPROF 패치 관리: apply | revert | status
# 패치: patches/0002-layerprof-cpu-op-timing.patch  (ggml-cpu.c per-op CPU 타이머, env GGML_LAYER_PROFILE=timing)
set -uo pipefail
ROOT="/home/iptv-infra/workspace/llamacpp-aos-benchmark"
LL="$ROOT/third_party/llama.cpp"
P="$ROOT/patches/0002-layerprof-cpu-op-timing.patch"
F="ggml/src/ggml-cpu/ggml-cpu.c"

applied() { grep -q "LAYERPROF" "$LL/$F"; }

case "${1:-status}" in
  apply)
    if applied; then echo "[skip] 이미 적용됨 (LAYERPROF 존재)"; exit 0; fi
    git -C "$LL" apply --check "$P" && git -C "$LL" apply "$P" && echo "[ok] 적용 완료" || { echo "[ERR] apply 실패"; exit 1; }
    ;;
  revert)
    if ! applied; then echo "[skip] 적용 안 됨"; exit 0; fi
    # 패치로 되돌리기 시도, 실패 시 git checkout 로 원복
    git -C "$LL" apply -R "$P" 2>/dev/null && echo "[ok] 패치 -R 원복" || \
      { git -C "$LL" checkout -- "$F" && echo "[ok] git checkout 원복"; }
    ;;
  status)
    applied && echo "적용됨 (LAYERPROF 있음)" || echo "미적용"
    git -C "$LL" diff --stat -- "$F"
    ;;
  *) echo "usage: $0 apply|revert|status"; exit 2 ;;
esac
