cat > scripts/05_run_bench.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="/data/local/tmp/llama"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="logs/llama-bench_${TS}.log"

echo "Writing log to $OUT"

for T in 1 2 4 6 8; do
  echo
  echo "=============================="
  echo "n_threads=$T"
  echo "=============================="

  adb shell "cd $DEVICE_DIR && ./llama-bench \
    -m model.gguf \
    -t $T \
    -p 128 \
    -n 128 \
    -c 512" | tee -a "$OUT"
done
EOF

chmod +x scripts/05_run_bench.sh