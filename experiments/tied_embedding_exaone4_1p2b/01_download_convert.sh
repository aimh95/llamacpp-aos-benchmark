#!/usr/bin/env bash
# EXAONE 4.0 1.2B 다운로드 및 GGUF 변환 (tied_on / tied_off 두 variant 생성).
# 실행 위치: llamacpp-aos-benchmark 루트
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
OUT_DIR="$SCRIPT_DIR"

HF_REPO="LGAI-EXAONE/EXAONE-4.0-1.2B"
MODEL_DIR="$ROOT_DIR/models/exaone4_1p2b_hf"
GGUF_TIED_ON="$ROOT_DIR/models/exaone4_1p2b_q8_0_tied_on.gguf"
GGUF_TIED_OFF="$ROOT_DIR/models/exaone4_1p2b_q8_0_tied_off.gguf"

QUANT_TYPE="${QUANT_TYPE:-Q8_0}"  # override with QUANT_TYPE=Q4_0 ./01_download_convert.sh

echo "=== EXAONE 4.0 1.2B download & convert ==="
echo "HF_REPO  : $HF_REPO"
echo "MODEL_DIR: $MODEL_DIR"
echo "QUANT    : $QUANT_TYPE"
echo ""

# ------------------------------------------------------------------ #
# Step 1: Download from HuggingFace
# ------------------------------------------------------------------ #
echo "[1/5] Downloading $HF_REPO ..."
mkdir -p "$MODEL_DIR"

if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'],
)
print('Download complete.')
"
elif command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR" \
      --exclude "*.msgpack" "*.h5" "flax_model*" "tf_model*"
else
    echo "[ERROR] huggingface_hub (pip) 또는 huggingface-cli 가 필요합니다."
    echo "  pip install huggingface_hub"
    exit 1
fi

# ------------------------------------------------------------------ #
# Step 2: Extract HF config summary
# ------------------------------------------------------------------ #
echo "[2/5] Extracting hf_config_summary.json ..."
python3 - <<PYEOF
import json, sys

try:
    with open("$MODEL_DIR/config.json") as f:
        cfg = json.load(f)
except FileNotFoundError:
    print("ERROR: config.json not found in $MODEL_DIR", file=sys.stderr)
    sys.exit(1)

summary = {
    "model_type":            cfg.get("model_type"),
    "architectures":         cfg.get("architectures"),
    "hidden_size":           cfg.get("hidden_size"),
    "num_hidden_layers":     cfg.get("num_hidden_layers"),
    "num_attention_heads":   cfg.get("num_attention_heads"),
    "num_key_value_heads":   cfg.get("num_key_value_heads"),
    "intermediate_size":     cfg.get("intermediate_size"),
    "vocab_size":            cfg.get("vocab_size"),
    "tie_word_embeddings":   cfg.get("tie_word_embeddings"),
    "max_position_embeddings": cfg.get("max_position_embeddings"),
    "rope_theta":            cfg.get("rope_theta"),
    "torch_dtype":           cfg.get("torch_dtype"),
}

out = {
    "hf_repo":  "$HF_REPO",
    "config_summary": summary,
    "tie_word_embeddings_value": cfg.get("tie_word_embeddings"),
    "notes": {
        "tied_on":  "output.weight absent in GGUF → llama.cpp uses token_embd.weight+TENSOR_DUPLICATED",
        "tied_off": "output.weight present in GGUF → llama.cpp uses it separately for final projection",
    },
}

for d in ["$OUT_DIR/tied_on", "$OUT_DIR/tied_off"]:
    with open(f"{d}/hf_config_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {d}/hf_config_summary.json")

tie = cfg.get("tie_word_embeddings")
print(f"  tie_word_embeddings = {tie}")
if not tie:
    print("  [WARN] HF config already has tie_word_embeddings=false — model may already have lm_head.weight")
PYEOF

# ------------------------------------------------------------------ #
# Step 3: Convert to GGUF (tied_on = default, no output.weight)
# ------------------------------------------------------------------ #
echo "[3/5] Converting to GGUF (tied_on = $QUANT_TYPE) ..."
GGUF_TIED_ON="$ROOT_DIR/models/exaone4_1p2b_${QUANT_TYPE,,}_tied_on.gguf"

python3 "$LLAMA_DIR/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$GGUF_TIED_ON" \
    --outtype "${QUANT_TYPE,,}" \
    2>&1 | tee "$OUT_DIR/tied_on/convert.log"

echo "  tied_on GGUF: $GGUF_TIED_ON  ($(du -sh "$GGUF_TIED_ON" | cut -f1))"

# ------------------------------------------------------------------ #
# Step 4: Dump tensor list (tied_on)
# ------------------------------------------------------------------ #
echo "[4/5] Dumping tensor list (tied_on) ..."
python3 "$SCRIPT_DIR/dump_gguf_tensors.py" \
    "$GGUF_TIED_ON" \
    "$OUT_DIR/tied_on/gguf_tensor_list.txt" \
    "$OUT_DIR/tied_on/tensor_type_summary.txt"

# ------------------------------------------------------------------ #
# Step 5: Create tied_off variant
# ------------------------------------------------------------------ #
echo "[5/5] Creating tied_off GGUF ..."
GGUF_TIED_OFF="$ROOT_DIR/models/exaone4_1p2b_${QUANT_TYPE,,}_tied_off.gguf"

python3 "$SCRIPT_DIR/create_tied_off_gguf.py" \
    --input  "$GGUF_TIED_ON" \
    --output "$GGUF_TIED_OFF" \
    --log    "$OUT_DIR/tied_off/convert.log"

if [ $? -eq 0 ]; then
    echo "  tied_off GGUF: $GGUF_TIED_OFF  ($(du -sh "$GGUF_TIED_OFF" | cut -f1))"
    python3 "$SCRIPT_DIR/dump_gguf_tensors.py" \
        "$GGUF_TIED_OFF" \
        "$OUT_DIR/tied_off/gguf_tensor_list.txt" \
        "$OUT_DIR/tied_off/tensor_type_summary.txt"
else
    echo "  [WARN] tied_off 생성 실패 — tied_off/convert.log 참조"
fi

echo ""
echo "=== Done ==="
echo "tied_on : $GGUF_TIED_ON"
echo "tied_off: $GGUF_TIED_OFF"
echo ""
echo "Next: $SCRIPT_DIR/02_run_experiment.sh --serial <device_serial>"
