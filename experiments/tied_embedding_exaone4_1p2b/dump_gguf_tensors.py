#!/usr/bin/env python3
"""GGUF 파일의 tensor list를 덤프한다.
token_embd.weight, output.weight, lm_head.weight의 존재 여부를 기록한다.

Usage: dump_gguf_tensors.py <gguf_path> <tensor_list_out> <summary_out>
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "third_party/llama.cpp/gguf-py"))
from gguf import GGUFReader

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <gguf_path> <tensor_list_out> <summary_out}")
        sys.exit(1)

    gguf_path   = Path(sys.argv[1])
    list_out    = Path(sys.argv[2])
    summary_out = Path(sys.argv[3])

    if not gguf_path.exists():
        print(f"[ERROR] {gguf_path} not found")
        sys.exit(1)

    reader = GGUFReader(str(gguf_path), "r")
    list_out.parent.mkdir(parents=True, exist_ok=True)

    KEY_TENSORS = ["token_embd.weight", "output.weight", "lm_head.weight", "output_norm.weight"]
    found = {k: None for k in KEY_TENSORS}
    all_tensors = []

    for tensor in reader.tensors:
        name  = tensor.name
        shape = list(tensor.shape)
        dtype = str(tensor.tensor_type)
        nbytes = tensor.nbytes if hasattr(tensor, "nbytes") else 0
        all_tensors.append({"name": name, "dtype": dtype, "shape": shape, "nbytes": nbytes})
        if name in found:
            found[name] = {"dtype": dtype, "shape": shape, "nbytes": nbytes}

    # --- tensor list txt ---
    with open(list_out, "w") as f:
        f.write(f"# GGUF tensor list: {gguf_path.name}  ({len(all_tensors)} tensors)\n")
        f.write(f"{'idx':<5}  {'name':<60}  {'dtype':<12}  shape\n")
        for i, t in enumerate(all_tensors):
            f.write(f"{i:<5}  {t['name']:<60}  {t['dtype']:<12}  {t['shape']}\n")

    # --- summary txt ---
    with open(summary_out, "w") as f:
        f.write(f"# tensor_type_summary: {gguf_path.name}\n\n")
        f.write("## Key tensors\n")
        for k in KEY_TENSORS:
            v = found[k]
            if v:
                f.write(f"  {k:40s}  FOUND   dtype={v['dtype']}  shape={v['shape']}\n")
            else:
                f.write(f"  {k:40s}  NOT_FOUND\n")
        f.write(f"\n## Summary\n")
        f.write(f"  total tensors     : {len(all_tensors)}\n")
        f.write(f"  token_embd.weight : {'FOUND' if found['token_embd.weight'] else 'NOT_FOUND'}\n")
        f.write(f"  output.weight     : {'FOUND' if found['output.weight'] else 'NOT_FOUND'}\n")
        f.write(f"  lm_head.weight    : {'FOUND' if found['lm_head.weight'] else 'NOT_FOUND'}\n")
        tie_inference = "TIED (output.weight absent → llama.cpp uses token_embd.weight+TENSOR_DUPLICATED)" \
            if not found["output.weight"] else "UNTIED (output.weight present → separate LM-head tensor)"
        f.write(f"  tie_inference     : {tie_inference}\n")

    print(f"  tensor list  → {list_out}")
    print(f"  type summary → {summary_out}")
    print(f"  token_embd.weight : {'FOUND' if found['token_embd.weight'] else 'NOT_FOUND'}")
    print(f"  output.weight     : {'FOUND' if found['output.weight'] else 'NOT_FOUND'}")


if __name__ == "__main__":
    main()
