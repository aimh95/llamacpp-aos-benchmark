#!/usr/bin/env python3
"""Compare per-tensor quantization types across Q4_K_S / Q4_K_M / pure-Q4_K GGUFs.

Reads the actual tensor types from each generated GGUF (not the quant log) and
emits three CSVs plus an aggregate JSON used by the analysis report.

Usage:
  python scripts/quantization/compare_q4k_recipes.py \
    --q4-k-s   models/quant_compare/ixi_gen_1p2b_q4_k_s.gguf \
    --q4-k-m   models/quant_compare/ixi_gen_1p2b_q4_k_m.gguf \
    --q4-k-pure models/quant_compare/ixi_gen_1p2b_q4_k_pure.gguf \
    --output-dir artifacts/quantization/q4k_compare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict

# gguf from the in-repo llama.cpp checkout
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

# The 7 projection roles we tabulate per layer, in a stable order.
PROJ_ROLES = ["attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down"]

_BLK_RE = re.compile(r"^blk\.(\d+)\.")


def classify(name: str):
    """Return (layer_index, role). layer_index is int or None."""
    m = _BLK_RE.match(name)
    layer = int(m.group(1)) if m else None
    base = name
    if m:
        base = name[m.end():]  # strip 'blk.N.'

    # non-layer tensors
    if name == "token_embd.weight":
        return None, "token_embd"
    if name == "output.weight":
        return None, "output"
    if name == "output_norm.weight":
        return None, "norm"

    role_map = {
        "attn_q.weight": "attn_q",
        "attn_k.weight": "attn_k",
        "attn_v.weight": "attn_v",
        "attn_output.weight": "attn_output",
        "ffn_gate.weight": "ffn_gate",
        "ffn_up.weight": "ffn_up",
        "ffn_down.weight": "ffn_down",
    }
    if base in role_map:
        return layer, role_map[base]
    if "norm" in base:
        return layer, "norm"
    return layer, "other"


def read_types(path: str):
    """name -> dict(type, shape_tuple), preserving file order."""
    r = GGUFReader(path)
    out = OrderedDict()
    for t in r.tensors:
        out[t.name] = {
            "type": t.tensor_type.name,
            "shape": tuple(int(x) for x in t.shape),
        }
    return out


def shape_str(shape):
    return "x".join(str(x) for x in shape) if shape else ""


def change_label(pure, s, m):
    """Human label describing how a tensor's type differs across recipes."""
    if pure == s == m:
        return "unchanged"
    # which recipes promote above the pure baseline
    promoted = []
    if s != pure:
        promoted.append("S")
    if m != pure:
        promoted.append("M")
    # if S and M agree but differ from pure
    tgt = m if m != pure else s
    if s == m and s != pure:
        where = "S+M"
    elif m != pure and s == pure:
        where = "M only"
    elif s != pure and m == pure:
        where = "S only"
    else:
        # s, m and pure all differ from each other
        return f"{pure} -> S:{s} / M:{m}"
    return f"{pure} -> {tgt} ({where})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4-k-s", required=True)
    ap.add_argument("--q4-k-m", required=True)
    ap.add_argument("--q4-k-pure", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = {
        "pure": read_types(args.q4_k_pure),
        "s": read_types(args.q4_k_s),
        "m": read_types(args.q4_k_m),
    }

    # --- integrity: identical tensor name sets and shapes ---
    names_pure = list(data["pure"].keys())
    integrity = {"name_sets_match": True, "shape_mismatches": []}
    for key in ("s", "m"):
        if set(data[key].keys()) != set(names_pure):
            integrity["name_sets_match"] = False
    for name in names_pure:
        shapes = {k: data[k][name]["shape"] for k in data if name in data[k]}
        if len(set(shapes.values())) > 1:
            integrity["shape_mismatches"].append({"tensor": name, "shapes": {k: shape_str(v) for k, v in shapes.items()}})

    # --- per-tensor comparison CSV ---
    comp_path = os.path.join(args.output_dir, "tensor_type_comparison.csv")
    rows = []
    for name in names_pure:
        layer, role = classify(name)
        pt = data["pure"][name]["type"]
        st = data["s"][name]["type"]
        mt = data["m"][name]["type"]
        rows.append({
            "tensor_name": name,
            "layer_index": "" if layer is None else layer,
            "tensor_role": role,
            "shape": shape_str(data["pure"][name]["shape"]),
            "q4_k_pure_type": pt,
            "q4_k_s_type": st,
            "q4_k_m_type": mt,
            "pure_vs_s_changed": int(pt != st),
            "s_vs_m_changed": int(st != mt),
            "pure_vs_m_changed": int(pt != mt),
        })
    with open(comp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- layer summary CSV (per recipe columns for the 7 proj roles) ---
    # collect per layer -> role -> type for each recipe
    layers = sorted({r["layer_index"] for r in rows if r["layer_index"] != "" and classify(r["tensor_name"])[1] in PROJ_ROLES})
    recipe_keys = [("pure", "q4_k_pure_type"), ("s", "q4_k_s_type"), ("m", "q4_k_m_type")]
    layer_path = os.path.join(args.output_dir, "layer_type_summary.csv")
    header = ["layer"]
    for rk, _ in recipe_keys:
        header += [f"{rk}_{role}" for role in PROJ_ROLES]
    # index rows by (layer, role)
    lut = {}
    for r in rows:
        if r["layer_index"] == "":
            continue
        lut[(r["layer_index"], r["tensor_role"])] = r
    with open(layer_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for L in layers:
            line = [L]
            for _, col in recipe_keys:
                for role in PROJ_ROLES:
                    cell = lut.get((L, role))
                    line.append(cell[col] if cell else "")
            w.writerow(line)

    # --- promoted (changed) tensors CSV ---
    prom_path = os.path.join(args.output_dir, "promoted_tensors.csv")
    prom_rows = []
    for r in rows:
        pt, st, mt = r["q4_k_pure_type"], r["q4_k_s_type"], r["q4_k_m_type"]
        if not (pt == st == mt):
            prom_rows.append({
                "tensor_name": r["tensor_name"],
                "layer_index": r["layer_index"],
                "tensor_role": r["tensor_role"],
                "q4_k_pure_type": pt,
                "q4_k_s_type": st,
                "q4_k_m_type": mt,
                "change_pattern": change_label(pt, st, mt),
            })
    with open(prom_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tensor_name", "layer_index", "tensor_role",
                                          "q4_k_pure_type", "q4_k_s_type", "q4_k_m_type", "change_pattern"])
        w.writeheader()
        w.writerows(prom_rows)

    # --- aggregates JSON (for report) ---
    def type_counts(key):
        return dict(Counter(data[key][n]["type"] for n in names_pure))

    def file_size(p):
        return os.path.getsize(p)

    # counts of promotions in each recipe vs pure baseline
    def promo_count(recipe_col, target):
        return sum(1 for r in rows if r["q4_k_pure_type"] != r[recipe_col] and r[recipe_col] == target)

    s_vs_m = [r for r in rows if r["q4_k_s_type"] != r["q4_k_m_type"]]
    agg = {
        "commit": None,
        "files": {
            "pure": {"path": args.q4_k_pure, "bytes": file_size(args.q4_k_pure)},
            "s": {"path": args.q4_k_s, "bytes": file_size(args.q4_k_s)},
            "m": {"path": args.q4_k_m, "bytes": file_size(args.q4_k_m)},
        },
        "type_counts": {"pure": type_counts("pure"), "s": type_counts("s"), "m": type_counts("m")},
        "n_tensors": len(names_pure),
        "integrity": integrity,
        "diff_counts": {
            "pure_vs_s": sum(1 for r in rows if r["q4_k_pure_type"] != r["q4_k_s_type"]),
            "pure_vs_m": sum(1 for r in rows if r["q4_k_pure_type"] != r["q4_k_m_type"]),
            "s_vs_m": len(s_vs_m),
        },
        "m_promotions": {
            "to_Q5_K": promo_count("q4_k_m_type", "Q5_K"),
            "to_Q6_K": promo_count("q4_k_m_type", "Q6_K"),
        },
        "s_promotions": {
            "to_Q5_K": promo_count("q4_k_s_type", "Q5_K"),
            "to_Q6_K": promo_count("q4_k_s_type", "Q6_K"),
        },
        "s_vs_m_tensors": [
            {"tensor": r["tensor_name"], "role": r["tensor_role"], "layer": r["layer_index"],
             "s": r["q4_k_s_type"], "m": r["q4_k_m_type"]} for r in s_vs_m
        ],
    }
    with open(os.path.join(args.output_dir, "aggregates.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # console recap
    print("n_tensors:", agg["n_tensors"])
    print("integrity name_sets_match:", integrity["name_sets_match"],
          "| shape_mismatches:", len(integrity["shape_mismatches"]))
    print("type_counts pure:", agg["type_counts"]["pure"])
    print("type_counts S   :", agg["type_counts"]["s"])
    print("type_counts M   :", agg["type_counts"]["m"])
    print("diff pure_vs_s:", agg["diff_counts"]["pure_vs_s"],
          "| pure_vs_m:", agg["diff_counts"]["pure_vs_m"],
          "| s_vs_m:", agg["diff_counts"]["s_vs_m"])
    print("M promotions -> Q5_K:", agg["m_promotions"]["to_Q5_K"],
          "| -> Q6_K:", agg["m_promotions"]["to_Q6_K"])
    print("wrote:", comp_path, layer_path, prom_path)


if __name__ == "__main__":
    main()
