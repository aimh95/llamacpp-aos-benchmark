#!/usr/bin/env python3
"""실험 결과 파싱 및 summary.md 생성.

각 variant(tied_on / tied_off) 폴더의 log를 파싱하여:
  - graph_nodes.txt에서 GET_ROWS / final logits MUL_MAT 노드 추출
  - buffer_placement.txt에서 token_embd.weight/output.weight 배치 추출
  - graph_schedule.txt에서 backend 배정 추출
  - benchmark_result.json에서 성능 수치 추출
  - summary.md 생성

Usage: parse_results.py <experiment_dir>
"""
import sys
import re
import json
from pathlib import Path
from typing import Optional


VARIANTS = ["tied_on", "tied_off"]


# ------------------------------------------------------------------ #
# Parsers
# ------------------------------------------------------------------ #

def parse_embtrace(buffer_placement_path: Path) -> dict:
    """EMBTRACE 로그에서 token_embd.weight 배치 정보 추출."""
    result = {
        "token_embd_get_rows_backend":  "unknown",
        "token_embd_get_rows_buft":     "unknown",
        "token_embd_mulmat_backend":    "unknown",
        "token_embd_mulmat_buft":       "unknown",
        "output_weight_buft":           "unknown",
        "cpu_buft_list":                [],
        "gpu_buft_list":                [],
    }
    if not buffer_placement_path.exists():
        return result

    text = buffer_placement_path.read_text()

    # token_embd.weight GET_ROWS selected_buft
    m = re.search(
        r'\[EMBTRACE\]\[LOAD\] name=token_embd\.weight.*?op=GET_ROWS.*?selected_buft=(\S+)',
        text)
    if m:
        result["token_embd_get_rows_buft"] = m.group(1)

    # token_embd.weight MUL_MAT selected_buft
    m = re.search(
        r'\[EMBTRACE\]\[LOAD\] name=token_embd\.weight.*?op=MUL_MAT.*?selected_buft=(\S+)',
        text)
    if m:
        result["token_embd_mulmat_buft"] = m.group(1)

    # gpu_buft_list
    gpu_bufts = re.findall(r'\[EMBTRACE\]\[BUFT\]\s+gpu_buft_list\[(\d+)\] dev=(\S+) buft=(\S+)', text)
    result["gpu_buft_list"] = [f"{dev}/{buft}" for _, dev, buft in gpu_bufts]

    # cpu_buft_list
    cpu_bufts = re.findall(r'\[EMBTRACE\]\[BUFT\]\s+cpu_buft_list\[(\d+)\] dev=(\S+) buft=(\S+)', text)
    result["cpu_buft_list"] = [f"{dev}/{buft}" for _, dev, buft in cpu_bufts]

    return result


def parse_optrace(graph_schedule_path: Path) -> dict:
    """OPTRACE 로그에서 GET_ROWS / MUL_MAT backend 배정 추출."""
    result = {
        "get_rows_backend":      "unknown",
        "final_mulmat_backend":  "unknown",
        "splits":                [],
    }
    if not graph_schedule_path.exists():
        return result

    text = graph_schedule_path.read_text()

    # split 정보
    splits = re.findall(
        r'\[OPTRACE\]\[SCHED\] split=(\d+) backend=(\S+) n_nodes=(\d+)',
        text)
    result["splits"] = [{"split": int(s), "backend": b, "n_nodes": int(n)} for s, b, n in splits]

    # GET_ROWS node backend (CPU or HTP)
    m = re.search(r'\[OPTRACE\]\[(CPU|HTP[^\]]*)\].*GET_ROWS', text)
    if m:
        result["get_rows_backend"] = m.group(1)

    return result


def parse_graph_nodes(graph_nodes_path: Path) -> dict:
    """graph_nodes.txt에서 GET_ROWS / final MUL_MAT 노드 추출."""
    result = {
        "get_rows_nodes":   [],
        "final_mulmat_src": "unknown",
        "n_nodes":          0,
    }
    if not graph_nodes_path.exists():
        return result

    lines = graph_nodes_path.read_text().splitlines()

    # n_nodes from header
    for line in lines:
        m = re.search(r'n_nodes=(\d+)', line)
        if m:
            result["n_nodes"] = int(m.group(1))
            break

    # parse table rows (skip header lines starting with #)
    last_mulmat = None
    for line in lines:
        if line.startswith('#') or line.startswith('idx'):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        op = parts[2] if len(parts) > 2 else ""
        name = parts[1] if len(parts) > 1 else ""
        src0 = parts[5] if len(parts) > 5 else "-"
        buf  = parts[7] if len(parts) > 7 else "-"

        if op == "GET_ROWS":
            result["get_rows_nodes"].append({
                "name": name, "src0": src0, "buf": buf
            })
        if op == "MUL_MAT":
            last_mulmat = {"name": name, "src0": src0, "buf": buf}

    if last_mulmat:
        result["final_mulmat_src"] = last_mulmat.get("src0", "unknown")
        result["final_mulmat_buf"] = last_mulmat.get("buf", "unknown")

    return result


def parse_gguf_tensor_summary(summary_path: Path) -> dict:
    """tensor_type_summary.txt에서 key tensor 존재 여부 추출."""
    result = {
        "token_embd_weight": "unknown",
        "output_weight":     "unknown",
        "lm_head_weight":    "unknown",
        "tie_inference":     "unknown",
    }
    if not summary_path.exists():
        return result

    text = summary_path.read_text()
    for line in text.splitlines():
        if "token_embd.weight" in line:
            result["token_embd_weight"] = "FOUND" if "FOUND" in line else "NOT_FOUND"
        if "output.weight" in line and "token_embd" not in line:
            result["output_weight"] = "FOUND" if "FOUND" in line else "NOT_FOUND"
        if "lm_head.weight" in line:
            result["lm_head_weight"] = "FOUND" if "FOUND" in line else "NOT_FOUND"
        if "tie_inference" in line:
            result["tie_inference"] = line.split(":", 1)[-1].strip()
    return result


def parse_benchmark(bench_path: Path) -> dict:
    """llama-bench JSON 결과 파싱."""
    result = {"pp_tps": "N/A", "tg_tps": "N/A", "pp_ms": "N/A", "tg_ms": "N/A"}
    if not bench_path.exists():
        return result
    try:
        data = json.loads(bench_path.read_text())
        if isinstance(data, list) and data:
            for entry in data:
                if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                    result["pp_tps"] = f"{entry.get('pp_ts', 'N/A'):.1f}"
                    result["pp_ms"]  = f"{entry.get('pp_ms', 'N/A'):.0f}"
                elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) <= 1:
                    result["tg_tps"] = f"{entry.get('tg_ts', 'N/A'):.1f}"
                    result["tg_ms"]  = f"{entry.get('tg_ms', 'N/A'):.0f}"
    except Exception:
        pass
    return result


def parse_hf_config(config_path: Path) -> dict:
    result = {"tie_word_embeddings": "unknown", "hf_repo": "unknown"}
    if not config_path.exists():
        return result
    try:
        data = json.loads(config_path.read_text())
        cfg = data.get("config_summary", {})
        result["tie_word_embeddings"] = str(cfg.get("tie_word_embeddings", "unknown"))
        result["hf_repo"] = data.get("hf_repo", "unknown")
    except Exception:
        pass
    return result


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    records = {}

    for variant in VARIANTS:
        vdir = exp_dir / variant
        if not vdir.exists():
            print(f"[SKIP] {variant} dir not found")
            continue

        rec = {
            "variant":  variant,
            "hf_cfg":   parse_hf_config(vdir / "hf_config_summary.json"),
            "tensors":  parse_gguf_tensor_summary(vdir / "tensor_type_summary.txt"),
            "emb":      parse_embtrace(vdir / "buffer_placement.txt"),
            "sched":    parse_optrace(vdir / "graph_schedule.txt"),
            "graph":    parse_graph_nodes(vdir / "graph_nodes.txt"),
            "bench":    parse_benchmark(vdir / "benchmark_result.json"),
        }
        records[variant] = rec

    # ---------------------------------------------------------------- #
    # Write summary.md
    # ---------------------------------------------------------------- #
    out_path = exp_dir / "summary.md"
    with open(out_path, "w") as f:
        f.write("# EXAONE 4.0 1.2B: tied_embedding experiment summary\n\n")
        f.write("## Key question\n")
        f.write("token_embd.weight의 GET_ROWS op가 tied/untied 여부에 따라 HTP0에 할당되는가?\n\n")

        # Comparison table
        cols = [
            ("variant",                    lambda r: r["variant"]),
            ("HF tie_word_embeddings",     lambda r: r["hf_cfg"]["tie_word_embeddings"]),
            ("GGUF token_embd.weight",     lambda r: r["tensors"]["token_embd_weight"]),
            ("GGUF output.weight",         lambda r: r["tensors"]["output_weight"]),
            ("GGUF lm_head.weight",        lambda r: r["tensors"]["lm_head_weight"]),
            ("tie_inference",              lambda r: r["tensors"]["tie_inference"]),
            ("GET_ROWS src0",              lambda r: r["graph"]["get_rows_nodes"][0]["src0"]
                                                     if r["graph"]["get_rows_nodes"] else "N/A"),
            ("GET_ROWS backend",           lambda r: r["sched"]["get_rows_backend"]),
            ("token_embd.weight buf (GET_ROWS)", lambda r: r["emb"]["token_embd_get_rows_buft"]),
            ("token_embd.weight buf (MUL_MAT)",  lambda r: r["emb"]["token_embd_mulmat_buft"]),
            ("final logits src0",          lambda r: r["graph"]["final_mulmat_src"]),
            ("final logits backend",       lambda r: r["sched"]["final_mulmat_backend"]),
            ("pp t/s",                     lambda r: r["bench"]["pp_tps"]),
            ("tg t/s",                     lambda r: r["bench"]["tg_tps"]),
        ]

        # Header
        f.write("## Comparison table\n\n")
        f.write("| " + " | ".join(c[0] for c in cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")

        for variant in VARIANTS:
            if variant not in records:
                row_vals = ["N/A"] * len(cols)
                row_vals[0] = variant
            else:
                r = records[variant]
                row_vals = []
                for _, fn in cols:
                    try:
                        row_vals.append(str(fn(r)))
                    except Exception:
                        row_vals.append("ERR")
            f.write("| " + " | ".join(row_vals) + " |\n")

        # Analysis
        f.write("\n## Analysis\n\n")

        for variant, rec in records.items():
            f.write(f"### {variant}\n")
            get_rows_buf = rec["emb"]["token_embd_get_rows_buft"]
            get_rows_be  = rec["sched"]["get_rows_backend"]
            mulmat_buf   = rec["emb"]["token_embd_mulmat_buft"]
            output_wt    = rec["tensors"]["output_weight"]

            if get_rows_be == "HTP0" or "HTP" in get_rows_buf:
                f.write("**결론: GET_ROWS HTP 할당 성공** — input embedding NPU 오프로드 가능.\n")
            elif get_rows_be == "CPU" or get_rows_buf in ("CPU", "CPU_REPACK", "unknown"):
                f.write("**결론: GET_ROWS CPU 고정** — tied/untied 무관하게 LAYER_INPUT → cpu_buft_list 경로.\n")
                f.write("HTP 오프로드를 위해서는 dev_input 버퍼 목록에 HTP 추가 필요.\n")
            else:
                f.write(f"**결론: 불확실** (GET_ROWS backend={get_rows_be}, buft={get_rows_buf})\n")

            if output_wt == "FOUND":
                f.write("- output.weight 존재 → MUL_MAT은 별도 HTP0 텐서 사용\n")
            else:
                f.write("- output.weight 없음 → token_embd.weight TENSOR_DUPLICATED → MUL_MAT=HTP0\n")
                if mulmat_buf not in ("unknown",):
                    f.write(f"- token_embd.weight MUL_MAT 배치: {mulmat_buf}\n")
            f.write("\n")

        # Known limitations
        f.write("## Known Limitations\n\n")
        f.write("- **GET_ROWS CPU 고정**: llama-model.cpp에서 token_embd.weight가 `LAYER_INPUT` → `dev_input` → `cpu_buft_list`로 배정. cpu_buft_list에는 HTP0/HTP0-REPACK이 없음.\n")
        f.write("- **tied_off 메모리 증가**: output.weight 복제로 token_embd.weight 크기만큼 추가 메모리 필요.\n")
        f.write("- **FORCE_GET_ROWS_HTP**: TENSOR_DUPLICATED 없이 GET_ROWS를 HTP에 강제 배치하는 실험 패치는 별도 구현 (FORCE_GET_ROWS_HTP=1 macro).\n")
        f.write("- **converter limitation**: convert_hf_to_gguf.py의 EXAONE4 converter는 tie_word_embeddings 명시적 처리 없음. tied_off GGUF는 post-processing으로 생성.\n")

        # TODO
        f.write("\n## TODO\n\n")
        f.write("- [ ] FORCE_GET_ROWS_HTP 패치 적용 후 re-run하여 GET_ROWS HTP 할당 가능 여부 확인\n")
        f.write("- [ ] Q4_0 variant로 동일 실험 반복 (tied_on vs tied_off)\n")
        f.write("- [ ] tied_on에서 token_embd.weight가 내부적으로 CPU/HTP 양쪽에 복제되는지 메모리 사용량으로 확인\n")
        f.write("- [ ] dev_input에 HTP buffer 추가하는 패치 작성 (FORCE_EMB_HTP)\n")

    print(f"[OK] summary.md → {out_path}")


if __name__ == "__main__":
    main()
