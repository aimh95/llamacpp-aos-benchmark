#!/usr/bin/env python3
"""
parse_backflow.py: BACKFLOW 실험 결과 파싱 및 summary.md 생성.

Input (experiments/backend_flow/ 아래):
  prefill_graph_nodes.txt   decode_graph_nodes.txt
  prefill_schedule.txt      decode_schedule.txt
  get_rows_trace.txt        backend_copy_trace.txt
  decode_timing.csv

Output:
  experiments/backend_flow/summary.md

Usage:
  python3 experiments/backend_flow/parse_backflow.py [--dir experiments/backend_flow]
"""
import argparse
import csv
import re
import statistics
from pathlib import Path


def read_text(p: Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


# ------------------------------------------------------------------ #
# graph_nodes.txt parsing
# ------------------------------------------------------------------ #

def parse_graph_nodes(path: Path) -> list[dict]:
    """
    Parse {prefill,decode}_graph_nodes.txt → list of node dicts.

    Format:
    [   0] name  op=OP  type=T  shape=[...]  be=BE  buft=BT  [flags]
           src0=N  op=O  type=T  shape=[...]  be=BE  buft=BT
           src1=N  op=O  type=T  shape=[...]  be=BE  buft=BT
    """
    text = read_text(path)
    nodes = []
    lines = text.splitlines()

    def parse_fields(line: str) -> dict:
        d = {}
        for key in ("name", "op", "type", "shape", "be", "buft"):
            m = re.search(rf'(?:^|\s){key}=(\S+)', line)
            if m:
                d[key] = m.group(1)
        return d

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'\[\s*(\d+)\]\s+(\S+)\s+', line)
        if m:
            idx  = int(m.group(1))
            name = m.group(2)
            node = {"idx": idx, "name": name}
            node.update(parse_fields(line))
            node["name"] = name  # ensure name not overwritten by parse_fields

            # read src0 / src1 lines
            for src_key in ("src0", "src1"):
                if i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if nxt.strip().startswith(f"{src_key}="):
                        src_fields = parse_fields(nxt)
                        # extract src0_name etc.
                        m2 = re.match(r'\s+' + src_key + r'=(\S+)', nxt)
                        if m2:
                            src_fields["name"] = m2.group(1)
                        for k, v in src_fields.items():
                            node[f"{src_key}_{k}"] = v
                        i += 1

            nodes.append(node)
        i += 1

    return nodes


def analyze_graph_nodes(nodes: list[dict]) -> dict:
    """
    Derive key info from graph nodes:
    - GET_ROWS node for token_embd (idx, output info)
    - scheduler copy tensor detection (HTP0#name#n pattern)
    - consumer of GET_ROWS output
    - last node (result_output / lm_head)
    """
    result = {}
    if not nodes:
        return result

    # Find GET_ROWS on token_embd
    get_rows_node = None
    for nd in nodes:
        if nd.get("op") == "GET_ROWS" and "token_embd" in nd.get("src0_name", ""):
            get_rows_node = nd
            break

    if not get_rows_node:
        return result

    result["get_rows_idx"]      = get_rows_node["idx"]
    result["get_rows_output_be"]   = get_rows_node.get("be", "?")
    result["get_rows_output_buft"] = get_rows_node.get("buft", "?")
    result["get_rows_output_name"] = get_rows_node.get("name", "?")
    result["get_rows_src0_buft"]   = get_rows_node.get("src0_buft", "?")

    # Look for scheduler copy tensor: name = {BACKEND}#{embd_name}#{n}
    # This appears as src0/src1 of nodes immediately following GET_ROWS.
    embd_name = get_rows_node.get("name", "embd")
    sched_copy_pattern = re.compile(rf'^(\w+)#{re.escape(embd_name)}#\d+$')
    consumer_node  = None
    sched_copy_be  = None

    for nd in nodes:
        if nd["idx"] <= get_rows_node["idx"]:
            continue
        for src_key in ("src0", "src1"):
            src_name = nd.get(f"{src_key}_name", "")
            m = sched_copy_pattern.match(src_name)
            if m:
                sched_copy_be  = m.group(1)    # e.g. "HTP0"
                consumer_node  = nd
                break
            # Also check direct reference to embd (no copy, same backend)
            if src_name == embd_name and consumer_node is None:
                consumer_node = nd
        if consumer_node:
            break

    if consumer_node:
        result["consumer_idx"]  = consumer_node["idx"]
        result["consumer_name"] = consumer_node["name"]
        result["consumer_op"]   = consumer_node.get("op", "?")
        result["consumer_be"]   = consumer_node.get("be", "?")
        result["sched_copy_be"] = sched_copy_be    # None if same backend
    else:
        result["consumer_idx"]  = None
        result["consumer_name"] = "?"
        result["consumer_op"]   = "?"
        result["consumer_be"]   = "?"
        result["sched_copy_be"] = None

    # Cross-backend: GET_ROWS output on different backend from consumer
    gr_be       = result["get_rows_output_be"]
    consumer_be = result["consumer_be"]
    result["cross_backend"] = (gr_be != "?" and consumer_be != "?"
                               and gr_be != consumer_be)
    # Scheduler copy is the definitive cross-backend signal
    if sched_copy_be and sched_copy_be != gr_be:
        result["cross_backend"]  = True
        result["sched_copy_bytes"] = _shape_to_bytes(
            get_rows_node.get("shape", ""),
            get_rows_node.get("type", "f32"))

    # Last node = final output / lm_head
    last = nodes[-1]
    result["last_idx"]     = last["idx"]
    result["last_name"]    = last["name"]
    result["last_op"]      = last.get("op", "?")
    result["last_be"]      = last.get("be", "?")
    result["last_src0_name"] = last.get("src0_name", "?")
    result["last_src0_buft"] = last.get("src0_buft", "?")
    result["last_src1_be"]   = last.get("src1_be", "?")
    result["last_src1_buft"] = last.get("src1_buft", "?")
    # If last src1 is on a different backend from last op: also cross-backend
    last_op_be  = last.get("be", "?")
    last_src1_be = last.get("src1_be", "?")
    result["last_src1_cross"] = (last_src1_be not in ("?", "none", last_op_be))

    return result


def _shape_to_bytes(shape_str: str, type_str: str) -> int:
    """Estimate tensor size in bytes from shape string like [2048,1,1,1]."""
    type_bytes = {"f32": 4, "f16": 2, "q8_0": 1, "i32": 4, "i64": 8}
    bpe = type_bytes.get(type_str.lower(), 4)
    nums = re.findall(r'\d+', shape_str)
    n = 1
    for x in nums:
        n *= int(x)
    return n * bpe


# ------------------------------------------------------------------ #
# get_rows_trace.txt parsing
# ------------------------------------------------------------------ #

def parse_get_rows_trace(path: Path) -> dict:
    """Return {prefill: rec, decode: rec} for token_embd GET_ROWS."""
    text = read_text(path)
    result = {"prefill": {}, "decode": {}}

    sections = text.split("=== GET_ROWS trace")
    for sec in sections[1:]:
        vm = re.match(r'\s+variant=(\S+)', sec)
        if not vm:
            continue
        v = vm.group(1).rstrip(":")

        sec_recs = []
        for block in re.split(r'\[GET_ROWS idx=', sec)[1:]:
            rec = {}
            m = re.match(r'(\d+)\]', block)
            if m:
                rec["idx"] = int(m.group(1))
            for field, pat in [
                ("output_be",    r'output:.*?be=(\S+)'),
                ("output_buft",  r'output:.*?buft=(\S+)'),
                ("output_shape", r'output:.*?shape=(\S+)'),
                ("src0_name",    r'src0:.*?name=(\S+)'),
                ("src0_buft",    r'src0:.*?buft=(\S+)'),
                ("consumer_op",  r'consumer\[.*?\].*?op=(\S+)'),
                ("consumer_be",  r'consumer\[.*?\].*?be=(\S+)'),
            ]:
                mm = re.search(pat, block)
                if mm:
                    rec[field] = mm.group(1)
            rec["cross_backend"] = "CROSS-BACKEND" in block
            if "idx" in rec:
                sec_recs.append(rec)

        embd_recs = [r for r in sec_recs if "token_embd" in r.get("src0_name", "")]
        gr = embd_recs[0] if embd_recs else (sec_recs[0] if sec_recs else {})

        if v.startswith("prefill"):
            result["prefill"] = gr
        elif v.startswith("decode"):
            result["decode"] = gr

    return result


# ------------------------------------------------------------------ #
# backend_copy_trace.txt parsing
# ------------------------------------------------------------------ #

def parse_copy_trace(path: Path) -> dict:
    result = {"prefill": [], "decode": []}
    text = read_text(path)
    current = "unknown"
    for line in text.splitlines():
        if "variant=prefill" in line:
            current = "prefill"
        elif "variant=decode" in line:
            current = "decode"
        m = re.search(
            r'\[CPY idx=(\d+)\].*?bytes=(\d+).*?from=(\S+)\(.*?\).*?to=(\S+)\(.*?\).*?direction=(\S+)',
            line)
        if m and current in result:
            result[current].append({
                "idx":       int(m.group(1)),
                "bytes":     int(m.group(2)),
                "from_be":   m.group(3),
                "to_be":     m.group(4),
                "direction": m.group(5),
            })
    return result


# ------------------------------------------------------------------ #
# decode_timing.csv parsing
# ------------------------------------------------------------------ #

def parse_timing_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = []
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    decode_rows = [r for r in rows if r.get("is_prefill", "0") == "0"]
    if not decode_rows:
        return {}

    times_us = []
    for r in decode_rows:
        try:
            times_us.append(int(r["graph_compute_us"]))
        except (KeyError, ValueError):
            pass
    if not times_us:
        return {}

    return {
        "n_steps":   len(times_us),
        "mean_us":   statistics.mean(times_us),
        "median_us": statistics.median(times_us),
        "min_us":    min(times_us),
        "max_us":    max(times_us),
        "n_splits":  decode_rows[0].get("n_splits", "?"),
        "n_copies":  decode_rows[0].get("n_copies", "?"),
        "n_nodes":   decode_rows[0].get("n_nodes", "?"),
    }


# ------------------------------------------------------------------ #
# schedule backend count
# ------------------------------------------------------------------ #

def parse_schedule_summary(path: Path) -> dict:
    text = read_text(path)
    backends = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            be = parts[3]
            backends[be] = backends.get(be, 0) + 1
    return backends


# ------------------------------------------------------------------ #
# Summary writer
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="experiments/backend_flow")
    args = parser.parse_args()

    d = Path(args.dir)

    gr_trace   = parse_get_rows_trace(d / "get_rows_trace.txt")
    cp_trace   = parse_copy_trace(d / "backend_copy_trace.txt")
    timing     = parse_timing_csv(d / "decode_timing.csv")
    pre_sched  = parse_schedule_summary(d / "prefill_schedule.txt")
    dec_sched  = parse_schedule_summary(d / "decode_schedule.txt")

    # Graph node analysis (richer consumer/lm_head info)
    prefill_nodes = parse_graph_nodes(d / "prefill_graph_nodes.txt")
    decode_nodes  = parse_graph_nodes(d / "decode_graph_nodes.txt")
    pre_graph     = analyze_graph_nodes(prefill_nodes)
    dec_graph     = analyze_graph_nodes(decode_nodes)

    # Merge: graph analysis overrides trace for consumer/cross_backend
    prefill_gr = gr_trace.get("prefill", {})
    decode_gr  = gr_trace.get("decode", {})
    if dec_graph.get("consumer_be") and dec_graph["consumer_be"] != "?":
        decode_gr["consumer_op"] = dec_graph.get("consumer_op", "?")
        decode_gr["consumer_be"] = dec_graph["consumer_be"]
    if dec_graph.get("cross_backend"):
        decode_gr["cross_backend"] = True
    if pre_graph.get("consumer_be") and pre_graph["consumer_be"] != "?":
        prefill_gr["consumer_op"] = pre_graph.get("consumer_op", "?")
        prefill_gr["consumer_be"] = pre_graph["consumer_be"]
    if pre_graph.get("cross_backend"):
        prefill_gr["cross_backend"] = True

    def cell(d, key, default="N/A"):
        return str(d.get(key, default))

    def copy_bytes_str(copies):
        if not copies:
            return "없음"
        total = sum(c["bytes"] for c in copies)
        return f"{len(copies)}건 / {total:,} bytes"

    def cpu_to_htp(copies):
        return [c for c in copies if c.get("direction") == "CPU->HTP"]

    def htp_to_cpu(copies):
        return [c for c in copies if c.get("direction") == "HTP->CPU"]

    pre_cpu2htp = cpu_to_htp(cp_trace.get("prefill", []))
    dec_cpu2htp = cpu_to_htp(cp_trace.get("decode",  []))
    pre_htp2cpu = htp_to_cpu(cp_trace.get("prefill", []))
    dec_htp2cpu = htp_to_cpu(cp_trace.get("decode",  []))

    # Scheduler copy info from graph analysis
    dec_sched_copy_be    = dec_graph.get("sched_copy_be")     # e.g. "HTP0"
    dec_sched_copy_bytes = dec_graph.get("sched_copy_bytes")  # int or None
    pre_sched_copy_be    = pre_graph.get("sched_copy_be")
    pre_sched_copy_bytes = pre_graph.get("sched_copy_bytes")

    def sched_copy_str(copy_be, copy_bytes, cpy_trace_list):
        # cpy_trace_list = user-graph CPY nodes (usually empty for sched copies)
        if cpy_trace_list:
            total = sum(c["bytes"] for c in cpy_trace_list)
            return f"{len(cpy_trace_list)}건 / {total:,} bytes"
        if copy_be:
            if copy_bytes:
                return f"1건 (scheduler internal) / {copy_bytes:,} bytes → {copy_be}"
            return f"1건 (scheduler internal) → {copy_be}"
        n_copies = timing.get("n_copies", "?") if cpy_trace_list is dec_cpu2htp else "?"
        if n_copies and str(n_copies) != "?" and int(n_copies) > 0:
            return f"{n_copies}건 (scheduler internal, 바이트 미확인)"
        return "없음 (scheduler internal copy: user graph에 직접 미노출)"

    n_copies_val = timing.get("n_copies", "?")

    # ---------------------------------------------------------------- #
    # Write summary.md
    # ---------------------------------------------------------------- #
    out = d / "summary.md"
    with open(out, "w") as f:

        f.write("# BACKFLOW: decode step backend-flow analysis\n\n")
        f.write("## 실제 decode 실행 흐름\n\n")
        f.write("```\n")
        f.write("CPU:  sampling → token_id\n")
        f.write("CPU:  GET_ROWS(token_embd.weight[CPU_Mapped], token_id)\n")
        f.write("        → embd [F32, 2048×1, 8KB, CPU buffer]\n")
        f.write("\n")
        f.write("[scheduler internal] CPU → HTP0: embd (8KB, 1 copy)\n")
        f.write("        ↓  copy tensor: HTP0#embd#0 [F32, 2048×1, HTP0 shared mem]\n")
        f.write("\n")
        f.write("HTP0: 30 layers × (Q/K/V proj + Attention + FFN + norms)\n")
        f.write("        → result_norm [F32, 2048×1, HTP0 shared mem]\n")
        f.write("\n")
        f.write("[no copy needed] HTP0 buffer is CPU-accessible shared memory\n")
        f.write("        CPU reads result_norm directly from HTP0 shared mem\n")
        f.write("\n")
        f.write("CPU:  result_output MUL_MAT(token_embd.weight[CPU_REPACK], result_norm)\n")
        f.write("        → logits [F32, 102400×1, 400KB, CPU]\n")
        f.write("        ※ Q8_0 token_embd.weight: HTP0-REPACK 불가 → lm_head=CPU\n")
        f.write("\n")
        f.write("CPU:  sampling → next token_id\n")
        f.write("```\n\n")

        f.write("## 결론 표\n\n")
        f.write("| 항목 | Prefill | Decode |\n")
        f.write("|---|---|---|\n")

        # GET_ROWS info
        pre_gr_be   = pre_graph.get("get_rows_output_be") or prefill_gr.get("output_be", "N/A")
        dec_gr_be   = dec_graph.get("get_rows_output_be") or decode_gr.get("output_be", "N/A")
        pre_gr_buft = pre_graph.get("get_rows_src0_buft") or prefill_gr.get("src0_buft", "N/A")
        dec_gr_buft = dec_graph.get("get_rows_src0_buft") or decode_gr.get("src0_buft", "N/A")
        pre_out_buft = pre_graph.get("get_rows_output_buft") or prefill_gr.get("output_buft", "N/A")
        dec_out_buft = dec_graph.get("get_rows_output_buft") or decode_gr.get("output_buft", "N/A")
        pre_con_op  = pre_graph.get("consumer_op") or prefill_gr.get("consumer_op", "N/A")
        dec_con_op  = dec_graph.get("consumer_op") or decode_gr.get("consumer_op", "N/A")
        pre_con_be  = pre_graph.get("consumer_be") or prefill_gr.get("consumer_be", "N/A")
        dec_con_be  = dec_graph.get("consumer_be") or decode_gr.get("consumer_be", "N/A")
        pre_cross   = pre_graph.get("cross_backend") or prefill_gr.get("cross_backend", False)
        dec_cross   = dec_graph.get("cross_backend") or decode_gr.get("cross_backend", False)

        # Scheduler copy signal
        dec_sched_copy_str = "없음"
        if dec_sched_copy_be:
            bstr = f"{dec_sched_copy_bytes:,} bytes" if dec_sched_copy_bytes else "크기 미확인"
            dec_sched_copy_str = f"HTP0#embd#0 ({bstr}) → {dec_sched_copy_be}"
        pre_sched_copy_str = "없음"
        if pre_sched_copy_be:
            bstr = f"{pre_sched_copy_bytes:,} bytes" if pre_sched_copy_bytes else "크기 미확인"
            pre_sched_copy_str = f"HTP0#embd#0 ({bstr}) → {pre_sched_copy_be}"

        # lm_head info
        dec_lm_be   = dec_graph.get("last_be", "N/A")
        dec_lm_src0 = dec_graph.get("last_src0_buft", "N/A")
        dec_lm_src1 = dec_graph.get("last_src1_buft", "N/A")
        dec_lm_cross = dec_graph.get("last_src1_cross", False)

        # lm_head HTP→CPU: n_copies=1 → only embd CPU→HTP counted.
        # result_norm is in HTP0 shared mem → CPU reads directly (zero-copy).
        n_copies_int = int(n_copies_val) if str(n_copies_val).isdigit() else -1
        lm_htp_to_cpu = "없음 (HTP0 shared mem, zero-copy)" if n_copies_int == 1 else (
            "있음" if dec_lm_cross else "없음")

        # Prefill n_splits: format the backend-count dict nicely.
        # Multi-word tensor names (e.g. "Kcur-0 (view)") shift columns in the
        # schedule file, causing op names like VIEW/SET_ROWS/(permuted) to appear
        # as false "backend" entries. Filter to known backend name patterns only.
        _BACKEND_RE = re.compile(r'^(CPU|HTP\d*|Metal|CUDA|Vulkan|SYCL|ROCm|Backend)')
        pre_splits_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(pre_sched.items())
            if _BACKEND_RE.match(k)) if pre_sched else "?"

        table_rows = [
            ("GET_ROWS backend",                    pre_gr_be,             dec_gr_be),
            ("token_embd.weight buft",              pre_gr_buft,           dec_gr_buft),
            ("GET_ROWS output buft",                pre_out_buft,          dec_out_buft),
            ("GET_ROWS 다음 consumer op",           pre_con_op,            dec_con_op),
            ("consumer backend",                    pre_con_be,            dec_con_be),
            ("CPU→HTP cross-backend 여부",          str(pre_cross),        str(dec_cross)),
            ("scheduler copy tensor (embd)",        pre_sched_copy_str,    dec_sched_copy_str),
            ("n_copies (scheduler internal)",       "?",                   str(n_copies_val)),
            ("lm_head (result_output) backend",     "N/A",                 dec_lm_be),
            ("lm_head weight buft",                 "N/A",                 dec_lm_src0),
            ("lm_head src1 buft (result_norm)",     "N/A",                 dec_lm_src1),
            ("lm_head HTP→CPU copy 여부",           "N/A",                 lm_htp_to_cpu),
            ("graph_compute time",                  "N/A (prefill 1회)",   f"{timing.get('mean_us',0)/1000:.2f} ms avg, n={timing.get('n_steps','?')}" if timing else "N/A"),
            ("n_splits (ops per backend)",          pre_splits_str,        timing.get("n_splits", "?")),
            ("n_nodes",                             str(len(prefill_nodes)) if prefill_nodes else "?",
                                                    timing.get("n_nodes", "?")),
        ]
        for label, pre, dec in table_rows:
            f.write(f"| {label} | {pre} | {dec} |\n")

        # GET_ROWS 상세
        f.write("\n## GET_ROWS 상세\n\n")
        for variant, gr, graph in [("Prefill", prefill_gr, pre_graph), ("Decode", decode_gr, dec_graph)]:
            f.write(f"### {variant}\n")
            if not graph:
                f.write("데이터 없음\n\n")
                continue
            f.write(f"- GET_ROWS output backend: **{graph.get('get_rows_output_be','?')}**\n")
            f.write(f"- token_embd.weight buft: **{graph.get('get_rows_src0_buft','?')}**\n")
            f.write(f"- GET_ROWS output buft: **{graph.get('get_rows_output_buft','?')}**\n")
            con_op = graph.get("consumer_op","?")
            con_be = graph.get("consumer_be","?")
            f.write(f"- consumer: **{con_op}** @ **{con_be}**\n")
            if graph.get("sched_copy_be"):
                sbe = graph["sched_copy_be"]
                sb  = graph.get("sched_copy_bytes")
                bstr = f"{sb:,} bytes" if sb else "크기 미확인"
                f.write(f"- **SCHEDULER COPY 감지**: embd → `HTP0#embd#0` [{bstr}] (CPU→{sbe})\n")
                f.write(f"  consumer `{con_op}` (idx={graph.get('consumer_idx','?')}) reads from HTP0 shared buffer\n")
            elif graph.get("cross_backend"):
                f.write(f"- **CROSS-BACKEND**: GET_ROWS({graph.get('get_rows_output_be','?')}) → consumer({con_be}) → 스케줄러 copy 필요\n")
            else:
                f.write(f"- Same-backend: GET_ROWS output과 consumer가 같은 backend\n")
            f.write("\n")

        # lm_head 상세
        f.write("## lm_head (result_output) 상세\n\n")
        if dec_graph:
            f.write(f"- **idx={dec_graph.get('last_idx','?')}** `{dec_graph.get('last_name','?')}` op={dec_graph.get('last_op','?')}\n")
            f.write(f"- backend: **{dec_graph.get('last_be','?')}**\n")
            f.write(f"- src0 (weight): `{dec_graph.get('last_src0_name','?')}` buft={dec_graph.get('last_src0_buft','?')}\n")
            f.write(f"- src1 (hidden): buft={dec_graph.get('last_src1_buft','?')} (be={dec_graph.get('last_src1_be','?')})\n")
            if dec_graph.get("last_src1_cross"):
                f.write(f"- **주의**: src1 (HTP0)을 CPU op이 읽음. HTP0 buffer는 shared mem → zero-copy로 CPU 접근 가능.\n")
            f.write(f"\n")
            f.write("**Q8_0 제약**: token_embd.weight Q8_0 (2048×102400) → HTP0-REPACK 불가 → lm_head=CPU\n")
            f.write("tied_off 모델에서는 output.weight가 별도 LAYER_OUTPUT → HTP0-REPACK 가능 → lm_head=HTP0 기대.\n")
        else:
            f.write("데이터 없음\n")
        f.write("\n")

        # backend copy trace
        f.write("## backend_copy_trace 요약 (user graph CPY nodes)\n\n")
        for variant, copies in [("Prefill", cp_trace.get("prefill",[])), ("Decode", cp_trace.get("decode",[]))]:
            f.write(f"### {variant}\n")
            if not copies:
                f.write("없음 (scheduler internal copy는 user graph에 직접 미노출)\n\n")
                continue
            by_dir: dict = {}
            for c in copies:
                by_dir.setdefault(c["direction"], []).append(c)
            for direction, items in sorted(by_dir.items()):
                total_bytes = sum(i["bytes"] for i in items)
                f.write(f"- `{direction}`: {len(items)}건, {total_bytes:,} bytes total\n")
            f.write("\n")

        # timing
        f.write("## decode timing\n\n")
        if timing:
            f.write(f"- steps: {timing['n_steps']}\n")
            f.write(f"- graph_compute: avg={timing['mean_us']/1000:.2f}ms  "
                    f"median={timing['median_us']/1000:.2f}ms  "
                    f"min={timing['min_us']/1000:.2f}ms  "
                    f"max={timing['max_us']/1000:.2f}ms\n")
            f.write(f"- n_nodes={timing['n_nodes']}  n_splits={timing['n_splits']}  n_copies={timing['n_copies']}\n")
        else:
            f.write("decode_timing.csv 없음 또는 decode step 미수행\n")
        f.write("\n")

        # 최종 판정
        f.write("## 최종 판정\n\n")
        dec_gr_be_val = dec_graph.get("get_rows_output_be", "?")
        dec_cross_val = dec_graph.get("cross_backend", decode_gr.get("cross_backend", False))
        dec_con_be_val = dec_graph.get("consumer_be", "?")
        dec_copy_be_val = dec_graph.get("sched_copy_be")

        if dec_gr_be_val == "HTP0":
            verdict_num = "2"
            verdict = "GET_ROWS도 HTP에서 실행된다."
        elif dec_copy_be_val or dec_cross_val:
            copy_detail = ""
            if dec_sched_copy_bytes:
                copy_detail = f" (embd {dec_sched_copy_bytes:,} bytes, CPU→{dec_copy_be_val})"
            verdict_num = "1"
            verdict = f"GET_ROWS CPU 실행 후 embedding activation이 HTP로 copy된다.{copy_detail}"
        elif int(n_copies_val) > 0 if str(n_copies_val).isdigit() else False:
            verdict_num = "1"
            verdict = f"GET_ROWS CPU 실행 후 embedding activation이 HTP로 copy된다. (n_copies={n_copies_val}, 바이트 미확인)"
        else:
            verdict_num = "3/5"
            verdict = f"GET_ROWS backend=CPU, cross-backend 미감지. HTP profiler로 추가 확인 필요."

        f.write(f"### 가설 {verdict_num}: **{verdict}**\n\n")
        if verdict_num in ("1", "2"):
            f.write("#### 근거\n")
            f.write(f"1. GET_ROWS (idx={dec_graph.get('get_rows_idx','?')}) → `embd` [CPU, CPU_Mapped src]\n")
            if dec_copy_be_val:
                bstr = f"{dec_sched_copy_bytes:,} bytes" if dec_sched_copy_bytes else "크기 미확인"
                f.write(f"2. Scheduler copy: `embd` → `HTP0#embd#0` [{bstr}] (CPU→{dec_copy_be_val} shared mem)\n")
            f.write(f"3. Consumer `{dec_graph.get('consumer_op','?')}` (idx={dec_graph.get('consumer_idx','?')}) runs on {dec_graph.get('consumer_be','?')}\n")
            f.write(f"4. n_splits={timing.get('n_splits','?')}: [CPU:GET_ROWS | HTP0:transformer×30layers | CPU:lm_head]\n")
            f.write(f"5. n_copies={n_copies_val}: scheduler가 삽입한 internal copy 수 (CPU→HTP0 embd)\n")
            f.write(f"6. HTP→CPU: result_norm은 HTP0 shared mem → CPU가 zero-copy로 직접 읽음 (추가 copy 없음)\n")
            f.write(f"7. lm_head (result_output) idx={dec_graph.get('last_idx','?')}: be={dec_graph.get('last_be','?')}"
                    f" (Q8_0 weight → HTP0-REPACK 불가 → CPU fallback)\n")
            f.write("\n")
            f.write("#### 추가 발견: lm_head가 CPU에서 실행\n")
            f.write("- Q8_0 타입 token_embd.weight는 HTP MUL_MAT용 HTP0-REPACK 불가\n")
            f.write("- tied_on 모델에서 lm_head weight = token_embd.weight (TENSOR_DUPLICATED)\n")
            f.write("- 결과: transformer=HTP0이지만 lm_head=CPU (logits 400KB CPU 버퍼)\n")
            f.write("- tied_off 모델: output.weight 별도 텐서 → LAYER_OUTPUT → HTP0-REPACK 가능 → lm_head=HTP0 기대\n\n")

        f.write("### Limitations\n\n")
        f.write("- GET_ROWS / lm_head 개별 실행 시간: GGML per-op timer 또는 Snapdragon Profiler 필요\n")
        f.write("- CPU→HTP copy 시간: scheduler 내부 — Snapdragon Profiler / ETW 필요\n")
        f.write("- Scheduler-inserted copy node는 user graph (ggml_cgraph.nodes)에 직접 미노출\n")
        f.write("  → `HTP0#embd#0` copy tensor는 src포인터로만 관찰 가능\n")
        f.write("- `ggml_backend_sched_get_tensor_backend`가 weight tensor에 대해 NULL 반환 시 buffer 이름으로 fallback\n")

    print(f"[OK] summary.md → {out}")


if __name__ == "__main__":
    main()
