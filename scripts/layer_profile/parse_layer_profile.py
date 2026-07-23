#!/usr/bin/env python3
"""Phase-1 (무패치) layer-profile 로그 파서.

capture_layer_profile.sh 가 남긴 raw 로그를 읽어 CSV 4종을 만든다.
  - assign_<cfg>_<mode>.log : GGML_SCHED_DEBUG=2 (+OPTRACE)  → assigned backend / splits
  - htpprof_<cfg>_HTP0.log  : GGML_HEXAGON_PROFILE=1         → HTP per-op usec
  - bench_<cfg>_<mode>.log  : llama-bench                     → graph-level prefill/decode t/s

출력:
  raw_backend_assignment.csv, backend_transition.csv, raw_node_timing.csv, layer_summary.csv

로그 포맷(확인됨):
  node 라인: "node #  1 (   MUL_MAT):  Qcur-0 (   8K) [ HTP0 ] use=1,c=1: blk.0.attn_q.weight ( 2M) [ HTP0 ] ..."
  split 라인: "## SPLIT #1: HTP0 # 6 inputs : [embd (8K)] [leaf_5 (0K)] ..."
  htp prof:  "ggml-hex: <sess> profile-op <OP>: ... : usec <U> cycles <C> ..."
             "ggml-hex: <sess> profile-batch n-ops <N> batch-dur-usec <D> htp-ops-usec <U>"
  ※ 로그가 GGML_LOG_DEBUG 이어붙임 때문에 한 줄에 여러 record 가 섞일 수 있어, 'node #'/'## SPLIT' 기준으로 재분할한다.
  ※ 실제 첫 실행 로그로 정규식 미세보정 필요할 수 있음(스페이싱/버전차).
"""
from __future__ import annotations
import argparse, csv, glob, os, re
from collections import defaultdict, OrderedDict

BLK = re.compile(r"blk\.(\d+)\.")
SUF = re.compile(r"-(\d+)(?:$|[^0-9])")
# 중간 텐서 출력명 -> role (weight 이름이 %20.20s 로 잘려 안 잡힐 때의 fallback)
_ROLE_OUT = [
    ("Qcur", "attn_q"), ("Kcur", "attn_k"), ("Vcur", "attn_v"),
    ("kqv_out", "attn_output"), ("kqv_merged", "attn_output"), ("attn_out", "attn_output"),
    ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"), ("ffn_par", "ffn_gate"),
    ("ffn_out", "ffn_down"), ("ffn_swiglu", "ffn_down"), ("ffn_moe", "ffn_down"),
]

def logical(name: str):
    """tensor/node 이름 -> (block_index, logical_layer, role). weight명 또는 출력명 모두 처리."""
    if not name:
        return "", "unknown", "other"
    if name.startswith("token_embd") or name in ("embd", "inp_embd", "inp_tokens", "inpL"):
        return "", "embedding", "token_embd"
    if name.startswith("output_norm") or name.startswith("result_norm"):
        return "", "output_norm", "norm"
    if name.startswith("result_output") or name.startswith("output.weight") or name == "result_output":
        return "", "lm_head", "output"
    # block index: blk.N 우선, 없으면 -N 접미사
    bi = ""
    m = BLK.search(name)
    if m:
        bi = int(m.group(1))
    else:
        m2 = SUF.search(name)
        if m2:
            bi = int(m2.group(1))
    # role: 정식 weight 이름 우선
    role = "other"
    for r in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        if f".{r}." in name:
            role = r
            break
    if role == "other":
        for pat, rr in _ROLE_OUT:
            if name.startswith(pat):
                role = rr
                break
    if role == "other" and "norm" in name:
        role = "norm"
    lname = f"blk.{bi}" if bi != "" else name
    return bi, lname, role

# node 세그먼트: "node #<idx> (<OP>): <out> (<sz>) [<backend>] ..."
NODE = re.compile(r"node #\s*(\d+)\s*\(\s*([A-Z_0-9]+)\s*\)\s*:\s*(\S+)\s*\([^)]*\)\s*\[\s*([A-Za-z0-9_]+)\s*\]")
# 세그먼트 내부의 (tensor (sz) [backend]) 조각들 (src 포함)
TB = re.compile(r"(\S+)\s*\(\s*[0-9.]+[KMG]?\s*\)\s*\[\s*([A-Za-z0-9_]+)\s*\]")
SPLIT = re.compile(r"##\s*SPLIT #(\d+)\s*:\s*([A-Za-z0-9_]+)\s*#\s*(\d+)\s*inputs")
# HTP profile-op: op 이름과 usec/cycles 만 견고하게 추출
POP = re.compile(r"profile-op\s+([A-Z_0-9+]+).*?usec\s+(\d+)\s+cycles\s+(\d+)")
PBATCH = re.compile(r"profile-batch\s+n-ops\s+(\d+)\s+batch-dur-usec\s+(\d+)\s+htp-ops-usec\s+(\d+)")
# bench md row: "| model | size | params | backend | ngl | dev | test | t/s |"
BENCH = re.compile(r"\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|\s*$")

def read(path):
    try:
        return open(path, errors="replace").read()
    except FileNotFoundError:
        return ""

# ---- [OPTRACE][HTP]: HTP0 세션에 배정된 op 목록 (full name, 그래프당 1회) ----
OPT_HDR = re.compile(r"\[OPTRACE\]\[HTP\]\s+\S+\s+n_htp_ops=(\d+)\s+graph_uid=(\d+)")
OPT_OP = re.compile(
    r'\[OPTRACE\]\[HTP\]\s+idx=(\d+)\s+op=(\S+)\s+dst="([^"]*)"\s+dst_type=(\S+)\s+dst_buft=(\S+)'
    r'(?:.*?src0="([^"]*)".*?src0_buft=(\S+))?'
)
# profile-op 포맷: profile-op <OP>|<names>|<dims>|<types>|<strides>|<kparams>|usec U cycles C ...
#  names 의 "-> <dst>" 가 OPTRACE dst 와 매칭. dims 의 마지막 "-> A:B" 의 B = 출력 토큰수(phase 판정).
#  주의: HTP 는 op 를 융합(MUL_MAT+MUL_MAT...)하며 dst 는 융합 결과의 마지막 텐서 1개.
PROF = re.compile(r"profile-op\s+([^|]+)\|([^|]*)\|([^|]*)\|.*?usec\s+(\d+)\s+cycles\s+(\d+)")
_DST = re.compile(r"->\s*([^\s|]+)")
_NTOK = re.compile(r"->\s*\d+:(\d+)")
DECODE_MAX_TOK = 8  # 출력 토큰 <=8 이면 decode, 그 이상은 prefill

def parse_profileop(text):
    """returns list of dict(op, dst, ntok, phase, usec, cycles)."""
    rows = []
    for m in PROF.finditer(text):
        op, names, dims = m.group(1).strip(), m.group(2), m.group(3)
        dmatch = _DST.findall(names)
        dst = dmatch[-1] if dmatch else ""
        tks = _NTOK.findall(dims)
        ntok = int(tks[-1]) if tks else 0
        phase = "decode" if 0 < ntok <= DECODE_MAX_TOK else "prefill"
        rows.append(dict(op=op, dst=dst, ntok=ntok, phase=phase,
                         usec=int(m.group(4)), cycles=int(m.group(5))))
    return rows

def parse_optrace_htp(text):
    """[OPTRACE][HTP] 블록들 중 op 수가 가장 많은(=완전한) 그래프 하나를 파싱.
    returns list of dict(idx, op, dst, dst_type, dst_buft, src0, src0_buft)."""
    hdrs = [(m.start(), int(m.group(1))) for m in OPT_HDR.finditer(text)]
    if not hdrs:
        return []
    # 가장 큰 n_htp_ops 블록의 시작 위치 선택
    hdrs.sort(key=lambda x: -x[1])
    start = hdrs[0][0]
    # 선택한 블록의 끝 = 그 다음 헤더 시작 위치
    ends = [m.start() for m in OPT_HDR.finditer(text) if m.start() > start]
    end = min(ends) if ends else len(text)
    seg = text[start:end]
    ops = []
    seen = set()
    for m in OPT_OP.finditer(seg):
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        ops.append(dict(idx=idx, op=m.group(2), dst=m.group(3), dst_type=m.group(4),
                        dst_buft=m.group(5), src0=m.group(6) or "", src0_buft=m.group(7) or ""))
    ops.sort(key=lambda x: x["idx"])
    return ops

# [LAYERPROF] (Phase-2 패치): CPU 백엔드 per-op 시간. g=그래프실행카운터, dst=출력명, us=시간.
LPROF = re.compile(r'\[LAYERPROF\]\s+g=(\d+)\s+op=(\S+)\s+dst="([^"]*)"\s+us=(\d+)\s+nf=(\d+)')

def parse_layerprof(text):
    """returns dst -> {'prefill':[us...], 'decode':[us...]}.
    phase 는 그래프(g)별 총 us 로 분류: 최대 총합 g = prefill, 나머지 = decode."""
    byg = defaultdict(list)
    for m in LPROF.finditer(text):
        byg[int(m.group(1))].append((m.group(3), int(m.group(4))))
    if not byg:
        return {}
    totals = {g: sum(u for _, u in rows) for g, rows in byg.items()}
    prefill_g = max(totals, key=totals.get)
    res = defaultdict(lambda: {"prefill": [], "decode": []})
    for g, rows in byg.items():
        ph = "prefill" if g == prefill_g else "decode"
        for dst, us in rows:
            res[dst][ph].append(us)
    return res

def parse_assign(text):
    """returns (nodes[list of dict], splits[list of dict])."""
    nodes = []
    # 'node #' 로 세그먼트 분할 (여러 record 가 한 물리적 줄에 이어붙어도 처리)
    segs = re.split(r"(?=node #\s*\d+\s*\()", text)
    for seg in segs:
        m = NODE.search(seg)
        if not m:
            continue
        idx, op, out, out_be = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        # 이 세그먼트의 tensor/backend 조각들 = out + srcs. 첫번째는 out, 이후는 src.
        pairs = TB.findall(seg)
        srcs = [p for p in pairs if p[0] != out]
        weight = next((n for (n, be) in srcs if ".weight" in n), "")
        wbe = next((be for (n, be) in srcs if ".weight" in n), "")
        bi, lname, role = logical(weight or out)
        nodes.append(OrderedDict(
            node_index=idx, op_type=op, node_name=out, assigned_backend=out_be,
            weight_tensor=weight, weight_backend=wbe,
            src_tensors=";".join(n for n, _ in srcs),
            block_index=bi, logical_layer=lname, tensor_role=role,
        ))
    splits = []
    for m in SPLIT.finditer(text):
        splits.append(OrderedDict(split_index=int(m.group(1)),
                                  executed_backend=m.group(2), n_inputs=int(m.group(3))))
    return nodes, splits

def parse_htpprof(text):
    """returns per-op HTP timing aggregated by op (name 미포함이 흔하므로 op 단위)."""
    rows = []
    for m in POP.finditer(text):
        rows.append(dict(op_type=m.group(1), usec=int(m.group(2)), cycles=int(m.group(3))))
    batches = [dict(n_ops=int(a), batch_dur_usec=int(b), htp_ops_usec=int(c))
               for (a, b, c) in PBATCH.findall(text)]
    return rows, batches

def parse_bench(text):
    """returns dict: test('pp255'/'tg64') -> (median_tps, stdev)."""
    out = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        mb = BENCH.search(line)
        if not mb:
            continue
        if "pp" in line:
            key = "prefill"
        elif "tg" in line:
            key = "decode"
        else:
            continue
        out[key] = (float(mb.group(1)), float(mb.group(2)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, O = a.raw, a.out
    os.makedirs(O, exist_ok=True)

    configs = []  # (cfg, mode)
    for f in sorted(glob.glob(os.path.join(R, "assign_*.log"))):
        b = os.path.basename(f)[len("assign_"):-4]
        cfg, mode = b.rsplit("_", 1)
        configs.append((cfg, mode))

    assign_rows, trans_rows = [], []
    for cfg, mode in configs:
        nodes, splits = parse_assign(read(os.path.join(R, f"assign_{cfg}_{mode}.log")))
        # split 실행 backend 를 node 에 매핑: node_index 오름차순으로 split 경계 배정
        # (split_index 는 순서대로, i_start 정보는 무패치 로그에 없으므로 순서 기반 근사)
        for n in nodes:
            n2 = OrderedDict(run_config=f"{cfg}_{mode}", quantization=cfg, execution_mode=mode)
            n2.update(n)
            n2["executed_backend_hint"] = n["assigned_backend"]  # 무패치: assigned=executed 가정(HTP미지원op는 split이 CPU로 분리됨)
            assign_rows.append(n2)
        # transitions: 인접 split backend 변화
        prev = None
        for s in splits:
            if prev is not None and prev != s["executed_backend"]:
                trans_rows.append(OrderedDict(run_config=f"{cfg}_{mode}", quantization=cfg,
                    execution_mode=mode, split_index=s["split_index"],
                    src_backend=prev, dst_backend=s["executed_backend"], n_inputs=s["n_inputs"]))
            prev = s["executed_backend"]

    # dedup: 로그에 그래프가 여러 번(reserve+prefill+decode) 찍혀 노드가 중복됨 → (config,node_name,op) 첫 등장만
    seen = set(); dd = []
    for r in assign_rows:
        k = (r["run_config"], r["node_name"], r["op_type"])
        if k in seen:
            continue
        seen.add(k); dd.append(r)
    assign_rows = dd
    seen = set(); dt = []
    for r in trans_rows:
        k = (r["run_config"], r["split_index"], r["src_backend"], r["dst_backend"])
        if k in seen:
            continue
        seen.add(k); dt.append(r)
    trans_rows = dt

    # ---- op_detail.csv + layer_timing.csv : [OPTRACE][HTP] 맵 + profile-op (phase 분리) ----
    def med(xs):
        xs = sorted(xs); return xs[len(xs)//2] if xs else ""
    op_rows, lt_rows, role_rows, batch_rows = [], [], [], []
    for cfg, mode in configs:
        if mode != "HTP0":
            continue  # OPTRACE[HTP] 는 HTP 세션 op 만; CPU모드는 assignment(전부 CPU)
        ops = parse_optrace_htp(read(os.path.join(R, f"assign_{cfg}_{mode}.log")))
        prof = parse_profileop(read(os.path.join(R, f"htpprof_{cfg}_HTP0.log")))
        # dst -> (block, role)
        dst_meta = {}
        for o in ops:
            bi, lname, role = logical(o["src0"] or o["dst"])
            if role == "other":
                bi, lname, role = logical(o["dst"])
            dst_meta[o["dst"]] = (bi, lname, role)
        # (dst, phase) -> [usec]
        by = defaultdict(list)
        for p in prof:
            by[(p["dst"], p["phase"])].append(p["usec"])
        # op_detail: op 별 prefill/decode usec (dst 매칭, median)
        for o in ops:
            bi, lname, role = dst_meta[o["dst"]]
            be = "HTP0" if o["dst_buft"].startswith("HTP0") else o["dst_buft"]
            pf, dc = med(by.get((o["dst"], "prefill"), [])), med(by.get((o["dst"], "decode"), []))
            op_rows.append(OrderedDict(
                quantization=cfg, execution_mode=mode, idx=o["idx"], op_type=o["op"],
                dst=o["dst"], dst_type=o["dst_type"], dst_buft=o["dst_buft"],
                src0=o["src0"], src0_buft=o["src0_buft"],
                block_index=bi, logical_layer=lname, tensor_role=role, assigned_backend=be,
                prefill_us=pf, decode_us=dc,
                timing_source=("profile-op(name)" if (pf != "" or dc != "") else "확인불가"),
                fused_note="HTP op-fusion 가능(dst 1개에 여러 matmul 융합) — 값은 융합그룹 기준일 수 있음",
            ))
        # block/role 별 집계: 각 dst median 을 block/role 로 합산 (iteration당 시간)
        blk = defaultdict(lambda: defaultdict(list))   # (block,phase)->dst->[usec]
        rol = defaultdict(lambda: defaultdict(list))   # (role,phase)->dst->[usec]
        for (dst, ph), us in by.items():
            bi, _, role = dst_meta.get(dst, ("", "", "other"))
            blk[(bi, ph)][dst] = us
            rol[(role, ph)][dst] = us
        blocks = sorted({k[0] for k in blk}, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
        for bi in blocks:
            r = OrderedDict(quantization=cfg, block=(f"blk.{bi}" if isinstance(bi, int) else bi))
            for ph in ("prefill", "decode"):
                dd = blk.get((bi, ph), {})
                r[f"{ph}_us"] = round(sum(med(v) for v in dd.values()), 1)
            lt_rows.append(r)
        for role in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down","norm","other"]:
            r = OrderedDict(quantization=cfg, role=role)
            for ph in ("prefill", "decode"):
                dd = rol.get((role, ph), {})
                r[f"{ph}_us"] = round(sum(med(v) for v in dd.values()), 1)
            role_rows.append(r)
        # batch overhead: profile-batch (batch-dur vs htp-ops). overhead = dispatch/sync.
        _, batches = parse_htpprof(read(os.path.join(R, f"htpprof_{cfg}_HTP0.log")))
        if batches:
            durs = [b["batch_dur_usec"] for b in batches]
            ops_us = [b["htp_ops_usec"] for b in batches]
            batch_rows.append(OrderedDict(quantization=cfg, n_batches=len(batches),
                median_batch_dur_us=med(durs), median_htp_ops_us=med(ops_us),
                median_overhead_us=(med(durs) - med(ops_us) if durs and ops_us else ""),
                note="overhead=dispatch/synchronize (host↔DSP). decode 병목 원인 판별용"))

    with open(os.path.join(O, "op_detail.csv"), "w", newline="") as fp:
        cols = ["quantization","execution_mode","idx","op_type","dst","dst_type","dst_buft",
                "src0","src0_buft","block_index","logical_layer","tensor_role",
                "assigned_backend","prefill_us","decode_us","timing_source","fused_note"]
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in op_rows: w.writerow(r)
    with open(os.path.join(O, "layer_timing.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["quantization","block","prefill_us","decode_us"]); w.writeheader()
        for r in lt_rows: w.writerow(r)
    with open(os.path.join(O, "layer_timing_by_role.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["quantization","role","prefill_us","decode_us"]); w.writeheader()
        for r in role_rows: w.writerow(r)
    with open(os.path.join(O, "htp_batch_overhead.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["quantization","n_batches","median_batch_dur_us",
                "median_htp_ops_us","median_overhead_us","note"]); w.writeheader()
        for r in batch_rows: w.writerow(r)

    # ---- op_compare.csv : op별 CPU모드 vs HTP모드 나란히 (buft + time). 임베딩/출력 CPU op 포함 ----
    #  HTP ops = OPTRACE[HTP] (full detail + profile-op time). CPU-fallback ops(임베딩 GET_ROWS/출력) = SCHED_DEBUG HTP모드 backend==CPU.
    #  (CPU Mode)inference_time 은 무패치로 확인불가 → Phase-2 패치 필요.
    cmp_rows = []
    for cfg in sorted({c for c, _ in configs}):
        if not os.path.exists(os.path.join(R, f"assign_{cfg}_HTP0.log")):
            continue
        htp_text = read(os.path.join(R, f"assign_{cfg}_HTP0.log"))
        optrace = parse_optrace_htp(htp_text)
        sched_nodes, _ = parse_assign(htp_text)
        prof = parse_profileop(read(os.path.join(R, f"htpprof_{cfg}_HTP0.log")))
        tby = defaultdict(lambda: defaultdict(list))
        for p in prof:
            tby[p["dst"]][p["phase"]].append(p["usec"])
        # Phase-2 패치 로그: CPU 모드 per-op 시간, HTP 모드 CPU-fallback op 시간
        cpu_lp = parse_layerprof(read(os.path.join(R, f"cpuprof_{cfg}_CPU.log")))
        htp_lp = parse_layerprof(read(os.path.join(R, f"cpuprof_{cfg}_HTP0.log")))
        def cput(dst):  # CPU 모드 decode us (없으면 prefill)
            d = cpu_lp.get(dst)
            if not d: return ""
            return med(d["decode"]) if d["decode"] else med(d["prefill"])
        def htp_cpufb(dst):  # HTP 모드에서 CPU 로 남은 op 의 decode us
            d = htp_lp.get(dst)
            return med(d["decode"]) if (d and d["decode"]) else ""
        NA_CPU = "확인불가(패치빌드/CPU캡처 필요)"
        rows = []
        htp_dsts = set()
        for o in optrace:
            bi, lname, role = logical(o["src0"] or o["dst"])
            if role == "other":
                bi, lname, role = logical(o["dst"])
            htp_dsts.add(o["dst"])
            rows.append((o["idx"], OrderedDict(
                quantization=cfg, idx=o["idx"], op_type=o["op"], dst=o["dst"], dst_type=o["dst_type"],
                src0=o["src0"], logical_layer=lname, tensor_role=role,
                cpu_mode_time_us=(cput(o["dst"]) or NA_CPU),
                htp_mode_time_us=(med(tby[o["dst"]].get("decode", [])) or med(tby[o["dst"]].get("prefill", [])) or ""),
                cpu_mode_src0_buft="CPU", htp_mode_src0_buft=o["src0_buft"])))
        # CPU-fallback ops (HTP 모드에서 CPU 로 남은 것: 임베딩 GET_ROWS, 출력 등). dedup by name.
        seen_cpu = set()
        for n in sched_nodes:
            nm = n["node_name"]
            if n["assigned_backend"] != "CPU" or nm in htp_dsts or nm in seen_cpu:
                continue
            seen_cpu.add(nm)
            bi, lname, role = logical(n["weight_tensor"] or nm)
            order = -1 if role == "token_embd" else 10 ** 6  # 임베딩은 맨 앞, 출력류는 맨 뒤
            rows.append((order, OrderedDict(
                quantization=cfg, idx="", op_type=n["op_type"], dst=nm, dst_type="",
                src0=n["weight_tensor"], logical_layer=lname, tensor_role=role,
                cpu_mode_time_us=(cput(nm) or NA_CPU),
                htp_mode_time_us=(htp_cpufb(nm) or "확인불가(CPU fallback, 패치캡처 필요)"),
                cpu_mode_src0_buft="CPU", htp_mode_src0_buft="CPU")))
        rows.sort(key=lambda x: x[0])
        cmp_rows.extend(r for _, r in rows)
    with open(os.path.join(O, "op_compare.csv"), "w", newline="") as fp:
        cols = ["quantization","idx","op_type","dst","dst_type","src0","logical_layer","tensor_role",
                "cpu_mode_time_us","htp_mode_time_us","cpu_mode_src0_buft","htp_mode_src0_buft"]
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in cmp_rows: w.writerow(r)

    # raw_backend_assignment.csv
    if assign_rows:
        with open(os.path.join(O, "raw_backend_assignment.csv"), "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(assign_rows[0].keys())); w.writeheader(); w.writerows(assign_rows)
    # backend_transition.csv
    with open(os.path.join(O, "backend_transition.csv"), "w", newline="") as fp:
        cols = ["run_config","quantization","execution_mode","split_index","src_backend","dst_backend","n_inputs"]
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader(); w.writerows(trans_rows)

    # raw_node_timing.csv (HTP op timing; CPU op 시간은 무패치로 확인불가)
    timing_rows = []
    for f in sorted(glob.glob(os.path.join(R, "htpprof_*.log"))):
        cfg = os.path.basename(f)[len("htpprof_"):-len("_HTP0.log")]
        ops, batches = parse_htpprof(read(f))
        agg = defaultdict(list)
        for r in ops: agg[r["op_type"]].append(r)
        for op, rs in agg.items():
            us = sorted(x["usec"] for x in rs); n = len(us)
            timing_rows.append(OrderedDict(quantization=cfg, execution_mode="HTP0", backend="HTP0",
                op_type=op, count=n,
                median_us=us[n//2], p90_us=us[min(n-1,int(n*0.9))], min_us=us[0], max_us=us[-1],
                note="HTP DSP per-op (HEXAGON_PROFILE=1)"))
        for i, bt in enumerate(batches[:5]):
            timing_rows.append(OrderedDict(quantization=cfg, execution_mode="HTP0", backend="HTP0",
                op_type=f"__batch_{i}", count=bt["n_ops"], median_us=bt["batch_dur_usec"],
                p90_us="", min_us="", max_us=bt["htp_ops_usec"], note="HTP batch dur / htp-ops-usec"))
    with open(os.path.join(O, "raw_node_timing.csv"), "w", newline="") as fp:
        cols = ["quantization","execution_mode","backend","op_type","count","median_us","p90_us","min_us","max_us","note"]
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in timing_rows: w.writerow(r)

    # layer_summary.csv : block 단위 backend 요약(assignment) + graph-level bench(prefill/decode)
    #  - block backend: 해당 block 노드들의 assigned backend 집합 -> HTP0/CPU/MIXED
    #  - graph 성능은 config 단위(레이어별 CPU 시간은 무패치 확인불가)
    block_be = defaultdict(lambda: defaultdict(set))  # (cfg_mode) -> block -> set(backend)
    op_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # cfgmode->block->backend->count
    for r in assign_rows:
        key = r["run_config"]; bi = r["block_index"] if r["block_index"] != "" else r["logical_layer"]
        block_be[key][bi].add(r["assigned_backend"])
        op_counts[key][bi][r["assigned_backend"]] += 1
    PPTOK = 255
    bench = {}
    for cfg, mode in configs:
        pf, dc = [], []
        for bf in sorted(glob.glob(os.path.join(R, f"bench_{cfg}_{mode}*.log"))):
            br = parse_bench(read(bf))
            if "prefill" in br: pf.append(br["prefill"][0])
            if "decode" in br: dc.append(br["decode"][0])
        d = {}
        if pf: d["prefill"] = (sorted(pf)[len(pf)//2], 0)
        if dc: d["decode"] = (sorted(dc)[len(dc)//2], 0)
        bench[f"{cfg}_{mode}"] = d

    # mode_comparison.csv : HTP vs CPU / Q4_0 vs Q8_0 시간 비교 (모드 병목 파악의 top-level 표)
    def g(key, which):
        v = bench.get(key, {}).get(which)
        return v[0] if v else None
    mc = []
    for cfg, mode in configs:
        k = f"{cfg}_{mode}"
        ptps, dtps = g(k, "prefill"), g(k, "decode")
        mc.append(OrderedDict(config=k, quantization=cfg, mode=mode,
            prefill_tps=(round(ptps,2) if ptps else ""),
            prefill_ms_255tok=(round(PPTOK/ptps*1000,2) if ptps else ""),
            decode_tps=(round(dtps,2) if dtps else ""),
            decode_ms_per_token=(round(1000/dtps,3) if dtps else "")))
    with open(os.path.join(O, "mode_comparison.csv"), "w", newline="") as fp:
        if mc:
            w = csv.DictWriter(fp, fieldnames=list(mc[0].keys())); w.writeheader(); w.writerows(mc)
    # speedup 요약 (console)
    def spd(a, b, which):
        va, vb = g(a, which), g(b, which)
        return round(va/vb, 2) if (va and vb) else "N/A"
    print("--- mode speedup (t/s ratio, >1 = 앞이 빠름) ---")
    for q in ("Q4_0", "Q8_0"):
        print(f"  {q}: prefill HTP/CPU={spd(q+'_HTP0', q+'_CPU','prefill')}  decode HTP/CPU={spd(q+'_HTP0', q+'_CPU','decode')}")
    for m in ("CPU", "HTP0"):
        print(f"  {m}: prefill Q4/Q8={spd('Q4_0_'+m,'Q8_0_'+m,'prefill')}  decode Q4/Q8={spd('Q4_0_'+m,'Q8_0_'+m,'decode')}")
    with open(os.path.join(O, "layer_summary.csv"), "w", newline="") as fp:
        cols = ["run_config","logical_layer","backend_set","cpu_ops","htp_ops",
                "graph_prefill_tps","graph_decode_tps","note"]
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for key in block_be:
            bres = bench.get(key, {})
            pf = bres.get("prefill", ("",""))[0]; dc = bres.get("decode", ("",""))[0]
            # 정렬: 숫자 block 먼저
            def sk(x):
                return (0, x) if isinstance(x, int) else (1, str(x))
            for bi in sorted(block_be[key], key=sk):
                bs = block_be[key][bi]
                tag = "MIXED" if len(bs) > 1 else next(iter(bs))
                w.writerow(dict(run_config=key, logical_layer=(f"blk.{bi}" if isinstance(bi,int) else bi),
                    backend_set=tag, cpu_ops=op_counts[key][bi].get("CPU",0),
                    htp_ops=op_counts[key][bi].get("HTP0",0),
                    graph_prefill_tps=pf, graph_decode_tps=dc,
                    note="graph_*_tps 는 config 단위(레이어별 CPU 시간은 무패치 확인불가)"))

    print("parsed configs:", configs)
    print("assignment rows:", len(assign_rows), "| transitions:", len(trans_rows),
          "| htp timing rows:", len(timing_rows))
    print("wrote: raw_backend_assignment.csv, backend_transition.csv, raw_node_timing.csv, layer_summary.csv ->", O)

if __name__ == "__main__":
    main()
