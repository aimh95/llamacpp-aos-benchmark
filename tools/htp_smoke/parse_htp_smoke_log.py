#!/usr/bin/env python3
"""Parse llama.cpp HTP0 smoke test raw logs into summary.md / summary.csv.

표준 라이브러리만 사용. 모든 판정은 로그 키워드 기반 candidate이며
"candidate based on logs"이다 — 최종 성공 여부는 사람이 판단한다.
"""
import csv
import os
import re
import sys
from datetime import datetime, timezone

RUNTIME_INIT_KEYWORDS = [
    "hexagon",
    "ggml-hex",
    "ggml_hex",
    "qnn backend",
    "qnnbackend",
    "backend_qnn",
    "htp backend",
    "htp session",
    "htp0 session",
    "qnn_interface",
    "qnninterface",
]

QNN_LOAD_ACTION_HINTS = ["load", "dlopen", "init"]
SKEL_STUB_KEYWORDS = ["skel", "stub"]
CDSP_RPC_KEYWORDS = ["cdsp", "adsprpc", "rpc"]

FAILURE_KEYWORDS = [
    "offloaded 0",
    "fallback",
    "cpu fallback",
    "device not found",
    "failed to load",
    "unsupported op",
    "unsupported tensor",
    "segmentation fault",
    "no such file or directory",
    "permission denied",
    "aborted",
]

OFFLOAD_RE = re.compile(r"offload(?:ed|ing)?\D{0,10}(\d+)\s*/\s*(\d+)\s*layers?", re.IGNORECASE)
HTP_BUFFER_RE = re.compile(r"htp[^\n]{0,40}(?:model )?buffer[^\n]{0,80}", re.IGNORECASE)

LOAD_TIME_RE = re.compile(
    r"^\s*llama_perf_context_print:\s*load time\s*=\s*([\d.]+)\s*ms",
    re.IGNORECASE | re.MULTILINE,
)
PROMPT_EVAL_RE = re.compile(
    r"^\s*llama_perf_context_print:\s*prompt eval time\s*=\s*([\d.]+)\s*ms"
    r"(?:[^\n]*?([\d.]+)\s*tokens per second)?",
    re.IGNORECASE | re.MULTILINE,
)
EVAL_RE = re.compile(
    r"^\s*llama_perf_context_print:\s*eval time\s*=\s*([\d.]+)\s*ms"
    r"(?:[^\n]*?([\d.]+)\s*tokens per second)?",
    re.IGNORECASE | re.MULTILINE,
)
TOTAL_RE = re.compile(
    r"^\s*llama_perf_context_print:\s*total time\s*=\s*([\d.]+)\s*ms",
    re.IGNORECASE | re.MULTILINE,
)

FIELD_ORDER = [
    "htp_device_visible",
    "htp_runtime_initialized",
    "qnn_library_load_detected",
    "skel_stub_keyword_detected",
    "cdsp_rpc_keyword_detected",
    "offload_detected",
    "offloaded_layers_raw",
    "cpu_fallback_suspected",
    "failure_keywords",
    "llama_perf_detected",
    "prompt_eval_time_ms",
    "eval_time_ms",
    "total_time_ms",
    "tok_per_sec",
]

RAW_ARTIFACT_NAMES = [
    "devices.txt",
    "run_params.txt",
    "wrapper_check.txt",
    "list_devices.log",
    "qnn_libs_before_run.txt",
    "search_errors.txt",
    "htp0_smoke.log",
    "exit_code.txt",
    "logcat_qnn_htp.txt",
    "logcat_errors.txt",
]


def read_text(path):
    if not os.path.isfile(path):
        return ""
    with open(path, "r", errors="replace") as f:
        return f.read()


def read_params(path):
    params = {}
    for line in read_text(path).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def any_keyword(text_lower, keywords):
    return any(k in text_lower for k in keywords)


def detect_qnn_library_load(text_lower):
    for line in text_lower.splitlines():
        if "qnn" in line and any(h in line for h in QNN_LOAD_ACTION_HINTS):
            return True
    return False


def detect_offload(text):
    m = OFFLOAD_RE.search(text)
    if m:
        return True, m.group(0).strip(), m.group(1)
    m = HTP_BUFFER_RE.search(text)
    if m:
        return True, m.group(0).strip(), None
    return False, "", None


def extract_perf(text):
    perf = {
        "prompt_eval_time_ms": "NOT_FOUND",
        "eval_time_ms": "NOT_FOUND",
        "total_time_ms": "NOT_FOUND",
        "tok_per_sec": "NOT_FOUND",
        "detected": False,
    }
    if LOAD_TIME_RE.search(text):
        perf["detected"] = True

    m = PROMPT_EVAL_RE.search(text)
    if m:
        perf["prompt_eval_time_ms"] = m.group(1)
        perf["detected"] = True

    m = EVAL_RE.search(text)
    if m:
        perf["eval_time_ms"] = m.group(1)
        perf["detected"] = True
        if m.group(2):
            perf["tok_per_sec"] = m.group(2)

    m = TOTAL_RE.search(text)
    if m:
        perf["total_time_ms"] = m.group(1)
        perf["detected"] = True

    return perf


def build_fields(out_dir):
    params = read_params(os.path.join(out_dir, "run_params.txt"))
    backend_device = params.get("backend_device") or "HTP0"

    list_devices_text = read_text(os.path.join(out_dir, "list_devices.log"))
    smoke_text = read_text(os.path.join(out_dir, "htp0_smoke.log"))
    logcat_text = read_text(os.path.join(out_dir, "logcat_qnn_htp.txt"))

    combined = "\n".join([smoke_text, logcat_text])
    combined_lower = combined.lower()
    has_combined = bool(combined.strip())

    if list_devices_text.strip():
        htp_device_visible = "YES" if backend_device.lower() in list_devices_text.lower() else "NO"
    else:
        htp_device_visible = "UNKNOWN"

    if not has_combined:
        htp_runtime_initialized = "UNKNOWN"
        qnn_library_load_detected = "UNKNOWN"
        skel_stub_keyword_detected = "UNKNOWN"
        cdsp_rpc_keyword_detected = "UNKNOWN"
        offload_detected = "UNKNOWN"
    else:
        htp_runtime_initialized = "YES" if any_keyword(combined_lower, RUNTIME_INIT_KEYWORDS) else "NO"
        qnn_library_load_detected = "YES" if detect_qnn_library_load(combined_lower) else "NO"
        skel_stub_keyword_detected = "YES" if any_keyword(combined_lower, SKEL_STUB_KEYWORDS) else "NO"
        cdsp_rpc_keyword_detected = "YES" if any_keyword(combined_lower, CDSP_RPC_KEYWORDS) else "NO"

    offload_found, offload_raw, offload_n = detect_offload(combined)
    if not has_combined:
        offload_detected = "UNKNOWN"
    elif offload_found and offload_n == "0":
        offload_detected = "NO"
    elif offload_found:
        offload_detected = "YES"
    else:
        offload_detected = "UNKNOWN"
    offloaded_layers_raw = offload_raw if offload_raw else "NOT_FOUND"

    found_failures = sorted({kw for kw in FAILURE_KEYWORDS if kw in combined_lower})
    failure_keywords = ", ".join(found_failures) if found_failures else "NONE"

    explicit_zero_offload = offload_found and offload_n == "0"
    if not has_combined:
        cpu_fallback_suspected = "UNKNOWN"
    elif "fallback" in combined_lower or explicit_zero_offload:
        cpu_fallback_suspected = "YES"
    elif htp_runtime_initialized == "YES" and offload_detected == "YES":
        cpu_fallback_suspected = "NO"
    else:
        cpu_fallback_suspected = "UNKNOWN"

    perf = extract_perf(smoke_text)

    fields = {
        "htp_device_visible": htp_device_visible,
        "htp_runtime_initialized": htp_runtime_initialized,
        "qnn_library_load_detected": qnn_library_load_detected,
        "skel_stub_keyword_detected": skel_stub_keyword_detected,
        "cdsp_rpc_keyword_detected": cdsp_rpc_keyword_detected,
        "offload_detected": offload_detected,
        "offloaded_layers_raw": offloaded_layers_raw,
        "cpu_fallback_suspected": cpu_fallback_suspected,
        "failure_keywords": failure_keywords,
        "llama_perf_detected": "YES" if perf["detected"] else "NO",
        "prompt_eval_time_ms": perf["prompt_eval_time_ms"],
        "eval_time_ms": perf["eval_time_ms"],
        "total_time_ms": perf["total_time_ms"],
        "tok_per_sec": perf["tok_per_sec"],
    }
    return fields, params, backend_device


def write_markdown(out_path, out_dir, params, backend_device, exit_code, fields):
    lines = []
    lines.append("# HTP0 Smoke Test Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Artifacts dir: `{out_dir}`")
    lines.append(f"- Backend device requested: `{backend_device}`")
    lines.append(f"- Run exit code: `{exit_code}`")
    lines.append("")
    lines.append("## Run Params")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for k, v in params.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Summary (candidate based on logs)")
    lines.append("")
    lines.append(
        "> 아래 판정은 로그 키워드 기반 candidate이며 확정값이 아닙니다. "
        "최종 성공/실패 판정은 사람이 raw log를 보고 직접 확인해야 합니다."
    )
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for name in FIELD_ORDER:
        lines.append(f"| {name} | {fields[name]} |")
    lines.append("")
    lines.append("## Raw Artifacts")
    lines.append("")
    for name in RAW_ARTIFACT_NAMES:
        if os.path.isfile(os.path.join(out_dir, name)):
            lines.append(f"- {name}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_csv(out_path, fields):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        for name in FIELD_ORDER:
            writer.writerow([name, fields[name]])


def main(argv):
    if len(argv) != 2:
        print("usage: parse_htp_smoke_log.py <artifacts_dir>", file=sys.stderr)
        return 1

    out_dir = argv[1]
    fields, params, backend_device = build_fields(out_dir)
    exit_code = read_text(os.path.join(out_dir, "exit_code.txt")).strip() or "UNKNOWN"

    write_markdown(
        os.path.join(out_dir, "summary.md"), out_dir, params, backend_device, exit_code, fields
    )
    write_csv(os.path.join(out_dir, "summary.csv"), fields)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
