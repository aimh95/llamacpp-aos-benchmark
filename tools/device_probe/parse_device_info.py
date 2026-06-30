#!/usr/bin/env python3
"""Parse raw android device probe artifacts into device_info.md / device_info.csv.

표준 라이브러리만 사용. collect_android_device_info.sh 가 생성한
<artifacts_dir>/{getprop.txt,qnn_libs.txt,host_env.txt,...} 를 읽어
같은 디렉터리에 device_info.md / device_info.csv 를 만든다.
"""
import csv
import os
import re
import sys
from datetime import datetime, timezone

GETPROP_LINE_RE = re.compile(r"^\[(?P<key>[^\]]+)\]:\s*\[(?P<value>.*)\]\s*$")
HTP_VERSION_RE = re.compile(r"[Vv](\d{2,3})")

# 단말마다 SoC 정보가 들어있는 prop 키가 다르므로 후보를 순서대로 시도한다.
FIELD_PROP_CANDIDATES = {
    "manufacturer": ["ro.product.manufacturer"],
    "device_model": ["ro.product.model"],
    "device_name": ["ro.product.device", "ro.product.name"],
    "android_release": ["ro.build.version.release"],
    "android_sdk": ["ro.build.version.sdk"],
    "build_fingerprint": ["ro.build.fingerprint"],
    "abi_primary": ["ro.product.cpu.abi"],
    "abi_list": ["ro.product.cpu.abilist"],
    "board_platform": ["ro.board.platform"],
    "hardware": ["ro.hardware"],
    "soc_model": [
        "ro.soc.model",
        "ro.boot.hardware.platform",
        "ro.hardware.chipname",
        "ro.product.board",
        "ro.boot.hardware",
    ],
    "soc_manufacturer": ["ro.soc.manufacturer"],
}

DERIVED_FIELD_ORDER = [
    "htp_candidate_versions_from_lib_name",
    "qnn_htp_skel_count",
    "qnn_htp_stub_count",
    "qnn_core_lib_count",
    "cdsprpc_found",
    "opencl_found",
]

FIELD_ORDER = list(FIELD_PROP_CANDIDATES.keys()) + DERIVED_FIELD_ORDER

RAW_ARTIFACT_NAMES = [
    "devices.txt",
    "getprop.txt",
    "cpuinfo.txt",
    "meminfo.txt",
    "uname.txt",
    "pm_features.txt",
    "host_env.txt",
    "qnn_libs.txt",
]


def parse_getprop(path):
    props = {}
    if not os.path.isfile(path):
        return props
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = GETPROP_LINE_RE.match(line.strip())
            if m:
                props[m.group("key")] = m.group("value")
    return props


def first_present(props, keys):
    for k in keys:
        v = props.get(k, "").strip()
        if v:
            return v
    return "NOT_FOUND"


def parse_qnn_libs(path):
    paths = set()
    if not os.path.isfile(path):
        return paths
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=="):
                continue
            paths.add(line)
    return paths


def categorize_libs(lib_paths):
    skel, stub, core = [], [], []
    cdsprpc = False
    opencl = False
    versions = set()

    for p in lib_paths:
        base = os.path.basename(p)
        lower = base.lower()
        is_htp = "htp" in lower
        is_qnn = "qnn" in lower
        is_skel = is_htp and "skel" in lower
        is_stub = is_htp and "stub" in lower

        if is_skel:
            skel.append(p)
        elif is_stub:
            stub.append(p)
        elif is_qnn:
            core.append(p)

        if is_skel or is_stub:
            m = HTP_VERSION_RE.search(base)
            if m:
                versions.add(f"V{m.group(1)}")

        if lower == "libcdsprpc.so":
            cdsprpc = True
        if lower == "libopencl.so":
            opencl = True

    return {
        "skel": skel,
        "stub": stub,
        "core": core,
        "cdsprpc": cdsprpc,
        "opencl": opencl,
        "versions": sorted(versions, key=lambda v: int(v[1:])),
    }


def build_fields(props, lib_info):
    fields = {}
    for name, keys in FIELD_PROP_CANDIDATES.items():
        fields[name] = first_present(props, keys)

    fields["htp_candidate_versions_from_lib_name"] = (
        ", ".join(lib_info["versions"]) if lib_info["versions"] else "NOT_FOUND"
    )
    fields["qnn_htp_skel_count"] = str(len(lib_info["skel"]))
    fields["qnn_htp_stub_count"] = str(len(lib_info["stub"]))
    fields["qnn_core_lib_count"] = str(len(lib_info["core"]))
    fields["cdsprpc_found"] = "FOUND" if lib_info["cdsprpc"] else "NOT_FOUND"
    fields["opencl_found"] = "FOUND" if lib_info["opencl"] else "NOT_FOUND"
    return fields


def read_host_env(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                rows.append((k, v))
    return rows


def write_markdown(out_path, artifacts_dir, fields, lib_paths, host_env_rows):
    lines = []
    lines.append("# Android Device Probe Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Artifacts dir: `{artifacts_dir}`")
    lines.append("")
    lines.append("## Device Info")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for name in FIELD_ORDER:
        lines.append(f"| {name} | {fields[name]} |")
    lines.append("")
    lines.append("## HTP Generation Notice")
    lines.append("")
    lines.append(
        "> **candidate only** — `htp_candidate_versions_from_lib_name` 값은 "
        "라이브러리 파일명(예: `libQnnHtpV75Skel.so`)에서 추출한 추정치이며 확정값이 아닙니다."
    )
    lines.append(
        "> 실제 NPU(HTP) 실행 가능 여부는 runtime에서 QNN/HTP backend "
        "initialization 로그를 직접 확인해야 합니다."
    )
    lines.append("")
    lines.append(f"## Found QNN / HTP / OpenCL / RPC Libraries ({len(lib_paths)})")
    lines.append("")
    if lib_paths:
        lines.append("```")
        for p in sorted(lib_paths):
            lines.append(p)
        lines.append("```")
    else:
        lines.append("NOT_FOUND")
    lines.append("")
    lines.append("## Host Environment")
    lines.append("")
    lines.append("| var | value |")
    lines.append("| --- | --- |")
    for k, v in host_env_rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Raw Artifacts")
    lines.append("")
    for name in RAW_ARTIFACT_NAMES:
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
        print("usage: parse_device_info.py <artifacts_dir>", file=sys.stderr)
        return 1

    artifacts_dir = argv[1]
    props = parse_getprop(os.path.join(artifacts_dir, "getprop.txt"))
    lib_paths = parse_qnn_libs(os.path.join(artifacts_dir, "qnn_libs.txt"))
    lib_info = categorize_libs(lib_paths)
    fields = build_fields(props, lib_info)
    host_env_rows = read_host_env(os.path.join(artifacts_dir, "host_env.txt"))

    write_markdown(
        os.path.join(artifacts_dir, "device_info.md"),
        artifacts_dir,
        fields,
        lib_paths,
        host_env_rows,
    )
    write_csv(os.path.join(artifacts_dir, "device_info.csv"), fields)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
