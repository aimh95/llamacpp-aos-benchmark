#!/usr/bin/env python3
"""tied_on GGUF에서 output.weight 텐서를 추가하여 tied_off GGUF를 생성한다.

Strategy:
  - token_embd.weight 의 raw data를 복사하여 output.weight 텐서로 추가
  - 나머지 메타데이터/텐서는 동일하게 복사
  - 결과: llama.cpp가 output.weight를 별도 LAYER_OUTPUT 텐서로 로드
    (token_embd.weight는 LAYER_INPUT/CPU, output.weight는 LAYER_OUTPUT/HTP0)

Limitation:
  - 메모리 사용량 증가: token_embd.weight 크기만큼 추가 (≈ vocab_size × embd_dim × dtype_bytes)
  - Q8_0 기준 ~256MB, Q4_0 기준 ~128MB 증가 예상

Usage:
  python3 create_tied_off_gguf.py --input tied_on.gguf --output tied_off.gguf [--log log.txt]
"""
import argparse
import sys
import struct
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "third_party/llama.cpp/gguf-py"))

try:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.constants import GGMLQuantizationType
except ImportError as e:
    print(f"[ERROR] gguf library not found: {e}")
    print("  Expected at: third_party/llama.cpp/gguf-py/")
    sys.exit(1)


def log(msg: str, logfile=None):
    print(msg)
    if logfile:
        print(msg, file=logfile)


def create_tied_off(input_path: Path, output_path: Path, logfile=None):
    log(f"[create_tied_off_gguf] input : {input_path}", logfile)
    log(f"[create_tied_off_gguf] output: {output_path}", logfile)

    reader = GGUFReader(str(input_path), "r")

    # --- Check preconditions ---
    token_embd = None
    output_exists = False
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            token_embd = tensor
        if tensor.name == "output.weight":
            output_exists = True

    if token_embd is None:
        log("[ERROR] token_embd.weight not found in input GGUF", logfile)
        return False

    if output_exists:
        log("[WARN] output.weight already exists in input GGUF — this is already a tied_off model", logfile)
        log("       Copying input to output as-is.", logfile)
        shutil.copy2(input_path, output_path)
        return True

    log(f"[INFO] token_embd.weight: shape={list(token_embd.shape)} dtype={token_embd.tensor_type}", logfile)
    log(f"[INFO] Adding output.weight = copy of token_embd.weight", logfile)

    # --- Build new GGUF ---
    writer = GGUFWriter(str(output_path), arch=None)

    # Copy all metadata key-value pairs
    kv_count = 0
    for key in reader.fields:
        field = reader.fields[key]
        if field.name == "general.architecture":
            # Must be added first; get arch string
            arch_val = bytes(field.parts[-1]).decode("utf-8").rstrip("\x00")
            writer.add_architecture()  # will be set from arch
            # Use direct API
            writer.add_string("general.architecture", arch_val)
            kv_count += 1
            continue

        # Re-add each field using the low-level API
        parts = field.parts
        if field.types and field.types[0] == GGUFValueType.STRING:
            val = bytes(parts[-1]).decode("utf-8").rstrip("\x00")
            writer.add_string(field.name, val)
        elif field.types and field.types[0] == GGUFValueType.UINT32:
            writer.add_uint32(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.INT32:
            writer.add_int32(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.FLOAT32:
            writer.add_float32(field.name, float(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.BOOL:
            writer.add_bool(field.name, bool(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.UINT64:
            writer.add_uint64(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.INT64:
            writer.add_int64(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.FLOAT64:
            writer.add_float64(field.name, float(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.UINT16:
            writer.add_uint16(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.INT16:
            writer.add_int16(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.UINT8:
            writer.add_uint8(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.INT8:
            writer.add_int8(field.name, int(parts[-1][0]))
        elif field.types and field.types[0] == GGUFValueType.ARRAY:
            # Arrays (e.g. tokenizer vocab, merges, etc.) — use raw numpy data
            # This is complex; log and skip non-critical arrays if needed
            try:
                arr_type = field.types[1] if len(field.types) > 1 else None
                if arr_type == GGUFValueType.STRING:
                    vals = [bytes(p).decode("utf-8").rstrip("\x00") for p in parts[3::2]]
                    writer.add_array(field.name, vals)
                elif arr_type in (GGUFValueType.UINT32, GGUFValueType.INT32,
                                  GGUFValueType.FLOAT32, GGUFValueType.UINT8,
                                  GGUFValueType.INT8, GGUFValueType.UINT16,
                                  GGUFValueType.FLOAT64):
                    writer.add_array(field.name, list(parts[-1]))
                else:
                    log(f"  [SKIP] array field {field.name!r} type={arr_type}", logfile)
                    continue
            except Exception as ex:
                log(f"  [SKIP] array field {field.name!r}: {ex}", logfile)
                continue
        else:
            log(f"  [SKIP] unknown field type {field.name!r} types={field.types}", logfile)
            continue
        kv_count += 1

    log(f"[INFO] copied {kv_count} metadata fields", logfile)

    # --- Add all original tensors ---
    n_tensors = 0
    embd_data = None
    embd_qtype = None

    for tensor in reader.tensors:
        data = tensor.data  # numpy array (raw bytes as uint8 for quantized types)
        writer.add_tensor(tensor.name, data, raw_shape=tensor.shape,
                          raw_dtype=tensor.tensor_type)
        if tensor.name == "token_embd.weight":
            embd_data  = data.copy()
            embd_qtype = tensor.tensor_type
        n_tensors += 1

    log(f"[INFO] copied {n_tensors} tensors", logfile)

    # --- Add output.weight as copy of token_embd.weight ---
    log(f"[INFO] adding output.weight (dtype={embd_qtype}, shape={list(token_embd.shape)})", logfile)
    writer.add_tensor("output.weight", embd_data, raw_shape=token_embd.shape,
                      raw_dtype=embd_qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = output_path.stat().st_size / (1024**3)
    in_size  = input_path.stat().st_size / (1024**3)
    log(f"[INFO] tied_off GGUF written: {output_path}", logfile)
    log(f"       size: {in_size:.2f} GB (tied_on) → {out_size:.2f} GB (tied_off)", logfile)
    log(f"       delta: +{(out_size - in_size)*1024:.0f} MB (output.weight copy)", logfile)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log",    default=None)
    args = parser.parse_args()

    logfile = open(args.log, "w") if args.log else None
    try:
        ok = create_tied_off(Path(args.input), Path(args.output), logfile)
    finally:
        if logfile:
            logfile.close()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
