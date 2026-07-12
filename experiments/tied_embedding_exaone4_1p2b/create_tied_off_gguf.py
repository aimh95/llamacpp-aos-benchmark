#!/usr/bin/env python3
"""
Binary-level GGUF modification: output.weight = copy of token_embd.weight.

Approach: direct byte manipulation instead of GGUFWriter re-encoding.
- KV section copied byte-for-byte (no re-encoding of tokenizer vocab etc.)
- Existing tensor_info offsets are relative to data_start → unchanged
- New output.weight tensor_info appended (offset = aligned end of existing data)
- Existing tensor data copied byte-for-byte, output.weight appended after

Usage:
  python3 create_tied_off_gguf.py --input tied_on.gguf --output tied_off.gguf [--log log.txt]
"""
import argparse
import struct
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "third_party/llama.cpp/gguf-py"))
try:
    from gguf import GGUFReader
except ImportError as e:
    print(f"[ERROR] gguf library not found: {e}")
    sys.exit(1)


# GGUFValueType scalar byte sizes (type_id → bytes)
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_kv(data: bytes, pos: int, n_kv: int) -> int:
    """Advance pos past n_kv KV entries and return new position."""
    for _ in range(n_kv):
        key_len, = struct.unpack_from('<Q', data, pos); pos += 8 + key_len
        vtype,   = struct.unpack_from('<I', data, pos); pos += 4
        if vtype == 8:    # STRING
            slen, = struct.unpack_from('<Q', data, pos); pos += 8 + slen
        elif vtype == 9:  # ARRAY
            etype, = struct.unpack_from('<I', data, pos); pos += 4
            alen,  = struct.unpack_from('<Q', data, pos); pos += 8
            if etype == 8:   # array of STRING
                for _ in range(alen):
                    slen, = struct.unpack_from('<Q', data, pos); pos += 8 + slen
            elif etype in _SCALAR:
                pos += alen * _SCALAR[etype]
            else:
                raise ValueError(f"Unknown ARRAY element type: {etype}")
        elif vtype in _SCALAR:
            pos += _SCALAR[vtype]
        else:
            raise ValueError(f"Unknown KV value type: {vtype}")
    return pos


def _info_entry_size(name: str, n_dims: int) -> int:
    return 8 + len(name.encode()) + 4 + n_dims * 8 + 4 + 8


def create_tied_off(input_path: Path, output_path: Path, log=print) -> bool:
    log(f"[create_tied_off_gguf] input : {input_path}")
    log(f"[create_tied_off_gguf] output: {output_path}")

    reader = GGUFReader(str(input_path), "r")

    embd = None
    for t in reader.tensors:
        if t.name == "output.weight":
            log("[WARN] output.weight already exists — copying as-is")
            shutil.copy2(input_path, output_path)
            return True
        if t.name == "token_embd.weight":
            embd = t

    if embd is None:
        log("[ERROR] token_embd.weight not found"); return False

    log(f"[INFO] token_embd.weight: shape={list(embd.shape)} "
        f"dtype={embd.tensor_type} n_bytes={embd.n_bytes}")

    with open(input_path, "rb") as f:
        src = bytearray(f.read())

    assert src[:4] == b'GGUF', "Not a GGUF file"
    version,   = struct.unpack_from('<I', src,  4)
    n_tensors, = struct.unpack_from('<Q', src,  8)
    n_kv,      = struct.unpack_from('<Q', src, 16)

    kv_end  = _skip_kv(src, 24, n_kv)
    kv_size = kv_end - 24
    log(f"[INFO] version={version} n_tensors={n_tensors} n_kv={n_kv} kv_size={kv_size}")

    # Alignment (default 32; may be stored as general.alignment KV)
    alignment = 32
    if "general.alignment" in reader.fields:
        alignment = int(reader.fields["general.alignment"].parts[-1][0])

    original_data_start = reader.data_offset

    # Total size of original tensor_info section (no padding)
    orig_info_size = sum(_info_entry_size(t.name, len(t.shape)) for t in reader.tensors)

    # Verify our KV parser is correct
    expected_ds = ((24 + kv_size + orig_info_size) + alignment - 1) // alignment * alignment
    if expected_ds != original_data_start:
        log(f"[WARN] parsed data_start={expected_ds} != reader.data_offset={original_data_start} "
            f"— KV parse may be off")

    # New tensor_info entry for output.weight
    new_name    = "output.weight"
    n_dims      = len(embd.shape)
    new_info_sz = _info_entry_size(new_name, n_dims)

    new_pre      = 24 + kv_size + orig_info_size + new_info_sz
    new_ds       = (new_pre + alignment - 1) // alignment * alignment
    offset_delta = new_ds - original_data_start
    log(f"[INFO] new_data_start={new_ds} (delta={offset_delta:+d} bytes)")

    # output.weight lives after all existing tensor data, aligned
    max_end           = max(t.data_offset + t.n_bytes for t in reader.tensors)
    existing_data_end = max_end - original_data_start  # relative to data_start
    out_weight_offset = (existing_data_end + alignment - 1) // alignment * alignment
    log(f"[INFO] output.weight data offset in section: {out_weight_offset}")

    # ---- Assemble new file ----
    out = bytearray()

    # 1. Header — n_tensors incremented, everything else unchanged
    out += src[:8]
    out += struct.pack('<Q', n_tensors + 1)
    out += src[16:24]

    # 2. KV section — exact bytes, no re-encoding
    out += src[24:kv_end]

    # 3. Original tensor_info entries — offsets relative to data_start, unchanged
    out += src[kv_end : kv_end + orig_info_size]

    # 4. New tensor_info entry for output.weight
    entry  = struct.pack('<Q', len(new_name.encode()))
    entry += new_name.encode()
    entry += struct.pack('<I', n_dims)
    for d in embd.shape:
        entry += struct.pack('<Q', int(d))
    entry += struct.pack('<I', embd.tensor_type.value)
    entry += struct.pack('<Q', out_weight_offset)
    out += entry

    # 5. Alignment padding → lands at new_ds
    pad = (alignment - len(out) % alignment) % alignment
    out += b'\x00' * pad
    assert len(out) == new_ds, f"Padding error: len={len(out)}, expected={new_ds}"

    # 6. Existing tensor data section (unchanged)
    out += src[original_data_start:]

    # 7. Inter-tensor alignment padding before output.weight
    existing_sz = len(src) - original_data_start
    pad2 = (alignment - existing_sz % alignment) % alignment
    out += b'\x00' * pad2

    # 8. output.weight data = raw bytes of token_embd.weight
    out += src[embd.data_offset : embd.data_offset + embd.n_bytes]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(out)

    in_gb  = len(src) / 1024**3
    out_gb = len(out) / 1024**3
    log(f"[OK] tied_off GGUF written: {output_path}")
    log(f"     {in_gb:.2f} GB → {out_gb:.2f} GB (+{(out_gb - in_gb)*1024:.0f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add output.weight to a tied GGUF")
    parser.add_argument("--input",  required=True, help="tied_on .gguf path")
    parser.add_argument("--output", required=True, help="tied_off .gguf output path")
    parser.add_argument("--log",    default=None,  help="optional log file")
    args = parser.parse_args()

    logfile = open(args.log, "w") if args.log else None

    def log_fn(msg):
        print(msg)
        if logfile:
            print(msg, file=logfile, flush=True)

    try:
        ok = create_tied_off(Path(args.input), Path(args.output), log=log_fn)
    finally:
        if logfile:
            logfile.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
