#!/usr/bin/env python3
"""Requant routed-expert tensors of the UD-IQ2_XS-mtp production GGUF.

Weights come from the same-base unsloth UD-Q4_K_XL GGUF shards (higher
precision than the 2-bit production pack); importance weighting uses the
unsloth imatrix.  Three stages:

  plan      scan the production template, apply {match, from, to} rules,
            emit a recipe JSON (explicit tensor -> target type map)
  quantize  read each replaced tensor from the source shards, dequantize
            through libggml-base's dequantize_row_*, importance-weight
            quantize with quants.c, write raw payloads + manifest
            (resumable, sha256-verified)
  assemble  rebuild the GGUF: byte-copy every untouched tensor, splice in
            the new payloads, re-lay-out offsets, read-back verify

GGUF type ids follow ggml (== ds4 DS4_TENSOR_*); type 42 is the ds4-private
q2_0 block format (64 elems / 18 bytes).  Quantize targets are limited to
what quants.c implements: IQ2_XXS, Q2_K, Q4_1, Q4_K, Q8_0.
"""
import argparse
import ctypes
import hashlib
import json
import math
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen4_pack import (GGMLQuantizer, QwenGGUFImatrix, f32_to_bf16,
                        bf16_to_f32)
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str

Ggml = ctypes.CDLL("/opt/homebrew/lib/libggml-base.0.25.3.dylib")

# ggml type id -> (name, dequantize_row_* symbol or None for raw types).
# Ids match ggml.h / GGUF; 42 is the ds4-private q2_0 extension.
_TYPES = {
    0: ("F32", None), 1: ("F16", None), 2: ("Q4_0", "dequantize_row_q4_0"),
    3: ("Q4_1", "dequantize_row_q4_1"), 6: ("Q5_0", "dequantize_row_q5_0"),
    7: ("Q5_1", "dequantize_row_q5_1"), 8: ("Q8_0", "dequantize_row_q8_0"),
    9: ("Q8_1", None), 10: ("Q2_K", "dequantize_row_q2_K"),
    11: ("Q3_K", "dequantize_row_q3_K"), 12: ("Q4_K", "dequantize_row_q4_K"),
    13: ("Q5_K", "dequantize_row_q5_K"), 14: ("Q6_K", "dequantize_row_q6_K"),
    15: ("Q8_K", "dequantize_row_q8_K"), 16: ("IQ2_XXS", "dequantize_row_iq2_xxs"),
    17: ("IQ2_XS", "dequantize_row_iq2_xs"), 18: ("IQ3_XXS", "dequantize_row_iq3_xxs"),
    19: ("IQ1_S", "dequantize_row_iq1_s"), 20: ("IQ4_NL", "dequantize_row_iq4_nl"),
    21: ("IQ3_S", "dequantize_row_iq3_s"), 22: ("IQ2_S", "dequantize_row_iq2_s"),
    23: ("IQ4_XS", "dequantize_row_iq4_xs"), 24: ("I8", None),
    25: ("I16", None), 26: ("I32", None), 27: ("I64", None), 28: ("F64", None),
    29: ("IQ1_M", "dequantize_row_iq1_m"), 30: ("BF16", None),
    34: ("TQ1_0", "dequantize_row_tq1_0"), 35: ("TQ2_0", "dequantize_row_tq2_0"),
    39: ("MXFP4", "dequantize_row_mxfp4"), 40: ("NVFP4", "dequantize_row_nvfp4"),
    41: ("Q1_0", "dequantize_row_q1_0"), 42: ("Q2_0_D4", None),
}
_NAME_TO_ID = {v[0]: k for k, v in _TYPES.items()}

# gguf type id -> (block_elems, block_bytes); mirrors ds4.c gguf_types[].
_BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
    17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
    21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2),
    26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
    34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (16, 68), 41: (64, 18),
    42: (64, 18),
}

# quants.c output targets (QTYPE_LAYOUT keys).
QUANT_TARGETS = {"IQ2_XXS", "Q2_K", "Q4_1", "Q4_K", "Q8_0"}


def gguf_bytes(kind, dims):
    n = math.prod(dims)
    be, bb = _BLOCKS[kind]
    if n % be:
        raise SystemExit(f"type {kind} ({_TYPES[kind][0]}): {n} values not "
                         f"block-aligned ({be})")
    return n // be * bb


def type_name(t):
    return _TYPES.get(t, (f"type{t}", None))[0]


def dequant_tensor(kind, raw, n_rows, n_cols):
    """Packed (n_rows, n_cols) tensor payload -> float32 (n_rows, n_cols).
    ggml dequantizers accept the full element count: rows are contiguous
    independent block runs, so one call covers the whole tensor."""
    name, sym = _TYPES[kind]
    total = n_rows * n_cols
    if kind == 0:
        return np.frombuffer(raw, dtype="<f4", count=total).reshape(n_rows, n_cols)
    if kind == 1:
        u = np.frombuffer(raw, dtype="<u2", count=total)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(n_rows, n_cols)
    if kind == 30:
        return bf16_to_f32(np.frombuffer(raw, dtype="<u2", count=total)
                           .reshape(n_rows, n_cols))
    if sym is None:
        raise SystemExit(f"no dequantizer for ggml type {kind} ({name})")
    fn = getattr(Ggml, sym)
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    fn.restype = None
    out = np.empty(total, dtype=np.float32)
    fn(ctypes.c_char_p(raw),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), total)
    return out.reshape(n_rows, n_cols)


class SourceIncomplete(Exception):
    pass


class ShardMap:
    """Tensor lookup across a set of GGUF shard files (multi-file model)."""

    def __init__(self, paths):
        self.readers = [Reader(str(p)) for p in paths]
        self.loc = {}
        for rd in self.readers:
            for name, (kind, dims, off) in rd.tensors.items():
                self.loc[name] = (rd, kind, dims, off)

    def has(self, name):
        return name in self.loc

    def read_f32(self, name, cols=None):
        rd, kind, dims, off = self.loc[name]
        rd.f.seek(rd.data_start + off)
        raw = rd.f.read(gguf_bytes(kind, dims))
        if len(raw) != gguf_bytes(kind, dims):
            raise SourceIncomplete(name)
        n0, n1, n2 = (list(dims) + [1, 1, 1])[:3]
        if cols is not None and n0 != cols:
            # padded source (e.g. down K=768 logical 640): drop trailing pad
            pass
        m = dequant_tensor(kind, raw, n1 * n2, n0)
        out = m.reshape(n2, n1, n0)
        if cols is not None and n0 != cols:
            if n0 < cols:
                raise SystemExit(f"{name}: source cols {n0} < target {cols}")
            out = out[:, :, :cols]
        return out


def plan(args):
    tpl = Reader(str(args.template))
    rules = json.loads(args.rules.read_text()).get("rules", [])
    repl = {}
    for name, (kind, dims, off) in tpl.tensors.items():
        hit = None
        for rule in rules:
            if re.search(rule["match"], name) and kind in rule.get("from", []):
                hit = rule["to"].upper()
        if not hit or hit == type_name(kind):
            continue
        to_id = _NAME_TO_ID.get(hit)
        if to_id is None:
            raise SystemExit(f"unknown target type {hit}")
        if hit not in QUANT_TARGETS:
            raise SystemExit(f"{hit}: quants.c cannot quantize to this type "
                             f"(supported: {sorted(QUANT_TARGETS)})")
        try:
            gguf_bytes(to_id, dims)
        except SystemExit as e:
            raise SystemExit(f"{name}: {e}")
        repl[name] = {"to": hit, "to_id": to_id, "from": type_name(kind),
                      "dims": dims}
    out = {"template": str(args.template.resolve()), "replacements": repl}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(repl)} tensors to requantize -> {args.out}")
    for n, r in sorted(repl.items()):
        print(f"  {n}: {r['from']} -> {r['to']} dims={r['dims']}")


def quantize(args):
    recipe = json.loads(args.recipe.read_text())
    repl = recipe["replacements"]
    payloads = args.payloads
    payloads.mkdir(parents=True, exist_ok=True)
    manifest_path = payloads / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "recipe": str(args.recipe.resolve()), "tensors": {}}
    src = ShardMap(sorted(Path(args.source_dir).glob("*.gguf")))
    imatrix = QwenGGUFImatrix(Path(args.imatrix))
    quant = GGMLQuantizer(args.library)
    quant.lib.ds4q_quantize_init(16)
    tpl = Reader(recipe["template"])
    threads = args.threads

    for name, spec in repl.items():
        tkind, tdims, _ = tpl.tensors[name]
        out_path = payloads / (name + ".bin")
        if out_path.is_file() and manifest["tensors"].get(name, {}).get("sha256") == \
                hashlib.sha256(out_path.read_bytes()).hexdigest():
            print(f"{name}: verified existing payload", flush=True)
            continue
        # Source lookup: same tensor name, or fused ffn_gate_up_exps split.
        src_name = name
        fused = None
        if not src.has(src_name):
            base = name.rsplit(".", 2)[0]  # blk.N.ffn
            fused = f"{base}.ffn_gate_up_exps.weight"
            if not src.has(fused):
                print(f"{name}: SKIP no source tensor", flush=True)
                continue
        t0 = time.monotonic()
        try:
            vals3 = src.read_f32(src_name if fused is None else fused,
                                 cols=tdims[0])
        except SourceIncomplete:
            print(f"{name}: SKIP source shard still downloading", flush=True)
            continue
        if fused is not None:
            # fused gate_up: first half of rows is gate, second is up
            half = vals3.shape[1] // 2
            vals3 = vals3[:, :half, :] if "_gate_" in name else vals3[:, half:, :]
        vals = f32_to_bf16(vals3)
        del vals3

        def convert(e):
            v = np.ascontiguousarray(vals[e])
            w = imatrix.expert_weights(name, e, v)
            return quant.encode(v, spec["to"], w)

        chunks = []
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for i in range(0, vals.shape[0], threads * 2):
                chunks.extend(ex.map(
                    convert, range(i, min(i + threads * 2, vals.shape[0]))))
        raw = b"".join(chunks)
        expected = gguf_bytes(spec["to_id"], spec["dims"])
        if len(raw) != expected:
            raise SystemExit(f"{name}: payload {len(raw)} != expected {expected}")
        tmp = out_path.with_suffix(".incomplete")
        tmp.write_bytes(raw)
        tmp.replace(out_path)
        manifest["tensors"][name] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "to": spec["to"],
            "source": src_name if fused is None else fused,
            "seconds": time.monotonic() - t0}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(raw)} bytes in {time.monotonic()-t0:.1f}s",
              flush=True)
        del vals, raw
    imatrix.close()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(args):
    recipe = json.loads(args.recipe.read_text())
    tpl_path = recipe["template"]
    base = Reader(tpl_path)
    manifest = json.loads((args.payloads / "manifest.json").read_text())
    repl = recipe["replacements"]
    if set(manifest["tensors"]) != set(repl):
        raise SystemExit("manifest does not cover the recipe tensor set")
    alignment = base.kv.get("general.alignment", 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    metadata = {k: (base.kv_types[k], v) for k, v in base.kv.items()}
    metadata["general.name"] = (8, args.name or "UD-IQ2_XS-mtp requant experiment")
    rows = []
    offset = 0
    for name, (kind, dims, old) in base.tensors.items():
        if name in repl:
            spec = repl[name]
            kind, dims = spec["to_id"], spec["dims"]
            payload = args.payloads / (name + ".bin")
            if payload.stat().st_size != gguf_bytes(kind, dims) or \
               digest(payload) != manifest["tensors"][name]["sha256"]:
                raise SystemExit(f"invalid payload for {name}")
        size = gguf_bytes(kind, dims)
        rows.append((name, kind, dims, offset, size, old))
        offset = align(offset + size)
    prefix = b"GGUF" + struct.pack("<IQQ", 3, len(rows), len(metadata))
    prefix += b"".join(kv_bytes(k, t, v) for k, (t, v) in metadata.items())
    for name, kind, dims, pos, size, old in rows:
        prefix += w_str(name) + struct.pack("<I", len(dims))
        prefix += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, pos)
    data_start = align(len(prefix))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + ".incomplete")
    records = {}
    with tmp.open("wb") as out:
        out.write(prefix)
        out.write(bytes(data_start - len(prefix)))
        for name, kind, dims, pos, size, old in rows:
            out.seek(data_start + pos)
            if name in repl:
                srcf = (args.payloads / (name + ".bin")).open("rb")
            else:
                srcf = base.f
                srcf.seek(base.data_start + old)
            h = hashlib.sha256()
            left = size
            while left:
                chunk = srcf.read(min(left, 8 << 20))
                if not chunk:
                    raise SystemExit(f"short read: {name}")
                out.write(chunk)
                h.update(chunk)
                left -= len(chunk)
            if name in repl:
                srcf.close()
            records[name] = {"type": kind, "bytes": size, "sha256": h.hexdigest(),
                             "replaced": name in repl}
        out.truncate(data_start + offset)
    check = Reader(str(tmp))
    for name, rec in records.items():
        check.f.seek(check.data_start + check.tensors[name][2])
        h = hashlib.sha256()
        left = rec["bytes"]
        while left:
            chunk = check.f.read(min(left, 8 << 20))
            if not chunk:
                raise SystemExit(f"short verify read: {name}")
            h.update(chunk)
            left -= len(chunk)
        if h.hexdigest() != rec["sha256"]:
            raise SystemExit(f"written tensor mismatch: {name}")
    check.f.close()
    base.f.close()
    tmp.replace(args.out)
    with open(str(args.out) + ".json", "w") as f:
        json.dump({"template": tpl_path, "recipe": str(args.recipe),
                   "tensors": records, "bytes": args.out.stat().st_size,
                   "sha256": digest(args.out)}, f)
    print(f"verified {args.out}: {args.out.stat().st_size} bytes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--rules", type=Path, required=True,
                   help='JSON: {"rules": [{"match": regex, "from": [type_ids], '
                        '"to": "IQ2_XXS"}]}')
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--recipe", type=Path, required=True)
    q.add_argument("--source-dir", type=Path, required=True)
    q.add_argument("--imatrix", type=Path, required=True)
    q.add_argument("--library", type=Path, required=True)
    q.add_argument("--payloads", type=Path, required=True)
    q.add_argument("--threads", type=int, default=8)
    a = sub.add_parser("assemble")
    a.add_argument("--recipe", type=Path, required=True)
    a.add_argument("--payloads", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--name", default=None)
    args = ap.parse_args()
    {"plan": plan, "quantize": quantize, "assemble": assemble}[args.stage](args)


if __name__ == "__main__":
    main()
