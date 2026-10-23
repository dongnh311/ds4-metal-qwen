#!/usr/bin/env python3
"""Check the DFlash2 mask-token row inside the target GGUF's token_embd.

DFlash2 drafter reads the target's mask-token embedding (id 248077). The model
card warns: GGUF builds of the target collapse that row (~1e-6 scale) to ~zero,
which halves draft acceptance. mask_embedding.safetensors ships an exact BF16
copy of the row for engines to load instead.

Outputs (CPU-only, no model/GPU load):
  - Q4_K block scales + dequanted stats for the mask row vs normal rows
  - mask_row_248077.f32 sidecar (E*4 bytes) next to the target GGUF
"""
import glob
import json
import os
import struct
import sys

import numpy as np
from gguf import GGUFReader

MASK_ID = 248077
E = 3072
BLOCK = 256            # elements per Q4_K block
BLOCK_BYTES = 144

MODELS_DIR = os.path.expanduser("~/.local/share/ai-gateway/ds4-models/ornith-122b")
ST_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--incoai--Ornith-1.5-122B-DFlash2/"
    "snapshots/9baefc1bec85c08193ab442e0ca5f7057669b61d/mask_embedding.safetensors")


def get_scale_min_k4(j, q):
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 15
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def deq_q4k_block(blk):
    d, dmin = struct.unpack_from("<ee", blk, 0)
    scales = blk[4:16]
    qs = blk[16:144]
    out = np.empty(BLOCK, dtype=np.float32)
    for c in range(4):
        d1, m1 = get_scale_min_k4(2 * c, scales)
        d2, m2 = get_scale_min_k4(2 * c + 1, scales)
        s1 = np.float32(d) * np.float32(d1)
        mn1 = np.float32(dmin) * np.float32(m1)
        s2 = np.float32(d) * np.float32(d2)
        mn2 = np.float32(dmin) * np.float32(m2)
        q = qs[32 * c:32 * c + 32]
        lo = np.frombuffer(q, dtype=np.uint8)
        out[64 * c:64 * c + 32] = s1 * (lo & 0xF).astype(np.float32) + mn1
        out[64 * c + 32:64 * c + 64] = s2 * (lo >> 4).astype(np.float32) + mn2
    return out, d, dmin


def tensor_bytes(reader, tensor, path, candidate_offsets):
    """Try candidate absolute file offsets; pick the one whose probe row decodes sanely."""
    size = tensor.n_bytes
    need = (MASK_ID + 1) * (E // BLOCK) * BLOCK_BYTES
    best = None
    for off in candidate_offsets:
        if off + size > os.path.getsize(path):
            print(f"  [offset {off}] exceeds file size, skip")
            continue
        with open(path, "rb") as fh:
            fh.seek(off)
            raw = fh.read(size)
        probe = row_blocks(memoryview(raw), 100)
        vals, _, _ = deq_q4k_block(probe[:BLOCK_BYTES])
        amax = float(np.abs(vals).max())
        bad = (not np.isfinite(vals).all()) or amax > 1e4 or amax < 1e-8
        print(f"  [offset {off}] token#100 probe absmax={amax:.4g} "
              f"finite={bool(np.isfinite(vals).all())} -> {'BAD' if bad else 'OK'}")
        if not bad:
            return raw
        best = bad and best is None and raw
    if best is not None:
        return best
    raise SystemExit("no sane tensor offset found")


def row_blocks(raw, token):
    base = token * (E // BLOCK) * BLOCK_BYTES
    return raw[base:base + (E // BLOCK) * BLOCK_BYTES]


def row_stats(raw, token, label):
    rawrow = row_blocks(raw, token)
    vals, ds, dmins = [], [], []
    for b in range(E // BLOCK):
        blk = rawrow[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        out, d, dmin = deq_q4k_block(blk)
        vals.append(out)
        ds.append(d)
        dmins.append(dmin)
    vals = np.concatenate(vals)
    print(f"  {label:14s} tok={token:6d}  "
          f"absmax={np.abs(vals).max():.6g}  std={vals.std():.3g}  "
          f"|d| fp16: {[f'{abs(x):.3g}' for x in ds]}  "
          f"|dmin|: {[f'{abs(x):.3g}' for x in dmins]}")
    return vals


def load_mask_safetensors():
    with open(ST_PATH, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hlen))
    name = next(k for k in hdr if k != "__metadata__")
    info = hdr[name]
    a, b = info["data_offsets"]
    with open(ST_PATH, "rb") as fh:
        fh.seek(8 + hlen + a)
        blob = fh.read(b - a)
    dtype = info["dtype"]
    if dtype == "BF16":
        row = (np.frombuffer(blob, dtype=np.uint16)
               .astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F32":
        row = np.frombuffer(blob, dtype=np.float32).copy()
    else:
        raise SystemExit(f"unexpected sidecar dtype {dtype}")
    return name, info["shape"], row


def main():
    targets = [p for p in glob.glob(os.path.join(MODELS_DIR, "*122B*Q4_K_M*.gguf"))
               if "DFlash2" not in os.path.basename(p)]
    if not targets:
        raise SystemExit("target Ornith-1.5-122B-Q4_K_M.gguf not found")
    gguf_path = targets[0]
    print(f"target: {os.path.basename(gguf_path)}")

    r = GGUFReader(gguf_path)
    te = next(t for t in r.tensors if t.name == "token_embd.weight")
    print(f"token_embd: type={te.tensor_type} shape={list(te.shape)} "
          f"n_bytes={getattr(te, 'n_bytes', '?')} "
          f"r.data_offset={r.data_offset} t.data_offset={te.data_offset}")
    raw = tensor_bytes(
        r, te, gguf_path,
        [int(te.data_offset),  # candidate: already absolute
         int(r.data_offset) + int(te.data_offset)])  # candidate: section-relative
    if len(raw) < (MASK_ID + 1) * (E // BLOCK) * BLOCK_BYTES:
        raise SystemExit(f"raw tensor view too small ({len(raw)})")

    print("row dequanted stats (Q4_K, 12 blocks each):")
    row_stats(raw, 100, "normal #100")
    row_stats(raw, 248076, "neighbor")
    mask_vals = row_stats(raw, MASK_ID, "MASK 248077")

    print(f"\nmask-row verdict: absmax = {np.abs(mask_vals).max():.3g} "
          f"(typical normal row ~1e-2..1e-1)")

    name, shape, side = load_mask_safetensors()
    print(f"\nsafetensors: {name} dtype-shape {shape} "
          f"absmax={np.abs(side).max():.6g} std={side.std():.3g} "
          f"mean={side.mean():.3g}")
    sidecar = os.path.join(MODELS_DIR, "mask_row_248077.f32")
    with open(sidecar, "wb") as fh:
        fh.write(side.astype(np.float32).tobytes())
    print(f"sidecar written: {sidecar} ({len(side) * 4} bytes, E={len(side)})")
    ref = side.view(np.uint32) >> 16          # original bf16 bits
    back = (ref.astype(np.uint32) << 16).view(np.float32)
    print(f"sidecar bf16 roundtrip err={np.abs(back - side).max():.3g} (expect 0)")


if __name__ == "__main__":
    main()
