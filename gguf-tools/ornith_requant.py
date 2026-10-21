#!/usr/bin/env python3
"""Phase-3.1 artifact: requantize the routed experts of the official Ornith-1.5-122B Q4_K_M GGUF into
the ds4-native IQ2 tier — routed gate/up into IQ2_XXS, routed down into Q2_K, guided per expert by
the unsloth llama.cpp imatrix (sha256-verified on load). Every other tensor (attention, GDN/ssm,
shared experts, MTP, norms, router, embeddings) is copied verbatim from the source, so the dense
path stays byte-identical to the Q4_K_M.

Facts this builds on (receipts in docs/Task/2026-10-21-ornith-122b-port.md):
- The checkpoint ships dead ff-rows: blocks whose d/dmin/scales are all zero (~74-81% of sampled
  gate rows at layer 0, fewer in later layers). Position agreement with unsloth's independent
  UD-IQ2_XXS file is 97.5% — the dead rows are a property of the checkpoint, not of any quantizer.
  They dequantize to exact zeros and requantize to zero blocks; no special casing.
- Dequant math is ggml upstream verbatim: q4_K = d*sc*nib - dmin*m with get_scale_min_k4 (repo
  fixture test_q4k_dot.c, the ds4q writer and ggml all agree); q6_K = d*scale*(q-32) with one int8
  scale per 16 values, d stored LAST in the block. Validated by gguf-tools/ornith_dequant_check.py
  against tests/quant_fixtures.h (Q6_K golden) and tests/q4k_dump.c (C reference dump).
- Encoding goes through libds4quants via qwen4_pack.GGMLQuantizer — the same packers that produced
  the prod qwen3.8 files (IQ2_XXS requires the imatrix; Q2_K also gets it: per-expert [in_dim] f32).
- Output written by tools/gguf_lite.write with a streaming proc callback per requantized tensor
  (no tens-of-GB buffering); alignment copied from the source header.

Usage:
  python3 gguf-tools/ornith_requant.py --src <Q4_K_M.gguf> --imatrix <imatrix.gguf> \
      --out <IQQ2.gguf> [--layers 0,5,11 | all] [--threads 8]
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent.parent / "tools"
GGUF_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(GGUF_TOOLS))
from gguf_lite import Reader  # noqa: E402
import gguf_lite  # noqa: E402
import ggml_tables  # noqa: E402
from qwen4_pack import GGMLQuantizer, default_quantizer_library  # noqa: E402

Q4_K, Q6_K, Q8_0, Q2_K, IQ2_XXS = 12, 14, 8, 10, 16
QTYPE_NAME = {Q2_K: "Q2_K", IQ2_XXS: "IQ2_XXS"}
IMATRIX_SHA256 = "6e5ada71fe8dc6b458d1087e1a753fa3e3b639c7ab207b644e8d11a14dc103ab"


def fail(msg):
    print(f"ornith-requant: {msg}", file=sys.stderr)
    sys.exit(1)


def _f16(b):
    return np.frombuffer(np.ascontiguousarray(b), dtype="<f2").astype("<f4")


def q4k_scale_min(s):
    """All eight (scale, min) pairs of the q4_K packed 12-byte scale field. Exact ggml
    get_scale_min_k4 for j = 0..7 (verified against ggml upstream, the repo fixture and writer)."""
    sc = np.empty((s.shape[0], 8), dtype=np.float32)
    mn = np.empty((s.shape[0], 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0x0F) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def deq_q4k(buf):
    """q4_K block bytes (nb*144) -> (nb, 256) float32, canonical ggml order: per 64-chunk the first
    32 values are low nibbles and the next 32 are high nibbles of the same 32 bytes; subblock
    s = 2*chunk + lo/hi uses get_scale_min_k4(s)."""
    b = np.frombuffer(np.ascontiguousarray(buf), np.uint8).reshape(-1, 144)
    nb = b.shape[0]
    if nb == 0:
        return np.zeros((0, 256), dtype=np.float32)
    d = _f16(b[:, 0:2])[:, None, None]
    dmin = _f16(b[:, 2:4])[:, None, None]
    sc, mn = q4k_scale_min(b[:, 4:16])
    qs = np.ascontiguousarray(b[:, 16:]).reshape(nb, 4, 32)
    nib = np.empty((nb, 8, 32), dtype=np.float32)
    nib[:, 0::2] = (qs & 15).astype(np.float32)
    nib[:, 1::2] = (qs >> 4).astype(np.float32)
    return (d * sc[:, :, None] * nib - dmin * mn[:, :, None]).reshape(nb, 256)


def deq_q6k(buf):
    """q6_K block bytes (nb*210) -> (nb, 256) float32, canonical ggml order: block is
    ql[128] | qh[64] | scales[16] (int8) | d (f16, LAST); per 128-half the four 32-groups are
    ql low/high x qh 2-bit planes minus 32, scaled by the int8 scale for each 16-group."""
    b = np.frombuffer(np.ascontiguousarray(buf), np.uint8).reshape(-1, 210)
    nb = b.shape[0]
    if nb == 0:
        return np.zeros((0, 256), dtype=np.float32)
    d = _f16(b[:, 208:210])[:, None]
    sc = b[:, 192:208].astype(np.int8).astype(np.float32)  # (nb, 16)
    ql = np.ascontiguousarray(b[:, 0:128]).reshape(nb, 2, 64)
    qh = np.ascontiguousarray(b[:, 128:192]).reshape(nb, 2, 32)
    vals = np.empty((nb, 2, 128), dtype=np.float32)
    for p in range(2):
        a = ql[:, p, 0:32].astype(np.int16)
        c = ql[:, p, 32:64].astype(np.int16)
        hb = qh[:, p].astype(np.int16)
        q = (
            (a & 15) | (((hb >> 0) & 3) << 4),
            (c & 15) | (((hb >> 2) & 3) << 4),
            ((a >> 4) & 15) | (((hb >> 4) & 3) << 4),
            ((c >> 4) & 15) | (((hb >> 6) & 3) << 4),
        )
        for k in range(4):
            s0 = np.repeat(sc[:, p * 8 + k * 2][:, None], 16, axis=1)
            s1 = np.repeat(sc[:, p * 8 + k * 2 + 1][:, None], 16, axis=1)
            scv = np.concatenate([s0, s1], axis=1)  # (nb, 32)
            vals[:, p, k * 32:(k + 1) * 32] = d * scv * (q[k] - 32)
    return vals.reshape(nb, 256)


def deq_q2k(buf):
    """Canonical ggml q2_K block bytes (nb*84) -> (nb, 256) float32: scales-first layout
    [scales[16] | qs[64] | d f16 | dmin f16]. Group g covers values [128p + 32j + 16i .. +15]
    (p = g//8, j = (g%8)//2, i = g%2) and reads the 16 bytes qs[32p + 16i + l] shifted by 2j;
    value = d*scale*q - dmin*min (canonical MINUS), scale = scales[g]&15, min = scales[g]>>4."""
    b = np.frombuffer(np.ascontiguousarray(buf), np.uint8).reshape(-1, 84)
    nb = b.shape[0]
    if nb == 0:
        return np.zeros((0, 256), dtype=np.float32)
    d = _f16(b[:, 80:82])[:, None]
    dmin = _f16(b[:, 82:84])[:, None]
    s = b[:, 0:16]
    qs = np.ascontiguousarray(b[:, 16:80])
    q = np.empty((nb, 16, 16), dtype=np.float32)
    for g in range(16):
        base = 32 * (g // 8) + 16 * (g % 2)
        shift = 2 * ((g % 8) // 2)
        q[:, g, :] = ((qs[:, base:base + 16] >> shift) & 3).astype(np.float32)
    sc = (s & 15).astype(np.float32)[:, :, None]
    mn = (s >> 4).astype(np.float32)[:, :, None]
    return (d[:, None] * sc * q - dmin[:, None] * mn).reshape(nb, 256)


def deq_iq2xxs(buf):
    """Canonical ggml iq2_xxs block bytes (nb*66) -> (nb, 256) float32, exact
    dequantize_row_iq2_xxs: d f16, then 8 groups of 8 bytes where the first u32 carries four
    grid-row indices and the second u32 carries four 7-bit sign indices + a 4-bit exponent;
    value = d*(0.5+exp)*0.25 * grid[j] * (-1 if sign bit j)."""
    grids = ggml_tables.load()
    b = np.frombuffer(np.ascontiguousarray(buf), np.uint8).reshape(-1, 66)
    nb = b.shape[0]
    if nb == 0:
        return np.zeros((0, 256), dtype=np.float32)
    d = _f16(b[:, 0:2])[:, None]
    words = np.ascontiguousarray(b[:, 2:66]).view("<u4").reshape(nb, 8, 2)  # (nb, ib32, aux pair)
    gidx = (words[:, :, 0:1] >>
            np.uint32(8) * np.arange(4, dtype=np.uint32).reshape(1, 1, 4)) & np.uint32(255)  # (nb,8,4)
    a1 = words[:, :, 1]
    expo = (a1 >> np.uint32(28)).astype(np.float32)  # (nb, 8)
    db = d * (0.5 + expo) * 0.25  # (nb, 8)
    signs_idx = (a1[:, :, None].astype(np.int64) >>
                 np.int64(7) * np.arange(4, dtype=np.int64)) & 127  # (nb, 8, 4)
    sl = grids["ksigns_iq2xs"][signs_idx.astype(np.int64)]  # (nb, 8, 4) sign bytes
    sign_bits = 1 << np.arange(8, dtype=np.int64)  # kmask_iq2xs
    tbl = grids["iq2xxs_grid"]
    gi = gidx.astype(np.int64)
    out = np.empty((nb, 8, 4, 8), dtype=np.float32)
    for j in range(8):
        out[:, :, :, j] = ((tbl[gi] >> np.uint64(8 * j)) & np.uint64(255)).astype(np.float32)
    for l in range(4):
        sb = np.repeat(sl[:, :, l][:, :, None], 8, axis=2).astype(np.int64)  # (nb, 8, 8)
        signs = np.where((sb & sign_bits[None, None, :]) != 0, -1.0, 1.0)
        out[:, :, l, :] = db[:, :, None] * out[:, :, l, :] * signs
    return out.reshape(nb, 256)


class ImatrixMap:
    """unsloth llama.cpp imatrix: '<name>.in_sum2' F32 with gguf dims [in, e] (numpy (e, in)) and
    '<name>.counts'. sha256-checked on load; families load lazily and stay resident."""

    def __init__(self, path, sha256):
        self.path = str(path)
        digest = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            fail(f"imatrix checksum mismatch; expected {sha256}, got {digest.hexdigest()}")
        self.r = Reader(self.path)
        self.by_name = {t["name"]: t for t in self.r.tensors}
        self.cache = {}

    def _ensure(self, tensor_name, n_experts):
        if tensor_name in self.cache:
            return self.cache[tensor_name]
        st = self.by_name.get(tensor_name + ".in_sum2")
        ct = self.by_name.get(tensor_name + ".counts")
        if st is None or ct is None:
            fail(f"imatrix entry missing for {tensor_name}")
        with open(self.path, "rb") as f:
            f.seek(st["abs"])
            sums = np.frombuffer(f.read(st["nbytes"]), "<f4").reshape(
                st["dims"][1], st["dims"][0])
            f.seek(ct["abs"])
            counts = np.frombuffer(f.read(ct["nbytes"]), "<f4").reshape(-1)
        if counts.shape[0] != n_experts or sums.shape[0] != n_experts:
            fail(f"imatrix geometry mismatch for {tensor_name}")
        self.cache[tensor_name] = (sums, counts)
        return self.cache[tensor_name]

    def pure_weights(self, tensor_name, n_experts, in_dim, e):
        """seam: the calibrated weights for expert e, or None when the calibration never saw
        that expert (the caller may then compare against its own fallback)."""
        sums, counts = self._ensure(tensor_name, n_experts)
        if sums.shape[1] != in_dim:
            fail(f"imatrix in-dim mismatch for {tensor_name}: {sums.shape[1]} != {in_dim}")
        if counts[e] == 0.0:
            return None
        return np.ascontiguousarray(sums[e] / counts[e], dtype="<f4")

    def expert_weights(self, tensor_name, n_experts, in_dim, e, values):
        """weights = sums[e] / counts[e] (llama.cpp convention, per input column); falls back to
        the sum of squares over the expert's own rows when calibration never saw that expert."""
        sums, counts = self._ensure(tensor_name, n_experts)
        if sums.shape[1] != in_dim:
            fail(f"imatrix in-dim mismatch for {tensor_name}: {sums.shape[1]} != {in_dim}")
        if counts[e] > 0.0:
            return np.ascontiguousarray(sums[e] / counts[e], dtype="<f4")
        w = np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float32)
        if not np.all(np.isfinite(w)):
            fail(f"{tensor_name}: nonfinite weight-energy fallback for expert {e}")
        return np.ascontiguousarray(w, dtype="<f4")


def build_plan(src, layers, threads, quantizer, imap, down_type="Q2_K"):
    """Per requantized routed-expert tensor: dims stay, type changes, nbytes shrink; data is
    produced by a streaming proc(out, h) that writes exactly nbytes bytes (and hashes them).
    down_type="COPY" leaves routed down tensors verbatim (the conservative fallback tier)."""
    plan = []
    for t in src.tensors:
        n = t["name"]
        if not (n.startswith("blk.") and n.endswith(".weight") and
                any(f in n for f in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")) and
                not n.startswith("blk.48.")):
            continue
        il = int(n.split(".")[1])
        if layers != "all" and str(il) not in layers:
            continue
        in_dim, out_dim, n_exp = t["dims"]
        is_down = "ffn_down_exps" in n
        if is_down and down_type == "COPY":
            continue
        if is_down:
            if t["type"] not in (Q4_K, Q6_K):
                fail(f"{n}: unexpected routed down source type {t['type']} (want Q4_K/Q6_K)")
            dst_type, src_block = Q2_K, (210 if t["type"] == Q6_K else 144)
        else:
            if t["type"] != Q4_K:
                fail(f"{n}: unexpected routed gate/up source type {t['type']} (want Q4_K)")
            dst_type, src_block = IQ2_XXS, 144
        blocks_per_row = in_dim // 256
        src_row_bytes = blocks_per_row * src_block
        dst_row_size = blocks_per_row * (84 if dst_type == Q2_K else 66)
        new_nbytes = dst_row_size * out_dim * n_exp
        src_path, src_abs, src_type, src_name = str(src.path), t["abs"], t["type"], n

        def proc(out, h, is_down=is_down, in_dim=in_dim, out_dim=out_dim, n_exp=n_exp,
                 src_path=src_path, src_abs=src_abs, src_type=src_type, src_name=src_name,
                 src_block=src_block, src_row_bytes=src_row_bytes, dst_row_size=dst_row_size):
            deq = deq_q6k if (is_down and src_type == Q6_K) else deq_q4k
            qtype = QTYPE_NAME[Q2_K if is_down else IQ2_XXS]

            def job(e):
                with open(src_path, "rb") as f:
                    f.seek(src_abs + e * out_dim * src_row_bytes)
                    buf = f.read(out_dim * src_row_bytes)
                if len(buf) != out_dim * src_row_bytes:
                    fail(f"{src_name}: short read at expert {e} ({len(buf)} bytes)")
                vals = deq(buf).reshape(out_dim, in_dim)
                w = imap.expert_weights(src_name, n_exp, in_dim, e, vals)
                return quantizer.encode(vals, qtype, w)

            written = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
                order = list(pool.map(job, range(n_exp)))
            for enc in order:
                if len(enc) != dst_row_size * out_dim:
                    fail(f"{src_name}: encoded expert bytes {len(enc)} != {dst_row_size * out_dim}")
                out.write(enc)
                h.update(enc)
                written += len(enc)
            return written

        plan.append({
            "name": n, "dims": t["dims"], "type": dst_type, "nbytes": new_nbytes, "proc": proc,
            "src_nbytes": t["nbytes"], "src_type": src_type, "layer": il, "is_down": is_down,
        })
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="'all' or a comma list like 0,5,11")
    ap.add_argument("--down-type", default="Q2_K", choices=("Q2_K", "COPY"),
                    help="routed down target: requant to Q2_K, or leave the source bytes verbatim")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trust-imatrix", action="store_true", help="skip the imatrix sha256 check")
    args = ap.parse_args()

    src = Reader(args.src)
    imap = ImatrixMap(args.imatrix, args.trust_imatrix and "0" * 64 or IMATRIX_SHA256)
    quantizer = GGMLQuantizer(default_quantizer_library())
    layers = args.layers.strip().lower()
    if layers != "all":
        layers = {x.strip() for x in layers.split(",") if x.strip()}
    print(f"ornith-requant: layers={args.layers} threads={args.threads}")

    plan = build_plan(src, layers, args.threads, quantizer, imap, args.down_type)
    kept = {e["name"]: e for e in plan}
    total_new = sum(e["nbytes"] for e in plan)
    total_old = sum(e["src_nbytes"] for e in plan)
    print(f"ornith-requant: {len(plan)} routed tensors, {total_old/2**30:.2f} GiB -> "
          f"{total_new/2**30:.2f} GiB")

    tensors_out = []
    for t in src.tensors:
        entry = {k: t[k] for k in ("name", "dims", "type", "nbytes")}
        hit = kept.get(t["name"])
        if hit:
            entry["type"] = hit["type"]
            entry["nbytes"] = hit["nbytes"]
            entry["proc"] = hit["proc"]
        else:
            entry["src"] = (args.src, t["abs"])
        tensors_out.append(entry)

    t0 = time.time()
    shas = gguf_lite.write(args.out, src.kv, tensors_out, alignment=src.alignment, nocache=True)
    dt = time.time() - t0
    size = os.path.getsize(args.out)
    print(f"ornith-requant: wrote {args.out} ({size/2**30:.2f} GiB) in {dt:.0f}s")

    out_reader = Reader(args.out)
    total = sum(x["nbytes"] for x in out_reader.tensors)
    print(f"ornith-requant: re-read OK: {len(out_reader.tensors)} tensors, "
          f"{total/2**30:.2f} GiB tensor bytes, data_start={out_reader.data_start}")

    receipt = {
        "src": args.src, "imatrix": args.imatrix, "out": args.out,
        "layers": args.layers, "threads": args.threads,
        "seconds": round(dt, 1), "bytes": size,
        "plan": [{"name": e["name"], "type": e["type"], "nbytes": e["nbytes"],
                  "src_nbytes": e["src_nbytes"], "src_type": e["src_type"],
                  "layer": e["layer"], "is_down": e["is_down"]} for e in plan],
        "sha256_of_output_tensors": shas,
    }
    with open(GGUF_TOOLS / "ornith_requant.receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print("ornith-requant: receipt -> gguf-tools/ornith_requant.receipt.json")


if __name__ == "__main__":
    main()
