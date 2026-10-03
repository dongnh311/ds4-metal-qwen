#!/usr/bin/env python3
"""Stage-0 log parser (spec 2026-10-02-ornith-decode-design.md section 3).

  parse_stage0.py DIR   -> one markdown row per <ctx>-<run>.log in DIR
"""
import glob
import os
import re
import sys

PREFILL_GPU_US = 50000      # a buffer with more GPU time than this is a prefill chunk, not a decode step
SPLIT_KEYS = ("gdn-mix", "gdn-moe", "attn-mix", "attn-moe", "head")


def gen_tps(text):
    m = re.findall(r"generation: ([\d.]+) t/s", text)
    return float(m[-1]) if m else None


def cb_decode(text):
    """Decode buffers: those after the last prefill chunk (a buffer over PREFILL_GPU_US of GPU
    time).  An --mtp prefill leaves small MTP catch-up buffers between its chunks, so a
    per-buffer size filter alone would count them as decode."""
    rows = [(float(g), float(gp)) for g, gp in
            re.findall(r"ds4: cb [^:]*: driver [\d.]+ us, queue-wait [\d.]+ us, gpu ([\d.]+) us, "
                       r"gap-from-prev-gpu-end ([\d.]+) us", text)]
    last_prefill = max((i for i, (g, _) in enumerate(rows) if g > PREFILL_GPU_US), default=-1)
    decode = rows[last_prefill + 1:]
    gpu = sum(g for g, _ in decode) / 1000.0
    gap = sum(gp for _, gp in decode) / 1000.0
    return {"n": len(decode), "gpu_ms": gpu, "gap_ms": gap, "busy": gpu / (gpu + gap) if gpu + gap > 0 else None}


def split_means(text, what):
    rows = [ln for ln in text.splitlines() if what in ln]
    out = {}
    for k in SPLIT_KEYS:
        vals = [float(m.group(1)) for ln in rows for m in [re.search(r"\b%s ([\d.]+)" % re.escape(k), ln)] if m]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def last_line(text, prefix):
    rows = [ln[ln.index(prefix):] for ln in text.splitlines() if prefix in ln]
    return rows[-1] if rows else None


def fmt(v, f="%.2f"):
    return "missing" if v is None else f % v


def main():
    d = sys.argv[1]
    print("| run | gen t/s | decode CBs | GPU busy | split ms/step (gdn-mix/gdn-moe/attn-mix/attn-moe/head) "
          "| verify split ms/chunk T=2 | spec stats | overlap |")
    print("|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(d, "*.log"))):
        text = open(path, errors="replace").read()
        cb = cb_decode(text)
        sd = split_means(text, "decode split ms/step")
        sv = split_means(text, "T=2): gdn-mix")
        print("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
            os.path.basename(path)[:-4], fmt(gen_tps(text)), cb["n"], fmt(cb["busy"], "%.3f"),
            "/".join(fmt(sd.get(k)) for k in SPLIT_KEYS) if sd else "missing",
            "/".join(fmt(sv.get(k)) for k in SPLIT_KEYS) if sv else "missing",
            last_line(text, "Ornith spec stats:") or "missing",
            last_line(text, "Ornith verify expert overlap:") or "missing"))


if __name__ == "__main__":
    main()
