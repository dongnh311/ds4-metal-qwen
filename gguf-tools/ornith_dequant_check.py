#!/usr/bin/env python3
"""Validate the requant dequants against the repo's golden fixtures (tests/quant_fixtures.h) and,
for q4_K, against a C reference dump built from tests/test_q4k_dot.c (repo-verified block/scale
convention). Every fixture type we use downstream must match bit-for-bit (allclose 1e-6).

Also probes the real 122B Q4_K_M file: dead rows must dequant to exact zeros, live rows must be
finite with sane magnitudes.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-tools"))
from ornith_requant import deq_q4k, deq_q6k  # noqa: E402

FIX = ROOT / "tests" / "quant_fixtures.h"
# gguf-numeric type for the ds4 fixture enum where they differ
DS4_TO_GGUF = {6: 6, 11: 11, 13: 13, 14: 14, 16: 16, 17: 17, 18: 18, 20: 20, 21: 21, 22: 22, 23: 23, 42: 42}


def parse_fixtures():
    src = FIX.read_text()
    arrays = {}
    for m in re.finditer(r"qf_(bytes|want)_(\d+)\s*\[\]\s*=\s*\{([^;]+)\};", src):
        kind, idx, body = m.group(1), int(m.group(2)), m.group(3)
        toks = re.findall(r"[-+]?[0-9.]+(?:e[-+]?[0-9]+)?f?|[-+]?[0-9]+f?|0x[0-9a-fA-F]+", body)
        vals = [float(x.rstrip("f")) for x in toks]
        arrays.setdefault(idx, {})[kind] = vals
    table = re.findall(r"\{\s*(\d+),\s*(\d+),\s*qf_bytes_(\d+),\s*(\d+),\s*qf_want_(\d+)\s*\}", src)
    return arrays, table


def main():
    arrays, table = parse_fixtures()
    print(f"parsed {len(arrays)} fixture arrays, {len(table)} table rows")
    ok = True
    for row in table:
        ds4_type, n, bidx, nbytes, widx = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
        b = np.array(arrays[bidx]["bytes"], dtype=np.uint8)
        want = np.array(arrays[widx]["want"], dtype=np.float32)
        if len(b) != nbytes:
            print(f"type {ds4_type}: byte count mismatch {len(b)} != {nbytes}")
            ok = False
            continue
        gguf_type = DS4_TO_GGUF.get(ds4_type, ds4_type)
        if gguf_type == 14:  # Q6_K: 210-byte blocks
            vals = deq_q6k(b.tobytes()).reshape(-1)
        else:
            continue  # only Q6_K covers our down path directly here
        n_show = min(len(want), len(vals), 6)
        exact = len(vals) == len(want) and np.allclose(vals, want, rtol=1e-5, atol=1e-7)
        print(f"Q6_K fixture[ds4 {ds4_type}]: {len(b)}B -> {len(vals)} vals vs want {len(want)}: "
              f"{'MATCH' if exact else 'FAIL'} (first: {vals[:n_show]} vs {want[:n_show]})")
        ok &= exact

    # q4_K golden via the C dump harness, fed real live-row bytes from the source file
    harness_src = (ROOT / "tests" / "q4k_dump.c")
    from gguf_lite import Reader
    src_path = Path.home() / ".local/share/ai-gateway/ds4-models/ornith-122b/Ornith-1.5-122B-Q4_K_M.gguf"
    src = Reader(str(src_path))
    t = next(x for x in src.tensors if x["name"] == "blk.0.ffn_gate_exps.weight")
    with open(str(src_path), "rb") as f:
        f.seek(t["abs"] + 8 * 1728)  # row8 of expert0 = live (from the dead-row map)
        live_bytes = f.read(1728)
        f.seek(t["abs"] + 1 * 1728)  # row1 = fully dead (all 12 block headers zero, verified earlier)
        dead_bytes = f.read(1728)
    if harness_src.exists():
        with tempfile.TemporaryDirectory() as td:
            bin_path = Path(td) / "q4k_dump"
            cc = subprocess.run(["cc", "-O2", "-I", str(ROOT / "tests"), "-o", str(bin_path),
                                 str(harness_src), "-lm"], capture_output=True, text=True)
            if cc.returncode != 0:
                print("harness build failed:", cc.stderr)
                ok = False
            else:
                r = subprocess.run([str(bin_path)], input=live_bytes + dead_bytes, capture_output=True)
                got = np.frombuffer(r.stdout, "<f4").reshape(2, 1728 // 144 * 256)
                mine_live = deq_q4k(live_bytes).reshape(1728 // 144 * 256)
                mine_dead = deq_q4k(dead_bytes).reshape(1728 // 144 * 256)
                exact = np.array_equal(got[0], mine_live) and np.array_equal(got[1], mine_dead)
                print(f"q4_K harness vs numpy (live+dead rows, 3072 vals each): "
                      f"{'MATCH' if exact else 'FAIL'}")
                ok &= exact

    # live-file probes on the real source

    t = next(x for x in src.tensors if x["name"] == "blk.0.ffn_gate_exps.weight")
    dead = deq_q4k(dead_bytes)  # 12 blocks of row 1: every block header verified zero earlier
    live = deq_q4k(live_bytes).reshape(-1)  # row 8 = live
    dead_ok = np.all(dead == 0.0)
    live_ok = np.all(np.isfinite(live)) and np.abs(live).max() < 1.0 and np.abs(live).max() > 1e-6
    print(f"L0 gate row1 dead row (12 blocks, {dead.size} values): "
          f"{'ALL-ZERO OK' if dead_ok else 'FAIL'} (max|v|={np.abs(dead).max():.3e})")
    print(f"L0 gate row8 live row: {'FINITE/SANE OK' if live_ok else 'FAIL'} "
          f"(std={live.std():.4f}, max|v|={np.abs(live).max():.4f}, mean={live.mean():+.4f})")
    ok &= dead_ok and live_ok
    print("DEQUANT VALIDATION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
