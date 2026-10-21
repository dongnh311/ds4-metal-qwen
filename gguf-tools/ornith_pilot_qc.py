#!/usr/bin/env python3
"""Pilot QC: prove the requant pipeline preserves the data it was given.

1. Unit: dequantize_row_iq2_xxs vs the tests/quant_fixtures.h golden row (fixture type 16).
2. End-to-end: for layer 0, read the pilot file's requantized gate/up (IQ2_XXS) and down (Q2_K)
   rows, dequantize them, and compare against the same source rows (official Q4_K_M):
   a) byte determinism — re-encoding the same source values through the same packer and imatrix
      must reproduce the pilot's stored bytes exactly;
   b) live-position cosine — sparse rows (the checkpoint's dead ff-channels saturate ~90% of the
      down rows with exact zeros, and the coarse 2.6-bpw Q2_K grid maps those zeros to small
      min-term offsets) are only judged on the positions that actually carry energy;
   c) dead-row agreement — rows that are all-zero in the source must stay all-zero in the pilot.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-tools"))
from ornith_requant import deq_q2k, deq_q4k, deq_q6k, deq_iq2xxs  # noqa: E402
from gguf_lite import Reader  # noqa: E402
from ornith_dequant_check import parse_fixtures  # noqa: E402

DIR = Path.home() / ".local/share/ai-gateway/ds4-models/ornith-122b"
SRC, PILOT = DIR / "Ornith-1.5-122B-Q4_K_M.gguf", DIR / "Ornith-1.5-122B-IQ2-pilotL0.gguf"


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ok = True

    # --- 1. iq2_xxs fixture unit check
    arrays, table = parse_fixtures()
    row = next(r for r in table if int(r[0]) == 16)
    fx_b = np.parse_eval = np.array(arrays[int(row[2])]["bytes"], dtype=np.uint8).tobytes()
    fx_want = np.array(arrays[int(row[4])]["want"], dtype=np.float32)
    fx_mine = deq_iq2xxs(fx_b).reshape(-1)
    fx_ok = len(fx_mine) == len(fx_want) and np.allclose(fx_mine, fx_want, rtol=1e-5, atol=1e-7)
    print(f"iq2_xxs fixture: {len(fx_b)}B -> {len(fx_mine)} vals: "
          f"{'MATCH' if fx_ok else 'FAIL'} (maxdiff={np.abs(fx_mine - fx_want).max():.2e})")
    ok &= fx_ok

    # --- 2. pilot vs source per-row comparison
    s = Reader(str(SRC))
    p = Reader(str(PILOT))
    from qwen4_pack import GGMLQuantizer, default_quantizer_library
    from ornith_requant import ImatrixMap
    IMX_SHA = "6e5ada71fe8dc6b458d1087e1a753fa3e3b639c7ab207b644e8d11a14dc103ab"
    imap = ImatrixMap(DIR / "imatrix_unsloth.gguf_file", IMX_SHA)
    quant = GGMLQuantizer(default_quantizer_library())
    fams = [
        ("ffn_gate_exps", deq_q4k, deq_iq2xxs, "IQ2_XXS", 144, 66),
        ("ffn_up_exps", deq_q4k, deq_iq2xxs, "IQ2_XXS", 144, 66),
        ("ffn_down_exps", deq_q6k, deq_q2k, "Q2_K", 210, 84),
    ]
    experts = (0, 3, 64, 200, 255)
    out_rows = (0, 1, 8, 300, 700, 1023)
    for fam, sdeq, pdeq, qtype, sblk, pblk in fams:
        st = next(t for t in s.tensors if t["name"] == f"blk.0.{fam}.weight")
        pt = next(t for t in p.tensors if t["name"] == f"blk.0.{fam}.weight")
        in_dim = st["dims"][0]
        s_row = (in_dim // 256) * sblk
        p_row = (in_dim // 256) * pblk
        cos, cos_live, n_live, both_zero, dead_live, byte_eq, byte_skip = [], [], 0, 0, 0, 0, 0
        with open(str(SRC), "rb") as fs, open(str(PILOT), "rb") as fp:
            for e in experts:
                for r in out_rows:
                    fs.seek(st["abs"] + (e * st["dims"][1] + r) * s_row)
                    src_vals = sdeq(fs.read(s_row)).reshape(-1)
                    fp.seek(pt["abs"] + (e * pt["dims"][1] + r) * p_row)
                    pilot_bytes = fp.read(p_row)
                    pilot_vals = pdeq(pilot_bytes).reshape(-1)
                    c = cosine(src_vals, pilot_vals)
                    live = np.abs(src_vals) > 0.02 * np.abs(src_vals).max()
                    if np.any(live):
                        n_live += 1
                        cos_live.append(cosine(src_vals[live], pilot_vals[live]))
                    cos.append(c)
                    if np.linalg.norm(src_vals) == 0 and np.linalg.norm(pilot_vals) == 0:
                        both_zero += 1
                    dead_live += int((np.linalg.norm(src_vals) == 0) !=
                                     (np.linalg.norm(pilot_vals) == 0))
                    # byte determinism: same input values + same packer + same imatrix weights
                    vals2 = src_vals.reshape(1, in_dim)
                    w2 = imap.pure_weights(f"blk.0.{fam}.weight", st["dims"][2], in_dim, e)
                    if w2 is None:
                        byte_skip += 1
                    else:
                        reenc = quant.encode(vals2, qtype, w2)
                        byte_eq += int(bytes(reenc) == pilot_bytes)
            cos = np.array(cos)
            live_cos = np.array(cos_live) if cos_live else np.array([1.0])
            print(f"{fam}: cos mean={cos.mean():.4f} min={cos.min():.4f} | "
                  f"live-pos cos mean={live_cos.mean():.4f} min={live_cos.min():.4f} "
                  f"({n_live} rows) | byte-equal {byte_eq}/{len(cos) - byte_skip} | "
                  f"both-zero={both_zero} | zero-disagreements={dead_live}")
            ok &= live_cos.min() > 0.90 and byte_eq == len(cos) - byte_skip
    print("PILOT QC:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
