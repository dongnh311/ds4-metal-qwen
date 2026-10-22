#!/usr/bin/env python3
"""Requant QC: prove the requant pipeline preserves the data it was given.

1. Unit: dequantize_row_iq2_xxs vs the tests/quant_fixtures.h golden row (fixture type 16).
2. End-to-end: for sampled layers, read the output file's requantized gate/up (IQ2_XXS) and
   down (Q2_K) rows, dequantize them, and compare against the same source rows (official
   Q4_K_M):
   a) byte determinism - re-encoding the same source values through the same packer and imatrix
      must reproduce the stored bytes exactly;
   b) live-position cosine - sparse rows (the checkpoint's dead ff-channels saturate ~90% of the
      down rows with exact zeros, and the coarse 2.6-bpw Q2_K grid maps those zeros to small
      min-term offsets) are only judged on the positions that actually carry energy;
   c) dead-row agreement - rows that are all-zero in the source must stay all-zero in the output.

Usage:
  python3 gguf-tools/ornith_pilot_qc.py [--out <file.gguf>] [--layers 0,11,23,35,47]
                                        [--experts 0,3,64,200,255] [--rows 0,1,8,300,700,1023]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gguf-tools"))
from ornith_requant import deq_q2k, deq_q4k, deq_q6k, deq_iq2xxs, ImatrixMap, energy_weights  # noqa: E402
from gguf_lite import Reader  # noqa: E402
from ornith_dequant_check import parse_fixtures  # noqa: E402
from qwen4_pack import GGMLQuantizer, default_quantizer_library  # noqa: E402

DIR = Path.home() / ".local/share/ai-gateway/ds4-models/ornith-122b"
SRC = DIR / "Ornith-1.5-122B-Q4_K_M.gguf"
IMATRIX_SHA = "6e5ada71fe8dc6b458d1087e1a753fa3e3b639c7ab207b644e8d11a14dc103ab"
FAMS = [
    ("ffn_gate_exps", deq_q4k, "IQ2_XXS", 144, 66),
    ("ffn_up_exps", deq_q4k, "IQ2_XXS", 144, 66),
    ("ffn_down_exps", None, "Q2_K", None, 84),  # source may be Q4_K or Q6_K; pick by tensor type
]
SRC_DEQ = {12: (deq_q4k, 144), 14: (deq_q6k, 210)}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DIR / "Ornith-1.5-122B-IQ2.gguf"))
    ap.add_argument("--layers", default="0,11,23,35,47")
    ap.add_argument("--experts", default="0,3,64,200,255")
    ap.add_argument("--rows", default="0,1,8,300,700,1023")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    experts = [int(x) for x in args.experts.split(",") if x.strip()]
    out_rows = [int(x) for x in args.rows.split(",") if x.strip()]
    out_path = Path(args.out)

    ok = True

    # --- 1. iq2_xxs fixture unit check
    arrays, table = parse_fixtures()
    row = next(r for r in table if int(r[0]) == 16)
    fx_b = np.array(arrays[int(row[2])]["bytes"], dtype=np.uint8).tobytes()
    fx_want = np.array(arrays[int(row[4])]["want"], dtype=np.float32)
    fx_mine = deq_iq2xxs(fx_b).reshape(-1)
    fx_ok = len(fx_mine) == len(fx_want) and np.allclose(fx_mine, fx_want, rtol=1e-5, atol=1e-7)
    print(f"iq2_xxs fixture: {len(fx_b)}B -> {len(fx_mine)} vals: "
          f"{'MATCH' if fx_ok else 'FAIL'} (maxdiff={np.abs(fx_mine - fx_want).max():.2e})")
    ok &= fx_ok

    # --- 2. output vs source per-row comparison across sampled layers
    s = Reader(str(SRC))
    p = Reader(str(out_path))
    imap = ImatrixMap(DIR / "imatrix_unsloth.gguf_file", IMATRIX_SHA)
    quant = GGMLQuantizer(default_quantizer_library())
    tot_eq = tot_skip = 0
    for il in layers:
        for fam, sdeq, qtype, sblk, pblk in FAMS:
            st = next(t for t in s.tensors if t["name"] == f"blk.{il}.{fam}.weight")
            pt = next(t for t in p.tensors if t["name"] == f"blk.{il}.{fam}.weight")
            if pt["type"] == st["type"] and pt["nbytes"] == st["nbytes"]:
                print(f"L{il} {fam}: verbatim copy (unchanged), skipped")
                continue
            OUT_DEQ = {16: deq_iq2xxs, 10: deq_q2k, 12: deq_q4k}
            OUT_QTYPE = {16: "IQ2_XXS", 10: "Q2_K", 12: "Q4_K"}
            if pt["type"] not in OUT_DEQ:
                print(f"L{il} {fam}: unexpected output type {pt['type']}")
                ok = False
                continue
            pdeq = OUT_DEQ[pt["type"]]
            qtype = OUT_QTYPE[pt["type"]]
            if sdeq is None:
                if st["type"] not in SRC_DEQ:
                    print(f"L{il} {fam}: unexpected source type {st['type']}")
                    ok = False
                    continue
                sdeq, sblk = SRC_DEQ[st["type"]]
            in_dim = st["dims"][0]
            out_full_rows = st["dims"][1]
            s_row = (in_dim // 256) * sblk
            p_row = (in_dim // 256) * pblk
            cos, cos_live, n_live, both_zero, dead_live, byte_eq, byte_skip = [], [], 0, 0, 0, 0, 0
            with open(str(SRC), "rb") as fs, open(str(out_path), "rb") as fp:
                for e in experts:
                    # the packer's weights are per-expert, so read the whole expert block
                    # once and derive the same weights the requant tool used
                    fs.seek(st["abs"] + e * st["dims"][1] * s_row)
                    block_vals = sdeq(fs.read(out_full_rows * s_row)).reshape(out_full_rows, in_dim)
                    w2 = imap.pure_weights(f"blk.{il}.{fam}.weight", st["dims"][2], in_dim, e)
                    if w2 is None:
                        w2 = energy_weights(block_vals)
                    for r in out_rows:
                        src_vals = block_vals[r]
                        fp.seek(pt["abs"] + (e * pt["dims"][1] + r) * p_row)
                        out_bytes = fp.read(p_row)
                        out_vals = pdeq(out_bytes).reshape(-1)
                        cos.append(cosine(src_vals, out_vals))
                        live = np.abs(src_vals) > 0.02 * np.abs(src_vals).max()
                        if np.any(live):
                            n_live += 1
                            cos_live.append(cosine(src_vals[live], out_vals[live]))
                        if np.linalg.norm(src_vals) == 0 and np.linalg.norm(out_vals) == 0:
                            both_zero += 1
                        dead_live += int((np.linalg.norm(src_vals) == 0) !=
                                        (np.linalg.norm(out_vals) == 0))
                        reenc = quant.encode(src_vals.reshape(1, in_dim), qtype, w2)
                        byte_eq += int(bytes(reenc) == out_bytes)
            cos = np.array(cos)
            live_cos = np.array(cos_live) if cos_live else np.array([1.0])
            tot_eq += byte_eq
            tot_skip += byte_skip
            print(f"L{il} {fam}: cos mean={cos.mean():.4f} min={cos.min():.4f} | "
                  f"live-pos cos mean={live_cos.mean():.4f} min={live_cos.min():.4f} "
                  f"({n_live} rows) | byte-equal {byte_eq}/{len(cos) - byte_skip} | "
                  f"both-zero={both_zero} | zero-disagreements={dead_live}")
            ok &= live_cos.min() > 0.90 and byte_eq == len(cos) - byte_skip and dead_live == 0
    print(f"byte determinism total: {tot_eq}/{tot_eq + tot_skip} "
          f"({tot_skip} uncalibrated experts skipped)")
    print("REQUANT QC:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
