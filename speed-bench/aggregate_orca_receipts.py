#!/usr/bin/env python3
"""aggregate_orca_receipts.py — rebuild speed-bench/orca-iq2-baseline.csv from raw receipts.

The sweep writes one bench CSV plus one .err (acceptance counters at session free)
per arm/rep/ladder; this merges them into the campaign table.  It exists because
the sweep script's own inline CSV append mixed columns on 2026-10-29 (it dropped
gen_steady_tokens) — the raw run files were never affected, so any aggregation
bug is repairable without re-benchmarking.

usage: python3 speed-bench/aggregate_orca_receipts.py [receipts_dir ...]
Default: every speed-bench/receipts/orca-iq2-* dir except ones named with "-discard".
"""
import csv, glob, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"run-(plain|mtp-force|mtp-d2|mtp-d3|mtp)-r(\d+)-(main|192k)\.csv$")
COLS = ["arm", "rep", "ladder", "ctx_tokens", "prefill_tokens", "prefill_tps",
        "gen_tokens", "gen_tps", "gen_first_ms", "gen_steady_tokens", "gen_steady_tps",
        "mtp_cycles", "mtp_accepted", "mtp_pct", "depth3_cycles", "depth3_accept2",
        "extra_tokens_per_cycle"]
ORDER = {"plain": 0, "mtp": 1, "mtp-d2": 2, "mtp-d3": 3, "mtp-force": 4}

def harvest(d):
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "run-*.csv"))):
        m = PAT.search(os.path.basename(path))
        if not m:
            continue
        acc = {}
        try:
            txt = open(path[:-4] + ".err", encoding="utf-8", errors="replace").read()
        except OSError:
            txt = ""
        mm = re.search(r"mtp: (\d+) verify cycles, (\d+) drafts accepted \(([\d.]+)%\)", txt)
        dd = re.search(r"depth3: (\d+) cycles, (\d+) second-draft accepts", txt)
        ss = re.search(r"([\d.]+) extra tokens per cycle", txt)
        if mm:
            acc = {"mtp_cycles": mm.group(1), "mtp_accepted": mm.group(2), "mtp_pct": mm.group(3)}
        if dd:
            acc["depth3_cycles"], acc["depth3_accept2"] = dd.group(1), dd.group(2)
        if ss:
            acc["extra_tokens_per_cycle"] = ss.group(1)
        for r in csv.DictReader(open(path)):
            r.update(arm=m.group(1), rep=m.group(2), ladder=m.group(3), **acc)
            rows.append(r)
    return rows

def main():
    dirs = sys.argv[1:] or [d for d in glob.glob(os.path.join(REPO, "speed-bench/receipts/orca-iq2-*"))
                            if "discard" not in os.path.basename(d)]
    rows = []
    for d in dirs:
        rows.extend(harvest(d))
    # later dirs win per (arm, rep, ladder, ctx): a re-run replaces its own rows.
    seen = {}
    for r in rows:
        seen[(r["arm"], r["rep"], r["ladder"], r["ctx_tokens"])] = r
    rows = sorted(seen.values(), key=lambda r: (r["rep"], ORDER.get(r["arm"], 9),
                                                r["ladder"], int(r["ctx_tokens"])))
    out = os.path.join(REPO, "speed-bench/orca-iq2-baseline.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in COLS])
    print(f"{len(rows)} rows -> {out}  (from {len(dirs)} receipt dirs)")

if __name__ == "__main__":
    main()
