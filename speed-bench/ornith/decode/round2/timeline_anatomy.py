#!/usr/bin/env python3
"""Per-MTP-cycle anatomy of a DS4_METAL_ENCODER_TIMELINE file (round 2, Plan A Task 4).

The timeline (ds4_metal.m, ds4_gpu_timeline_resolve) writes one B line per
command buffer and one E line per encoder (dispatch group):
    B <seq> <n_encoders> <gpu_start_ns> <gpu_end_ns>
    E <seq> <idx> <start_ns> <end_ns> <dur_us> <gap_us> <caller> <n_dispatch> <tg> <tpt> <kernel>
B times come from the command buffer's GPUStartTime/GPUEndTime, E times from
counter samples: different clocks, so the two are never subtracted from each
other. Decode buffers are those after the last buffer longer than
prefill_ms; an MTP cycle is counted per encoder named *mtp_concat* (one per
draft). Usage: timeline_anatomy.py FILE [--small-us 10] [--top 25]
"""
import argparse
import collections


def parse(lines):
    bufs, by_seq = [], {}
    for line in lines:
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] == "B" and len(f) >= 5:
            b = {"seq": int(f[1]), "start_ns": float(f[3]), "end_ns": float(f[4]), "encs": []}
            bufs.append(b)
            by_seq[b["seq"]] = b
        elif f[0] == "E" and len(f) >= 12:
            b = by_seq.get(int(f[1]))
            if b is not None:
                b["encs"].append({"dur_us": float(f[5]), "gap_us": float(f[6]), "n_dispatch": int(f[8]),
                                  "kernel": f[11]})
    return bufs


def decode_buffers(bufs, prefill_ms=50.0):
    last = -1
    for i, b in enumerate(bufs):
        if (b["end_ns"] - b["start_ns"]) / 1e6 > prefill_ms:
            last = i
    return bufs[last + 1:]


def anatomy(bufs, small_us=10.0):
    cycles = sum(1 for b in bufs for e in b["encs"] if "mtp_concat" in e["kernel"])
    if cycles == 0:
        raise ValueError("no mtp_concat encoder: not an --mtp decode timeline")
    tot = collections.Counter()
    kern_n, kern_us = collections.Counter(), collections.Counter()
    for i, b in enumerate(bufs):
        tot["buffer_gpu_us"] += (b["end_ns"] - b["start_ns"]) / 1e3
        if i > 0:
            tot["inter_idle_us"] += max(0.0, (b["start_ns"] - bufs[i - 1]["end_ns"]) / 1e3)
        for e in b["encs"]:
            dur = max(0.0, e["dur_us"])
            tot["busy_us"] += dur
            tot["intra_gap_us"] += e["gap_us"]
            tot["encoders"] += 1
            tot["dispatches"] += e["n_dispatch"]
            if dur < small_us:
                tot["small_encoders"] += 1
                tot["small_busy_us"] += dur
                tot["small_gap_us"] += e["gap_us"]
            kern_n[e["kernel"]] += 1
            kern_us[e["kernel"]] += dur
    out = {k: v / cycles for k, v in tot.items()}
    for k in ("buffer_gpu_us", "inter_idle_us", "busy_us", "intra_gap_us", "encoders", "dispatches",
              "small_encoders", "small_busy_us", "small_gap_us"):
        out.setdefault(k, 0.0)
    out["cycles"] = cycles
    out["buffers"] = len(bufs) / cycles
    out["kernels"] = sorted(((k, kern_n[k] / cycles, kern_us[k] / cycles) for k in kern_n),
                            key=lambda r: -r[2])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--small-us", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)
    with open(args.file, encoding="utf-8", errors="replace") as fp:
        a = anatomy(decode_buffers(parse(fp)), small_us=args.small_us)
    print("anatomy: cycles %d, per cycle: %.1f buffers, %.0f encoders, %.0f dispatches" % (
        a["cycles"], a["buffers"], a["encoders"], a["dispatches"]))
    print("anatomy: per cycle us: buffer GPU %.0f, encoder busy %.0f, gaps inside buffers %.0f, "
          "idle between buffers %.0f" % (a["buffer_gpu_us"], a["busy_us"], a["intra_gap_us"], a["inter_idle_us"]))
    print("anatomy: small encoders (<%.0f us): %.0f per cycle, busy %.0f us, their gaps %.0f us" % (
        args.small_us, a["small_encoders"], a["small_busy_us"], a["small_gap_us"]))
    for k, n, us in a["kernels"][:args.top]:
        print("anatomy: kernel %-56s %6.1f/cycle %8.1f us/cycle %7.2f us each" % (k, n, us, us / n if n else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
