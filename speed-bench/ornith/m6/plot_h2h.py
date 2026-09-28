"""Charts and CSV for the Ornith head-to-head (h2h.py output).

  uv run --with matplotlib python speed-bench/ornith/m6/plot_h2h.py h2h/h2h.json \
      [--out-dir DIR] [--ds4-label TEXT] [--png-dir DIR]

Writes h2h.csv (every request), h2h-throughput.svg (prefill and decode t/s per
context) and h2h-ttft-memory.svg (time to first token and peak footprint per
context) next to this script, or into --out-dir. A request that failed is drawn as an x on the
axis floor and labelled.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
# ds4 memory-maps the GGUF and makes it resident through Metal, so the model's
# 21.26 GiB never shows in the process footprint (h2h-logs/ds4-memory.txt:
# "KV 5.50 GiB + buffers 1.00 GiB + resident model 21.26 GiB"); oMLX loads the
# weights into its own process memory. total_gib puts both on the same basis.
DS4_RESIDENT_MODEL_GIB = 21.26
STYLE = {"ds4": ("ds4 (Ornith 23G, M6)", "#2563eb", "o"), "omlx": ("oMLX 0.6.4 (live)", "#64748b", "s")}
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})


def load(path):
    rows = json.load(open(path))["rows"]
    for r in rows:
        if r.get("peak_footprint_gib"):
            extra = DS4_RESIDENT_MODEL_GIB if r["runtime"] == "ds4" else 0.0
            r["total_gib"] = round(r["peak_footprint_gib"] + extra, 2)
    return rows


def write_csv(rows, out):
    keys = ["runtime", "context", "rep", "prompt_tokens", "ttft_s", "prefill_tps", "decode_tps",
            "completion_tokens", "peak_footprint_gib", "total_gib", "idle_footprint_gib", "finish_reason", "error"]
    with (out / "h2h.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def series(rows, runtime, metric):
    xs, ys, fails = [], [], []
    for ctx in sorted({r["context"] for r in rows if r["runtime"] == runtime}):
        got = [r for r in rows if r["runtime"] == runtime and r["context"] == ctx]
        ok = [r[metric] for r in got if not r.get("error") and r.get(metric)]
        if ok:
            xs.append(ctx)
            ys.append(statistics.median(ok))
        elif any(r.get("error") for r in got):
            fails.append(ctx)
    return xs, ys, fails


def panel(ax, rows, metric, title, ylabel, fmt, log=False):
    top = 0.0
    for runtime, (label, color, marker) in STYLE.items():
        xs, ys, fails = series(rows, runtime, metric)
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=label, zorder=3)
            for x, y in zip(xs, ys):
                ax.annotate(fmt(y), (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=9, color=color)
            top = max(top, max(ys))
        for x in fails:
            ax.scatter([x], [0], marker="x", s=80, color=color, zorder=4)
            ax.annotate("failed", (x, 0), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=9, color=color)
    ax.set_xscale("log", base=2)
    ctxs = sorted({r["context"] for r in rows})
    ax.set_xticks(ctxs)
    ax.set_xticklabels(["%dK" % (c // 1024 if c % 1024 == 0 else round(c / 1000)) for c in ctxs],
                       rotation=30)
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, top * 1.2 if top else 1)
    ax.set(title=title, xlabel="Prompt tokens", ylabel=ylabel)
    ax.grid(color="#e2e8f0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def chart(name, rows, panels, out, png_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, args in zip(axes, panels):
        panel(ax, rows, *args)
    fig.tight_layout()
    fig.savefig(out / name)
    if png_dir:
        fig.savefig(Path(png_dir) / name.replace(".svg", ".png"), dpi=110)
    plt.close(fig)


def take_opt(argv, name):
    if name not in argv:
        return None, argv
    i = argv.index(name)
    return argv[i + 1], argv[:i] + argv[i + 2:]


def main(argv):
    png_dir, argv = take_opt(argv, "--png-dir")
    out_dir, argv = take_opt(argv, "--out-dir")
    label, argv = take_opt(argv, "--ds4-label")
    out = Path(out_dir) if out_dir else ROOT
    if label:
        STYLE["ds4"] = (label,) + STYLE["ds4"][1:]
    rows = load(argv[0] if argv else ROOT / "h2h.json")
    write_csv(rows, out)
    chart("h2h-throughput.svg", rows, [
        ("prefill_tps", "Prefill", "Tokens / second", lambda v: "%.0f" % v),
        ("decode_tps", "Decode (MTP on both)", "Tokens / second", lambda v: "%.1f" % v)], out, png_dir)
    chart("h2h-ttft-memory.svg", rows, [
        ("ttft_s", "Time to first token (lower is better)", "Seconds", lambda v: "%.1f" % v, True),
        ("total_gib", "Peak memory (weights + KV + buffers)", "GiB", lambda v: "%.1f" % v)], out, png_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
