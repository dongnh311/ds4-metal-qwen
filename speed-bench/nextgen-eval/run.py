"""Run one arm (a ds4-server configuration) through the next-gen evaluation suites.

python3 speed-bench/nextgen-eval/run.py --config speed-bench/nextgen-eval/configs/prod.json [--suites code,vi]

The server suites share one ds4-server process. The reasoning suite runs ds4-eval afterwards, once the
server has stopped, so only one model is ever loaded. Pause the gateway stack before running (README).
"""
import argparse
import datetime
import json
import pathlib
import shutil
import statistics
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ds4eval  # noqa: E402
import eval_suites  # noqa: E402
import server  # noqa: E402
from compare import ACCURACY, SPEED_SUITES  # noqa: E402
from fetch_data import DATA  # noqa: E402

ALL_SUITES = list(eval_suites.SERVER_SUITES) + ["reason"]


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def _speed_group(suite):
    if suite.startswith("vi_"):
        return "vi"
    if suite.startswith("uncensor_"):
        return "uncensor"
    return suite


def summarize(rows):
    suites = {}
    for r in rows:
        if r.get("passed") is None or r["suite"] not in ACCURACY:
            continue
        s = suites.setdefault(r["suite"], {"n": 0, "passed": 0})
        s["n"] += 1
        s["passed"] += int(bool(r["passed"]))
    vi_rows = [r for r in rows if r["suite"].startswith("vi_")]
    uncensor = {}
    for kind in ("harmful", "harmless"):
        kind_rows = [r for r in rows if r["suite"] == "uncensor_" + kind]
        if kind_rows:
            uncensor[kind + "_refusals"] = sum(1 for r in kind_rows if r["refused"])
            uncensor[kind + "_n"] = len(kind_rows)
    needle = {r["id"][len("needle-"):]: r["passed"] for r in rows
              if r["suite"] == "longctx" and r["id"].startswith("needle-")}
    mem = next((r for r in rows if r["suite"] == "longctx_mem"), {})
    totals = {}
    for r in rows:
        group = _speed_group(r["suite"])
        if group in SPEED_SUITES and r.get("seconds") is not None:
            totals[group] = round(totals.get(group, 0.0) + r["seconds"], 1)
    return {
        "suites": suites,
        "vi_cjk_leaks": sum(1 for r in vi_rows if (r.get("cjk") or 0) > 0) if vi_rows else None,
        "uncensor": uncensor,
        "longctx": {"needle": needle, "peak_wired_gib": mem.get("peak_wired_gib"), "swapouts": mem.get("swapouts")},
        "speed": {"total_seconds": totals,
                  "think_tokens_median": _median([r.get("think_tokens") for r in rows]),
                  "decode_tps_median": _median([r.get("decode_tps") for r in rows]),
                  "prefill_tps_median": _median([r.get("prefill_tps") for r in rows])},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suites", default=",".join(ALL_SUITES))
    ap.add_argument("--port", type=int, default=18299)
    ap.add_argument("--out")
    args = ap.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text())
    wanted = [s for s in args.suites.split(",") if s]
    unknown = sorted(set(wanted) - set(ALL_SUITES))
    if unknown:
        raise SystemExit("unknown suites: %s (known: %s)" % (", ".join(unknown), ", ".join(ALL_SUITES)))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out) if args.out else DATA / "runs" / ("%s-%s" % (config["name"], stamp))
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads(server.REGISTRY.read_text())
    kv_dir = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=str(out)))
    env, argv = server.resolve(config, registry, server.ROOT, args.port, kv_dir)
    (out / "command.json").write_text(json.dumps({"config": config, "env": env, "argv": argv}, indent=1) + "\n")
    rows = []

    def keep(new_rows):
        rows.extend(new_rows)
        with open(out / "rows.jsonl", "a") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    server_suites = [s for s in wanted if s in eval_suites.SERVER_SUITES]
    try:
        if server_suites:
            srv = server.Ds4Server(env, argv, out / "server.log", args.port)
            srv.start()
            try:
                ctx = eval_suites.Ctx(srv.base_url, server.LogCursor(out / "server.log"), DATA,
                                      int(server.argv_value(argv, "-c") or 0), server.ROOT)
                for name in server_suites:
                    print("== suite %s" % name, flush=True)
                    keep(eval_suites.SERVER_SUITES[name](ctx))
            finally:
                srv.stop()
        if "reason" in wanted:
            print("== suite reason", flush=True)
            keep(ds4eval.run_reason(env, argv, server.ROOT, out))
    finally:
        shutil.rmtree(kv_dir, ignore_errors=True)
    summary = dict(summarize(rows), arm=config["name"], config=config, suites_run=wanted)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("run directory: %s" % out)


if __name__ == "__main__":
    main()
