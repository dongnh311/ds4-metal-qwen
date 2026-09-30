"""Run one arm (a ds4-server configuration) through the next-gen evaluation suites.

python3 speed-bench/nextgen-eval/run.py --config speed-bench/nextgen-eval/configs/prod.json [--suites code,vi]

The server suites share one ds4-server process. The reasoning suite runs ds4-eval afterwards, once the
server has stopped, so only one model is ever loaded. Pause the gateway stack before running (README).
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ds4eval  # noqa: E402
import eval_suites  # noqa: E402
import graders  # noqa: E402
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
    ids = {}
    for r in rows:
        if r.get("passed") is not None and r["suite"] in ACCURACY:
            ids.setdefault(r["suite"], []).append(str(r["id"]))
    case_ids = {k: hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()[:16] for k, v in ids.items()}
    docqa = {r["id"]: r["passed"] for r in rows if r["suite"] == "longctx" and r["id"].startswith("docqa-")}
    vi_rows = [r for r in rows if r["suite"].startswith("vi_")]
    uncensor = {}
    for kind in ("harmful", "harmless"):
        kind_rows = [r for r in rows if r["suite"] == "uncensor_" + kind]
        if kind_rows:
            uncensor[kind + "_refusals"] = sum(1 for r in kind_rows if r["refused"])
            uncensor[kind + "_n"] = len(kind_rows)
    needle = {r["id"][len("needle-"):]: r["passed"] for r in rows
              if r["suite"] == "longctx" and r["id"].startswith("needle-")}
    short_tiers = sorted(r["id"][len("needle-"):] for r in rows
                         if r["suite"] == "longctx" and r["id"].startswith("needle-")
                         and r.get("prompt_tokens") and r.get("target_tokens")
                         and r["prompt_tokens"] < eval_suites.SHORT_TIER * r["target_tokens"])
    mem = next((r for r in rows if r["suite"] == "longctx_mem"), {})
    totals = {}
    for r in rows:
        group = _speed_group(r["suite"])
        if group in SPEED_SUITES and r.get("seconds") is not None:
            totals[group] = round(totals.get(group, 0.0) + r["seconds"], 1)
    return {
        "suites": suites,
        "case_ids": case_ids,
        "errors": [{"suite": r["suite"], "error": r.get("error")} for r in rows if r.get("id") == "suite-error"],
        "vi_cjk_leaks": sum(1 for r in vi_rows if (r.get("cjk") or 0) > 0) if vi_rows else None,
        "uncensor": uncensor,
        "longctx": {"needle": needle, "docqa": docqa, "peak_wired_gib": mem.get("peak_wired_gib"),
                    "swapouts": mem.get("swapouts"), "short_tiers": short_tiers},
        "speed": {"total_seconds": totals,
                  "think_tokens_median": _median([r.get("think_tokens") for r in rows]),
                  "decode_tps_median": _median([r.get("decode_tps") for r in rows]),
                  "prefill_tps_median": _median([r.get("prefill_tps") for r in rows])},
    }


def _git(repo, *args):
    try:
        return subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def provenance(env, argv, root, data, gateway, git=_git):
    """What this arm ran: commit, binaries, command, frozen data. Two arms run days apart can differ in
    any of these; compare.py prints them side by side."""
    manifest = data / "manifest.json"
    frozen = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {"git_head": git(root, "rev-parse", "HEAD"), "git_dirty": bool(git(root, "status", "--porcelain")),
            "gateway_head": git(gateway, "rev-parse", "HEAD"),
            "ds4_server_sha256": _sha256(root / "ds4-server"), "ds4_eval_sha256": _sha256(root / "ds4-eval"),
            "env": env, "argv": argv, "data": {k: v.get("sha256") for k, v in frozen.items()}}


# The row suites each arm-level suite produces; its error rows carry the arm-level name.
ROW_SUITES = {"code": ("code",), "ifeval": ("ifeval",), "vi": ("vi_knowledge", "vi_writing", "vi_speed"),
              "uncensor": ("uncensor_harmful", "uncensor_harmless"),
              "tools": ("tools_pos", "tools_neg", "tools_xfer", "faithfulness"),
              "longctx": ("longctx", "longctx_mem"), "reason": ("reason",)}


def rows_outside(rows, suites):
    """The rows that belong to none of the arm-level `suites` (their error rows included)."""
    drop = set(suites) | {s for name in suites for s in ROW_SUITES[name]}
    return [r for r in rows if r["suite"] not in drop]


def prepare_rerun(out, wanted, arm):
    """Drop the rows of `wanted` from an earlier run of `arm` in `out` (old rows.jsonl kept as a backup);
    returns the remaining rows and the earlier summary."""
    previous = json.loads((out / "summary.json").read_text())
    if previous.get("arm") != arm:
        raise SystemExit("%s holds arm %r, not %r" % (out, previous.get("arm"), arm))
    old = (out / "rows.jsonl").read_text()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / ("rows.before-rerun-%s.jsonl" % stamp)).write_text(old)
    rows = rows_outside([json.loads(line) for line in old.splitlines() if line.strip()], wanted)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows, previous


def rerun_provenance(previous, new, wanted):
    """The first run's provenance, plus what each rerun used."""
    keys = ("git_head", "git_dirty", "ds4_server_sha256", "ds4_eval_sha256", "data")
    return dict(previous, reruns=previous.get("reruns", []) + [dict({"suites": wanted}, **{k: new[k] for k in keys})])


def _error_row(suite, error):
    return {"suite": suite, "id": "suite-error", "passed": None, "error": error}


def run_arm(env, argv, wanted, out, rows, port=18299, server_cls=None, suites=None, reason=None):
    """Run the wanted suites. Every row goes to `rows` and to out/rows.jsonl as soon as it exists. A
    failing suite leaves an error row and the run goes on; a dead server ends the server suites."""
    server_cls = server_cls or server.Ds4Server
    suites = eval_suites.SERVER_SUITES if suites is None else suites
    reason = reason or ds4eval.run_reason
    if "reason" in wanted:
        ds4eval.eval_argv(argv, server.ROOT, "core", "-", 1, out / "-")  # refuse unknown flags before GPU time

    def keep(row):
        rows.append(row)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def drain(name, produce):
        print("== suite %s" % name, flush=True)
        try:
            for row in produce():
                keep(row)
        except Exception as e:  # recorded as an error row; the gate fails on it
            print("   suite %s failed: %r" % (name, e), flush=True)
            keep(_error_row(name, repr(e)[:500]))

    server_suites = [s for s in wanted if s in suites]
    if server_suites:
        srv = server_cls(env, argv, out / "server.log", port)
        srv.start()
        try:
            ctx = eval_suites.Ctx(srv.base_url, server.LogCursor(out / "server.log"), DATA,
                                  int(server.argv_value(argv, "-c") or 0), server.ROOT)
            for name in server_suites:
                code = srv.proc.poll()
                if code is not None:
                    keep(_error_row(name, "ds4-server exited with %s; see server.log" % code))
                    continue
                drain(name, lambda: suites[name](ctx))
        finally:
            srv.stop()
    if "reason" in wanted:
        drain("reason", lambda: reason(env, argv, server.ROOT, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suites", default=",".join(ALL_SUITES))
    ap.add_argument("--port", type=int, default=18299)
    ap.add_argument("--out")
    ap.add_argument("--rerun", action="store_true",
                    help="--out is an earlier run of this arm: replace the rows of --suites there, keep the "
                         "others, and summarize them all")
    args = ap.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text())
    wanted = [s for s in args.suites.split(",") if s]
    unknown = sorted(set(wanted) - set(ALL_SUITES))
    if unknown:
        raise SystemExit("unknown suites: %s (known: %s)" % (", ".join(unknown), ", ".join(ALL_SUITES)))
    if args.rerun and not (args.out and (pathlib.Path(args.out) / "summary.json").exists()):
        raise SystemExit("--rerun needs --out pointing at an earlier run of this arm")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out) if args.out else DATA / "runs" / ("%s-%s" % (config["name"], stamp))
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads(server.REGISTRY.read_text())
    kv_dir = pathlib.Path(tempfile.mkdtemp(prefix="kv-", dir=str(out)))
    env, argv = server.resolve(config, registry, server.ROOT, args.port, kv_dir)
    prov = provenance(env, argv, server.ROOT, DATA, graders.GATEWAY_REPO)
    if args.rerun:
        rows, previous = prepare_rerun(out, wanted, config["name"])
        suites_run = [s for s in ALL_SUITES if s in set(previous["suites_run"]) | set(wanted)]
        prov = rerun_provenance(previous["provenance"], prov, wanted)
    else:
        (out / "command.json").write_text(json.dumps({"config": config, "env": env, "argv": argv}, indent=1) + "\n")
        rows, suites_run = [], wanted
    try:
        run_arm(env, argv, wanted, out, rows, port=args.port)
    finally:
        shutil.rmtree(kv_dir, ignore_errors=True)
        summary = dict(summarize(rows), arm=config["name"], config=config, suites_run=suites_run, provenance=prov)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("run directory: %s" % out)


if __name__ == "__main__":
    main()
