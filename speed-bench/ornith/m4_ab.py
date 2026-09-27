#!/usr/bin/env python3
"""Interleaved A/B harness for Ornith (spec Sec8 gate 3, DECISIONS.md Sec4.2).

One model process at a time on this 64 GB box. Every prompt carries a fresh
nonce at its START (defeats both prefix caches) and cached_tokens==0 is
asserted. Client-side timing is qwen_gate.sse_timings/request_rates
(imported, unchanged).

Two independent modes, never combined in one run:

  baseline: ds4-server (this Ornith build) vs the live oMLX backend, one
            process at a time, A-B-B-A over ("omlx","ds4","ds4","omlx").
            Used only for the Task 3 baseline and the final Task 12 run.
            The live production oMLX watchdog must already be paused by the
            caller (this script never unloads LaunchAgents, never edits the
            gateway registry, and never touches :8090).

  lever:    two ds4-server arms in the SAME run, differing only by process
            environment (--base-env / --lever-env), A-B-B-A over
            ("base","lever","lever","base"). Used for every lever A/B
            (ds4 vs ds4); oMLX is not started or measured in this mode, but
            guard_free() still requires the live oMLX to be paused (the box
            must be free of any model process before either arm starts).

  m4_ab.py --mode baseline --ds4-model PATH --out DIR
           [--contexts 2048,32768,131072] [--cold-tokens 31000]
           [--max-tokens 256] [--warmup 1] [--ds4-env K=V,...] [--ds4-args "..."]

  m4_ab.py --mode lever --ds4-model PATH --out DIR
           [--base-env K=V,...] [--lever-env K=V,...] [--ds4-args "..."]
           [--contexts ...] [--cold-tokens ...] [--max-tokens ...] [--warmup ...]

Arms are stopped with SIGTERM and a bounded wait only. A process that
survives SIGTERM is never SIGKILLed: _terminate() raises SystemExit naming
the pid and leaves it running for a human to investigate.
"""
import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "qwen-regression"))
sys.path.insert(0, os.path.join(ROOT, "speed-bench", "lib"))
import qwen_gate  # noqa: E402  (sse_timings, request_rates)
import machine    # noqa: E402
import wired      # noqa: E402

DS4_PORT = 18296
OMLX_PORT = 18085
OMLX_MODEL = "Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx"
FILLER = os.path.join(ROOT, "speed-bench", "promessi_sposi.txt")
BASELINE_ORDER = ("omlx", "ds4", "ds4", "omlx")
LEVER_ORDER = ("base", "lever", "lever", "base")
STOP_TIMEOUT_S = 60


class CacheHit(Exception):
    pass


def cached_tokens(usage):
    det = (usage or {}).get("prompt_tokens_details") or {}
    return int(det.get("cached_tokens", 0) or 0)


def assert_cache_cold(usage):
    if cached_tokens(usage) != 0:
        raise CacheHit("cached_tokens=%d (prefix cache defeated the prefill)" % cached_tokens(usage))


def build_prompt(nonce, target_tokens, filler, chars_per_token):
    chars = max(0, int(round(target_tokens * chars_per_token)) - len(nonce) - 40)
    if chars > len(filler):
        raise ValueError("filler has %d chars, need %d for %d tokens" % (len(filler), chars, target_tokens))
    return nonce + " " + filler[:chars] + "\n\nSummarize the text above in two sentences."


def guard_free(ds4_running=None, omlx_running=None, estate=None):
    """Raise SystemExit unless the box is free of every model process and no
    process is wedged in Metal state E/U. Callers pass fakes in tests; the
    real checks default to machine.ds4_running, pgrep for omlx-server, and a
    `ps` scan for state E/U."""
    ds4_running = ds4_running or machine.ds4_running
    omlx_running = omlx_running or _omlx_running
    estate = estate or _procs_in_estate
    d, o, e = ds4_running(), omlx_running(), estate()
    if d:
        raise SystemExit("m4_ab: a ds4 process is running; the box must be free:\n" + d)
    if o:
        raise SystemExit("m4_ab: an omlx process is running; the box must be free "
                          "(pause the live oMLX watchdog first):\n" + o)
    if e:
        raise SystemExit("m4_ab: a process is in state E/U (wedged Metal); reboot before running:\n" + e)


def _omlx_running():
    """True (non-empty) if the live oMLX is up, whether or not it has
    setproctitle-renamed itself to omlx-server yet. A freshly spawned oMLX
    starts as .../omlx-venv/bin/omlx serve ... and may still be in that
    window when guard_free() checks, so both forms are matched."""
    hits = []
    for cmd in (["pgrep", "-x", "omlx-server"], ["pgrep", "-f", "omlx-venv/bin/omlx"]):
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout.strip()
        if out:
            hits.append(out)
    return "\n".join(hits)


MODEL_COMMS = {"ds4", "ds4-server", "ds4-agent", "ds4-bench", "ds4-eval", "ds4_test",
              "omlx-server", "omlx", "llama-server"}
_PYTHON_COMM_RE = re.compile(r"^python\d*(\.\d+)?$")


def _is_model_command(command):
    """True if `command` (a full ps command line, argv[0] + args) names a
    model/Metal process we care about: one of MODEL_COMMS by argv[0]
    basename, or a python interpreter whose command line mentions omlx."""
    argv0 = command.split(None, 1)[0] if command.strip() else ""
    name = os.path.basename(argv0)
    if name in MODEL_COMMS:
        return True
    if _PYTHON_COMM_RE.match(name) and "omlx" in command:
        return True
    return False


def _ps_estate_snapshot():
    """pid -> (stat, command) for every process currently in state E or U,
    using the full command line (needed to spot a python/omlx process)."""
    p = subprocess.run(["ps", "-axo", "pid,stat,command"], capture_output=True, text=True)
    snap = {}
    for line in p.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, stat, command = parts
        if re.search(r"[EU]", stat):
            snap[pid] = (stat, command)
    return snap


def _procs_in_estate():
    """Report a process only if it is one of the model/Metal processes we
    care about (see _is_model_command) AND its E/U state persists across a
    second check ~2s later (same pid still E/U). A momentary U (normal for
    unrelated system processes like macmon in disk/IO wait) never trips
    this, and unrelated processes are never reported at all."""
    snap1 = _ps_estate_snapshot()
    candidates = {pid: cmd for pid, (stat, cmd) in snap1.items() if _is_model_command(cmd)}
    if not candidates:
        return ""
    time.sleep(2)
    snap2 = _ps_estate_snapshot()
    hits = []
    for pid, cmd in candidates.items():
        if pid in snap2:
            stat2, cmd2 = snap2[pid]
            hits.append("%s %s %s" % (pid, stat2, cmd2))
    return "\n".join(hits)


def _get_json(url, timeout=2):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def _model_discovered(models, model_id):
    """True if `model_id` appears in a /v1/models response's data list,
    whether or not the live oMLX has actually loaded it yet."""
    return any(entry.get("id") == model_id for entry in (models or {}).get("data") or [])


def _parse_env(spec):
    """'K=V,K2=V2' -> {'K': 'V', 'K2': 'V2'}; '' -> {}."""
    return dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv)


def calibrate_chars_per_token(arm):
    """One max_tokens=1 request; usage.prompt_tokens over the prompt char count."""
    with open(FILLER, encoding="utf-8", errors="replace") as fp:
        probe = "nonce-cal " + fp.read(20000)
    _, usage = arm.stream(probe, max_tokens=1)
    pt = int((usage or {}).get("prompt_tokens", 0) or 0)
    return (len(probe) / pt) if pt > 0 else 3.46


def _terminate(proc, label, timeout=STOP_TIMEOUT_S):
    """SIGTERM + bounded wait. Never SIGKILL: a survivor is left running and
    named in a SystemExit for a human to investigate."""
    if not proc or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "m4_ab: pid %d (%s) did not exit within %ds of SIGTERM; left it running, "
            "investigate and stop it by hand (never SIGKILL a Metal process)"
            % (proc.pid, label, timeout))


class Ds4Arm:
    def __init__(self, model_path, out, extra_env=None, extra_args=None, label="ds4"):
        self.model_path = model_path
        self.out = out
        self.extra_env = extra_env or {}
        self.extra_args = extra_args or []
        self.name = label
        self.model_id = "ornith-1.5-35b-a3b"
        self.base_url = "http://127.0.0.1:%d" % DS4_PORT
        self.proc = None

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        kv = os.path.join(self.out, "ds4-kv")
        shutil.rmtree(kv, ignore_errors=True)
        os.makedirs(kv, exist_ok=True)
        cmd = ["caffeinate", "-i", "-s", os.path.join(ROOT, "ds4-server"), "--metal",
               "-m", self.model_path, "-c", "262144", "--mtp",
               "--kv-disk-dir", kv, "--host", "127.0.0.1", "--port", str(DS4_PORT)] + self.extra_args
        with open(os.path.join(self.out, "ds4-server.log"), "w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, **self.extra_env})

    def ready(self, timeout=900):
        for _ in range(timeout):
            if self.proc.poll() is not None:
                raise SystemExit("m4_ab: ds4-server (%s) exited during startup, see ds4-server.log"
                                 % self.name)
            try:
                urllib.request.urlopen(self.base_url + "/v1/models", timeout=2)
                return True
            except OSError:
                time.sleep(1)
        raise SystemExit("m4_ab: ds4-server (%s) did not come up in %d s" % (self.name, timeout))

    def stream(self, prompt, max_tokens):
        return _chat_stream(self.base_url, self.model_id, prompt, max_tokens)

    def stop(self):
        _terminate(self.proc, "ds4:%s" % self.name)


class OmlxArm:
    name = "omlx"
    base_url = "http://127.0.0.1:%d" % OMLX_PORT
    model_id = OMLX_MODEL

    def __init__(self, out):
        self.out = out
        self.proc = None

    def _registry_command(self):
        reg = os.path.expanduser(os.environ.get("DS4_GATEWAY_REGISTRY",
                                                 "~/.local/ai-gateway/runtime-registry.json"))
        with open(reg) as fp:
            models = json.load(fp)["models"]
        entry = models[OMLX_MODEL]["runtimes"]["omlx"]
        return list(entry["process_command"])

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "omlx-server.log"), "w") as log:
            self.proc = subprocess.Popen(["caffeinate", "-i", "-s"] + self._registry_command(),
                                         stdout=log, stderr=subprocess.STDOUT)

    def ready(self, timeout=900):
        """True once the server answers /api/status with status=='ok' AND
        /v1/models lists OMLX_MODEL as discovered. The live oMLX (0.6.4)
        loads models lazily on the first request: right after startup
        /api/status returns loaded_models=[] forever until a request comes
        in, so waiting on loaded_models (as this used to) never returns.
        Once discovered, sends one unmeasured load request so the model is
        actually resident before any measured request runs."""
        for _ in range(timeout):
            if self.proc and self.proc.poll() is not None:
                raise SystemExit("m4_ab: omlx exited during startup, see omlx-server.log")
            try:
                st = _get_json(self.base_url + "/api/status")
                models = _get_json(self.base_url + "/v1/models")
            except OSError:
                time.sleep(1)
                continue
            except Exception:
                time.sleep(1)
                continue
            if st.get("status") == "ok" and _model_discovered(models, self.model_id):
                self._load()
                return True
            time.sleep(1)
        raise SystemExit("m4_ab: omlx not ready in %d s" % timeout)

    def _load(self):
        """One unmeasured request that forces the lazily-loading oMLX to
        actually load self.model_id, so the first measured request (sent by
        measure_arm's warmup loop, or the first ctx row if --warmup 0) never
        includes model load time."""
        self.stream("nonce-omlx-load ping", max_tokens=1)

    def stream(self, prompt, max_tokens):
        return _chat_stream(self.base_url, self.model_id, prompt, max_tokens)

    def stop(self):
        _terminate(self.proc, "omlx")


def _last_finish_reason(events):
    fr = None
    for _, chunk in events:
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                fr = choice["finish_reason"]
    return fr


def _chat_stream(base, model, text, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": True},
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    events = []
    t0 = time.time()
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                events.append((time.time() - t0, chunk))
    timing = qwen_gate.sse_timings(events)
    timing["finish_reason"] = _last_finish_reason(events)
    return timing, usage


def measure_arm(arm, contexts, filler, max_tokens, warmup=1, cold_tokens=0,
                swap_used=None, wait_idle=None):
    swap_used = swap_used or machine.swap_used_mib
    wait_idle = wait_idle or (lambda: wired.wait_idle_gib())
    wait_idle()
    swap_before = swap_used()
    cpt = getattr(arm, "_cpt", None)
    if cpt is None:
        cpt = calibrate_chars_per_token(arm)
        arm._cpt = cpt
    for _ in range(warmup):
        arm.stream(build_prompt("nonce-warm", 512, filler, cpt), max_tokens)
    rows = {}
    targets = list(contexts) + (["cold31k"] if cold_tokens else [])
    for ctx in targets:
        n = cold_tokens if ctx == "cold31k" else ctx
        nonce = "nonce-%s-%d" % (arm.name, int(time.time() * 1000) % 100000)
        timing, usage = arm.stream(build_prompt(nonce, n, filler, cpt), max_tokens)
        assert_cache_cold(usage)
        rates = qwen_gate.request_rates(timing)
        rows[str(ctx)] = {**timing, **rates, "cached_tokens": cached_tokens(usage),
                          "finish_reason": timing.get("finish_reason")}
    swap_after = swap_used()
    return {"arm": arm.name, "rows": rows, "chars_per_token": cpt,
            "swap_before_mib": swap_before, "swap_after_mib": swap_after,
            "swap_delta_mib": swap_after - swap_before}


def _wait_free():
    for _ in range(120):
        if not machine.ds4_running() and not _omlx_running():
            return
        time.sleep(1)


def interleave(makers, contexts, cold_tokens, filler, max_tokens, warmup, order,
              guard=None, wait_free=None, swap_used=None, wait_idle=None):
    """Run `order` (a tuple of roles, each a key of `makers`), one process at
    a time: guard the box is free, start the role's arm, measure it, stop it,
    wait for the box to be free again, next role. Returns the summary dict
    (per-role means per context, plus the raw per-run rows)."""
    guard = guard if guard is not None else guard_free
    wait_free = wait_free if wait_free is not None else _wait_free
    runs = {role: [] for role in makers}
    for role in order:
        guard()
        arm = makers[role]()
        arm.start()
        try:
            arm.ready()
            runs[role].append(measure_arm(arm, contexts, filler, max_tokens, warmup, cold_tokens,
                                          swap_used=swap_used, wait_idle=wait_idle))
        finally:
            arm.stop()
        wait_free()
    return _summarize(runs, contexts, cold_tokens, list(makers.keys()))


def _summarize(runs, contexts, cold_tokens, roles):
    summary = {"runs": runs}
    keys = [str(c) for c in contexts] + (["cold31k"] if cold_tokens else [])
    for ctx in keys:
        row = {}
        for role in roles:
            dec = [r["rows"][ctx]["decode_tps"] for r in runs[role] if ctx in r["rows"]]
            pre = [r["rows"][ctx]["prefill_tps"] for r in runs[role] if ctx in r["rows"]]
            ttft = [r["rows"][ctx]["ttft_s"] for r in runs[role] if ctx in r["rows"]]
            if dec:
                row[role + "_decode"] = statistics.fmean(dec)
                row[role + "_prefill"] = statistics.fmean(pre)
                row[role + "_ttft_s"] = statistics.fmean(ttft)
        summary[ctx] = row
    return summary


def verdict(summary):
    """Gate-3 failures (ds4 vs oMLX only; baseline mode)."""
    failures = []
    for ctx, row in summary.items():
        if ctx in ("runs",):
            continue
        if ctx == "cold31k":
            if "ds4_ttft_s" in row and "omlx_ttft_s" in row and row["ds4_ttft_s"] > row["omlx_ttft_s"]:
                failures.append("cold31k: ds4 TTFT %.2fs > oMLX %.2fs" % (row["ds4_ttft_s"], row["omlx_ttft_s"]))
            continue
        if "ds4_decode" in row and "omlx_decode" in row and row["ds4_decode"] < row["omlx_decode"]:
            failures.append("%s: ds4 decode %.1f < oMLX %.1f t/s" % (ctx, row["ds4_decode"], row["omlx_decode"]))
        if "ds4_prefill" in row and "omlx_prefill" in row and row["ds4_prefill"] < row["omlx_prefill"]:
            failures.append("%s: ds4 prefill %.0f < oMLX %.0f t/s" % (ctx, row["ds4_prefill"], row["omlx_prefill"]))
    return failures


def _raise_signal_exit(signum, frame):
    """SIGTERM/SIGHUP default to a bare process exit that skips `finally:`
    blocks; turning them into SystemExit lets interleave()'s
    `finally: arm.stop()` run its bounded SIGTERM stop of the current arm
    instead of orphaning it."""
    raise SystemExit(128 + signum)


def install_signal_exit_handlers():
    signal.signal(signal.SIGTERM, _raise_signal_exit)
    signal.signal(signal.SIGHUP, _raise_signal_exit)


def main():
    install_signal_exit_handlers()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "lever"], required=True)
    ap.add_argument("--ds4-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="2048,32768,131072")
    ap.add_argument("--cold-tokens", type=int, default=31000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ds4-args", default="")   # space-joined extra ds4-server args (both arms)
    ap.add_argument("--ds4-env", default="")    # baseline mode: applied to the sole ds4 arm
    ap.add_argument("--base-env", default="")   # lever mode: applied to the "base" ds4 arm
    ap.add_argument("--lever-env", default="")  # lever mode: applied to the "lever" ds4 arm only
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    contexts = [int(x) for x in args.contexts.split(",") if x]
    extra_args = args.ds4_args.split() if args.ds4_args else []
    with open(FILLER, encoding="utf-8", errors="replace") as fp:
        filler = fp.read()

    if args.mode == "baseline":
        env = _parse_env(args.ds4_env)
        makers = {
            "ds4": lambda: Ds4Arm(args.ds4_model, os.path.join(out, "ds4"), env, extra_args, label="ds4"),
            "omlx": lambda: OmlxArm(os.path.join(out, "omlx")),
        }
        order = BASELINE_ORDER
    else:
        base_env = _parse_env(args.base_env)
        lever_env = _parse_env(args.lever_env)
        makers = {
            "base": lambda: Ds4Arm(args.ds4_model, os.path.join(out, "base"), base_env, extra_args,
                                   label="base"),
            "lever": lambda: Ds4Arm(args.ds4_model, os.path.join(out, "lever"), lever_env, extra_args,
                                    label="lever"),
        }
        order = LEVER_ORDER

    summary = interleave(makers, contexts, args.cold_tokens, filler, args.max_tokens, args.warmup, order)
    ctx_keys = [str(c) for c in contexts] + (["cold31k"] if args.cold_tokens else [])
    if args.mode == "baseline":
        summary["failures"] = verdict(summary)
        for ctx in ctx_keys:
            row = summary.get(ctx, {})
            print("m4_ab: %-8s ds4 decode %.1f / oMLX %.1f  ds4 prefill %.0f / oMLX %.0f" % (
                ctx, row.get("ds4_decode", 0), row.get("omlx_decode", 0),
                row.get("ds4_prefill", 0), row.get("omlx_prefill", 0)))
        for f in summary["failures"]:
            print("m4_ab: FAIL", f)
        print("m4_ab:", "FAIL" if summary["failures"] else "PASS")
        rc = 1 if summary["failures"] else 0
    else:
        for ctx in ctx_keys:
            row = summary.get(ctx, {})
            print("m4_ab: %-8s base decode %.1f / lever %.1f  base prefill %.0f / lever %.0f" % (
                ctx, row.get("base_decode", 0), row.get("lever_decode", 0),
                row.get("base_prefill", 0), row.get("lever_prefill", 0)))
        print("m4_ab: lever run done (no gate; compare against the Task 3 baseline by hand)")
        rc = 0

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "m4_ab.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
