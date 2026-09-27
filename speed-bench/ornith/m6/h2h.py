#!/usr/bin/env python3
"""Ornith head-to-head: ds4 vs the live oMLX across context lengths.

For each runtime in turn (one model process at a time, the M4 harness's arms),
sends one cold request per context length, ascending, and records TTFT, the
prefill and decode rates, the finish reason and the model process's peak
memory footprint during that request (macOS `footprint`, sampled every 2 s).
A runtime that errors or dies at some length gets that length (and every
longer one) recorded as failed instead of stopping the whole run.

Memory guard: if system swap grows by more than --swap-guard-gib during a
request, the model process gets SIGTERM (never SIGKILL: a killed Metal
process can wedge its model file) and the length is recorded as aborted.

  python3 speed-bench/ornith/m6/h2h.py --ds4-model GGUF --out DIR \
      --ds4-env "DS4_QWEN35_ATTN_NAX=1,DS4_QWEN35_MTP_DRAFT_VOCAB=..." \
      --contexts 2048,8192,32768,65536,131072,196608,250000 --reps-short 2
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # speed-bench/ornith (m4_ab)
import m4_ab                                          # noqa: E402

PROC_NAMES = {"ds4": ["ds4-server"], "omlx": ["omlx-server"]}


def model_pid(role):
    for name in PROC_NAMES[role]:
        out = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout.split()
        if out:
            return int(out[0])
    if role == "omlx":
        out = subprocess.run(["pgrep", "-f", "omlx-venv/bin/omlx"], capture_output=True, text=True).stdout.split()
        if out:
            return int(out[0])
    return None


_FOOT = re.compile(r"Footprint:\s*([\d.]+)\s*(KB|MB|GB)")


def footprint_gib(pid):
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _FOOT.search(out)
    if not m:
        return None
    scale = {"KB": 1.0 / (1024 * 1024), "MB": 1.0 / 1024, "GB": 1.0}[m.group(2)]
    return float(m.group(1)) * scale


def swap_used_gib():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


class Sampler:
    """Polls the model process's footprint and the system swap while a request runs."""

    def __init__(self, role, swap_guard_gib):
        self.role, self.swap_guard = role, swap_guard_gib
        self.peak = 0.0
        self.aborted = False
        self._stop = threading.Event()
        self._swap0 = swap_used_gib()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            pid = model_pid(self.role)
            if pid:
                g = footprint_gib(pid)
                if g:
                    self.peak = max(self.peak, g)
                if swap_used_gib() - self._swap0 > self.swap_guard and not self.aborted:
                    self.aborted = True
                    try:
                        os.kill(pid, 15)   # SIGTERM only
                    except OSError:
                        pass
            self._stop.wait(2.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(30)


def run_arm(role, arm, contexts, reps_short, max_tokens, swap_guard):
    filler = open(m4_ab.FILLER, encoding="utf-8", errors="replace").read()
    rows, dead = [], None
    arm.start()
    try:
        arm.ready()
        pid = model_pid(role)
        idle = footprint_gib(pid) if pid else None
        cpt = m4_ab.calibrate_chars_per_token(arm)
        arm.stream(m4_ab.build_prompt("nonce-warm", 512, filler, cpt), max_tokens)
        for ctx in contexts:
            reps = reps_short if ctx <= 32768 else 1
            for rep in range(reps):
                row = {"runtime": role, "context": ctx, "rep": rep}
                if dead:
                    row["error"] = "not run: " + dead
                    rows.append(row)
                    continue
                nonce = "nonce-h2h-%s-%d" % (role, int(time.time() * 1000) % 1000000)
                t0 = time.time()
                with Sampler(role, swap_guard) as s:
                    try:
                        timing, usage = arm.stream(m4_ab.build_prompt(nonce, ctx, filler, cpt), max_tokens)
                        rates = m4_ab.qwen_gate.request_rates(timing)
                        row.update({"ttft_s": timing.get("ttft_s"), "prefill_tps": rates.get("prefill_tps"),
                                    "decode_tps": rates.get("decode_tps"),
                                    "prompt_tokens": (usage or {}).get("prompt_tokens"),
                                    "completion_tokens": (usage or {}).get("completion_tokens"),
                                    "cached_tokens": m4_ab.cached_tokens(usage),
                                    "finish_reason": timing.get("finish_reason")})
                    except Exception as e:   # HTTP error, reset connection, server death
                        row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                row["peak_footprint_gib"] = round(s.peak, 2) if s.peak else None
                row["idle_footprint_gib"] = round(idle, 2) if idle else None
                row["wall_s"] = round(time.time() - t0, 1)
                if s.aborted:
                    row["error"] = "aborted by the swap guard (SIGTERM)"
                if "error" in row and model_pid(role) is None:
                    dead = "server gone after %d-token request" % ctx
                elif "error" in row and s.aborted:
                    dead = "aborted at %d tokens" % ctx
                rows.append(row)
                print("h2h: %s" % json.dumps(row), flush=True)
    finally:
        try:
            arm.stop()
        except SystemExit as e:
            print("h2h: stop: %s" % e, flush=True)
    return rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4-model", required=True)
    ap.add_argument("--ds4-env", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default="2048,8192,32768,65536,131072,196608,250000")
    ap.add_argument("--reps-short", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--swap-guard-gib", type=float, default=8.0)
    ap.add_argument("--order", default="omlx,ds4")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    contexts = [int(x) for x in a.contexts.split(",")]
    env = m4_ab._parse_env(a.ds4_env)
    makers = {"ds4": lambda: m4_ab.Ds4Arm(a.ds4_model, os.path.join(a.out, "ds4"), extra_env=env),
              "omlx": lambda: m4_ab.OmlxArm(os.path.join(a.out, "omlx"))}
    rows = []
    for role in a.order.split(","):
        m4_ab.guard_free()
        rows += run_arm(role, makers[role](), contexts, a.reps_short, a.max_tokens, a.swap_guard_gib)
        m4_ab._wait_free()
    with open(os.path.join(a.out, "h2h.json"), "w") as fp:
        json.dump({"contexts": contexts, "ds4_env": a.ds4_env, "rows": rows}, fp, indent=1)
    print("h2h: wrote %s" % os.path.join(a.out, "h2h.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
