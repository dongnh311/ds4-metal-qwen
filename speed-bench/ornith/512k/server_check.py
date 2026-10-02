#!/usr/bin/env python3
"""Ornith 512K server check: the disk KV cache is keyed by the YaRN factor.

Three server runs on one fresh --kv-disk-dir, the same ~30K-token prompt each:
  1. 262k-store:   -c 262144 (no YaRN); the prompt is stored (cold store + shutdown store);
  2. 512k-cold:    -c 524288 (YaRN 2); the log names <kv-dir>/yarn-2 and the prompt is NOT
                   restored from the 262K cache (different rope);
  3. 512k-restore: -c 524288 again; the prompt IS restored (>= 90% of it cached).
Servers stop with SIGTERM only and are waited for (a Metal process is never SIGKILLed).

  server_check.py --bin REPO --model GGUF --draft-vocab TXT --kv-dir DIR --out DIR
                  [--port 18298] [--chars 120000]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "nextgen-eval"))


def keyed_dir(log_text):
    m = re.search(r"keyed kv cache directory (\S+)", log_text)
    return m.group(1) if m else None


def cached(resp):
    usage = resp.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return int(details.get("cached_tokens") or 0), int(usage.get("prompt_tokens") or 0)


def verdict(phases):
    why = []
    by = {p["phase"]: p for p in phases}
    cold, restore = by.get("512k-cold"), by.get("512k-restore")
    if not cold or not restore:
        return False, ["missing phase"]
    if cold["cached"] > 0:
        why.append("512k-cold restored %d tokens written at another rope" % cold["cached"])
    for p in (cold, restore):
        if not (p["dir"] or "").endswith("yarn-2"):
            why.append("%s did not key its cache yarn-2 (dir %r)" % (p["phase"], p["dir"]))
    if restore["cached"] < 0.9 * restore["prompt"]:
        why.append("512k-restore cached only %d of %d" % (restore["cached"], restore["prompt"]))
    return not why, why


def run_phase(a, name, ctx, prompt, extra):
    log_path = os.path.join(a.out, name + ".log")
    cmd = [os.path.join(a.bin, "ds4-server"), "--metal", "-m", a.model, "-c", str(ctx), "--mtp",
           "--kv-disk-dir", a.kv_dir, "--kv-disk-space-mb", "32768",
           "--kv-cache-cold-max-tokens", "262144", "--host", "127.0.0.1", "--port", str(a.port)] + extra
    env = dict(os.environ, DS4_QWEN35_MTP_DRAFT_VOCAB=a.draft_vocab)
    env.pop("DS4_QWEN4_YARN_FACTOR", None)
    with open(log_path, "w") as log:
        srv = subprocess.Popen(cmd, cwd=a.bin, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = "http://127.0.0.1:%d" % a.port
    try:
        for _ in range(900):
            if srv.poll() is not None:
                sys.exit("%s: ds4-server exited during startup, see %s" % (name, log_path))
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(1)
        body = json.dumps({"model": "ds4", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 8, "temperature": 0, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    finally:
        srv.terminate()
        try:
            srv.wait(300)          # the shutdown store writes the KV
        except subprocess.TimeoutExpired:
            sys.exit("%s: ds4-server pid %d did not exit 300 s after SIGTERM; left running" % (name, srv.pid))
    c, n = cached(resp)
    with open(log_path) as f:
        return {"phase": name, "ctx": ctx, "dir": keyed_dir(f.read()), "cached": c, "prompt": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-vocab", required=True)
    ap.add_argument("--kv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18298)
    ap.add_argument("--chars", type=int, default=120000)
    a = ap.parse_args()
    if os.path.exists(a.kv_dir) and os.listdir(a.kv_dir):
        sys.exit("--kv-dir must be empty: " + a.kv_dir)
    os.makedirs(a.kv_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    from eval_suites import haystack
    data = os.environ.get("NEXTGEN_EVAL_DATA", os.path.expanduser("~/orca/workspaces/ds4-metal-data/evals/nextgen"))
    prompt = ("Here is a C source file.\n\n" + haystack(os.path.join(data, "haystack.c"), a.chars)
              + "\n\nIn one word, which language is this file written in?")
    interval0 = ["--kv-cache-continued-interval-tokens", "0"]
    phases = [run_phase(a, "262k-store", 262144, prompt, []),
              run_phase(a, "512k-cold", 524288, prompt, interval0),
              run_phase(a, "512k-restore", 524288, prompt, interval0)]
    ok, why = verdict(phases)
    with open(os.path.join(a.out, "server_check.json"), "w") as f:
        json.dump({"phases": phases, "ok": ok, "why": why}, f, indent=1)
    for p in phases:
        print("%-13s ctx %-7d dir %-40s cached %d/%d" % (p["phase"], p["ctx"], p["dir"], p["cached"], p["prompt"]))
    print("server_check:", "PASS" if ok else "FAIL " + "; ".join(why))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
