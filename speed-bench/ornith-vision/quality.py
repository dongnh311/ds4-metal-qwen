#!/usr/bin/env python3
"""Ornith native vision vs the describe sidecar, scored on the facts frozen in
QUALITY-PRE-REGISTRATION.md (plan 2026-10-10, Task 9).

Starts the scratch ds4-server of e2e_vision.py (production Ornith-512K argv + --vision) and a second sidecar
(production script and venv, port 8182), runs both arms on the six fixtures, and writes quality-<date>.json with
facts found, latencies and answer lengths only. Run it with the gateway's ds4 slot held down.
argv: --bin DIR --mmproj FILE
"""
import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_vision as e2e  # noqa: E402

SIDECAR_PY = os.path.expanduser("~/.local/mlx-vlm-env/bin/python")
SIDECAR = os.path.expanduser("~/Documents/GitHub/AI-Gateway-MLX/scripts/vision-sidecar.py")
SIDECAR_PORT = 8182
QUESTION = "Describe this image in detail, including any text you can read."
WRAP = ("[Attached image — described by the vision model (the main model cannot see pixels directly): %s]")


def frozen_facts():
    text = open(os.path.join(HERE, "QUALITY-PRE-REGISTRATION.md"), encoding="utf-8").read()
    return json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))


def found(answer, facts):
    low = (answer or "").lower().translate(e2e.DASHES)
    return [any(alt in low for alt in f.split("|")) for f in facts]


def b64_of(rel):
    path = os.path.normpath(os.path.join(HERE, rel))
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    with open(path, "rb") as fh:
        return mime, base64.b64encode(fh.read()).decode("ascii")


def describe(mime, b64):
    req = urllib.request.Request("http://127.0.0.1:%d/describe" % SIDECAR_PORT,
                                 json.dumps({"image_b64": b64, "media_type": mime}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        text = (json.loads(r.read().decode()).get("text") or "").strip()
    return text, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--mmproj", required=True)
    a = ap.parse_args()
    facts = frozen_facts()
    logdir = tempfile.mkdtemp(prefix="ovq-")
    side_log = open(os.path.join(logdir, "sidecar.log"), "w")
    side = subprocess.Popen([SIDECAR_PY, SIDECAR], stdout=side_log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, MLX_VISION_PORT=str(SIDECAR_PORT), MLX_VISION_IDLE="0"))
    kv = tempfile.mkdtemp(prefix="ovq-kv-")
    _key, argv = e2e.server_argv(a.bin, kv, a.mmproj, mtp=True)
    srv = e2e.Server(argv, a.bin, os.path.join(logdir, "server.log"))
    rows = {}
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % SIDECAR_PORT, timeout=2)
                break
            except Exception:
                time.sleep(1)
        for rel, fl in facts.items():
            mime, b64 = b64_of(rel)
            img = {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}}
            ans_n, _p, _c, t_n = e2e.chat([{"role": "user", "content": [img, {"type": "text", "text": QUESTION}]}],
                                          max_tokens=600)
            desc, t_d = describe(mime, b64)
            ans_s, _p, _c, t_s = e2e.chat([{"role": "user", "content": [
                {"type": "text", "text": WRAP % desc}, {"type": "text", "text": QUESTION}]}], max_tokens=600)
            fn, fs = found(ans_n, fl), found(ans_s, fl)
            rows[rel] = {"facts": len(fl), "native": sum(fn), "sidecar": sum(fs),
                         "native_hits": fn, "sidecar_hits": fs,
                         "native_s": round(t_n, 2), "sidecar_describe_s": round(t_d, 2),
                         "sidecar_answer_s": round(t_s, 2),
                         "native_chars": len(ans_n), "sidecar_chars": len(ans_s), "description_chars": len(desc)}
            print("%-48s native %d/%d (%.1f s)  sidecar %d/%d (%.1f + %.1f s)" % (
                rel[-48:], sum(fn), len(fl), t_n, sum(fs), len(fl), t_d, t_s), flush=True)
    finally:
        srv.stop()
        side.send_signal(signal.SIGTERM)
        try:
            side.wait(60)
        except subprocess.TimeoutExpired:
            side.kill()
        side_log.close()
    tn = sum(r["native"] for r in rows.values())
    ts = sum(r["sidecar"] for r in rows.values())
    verdict = "PASS" if tn >= ts else "FAIL"
    out = {"rows": rows, "native_total": tn, "sidecar_total": ts,
           "facts_total": sum(r["facts"] for r in rows.values()), "verdict": verdict}
    with open(os.path.join(HERE, "quality-%s.json" % time.strftime("%Y-%m-%d")), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("TOTAL native %d sidecar %d of %d -> %s" % (tn, ts, out["facts_total"], verdict))


if __name__ == "__main__":
    main()
