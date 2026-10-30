#!/usr/bin/env python3
"""run-loop-attract-ab.py — Campaign Task-2b: the loop-attractor half of the
temperature A/B (the "cheap version" the gateway profile calls still OPEN).

Question: does the LIVE sampling choice (temp 0.6 / top_p 0.95 / top_k 20)
escape decoder loops that greedy falls into, and at what acceptance cost?
The profile's reliability argument for 0.6 rests on guards.py citing the Qwen3
card ("greedy runs into repeated-action loops") — this measures it on THIS
model instead of citing the card.

Method (invariants inherited from reports/loop-sampling-ab-2026-07-27.md):
  * The backend is deterministic per (prompt, temp, seed): repeating the SAME
    config adds no information; sample the distribution by varying the SEED.
  * No judge. Loop = structural (repeated-line ratio); correctness =
    identifier spelling via difflib-close-to-known (the old report's
    escape-but-typo failure mode, `aes_gccm::{Nec256Gcmm}`, is counted
    separately from LOOP).
  * Prompt = the real loop attractor captured from the failed 27/07 agent
    turn (secure-store `aes-gcm` `aead` `OsRng` ×6) + two synthetic
    agent-door attractors of the same shape (edit-echo, vi enumeration).
  * Requests go to the EXACT registry command (PROD binary + steering +
    --mtp --mtp-timing), same server per arm, counters read at SIGTERM drain.

Arms: temperature 0.0 (greedy control), 0.3, 0.6 (LIVE), 1.0; top_p 0.95 /
top_k 20 always sent (greedy ignores them; the 0.7-era report had no top_p).

Output: speed-bench/loop-attract-ab.csv + per-arm server logs under
speed-bench/receipts/loop-ab-<stamp>/.  Run inside a freeze window.
"""
import csv
import difflib
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.request

HOME = os.path.expanduser("~")
REG_PATH = os.environ.get("ORCA_REGISTRY", HOME + "/.local/ai-gateway/runtime-registry.json")
ROW = os.environ.get("ORCA_ROW", "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2")
PORT = int(os.environ.get("DS4_AB_PORT", "18300"))
LOCK = os.environ.get("DS4_AB_LOCK", "/tmp/ds4-ab.lock")
TEMPS = [float(t) for t in (os.environ.get("DS4_AB_TEMPS") or "0 0.3 0.6 1.0").split()]
MAX_TOKENS = int(os.environ.get("DS4_AB_MAXTOK", "200"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = time.strftime("%Y%m%d-%H%M%S")
RECDIR = os.path.join(REPO, "speed-bench", "receipts", "loop-ab-" + STAMP)
CSV_PATH = os.environ.get("DS4_AB_CSV") or os.path.join(REPO, "speed-bench", "loop-attract-ab.csv")

# ── attractors ───────────────────────────────────────────────────────────────
_HEAD = ("The `rand::Rng` trait import was missing. Let me fix the nonce generation.\n\n")
_PAIR = ("I'll use `aes-gcm` `aead` `OsRng` via `aes-gcm` `aead` `Aead`:\n"
         "OK let me use `aes-gcm` `aead` `OsRng` via `aes-gcm` `aead` `Aead`:\n")
ATTRACTOR = _HEAD + _PAIR * 6                      # verbatim shape of the 27/07 failure

_EDIT_HEAD = ("Task: rename `old_timeout` to `request_timeout_ms`. Now apply the edit.\n")
_EDIT_PAIR = ("Applying the edit:\n"
              "Now apply the edit.\n")
EDIT_ECHO = _EDIT_HEAD + _EDIT_PAIR * 5

_VI_HEAD = "Liệt kê các tỉnh thành theo đúng mẫu đã mở đầu:\n1. Hà Nội\n2. Hải Phòng\n"
_VI_PAIR = "3. Hà Nội\n4. Hải Phòng\n"
VI_ENUM = _VI_HEAD + _VI_PAIR * 4

# (name, prompt) x seeds.  Seeds vary the DRAW, prompts vary the ATTRACTOR.
CASES = [
    ("aesgcm", ATTRACTOR, [901, 902, 903]),
    ("editecho", EDIT_ECHO, [911, 912]),
    ("vienum", VI_ENUM, [921, 922]),
]

# identifiers that count as "right" when continuing the attractor
KNOWN = ["aes-gcm", "aes_gcm", "aesgcm", "OsRng", "Aes256Gcm", "Aes128Gcm",
         "KeyInit", "Rng", "Nonce", "rand", "Payload", "encrypt", "decrypt",
         "request_timeout_ms", "old_timeout", "config", "tests", "README"]
_KNOWN_L = {k.lower(): k for k in KNOWN}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{3,}")


def score_text(text):
    """Deterministic, no judge: (ok_ids, bad_ids, uniq_ratio, loop)."""
    ok = bad = 0
    for tok in TOKEN_RE.findall(text):
        t = tok.lower()
        if t in _KNOWN_L:
            ok += 1
        else:
            close = difflib.get_close_matches(t, _KNOWN_L.keys(), n=1, cutoff=0.75)
            if close:
                bad += 1
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    uniq = len(set(lines)) / len(lines) if lines else 1.0
    loop = (len(lines) >= 4 and uniq < 0.8)
    return ok, bad, uniq, loop


def registry_command(kv_dir):
    pc = json.load(open(REG_PATH))["models"][ROW]["runtimes"]["ds4"]["process_command"]
    cmd = list(pc)
    for i, tok in enumerate(cmd):
        if tok == "--port" and i + 1 < len(cmd):
            cmd[i + 1] = str(PORT)
        elif tok == "--kv-disk-dir" and i + 1 < len(cmd):
            cmd[i + 1] = kv_dir
    return cmd


def wait_ready(proc, logf, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("ds4-server exited during startup, see " + logf.name)
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                return
        except OSError:
            time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    raise SystemExit("ds4-server not ready within %ss" % timeout)


def stream_completion(temp, seed, prompt):
    body = {"model": "ds4", "prompt": prompt, "temperature": temp,
            "top_p": 0.95, "top_k": 20, "max_tokens": MAX_TOKENS,
            "seed": seed, "stream": True}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % PORT,
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t_start = time.time()
    parts, first, last = [], None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                o = json.loads(payload)
            except ValueError:
                continue
            for ch in o.get("choices") or []:
                txt = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                if txt:
                    now = time.time()
                    first = first or now
                    last = now
                    parts.append(txt)
    text = "".join(parts)
    ttft = (first - t_start) if first else float("nan")
    dur = (last - first) if first and last and last > first else 0.0
    tps = (len(text) / 4) / dur if dur > 0 else float("nan")
    return text, ttft, tps


ACCEPT_RE = re.compile(r"mtp: (\d+) verify cycles, (\d+) drafts accepted \(([\d.]+)%\)")


def parse_counters(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return "", "", ""
    m = ACCEPT_RE.search(txt)
    return (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")


def main():
    os.makedirs(RECDIR, exist_ok=True)
    outf = open(CSV_PATH, "a", newline="")
    w = csv.writer(outf)
    if outf.tell() == 0:
        w.writerow(["ts", "temp", "req", "seed", "sse_chars", "ttft_s", "decode_tps",
                    "uniq_ratio", "loop", "ok_ids", "bad_ids",
                    "mtp_cycles", "mtp_accepted", "mtp_pct", "head"])
    print("=== loop-attract A/B %s temps=%s port=%d ===" % (STAMP, TEMPS, PORT), flush=True)
    for temp in TEMPS:
        kv_dir = os.path.join(RECDIR, "kv-%s" % str(temp).replace(".", "p"))
        log_path = os.path.join(RECDIR, "server-%s.log" % str(temp).replace(".", "p"))
        cmd = registry_command(kv_dir)
        logf = open(log_path, "wb")
        proc = subprocess.Popen(cmd, env=dict(os.environ, DS4_LOCK_FILE=LOCK),
                                stdout=logf, stderr=subprocess.STDOUT)
        wait_ready(proc, logf)
        print("[%s] server up (%s)" % (temp, time.strftime("%H:%M:%S")), flush=True)
        rows = []
        t0 = time.time()
        for name, prompt, seeds in CASES:
            for seed in seeds:
                try:
                    text, ttft, tps = stream_completion(temp, seed, prompt)
                except Exception as exc:
                    print("  %s/%d FAILED: %s" % (name, seed, exc), flush=True)
                    text, ttft, tps = "", float("nan"), float("nan")
                ok, bad, uniq, loop = score_text(text)
                rows.append((name, seed, len(text), ttft, tps, uniq, loop, ok, bad,
                             text[:60].replace("\n", " ")))
                print("  %s seed=%d chars=%d uniq=%.2f %s ok=%d bad=%d | %s"
                      % (name, seed, len(text), uniq, "LOOP" if loop else "free ",
                         ok, bad, text[:50].replace("\n", " ")), flush=True)
        proc.send_signal(signal.SIGTERM)
        tw = time.time()
        while proc.poll() is None and time.time() - tw < 120:
            time.sleep(1)
        if proc.poll() is None:
            proc.kill()
        logf.close()
        cycles, accepted, pct = parse_counters(log_path)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for name, seed, nch, ttft, tps, uniq, loop, ok, bad, head in rows:
            w.writerow([ts, temp, name, seed, nch, "%.3f" % ttft, "%.2f" % tps,
                        "%.3f" % uniq, int(loop), ok, bad, cycles, accepted, pct, head])
        outf.flush()
        import shutil
        shutil.rmtree(kv_dir, ignore_errors=True)
        n = len(rows)
        nloop = sum(1 for x in rows if x[6])
        print("[%s] DONE %d reqs, LOOP=%d/%d, acc=%s%%, wall=%.0fs\n"
              % (temp, n, nloop, n, pct, time.time() - t0), flush=True)
    outf.close()
    print("rows -> %s ; logs -> %s" % (CSV_PATH, RECDIR))


if __name__ == "__main__":
    main()
