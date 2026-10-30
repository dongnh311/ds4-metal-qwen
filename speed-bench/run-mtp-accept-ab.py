#!/usr/bin/env python3
"""run-mtp-accept-ab.py — Campaign Task 2: MTP acceptance at the SAMPLING distribution.

Why the server and not ds4-bench:
  * bench decode is greedy-only (no temperature knob);
  * --mtp-margin was RETRACTED as a Qwen3.8 lever — grep shows zero margin
    references in the qwen4 spec path (77900-78700); the gate at 88926 belongs
    to the Ornith/dspark drafting loop.  The knob that actually moves the
    production accept rate (83.3% observed vs 73.2% greedy) is temperature
    itself — exactly the 0.3/0.6/1.0 A/B the gateway profile pre-registered.

Method per arm (temp in {0, 0.3, 0.6, 1.0}; 0 = greedy control):
  launch the EXACT registry process_command (PROD binary, steering, draft
  vocab, KV_GROW, --mtp --mtp-timing) with only --port and --kv-disk-dir
  swapped and DS4_LOCK_FILE pointed at a scratch lock; send 6 fixed
  streaming /v1/completions requests with fixed seeds (same every arm);
  SIGTERM the server — the acceptance counters print at session-free during
  the drain — and parse cycles/accepts from its stderr log.
  Recorded per request: TTFT (pre-first-byte) and decode t/s (first→last
  token), so one run answers the temperature question AND the prefill-facing
  TTFT question on real traffic.

Output: speed-bench/mtp-accept-task2.csv (append; header on first write)
plus one full server log per arm under speed-bench/receipts/accept-ab-<stamp>/.

Run inside a freeze window (kills the slot is NOT needed — this uses its own
lock file but still cannot coexist with the gateway slot in RAM; the caller
must hold /tmp/ds4.lock down and keep ds4-server otherwise off).
"""
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
REG_PATH = os.environ.get("ORCA_REGISTRY", HOME + "/.local/ai-gateway/runtime-registry.json")
ROW = os.environ.get("ORCA_ROW", "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2")
PORT = int(os.environ.get("DS4_AB_PORT", "18298"))
LOCK = os.environ.get("DS4_AB_LOCK", "/tmp/ds4-ab.lock")
TEMPS = [float(t) for t in (os.environ.get("DS4_AB_TEMPS") or "0 0.3 0.6 1.0").split()]
MAX_TOKENS = int(os.environ.get("DS4_AB_MAXTOK", "300"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = time.strftime("%Y%m%d-%H%M%S")
RECDIR = os.path.join(REPO, "speed-bench", "receipts", "accept-ab-" + STAMP)
CSV_PATH = os.environ.get("DS4_AB_CSV") or os.path.join(REPO, "speed-bench", "mtp-accept-task2.csv")

# Fixed prompts spanning the real doors (code, Vietnamese prose, structured
# output, agent-ish continuation).  Same text + same seed on every arm.
PROMPTS = [
    ("py", 701, "def merge_intervals(intervals):\n    \"\"\"Merge overlapping [start, end] intervals and return the sorted merged list.\"\"\"\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for cur in intervals[1:]:\n"),
    ("c", 702, "static bool qwen4_mtp_draft_rows(ds4_session *s, int *drafts, int cap) {\n    /* Verify width: the draft head proposes up to two tokens per cycle;\n     * the target model re-scores them in one batched pass. */\n    if (!s || !s->mtp_draft_valid) return false;\n    int n = 0;\n"),
    ("vi", 703, "Giải thích ngắn gọn vì sao bộ nhớ đệm KV (KV cache) giúp suy luận mô hình ngôn ngữ nhanh hơn, và đánh đổi bộ nhớ nào khi ngữ cảnh dài 200K token:\n"),
    ("json", 704, "Trả về duy nhất một JSON object mô tả thành phố Hà Nội với các khóa name, country, population, landmarks (mảng chuỗi). Không thêm văn bản nào khác.\n{\n"),
    ("math", 705, "A train leaves station A at 7:00 going 80 km/h toward station B, 260 km away. Another leaves B at 7:30 going 60 km/h. Compute step by step the time and the distance from A where they meet.\n"),
    ("agent", 706, "Task: rename the config key `old_timeout` to `request_timeout_ms` across the repo. Files: gateway/config.py (reads env AI_GATEWAY_OLD_TIMEOUT default 30), tests/test-suite-config-keys.py asserts the key name, README.md documents it. List the exact edits, one per line, in the form FILE: old -> new.\n"),
]


def registry_command(kv_dir):
    pc = json.load(open(REG_PATH))["models"][ROW]["runtimes"]["ds4"]["process_command"]
    cmd = list(pc)
    for i, tok in enumerate(cmd):
        if tok == "--port" and i + 1 < len(cmd):
            cmd[i + 1] = str(PORT)
        elif tok == "--kv-disk-dir" and i + 1 < len(cmd):
            cmd[i + 1] = kv_dir
    if "--mtp-timing" not in cmd:           # the live row carries it; be explicit anyway
        cmd.insert(cmd.index("--mtp") + 1, "--mtp-timing")
    return cmd


def wait_ready(proc, logf, timeout=240):
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


def stream_completion(temp, name, seed, prompt):
    body = {"model": "ds4", "prompt": prompt, "temperature": temp,
            "top_p": 0.95, "top_k": 20, "max_tokens": MAX_TOKENS,
            "seed": seed, "stream": True}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % PORT,
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    nev = 0
    usage = None
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
            if o.get("usage"):
                usage = o["usage"]
            for ch in o.get("choices") or []:
                txt = ch.get("text") or (ch.get("delta") or {}).get("content") or ""
                if txt:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    nev += 1
    ntok = (usage or {}).get("completion_tokens") or nev
    ttft = (first - t0) if first else float("nan")
    tps = (ntok - 1) / (last - first) if first and last and last > first and ntok > 1 else float("nan")
    return ntok, ttft, tps, nev


ACCEPT_RE = re.compile(r"mtp: (\d+) verify cycles, (\d+) drafts accepted \(([\d.]+)%\)")
DEPTH_RE = re.compile(r"depth3: (\d+) cycles, (\d+) second-draft accepts")


def parse_counters(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ("", "", "", "", "")
    m = ACCEPT_RE.search(txt)
    d = DEPTH_RE.search(txt)
    return ((m.group(1), m.group(2), m.group(3)) if m else ("", "", "")) + \
           ((d.group(1), d.group(2)) if d else ("", ""))


def main():
    os.makedirs(RECDIR, exist_ok=True)
    fresh = not os.path.exists(CSV_PATH)
    outf = open(CSV_PATH, "a", newline="")
    w = csv.writer(outf)
    if fresh:
        w.writerow(["ts", "temp", "req", "completion_tokens", "ttft_s", "decode_tps",
                    "sse_text_events", "mtp_cycles", "mtp_accepted", "mtp_pct",
                    "depth3_cycles", "depth3_accept2", "server_log"])
    print("=== accept A/B %s temps=%s port=%d ===" % (STAMP, TEMPS, PORT), flush=True)
    for temp in TEMPS:
        kv_dir = os.path.join(RECDIR, "kv-%s" % str(temp).replace(".", "p"))
        log_path = os.path.join(RECDIR, "server-%s.log" % str(temp).replace(".", "p"))
        cmd = registry_command(kv_dir)
        env = dict(os.environ, DS4_LOCK_FILE=LOCK)
        logf = open(log_path, "wb")
        binary = next(t for t in cmd if t.startswith("/") and not t.startswith("/usr/bin/env"))
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(binary))
        wait_ready(proc, logf)
        print("[%s] server up (%s)" % (temp, time.strftime("%H:%M:%S")), flush=True)
        rows = []
        for name, seed, prompt in PROMPTS:
            try:
                ntok, ttft, tps, nev = stream_completion(temp, name, seed, prompt)
            except Exception as exc:            # a failed request must not kill the arm
                print("[%s] %s FAILED: %s" % (temp, name, exc), flush=True)
                ntok, ttft, tps, nev = 0, float("nan"), float("nan"), 0
            rows.append((name, ntok, ttft, tps, nev))
            print("  %s temp=%s tok=%d ttft=%.2fs tps=%.1f" % (name, temp, ntok, ttft, tps), flush=True)
        proc.send_signal(signal.SIGTERM)
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < 120:
            time.sleep(1)
        if proc.poll() is None:
            proc.kill()
        logf.close()
        cycles, accepted, pct, d3, a2 = parse_counters(log_path)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for name, ntok, ttft, tps, nev in rows:
            w.writerow([ts, temp, name, ntok, "%.3f" % ttft, "%.2f" % tps, nev,
                        cycles, accepted, pct, d3, a2, os.path.relpath(log_path, REPO)])
        outf.flush()
        import shutil
        shutil.rmtree(kv_dir, ignore_errors=True)
        good = [r for r in rows if r[1] > 1]
        if good:
            print("[%s] DONE cycles=%s acc=%s%% mean_ttft=%.2fs mean_tps=%.1f\n"
                  % (temp, cycles, pct,
                     sum(r[2] for r in good) / len(good),
                     sum(r[3] for r in good) / len(good)), flush=True)
    outf.close()
    print("rows -> %s ; logs -> %s" % (CSV_PATH, RECDIR))


if __name__ == "__main__":
    main()
