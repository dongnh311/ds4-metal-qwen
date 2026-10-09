#!/usr/bin/env python3
"""Re-send / compaction TTFT on Ornith: baseline binary vs rewind-point binary.

Starts ONE ds4-server (the registry's Ornith-512K process_command, scratch port, scratch KV
dir) and grows a conversation the way production does: a ~16K cold first prompt (its cold
checkpoint is the only disk state, as in production where continued checkpoints are off on the
512K slots), then live continuations that append the model's own reply and ~6K tokens of new
user text until ~N tokens. Then, with temperature 0 and streaming:
  r1       the last turn (a normal live continuation)
  resend   r1 again (the client dropped r1's reply)              <- the 16:13 shape
  resend2  r1 a third time                                        <- a point must survive restore
  compact  r1 + a user "summarize" message (reply dropped)        <- the 16:46 shape
TTFT = first streamed delta. The rewind arm must also reproduce r1's reply byte for byte.
Thinking is off (enable_thinking=false) so replies are short and deterministic.
Prompt content is synthetic (a public story file); nothing private is read or printed.
argv: --bin DIR --label NAME --tokens 64000 [--rewind 16384] [--row-match ornith --row-suffix 512K]
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
HERE = os.path.dirname(os.path.abspath(__file__))
STORY = os.path.join(HERE, "..", "..", "tests", "long_context_story_prompt.txt")
PORT = 18199
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}}]


def server_argv(bin_dir, kv_dir, rewind, extra, row_match, row_suffix):
    reg = json.load(open(REG))["models"]
    key = next(k for k in reg if row_match in k.lower() and k.endswith(row_suffix))
    argv = list(reg[key]["runtimes"]["ds4"]["process_command"])
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.endswith("/ds4-server"):
            out.append(os.path.join(bin_dir, "ds4-server"))
        elif a in ("--port", "--kv-disk-dir", "--rewind-point-min-tokens"):
            i += 1
        else:
            out.append(a)
        i += 1
    out += ["--port", str(PORT), "--kv-disk-dir", kv_dir]
    if rewind:
        out += ["--rewind-point-min-tokens", str(rewind)]
    return key, out + list(extra)


SYSTEM = "You are a careful reading assistant. After each part, reply with the single word NEXT."


def user_turn(chunks, k, n):
    body = "\n\n".join(chunks[(k + i) % len(chunks)] for i in range(n))
    return {"role": "user", "content": "Story part %d:\n%s" % (k, body)}


def ask(msgs, max_tokens=48):
    body = json.dumps({"model": "x", "messages": msgs, "tools": TOOLS, "temperature": 0,
                       "max_tokens": max_tokens, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    out = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or [{}]
            d = choices[0].get("delta") or {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + \
                (json.dumps(d["tool_calls"]) if d.get("tool_calls") else "")
            if ttft is None and piece:
                ttft = time.time() - t0
            out.append(piece)
    return ttft, time.time() - t0, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=64000)
    ap.add_argument("--rewind", type=int, default=0)
    ap.add_argument("--row-match", default="ornith")
    ap.add_argument("--row-suffix", default="512K")
    ap.add_argument("--extra", default="", help='extra ds4-server args as ONE string, e.g. --extra="--flag 1"')
    a = ap.parse_args()
    kv = tempfile.mkdtemp(prefix="rp-kv-")
    logp = os.path.join(HERE, "server-%s.log" % a.label)
    log = open(logp, "w")
    key, argv = server_argv(a.bin, kv, a.rewind, shlex.split(a.extra), a.row_match, a.row_suffix)
    print("%-8s row %s" % (a.label, key[-40:]), flush=True)
    srv = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=a.bin)
    try:
        for _ in range(600):
            if srv.poll() is not None:
                sys.exit("server exited early; see %s" % logp)
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2)
                break
            except Exception:
                time.sleep(1)
        text = open(STORY).read()
        chunks = [text[i:i + 6000] for i in range(0, len(text), 6000)]
        msgs = [{"role": "system", "content": SYSTEM}, user_turn(chunks, 0, 10)]   # ~16K cold
        k, approx = 10, 60000
        while approx // 4 < a.tokens:
            ttft, dur, reply = ask(msgs, max_tokens=8)
            print("%-8s grow     ~%6d tok  ttft %5.1f s" % (a.label, approx // 4, ttft or -1), flush=True)
            msgs.append({"role": "assistant", "content": reply})
            msgs.append(user_turn(chunks, k, 4))                                   # ~6K per step
            k += 4
            approx += 24000
        res = {}
        res["r1"] = ask(msgs)
        res["resend"] = ask(msgs)
        res["resend2"] = ask(msgs)
        # Claude Code compaction: the last prompt up to its last message, reply dropped, plus a user ask
        res["compact"] = ask(msgs + [{"role": "user", "content": "Summarize the conversation so far in two sentences."}])
        for k, (ttft, dur, _) in res.items():
            print("%-8s %-8s ttft %7.1f s  total %7.1f s" % (a.label, k, ttft if ttft is not None else -1, dur),
                  flush=True)
        same = res["resend"][2] == res["r1"][2] and res["resend2"][2] == res["r1"][2]
        print("%-8s reply identical across r1/resend/resend2: %s" % (a.label, same))
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(120)
        except subprocess.TimeoutExpired:
            print("server did not exit in 120 s; left running (never SIGKILL Metal)", file=sys.stderr)
        shutil.rmtree(kv, ignore_errors=True)
        log.close()
    lines = open(logp).read().splitlines()
    hits = [l for l in lines if "rewind point hit" in l]
    misses = [l for l in lines if "live kv cache miss" in l]
    print("%-8s rewind point hits: %d, live misses: %d" % (a.label, len(hits), len(misses)))
    for l in hits + misses:
        print("   ", l[l.find("ds4-server:"):][:150])


if __name__ == "__main__":
    main()
