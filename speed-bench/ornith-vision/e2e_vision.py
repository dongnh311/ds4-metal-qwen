#!/usr/bin/env python3
"""Ornith native vision, end to end on the production argv (plan 2026-10-10, Task 6).

Starts ONE ds4-server from the registry's Ornith-512K process_command (scratch port, scratch
KV dir, --rewind-point-min-tokens 256 so the short prompts mark, plus --vision MMPROJ), runs
checks C1-C7, and writes e2e-<date>.json with outcomes, timings and log-line counts only.
The images are the synthetic fixtures in fixtures/; nothing private is read or printed.
Run it with the gateway's ds4 slot held down (DS4_LOCK_FILE pointed elsewhere).
argv: --bin DIR --mmproj FILE
"""
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
PORT = 18189
TOOLS = [{"type": "function", "function": {"name": "send_email", "description": "Send an email.",
          "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                         "required": ["to", "body"]}}}]
SYSTEM_TOOLS = "Answer the user's question directly from what you see. Never call tools."
DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def server_argv(bin_dir, kv_dir, mmproj, mtp=True, row_match="ornith", row_suffix="512K"):
    reg = json.load(open(REG))["models"]
    key = next(k for k in reg if row_match in k.lower() and k.endswith(row_suffix))
    argv = list(reg[key]["runtimes"]["ds4"]["process_command"])
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a.endswith("/ds4-server"):
            out.append(os.path.join(bin_dir, "ds4-server"))
        elif a in ("--port", "--kv-disk-dir", "--rewind-point-min-tokens", "--vision"):
            i += 1
        elif a == "--mtp" and not mtp:
            pass
        else:
            out.append(a)
        i += 1
    return key, out + ["--port", str(PORT), "--kv-disk-dir", kv_dir,
                       "--rewind-point-min-tokens", "256", "--vision", mmproj]


def data_uri(name):
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    with open(os.path.join(FIX, name), "rb") as f:
        return mime, base64.b64encode(f.read()).decode("ascii")


def image_part(name):
    mime, b64 = data_uri(name)
    return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}}


def post(path, body, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return out, time.time() - t0


def chat(messages, tools=None, max_tokens=200):
    body = {"model": "x", "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}
    if tools:
        body["tools"] = tools
    out, dt = post("/v1/chat/completions", body)
    msg = (out.get("choices") or [{}])[0].get("message") or {}
    usage = out.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return (msg.get("content") or ""), usage.get("prompt_tokens", 0), cached, dt


def has(text, *needles):
    low = (text or "").lower().translate(DASHES)
    return all(n in low for n in needles)


class Server:
    def __init__(self, argv, bin_dir, logp):
        self.logp = logp
        self.log = open(logp, "w")
        self.p = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, cwd=bin_dir)
        for _ in range(900):
            if self.p.poll() is not None:
                sys.exit("server exited early; see %s" % logp)
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2)
                return
            except Exception:
                time.sleep(1)
        sys.exit("server did not come up")

    def count(self, needle):
        self.log.flush()
        return sum(1 for line in open(self.logp, errors="replace") if needle in line)

    def stop(self):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(120)
        except subprocess.TimeoutExpired:
            print("server did not exit in 120 s; left running (never SIGKILL Metal)", file=sys.stderr)
        self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--mmproj", required=True)
    a = ap.parse_args()
    res = {}
    kv = tempfile.mkdtemp(prefix="ov-kv-")
    logp = os.path.join(tempfile.mkdtemp(prefix="ov-log-"), "server.log")
    key, argv = server_argv(a.bin, kv, a.mmproj, mtp=True)
    print("row", key[-40:], flush=True)
    srv = Server(argv, a.bin, logp)
    try:
        news = [{"role": "user", "content": [
            image_part("newspaper.jpg"),
            {"type": "text", "text": "Describe this front page: newspaper, date, main headline."}]}]
        text, ptok, _, dt = chat(news)
        res["C1"] = {"pass": has(text, "new york times", "1969", "men walk on moon") and
                     srv.count("request images=1") >= 1, "s": round(dt, 2), "prompt_tokens": ptok}
        c1_text = text

        code = [{"role": "user", "content": [
            image_part("code.png"),
            {"type": "text", "text": "Transcribe the function name and the string literal in this screenshot."}]}]
        text, _, _, dt = chat(code)
        res["C2"] = {"pass": has(text, "parse_invoice_total", "inv-2041"), "s": round(dt, 2)}

        base = [{"role": "system", "content": SYSTEM_TOOLS},
                {"role": "user", "content": [image_part("newspaper.jpg"),
                                             {"type": "text", "text": "Describe this front page."}]}]
        r1, p1, _, _ = chat(base, TOOLS)
        turn = base + [{"role": "assistant", "content": r1},
                       {"role": "user", "content": "Answer with only the main headline."}]
        hits0 = srv.count("rewind point hit")
        text, _, cached, dt = chat(turn, TOOLS)
        res["C3"] = {"pass": has(text, "men walk on moon") and cached >= p1, "s": round(dt, 2),
                     "cached": cached, "first_prompt": p1}
        c3_text = text

        text, _, _, dt = chat(turn, TOOLS)
        res["C4"] = {"pass": srv.count("rewind point hit") == hits0 + 1 and text == c3_text, "s": round(dt, 2)}

        compact = turn + [{"role": "user", "content": "Summarize the conversation in one sentence."}]
        text, _, _, dt = chat(compact, TOOLS)
        res["C5"] = {"pass": srv.count("rewind point hit") == hits0 + 2 and bool(text.strip()), "s": round(dt, 2)}

        mime, b64 = data_uri("code.png")
        anth = {"model": "x", "max_tokens": 200, "temperature": 0, "thinking": {"type": "disabled"},
                "tools": [{"name": "Read", "description": "Read a file.",
                           "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}},
                                            "required": ["file_path"]}}],
                "messages": [
                    {"role": "user", "content": "Open the screenshot file /tmp/shot.png."},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "Read",
                                                       "input": {"file_path": "/tmp/shot.png"}}]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "tu1", "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}]},
                        {"type": "text", "text": "What function is in the image the tool returned? Name it."}]}]}
        out, dt = post("/v1/messages", anth)
        text = "".join(b.get("text", "") for b in out.get("content") or [] if b.get("type") == "text")
        res["C7"] = {"pass": has(text, "parse_invoice_total"), "s": round(dt, 2)}
        res["log"] = {k: srv.count(k) for k in ("request images=", "rewind point remembered",
                                                  "rewind point hit", "multimodal live kv hit")}
    finally:
        srv.stop()

    _, argv = server_argv(a.bin, kv, a.mmproj, mtp=False)
    srv = Server(argv, a.bin, logp + ".nomtp")
    try:
        text, _, _, dt = chat(news)
        res["C6"] = {"pass": text == c1_text, "s": round(dt, 2)}
    finally:
        srv.stop()

    out = os.path.join(HERE, "e2e-%s.json" % time.strftime("%Y-%m-%d"))
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    for k in sorted(res):
        print(k, json.dumps(res[k], sort_keys=True))
    ok = all(v.get("pass") for k, v in res.items() if k.startswith("C"))
    print("E2E", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
