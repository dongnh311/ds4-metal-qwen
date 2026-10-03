#!/usr/bin/env python3
"""Model-free tests for `deploy-ai-gateway.sh smoke`.

The smoke command reads the gateway registry, starts one registry row's ds4 command on a scratch
port and talks to it. These tests point it at a scratch registry whose rows start this file as a
fake OpenAI server (`--serve`), so they need no GPU and never touch the live registry:

  - with several enabled ds4 rows the model must be named with --model;
  - --model picks that row's command;
  - a single enabled ds4 row still needs no --model;
  - a server that ignores SIGTERM is reported and left running, never SIGKILLed;
  - the tool round-trip records the leg-1 call and the leg-2 reply, without the random call id;
  - a missing tool call or a leg-2 reply without the tool result fails the smoke;
  - the round-trip output is compared with --ref like the other prompts.

Run: python3 tests/test_deploy_smoke.py
"""
import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "deploy-ai-gateway.sh")
PORT = 18397


def serve(argv):
    port, tag, ignore_term, no_tool, tool_reply = None, "", False, False, "Huế: 31°C, nắng nhẹ."
    issued = set()
    i = 0
    while i < len(argv):
        if argv[i] == "--port":
            port = int(argv[i + 1]); i += 2
        elif argv[i] == "--tag":
            tag = argv[i + 1]; i += 2
        elif argv[i] == "--kv-disk-dir":
            i += 2
        elif argv[i] == "--ignore-term":
            ignore_term = True; i += 1
        elif argv[i] == "--no-tool":
            no_tool = True; i += 1
        elif argv[i] == "--tool-reply":
            tool_reply = argv[i + 1]; i += 2
        else:
            i += 1
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"data": [{"id": tag}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            msgs = req["messages"]
            msg = {"content": "reply from " + tag}
            if req.get("tools") and msgs[-1]["role"] == "user" and not no_tool:
                # Like ds4-server: a fresh random id on every call.
                cid = "call_" + os.urandom(8).hex()
                issued.add(cid)
                msg = {"role": "assistant", "content": "", "reasoning_content": "Cần gọi get_weather.",
                       "tool_calls": [{"id": cid, "type": "function", "function": {
                           "name": "get_weather", "arguments": json.dumps({"city": "Huế"}, ensure_ascii=False)}}]}
            elif msgs[-1]["role"] == "tool":
                # Answer only a well-formed round trip: our own call, sent back with its result.
                call = (msgs[-2].get("tool_calls") or [{}])[0]
                ok = (call.get("id") in issued and msgs[-1].get("tool_call_id") == call["id"] and
                      json.loads(msgs[-1]["content"]).get("temp_c") == 31)
                msg = {"role": "assistant", "content": tool_reply if ok else "bad round trip",
                       "reasoning_content": "Có kết quả tool."}
            self.reply({"choices": [{"message": msg}], "usage": {"completion_tokens": 3}})

    http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def row(tag, enabled=True, extra=()):
    return {"runtimes": {"ds4": {
        "enabled": enabled,
        "process_command": [sys.executable, os.path.abspath(__file__), "--serve", "--tag", tag,
                            "--kv-disk-dir", "/nonexistent", "--port", "1"] + list(extra)}}}


def smoke(tmp, models, *args, env=None):
    reg = os.path.join(tmp, "registry.json")
    with open(reg, "w") as f:
        json.dump({"models": models, "version": 1}, f)
    out = tempfile.mkdtemp(dir=tmp)
    # The smoke refuses to start while any ds4 runs; a pgrep that finds nothing lets these tests run
    # next to a live PROD server.
    shim = os.path.join(tmp, "bin")
    if not os.path.exists(shim):
        os.makedirs(shim)
        with open(os.path.join(shim, "pgrep"), "w") as f:
            f.write("#!/bin/sh\nexit 1\n")
        os.chmod(os.path.join(shim, "pgrep"), 0o755)
    e = dict(os.environ, DS4_GATEWAY_REGISTRY=reg, DS4_SMOKE_PORT=str(PORT),
             PATH=shim + os.pathsep + os.environ["PATH"])
    e.update(env or {})
    p = subprocess.run(["bash", SCRIPT, "smoke", "--out", out] + list(args),
                       capture_output=True, text=True, env=e, timeout=120)
    return p, out


def read(path):
    with open(path) as f:
        return f.read()


def line(text, marker):
    """The first output line containing marker, or ''."""
    return next((l for l in text.splitlines() if marker in l), "")


failures = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else ": " + detail))
    if not cond:
        failures.append(name)


def fake_pids():
    p = subprocess.run(["pgrep", "-f", os.path.abspath(__file__) + " --serve"],
                       capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        two = {"ornith": row("ornith"), "qwen": row("qwen")}

        p, _ = smoke(tmp, two)
        check("several ds4 rows need --model", p.returncode != 0 and
              "--model" in p.stderr and "ornith" in p.stderr and "qwen" in p.stderr,
              "rc=%d stderr=%r" % (p.returncode, p.stderr[-400:]))

        p, out = smoke(tmp, two, "--model", "qwen")
        ok = p.returncode == 0 and os.path.exists(os.path.join(out, "vi.txt"))
        check("--model picks that row", ok and "reply from qwen" in read(os.path.join(out, "vi.txt")),
              "rc=%d out=%r err=%r" % (p.returncode, p.stdout[-400:], p.stderr[-400:]))

        p, _ = smoke(tmp, two, "--model", "absent")
        check("unknown --model is refused", p.returncode != 0 and "absent" in p.stderr,
              "rc=%d stderr=%r" % (p.returncode, p.stderr[-400:]))

        p, out = smoke(tmp, {"ornith": row("ornith", enabled=False), "qwen": row("qwen")})
        ok = p.returncode == 0 and os.path.exists(os.path.join(out, "code.txt"))
        check("one enabled ds4 row needs no --model",
              ok and "reply from qwen" in read(os.path.join(out, "code.txt")),
              "rc=%d err=%r" % (p.returncode, p.stderr[-400:]))

        before = set(fake_pids())
        p, _ = smoke(tmp, {"stubborn": row("stubborn", extra=["--ignore-term"])},
                     env={"DS4_SMOKE_STOP_TIMEOUT": "2"})
        left = [pid for pid in fake_pids() if pid not in before]
        check("a server that ignores SIGTERM is reported and left running",
              p.returncode != 0 and "SIGTERM" in p.stderr and len(left) == 1,
              "rc=%d left=%r stderr=%r" % (p.returncode, left, p.stderr[-400:]))
        for pid in left:
            os.kill(pid, signal.SIGKILL)   # the fake server is plain Python, not a Metal process

        one = {"ornith": row("ornith")}
        p, ref = smoke(tmp, one)
        tool_txt = os.path.join(ref, "tool.txt")
        tool = read(tool_txt) if os.path.exists(tool_txt) else ""
        check("tool round-trip records the call and the leg-2 reply",
              p.returncode == 0 and 'get_weather {"city": "Huế"}' in tool and "Huế: 31°C, nắng nhẹ." in tool
              and "call_" not in tool,
              "rc=%d tool=%r out=%r err=%r" % (p.returncode, tool, p.stdout[-400:], p.stderr[-400:]))

        p, _ = smoke(tmp, {"ornith": row("ornith", extra=["--no-tool"])})
        check("a reply without a tool call fails the smoke", p.returncode != 0 and "smoke tool" in p.stdout,
              "rc=%d out=%r err=%r" % (p.returncode, p.stdout[-400:], p.stderr[-400:]))

        p, _ = smoke(tmp, {"ornith": row("ornith", extra=["--tool-reply", "Trời nắng nhẹ."])})
        check("a leg-2 reply without the tool result fails the smoke",
              p.returncode != 0 and "smoke tool" in p.stdout,
              "rc=%d out=%r err=%r" % (p.returncode, p.stdout[-400:], p.stderr[-400:]))

        p, _ = smoke(tmp, one, "--ref", ref)
        check("the same round-trip matches --ref (the random call id is not recorded)",
              p.returncode == 0 and "same as ref" in line(p.stdout, "smoke tool"),
              "rc=%d out=%r err=%r" % (p.returncode, p.stdout[-400:], p.stderr[-400:]))

        p, _ = smoke(tmp, {"ornith": row("ornith", extra=["--tool-reply", "Huế: 31°C, có mưa."])}, "--ref", ref)
        check("a different leg-2 reply DIFFERS from --ref and fails",
              p.returncode != 0 and "DIFFERS from ref" in line(p.stdout, "smoke tool") and
              "same as ref" in line(p.stdout, "smoke vi"),
              "rc=%d out=%r err=%r" % (p.returncode, p.stdout[-600:], p.stderr[-400:]))

    print("%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve(sys.argv[2:])
    else:
        sys.exit(main())
