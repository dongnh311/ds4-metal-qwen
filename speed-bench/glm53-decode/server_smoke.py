#!/usr/bin/env python3
"""GLM-5.3 SP1 server smoke on ds4-server --ssd-streaming: decode and prefill
speed, multi-turn reuse, and (with --stall) an injected gate timeout.

usage: server_smoke.py OUT_DIR LABEL [--stall N:MS] [--env K=V ...]

Writes OUT_DIR/LABEL.json and OUT_DIR/server-LABEL.log. Stops the server with
exactly one SIGTERM (never SIGKILL a Metal process).
"""
import argparse, http.client, json, os, subprocess, time, urllib.error, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("GLM_MODEL", os.path.expanduser(
    "~/orca/workspaces/ds4-metal-data/gguf/glm53/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf"))
PORT = 18191
FILLER = open(os.path.join(REPO, "speed-bench/promessi_sposi.txt"), encoding="utf-8").read()
CODE_Q = ("Write a Python function that parses an ISO-8601 duration string such as "
          "'P3DT4H5M' into total seconds, with a short docstring and three doctest examples.")
DOC_Q = "\n\nQuestion: summarize the passage above in eight bullet points."


def start(out, label, env):
    cmd = [os.path.join(REPO, "ds4-server"), "--metal", "-m", MODEL, "--ssd-streaming",
           "--power", "100", "--host", "127.0.0.1", "--port", str(PORT), "-c", "262144"]
    log = open(os.path.join(out, f"server-{label}.log"), "w")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **env})
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit(f"server exited during startup (rc {proc.returncode})")
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2)
            return proc, log, base
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise SystemExit("server did not come up")


def stop(proc, log):
    proc.terminate()  # exactly one SIGTERM
    try:
        proc.wait(120)
    except subprocess.TimeoutExpired:
        print("WARNING: server still alive after 120 s; not killing", flush=True)
    log.close()


def chat(base, messages, max_tokens, model="glm-5.3-flash-chat", extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra or {})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = {}; content = []; finish = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[6:]); t = time.time() - t0
                usage = ch.get("usage") or usage
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    finish = c.get("finish_reason") or finish
                    if d.get("content") or d.get("reasoning_content"):
                        first = t if first is None else first; last = t
                    content.append(d.get("content") or "")
    except (urllib.error.URLError, ConnectionError, http.client.HTTPException,
            json.JSONDecodeError) as e:
        return {"error": repr(e), "finish": finish, "content": "".join(content)}
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(first or 0, 2),
            "prefill_tps": round(pt / first, 1) if first else 0,
            "decode_tps": round((ct - 1) / (last - first), 2) if first and last and last > first and ct > 1 else 0,
            "finish": finish, "content": "".join(content)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("label")
    ap.add_argument("--stall"); ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = dict(kv.split("=", 1) for kv in a.env)
    if a.stall:
        env["DS4_GLM_STREAM_GATE_TEST_STALL"] = a.stall
    proc, log, base = start(a.out, a.label, env)
    rec = {"label": a.label, "env": env}
    try:
        user = lambda text: [{"role": "user", "content": text}]
        if a.stall:
            # The held gate falls in the first request's decode: it must fail
            # without a wrong token; the next request runs on the drain path.
            rec["stalled"] = chat(base, user(CODE_Q), 256)
            rec["after"] = chat(base, user(CODE_Q), 64)
        else:
            for name, msgs, n in [("warm", user(CODE_Q), 64), ("short1", user(CODE_Q), 256),
                                  ("short2", user(CODE_Q), 256),
                                  ("doc8k", user(FILLER[:30000] + DOC_Q), 256)]:
                rec[name] = chat(base, msgs, n)
            msgs = user("Explain briefly what an LRU cache is.")
            for turn, follow in enumerate(["Give a Python example.", "What is the complexity of get and put?"]):
                r = chat(base, msgs, 300, model="glm-5.3-flash", extra={"reasoning_effort": "low"})
                rec[f"chat{turn + 1}"] = {k: r.get(k) for k in ("prompt_tokens", "ttft_s", "finish")}
                msgs += [{"role": "assistant", "content": r.get("content", "")},
                         {"role": "user", "content": follow}]
            r = chat(base, msgs, 300, model="glm-5.3-flash", extra={"reasoning_effort": "low"})
            rec["chat3"] = {k: r.get(k) for k in ("prompt_tokens", "ttft_s", "finish")}
    finally:
        stop(proc, log)
    text = open(os.path.join(a.out, f"server-{a.label}.log"), errors="replace").read()
    rec["log"] = {"requires_rebuild": text.count("requires rebuild"),
                  "gate_timed_out": text.count("poll timed out"),
                  "gates_off": text.count("gates are now off")}
    for k, v in rec.items():
        if isinstance(v, dict) and "content" in v:
            v["content"] = v["content"][:200]
    json.dump(rec, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
