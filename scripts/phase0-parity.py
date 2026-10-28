#!/usr/bin/env python3
"""Phase 0 parity: same model, same prompt, different ds4-server binaries.

Usage: phase0-parity.py tag=path [tag=path ...]
Measures cold 24k prefill (backend "prompt done") and decode median (log avg).
Stops prod, benches each binary, leaves nothing running (caller restores).
"""
import json, os, random, re, statistics, subprocess, sys, time, urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
MID = "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2"
PORT = 18086
UNIT = ("Kỹ thuật viên đang đọc tài liệu kiến trúc hệ thống. Dòng này lặp lại để tạo "
        "ngữ cảnh dài và mỗi dòng có số khác nhau. ")
LINES = 600
DECODE_PROMPT = ("Viết một đoạn văn tiếng Việt khoảng 300 từ về kiến trúc microservices, "
                 "ưu nhược điểm và cách triển khai trên một máy chủ.")


def stop():
    subprocess.run(["pkill", "-TERM", "-f", "ds4-server"], capture_output=True)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", "ds4-server"], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def start(tag, binary, extra_env=None):
    cfg = json.load(open(REG))["models"][MID]["runtimes"]["ds4"]
    cmd = list(cfg["process_command"])
    idx = next(i for i, a in enumerate(cmd) if a.endswith("/ds4-server") or a == "ds4-server")
    cmd[idx] = binary
    if extra_env and cmd and cmd[0].endswith("/env"):
        cmd = [cmd[0]] + [x for x in cmd[1:]
                          if not ("=" in x and x.split("=", 1)[0] in extra_env)]
        cmd[1:1] = ["%s=%s" % (k, v) for k, v in extra_env.items()]
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    env.update(extra_env or {})
    lp = "/tmp/phase0/run-%s.log" % tag
    log = open(lp, "w")
    p = subprocess.Popen(cmd, cwd=os.path.expanduser(cfg.get("process_cwd") or "~"),
                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2).read()
            return p, lp
        except Exception:
            if p.poll() is not None:
                return None, lp
            time.sleep(1)
    return p, lp


def chat(prompt_text, max_tokens):
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": prompt_text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def last(logpath, pat, cast=float):
    hits = re.findall(pat, open(logpath, errors="replace").read())
    return cast(hits[-1]) if hits else None


def main():
    specs = [s.split("=", 1) for s in sys.argv[1:]]
    rows = []
    for i, (tag, spec) in enumerate(specs):
        binary, _, envs = spec.partition("|")
        extra_env = {}
        for kv in filter(None, envs.split(",")):
            k, _, v = kv.partition("=")
            extra_env[k] = v
        stop()
        p, lp = start(tag, binary, extra_env)
        if p is None:
            print("%-14s START-FAIL (%s)" % (tag, lp), flush=True)
            rows.append((tag, None, None, None)); continue
        try:
            salt = random.randrange(10000, 999999)
            text = "".join("[%d-%d] %s" % (salt, k * 7 % 9999, UNIT) for k in range(LINES))
            t0 = time.time()
            out = chat(text, 8)
            wall = time.time() - t0
            pt = out.get("usage", {}).get("prompt_tokens") or 0
            bp = last(lp, r"prompt done ([0-9.]+)s")
            tps = pt / bp if bp else None
            rates = []
            for _ in range(3):
                chat(DECODE_PROMPT, 160)
                r = last(lp, r"avg=([0-9.]+) t/s")
                if r:
                    rates.append(r)
            dec = statistics.median(rates) if rates else None
            ans = (out["choices"][0]["message"].get("content") or "").strip()
            rows.append((tag, tps, dec, pt))
            print("%-14s prefill %6.1f t/s (%ss, %d tok)  decode %5.1f t/s  ans=%r"
                  % (tag, tps or -1, round(bp or wall, 1), pt, dec or -1, ans[:14]), flush=True)
        except Exception as e:
            print("%-14s FAIL %s" % (tag, str(e)[:80]), flush=True)
            rows.append((tag, None, None, None))
        finally:
            stop()
    base = rows[0]
    print("\n=== vs %s ===" % base[0])
    for tag, tps, dec, _ in rows[1:]:
        if tps and base[1]:
            print("%-14s prefill %+5.1f%%   decode %+5.1f%%" % (
                tag, 100 * (tps / base[1] - 1),
                100 * (dec / base[2] - 1) if dec and base[2] else 0))


if __name__ == "__main__":
    main()