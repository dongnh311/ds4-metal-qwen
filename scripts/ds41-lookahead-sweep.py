#!/usr/bin/env python3
"""ds41f lookahead sweep: prefill 30k cold + decode at 30k for k=0/1/3/6."""
import json, os, random, re, subprocess, sys, time, urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
MID = "local--DeepSeek-V4.1-Flash-1017-ds41f"
PORT = 18090
UNIT = ("Kỹ thuật viên đang đọc tài liệu kiến trúc hệ thống. Dòng này lặp lại để tạo "
        "ngữ cảnh dài và mỗi dòng có số khác nhau. ")


def stop():
    subprocess.run(["pkill", "-TERM", "-f", "ds4-server"], capture_output=True)
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "ds4-server"], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def start(tag, extra):
    cfg = json.load(open(REG))["models"][MID]["runtimes"]["ds4"]
    env = dict(os.environ); env.update(cfg.get("env") or {}); env.update(extra)
    lp = "/tmp/ds41la-%s.log" % tag
    log = open(lp, "w")
    p = subprocess.Popen(cfg["process_command"], cwd=os.path.expanduser(cfg.get("process_cwd") or "~"),
                         env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2).read()
            return p, lp
        except Exception:
            if p.poll() is not None:
                return None, lp
            time.sleep(1)
    return p, lp


def chat(text, mt):
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": text}],
                       "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return time.time() - t, out


def last(lp, pat):
    hits = re.findall(pat, open(lp, errors="replace").read())
    return float(hits[-1]) if hits else None


def main():
    ks = sys.argv[1:] or ["0", "1", "3", "6"]
    rows = []
    for k in ks:
        stop()
        env = {"DS4_METAL_V41_LOOKAHEAD_K": k} if k != "0" else {"DS4_METAL_V41_LOOKAHEAD_K": "0"}
        p, lp = start("k" + k, env)
        if p is None:
            print("k=%s START-FAIL %s" % (k, open(lp, errors="replace").read()[-200:]), flush=True)
            rows.append((k, None, None, None)); continue
        try:
            salt = random.SystemRandom().randrange(100000, 999999)
            doc = "".join("[%d-%d] %s" % (salt, i * 7 % 9999, UNIT) for i in range(780))
            t0 = time.time()
            _, out = chat(doc + "\n\nTóm tắt tài liệu trên trong một câu.", 8)
            pt = out.get("usage", {}).get("prompt_tokens") or 0
            bp = last(lp, r"prompt done ([0-9.]+)s")
            dt, out2 = chat(doc + "\n\nViết một đoạn tóm tắt khoảng 150 từ bằng tiếng Việt.", 150)
            dec = last(lp, r"avg=([0-9.]+) t/s")
            rows.append((k, bp, pt, dec))
            print("k=%-2s prefill %6.1fs (%d tok, %.1f t/s)  decode@30k %.1f t/s  wall %.0fs"
                  % (k, bp or -1, pt, (pt / bp) if bp else -1, dec or -1, dt), flush=True)
            la = re.findall(r"lookahead:[^\n]*", open(lp, errors="replace").read())
            for line in la[-2:]:
                print("   ", line.strip()[:220], flush=True)
        except Exception as e:
            print("k=%s FAIL %s" % (k, str(e)[:100]), flush=True)
            rows.append((k, None, None, None))
        finally:
            stop()
    base = next((r for r in rows if r[0] == "0"), None)
    if base and base[1] and base[3]:
        print("\n=== vs k=0 ===")
        for k, bp, pt, dec in rows:
            if k != "0" and bp and dec:
                print("k=%-2s prefill %+5.1f%%   decode %+5.1f%%" % (
                    k, 100 * (pt / bp) / (base[2] / base[1]) - 100,
                    100 * dec / base[3] - 100))


if __name__ == "__main__":
    main()