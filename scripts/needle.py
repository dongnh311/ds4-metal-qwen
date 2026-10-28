#!/usr/bin/env python3
"""Needle-in-a-haystack gate for Qwen3.8 (cold prompt, configurable binary/env).

Usage:
  scripts/needle.py --tag kva36 --binary /path/ds4-server --tokens 52000 \
                    --ctx 131072 --env DS4_QWEN4_KVA_SPLIT=36,DS4_QWEN4_KVA_TAIL=2048

Prints hits/3 plus time.  The haystack carries a random session marker so the
KV disk cache never short-circuits the prefill being measured.
"""
import argparse, json, os, random, re, subprocess, sys, time, urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
MID = "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2"
PORT = 18086

SUBJ = ["dịch vụ thanh toán", "bộ đối soát", "luồng hoàn tiền", "cổng webhook", "hàng đợi sự kiện",
        "bộ định tuyến", "kho lưu trữ hoá đơn", "dịch vụ xác thực", "bộ giám sát độ trễ",
        "tầng lưu đệm", "dịch vụ thông báo", "bộ chuyển đổi định dạng"]
VERB = ["ghi log có tương quan", "thử lại theo exponential backoff", "đối soát theo lô mỗi 5 phút",
        "xác thực chữ ký HMAC", "lưu bản ghi bất biến", "đẩy sự kiện qua Kafka",
        "giới hạn 120 yêu cầu mỗi phút", "che dữ liệu thẻ trước khi lưu", "đo p95 và p99",
        "chuyển sang trạng thái chờ khi lỗi", "tách hàng đợi theo merchant", "cảnh báo khi lệch sổ"]
NUM = ["1472", "3305", "9077", "4821", "6120", "7355", "2048", "8891", "5510", "1190"]


def build(target):
    rnd = random.Random(42)
    codes = ["ZX-4821", "QK-9077", "MB-3350"]
    per = 35
    n_lines = max(60, target // per)
    marks = {int(n_lines * 0.15): ("thanh toán", codes[0]),
             int(n_lines * 0.50): ("đối soát", codes[1]),
             int(n_lines * 0.85): ("hoàn tiền", codes[2])}
    lines = ["PHIÊN BẢN TÀI LIỆU: %06x (nội bộ).\n" % random.SystemRandom().randrange(1 << 24)]
    for i in range(n_lines):
        if i in marks:
            comp, code = marks[i]
            lines.append("GHI CHÚ QUAN TRỌNG: mã dự án của thành phần %s là %s.\n" % (comp, code))
        lines.append("%d. %s %s %s, dùng cổng %s và mã %s.\n" % (
            i, rnd.choice(SUBJ), rnd.choice(VERB), rnd.choice(SUBJ),
            rnd.choice(NUM), "%04d" % (i * 37 % 10000)))
    lines.append("\nCâu hỏi: Ba mã dự án (định dạng XX-0000) xuất hiện trong tài liệu là gì? "
                 "Trả lời mỗi mã một dòng, không giải thích.\n")
    return "".join(lines), codes


def stop():
    subprocess.run(["pkill", "-TERM", "-f", "ds4-server"], capture_output=True)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", "ds4-server"], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def start(tag, binary, extra, ctx):
    cfg = json.load(open(REG))["models"][MID]["runtimes"]["ds4"]
    cmd = list(cfg["process_command"])
    if binary:
        idx = next(i for i, a in enumerate(cmd) if a.endswith("/ds4-server") or a == "ds4-server")
        cmd[idx] = binary
    cmd[cmd.index("-c") + 1] = str(ctx)
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    env.update(extra)
    lp = "/tmp/needle-%s.log" % tag
    log = open(lp, "w")
    p = subprocess.Popen(cmd, cwd=os.path.expanduser(cfg.get("process_cwd") or "~"), env=env,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2).read()
            return lp, p
        except Exception:
            if p.poll() is not None:
                return lp, p
            time.sleep(1)
    return lp, p


def ask(text, max_tokens=400):
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=2400).read())
    m = out["choices"][0]["message"]
    return time.time() - t, (m.get("content") or "") + "\n" + (m.get("reasoning_content") or ""), out.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tokens", type=int, default=52000)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    extra = {}
    for kv in filter(None, a.env.split(",")):
        k, _, v = kv.partition("=")
        extra[k] = v
    stop()
    lp, p = start(a.tag, a.binary, extra, a.ctx)
    if p.poll() is not None:
        print(a.tag, "START-FAIL", open(lp, errors="replace").read()[-300:])
        return
    time.sleep(0.5)
    text, codes = build(a.tokens)
    try:
        dt, ans, usage = ask(text)
        hit = [c for c in codes if c in ans]
        print("%-10s ctx=%d prompt_tokens=%s time=%.0fs hits=%d/3 %s"
              % (a.tag, a.ctx, usage.get("prompt_tokens"), dt, len(hit), hit), flush=True)
        print("   answer:", ans.replace("\n", " ")[:300], flush=True)
    except Exception as e:
        print(a.tag, "FAIL", e, flush=True)
    txt = open(lp, errors="replace").read()
    for pat in (r"KVA approx[^\n]*", r"prompt done [0-9.]+s", r"memory: KV[^\n]*"):
        m = re.findall(pat, txt)
        if m:
            print("   log:", m[-1][:160], flush=True)
    stop()


if __name__ == "__main__":
    main()