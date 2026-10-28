#!/usr/bin/env python3
"""12-fact grid across the haystack: a stricter recall gate than 3 needles.

Usage:
  scripts/needle-grid.py --tag kva36-128k --binary ./ds4-server --tokens 128000 \
      --ctx 262144 --env DS4_QWEN4_KVA_SPLIT=36
"""
import argparse, json, os, random, subprocess, time, urllib.request

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
NFACT = 12


def build(target):
    rnd = random.Random(4242)
    per = 35
    n_lines = max(120, target // per)
    codes = ["NB-%04d" % (1000 + k * 137) for k in range(NFACT)]
    marks = {int(n_lines * (0.05 + 0.075 * k)): codes[k] for k in range(NFACT)}
    lines = ["PHIÊN BẢN TÀI LIỆU: %06x (nội bộ).\n" % random.SystemRandom().randrange(1 << 24)]
    for i in range(n_lines):
        if i in marks:
            lines.append("GHI CHÚ QUAN TRỌNG: mã thành phần số %d là %s.\n"
                         % (list(marks).index(i) + 1, marks[i]))
        lines.append("%d. %s %s %s, dùng cổng %s và mã %s.\n" % (
            i, rnd.choice(SUBJ), rnd.choice(VERB), rnd.choice(SUBJ),
            rnd.choice(NUM), "%04d" % (i * 37 % 10000)))
    lines.append("\nCâu hỏi: trong tài liệu có 12 ghi chú quan trọng, mỗi ghi chú nêu một mã "
                 "dạng NB-xxxx. Liệt kê đủ 12 mã, mỗi dòng một mã, không giải thích.\n")
    return "".join(lines), codes


def stop():
    subprocess.run(["pkill", "-TERM", "-f", "ds4-server"], capture_output=True)
    for _ in range(50):
        if subprocess.run(["pgrep", "-f", "ds4-server"], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def start(tag, binary, extra, ctx):
    cfg = json.load(open(REG))["models"][MID]["runtimes"]["ds4"]
    cmd = list(cfg["process_command"])
    idx = next(i for i, a in enumerate(cmd) if a.endswith("/ds4-server") or a == "ds4-server")
    if binary:
        cmd[idx] = binary
    cmd[cmd.index("-c") + 1] = str(ctx)
    if extra and cmd and cmd[0].endswith("/env"):
        # our overrides must beat the registry's inline /usr/bin/env assignments
        cmd = [cmd[0]] + [a for a in cmd[1:]
                          if not ("=" in a and a.split("=", 1)[0] in extra)]
        cmd[1:1] = ["%s=%s" % (k, v) for k, v in extra.items()]
    env = dict(os.environ); env.update(cfg.get("env") or {}); env.update(extra)
    lp = "/tmp/grid-%s.log" % tag
    log = open(lp, "w")
    p = subprocess.Popen(cmd, cwd=os.path.expanduser(cfg.get("process_cwd") or "~"), env=env,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(300):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2).read()
            return lp, p
        except Exception:
            if p.poll() is not None:
                return lp, p
            time.sleep(1)
    return lp, p


def ask(text):
    body = json.dumps({"model": "local", "messages": [{"role": "user", "content": text}],
                       "max_tokens": 600, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    m = out["choices"][0]["message"]
    return time.time() - t, (m.get("content") or "") + "\n" + (m.get("reasoning_content") or ""), out.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--tokens", type=int, default=128000)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    extra = {}
    for kv in filter(None, a.env.split(",")):
        k, _, v = kv.partition("=")
        extra[k] = v
    stop()
    lp, p = start(a.tag, a.binary, extra, a.ctx)
    if p.poll() is not None:
        print(a.tag, "START-FAIL", open(lp, errors="replace").read()[-300:]); return
    time.sleep(0.5)
    text, codes = build(a.tokens)
    dt, ans, usage = ask(text)
    hit = [c for c in codes if c in ans]
    miss = [c for c in codes if c not in ans]
    print("%-12s ctx=%d prompt=%s time=%.0fs hits=%d/%d missing=%s"
          % (a.tag, a.ctx, usage.get("prompt_tokens"), dt, len(hit), NFACT, miss), flush=True)
    print("   answer:", ans.replace("\n", " ")[:240], flush=True)
    txt = open(lp, errors="replace").read()
    import re
    for pat in (r"KVA approx[^\n]*", r"prompt done [0-9.]+s", r"memory: KV[^\n]*"):
        m = re.findall(pat, txt)
        if m:
            print("   log:", m[-1][:150], flush=True)
    stop()


if __name__ == "__main__":
    main()