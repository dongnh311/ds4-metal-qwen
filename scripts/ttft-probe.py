#!/usr/bin/env python3
"""ttft-probe — measure time-to-first-token and prefill/decode rates.

Gateway mode (prod 35B / any OpenAI-compatible endpoint):
    scripts/ttft-probe.py --url http://127.0.0.1:8090/v1/chat/completions \
        --model local --prompt-file /tmp/p2k.txt --max-tokens 64

CLI mode (local ds4, e.g. ds41f): runs the binary and timestamps the first
byte of generated text.
    scripts/ttft-probe.py --cli --bin ./ds4 --model-path MODEL.gguf \
        --ctx 524288 --cache 4300 --prompt-file /tmp/p2k.txt --max-tokens 64

Both modes print: prompt tokens (if reported), TTFT seconds, total seconds,
decode tok/s (estimated from the client side).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


def read_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return args.prompt or "Hello"


def gateway(args, prompt):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(args.url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + (args.api_key or ""),
    })
    t0 = time.time()
    ttft = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
                text.append(piece)
    total = time.time() - t0
    return ttft or total, total, chunks, "".join(text)


def cli(args, prompt):
    cmd = [args.bin, "-m", args.model_path, "--metal", "--ssd-streaming",
           "-c", str(args.ctx), "--ssd-streaming-cache-experts", str(args.cache)]
    if args.kv_disk:
        cmd += ["--kv-disk-dir", args.kv_disk, "--kv-disk-space-mb", "32768"]
    if args.prompt_file:
        cmd += ["--prompt-file", args.prompt_file]
    else:
        cmd += ["-p", prompt]
    cmd += ["-n", str(args.max_tokens), "--temp", "0"]

    t0 = time.time()
    ttft = None
    out = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    # first stdout byte marks the first generated token (ds4 streams tokens)
    while True:
        b = proc.stdout.read(1)
        if not b:
            break
        if ttft is None:
            ttft = time.time() - t0
        out.append(b)
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.wait()
    total = time.time() - t0
    m = re.search(r"prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s", err)
    prefill_tps = float(m.group(1)) if m else None
    gen_tps = float(m.group(2)) if m else None
    ntok = re.search(r"processing (\d+) input tokens", err)
    return ttft or total, total, int(ntok.group(1)) if ntok else 0, prefill_tps, gen_tps, b"".join(out).decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default=os.environ.get("TTFT_API_KEY", ""))
    ap.add_argument("--bin", default="./ds4")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--ctx", type=int, default=524288)
    ap.add_argument("--cache", type=int, default=4300)
    ap.add_argument("--kv-disk", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args()

    prompt = read_prompt(args)
    if args.cli:
        ttft, total, ntok, ptps, gtps, text = cli(args, prompt)
        print(f"prompt_tokens={ntok} ttft_s={ttft:.2f} total_s={total:.2f} "
              f"prefill_tps={ptps if ptps is not None else '?'} decode_tps={gtps if gtps is not None else '?'}")
        print("first_output:", text[:120].replace("\n", " "))
    else:
        ttft, total, chunks, text = gateway(args, prompt)
        print(f"ttft_s={ttft:.2f} total_s={total:.2f} chunks={chunks}")
        print("first_output:", text[:120].replace("\n", " "))


if __name__ == "__main__":
    sys.exit(main())