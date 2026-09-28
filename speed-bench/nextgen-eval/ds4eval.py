"""The reasoning suite: ds4-eval's embedded GPQA Diamond / SuperGPQA / AIME 2025 / MMLU-Pro cases."""
import os
import pathlib
import re
import subprocess

import server

EVAL_WITH_VALUE = {"-m", "--model", "--ple", "--prefill-chunk", "--ssd-streaming-cache-experts",
                   "--ssd-streaming-full-layers", "--ssd-streaming-preload-experts", "--threads"}
EVAL_NO_VALUE = {"--metal", "--ssd-streaming", "--ssd-streaming-cold", "--quality"}
REASON_RUNS = [("core", "GPQA Diamond", 8), ("core", "SuperGPQA", 8), ("core", "AIME2025", 8),
               ("hard", "MMLU-Pro", 20)]
TOKENS = 32768
CTX_CAP = 65536
_STATES = {"PASSED", "FAILED", "INCOMPLETE", "SKIPPED", "STOPPED", "PREFILL", "RUNNING", "PENDING"}
_ROW = re.compile(r"^\s*(\d+) (\S+)\s+(\d+)\s+(\d+)\s+(\d+) (.{20}) (.{20}) (.+)$")


def eval_argv(server_argv, root, suite, source, questions, trace):
    """ds4-eval arguments: the server's model/PLE/streaming flags plus the case selection."""
    out = [str(pathlib.Path(root) / "ds4-eval"), "--plain"]
    i = 1
    while i < len(server_argv):
        arg = server_argv[i]
        if arg in EVAL_WITH_VALUE and i + 1 < len(server_argv):
            out += [arg, server_argv[i + 1]]
            i += 2
            continue
        if arg in EVAL_NO_VALUE:
            out.append(arg)
        i += 1
    ctx = int(server.argv_value(server_argv, "-c") or CTX_CAP)
    out += ["-c", str(min(ctx, CTX_CAP)), "--suite", suite, "--source", source,
            "--questions", str(questions), "--tokens", str(TOKENS), "--trace", str(trace)]
    return out


def parse_report(stdout):
    rows = []
    for line in stdout.splitlines():
        m = _ROW.match(line)
        if not m or m.group(2) not in _STATES:
            continue
        source, _, case_id = m.group(8).strip().partition("/")
        rows.append({"idx": int(m.group(1)), "state": m.group(2), "prompt_tokens": int(m.group(3)),
                     "gen_tokens": int(m.group(4)), "given": m.group(6).strip(),
                     "correct": m.group(7).strip(), "source": source, "case_id": case_id})
    return rows


def run_reason(env, server_argv, root, out_dir):
    rows = []
    for suite, source, questions in REASON_RUNS:
        stem = "ds4-eval-%s" % source.replace(" ", "_")
        trace = pathlib.Path(out_dir) / (stem + ".trace")
        argv = eval_argv(server_argv, root, suite, source, questions, trace)
        print("   %s" % " ".join(argv), flush=True)
        proc = subprocess.run(argv, cwd=str(root), env={**os.environ, **env}, capture_output=True,
                              text=True, timeout=6 * 3600)
        (pathlib.Path(out_dir) / (stem + ".log")).write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        parsed = parse_report(proc.stdout)
        if len(parsed) != questions:
            raise RuntimeError("ds4-eval %s: expected %d report rows, got %d (exit %s), see %s.log" % (
                source, questions, len(parsed), proc.returncode, stem))
        rows += [dict(r, suite="reason", id="%s/%s" % (r["source"], r["case_id"]),
                      passed=r["state"] == "PASSED") for r in parsed]
    return rows
