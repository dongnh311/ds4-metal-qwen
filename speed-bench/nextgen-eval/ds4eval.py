"""The reasoning suite: ds4-eval's embedded GPQA Diamond / SuperGPQA / AIME 2025 / MMLU-Pro cases."""
import os
import pathlib
import re
import subprocess

import server

EVAL_WITH_VALUE = {"-m", "--model", "--ple", "--prefill-chunk", "--ssd-streaming-cache-experts",
                   "--ssd-streaming-full-layers", "--ssd-streaming-preload-experts", "--threads",
                   "--dir-steering-file", "--dir-steering-ffn", "--dir-steering-attn"}
EVAL_NO_VALUE = {"--metal", "--ssd-streaming", "--ssd-streaming-cold", "--quality"}
# Serving flags that do not apply to ds4-eval's offline runs. Every other flag must be classified here
# or above: a new candidate flag (runtime projection, YaRN, ...) is refused, never dropped silently.
SERVER_ONLY_WITH_VALUE = {"-c", "--ctx", "--host", "--port", "--kv-disk-dir", "--kv-disk-space-mb",
                          "--kv-cache-cold-max-tokens", "--kv-cache-continued-interval-tokens",
                          "--think-budget"}
SERVER_ONLY_NO_VALUE = {"--mtp"}
REASON_RUNS = [("core", "GPQA Diamond", 8), ("core", "SuperGPQA", 8), ("core", "AIME2025", 8),
               ("hard", "MMLU-Pro", 20)]
TOKENS = 32768
CTX_CAP = 65536
QWEN4_NATIVE_CTX = 262144  # qwen4exp.context_length of every Qwen3.8-Flash-Next GGUF this harness runs


def eval_env(env, server_argv):
    """The arm's env for ds4-eval. ds4-eval runs at -c <= CTX_CAP, below the native context, so it
    would never derive YaRN from -c the way the server does: hand it the server's factor, by ds4.c's
    rule (the smallest power of two covering -c / native). An explicit DS4_QWEN4_YARN_FACTOR wins,
    from the arm's env or from the shell (the server runs with {**os.environ, **env} too)."""
    out = dict(env)
    if out.get("DS4_QWEN4_YARN_FACTOR") or os.environ.get("DS4_QWEN4_YARN_FACTOR"):
        return out
    ctx = int(server.argv_value(server_argv, "-c") or 0)
    if ctx <= QWEN4_NATIVE_CTX:
        return out
    factor = 2
    while QWEN4_NATIVE_CTX * factor < ctx:
        factor *= 2
    out["DS4_QWEN4_YARN_FACTOR"] = str(factor)
    return out
_STATES = {"PASSED", "FAILED", "INCOMPLETE", "SKIPPED", "STOPPED", "PREFILL", "RUNNING", "PENDING"}
_GRADED = {"PASSED", "FAILED", "INCOMPLETE"}  # INCOMPLETE counts as a fail; anything else means the run broke
# C pads %-20.20s by bytes, so the report is matched as bytes and each field decoded afterwards.
_ROW = re.compile(rb"^\s*(\d+) (\S+)\s+(\d+)\s+(\d+)\s+(\d+) (.{20}) (.{20}) (.+)$")


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
        if arg in SERVER_ONLY_WITH_VALUE:
            i += 2
            continue
        if arg in EVAL_NO_VALUE:
            out.append(arg)
        elif arg not in SERVER_ONLY_NO_VALUE:
            raise ValueError("ds4-eval cannot place server flag %r: add it to ds4eval.EVAL_* (passed to "
                             "ds4-eval) or SERVER_ONLY_* (serving only)" % arg)
        i += 1
    ctx = int(server.argv_value(server_argv, "-c") or CTX_CAP)
    out += ["-c", str(min(ctx, CTX_CAP)), "--suite", suite, "--source", source,
            "--questions", str(questions), "--tokens", str(TOKENS), "--trace", str(trace)]
    return out


def _text(raw):
    return raw.decode("utf-8", errors="replace")


def parse_report(stdout):
    """Report rows from ds4-eval's stdout (bytes, or str for convenience)."""
    if isinstance(stdout, str):
        stdout = stdout.encode()
    rows = []
    for line in stdout.splitlines():
        m = _ROW.match(line)
        if not m or _text(m.group(2)) not in _STATES:
            continue
        source, _, case_id = _text(m.group(8)).strip().partition("/")
        rows.append({"idx": int(m.group(1)), "state": _text(m.group(2)), "prompt_tokens": int(m.group(3)),
                     "gen_tokens": int(m.group(4)), "given": _text(m.group(6)).strip(),
                     "correct": _text(m.group(7)).strip(), "source": source, "case_id": case_id})
    return rows


def _run(argv, cwd, env):
    """Run ds4-eval to the end. No timeout, and on Ctrl-C SIGTERM and wait: a Metal process killed with
    -9 can wedge its GGUF until reboot."""
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate()
    except BaseException:
        proc.terminate()
        proc.wait()
        raise
    return proc.returncode, out, err


def run_reason(env, server_argv, root, out_dir):
    """Yields the rows of each ds4-eval run as soon as the run is checked."""
    for suite, source, questions in REASON_RUNS:
        stem = "ds4-eval-%s" % source.replace(" ", "_")
        trace = pathlib.Path(out_dir) / (stem + ".trace")
        argv = eval_argv(server_argv, root, suite, source, questions, trace)
        print("   %s" % " ".join(argv), flush=True)
        code, out, err = _run(argv, str(root), {**os.environ, **eval_env(env, server_argv)})
        (pathlib.Path(out_dir) / (stem + ".log")).write_text(
            " ".join(argv) + "\n" + _text(out) + "\n--- stderr ---\n" + _text(err))
        parsed = parse_report(out)
        states = sorted({r["state"] for r in parsed} - _GRADED)
        # ds4_eval.c exits `rc || failed || incomplete ? 1 : 0`, so a wrong answer alone exits 1; an engine
        # error shows as ungraded rows or as an exit code the rows do not explain.
        expected = 1 if any(r["state"] in ("FAILED", "INCOMPLETE") for r in parsed) else 0
        if code != expected or len(parsed) != questions or states:
            raise RuntimeError("ds4-eval %s: exit %s, %d/%d report rows, ungraded states %s; see %s.log" % (
                source, code, len(parsed), questions, states or "none", stem))
        for r in parsed:
            yield dict(r, suite="reason", id="%s/%s" % (r["source"], r["case_id"]), passed=r["state"] == "PASSED")
