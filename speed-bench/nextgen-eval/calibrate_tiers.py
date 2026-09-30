"""Calibrate the long-context tiers in real tokens (stdlib only; CPU, no GPU).

python3 speed-bench/nextgen-eval/calibrate_tiers.py --model GGUF [--targets 480000,960000]

The needle prompt is the frozen haystack's first N characters plus the question, and the harness
used to size N as tokens x 3.03, measured on the head of ds4.c. Deeper text tokenizes at about
3.5-3.7 characters per token, so the deep tiers came out short (the "480K" prompt was 415K tokens).
This tool finds N for each target with `ds4 --dump-tokens` (the chat template, like the server) and
prints the TIERS entries to paste into eval_suites.py.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import eval_suites  # noqa: E402
import fetch_data  # noqa: E402


def count_dump(text):
    """ds4 --dump-tokens prints the token ids as a JSON array on its first line."""
    return len(json.loads(text.splitlines()[0]))


def search(count_tokens, target, lo, hi, tol):
    """Bisection on the character count: returns chars whose token count is within tol of target."""
    if not (count_tokens(lo) <= target <= count_tokens(hi)):
        raise ValueError("target %d outside [%d, %d] tokens" % (target, count_tokens(lo), count_tokens(hi)))
    while True:
        mid = (lo + hi) // 2
        n = count_tokens(mid)
        if abs(n - target) <= tol or hi - lo <= 1:
            return mid
        if n < target:
            lo = mid
        else:
            hi = mid


def ds4_counter(model, source):
    def count(chars):
        doc = "Here is a C source file.\n\n" + eval_suites.haystack(source, chars) + "\n\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(doc + eval_suites.NEEDLE_Q)
        r = subprocess.run([str(ROOT / "ds4"), "-m", str(model), "--nothink", "--dump-tokens",
                            "--prompt-file", f.name], capture_output=True, text=True, check=True)
        pathlib.Path(f.name).unlink()
        n = count_dump(r.stdout)
        print("  %9d chars -> %7d tokens" % (chars, n), flush=True)
        return n
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="480000,960000")
    args = ap.parse_args()
    source = fetch_data.DATA / eval_suites.HAYSTACK
    count = ds4_counter(args.model, source)
    size = len(source.read_text(errors="replace"))
    for target in (int(t) for t in args.targets.split(",")):
        chars = search(count, target, lo=int(target * 3.0), hi=min(size - 1000, int(target * 4.2)),
                       tol=max(500, target // 1000))
        print('    ("%dk", %d, %d),' % (target // 1000, target, chars))


if __name__ == "__main__":
    main()
