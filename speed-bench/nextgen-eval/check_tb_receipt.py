"""Check a PROD code+vi run against the think-budget receipt (same prompts, same PROD model).

python3 speed-bench/nextgen-eval/check_tb_receipt.py RUN_DIR/rows.jsonl
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "speed-bench/think-budget/runs/budget-0.json"


def main():
    rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    receipt = json.loads(RECEIPT.read_text())
    he_new = [r["passed"] for r in rows if r["suite"] == "code" and r["id"].startswith("HE/")]
    he_old = [r["passed"] for r in receipt if r["kind"] == "humaneval"]
    vi_new = [r["think_tokens"] for r in rows if r["suite"] == "vi_speed" and r["think_tokens"] is not None]
    vi_old = [r["think_tokens"] for r in receipt if r["kind"] == "vi" and r["think_tokens"] is not None]
    med_new, med_old = statistics.median(vi_new), statistics.median(vi_old)
    ok_pass = he_new == he_old
    ok_think = abs(med_new - med_old) <= 0.05 * med_old
    print("humaneval passes new=%d/%d receipt=%d/%d identical=%s" % (
        sum(he_new), len(he_new), sum(he_old), len(he_old), ok_pass))
    print("vi thinking median new=%s receipt=%s within5%%=%s" % (med_new, med_old, ok_think))
    sys.exit(0 if ok_pass and ok_think else 1)


if __name__ == "__main__":
    main()
