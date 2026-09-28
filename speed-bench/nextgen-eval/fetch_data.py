"""Download the public evaluation datasets into $NEXTGEN_EVAL_DATA (stdlib only).

python3 speed-bench/nextgen-eval/fetch_data.py

Subsets are deterministic:
- MBPP+: first 50 by numeric task_id;
- IFEval: first 60 by key among prompts whose every instruction id is supported;
- harmful / harmless: first 50 rows of each test split;
- mcp_tools.json: the gateway's MCP tool catalog, frozen (run this while the gateway stack is up);
- haystack.c: this checkout's ds4.c, frozen as the long-context document.
"""
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import graders  # noqa: E402
import ifeval_checks  # noqa: E402

DATA = pathlib.Path(os.environ.get(
    "NEXTGEN_EVAL_DATA", str(pathlib.Path.home() / "orca/workspaces/ds4-metal-data/evals/nextgen")))
ROWS_API = "https://datasets-server.huggingface.co/rows"
IFEVAL_URL = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"


def _get(url, attempts=5):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # network hiccup or HF rate limit: back off and retry
            if attempt == attempts - 1:
                raise
            print("  retry %d: %s" % (attempt + 1, e), flush=True)
            time.sleep(10 * (attempt + 1))


def fetch_rows(dataset, split, config="default", limit=None):
    """Rows of one split via the datasets-server API, 100 per page; stop early at `limit`."""
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": 100})
        page = json.loads(_get(ROWS_API + "?" + query))
        rows += [r["row"] for r in page["rows"]]
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"] or (limit and len(rows) >= limit):
            return rows[:limit] if limit else rows


def _task_number(task_id):
    return int(re.sub(r"\D", "", str(task_id)))


def select_mbpp(rows, n=50):
    out = []
    for r in sorted(rows, key=lambda r: _task_number(r["task_id"])):
        out.append({"task_id": str(_task_number(r["task_id"])), "prompt": r["prompt"], "code": r["code"],
                    "test_imports": graders.as_list(r.get("test_imports")),
                    "test_list": graders.as_list(r["test_list"])})
    return out[:n]


def select_ifeval(rows, n=60):
    ok = [r for r in rows if set(r["instruction_id_list"]) <= ifeval_checks.SUPPORTED]
    return sorted(ok, key=lambda r: r["key"])[:n]


def select_first(rows, n):
    return [{"text": r["text"]} for r in rows[:n]]


def snapshot_haystack(root=ROOT):
    """ds4.c as the long-context haystack, frozen so baseline and candidate read the same document."""
    return (pathlib.Path(root) / "ds4.c").read_bytes()


def fetch_mcp_catalog(client=None):
    """The gateway's MCP tool catalog (tools/list response). ds4-server has no /mcp, so the toolcall
    suite grades every arm against this frozen copy. Needs the gateway stack up."""
    if client is None:
        evals = str(graders.GATEWAY_REPO / "evals")
        if evals not in sys.path:
            sys.path.append(evals)
        import harness
        client = harness.Client()
    response, _ = client.post("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, timeout=30)
    if not ((response or {}).get("result") or {}).get("tools"):
        raise SystemExit("gateway /mcp tools/list returned no tools; is the gateway stack up?")
    return response


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    catalog = fetch_mcp_catalog()
    ifeval_rows = [json.loads(line) for line in _get(IFEVAL_URL).decode().splitlines() if line.strip()]
    sets = {
        "mbpp.jsonl": select_mbpp(fetch_rows("evalplus/mbppplus", "test")),
        "ifeval.jsonl": select_ifeval(ifeval_rows),
        "harmful.jsonl": select_first(fetch_rows("mlabonne/harmful_behaviors", "test", limit=50), 50),
        "harmless.jsonl": select_first(fetch_rows("mlabonne/harmless_alpaca", "test", limit=50), 50),
    }
    manifest = {}
    for name, rows in sets.items():
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        (DATA / name).write_bytes(blob)
        manifest[name] = {"rows": len(rows), "sha256": hashlib.sha256(blob).hexdigest()}
        print("%-16s %3d rows" % (name, len(rows)))
    blob = snapshot_haystack()
    (DATA / "haystack.c").write_bytes(blob)
    manifest["haystack.c"] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    print("%-16s %d bytes" % ("haystack.c", len(blob)))
    blob = (json.dumps(catalog, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
    (DATA / "mcp_tools.json").write_bytes(blob)
    tools = len(catalog["result"]["tools"])
    manifest["mcp_tools.json"] = {"tools": tools, "sha256": hashlib.sha256(blob).hexdigest()}
    print("%-16s %3d tools" % ("mcp_tools.json", tools))
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
