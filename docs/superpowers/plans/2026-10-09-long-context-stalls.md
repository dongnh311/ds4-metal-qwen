# Long-context stalls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two outcomes:
- Re-sent, retried and compaction requests on Ornith continue from the live session instead of
  re-prefilling 100K+ tokens.
- Re-stage, Tier L and ds4 slot stops never run while a real user session is live.

**Architecture:**
- **ds4:** the session keeps one in-memory copy of Ornith's fixed-size state (GDN states and conv
  histories, MTP carry, position, logits) at the prompt end of the latest request.
  `ds4_session_rewind` restores it, and a new server reuse tier selects it when a request's text
  extends that prompt end.
- **Gateway:** a guard script decides "quiet" from `/status` and the analytics store. A
  `restage.sh` wrapper and the Tier L/A gate use it.

**Tech Stack:**
- ds4-metal: C, Metal. Model-free tests run with `./ds4_test --server`. The model-backed test is a
  standalone Metal binary.
- AI-Gateway-MLX: Python 3 and bash; hermetic suites run via `tests/run-success-gate.py`.

**Spec:** `docs/superpowers/specs/2026-10-09-long-context-stalls-design.md` (ds4-metal, branch `feature/rewind-point`).

## Global Constraints

- **Worktrees:**
  - ds4: `~/orca/workspaces/ds4-metal/rewind-point`, branch `feature/rewind-point` (from develop `8acc1083`).
  - Gateway: `~/Documents/GitHub/AI-Gateway-MLX/.claude/worktrees/quiet-window`, branch
    `dongnh311/quiet-window` (from develop `b27f01cd`).
- **ds4 push:** only to `origin` (dongnh311/ds4-metal-qwen). Never push to `antirez` or `upstream`.
- **ds4 PROD:** built only by `deploy-ai-gateway.sh` from a `prod/<feature>-YYYYMMDD` branch
  contained in develop. Never edit the PROD checkout `~/.local/share/ai-gateway/ds4-metal`.
- **Memory:** one model process at a time on this 64 GB box. Any model-backed test, smoke or bench
  needs the gateway's ds4 slot stopped (SIGTERM, never SIGKILL). Stop it only after
  `scripts/quiet-window.py --wait-s` says quiet.
- **Gateway commits:** `git add <own paths>` only, never `-A` (repo rule #12). Never delete
  `.git/index.lock`. Tier L after every re-stage (rule #11), using the recorded
  `ALLOW_LIVE_PRODUCTION_TESTS=1` exception.
- **Privacy:** ds4 KV files and server traces hold prompt text. Never print prompt content into
  reports or commits.
- **Defaults stay off:** `--rewind-point-min-tokens` defaults to 0. Production changes only when
  the registry rows add the flag.
- **Commits:** end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01SxJu7BBYHNvVnivN8NTMM5`.

## Review Focus

1. **Rewind after a disk load:** a session restored from a KV file must not keep a stale point.
   The rewind tier must miss, not produce wrong state (Task 5 test `stale after load`).
2. **Re-send twice in a row:** the second re-send of the same prompt must hit the same point again,
   because restore is a copy, not a swap (Task 5 test `second rewind`, Task 7 bench `resend2`).
3. **Different conversation on the same slot:** the probe must not select the point when the
   request text does not byte-extend it, even if the token count is larger (Task 6 tier test
   `alien text`).
4. **Gateway down during re-stage:** `quiet-window.py` must report quiet, so a restore is never
   blocked (Task 1 case `gateway down`).
5. **Tier L's own traffic:** `ai-journey-suite`, `ai-gateway-eval` and `is_test` rows must not keep
   the box "busy" forever (Task 1 cases `test UA ignored`, `is_test ignored`).

---

## Part B — quiet-window guard (AI-Gateway-MLX)

### Task 1: `scripts/quiet-window.py` and its hermetic suite

**Files:**
- Create: `scripts/quiet-window.py`
- Create: `tests/test-suite-quiet-window.py`
- Modify: `tests/run-success-gate.py` (register `H.quiet_window` next to `H.grade_arm`)

**Interfaces:**
- Produces: the CLI `python3 scripts/quiet-window.py [--quiet-s 600] [--wait-s 0] [--poll-s 15]`.
  - Exit codes: 0 quiet, 3 busy.
  - Output: one line on stdout, starting with `quiet:`, `busy:`, or `quiet check BYPASSED`.
  - Env: `AI_GW_QUIET_STATUS_URL` (default `http://127.0.0.1:8090/status`),
    `AI_GW_QUIET_DB` (default `~/.local/share/ai-gateway/analytics.sqlite3`),
    `AI_GW_IGNORE_QUIET=1`.

- [ ] **Step 1: Create the gateway worktree**

```bash
cd ~/Documents/GitHub/AI-Gateway-MLX && git status --short | grep -v '^??' ; \
git worktree add -q .claude/worktrees/quiet-window -b dongnh311/quiet-window develop && \
cd .claude/worktrees/quiet-window && git log --oneline -1
```
Expected: the HEAD line is `b27f01cd …`.

- [ ] **Step 2: Write the failing suite** `tests/test-suite-quiet-window.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hermetic guard: re-stage and Tier L/A must wait for a QUIET box, not a 20 s lull.

THE INCIDENT (2026-10-09 15:38). The pre-re-stage check was "/status shows nothing active for
10 polls of 2 s". It passed inside a 22 s tool gap of a live Claude Code session; the re-stage
cut the in-flight request (client_disconnect) and Tier L then took the ds4 slot, so the user's
next 122K-token turn re-prefilled from the 32K checkpoint (116 s).

WHAT THIS LOCKS (scripts/quiet-window.py): busy while anything non-canary is in flight, and for
--quiet-s after the newest REAL request (is_test 0, is_canary 0, not one of our suites' UAs);
a gateway that does not answer is quiet (a restore must never wait); --wait-s polls; the
bypass is explicit and loud.

Hermetic: a temp sqlite store and a stub /status on an ephemeral loopback port.
"""
import http.server
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "quiet-window.py")
RESULTS = []
STATUS = {"active": []}


def case(name, ok, note=""):
    RESULTS.append(bool(ok))
    print("%s %-62s %s" % ("✅" if ok else "❌", name, note))


class _Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(STATUS).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_db(path, rows):
    c = sqlite3.connect(path)
    c.execute("DROP TABLE IF EXISTS events")
    c.execute("CREATE TABLE events (ts REAL, ts_start REAL, client_door TEXT, conv_id TEXT, "
              "client_ua TEXT, is_test INTEGER, is_canary INTEGER)")
    now = time.time()
    for age, door, ua, is_test, is_canary in rows:
        c.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                  (now - age, now - age - 1, door, "conv-%s" % door, ua, is_test, is_canary))
    c.commit()
    c.close()


def run(db, url, *args, env_extra=None):
    env = dict(os.environ, AI_GW_QUIET_DB=db, AI_GW_QUIET_STATUS_URL=url)
    env.pop("AI_GW_IGNORE_QUIET", None)
    env.update(env_extra or {})
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT] + list(args), env=env, capture_output=True,
                       text=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).strip(), time.time() - t0


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="quietwin-")
    db = os.path.join(tmp, "a.sqlite3")
    port = free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Status)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/status" % port

    make_db(db, [])
    rc, out, _ = run(db, url)
    case("empty store, nothing active -> quiet", rc == 0 and out.startswith("quiet:"), out)

    make_db(db, [(30, "anthropic", None, 0, 0)])
    rc, out, _ = run(db, url)
    case("real request 30 s ago -> busy", rc == 3 and "last real request" in out, out)

    make_db(db, [(700, "anthropic", None, 0, 0)])
    rc, out, _ = run(db, url)
    case("real request 700 s ago (window 600) -> quiet", rc == 0, out)

    make_db(db, [(10, "chat-local", "ai-journey-suite/1.0", 0, 0),
                 (10, "chat-local", "ai-gateway-eval/1.0", 0, 0),
                 (10, "chat-local", "ai-test-suite/1.0", 0, 0)])
    rc, out, _ = run(db, url)
    case("test UA ignored", rc == 0, out)

    make_db(db, [(10, "anthropic", None, 1, 0)])
    rc, out, _ = run(db, url)
    case("is_test ignored", rc == 0, out)

    make_db(db, [(10, "chat-local", "curl/8.7.1", 0, 1)])
    rc, out, _ = run(db, url)
    case("is_canary ignored", rc == 0, out)

    make_db(db, [(10, "anthropic", None, None, None)])
    rc, out, _ = run(db, url)
    case("NULL is_test/is_canary count as real", rc == 3, out)

    make_db(db, [])
    STATUS["active"] = [{"canary": False, "idle_s": 3, "model": "ornith"}]
    rc, out, _ = run(db, url)
    case("non-canary request in flight -> busy", rc == 3 and "in flight" in out, out)
    STATUS["active"] = [{"canary": True, "idle_s": 1}]
    rc, out, _ = run(db, url)
    case("canary in flight only -> quiet", rc == 0, out)
    STATUS["active"] = []

    rc, out, _ = run(db, "http://127.0.0.1:%d/status" % free_port())
    case("gateway down -> quiet (restores never wait)", rc == 0 and "does not answer" in out, out)

    make_db(db, [(1, "anthropic", None, 0, 0)])
    rc, out, dt = run(db, url, "--quiet-s", "4", "--wait-s", "20", "--poll-s", "1")
    case("--wait-s polls until the window passes", rc == 0 and 2 <= dt < 15, "%.1fs %s" % (dt, out))

    make_db(db, [(1, "anthropic", None, 0, 0)])
    rc, out, dt = run(db, url, "--quiet-s", "600", "--wait-s", "2", "--poll-s", "1")
    case("--wait-s timeout -> busy", rc == 3 and dt < 15, "%.1fs %s" % (dt, out))

    rc, out, _ = run(db, url, env_extra={"AI_GW_IGNORE_QUIET": "1"})
    case("bypass is explicit and loud", rc == 0 and "BYPASSED" in out, out)

    srv.shutdown()
    ok = sum(RESULTS)
    print("\n%d/%d passed" % (ok, len(RESULTS)))
    sys.exit(0 if ok == len(RESULTS) else 1)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python3 tests/test-suite-quiet-window.py`
Expected: FAIL. Every case is ❌ (the script does not exist: `can't open file …quiet-window.py`).

- [ ] **Step 4: Write `scripts/quiet-window.py`**

```python
#!/usr/bin/env python3
"""Is the box quiet enough to re-stage, run Tier L/A, or stop a ds4 slot?

Quiet = nothing non-canary in flight on the gateway's /status AND no REAL request in the
analytics store for the last --quiet-s seconds. Real = is_test 0, is_canary 0 (NULL counts as
0), and a user agent that is not one of our own suites. A gateway that does not answer is
quiet: there is nothing to protect, and a restore must never wait on it.

THE INCIDENT (2026-10-09 15:38): a 20 s /status-idle check passed inside a 22 s tool gap of a
live Claude Code session. The re-stage cut the in-flight request and Tier L took the ds4 slot;
the next 122K-token turn re-prefilled from the 32K checkpoint (116 s).

Exit 0 quiet, 3 busy (after --wait-s). AI_GW_IGNORE_QUIET=1 bypasses (prints BYPASSED, exit 0).
Env AI_GW_QUIET_STATUS_URL / AI_GW_QUIET_DB point it elsewhere (tests).
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request

TEST_UA_PREFIXES = ("ai-test-suite", "ai-journey-suite", "ai-gateway-eval")
STATUS_URL = os.environ.get("AI_GW_QUIET_STATUS_URL", "http://127.0.0.1:8090/status")
DB = os.environ.get("AI_GW_QUIET_DB",
                    os.path.expanduser("~/.local/share/ai-gateway/analytics.sqlite3"))


def in_flight(url):
    """Non-canary requests in flight, or None when the gateway does not answer."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            status = json.loads(r.read())
    except Exception:
        return None
    return [a for a in (status.get("active") or []) if not a.get("canary")]


def last_real(db, now, quiet_s):
    """(age_s, door, conv) of the newest real request inside the window, or None."""
    if not os.path.exists(db):
        return None
    c = sqlite3.connect("file:%s?mode=ro" % db, uri=True, timeout=5)
    try:
        rows = c.execute(
            "SELECT MAX(ts, COALESCE(ts_start, 0)) AS t, client_door, conv_id, client_ua "
            "FROM events WHERE COALESCE(is_test, 0) = 0 AND COALESCE(is_canary, 0) = 0 "
            "AND MAX(ts, COALESCE(ts_start, 0)) >= ? ORDER BY t DESC LIMIT 500",
            (now - quiet_s,)).fetchall()
    finally:
        c.close()
    for t, door, conv, ua in rows:
        if (ua or "").startswith(TEST_UA_PREFIXES):
            continue
        return now - t, door, conv
    return None


def check(quiet_s):
    active = in_flight(STATUS_URL)
    if active is None:
        return True, "quiet: gateway at %s does not answer (nothing to protect)" % STATUS_URL
    if active:
        a = active[0]
        return False, "busy: %d request(s) in flight (%s, idle %s s)" % (
            len(active), a.get("door") or a.get("model") or "?", a.get("idle_s", "?"))
    real = last_real(DB, time.time(), quiet_s)
    if real:
        age, door, conv = real
        return False, "busy: last real request %.0f s ago (%s, conv %s); window %d s" % (
            age, door or "?", (conv or "?")[:8], quiet_s)
    return True, "quiet: no real request in the last %d s, nothing in flight" % quiet_s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet-s", type=int, default=600)
    ap.add_argument("--wait-s", type=int, default=0)
    ap.add_argument("--poll-s", type=int, default=15)
    args = ap.parse_args()
    if os.environ.get("AI_GW_IGNORE_QUIET") == "1":
        print("quiet check BYPASSED (AI_GW_IGNORE_QUIET=1)")
        return 0
    deadline = time.time() + max(0, args.wait_s)
    while True:
        ok, line = check(args.quiet_s)
        if ok or time.time() + args.poll_s > deadline:
            print(line, flush=True)
            return 0 if ok else 3
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the suite to verify it passes**

Run: `chmod +x scripts/quiet-window.py && python3 tests/test-suite-quiet-window.py`
Expected: PASS with `13/13 passed`.

- [ ] **Step 6: Register `H.quiet_window`** in `tests/run-success-gate.py`, directly after the
  `("H.grade_arm", …)` tuple, in the same style:

```python
        ("H.quiet_window", "Stub suite: re-stage and Tier L/A wait for a QUIET box. A 20 s "
         "/status lull passed inside a 22 s tool gap of a live session on 2026-10-09; the re-stage "
         "cut the request and Tier L took the ds4 slot (next turn: 116 s re-prefill). Busy while "
         "anything non-canary is in flight and for 600 s after a real request; test UAs, is_test "
         "and canaries never count; a gateway that does not answer is quiet",
         lambda: [py, "tests/test-suite-quiet-window.py"], []),
```

- [ ] **Step 7: Run Tier H** (worktree; `H.scratchpad_tracked` and journey F81 are known
  environment-only reds in a linked worktree)

Run: `python3 tests/run-success-gate.py --tier H > /tmp/qw-H.log 2>&1; tail -5 /tmp/qw-H.log; grep -E "❌" /tmp/qw-H.log | head`
Expected: `H.quiet_window` ✅, and no ❌ other than those two environment-only reds.

- [ ] **Step 8: Commit**

```bash
git add scripts/quiet-window.py tests/test-suite-quiet-window.py tests/run-success-gate.py
git commit -m "quiet-window: re-stage and Tier L/A wait for minutes without real traffic, not a 20 s lull"
```

### Task 2: `scripts/restage.sh` and the Tier L/A quiet check

**Files:**
- Create: `scripts/restage.sh`
- Modify: `tests/run-success-gate.py` (in `main()`, after the `L-safety` refusal block; add `_quiet_window_error()` next to `_live_target_safety_error()`)
- Modify: `tests/test-suite-quiet-window.py` (two cases for the gate hook)
- Modify: `CLAUDE.md` (rule #11 text and the Tier table)

**Interfaces:**
- Consumes: `scripts/quiet-window.py` (Task 1).
- Produces:
  - `scripts/restage.sh`, the supported re-stage path. Exit 3 means refused because busy;
    otherwise it returns the exit code of `install-launchd.sh`.
  - `_quiet_window_error() -> str | None` in `tests/run-success-gate.py`.

- [ ] **Step 1: Add two failing cases** to `tests/test-suite-quiet-window.py`, before `srv.shutdown()`:

```python
    import importlib.util
    spec = importlib.util.spec_from_file_location("gate", os.path.join(REPO, "tests", "run-success-gate.py"))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    os.environ.update(AI_GW_QUIET_DB=db, AI_GW_QUIET_STATUS_URL=url, AI_GW_QUIET_WAIT_S="0")
    os.environ.pop("AI_GW_IGNORE_QUIET", None)
    make_db(db, [(5, "anthropic", None, 0, 0)])
    err = getattr(gate, "_quiet_window_error", lambda: "missing")()
    case("gate: Tier L/A refused while a real session is live",
         isinstance(err, str) and "busy" in err, str(err))
    make_db(db, [])
    err = getattr(gate, "_quiet_window_error", lambda: "missing")()
    case("gate: quiet box lets Tier L/A run", err is None, str(err))
```

Run: `python3 tests/test-suite-quiet-window.py`
Expected: FAIL on the second new case: `gate: quiet box lets Tier L/A run` is ❌ with `missing`.

- [ ] **Step 2: Implement the gate hook.** In `tests/run-success-gate.py`, next to
  `_live_target_safety_error()`:

```python
def _quiet_window_error():
    """Tier L/A drive production's ds4 slot and GPU: refuse while a real user session is live.

    2026-10-09 15:38: Tier L right after a re-stage took the ds4 slot from a live 170K-token
    session; its next turn re-prefilled from the 32K checkpoint (116 s). scripts/quiet-window.py
    decides (exit 0 quiet, 3 busy); AI_GW_QUIET_WAIT_S lets it wait, AI_GW_IGNORE_QUIET=1 bypasses.
    """
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "quiet-window.py")
    r = subprocess.run([sys.executable, script, "--wait-s",
                        os.environ.get("AI_GW_QUIET_WAIT_S", "0")],
                       capture_output=True, text=True)
    line = (r.stdout or r.stderr).strip()
    if r.returncode == 0:
        print("  " + line)
        return None
    return ("Tier L/A refused: %s. Retry when the box is quiet, set AI_GW_QUIET_WAIT_S=<s> to "
            "wait, or AI_GW_IGNORE_QUIET=1 to override." % line)
```

In `main()`, directly after the existing `L-safety` refusal block, which does
`_write_summary("L-safety", …)` and then exits, add the following. Mirror that block's exit
statement exactly.

```python
    if {"L", "A"} & set(args.tier.upper().replace(" ", "").split(",")):
        _quiet_error = _quiet_window_error()
        if _quiet_error:
            print("❌ " + _quiet_error, file=sys.stderr)
            _write_summary("L-quiet", [], 0, 2, requested_tier=args.tier.upper(), omitted=[])
            sys.exit(2)
```

If `subprocess` is not imported at the top, add `import subprocess`.

- [ ] **Step 3: Run the suite**

Run: `python3 tests/test-suite-quiet-window.py`
Expected: PASS with `15/15 passed`.

- [ ] **Step 4: Write `scripts/restage.sh`**

```bash
#!/usr/bin/env bash
#
# restage.sh — the supported way to re-stage the gateway (repo CLAUDE.md rule #11).
#
#   1. wait for a quiet window (scripts/quiet-window.py): nothing non-canary in flight and no
#      real request for 600 s — a 20 s lull is a Claude Code tool gap, not an idle box
#      (2026-10-09 15:38: a re-stage cut a live request; the next turn re-prefilled 116 s)
#   2. back up the staged gateway (last 3 kept)
#   3. run install-launchd.sh
#   4. print the live runtime_code_hash
#
# AI_GW_QUIET_WAIT_S (default 1800) bounds the wait; AI_GW_IGNORE_QUIET=1 bypasses it.
# install-launchd.sh itself is unchanged: restore-known-good.sh and the power scripts call it
# directly and are never held up by this check. Tier L comes after, as its own command.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGED="$HOME/.local/ai-gateway/gateway"
BKDIR="${AI_GW_RESTAGE_BACKUP_DIR:-$HOME/.local/ai-gateway/restage-backups}"

if ! python3 "$REPO/scripts/quiet-window.py" --wait-s "${AI_GW_QUIET_WAIT_S:-1800}"; then
    echo "restage: refused — a user session is live (AI_GW_IGNORE_QUIET=1 to override)" >&2
    exit 3
fi
mkdir -p "$BKDIR"
BK="$BKDIR/gateway-$(date +%Y%m%d-%H%M%S)"
if [ -d "$STAGED" ]; then
    cp -Rp "$STAGED" "$BK" && echo "restage: backup $BK"
    ls -1dt "$BKDIR"/gateway-* 2>/dev/null | tail -n +4 | while read -r old; do rm -rf "$old"; done
fi
bash "$REPO/scripts/install-launchd.sh"
rc=$?
sleep 3
hash=$(curl -s -m 10 http://127.0.0.1:8090/status |
       python3 -c 'import json,sys; print(json.load(sys.stdin).get("runtime_code_hash","")[:8])' 2>/dev/null)
echo "restage: install-launchd.sh exit $rc; live runtime_code_hash ${hash:-unknown}"
exit $rc
```

Run: `chmod +x scripts/restage.sh && bash -n scripts/restage.sh && AI_GW_QUIET_STATUS_URL=http://127.0.0.1:9/status AI_GW_QUIET_DB=/nonexistent python3 scripts/quiet-window.py`
Expected: no syntax error, then `quiet: gateway at http://127.0.0.1:9/status does not answer …`.

- [ ] **Step 5: Docs.** In `CLAUDE.md`:
  - In rule #11, after the first sentence, add: `Re-stage with **scripts/restage.sh** (it waits for
    scripts/quiet-window.py: nothing in flight and no real request for 600 s), and Tier L/A
    refuse on their own while a real session is live (AI_GW_QUIET_WAIT_S to wait,
    AI_GW_IGNORE_QUIET=1 to override).`
  - In rule #12's sentence "Only the deploy owner runs `install-launchd.sh`…", add `(via
    scripts/restage.sh)` after `install-launchd.sh`.
  - Add the line `python3 tests/test-suite-quiet-window.py` to the hermetic suite list, if the
    file keeps one.

- [ ] **Step 6: Tier H, then commit**

Run: `python3 tests/run-success-gate.py --tier H > /tmp/qw-H2.log 2>&1; tail -3 /tmp/qw-H2.log`
Expected: green, apart from the two worktree environment-only reds.

```bash
git add scripts/restage.sh tests/run-success-gate.py tests/test-suite-quiet-window.py CLAUDE.md
git commit -m "restage.sh + Tier L/A quiet check: no re-stage or live tier while a real session is live"
```

### Task 3: Ship Part B

**Files:** none new. This task reviews, merges, re-stages and records the receipt.

- [ ] **Step 0: Review Part B before it ships.** Use `superpowers:requesting-code-review`: one
  reviewer subagent with the model left empty, over `b27f01cd..dongnh311/quiet-window`, given the
  spec §4 and this plan's Review Focus items 4-5. Fix Critical and Important findings RED→GREEN
  and ledger the Minor ones.

- [ ] **Step 1: Merge to develop** in the main checkout. Before merging, check that no foreign
  dirty tracked files are present (rule #12).

```bash
cd ~/Documents/GitHub/AI-Gateway-MLX && git status --short | grep -v '^??'
git merge --no-ff -q dongnh311/quiet-window -m "Merge dongnh311/quiet-window: quiet-window guard for re-stage and Tier L/A"
```
Expected: the status line shows nothing tracked as dirty, and the merge lands.

- [ ] **Step 2: Tier H on develop**

Run: `python3 tests/run-success-gate.py --tier H > /tmp/qw-dev-H.log 2>&1; tail -3 /tmp/qw-dev-H.log`
Expected: all green, because the main checkout has no worktree reds.

- [ ] **Step 3: Re-stage through the new path.** This is the first real use of the guard, so it
  waits by itself.

Run: `AI_GW_QUIET_WAIT_S=7200 bash scripts/restage.sh > /tmp/qw-restage.log 2>&1; tail -3 /tmp/qw-restage.log`
Expected: a `quiet:` line, a backup line, then `install-launchd.sh exit 0; live runtime_code_hash <new>`.

- [ ] **Step 4: Tier L** (rule #11, recorded exception)

Run: `ALLOW_LIVE_PRODUCTION_TESTS=1 python3 tests/run-success-gate.py --tier L > /tmp/qw-L.log 2>&1; grep -E "steps passed|quiet:|❌" /tmp/qw-L.log | head`
Expected:
- the run prints a `quiet:` line first;
- the result is `16/17`, the same as the last baseline, where the only red is the vision step
  (red since 10-03).

- [ ] **Step 5: Receipt.** Write `reports/quiet-window-2026-10-09/DEPLOY.md`. It records the code
  hash, Tier H and Tier L results, and the `quiet:` line. Copy the Tier L summary lines into
  `evidence/`. Then commit and push `origin develop`.

```bash
git add reports/quiet-window-2026-10-09
git commit -m "report: quiet-window guard deployed (re-stage via restage.sh, Tier L under the quiet check)"
git push -q origin develop
```

---

## Part A — the rewind point (ds4-metal, worktree `~/orca/workspaces/ds4-metal/rewind-point`)

### Task 4: Merge the prompt-end branch

**Files:** `ds4_kvstore.c` (the conflict), plus everything bd55adcf, 429831ae and a84c5688 touch.

**Interfaces:**
- Produces, on this branch:
  - `prompt_end_text_len(const request *r)`;
  - `prompt_tail_token_start(const ds4_tokens *prompt, const ds4_tokens *tail)`;
  - the cold-prefill split block in `generate_job_inner`;
  - `--kv-cache-prompt-end-min-tokens`.

- [ ] **Step 1: Merge**

```bash
cd ~/orca/workspaces/ds4-metal/rewind-point && git merge --no-ff feature/prompt-end-checkpoint
```
Expected: `CONFLICT (content): Merge conflict in ds4_kvstore.c`. It is one hunk: develop's
`ds4_kvstore_reuse_existing` against the branch's `ds4_kvstore_incoming_context`.

- [ ] **Step 2: Resolve by keeping both functions** (develop's first, then the branch's), and
  remove the markers. Then:

```bash
grep -c '^<<<<<<<\|^>>>>>>>' ds4_kvstore.c; git add ds4_kvstore.c && git commit -q --no-edit
```
Expected: `0`, and the merge commit is created.

- [ ] **Step 3: Build and run the model-free tests**

Run: `make -j8 ds4 ds4-server ds4_test > /tmp/rp-build.log 2>&1 && ./ds4_test --server > /tmp/rp-t4.log 2>&1; echo rc=$?; tail -3 /tmp/rp-t4.log`
Expected: `rc=0`. The output includes `test_kv_cache_prompt_end_option` and
`test_prompt_end_split_plan` passing.

### Task 5: Engine rewind point (Ornith)

**Files:**
- Modify: `ds4.h` (declarations next to `ds4_session_rewind`, plus a test helper next to
  `ds4_session_set_test_images`)
- Modify: `ds4.c`. Touch points:
  - `ds4_qwen4_gpu_graph`: fields after `snap_lin_hist`;
  - `qwen4_graph_free`: free them;
  - `struct ds4_session`: fields after `checkpoint_valid`;
  - `ds4_session_free`;
  - `ds4_session_invalidate`;
  - `ds4_session_load_payload`;
  - the qwen35 branch of `ds4_session_sync`;
  - `ds4_session_rewind`;
  - the new API functions after `ds4_session_rewind`.
- Modify: `ds4_qwen35moe.inc` (two helpers after `qwen35_graph_set_h_last`)
- Create: `tests/test_qwen35_rewind_point.c`
- Modify: `Makefile` (target `test-qwen35-rewind-point`, and the binary in `clean`)

**Interfaces:**
- Produces:
  - `bool ds4_session_mark_rewind_point(ds4_session *s);`
  - `int ds4_session_rewind_point_pos(const ds4_session *s);`
  - `ds4_session_rewind(s, pos)` restores the point when
    `pos == ds4_session_rewind_point_pos(s)` and keeps `checkpoint_valid`.
  - The test-only `void ds4_session_set_test_rewind_point(ds4_session *s, int pos);` (-1 clears).

- [ ] **Step 1: Write the failing model-backed test** `tests/test_qwen35_rewind_point.c`

```c
/* Model-backed oracle for the Ornith rewind point (ds4_session_mark_rewind_point).
 *
 * Run with the production ds4 slot stopped (one model process on 64 GB):
 *   DS4_TEST_MODEL=/path/to/Ornith.gguf make test-qwen35-rewind-point
 *   DS4_TEST_MTP=1 also opens the embedded MTP block (ds4-server --mtp).
 *
 * A = about 6K prompt tokens, B = 300 greedy reply tokens, C = 64 new tokens.
 * Session X: sync A, mark, decode B, rewind to |A|, eval C one token at a time.
 * Session Y: sync A, eval C one token at a time (never rewound).
 * Every row of C must match: same argmax, max |dlogit| <= 1e-2.  X is then
 * rewound to |A| a second time and must match again (restore is a copy).
 * Finally a payload load must drop the point. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CTX 16384
#define A_TOKENS 6000
#define B_TOKENS 300
#define C_TOKENS 64

static void fail(const char *what) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) fail("open tests/long_context_story_prompt.txt");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)n + 1);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) fail("read prompt file");
    p[n] = 0;
    fclose(f);
    return p;
}

static void eval_compare(ds4_session *x, ds4_session *y, const ds4_tokens *c, int vocab,
                         float *lx, float *ly, const char *pass) {
    char err[256];
    float worst = 0.0f;
    for (int i = 0; i < c->len; i++) {
        if (ds4_session_eval(x, c->v[i], err, sizeof(err))) fail(err);
        if (ds4_session_eval(y, c->v[i], err, sizeof(err))) fail(err);
        if (ds4_session_copy_logits(x, lx, vocab) != vocab ||
            ds4_session_copy_logits(y, ly, vocab) != vocab) fail("copy logits");
        if (ds4_session_argmax(x) != ds4_session_argmax(y)) {
            fprintf(stderr, "FAIL: %s argmax differs at row %d\n", pass, i);
            exit(1);
        }
        for (int k = 0; k < vocab; k++) {
            float d = fabsf(lx[k] - ly[k]);
            if (d > worst) worst = d;
        }
    }
    printf("%s: %d rows, max |dlogit| %.6g\n", pass, c->len, worst);
    if (!(worst <= 1e-2f)) fail("logits differ after rewind");
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) fail("set DS4_TEST_MODEL to the Ornith GGUF");
    const char *mtp_env = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = CTX,
        .glm_mtp = mtp_env && mtp_env[0] == '1',
        .mtp_draft_tokens = 1,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(e);

    char *text = read_file("tests/long_context_story_prompt.txt");
    ds4_tokens all = {0}, a = {0}, c = {0};
    ds4_tokenize_text(e, text, &all);
    if (all.len < A_TOKENS + C_TOKENS) fail("prompt file too short");
    for (int i = 0; i < A_TOKENS; i++) ds4_tokens_push(&a, all.v[i]);
    for (int i = 0; i < C_TOKENS; i++) ds4_tokens_push(&c, all.v[all.len - C_TOKENS + i]);

    ds4_session *x = NULL, *y = NULL;
    char err[256];
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync(x, &a, err, sizeof(err))) fail(err);
    if (ds4_session_sync(y, &a, err, sizeof(err))) fail(err);

    if (!ds4_session_mark_rewind_point(x)) fail("mark rewind point");
    if (ds4_session_rewind_point_pos(x) != A_TOKENS) fail("rewind point position");

    for (int i = 0; i < B_TOKENS; i++)
        if (ds4_session_eval(x, ds4_session_argmax(x), err, sizeof(err))) fail(err);

    ds4_session_rewind(x, A_TOKENS);
    if (!ds4_session_checkpoint_valid(x) || ds4_session_pos(x) != A_TOKENS)
        fail("rewind to the point did not keep a valid checkpoint");
    float *lx = malloc((size_t)vocab * sizeof(float)), *ly = malloc((size_t)vocab * sizeof(float));
    eval_compare(x, y, &c, vocab, lx, ly, "first rewind");

    /* second rewind: X back to |A|, Y re-synced to A from scratch */
    ds4_session_rewind(x, A_TOKENS);
    if (!ds4_session_checkpoint_valid(x)) fail("second rewind lost the checkpoint");
    ds4_session_invalidate(y);
    if (ds4_session_sync(y, &a, err, sizeof(err))) fail(err);
    eval_compare(x, y, &c, vocab, lx, ly, "second rewind");

    /* rewinding below the point drops it */
    ds4_session_rewind(x, A_TOKENS - 1);
    if (ds4_session_rewind_point_pos(x) != -1) fail("rewind below the point kept it");

    /* stale after load: a payload loaded into a session drops its point */
    ds4_session_invalidate(x);
    if (ds4_session_sync(x, &a, err, sizeof(err))) fail(err);
    if (!ds4_session_mark_rewind_point(x)) fail("re-mark");
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(y, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_load_snapshot(x, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_rewind_point_pos(x) != -1) fail("stale after load: point survived a payload load");
    ds4_session_snapshot_free(&snap);

    printf("PASS: rewind point (mtp=%d)\n", opt.glm_mtp ? 1 : 0);
    free(lx); free(ly); free(text);
    ds4_tokens_free(&all); ds4_tokens_free(&a); ds4_tokens_free(&c);
    ds4_session_free(x); ds4_session_free(y);
    ds4_engine_close(e);
    return 0;
}
```

Makefile additions, next to `test-metal-session-batch`:

```make
tests/test_qwen35_rewind_point.o: tests/test_qwen35_rewind_point.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_rewind_point.c

tests/test_qwen35_rewind_point: tests/test_qwen35_rewind_point.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ tests/test_qwen35_rewind_point.o $(CORE_OBJS) $(LDLIBS)

test-qwen35-rewind-point: tests/test_qwen35_rewind_point
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_qwen35_rewind_point
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP=1 ./tests/test_qwen35_rewind_point
```
Copy the link line's exact flag variables from the `tests/test_metal_session_batch:` rule
(Makefile line ~118), and add `tests/test_qwen35_rewind_point` to the `clean` list.

- [ ] **Step 2: Build it to verify it fails**

Run: `make tests/test_qwen35_rewind_point 2>&1 | grep -E "error|undefined" | head -3`
Expected: FAIL with `undefined symbol` or `implicit declaration` for
`ds4_session_mark_rewind_point`.

- [ ] **Step 3: Declarations** in `ds4.h`, after `void ds4_session_rewind(ds4_session *s, int pos);`:

```c
/* Rewind point (Ornith/qwen35 only): keep the session's fixed-size state at its
 * current position (GDN states and conv histories, MTP carry, position, logits)
 * so that ds4_session_rewind(s, ds4_session_rewind_point_pos(s)) restores it
 * exactly; the append-only rows before it are reused as they are.  One point
 * per session; a new mark replaces it; a reset, a payload load, an invalidate
 * or a rewind below it drops it.  Returns false (and keeps no point) for other
 * models, distributed sessions or on allocation failure. */
bool ds4_session_mark_rewind_point(ds4_session *s);
/* Position of the valid rewind point, or -1. */
int ds4_session_rewind_point_pos(const ds4_session *s);
```

And next to `ds4_session_set_test_images`:

```c
/* Test helper: pretend a rewind point exists at pos on a test checkpoint
 * session (pos < 0 clears it). */
void ds4_session_set_test_rewind_point(ds4_session *s, int pos);
```

- [ ] **Step 4: Fields.**

In `ds4_qwen4_gpu_graph`, after `ds4_gpu_tensor *snap_lin_hist[DS4_MAX_LAYER];`:

```c
    /* Rewind point (qwen35): GDN states and conv histories at the point,
     * allocated on the first mark (ds4_session_mark_rewind_point). */
    ds4_gpu_tensor *rw_lin_state[DS4_MAX_LAYER];
    ds4_gpu_tensor *rw_lin_hist[DS4_MAX_LAYER];
    bool rw_ready;
```

In `qwen4_graph_free`, inside the loop that frees `layer_lin_state[il]`:

```c
        ds4_gpu_tensor_free(g->rw_lin_state[il]);
        ds4_gpu_tensor_free(g->rw_lin_hist[il]);
```

In `struct ds4_session`, after `bool checkpoint_valid;`:

```c
    /* Rewind point: see ds4_session_mark_rewind_point(). */
    bool rewind_valid;
    int rewind_pos;
    uint32_t rewind_mtp_pos;
    float *rewind_logits;   /* DS4_N_VOCAB, the logits at rewind_pos */
    float *rewind_h_last;   /* DS4_N_EMBD, MTP carry h_{rewind_pos-1}; NULL without MTP */
```

In `ds4_session_free`, before the struct itself is freed: `free(s->rewind_logits); free(s->rewind_h_last);`.

- [ ] **Step 5: GPU helpers** in `ds4_qwen35moe.inc`, after `qwen35_graph_set_h_last`:

```c
/* Rewind point buffers: one GDN state + conv history per recurrent layer,
 * allocated on the first mark and kept for the graph's life. */
static bool qwen35_graph_rw_ensure(ds4_qwen4_gpu_graph *g) {
    if (g->rw_ready) return true;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->layer_lin_state[il]) continue;
        g->rw_lin_state[il] = qwen4_graph_alloc_f32(v_dim * DS4_N_LIN_HEAD_DIM);
        g->rw_lin_hist[il] = qwen4_graph_alloc_f32((uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
        if (!g->rw_lin_state[il] || !g->rw_lin_hist[il]) {
            for (uint32_t j = 0; j <= il; j++) {
                ds4_gpu_tensor_free(g->rw_lin_state[j]);
                ds4_gpu_tensor_free(g->rw_lin_hist[j]);
                g->rw_lin_state[j] = g->rw_lin_hist[j] = NULL;
            }
            return false;
        }
    }
    g->rw_ready = true;
    return true;
}

/* Copy every GDN state and conv history into (save) or out of the rewind
 * point buffers.  A copy, not a swap: the point survives a restore, so the
 * same prompt can be re-sent any number of times. */
static bool qwen35_graph_rw_copy(ds4_qwen4_gpu_graph *g, bool save) {
    if (!g->rw_ready) return false;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t state_bytes = v_dim * DS4_N_LIN_HEAD_DIM * sizeof(float);
    const uint64_t hist_bytes = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim * sizeof(float);
    if (!glm_graph_begin_commands_if_needed()) return false;
    bool ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (!g->rw_lin_state[il]) continue;
        ds4_gpu_tensor *live_s = g->layer_lin_state[il], *rw_s = g->rw_lin_state[il];
        ds4_gpu_tensor *live_h = g->layer_lin_hist[il], *rw_h = g->rw_lin_hist[il];
        ok = ds4_gpu_tensor_copy(save ? rw_s : live_s, 0, save ? live_s : rw_s, 0, state_bytes) != 0 &&
             ds4_gpu_tensor_copy(save ? rw_h : live_h, 0, save ? live_h : rw_h, 0, hist_bytes) != 0;
    }
    if (!ds4_gpu_end_commands()) ok = false;
    return ok;
}
```

- [ ] **Step 6: Session API and rewind.** In `ds4.c`, directly after the body of
  `ds4_session_rewind`:

```c
bool ds4_session_mark_rewind_point(ds4_session *s) {
    if (!s) return false;
    s->rewind_valid = false;
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_QWEN4_METAL)
    if (s->distributed || !s->checkpoint_valid || !ds4_session_is_qwen35(s)) return false;
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    if (g->pos != (uint32_t)s->checkpoint.len || !qwen35_graph_rw_ensure(g)) return false;
    if (!s->rewind_logits) s->rewind_logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (g->mtp_h && !s->rewind_h_last) s->rewind_h_last = malloc((size_t)DS4_N_EMBD * sizeof(float));
    if (!s->rewind_logits || (g->mtp_h && !s->rewind_h_last)) return false;
    if (g->mtp_h && !qwen35_graph_h_last(g, s->rewind_h_last)) return false;
    if (!qwen35_graph_rw_copy(g, true)) return false;
    memcpy(s->rewind_logits, s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
    s->rewind_mtp_pos = g->mtp_pos;
    s->rewind_pos = s->checkpoint.len;
    s->rewind_valid = true;
    return true;
#else
    return false;
#endif
}

int ds4_session_rewind_point_pos(const ds4_session *s) {
    return s && s->rewind_valid ? s->rewind_pos : -1;
}

void ds4_session_set_test_rewind_point(ds4_session *s, int pos) {
    if (!s) return;
    s->rewind_valid = pos >= 0;
    s->rewind_pos = pos >= 0 ? pos : 0;
}
```

Inside `ds4_session_rewind`, in the `#ifdef DS4_HAS_QWEN4_METAL` block and **before** the
existing `if (s->checkpoint_valid && ds4_session_is_qwen35(s)) {` one-token-back block, add the
following. Then make that existing block's condition start with `!state_ok &&`.

```c
    if (s->rewind_valid && pos < s->rewind_pos) s->rewind_valid = false;
    if (s->checkpoint_valid && ds4_session_is_qwen35(s) && s->rewind_valid &&
        pos == s->rewind_pos) {
        /* Rewind point: rows [0, pos) of the attention, MTP and mrope caches
         * are append-only and untouched since the mark; restore the rest. */
        ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
        bool ok = qwen35_graph_rw_copy(g, false);
        g->pos = (uint32_t)pos;
        if (ok && g->mtp_h) ok = qwen35_graph_set_h_last(g, s->rewind_h_last);
        if (ok) {
            g->mtp_pos = s->rewind_mtp_pos;
            g->snap_valid = false;
            g->snap_after_first = false;
            memcpy(s->logits, s->rewind_logits, (size_t)DS4_N_VOCAB * sizeof(float));
            state_ok = true;
        }
    }
```

- [ ] **Step 7: Invalidation points.** Add `s->rewind_valid = false;` at each of these:
  - (a) the top of `ds4_session_invalidate`, after `if (!s) return;`;
  - (b) the top of `ds4_session_load_payload`, after its argument checks (every model type);
  - (c) the qwen35 branch of `ds4_session_sync`, inside the `else { qwen35_graph_reset(g); …`
    replay branch.

- [ ] **Step 8: Build and run the model-backed test.** Do this only when the box is quiet, and
  stop the gateway's ds4 slot first.

```bash
python3 ~/Documents/GitHub/AI-Gateway-MLX/scripts/quiet-window.py --wait-s 7200 && \
pid=$(pgrep -f "ds4-metal/ds4-server"); [ -n "$pid" ] && kill -TERM $pid; \
while pgrep -f "ds4-metal/ds4-server" >/dev/null; do sleep 2; done; \
make -j8 tests/test_qwen35_rewind_point > /tmp/rp-b5.log 2>&1 && \
DS4_TEST_MODEL=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf \
  make test-qwen35-rewind-point > /tmp/rp-t5.log 2>&1; echo rc=$?; grep -E "rewind:|PASS|FAIL" /tmp/rp-t5.log
```
Expected: `rc=0`. Output has `first rewind: 64 rows, max |dlogit| 0` (or below 1e-2),
`second rewind: …`, and `PASS: rewind point (mtp=0)` and `(mtp=1)`. The gateway relaunches its
slot on the next request.

- [ ] **Step 9: Model-free regression**

Run: `make -j8 ds4_test && ./ds4_test --server > /tmp/rp-t5s.log 2>&1; echo rc=$?`
Expected: `rc=0`.

- [ ] **Step 10: Commit**

```bash
git add ds4.h ds4.c ds4_qwen35moe.inc tests/test_qwen35_rewind_point.c Makefile
git commit -m "ornith: in-memory rewind point -- restore GDN state, MTP carry and logits at a marked prompt end"
```

### Task 6: Server — mark at the prompt end, `REUSE_REWIND_POINT`, option

**Files:**
- Modify: `ds4_server.c`. Touch points:
  - `server_config`, `struct server`, `parse_options`, the copy into `s`, and a startup log;
  - `struct server_slot` and slot free;
  - `prompt_end_text_len`, which gains the `allow_ornith` parameter (2 call sites);
  - the pure helper `rewind_point_cut_plan` and `rewind_point_cut`;
  - the mark in `generate_job_inner` after the prompt-end block;
  - the `slot_reuse_kind` enum, `slot_probe_reuse_locked` and its materialization `switch`;
  - tests in the `DS4_SERVER_TEST` section.
- Modify: `ds4_help.c` (one line), `docs/SERVER.md` (one paragraph)

**Interfaces:**
- Consumes: `ds4_session_mark_rewind_point`, `ds4_session_rewind_point_pos`, the rewind
  behavior of `ds4_session_rewind`, and `ds4_session_set_test_rewind_point` (Task 5).
- Produces:
  - `--rewind-point-min-tokens N`;
  - `REUSE_REWIND_POINT`;
  - the log lines `ds4-server: rewind point remembered pos=%d text=%zu` and
    `ds4-server: rewind point hit pos=%d live=%d prompt=%d`.

- [ ] **Step 1: Write failing model-free tests** in the `DS4_SERVER_TEST` section, next to
  `test_kv_cache_prompt_end_option`, and register them where that test is called:

```c
static void test_rewind_point_option(void) {
    char *d[] = {"ds4-server"};
    TEST_ASSERT(parse_options(1, d).rewind_point_min_tokens == 0);
    char *c[] = {"ds4-server", "--rewind-point-min-tokens", "16384"};
    TEST_ASSERT(parse_options(3, c).rewind_point_min_tokens == 16384);
}

static void test_rewind_point_cut_plan(void) {
    ds4_tokens prompt = {0}, tail = {0};
    for (int i = 0; i < 100; i++) ds4_tokens_push(&prompt, i + 1);
    for (int i = 96; i < 100; i++) ds4_tokens_push(&tail, i + 1);   /* generation prompt */
    TEST_ASSERT(rewind_point_cut_plan(50, &prompt, &tail, 10) == 96);
    TEST_ASSERT(rewind_point_cut_plan(50, &prompt, &tail, 96) == 0);   /* nothing new before the cut */
    TEST_ASSERT(rewind_point_cut_plan(97, &prompt, &tail, 10) == 0);   /* below the minimum */
    TEST_ASSERT(rewind_point_cut_plan(0, &prompt, &tail, 10) == 0);    /* option off */
    tail.v[0] = 999;
    TEST_ASSERT(rewind_point_cut_plan(50, &prompt, &tail, 10) == 0);   /* not the token tail */
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&tail);
}

static void test_rewind_point_reuse_tier(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    int live[300];
    for (int i = 0; i < 300; i++) live[i] = 7000 + i;
    server_slot slot = {0};
    slot.session = ds4_session_new_test_checkpoint(live, 300);
    slot.rewind_point.valid = true;
    slot.rewind_point.live_tokens = 200;
    slot.rewind_point.text = (char *)"<sys>turn1<tool>";
    slot.rewind_point.text_len = strlen(slot.rewind_point.text);
    ds4_session_set_test_rewind_point(slot.session, 200);
    job j = {0};
    ds4_tokens_push(&j.req.prompt, 1);                       /* token prefix diverges at once */
    j.req.kind = REQ_CHAT;
    j.req.prompt_text = (char *)"<sys>turn1<tool><user>again";
    slot_reuse pr = slot_probe_reuse_locked(&s, &slot, &j.req);
    TEST_ASSERT(pr.kind == REUSE_REWIND_POINT);
    TEST_ASSERT(pr.reuse_tokens == 200);
    TEST_ASSERT(pr.suffix_off == slot.rewind_point.text_len);
    /* alien text: same length class, different bytes */
    j.req.prompt_text = (char *)"<sys>turnX<tool><user>again";
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind == REUSE_NONE);
    /* equal text: nothing to add past the point */
    j.req.prompt_text = (char *)"<sys>turn1<tool>";
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind == REUSE_NONE);
    /* stale: the engine dropped its point */
    j.req.prompt_text = (char *)"<sys>turn1<tool><user>again";
    ds4_session_set_test_rewind_point(slot.session, -1);
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind == REUSE_NONE);
    /* engine point at another position */
    ds4_session_set_test_rewind_point(slot.session, 150);
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind == REUSE_NONE);
    /* images never take the tier */
    ds4_session_set_test_rewind_point(slot.session, 200);
    j.req.image_count = 1;
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &j.req).kind != REUSE_REWIND_POINT);
    ds4_session_free_test_checkpoint(slot.session);
    ds4_tokens_free(&j.req.prompt);
    pthread_mutex_destroy(&s.tool_mu);
}
```

If the vision gate (`ds4_session_vision_fingerprint_prefix_matches`) already rejects the
image case before the tier, that assertion still holds: the tier is not chosen.

- [ ] **Step 2: Run to verify they fail**

Run: `make ds4_test 2>&1 | grep -E "error" | head -3`
Expected: FAIL to compile. The errors are `no member named 'rewind_point_min_tokens'`, an
undeclared `rewind_point_cut_plan`, and an undeclared `REUSE_REWIND_POINT`.

- [ ] **Step 3: Option plumbing.**
  - Add `int rewind_point_min_tokens;` to `server_config` and to `struct server`.
  - In `parse_options`, next to `--kv-cache-prompt-end-min-tokens`:

```c
        } else if (!strcmp(arg, "--rewind-point-min-tokens")) {
            c.rewind_point_min_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
```

  - In main, next to `s.think_budget = cfg.think_budget;`, add
    `s.rewind_point_min_tokens = cfg.rewind_point_min_tokens;`.
  - After the prompt-end startup log:

```c
    if (s.rewind_point_min_tokens > 0)
        server_log(DS4_LOG_KVCACHE, "ds4-server: rewind point at the prompt end of chat prompts >= %d tokens",
                   s.rewind_point_min_tokens);
```

  - In `ds4_help.c`, next to the prompt-end line:
    `  --rewind-point-min-tokens N  keep an in-memory rewind point at the prompt end (Ornith; 0 = off)`.

- [ ] **Step 4: Slot state.** In `struct server_slot`, after `thinking_live`:

```c
    /* The prompt end of the latest request, where the engine holds a rewind
     * point: that request's rendered text up to the end of its last message
     * and the live token count there.  The engine is the authority on whether
     * the point still exists (ds4_session_rewind_point_pos). Guarded by tool_mu. */
    struct {
        bool valid;
        int live_tokens;
        char *text;
        size_t text_len;
    } rewind_point;
```

Add these helpers after `thinking_live_remember`:

```c
static void rewind_point_clear(server *s, server_slot *slot) {
    if (!s || !slot) return;
    pthread_mutex_lock(&s->tool_mu);
    free(slot->rewind_point.text);
    memset(&slot->rewind_point, 0, sizeof(slot->rewind_point));
    pthread_mutex_unlock(&s->tool_mu);
}

static void rewind_point_remember(server *s, server_slot *slot, const char *text,
                                  size_t text_len, int live_tokens) {
    char *copy = xstrndup(text, text_len);
    pthread_mutex_lock(&s->tool_mu);
    free(slot->rewind_point.text);
    slot->rewind_point.text = copy;
    slot->rewind_point.text_len = text_len;
    slot->rewind_point.live_tokens = live_tokens;
    slot->rewind_point.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
    server_log(DS4_LOG_KVCACHE, "ds4-server: rewind point remembered pos=%d text=%zu",
               live_tokens, text_len);
}
```

At the slot free path, where `visible_live_free(&slot->thinking_live)` is called, also call
`free(slot->rewind_point.text);`.

- [ ] **Step 5: Eligibility and cut.** Change `prompt_end_text_len(const request *r)` to
  `prompt_end_text_len(const request *r, bool allow_ornith)`. Its Ornith check becomes
  `(!allow_ornith && server_qwen_is_ornith())`, and every existing call passes `false`. Then add
  this after `kv_cache_prompt_end_split`:

```c
/* The in-memory rewind point's cut: the same clean boundary as the disk
 * prompt-end checkpoint (the generation prompt is the token tail), with new
 * tokens before it and at least min_tokens of prompt.  0 = no cut.  Pure. */
static int rewind_point_cut_plan(int min_tokens, const ds4_tokens *prompt,
                                 const ds4_tokens *tail, int cached) {
    if (min_tokens <= 0 || !prompt || prompt->len < min_tokens) return 0;
    const int cut = prompt_tail_token_start(prompt, tail);
    if (cut <= 0 || cut <= cached || cut < min_tokens) return 0;
    return cut;
}

/* Ornith admitted: the key is byte-exact, so a template that does not replay
 * a past turn as these bytes simply never matches. */
static int rewind_point_cut(server *s, const request *r, const ds4_tokens *prompt,
                            int cached, size_t *text_len) {
    *text_len = 0;
    if (!s || s->rewind_point_min_tokens <= 0 || !prompt ||
        prompt->len < s->rewind_point_min_tokens) return 0;
    const size_t n = prompt_end_text_len(r, true);
    if (!n) return 0;
    ds4_tokens tail = {0};
    ds4_tokenize_rendered_chat(s->engine, r->prompt_text + n, &tail);
    const int cut = rewind_point_cut_plan(s->rewind_point_min_tokens, prompt, &tail, cached);
    ds4_tokens_free(&tail);
    if (cut > 0) *text_len = n;
    return cut;
}
```

- [ ] **Step 6: Mark at the cut.** In `generate_job_inner`, directly after
  `prompt_end_split_free(&prompt_end);`:

```c
    /* The rewind point sits on the same boundary: prefill stops there (if the
     * disk prompt-end split did not already), the engine keeps the state, and
     * only then is the generation prompt evaluated.  A re-sent, retried or
     * compaction request that starts with these bytes rewinds here. */
    size_t rewind_text_len = 0;
    const int rewind_cut = (prompt_sync_rc == 0 && !multimodal) ?
        rewind_point_cut(s, &j->req, prompt_for_sync, cached, &rewind_text_len) : 0;
    if (rewind_cut > 0) {
        if (ds4_session_pos(slot->session) != rewind_cut) {
            ds4_tokens prefix = {0};
            tokens_copy_prefix(&prefix, prompt_for_sync, rewind_cut);
            prompt_sync_rc = server_session_sync(s, slot, &prefix, err, sizeof(err));
            ds4_tokens_free(&prefix);
        }
        if (prompt_sync_rc == 0) {
            if (ds4_session_mark_rewind_point(slot->session))
                rewind_point_remember(s, slot, j->req.prompt_text, rewind_text_len, rewind_cut);
            else
                rewind_point_clear(s, slot);
        }
    }
```

- [ ] **Step 7: The tier.** Append `REUSE_REWIND_POINT,` to `slot_reuse_kind`, after
  `REUSE_MEMORY_TEXT`. In `slot_probe_reuse_locked`, after the `REUSE_MEMORY_TEXT` block and
  before the final `return pr;`:

```c
    if (ptext && req->kind == REQ_CHAT && req->image_count == 0 &&
        slot->rewind_point.valid && slot->rewind_point.text &&
        slot->rewind_point.text_len < plen &&
        slot->rewind_point.live_tokens <= live_pos &&
        ds4_session_rewind_point_pos(slot->session) == slot->rewind_point.live_tokens &&
        byte_prefix_match(ptext, plen, slot->rewind_point.text, slot->rewind_point.text_len))
    {
        pr.kind = REUSE_REWIND_POINT;
        pr.reuse_tokens = slot->rewind_point.live_tokens;
        pr.suffix_off = slot->rewind_point.text_len;
        return pr;
    }
```

In the materialization `switch`, before `case REUSE_NONE:`:

```c
    case REUSE_REWIND_POINT: {
        const int live_before = ds4_session_pos(slot->session);
        pthread_mutex_lock(&s->inference_mu);
        ds4_session_rewind(slot->session, reuse.reuse_tokens);
        const bool rewound = ds4_session_checkpoint_valid(slot->session) &&
            ds4_session_pos(slot->session) == reuse.reuse_tokens &&
            ds4_session_rewind_point_pos(slot->session) == reuse.reuse_tokens;
        pthread_mutex_unlock(&s->inference_mu);
        if (!rewound || !build_live_prompt_suffix(s, slot, &j->req,
                j->req.prompt_text + reuse.suffix_off, &effective_prompt)) {
            cached = 0;
            cache_source = "none";
            break;
        }
        live_materialized = true;
        cache_source = "rewind-point";
        prompt_for_sync = &effective_prompt;
        server_log(DS4_LOG_KVCACHE, "ds4-server: rewind point hit pos=%d live=%d prompt=%d",
                   reuse.reuse_tokens, live_before, effective_prompt.len);
        break;
    }
```

Add a `"rewind-point"` entry wherever `cache_source` strings or `slot_reuse_kind` values are
enumerated for logging or tracing (grep `"memory-text"` to find them).

- [ ] **Step 8: Run the model-free tests**

Run: `make -j8 ds4 ds4-server ds4_test > /tmp/rp-b6.log 2>&1 && ./ds4_test --server > /tmp/rp-t6.log 2>&1; echo rc=$?; grep -c PASS /tmp/rp-t6.log`
Expected: `rc=0`, including the three new tests.

- [ ] **Step 9: Docs.** In `docs/SERVER.md`, after the `--kv-cache-prompt-end-min-tokens` paragraph:

```markdown
`--rewind-point-min-tokens N` (default 0, off) keeps an in-memory rewind point on Ornith. For
every OpenAI-chat request with tools whose prompt is at least N tokens, prefill stops at the end
of the last message (the same boundary as the prompt-end checkpoint), and the session keeps its
fixed-size state there: GDN states and conv histories, the MTP carry, the position and the
logits. A later request whose text starts with that prompt up to the cut is served by restoring
the point and prefilling only what follows. This covers a client re-sending or retrying the last
request, and a Claude Code compaction, which drops the last reply and adds a user message. It
writes nothing to disk and needs no disk cache. The log shows `rewind point remembered` and
`rewind point hit`. Other models ignore the flag.
```

- [ ] **Step 10: Commit**

```bash
git add ds4_server.c ds4_help.c docs/SERVER.md
git commit -m "server: --rewind-point-min-tokens -- mark the prompt end, REUSE_REWIND_POINT serves re-sends and compactions"
```

### Task 7: End-to-end benchmark (baseline PROD binary vs this branch)

**Files:**
- Create: `speed-bench/rewind-point/bench_resend.py`
- Create: `speed-bench/rewind-point/REPORT.md`

**Interfaces:**
- Consumes: `ds4-server --rewind-point-min-tokens` (Task 6) and the registry row of
  `…-ornith-…-512K`.

- [ ] **Step 1: Write `speed-bench/rewind-point/bench_resend.py`**

```python
#!/usr/bin/env python3
"""Re-send / compaction TTFT on Ornith: baseline binary vs rewind-point binary.

Starts ONE ds4-server (the registry's Ornith-512K argv, scratch port, scratch KV dir), builds a
Claude-Code-shaped OpenAI chat (system + tools + ~N tokens of tool turns from
tests/long_context_story_prompt.txt), then times, with temperature 0 and streaming:
  r1       the full conversation (cold prefill)
  resend   r1 again (the client dropped r1's reply)              <- the 16:13 shape
  resend2  r1 a third time                                        <- a point must survive restore
  compact  r1 + a user "summarize" message (reply dropped)        <- the 16:46 shape
TTFT = first streamed delta. The rewind arm must also reproduce r1's reply byte for byte.
Prompt content is synthetic (a public story file); nothing private is read or printed.
argv: --bin DIR --label NAME --tokens 64000 [--rewind 16384]
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, tempfile, time, urllib.request

REG = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
HERE = os.path.dirname(os.path.abspath(__file__))
STORY = os.path.join(HERE, "..", "..", "tests", "long_context_story_prompt.txt")
PORT = 18199
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}}]


def ornith_argv(bin_dir, kv_dir, rewind):
    reg = json.load(open(REG))
    row = next(v for k, v in reg.items() if isinstance(v, dict) and "ornith" in k.lower()
               and k.endswith("512K"))
    rt = row["runtimes"]["ds4"]
    argv = list(rt.get("argv") or rt.get("command"))
    argv[0] = os.path.join(bin_dir, "ds4-server")
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--port", "--kv-disk-dir", "--rewind-point-min-tokens"):
            i += 2
            continue
        out.append(a)
        i += 1
    out += ["--port", str(PORT), "--kv-disk-dir", kv_dir]
    if rewind:
        out += ["--rewind-point-min-tokens", str(rewind)]
    return out


def conversation(tokens):
    text = open(STORY).read()
    chunks = [text[i:i + 6000] for i in range(0, len(text), 6000)]
    msgs = [{"role": "system", "content": "You are a careful reading assistant. Answer briefly."},
            {"role": "user", "content": "Read the story files one by one, then tell me who the narrator is."}]
    approx, k = 0, 0
    while approx < tokens * 4:   # ~4 bytes per token
        cid = "call_%d" % k
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function", "function": {
                         "name": "read_file", "arguments": json.dumps({"path": "story/part%d.txt" % k})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": chunks[k % len(chunks)]})
        approx += len(chunks[k % len(chunks)])
        k += 1
    return msgs


def ask(msgs, max_tokens=48):
    body = json.dumps({"model": "x", "messages": msgs, "tools": TOOLS, "temperature": 0,
                       "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])["choices"][0].get("delta") or {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "") + \
                    json.dumps(d.get("tool_calls") or "")
            if ttft is None and piece.strip('"'):
                ttft = time.time() - t0
            out.append(piece)
    return ttft, time.time() - t0, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=64000)
    ap.add_argument("--rewind", type=int, default=0)
    a = ap.parse_args()
    kv = tempfile.mkdtemp(prefix="rp-kv-")
    log = open(os.path.join(HERE, "server-%s.log" % a.label), "w")
    srv = subprocess.Popen(ornith_argv(a.bin, kv, a.rewind), stdout=log, stderr=subprocess.STDOUT,
                           cwd=a.bin)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=2)
                break
            except Exception:
                time.sleep(1)
        msgs = conversation(a.tokens)
        res = {}
        res["r1"] = ask(msgs)
        res["resend"] = ask(msgs)
        res["resend2"] = ask(msgs)
        res["compact"] = ask(msgs + [{"role": "user", "content": "Summarize the conversation so far in two sentences."}])
        for k, (ttft, dur, _) in res.items():
            print("%-8s %-8s ttft %7.1f s  total %7.1f s" % (a.label, k, ttft or -1, dur))
        same = res["resend"][2] == res["r1"][2] and res["resend2"][2] == res["r1"][2]
        print("%-8s reply identical across r1/resend/resend2: %s" % (a.label, same))
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(120)
        except subprocess.TimeoutExpired:
            print("server did not exit in 120 s; left running (never SIGKILL Metal)", file=sys.stderr)
        shutil.rmtree(kv, ignore_errors=True)
    log.close()
    hits = sum(1 for l in open(os.path.join(HERE, "server-%s.log" % a.label)) if "rewind point hit" in l)
    print("%-8s rewind point hits: %d" % (a.label, hits))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run both arms in one quiet window, with the slot stopped.** The baseline uses the
  PROD tree's binary, so nothing is rebuilt.

```bash
python3 ~/Documents/GitHub/AI-Gateway-MLX/scripts/quiet-window.py --wait-s 7200 && \
pid=$(pgrep -f "ai-gateway/ds4-metal/ds4-server"); [ -n "$pid" ] && kill -TERM $pid; \
while pgrep -f "ds4-server" >/dev/null; do sleep 2; done; cd speed-bench/rewind-point && \
python3 bench_resend.py --bin ~/.local/share/ai-gateway/ds4-metal --label base --tokens 64000 | tee base.txt && \
python3 bench_resend.py --bin ~/orca/workspaces/ds4-metal/rewind-point --label rewind --tokens 64000 --rewind 16384 | tee rewind.txt
```
Expected:
- **base:** `resend` and `compact` TTFT are tens of seconds or more (a full re-prefill), and there
  are 0 rewind hits.
- **rewind:** `resend` and `resend2` TTFT are under 5 s, `compact` TTFT is a few seconds, there
  are at least 3 rewind point hits, and the reply is identical (`True`).

If the rewind arm's reply is not identical, stop and debug with
superpowers:systematic-debugging. Do not ship.

- [ ] **Step 3: Write `REPORT.md`.** It covers the two tables, the hardware and context, the
  conclusion, and the commands. Then commit.

```bash
git add speed-bench/rewind-point/bench_resend.py speed-bench/rewind-point/REPORT.md speed-bench/rewind-point/base.txt speed-bench/rewind-point/rewind.txt
git commit -m "bench: rewind point -- re-send and compaction TTFT, baseline vs branch"
```

Do not commit `server-*.log`. It holds only synthetic text, but it is noise.

### Task 8: Ship Part A

**Files:** registry (production config, backed up first), ds4 PROD via `deploy-ai-gateway.sh`.

- [ ] **Step 1: Whole-branch review** (executing-plans "Final Review", see below), before merging.

- [ ] **Step 2: Merge to develop and push origin**

```bash
cd ~/orca/workspaces/ds4-metal/develop && git pull -q --ff-only origin develop && \
git merge --no-ff -q feature/rewind-point -m "Merge feature/rewind-point: in-memory rewind point for Ornith re-sends and compactions (+ prompt-end checkpoint branch)" && \
make -j8 ds4_test && ./ds4_test --server > /tmp/rp-dev.log 2>&1; echo rc=$?; git push -q origin develop feature/rewind-point
```
Expected: `rc=0` and the push succeeds.

- [ ] **Step 3: Cut and install PROD** in a quiet window with the slot stopped.

```bash
cd ~/.local/share/ai-gateway/ds4-metal && \
python3 ~/Documents/GitHub/AI-Gateway-MLX/scripts/quiet-window.py --wait-s 7200 && \
pid=$(pgrep -f "ai-gateway/ds4-metal/ds4-server"); [ -n "$pid" ] && kill -TERM $pid; \
while pgrep -f "ds4-server" >/dev/null; do sleep 2; done; \
./deploy-ai-gateway.sh cut rewind-point && ./deploy-ai-gateway.sh install prod/rewind-point-$(date +%Y%m%d) && \
for m in $(python3 -c "import json,os;r=json.load(open(os.path.expanduser('~/.local/ai-gateway/runtime-registry.json')));print(' '.join(k for k,v in r.items() if isinstance(v,dict) and v.get('active_runtime')=='ds4'))"); do ./deploy-ai-gateway.sh smoke --model "$m" || echo "SMOKE FAIL $m"; done
```
Expected: `deploy: … now at prod/rewind-point-YYYYMMDD`, and every smoke prints `ok`, with no
`SMOKE FAIL`.

- [ ] **Step 4: Registry flag.** Back up the registry, then add `--rewind-point-min-tokens 16384`
  to both Ornith rows.

```bash
R=~/.local/ai-gateway/runtime-registry.json; cp -p $R $R.bak-rewind-$(date +%Y%m%d-%H%M%S) && python3 - <<'EOF'
import json, os
p = os.path.expanduser("~/.local/ai-gateway/runtime-registry.json")
r = json.load(open(p))
for k, v in r.items():
    if isinstance(v, dict) and "ornith" in k.lower() and v.get("active_runtime") == "ds4":
        rt = v["runtimes"]["ds4"]
        key = "argv" if "argv" in rt else "command"
        if "--rewind-point-min-tokens" not in rt[key]:
            rt[key] += ["--rewind-point-min-tokens", "16384"]
        print(k, "->", " ".join(rt[key][-2:]))
json.dump(r, open(p, "w"), indent=2)
EOF
```
Expected: two lines, each ending `--rewind-point-min-tokens 16384`. Check how the gateway reads
the registry: `grep -n "runtime-registry" ~/Documents/GitHub/AI-Gateway-MLX/gateway/*.py`. If it
caches the registry at boot, a quiet-window `scripts/restage.sh` is needed (then Tier L);
otherwise the next slot start picks the flag up.

- [ ] **Step 5: Production check.** After the next real Ornith request, confirm the slot runs with
  the flag (`ps -axo command | grep rewind-point-min-tokens`) and that the log has
  `rewind point remembered`. Over the next sessions, collect `rewind point hit` lines, and check
  that no big miss lands its `common` at a previous prompt end. Use
  `scratchpad/boundary_check.py`, re-created from the report if it is gone.

- [ ] **Step 6: Receipt.** Write `speed-bench/rewind-point/DEPLOY.md` (ds4 repo). It records the prod branch, the smoke
  results, the registry diff (only the flag), and the first production `rewind point hit` lines
  with their prompt sizes, never content. Then commit to develop and push origin.

### Task 9: Qwen3.8 prompt-end measurement (G4, gated)

- [ ] **Step 1: Measure** in a quiet window with the slot stopped. Use the Qwen3.8-512K registry
  argv on a scratch port, with `--kv-cache-prompt-end-min-tokens 32768` and a scratch KV dir.
  Send the same synthetic tool conversation at 64K and 128K (reuse `bench_resend.py` with the
  row filter changed to `qwen3.8` / `Flash-Next` and the flag swapped), then read
  `kv cache stored … reason=prompt-end size=… save=…`.
- [ ] **Step 2: Decide by the spec's gate.** If a store is ≤ 1.5 GB and ≤ 0.5 s at 128K, back up
  the registry, add `--kv-cache-prompt-end-min-tokens 65536` to both Qwen3.8 rows, and restart
  that slot when quiet. Otherwise leave it off. Record the numbers and the decision in
  `speed-bench/rewind-point/REPORT.md` (section "Qwen3.8"), then commit and push.

---

## Final Review (after Task 7, before Task 8 step 2)

- Run `review-package` over `8acc1083..feature/rewind-point` (ds4 only; Part B was reviewed in
  Task 3 step 0).
- Dispatch one reviewer with `superpowers:requesting-code-review`'s template. Leave the
  model empty, which means the session model, never Fable. Give each reviewer the spec, this plan,
  this Review Focus, and the ledger's `Ruling:` lines.
- Fix every Critical and Important finding with RED→GREEN. Ledger the Minor ones.
