#!/bin/bash
# Deploy ds4 to the local AI-Gateway from a prod/<feature>-YYYYMMDD branch.
# See docs/DEPLOY_AI_GATEWAY.md for the full procedure and the rules.
#
#   deploy-ai-gateway.sh cut FEATURE           create prod/FEATURE-YYYYMMDD from the remote develop
#   deploy-ai-gateway.sh install BRANCH        check out BRANCH in the PROD checkout and clean-build it
#   deploy-ai-gateway.sh smoke [--model KEY] [--bin DIR] [--out DIR] [--ref DIR]
#                                              run a registry row's ds4 command on a scratch port + KV dir
#                                              (--model is required when several rows run on ds4)
set -euo pipefail

PROD_DIR=${DS4_PROD_DIR:-$HOME/.local/share/ai-gateway/ds4-metal}
REGISTRY=${DS4_GATEWAY_REGISTRY:-$HOME/.local/ai-gateway/runtime-registry.json}
REPO_SLUG=dongnh311/ds4-metal-qwen
BINARIES="ds4 ds4-server ds4-bench ds4-eval ds4-agent"
BRANCH_RE='^prod/[a-z0-9][a-z0-9-]*-[0-9]{8}$'

die() { echo "deploy: $*" >&2; exit 1; }

# The remote that points at our repo; every other remote (Ivan's origin, antirez) is read-only for us.
repo_remote() {
    local r
    r=$(git remote -v | awk -v s="$REPO_SLUG" 'index($2, s) && $3 == "(push)" {print $1; exit}')
    [ -n "$r" ] || die "no remote points at $REPO_SLUG in $(pwd)"
    echo "$r"
}

ds4_running() { pgrep -fl "(^|/)ds4(-server|-agent|-bench|-eval)?( |$)" || true; }

cmd_cut() {
    local feature=${1:-}
    [[ "$feature" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "usage: cut FEATURE (lowercase, digits, dashes)"
    local r branch
    r=$(repo_remote)
    branch="prod/$feature-$(date +%Y%m%d)"
    git fetch -q "$r"
    git rev-parse -q --verify "refs/remotes/$r/$branch" >/dev/null && die "$branch already exists on $r"
    echo "deploy: $branch <- $r/develop $(git log -1 --format='%h %s' "$r/develop")"
    git push "$r" "refs/remotes/$r/develop:refs/heads/$branch"
    echo "deploy: next: $0 install $branch"
}

cmd_install() {
    local branch=${1:-}
    [[ "$branch" =~ $BRANCH_RE ]] || die "usage: install prod/<feature>-YYYYMMDD"
    cd "$PROD_DIR"
    local r running prev
    r=$(repo_remote)
    [ -z "$(git status --porcelain --untracked-files=no)" ] ||
        die "$PROD_DIR has local edits to tracked files; PROD is never edited in place (fix on develop, cut a new prod branch)"
    running=$(ds4_running)
    [ -z "$running" ] || die "ds4 is running, stop the gateway ds4 slot first:"$'\n'"$running"
    git fetch -q "$r"
    git rev-parse -q --verify "refs/remotes/$r/$branch" >/dev/null || die "$branch is not on $r (cut it first)"
    git merge-base --is-ancestor "$r/$branch" "$r/develop" ||
        die "$branch is not contained in develop; prod branches are cut from develop"
    if git rev-parse -q --verify "refs/heads/$branch" >/dev/null &&
       ! git merge-base --is-ancestor "$branch" "$r/$branch"; then
        die "local $branch has commits that are not on $r; prod branches only come from the remote"
    fi
    prev="$(git rev-parse --abbrev-ref HEAD)@$(git rev-parse --short HEAD)"
    git checkout -q -B "$branch" --track "$r/$branch"
    make clean >/dev/null
    make -j"$(sysctl -n hw.ncpu)" $BINARIES >/dev/null
    for b in $BINARIES; do [ -x "$b" ] || die "build did not produce $b"; done
    echo "$(date '+%Y-%m-%d %H:%M:%S') $prev -> $branch@$(git rev-parse --short HEAD)" >> .deploy-history
    echo "deploy: $PROD_DIR now at $branch $(git log -1 --format='%h %s')"
    echo "deploy: previous $prev (recorded in $PROD_DIR/.deploy-history)"
    echo "deploy: next: $0 smoke, then start the gateway ds4 slot"
}

cmd_smoke() {
    local model="" bin="" out="" ref=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model=$2; shift 2 ;;
            --bin) bin=$2; shift 2 ;;
            --out) out=$2; shift 2 ;;
            --ref) ref=$2; shift 2 ;;
            *) die "usage: smoke [--model KEY] [--bin DIR] [--out DIR] [--ref DIR]" ;;
        esac
    done
    local running
    running=$(ds4_running)
    [ -z "$running" ] || die "ds4 is running; one model process at a time on this machine:"$'\n'"$running"
    out=${out:-$(mktemp -d "${TMPDIR:-/tmp}/ds4-smoke.XXXXXX")}
    mkdir -p "$out"
    python3 - "$REGISTRY" "$bin" "$out" "$ref" "$model" <<'PY'
import json, os, shutil, subprocess, sys, time, urllib.request

registry, bin_dir, out, ref, want = sys.argv[1:6]
port = int(os.environ.get("DS4_SMOKE_PORT", "18297"))
stop_timeout = float(os.environ.get("DS4_SMOKE_STOP_TIMEOUT", "60"))
# Every model row runs its own ds4 process, so with several rows the one to test must be named.
rows = [(key, model["runtimes"]["ds4"]) for key, model in json.load(open(registry))["models"].items()
        if model.get("runtimes", {}).get("ds4", {}).get("enabled")]
if want:
    rows = [(key, rt) for key, rt in rows if key == want]
    if not rows:
        sys.exit("deploy: no enabled ds4 runtime for model %s in %s" % (want, registry))
elif len(rows) > 1:
    sys.exit("deploy: several enabled ds4 runtimes in %s, pick one with --model: %s"
             % (registry, ", ".join(key for key, _ in rows)))
elif not rows:
    sys.exit("deploy: no enabled ds4 runtime in " + registry)
key, entry = rows[0]
print("deploy: smoke model:", key)

cmd = list(entry["process_command"])
kv = os.path.join(out, "kv")
shutil.rmtree(kv, ignore_errors=True)
os.makedirs(kv)
for i, a in enumerate(cmd):
    if a == "--port":
        cmd[i + 1] = str(port)
    elif a == "--kv-disk-dir":
        cmd[i + 1] = kv
    elif bin_dir and os.path.basename(a) == "ds4-server":
        cmd[i] = os.path.join(os.path.abspath(bin_dir), "ds4-server")
cwd = os.path.abspath(bin_dir) if bin_dir else entry.get("process_cwd") or None
print("deploy: smoke:", " ".join(cmd))

log = open(os.path.join(out, "server.log"), "w")
srv = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{port}"
prompts = {
    "vi": "Giải thích ngắn gọn cách bộ nhớ đệm KV giúp mô hình ngôn ngữ sinh văn bản nhanh hơn.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests.",
}
weather_tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
failed = False


def chat(messages, max_tokens, tools=None):
    body = {"model": "ds4", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    if tools:
        body["tools"] = tools
    t0 = time.time()
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return o["choices"][0]["message"], o.get("usage", {}).get("completion_tokens", 0), time.time() - t0


def record(name, reply):
    """Save a reply and compare it with the reference: (status suffix, differs)."""
    open(os.path.join(out, name + ".txt"), "w").write(reply)
    if not ref:
        return "", False
    try:
        same = open(os.path.join(ref, name + ".txt")).read() == reply
    except OSError:
        return ", no ref file", False
    return (", same as ref", False) if same else (", DIFFERS from ref", True)


try:
    for _ in range(900):
        if srv.poll() is not None:
            sys.exit("deploy: ds4-server exited during startup, see " + log.name)
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        sys.exit("deploy: ds4-server did not come up in 900 s")
    for name, text in prompts.items():
        msg, n, dt = chat([{"role": "user", "content": text}], 300)
        reply = (msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "")
        status = "ok" if n > 0 and reply.strip("\n-") else "EMPTY"
        suffix, differs = record(name, reply)
        failed |= differs or n == 0
        print(f"deploy: smoke {name}: {n} tokens in {dt:.1f} s ({n / dt:.1f} t/s incl. prefill), {status}{suffix}")

    # A greedy tool round trip: leg 2 answers a tool result on top of leg 1's KV checkpoint, a turn the
    # single-turn prompts never reach. Its wording is judged only against the reference: Ornith garbles
    # the first word here ("Thú Huế") and llama.cpp on the same GGUF gives the same text. The call id is
    # random, so it is not recorded. 4608 tokens leave room for a full --think-budget 4096 before the call.
    msgs = [{"role": "user", "content": "Thời tiết Huế? Dùng tool rồi tóm tắt."}]
    msg, n1, dt = chat(msgs, 4608, weather_tools)
    calls = msg.get("tool_calls") or []
    reply = ((msg.get("reasoning_content") or "") + "\n---\n" + (msg.get("content") or "") + "\n---\n" +
             "\n".join(c["function"]["name"] + " " + c["function"]["arguments"] for c in calls))
    n2 = 0
    if not any(c["function"]["name"] == "get_weather" for c in calls):
        status = "NO TOOL CALL in leg 1"
    else:
        msgs += [msg, {"role": "tool", "tool_call_id": calls[0]["id"],
                       "content": '{"temp_c": 31, "condition": "nắng nhẹ"}'}]
        msg, n2, dt2 = chat(msgs, 4608, weather_tools)
        dt += dt2
        content = msg.get("content") or ""
        reply += "\n===\n" + (msg.get("reasoning_content") or "") + "\n---\n" + content
        status = "ok" if "31" in content else "leg 2 does not use the tool result (no 31)"
    suffix, differs = record("tool", reply)
    failed |= differs or status != "ok"
    print(f"deploy: smoke tool: leg 1 {n1} + leg 2 {n2} tokens in {dt:.1f} s, {status}{suffix}")
finally:
    srv.terminate()
    try:
        srv.wait(stop_timeout)
    except subprocess.TimeoutExpired:
        # Never SIGKILL a Metal process: a killed ds4 can leave its GGUF unopenable until a reboot.
        print("deploy: ds4-server pid %d did not exit %g s after SIGTERM; it is left running, "
              "stop it before the next step" % (srv.pid, stop_timeout), file=sys.stderr)
        failed = True
print("deploy: smoke outputs in", out)
sys.exit(1 if failed else 0)
PY
}

case "${1:-}" in
    cut) shift; cmd_cut "$@" ;;
    install) shift; cmd_install "$@" ;;
    smoke) shift; cmd_smoke "$@" ;;
    *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
