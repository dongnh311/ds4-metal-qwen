#!/usr/bin/env bash
# run-orca-iq2-baseline.sh — FAST Campaign Task 0: the fixed baseline sweep.
#
# Protocol (docs/Task/2026-09-15-ds4-iq2-m5pro-fast-campaign.md, Task 0):
#   five context points (32K, 64K, 128K, 192K, 225K), greedy 128-token decode
#   per frontier (ds4-bench: temp 0, EOS suppressed), MTP arms on/off, 3 reps,
#   one provenance receipt per run.  next_frontier() clamps the doubling ladder
#   to --ctx-max, so ONE run walks 32768 -> 65536 -> 131072 -> 225280; the 192K
#   point is one extra single-frontier run per arm/rep (CTX_EXTRA, optional).
#
# Output: speed-bench/orca-iq2-baseline.csv (the campaign's authoritative rows)
# and speed-bench/receipts/orca-iq2-<stamp>/run-<arm>-r<rep>*.{csv,err} plus
# receipt-<arm>-r<rep>-<ladder>.json per run.
#
# SAFETY: ds4-bench loads the FULL resident model (~41 GiB).  The live
# ds4-server already plans 53 GiB on this 64 GB box — running both is the
# AGENT.md forbidden case.  This script REFUSES to start while any ds4 model
# process is up.  Stop the gateway's ds4 slot first, run, restart: the KV disk
# cache survives; the first turns re-prefill.
#
# Tunables (env): ORCA_MODEL, ORCA_PLE, ORCA_RUNS, ORCA_ARMS, ORCA_CTX_MAIN,
# ORCA_CTX_EXTRA, ORCA_PREFILL_CHUNK, ORCA_PROMPT_BYTES, ORCA_SKIP_192K=1.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL="${ORCA_MODEL:-/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf}"
PLE="${ORCA_PLE:-/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="speed-bench/receipts/orca-iq2-$STAMP"
CSV="speed-bench/orca-iq2-baseline.csv"
PROMPT="speed-bench/.cache/orca-iq2-frozen-prompt.txt"
PROMPT_BYTES="${ORCA_PROMPT_BYTES:-1200000}"   # ~340K tokens of code text
GEN=128
CTX_MAIN="${ORCA_CTX_MAIN:---ctx-start 32768 --ctx-max 225280 --step-mul 2}"
CTX_EXTRA="${ORCA_CTX_EXTRA:---ctx-start 196608 --ctx-max 196608}"
PREFILL_CHUNK="${ORCA_PREFILL_CHUNK:-8192}"    # mirror production (4096 -> 8192 flipped 2026-10-30, see ORCA_IQ2_FAST_CAMPAIGN.md; reps 1-3 ran at 4096)
RUNS="${ORCA_RUNS:-3}"
REP_FROM="${ORCA_REP_FROM:-1}"        # e.g. ORCA_ARMS=plain ORCA_REP_FROM=2 ORCA_RUNS=2 reruns only rep 2
ARMS="${ORCA_ARMS:-plain mtp mtp-d2 mtp-d3 mtp-force}"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "BASELINE ABORT: $*" >&2; exit 2; }

# ── guard: one huge model process at a time ────────────────────────────────────
if pgrep -x ds4-server >/dev/null 2>&1 || pgrep -x ds4-bench >/dev/null 2>&1 || pgrep -x ds4-agent >/dev/null 2>&1; then
    die "a ds4 model process is running (AGENT.md instance rule). Stop the gateway ds4 slot first. Override is deliberately absent: this box cannot fit two."
fi
[ -f "$MODEL" ] || die "model not found: $MODEL"
[ -f "$PLE" ]   || die "PLE sidecar not found: $PLE"

# Mirror the production launch environment. The live ds4 row runs the engine
# under /usr/bin/env VAR=value pairs (MTP draft vocab, grow-on-demand KV,
# streamed full layers); measuring without them measures a configuration
# nobody runs. ORCA_SKIP_PROD_ENV=1 opts out (e.g. a machine with no registry).
PROD_ENV=()
if [ -z "${ORCA_SKIP_PROD_ENV:-}" ]; then
    REG="${ORCA_REGISTRY:-$HOME/.local/ai-gateway/runtime-registry.json}"
    ROW="${ORCA_ROW:-ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2}"
    [ -f "$REG" ] || die "registry not found: $REG (or set ORCA_SKIP_PROD_ENV=1)"
    while IFS= read -r kv; do
        [ -n "$kv" ] && PROD_ENV+=("$kv")
    done < <(python3 - "$REG" "$ROW" <<'PY'
import json, sys
reg, row = sys.argv[1:3]
pc = json.load(open(reg))["models"][row]["runtimes"]["ds4"]["process_command"]
for tok in pc[1:]:
    if "=" in tok and not tok.startswith("/") and tok.split("=", 1)[0].replace("_", "").isalnum():
        print(tok)
    else:
        break
PY
    ) || die "could not parse the env prefix from $REG row $ROW"
    log "production env mirrored: ${PROD_ENV[*]:-none}"
fi

mkdir -p "$OUTDIR" "$(dirname "$PROMPT")"
if [ ! -s "$PROMPT" ]; then
    head -c "$PROMPT_BYTES" ds4.c > "$PROMPT"
    log "frozen prompt built from ds4.c @ $(git rev-parse --short HEAD) ($PROMPT_BYTES bytes)"
fi

CHIP="$(sysctl -n machdep.cpu.brand_string)"
MACOS="$(sw_vers -productVersion)"
GIT="$(git rev-parse HEAD)"
MODEL_TAG="$(basename "$MODEL" .gguf)"
PROMPT_SHA="$(shasum -a 256 "$PROMPT" | cut -d' ' -f1)"
MODEL_SHA_FILE="$OUTDIR/.model-sha256"
if [ -f "$MODEL_SHA_FILE" ]; then MODEL_SHA="$(cat "$MODEL_SHA_FILE")"; else
    log "hashing model (one-time per sweep)…"
    MODEL_SHA="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)"; printf '%s' "$MODEL_SHA" > "$MODEL_SHA_FILE"
fi

[ -f "$CSV" ] || echo "ts,arm,rep,ladder,git,model_tag,ctx,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,mtp_cycles,mtp_accepted,mtp_pct,depth3_cycles,depth3_accept2,swap_peak_mb" >> "$CSV"

swap_used_mb() {
    # vm.swapusage: "total = 2048.00M used = 1062.00M free = ..."; also byte form
    sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)([KMGT]?)/\1 \2/' | awk '
        { v = $1 + 0; u = $2
          if (u ~ /G/) print int(v * 1024)
          else if (u ~ /M/) print int(v)
          else if (u ~ /K/) print int(v / 1024)
          else print int(v / 1048576) }'
}

run_ladder() {
    local arm="$1" rep="$2" ctxflags="$3" label="$4"
    local out="$OUTDIR/run-$arm-r$rep-$label.csv" err="$OUTDIR/run-$arm-r$rep-$label.err"
    # the engine takes DS4_LOCK_FILE (default /tmp/ds4.lock); the deploy window
    # holds that lock to keep the gateway's slot down, so benches need their own
    local -a envs=("${PROD_ENV[@]+"${PROD_ENV[@]}"}" "DS4_LOCK_FILE=${ORCA_BENCH_LOCK:-/tmp/ds4.lock}") flags=()
    # The acceptance counters need BOTH flags (bench: --mtp enables the spec path,
    # --mtp-timing only prints; decoupled 2026-10-29 because the first sweep's
    # "plain" arm silently ran MTP via the CLI's imply-from-timing behaviour).
    case "$arm" in
        plain) flags=() ;;
        mtp) flags=(--mtp --mtp-timing) ;;
        mtp-d2) flags=(--mtp --mtp-timing); envs+=(DS4_QWEN4_MTP_DEPTH=2) ;;
        mtp-d3) flags=(--mtp --mtp-timing); envs+=(DS4_QWEN4_MTP_DEPTH=3) ;;
        mtp-force) flags=(--mtp --mtp-timing); envs+=(DS4_QWEN4_SPEC_FORCE_ACCEPT=1) ;;
        *) die "unknown arm $arm" ;;
    esac
    local swap0 swap1 peak; swap0="$(swap_used_mb)"
    log "arm=$arm rep=$rep ladder=$label swap0=${swap0}MB"
    # shellcheck disable=SC2086  # CTX_* are word-split flags by design
    env ${envs[@]+"${envs[@]}"} ./ds4-bench -m "$MODEL" --ple "$PLE" \
        --prompt-file "$PROMPT" --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens "$GEN" ${flags[@]+"${flags[@]}"} $ctxflags --csv "$out" 2>"$err"
    swap1="$(swap_used_mb)"; peak=$(( swap0 > swap1 ? swap0 : swap1 ))
    local mc ma mp d3c d3a
    mc=$(sed -n 's/.*mtp: \([0-9]*\) verify cycles.*/\1/p' "$err" | tail -1)
    ma=$(sed -n 's/.*cycles, \([0-9]*\) drafts accepted.*/\1/p' "$err" | tail -1)
    mp=$(sed -n 's/.*drafts accepted (\([0-9.]*\)%.*/\1/p' "$err" | tail -1)
    d3c=$(sed -n 's/.*depth3: \([0-9]*\) cycles.*/\1/p' "$err" | tail -1)
    d3a=$(sed -n 's/.*depth3: [0-9]* cycles, \([0-9]*\) second-draft.*/\1/p' "$err" | tail -1)
    while IFS=, read -r ctx pt ptps gt gtps gfms gstk gst skv; do
        echo "$(date -u +%FT%TZ),$arm,$rep,$label,$GIT,$MODEL_TAG,$ctx,$pt,$ptps,$gt,$gtps,$gfms,$gstk,$gst,$skv,${mc:-},${ma:-},${mp:-},${d3c:-},${d3a:-},$peak" >> "$CSV"
    done < <(tail -n +2 "$out")
    cat > "$OUTDIR/receipt-$arm-r$rep-$label.json" <<JSON
{
 "protocol": "orca-iq2-fast-campaign-task0",
 "git_commit": "$GIT",
 "model": "$MODEL_TAG",
 "model_sha256": "$MODEL_SHA",
 "prompt_sha256": "$PROMPT_SHA",
 "macos": "$MACOS",
 "chip": "$CHIP",
 "gen_tokens": $GEN,
 "temp": 0,
 "prefill_chunk": $PREFILL_CHUNK,
 "arm": "$arm",
 "rep": $rep,
 "ladder": "$label",
 "ctx_flags": "$ctxflags",
 "mtp_env": "${envs[*]:-none}",
 "dir_steering": "none — bench has no --dir-steering-file; production carries refusal-4-44 ffn=0.5, measure that delta separately",
 "swap_peak_mb_coarse": $peak,
 "golden_checksum": "deferred — run tests/test_qwen4_logit_dump.py once for the Phase-1 bit-identical fixture"
}
JSON
    log "arm=$arm rep=$rep ladder=$label done swap1=${swap1}MB cycles=${mc:-0}"
}

log "sweep start: arms=[$ARMS] runs=$RUNS main=[$CTX_MAIN]"
for arm in $ARMS; do
    for rep in $(seq "$REP_FROM" "$RUNS"); do
        run_ladder "$arm" "$rep" "$CTX_MAIN" main
        [ -n "${ORCA_SKIP_192K:-}" ] || run_ladder "$arm" "$rep" "$CTX_EXTRA" 192k
    done
done
log "done. CSV: $CSV  receipts: $OUTDIR/"
