#!/usr/bin/env bash
# run-prefill-task1.sh — Campaign Task 1: the prefill/TTFT measurements.
#
# Three parts, one process family (ds4-bench), all under the Task-0 harness
# rules (production launch env mirrored from the live registry row):
#
#  [P1] rep 3 of the Task-0 main ladder (plain + auto-MTP) — delegated to
#       run-orca-iq2-baseline.sh so the rows land in the campaign table
#       (ORCA_REP_FROM=3).  Third day-part sample of the prefill columns.
#  [P2] EQUAL-STEP ladder: --step-incr 32768 from 32K to 225K, auto-MTP,
#       gen 128, production chunk 4096.  Every prefill segment has the SAME
#       token count while depth grows -> prefill_tps here is the pure depth
#       curve (segment-size confound removed), i.e. what a cache-miss prompt
#       actually pays per chunk at depth d.
#  [P3] --prefill-chunk sweep {2048,4096,8192,16384} on the P2 ladder with
#       --gen-tokens 0 (pure prefill): the one engine-side prefill lever a
#       client cannot set.  Production runs 4096.
#
# Output: speed-bench/prefill-task1.csv (own table; Task-0 CSV untouched by
# P2/P3) + receipts under speed-bench/receipts/prefill-<stamp>/ (the Task-0
# aggregator globs orca-iq2-* only, so these never mix).
#
# SAFETY: refuses to run while any ds4 process is up, same rule as Task 0.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODEL="${ORCA_MODEL:-/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf}"
PLE="${ORCA_PLE:-/Users/dongnh/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-PLE-Q4_1.gguf}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="speed-bench/receipts/prefill-$STAMP"
CSV="speed-bench/prefill-task1.csv"
PROMPT="speed-bench/.cache/orca-iq2-frozen-prompt.txt"
LADDER_EQ="--ctx-start 32768 --ctx-max 225280 --step-incr 32768"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "PREFILL ABORT: $*" >&2; exit 2; }

if pgrep -x ds4-server >/dev/null 2>&1 || pgrep -x ds4-bench >/dev/null 2>&1 || pgrep -x ds4-agent >/dev/null 2>&1; then
    die "a ds4 model process is running. Take the freeze window first (see night-task12.sh)."
fi
[ -s "$PROMPT" ] || die "frozen prompt missing — run the Task-0 sweep first: $PROMPT"

# ── production env mirror (identical rule to run-orca-iq2-baseline.sh) ────────
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
fi

mkdir -p "$OUTDIR"
[ -f "$CSV" ] || echo "ts,label,prefill_chunk,gen_tokens,ctx,prefill_tokens,prefill_tps,bench_gen_tokens,gen_tps,mtp_pct,depth3_cycles,depth3_accept2" >> "$CSV"

run_case() {
    local label="$1" chunk="$2" gen="$3"; shift 3
    local out="$OUTDIR/prefill-$label.csv" err="$OUTDIR/prefill-$label.err"
    log "case=$label chunk=$chunk gen=$gen"
    # shellcheck disable=SC2086  # $LADDER_EQ / remaining args are word-split flags by design
    env ${PROD_ENV[@]+"${PROD_ENV[@]}"} DS4_LOCK_FILE="${ORCA_BENCH_LOCK:-/tmp/ds4-bench.lock}" \
        ./ds4-bench -m "$MODEL" --ple "$PLE" \
        --prompt-file "$PROMPT" --prefill-chunk "$chunk" \
        --gen-tokens "$gen" --mtp --mtp-timing "$@" --csv "$out" 2>"$err"
    local mp d3 a2
    mp=$(sed -n 's/.*drafts accepted (\([0-9.]*\)%.*/\1/p' "$err" | tail -1)
    d3=$(sed -n 's/.*depth3: \([0-9]*\) cycles.*/\1/p' "$err" | tail -1)
    a2=$(sed -n 's/.*depth3: [0-9]* cycles, \([0-9]*\) second-draft.*/\1/p' "$err" | tail -1)
    while IFS=, read -r ctx pt ptps gt gtps gfms gstk gst skv; do
        [ "$ctx" = "ctx_tokens" ] && continue
        echo "$(date -u +%FT%TZ),$label,$chunk,$gen,$ctx,$pt,$ptps,$gt,$gtps,$mp,$d3,$a2" >> "$CSV"
    done < <(cat "$out")
    log "case=$label done"
}

# ── P1: rep 3 into the Task-0 table (own receipts dir, own OUTDIR stamp) ──────
if [ -z "${SKIP_P1:-}" ]; then
    log "P1: Task-0 main ladder rep 3 (plain + mtp)"
    ORCA_RUNS=3 ORCA_REP_FROM=3 ORCA_ARMS="plain mtp" \
        ORCA_BENCH_LOCK="${ORCA_BENCH_LOCK:-/tmp/ds4-bench.lock}" \
        ./speed-bench/run-orca-iq2-baseline.sh
fi

# ── P2: equal-step depth curve, production chunk, decode on ───────────────────
if [ -z "${SKIP_P2:-}" ]; then
    log "P2: equal-32K-step ladder (depth curve, chunk 4096, gen 128)"
    run_case equal32k-4096 4096 128 $LADDER_EQ
fi

# ── P3: prefill-chunk sweep, pure prefill (gen 0) ─────────────────────────────
for chunk in ${CHUNKS:-2048 4096 8192 16384}; do
    log "P3: pure-prefill chunk $chunk"
    run_case "pure-chunk$chunk" "$chunk" 0 $LADDER_EQ
done

log "done. CSV: $CSV  receipts: $OUTDIR/"
