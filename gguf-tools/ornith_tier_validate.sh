#!/usr/bin/env sh
# One GPU window, one model: layout acceptance + behaviour battery + matched-window
# perplexity + a 2K/8K/32K bench sweep. Run with the prod ds4 slot stopped; the
# gateway canary relaunches it as soon as this process releases /tmp/ds4.lock.
#
#   sh gguf-tools/ornith_tier_validate.sh <model.gguf> [tag]
set -e
MODEL="$1"
TAG="${2:-tier}"
REPO="$(dirname "$0")/.."
QA='What is the capital of Japan?
What is 123 * 45?
Who wrote the novel Dune?
What is the boiling point of water in Celsius?'

echo "== inspect =="
"$REPO/ds4" --inspect --metal -m "$MODEL" -c 512 2>&1 | sed -n '1,12p'

echo "== behaviour battery (temp 0, 60 tokens) =="
i=0
for q in $(printf '%s\n' "$QA"); do
    i=$((i + 1))
    out="/tmp/${TAG}_qa${i}.log"
    "$REPO/ds4" --metal -m "$MODEL" -c 2048 -n 60 --temp 0 -p "$q" >"$out" 2>&1
    printf '  %s -> ' "$q"
    grep -v '^ds4: ' "$out" | head -2 | tr '\n' ' '
    grep -E 'prefill: |generation: ' "$out" | sed 's/^/    /'
done

echo "== matched-window perplexity (prefix 32, 200 tokens) =="
"$REPO/ds4" --metal -m "$MODEL" -c 1024 -n 200 --perplexity-file /tmp/ppl_probe.txt 2>&1 |
    grep -E 'tokens=' | sed 's/^/  /'

echo "== mtp draft path (spec stats over ~640 greedy tokens) =="
"$REPO/ds4" --metal -m "$MODEL" -c 4096 -n 640 --temp 0 --mtp \
    -p 'Summarise how a bicycle works, then explain why gears matter, then describe one long ride you would plan through the mountains, then list three maintenance tips, then finish with a short poem about wheels.' 2>&1 |
    grep -E 'spec stats|committed|draft' | sed 's/^/  /'

echo "== decode-consistency (graph vs fresh re-prefill) =="
"$REPO/ds4" --metal -m "$MODEL" -c 4096 --decode-consistency 8 --temp 0 \
    -p 'The capital of Australia is' 2>&1 | grep -E 'decode-consistency compared' | sed 's/^/  /'

echo "== bench sweep =="
"$REPO/ds4-bench" --metal -m "$MODEL" --prompt-file /tmp/bench_122b.txt \
    --ctx-start 2048 --ctx-max 32768 --step-mul 4 --gen-tokens 128 --csv "/tmp/bench_${TAG}.csv" 2>&1 |
    grep -E 'memory: KV|planned' | sed 's/^/  /'
cat "/tmp/bench_${TAG}.csv"