#!/usr/bin/env bash
# ds41f-ctx.sh — run DeepSeek-V4.1-Flash-1017 (ds41f) on the ds4 Metal fork at a
# chosen context size.  Defaults are tuned for the 64 GB M5 Pro: 512k context,
# expert cache that the fork auto-fits to the remaining budget.
#
# Usage:
#   scripts/ds41f-ctx.sh -p "Xin chào"                    # 512k ctx, short prompt
#   scripts/ds41f-ctx.sh --prompt-file /tmp/p128k.txt -n 64
#   scripts/ds41f-ctx.sh -c 131072 --prompt-file f.txt    # smaller ctx = bigger expert cache
#   scripts/ds41f-ctx.sh --kv-disk /tmp/ds41f-kv -p hi    # persist KV on disk
#
# Notes:
#  * ds4 takes an exclusive lock (/tmp/ds4.lock); stop the prod stack first:
#        bash ~/.local/bin/server-power.sh off   …   on
#    (the prod 35B gateway keeps its own KV dir: ds4-kv-cache-ornith — never reuse it here)
#  * Measured on this machine (cache 4300 auto-fitted to 2827, ctx 524288):
#        ctx 2k → 8.6 t/s decode, 7.5k → 8.5, 30k → 7.4, 60k → 7.0 (see
#        docs/Task/2026-10-27-v41-flash-ds41f-512k-context.md for the full table)
#  * Anything above the prompt length still allocates KV/buffers: a 512k run costs
#    ~13 GiB of context buffers and shrinks the expert cache (4900 → 2827 experts),
#    which is why short-context decode is ~12.9 t/s instead of 15.6 t/s.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
bin=${DS41F_BIN:-$repo/ds4}
model=${DS41F_MODEL:-$HOME/.local/share/ai-gateway/ds4-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf}

ctx=524288
n_predict=64
cache=4300
kv_disk=""
args=()

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--ctx) ctx=$2; shift 2 ;;
        -n) n_predict=$2; shift 2 ;;
        --cache) cache=$2; shift 2 ;;
        --kv-disk) kv_disk=$2; shift 2 ;;
        -p|--prompt|--prompt-file|-t|--threads|--temp|--prefill-chunk|--think-max)
            args+=("$1" "$2"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

[ -x "$bin" ] || { echo "ds4 binary not found: $bin" >&2; exit 1; }
[ -f "$model" ] || { echo "model not found: $model" >&2; exit 1; }

if pgrep -f "ds4-server" >/dev/null 2>&1; then
    echo "warning: ds4-server is running and holds /tmp/ds4.lock." >&2
    echo "         stop the prod stack first: bash ~/.local/bin/server-power.sh off" >&2
fi

kv_args=()
if [ -n "$kv_disk" ]; then
    mkdir -p "$kv_disk"
    kv_args=(--kv-disk-dir "$kv_disk" --kv-disk-space-mb 32768)
fi

echo "ds41f: ctx=$ctx cache=$cache n=$n_predict model=$(basename "$model")" >&2
exec "$bin" -m "$model" --metal --ssd-streaming \
    -c "$ctx" --ssd-streaming-cache-experts "$cache" \
    "${kv_args[@]}" "${args[@]}" -n "$n_predict" --temp 0