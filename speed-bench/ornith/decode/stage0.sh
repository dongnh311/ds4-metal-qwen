# Ornith decode Stage 0 (spec section 3).  Sourced by the scratch gpuwin.sh: $S (out dir), say, run_to;
# cwd = the repo.  CLI one-shots on promessi_sposi prefixes plus a summary request, 256 tokens.
ORNITH=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
export DS4_QWEN35_MTP_DRAFT_VOCAB=$HOME/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
unset DS4_QWEN4_YARN_FACTOR DS4_QWEN35_PROFILE DS4_QWEN35_SPEC_STATS DS4_QWEN35_SPEC_OVERLAP DS4_METAL_CB_TIMES
ok=1
cli() {   # cli NAME PROMPTFILE ENV... -- ARGS...
    local name=$1 pf=$2; shift 2
    local envs=()   # bash 3.2 + set -u: an empty array must expand as ${envs[@]+...}
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    run_to "$name" 1800 env ${envs[@]+"${envs[@]}"} ./ds4 -m "$ORNITH" --metal -c 262144 --prefill-chunk 2048 \
        --prompt-file "$pf" -n 256 "$@" || ok=0
}
for c in 2k:7000 32k:112000 128k:450000; do
    ctx=${c%%:*}; chars=${c#*:}
    pf="$S/prompt-$ctx.txt"
    { head -c "$chars" speed-bench/promessi_sposi.txt; printf '\n\nSummarize the story so far in a few sentences.\n'; } > "$pf"
    cli "$ctx-plain-cb" "$pf" DS4_METAL_CB_TIMES=1 -- --temp 0
    cli "$ctx-mtp-cb" "$pf" DS4_METAL_CB_TIMES=1 DS4_QWEN35_SPEC_STATS=1 -- --temp 0 --mtp
    cli "$ctx-plain-split" "$pf" DS4_QWEN35_PROFILE=3 -- --temp 0
    cli "$ctx-mtp-split" "$pf" DS4_QWEN35_PROFILE=3 DS4_QWEN35_SPEC_OVERLAP=1 -- --temp 0 --mtp
    if [ "$ctx" != 128k ]; then
        cli "$ctx-mtp-t07" "$pf" DS4_QWEN35_SPEC_STATS=1 -- --temp 0.7 --seed 1 --mtp
        cli "$ctx-mtp-plain" "$pf" -- --temp 0 --mtp
    fi
done
cli "2k-plain-l2" "$S/prompt-2k.txt" DS4_QWEN35_PROFILE=2 -- --temp 0
say "stage0 ok=$ok"
[ "$ok" = 1 ]
