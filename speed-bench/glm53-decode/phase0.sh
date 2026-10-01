#!/bin/zsh
# Phase 0 of the GLM-5.3 decode program
# (docs/superpowers/specs/2026-10-01-glm53-decode-20tps-design.md): env-only
# measurements on the current tree. Run only in a GPU window the user
# authorized, with PROD paused and no peer session on the GPU.
# Never kill -9 a Metal process.
set -u
cd "${0:A:h}/../.."
R=${1:?usage: phase0.sh OUT_DIR}
M=${GLM_MODEL:-$HOME/orca/workspaces/ds4-metal-data/gguf/glm53/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf}
mkdir -p "$R"
Q="Write a Python function that parses an ISO-8601 duration string such as 'P3DT4H5M' into total seconds, with a short docstring and three doctest examples."
log() { print -r -- "STEP $* $(date +%T)" | tee -a "$R/steps.log"; }
# gen NAME N [K=V ...] [-- EXTRA_ARGS ...]
gen() {
    local name=$1 n=$2; shift 2
    local -a envs args
    while (( $# )) && [[ $1 != -- ]]; do envs+=("$1"); shift; done
    (( $# )) && shift
    args=("$@")
    env "${envs[@]}" /usr/bin/time -p ./ds4 -m "$M" --ssd-streaming --power 100 -c 262144 \
        --nothink --temp 0 -n "$n" "${args[@]}" -p "$Q" > "$R/$name.out" 2> "$R/$name.err"
    log "$name rc=$? $(grep -a -o 'generation: [0-9.]* t/s' "$R/$name.err" | tail -1)"
}
log "tree $(git rev-parse --short HEAD)"
# Readahead A/B, interleaved: F_RDADVISE per miss runs on the main thread first.
for rep in 1 2; do
    gen base-$rep 256
    gen nora-$rep 256 DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1
done
# Where the per-layer selected-load time goes (sync / copy / bind / pread).
gen timing 256 DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1
grep -a -i "timing\|selected\|pread" "$R/timing.err" | tail -40 > "$R/timing-summary.txt"
# MTP byte-identity under streaming (decides SP4's verify design).
gen mtp-off 256
gen mtp-on 256 -- --mtp
cmp -s "$R/mtp-off.out" "$R/mtp-on.out" && log "mtp greedy IDENTICAL" || log "mtp greedy DIFFERS"
for rep in 1 2; do cmp -s "$R/base-$rep.out" "$R/nora-$rep.out" && log "readahead rep$rep IDENTICAL" || log "readahead rep$rep DIFFERS"; done
log "all done"
