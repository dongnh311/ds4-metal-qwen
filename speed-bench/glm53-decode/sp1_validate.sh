#!/bin/zsh
# SP1 validation (docs/superpowers/plans/2026-10-01-glm53-decode-gates.md,
# Task 8): byte identity, decode speed, GPU busy. Run only in a GPU window the
# user authorized, PROD paused, no peer session on the GPU.
# Never kill -9 a Metal process.
set -u
cd "${0:A:h}/../.."
R=${1:?usage: sp1_validate.sh OUT_DIR}
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
    env "${envs[@]}" ./ds4 -m "$M" --ssd-streaming --power 100 -c 262144 \
        --nothink --temp 0 -n "$n" "${args[@]}" -p "$Q" > "$R/$name.out" 2> "$R/$name.err"
    log "$name rc=$? $(grep -a -o 'generation: [0-9.]* t/s' "$R/$name.err" | tail -1)"
}
same() { cmp -s "$R/$1.out" "$R/$2.out" && log "$2 vs $1 IDENTICAL" || log "$2 vs $1 DIFFERS"; }
log "tree $(git rev-parse --short HEAD)"
# 1-2. Byte identity: drain vs one-pass gates vs split gates.
for n in 16 256; do
    gen drain-$n $n DS4_GLM_STREAM_GATE=0
    gen gate1-$n $n DS4_GLM_STREAM_SPLIT=0 DS4_GLM_STREAM_TIMING=1
    gen split-$n $n DS4_GLM_STREAM_TIMING=1
    same drain-$n gate1-$n
    same drain-$n split-$n
done
# 3. MTP: the 2-token verify drains, the single-token replay is gated.
gen mtp-drain 256 DS4_GLM_STREAM_GATE=0 -- --mtp
gen mtp-split 256 -- --mtp
same mtp-drain mtp-split
# 4. Decode speed, interleaved reps.
for rep in 1 2; do
    gen speed-drain-$rep 256 DS4_GLM_STREAM_GATE=0
    gen speed-gate1-$rep 256 DS4_GLM_STREAM_SPLIT=0
    gen speed-split-$rep 256
done
# 5. GPU busy per decode token: 136 vs 8 generated tokens differenced.
for v in drain split; do
    local -a ev=()
    [[ $v == drain ]] && ev=(DS4_GLM_STREAM_GATE=0)
    gen busy-$v-8 8 DS4_METAL_GPU_BUSY_PROFILE=1 "${ev[@]}"
    gen busy-$v-136 136 DS4_METAL_GPU_BUSY_PROFILE=1 "${ev[@]}"
done
python3 - "$R" <<'EOF' | tee -a "$R/steps.log"
import re, sys
r = sys.argv[1]
def last(path, pat):
    m = re.findall(pat, open(path, errors="replace").read())
    return float(m[-1]) if m else None
for v in ("drain", "split"):
    b8 = last(f"{r}/busy-{v}-8.err", r"gpu busy accum ([0-9.]+) ms")
    b136 = last(f"{r}/busy-{v}-136.err", r"gpu busy accum ([0-9.]+) ms")
    tps = last(f"{r}/busy-{v}-136.err", r"generation: ([0-9.]+) t/s")
    if None in (b8, b136, tps):
        print(f"STEP busy {v}: missing numbers"); continue
    busy = (b136 - b8) / 128.0
    wall = 1000.0 / tps
    print(f"STEP busy {v}: {busy:.1f} ms GPU of {wall:.1f} ms per token = {100 * busy / wall:.0f}%")
EOF
grep -a "GLM stream gates" "$R/split-256.err" | tail -2 >> "$R/steps.log"
log "cli done"
