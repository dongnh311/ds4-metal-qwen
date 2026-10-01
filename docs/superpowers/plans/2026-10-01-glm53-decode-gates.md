# GLM-5.3 decode gates (Phase 0 + SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the per-layer host wait from GLM-5.3 streamed decode on Metal, so a token is encoded without
waiting and only its end waits. Target: ≥ 12.5 t/s from ~9.2-9.4 t/s, with byte-identical output.

**Architecture:** The qwen4 "stream gates" (qgate in `ds4_metal.m`) become a shared core with two clients. The
main thread publishes a layer's router selection into a mailbox. It encodes the shared expert, then commits
without waiting. The next batch opens with a GPU poll. A service thread resolves the experts into the stream
expert cache and writes the gate's own address tables, then releases the poll. The routed GLM kernels are
unchanged. In gate mode they read the gate's tables. An address of 0 makes them skip a slot, which also gives
split gates (cached experts first, misses second) with no kernel change.

**Tech Stack:** C (ds4.c), Objective-C/Metal (ds4_metal.m, metal/*.metal), zsh and Python bench scripts, make.

**Spec:** `docs/superpowers/specs/2026-10-01-glm53-decode-20tps-design.md` (commit 8881537d; this plan's
commit adds a "Planning amendments" section to it, listed below).

## Planning amendments (rulings made while reading the code; the spec is updated to match)

1. **Tables.** Gated GLM layers read per-gate ring tables in both passes. The layer's own address table is
   not used. Pass 1 holds every expert, or only the cached ones when the gate is split. Pass 2 holds the
   misses. A 0 address makes `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` and `kernel_mul_mv_addr_q2_K_f32`
   return early (`metal/moe.metal` ~4001, ~4143), so split gates need no new kernel. Cost if wrong: a
   kernel variant with a pending mask, which is what qwen4 has.
2. **Failure scope.** A gate failure disables gates for the process: it is the existing qgate latch, and
   there is one model per process. The spec said "for the session". For a server this has the same effect.
3. **Start condition.** A token is gated only with the static decode map. The measured config has it (log:
   "GLM SSD streaming decode map: global 8.24 GiB"). Without it, every layer remaps, and with
   `sync_each_layer` it also waits.
4. **Cold start.** With gates requested, the cache allocates one slab. The existing slab sizing at
   `ds4_metal.m` ~14660 keys on `qgate_requested()`, which is already true for every model. So gates start
   right after the first decode miss, not after ~65 tokens.
5. **Cache ownership.** `end_commands` now also waits until the service thread is idle. This covers the
   bookkeeping it does after a release, and closes a small race that qwen4 has today. It is a host-side wait
   only and does not change output.
6. **GLM fallback buffer:** 8 slots (~54 MiB), because GLM routes 8 experts. qwen4 keeps 64. The poll regions
   (48 MiB) are shared with qwen4. In total, GLM adds about 104 MiB.
7. **Lookahead prefetch** stays qwen4-only in SP1. GLM prefetch belongs to SP2.
8. **Metric.** "Waited command buffers per token" has no clean counter. SP1 instead reports GPU busy %
   (target ≥ 70%) and gated layers per token from `DS4_GLM_STREAM_TIMING`.

## Global Constraints

- **Byte-identical output is a hard requirement.** Greedy output with gates (one pass or split) must match
  `DS4_GLM_STREAM_GATE=0` token for token, at 16 and 256 tokens, with and without `--mtp`.
- **GLM and Metal only.** qwen4 keeps the same command stream. The Qwen regression gate
  (`speed-bench/qwen-regression/run.sh full`) must PASS, with Qwen decode ≥ 97% of PROD. ROCm and CUDA are
  untouched: every ds4.c call into the new API sits under `#if defined(__APPLE__) && !defined(DS4_ROCM_BUILD)`.
- **Memory:** stay inside the 256K plan (~48 GiB). The new buffers are the GLM fallback (8 slots, ~54 MiB)
  and 96 tables of 4 KiB each.
- **Switches:** `DS4_GLM_STREAM_GATE=0` restores the drain path. `DS4_GLM_STREAM_SPLIT=0` gives one-pass
  gates. `DS4_GLM_STREAM_TIMING=1` prints gate stats. `DS4_GLM_STREAM_GATE_TEST_STALL=N:MS` is a test-only
  stall.
- **SP1 gate:** ≥ 12.5 t/s, byte-identical, GPU busy ≥ 70%, Qwen gate full PASS. Stop rule: below 10.8 t/s
  (+15%), stop and re-evaluate the program with the user.
- **Language:** code, comments, docs and commit messages in English. Every commit ends with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc`.
- **GPU rules (user standing orders):**
  - Anything that calls `ds4_gpu_init` (Metal unit tests, `ds4`, `ds4-server`) uses the GPU. Run it only when
    no peer session holds the GPU. When the GPU is held, build only and run at the next window.
  - Model runs also need the user's go and PROD paused: `~/.local/bin/server-power.sh off`, then `on`
    afterwards. Always restore and confirm `backend_ok`, even if the run is cut short.
  - Notify the sessions "Mac16-Ai-Gateway" and "Mac16-DS4-Metal-Qwen" when PROD goes off or on, and when you
    take or free the GPU.
  - Never `kill -9` a Metal process; send exactly one SIGTERM. Never open GGUF files outside the ds4 binary.
- **Git:** work on `feature/glm53-decode-gates` in the dugong worktree. Never `git worktree remove` this
  worktree. Push only when the user asks. No deploy.

## Review Focus

1. **A token mixing gated and drained layers.** This happens when a publish is refused because the queue is
   full or a slab is missing. Expected: byte-identical output. Pinned by the mixed-token section in Task 4.
2. **A publish never committed,** for example when the shared expert fails. Expected: the next batch starts
   clean, and no stale gate reaches a later layer. Pinned by `check_uncommitted_publish` in Task 4.
3. **The cache budget grows mid-run (a new slab).** Expected: gates pause until every slab exists, then
   resume, with identical output. Pinned by the budget-growth section in Task 4.
4. **`--mtp` decoding with gates.** The 2-token verify is not gated; the single-token replay is. Expected:
   greedy identical to gates off. Pinned by Task 8, step 3.
5. **A server request after a gate failure.** Expected: the failed request errors, no wrong token is
   emitted, and the next request succeeds on the drain path. Pinned by Task 8, step 6 (stall smoke).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `ds4_metal.m` | modify | qgate core refactor (Task 3), GLM gate client, service and tables (Task 4), stall (Task 5), split pass 2 (Task 6), test hooks (Task 2) |
| `ds4_gpu.h` | modify | `ds4_gpu_glm_stream_gate_publish/commit` + test hooks (Apple only) |
| `ds4.c` | modify | `stream_gate_token` field, `glm_graph_stream_gate_token_allowed`, publish/commit in `glm_graph_encode_sparse_ffn_one` (Task 7) |
| `tests/test_metal_stream_gate.c` | create | model-free gate vs drain checks: `--qwen4`, `--glm`, `--glm-timeout`, `--glm-split` |
| `tests/test_glm53_stream_layout.c` | modify | `check_glm_stream_gate_token_allowed` (Task 7) |
| `Makefile` | modify | build, `test-metal-stream-gate` target, clean |
| `speed-bench/glm53-decode/phase0.sh` | create | Phase 0 measurements (Task 1) |
| `speed-bench/glm53-decode/sp1_validate.sh` | create | SP1 model validation (Task 8) |
| `speed-bench/glm53-decode/server_smoke.py` | create | server speed / reuse / stall smoke (Task 8) |
| `speed-bench/glm53-decode/RESULTS.md` | create | Phase 0 and SP1 numbers |
| `docs/superpowers/specs/2026-10-01-glm53-decode-20tps-design.md` | modify | planning amendments (this commit) |

Line numbers below are from commit 8881537d; locate by the quoted anchor text, not the number.

---

### Task 1: Phase 0 measurements (GPU window, no code change)

**Files:**
- Create: `speed-bench/glm53-decode/phase0.sh`
- Create: `speed-bench/glm53-decode/RESULTS.md`

**Interfaces:** none. Independent of Tasks 2-7; run at the first authorized GPU window.

- [ ] **Step 1: Write the script**

```zsh
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
```

- [ ] **Step 2: Syntax check and commit (no GPU)**

Run: `zsh -n speed-bench/glm53-decode/phase0.sh && chmod +x speed-bench/glm53-decode/phase0.sh`
Expected: no output, exit 0.

Create `speed-bench/glm53-decode/RESULTS.md`:

```markdown
# GLM-5.3 decode program: measurements

Model: GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf, `--ssd-streaming --power 100 -c 262144`,
M5 Pro 64 GB, greedy, 256 tokens, prompt: the ISO-8601 duration task.

## Phase 0 (env only)

| Run | rep 1 t/s | rep 2 t/s | Output vs base |
|---|---|---|---|
| base | | | - |
| readahead off | | | |
| `--mtp` | | - | |

Selected-load time split (from timing-summary.txt):

## SP1 (decode gates)
```

```bash
git add speed-bench/glm53-decode/phase0.sh speed-bench/glm53-decode/RESULTS.md
git commit -m "bench: GLM-5.3 decode Phase 0 measurement script

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

- [ ] **Step 3: Run in a GPU window (needs the user's go)**

1. Confirm the Qwen session has sent "GPU free" and the user authorized the window.
2. Notify both peers that PROD is going off and the GPU is taken.
3. Run `~/.local/bin/server-power.sh off`.
4. Run `make ds4 && speed-bench/glm53-decode/phase0.sh $SCRATCH/phase0` from the scratchpad, in the
   background, then wait for "all done".
5. Run `~/.local/bin/server-power.sh on` and confirm `backend_ok`.
6. Notify both peers that PROD is back on and the GPU is free.

Expected: 7 runs with rc=0 and a line saying MTP greedy IDENTICAL or DIFFERS. Either answer is data.

- [ ] **Step 4: Record and commit**

Fill the Phase 0 table and the time split in RESULTS.md from `steps.log` and `timing-summary.txt`.

```bash
git add speed-bench/glm53-decode/RESULTS.md
git commit -m "results: GLM-5.3 decode Phase 0 (readahead A/B, load time split, MTP identity)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 2: Gate test hooks and the qwen4 gate characterization test

Pins today's qwen4 gate behaviour (gated output == drain output) before the core is refactored.

**Files:**
- Modify: `ds4_metal.m` (qgate block ~52695-52810, `qgate_encode` ~53487)
- Modify: `ds4_gpu.h` (after the `ds4_gpu_stream_expert_cache_note_service_thread` declaration, ~428)
- Create: `tests/test_metal_stream_gate.c`
- Modify: `Makefile` (after the `test-metal-ssd-experts` rules ~221; `clean` ~1180)

**Interfaces:**
- Produces (ds4_gpu.h, `#ifdef __APPLE__`):
  - `void ds4_gpu_stream_gate_test_set_mode(int mode, int split);`
    `-1` follows the env, `0` is off, `1` is on. Applies to both clients.
  - `void ds4_gpu_stream_gate_stats(uint64_t *committed, uint64_t *split, uint64_t *fallback, int *failed);`
    Counts are cumulative. The call waits for the service thread to go idle first.
- Produces (test file): the fixture `setup()`, `route()`, `run_step()` and `compare_modes()`, all reused by
  Tasks 4-6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metal_stream_gate.c`:

```c
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * Stream gates against the per-layer drain on a synthetic expert file
 * (IQ2_XXS gate/up, Q2_K down; GLM-5.3 routes 8 of 288 experts). Every step
 * encodes LAYERS streamed layers in one batch, gated first so its layers
 * miss, then drained; the outputs must be byte-identical while a cache of
 * BUDGET experts evicts on every step.
 */
enum { D = 256, H = 512, E = 288, N = 8, LAYERS = 4, STEPS = 24, BUDGET = 24 };
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void *model;
static size_t model_bytes;
static FILE *model_file;
static uint64_t row_bytes, down_row_bytes, expert_bytes, down_expert_bytes, tensor_bytes;
static ds4_gpu_tensor *xt, *wt, *gate, *up, *mid, *down, *shared_t;
static ds4_gpu_tensor *ids_t[LAYERS], *out_t[LAYERS];
static float ref[LAYERS][N * D], got[LAYERS][N * D];
/* Elements each layer writes: qwen4 partials are N x D, GLM's summed output D. */
static int out_elems = N * D;

static int setup(void) {
    row_bytes = D / 256 * sizeof(iq2_block);
    down_row_bytes = H / 256 * sizeof(q2_block);
    expert_bytes = (uint64_t)H * row_bytes;
    down_expert_bytes = (uint64_t)D * down_row_bytes;
    tensor_bytes = (uint64_t)E * expert_bytes;
    model_bytes = 2 * tensor_bytes + (uint64_t)E * down_expert_bytes;
    model_file = tmpfile();
    if (!model_file || ftruncate(fileno(model_file), (off_t)model_bytes)) return 0;
    model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(model_file), 0);
    if (model == MAP_FAILED) return 0;
    iq2_block *g = model;
    for (uint64_t i = 0; i < 2 * tensor_bytes / sizeof(iq2_block); i++) {
        g[i].d = 0x1400;
        for (size_t j = 0; j < sizeof(g[i].qs); j++) g[i].qs[j] = (uint8_t)random_u32();
    }
    q2_block *q = (q2_block *)((char *)model + 2 * tensor_bytes);
    for (uint64_t i = 0; i < (uint64_t)E * down_expert_bytes / sizeof(q2_block); i++) {
        q[i].d = 0x1400;
        q[i].dmin = 0x1000;
        for (size_t j = 0; j < sizeof(q[i].scales); j++) q[i].scales[j] = (uint8_t)random_u32();
        for (size_t j = 0; j < sizeof(q[i].qs); j++) q[i].qs[j] = (uint8_t)random_u32();
    }
    if (msync(model, model_bytes, MS_SYNC) || !ds4_gpu_init()) return 0;
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * expert_bytes + down_expert_bytes);
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_set_model_fd(fileno(model_file))) return 0;
    float x[D], w[N];
    for (int i = 0; i < D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int i = 0; i < N; i++) w[i] = (i + 1) / 36.0f;
    xt = ds4_gpu_tensor_alloc(sizeof(x));
    wt = ds4_gpu_tensor_alloc(sizeof(w));
    gate = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    up = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    mid = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    down = ds4_gpu_tensor_alloc((uint64_t)N * D * sizeof(float));
    shared_t = ds4_gpu_tensor_alloc(sizeof(x));
    int ok = xt && wt && gate && up && mid && down && shared_t &&
             ds4_gpu_tensor_write(xt, 0, x, sizeof(x)) &&
             ds4_gpu_tensor_write(wt, 0, w, sizeof(w));
    for (int l = 0; ok && l < LAYERS; l++) {
        ids_t[l] = ds4_gpu_tensor_alloc(N * sizeof(int32_t));
        out_t[l] = ds4_gpu_tensor_alloc((uint64_t)N * D * sizeof(float));
        ok = ids_t[l] && out_t[l];
    }
    return ok;
}

/* Four hot experts per layer and four that rotate; the slot order moves so a
 * miss lands in every slot position. */
static void route(int step, int layer, int32_t ids[N]) {
    for (int i = 0; i < N; i++) {
        const int slot = (i + step + layer) % N;
        ids[slot] = i < 4 ? layer * 4 + i : 64 + (step * 4 + i + layer * 37) % (E - 64);
    }
}

typedef int (*layer_fn)(int layer, int gated);

/* One step: LAYERS layers in one batch, then every layer's output. */
static int run_step(layer_fn fn, int gated, float out[LAYERS][N * D]) {
    for (int l = 0; l < LAYERS; l++)
        if (!ds4_gpu_tensor_fill_f32(out_t[l], NAN, (uint64_t)N * D)) return 0;
    if (!ds4_gpu_begin_commands()) return 0;
    int ok = 1;
    for (int l = 0; ok && l < LAYERS; l++) ok = fn(l, gated);
    ok = ds4_gpu_end_commands() && ok;
    for (int l = 0; ok && l < LAYERS; l++)
        ok = ds4_gpu_tensor_read(out_t[l], 0, out[l], sizeof(out[l]));
    return ok;
}

static int write_routes(int step) {
    for (int l = 0; l < LAYERS; l++) {
        int32_t ids[N];
        route(step, l, ids);
        if (!ds4_gpu_tensor_write(ids_t[l], 0, ids, sizeof(ids))) return 0;
    }
    return 1;
}

static int same_outputs(const char *name, int step) {
    for (int l = 0; l < LAYERS; l++) {
        for (int i = 0; i < out_elems; i++) {
            if (!isfinite(ref[l][i]) || memcmp(&ref[l][i], &got[l][i], sizeof(float))) {
                fprintf(stderr, "%s: step %d layer %d element %d: drain %g gated %g\n",
                        name, step, l, i, ref[l][i], got[l][i]);
                return 0;
            }
        }
    }
    return 1;
}

/* STEPS steps, gated then drained; counts the gates committed. */
static int compare_modes(const char *name, layer_fn fn, int split, uint64_t *gated_layers,
                         uint64_t *split_gates, uint64_t *fallback) {
    uint64_t c0 = 0, s0 = 0, f0 = 0, c1 = 0, s1 = 0, f1 = 0;
    int failed = 0;
    ds4_gpu_stream_gate_stats(&c0, &s0, &f0, &failed);
    int ok = !failed;
    for (int step = 0; ok && step < STEPS; step++) {
        ok = write_routes(step);
        ds4_gpu_stream_gate_test_set_mode(1, split);
        ok = ok && run_step(fn, 1, got);
        ds4_gpu_stream_gate_test_set_mode(0, split);
        ok = ok && run_step(fn, 0, ref) && same_outputs(name, step);
    }
    ds4_gpu_stream_gate_stats(&c1, &s1, &f1, &failed);
    *gated_layers = c1 - c0;
    if (split_gates) *split_gates = s1 - s0;
    if (fallback) *fallback = f1 - f0;
    if (ok && failed) {
        fprintf(stderr, "%s: a gate failed\n", name);
        ok = 0;
    }
    /* Step 0 drains until the first miss allocates the cache slab. */
    if (ok && *gated_layers < (uint64_t)(STEPS - 1) * LAYERS) {
        fprintf(stderr, "%s: only %llu gated layers\n", name, (unsigned long long)*gated_layers);
        ok = 0;
    }
    fprintf(stderr, "%s: %llu gated layers: %s\n", name, (unsigned long long)*gated_layers,
            ok ? "PASS" : "FAIL");
    return ok;
}

static int qwen4_layer(int layer, int gated) {
    (void)gated;   /* ds4_gpu_stream_gate_test_set_mode picks the path */
    return ds4_gpu_qwen4_moe_stream_layer(
               mid, out_t[layer], xt, ids_t[layer], model, model_bytes, 3u + (uint32_t)layer,
               0, tensor_bytes, 2 * tensor_bytes, 16u, 10u,
               E, 1u, N, D, H, D, 0, 0, 0, UINT32_MAX, UINT32_MAX) != 0;
}

static int qwen4_suite(void) {
    uint64_t gated = 0, split = 0;
    ds4_gpu_set_glm_model(false);
    int ok = compare_modes("qwen4 one pass", qwen4_layer, 0, &gated, &split, NULL);
    ok = ok && compare_modes("qwen4 split", qwen4_layer, 1, &gated, &split, NULL);
    if (ok && split == 0) {
        fprintf(stderr, "qwen4 split: no split gate ran\n");
        ok = 0;
    }
    return ok;
}

int main(int argc, char **argv) {
    const char *mode = argc == 2 ? argv[1] : "";
    if (strcmp(mode, "--qwen4")) {
        fprintf(stderr, "usage: %s --qwen4\n", argv[0]);
        return 1;
    }
    int ok = setup();
    if (ok) ok = qwen4_suite();
    ds4_gpu_stream_gate_test_set_mode(-1, -1);
    ds4_gpu_cleanup();
    if (model && model != MAP_FAILED) munmap(model, model_bytes);
    if (model_file) fclose(model_file);
    fprintf(stderr, "Metal stream gate %s: %s\n", mode + 2, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
```

Add to `Makefile` after the `test-metal-ssd-experts` recipe lines:

```make
tests/test_metal_stream_gate.o: tests/test_metal_stream_gate.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_stream_gate: tests/test_metal_stream_gate.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: test-metal-stream-gate
test-metal-stream-gate: tests/test_metal_stream_gate
	./tests/test_metal_stream_gate --qwen4
```

and in `clean:` after `rm -f tests/test_metal_ssd_experts` add `rm -f tests/test_metal_stream_gate`.

- [ ] **Step 2: Verify RED (build only, no GPU)**

Run: `make tests/test_metal_stream_gate`
Expected: FAIL at compile or link, with `ds4_gpu_stream_gate_test_set_mode` and `ds4_gpu_stream_gate_stats`
undeclared or undefined.

- [ ] **Step 3: Add the hooks**

In `ds4_metal.m`, right after
`static double g_qgate_stat_miss_load_ms, g_qgate_stat_miss_count; /* split gates with misses */`
add:

```objc
static uint64_t g_qgate_stat_committed;      /* gates committed (main thread) */
/* tests/test_metal_stream_gate.c: -1 follows the env switches, 0/1 force
 * gates (both clients) or two-pass gates off/on. */
static int g_qgate_test_mode = -1;
static int g_qgate_test_split = -1;
```

In `qgate_requested()` add as the first line: `if (g_qgate_test_mode >= 0) return g_qgate_test_mode;`
In `qgate_split_requested()` add as the first line: `if (g_qgate_test_split >= 0) return g_qgate_test_split;`

In `qgate_encode`, right after `g_qgate_last_seq = seq;` add `g_qgate_stat_committed++;`.

After the `qgate_check_after_wait` function add:

```objc
void ds4_gpu_stream_gate_test_set_mode(int mode, int split) {
    g_qgate_test_mode = mode < 0 ? -1 : mode != 0;
    g_qgate_test_split = split < 0 ? -1 : split != 0;
}

void ds4_gpu_stream_gate_stats(uint64_t *committed, uint64_t *split, uint64_t *fallback, int *failed) {
    /* The service thread keeps counting after a gate's release: let it go idle. */
    for (;;) {
        pthread_mutex_lock(&g_qgate_mutex);
        const uint32_t pending = g_qgate_queue_count;
        pthread_mutex_unlock(&g_qgate_mutex);
        if (pending == 0) break;
        sched_yield();
    }
    if (committed) *committed = g_qgate_stat_committed;
    if (split) *split = g_qgate_stat_split_gates;
    if (fallback) *fallback = g_qgate_stat_fallback;
    if (failed) *failed = g_qgate_failed != 0;
}
```

In `ds4_gpu.h`, after the `#ifdef __APPLE__ ... ds4_gpu_stream_expert_cache_note_service_thread(void); #endif`
block add:

```c
#ifdef __APPLE__
/* Stream gates, tests only. mode: -1 follows DS4_QWEN4_STREAM_GATE /
 * DS4_GLM_STREAM_GATE, 0 forces the per-layer drain, 1 forces gates; split
 * likewise for two-pass gates. Stats are cumulative since start and are read
 * once the gate service thread is idle. */
void ds4_gpu_stream_gate_test_set_mode(int mode, int split);
void ds4_gpu_stream_gate_stats(uint64_t *committed, uint64_t *split,
                               uint64_t *fallback, int *failed);
#endif
```

- [ ] **Step 4: Build, then run when the GPU is free**

Run: `make ds4 tests/test_metal_stream_gate` (exit 0), then, when no peer holds the GPU,
`./tests/test_metal_stream_gate --qwen4`.
Expected: `qwen4 one pass: N gated layers: PASS` and `qwen4 split: N gated layers: PASS` (N ≥ 92), then
`Metal stream gate qwen4: PASS`.
If it FAILS, the existing qwen4 gates differ from the drain on this fixture. Stop and use
superpowers:systematic-debugging. Do not adjust the test to pass.

- [ ] **Step 5: Commit**

```bash
git add ds4_metal.m ds4_gpu.h tests/test_metal_stream_gate.c Makefile
git commit -m "test: qwen4 stream gates match the per-layer drain (model-free)

Adds test hooks to force gates and split gates on or off and to read gate
counters, and a synthetic IQ2_XXS/Q2_K expert test that runs four streamed
layers per batch through gates and through the drain, byte-identical.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 3: Shared stream-gate core (refactor, no behaviour change)

Splits `qgate_encode` into a model-neutral publish and commit. Adds a client tag to `qgate_req`, pulls the
mailbox wait and the previous-gate status check out of `qgate_service`, sizes the fallback per client, and
makes `end_commands` wait for an idle service thread. qwen4 keeps the same command stream.

**Files:**
- Modify: `ds4_metal.m` (qgate block ~52695-53600)

**Interfaces:**
- Consumes: the Task 2 hooks and the test.
- Produces (static, in `ds4_metal.m`):
  - Constants `QGATE_CLIENT_QWEN4 = 0` and `QGATE_CLIENT_GLM = 1`, and the field `uint32_t client;` in
    `qgate_req`.
  - `static const char *qgate_client_name(const qgate_req *r);` returns `"qwen4"` or `"GLM"`.
  - `static int qgate_wait_mailbox(const qgate_req *r, int32_t *unique_ids, uint32_t *n_unique, double *t_arrived);`
  - `static void qgate_check_prev(const qgate_req *r);`
  - `static void qgate_wait_idle(void);`
  - `static uint64_t qgate_publish(const ds4_gpu_tensor *selected, uint32_t n_sel);` returns 0 if nothing
    was encoded.
  - `static int qgate_commit(uint64_t seq, qgate_req *req);`
  - `static int qgate_setup(uint64_t slot_bytes, uint32_t fallback_slots, const char *label);`
  - `static uint32_t g_qgate_fallback_slots;`

- [ ] **Step 1: Baseline**

When the GPU is free, run `make tests/test_metal_stream_gate && ./tests/test_metal_stream_gate --qwen4`.
Expected: PASS. This is the refactor's safety net.

- [ ] **Step 2: Client tag and fallback slots**

At the top of the `qgate_req` struct (before `uint64_t seq;`) add:

```objc
    uint32_t client;                           /* QGATE_CLIENT_* */
```

Before `typedef struct {` of `qgate_req` add:

```objc
enum { QGATE_CLIENT_QWEN4 = 0, QGATE_CLIENT_GLM = 1 };
```

After `static uint64_t g_qgate_fallback_slot_bytes;` add
`static uint32_t g_qgate_fallback_slots;          /* fallback experts, per client at setup */`.

After `qgate_status_timed_out` add:

```objc
static const char *qgate_client_name(const qgate_req *r) {
    return r->client == QGATE_CLIENT_GLM ? "GLM" : "qwen4";
}
```

In `qgate_fallback`, replace `n > QGATE_MAX_IDS ||` with `n > g_qgate_fallback_slots ||`.

- [ ] **Step 3: Extract the mailbox wait and the previous-gate check**

Add before `qgate_service`:

```objc
/* Spins until the GPU has published gate r (every mailbox word carries its
 * tag once the committed batch's lines are in memory), then collects the
 * unique selected ids. 0: the gate never arrived (latched as a failure) or
 * an id is out of range. */
static int qgate_wait_mailbox(const qgate_req *r, int32_t *unique_ids, uint32_t *n_unique,
                              double *t_arrived) {
    const double t0 = ds4_gpu_now_ms();
    const uint32_t ring = (uint32_t)(r->seq % QGATE_RING);
    const uint32_t tag = (uint32_t)(r->seq & 0xffffu);
    volatile const uint32_t *mb = (volatile const uint32_t *)[g_qgate_mailbox contents] + (size_t)ring * QGATE_MAX_IDS;
    uint32_t spins = 0;
    for (;;) {
        uint32_t i = 0;
        while (i < r->n_sel && (mb[i] >> 16) == tag) i++;
        if (i == r->n_sel) break;
        if (++spins > (1u << 20)) {
            sched_yield();
            spins = 0;
            if (ds4_gpu_now_ms() - t0 > 20000.0) {
                fprintf(stderr, "ds4: %s stream gate %llu (layer %u) never arrived\n",
                        qgate_client_name(r), (unsigned long long)r->seq, r->layer);
                g_qgate_failed = 1;
                break;
            }
        }
    }
    *t_arrived = ds4_gpu_now_ms();
    g_qgate_seen_host[ring] = ds4_gpu_host_seconds();
    uint8_t seen[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    memset(seen, 0, sizeof(seen));
    *n_unique = 0;
    int ok = !g_qgate_failed;
    for (uint32_t i = 0; ok && i < r->n_sel; i++) {
        const uint32_t e = mb[i] & 0xffffu;
        if (e >= r->n_total_expert) { ok = 0; break; }
        if (!seen[e]) { seen[e] = 1; unique_ids[(*n_unique)++] = (int32_t)e; }
    }
    return ok;
}

/* A poll that timed out let the GPU run the previous gate's experts on a
 * stale address table. Its buffer has completed by now, so the status is
 * final. Also feeds the poll statistics. */
static void qgate_check_prev(const qgate_req *r) {
    if (r->seq <= 1u) return;
    const uint32_t prev = (uint32_t)((r->seq - 1u) % QGATE_RING);
    const uint32_t *st = (const uint32_t *)[g_qgate_status contents];
    for (uint32_t k = 0; k < (g_qgate_ring_split[prev] == 2u ? 3u : g_qgate_ring_split[prev] ? 2u : 1u); k++) {
        const uint32_t line = __atomic_load_n(&st[(k * QGATE_RING + prev) * 2u], __ATOMIC_ACQUIRE);
        /* slot 1: second polls of gates without misses, 2: with misses */
        const uint32_t slot = k == 0 ? 0u : (g_qgate_ring_missed[prev] ? 2u : 1u);
        if (line != 0xffffffffu) {
            g_qgate_stat_poll_n[slot]++;
            g_qgate_stat_poll_line[slot] += (double)line;
            if (line >= 32u) g_qgate_stat_poll_waited[slot]++;
        }
    }
    if (qgate_status_timed_out(prev) ||
        (g_qgate_ring_split[prev] && qgate_status_timed_out(QGATE_RING + prev)) ||
        (g_qgate_ring_split[prev] == 2u && qgate_status_timed_out(2u * QGATE_RING + prev))) {
        fprintf(stderr, "ds4: %s stream gate %llu poll timed out\n", qgate_client_name(r),
                (unsigned long long)(r->seq - 1u));
        g_qgate_failed = 1;
    }
}
```

In `qgate_service`, replace everything from its first line `const double t0 = ds4_gpu_now_ms();` through
the end of the id-dedupe loop (the loop ending with
`if (!seen[e]) { seen[e] = 1; unique_ids[n_unique++] = (int32_t)e; }` and its closing brace) with:

```objc
    const double t0 = ds4_gpu_now_ms();
    const uint32_t ring = (uint32_t)(r->seq % QGATE_RING);
    int32_t unique_ids[QGATE_MAX_IDS];
    uint32_t n_unique = 0;
    double t1 = t0;
    int ok = qgate_wait_mailbox(r, unique_ids, &n_unique, &t1);
    const uint64_t guard_clock = g_stream_expert_cache_clock;
```

Then replace the whole block that starts with the comment "A poll that timed out let the GPU run the
previous layer's experts" and ends after `g_qgate_failed = 1; } }` (the `if (r->seq > 1u) { ... }` block)
with `qgate_check_prev(r);`.

- [ ] **Step 4: Publish and commit, and the idle wait**

Add before `qgate_encode`:

```objc
/* The caller of end_commands owns the expert cache afterwards: wait until
 * the service thread has finished the bookkeeping it does after a release. */
static void qgate_wait_idle(void) {
    for (;;) {
        pthread_mutex_lock(&g_qgate_mutex);
        const uint32_t pending = g_qgate_queue_count;
        pthread_mutex_unlock(&g_qgate_mutex);
        if (pending == 0) return;
        sched_yield();
    }
}

/* Encodes the publish kernel of a new gate into the open batch: the selected
 * ids go to the gate's mailbox slot, each word tagged with its sequence. */
static uint64_t qgate_publish(const ds4_gpu_tensor *selected, uint32_t n_sel) {
    id<MTLBuffer> selbuf = ds4_gpu_tensor_buffer(selected);
    if (!selbuf || ds4_gpu_tensor_bytes(selected) < (uint64_t)n_sel * sizeof(int32_t)) return 0;
    const uint64_t seq = ++g_qgate_seq;
    const uint32_t ring = (uint32_t)(seq % QGATE_RING);
    const uint32_t tag = (uint32_t)(seq & 0xffffu);
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(g_batch_cb);
        [enc setComputePipelineState:ds4_gpu_get_pipeline("kernel_qwen4_stream_gate_publish")];
        [enc setBuffer:selbuf offset:ds4_gpu_tensor_offset(selected) atIndex:0];
        [enc setBuffer:g_qgate_mailbox offset:(NSUInteger)ring * QGATE_MAX_IDS * sizeof(uint32_t) atIndex:1];
        [enc setBytes:&n_sel length:sizeof(n_sel) atIndex:2];
        [enc setBytes:&tag length:sizeof(tag) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(QGATE_MAX_IDS, 1, 1)];
        ds4_gpu_end_compute_encoder(g_batch_cb, enc);
    }
    return seq;
}

/* Commits the batch that ends in gate seq's publish (committing is what
 * writes the mailbox back to memory), opens the next batch with the gate's
 * poll and queues its service. */
static int qgate_commit(uint64_t seq, qgate_req *req) {
    const uint32_t ring = (uint32_t)(seq % QGATE_RING);
    const uint32_t value = (uint32_t)seq;
    const uint32_t nlines = QGATE_POLL_LINES;
    if (!ds4_gpu_flush_commands()) return 0;
    @autoreleasepool {
        /* First work of the new batch, so no line of its region is cached. */
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(g_batch_cb);
        [enc setComputePipelineState:ds4_gpu_get_pipeline("kernel_dsv4_tp_poll_release")];
        [enc setBuffer:g_qgate_region offset:(NSUInteger)ring * QGATE_POLL_LINES * QGATE_POLL_LINE_BYTES atIndex:0];
        [enc setBytes:&value length:sizeof(value) atIndex:1];
        [enc setBytes:&nlines length:sizeof(nlines) atIndex:2];
        [enc setBuffer:g_qgate_status offset:(NSUInteger)ring * 2u * sizeof(uint32_t) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(g_batch_cb, enc);
        ds4_gpu_close_batch_encoder();
    }
    g_qgate_ring_split[ring] = (uint8_t)(req->split ? (req->staged ? 2u : 1u) : 0u);
    req->seq = seq;
    pthread_mutex_lock(&g_qgate_mutex);
    const uint32_t tail = (g_qgate_queue_head + g_qgate_queue_count) % QGATE_QUEUE;
    g_qgate_queue[tail] = *req;
    g_qgate_queue_count++;
    pthread_cond_signal(&g_qgate_cond);
    pthread_mutex_unlock(&g_qgate_mutex);
    g_qgate_last_seq = seq;
    g_qgate_stat_committed++;
    return 1;
}
```

Rewrite the body of `qgate_encode`, after the queue-full check and up to (not including) the residency loop
`uint32_t n = 0; for (uint32_t s = 0; ...`, as follows. The order of GPU work stays the same: publish,
lookahead hidden copy, flush, poll.

```objc
    const uint64_t seq = qgate_publish(selected, n_sel);
    if (!seq) return 0;
    const uint32_t ring = (uint32_t)(seq % QGATE_RING);
    /* Lookahead: copy this layer's router input for the service thread. */
    const uint32_t la_floats = n_rows * in_dim;
    const int la = g_qgate_la.top && x && la_floats && la_floats <= QGATE_HIDDEN_MAX &&
                   ds4_gpu_tensor_buffer(x) != nil &&
                   ds4_gpu_tensor_bytes(x) >= (uint64_t)la_floats * sizeof(float);
    if (la) @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(g_batch_cb);
        [enc setComputePipelineState:ds4_gpu_get_pipeline("kernel_qwen4_stream_gate_hidden")];
        [enc setBuffer:ds4_gpu_tensor_buffer(x) offset:ds4_gpu_tensor_offset(x) atIndex:0];
        [enc setBuffer:g_qgate_hidden offset:(NSUInteger)ring * QGATE_HIDDEN_MAX * sizeof(float) atIndex:1];
        [enc setBytes:&la_floats length:sizeof(la_floats) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((la_floats + 255u) / 256u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(g_batch_cb, enc);
    }
    if (getenv("DS4_QWEN4_STREAM_TIMING"))
        objc_setAssociatedObject(g_batch_cb, &kQgateSeqKey, @(seq), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    const uint32_t n_slabs = g_stream_expert_cache_slab_count;
    const int split = qgate_split_requested() && n_total_expert <= QGATE_MAX_EXPERT;
    const int staged = split && qgate_staged_requested();
    qgate_req req = {
        .client = QGATE_CLIENT_QWEN4,
        .model_map = model_map, .model_size = model_size,
        .gate_offset = gate_offset, .up_offset = up_offset, .down_offset = down_offset,
        .gate_expert_bytes = gate_expert_bytes, .down_expert_bytes = down_expert_bytes,
        .layer = layer, .n_sel = n_sel, .n_total_expert = n_total_expert, .n_slabs = n_slabs,
        .split = split, .staged = staged,
        .pf_layer = la ? g_qgate_la.layer : 0u, .pf_rows = la ? n_rows : 0u,
        .pf_in_dim = la ? in_dim : 0u, .pf_top = la ? g_qgate_la.top : 0u,
        .pf_router_offset = g_qgate_la.router_offset, .pf_gate_offset = g_qgate_la.gate_offset,
        .pf_up_offset = g_qgate_la.up_offset, .pf_down_offset = g_qgate_la.down_offset,
    };
    if (!qgate_commit(seq, &req)) return 0;
```

Delete the now-unused locals in `qgate_encode`: `selbuf` and its size check, `tag`, `value` and `nlines`.
Delete the old publish, flush, poll and enqueue code as well, together with the `g_qgate_stat_committed++`
from Task 2. Keep the residency loop and the outputs. Their `*split_out` line uses `split` and `staged`
exactly as before.

- [ ] **Step 5: Setup takes the fallback size and a label; end_commands waits for idle**

Change the definition to `static int qgate_setup(uint64_t slot_bytes, uint32_t fallback_slots, const char *label) {`. Then:

- Replace `if (g_qgate_thread_running) return g_qgate_fallback_slot_bytes >= slot_bytes;` with
  `if (g_qgate_thread_running) return g_qgate_fallback_slot_bytes >= slot_bytes && g_qgate_fallback_slots >= fallback_slots;`.
- After `g_qgate_fallback_slot_bytes = round_up_u64(...)`, add `g_qgate_fallback_slots = fallback_slots;`.
- In the fallback allocation, replace `(NSUInteger)(QGATE_MAX_IDS * g_qgate_fallback_slot_bytes)` with
  `(NSUInteger)((uint64_t)fallback_slots * g_qgate_fallback_slot_bytes)`.
- Change the three messages to use the label:
  - `"ds4: %s stream gates unavailable; keeping the per-layer drain\n", label`
  - `"ds4: %s stream gate thread failed to start; keeping the per-layer drain\n", label`
  - `"ds4: %s stream gates on (fallback %.1f MiB)\n", label, (double)fallback_slots * (double)g_qgate_fallback_slot_bytes / 1048576.0`

In `qgate_encode`, change the call to
`qgate_setup(gate_expert_bytes * 2ull + down_expert_bytes, QGATE_MAX_IDS, "qwen4")`.

In `qgate_check_after_wait`, after `if (!g_qgate_thread_running) return 1;`, add `qgate_wait_idle();`.

In `ds4_gpu_stream_gate_stats`, replace the open-coded wait loop with `if (g_qgate_thread_running) qgate_wait_idle();`.

- [ ] **Step 6: Build and run**

Run: `make ds4 ds4-server tests/test_metal_stream_gate tests/test_metal_ssd_experts`.
Expected: exit 0, no new warnings.

When the GPU is free, run `./tests/test_metal_stream_gate --qwen4 && make test-metal-ssd-experts`.
Expected: `Metal stream gate qwen4: PASS`, and every `Metal SSD ... PASS` line as before.

- [ ] **Step 7: Commit**

```bash
git add ds4_metal.m
git commit -m "metal: shared stream-gate core (publish, commit, mailbox wait, status check)

qgate_encode becomes qgate_publish + qgate_commit with the same command
stream; the service's mailbox wait and previous-gate check move into
helpers; requests carry a client tag; the fallback size is set per client;
end_commands also waits for the service thread's post-release bookkeeping
so its caller owns the cache. qwen4 output unchanged
(test_metal_stream_gate --qwen4).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 4: GLM gates, one pass (Metal backend)

**Files:**
- Modify: `ds4_metal.m`:
  - near the slab globals (~1030);
  - the forward declarations near `static int qgate_check_after_wait(void);` (~11829);
  - `ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu` (~33540) and `ds4_gpu_encode_mul_mv_addr_iq2` (~33619);
  - `ds4_gpu_routed_moe_one_tensor` (~41347-44030);
  - the qgate block.
- Modify: `ds4_gpu.h` (the Apple block from Task 2)
- Modify: `tests/test_metal_stream_gate.c`, `Makefile`

**Interfaces:**
- Consumes (Task 3): `qgate_publish`, `qgate_commit`, `qgate_wait_mailbox`, `qgate_check_prev`, `qgate_setup`,
  `QGATE_CLIENT_GLM`.
- Produces (ds4_gpu.h, Apple):
  - `int ds4_gpu_glm_stream_gate_publish(const ds4_gpu_stream_expert_table *table, const ds4_gpu_tensor *selected, uint32_t n_selected);`
    Returns 1 when published. The caller must then call commit and dispatch `table->layer`'s routed MoE.
    Returns 0 when it is not available; use the drain.
  - `int ds4_gpu_glm_stream_gate_commit(void);`
  - `void ds4_gpu_stream_gate_test_force_fallback(int on);`
- Produces (static): `qgate_glm_pending_for(uint32_t layer)`,
  `qgate_glm_take(uint32_t layer, uint64_t *seq, int *split)`,
  `qgate_glm_tab(uint64_t seq, uint32_t pass, uint32_t kind)` returning `id<MTLBuffer>`,
  `qgate_set_dispatch_resident(int on)`, `glm_gate_split_requested()` (returns 0 until Task 6).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metal_stream_gate.c`, before `main`:

```c
static int drain_layer = -1;   /* this layer never publishes (mixed tokens) */

static int glm_layer(int layer, int gated) {
    const ds4_gpu_stream_expert_table table = {
        .model_map = model, .model_size = model_bytes, .layer = 3u + (uint32_t)layer,
        .n_total_expert = E, .gate_offset = 0, .up_offset = tensor_bytes,
        .down_offset = 2 * tensor_bytes, .gate_expert_bytes = expert_bytes,
        .down_expert_bytes = down_expert_bytes,
    };
    const int published = gated && layer != drain_layer &&
                          ds4_gpu_glm_stream_gate_publish(&table, ids_t[layer], N);
    /* GPU work between publish and commit, where ds4.c encodes the shared expert. */
    if (!ds4_gpu_add_tensor(shared_t, xt, xt, D)) return 0;
    if (published && !ds4_gpu_glm_stream_gate_commit()) return 0;
    return ds4_gpu_routed_moe_one_tensor(
               out_t[layer], gate, up, mid, down, model, model_bytes,
               0, tensor_bytes, 2 * tensor_bytes, 16u, 10u,
               expert_bytes, row_bytes, down_expert_bytes, down_row_bytes, D, H, D,
               ids_t[layer], wt, E, N, 7.0f, xt, NULL, 3u + (uint32_t)layer, false) != 0;
}

/* A publish whose layer fails before commit must not leak into the next batch. */
static int check_uncommitted_publish(int split) {
    const ds4_gpu_stream_expert_table table = {
        .model_map = model, .model_size = model_bytes, .layer = 3u,
        .n_total_expert = E, .gate_offset = 0, .up_offset = tensor_bytes,
        .down_offset = 2 * tensor_bytes, .gate_expert_bytes = expert_bytes,
        .down_expert_bytes = down_expert_bytes,
    };
    ds4_gpu_stream_gate_test_set_mode(1, split);
    int ok = write_routes(STEPS) && ds4_gpu_begin_commands();
    const int published = ok && ds4_gpu_glm_stream_gate_publish(&table, ids_t[0], N);
    ok = ds4_gpu_end_commands() && ok && published;
    ok = ok && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_set_mode(0, split);
    ok = ok && run_step(glm_layer, 0, ref) && same_outputs("glm uncommitted publish", STEPS);
    fprintf(stderr, "glm uncommitted publish: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int glm_suite(int split) {
    const char *tag = split ? "glm split" : "glm one pass";
    char name[64];
    uint64_t gated = 0, splits = 0, fallback = 0;
    ds4_gpu_set_glm_model(true);
    out_elems = D;
    int ok = compare_modes(tag, glm_layer, split, &gated, &splits, &fallback);
    if (ok && split && splits == 0) {
        fprintf(stderr, "%s: no split gate ran\n", tag);
        ok = 0;
    }
    /* Mixed token: layer 2 drains between gated layers. */
    snprintf(name, sizeof(name), "%s mixed", tag);
    drain_layer = 2;
    uint64_t mixed = 0;
    ok = ok && compare_modes(name, glm_layer, split, &mixed, NULL, NULL);
    drain_layer = -1;
    /* Fallback buffer: every miss (and, split, every hit) treated as uncached. */
    snprintf(name, sizeof(name), "%s fallback", tag);
    ds4_gpu_stream_gate_test_force_fallback(1);
    ok = ok && compare_modes(name, glm_layer, split, &gated, NULL, &fallback);
    ds4_gpu_stream_gate_test_force_fallback(0);
    if (ok && fallback == 0) {
        fprintf(stderr, "%s: the fallback buffer was never used\n", name);
        ok = 0;
    }
    ok = ok && check_uncommitted_publish(split);
    /* A larger budget adds a slab: gates pause until it exists, then resume. */
    snprintf(name, sizeof(name), "%s budget growth", tag);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET + 16);
    ok = ok && compare_modes(name, glm_layer, split, &gated, NULL, NULL);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    return ok;
}
```

The mixed section expects ≥ `(STEPS - 1) * LAYERS` gated layers, but one layer in four is drained.
Change the threshold in `compare_modes` to account for this:

```c
    const uint64_t want = (uint64_t)(STEPS - 1) * (drain_layer >= 0 ? LAYERS - 1 : LAYERS);
    if (ok && *gated_layers < want) {
```

(and keep the message). `drain_layer` must be declared above `compare_modes`. Move the
`static int drain_layer = -1;` line to just after the `ref`/`got` arrays.

Replace `main`'s mode check and dispatch with:

```c
    const char *mode = argc == 2 ? argv[1] : "";
    if (strcmp(mode, "--qwen4") && strcmp(mode, "--glm")) {
        fprintf(stderr, "usage: %s --qwen4 | --glm\n", argv[0]);
        return 1;
    }
    int ok = setup();
    if (ok && !strcmp(mode, "--qwen4")) ok = qwen4_suite();
    if (ok && !strcmp(mode, "--glm")) ok = glm_suite(0);
```

Makefile: add `./tests/test_metal_stream_gate --glm` to the `test-metal-stream-gate` recipe.

In `ds4_gpu.h`, inside the Apple block from Task 2, add:

```c
/* GLM decode gates (SP1). publish: 1 when the layer's routed selection was
 * published into the open batch; the caller then encodes the shared expert,
 * calls commit and dispatches table->layer's routed MoE, which reads the
 * gate's own address tables. 0: gates are off or not ready; drain as before.
 * DS4_GLM_STREAM_GATE=0 keeps the drain. */
int ds4_gpu_glm_stream_gate_publish(const ds4_gpu_stream_expert_table *table,
                                    const ds4_gpu_tensor *selected,
                                    uint32_t n_selected);
int ds4_gpu_glm_stream_gate_commit(void);
/* Tests: resolve every gated expert through the fallback buffer. */
void ds4_gpu_stream_gate_test_force_fallback(int on);
```

- [ ] **Step 2: Verify RED (build only)**

Run: `make tests/test_metal_stream_gate`
Expected: link FAIL with `ds4_gpu_glm_stream_gate_publish`, `ds4_gpu_glm_stream_gate_commit` and
`ds4_gpu_stream_gate_test_force_fallback` undefined.

- [ ] **Step 3: Residency hook (encode helpers)**

In `ds4_metal.m`, right after `static uint32_t g_stream_expert_cache_slab_total_slots;` (~1030) add:

```objc
/* Stream-gated expert dispatches cannot name their experts at encode time
 * (a service thread resolves them after commit): every cache slab and the
 * gate fallback buffer are made resident for those dispatches instead. */
static __unsafe_unretained id<MTLResource> g_qgate_dispatch_resident[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS + 1u];
static uint32_t g_qgate_dispatch_resident_n;
```

In `ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu`:
- Replace `(n_entries == 0 && !overflow_gate) ||` with
  `(n_entries == 0 && !overflow_gate && g_qgate_dispatch_resident_n == 0) ||`.
- Replace `if (!ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,` with
  `if (n_entries && !ds4_gpu_stream_expert_cache_mark_entries_inflight(entries,` (same arguments).
- After `[enc setComputePipelineState:pipeline];` add:

```objc
    if (g_qgate_dispatch_resident_n)
        [enc useResources:g_qgate_dispatch_resident count:g_qgate_dispatch_resident_n usage:MTLResourceUsageRead];
```

In `ds4_gpu_encode_mul_mv_addr_iq2`:
- Replace `(n_entries == 0 && !overflow_resource) ||` with
  `(n_entries == 0 && !overflow_resource && g_qgate_dispatch_resident_n == 0) ||`.
- Add the same `useResources` lines after its `setComputePipelineState`.

Next to `static int qgate_check_after_wait(void);` (~11829) add the forward declarations:

```objc
static int qgate_glm_pending_for(uint32_t layer);
static int qgate_glm_take(uint32_t layer, uint64_t *seq, int *split);
static id<MTLBuffer> qgate_glm_tab(uint64_t seq, uint32_t pass, uint32_t kind);
static void qgate_set_dispatch_resident(int on);
```

- [ ] **Step 4: GLM client state, tables and service (qgate block)**

Right after the Task 2 test statics (`static int g_qgate_test_split = -1;`) add:

```objc
/*
 * SP1: GLM-5.3 decode gates. ds4.c publishes a streamed layer's selection
 * right after its router, encodes the shared expert, then commits; the
 * layer's routed MoE opens with the poll and reads this gate's own address
 * tables: pass 1 holds every expert (or only the cached ones when the gate
 * is split), pass 2 the misses. An address of 0 makes the routed kernels skip
 * that slot, so each (slot, row) is computed by exactly one pass with
 * unchanged arithmetic and the expert sum still adds the slots in order.
 * DS4_GLM_STREAM_GATE=0 keeps the per-layer drain.
 */
#define QGATE_GLM_FALLBACK_SLOTS 8u
static id<MTLBuffer> g_qgate_glmtab[QGATE_RING][2][3];   /* ring x pass x gate/up/down */
static int g_qgate_glm_ready;
static struct {
    int published;                 /* publish encoded, commit pending */
    int committed;                 /* committed, routed MoE not encoded yet */
    uint64_t seq;
    int split;
    uint32_t n_selected;
    ds4_gpu_stream_expert_table table;
} g_qgate_glm;
static int g_qgate_test_force_fallback;

static int glm_gate_requested(void) {
    if (g_qgate_test_mode >= 0) return g_qgate_test_mode;
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_GLM_STREAM_GATE");
        v = !(e && e[0] == '0');
    }
    return v;
}

/* Two-pass GLM gates arrive in a later change. */
static int glm_gate_split_requested(void) {
    return 0;
}

static id<MTLBuffer> qgate_glm_tab(uint64_t seq, uint32_t pass, uint32_t kind) {
    return g_qgate_glmtab[seq % QGATE_RING][pass][kind];
}

/* Writes one expert's gate/up/down addresses into a gate table; nil
 * buffers write 0 (the routed kernels skip that slot). */
static int qgate_glmtab_write(uint32_t ring, uint32_t pass, uint32_t expert,
                              id<MTLBuffer> gb, NSUInteger gi, id<MTLBuffer> ub, NSUInteger ui,
                              id<MTLBuffer> db, NSUInteger di) {
    if (expert >= QGATE_MAX_EXPERT) return 0;
    const uint64_t g = gb ? ds4_gpu_buffer_address(gb, gi) : 0;
    const uint64_t u = ub ? ds4_gpu_buffer_address(ub, ui) : 0;
    const uint64_t d = db ? ds4_gpu_buffer_address(db, di) : 0;
    if (gb && (!g || !u || !d)) return 0;
    ((uint64_t *)[g_qgate_glmtab[ring][pass][0] contents])[expert] = g;
    ((uint64_t *)[g_qgate_glmtab[ring][pass][1] contents])[expert] = u;
    ((uint64_t *)[g_qgate_glmtab[ring][pass][2] contents])[expert] = d;
    return 1;
}

void ds4_gpu_stream_gate_test_force_fallback(int on) {
    g_qgate_test_force_fallback = on != 0;
}
```

In `qgate_fallback`, replace the `const int set = r->split ? ... ;` expression with one that handles GLM
first:

```objc
        const int set = r->client == QGATE_CLIENT_GLM ?
            qgate_glmtab_write(ring, r->split ? 1u : 0u, (uint32_t)ids[i],
                g_qgate_fallback, off, g_qgate_fallback, off + (NSUInteger)r->gate_expert_bytes,
                g_qgate_fallback, off + (NSUInteger)(2u * r->gate_expert_bytes)) :
            r->split ?
            qgate_misstab_set(ring, (uint32_t)ids[i],
                g_qgate_fallback, off, g_qgate_fallback, off + (NSUInteger)r->gate_expert_bytes,
                g_qgate_fallback, off + (NSUInteger)(2u * r->gate_expert_bytes)) :
            ds4_gpu_stream_expert_cache_set_addr_slot_raw(r->layer, (uint32_t)ids[i],
                g_qgate_fallback, off, g_qgate_fallback, off + (NSUInteger)r->gate_expert_bytes,
                g_qgate_fallback, off + (NSUInteger)(2u * r->gate_expert_bytes));
```

After `qgate_service` (before `qgate_thread_main`) add the GLM service:

```objc
static void glm_gate_service(const qgate_req *r) {
    const double t0 = ds4_gpu_now_ms();
    const uint32_t ring = (uint32_t)(r->seq % QGATE_RING);
    int32_t unique_ids[QGATE_MAX_IDS];
    uint32_t n_unique = 0;
    double t1 = t0;
    int ok = qgate_wait_mailbox(r, unique_ids, &n_unique, &t1);
    uint8_t is_miss[QGATE_MAX_IDS];
    memset(is_miss, 0, sizeof(is_miss));
    uint32_t n_miss = 0;
    int released_a = 0;
    if (ok) {
        /* The drain path does both at the top of the routed MoE. */
        ds4_gpu_stream_expert_cache_note_token(r->layer);
        ds4_gpu_stream_expert_cache_note_selected_hotness(r->layer, unique_ids, n_unique);
    }
    if (ok && r->split) {
        /* Pass 1 gets the cached experts (0 for the misses), pass 2 the
         * reverse; release pass 1 before any read. */
        for (uint32_t i = 0; ok && i < n_unique; i++) {
            const uint64_t eid = (uint32_t)unique_ids[i];
            const ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[r->layer][eid];
            const int hit = !g_qgate_test_force_fallback &&
                ds4_gpu_stream_expert_cache_entry_matches(e, r->model_map, r->model_size,
                    r->gate_offset + eid * r->gate_expert_bytes,
                    r->up_offset + eid * r->gate_expert_bytes,
                    r->down_offset + eid * r->down_expert_bytes,
                    r->gate_expert_bytes, r->down_expert_bytes) &&
                qgate_entry_resident(e, r->n_slabs);
            is_miss[i] = (uint8_t)!hit;
            if (!hit) n_miss++;
            ok = hit ?
                qgate_glmtab_write(ring, 0u, (uint32_t)eid, e->gate_buffer, e->gate_inner,
                                   e->up_buffer, e->up_inner, e->down_buffer, e->down_inner) &&
                qgate_glmtab_write(ring, 1u, (uint32_t)eid, nil, 0, nil, 0, nil, 0) :
                qgate_glmtab_write(ring, 0u, (uint32_t)eid, nil, 0, nil, 0, nil, 0);
        }
        if (ok) {
            qgate_release_lines(ring, (uint32_t)r->seq, 0, QGATE_RELEASE_HEAD);
            released_a = 1;
        }
    }
    g_qgate_ring_missed[ring] = (uint8_t)(n_miss != 0);
    if (ok && (!r->split || n_miss != 0)) {
        ds4_gpu_stream_expert_cache_entry *entries[QGATE_MAX_IDS];
        g_glm_stream_expert_addr_table_building++;
        const int resolved = ds4_gpu_stream_expert_cache_load_batch(r->model_map, r->model_size, r->layer,
                                 unique_ids, n_unique, r->n_total_expert, r->gate_offset, r->up_offset,
                                 r->down_offset, r->gate_expert_bytes, r->down_expert_bytes,
                                 NULL, NULL, entries);
        g_glm_stream_expert_addr_table_building--;
        const uint32_t pass = r->split ? 1u : 0u;
        int32_t fb_ids[QGATE_MAX_IDS];
        uint32_t n_fb = 0;
        for (uint32_t i = 0; i < n_unique; i++) {
            /* A released (cached) expert needs nothing more: this gate's ids
             * keep it from being chosen as a victim. */
            if (r->split && !is_miss[i]) continue;
            ds4_gpu_stream_expert_cache_entry *e = resolved ? entries[i] : NULL;
            if (!g_qgate_test_force_fallback && e && qgate_entry_resident(e, r->n_slabs) &&
                qgate_glmtab_write(ring, pass, (uint32_t)unique_ids[i], e->gate_buffer, e->gate_inner,
                                   e->up_buffer, e->up_inner, e->down_buffer, e->down_inner)) {
                continue;
            }
            fb_ids[n_fb++] = unique_ids[i];
        }
        if (n_fb) {
            g_qgate_stat_fallback += n_fb;
            if (!qgate_fallback(r, fb_ids, n_fb)) {
                fprintf(stderr, "ds4: GLM stream gate layer %u: cache and fallback reads both failed\n",
                        r->layer);
                ok = 0;
            }
        }
    }
    if (!ok) g_qgate_failed = 1;
    qgate_check_prev(r);
    /* Release even on failure so the GPU drains; the failure surfaces at the
     * token's end_commands. */
    if (!released_a) qgate_release_lines(ring, (uint32_t)r->seq, 0, QGATE_RELEASE_HEAD);
    if (r->split) qgate_release_lines(QGATE_RING + ring, (uint32_t)r->seq, 0, QGATE_RELEASE_HEAD);
    qgate_release_lines(ring, (uint32_t)r->seq, QGATE_RELEASE_HEAD, QGATE_POLL_LINES);
    if (r->split) qgate_release_lines(QGATE_RING + ring, (uint32_t)r->seq, QGATE_RELEASE_HEAD, QGATE_POLL_LINES);
    if (ok && !g_qgate_failed) {
        if (r->split && n_miss == 0) {
            /* Count the hits and refresh their recency, as load_batch does. */
            ds4_gpu_stream_expert_cache_entry *entries[QGATE_MAX_IDS];
            (void)ds4_gpu_stream_expert_cache_load_batch(r->model_map, r->model_size, r->layer,
                      unique_ids, n_unique, r->n_total_expert, r->gate_offset, r->up_offset,
                      r->down_offset, r->gate_expert_bytes, r->down_expert_bytes, NULL, NULL, entries);
        }
        ds4_gpu_stream_expert_cache_prune_layer(r->layer, r->n_total_expert, n_unique, unique_ids, n_unique);
        ds4_gpu_stream_expert_cache_prune_global(r->layer, unique_ids, n_unique);
    }
    g_qgate_stat_gates++;
    if (r->split) {
        g_qgate_stat_split_gates++;
        if (n_miss) g_qgate_stat_split_miss_gates++;
    }
    g_qgate_stat_wait_ms += t1 - t0;
    g_qgate_stat_service_ms += ds4_gpu_now_ms() - t1;
    if (getenv("DS4_GLM_STREAM_TIMING") && g_qgate_stat_gates % 420u == 0u) {
        fprintf(stderr, "ds4: GLM stream gates %llu: avg service %.1f us, gpu arrival wait %.1f us, "
                "split %.1f%% with misses, fallback experts %llu\n",
                (unsigned long long)g_qgate_stat_gates,
                g_qgate_stat_service_ms / (double)g_qgate_stat_gates * 1000.0,
                g_qgate_stat_wait_ms / (double)g_qgate_stat_gates * 1000.0,
                g_qgate_stat_split_gates ?
                    100.0 * (double)g_qgate_stat_split_miss_gates / (double)g_qgate_stat_split_gates : 0.0,
                (unsigned long long)g_qgate_stat_fallback);
    }
}
```

In `qgate_thread_main`, replace `@autoreleasepool { qgate_service(&r); }` with:

```objc
        @autoreleasepool {
            if (r.client == QGATE_CLIENT_GLM) glm_gate_service(&r);
            else qgate_service(&r);
        }
```

After `qgate_encode_poll`, add the GLM entry points:

```objc
static int glm_gate_setup(uint64_t slot_bytes) {
    if (g_qgate_glm_ready) return g_qgate_fallback_slot_bytes >= slot_bytes;
    if (!qgate_setup(slot_bytes, QGATE_GLM_FALLBACK_SLOTS, "GLM")) return 0;
    for (uint32_t r = 0; r < QGATE_RING; r++) {
        for (uint32_t p = 0; p < 2u; p++) {
            for (uint32_t k = 0; k < 3u; k++) {
                id<MTLBuffer> b = [g_device newBufferWithLength:QGATE_MAX_EXPERT * sizeof(uint64_t)
                                                        options:MTLResourceStorageModeShared];
                if (!b) {
                    fprintf(stderr, "ds4: GLM stream gate tables unavailable; keeping the per-layer drain\n");
                    g_qgate_failed = 1;
                    g_qgate_failed_reported = 1;
                    return 0;
                }
                memset([b contents], 0, [b length]);
                b.label = @"ds4_glm_gate_addrs";
                g_qgate_glmtab[r][p][k] = b;
            }
        }
    }
    g_qgate_glm_ready = 1;
    return 1;
}

int ds4_gpu_glm_stream_gate_publish(const ds4_gpu_stream_expert_table *table,
                                    const ds4_gpu_tensor *selected, uint32_t n_selected) {
    if (!table || !selected || g_qgate_glm.published || g_qgate_glm.committed ||
        !g_ssd_streaming_mode || !glm_gate_requested() || g_qgate_failed || !g_batch_cb ||
        g_tp_split_world > 1 ||
        n_selected == 0 || n_selected > QGATE_GLM_FALLBACK_SLOTS ||
        table->layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        table->n_total_expert == 0 || table->n_total_expert > QGATE_MAX_EXPERT ||
        /* These diagnostics read the selection on the host. */
        getenv("DS4_MOE_RECORD_SELECTED_IDS") || getenv("DS4_MOE_REPLAY_SELECTED_IDS") ||
        getenv("DS4_MOE_RECORD_SELECTED_HOTLIST")) {
        return 0;
    }
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (!ds4_gpu_stream_expert_slab_enabled() || g_stream_expert_cache_slab_count == 0 ||
        budget < n_selected || g_stream_expert_cache_slab_total_slots < budget ||
        !ds4_gpu_stream_expert_cache_note_expert_size(table->gate_expert_bytes,
                                                      table->down_expert_bytes)) {
        return 0;   /* every slab must exist: ids are unknown at encode time */
    }
    if (!glm_gate_setup(table->gate_expert_bytes * 2ull + table->down_expert_bytes)) return 0;
    pthread_mutex_lock(&g_qgate_mutex);
    const int full = g_qgate_queue_count >= QGATE_QUEUE;
    pthread_mutex_unlock(&g_qgate_mutex);
    if (full) return 0;
    const uint64_t seq = qgate_publish(selected, n_selected);
    if (!seq) return 0;
    g_qgate_glm.published = 1;
    g_qgate_glm.seq = seq;
    g_qgate_glm.n_selected = n_selected;
    g_qgate_glm.table = *table;
    return 1;
}

int ds4_gpu_glm_stream_gate_commit(void) {
    if (!g_qgate_glm.published) return 0;
    g_qgate_glm.published = 0;
    const ds4_gpu_stream_expert_table *t = &g_qgate_glm.table;
    const int split = glm_gate_split_requested();
    qgate_req req = {
        .client = QGATE_CLIENT_GLM,
        .model_map = t->model_map, .model_size = t->model_size,
        .gate_offset = t->gate_offset, .up_offset = t->up_offset, .down_offset = t->down_offset,
        .gate_expert_bytes = t->gate_expert_bytes, .down_expert_bytes = t->down_expert_bytes,
        .layer = t->layer, .n_sel = g_qgate_glm.n_selected, .n_total_expert = t->n_total_expert,
        .n_slabs = g_stream_expert_cache_slab_count, .split = split, .staged = 0,
    };
    if (!qgate_commit(g_qgate_glm.seq, &req)) return 0;
    g_qgate_glm.committed = 1;
    g_qgate_glm.split = split;
    return 1;
}

static int qgate_glm_pending_for(uint32_t layer) {
    return g_qgate_glm.committed && g_qgate_glm.table.layer == layer;
}

static int qgate_glm_take(uint32_t layer, uint64_t *seq, int *split) {
    if (!qgate_glm_pending_for(layer)) return 0;
    g_qgate_glm.committed = 0;
    *seq = g_qgate_glm.seq;
    *split = g_qgate_glm.split;
    return 1;
}

static void qgate_set_dispatch_resident(int on) {
    g_qgate_dispatch_resident_n = 0;
    if (!on) return;
    for (uint32_t s = 0; s < g_stream_expert_cache_slab_count; s++)
        g_qgate_dispatch_resident[g_qgate_dispatch_resident_n++] = g_stream_expert_cache_slabs[s];
    if (g_qgate_fallback) g_qgate_dispatch_resident[g_qgate_dispatch_resident_n++] = g_qgate_fallback;
}
```

In `qgate_check_after_wait`, as the first statements (before `if (!g_qgate_thread_running) return 1;`), add:

```objc
    /* A GLM gate published or committed but never taken (its layer failed
     * before the routed MoE) must not reach a later batch. */
    g_qgate_glm.published = 0;
    g_qgate_glm.committed = 0;
```

- [ ] **Step 5: The routed MoE takes the gate**

In `ds4_gpu_routed_moe_one_tensor`:

(a) Replace `ds4_gpu_stream_expert_cache_note_token(layer_index);` (right after the dim check, ~41399)
with:

```objc
    /* A gated layer's service thread advances the aging clock itself. */
    if (!qgate_glm_pending_for(layer_index)) ds4_gpu_stream_expert_cache_note_token(layer_index);
```

(b) Next to `bool use_stream_expert_cache = false;` add:

```objc
        /* SP1: this layer's GLM gate was committed; the gate's service
         * thread resolves the experts and writes its address tables. */
        bool glm_gated = false;
        uint64_t glm_gate_seq = 0;
        int glm_gate_split = 0;
```

(c) In `} else if (use_selected_slots) {`, before `const bool selected_timing =`, add:

```objc
            glm_gated = use_iq2_stream_addr_table &&
                        qgate_glm_take(layer_index, &glm_gate_seq, &glm_gate_split);
```

(d) Change `use_stream_expert_cache =` to start with `!glm_gated &&` (i.e.
`use_stream_expert_cache = !glm_gated && !use_iq2_full_expert_addr_table && ...`). Change
`if (use_iq2_stream_addr_table && !use_stream_expert_cache) {` to
`if (use_iq2_stream_addr_table && !use_stream_expert_cache && !glm_gated) {`.

(e) Replace `if (use_iq2_full_expert_addr_table) {` (the one that sets `selected_id_source = "gpu-full-addr";`)
with:

```objc
            if (glm_gated) {
                selected_id_source = "gate";
                selected_ids_available = false;
                stream_gate_addr_buf = qgate_glm_tab(glm_gate_seq, 0u, 0u);
                stream_up_addr_buf = qgate_glm_tab(glm_gate_seq, 0u, 1u);
                stream_down_addr_buf = qgate_glm_tab(glm_gate_seq, 0u, 2u);
                use_stream_expert_addr_table = true;
            } else if (use_iq2_full_expert_addr_table) {
```

(f) In the mid dispatch, find the non-masked `ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu(cb,` call
(the one passing `stream_slot_entries, n_expert,` and ending `false, nil, nil);`):
- Put `if (glm_gated) qgate_set_dispatch_resident(1);` before it.
- Put `qgate_set_dispatch_resident(0);` after it.
- Change its `n_expert,` argument (right after `stream_slot_entries,`) to `glm_gated ? 0u : n_expert,`.

(g) In the down dispatch, the `ok = ds4_gpu_encode_mul_mv_addr_iq2(cb, down_type == DS4_METAL_TENSOR_Q2_K ? ...`
call under `if (use_stream_expert_addr_table) {`: wrap it in the same way, and change its `n_expert,`
argument (after `stream_slot_entries,`) to `glm_gated ? 0u : n_expert,`.

- [ ] **Step 6: Build and run when the GPU is free**

Run: `make ds4 ds4-server tests/test_metal_stream_gate` (exit 0). Then run
`./tests/test_metal_stream_gate --glm && ./tests/test_metal_stream_gate --qwen4`.
Expected:
- PASS lines for `glm one pass`, `glm one pass mixed`, `glm one pass fallback`, `glm uncommitted publish`
  and `glm one pass budget growth`;
- `Metal stream gate glm: PASS`;
- qwen4 still PASS.

If a section fails, use superpowers:systematic-debugging. Never loosen `same_outputs`.

- [ ] **Step 7: Commit**

```bash
git add ds4_metal.m ds4_gpu.h tests/test_metal_stream_gate.c Makefile
git commit -m "metal: GLM decode gates, one pass

ds4_gpu_glm_stream_gate_publish/commit put a GLM streamed layer behind a
stream gate: the service thread resolves the 8 routed experts into the
cache (or the gate fallback) and writes the gate's own address tables; the
routed MoE takes the committed gate, makes every slab resident and runs the
unchanged addr kernels. Model-free test: gated == drain bit for bit with
evictions, mixed tokens, fallback, an uncommitted publish and a new slab.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 5: Gate timeout fails loudly and turns gates off

**Files:**
- Modify: `ds4_metal.m` (qgate block), `ds4_gpu.h`, `tests/test_metal_stream_gate.c`, `Makefile`

**Interfaces:**
- Produces (ds4_gpu.h, Apple):
  - `void ds4_gpu_stream_gate_test_stall(uint64_t nth, uint32_t ms);` The service holds the `nth` gate
    published from now for `ms` before releasing it. 0 clears.
  - `void ds4_gpu_stream_gate_test_clear_failure(void);`
- Produces (env): `DS4_GLM_STREAM_GATE_TEST_STALL=N:MS`, read once at GLM gate setup.

- [ ] **Step 1: Write the failing test**

Add before `main`:

```c
/* A gate held past the poll's ~600 ms window must fail its batch, latch
 * gates off, and leave later steps exact on the drain path. */
static int check_timeout(void) {
    uint64_t c0 = 0, c1 = 0;
    int failed = 0;
    ds4_gpu_set_glm_model(true);
    out_elems = D;
    ds4_gpu_stream_gate_test_set_mode(1, 0);
    int ok = write_routes(0) && run_step(glm_layer, 1, got);   /* warm: slabs and gates up */
    ds4_gpu_stream_gate_test_stall(2, 900);
    const int stalled_ok = ok && write_routes(1) && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_stall(0, 0);
    ds4_gpu_stream_gate_stats(&c0, NULL, NULL, &failed);
    if (ok && (stalled_ok || !failed)) {
        fprintf(stderr, "glm timeout: the held gate was not reported (step ok %d, failed %d)\n",
                stalled_ok, failed);
        ok = 0;
    }
    ok = ok && write_routes(2) && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_set_mode(0, 0);
    ok = ok && run_step(glm_layer, 0, ref) && same_outputs("glm after timeout", 2);
    ds4_gpu_stream_gate_stats(&c1, NULL, NULL, &failed);
    if (ok && c1 != c0) {
        fprintf(stderr, "glm timeout: %llu gates ran after the failure\n",
                (unsigned long long)(c1 - c0));
        ok = 0;
    }
    ds4_gpu_stream_gate_test_clear_failure();
    fprintf(stderr, "glm timeout: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}
```

Add `--glm-timeout` to the mode check and usage, plus `if (ok && !strcmp(mode, "--glm-timeout")) ok = check_timeout();`.
In the Makefile recipe, add `./tests/test_metal_stream_gate --glm-timeout`. In `ds4_gpu.h` (Apple block) add:

```c
/* Tests: the service thread holds the nth gate published from now for ms
 * milliseconds before releasing it (0 clears); clear a latched failure. */
void ds4_gpu_stream_gate_test_stall(uint64_t nth, uint32_t ms);
void ds4_gpu_stream_gate_test_clear_failure(void);
```

- [ ] **Step 2: Verify RED (build only)**

Run: `make tests/test_metal_stream_gate`
Expected: link FAIL with `ds4_gpu_stream_gate_test_stall` and `ds4_gpu_stream_gate_test_clear_failure`
undefined.

- [ ] **Step 3: Implement**

Next to `static int g_qgate_test_force_fallback;` add:

```objc
static uint64_t g_qgate_test_stall_seq;   /* gate sequence to hold, 0: none */
static uint32_t g_qgate_test_stall_ms;

static void qgate_test_stall_maybe(uint64_t seq) {
    const uint64_t at = __atomic_load_n(&g_qgate_test_stall_seq, __ATOMIC_ACQUIRE);
    if (at && seq == at) usleep((useconds_t)g_qgate_test_stall_ms * 1000u);
}

void ds4_gpu_stream_gate_test_stall(uint64_t nth, uint32_t ms) {
    g_qgate_test_stall_ms = ms;
    __atomic_store_n(&g_qgate_test_stall_seq, nth ? g_qgate_seq + nth : 0, __ATOMIC_RELEASE);
}

void ds4_gpu_stream_gate_test_clear_failure(void) {
    if (g_qgate_thread_running) qgate_wait_idle();
    g_qgate_failed = 0;
    g_qgate_failed_reported = 0;
}
```

`qgate_wait_idle` is defined later in the file. Add `static void qgate_wait_idle(void);` above this block.

In `glm_gate_service`, right after `int ok = qgate_wait_mailbox(r, unique_ids, &n_unique, &t1);`, add
`qgate_test_stall_maybe(r->seq);`.

In `glm_gate_setup`, before `g_qgate_glm_ready = 1;`, add:

```objc
    /* Server-level timeout check: DS4_GLM_STREAM_GATE_TEST_STALL=N:MS holds
     * the Nth GLM gate for MS milliseconds. */
    const char *stall = getenv("DS4_GLM_STREAM_GATE_TEST_STALL");
    unsigned long long stall_n = 0;
    unsigned stall_ms = 0;
    if (stall && sscanf(stall, "%llu:%u", &stall_n, &stall_ms) == 2 && stall_n) {
        ds4_gpu_stream_gate_test_stall((uint64_t)stall_n, stall_ms);
        fprintf(stderr, "ds4: GLM stream gate %llu will be held %u ms (test)\n", stall_n, stall_ms);
    }
```

- [ ] **Step 4: Build and run when the GPU is free**

Run: `make tests/test_metal_stream_gate`, then
`./tests/test_metal_stream_gate --glm-timeout && ./tests/test_metal_stream_gate --glm`.
Expected:
- stderr shows `GLM stream gate ... poll timed out` and
  `stream gate failed; this batch is invalid and gates are now off`;
- `glm timeout: PASS`, and `--glm` still PASS.

- [ ] **Step 5: Commit**

```bash
git add ds4_metal.m ds4_gpu.h tests/test_metal_stream_gate.c Makefile
git commit -m "metal: GLM gate timeout fails the batch and turns gates off (tested)

A stall hook (tests, or DS4_GLM_STREAM_GATE_TEST_STALL=N:MS for server
checks) holds one gate past the poll window: the batch's end_commands fails,
gates stay off and the following steps run the drain path, exact.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 6: GLM split gates (cached experts first, misses second)

**Files:**
- Modify: `ds4_metal.m`:
  - `glm_gate_split_requested`;
  - the forward declarations;
  - `ds4_gpu_routed_moe_one_tensor`, before `DS4_METAL_PROFILE_MOE_ONE_STAGE("down");` (~44012).
- Modify: `tests/test_metal_stream_gate.c`, `Makefile`

**Interfaces:**
- Consumes: Task 4's tables (pass 1 index 0, pass 2 index 1), and `qgate_encode_poll(uint64_t seq, uint32_t stage)`,
  which exists in the qwen4 code.
- Produces (env): `DS4_GLM_STREAM_SPLIT=0` gives one pass. The default is on; Task 8 confirms it.

- [ ] **Step 1: Write the failing test**

In `main`, add `--glm-split` to the mode check and usage, plus
`if (ok && !strcmp(mode, "--glm-split")) ok = glm_suite(1);`. In the Makefile recipe, add
`./tests/test_metal_stream_gate --glm-split`.

- [ ] **Step 2: Verify RED (GPU free)**

Run: `make tests/test_metal_stream_gate && ./tests/test_metal_stream_gate --glm-split`
Expected: FAIL with `glm split: no split gate ran`, because `glm_gate_split_requested()` returns 0.

- [ ] **Step 3: Implement**

Replace `glm_gate_split_requested`:

```objc
/* Split gates (DS4_GLM_STREAM_SPLIT=0 keeps one pass): the GPU runs the
 * cached experts while the service thread reads the misses, then a second
 * poll in the same command buffer waits for them. */
static int glm_gate_split_requested(void) {
    if (g_qgate_test_split >= 0) return g_qgate_test_split;
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("DS4_GLM_STREAM_SPLIT");
        v = !(e && e[0] == '0');
    }
    return v;
}
```

Add `static void qgate_encode_poll(uint64_t seq, uint32_t stage);` to the forward declarations next to
`qgate_check_after_wait`.

In `ds4_gpu_routed_moe_one_tensor`, immediately before `DS4_METAL_PROFILE_MOE_ONE_STAGE("down");`, add:

```objc
        if (ok && glm_gated && glm_gate_split) {
            /* Split GLM gate, pass 2: the misses, after the gate's second
             * poll, from the gate's pass-2 tables (0 for the cached experts,
             * which pass 1 computed). Same act values as act_args above. */
            const ds4_gpu_dsv4_moe_swiglu_weight_args pass2_act = {
                .width = expert_mid_dim,
                .rows = pair_rows,
                .gate_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .up_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .mid_row_stride = (uint64_t)expert_mid_dim * sizeof(float),
                .weight_stride = sizeof(float),
                .write_clamped = 0,
                .clamp_value = clamp,
            };
            qgate_encode_poll(glm_gate_seq, 1u);
            qgate_set_dispatch_resident(1);
            ok = ds4_gpu_encode_mul_mv_addr_iq2_pair_swiglu(cb,
                     g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline,
                     &gate_args, &pass2_act, stream_slot_entries, 0u,
                     qgate_glm_tab(glm_gate_seq, 1u, 0u), qgate_glm_tab(glm_gate_seq, 1u, 1u),
                     xbuf, ds4_gpu_tensor_offset(x),
                     gatebuf, ds4_gpu_tensor_offset(gate),
                     upbuf, ds4_gpu_tensor_offset(up),
                     midbuf, ds4_gpu_tensor_offset(mid),
                     selected_exec_buf, selected_exec_off,
                     weightsbuf, ds4_gpu_tensor_offset(weights),
                     gate_smem, 2, false, nil, nil) &&
                 ds4_gpu_encode_mul_mv_addr_iq2(cb,
                     down_type == DS4_METAL_TENSOR_Q2_K ?
                         g_moe_mul_mv_addr_q2_k_pipeline : g_moe_mul_mv_addr_iq2_xxs_pipeline,
                     &down_args, stream_slot_entries, 0u,
                     qgate_glm_tab(glm_gate_seq, 1u, 2u),
                     midbuf, ds4_gpu_tensor_offset(mid),
                     down_dst, down_dst_off,
                     selected_exec_buf, selected_exec_off,
                     down_smem, 2, false, 2, nil);
            qgate_set_dispatch_resident(0);
        }
```

Check the context before building. `pair_rows`, `clamp`, `gate_args`, `down_args`, `gate_smem`, `down_smem`,
the buffers, `selected_exec_buf/off`, `down_dst` and `down_dst_off` must all be in scope at this point (the
sum call right below uses `down_dst`). `cb` must be the batch command buffer: gates require an open batch.
If `act_args` in the mid dispatch has different field values from those above, copy its exact initializer.
The byte-identity test catches any drift.

- [ ] **Step 4: Build and run when the GPU is free**

Run: `make ds4 ds4-server tests/test_metal_stream_gate && make test-metal-stream-gate`
Expected:
- the qwen4, glm, glm-timeout and glm-split runs all PASS;
- `--glm-split` prints `glm split: N gated layers: PASS`, plus PASS for `glm split mixed`,
  `glm split fallback` and `glm split budget growth`.

- [ ] **Step 5: Commit**

```bash
git add ds4_metal.m tests/test_metal_stream_gate.c Makefile
git commit -m "metal: GLM split gates run cached experts while misses load

The service writes pass-1 tables (cached experts, 0 for misses) and releases
the first poll before reading; the routed MoE encodes a second poll and a
second pass over the pass-2 tables (misses only). Each slot is computed once
with unchanged kernels; the expert sum adds slots in order. Default on,
DS4_GLM_STREAM_SPLIT=0 keeps one pass.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 7: ds4.c GLM client (publish, shared expert, commit)

**Files:**
- Modify: `ds4.c`:
  - the struct `ds4_glm_gpu_graph` (~46683-46878);
  - after `glm_graph_streaming_decode_sync_each_layer` (~57296);
  - `glm_graph_forward_token` (~57308);
  - `glm_graph_encode_sparse_ffn_one` (~50691).
- Modify: `tests/test_glm53_stream_layout.c`

**Interfaces:**
- Consumes (Task 4): `ds4_gpu_glm_stream_gate_publish`, `ds4_gpu_glm_stream_gate_commit`.
- Produces: the field `bool stream_gate_token;` in `ds4_glm_gpu_graph`, and
  `static bool glm_graph_stream_gate_token_allowed(bool static_decode_map, bool imatrix, bool expert_profile, uint32_t ablate_mask, bool async_profile);`.

- [ ] **Step 1: Write the failing test**

In `tests/test_glm53_stream_layout.c`, before `int main(void)`, add:

```c
/* SP1 decode gates encode a whole token without waiting: only tokens whose
 * weights are mapped up front and that read no routed selection on the host
 * mid-token may publish gates. */
static void check_glm_stream_gate_token_allowed(void) {
    CHECK(glm_graph_stream_gate_token_allowed(true, false, false, 0u, false));
    CHECK(!glm_graph_stream_gate_token_allowed(false, false, false, 0u, false));
    CHECK(!glm_graph_stream_gate_token_allowed(true, true, false, 0u, false));
    CHECK(!glm_graph_stream_gate_token_allowed(true, false, true, 0u, false));
    CHECK(!glm_graph_stream_gate_token_allowed(true, false, false, DS4_GLM_ABLATE_ROUTED, false));
    CHECK(!glm_graph_stream_gate_token_allowed(true, false, false, 0u, true));
}
```

and call `check_glm_stream_gate_token_allowed();` in `main` after `check_glm_streaming_async_load_default();`.

- [ ] **Step 2: Verify RED (no GPU)**

Run: `make tests/test_glm53_stream_layout`
Expected: compile FAIL: implicit declaration of `glm_graph_stream_gate_token_allowed`.

- [ ] **Step 3: Implement**

(a) In `ds4_glm_gpu_graph`, after `bool streaming_static_decode_map_current;`, add:

```c
    /* SP1: this decode token may publish stream gates (set per token by
     * glm_graph_forward_token, false outside it). */
    bool stream_gate_token;
```

(b) After `glm_graph_streaming_decode_sync_each_layer` add:

```c
/* SP1 decode gates (Metal): a gated token is encoded without waiting, so all
 * its weights must be mapped before encoding starts (the static decode map)
 * and no diagnostic may read the routed selection on the host mid-token. */
static bool glm_graph_stream_gate_token_allowed(bool     static_decode_map,
                                                bool     imatrix,
                                                bool     expert_profile,
                                                uint32_t ablate_mask,
                                                bool     async_profile) {
    return static_decode_map && !imatrix && !expert_profile &&
           ablate_mask == 0 && !async_profile;
}
```

(c) In `glm_graph_forward_token`, right after the `const bool streaming_decode_sync_each_layer = ...;`
statement, add:

```c
    g->stream_gate_token =
        g->ssd_streaming &&
        glm_graph_stream_gate_token_allowed(static_decode_map,
                                            g->imatrix != NULL,
                                            g_expert_profile.active,
                                            glm_decode_ablate_mask(),
                                            glm_graph_streaming_async_profile_enabled());
```

Then, right after the layer loop (immediately before `#undef DS4_GLM_PROFILE_DECODE_STAGE`), add
`g->stream_gate_token = false;`.

(d) In `glm_graph_encode_sparse_ffn_one`:
- Before `if (ok && streaming_selected_cache) {`, add `bool gate_published = false;`.
- Inside that block, right after the `const ds4_gpu_stream_expert_table table = { ... };` initializer, add:

```c
#if defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        /* SP1 decode gates: publish the selection now and commit after the
         * shared expert; the routed MoE below reads the gate's tables. */
        gate_published = g->stream_gate_token &&
                         generic_streaming_selected_cache &&
                         ds4_gpu_glm_stream_gate_publish(&table,
                                                         g->router_selected,
                                                         DS4_N_EXPERT_USED) != 0;
#endif
```

- Change `const bool async_selected_load =` so that its expression starts with `!gate_published &&`
  (before the `#ifdef DS4_ROCM_BUILD` line).
- Change `if (!async_load_started) {` (the one that calls
  `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor`) to
  `if (!async_load_started && !gate_published) {`.
- After the `if (ok && shared_first) { ... shared_down ... }` block, and before the `if (async_profile) {`
  that follows it, add:

```c
#if defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    if (gate_published) {
        /* Commit even after a failure so the gate never outlives its layer;
         * the token then fails at its end_commands. */
        const bool committed = ds4_gpu_glm_stream_gate_commit() != 0;
        ok = ok && committed;
    }
#endif
```

- [ ] **Step 4: Build and run**

Run: `make ds4 ds4-server tests/test_glm53_stream_layout && ./tests/test_glm53_stream_layout`.
Expected: build exit 0, `test_glm53_stream_layout: PASS`. This test is model-free with no GPU work.

Run: `make test-session-state` (CPU).
Expected: PASS as before.

When the GPU is free, run `make test-metal-stream-gate test-metal-ssd-experts`.
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ds4.c tests/test_glm53_stream_layout.c
git commit -m "glm: decode tokens publish stream gates on Metal

A token with the static decode map and no host-side selection diagnostics
publishes each streamed IQ2 layer's selection after its router, encodes the
shared expert, then commits; the routed MoE runs behind the gate instead of
draining the batch. DS4_GLM_STREAM_GATE=0 keeps the drain.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

---

### Task 8: SP1 validation on the model (GPU window)

**Files:**
- Create: `speed-bench/glm53-decode/sp1_validate.sh`
- Create: `speed-bench/glm53-decode/server_smoke.py`
- Modify: `speed-bench/glm53-decode/RESULTS.md`

**Interfaces:** consumes the env switches from Tasks 4-6 and the stall env from Task 5.

- [ ] **Step 1: Write the CLI validation script**

```zsh
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
```

Run: `zsh -n speed-bench/glm53-decode/sp1_validate.sh && chmod +x speed-bench/glm53-decode/sp1_validate.sh`
Expected: exit 0.

- [ ] **Step 2: Write the server smoke script**

```python
#!/usr/bin/env python3
"""GLM-5.3 SP1 server smoke on ds4-server --ssd-streaming: decode and prefill
speed, multi-turn reuse, and (with --stall) an injected gate timeout.

usage: server_smoke.py OUT_DIR LABEL [--stall N:MS] [--env K=V ...]

Writes OUT_DIR/LABEL.json and OUT_DIR/server-LABEL.log. Stops the server with
exactly one SIGTERM (never SIGKILL a Metal process).
"""
import argparse, http.client, json, os, subprocess, time, urllib.error, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("GLM_MODEL", os.path.expanduser(
    "~/orca/workspaces/ds4-metal-data/gguf/glm53/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf"))
PORT = 18191
FILLER = open(os.path.join(REPO, "speed-bench/promessi_sposi.txt"), encoding="utf-8").read()
CODE_Q = ("Write a Python function that parses an ISO-8601 duration string such as "
          "'P3DT4H5M' into total seconds, with a short docstring and three doctest examples.")
DOC_Q = "\n\nQuestion: summarize the passage above in eight bullet points."


def start(out, label, env):
    cmd = [os.path.join(REPO, "ds4-server"), "--metal", "-m", MODEL, "--ssd-streaming",
           "--power", "100", "--host", "127.0.0.1", "--port", str(PORT), "-c", "262144"]
    log = open(os.path.join(out, f"server-{label}.log"), "w")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **env})
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit(f"server exited during startup (rc {proc.returncode})")
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=2)
            return proc, log, base
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise SystemExit("server did not come up")


def stop(proc, log):
    proc.terminate()  # exactly one SIGTERM
    try:
        proc.wait(120)
    except subprocess.TimeoutExpired:
        print("WARNING: server still alive after 120 s; not killing", flush=True)
    log.close()


def chat(base, messages, max_tokens, model="glm-5.3-flash-chat", extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra or {})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = {}; content = []; finish = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[6:]); t = time.time() - t0
                usage = ch.get("usage") or usage
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    finish = c.get("finish_reason") or finish
                    if d.get("content") or d.get("reasoning_content"):
                        first = t if first is None else first; last = t
                    content.append(d.get("content") or "")
    except (urllib.error.URLError, ConnectionError, http.client.HTTPException,
            json.JSONDecodeError) as e:
        return {"error": repr(e), "finish": finish, "content": "".join(content)}
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(first or 0, 2),
            "prefill_tps": round(pt / first, 1) if first else 0,
            "decode_tps": round((ct - 1) / (last - first), 2) if first and last and last > first and ct > 1 else 0,
            "finish": finish, "content": "".join(content)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("label")
    ap.add_argument("--stall"); ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = dict(kv.split("=", 1) for kv in a.env)
    if a.stall:
        env["DS4_GLM_STREAM_GATE_TEST_STALL"] = a.stall
    proc, log, base = start(a.out, a.label, env)
    rec = {"label": a.label, "env": env}
    try:
        user = lambda text: [{"role": "user", "content": text}]
        if a.stall:
            # The held gate falls in the first request's decode: it must fail
            # without a wrong token; the next request runs on the drain path.
            rec["stalled"] = chat(base, user(CODE_Q), 256)
            rec["after"] = chat(base, user(CODE_Q), 64)
        else:
            for name, msgs, n in [("warm", user(CODE_Q), 64), ("short1", user(CODE_Q), 256),
                                  ("short2", user(CODE_Q), 256),
                                  ("doc8k", user(FILLER[:30000] + DOC_Q), 256)]:
                rec[name] = chat(base, msgs, n)
            msgs = user("Explain briefly what an LRU cache is.")
            for turn, follow in enumerate(["Give a Python example.", "What is the complexity of get and put?"]):
                r = chat(base, msgs, 300, model="glm-5.3-flash", extra={"reasoning_effort": "low"})
                rec[f"chat{turn + 1}"] = {k: r.get(k) for k in ("prompt_tokens", "ttft_s", "finish")}
                msgs += [{"role": "assistant", "content": r.get("content", "")},
                         {"role": "user", "content": follow}]
            r = chat(base, msgs, 300, model="glm-5.3-flash", extra={"reasoning_effort": "low"})
            rec["chat3"] = {k: r.get(k) for k in ("prompt_tokens", "ttft_s", "finish")}
    finally:
        stop(proc, log)
    text = open(os.path.join(a.out, f"server-{a.label}.log"), errors="replace").read()
    rec["log"] = {"requires_rebuild": text.count("requires rebuild"),
                  "gate_timed_out": text.count("poll timed out"),
                  "gates_off": text.count("gates are now off")}
    for k, v in rec.items():
        if isinstance(v, dict) and "content" in v:
            v["content"] = v["content"][:200]
    json.dump(rec, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
```

Run: `python3 -m py_compile speed-bench/glm53-decode/server_smoke.py`
Expected: exit 0.

Commit both scripts:

```bash
git add speed-bench/glm53-decode/sp1_validate.sh speed-bench/glm53-decode/server_smoke.py
git commit -m "bench: GLM-5.3 SP1 validation and server smoke scripts

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

- [ ] **Step 3: Run in a GPU window (needs the user's go)**

1. Wait for "GPU free" from the Qwen session and the user's go.
2. Notify both peers.
3. Run `~/.local/bin/server-power.sh off`.
4. Run `make ds4 ds4-server`.
5. From the scratchpad, in the background, run in order:
   - `speed-bench/glm53-decode/sp1_validate.sh $R`
   - `python3 speed-bench/glm53-decode/server_smoke.py $R drain --env DS4_GLM_STREAM_GATE=0`
   - `python3 speed-bench/glm53-decode/server_smoke.py $R split`
   - `python3 speed-bench/glm53-decode/server_smoke.py $R stall --stall 3000:900`
   - `speed-bench/qwen-regression/run.sh full > $R/qwen-full.log 2>&1`
6. Then run `~/.local/bin/server-power.sh on`, confirm `backend_ok`, and notify both peers. Restore PROD
   even if a step fails or the window is cut short.

- [ ] **Step 4: Check every gate**

| Check | Pass condition |
|---|---|
| Identity | `gate1` and `split` vs `drain` IDENTICAL at 16 and 256 tokens; `mtp-split` vs `mtp-drain` IDENTICAL |
| Speed | mean of `speed-split-1/2` ≥ 12.5 t/s (SP1 gate). Below 10.8: stop and re-evaluate with the user. In between: report and ask |
| GPU busy | `busy split` ≥ 70% |
| Split default | if `speed-gate1` > `speed-split` by more than 2% in both reps, make one pass the default (Step 5) |
| Server | `split.json`: `short2.decode_tps` ≥ `drain.json`'s; `doc8k.prefill_tps` within 3% of drain; chat2/chat3 `prompt_tokens` grow and `requires_rebuild` is 0 (or no more than drain's) |
| Timeout | `stall.json`: `stalled` has an error or a finish other than stop/length; `after.finish` is stop or length with content; log shows `poll timed out` and `gates are now off` |
| Qwen | `qwen-full.log` contains `qwen_gate: PASS`, and Qwen decode ≥ 97% of PROD |

If any identity check DIFFERS, stop and use superpowers:systematic-debugging, then rerun the window.

- [ ] **Step 5 (only if the split-default row says so): make one pass the default**

In `glm_gate_split_requested`, replace `v = !(e && e[0] == '0');` with `v = e && e[0] && e[0] != '0';`, and
change the comment to "Split gates (DS4_GLM_STREAM_SPLIT=1)". Rebuild. When the GPU is free, run
`make test-metal-stream-gate`; expected PASS.
Commit with the message `glm: one-pass stream gates by default (split measured slower)`, plus the trailers.

- [ ] **Step 6: Record and commit**

Fill the SP1 section of `speed-bench/glm53-decode/RESULTS.md` with one row per check above (numbers from
`steps.log` and the JSON files), the gate verdict, and the `DS4_GLM_STREAM_TIMING` lines.

```bash
git add speed-bench/glm53-decode/RESULTS.md
git commit -m "results: GLM-5.3 SP1 decode gates on M5 Pro (identity, speed, busy, server, Qwen gate)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015npc1LHTBTYaTXLaJt8Uqc"
```

Update the memory file `glm53-flash-next-build.md` and its MEMORY.md line. Record:
- the SP1 result (t/s, busy %, identity);
- the branch head;
- the next step: the SP2 spec if the SP1 gate passed, or a re-evaluation if it did not.

Then hand over to superpowers:finishing-a-development-branch after the final whole-branch review. Do not
merge or push without the user.
