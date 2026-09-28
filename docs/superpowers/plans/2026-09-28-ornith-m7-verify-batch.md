# Ornith M7: Exact batched MTP verify — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Ornith's `--mtp` decode rate at 2K and 32K to at least the live oMLX's by making the 2-row MTP verify read each Q8_0 weight once for both rows, while every row stays bit-identical to a plain 1-row step.

**Architecture:** Ornith-only and additive. A two-row copy of `kernel_mul_mv_q8_0_f32_impl` (`kernel_qwen35_mv_q8_0_rows2` in `metal/qwen35.metal`) loads each weight block once and keeps the one-row kernel's per-row multiply-adds and reduction tree; `ds4_gpu_qwen35_matmul_q8_0_rows_tensor` in `ds4_metal.m` dispatches it with the T=1 path's NR0/NSG and counts its dispatches. `ds4_qwen35moe.inc` routes the verify's Q8_0 projections (`qwen35_gemv`: attention q/output, lm head) and a new GDN verify layer (`qwen35_graph_linear_verify`: batched `lin_qkv`/`lin_gate`/`lin_out`, per-row front/scan/norm) through it behind `DS4_QWEN35_VERIFY_BATCH`. A model-free microbench decides whether the existing `decode_rows_exact` dispatch already suffices and applies the stop rule; the M4 harness decides adoption.

**Tech Stack:** C, Objective-C (`ds4_metal.m`), Metal Shading Language, Python 3 (M4 harness, gate scripts).

**Spec:** `docs/superpowers/specs/2026-09-28-ornith-m7-verify-batch-design.md` (parent: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today. Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge. `qwen4_graph_linear` and `qwen4_gemv_rows` in `ds4.c` are not modified.
- Metal only for Ornith. Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no `DS4_QWEN4_*` knob.
- Kernel changes are additive: new kernels in `metal/qwen35.metal`; `metal/dense.metal` and `metal/qwen4.metal` are never edited.
- `--mtp` greedy output stays byte-identical to plain decoding (user ruling 2026-09-28). No lever in this plan trades exactness for speed. Plain T=1 decode keeps `kernel_mul_mv_q8_0_f32` unless the spec §5 fallback is taken (a ledgered ruling).
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process; scripts stop processes with SIGTERM and a bounded wait.
- GPU windows (model runs, timed benchmark runs, speed measurements) pause the live stack and restore it afterwards (the controller's `gpuwin.sh OUT STEPS.sh` from M4-M6, run from snapshots built with `git archive <sha>` into the scratchpad); builds and model-free kernel tests run outside windows.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Plan-specific constraints:
- Branch `feature/ornith-m7` in worktree `foxface` (from develop `b764a66`, holds the spec commit `5a105ba`); merging or pushing needs the user's explicit OK. No deploy.
- Adoption rule: `DS4_QWEN35_VERIFY_BATCH` starts off and becomes the default only after Task 5's checks pass and its lever A/B gains at least 5% at 2K (spec §6).
- Existing unit tests are never edited; new tests are additions (a new test file, new functions and calls in `tests/test_qwen35_kernels.c`).
- The MTP draft vocabulary used by every speed run: `$SNAP/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt` (`$SNAP` = the window's snapshot directory).
- **Stop rules (spec §6):** after Task 2, the two-row path must cost < 1.5x one T=1 call on 2048->8192 and on the lm head, else stop and report; after Task 5, the lever A/B must gain >= 5% at 2K, else stop before Task 7 (the flip in Task 6 still needs the gain). Task 7 (MoE) only if Task 6's gate 3 misses criterion 1 and its overlap measurement clears ~40%, and then only as a design addendum for the user.

## Review Focus

1. **NSG agreement with the T=1 path.** Row r is bit-identical only if the two-row kernel runs with the same NR0 and NSG as the T=1 dispatch of the same weight: 4 by default, 8 past 65536 outputs (the lm head), and `DS4_METAL_Q8_MV_NSG` overriding both. Pinned by `test_q8_rows2_exact` at out_dim 8192 (NSG 4) and 65600 (NSG 8) (Task 2); the env override is a reviewer check (the wrapper must take NSG from `ds4_gpu_make_q8_0_mv_dispatch()`).
2. **Threadgroup-memory race between the two rows' reductions.** Each row's reduction must use its own threadgroup-memory region (`shmem + r * NR0 * 32` floats); a shared region lets simdgroup 0 zero row 1's slots while other simdgroups still read row 0's. Pinned by `test_q8_rows2_exact` repeating the batched call 3 times per shape and comparing each (Task 2).
3. **Odd out_dim.** The last threadgroup of an odd out_dim has one valid output row; nothing past row out_dim - 1 of either output row, and nothing past the second output row, may be written. Pinned by the (2048, 4099) case with a sentinel third output row (Task 2).
4. **Row views and the snapshot in the GDN verify layer.** Row 1's front/scan/norm must read and write only row 1's slots of `qkv`/`z`/`ga`/`gb`/`lin_o`, and only row 0's calls may receive the after-first-row snapshot pointers (a rejected draft restores it). Pinned by `test_qwen35_mtp` (rejected and forced-accepted drafts, memcmp against plain) and `test_qwen35_verify_batch` with the knob on (Task 4).
5. **Knob off and prefill unchanged.** With `DS4_QWEN35_VERIFY_BATCH` unset (before Task 6) or `=0` (after), the verify must dispatch no two-row matvec, and prefill/plain decode never reach it (`T == 2u` and `g->verify_rows_exact` only). Pinned by `test_qwen35_verify_batch --knob-off` (Task 6) and gate 1's dumps against develop (Task 5).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/bench_qwen35_verify.c` | Create | model-free timing of T=1, 2x T=1, `decode_rows_exact`, two-row kernel on Ornith's Q8_0 shapes with cold weights |
| `metal/qwen35.metal` | Modify (append) | `kernel_qwen35_mv_q8_0_rows2` |
| `ds4_metal.m`, `ds4_gpu.h` | Modify | `ds4_gpu_qwen35_matmul_q8_0_rows_tensor`, `ds4_gpu_qwen35_q8_rows_dispatches` |
| `ds4_qwen35moe.inc` | Modify | `DS4_QWEN35_VERIFY_BATCH`, batched branch in `qwen35_gemv`, `qwen35_graph_linear_verify` |
| `tests/test_qwen35_kernels.c` | Modify (additions) | `test_q8_rows2_exact` and its cases; arena 1 GiB |
| `tests/test_qwen35_verify_batch.c` | Create | real-model identity + dispatch-count check of the batched verify |
| `Makefile` | Modify | bench and test targets |
| `speed-bench/ornith/m7/` | Create | `BENCH.md`, `LEVERS.md`, `speed/GATE3.md`, `h2h/`, `QWEN_GATE.md`, `REPORT.md` |
| `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` | Modify (Task 6) | §5 verify text, §7.3 knob list |

---

### Task 1: Microbench of the existing paths

**Files:**
- Create: `tests/bench_qwen35_verify.c`
- Modify: `Makefile` (after the `bench-qwen35-attn` target)
- Create: `speed-bench/ornith/m7/BENCH.md`

**Interfaces:**
- Consumes: `ds4_gpu_qwen4_matmul_q8_0_tensor(out, map, size, off, in_dim, out_dim, x, n_tok)`, `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, map, size, off, in_dim, out_dim, x, n_rows)`, `ds4_gpu_begin_commands/end_commands/synchronize`, `ds4_gpu_set_model_map`.
- Produces: `./tests/bench_qwen35_verify`, printing one line per (shape, method): `bench: <shape> <method> <ms> ms  x<ratio vs t1>  <GB/s>`; and the verdict line `bench: verdict decode_rows_exact <use|build-kernel>`. Task 2 adds a method to it.

- [ ] **Step 1: Write the bench**

```c
/* Model-free micro-benchmark for the M7 verify (Ornith's Q8_0 projection
 * shapes): one T=1 matvec, two T=1 matvecs (today's per-row verify), the
 * existing exact two-row dispatch (ds4_gpu_matmul_q8_0_decode_rows_exact_tensor)
 * and, from M7 Task 2, the two-row kernel.  Every timed call reads a
 * different copy of the weights (>= 256 MB per shape, above the system
 * cache): a decode step streams ~2.7 GB and never finds its weights cached.
 * Needs the GPU but no model; run it with the live stack paused. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

/* Stub for the symbol ds4_metal.o expects from ds4.o, which this
 * GPU-kernel-only bench does not link (same pattern as tests/bench_qwen35_attn.c). */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

typedef struct { const char *name; uint32_t in_dim, out_dim; } shape_t;
static const shape_t SHAPES[] = {
    { "2048x8192(lin_qkv,attn_q)", 2048u, 8192u },
    { "2048x4096(lin_gate)", 2048u, 4096u },
    { "4096x2048(lin_out,attn_output)", 4096u, 2048u },
    { "2048x248320(lm_head)", 2048u, 248320u },
};
enum { N_SHAPES = 4, CALLS = 32, REPS = 5, MAX_COPIES = 64 };
static const uint64_t COLD_BYTES = 256ull << 20;

typedef enum { M_T1, M_T1X2, M_ROWS_EXACT, N_METHODS } method_t;
static const char *METHOD_NAMES[N_METHODS] = { "t1", "t1x2", "rows_exact" };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_verify: %s failed\n", what); exit(1); }
}

static uint32_t g_rng = 0x6c8e9cf5u;
static uint32_t next_rng(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t off[MAX_COPIES];
    uint32_t copies, in_dim, out_dim;
    uint64_t bytes;
    ds4_gpu_tensor *x, *x0, *x1, *o, *o0, *o1;
} ctx_t;

static uint32_t copies_for(uint64_t bytes) {
    uint64_t c = (COLD_BYTES + bytes - 1u) / bytes;
    if (c < 2u) c = 2u;
    return c > MAX_COPIES ? MAX_COPIES : (uint32_t)c;
}

/* Q8_0 blocks: scale 2^-7 (half 0x2000), quants uniform in [-64, 63] */
static void fill_q8(uint8_t *p, uint64_t bytes) {
    for (uint64_t b = 0; b + 34u <= bytes; b += 34u) {
        p[b] = 0x00; p[b + 1] = 0x20;
        for (int j = 0; j < 32; j++) p[b + 2 + j] = (uint8_t)((next_rng() >> 8) & 0x7fu) - 64u;
    }
}

static int call(ctx_t *c, method_t m, uint32_t k) {
    const uint64_t off = c->off[k % c->copies];
    switch (m) {
    case M_T1:
        return ds4_gpu_qwen4_matmul_q8_0_tensor(c->o0, c->base, c->size, off, c->in_dim, c->out_dim, c->x0, 1u);
    case M_T1X2:
        return ds4_gpu_qwen4_matmul_q8_0_tensor(c->o0, c->base, c->size, off, c->in_dim, c->out_dim, c->x0, 1u) &&
               ds4_gpu_qwen4_matmul_q8_0_tensor(c->o1, c->base, c->size, off, c->in_dim, c->out_dim, c->x1, 1u);
    case M_ROWS_EXACT:
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(c->o, c->base, c->size, off, c->in_dim, c->out_dim,
                                                            c->x, 2u);
    default:
        return 0;
    }
}

/* median over REPS of (CALLS calls encoded into one command buffer) / CALLS,
 * after one untimed repetition */
static double time_ms(ctx_t *c, method_t m) {
    double t[REPS];
    for (int r = -1; r < REPS; r++) {
        const double t0 = now_s();
        need(ds4_gpu_begin_commands(), "begin commands");
        for (uint32_t k = 0; k < CALLS; k++) need(call(c, m, k), METHOD_NAMES[m]);
        need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
        if (r >= 0) t[r] = (now_s() - t0) * 1e3 / CALLS;
    }
    for (int i = 1; i < REPS; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[REPS / 2];
}

static ds4_gpu_tensor *rand_rows(uint64_t n) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = (float)((int)(next_rng() & 0xffffu) - 32768) / 32768.0f;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "input upload");
    free(h);
    return t;
}

int main(void) {
    uint64_t total = 0;
    for (int s = 0; s < N_SHAPES; s++) {
        const uint64_t bytes = (uint64_t)SHAPES[s].out_dim * (SHAPES[s].in_dim / 32u) * 34u;
        total += ((bytes + 63u) & ~63ull) * copies_for(bytes);
    }
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    need(base != MAP_FAILED, "weight arena mmap");
    need(ds4_gpu_init(), "GPU initialization");
    need(ds4_gpu_set_model_map(base, total), "model map registration");
    uint64_t used = 0;
    double ratio_exact[N_SHAPES];
    for (int s = 0; s < N_SHAPES; s++) {
        ctx_t c = { .base = base, .size = total, .in_dim = SHAPES[s].in_dim, .out_dim = SHAPES[s].out_dim };
        c.bytes = (uint64_t)c.out_dim * (c.in_dim / 32u) * 34u;
        c.copies = copies_for(c.bytes);
        for (uint32_t k = 0; k < c.copies; k++) {
            c.off[k] = used;
            fill_q8(base + used, c.bytes);
            used += (c.bytes + 63u) & ~63ull;
        }
        c.x = rand_rows(2ull * c.in_dim);
        c.x0 = ds4_gpu_tensor_view(c.x, 0, (uint64_t)c.in_dim * sizeof(float));
        c.x1 = ds4_gpu_tensor_view(c.x, (uint64_t)c.in_dim * sizeof(float), (uint64_t)c.in_dim * sizeof(float));
        c.o = ds4_gpu_tensor_alloc(2ull * c.out_dim * sizeof(float));
        c.o0 = ds4_gpu_tensor_view(c.o, 0, (uint64_t)c.out_dim * sizeof(float));
        c.o1 = ds4_gpu_tensor_view(c.o, (uint64_t)c.out_dim * sizeof(float), (uint64_t)c.out_dim * sizeof(float));
        need(c.x0 && c.x1 && c.o && c.o0 && c.o1, "tensor views");
        double t1 = 0.0;
        for (int m = 0; m < N_METHODS; m++) {
            const double ms = time_ms(&c, (method_t)m);
            if (m == M_T1) t1 = ms;
            if (m == M_ROWS_EXACT) ratio_exact[s] = ms / t1;
            printf("bench: %-32s %-10s %8.4f ms  x%.2f  %6.1f GB/s\n", SHAPES[s].name, METHOD_NAMES[m], ms,
                   ms / t1, (double)c.bytes / (ms * 1e6));
        }
        ds4_gpu_tensor_free(c.x0); ds4_gpu_tensor_free(c.x1); ds4_gpu_tensor_free(c.x);
        ds4_gpu_tensor_free(c.o0); ds4_gpu_tensor_free(c.o1); ds4_gpu_tensor_free(c.o);
    }
    bool use = true;
    for (int s = 0; s < N_SHAPES; s++) use = use && ratio_exact[s] <= 1.2;
    printf("bench: verdict decode_rows_exact %s\n", use ? "use" : "build-kernel");
    return 0;
}
```

- [ ] **Step 2: Add the Makefile targets** (after the `bench-qwen35-attn` target, same block)

```make
tests/bench_qwen35_verify.o: tests/bench_qwen35_verify.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/bench_qwen35_verify: tests/bench_qwen35_verify.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: bench-qwen35-verify
bench-qwen35-verify: tests/bench_qwen35_verify
	./tests/bench_qwen35_verify
```

- [ ] **Step 3: Build**

Run: `make -j8 tests/bench_qwen35_verify 2>&1 | grep -ciE 'warning|error'`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add tests/bench_qwen35_verify.c Makefile
git commit -m "tests: model-free microbench of Ornith's verify Q8_0 projections"
```

- [ ] **Step 5 (controller, GPU window): run the bench**

Snapshot the Task 1 commit, build `tests/bench_qwen35_verify` there, run `./tests/bench_qwen35_verify` in a window (twice, to see the spread).
Expected: 12 `bench:` lines per run and one verdict line. Sanity: `t1x2` ~2x `t1` on every shape; `t1` on the lm head reads ~540 MB at > 150 GB/s.

- [ ] **Step 6: Write `speed-bench/ornith/m7/BENCH.md` and commit**

The table of both runs, the verdict, and what it decides: `use` -> Task 2 takes its "existing dispatch" variant (Step 3b); `build-kernel` -> Task 2 builds the kernel.

```bash
git add speed-bench/ornith/m7/BENCH.md
git commit -m "speed-bench/ornith: M7 microbench of the verify's Q8_0 projections"
```

---

### Task 2: Two-row Q8_0 matvec, exactness tests, stop rule

**Files:**
- Modify: `metal/qwen35.metal` (append after the final `#endif /* DS4_METAL_HAS_TENSOR */`)
- Modify: `ds4_metal.m` (after `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`), `ds4_gpu.h` (after its declaration)
- Modify: `tests/test_qwen35_kernels.c` (new function before `main`, calls in `main`, arena 1 GiB)
- Modify: `tests/bench_qwen35_verify.c` (method `rows2`)
- Modify: `speed-bench/ornith/m7/BENCH.md`

**Interfaces:**
- Consumes: Task 1's bench; `kernel_mul_mv_q8_0_f32_impl` and `helper_mv_reduce_and_write` in `metal/dense.metal` (same library, read-only reference).
- Produces:
  ```c
  /* Two rows of x through one Q8_0 weight in one dispatch; row r is
   * bit-identical to ds4_gpu_qwen4_matmul_q8_0_tensor(..., row r, 1).
   * n_rows must be 2; returns 0 (nothing encoded) otherwise. */
  int ds4_gpu_qwen35_matmul_q8_0_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                             const ds4_gpu_tensor *x, uint32_t n_rows);
  /* Successful ds4_gpu_qwen35_matmul_q8_0_rows_tensor calls since process start. */
  uint64_t ds4_gpu_qwen35_q8_rows_dispatches(void);
  ```

- [ ] **Step 1: Write the failing test** (in `tests/test_qwen35_kernels.c`, before `main`)

```c
/* M7: the verify's two-row Q8_0 matvec must equal a one-row
 * kernel_mul_mv_q8_0_f32 dispatch bit for bit on each row (NSG 4, and 8
 * past 65536 outputs), so batching the verify cannot change --mtp output.
 * The batched call runs 3 times (a reduction race would not repeat
 * reliably), a third output row keeps its sentinel, and n_rows != 2 is
 * refused. */
static void test_q8_rows2_exact(arena_t *a, uint32_t in_dim, uint32_t out_dim) {
    double *shadow;
    const uint64_t off = arena_q8_0(a, out_dim, in_dim, &shadow, 0.05f);
    free(shadow);
    float *x = rand_vec(2ull * in_dim, 1.0f);
    ds4_gpu_tensor *gx = upload(x, 2ull * in_dim);
    ds4_gpu_tensor *g3 = upload(NULL, 3ull * out_dim);
    float *ref[2];
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(gx, (uint64_t)r * in_dim * sizeof(float), (uint64_t)in_dim * sizeof(float));
        ds4_gpu_tensor *o1 = upload(NULL, out_dim);
        require_ok(xr && ds4_gpu_qwen4_matmul_q8_0_tensor(o1, a->base, a->size, off, in_dim, out_dim, xr, 1u),
                   "q8 T=1 reference");
        ref[r] = download(o1, out_dim);
        ds4_gpu_tensor_free(o1);
        ds4_gpu_tensor_free(xr);
    }
    for (int rep = 0; rep < 3; rep++) {
        require_ok(ds4_gpu_tensor_fill_f32(g3, -1234.5f, 3ull * out_dim), "q8 rows2 sentinel");
        const uint64_t before = ds4_gpu_qwen35_q8_rows_dispatches();
        require_ok(ds4_gpu_qwen35_matmul_q8_0_rows_tensor(g3, a->base, a->size, off, in_dim, out_dim, gx, 2u),
                   "q8 rows2");
        require_ok(ds4_gpu_qwen35_q8_rows_dispatches() == before + 1u, "q8 rows2 dispatch counted");
        float *got = download(g3, 3ull * out_dim);
        for (uint32_t r = 0; r < 2u; r++)
            require_ok(memcmp(got + (uint64_t)r * out_dim, ref[r], (uint64_t)out_dim * sizeof(float)) == 0,
                       "q8 rows2 row bit-identical to T=1");
        for (uint64_t i = 2ull * out_dim; i < 3ull * out_dim; i++) require_ok(got[i] == -1234.5f, "q8 rows2 third row untouched");
        free(got);
    }
    const uint64_t before = ds4_gpu_qwen35_q8_rows_dispatches();
    require_ok(!ds4_gpu_qwen35_matmul_q8_0_rows_tensor(g3, a->base, a->size, off, in_dim, out_dim, gx, 1u) &&
               !ds4_gpu_qwen35_matmul_q8_0_rows_tensor(g3, a->base, a->size, off, in_dim, out_dim, gx, 3u) &&
               ds4_gpu_qwen35_q8_rows_dispatches() == before, "q8 rows2 refuses n_rows != 2");
    printf("  q8_0 rows2 %ux%u: both rows bit-identical to T=1 (x3)\n", in_dim, out_dim);
    free(ref[0]); free(ref[1]); free(x);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(g3);
}
```

In `main`, change `arena.size = (uint64_t)512 << 20;` to `arena.size = (uint64_t)1024 << 20;   /* M7's rows2 cases add ~190 MB of Q8_0 */` and add, right after `require_ok(ds4_gpu_set_model_map(...))`:

```c
    printf("qwen35 two-row Q8_0 matvec for the MTP verify (M7)\n");
    test_q8_rows2_exact(&arena, 2048u, 8192u);     /* lin_qkv, attn_q */
    test_q8_rows2_exact(&arena, 2048u, 4096u);     /* lin_gate */
    test_q8_rows2_exact(&arena, 4096u, 2048u);     /* lin_out, attn_output */
    test_q8_rows2_exact(&arena, 2048u, 65600u);    /* past 65536 outputs: NSG 8, as the lm head */
    test_q8_rows2_exact(&arena, 2048u, 4099u);     /* odd out_dim: half-empty last tile */
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make tests/test_qwen35_kernels 2>&1 | grep -E 'error' | head -3`
Expected: compile errors, implicit declaration of `ds4_gpu_qwen35_q8_rows_dispatches` / `ds4_gpu_qwen35_matmul_q8_0_rows_tensor`.

- [ ] **Step 3a (Task 1 verdict `build-kernel`): the kernel** (append to `metal/qwen35.metal`)

```metal
/* ---- M7: two-row Q8_0 matvec for the MTP verify ----------------------------
 * kernel_mul_mv_q8_0_f32_impl (dense.metal) for R input rows at once: the
 * same K walk, the same per-row multiply-adds (sumq over NQ quants, then
 * sumf += sumq * d) and the same reduction tree (simd_sum, threadgroup
 * partials, simd_sum) with the same NR0 and NSG, while each weight block is
 * loaded once for all rows.  Row r of dst is therefore bit-identical to a
 * kernel_mul_mv_q8_0_f32 dispatch on row r alone.  Each row reduces in its
 * own NR0 x 32-float threadgroup region, so row 1's zeroing cannot race row
 * 0's reads. */
template<short NR0, short R>
void kernel_qwen35_mv_q8_0_rows_impl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x*NR0;

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q8_0 *) (src0 + (r0 + row)*args.nb01);
    }

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float sumf[R][NR0];
    device const float * yb[R];
    FOR_UNROLL (short r = 0; r < R; ++r) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) sumf[r][row] = 0.f;
        yb[r] = (device const float *) (src1 + (uint64_t)r*args.nb11) + ib0*QK8_0 + il*NQ;
    }

    float yl[R][NQ];

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        FOR_UNROLL (short r = 0; r < R; ++r) {
            for (short i = 0; i < NQ; ++i) {
                yl[r][i] = yb[r][i];
            }
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            const half d = ax[row][ib].d;

            FOR_UNROLL (short r = 0; r < R; ++r) {
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[r][i];
                }

                sumf[r][row] += sumq*d;
            }
        }

        FOR_UNROLL (short r = 0; r < R; ++r) yb[r] += NSG*NQ*QK8_0;
    }

    FOR_UNROLL (short r = 0; r < R; ++r) {
        device float * dst_f32 = (device float *) dst + (uint64_t)r*args.ne0;
        helper_mv_reduce_and_write<NR0, false>(dst_f32, sumf[r], r0, args.ne01, tiisg, sgitg,
                                               shmem + r*NR0*NW*sizeof(float));
    }
}

[[host_name("kernel_qwen35_mv_q8_0_rows2")]]
kernel void kernel_qwen35_mv_q8_0_rows2(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_qwen35_mv_q8_0_rows_impl<N_R0_Q8_0, 2>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
```

(`const half d` keeps `sumq*d` a float x half product exactly as `sumq*ax[row][ib].d` in the one-row kernel.)

Host side, in `ds4_metal.m` after `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`:

```objc
static uint64_t g_qwen35_q8_rows_dispatches;

uint64_t ds4_gpu_qwen35_q8_rows_dispatches(void) {
    return g_qwen35_q8_rows_dispatches;
}

/* Ornith's MTP verify: two rows through one Q8_0 weight, each weight block
 * loaded once (kernel_qwen35_mv_q8_0_rows2).  NR0, NSG and the arguments
 * are the T=1 path's (ds4_gpu_matmul_q8_0_legacy_tensor, n_tok == 1), so row
 * r equals that path's output for row r bit for bit. */
int ds4_gpu_qwen35_matmul_q8_0_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map || n_rows != 2u || in_dim == 0 || out_dim == 0 ||
        (in_dim & 31u) != 0 || in_dim > INT32_MAX || out_dim > INT32_MAX ||
        ds4_gpu_tensor_bytes(x) < (uint64_t)n_rows * in_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(out) < (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }

    @autoreleasepool {
        id<MTLBuffer> xbuf = ds4_gpu_tensor_buffer(x);
        id<MTLBuffer> outbuf = ds4_gpu_tensor_buffer(out);
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        if (!xbuf || !outbuf || out_dim > UINT64_MAX / row_bytes) return 0;
        const uint64_t weight_bytes = out_dim * row_bytes;
        if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;

        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(model_map, model_size, weight_offset, weight_bytes,
                                                      &inner_offset);
        if (!wbuf) return 0;

        ds4_gpu_mv_dispatch dispatch = ds4_gpu_make_q8_0_mv_dispatch();
        if (out_dim > 65536u) dispatch.nsg = 8;
        ds4_gpu_q8_0_matvec_args args = ds4_gpu_make_q8_0_mv_args(in_dim, out_dim);
        args.nr0 = dispatch.nr0;

        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_mul_mv_pipeline("kernel_qwen35_mv_q8_0_rows2", dispatch.nsg);
        if (!pipeline) return 0;

        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:xbuf offset:ds4_gpu_tensor_offset(x) atIndex:2];
        [enc setBuffer:outbuf offset:ds4_gpu_tensor_offset(out) atIndex:3];
        [enc setThreadgroupMemoryLength:(NSUInteger)n_rows * dispatch.smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_dim + (NSUInteger)dispatch.nr0 - 1u) /
                                                  (NSUInteger)dispatch.nr0, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);

        if (!ds4_gpu_finish_command_buffer(cb, owned, "Qwen35 Q8_0 two-row matvec")) return 0;
        g_qwen35_q8_rows_dispatches++;
        return 1;
    }
}
```

- [ ] **Step 3b (Task 1 verdict `use`): the existing dispatch instead** — no kernel; the same two functions in `ds4_metal.m`, with the body of `ds4_gpu_qwen35_matmul_q8_0_rows_tensor` reduced to:

```objc
    if (n_rows != 2u) return 0;
    if (!ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model_map, model_size, weight_offset, in_dim, out_dim,
                                                      x, n_rows)) return 0;
    g_qwen35_q8_rows_dispatches++;
    return 1;
```

- [ ] **Step 4: Declare both functions in `ds4_gpu.h`** (after `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`), with the doc comments from the Interfaces block.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `make -j8 tests/test_qwen35_kernels && ./tests/test_qwen35_kernels 2>&1 | tail -12`
Expected: five `q8_0 rows2 ...: both rows bit-identical to T=1 (x3)` lines and `qwen35 kernels: ok`.

If a row differs: use superpowers:systematic-debugging (compare the one-row and two-row kernels' generated order; check `d`'s type, the shmem regions and NSG); if bit-exactness cannot be reached, take spec §5's fallback (plain Ornith decode moves to an R = 1 instance of the same template) as a ledgered ruling and re-plan Tasks 3-5's identity expectations accordingly.

- [ ] **Step 6: Add the `rows2` method to the bench**

In `tests/bench_qwen35_verify.c`: `typedef enum { M_T1, M_T1X2, M_ROWS_EXACT, M_ROWS2, N_METHODS } method_t;`, `METHOD_NAMES` gains `"rows2"`, and `call()` gains:

```c
    case M_ROWS2:
        return ds4_gpu_qwen35_matmul_q8_0_rows_tensor(c->o, c->base, c->size, off, c->in_dim, c->out_dim, c->x, 2u);
```

and after the verdict line:

```c
    printf("bench: stop-rule rows2 %s (2048x8192 x%.2f, lm_head x%.2f; limit x1.50)\n",
           ratio_rows2[0] < 1.5 && ratio_rows2[3] < 1.5 ? "pass" : "STOP", ratio_rows2[0], ratio_rows2[3]);
```

with `double ratio_rows2[N_SHAPES];` filled like `ratio_exact` (`if (m == M_ROWS2) ratio_rows2[s] = ms / t1;`).

- [ ] **Step 7: Build, Qwen kernel tests, commit**

Run: `make -j8 tests/test_qwen35_kernels tests/bench_qwen35_verify ds4 ds4-server 2>&1 | grep -ciE 'warning|error'` -> `0`; `make test-qwen4-kernels 2>&1 | tail -2` -> ok.

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_verify.c
git commit -m "qwen35: two-row Q8_0 matvec for the MTP verify, bit-identical to the one-row kernel per row"
```

- [ ] **Step 8 (controller, GPU window): stop rule**

Window steps: `./tests/bench_qwen35_verify` (twice) on the Task 2 snapshot; `make test-qwen4-q2`; `speed-bench/qwen-regression/run.sh fast`.
Expected: `bench: stop-rule rows2 pass` in both runs; Qwen gates PASS. `STOP` -> stop the plan, add the numbers to `BENCH.md`, report to the user.

- [ ] **Step 9: Update `BENCH.md` (rows2 rows, stop rule, Qwen fast gate) and commit**

```bash
git add speed-bench/ornith/m7/BENCH.md
git commit -m "speed-bench/ornith: M7 two-row matvec timing and stop rule"
```

---

### Task 3: Knob and batched dense projections in the verify

**Files:**
- Modify: `ds4_qwen35moe.inc` (above and inside `qwen35_gemv`)
- Create: `tests/test_qwen35_verify_batch.c`
- Modify: `Makefile` (object/link rules next to `tests/test_qwen35_mtp`, phony target next to `test-qwen35-mtp`, `clean` list)

**Interfaces:**
- Consumes: `ds4_gpu_qwen35_matmul_q8_0_rows_tensor`, `ds4_gpu_qwen35_q8_rows_dispatches` (Task 2).
- Produces: `static bool qwen35_verify_batch_env(void)` in `ds4_qwen35moe.inc` (Task 4 and Task 6 use it); `./tests/test_qwen35_verify_batch MODEL` with `VERIFY_ROWS` per verify.

- [ ] **Step 1: Write the failing test** (`tests/test_qwen35_verify_batch.c`)

```c
/* Real-model check of the M7 batched verify.
 * Usage: test_qwen35_verify_batch MODEL
 * Sets DS4_QWEN35_VERIFY_BATCH=1 before the engine opens (the knob is read
 * once).  150 greedy speculative cycles from a 1500-token prompt: after
 * every cycle the logits are bit-identical to a plain session fed the
 * committed tokens, and every verify issues exactly VERIFY_ROWS two-row
 * Q8_0 matvecs (ds4_gpu_qwen35_q8_rows_dispatches): attn_q and attn_output
 * in each of Ornith's 10 attention layers, plus the lm head. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CTX = 4160, CHUNK = 512, CYCLES = 150, VERIFY_ROWS = 10 * 2 + 1 };

static int g_vocab;
static float *g_a, *g_b;

static void sync_len(ds4_session *s, const ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char err[256] = {0};
    const int rc = ds4_session_sync(s, &prefix, err, sizeof(err));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, err);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

static void same_logits(ds4_session *spec, ds4_session *plain) {
    assert(ds4_session_copy_logits(spec, g_a, g_vocab) == g_vocab);
    assert(ds4_session_copy_logits(plain, g_b, g_vocab) == g_vocab);
    assert(memcmp(g_a, g_b, (size_t)g_vocab * sizeof(float)) == 0);
    assert(ds4_session_pos(spec) == ds4_session_pos(plain));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    setenv("DS4_QWEN35_VERIFY_BATCH", "1", 1);
    unsetenv("DS4_QWEN35_SPEC_FORCE_ACCEPT");
    ds4_engine_options opt = {.model_path = argv[1], .context_size = CTX, .prefill_chunk = CHUNK,
                              .backend = DS4_BACKEND_METAL, .glm_mtp = true};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen35moe(engine));
    const int eos = ds4_token_eos(engine);
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    assert(fp);
    char *text = calloc(40001, 1);
    assert(fread(text, 1, 40000, fp) > 0);
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len >= 1500);
    g_vocab = ds4_engine_vocab_size(engine);
    g_a = malloc((size_t)g_vocab * sizeof(float));
    g_b = malloc((size_t)g_vocab * sizeof(float));
    ds4_session *spec = NULL, *plain = NULL;
    assert(ds4_session_create(&spec, engine, CTX) == 0);
    assert(ds4_session_create(&plain, engine, CTX) == 0);
    char err[256] = {0};
    sync_len(spec, &tokens, 1500);
    sync_len(plain, &tokens, 1500);
    same_logits(spec, plain);
    int verifies = 0, accepts = 0;
    for (int c = 0; c < CYCLES; c++) {
        const uint64_t before = ds4_gpu_qwen35_q8_rows_dispatches();
        const int first = ds4_session_argmax(spec);
        int acc[17];
        const int n = ds4_session_eval_speculative_argmax(spec, first, 16, eos, acc, 17, err, sizeof(err));
        if (n < 0) fprintf(stderr, "cycle %d: %s\n", c, err);
        assert(n >= 1 && n <= 2 && acc[0] == first);
        const uint64_t d = ds4_gpu_qwen35_q8_rows_dispatches() - before;
        if (d != 0u && d != (uint64_t)VERIFY_ROWS) {
            fprintf(stderr, "cycle %d: %llu two-row matvecs, expected 0 or %d\n", c, (unsigned long long)d,
                    VERIFY_ROWS);
            return 1;
        }
        verifies += d == (uint64_t)VERIFY_ROWS;
        accepts += n == 2;
        for (int i = 0; i < n; i++) {
            assert(ds4_session_argmax(plain) == acc[i]);
            char perr[256] = {0};
            if (ds4_session_eval(plain, acc[i], perr, sizeof(perr)) != 0) {
                fprintf(stderr, "plain eval: %s\n", perr);
                return 1;
            }
        }
        same_logits(spec, plain);
    }
    printf("  %d cycles, %d verifies with %d two-row matvecs each, %d accepted, bit-identical to plain\n",
           CYCLES, verifies, VERIFY_ROWS, accepts);
    assert(verifies >= CYCLES - 5 && accepts >= 10);
    ds4_session_free(plain);
    ds4_session_free(spec);
    ds4_tokens_free(&tokens);
    free(g_a);
    free(g_b);
    ds4_engine_close(engine);
    printf("qwen35 verify batch: ok\n");
    return 0;
}
```

Makefile, next to the `tests/test_qwen35_mtp` rules (same conditional block):

```make
tests/test_qwen35_verify_batch.o: tests/test_qwen35_verify_batch.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen35_verify_batch.c

tests/test_qwen35_verify_batch: tests/test_qwen35_verify_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
```

next to `test-qwen35-mtp`:

```make
.PHONY: test-qwen35-verify-batch
test-qwen35-verify-batch: tests/test_qwen35_verify_batch
	./tests/test_qwen35_verify_batch "$(DS4_ORNITH_MODEL)"
```

and add `tests/test_qwen35_verify_batch tests/bench_qwen35_verify` to the `rm -f tests/test_qwen35_kernels ...` line of `clean`.

- [ ] **Step 2: Build it and commit the test** (the RED run happens in Step 6's window)

Run: `make -j8 tests/test_qwen35_verify_batch 2>&1 | grep -ciE 'warning|error'`
Expected: `0`.

```bash
git add tests/test_qwen35_verify_batch.c Makefile
git commit -m "tests: Ornith batched-verify identity and dispatch-count check"
```

Record this commit as `RED3`.

- [ ] **Step 3: The knob and the batched branch** (`ds4_qwen35moe.inc`)

Above the `qwen35_gemv` comment:

```c
/* DS4_QWEN35_VERIFY_BATCH (M7, default off until its gates pass): a
 * verify's Q8_0 projections run both rows through one two-row matvec
 * (ds4_gpu_qwen35_matmul_q8_0_rows_tensor) that loads each weight block
 * once and keeps kernel_mul_mv_q8_0_f32's arithmetic for every row, so each
 * row stays bit-identical to the T=1 dispatch qwen35_gemv would otherwise
 * issue for it.  Other weight types keep the per-row loop. */
static bool qwen35_verify_batch_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_QWEN35_VERIFY_BATCH"); v = e && e[0] && e[0] != '0'; }
    return v;
}
```

In `qwen35_gemv`, after the `xrow`/`orow` line:

```c
    if (T == 2u && w->type == DS4_TENSOR_Q8_0 && qwen35_verify_batch_env()) {
        if (ds4_gpu_qwen35_matmul_q8_0_rows_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, T))
            return true;
        fprintf(stderr, "ds4: Ornith two-row verify matvec failed for %.*s\n", (int)w->name.len, w->name.ptr);
        return false;
    }
```

and extend the `qwen35_gemv` comment's last sentence: "With DS4_QWEN35_VERIFY_BATCH a 2-row verify's Q8_0 weights take one two-row matvec instead (same per-row arithmetic, weights read once)."

- [ ] **Step 4: Build and commit**

Run: `make -j8 ds4 ds4-server tests/test_qwen35_verify_batch tests/test_qwen35_mtp 2>&1 | grep -ciE 'warning|error'`
Expected: `0`.

```bash
git add ds4_qwen35moe.inc
git commit -m "qwen35: DS4_QWEN35_VERIFY_BATCH runs the verify's Q8_0 projections as two-row matvecs"
```

Record this commit as `GREEN3`.

- [ ] **Step 5: Model-free tests**

Run: `./tests/test_qwen35_kernels 2>&1 | tail -1`
Expected: `qwen35 kernels: ok`.

- [ ] **Step 6 (controller, GPU window): RED then GREEN**

Snapshots of `RED3` and `GREEN3`, each built (`make -j8 ds4 tests/test_qwen35_verify_batch tests/test_qwen35_mtp`). Window steps:
1. `RED3`: `./tests/test_qwen35_verify_batch "$DS4_ORNITH_MODEL"` -> expected FAIL on `verifies >= CYCLES - 5` (0 verifies counted).
2. `GREEN3`: same command -> expected `150 cycles, ~149 verifies with 21 two-row matvecs each, ... bit-identical to plain` and `qwen35 verify batch: ok`.
3. `GREEN3`: `DS4_QWEN35_VERIFY_BATCH=1 ./tests/test_qwen35_mtp "$DS4_ORNITH_MODEL"` -> `qwen35 mtp: ok`.
4. `GREEN3`: `./tests/test_qwen35_mtp "$DS4_ORNITH_MODEL"` (knob off) -> `qwen35 mtp: ok`.

A failure in 2-4 that is a memcmp mismatch -> superpowers:systematic-debugging before anything else.

- [ ] **Step 7: Ledger the window result** (no commit; receipts go into Task 5's `LEVERS.md`).

---

### Task 4: GDN verify layer

**Files:**
- Modify: `ds4_qwen35moe.inc` (new function before `qwen35_graph_linear_rows`, one branch inside it)
- Modify: `tests/test_qwen35_verify_batch.c` (`VERIFY_ROWS` and its comment)

**Interfaces:**
- Consumes: `qwen35_verify_batch_env()`, `qwen35_gemv` (Task 3); `qwen4_graph_fused`, `sub_seal`, `SUB_GDN_*` (static in `ds4.c`, visible because `ds4.c` includes this file after them).
- Produces: `static bool qwen35_graph_linear_verify(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l, uint32_t il)`.

- [ ] **Step 1: Update the test's expectation**

```c
enum { CTX = 4160, CHUNK = 512, CYCLES = 150, VERIFY_ROWS = 10 * 2 + 30 * 3 + 1 };
```

and the header comment's last sentence: "attn_q and attn_output in each of Ornith's 10 attention layers, lin_qkv, lin_gate and lin_out in each of its 30 GDN layers, plus the lm head."

- [ ] **Step 2: Build and commit** (RED runs in Step 6's window: 21 per verify, expected 111)

```bash
make -j8 tests/test_qwen35_verify_batch 2>&1 | grep -ciE 'warning|error'   # 0
git add tests/test_qwen35_verify_batch.c
git commit -m "tests: the batched verify also covers the GDN projections"
```

Record this commit as `RED4`.

- [ ] **Step 3: The GDN verify layer** (`ds4_qwen35moe.inc`, before the `qwen35_graph_linear_rows` comment)

```c
/* A verify's 2-row GDN layer under DS4_QWEN35_VERIFY_BATCH: the sequence of
 * two T=1 qwen4_graph_linear calls (ds4.c), with its three Q8_0 projections
 * (lin_qkv, lin_gate, lin_out) batched over both rows through qwen35_gemv
 * and everything recurrent -- front (conv + alpha/beta), scan and gated
 * norm -- run one row at a time, in row order, on row views of the scratch.
 * An MTP graph never takes the q8_pair fusion (g->mtp_R is set), so the
 * projections are the same kernel_mul_mv_q8_0_f32 arithmetic either way.
 * Only row 0's front and scan receive the after-first-row snapshot. */
static ds4_gpu_tensor *qwen35_row_view(const ds4_gpu_tensor *t, uint32_t row, uint64_t n) {
    return ds4_gpu_tensor_view(t, (uint64_t)row * n * sizeof(float), n * sizeof(float));
}

static bool qwen35_graph_linear_verify(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                       uint32_t il) {
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    bool ok = qwen35_gemv(g, g->qkv, m, l->lin_qkv, g->mixed, 2u);
    sub_seal(SUB_GDN_QKV);
    ok = ok && qwen35_gemv(g, g->z, m, l->lin_gate, g->mixed, 2u);
    sub_seal(SUB_GDN_Z);
    for (uint32_t t = 0; t < 2u && ok; t++) {
        ds4_gpu_tensor *qkv = qwen35_row_view(g->qkv, t, conv_dim);
        ds4_gpu_tensor *z = qwen35_row_view(g->z, t, v_dim);
        ds4_gpu_tensor *ga = qwen35_row_view(g->ga, t, DS4_N_LIN_V_HEAD);
        ds4_gpu_tensor *gb = qwen35_row_view(g->gb, t, DS4_N_LIN_V_HEAD);
        ds4_gpu_tensor *lo = qwen35_row_view(g->lin_o, t, v_dim);
        ds4_gpu_tensor *mx = qwen35_row_view(g->mixed, t, DS4_N_EMBD);
        const bool snap = t == 0u && g->snap_after_first;
        ok = qkv && z && ga && gb && lo && mx;
        if (ok && qwen4_graph_fused(g, 1u)) {
            ok = ds4_gpu_qwen4_gdn_front_tensor(qkv, g->layer_lin_hist[il], mx, ga, gb, m->map, m->size,
                                                l->lin_conv->abs_offset, l->lin_alpha->abs_offset,
                                                l->lin_beta->abs_offset, l->lin_a->abs_offset,
                                                l->lin_dt_bias->abs_offset, l->lin_alpha->type, 1u,
                                                DS4_N_LIN_K_HEAD, DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM,
                                                DS4_N_LIN_CONV, DS4_N_EMBD, snap ? g->snap_lin_hist[il] : NULL, 0u,
                                                NULL, 1u) != 0;
        } else if (ok) {
            ok = qwen4_gemv(ga, m, l->lin_alpha, mx, 1u) &&
                 qwen4_gemv(gb, m, l->lin_beta, mx, 1u) &&
                 ds4_gpu_qwen4_conv_stream_tensor(qkv, g->layer_lin_hist[il], m->map, m->size,
                                                  l->lin_conv->abs_offset, 1u, (uint32_t)conv_dim, DS4_N_LIN_CONV,
                                                  true) &&
                 ds4_gpu_qwen4_gdn_prep_tensor(qkv, ga, gb, m->map, m->size, l->lin_a->abs_offset,
                                               l->lin_dt_bias->abs_offset, 1u, DS4_N_LIN_K_HEAD, DS4_N_LIN_V_HEAD,
                                               DS4_N_LIN_HEAD_DIM);
        }
        sub_seal(SUB_GDN_FRONT);
        if (ok) {
            ok = ds4_gpu_qwen4_gdn_scan_tensor(lo, g->layer_lin_state[il], qkv, ga, gb, 1u, DS4_N_LIN_K_HEAD,
                                               DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM,
                                               snap ? g->snap_lin_state[il] : NULL, 0u, NULL, 1u) != 0;
        }
        sub_seal(SUB_GDN_SCAN);
        if (ok) {
            ok = g->gdn_silu
                 ? ds4_gpu_qwen35_gdn_out_tensor(lo, z, m->map, m->size, l->lin_norm->abs_offset, 1u,
                                                 DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM, DS4_RMS_EPS) != 0
                 : ds4_gpu_qwen4_gdn_out_tensor(lo, z, m->map, m->size, l->lin_norm->abs_offset, 1u,
                                                DS4_N_LIN_V_HEAD, DS4_N_LIN_HEAD_DIM, DS4_RMS_EPS) != 0;
        }
        sub_seal(SUB_GDN_OUT);
        ds4_gpu_tensor_free(qkv); ds4_gpu_tensor_free(z); ds4_gpu_tensor_free(ga);
        ds4_gpu_tensor_free(gb); ds4_gpu_tensor_free(lo); ds4_gpu_tensor_free(mx);
    }
    ok = ok && qwen35_gemv(g, g->blk, m, l->lin_out, g->lin_o, 2u);
    sub_seal(SUB_GDN_PROJ);
    return ok;
}
```

Check against `qwen4_graph_linear` (ds4.c) line by line while writing: argument order of `gdn_front`, `conv_stream`, `gdn_prep`, `gdn_scan`, `gdn_out`; the non-fused branch takes no snapshot there either.

In `qwen35_graph_linear_rows`, after the `snap_after_second` refusal block:

```c
    if (T == 2u && qwen35_verify_batch_env()) return qwen35_graph_linear_verify(g, m, l, il);
```

and add to its comment: "With DS4_QWEN35_VERIFY_BATCH a 2-row verify takes qwen35_graph_linear_verify instead (projections batched, the recurrence still one row at a time)."

- [ ] **Step 4: Build and commit**

Run: `make -j8 ds4 ds4-server tests/test_qwen35_verify_batch tests/test_qwen35_mtp 2>&1 | grep -ciE 'warning|error'`
Expected: `0`.

```bash
git add ds4_qwen35moe.inc
git commit -m "qwen35: the batched verify's GDN layers read lin_qkv, lin_gate and lin_out once for both rows"
```

Record this commit as `GREEN4`.

- [ ] **Step 5: Model-free tests** — `./tests/test_qwen35_kernels 2>&1 | tail -1` -> `qwen35 kernels: ok`.

- [ ] **Step 6 (controller, GPU window): RED then GREEN**

Snapshots of `RED4` and `GREEN4`. Window steps:
1. `RED4`: `./tests/test_qwen35_verify_batch "$DS4_ORNITH_MODEL"` -> expected FAIL: `cycle N: 21 two-row matvecs, expected 0 or 111`.
2. `GREEN4`: same -> `... with 111 two-row matvecs each ... bit-identical to plain`, `qwen35 verify batch: ok`.
3. `GREEN4`: `DS4_QWEN35_VERIFY_BATCH=1 ./tests/test_qwen35_mtp "$DS4_ORNITH_MODEL"` -> `qwen35 mtp: ok`.
4. `GREEN4`: profile, on a ~2K-token prefix of the M4 profile text (`head -c 8000 speed-bench/promessi_sposi.txt > <out>/prompt-2k.txt`): `DS4_QWEN35_VERIFY_BATCH=1 DS4_QWEN35_PROFILE=2 ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --mtp -c 8192 -n 256 --temp 0 --prompt-file <out>/prompt-2k.txt` and the same with `DS4_QWEN35_VERIFY_BATCH=0`; save both logs under `speed-bench/ornith/m7/profile/`. Level 2 prints the T=2 verify's stage line (`Ornith stage ms/chunk (pos=... T=2)`) per verify and the T=1 decode average every 64 plain steps.

Expected in step 4: mean verify time over the mean plain-step time <= 1.35 with the knob on (today ~1.74).

- [ ] **Step 7: Ledger the window result** (receipts go into Task 5's `LEVERS.md`).

---

### Task 5: Gates and lever A/B with the knob on

**Files:**
- Create: `speed-bench/ornith/m7/LEVERS.md`, `speed-bench/ornith/m7/levers/`, `speed-bench/ornith/m7/profile/`

**Interfaces:**
- Consumes: `GREEN4` (or later) build; `tests/ornith/test_mtp_cli.py OUT_DIR`, `tests/ornith/gate1.py [--ds4-arg ARG ...] OUT_DIR`, `speed-bench/ornith/m4_ab.py`, `speed-bench/qwen-regression/run.sh`.
- Produces: the adoption decision for Task 6.

- [ ] **Step 1 (controller, GPU window): correctness with the knob on**

On the `GREEN4` snapshot, `DS4_QWEN35_VERIFY_BATCH=1` in every step's env:
1. `python3 tests/ornith/test_mtp_cli.py <out>/mtp_cli` -> PASS (`--mtp` byte-identical to plain, chunks 1/2/64 + long_copy).
2. `DS4_TEST_MODEL="$DS4_ORNITH_MODEL" ./ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` -> `ds4 tests: ok`; the same with `DS4_TEST_GLM_MTP=1` -> ok.
3. `make test-qwen35-graph` -> ok.
4. For chunks 2048, 64, 65: `python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg $c <out>/gate1-$c` -> PASS x3; for chunk 2048, compare its `ds4/*.json` dumps with a develop `b764a66` snapshot's run of the same command: 0 differ.
5. `speed-bench/qwen-regression/run.sh fast` -> PASS.

- [ ] **Step 2 (controller, GPU window): lever A/B**

`python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m7/levers/ab-verify-batch --base-env "DS4_QWEN35_VERIFY_BATCH=0 DS4_QWEN35_MTP_DRAFT_VOCAB=$SNAP/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt" --lever-env "DS4_QWEN35_VERIFY_BATCH=1 DS4_QWEN35_MTP_DRAFT_VOCAB=$SNAP/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt" --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`
Expected: decode lever/base >= 1.05 at 2K (stop rule), no context below 0.97.

- [ ] **Step 3: Write `LEVERS.md` and commit**

Sections: correctness table (Tasks 3-5 windows: RED/GREEN runs, test_qwen35_mtp on/off, test_mtp_cli, ds4_test, gate 1 x3 + dumps, Qwen fast gate), profile (verify/plain ratio before/after), A/B table, verdict (adopt / stop).

```bash
git add speed-bench/ornith/m7/LEVERS.md speed-bench/ornith/m7/levers speed-bench/ornith/m7/profile
git commit -m "speed-bench/ornith: M7 batched-verify gates and lever A/B"
```

Stop rule: A/B < +5% at 2K -> stop here, report to the user (Task 6's flip and Task 7 are not done).

---

### Task 6: Default on, gate 3, head-to-head, Qwen full gate, report

**Files:**
- Modify: `ds4_qwen35moe.inc` (`qwen35_verify_batch_env` and its comment)
- Modify: `tests/test_qwen35_verify_batch.c` (argument modes `--default` and `--knob-off`)
- Modify: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (§5 verify paragraph, §7.3 knob list)
- Create: `speed-bench/ornith/m7/speed/GATE3.md`, `speed-bench/ornith/m7/h2h/`, `speed-bench/ornith/m7/QWEN_GATE.md`, `speed-bench/ornith/m7/REPORT.md`

**Interfaces:**
- Consumes: `qwen35_verify_batch_env()` (Task 3), `test_qwen35_verify_batch` (Task 4 state, `VERIFY_ROWS` = 111).
- Produces: the default-on build and the M7 report.

- [ ] **Step 1: Test modes** — `main` accepts an optional second argument: none -> `setenv("DS4_QWEN35_VERIFY_BATCH", "1", 1)` (as today); `--default` -> `unsetenv("DS4_QWEN35_VERIFY_BATCH")` and the same expectations; `--knob-off` -> `setenv(..., "0", 1)` and every cycle must count 0 two-row matvecs (identity still checked, `verifies == 0`, `accepts >= 10`). `argc` 2 or 3; the usage line names both modes.

```c
    const char *mode = argc == 3 ? argv[2] : "";
    const bool knob_off = strcmp(mode, "--knob-off") == 0;
    if (argc > 3 || (argc == 3 && !knob_off && strcmp(mode, "--default") != 0)) {
        fprintf(stderr, "usage: %s MODEL [--default | --knob-off]\n", argv[0]);
        return 1;
    }
    if (knob_off) setenv("DS4_QWEN35_VERIFY_BATCH", "0", 1);
    else if (argc == 3) unsetenv("DS4_QWEN35_VERIFY_BATCH");
    else setenv("DS4_QWEN35_VERIFY_BATCH", "1", 1);
    const uint64_t expect = knob_off ? 0u : (uint64_t)VERIFY_ROWS;
```

with the per-cycle check becoming `if (d != 0u && d != expect)` (message "expected 0 or %llu"), `verifies += expect && d == expect;`, and the final assertion `assert((knob_off ? verifies == 0 : verifies >= CYCLES - 5) && accepts >= 10);`. Commit as `RED6` ("tests: batched-verify default and knob-off modes").

- [ ] **Step 2: Flip the default**

```c
/* DS4_QWEN35_VERIFY_BATCH (M7, default on; =0 restores the per-row
 * verify): a verify's Q8_0 projections run both rows through one two-row
 * matvec (ds4_gpu_qwen35_matmul_q8_0_rows_tensor) that loads each weight
 * block once and keeps kernel_mul_mv_q8_0_f32's arithmetic for every row,
 * so each row stays bit-identical to the T=1 dispatch qwen35_gemv would
 * otherwise issue for it.  Other weight types keep the per-row loop. */
static bool qwen35_verify_batch_env(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_QWEN35_VERIFY_BATCH"); v = !(e && strcmp(e, "0") == 0); }
    return v;
}
```

Parent spec §5: after the sentence ending "`lin_qkv`/`lin_gate`/`lin_out` projections).", add: "Since M7 (`DS4_QWEN35_VERIFY_BATCH`, default on; `=0` restores the per-row dispatch) the verify's Q8_0 projections — attention q/output, the GDN `lin_qkv`/`lin_gate`/`lin_out` and the lm head — run as one two-row matvec (`kernel_qwen35_mv_q8_0_rows2`) that loads each weight once and keeps the one-row kernel's arithmetic per row (memcmp-tested); F16/F32 projections, the GDN recurrence and the experts stay per row." §7.3 list: add "- `DS4_QWEN35_VERIFY_BATCH` — two-row Q8_0 matvecs in the MTP verify (M7; default 1, 0 = one T=1 dispatch per row)." Commit as `GREEN6` ("qwen35: the batched verify is the Ornith default").

- [ ] **Step 3 (controller, GPU window): RED/GREEN + gate 3 + h2h + Qwen full**

1. `RED6`: `./tests/test_qwen35_verify_batch "$DS4_ORNITH_MODEL" --default` -> FAIL (0 verifies counted).
2. `GREEN6`: `--default` -> ok; `--knob-off` -> ok; no argument -> ok; `./tests/test_qwen35_mtp "$DS4_ORNITH_MODEL"` -> ok; `python3 tests/ornith/test_mtp_cli.py <out>/mtp_cli` -> PASS.
3. Gate 3: `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m7/speed/final --ds4-env "DS4_QWEN35_MTP_DRAFT_VOCAB=$SNAP/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt" --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`.
4. Head-to-head: `python3 speed-bench/ornith/m6/h2h.py --ds4-model "$DS4_ORNITH_MODEL" --out speed-bench/ornith/m7/h2h --ds4-env "DS4_QWEN35_MTP_DRAFT_VOCAB=$SNAP/speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt" --contexts 2048,8192,32768,65536,131072 --reps-short 2` (192K/250K stay M6's rows: the verify change does not touch prefill, and the long rows cost ~2 h), then `python3 speed-bench/ornith/m6/plot_h2h.py` on it.
5. `speed-bench/qwen-regression/run.sh full` -> PASS.

- [ ] **Step 4: Receipts, report, commit**

`speed/GATE3.md` against spec §1 criteria 1-2 (decode vs the live oMLX at 2K and 32K; 128K within 3% of 40.1); `h2h/` (json, csv, txt, svg); `QWEN_GATE.md`; `REPORT.md` (what changed, identity evidence, speed tables vs M6 and oMLX, criterion 1 met or not, the Task 7 decision). Gate 2: M7 output is byte-identical to M6's (plain decode unchanged: gate 1 dumps; `--mtp` identical to plain), so M6's `quality/GATE2.md` stands; `REPORT.md` says so.

```bash
git add speed-bench/ornith/m7
git commit -m "speed-bench/ornith: M7 gate 3, head-to-head and report"
```

---

### Task 7 (conditional): MoE overlap measurement and stage 2 addendum

Only if Task 6's gate 3 misses criterion 1 (decode >= oMLX at 2K and 32K).

**Files:**
- Create: `speed-bench/ornith/m7/OVERLAP.md`
- Modify: `docs/superpowers/specs/2026-09-28-ornith-m7-verify-batch-design.md` (addendum section, only if the overlap clears the bar)

- [ ] **Step 1: Debug patch (not committed)** — in a snapshot of the Task 6 commit, `qwen35_graph_moe` under `g->verify_rows_exact && T == 2u && getenv("DS4_QWEN35_VERIFY_OVERLAP")`: after `ds4_gpu_qwen4_router_topk_tensor`, `ds4_gpu_synchronize()`, read `g->selected` (2 x 8 int32), count the experts both rows select, accumulate per layer, and print `ds4: verify overlap layer L: mean S/8 over N verifies` every 64 verifies.
- [ ] **Step 2 (controller, GPU window):** `DS4_QWEN35_VERIFY_OVERLAP=1 ./ds4 --mtp ...` on three prompts (English prose, Vietnamese, code) at 2K, 256 tokens each.
- [ ] **Step 3:** `OVERLAP.md` with the per-layer means and the overall shared fraction. >= ~40% -> write the spec addendum (a 2-row MoE path that loads each shared expert once, per-row arithmetic of the row kernels kept, its exactness test and its A/B) and stop for the user's review; < ~40% -> record that stage 2 is dropped and stop. Commit `OVERLAP.md` (and the addendum if written).

---

## Execution notes

- Tasks 1-2 are model-free until their GPU-window steps; Tasks 3-4 build RED and GREEN commits first and run both in one window each; Task 5 and Task 6 are window-heavy (the lever A/B and gate 3 take ~1 h each, the head-to-head ~1.5 h).
- Every window restores the live stack, including after a failure (the `gpuwin.sh` trap).
