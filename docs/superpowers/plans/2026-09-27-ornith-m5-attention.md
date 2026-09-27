# Ornith M5: Attention kernels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the long-context gap to the live oMLX by replacing Ornith's attention kernels: a query-token-tiled flash prefill kernel (`kernel_qwen35_attn_flash`) with key split, and a decode/verify kernel (`kernel_qwen35_attn_decode3`) with more splits, keys spread across simdgroups and a parallel merge — while keeping `--mtp` byte-identical to plain decoding and Qwen3.8 byte-identical.

**Architecture:** Ornith-only and additive: kernels in `metal/qwen35.metal`, wrappers in `ds4_metal.m` (declared in `ds4_gpu.h`), dispatch in `ds4_qwen35moe.inc` behind `DS4_QWEN35_ATTN_FLASH` and `DS4_QWEN35_ATTN_DECODE`. A model-free micro-benchmark (`tests/bench_qwen35_attn.c`) measures every kernel against the K/V bandwidth floor and picks the tile parameters; the M4 harness (`speed-bench/ornith/m4_ab.py`) decides adoption and the final gates.

**Tech Stack:** C, Objective-C (`ds4_metal.m`), Metal Shading Language (`metal/qwen35.metal`), Python 3 (M4 harness, gate scripts).

**Spec:** `docs/superpowers/specs/2026-09-27-ornith-m5-attention-design.md` (parent: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today. Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`, `ds4_agent.c`, `ds4_kvstore.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge.
- Metal only for Ornith. Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no `DS4_QWEN4_*` knob.
- Kernel changes are additive: new kernels in `metal/qwen35.metal`; `metal/qwen4.metal` is never edited. New kernels must compile with `DS4_METAL_DISABLE_METAL4=1` (no Metal 4 tensor API in M5).
- `--mtp` greedy output stays byte-identical to plain decoding (M2). Plain decode and each 2-row verify row run the same decode kernel at the same per-row split geometry (M4's L12 contract).
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`; the 25G tier sits next to it.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process; scripts stop processes with SIGTERM and a bounded wait.
- GPU windows (model runs, speed measurements, the benchmark's timed runs) pause the live stack and restore it afterwards (the controller's `gpuwin.sh` pattern from M4); implementers build and run model-free kernel tests only.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Plan-specific constraints:
- Branch `feature/ornith-m5` (holds the spec commit `df108c1`); merging or pushing needs the user's explicit OK.
- Adoption rule (M4): a new kernel becomes the default only after gate 1 passes at prefill chunks 2048/64/65, the MTP identity tests pass (`make test-qwen35-graph test-qwen35-mtp`, `tests/ornith/test_mtp_cli.py`) and a `m4_ab.py --mode lever` A/B shows a gain; otherwise it stays opt-in and the receipt says why.
- Existing unit tests are never edited; new tests are additions.
- Kernel bodies are developed test-first in their task: the task gives the complete tests, the exact interfaces, the tile structure and the performance target; tile parameters are chosen with the benchmark, so the kernel source is not fixed in advance.

## Review Focus

1. **Causal mask inside a query-token block.** A block of Br tokens straddles several positions; token t may see keys up to pos0 + t only, so rows of one block have different key counts and the last tiles are partly masked. Pinned by `test_attn_flash` at T = 9, 65, 200 with pos0 = 0 and pos0 = 37 against a host double reference (Task 3).
2. **decode3 rows contract at split-count boundaries.** When the verify's two rows (key counts n0 = pos0+1, n1 = pos0+2) fall on different sides of a split boundary (ns0 != ns1) or kps0 != kps1, row 0 must still equal a solo decode bit for bit. Pinned by `test_attn_decode3_rows` at pos0 values chosen at the boundaries of the forced split size (Task 2).
3. **Partial-buffer sizing at long context.** 256 splits x 16 heads x (2 + 256) floats x rows must fit the scratch the graph allocates, for both kernels. Pinned by the benchmark at 131,072 keys and by `test_attn_decode3_rows` at the split cap (Tasks 1-2, 4).
4. **Prefill tails and chunk sizes 1/2/64.** T <= 8 tails keep today's path and chunked prefill must stay identical between `--mtp` and plain runs. Pinned by `tests/ornith/test_mtp_cli.py` (chunks 1, 2, 64 + long_copy) with the knobs on (Tasks 5-6).
5. **fp8/q4 K/V (opt-in) keeps working.** The new kernels read F16 K/V only; the dispatch must fall back for `DS4_QWEN35_KV=fp8|q4`. Pinned by `ds4_test --qwen35-payloads` (f16/fp8/q4 loop) with the knobs on (Tasks 5-6).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `tests/bench_qwen35_attn.c` | Create | model-free timing of attention kernels vs the K/V bandwidth floor |
| `Makefile` | Modify | `tests/bench_qwen35_attn`, `bench-qwen35-attn` |
| `metal/qwen35.metal` | Modify | `kernel_qwen35_attn_decode3`, `kernel_qwen35_attn_merge3`, `kernel_qwen35_attn_flash` (+ split variant) |
| `ds4_metal.m`, `ds4_gpu.h` | Modify | enums/names, `ds4_gpu_qwen35_attn_decode3_tensor`, `ds4_gpu_qwen35_attn_flash_tensor`, `ds4_gpu_qwen35_attn_part3_floats` |
| `ds4_qwen35moe.inc` | Modify | knobs, dispatch in `qwen35_graph_attend`, scratch sizing |
| `tests/test_qwen35_kernels.c` | Modify (additions) | `test_attn_decode3_rows`, `test_attn_flash` |
| `speed-bench/ornith/m5/` | Create | `BENCH.md`, `LEVERS.md`, `speed/GATE3.md`, `quality/GATE2.md`, `QWEN_GATE.md`, `REPORT.md` |
| `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` | Modify | §4 attention text for the new kernels |

---

### Task 1: Benchmark harness and baseline

**Files:**
- Create: `tests/bench_qwen35_attn.c`
- Modify: `Makefile`
- Create (controller, after the GPU run): `speed-bench/ornith/m5/BENCH.md`

**Interfaces:**
- Consumes: `ds4_gpu_qwen4_attn_decode_tensor` (today's prefill path for T > 8 via `kernel_qwen4_attn_mm`), `ds4_gpu_qwen35_attn_decode2_tensor` (today's decode), `ds4_gpu_qwen4_attn_part_floats`.
- Produces: `./tests/bench_qwen35_attn [prefill|decode|all]` printing one line per case: `bench <kernel> <mode> pos=<P> T=<T> rows=<R>: <ms> ms, K/V read <GB> GB -> <GB/s> GB/s (floor <ms_floor> ms)`. Later tasks add their kernels to the same table.

- [ ] **Step 1: Write the benchmark**

Create `tests/bench_qwen35_attn.c`:

```c
/* Model-free attention micro-benchmark for Ornith's shape (16 query heads,
 * 2 KV heads, head dim 256, F16 K/V).  Times each attention kernel at long
 * key counts and reports the K/V bytes it must read against a bandwidth
 * floor, so tile parameters can be chosen by measurement.  Needs the GPU but
 * no model; run it with the live stack paused for stable numbers. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

enum { H = 16, HKV = 2, D = 256 };
static const double FLOOR_GBS = 250.0;   /* assumed sustained read for the M5 Pro (~273 GB/s peak); the ratio is indicative */

static uint32_t g_rng = 0x2545f491u;
static float frand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xffffffu) / 16777216.0f * 2.0f - 1.0f;
}
static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t s = (v.u >> 16) & 0x8000u;
    int e = (int)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t m = v.u & 0x7fffffu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_attn: %s failed\n", what); exit(1); }
}
static ds4_gpu_tensor *rand_f32(uint64_t n, float scale) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = frand() * scale;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "f32 upload");
    free(h);
    return t;
}
static ds4_gpu_tensor *rand_f16(uint64_t n) {
    uint16_t *h = malloc(n * sizeof(uint16_t));
    for (uint64_t i = 0; i < n; i++) h[i] = f32_to_f16(frand());
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(uint16_t));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(uint16_t)), "f16 upload");
    free(h);
    return t;
}

/* median of `reps` timed calls after `warm` warm-up calls; each call ends in a
 * synchronize so the wall time is the kernel time plus one submit */
typedef int (*bench_fn)(void *ctx);
static double time_ms(bench_fn fn, void *ctx, int warm, int reps) {
    double t[16];
    for (int i = 0; i < warm; i++) need(fn(ctx) && ds4_gpu_synchronize(), "warm-up call");
    for (int i = 0; i < reps; i++) {
        const double t0 = now_s();
        need(fn(ctx) && ds4_gpu_synchronize(), "timed call");
        t[i] = (now_s() - t0) * 1e3;
    }
    for (int i = 1; i < reps; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[reps / 2];
}

typedef struct {
    ds4_gpu_tensor *q, *gate, *out, *part, *kc, *vc;
    uint32_t pos0, T, rows;
    float scale;
} bench_ctx;

static int run_prefill_mm(void *p) {       /* today's prefill: kernel_qwen4_attn_mm via the qwen4 wrapper */
    bench_ctx *c = p;
    return ds4_gpu_qwen4_attn_decode_tensor(c->out, c->q, c->gate, c->kc, c->vc, NULL, NULL, NULL,
                                            c->T, H, HKV, D, c->pos0, 0u, 0u, c->scale,
                                            NULL, NULL, NULL, NULL, 0u);
}
static int run_decode2(void *p) {          /* today's decode / verify: L12 */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_decode2_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part,
                                              H, HKV, D, c->pos0, c->rows, c->scale, NULL, NULL, NULL, NULL, 0u);
}

/* K/V bytes a kernel must read at least once: every key <= the last query's
 * position, K and V, 2 KV heads x 256 halves each */
static double kv_gb(uint32_t pos0, uint32_t T) {
    return (double)(pos0 + T) * HKV * D * 2.0 * 2.0 / 1e9;
}
static void report(const char *kernel, const char *mode, uint32_t pos0, uint32_t T, uint32_t rows, double ms) {
    const double gb = kv_gb(pos0, T);
    printf("bench %-14s %-7s pos=%-6u T=%-4u rows=%u: %8.3f ms, K/V read %.3f GB -> %6.1f GB/s (floor %.3f ms)\n",
           kernel, mode, pos0, T, rows, ms, gb, gb / (ms * 1e-3), gb / FLOOR_GBS * 1e3);
}

int main(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    const int do_prefill = !strcmp(what, "all") || !strcmp(what, "prefill");
    const int do_decode = !strcmp(what, "all") || !strcmp(what, "decode");
    need(ds4_gpu_init(), "GPU initialization");
    const uint32_t cap = 131072u + 2048u + 8u;
    bench_ctx c = {0};
    c.scale = 1.0f / sqrtf((float)D);
    c.kc = rand_f16((uint64_t)cap * HKV * D);
    c.vc = rand_f16((uint64_t)cap * HKV * D);
    c.q = rand_f32(2048ull * H * D, 1.0f);
    c.gate = rand_f32(2048ull * H * D, 1.0f);
    c.out = ds4_gpu_tensor_alloc(2048ull * H * D * sizeof(float));
    c.part = ds4_gpu_tensor_alloc(ds4_gpu_qwen4_attn_part_floats(2048u, H, D) * sizeof(float));
    need(c.out && c.part, "output buffers");

    if (do_prefill) {
        const uint32_t pos[3] = { 0u, 30720u, 122880u };
        for (int i = 0; i < 3; i++) {
            c.pos0 = pos[i]; c.T = 2048u; c.rows = 0;
            report("qwen4_attn_mm", "prefill", c.pos0, c.T, 0, time_ms(run_prefill_mm, &c, 1, 3));
        }
    }
    if (do_decode) {
        const uint32_t pos[3] = { 2048u, 32768u, 131072u };
        for (int i = 0; i < 3; i++)
            for (uint32_t rows = 1; rows <= 2; rows++) {
                c.pos0 = pos[i]; c.T = rows; c.rows = rows;
                report("decode2", "decode", c.pos0, c.T, rows, time_ms(run_decode2, &c, 3, 9));
            }
    }
    ds4_gpu_tensor_free(c.kc); ds4_gpu_tensor_free(c.vc); ds4_gpu_tensor_free(c.q);
    ds4_gpu_tensor_free(c.gate); ds4_gpu_tensor_free(c.out); ds4_gpu_tensor_free(c.part);
    return 0;
}
```

- [ ] **Step 2: Makefile target**

In `Makefile`, next to the `tests/test_qwen35_kernels` rules (anchor: `tests/test_qwen35_kernels: tests/test_qwen35_kernels.o ds4_metal.o ds4_image.o`), add (recipe lines start with a tab):

```make
tests/bench_qwen35_attn.o: tests/bench_qwen35_attn.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/bench_qwen35_attn: tests/bench_qwen35_attn.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: bench-qwen35-attn
bench-qwen35-attn: tests/bench_qwen35_attn
	./tests/bench_qwen35_attn all
```

- [ ] **Step 3: Build and smoke-run the decode part**

```bash
make tests/bench_qwen35_attn 2>&1 | grep -c -i warning
./tests/bench_qwen35_attn decode | head -2
```
Expected: `0` warnings; two `bench decode2 decode pos=2048 ...` lines (the implementer may run this smoke with the live stack up — the numbers are not the receipt).

- [ ] **Step 4: Commit**

```bash
git add tests/bench_qwen35_attn.c Makefile
git commit -m "tests: Ornith attention micro-benchmark

Times the prefill (qwen4_attn_mm) and decode/verify (decode2) kernels at
Ornith's shape against the K/V bandwidth floor; later M5 kernels join the
same table.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- tests/bench_qwen35_attn.c Makefile
```

- [ ] **Step 5 (controller, GPU window): baseline receipt**

Run `./tests/bench_qwen35_attn all` with the live stack paused and write `speed-bench/ornith/m5/BENCH.md` with the table (baseline column) and the commit hash; commit it.

---

### Task 2: `kernel_qwen35_attn_decode3` and its merge

**Files:**
- Modify: `metal/qwen35.metal` (append after `kernel_qwen35_attn_decode2`), `ds4_metal.m`, `ds4_gpu.h`, `tests/test_qwen35_kernels.c` (additions), `tests/bench_qwen35_attn.c` (add decode3 cases)

**Interfaces:**
- Consumes: the L12 kernel as the reference design (`kernel_qwen35_attn_decode2`, metal/qwen35.metal ~851) and its host split helper `qwen4_attn_row_splits_host` (ds4_metal.m ~50778).
- Produces:
  - `uint64_t ds4_gpu_qwen35_attn_part3_floats(uint32_t rows, uint32_t n_head, uint32_t head_dim);` — scratch floats for `rows` rows at the decode3 split cap.
  - `int ds4_gpu_qwen35_attn_decode3_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, uint32_t rows, float scale);` — same meaning as decode2 (rows 1 = plain decode at `pos0`, rows 2 = verify rows at `pos0`, `pos0 + 1`), F16 K/V only, head_dim 256.
  - Knob (tuning/test only): `DS4_QWEN35_ATTN_SPLIT_KEYS` (minimum keys per split, default 64) and the constant `QWEN35_ATTN_MAX_SPLITS 256`.

**Design (binding):**
- Grid `(n_splits_max_of_rows, n_head_kv, 1)`, threadgroup `NSG = 4` simdgroups x 32 threads. Each thread holds `NPT = 8` dims of a head (256 = 32 x 8).
- Row r's split geometry depends only on its own key count n_r = pos0 + 1 + r: `ns_r = min(QWEN35_ATTN_MAX_SPLITS, ceil(n_r / split_keys))`, `kps_r = ceil(n_r / ns_r)`, split s covers `[s*kps_r, min(n_r, (s+1)*kps_r))`. Computed on the host (like decode2) and passed as `ns0, ns1`.
- Inside a threadgroup, simdgroup `sg` processes keys `lo_r + sg + NSG*j` of row r's range (striding relative to **the row's own** `lo_r`, never the union), for all 8 query heads of the KV head (`qv[8][NPT]`, `acc[8][NPT]`, `m[8]`, `l[8]` per row). A key both rows need is read once when both rows assign it to the same simdgroup (the common case kps0 == kps1); otherwise each row reads it separately — correctness never depends on sharing.
- K/V loads: one `half4` x 2 (or `uint4`) per thread per key row, converted to float.
- The NSG simdgroup partial states of a row are combined in threadgroup memory in the fixed order sg = 0, 1, 2, 3 (the standard online-softmax merge: m = max, l = sum l_i·exp(m_i − m), acc likewise). A row with `ns_r == 1` then writes the gated output directly; otherwise it writes (m, l, acc) partials to `part` in the layout `[row][Hkv][ns_r][group][2 + D]` (row 0's block first).
- `kernel_qwen35_attn_merge3`: grid `(n_head, rows, 1)`, one threadgroup of 256 threads per (head, row): each thread owns one dim; the threadgroup reduces the row's `ns_r` partials with a fixed-shape tree (pairs (0,1),(2,3)… then pairs of pairs), applies 1/l and the sigmoid gate. The tree shape depends only on `ns_r`.
- Because a row's key assignment, simdgroup merge order and split tree depend only on that row's own key count, a rows == 2 call equals two rows == 1 calls bit for bit.

- [ ] **Step 1: Write the failing kernel test**

Append to `tests/test_qwen35_kernels.c` before `int main` (reuses this file's helpers `arena_f32`, `rand_vec`, `upload`, `download`, `require_ok`, `f16_to_f32`):

```c
/* decode3 (M5): the rows==2 verify call equals two rows==1 calls of the same
 * kernel bit for bit, and both equal a host double reference within FP noise.
 * `split_keys` forces small splits so short caches exercise many splits and
 * the split-count boundaries (ns0 != ns1, kps0 != kps1). Outputs are
 * sentinel-filled before every call so a kernel that writes nothing fails. */
static void test_attn_decode3_rows(arena_t *a, uint32_t pos0, uint32_t split_keys) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, fill = pos0 + 2u;
    const uint32_t cap = fill + 8u;
    const float scale = 1.0f / sqrtf((float)D);
    char sk[16];
    snprintf(sk, sizeof(sk), "%u", split_keys);
    setenv("DS4_QWEN35_ATTN_SPLIT_KEYS", sk, 1);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "decode3 fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, 1.0e7f, 1e-6f, kc, vc, kc, vc, 0u),
               "decode3 fill");

    float *q2 = rand_vec(2ull * H * D, 1.0f), *g2 = rand_vec(2ull * H * D, 1.0f);
    ds4_gpu_tensor *gq2 = upload(q2, 2ull * H * D), *gg2 = upload(g2, 2ull * H * D);
    ds4_gpu_tensor *o2 = upload(NULL, 2ull * H * D);
    ds4_gpu_tensor *part2 = upload(NULL, ds4_gpu_qwen35_attn_part3_floats(2u, H, D));
    require_ok(o2 && part2 && ds4_gpu_tensor_fill_f32(o2, -1234.5f, 2ull * H * D), "decode3 sentinel rows=2");
    require_ok(ds4_gpu_qwen35_attn_decode3_tensor(o2, gq2, gg2, kc, vc, part2, H, Hkv, D, pos0, 2u, scale),
               "decode3 rows=2");
    float *shared = download(o2, 2ull * H * D);

    float *solo = malloc(2ull * H * D * sizeof(float));
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(gq2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *gr = ds4_gpu_tensor_view(gg2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *orow = upload(NULL, (uint64_t)H * D);
        ds4_gpu_tensor *part1 = upload(NULL, ds4_gpu_qwen35_attn_part3_floats(1u, H, D));
        require_ok(qr && gr && orow && part1 && ds4_gpu_tensor_fill_f32(orow, -1234.5f, (uint64_t)H * D),
                   "decode3 sentinel rows=1");
        require_ok(ds4_gpu_qwen35_attn_decode3_tensor(orow, qr, gr, kc, vc, part1, H, Hkv, D, pos0 + r, 1u, scale),
                   "decode3 rows=1");
        float *rr = download(orow, (uint64_t)H * D);
        memcpy(solo + (uint64_t)r * H * D, rr, (uint64_t)H * D * sizeof(float));
        free(rr);
        ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(gr); ds4_gpu_tensor_free(orow); ds4_gpu_tensor_free(part1);
    }
    const int same = memcmp(shared, solo, 2ull * H * D * sizeof(float)) == 0;
    printf("  attn decode3 pos0=%u split_keys=%u: rows=2 vs two rows=1 memcmp %s\n",
           pos0, split_keys, same ? "== 0" : "DIFFERS");
    require_ok(same, "decode3 rows=2 matches two rows=1 calls of the same kernel");

    /* host double reference over the packed F16 cache */
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u) &&
               ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "decode3 cache read");
    float *qraw = download(gq2, 2ull * H * D), *graw = download(gg2, 2ull * H * D);
    double worst = 0.0, sc = 1e-6;
    for (uint32_t r = 0; r < 2u; r++) {
        const uint32_t n_keys = pos0 + r + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double acc[256], m = -1e300, l = 0.0;
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            for (uint32_t idx = 0; idx < n_keys; idx++) {
                double s = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    s += (double)qraw[((uint64_t)r * H + h) * D + d] * scale *
                         (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                l = l * corr + w;
                for (uint32_t d = 0; d < D; d++)
                    acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                m = mn;
            }
            for (uint32_t d = 0; d < D; d++) {
                const double sig = 1.0 / (1.0 + exp(-(double)graw[((uint64_t)r * H + h) * D + d]));
                const double refv = acc[d] / l * sig;
                const double got = (double)shared[((uint64_t)r * H + h) * D + d];
                if (fabs(refv) > sc) sc = fabs(refv);
                if (fabs(got - refv) > worst) worst = fabs(got - refv);
            }
        }
    }
    printf("  attn decode3 pos0=%u: vs host double ref max|d| %.3e (rel %.3e)\n", pos0, worst, worst / sc);
    require_ok(worst <= 1e-5 * sc, "decode3 within FP noise of the host double reference");
    unsetenv("DS4_QWEN35_ATTN_SPLIT_KEYS");
    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(q2); free(g2);
    free(shared); free(solo); free(kh); free(vh); free(qraw); free(graw);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq2); ds4_gpu_tensor_free(gg2); ds4_gpu_tensor_free(o2); ds4_gpu_tensor_free(part2);
}
```

In `main`, after the decode2 calls, add:

```c
    printf("qwen35 attention decode3 (M5)\n");
    test_attn_decode3_rows(&arena, 6u, 64u);      /* one split per row */
    test_attn_decode3_rows(&arena, 62u, 64u);     /* n0 = 63 (1 split), n1 = 64 (1 split, full) */
    test_attn_decode3_rows(&arena, 63u, 64u);     /* n0 = 64 (1 split), n1 = 65 (2 splits): ns0 != ns1 */
    test_attn_decode3_rows(&arena, 200u, 16u);    /* many splits, kps0 != kps1 around the boundary */
    test_attn_decode3_rows(&arena, 4200u, 16u);   /* at the 256-split cap: n / 16 > 256 */
```

- [ ] **Step 2: Build and confirm it fails**

```bash
make tests/test_qwen35_kernels 2>&1 | tail -3
```
Expected: a link or compile error naming `ds4_gpu_qwen35_attn_decode3_tensor` / `ds4_gpu_qwen35_attn_part3_floats` (not defined yet).

- [ ] **Step 3: Implement the kernels, the wrapper and the scratch helper**

Following the Design above:
- `metal/qwen35.metal`: `struct ds4_metal_args_qwen35_attn_decode3 { uint32_t n_head, n_head_kv, head_dim, pos0, rows, ns0, ns1; float scale; };`, `kernel_qwen35_attn_decode3` (template on NPT, instantiate `_npt8` only) and `kernel_qwen35_attn_merge3`.
- `ds4_metal.m`: enum entries `QWEN4_K_QWEN35_ATTN_DECODE3`, `QWEN4_K_QWEN35_ATTN_MERGE3` appended before `QWEN4_K_COUNT` with matching names in the name table (same relative order); `#define QWEN35_ATTN_MAX_SPLITS 256u`; a host `qwen35_attn_row_splits3(n_keys, split_keys, &ns, &kps)` mirroring the kernel's rule; `qwen35_attn_split_keys3()` reading `DS4_QWEN35_ATTN_SPLIT_KEYS` (default 64); `ds4_gpu_qwen35_attn_part3_floats(rows, H, D) = rows * H * QWEN35_ATTN_MAX_SPLITS * (2 + D)`; `ds4_gpu_qwen35_attn_decode3_tensor` dispatching decode3 then (for any row with ns > 1) merge3. Refuse (return 0 with a stderr message) head_dim != 256 or rows not in {1, 2}.
- `ds4_gpu.h`: the two declarations with a comment on the rows contract.

- [ ] **Step 4: Run the kernel tests (both Metal settings)**

```bash
make test-qwen35-kernels 2>&1 | grep -E 'decode3|qwen35 kernels' ; DS4_METAL_DISABLE_METAL4=1 ./tests/test_qwen35_kernels | tail -1
make test-qwen4-kernels test-qwen4-q2 2>&1 | tail -2
```
Expected: five `memcmp == 0` lines and five host-reference lines within tolerance; `qwen35 kernels: ok` both ways; qwen4 tests pass.

- [ ] **Step 5: Add decode3 to the benchmark**

In `tests/bench_qwen35_attn.c` add `run_decode3` (same shape as `run_decode2`, calling `ds4_gpu_qwen35_attn_decode3_tensor` with `c->part` sized by `ds4_gpu_qwen35_attn_part3_floats(2u, H, D)` — allocate a separate `part3` buffer in the context) and report it after each decode2 line with the kernel name `decode3`. Build warning-free.

- [ ] **Step 6: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
git commit -m "qwen35: decode3 attention kernel (more splits, keys across simdgroups, parallel merge)

Plain decode is a rows==1 call and the MTP verify one rows==2 call; each
row's split geometry, simdgroup key assignment and merge tree depend only on
its own key count, so the verify rows equal plain decode bit for bit
(memcmp-tested at split boundaries and the 256-split cap).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
```

- [ ] **Step 7 (controller, GPU window): benchmark**

Run `./tests/bench_qwen35_attn decode` (stack paused); add the decode3 column to `BENCH.md`. Target: decode3 at 131,072 keys within 1.5x of the floor (~11 ms floor for 10 layers -> the benchmark reports one layer: ~1.1 ms floor per layer; target <= 1.7 ms). If the target is missed, tune `split_keys` / NSG / cap with the benchmark before Task 5 and record the chosen values.

---

### Task 3: `kernel_qwen35_attn_flash` (prefill, no key split)

**Files:**
- Modify: `metal/qwen35.metal` (append), `ds4_metal.m`, `ds4_gpu.h`, `tests/test_qwen35_kernels.c` (additions), `tests/bench_qwen35_attn.c`

**Interfaces:**
- Consumes: `kernel_qwen4_attn_mm` (metal/qwen4.metal ~2603) as the reference design — same simdgroup-matrix QKᵀ/PV structure and online softmax, same epilogue (normalize, sigmoid gate).
- Produces: `int ds4_gpu_qwen35_attn_flash_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part, uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, float scale);` — T query tokens at positions pos0..pos0+T-1 (their K/V already written), causal, F16 K/V, head_dim 256, group 8; `part == NULL` means no key split (Task 4 adds the split and `ds4_gpu_qwen35_attn_flash_part_floats`).

**Design (binding):**
- Template parameters `TOK` (query tokens per threadgroup) and `KT` (keys per tile). Rows per threadgroup = `TOK * 8` (the 8 query heads of one KV head for each token), laid out token-major; no padding rows (today's kernel pads 8 heads to 16 rows).
- Grid `(n_head_kv, ceil(T / TOK), 1)`. Row tile r (8 rows) = the 8 query heads of token `blk*TOK + r`. Threadgroup = `2*TOK` simdgroups: simdgroup `sg` owns row tile `sg % TOK` and dim half `sg / TOK` (128 dims = 16 `simdgroup_float8x8` accumulators, as today).
  - TOK=2, KT=16 (4 simdgroups, 128 threads) is today's `kernel_qwen4_attn_mm` with its padding row tile replaced by the second token: the two simdgroups of a row tile each score one 8-key half of the tile, exchange scores through threadgroup memory and run the same online softmax.
  - TOK=4, KT=8 (8 simdgroups, 256 threads): the two simdgroups of a row tile each compute the 8x8 score tile over their half of the 256 dims and sum the two halves through threadgroup memory before the softmax.
- Per row-tile causal mask: row tile r (token t = blk*TOK + r, position pos0 + t) sees keys `idx <= pos0 + t`; keys beyond it get score −inf (probability 0). The key loop runs to `pos0 + t_last + 1` (t_last = the block's last valid token, clamped at T − 1); a block's rows past T are not written.
- Threadgroup memory: `Qs[TOK*8*256]` half + `K/V tile 2*KT*256` half + scores/probabilities scratch must stay <= 32 KB: TOK=2, KT=16 (8 KB + 16 KB) and TOK=4, KT=8 (16 KB + 8 KB) are the two instances to build; the benchmark chooses.
- Output written directly (no split) with the sigmoid gate, as today.

- [ ] **Step 1: Write the failing kernel test**

Append before `int main`:

```c
/* attn_flash (M5): causal prefill attention for T tokens at pos0 against a
 * host double reference and against today's kernel (qwen4 attn_mm via the
 * qwen4 wrapper). */
static void test_attn_flash(arena_t *a, uint32_t pos0, uint32_t T, int use_part) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, fill = pos0 + T, cap = fill + 8u;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "flash fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, 1.0e7f, 1e-6f, kc, vc, kc, vc, 0u),
               "flash fill");
    /* the last T rows of the prepped q/gate are the queries */
    ds4_gpu_tensor *qT = ds4_gpu_tensor_view(pq, (uint64_t)pos0 * H * D * sizeof(float), (uint64_t)T * H * D * sizeof(float));
    ds4_gpu_tensor *gT = ds4_gpu_tensor_view(pg, (uint64_t)pos0 * H * D * sizeof(float), (uint64_t)T * H * D * sizeof(float));
    ds4_gpu_tensor *of = upload(NULL, (uint64_t)T * H * D), *om = upload(NULL, (uint64_t)T * H * D);
    /* key-split scratch only for the Task 4 cases: its size follows the split rule */
    ds4_gpu_tensor *partf = use_part ? upload(NULL, ds4_gpu_qwen35_attn_flash_part_floats(T, H, D)) : NULL;
    require_ok(qT && gT && of && om && (!use_part || partf) &&
               ds4_gpu_tensor_fill_f32(of, -1234.5f, (uint64_t)T * H * D), "flash buffers");
    require_ok(ds4_gpu_qwen35_attn_flash_tensor(of, qT, gT, kc, vc, partf, T, H, Hkv, D, pos0, scale), "flash call");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(om, qT, gT, kc, vc, NULL, NULL, NULL, T, H, Hkv, D, pos0,
                   0u, 0u, scale, NULL, NULL, NULL, NULL, 0u), "qwen4 attn reference");
    float *gotf = download(of, (uint64_t)T * H * D), *gotm = download(om, (uint64_t)T * H * D);

    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u) &&
               ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "flash cache read");
    float *qraw = download(qT, (uint64_t)T * H * D), *graw = download(gT, (uint64_t)T * H * D);
    double worst_ref = 0.0, worst_mm = 0.0, sc = 1e-6;
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t n_keys = pos0 + t + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double acc[256], m = -1e300, l = 0.0;
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            for (uint32_t idx = 0; idx < n_keys; idx++) {
                double s = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    s += (double)qraw[((uint64_t)t * H + h) * D + d] * scale *
                         (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                l = l * corr + w;
                for (uint32_t d = 0; d < D; d++)
                    acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                m = mn;
            }
            for (uint32_t d = 0; d < D; d++) {
                const uint64_t o = ((uint64_t)t * H + h) * D + d;
                const double refv = acc[d] / l / (1.0 + exp(-(double)graw[o]));
                if (fabs(refv) > sc) sc = fabs(refv);
                if (fabs((double)gotf[o] - refv) > worst_ref) worst_ref = fabs((double)gotf[o] - refv);
                if (fabs((double)gotf[o] - (double)gotm[o]) > worst_mm) worst_mm = fabs((double)gotf[o] - (double)gotm[o]);
            }
        }
    }
    printf("  attn flash pos0=%u T=%u: vs host double ref max|d| %.3e (rel %.3e), vs attn_mm %.3e\n",
           pos0, T, worst_ref, worst_ref / sc, worst_mm);
    /* queries and K/V are staged as half on the matrix path, like attn_mm */
    require_ok(worst_ref <= 3e-3 * sc, "flash within half-precision staging of the host double reference");
    require_ok(worst_mm <= 3e-3 * sc, "flash within half-precision staging of qwen4 attn_mm");
    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(gotf); free(gotm);
    free(kh); free(vh); free(qraw); free(graw);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(qT); ds4_gpu_tensor_free(gT); ds4_gpu_tensor_free(of); ds4_gpu_tensor_free(om);
    if (partf) ds4_gpu_tensor_free(partf);
}
```

In `main`:

```c
    printf("qwen35 attention flash prefill (M5)\n");
    test_attn_flash(&arena, 0u, 9u, 0);
    test_attn_flash(&arena, 0u, 65u, 0);
    test_attn_flash(&arena, 37u, 65u, 0);
    test_attn_flash(&arena, 37u, 200u, 0);
    test_attn_flash(&arena, 4096u, 64u, 0);
```

Run the test once per instance: the wrapper reads `DS4_QWEN35_ATTN_FLASH_TOK` (2 or 4, default 2) so both instances are exercised: `DS4_QWEN35_ATTN_FLASH_TOK=4 ./tests/test_qwen35_kernels`.

- [ ] **Step 2: Build and confirm it fails** (link error naming `ds4_gpu_qwen35_attn_flash_tensor`).

- [ ] **Step 3: Implement** the kernel instances (`kernel_qwen35_attn_flash_tok2_kt16`, `kernel_qwen35_attn_flash_tok4_kt8`), enum/name entries, `ds4_gpu_qwen35_attn_flash_tensor` (chooses the instance from `DS4_QWEN35_ATTN_FLASH_TOK`, refuses head_dim != 256 or group != 8 or T <= 8), and the header declaration, following the Design.

- [ ] **Step 4: Run the tests** — `make test-qwen35-kernels`, then `DS4_QWEN35_ATTN_FLASH_TOK=4 ./tests/test_qwen35_kernels`, then both again with `DS4_METAL_DISABLE_METAL4=1`; `make test-qwen4-kernels test-qwen4-q2`. Expected: all five flash lines within tolerance for both TOK values; everything else unchanged.

- [ ] **Step 5: Benchmark cases** — add `run_flash` to `tests/bench_qwen35_attn.c` (prefill rows `attn_flash` for TOK 2 and 4 at the same positions as `qwen4_attn_mm`).

- [ ] **Step 6: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
git commit -m "qwen35: flash prefill attention (query-token tiles, causal, no key split)

TOK query tokens x 8 query heads share every K/V tile (instances TOK=2/KT=16
and TOK=4/KT=8), removing today's per-token K/V re-read and the 8-to-16 row
padding; tested against a host double reference and qwen4 attn_mm.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
```

---

### Task 4: Key split for flash prefill

**Files:** `metal/qwen35.metal`, `ds4_metal.m`, `tests/test_qwen35_kernels.c` (additions), `tests/bench_qwen35_attn.c`

**Interfaces:**
- Consumes: Task 3's kernel and wrapper, Task 2's `kernel_qwen35_attn_merge3` and `ds4_gpu_qwen35_attn_part3_floats`.
- Produces:
  - `ds4_gpu_qwen35_attn_flash_tensor` splits the key range into `Ks` parts when `part != NULL` and `n_head_kv * ceil(T/TOK) < DS4_QWEN35_ATTN_FLASH_MIN_TG` (default 256 threadgroups): `Ks = min(QWEN35_ATTN_MAX_SPLITS, ceil(min_tg / (Hkv * ceil(T/TOK))))`, reduced so each part keeps >= 1024 keys. A 2048-token chunk already gives 2048 threadgroups (TOK=2), so the split only engages for short chunks (T < 256) at long context. Partials use merge3's per-row layout; merge3 combines them.
  - `uint64_t ds4_gpu_qwen35_attn_flash_part_floats(uint32_t n_tokens, uint32_t n_head, uint32_t head_dim);` — scratch floats for the worst-case Ks of that T under the split rule: `n_tokens * n_head * Ks_max(T) * (2 + head_dim)` (a few MB: T=64 -> Ks <= 8 -> ~8.5 MB).
  - `uint32_t ds4_gpu_qwen35_attn_flash_last_splits(void);` — Ks used by the last flash call (test hook).

**Design (binding):** split s of the key range `[0, n_last)` (n_last = pos0 + T) covers `[s*kps, (s+1)*kps)`; each row still masks keys above its own diagonal, so a row whose whole part is masked writes a neutral partial (m = −inf, l = 0) that the merge ignores. The split rule depends on T and pos0 only (prefill needs no rows contract).

- [ ] **Step 1: Failing test** — add to `main`:

```c
    setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", "4096", 1);   /* force key splits on short caches */
    test_attn_flash(&arena, 4096u, 64u, 1);
    require_ok(ds4_gpu_qwen35_attn_flash_last_splits() > 1, "flash key split taken (T=64)");
    test_attn_flash(&arena, 8000u, 200u, 1);
    require_ok(ds4_gpu_qwen35_attn_flash_last_splits() > 1, "flash key split taken (T=200)");
    unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
```
- [ ] **Step 2: Build** — fails to link: `ds4_gpu_qwen35_attn_flash_part_floats` / `ds4_gpu_qwen35_attn_flash_last_splits` undefined.
- [ ] **Step 3: Implement** the split kernel variant (partial output instead of the gated write), the host split rule, `ds4_gpu_qwen35_attn_flash_last_splits`, and the merge3 dispatch.
- [ ] **Step 4: Tests** — all flash lines within tolerance with and without the forced split, both TOK values, both Metal settings; qwen4 tests pass.
- [ ] **Step 5: Benchmark** — flash with key split at pos 30,720 and 122,880 (T = 2048).
- [ ] **Step 6: Commit** — `qwen35: key split for flash prefill attention` with the trailer, files by path.
- [ ] **Step 7 (controller, GPU window): benchmark receipt** — prefill columns for TOK 2/4 with and without split in `BENCH.md`; choose the default TOK and split threshold. Target: per-chunk attention at pos 30,720 <= 0.6 s (today 4.3 s) and at 122,880 <= 2.5 s (today 19.7 s), measured through the profile in Task 6.

---

### Task 5: decode3 dispatch

**Files:** `ds4_qwen35moe.inc` (`qwen35_graph_attend`, the scratch allocation in `qwen35_graph_alloc`)

**Interfaces:**
- Consumes: `ds4_gpu_qwen35_attn_decode3_tensor`, `ds4_gpu_qwen35_attn_part3_floats`.
- Produces: knob `DS4_QWEN35_ATTN_DECODE` = `2` (default; L12 decode2) or `3` (decode3), read once; with `3`, plain T=1 decode calls decode3 rows=1 and the verify (`verify_rows_exact && T == 2`) calls decode3 rows=2; fp8/q4 K/V modes and `DS4_QWEN35_ATTN_DECODE2=0` keep today's paths. The graph's attention scratch (`part`) grows to `max(existing, ds4_gpu_qwen35_attn_part3_floats(2, H, D))` floats when the graph is allocated (cost: 2 x 16 x 256 x 258 x 4 B ≈ 8.5 MB).

- [ ] **Step 1:** Read `qwen35_graph_attend` and the decode2 branch M4 added (search `qwen35_attn_decode2_env`); add the knob parse (`static int qwen35_attn_decode_env(void)` returning 2 or 3) and the decode3 branch in both places decode2 is called (plain T=1 and the verify rows=2), guarded by `!qwen4_kv_mode(g)` (F16 only).
- [ ] **Step 2:** Grow the scratch allocation as stated.
- [ ] **Step 3:** Build warning-free (`make ds4 ds4-server ds4_test`), `make test-qwen35-kernels`, `./ds4_test --server`.
- [ ] **Step 4:** Commit `qwen35: DS4_QWEN35_ATTN_DECODE selects the decode3 kernel (default 2)`.
- [ ] **Step 5 (controller, GPU window):** with `DS4_QWEN35_ATTN_DECODE=3`: gate 1 at chunks 2048/64/65 (`tests/ornith/gate1.py`), `make test-qwen35-graph test-qwen35-mtp`, `tests/ornith/test_mtp_cli.py`, `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` with and without `DS4_TEST_GLM_MTP=1`, and `DS4_QWEN35_KV=q4` payload loop; then `m4_ab.py --mode lever --base-env DS4_QWEN35_ATTN_DECODE=2 --lever-env DS4_QWEN35_ATTN_DECODE=3 --contexts 32768,131072 --cold-tokens 0`; Qwen fast gate. If all pass and decode improves, flip the default to 3 in a one-line commit (`qwen35: decode3 is the Ornith default`), otherwise record why in `speed-bench/ornith/m5/LEVERS.md`.

---

### Task 6: flash dispatch

**Files:** `ds4_qwen35moe.inc` (`qwen35_graph_attend` prefill branch, scratch)

**Interfaces:**
- Consumes: `ds4_gpu_qwen35_attn_flash_tensor`, the defaults chosen in Task 4 Step 7.
- Produces: knob `DS4_QWEN35_ATTN_FLASH` (0 default, 1 on), read once; with 1, prefill chunks with T > 8 (not the verify, not T <= 8 tails) and F16 K/V call flash. The graph allocates a flash scratch of `max over T in (8, prefill_chunk] of ds4_gpu_qwen35_attn_flash_part_floats(T, H, D)` floats once (a few MB: the split only engages for T < 256), and the wrapper refuses a split that would exceed the scratch it was given.

- [ ] **Step 1:** Add the knob and the prefill branch; allocate the flash scratch as stated (document its bytes in the comment).
- [ ] **Step 2:** Build warning-free; `make test-qwen35-kernels`; `./ds4_test --server`.
- [ ] **Step 3:** Commit `qwen35: DS4_QWEN35_ATTN_FLASH selects the flash prefill kernel (default off)`.
- [ ] **Step 4 (controller, GPU window):** with `DS4_QWEN35_ATTN_FLASH=1` (and the decode default from Task 5): gate 1 at 2048/64/65, the MTP identity tests, `test_mtp_cli.py`, payload loops, `DS4_QWEN35_PROFILE=1` at 32K and 128K prefill (attention ms per chunk vs the Task 4 target), `m4_ab.py --mode lever --base-env DS4_QWEN35_ATTN_FLASH=0 --lever-env DS4_QWEN35_ATTN_FLASH=1 --contexts 32768,131072 --cold-tokens 31000`; Qwen fast gate. Flip the default to 1 on success (one-line commit), else record why.

---

### Task 7: Final gates, receipts and report

**Files:** `speed-bench/ornith/m5/{LEVERS.md,speed/GATE3.md,quality/GATE2.md,QWEN_GATE.md,REPORT.md}`, `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (§4 attention)

- [ ] **Step 1 (controller, GPU window): gate 3** — `m4_ab.py --mode baseline --ds4-env "DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --contexts 2048,32768,131072 --cold-tokens 31000` on the final defaults; write `speed/GATE3.md` (pass rule: spec section 1 items 1-2; the 2K decode gap is out of scope and reported, not judged).
- [ ] **Step 2 (controller, GPU window): gate 2** — the M4 gate-2 scripts (scratch copy of the eval, matrix, probes) on 23G and 25G plus the same eval against the live oMLX the same day; pass rule: each tier's index >= the oMLX index − 3.0, truncated = errored = 0, no matrix quality failure, probes match. Write `quality/GATE2.md`.
- [ ] **Step 3 (controller, GPU window): Qwen full gate** — `speed-bench/qwen-regression/run.sh full`; write `QWEN_GATE.md`.
- [ ] **Step 4: spec** — rewrite the parent spec's §4 attention paragraph for `attn_flash` (prefill T > 8) and `decode3` (decode and verify), their knobs and defaults; list the new knobs in §7.3.
- [ ] **Step 5: report** — `REPORT.md`: baseline (M4 final) vs M5 per context against the live oMLX, the kernel benchmark table, each kernel's adoption verdict, gate 1/2/3 and Qwen results, open items.
- [ ] **Step 6: commit** the receipts, spec and report by path with the trailer.

---

## Self-Review

1. **Spec coverage.** §3.1 flash prefill: Tasks 3-4 (kernel, key split), 6 (dispatch). §3.2 decode3: Tasks 2 (kernel, merge, rows contract), 5 (dispatch). §3.3 knobs and fallbacks: Tasks 5-6. §4 exactness: Task 2's memcmp test and Task 5's MTP identity runs. §5 testing and benchmark: Tasks 1-4 (kernel tests, benchmark), 5-7 (model checks, gates). §1 success criteria: Task 7. §8 gate-2 rule: Task 7 Step 2.
2. **Placeholders.** Kernel bodies are specified by binding design + complete tests rather than source (plan-specific constraint: tile parameters are benchmark-chosen); every test, the benchmark and the commands are complete.
3. **Type consistency.** `ds4_gpu_qwen35_attn_decode3_tensor` / `_part3_floats` / `_flash_tensor` / `_flash_last_splits` are declared in Tasks 2-4 and used with the same parameter lists in the tests, the benchmark and the dispatch tasks.
4. **Review Focus.** Each of the five lines has its test in the owning task (Tasks 2, 3, 4, 5, 6).
