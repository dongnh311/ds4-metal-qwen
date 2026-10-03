# Ornith decode round 2, Plan B: the attention merge folded in registers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kernel_qwen35_attn_merge3` cheap without changing one output bit. Its per-thread arrays of
3 × 256 floats live in device memory, so at 2K a merge costs as much as the attention it merges. From 32K up
every attention layer merges 256 splits.

**Architecture:**
- **A new kernel, `kernel_qwen35_attn_merge3_fold`.** It walks the same leaves in the same order, padded with
  the same neutral leaves to the next power of two. It folds them with a binary carry: slot `lv` holds the
  pending left node of level `lv`.
  - This makes exactly the pairs of merge3's level-by-level loop: (0,1), (2,3), then pairs of pairs. The
    lower-index node is always operand 0, and the per-node formula is the same.
  - At most 9 nodes are live, with constant slot indices after unrolling, so they stay in registers.
- **Host selection.** In the three merge3 dispatch sites of `ds4_metal.m` (decode3 solo, decode3 shared, flash
  merge), behind `DS4_QWEN35_ATTN_MERGE_FOLD=1` (default off).
- **A self-check** at Ornith session creation compares the fold with merge3 on fixed partials. A mismatch turns
  the fold off for the process.

**Tech Stack:** Metal (`metal/qwen35.metal`), Objective-C (`ds4_metal.m`), C (`ds4_gpu.h`, `ds4.c`), C tests
(`tests/test_qwen35_kernels.c`) and the bench (`tests/bench_qwen35_attn.c`).

**Spec:** `docs/superpowers/specs/2026-10-03-ornith-decode-round2-design.md` (§2 lossless, §5 Stage R, §1
gate). Evidence: `speed-bench/ornith/decode/round2/ANATOMY.md`, ruling 2.

## Global Constraints

- **Bit-identical.**
  - The fold's `out` `memcmp`-equals merge3's for every split count 2-256, for rows 1 and 2, and at unequal
    row split counts.
  - The gate-1 dumps (chunks 64/512/2048), `test_mtp_cli` and `test-qwen35-verify-batch` stay identical with
    the knob on.
- **Ornith-only kernel.** The host change touches only the Ornith wrappers' merge dispatches in `ds4_metal.m`,
  a shared file, so the Qwen session's gate runs before merge.
- **Knob.** `DS4_QWEN35_ATTN_MERGE_FOLD`, read fresh on every dispatch, as `DS4_QWEN35_ATTN_SPLIT_KEYS` is.
  Only `1` turns it on. It flips to default-on only after spec §1's gate passes.
- **GPU use only inside an announced window,** including the kernel tests, because peers are measuring today.

## Review Focus

1. **Split count exactly a power of two (2, 32, 256) against just above one (3, 33, 129).** The pad leaves must
   be folded exactly as merge3 folds them. Test: the exactness cases at ns 3, 32 and 256.
2. **A row with one split next to a row with several.** decode3 writes the one-split row directly, and the
   merge skips it. Test: pos0 = 63 at split_keys 64 (n0 = 64 → 1 split, n1 = 65 → 2 splits; the solo path).
3. **Neutral partials (l = 0) mixed with live ones.** Real decode3 partials have l > 0, so the self-check feeds
   l = 0 entries to exercise the `c = 0` branch. Test: the self-check data.
4. **The knob is toggled between calls in one process.** The test does this, so the selection must not be
   cached. Test: the exactness test runs off, then on, then reads the dispatch counter.
5. **A self-check failure.** The fold turns off with a warning, and the output stays merge3's. Test:
   `ds4_gpu_qwen35_attn_merge_fold_selfcheck` returns 1 on this toolchain. The failure branch is a single
   flag set, read in the selection helper.

---

### Task 1: The fold kernel, selection, self-check and exactness test

**Files:**
- Modify: `metal/qwen35.metal`. Add `qwen35_merge_node` and `kernel_qwen35_attn_merge3_fold` after
  `kernel_qwen35_attn_merge3` (which ends at line 1320).
- Modify: `ds4_metal.m`:
  - the enum entry after `QWEN4_K_QWEN35_ATTN_MERGE3` (line 49925) and the name after
    `"kernel_qwen35_attn_merge3"` (line 50094);
  - the selection helper, the counter and the self-check after `qwen35_attn_row_splits3` (line 51484);
  - the three dispatch sites (lines 51534, 51606 and 51750) use `qwen35_merge3_kernel()`.
- Modify: `ds4_gpu.h`. Declare `ds4_gpu_qwen35_attn_merge_fold_dispatches` and
  `ds4_gpu_qwen35_attn_merge_fold_selfcheck`.
- Modify: `ds4.c`. Call the self-check once next to `qwen35_verify_batch_selfcheck` (line 76500), for every
  Ornith session, not only MTP ones.
- Test: `tests/test_qwen35_kernels.c`. Add `test_attn_merge_fold(a, pos0, split_keys)` and its calls after
  line 1185.

**Interfaces:**
- Produces:
  - `uint64_t ds4_gpu_qwen35_attn_merge_fold_dispatches(void)`, which counts the fold dispatches;
  - `int ds4_gpu_qwen35_attn_merge_fold_selfcheck(void)`. It returns 1 if the knob is off or the fold matches,
    and 0 on a mismatch or error, after turning the fold off. It runs once per process.

- [ ] **Step 1: Write the failing test** in `tests/test_qwen35_kernels.c`, before `test_attn_decode3_rows`'s
  callers in `main`, as a function placed after `test_attn_decode3_rows`:

```c
/* Plan B (round 2): kernel_qwen35_attn_merge3_fold folds merge3's fixed
 * binary tree with one register slot per level; the decode3 output with
 * DS4_QWEN35_ATTN_MERGE_FOLD=1 must memcmp-equal the merge3 output, rows
 * 1 and 2, and the fold must actually have run (dispatch counter). */
static void test_attn_merge_fold(arena_t *a, uint32_t pos0, uint32_t split_keys) {
    (void)a;
    const uint32_t H = 16u, Hkv = 2u, D = 256u, cap = pos0 + 2u;
    const float scale = 0.0625f;
    char sk[16];
    snprintf(sk, sizeof(sk), "%u", split_keys);
    setenv("DS4_QWEN35_ATTN_SPLIT_KEYS", sk, 1);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(2ull * H * D * sizeof(float));
    ds4_gpu_tensor *g = ds4_gpu_tensor_alloc(2ull * H * D * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_qwen35_attn_part3_floats(2u, H, D) * sizeof(float));
    ds4_gpu_tensor *o_ref = ds4_gpu_tensor_alloc(2ull * H * D * sizeof(float));
    ds4_gpu_tensor *o_new = ds4_gpu_tensor_alloc(2ull * H * D * sizeof(float));
    require_ok(kc && vc && q && g && part && o_ref && o_new, "merge fold buffers");
    uint16_t *kv = malloc((uint64_t)cap * Hkv * D * 2u);
    float *f = malloc(2ull * H * D * sizeof(float));
    uint32_t r = 0x51ed270bu ^ pos0;
    for (uint64_t i = 0; i < (uint64_t)cap * Hkv * D; i++) {
        r = r * 1664525u + 1013904223u;
        kv[i] = (uint16_t)(0x3800u + ((r >> 9) & 0x3ffu) - 0x200u);   /* halves around +/-0.5 */
    }
    require_ok(ds4_gpu_tensor_write(kc, 0, kv, (uint64_t)cap * Hkv * D * 2u), "merge fold k");
    for (uint64_t i = 0; i < (uint64_t)cap * Hkv * D; i++) kv[i] ^= 0x8000u;
    require_ok(ds4_gpu_tensor_write(vc, 0, kv, (uint64_t)cap * Hkv * D * 2u), "merge fold v");
    for (uint64_t i = 0; i < 2ull * H * D; i++) {
        r = r * 1664525u + 1013904223u;
        f[i] = (float)((int32_t)(r >> 8) - (1 << 23)) / (float)(1 << 22);
    }
    require_ok(ds4_gpu_tensor_write(q, 0, f, 2ull * H * D * sizeof(float)) &&
               ds4_gpu_tensor_write(g, 0, f, 2ull * H * D * sizeof(float)), "merge fold q/gate");
    for (uint32_t rows = 1u; rows <= 2u; rows++) {
        unsetenv("DS4_QWEN35_ATTN_MERGE_FOLD");
        require_ok(ds4_gpu_tensor_fill_f32(o_ref, -1234.5f, 2ull * H * D) &&
                   ds4_gpu_qwen35_attn_decode3_tensor(o_ref, q, g, kc, vc, part, H, Hkv, D, pos0, rows, scale),
                   "decode3 merge3");
        const uint64_t before = ds4_gpu_qwen35_attn_merge_fold_dispatches();
        setenv("DS4_QWEN35_ATTN_MERGE_FOLD", "1", 1);
        require_ok(ds4_gpu_tensor_fill_f32(o_new, -1234.5f, 2ull * H * D) &&
                   ds4_gpu_qwen35_attn_decode3_tensor(o_new, q, g, kc, vc, part, H, Hkv, D, pos0, rows, scale),
                   "decode3 merge3_fold");
        unsetenv("DS4_QWEN35_ATTN_MERGE_FOLD");
        const bool ran = ds4_gpu_qwen35_attn_merge_fold_dispatches() > before;
        float *x = malloc(2ull * H * D * sizeof(float)), *y = malloc(2ull * H * D * sizeof(float));
        require_ok(ds4_gpu_tensor_read(o_ref, 0, x, (uint64_t)rows * H * D * sizeof(float)) &&
                   ds4_gpu_tensor_read(o_new, 0, y, (uint64_t)rows * H * D * sizeof(float)), "merge fold read");
        const bool same = memcmp(x, y, (uint64_t)rows * H * D * sizeof(float)) == 0;
        printf("  attn merge fold pos0=%u split_keys=%u rows=%u: %s%s\n", pos0, split_keys, rows,
               same ? "bit-identical" : "DIFFERS", ran ? "" : " (fold did not run)");
        require_ok(same, "merge3_fold output equals merge3");
        require_ok(ran, "merge3_fold dispatched under DS4_QWEN35_ATTN_MERGE_FOLD=1");
        free(x); free(y);
    }
    unsetenv("DS4_QWEN35_ATTN_SPLIT_KEYS");
    free(kv); free(f);
    ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(g);
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(o_ref); ds4_gpu_tensor_free(o_new);
}
```

  And in `main`, after line 1185 (`test_attn_decode3_rows(&arena, 200u, 16u);`):

```c
    printf("qwen35 attention merge fold (round 2 Plan B)\n");
    require_ok(ds4_gpu_qwen35_attn_merge_fold_selfcheck(), "merge fold self-check (knob off: trivially 1)");
    test_attn_merge_fold(&arena, 40u, 16u);       /* n = 41/42: ns 3 (pad to 4) */
    test_attn_merge_fold(&arena, 2046u, 64u);     /* n = 2047/2048: ns 32 (a power of two) */
    test_attn_merge_fold(&arena, 2100u, 64u);     /* ns 33 (pad to 64) */
    test_attn_merge_fold(&arena, 8250u, 64u);     /* ns 129 (pad to 256) */
    test_attn_merge_fold(&arena, 32766u, 64u);    /* ns 256 (the cap) */
    test_attn_merge_fold(&arena, 63u, 64u);       /* n0 = 64 (1 split), n1 = 65 (2): the solo path */
    setenv("DS4_QWEN35_ATTN_MERGE_FOLD", "1", 1);
    require_ok(ds4_gpu_qwen35_attn_merge_fold_selfcheck(), "merge fold self-check on this compiler");
    unsetenv("DS4_QWEN35_ATTN_MERGE_FOLD");
```

- [ ] **Step 2: RED (a compile failure, no GPU).**
  Run: `make tests/test_qwen35_kernels 2>&1 | grep -m2 error`
  Expected: `error: call to undeclared function 'ds4_gpu_qwen35_attn_merge_fold_dispatches'` (or the
  implicit-declaration error).

- [ ] **Step 3: Implement.**
  - **`metal/qwen35.metal`**, after `kernel_qwen35_attn_merge3`:

```metal
/* One node of kernel_qwen35_attn_merge3's tree, operand 0 the lower index:
 * the same expressions as merge3's loop body. */
static inline void qwen35_merge_node(float m0, float l0, float o0, float m1, float l1, float o1,
                                     thread float &m, thread float &l, thread float &o) {
    const float nm = max(m0, m1);
    const float c0 = l0 > 0.0f ? exp(m0 - nm) : 0.0f;
    const float c1 = l1 > 0.0f ? exp(m1 - nm) : 0.0f;
    m = nm; l = l0 * c0 + l1 * c1; o = o0 * c0 + o1 * c1;
}

/* kernel_qwen35_attn_merge3 with its tree folded in leaf order (round 2
 * Plan B, DS4_QWEN35_ATTN_MERGE_FOLD): slot lv holds the pending left node
 * of level lv; an arriving node merges with it (the slot as operand 0, as
 * merge3 pairs (2i, 2i+1)) and carries up, or parks.  Every neutral pad up
 * to the next power of two is folded too, so the pairs, the operand order
 * and the per-node arithmetic are merge3's and the output is bit-identical,
 * with at most QWEN35_MERGE_LEVELS live nodes in registers instead of three
 * QWEN35_ATTN_MAX_SPLITS-float private arrays in device memory. */
#define QWEN35_MERGE_LEVELS 9u   /* 256 leaves -> levels 0..8 */
kernel void kernel_qwen35_attn_merge3_fold(
        constant ds4_metal_args_qwen35_attn_decode3 & args,
        device const float *part,
        device const float *gate,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x, r = tgpig.y;
    if (h >= args.n_head || r >= args.rows) return;
    const uint ns = r == 0u ? args.ns0 : args.ns1;
    if (ns <= 1u) return;
    constexpr uint D = 256u;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t row_base = (r == 0u ? 0u : (uint64_t)args.ns0 + (uint64_t)(r - 1u) * args.ns1) * H * (2u + D);
    const uint64_t stride = (uint64_t)group * (2u + D);
    device const float *base = part + row_base + ((uint64_t)kvh * ns * group + g) * (2u + D);

    uint p = 1u;
    while (p < ns) p <<= 1u;
    float sm[QWEN35_MERGE_LEVELS], sl[QWEN35_MERGE_LEVELS], so[QWEN35_MERGE_LEVELS];
    uint used = 0u;
    for (uint s = 0; s < p; s++) {
        float m, l, o;
        if (s < ns) {
            device const float *ps = base + s * stride;
            m = ps[0]; l = ps[1]; o = ps[2u + tid];
        } else {
            m = -3.0e38f; l = 0.0f; o = 0.0f;   /* neutral pad, as merge3 */
        }
        bool carry = true;
#pragma unroll
        for (uint lv = 0; lv < QWEN35_MERGE_LEVELS; lv++) {
            if (carry) {
                if (used & (1u << lv)) {
                    qwen35_merge_node(sm[lv], sl[lv], so[lv], m, l, o, m, l, o);
                    used &= ~(1u << lv);
                } else {
                    sm[lv] = m; sl[lv] = l; so[lv] = o;
                    used |= 1u << lv;
                    carry = false;
                }
            }
        }
    }
    float rl = 0.0f, ro = 0.0f;   /* p is a power of two: one slot holds the root */
#pragma unroll
    for (uint lv = 0; lv < QWEN35_MERGE_LEVELS; lv++)
        if (used == (1u << lv)) { rl = sl[lv]; ro = so[lv]; }
    const float inv = rl > 0.0f ? 1.0f / rl : 0.0f;
    const uint64_t at = ((uint64_t)r * H + h) * D + tid;
    out[at] = ro * inv * qwen4_sigmoid(gate[at]);
}
```

  - **`ds4_metal.m` enum and name table:**
    - add `QWEN4_K_QWEN35_ATTN_MERGE3_FOLD,` after `QWEN4_K_QWEN35_ATTN_MERGE3,`;
    - add `"kernel_qwen35_attn_merge3_fold",` after `"kernel_qwen35_attn_merge3",`.

    They are matched by position, so check that both lists stay aligned.
  - **`ds4_metal.m`**, after `qwen35_attn_row_splits3`:

```objc
/* Round 2 Plan B: kernel_qwen35_attn_merge3_fold (bit-identical to merge3)
 * when DS4_QWEN35_ATTN_MERGE_FOLD=1, read fresh on every dispatch like the
 * split knob above; a failed startup self-check pins merge3. */
static bool g_qwen35_merge_fold_off;
static uint64_t g_qwen35_merge_fold_dispatches;

static int qwen35_merge3_kernel(void) {
    const char *e = getenv("DS4_QWEN35_ATTN_MERGE_FOLD");
    if (!g_qwen35_merge_fold_off && e && strcmp(e, "1") == 0) {
        g_qwen35_merge_fold_dispatches++;
        return QWEN4_K_QWEN35_ATTN_MERGE3_FOLD;
    }
    return QWEN4_K_QWEN35_ATTN_MERGE3;
}

uint64_t ds4_gpu_qwen35_attn_merge_fold_dispatches(void) {
    return g_qwen35_merge_fold_dispatches;
}

/* Once per process: the shader library is compiled at run time with fast
 * math, so the fold's equality with merge3 -- pinned by the kernel tests on
 * the development toolchain -- is checked again here on fixed partials (two
 * rows, 256 and 37 splits, every eighth partial neutral with l == 0).  A
 * mismatch or an error turns the fold off for the process. */
int ds4_gpu_qwen35_attn_merge_fold_selfcheck(void) {
    static int done = -1;
    if (done >= 0) return done;
    const char *e = getenv("DS4_QWEN35_ATTN_MERGE_FOLD");
    if (!(e && strcmp(e, "1") == 0)) return 1;   /* knob off: nothing to check yet */
    const uint32_t H = 16u, Hkv = 2u, D = 256u, ns0 = 256u, ns1 = 37u;
    const uint64_t pf = ((uint64_t)ns0 + ns1) * H * (2u + D), of = 2ull * H * D;
    float *hp = malloc(pf * sizeof(float)), *hg = malloc(of * sizeof(float));
    float *ha = malloc(of * sizeof(float)), *hb = malloc(of * sizeof(float));
    ds4_gpu_tensor *tp = ds4_gpu_tensor_alloc(pf * sizeof(float)), *tg = ds4_gpu_tensor_alloc(of * sizeof(float));
    ds4_gpu_tensor *ta = ds4_gpu_tensor_alloc(of * sizeof(float)), *tb = ds4_gpu_tensor_alloc(of * sizeof(float));
    int ok = hp && hg && ha && hb && tp && tg && ta && tb;
    uint32_t r = 0x2b7e1516u;
    for (uint64_t i = 0; ok && i < pf; i++) {
        r = r * 1664525u + 1013904223u;
        const float v = (float)((int32_t)(r >> 8) - (1 << 23)) / (float)(1 << 23);
        const uint64_t k = i % (2u + D), slot = i / (2u + D);
        hp[i] = k == 0 ? 4.0f * v : k == 1 ? (slot % 8u == 7u ? 0.0f : 1.0f + fabsf(v) * 30.0f) : v;
    }
    for (uint64_t i = 0; ok && i < of; i++) { r = r * 1664525u + 1013904223u; hg[i] = (float)(int32_t)(r >> 8) / (float)(1 << 23); }
    struct ds4_qwen35_attn_decode3_args args = { H, Hkv, D, 0u, 2u, ns0, 1u, ns1, 1u, 0.0625f };
    ok = ok && ds4_gpu_tensor_write(tp, 0, hp, pf * sizeof(float)) && ds4_gpu_tensor_write(tg, 0, hg, of * sizeof(float));
    for (int pass = 0; pass < 2 && ok; pass++) {
        qwen4_bind mb[3];
        ok = qwen4_bind_tensor(&mb[0], tp, pf * sizeof(float), "fold check part") &&
             qwen4_bind_tensor(&mb[1], tg, of * sizeof(float), "fold check gate") &&
             qwen4_bind_tensor(&mb[2], pass ? tb : ta, of * sizeof(float), "fold check out") &&
             ds4_gpu_begin_commands() &&
             qwen4_dispatch(pass ? QWEN4_K_QWEN35_ATTN_MERGE3_FOLD : QWEN4_K_QWEN35_ATTN_MERGE3, &args,
                            sizeof(args), mb, 3, MTLSizeMake(H, 2, 1), MTLSizeMake(256, 1, 1), 0) &&
             ds4_gpu_end_commands() && ds4_gpu_synchronize();
    }
    ok = ok && ds4_gpu_tensor_read(ta, 0, ha, of * sizeof(float)) && ds4_gpu_tensor_read(tb, 0, hb, of * sizeof(float)) &&
         memcmp(ha, hb, of * sizeof(float)) == 0;
    free(hp); free(hg); free(ha); free(hb);
    ds4_gpu_tensor_free(tp); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(ta); ds4_gpu_tensor_free(tb);
    if (!ok) {
        g_qwen35_merge_fold_off = true;
        fprintf(stderr, "ds4: Ornith attention merge fold off: it failed its startup check against merge3\n");
    }
    done = ok ? 1 : 0;
    return done;
}
```

  - **`ds4_metal.m` dispatch sites.** In the three merge3 dispatches (the decode3 solo row, the decode3 shared
    rows and the flash merge), replace the first argument `QWEN4_K_QWEN35_ATTN_MERGE3` of `qwen4_dispatch` with
    `qwen35_merge3_kernel()`. Nothing else in those calls changes.
  - **`ds4_gpu.h`**, next to `ds4_gpu_qwen35_attn_part3_floats`:

```c
/* Round 2 Plan B: number of kernel_qwen35_attn_merge3_fold dispatches so far
 * (tests), and the once-per-process fold-vs-merge3 startup check (1 = knob
 * off or match; 0 = mismatch, fold turned off). */
uint64_t ds4_gpu_qwen35_attn_merge_fold_dispatches(void);
int ds4_gpu_qwen35_attn_merge_fold_selfcheck(void);
```

  - **`ds4.c`**, at line 76500, before the `if (e->glm_mtp) qwen35_verify_batch_selfcheck(...)` line:
    `(void)ds4_gpu_qwen35_attn_merge_fold_selfcheck();`. If the non-Metal build has no such symbol, guard it
    as the neighbouring Metal-only calls are guarded. Check with `grep -n "ds4_gpu_qwen35_matmul_q8_0_rows_tensor" ds4_cuda* ds4_gpu_stub* 2>/dev/null`;
    if Ornith calls are compiled into non-Metal builds, add stubs that return 1 and 0 next to the other
    `ds4_gpu_qwen35_*` stubs.
  - **Any non-Metal backend files** that implement `ds4_gpu.h`, if present, get stub definitions:
    `ds4_gpu_qwen35_attn_merge_fold_dispatches` returns 0, and `..._selfcheck` returns 1.

- [ ] **Step 4: Build.**
  Run: `make ds4 ds4-server tests/test_qwen35_kernels 2>&1 | grep -E "error|warning" | head`
  Expected: nothing.
- [ ] **Step 5: Commit** the test and the implementation, path-scoped, as two commits: the test first, then
  the implementation. Keep the RED evidence (Step 2's error line) in the ledger.
- [ ] **Step 6: GREEN in window r2w3,** the first step there.
  Run: `./tests/test_qwen35_kernels 2>&1 | grep -E "merge fold|FAIL|passed"`
  Expected:
  - 12 lines `attn merge fold ... bit-identical`;
  - both self-check requires pass;
  - the suite's final pass line, with no FAIL.

### Task 2: The bench row

**Files:**
- Modify: `tests/bench_qwen35_attn.c`, the `do_decode` loop at lines 247-255.

- [ ] **Step 1:** After the `report("decode3", ...)` line, add:

```c
                setenv("DS4_QWEN35_ATTN_MERGE_FOLD", "1", 1);
                report("decode3-fold", "decode", c.pos0, c.T, rows, time_ms(run_decode3, &c, 3, 9));
                unsetenv("DS4_QWEN35_ATTN_MERGE_FOLD");
```

- [ ] **Step 2:** Run `make tests/bench_qwen35_attn 2>&1 | grep -E "error|warning"`. Expected: nothing. Then
  commit.

### Task 3: Window r2w3 (exactness, bench, identity, paired A/B) and the records

**Files:**
- Create: `speed-bench/ornith/decode/round2/levers/merge-fold.md` and `round2/receipts/r2w3-*`.
- Create (scratch): `$SCR/ornith-decode/steps-r2w3.sh`.

- [ ] **Step 1: The steps file.** It uses `run_to` and `say` from `gpuwin.sh`, as in round 1's window C, and
  stops at the first failed check:

```bash
# Ornith decode round 2, window 3: merge fold exactness, bench, identity, paired A/B.
M1=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/a18593e5-414b-4337-9d32-8629ac82447a/scratchpad/ornith512k
export DS4_ORNITH_MODEL=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
export DS4_QWEN35_MTP_DRAFT_VOCAB=$HOME/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
unset DS4_QWEN4_YARN_FACTOR DS4_QWEN35_ATTN_MERGE_FOLD DS4_QWEN35_ATTN_SPLIT_KEYS
ok=1
run_to kernels 1800 ./tests/test_qwen35_kernels || ok=0
grep -E "merge fold" "$S/kernels.log" | tail -14 >> "$S/progress"
[ "$ok" = 1 ] || { say "kernel tests failed: stop"; false; }
run_to bench-attn 1800 ./tests/bench_qwen35_attn decode || ok=0
grep -E "decode3" "$S/bench-attn.log" >> "$S/progress"
L="DS4_QWEN35_ATTN_MERGE_FOLD=1"
for c in 64 512 2048; do
    rm -rf "$S/g1-$c"
    run_to "g1-$c" 3600 env $L python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg "$c" "$S/g1-$c" || ok=0
    n=0; for f in "$M1/m1-merge-$c"/ds4/*.json; do cmp -s "$f" "$S/g1-$c/ds4/$(basename "$f")" || n=$((n + 1)); done
    say "identity chunk $c (merge fold) vs window M: $n differ"; [ "$n" = 0 ] || ok=0
done
rm -rf "$S/mtp-cli"
run_to mtp-cli 7200 env $L python3 tests/ornith/test_mtp_cli.py "$S/mtp-cli" || ok=0
run_to verify-batch 3600 env $L make test-qwen35-verify-batch || ok=0
[ "$ok" = 1 ] || { say "identity failed: no A/B"; false; }
rm -rf "$S/ab"
run_to ab 3600 python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model "$DS4_ORNITH_MODEL" --out "$S/ab" \
    --base-env "" --lever-env "$L" --contexts 2048,32768 --cold-tokens 0 --paired --blocks 2 --reps 2 || ok=0
grep "m4_ab:" "$S/ab.log" >> "$S/progress"
rm -rf "$S/ab128"
run_to ab128 3600 python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model "$DS4_ORNITH_MODEL" --out "$S/ab128" \
    --base-env "" --lever-env "$L" --contexts 131072 --cold-tokens 0 --paired --blocks 1 --reps 1 || ok=0
grep "m4_ab:" "$S/ab128.log" | sed "s/^/128k /" >> "$S/progress"
say "window r2w3 ok=$ok"
[ "$ok" = 1 ]
```

- [ ] **Step 2: Run the window.** Announce it to Qwen, DS41F and CodeTab, ask DS41F for SSD quiet, run
  `gpuwin.sh` in the background, and send "done" afterwards.
  Expected: `ok=1`, 39 of 39 dumps identical (13 per chunk), `test_mtp_cli` and `verify-batch` pass, and the
  paired lines show `mismatches=0`.
- [ ] **Step 3: Write `levers/merge-fold.md`.** It records:
  - the bench `decode3` against `decode3-fold` at 2K/32K/128K, for rows 1 and 2;
  - the identity results;
  - the paired A/B table;
  - the §1 ruling: on by default if it gains at least 3% at 2K or 32K, at least twice the A/A noise
    (`AA.md`: 1.4% at 2K, 2.6% at 32K), and no context loses more than 3%.

  Copy the receipts and commit.
- [ ] **Step 4: If §1 passes,** flip the default in `qwen35_merge3_kernel`, so that unset means on and only
  `0` turns it off. Update the test so it toggles `=0` and `=1` explicitly. Re-run the kernel tests and the
  gate-1 identity in the next window. Then spec §1.2's final paired A/B (3 blocks), the whole-branch review,
  the report, and a question to the user for merge, push and deploy.
