# GSQ-RCO prefill tiled GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ISTA's prefill leaves the per-token kernels: routed GSQ-RCO experts run the tiled MoE GEMM
(`kernel_qwen4_moe_mm_{mid,down}`) and dense GSQ-RCO and BF16 tensors run the tiled dense GEMM
(`kernel_qwen4_dense_mm`), so a prefill chunk reads each weight row once per token tile.

**Architecture:** Both tiled kernels stage weights through one per-type dequantizer,
`qwen4_mm_stage8` (8 consecutive values of a 32-wide block). Adding the nine GSQ-RCO types there
gives both GEMMs the types; the host dispatchers widen their type checks, the M5 specialized
pipeline accepts type 42, and the graph routes GSQ-RCO experts (T > 64) and GSQ-RCO/BF16 dense rows
(T > 8) to the GEMMs. The Metal 4 tensor-op tiles (`_nax`) stay on their four types; a tensor-op
variant for the new types is a follow-up only if the measured prefill is still short.

**Tech Stack:** C (ds4.c, tests/ds4_test.c), Objective-C (ds4_metal.m), Metal (metal/qwen4.metal).

**Spec:** `docs/superpowers/specs/2026-09-30-gsq-rco-types-design.md` (Engine items 3-4: prefill for
routed and dense types). Open item 1 of `speed-bench/nextgen-eval/results/2026-09-30-sp3-gsq-rco.md`.

## Global Constraints

- Models without the new types run byte-identically: every change is behind the tensor type (spec, Engine 8).
- Mac only; CUDA/ROCm unchanged (`qwen4_graph_gsq_ok` is false off Metal).
- No push, no deploy, no HF upload without the user. Git holds source and small result files only.
- GPU model runs take the instance lock: they run inside a gateway pause (hold-gateway.sh), restored after.
- Never `kill -9` a Metal process.
- Code, docs and commits in English.

## Review Focus

1. Q2_0 is GGML type 42; the M5 MoE GEMM pipelines are specialized per type and refused `type >= 40`.
2. Partial token tiles: an expert whose pair count is not a multiple of the tile, and experts with no pairs.
3. The dense GEMM's k-split (narrow outputs, few token tiles): split boundaries fall on 32-wide
   blocks, inside a 256-wide super-block or a 64-wide Q2_0 block.
4. BF16 dense rows whose width is not a multiple of 32 must honour the row end, like F16.
5. MTP verify rows (2-3 tokens) and decode stay on the verified per-token kernels.

---

### Task 1: stage8 dequantizers for the GSQ-RCO types; the tiled MoE GEMM accepts them

**Files:**
- Modify: `metal/qwen4.metal` (`qwen4_mm_stage8`)
- Modify: `ds4_metal.m` (`qwen4_mm_gsq_block`, `ds4_gpu_qwen4_moe_mm_mid_tensor`, `ds4_gpu_qwen4_moe_mm_down_tensor`, MoE GEMM pipeline type bound)
- Test: `tests/ds4_test.c` (`test_metal_qwen4_quant_moe_mm`, called from the `__APPLE__` block of `test_metal_kernel_group`)

**Interfaces:**
- Produces: `qwen4_mm_stage8<D>(row, b, q, type, dst)` handles types 13, 14, 17, 18, 20, 21, 22, 23, 42;
  `static uint32_t qwen4_mm_gsq_block(uint32_t type)` in ds4_metal.m (256, 32 for IQ4_NL, 64 for Q2_0, 0 otherwise).

- [ ] **Step 1: Write the failing test** (after `test_metal_qwen4_quant_moe`)

```c
/* The tiled prefill GEMMs (kernel_qwen4_moe_mm_mid/down) on 3 experts and 40 tokens x 2 slots:
 * expert 0 gets 40 pairs (two token tiles), 1 and 2 the rest. Weights and activations are staged
 * as half, so the bound is 1.5e-3 of the magnitude. */
static void test_metal_qwen4_quant_moe_mm_case(uint32_t gu, uint32_t dn) {
    const uint32_t in_dim = 2560u, ff = 640u, out_dim = 2560u, n_exp = 3u, T = 40u, slots = 2u;
    const ds4_quant_fixture *fg = test_quant_fixture(gu), *fd = test_quant_fixture(dn);
    TEST_ASSERT(fg && fd);
    if (!fg || !fd) return;
    const uint64_t g_row = test_quant_row_bytes(gu, in_dim), g_exp = g_row * ff;
    const uint64_t d_row = test_quant_row_bytes(dn, ff), d_exp = d_row * out_dim;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t off_up = test_round_up_u64(g_exp * n_exp, page);
    const uint64_t off_dn = off_up + test_round_up_u64(g_exp * n_exp, page);
    const uint64_t alloc = off_dn + test_round_up_u64(d_exp * n_exp, page);
    uint8_t *w = NULL;
    TEST_ASSERT(posix_memalign((void **)&w, (size_t)page, (size_t)alloc) == 0);
    if (!w) return;
    memset(w, 0, (size_t)alloc);
    test_quant_fill_rows(w, fg, in_dim, ff * n_exp);
    test_quant_fill_rows(w + off_up, fg, in_dim, ff * n_exp);
    memcpy(w + off_up, w + off_up + g_row, (size_t)g_row);           /* up row 0 differs from gate row 0 */
    test_quant_fill_rows(w + off_dn, fd, ff, out_dim * n_exp);
    int32_t sel[80];
    for (uint32_t t = 0; t < T; t++) { sel[t * 2u] = 0; sel[t * 2u + 1u] = (t % 3u) == 0 ? 2 : 1; }
    float *xh = malloc((size_t)T * in_dim * sizeof(float));
    float *mh = malloc((size_t)T * slots * ff * sizeof(float));
    float *ph = malloc((size_t)T * slots * out_dim * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)T * in_dim * sizeof(float));
    ds4_gpu_tensor *s = ds4_gpu_tensor_alloc(sizeof(sel));
    ds4_gpu_tensor *lists = ds4_gpu_tensor_alloc((uint64_t)n_exp * T * sizeof(int32_t));
    ds4_gpu_tensor *counts = ds4_gpu_tensor_alloc((uint64_t)n_exp * sizeof(int32_t));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)T * slots * ff * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc((uint64_t)T * slots * out_dim * sizeof(float));
    TEST_ASSERT(xh && mh && ph && x && s && lists && counts && mid && part);
    if (xh && mh && ph && x && s && lists && counts && mid && part) {
        for (uint32_t t = 0; t < T; t++) test_quant_x(xh + (uint64_t)t * in_dim, in_dim, t + 11u);
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, xh, (uint64_t)T * in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(s, 0, sel, sizeof(sel)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(w, alloc) != 0);
        TEST_ASSERT(ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, s, T, slots, n_exp, T) != 0);
        TEST_ASSERT(ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, w, alloc, 0, off_up, gu, n_exp, T, slots,
                                                    slots, in_dim, ff, T) != 0);
        TEST_ASSERT(ds4_gpu_qwen4_moe_mm_down_tensor(part, mid, lists, counts, w, alloc, off_dn, dn, n_exp, T, slots,
                                                     slots, ff, out_dim, T) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(mid, 0, mh, (uint64_t)T * slots * ff * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(part, 0, ph, (uint64_t)T * slots * out_dim * sizeof(float)) != 0);
        uint32_t bad_mid = 0, bad_down = 0;
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t k = 0; k < slots; k++) {
                const uint64_t e = (uint64_t)sel[t * slots + k];
                const float *xt = xh + (uint64_t)t * in_dim, *mt = mh + ((uint64_t)t * slots + k) * ff;
                for (uint32_t r = 0; r < ff; r++) {
                    double mg = 0.0, mu = 0.0;
                    const double gv = test_quant_row_ref(gu, w + e * g_exp + r * g_row, in_dim, xt, &mg);
                    const double uv = test_quant_row_ref(gu, w + off_up + e * g_exp + r * g_row, in_dim, xt, &mu);
                    const double silu = gv / (1.0 + exp(-gv)), ref = silu * uv;
                    if (fabs((double)mt[r] - ref) > 1.5e-3 * (1.1 * fabs(uv) * mg + fabs(silu) * mu) + 1e-5) bad_mid++;
                }
                for (uint32_t r = 0; r < out_dim; r++) {
                    double mag = 0.0;
                    const double ref = test_quant_row_ref(dn, w + off_dn + e * d_exp + r * d_row, ff, mt, &mag);
                    if (fabs((double)ph[((uint64_t)t * slots + k) * out_dim + r] - ref) > 1.5e-3 * mag + 1e-5) bad_down++;
                }
            }
        }
        if (bad_mid || bad_down) fprintf(stderr, "ds4-test: Qwen MoE GEMM %u/%u: mid %u, down %u rows off\n",
                                         gu, dn, bad_mid, bad_down);
        TEST_ASSERT(bad_mid == 0 && bad_down == 0);
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(s);
    ds4_gpu_tensor_free(lists);
    ds4_gpu_tensor_free(counts);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part);
    free(xh);
    free(mh);
    free(ph);
    free(w);
}

static void test_metal_qwen4_quant_moe_mm(void) {
    test_metal_qwen4_quant_moe_mm_case(17, 42);   /* IQ2_XS gate/up, Q2_0 down */
    test_metal_qwen4_quant_moe_mm_case(22, 20);   /* IQ2_S, IQ4_NL */
    test_metal_qwen4_quant_moe_mm_case(18, 42);   /* IQ3_XXS, Q2_0 */
    test_metal_qwen4_quant_moe_mm_case(21, 20);   /* IQ3_S, IQ4_NL */
    test_metal_qwen4_quant_moe_mm_case(16, 20);   /* IQ2_XXS (existing tiles), IQ4_NL down */
}
```

Call it right after `test_metal_qwen4_quant_moe();` in `test_metal_kernel_group`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: FAIL: `ds4_gpu_qwen4_moe_mm_mid_tensor(...) != 0` asserts for type 17 (the host refuses the type).

- [ ] **Step 3: Implement**

`metal/qwen4.metal`, in `qwen4_mm_stage8` after the `type == 12` branch:

```metal
    /* GSQ-RCO types (ggml-quants.c dequantizers @931351ea, MIT), the same
     * coordinates as the qwen4_lane_* helpers: super-block b / 8, 32-wide
     * sub-block b % 8, and the quarter's 8 values */
    if (type == 17) {   /* IQ2_XS */
        const uint ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 74);
        const uint sc = blk[66 + ib32];
        const float db = (float)(*(device const half *)blk) * (0.5f + (float)(q < 2 ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
        const uint v = ((device const ushort *)(blk + 2))[4 * ib32 + q];
        constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xs_grid + (v & 511u));
        const uint signs = ds4_metal_ksigns_iq2xs[v >> 9];
        for (uint i = 0; i < 8; i++) dst[i] = (D)(db * (float)grid[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
        return;
    }
    if (type == 22) {   /* IQ2_S */
        const uint ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 82);
        const uint qh = blk[66 + ib32], sc = blk[74 + ib32];
        const float db = (float)(*(device const half *)blk) * (0.5f + (float)(q < 2 ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
        constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2s_grid +
            ((uint)blk[2 + 4 * ib32 + q] | ((qh << (8u - 2u * q)) & 0x300u)));
        const uint s = blk[34 + 4 * ib32 + q];
        for (uint i = 0; i < 8; i++) dst[i] = (D)(db * (float)grid[i] * ((s >> i) & 1u ? -1.0f : 1.0f));
        return;
    }
    if (type == 18) {   /* IQ3_XXS */
        const uint ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 98);
        device const ushort *ss = (device const ushort *)(blk + 66) + 2 * ib32;
        const uint aux = (uint)ss[0] | ((uint)ss[1] << 16);
        const float db = (float)(*(device const half *)blk) * (0.5f + (float)(aux >> 28)) * 0.5f;
        const uint signs = ds4_metal_ksigns_iq2xs[(aux >> (7u * q)) & 127u];
        device const uchar *qs = blk + 2 + 8 * ib32 + 2 * q;
        constant const uchar *g1 = (constant const uchar *)(ds4_metal_iq3xxs_grid + qs[0]);
        constant const uchar *g2 = (constant const uchar *)(ds4_metal_iq3xxs_grid + qs[1]);
        for (uint i = 0; i < 4; i++) {
            dst[i] = (D)(db * (float)g1[i] * ((signs >> i) & 1u ? -1.0f : 1.0f));
            dst[4 + i] = (D)(db * (float)g2[i] * ((signs >> (i + 4u)) & 1u ? -1.0f : 1.0f));
        }
        return;
    }
    if (type == 21) {   /* IQ3_S */
        const uint ib32 = b % 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 110);
        const uint qh = blk[66 + ib32], s = blk[74 + 4 * ib32 + q];
        const float db = (float)(*(device const half *)blk) *
                         (float)(1u + 2u * ((blk[106 + ib32 / 2] >> (4u * (ib32 & 1u))) & 0xFu));
        device const uchar *qs = blk + 2 + 8 * ib32 + 2 * q;
        constant const uchar *g1 = (constant const uchar *)(ds4_metal_iq3s_grid + ((uint)qs[0] | ((qh << (8u - 2u * q)) & 256u)));
        constant const uchar *g2 = (constant const uchar *)(ds4_metal_iq3s_grid + ((uint)qs[1] | ((qh << (7u - 2u * q)) & 256u)));
        for (uint i = 0; i < 4; i++) {
            dst[i] = (D)(db * (float)g1[i] * ((s >> i) & 1u ? -1.0f : 1.0f));
            dst[4 + i] = (D)(db * (float)g2[i] * ((s >> (i + 4u)) & 1u ? -1.0f : 1.0f));
        }
        return;
    }
    if (type == 20 || type == 23) {   /* IQ4_NL (18-byte blocks of 32), IQ4_XS (136-byte super-blocks) */
        device const uchar *qs;
        float dl;
        if (type == 20) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)b * 18);
            dl = (float)(*(device const half *)blk);
            qs = blk + 2;
        } else {
            const uint ib32 = b % 8;
            device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 136);
            const uint scales_h = (uint)*(device const ushort *)(blk + 2);
            const int ls = (int)(((blk[4 + ib32 / 2] >> (4u * (ib32 % 2u))) & 0xFu) | (((scales_h >> (2u * ib32)) & 3u) << 4));
            dl = (float)(*(device const half *)blk) * (float)(ls - 32);
            qs = blk + 8 + 16 * ib32;
        }
        qs += (q & 1u) * 8;
        const bool hi = q >= 2;
        for (uint i = 0; i < 8; i++) dst[i] = (D)(dl * (float)ds4_metal_kvalues_iq4nl[hi ? (qs[i] >> 4) : (qs[i] & 0xFu)]);
        return;
    }
    if (type == 42) {   /* Q2_0: 18-byte blocks of 64, 2-bit codes minus one, four per byte */
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 2) * 18);
        const float d = (float)(*(device const half *)blk);
        device const uchar *qs = blk + 2 + ((b % 2) * 32 + q * 8) / 4;
        for (uint i = 0; i < 8; i++) dst[i] = (D)(d * (float)((int)((qs[i / 4] >> ((i % 4) * 2)) & 3u) - 1));
        return;
    }
    if (type == 13) {   /* Q5_K */
        const uint g = b % 8, l = q * 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 176);
        const float d = (float)(*(device const half *)blk), dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (g < 4) { s = sc[g] & 63u; mn = sc[g + 4] & 63u; }
        else { s = (sc[g + 4] & 0xFu) | ((sc[g - 4] >> 6) << 4); mn = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        device const uchar *qh = blk + 16 + l, *ql = blk + 48 + 32 * (g / 2) + l;
        for (uint i = 0; i < 8; i++) {
            const uint lo = (g & 1u) ? (uint)(ql[i] >> 4) : (uint)(ql[i] & 0xFu);
            dst[i] = (D)(ds * (float)(lo + (((qh[i] >> g) & 1u) << 4)) - dm);
        }
        return;
    }
    if (type == 14) {   /* Q6_K */
        const uint g = b % 8, h = g / 4, k = g % 4, l = q * 8;
        device const uchar *blk = (device const uchar *)(row + (uint64_t)(b / 8) * 210);
        device const uchar *ql = blk + 64 * h + ((k & 1u) ? 32 : 0) + l, *qh = blk + 128 + 32 * h + l;
        const float ds = (float)(*(device const half *)(blk + 208)) *
                         (float)((device const char *)(blk + 192 + 8 * h))[2 * k + (l >= 16 ? 1 : 0)];
        for (uint i = 0; i < 8; i++) {
            const uint lo = (k < 2) ? (uint)(ql[i] & 0xFu) : (uint)(ql[i] >> 4);
            dst[i] = (D)(ds * (float)((int)(lo | (((qh[i] >> (2u * k)) & 3u) << 4)) - 32));
        }
        return;
    }
```

`ds4_metal.m`, before `ds4_gpu_qwen4_moe_mm_mid_tensor`:

```c
/* GSQ-RCO types the tiled prefill GEMMs stage (qwen4_mm_stage8), by block
 * width: the K-quant and IQ super-blocks, IQ4_NL's 32 and Q2_0's 64.  Zero
 * for every other type. */
static uint32_t qwen4_mm_gsq_block(uint32_t type) {
    switch (type) {
    case 13u: case 14u: case 17u: case 18u: case 21u: case 22u: case 23u: return 256u;
    case 20u: return 32u;
    case 42u: return 64u;
    default: return 0u;
    }
}
```

In both `ds4_gpu_qwen4_moe_mm_mid_tensor` and `ds4_gpu_qwen4_moe_mm_down_tensor`, the type check becomes

```c
        (weight_type != 8u && weight_type != 39u && weight_type != 12u && weight_type != 10u && weight_type != 16u &&
         weight_type != 2u && !qwen4_mm_gsq_block(weight_type)) ||
        (qwen4_mm_gsq_block(weight_type) && (in_dim % qwen4_mm_gsq_block(weight_type)) != 0) ||
```

(`ff_dim` in place of `in_dim` in the down function.) In the MoE GEMM pipeline creation, `if (type >= 40u) return 0;`
becomes `if (type >= 40u && type != 42u) return 0;   /* 42: Q2_0 */`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK`.

- [ ] **Step 5: Commit**

```bash
git add metal/qwen4.metal ds4_metal.m tests/ds4_test.c
git commit -m "metal: the tiled qwen4 MoE GEMM stages the GSQ-RCO types"
```

### Task 2: the tiled dense GEMM takes GSQ-RCO and BF16 rows

**Files:**
- Modify: `metal/qwen4.metal` (`qwen4_dm_stage8`, the args comment)
- Modify: `ds4_metal.m` (`ds4_gpu_qwen4_dense_mm_tensor`), `ds4_gpu.h` (comment)
- Test: `tests/ds4_test.c` (`test_metal_qwen4_quant_dense_mm`)

**Interfaces:**
- Consumes: `qwen4_mm_stage8` types and `qwen4_mm_gsq_block` from Task 1.
- Produces: `ds4_gpu_qwen4_dense_mm_tensor` accepts types 13, 14, 17, 18, 20, 21, 22, 23, 42 (in_dim a
  multiple of the block) and 30 (BF16, in_dim a multiple of 8).

- [ ] **Step 1: Write the failing test** (after `test_metal_qwen4_quant_moe_mm`)

```c
/* The tiled dense GEMM on 37 rows: 40 tokens (one full and one partial token tile) and 9 tokens
 * (few threadgroups, so the k-split planes and their reduce run). fp32 staging, fp32 bound. */
static void test_metal_qwen4_dense_mm_case(uint32_t type, const uint8_t *rows_bytes, uint64_t row_bytes,
                                           uint32_t in_dim, uint32_t rows) {
    const uint64_t page = (uint64_t)getpagesize(), alloc = test_round_up_u64(row_bytes * rows, page);
    void *wraw = NULL;
    TEST_ASSERT(posix_memalign(&wraw, (size_t)page, (size_t)alloc) == 0);
    if (!wraw) return;
    memcpy(wraw, rows_bytes, (size_t)(row_bytes * rows));
    const uint32_t toks[2] = { 40u, 9u };
    for (int ti = 0; ti < 2; ti++) {
        const uint32_t n_tok = toks[ti];
        float *xh = malloc((size_t)n_tok * in_dim * sizeof(float));
        float *oh = malloc((size_t)n_tok * rows * sizeof(float));
        ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *o = ds4_gpu_tensor_alloc((uint64_t)n_tok * rows * sizeof(float));
        TEST_ASSERT(xh && oh && x && o);
        if (xh && oh && x && o) {
            for (uint32_t t = 0; t < n_tok; t++) test_quant_x(xh + (uint64_t)t * in_dim, in_dim, t + 3u);
            TEST_ASSERT(ds4_gpu_tensor_write(x, 0, xh, (uint64_t)n_tok * in_dim * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_set_model_map(wraw, alloc) != 0);
            TEST_ASSERT(ds4_gpu_qwen4_dense_mm_tensor(o, x, wraw, alloc, 0, type, n_tok, in_dim, rows) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(o, 0, oh, (uint64_t)n_tok * rows * sizeof(float)) != 0);
            uint32_t bad = 0;
            for (uint32_t t = 0; t < n_tok; t++) {
                for (uint32_t r = 0; r < rows; r++) {
                    double mag = 0.0;
                    const double ref = test_quant_row_ref(type, (const uint8_t *)wraw + r * row_bytes, in_dim,
                                                          xh + (uint64_t)t * in_dim, &mag);
                    if (fabs((double)oh[(uint64_t)t * rows + r] - ref) > 1e-5 * mag + 1e-6) bad++;
                }
            }
            if (bad) fprintf(stderr, "ds4-test: Qwen dense GEMM type %u dim %u tokens %u: %u/%u rows off\n",
                             type, in_dim, n_tok, bad, n_tok * rows);
            TEST_ASSERT(bad == 0);
        }
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(o);
        free(xh);
        free(oh);
    }
    free(wraw);
}

static void test_metal_qwen4_quant_dense_mm(void) {
    const uint32_t rows = 37u;
    for (size_t i = 0; i < sizeof(quant_fixtures) / sizeof(quant_fixtures[0]); i++) {
        const ds4_quant_fixture *f = &quant_fixtures[i];
        if (f->type == 16) continue;                        /* IQ2_XXS dense rows keep their kernels */
        const uint32_t in_dim = (f->type == 20 || f->type == 42) ? 640u : 2560u;
        const uint64_t row_bytes = test_quant_row_bytes(f->type, in_dim);
        uint8_t *buf = malloc((size_t)(row_bytes * rows));
        TEST_ASSERT(buf != NULL);
        if (!buf) return;
        test_quant_fill_rows(buf, f, in_dim, rows);
        test_metal_qwen4_dense_mm_case(f->type, buf, row_bytes, in_dim, rows);
        free(buf);
    }
    /* BF16 rows 2568 wide: the last k tile ends mid-tile */
    const uint32_t in_dim = 2568u;
    uint16_t *bf = malloc((size_t)in_dim * rows * sizeof(uint16_t));
    float *v = malloc((size_t)in_dim * sizeof(float));
    TEST_ASSERT(bf && v);
    if (bf && v) {
        for (uint32_t r = 0; r < rows; r++) {
            test_quant_x(v, in_dim, r + 101u);
            for (uint32_t k = 0; k < in_dim; k++) {
                uint32_t u;
                memcpy(&u, &v[k], sizeof(u));
                bf[(uint64_t)r * in_dim + k] = (uint16_t)(u >> 16);
            }
        }
        test_metal_qwen4_dense_mm_case(30u, (const uint8_t *)bf, (uint64_t)in_dim * 2u, in_dim, rows);
    }
    free(bf);
    free(v);
}
```

`test_quant_row_ref` reads through `ds4_dequant_row`; check that it covers BF16 (type 30). If it does
not, the BF16 case computes its reference inline from the same `bf` values (`u << 16`).
Call it right after `test_metal_qwen4_quant_moe_mm();`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: FAIL: `ds4: Qwen3.8 dense mm rejected type 13 ...` and the `!= 0` assert.

- [ ] **Step 3: Implement**

`metal/qwen4.metal`, `qwen4_dm_stage8`:

```metal
/* 8 consecutive weights of row `row` starting at element k0 (k0 % 8 == 0);
 * f32/f16/bf16 rows may end mid-tile (in_dim % 32 != 0), quantized rows
 * cannot */
static inline void qwen4_dm_stage8(device const char *row, uint k0, uint k_end, uint type, threadgroup float *dst) {
    if (type == 1) {
        device const half *w = (device const half *)row + k0;
        for (uint i = 0; i < 8; i++) dst[i] = k0 + i < k_end ? (float)w[i] : 0.0f;
    } else if (type == 0) {
        device const float *w = (device const float *)row + k0;
        for (uint i = 0; i < 8; i++) dst[i] = k0 + i < k_end ? w[i] : 0.0f;
    } else if (type == 30) {
        device const ushort *w = (device const ushort *)row + k0;
        for (uint i = 0; i < 8; i++) dst[i] = k0 + i < k_end ? as_type<float>((uint)w[i] << 16) : 0.0f;
    } else {
        qwen4_mm_stage8<float>(row, k0 / 32, (k0 % 32) / 8, type, dst);
    }
}
```

The args comment becomes `/* 0 f32, 1 f16, 8 q8_0, 30 bf16, or a GSQ-RCO type */` and the section header
`dense tiled GEMM for f32/f16/bf16/q8_0 and GSQ-RCO weights`.

`ds4_metal.m`, `ds4_gpu_qwen4_dense_mm_tensor`:

```c
    const uint32_t gsq = qwen4_mm_gsq_block(weight_type);
    const uint32_t row_bytes = weight_type == 0u ? in_dim * 4u :
                               (weight_type == 1u || weight_type == 30u) ? in_dim * 2u :
                               qwen4_expert_row_bytes(weight_type, in_dim);
    struct { ... } args = ...;   /* unchanged */
    qwen4_bind b[3];
    if (n_tokens == 0 || row_bytes == 0 ||
        (weight_type != 0u && weight_type != 1u && weight_type != 8u && weight_type != 30u && !gsq) ||
        (weight_type == 8u && (in_dim % 32) != 0) || (gsq && (in_dim % gsq) != 0) ||
        (in_dim % 8) != 0 || out_rows == 0) {
```

`qwen4_mm_gsq_block` moves above `ds4_gpu_qwen4_dense_mm_tensor` if it is defined later in the file.
`ds4_gpu.h`: the comment above `ds4_gpu_qwen4_dense_mm_tensor` reads
`prefill dense GEMM (f32/f16/bf16/q8_0 and GSQ-RCO rows, 32x32 tiles)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK`.

- [ ] **Step 5: Commit**

```bash
git add metal/qwen4.metal ds4_metal.m ds4_gpu.h tests/ds4_test.c
git commit -m "metal: the tiled qwen4 dense GEMM takes GSQ-RCO and BF16 rows"
```

### Task 3: the Qwen3.8 graph routes GSQ-RCO prefill to the GEMMs; model-level checks

**Files:**
- Modify: `ds4.c` (`qwen4_expert_type_has_mm`, `qwen4_dense_mm_rows_ok` + its use in `qwen4_gemv_rows`, test hooks)
- Modify: `ds4.h` (test hook declarations next to `ds4_test_routed_expert_type_ok`)
- Test: `tests/ds4_test.c` (`test_quant_types`)
- Create: `speed-bench/nextgen-eval/results/2026-10-01-sp3-prefill-gemm.md`
- Modify: `docs/QWEN38_FLASH_NEXT.md` (the GSQ-RCO subsection's "not supported" list)

**Interfaces:**
- Consumes: Tasks 1-2 (`ds4_gpu_qwen4_moe_mm_*_tensor` and `ds4_gpu_qwen4_dense_mm_tensor` accept the types).
- Produces: `int ds4_test_qwen4_expert_has_mm(uint32_t type)`, `int ds4_test_qwen4_dense_mm_rows(uint32_t type, uint32_t n_tok, uint64_t in_dim)`.

- [ ] **Step 1: Write the failing test** (end of `test_quant_types`)

```c
#if defined(__APPLE__)
    /* prefill routing: GSQ-RCO experts take the tiled MoE GEMM, GSQ-RCO and BF16 dense rows the
     * tiled dense GEMM above 8 rows; PROD's types keep their kernels */
    TEST_ASSERT(ds4_test_qwen4_expert_has_mm(16) == 1);
    TEST_ASSERT(ds4_test_qwen4_expert_has_mm(18) == 1);
    TEST_ASSERT(ds4_test_qwen4_expert_has_mm(42) == 1);
    TEST_ASSERT(ds4_test_qwen4_expert_has_mm(30) == 0);
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(21, 40, 2560) == 1);
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(30, 40, 2560) == 1);
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(21, 8, 2560) == 0);   /* decode and MTP verify rows */
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(21, 40, 2600) == 0);  /* not whole super-blocks */
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(8, 40, 2560) == 0);   /* Q8_0 keeps its kernels */
    TEST_ASSERT(ds4_test_qwen4_dense_mm_rows(12, 40, 2560) == 0);  /* Q4_K keeps the ggml GEMM */
#endif
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make ds4_test`
Expected: compile error, `ds4_test_qwen4_expert_has_mm` undeclared (the hooks do not exist yet).

- [ ] **Step 3: Implement**

`ds4.c`:

```c
/* expert types the tiled prefill GEMM stages (kernel_qwen4_moe_mm_*) */
static bool qwen4_expert_type_has_mm(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 || type == DS4_TENSOR_MXFP4 || type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q2_K || type == DS4_TENSOR_IQ2_XXS || qwen4_graph_gsq_ok(type);
}

/* GSQ-RCO and BF16 dense rows take the tiled dense GEMM once a batch has more
 * than 8 rows; the multi-row gemv reads their weights once per token.
 * Decode and MTP verify rows keep the gemv. */
static bool qwen4_dense_mm_rows_ok(uint32_t type, uint32_t n_tok, uint64_t in_dim) {
#ifdef DS4_HAS_QWEN4_METAL
    if (n_tok <= 8u || (in_dim % 8u) != 0 || in_dim > UINT32_MAX) return false;
    if (type == DS4_TENSOR_BF16) return true;
    return qwen4_graph_gsq_ok(type) && tensor_type(type) && (in_dim % tensor_type(type)->block_elems) == 0;
#else
    (void)type; (void)n_tok; (void)in_dim;
    return false;
#endif
}

int ds4_test_qwen4_expert_has_mm(uint32_t type) { return qwen4_expert_type_has_mm(type) ? 1 : 0; }
int ds4_test_qwen4_dense_mm_rows(uint32_t type, uint32_t n_tok, uint64_t in_dim) {
    return qwen4_dense_mm_rows_ok(type, n_tok, in_dim) ? 1 : 0;
}
```

In `qwen4_gemv_rows` (Apple branch), after the F32/F16 dense-mm rule:

```c
    if (!legacy && qwen4_dense_mm_rows_ok(w->type, n_tok, in_dim)) {
        rc = ds4_gpu_qwen4_dense_mm_tensor(out, x, m->map, m->size, w->abs_offset, w->type, n_tok,
                                           (uint32_t)in_dim, (uint32_t)out_dim);
        if (rc) return true;
    }
```

`ds4.h`, after the `ds4_test_routed_expert_type_ok` declaration:

```c
int ds4_test_qwen4_expert_has_mm(uint32_t type);
int ds4_test_qwen4_dense_mm_rows(uint32_t type, uint32_t n_tok, uint64_t in_dim);
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make ds4 ds4_test ds4-bench && ./ds4_test --quant-types && ./ds4_test --metal-kernels`
Expected: `quant-types: OK`, `metal-kernels: OK`.

- [ ] **Step 5: Commit**

```bash
git add ds4.c ds4.h tests/ds4_test.c
git commit -m "qwen4: GSQ-RCO prefill takes the tiled MoE and dense GEMMs"
```

- [ ] **Step 6: Model-level checks** (one GPU window: `hold-gateway.sh`, restore after; `vm_stat` logged)

Baseline binaries: develop `6985bdb` (before this plan) built from `git archive` in
`ds4-metal-data/sp3/baseline-gemm`. Prompt: `ds4-metal-data/sp3/prompt-gemm.txt`, about 140 tokens of
mixed prose and code.

1. Correctness, `DS4_QWEN4_GPU=1 ./ds4 --first-token-test` on the 140-token prompt:
   - ISTA, `DS4_QWEN4_GPU_CHUNK=64` (every chunk at or below 64 rows: the verified per-token path);
   - ISTA, `DS4_QWEN4_GPU_CHUNK=256` (one chunk: the tiled GEMMs);
   - Ivan IQ2, `DS4_QWEN4_GPU_CHUNK=256` (PROD's kernel family on the tiled path: the accepted gap).
   Pass: ISTA's chunk-256 worst max|gpu-cpu| is at most max(Ivan chunk 256, ISTA chunk 64) x 1.25, and its
   top-1 agreement is at most one position below the better of the two.
2. Coherence: the SP3 three prompts, `ds4 --metal -c 32768 --temp 0 -n 256 --mtp`: coherent, MTP acceptance reported.
3. Speed: `ds4-bench` 8K story, 128 tokens, 3 rounds, ISTA resident: prefill and decode t/s against SP3's 59.4 / 25.2.
4. PROD: identity (3 prompts, 128 greedy tokens, `-c 262144`, PROD flags) byte-identical to the baseline
   binaries; `ds4-bench` A/B 3 rounds each: decode and prefill at least 0.98 x baseline.

Results go to `speed-bench/nextgen-eval/results/2026-10-01-sp3-prefill-gemm.md`; the docs' "not supported"
list drops the tiled prefill GEMM. Commit:

```bash
git add speed-bench/nextgen-eval/results/2026-10-01-sp3-prefill-gemm.md docs/QWEN38_FLASH_NEXT.md
git commit -m "sp3 results: ISTA prefill on the tiled GEMMs"
```
