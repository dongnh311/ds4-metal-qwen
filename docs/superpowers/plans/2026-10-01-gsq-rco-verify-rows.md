# GSQ-RCO decode and MTP verify rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MTP pays off on GSQ-RCO models: a dense GSQ-RCO or BF16 row is dequantized once per decode
or verify step, however many tokens the step carries.

**Architecture:** Dense GSQ-RCO and BF16 rows currently run `kernel_qwen4_multi_gemv` at 2-8 tokens.
That kernel runs the `qwen4_lane_*` helpers once per (row, token), so a 3-row MTP verify dequantizes
every weight three times.

Microbenchmark: a 2560 x 24576 Q6_K matrix on an M5 Pro.

| tokens | current | chunked prototype | PROD's Q4_K kernel |
|---|---|---|---|
| 1 | 0.238 ms | 0.202 ms | — |
| 2 | 0.451 ms | 0.220 ms | — |
| 3 | 0.665 ms (×2.8) | 0.272 ms | ×2.1 |

The new `kernel_qwen4_gsq_mv<R1>` gives each lane a 16-value chunk. The lane dequantizes it once into
registers through a register dequantizer, `qwen4_gsq_deq8`, and dots it with R1 activation columns.
`qwen4_mm_stage8`'s GSQ-RCO branches move into `qwen4_gsq_deq8`, so the tiled GEMMs and the new
kernel share one dequantizer. One-token decode uses the same kernel (R1 = 1). As a result, a verify
row matches the decode row bit for bit.

**Tech Stack:** Metal (metal/qwen4.metal), Objective-C (ds4_metal.m, ds4_gpu.h), C tests (tests/ds4_test.c).

**Spec:** `docs/superpowers/specs/2026-09-28-qwen-nextgen-design.md`:
- sub-project 5's speed gate ("total answer time lower");
- the risk row "IQ-type decode slower on Metal".

`docs/superpowers/specs/2026-09-30-gsq-rco-types-design.md` lists "fused kernels and M5 tuning … only
if sub-project 5 shows speed is short". The 2026-10-01 arm did show that: ISTA ran 27.3 t/s with MTP
against 40.6 for Ivan with the projection.

## Global Constraints

- Models without GSQ-RCO or BF16 dense rows run byte-identically. The new kernel is chosen only when
  every output of the call is GSQ-RCO or BF16.
- Mac only; CUDA unchanged. No push, deploy or HF upload without the user.
- GPU model runs happen inside a gateway pause; never `kill -9` a Metal process.
- Code, docs and commits in English.

## Review Focus

1. A verify column must equal the one-token decode row bit for bit (MTP verify commits autoregressive-identical tokens).
2. The tail of a 5-7 token batch: R1 = 4 with masked columns must not write or read past `n_tokens`.
3. Fused multi-output calls (`n_out` up to 4, mixed GSQ-RCO/BF16 types) map rows to outputs as `kernel_qwen4_multi_gemv` does.
4. BF16 rows whose width is not a multiple of 16 keep the old kernel.
5. The MoE GEMM, dense GEMM and decode MoE kernels keep their numbers after the dequantizer move.

---

### Task 1: one register dequantizer for the GSQ-RCO and BF16 rows

**Files:**
- Modify: `metal/qwen4.metal` (new `qwen4_gsq_deq8` above `qwen4_mm_stage8`; the stage8 GSQ-RCO branches call it)

**Interfaces:**
- Produces: `static inline void qwen4_gsq_deq8(device const char *row, uint b, uint q, uint type, thread float *v)`.
  It writes values `[b*32 + q*8, +8)` of the row for types 13, 14, 17, 18, 20, 21, 22, 23, 42 and 30 (BF16).

- [ ] **Step 1: Move the nine GSQ-RCO branches** of `qwen4_mm_stage8` into `qwen4_gsq_deq8`.
  - Write `v[i] = <float expr>` where the branches wrote `dst[i] = (D)(<float expr>)`.
  - Add BF16: `device const ushort *w = (device const ushort *)row + b * 32 + q * 8; v[i] = as_type<float>((uint)w[i] << 16);`.
  - In `qwen4_mm_stage8`, the moved branches become:

```metal
    if (type == 13 || type == 14 || type == 17 || type == 18 || (type >= 20 && type <= 23) || type == 42) {
        float v[8];
        qwen4_gsq_deq8(row, b, q, type, v);
        for (uint i = 0; i < 8; i++) dst[i] = (D)v[i];
        return;
    }
```

- [ ] **Step 2: Run the kernel tests** (a refactor under green tests: the MoE GEMM and dense GEMM tests cover every moved branch)

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK`.

- [ ] **Step 3: Commit** `metal: one register dequantizer for the GSQ-RCO rows`

### Task 2: `kernel_qwen4_gsq_mv` for decode, verify and small batches

**Files:**
- Modify: `metal/qwen4.metal` (kernel + 4 instances), `ds4_metal.m` (enum, names, routing in `ds4_gpu_qwen4_multi_gemv_tensor`, test hook), `ds4_gpu.h` (hook declaration)
- Test: `tests/ds4_test.c` (`test_metal_qwen4_gsq_mv`, called after `test_metal_qwen4_quant_gemv`)

**Interfaces:**
- Consumes: `qwen4_gsq_deq8`, `qwen4_mm_gsq_block`.
- Produces: `int ds4_gpu_qwen4_gsq_mv_selected(uint32_t n_tokens, uint32_t in_dim, uint32_t n_out, const uint32_t *types)`.
  It returns 1 when `ds4_gpu_qwen4_multi_gemv_tensor` will run `kernel_qwen4_gsq_mv`.

- [ ] **Step 1: Write the failing test**

```c
/* kernel_qwen4_gsq_mv: dense GSQ-RCO and BF16 rows at 1-8 tokens dequantize each chunk once.
 * Against the CPU rows, and column for column bit-equal to the one-token call. */
static void test_metal_qwen4_gsq_mv_case(uint32_t type, const uint8_t *rows_bytes, uint64_t row_bytes,
                                         uint32_t in_dim, uint32_t rows) {
    const uint64_t page = (uint64_t)getpagesize(), alloc = test_round_up_u64(row_bytes * rows, page);
    void *wraw = NULL;
    TEST_ASSERT(posix_memalign(&wraw, (size_t)page, (size_t)alloc) == 0);
    if (!wraw) return;
    memcpy(wraw, rows_bytes, (size_t)(row_bytes * rows));
    const uint32_t nmax = 6u;
    float *xh = malloc((size_t)nmax * in_dim * sizeof(float));
    float *one = malloc((size_t)nmax * rows * sizeof(float));
    float *many = malloc((size_t)nmax * rows * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)nmax * in_dim * sizeof(float));
    ds4_gpu_tensor *x1 = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *o = ds4_gpu_tensor_alloc((uint64_t)nmax * rows * sizeof(float));
    TEST_ASSERT(xh && one && many && x && x1 && o);
    if (xh && one && many && x && x1 && o) {
        for (uint32_t t = 0; t < nmax; t++) test_quant_x(xh + (uint64_t)t * in_dim, in_dim, t + 17u);
        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, xh, (uint64_t)nmax * in_dim * sizeof(float)) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(wraw, alloc) != 0);
        ds4_gpu_tensor *outs[1] = { o };
        const uint64_t offs[1] = { 0 };
        const uint32_t types[1] = { type }, out_rows[1] = { rows };
        for (uint32_t t = 0; t < nmax; t++) {           /* one-token decode rows */
            TEST_ASSERT(ds4_gpu_tensor_write(x1, 0, xh + (uint64_t)t * in_dim, (uint64_t)in_dim * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_qwen4_multi_gemv_tensor(x1, 1, in_dim, 1, outs, wraw, alloc, offs, types, out_rows) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(o, 0, one + (uint64_t)t * rows, (uint64_t)rows * sizeof(float)) != 0);
        }
        uint32_t bad = 0;
        for (uint32_t t = 0; t < nmax; t++)
            for (uint32_t r = 0; r < rows; r++) {
                double mag = 0.0;
                const double ref = test_quant_row_ref(type, (const uint8_t *)wraw + r * row_bytes, in_dim,
                                                      xh + (uint64_t)t * in_dim, &mag);
                if (fabs((double)one[(uint64_t)t * rows + r] - ref) > 1e-5 * mag + 1e-6) bad++;
            }
        const uint32_t ns[3] = { 2u, 3u, 6u };
        uint32_t differ = 0;
        for (int k = 0; k < 3; k++) {
            TEST_ASSERT(ds4_gpu_qwen4_multi_gemv_tensor(x, ns[k], in_dim, 1, outs, wraw, alloc, offs, types, out_rows) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(o, 0, many, (uint64_t)ns[k] * rows * sizeof(float)) != 0);
            if (memcmp(many, one, (size_t)ns[k] * rows * sizeof(float)) != 0) differ++;
        }
        if (bad || differ) fprintf(stderr, "ds4-test: GSQ mv type %u dim %u: %u rows off, %u batch sizes differ from one-token rows\n",
                                   type, in_dim, bad, differ);
        TEST_ASSERT(bad == 0 && differ == 0);
    }
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(x1);
    ds4_gpu_tensor_free(o);
    free(xh);
    free(one);
    free(many);
    free(wraw);
}

static void test_metal_qwen4_gsq_mv(void) {
    const uint32_t gsq[2] = { 21u, 14u }, prod[2] = { 8u, 12u };
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(1, 2560, 1, gsq) == 1);
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(3, 2560, 2, gsq) == 1);
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(8, 2560, 1, gsq) == 1);
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(9, 2560, 1, gsq) == 0);      /* the dense GEMM's batch sizes */
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(3, 2560, 1, prod) == 0);     /* Q8_0 keeps its kernels */
    const uint32_t bf[1] = { 30u };
    TEST_ASSERT(ds4_gpu_qwen4_gsq_mv_selected(3, 2568, 1, bf) == 0);       /* not whole 16-value chunks */
    const uint32_t rows = 37u;
    for (size_t i = 0; i < sizeof(quant_fixtures) / sizeof(quant_fixtures[0]); i++) {
        const ds4_quant_fixture *f = &quant_fixtures[i];
        if (f->type == 16) continue;
        const uint32_t in_dim = (f->type == 20 || f->type == 42) ? 640u : 2560u;
        const uint64_t row_bytes = test_quant_row_bytes(f->type, in_dim);
        uint8_t *buf = malloc((size_t)(row_bytes * rows));
        TEST_ASSERT(buf != NULL);
        if (!buf) return;
        test_quant_fill_rows(buf, f, in_dim, rows);
        test_metal_qwen4_gsq_mv_case(f->type, buf, row_bytes, in_dim, rows);
        free(buf);
    }
    const uint32_t in_dim = 2560u;                     /* BF16 rows */
    uint16_t *bfr = malloc((size_t)in_dim * rows * sizeof(uint16_t));
    float *v = malloc((size_t)in_dim * sizeof(float));
    TEST_ASSERT(bfr && v);
    if (bfr && v) {
        for (uint32_t r = 0; r < rows; r++) {
            test_quant_x(v, in_dim, r + 211u);
            for (uint32_t k = 0; k < in_dim; k++) {
                uint32_t u;
                memcpy(&u, &v[k], sizeof(u));
                bfr[(uint64_t)r * in_dim + k] = (uint16_t)(u >> 16);
            }
        }
        test_metal_qwen4_gsq_mv_case(30u, (const uint8_t *)bfr, (uint64_t)in_dim * 2u, in_dim, rows);
    }
    free(bfr);
    free(v);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `make ds4_test`
Expected: compile error, `ds4_gpu_qwen4_gsq_mv_selected` undeclared.

- [ ] **Step 3: Implement**

`metal/qwen4.metal`, after `kernel_qwen4_multi_gemv`:

```metal
/* Dense GSQ-RCO and BF16 rows over R1 tokens (decode, MTP verify, batches up
 * to 8): each lane dequantizes a 16-value chunk once into registers and dots
 * it with the R1 activation columns; one simdgroup per row.  A column's sum
 * does not depend on R1, so a verify row matches the one-token row. */
template <uint R1>
kernel void kernel_qwen4_gsq_mv(
        constant ds4_metal_args_qwen4_gemv & args,
        device const float *x,          /* [T][in_dim] */
        device const char  *w0,
        device const char  *w1,
        device const char  *w2,
        device const char  *w3,
        device float       *o0,         /* [T][out_rows[i]] */
        device float       *o1,
        device float       *o2,
        device float       *o3,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint total = args.out_rows[0] + args.out_rows[1] + args.out_rows[2] + args.out_rows[3];
    const uint r = tgpig.x * 4 + (uint)sgitg;
    if (r >= total) return;
    uint i = 0, local = r;
    while (i + 1 < args.n_out && local >= args.out_rows[i]) { local -= args.out_rows[i]; i++; }
    device const char *row = (i == 0 ? w0 : i == 1 ? w1 : i == 2 ? w2 : w3) + (uint64_t)local * args.row_bytes[i];
    device float *o = i == 0 ? o0 : i == 1 ? o1 : i == 2 ? o2 : o3;
    const uint type = args.types[i];
    const uint t0 = tgpig.y * R1;
    const uint nt = min(R1, args.n_tokens - t0);
    float acc[R1];
    for (uint t = 0; t < R1; t++) acc[t] = 0.0f;
    for (uint c = tiisg; c < args.in_dim / 16u; c += 32u) {
        float wv[16];
        qwen4_gsq_deq8(row, c / 2u, (c % 2u) * 2u, type, wv);
        qwen4_gsq_deq8(row, c / 2u, (c % 2u) * 2u + 1u, type, wv + 8);
        for (uint t = 0; t < R1; t++) {
            if (t < nt) {
                device const float4 *xv = (device const float4 *)(x + (uint64_t)(t0 + t) * args.in_dim + c * 16u);
                float s = 0.0f;
                for (uint j = 0; j < 4u; j++) {
                    const float4 y = xv[j];
                    s += wv[4u * j] * y.x + wv[4u * j + 1u] * y.y + wv[4u * j + 2u] * y.z + wv[4u * j + 3u] * y.w;
                }
                acc[t] += s;
            }
        }
    }
    for (uint t = 0; t < R1; t++) {
        const float v = simd_sum(acc[t]);
        if (tiisg == 0 && t < nt) o[(uint64_t)(t0 + t) * args.out_rows[i] + local] = v;
    }
}

#define QWEN4_GSQ_MV_SIG constant ds4_metal_args_qwen4_gemv &, device const float *, device const char *, \
    device const char *, device const char *, device const char *, device float *, device float *, \
    device float *, device float *, uint3, ushort, ushort
template [[host_name("kernel_qwen4_gsq_mv_r1")]] kernel void kernel_qwen4_gsq_mv<1>(QWEN4_GSQ_MV_SIG);
template [[host_name("kernel_qwen4_gsq_mv_r2")]] kernel void kernel_qwen4_gsq_mv<2>(QWEN4_GSQ_MV_SIG);
template [[host_name("kernel_qwen4_gsq_mv_r3")]] kernel void kernel_qwen4_gsq_mv<3>(QWEN4_GSQ_MV_SIG);
template [[host_name("kernel_qwen4_gsq_mv_r4")]] kernel void kernel_qwen4_gsq_mv<4>(QWEN4_GSQ_MV_SIG);
#undef QWEN4_GSQ_MV_SIG
```

`ds4_metal.m`:
- Add `QWEN4_K_GSQ_MV_R1, QWEN4_K_GSQ_MV_R2, QWEN4_K_GSQ_MV_R3, QWEN4_K_GSQ_MV_R4` after `QWEN4_K_MULTI_GEMV` in
  the enum, and the four names after `"kernel_qwen4_multi_gemv"` in the name table.
- `qwen4_mm_gsq_block` moves above `ds4_gpu_qwen4_multi_gemv_tensor` if it is defined later.
- Add the selector and its test hook:

```c
/* Dense GSQ-RCO and BF16 rows at 1-8 tokens take kernel_qwen4_gsq_mv (one
 * dequant per chunk for every token); larger batches take the dense GEMM.
 * DS4_QWEN4_GSQ_MV_LEGACY=1 keeps the per-token multi-row gemv. */
static bool qwen4_gsq_mv_ok(uint32_t n_tokens, uint32_t in_dim, uint32_t n_out, const uint32_t *types) {
    if (n_tokens == 0 || n_tokens > 8u || (in_dim % 16u) != 0 || getenv("DS4_QWEN4_GSQ_MV_LEGACY")) return false;
    for (uint32_t i = 0; i < n_out; i++)
        if (!qwen4_mm_gsq_block(types[i]) && types[i] != 30u) return false;
    return true;
}

int ds4_gpu_qwen4_gsq_mv_selected(uint32_t n_tokens, uint32_t in_dim, uint32_t n_out, const uint32_t *types) {
    return qwen4_gsq_mv_ok(n_tokens, in_dim, n_out, types) ? 1 : 0;
}
```

- In `ds4_gpu_qwen4_multi_gemv_tensor`, before the final dispatch:

```c
    if (qwen4_gsq_mv_ok(n_tokens, in_dim, n_out, types)) {
        const uint32_t r1 = n_tokens < 4u ? n_tokens : 4u;
        return qwen4_dispatch(QWEN4_K_GSQ_MV_R1 + (int)r1 - 1, &args, sizeof(args), b, 9,
                              MTLSizeMake((total + 3) / 4, (n_tokens + r1 - 1) / r1, 1), MTLSizeMake(128, 1, 1), 0);
    }
```

`ds4_gpu.h`, after `ds4_gpu_qwen4_multi_gemv_tensor`:
`int ds4_gpu_qwen4_gsq_mv_selected(uint32_t n_tokens, uint32_t in_dim, uint32_t n_out, const uint32_t *types);`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK`. The MoE decode tests, `test_metal_qwen4_quant_gemv` and the new test all pass.

- [ ] **Step 5: Throughput check** (throwaway microbenchmark, no model): Q6_K 2560 x 24576 at 1/2/3/8 tokens, plus IQ3_S and IQ4_XS.
  Expected: at or below the prototype's 0.202/0.220/0.272 ms for Q6_K, and no type slower than the old kernel at 1 token.

- [ ] **Step 6: Commit** `metal: GSQ-RCO and BF16 rows dequantize once per decode or verify step`

### Task 3: model-level checks (next GPU window)

1. **Correctness:** `DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_CHUNK=1 ./ds4 --first-token-test` on ISTA, against SP3's chunk-1 numbers
   (3.098, 30/31 on the 31-token prompt). Pass: worst max|diff| ≤ 6.697 (Ivan's chunk-1 gap) and top-1 ≥ 30/31.
2. **ISTA, the SP3 prompts:** `--mtp`, coherent, with MTP acceptance and generation t/s.
   - Also run `DS4_QWEN4_MTP_DRAFT_ROWS=65536` and record its acceptance and t/s.
   - Run without `--mtp`.
   - `ds4-bench` decode, 3 rounds.
3. **PROD:** identity 3/3 and `ds4-bench` A/B against the pre-plan build, at least 0.98x.
4. **Results file:** `speed-bench/nextgen-eval/results/2026-10-01-sp5-gsq-verify.md`.
