# Ornith M6: Prefill attention on the M5 neural accelerators — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach the live oMLX's prefill speed at 32K and 128K by running Ornith's prefill attention on the M5 neural accelerators (`matmul2d`), with the M5 simdgroup flash as the fallback and every exactness gate intact.

**Architecture:** Ornith-only and additive. A query-pack kernel (`kernel_qwen35_attn_qpack`) and a fused accelerator flash kernel (`kernel_qwen35_attn_flash_nax`, 8 tokens x 8 heads per threadgroup, K/V read straight from the cache, S and O on `matmul2d` fragments, online softmax in registers) live in `metal/qwen35.metal` inside `#ifdef DS4_METAL_HAS_TENSOR`; `ds4_gpu_qwen35_attn_flash_nax_tensor` in `ds4_metal.m` mirrors the M5 flash wrapper (same key split, same merge); `ds4_qwen35moe.inc` dispatches it behind `DS4_QWEN35_ATTN_NAX` when the tensor API is available. The model-free benchmark decides the stop rule and the short-chunk crossover; the M4 harness decides adoption and the final gates.

**Tech Stack:** C, Objective-C (`ds4_metal.m`), Metal Shading Language with Metal 4 `MetalPerformancePrimitives` (`matmul2d`, cooperative tensors), Python 3 (M4 harness, gate scripts).

**Spec:** `docs/superpowers/specs/2026-09-27-ornith-m6-nax-attention-design.md` (parent: `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`; previous milestone: `docs/superpowers/specs/2026-09-27-ornith-m5-attention-design.md`).

## Global Constraints

- The Qwen3.8 production path stays byte-identical and as fast as today. Every commit that touches a shared file (`ds4.c` outside Ornith-only functions, `ds4_metal.m`, `ds4_gpu.h`, `metal/*.metal`, `ds4_server.c`, `ds4_agent.c`, `ds4_kvstore.c`) passes `make test-qwen4-kernels test-qwen4-q2` and `speed-bench/qwen-regression/run.sh fast`; the branch passes `run.sh full` before merge.
- Metal only for Ornith. Ornith knobs use the `DS4_QWEN35_*` prefix; Ornith code reads no `DS4_QWEN4_*` knob.
- Kernel changes are additive: new kernels in `metal/qwen35.metal`; `metal/qwen4.metal` is never edited. Accelerator code sits inside `#ifdef DS4_METAL_HAS_TENSOR`, so the library still compiles with `DS4_METAL_DISABLE_METAL4=1`.
- `--mtp` greedy output stays byte-identical to plain decoding (M2). M6 does not touch decode (`kernel_qwen35_attn_decode3`).
- Code, comments, docs and commit messages in English. Model files never go into git. The 23G GGUF: `DS4_ORNITH_MODEL=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf`; the 25G tier sits next to it.
- One model process at a time on this 64 GB machine. Never `kill -9` a Metal process; scripts stop processes with SIGTERM and a bounded wait.
- GPU windows (model runs, speed measurements, the benchmark's timed runs) pause the live stack and restore it afterwards (the controller's `gpuwin.sh` pattern from M4/M5); implementers build and run model-free kernel tests only.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2
  ```

Plan-specific constraints:
- Branch `feature/ornith-m6` (from develop `696d328`, holds the spec commit `0ba6632`); merging or pushing needs the user's explicit OK.
- Adoption rule (M4): `DS4_QWEN35_ATTN_NAX` starts off and becomes the default only after gate 1 passes at prefill chunks 2048/64/65, the MTP identity tests pass (`make test-qwen35-graph test-qwen35-mtp`, `tests/ornith/test_mtp_cli.py`), the fallback identity check passes and a `m4_ab.py --mode lever` A/B shows a gain; otherwise it stays opt-in and the receipt says why.
- Existing unit tests are never edited; new tests are additions.
- Reference designs (read, do not copy wholesale): MLX's fused accelerator attention, MIT licence, at `~/.local/omlx-venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` and `.../steel/attn/nax.h` (fragment layout, `mma`); the repo's accelerator MoE kernels `kernel_qwen35_moe_mm_mid_q5k_nax_t` in `metal/qwen35.metal`.
- **Stop rule (Task 1 Step 8):** the unsplit accelerator kernel must be at least 2.5x faster than `attn_flash_tok2` at both pos 30720 and 122880 (T = 2048). A1 missing -> Task 2 (A2). A2 missing too -> stop and report to the user; the unfused approach is their decision.

## Review Focus

1. **Keys past the fill in the last 32-key block.** The kernel reads K/V in 32-key blocks straight from the cache; keys at or past pos0 + T must load as zero, because a NaN or stale V row times a zero probability still poisons O. Pinned by `test_attn_flash_nax`, which fills every K/V row past the fill with NaN (cache 40 rows longer than the fill) at T = 9, 65, 200, 64, 2048 (Task 1).
2. **Token counts that are not a multiple of 8.** The last 8-token block has padding rows (zero queries); they must never be written. Pinned by the guard token after the output, which must keep its sentinel, at T = 9 and 65 (Task 1).
3. **Key split with rows that have no keys in a split.** A causal row whose diagonal lies before a split's start must write a neutral partial (l = 0) that merge3 ignores; m must be written in natural-log units. Pinned by the forced-split cases (4096, 64), (8000, 200) and (1000, 1100) against the host double reference (Task 3).
4. **Long-context drift of the accelerator accumulate path on M5.** Pinned by (30720, 2048) against the M5 simdgroup flash within 2e-3 (Task 1), then gate 1 at chunks 2048/64/65 and gate 2 (Tasks 4-5).
5. **Fallback path unchanged.** With `DS4_METAL_DISABLE_METAL4=1` (and with the knob off before the flip) the gate-1 dumps must equal develop `696d328`'s byte for byte; the accelerator test cases print "skipped" without the tensor API while the M5 flash cases still run. Pinned in Task 4 Step 5 and by `make test-qwen35-kernels` under both Metal settings (Tasks 1, 3).

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `metal/qwen35.metal` | Modify (append) | accelerator fragment helpers, `kernel_qwen35_attn_qpack`, `kernel_qwen35_attn_flash_nax` (+ split instance) |
| `ds4_metal.m`, `ds4_gpu.h` | Modify | enum/name entries, `ds4_gpu_qwen35_attn_flash_nax_tensor`, shared `qwen35_attn_flash_merge3`, `ds4_gpu_qwen35_attn_flash_part_floats` bound for 8-token blocks |
| `ds4_qwen35moe.inc` | Modify | `DS4_QWEN35_ATTN_NAX` knob and dispatch in `qwen35_graph_attend`, scratch comment |
| `ds4.c` | Modify (Ornith branch) | memory estimate: packed-query scratch |
| `tests/test_qwen35_kernels.c` | Modify (additions) | `test_attn_flash_nax` and its cases |
| `tests/bench_qwen35_attn.c` | Modify | accelerator rows, short-chunk sweep |
| `speed-bench/ornith/m6/` | Create | `BENCH.md`, `LEVERS.md`, `speed/GATE3.md`, `quality/GATE2.md`, `QWEN_GATE.md`, `REPORT.md` |
| `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` | Modify | §4 attention text and §7.3 knob list |

---

### Task 1: Accelerator flash kernel A1 (no key split), wrapper, tests, benchmark rows

**Files:**
- Modify: `metal/qwen35.metal` (append), `ds4_metal.m`, `ds4_gpu.h`, `tests/test_qwen35_kernels.c` (additions), `tests/bench_qwen35_attn.c`

**Interfaces:**
- Consumes: `qwen4_nax_scratch` (ds4_metal.m, grows a scratch slot), `ds4_gpu_mpp_available()` (tensor API on and not `--quality`), `qwen4_bind_tensor` / `qwen4_dispatch`, `g_qwen35_attn_flash_last_splits`, `QWEN35_ATTN_GROUP` (8) and `qwen4_sigmoid` (metal), `ds4_gpu_qwen35_attn_flash_tensor` (M5, the reference in tests and bench).
- Produces:
  - `int ds4_gpu_qwen35_attn_flash_nax_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part, uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, float scale);` — same contract as the M5 flash wrapper; in this task `part` must be NULL (Task 3 adds the key split).
  - Kernels `kernel_qwen35_attn_qpack`, `kernel_qwen35_attn_flash_nax` (the `<false>` instance of template `kernel_qwen35_attn_flash_nax<bool SPLIT>`; Task 3 adds `<true>`).
  - Enum entries `QWEN4_K_QWEN35_ATTN_QPACK`, `QWEN4_K_QWEN35_ATTN_FLASH_NAX` (appended before `QWEN4_K_COUNT`, names appended at the end of `qwen4_kernel_names`).

- [ ] **Step 1: Write the failing kernel test**

In `tests/test_qwen35_kernels.c`, append before `int main`:

```c
/* flash_nax (M6): causal prefill attention on the accelerator path.  Every
 * K/V row past the fill is NaN (the cache is 40 rows longer than the fill,
 * so a 32-key block past it stays inside the buffer): a kernel that
 * multiplies such a row, even by a zero probability, turns the output
 * non-finite.  One guard token after the output must keep its sentinel (no
 * write past T).  Checked against a host double reference (host_ref) and
 * always against the M5 simdgroup flash.  Returns the Ks the accelerator
 * call used (0 when the tensor API is unavailable). */
static uint32_t test_attn_flash_nax(uint32_t pos0, uint32_t T, int use_part, int host_ref) {
    const uint32_t H = 16, Hkv = 2, D = 256, fill = pos0 + T, cap = fill + 40u;
    const uint64_t n_out = (uint64_t)T * H * D, row = (uint64_t)H * D, n_kv = (uint64_t)cap * Hkv * D;
    const float scale = 1.0f / sqrtf((float)D);
    if (!ds4_gpu_tensor_api_available()) {
        printf("  attn flash nax pos0=%u T=%u skipped (tensor API unavailable)\n", pos0, T);
        return 0;
    }
    float *qv = rand_vec(n_out, 3.0f), *gv = rand_vec(n_out, 1.0f);
    uint16_t *kh = malloc(n_kv * sizeof(uint16_t)), *vh = malloc(n_kv * sizeof(uint16_t));
    for (uint64_t i = 0; i < n_kv; i++) {
        const int past = i >= (uint64_t)fill * Hkv * D;
        kh[i] = past ? 0x7e00u : f32_to_f16(frand());   /* half NaN past the fill */
        vh[i] = past ? 0x7e00u : f32_to_f16(frand());
    }
    ds4_gpu_tensor *q = upload(qv, n_out), *gt = upload(gv, n_out);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc(n_kv * sizeof(uint16_t));
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc(n_kv * sizeof(uint16_t));
    require_ok(kc && vc && ds4_gpu_tensor_write(kc, 0, kh, n_kv * sizeof(uint16_t)) &&
               ds4_gpu_tensor_write(vc, 0, vh, n_kv * sizeof(uint16_t)), "nax cache upload");
    ds4_gpu_tensor *on = upload(NULL, n_out + row), *of = upload(NULL, n_out);
    ds4_gpu_tensor *part = use_part ? upload(NULL, ds4_gpu_qwen35_attn_flash_part_floats(T, H, D)) : NULL;
    require_ok(ds4_gpu_tensor_fill_f32(on, -1234.5f, n_out + row), "nax sentinel");
    require_ok(ds4_gpu_qwen35_attn_flash_nax_tensor(on, q, gt, kc, vc, part, T, H, Hkv, D, pos0, scale),
               "flash nax call");
    const uint32_t splits = ds4_gpu_qwen35_attn_flash_last_splits();
    require_ok(ds4_gpu_qwen35_attn_flash_tensor(of, q, gt, kc, vc, NULL, T, H, Hkv, D, pos0, scale),
               "flash reference");
    float *gotn = download(on, n_out + row), *gotf = download(of, n_out);

    uint64_t nonfinite = 0, guard_bad = 0;
    double worst_f = 0.0, scf = 1e-6, worst_ref = 0.0, sc = 1e-6;
    for (uint64_t o = 0; o < n_out; o++) {
        if (!isfinite(gotn[o])) nonfinite++;
        if (fabs((double)gotf[o]) > scf) scf = fabs((double)gotf[o]);
        if (fabs((double)gotn[o] - (double)gotf[o]) > worst_f) worst_f = fabs((double)gotn[o] - (double)gotf[o]);
    }
    for (uint64_t o = n_out; o < n_out + row; o++) guard_bad += gotn[o] != -1234.5f;
    if (host_ref) {
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t n_keys = pos0 + t + 1u;
            for (uint32_t h = 0; h < H; h++) {
                const uint32_t kvh = h / (H / Hkv);
                double acc[256], m = -1e300, l = 0.0;
                for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
                for (uint32_t idx = 0; idx < n_keys; idx++) {
                    double s = 0.0;
                    for (uint32_t d = 0; d < D; d++)
                        s += (double)qv[((uint64_t)t * H + h) * D + d] * scale *
                             (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                    const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                    l = l * corr + w;
                    for (uint32_t d = 0; d < D; d++)
                        acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                    m = mn;
                }
                for (uint32_t d = 0; d < D; d++) {
                    const uint64_t o = ((uint64_t)t * H + h) * D + d;
                    const double refv = acc[d] / l * sigmoid_d((double)gv[o]);
                    if (fabs(refv) > sc) sc = fabs(refv);
                    if (fabs((double)gotn[o] - refv) > worst_ref) worst_ref = fabs((double)gotn[o] - refv);
                }
            }
        }
    }
    printf("  attn flash nax pos0=%u T=%u splits=%u: vs M5 flash max|d| %.3e (rel %.3e)", pos0, T, splits,
           worst_f, worst_f / scf);
    if (host_ref) printf(", vs host double ref rel %.3e", worst_ref / sc);
    printf("\n");
    require_ok(nonfinite == 0, "flash nax output finite (no K/V row past the fill multiplied)");
    require_ok(guard_bad == 0, "flash nax writes nothing past token T");
    require_ok(worst_f <= 2e-3 * scf, "flash nax within 2e-3 of the M5 simdgroup flash");
    if (host_ref) require_ok(worst_ref <= 1e-3 * sc, "flash nax within 1e-3 of the host double reference");
    free(qv); free(gv); free(kh); free(vh); free(gotn); free(gotf);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(gt); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc);
    ds4_gpu_tensor_free(on); ds4_gpu_tensor_free(of);
    if (part) ds4_gpu_tensor_free(part);
    return splits;
}
```

In `main`, insert before `printf("qwen35 kernels: ok\n");`:

```c
    printf("qwen35 attention flash on the neural accelerators (M6)\n");
    test_attn_flash_nax(0u, 9u, 0, 1);
    test_attn_flash_nax(0u, 65u, 0, 1);
    test_attn_flash_nax(37u, 65u, 0, 1);
    test_attn_flash_nax(37u, 200u, 0, 1);
    test_attn_flash_nax(4096u, 64u, 0, 1);
    test_attn_flash_nax(30720u, 2048u, 0, 0);   /* long context: accelerator drift vs the simdgroup flash */
```

- [ ] **Step 2: Build and confirm it fails**

Run: `make tests/test_qwen35_kernels`
Expected: link error naming `ds4_gpu_qwen35_attn_flash_nax_tensor`.

- [ ] **Step 3: Kernels**

Append to `metal/qwen35.metal`:

```metal
/* --- M6: prefill attention on the M5 neural accelerators ------------------ */

#ifdef DS4_METAL_HAS_TENSOR

/* A 16x16 accelerator fragment held as 8 values per lane: rows c.y and
 * c.y + 8, four consecutive columns from c.x (qwen35_nax_coord).  This is
 * the register layout of MLX's BaseNAXFrag (mlx/backend/metal/kernels/steel/
 * attn/nax.h, MIT licence, Copyright (c) 2025 Apple Inc.), which matmul2d's
 * cooperative tensors use for a 16x32x16 simdgroup op.  The four lanes that
 * share a row differ in lane bits 0 and 3, so row reductions shuffle over
 * xor 1 and xor 8. */
typedef vec<float, 8> qwen35_nax_f;
typedef vec<half, 8>  qwen35_nax_h;

/* relaxed_precision of every accelerator product below; false matches the
 * Ornith MoE accelerator kernels (Task 1 Step 8 measured both settings) */
#define QWEN35_NAX_RELAXED false

static inline short2 qwen35_nax_coord(ushort lane) {
    const short qid = (short)(lane >> 2);
    const short fm = (qid & 4) | ((short)(lane >> 1) & 3);
    const short fn = ((qid & 2) | (short)(lane & 1)) * 4;
    return short2(fn, fm);   /* (column, row) of this lane's first value */
}

static inline qwen35_nax_f qwen35_nax_zero(void) {
    qwen35_nax_f f;
#pragma unroll
    for (short e = 0; e < 8; e++) f[e] = 0.0f;
    return f;
}

/* 16x16 half fragment at p, row stride str */
static inline qwen35_nax_h qwen35_nax_load(device const half *p, uint str, short2 c) {
    qwen35_nax_h f;
#pragma unroll
    for (short i = 0; i < 2; i++) {
        const half4 v = *(device const half4 *)(p + (uint64_t)(c.y + i * 8) * str + c.x);
        f[i * 4 + 0] = v.x; f[i * 4 + 1] = v.y; f[i * 4 + 2] = v.z; f[i * 4 + 3] = v.w;
    }
    return f;
}

/* same, rows at or past lim load as zero (lim <= 0: all zero) */
static inline qwen35_nax_h qwen35_nax_load_rows(device const half *p, uint str, short2 c, int lim) {
    qwen35_nax_h f;
#pragma unroll
    for (short i = 0; i < 2; i++) {
        const int r = c.y + i * 8;
        const half4 v = r < lim ? *(device const half4 *)(p + (uint64_t)r * str + c.x) : half4(0.0h);
        f[i * 4 + 0] = v.x; f[i * 4 + 1] = v.y; f[i * 4 + 2] = v.z; f[i * 4 + 3] = v.w;
    }
    return f;
}

/* c0|c1 (16x32, F32) += a (16x16) x b0|b1 (the right operand's two 16-column
 * halves).  TR: the right fragments hold the operand transposed (rows = its
 * columns), as K does for S = Q K^T. */
template <bool TR>
static inline void qwen35_nax_mma(thread qwen35_nax_f &c0, thread qwen35_nax_f &c1, qwen35_nax_h a,
                                  qwen35_nax_h b0, qwen35_nax_h b1) {
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, TR, QWEN35_NAX_RELAXED,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
#pragma unroll
    for (short i = 0; i < 8; i++) {
        ca[i] = a[i];
        cb[i] = b0[i]; cb[8 + i] = b1[i];
        cc[i] = c0[i]; cc[8 + i] = c1[i];
    }
    op.run(ca, cb, cc);
#pragma unroll
    for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

struct ds4_metal_args_qwen35_attn_qpack {
    uint32_t n_tokens, n_tokens_pad, n_head, n_head_kv;
    float scale2;                      /* scale * log2(e): scores come out in log2 units */
};

/* q [T][H*D] F32 -> qh [Hkv][Tpad][8][256] F16 times scale2, tokens
 * T..Tpad-1 zero: the 8 * Tpad query rows of one KV head become contiguous
 * (row stride D), the shape the accelerator fragments load directly.  One
 * thread per 4 values. */
kernel void kernel_qwen35_attn_qpack(
        constant ds4_metal_args_qwen35_attn_qpack & args,
        device const float *q,
        device half        *qh,
        uint gid [[thread_position_in_grid]]) {
    constexpr uint D = 256u, G = QWEN35_ATTN_GROUP;
    const uint per_kvh = args.n_tokens_pad * G * D;
    const uint e = gid * 4u;
    if (e >= args.n_head_kv * per_kvh) return;
    const uint kvh = e / per_kvh, rem = e % per_kvh;
    const uint t = rem / (G * D), g = (rem / D) % G, d = rem % D;
    half4 v = half4(0.0h);
    if (t < args.n_tokens) {
        const float4 x = *(device const float4 *)(q + ((uint64_t)t * args.n_head + kvh * G + g) * D + d);
        v = half4(x * args.scale2);
    }
    *(device half4 *)(qh + e) = v;
}

struct ds4_metal_args_qwen35_attn_flash_nax {
    uint32_t n_tokens, n_tokens_pad, n_head, n_head_kv, pos0;
    uint32_t kps, n_splits;            /* SPLIT: keys per split and split count */
};

/* M6 flash prefill on the accelerators (T > 8, F16 K/V, head_dim 256, group
 * 8).  Grid (Hkv, Tpad/8, SPLIT ? Ks : 1), 4 simdgroups: simdgroup sg owns
 * the 16 query rows of tokens t0 = blk*8 + 2*sg and t0 + 1 (row i*8 + g =
 * token t0 + i, query head kvh*8 + g), so every K/V fragment it loads serves
 * all 8 heads.  K/V are read straight from the cache in 32-key blocks; S =
 * Q K^T and O += P V run on matmul2d (qwen35_nax_mma), P as half; the online
 * softmax runs in log2 units (scale * log2e folded into qh).  No threadgroup
 * memory and no barriers: each simdgroup walks its own key range [lo, hi)
 * (causal: token t sees keys <= pos0 + t), and a lane holds rows c.y and
 * c.y + 8, i.e. head c.y of both tokens.  Keys at or past the fill
 * (pos0 + T) load as zero, so V never multiplies memory past the cache fill.
 * !SPLIT writes the gated output like kernel_qwen35_attn_flash; SPLIT writes
 * kernel_qwen35_attn_merge3 partials ([T][Hkv][Ks][8][2+D], m in natural-log
 * units, a row without keys in its split writes the neutral l = 0). */
template <bool SPLIT>
kernel void kernel_qwen35_attn_flash_nax(
        constant ds4_metal_args_qwen35_attn_flash_nax & args,
        device const half  *qh,       /* [Hkv][Tpad][8][256] */
        device const float *gate,     /* [n_tokens][H*D] */
        device const half  *k_cache,  /* [cap][Hkv*D] */
        device const half  *v_cache,  /* [cap][Hkv*D] */
        device float       *out,      /* !SPLIT: [n_tokens][H*D]; SPLIT: merge3 partials */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint D = 256u, G = QWEN35_ATTN_GROUP, KB = 32u;
    constexpr float NEG = -3.0e38f;    /* finite: this file builds under fast math */
    const uint kvh = tgpig.x, split = tgpig.z;
    const uint H = args.n_head, Hkv = args.n_head_kv, T = args.n_tokens;
    const uint t0 = tgpig.y * 8u + sgitg * 2u;
    if (kvh >= Hkv || t0 >= T) return;
    const uint n_fill = args.pos0 + T;
    const uint lim_row[2] = { args.pos0 + t0, args.pos0 + t0 + 1u };   /* row i sees keys <= lim_row[i] */
    uint lo = 0u, hi = min(lim_row[1], n_fill - 1u) + 1u;             /* this simdgroup's keys [lo, hi) */
    if (SPLIT) { lo = split * args.kps; hi = min(hi, lo + args.kps); }

    const short2 c = qwen35_nax_coord(tiisg);
    const uint kstr = Hkv * D;
    device const half *qrow = qh + ((uint64_t)kvh * args.n_tokens_pad + t0) * G * D;
    device const half *kbase = k_cache + (uint64_t)kvh * D;
    device const half *vbase = v_cache + (uint64_t)kvh * D;

    qwen35_nax_f O[16];
#pragma unroll
    for (short j = 0; j < 16; j++) O[j] = qwen35_nax_zero();
    float m[2] = { NEG, NEG }, l[2] = { 0.0f, 0.0f };

    for (uint kb0 = lo; kb0 < hi; kb0 += KB) {
        const bool edge = kb0 + KB > n_fill;           /* the block reaches past the fill */
        const int lim = (int)n_fill - (int)kb0;        /* key rows of this block below the fill */
        device const half *kp = kbase + (uint64_t)kb0 * kstr;
        device const half *vp = vbase + (uint64_t)kb0 * kstr;

        qwen35_nax_f S0 = qwen35_nax_zero(), S1 = qwen35_nax_zero();
#pragma unroll
        for (short id = 0; id < 16; id++) {
            const qwen35_nax_h qa = qwen35_nax_load(qrow + id * 16, D, c);
            const qwen35_nax_h ka = edge ? qwen35_nax_load_rows(kp + id * 16, kstr, c, lim)
                                         : qwen35_nax_load(kp + id * 16, kstr, c);
            const qwen35_nax_h kb = edge ? qwen35_nax_load_rows(kp + 16u * kstr + id * 16, kstr, c, lim - 16)
                                         : qwen35_nax_load(kp + 16u * kstr + id * 16, kstr, c);
            qwen35_nax_mma<true>(S0, S1, qa, ka, kb);
        }

        /* mask keys past [lo, hi) and past each row's diagonal */
        if (kb0 + KB > min(hi, lim_row[0] + 1u)) {
#pragma unroll
            for (short i = 0; i < 2; i++) {
#pragma unroll
                for (short j = 0; j < 4; j++) {
                    const uint a = kb0 + (uint)c.x + (uint)j, b = a + 16u;
                    if (a >= hi || a > lim_row[i]) S0[i * 4 + j] = NEG;
                    if (b >= hi || b > lim_row[i]) S1[i * 4 + j] = NEG;
                }
            }
        }

        /* online softmax, rows c.y (i = 0) and c.y + 8 (i = 1) */
#pragma unroll
        for (short i = 0; i < 2; i++) {
            float mx = NEG;
#pragma unroll
            for (short j = 0; j < 4; j++) mx = max(mx, max(S0[i * 4 + j], S1[i * 4 + j]));
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 8));
            const float mn = max(m[i], mx);
            float s = 0.0f;
#pragma unroll
            for (short j = 0; j < 4; j++) {
                const float pa = S0[i * 4 + j] > NEG ? exp2(S0[i * 4 + j] - mn) : 0.0f;
                const float pb = S1[i * 4 + j] > NEG ? exp2(S1[i * 4 + j] - mn) : 0.0f;
                S0[i * 4 + j] = pa;
                S1[i * 4 + j] = pb;
                s += pa + pb;
            }
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 8);
            const float f = m[i] > NEG ? exp2(m[i] - mn) : 0.0f;
            l[i] = l[i] * f + s;
            m[i] = mn;
#pragma unroll
            for (short j = 0; j < 16; j++) {
#pragma unroll
                for (short jj = 0; jj < 4; jj++) O[j][i * 4 + jj] *= f;
            }
        }

        /* O += P V, 16 keys at a time */
#pragma unroll
        for (short ik = 0; ik < 2; ik++) {
            qwen35_nax_h P;
#pragma unroll
            for (short e = 0; e < 8; e++) P[e] = (half)(ik == 0 ? S0[e] : S1[e]);
            device const half *vr = vp + (uint64_t)(ik * 16) * kstr;
            const int vlim = lim - ik * 16;
#pragma unroll
            for (short id = 0; id < 16; id += 2) {
                const qwen35_nax_h v0 = edge ? qwen35_nax_load_rows(vr + id * 16, kstr, c, vlim)
                                             : qwen35_nax_load(vr + id * 16, kstr, c);
                const qwen35_nax_h v1 = edge ? qwen35_nax_load_rows(vr + id * 16 + 16, kstr, c, vlim)
                                             : qwen35_nax_load(vr + id * 16 + 16, kstr, c);
                qwen35_nax_mma<false>(O[id], O[id + 1], P, v0, v1);
            }
        }
    }

    const uint g = (uint)c.y;
#pragma unroll
    for (short i = 0; i < 2; i++) {
        const uint t = t0 + (uint)i;
        if (t >= T) continue;
        if (!SPLIT) {
            const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
            const uint64_t at = ((uint64_t)t * H + kvh * G + g) * D;
#pragma unroll
            for (short id = 0; id < 16; id++) {
                const uint d = (uint)(id * 16 + c.x);
                const float4 gv = *(device const float4 *)(gate + at + d);
                float4 o;
                o.x = O[id][i * 4 + 0] * inv * qwen4_sigmoid(gv.x);
                o.y = O[id][i * 4 + 1] * inv * qwen4_sigmoid(gv.y);
                o.z = O[id][i * 4 + 2] * inv * qwen4_sigmoid(gv.z);
                o.w = O[id][i * 4 + 3] * inv * qwen4_sigmoid(gv.w);
                *(device float4 *)(out + at + d) = o;
            }
        } else {
            device float *dst = out + (uint64_t)t * args.n_splits * H * (2u + D) +
                                (((uint64_t)kvh * args.n_splits + split) * G + g) * (2u + D);
            if (c.x == 0) { dst[0] = m[i] * 0.6931471805599453f; dst[1] = l[i]; }
#pragma unroll
            for (short id = 0; id < 16; id++) {
#pragma unroll
                for (short jj = 0; jj < 4; jj++) dst[2u + (uint)(id * 16 + c.x + jj)] = O[id][i * 4 + jj];
            }
        }
    }
}
template [[host_name("kernel_qwen35_attn_flash_nax")]]
kernel void kernel_qwen35_attn_flash_nax<false>(
        constant ds4_metal_args_qwen35_attn_flash_nax &, device const half *, device const float *,
        device const half *, device const half *, device float *, uint3, ushort, ushort);

#endif /* DS4_METAL_HAS_TENSOR */
```

- [ ] **Step 4: Host wrapper**

In `ds4_metal.m`: append `QWEN4_K_QWEN35_ATTN_QPACK,` and `QWEN4_K_QWEN35_ATTN_FLASH_NAX,` to the kernel enum just before `QWEN4_K_COUNT`, and `"kernel_qwen35_attn_qpack",` and `"kernel_qwen35_attn_flash_nax",` at the end of `qwen4_kernel_names` (same order). Then add, directly after `ds4_gpu_qwen35_attn_flash_tensor`:

```c
/* Defined with the MoE accelerator helpers further down. */
static ds4_gpu_tensor *qwen4_nax_scratch(ds4_gpu_tensor **slot, uint64_t *slot_bytes, uint64_t bytes);

/* Mirrors metal/qwen35.metal's ds4_metal_args_qwen35_attn_qpack. */
struct ds4_qwen35_attn_qpack_args {
    uint32_t n_tokens, n_tokens_pad, n_head, n_head_kv;
    float scale2;
};

/* Mirrors metal/qwen35.metal's ds4_metal_args_qwen35_attn_flash_nax. */
struct ds4_qwen35_attn_flash_nax_args {
    uint32_t n_tokens, n_tokens_pad, n_head, n_head_kv, pos0;
    uint32_t kps, n_splits;
};

/* Packed queries of the accelerator flash: [Hkv][ceil8(T)][8][256] half,
 * grown on first use and kept (one buffer serves every layer). */
static ds4_gpu_tensor *g_qwen35_attn_qh;
static uint64_t g_qwen35_attn_qh_bytes;

/* M6 flash prefill on the M5 neural accelerators: the contract of
 * ds4_gpu_qwen35_attn_flash_tensor (T > 8 tokens at pos0.., causal, F16
 * K/V, head_dim 256, group 8, gated output) computed by
 * kernel_qwen35_attn_qpack + kernel_qwen35_attn_flash_nax.  Needs the Metal
 * 4 tensor API (ds4_gpu_mpp_available(): not under --quality, not before
 * M5/A19, not with DS4_METAL_DISABLE_METAL4=1) and refuses otherwise, so
 * the caller picks the M5 flash instead.  `part` must be NULL until the key
 * split lands (M6 Task 3). */
int ds4_gpu_qwen35_attn_flash_nax_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, float scale) {
    if (!ds4_gpu_mpp_available() || head_dim != 256u || n_head_kv == 0u || (n_head % n_head_kv) != 0u ||
        n_head / n_head_kv != 8u || n_tokens <= 8u || part != NULL) {
        fprintf(stderr, "ds4: qwen35 attn flash nax refuses tensor_api=%d head_dim=%u n_head=%u n_head_kv=%u "
                        "n_tokens=%u part=%s\n", ds4_gpu_mpp_available(), head_dim, n_head, n_head_kv, n_tokens,
                part ? "set" : "null");
        return 0;
    }
    const uint32_t t_pad = (n_tokens + 7u) & ~7u;
    const uint64_t qh_count = (uint64_t)n_head_kv * t_pad * 8u * head_dim;
    ds4_gpu_tensor *qh = qwen4_nax_scratch(&g_qwen35_attn_qh, &g_qwen35_attn_qh_bytes, qh_count * sizeof(uint16_t));
    if (!qh) return 0;
    const uint64_t q_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)(pos0 + n_tokens) * n_head_kv * head_dim * 2u;

    struct ds4_qwen35_attn_qpack_args pargs = { n_tokens, t_pad, n_head, n_head_kv, scale * 1.4426950408889634f };
    qwen4_bind pb[2];
    if (!qwen4_bind_tensor(&pb[0], q, q_bytes, "flash nax q") ||
        !qwen4_bind_tensor(&pb[1], qh, qh_count * sizeof(uint16_t), "flash nax packed q")) {
        return 0;
    }
    const uint32_t n4 = (uint32_t)(qh_count / 4u);
    if (!qwen4_dispatch(QWEN4_K_QWEN35_ATTN_QPACK, &pargs, sizeof(pargs), pb, 2,
                        MTLSizeMake((n4 + 255u) / 256u, 1, 1), MTLSizeMake(256, 1, 1), 0)) {
        return 0;
    }

    const uint32_t blocks = t_pad / 8u;
    g_qwen35_attn_flash_last_splits = 1u;
    struct ds4_qwen35_attn_flash_nax_args args = { n_tokens, t_pad, n_head, n_head_kv, pos0, pos0 + n_tokens, 1u };
    qwen4_bind b[5];
    b[0] = pb[1];   /* packed queries, already bound above */
    if (!qwen4_bind_tensor(&b[1], gate, q_bytes, "flash nax gate") ||
        !qwen4_bind_tensor(&b[2], k_cache, cache_bytes, "flash nax k cache") ||
        !qwen4_bind_tensor(&b[3], v_cache, cache_bytes, "flash nax v cache") ||
        !qwen4_bind_tensor(&b[4], out, q_bytes, "flash nax out")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_ATTN_FLASH_NAX, &args, sizeof(args), b, 5,
                          MTLSizeMake(n_head_kv, blocks, 1), MTLSizeMake(128, 1, 1), 0)
           ? 1 : 0;
}
```

In `ds4_gpu.h`, after the `ds4_gpu_qwen35_attn_flash_last_splits` declaration:

```c
/* M6: ds4_gpu_qwen35_attn_flash_tensor on the M5 neural accelerators (Metal 4
 * tensor API): same arguments, contract and key-split scratch.  Returns 0
 * without the tensor API (see ds4_gpu_tensor_api_available). */
int ds4_gpu_qwen35_attn_flash_nax_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache, ds4_gpu_tensor *part,
        uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, float scale);
```

- [ ] **Step 5: Run the tests**

Run: `make test-qwen35-kernels`, then `DS4_METAL_DISABLE_METAL4=1 ./tests/test_qwen35_kernels`, then `make test-qwen4-kernels test-qwen4-q2`.
Expected: the six `attn flash nax` lines print within tolerance (finite, guard intact, <= 2e-3 vs the M5 flash, <= 1e-3 vs the host reference where computed) and `qwen35 kernels: ok`; with Metal 4 disabled the six lines print `skipped (tensor API unavailable)` and every other line is unchanged; the qwen4 tests pass. If the Metal compiler rejects `vec<half, 8>` / `vec<float, 8>` or the cooperative-tensor calls, compare with MLX's `nax.h` (`BaseNAXFrag::mma`, `dtype_frag_t`) and match its spelling; the arithmetic stays as written.

- [ ] **Step 6: Benchmark rows**

In `tests/bench_qwen35_attn.c`, add after `run_flash_split`:

```c
static int run_flash_nax(void *p) {        /* M6: flash on the neural accelerators (tensor API) */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_nax_tensor(c->out, c->q, c->gate, c->kc, c->vc, NULL,
                                                c->T, H, HKV, D, c->pos0, c->scale);
}
```

and in the first prefill loop, after the `attn_flash_tok4` report line:

```c
            if (ds4_gpu_tensor_api_available())
                report("attn_flash_nax", "prefill", c.pos0, c.T, 0, time_ms(run_flash_nax, &c, 1, 3));
```

Run: `make tests/bench_qwen35_attn` (build only; timed runs happen in the controller's GPU window).

- [ ] **Step 7: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
git commit -m "qwen35: flash prefill attention on the M5 neural accelerators (no key split)

kernel_qwen35_attn_qpack packs scaled queries as half [Hkv][ceil8(T)][8][256];
kernel_qwen35_attn_flash_nax runs S = Q K^T and O += P V on matmul2d
fragments (8 tokens x 8 heads per threadgroup, K/V straight from the cache,
online softmax in registers) behind ds4_gpu_qwen35_attn_flash_nax_tensor;
tested against a host double reference and the M5 simdgroup flash, with
NaN K/V past the fill and a guard token.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_gpu.h tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
```

- [ ] **Step 8 (controller, GPU window): stop rule and precision setting**

On a snapshot of the Task 1 commit, live stack paused:
1. `./tests/bench_qwen35_attn prefill` twice; `make test-qwen35-kernels` and `DS4_METAL_DISABLE_METAL4=1 ./tests/test_qwen35_kernels`.
2. In the snapshot only, change `#define QWEN35_NAX_RELAXED false` to `true`; rerun `./tests/bench_qwen35_attn prefill` and `./tests/test_qwen35_kernels` (the Metal library compiles from `metal/` at startup, no rebuild needed).
3. Write `speed-bench/ornith/m6/BENCH.md`: per-layer ms for `qwen4_attn_mm`, `attn_flash_tok2`, `attn_flash_tok4`, `attn_flash_nax` (both precision settings) at pos 0 / 30720 / 122880, effective TFLOP/s at 30720 (1.07 TFLOP per layer), and the stop-rule verdict: pass when `attn_flash_nax` <= `attn_flash_tok2` / 2.5 at both 30720 and 122880.
4. Keep `QWEN35_NAX_RELAXED false` unless `true` is at least 5% faster at both positions and its test lines stay within tolerance; if `true` wins, commit the one-line change (`qwen35: accelerator flash uses relaxed-precision products`) with the measurement in the message.
5. Pass -> skip Task 2 (ledger: "A1 met the stop rule") and continue with Task 3. Fail -> Task 2.

---

### Task 2 (only if A1 misses the stop rule): variant A2, D split across simdgroup pairs

**Files:** `metal/qwen35.metal`, `ds4_metal.m`, `tests/bench_qwen35_attn.c`

**Interfaces:**
- Consumes: Task 1's helpers, args structs, wrapper and tests.
- Produces: kernel instance `kernel_qwen35_attn_flash_nax_d2` (enum `QWEN4_K_QWEN35_ATTN_FLASH_NAX_D2`, appended before `QWEN4_K_COUNT`); `ds4_gpu_qwen35_attn_flash_nax_tensor` picks it when `DS4_QWEN35_ATTN_NAX_KERNEL=d2` (read per call, benchmark and test only).

**Design (binding):**
- Same grid, args, output and split contract as A1, but 8 simdgroups (256 threads). Simdgroups `sg` and `sg + 4` share the 16 rows of tokens `t0 = blk*8 + 2*(sg % 4)` and `t0 + 1`; `dh = sg / 4` selects a 128-dim half.
- Scores: each simdgroup accumulates S0/S1 over its own 8 dim fragments (`id` in `[dh*8, dh*8 + 8)`), stores them to `threadgroup float Sx[4][2][32][16]` (16 KB: row group, dh, lane, 16 values), `threadgroup_barrier`, adds the partner's values (`Sx[sg % 4][1 - dh][lane]`), `threadgroup_barrier`. Both simdgroups then hold the full S and run the identical mask and softmax code of A1.
- Output: each simdgroup keeps only `O[8]` (its own dims `dh*128 ..`), accumulates `O += P V` for those 8 fragments, and writes its own 128 dims in the epilogue (the partial's `m`/`l` by the `dh == 0` simdgroup).
- Barriers need every simdgroup to run the same number of blocks: the key loop uses the threadgroup-wide range (`hi` from the block's last valid token, `lo` / split clamp as A1); per-row masking keeps rows correct; simdgroups whose tokens are all past T keep running (all rows masked) and skip the epilogue.

- [ ] **Step 1:** Add the `DS4_QWEN35_ATTN_NAX_KERNEL=d2` selection to the wrapper (threadgroup 256, same binds) and the kernel instance per the Design.
- [ ] **Step 2:** Run the Task 1 test cases with `DS4_QWEN35_ATTN_NAX_KERNEL=d2 ./tests/test_qwen35_kernels`; expected: the same tolerances pass.
- [ ] **Step 3:** Bench rows `attn_flash_nax_d2` next to `attn_flash_nax`.
- [ ] **Step 4:** Commit `qwen35: accelerator flash variant A2 (dims split across simdgroup pairs)`.
- [ ] **Step 5 (controller, GPU window):** bench both variants; A2 passes the stop rule -> it replaces A1 (commit: the `d2` instance takes the `kernel_qwen35_attn_flash_nax` host names, the A1 body and the env selection are removed; tests rerun). A2 misses -> stop the plan, write `BENCH.md` with both variants and report to the user.

---

### Task 3: Key split for the accelerator flash, short-chunk crossover

**Files:** `metal/qwen35.metal`, `ds4_metal.m`, `ds4_qwen35moe.inc` (comment only), `ds4.c` (comment only), `tests/test_qwen35_kernels.c` (additions), `tests/bench_qwen35_attn.c`

**Interfaces:**
- Consumes: Task 1's kernel template and wrapper; M5's `qwen35_attn_flash_splits`, `kernel_qwen35_attn_merge3`, `ds4_gpu_qwen35_attn_flash_part_floats`.
- Produces:
  - `ds4_gpu_qwen35_attn_flash_nax_tensor` accepts `part`: Ks from `qwen35_attn_flash_splits(pos0 + T, Hkv, ceil8(T)/8, ...)` (the M5 rule with 8-token blocks), `ds4_gpu_qwen35_attn_flash_last_splits()` reports it, Ks > 1 dispatches `kernel_qwen35_attn_flash_nax_split` over `(Hkv, blocks, Ks)` into `part` and merges.
  - `static int qwen35_attn_flash_merge3(const qwen4_bind *part_bind, const ds4_gpu_tensor *gate, ds4_gpu_tensor *out, uint64_t q_bytes, uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t pos0, uint32_t ks, uint32_t kps, float scale)` — the merge dispatch, shared by both flash wrappers (M5 behaviour unchanged).
  - `ds4_gpu_qwen35_attn_flash_part_floats` bounds 8-token blocks (`ceil(T/8)`), so one scratch serves both kernels (worst case ~34 MB instead of ~17 MB).

- [ ] **Step 1: Failing test**

In `main`, after the Task 1 lines:

```c
    setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", "4096", 1);   /* force key splits on short caches */
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(4096u, 64u, 1, 1) > 1,
               "flash nax key split taken (T=64)");
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(8000u, 200u, 1, 1) > 1,
               "flash nax key split taken (T=200)");
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(1000u, 1100u, 1, 1) > 1,
               "flash nax key split taken (T=1100, neutral partials)");
    unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
```

- [ ] **Step 2: Run and confirm it fails**

Run: `make test-qwen35-kernels`
Expected: `ds4: qwen35 attn flash nax refuses ... part=set` then `flash nax call failed`.

- [ ] **Step 3: Kernel instance**

Append inside the `#ifdef DS4_METAL_HAS_TENSOR` block of `metal/qwen35.metal`, after the `<false>` instance:

```metal
template [[host_name("kernel_qwen35_attn_flash_nax_split")]]
kernel void kernel_qwen35_attn_flash_nax<true>(
        constant ds4_metal_args_qwen35_attn_flash_nax &, device const half *, device const float *,
        device const half *, device const half *, device float *, uint3, ushort, ushort);
```

and add enum `QWEN4_K_QWEN35_ATTN_FLASH_NAX_SPLIT` / name `"kernel_qwen35_attn_flash_nax_split"` at the end of the lists.

- [ ] **Step 4: Shared merge, split path, scratch bound**

In `ds4_metal.m`, add before `ds4_gpu_qwen35_attn_flash_tensor`:

```c
/* Folds Ks flash key-split partials per token and applies the gate
 * (kernel_qwen35_attn_merge3 with rows = n_tokens, every row sharing Ks and
 * kps); shared by the simdgroup and accelerator flash wrappers. */
static int qwen35_attn_flash_merge3(const qwen4_bind *part_bind, const ds4_gpu_tensor *gate, ds4_gpu_tensor *out,
                                    uint64_t q_bytes, uint32_t n_tokens, uint32_t n_head, uint32_t n_head_kv,
                                    uint32_t head_dim, uint32_t pos0, uint32_t ks, uint32_t kps, float scale) {
    struct ds4_qwen35_attn_decode3_args margs =
        { n_head, n_head_kv, head_dim, pos0, n_tokens, ks, kps, ks, kps, scale };
    qwen4_bind mb[3];
    mb[0] = *part_bind;
    if (!qwen4_bind_tensor(&mb[1], gate, q_bytes, "flash merge gate") ||
        !qwen4_bind_tensor(&mb[2], out, q_bytes, "flash merge out")) {
        return 0;
    }
    return qwen4_dispatch(QWEN4_K_QWEN35_ATTN_MERGE3, &margs, sizeof(margs), mb, 3,
                          MTLSizeMake(n_head, n_tokens, 1), MTLSizeMake(256, 1, 1), 0)
           ? 1 : 0;
}
```

Replace the merge block at the end of `ds4_gpu_qwen35_attn_flash_tensor` (from `struct ds4_qwen35_attn_decode3_args margs =` to the final `? 1 : 0;`) with:

```c
    return qwen35_attn_flash_merge3(&sb[3], gate, out, q_bytes, n_tokens, n_head, n_head_kv, head_dim, pos0,
                                    ks, kps, scale);
```

In `ds4_gpu_qwen35_attn_flash_nax_tensor`: drop `|| part != NULL` from the refusal (and `part=%s` from its message), update the doc comment (the key split now follows the M5 rule with 8-token blocks), and replace everything from `const uint32_t blocks = t_pad / 8u;` to the end with:

```c
    const uint32_t blocks = t_pad / 8u;
    uint32_t ks = 1u, kps = pos0 + n_tokens;
    if (part) qwen35_attn_flash_splits(pos0 + n_tokens, n_head_kv, blocks, &ks, &kps);
    g_qwen35_attn_flash_last_splits = ks;
    struct ds4_qwen35_attn_flash_nax_args args = { n_tokens, t_pad, n_head, n_head_kv, pos0, kps, ks };
    qwen4_bind b[5];
    b[0] = pb[1];   /* packed queries, already bound above */
    if (!qwen4_bind_tensor(&b[1], gate, q_bytes, "flash nax gate") ||
        !qwen4_bind_tensor(&b[2], k_cache, cache_bytes, "flash nax k cache") ||
        !qwen4_bind_tensor(&b[3], v_cache, cache_bytes, "flash nax v cache")) {
        return 0;
    }
    if (ks <= 1u) {
        if (!qwen4_bind_tensor(&b[4], out, q_bytes, "flash nax out")) return 0;
        return qwen4_dispatch(QWEN4_K_QWEN35_ATTN_FLASH_NAX, &args, sizeof(args), b, 5,
                              MTLSizeMake(n_head_kv, blocks, 1), MTLSizeMake(128, 1, 1), 0)
               ? 1 : 0;
    }
    const uint64_t part_floats = (uint64_t)n_tokens * n_head * ks * (2u + head_dim);
    if (!qwen4_bind_tensor(&b[4], part, part_floats * sizeof(float), "flash nax split part")) return 0;
    if (!qwen4_dispatch(QWEN4_K_QWEN35_ATTN_FLASH_NAX_SPLIT, &args, sizeof(args), b, 5,
                        MTLSizeMake(n_head_kv, blocks, ks), MTLSizeMake(128, 1, 1), 0)) {
        return 0;
    }
    return qwen35_attn_flash_merge3(&b[4], gate, out, q_bytes, n_tokens, n_head, n_head_kv, head_dim, pos0,
                                    ks, kps, scale);
```

In `ds4_gpu_qwen35_attn_flash_part_floats`, change `const uint32_t blocks = (n_tokens + 3u) / 4u;   /* ceil(T / 4) */` to `const uint32_t blocks = (n_tokens + 7u) / 8u;   /* ceil(T / 8) */` and rewrite its doc comment's sizing sentence: the bound now covers the accelerator kernel's 8-token blocks (fewer blocks than either simdgroup instance, hence the largest Ks), about 34 MB at worst for the default min_tg = 256. Update the size figures in the matching comments in `ds4_qwen35moe.inc` (`~17 MB` -> `~34 MB`, and name both kernels) and `ds4.c` (the attn_flash_part comment).

- [ ] **Step 5: Run the tests**

Run: `make test-qwen35-kernels`, `DS4_METAL_DISABLE_METAL4=1 ./tests/test_qwen35_kernels`, `make test-qwen4-kernels test-qwen4-q2`, `make ds4 ds4-server ds4_test` (warning-free).
Expected: all `attn flash nax` lines within tolerance, the three forced-split cases report `splits=` > 1; the M5 flash lines unchanged; qwen4 tests pass.

- [ ] **Step 6: Benchmark rows**

In `tests/bench_qwen35_attn.c`: add

```c
static int run_flash_nax_split(void *p) {  /* same, with the key split engaged via `part` */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_nax_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part_flash,
                                                c->T, H, HKV, D, c->pos0, c->scale);
}
```

extend the scratch-sizing loop so `pf_max` also covers T = 16, 32, 64 and 256 (next to 2048 and 128), and append at the end of the `do_prefill` block:

```c
        if (ds4_gpu_tensor_api_available()) {
            /* short chunks at long context, default split rule: where the
             * accelerator kernel starts to beat the simdgroup flash */
            const uint32_t ts[5] = { 16u, 32u, 64u, 128u, 256u };
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "2", 1);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 5; j++) {
                    c.pos0 = split_pos[i]; c.T = ts[j]; c.rows = 0;
                    report("flash_tok2_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
                    report("flash_nax_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_nax_split, &c, 1, 3));
                }
            unsetenv("DS4_QWEN35_ATTN_FLASH_TOK");
        }
```

- [ ] **Step 7: Commit**

```bash
git add metal/qwen35.metal ds4_metal.m ds4_qwen35moe.inc ds4.c tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
git commit -m "qwen35: key split for the accelerator flash prefill

Short chunks at long context split the key range with the M5 rule over
8-token blocks; kernel_qwen35_attn_flash_nax_split writes merge3 partials
(natural-log m, neutral l = 0 rows) and both flash wrappers share one merge
dispatch; the flash key-split scratch bound now covers 8-token blocks.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBHu85zs7NPR1xpPcNEpG2" -- metal/qwen35.metal ds4_metal.m ds4_qwen35moe.inc ds4.c tests/test_qwen35_kernels.c tests/bench_qwen35_attn.c
```

- [ ] **Step 8 (controller, GPU window): crossover**

`./tests/bench_qwen35_attn prefill` twice; the kernel tests under both Metal settings. Add the short-chunk table to `BENCH.md`. Crossover rule: if `flash_nax_split` is slower than `flash_tok2_split` at some benched T at either position, the dispatch minimum `QWEN35_ATTN_NAX_MIN_T` (Task 4) is the smallest benched T from which `flash_nax_split` wins at both positions for every larger benched T; if it wins everywhere, no minimum is added. Ledger the value.

---

### Task 4: Dispatch, knob, memory estimate, adoption

**Files:** `ds4_qwen35moe.inc` (`qwen35_graph_attend`, knob), `ds4.c` (Ornith memory estimate)

**Interfaces:**
- Consumes: `ds4_gpu_qwen35_attn_flash_nax_tensor`, `ds4_gpu_tensor_api_available()`, Task 3's crossover value.
- Produces: knob `DS4_QWEN35_ATTN_NAX` (read once; off until Step 8); with it on, prefill chunks with T > 8 (and T >= `QWEN35_ATTN_NAX_MIN_T` if Task 3 set one), `DS4_QWEN35_ATTN_FLASH` on, F16 K/V and the tensor API available run the accelerator flash on `g->attn_flash_part`; otherwise the M5 flash runs unchanged.

- [ ] **Step 1: Knob and dispatch**

In `ds4_qwen35moe.inc`, after `qwen35_attn_flash_env`:

```c
/* M6 (DS4_QWEN35_ATTN_NAX): runs flash prefill on the M5 neural accelerators
 * (kernel_qwen35_attn_flash_nax), read once.  Off until the M6 adoption
 * checks pass; "1" turns it on, "0" keeps the M5 simdgroup flash.  Takes
 * effect only with DS4_QWEN35_ATTN_FLASH on and the Metal 4 tensor API
 * available (not under --quality, before M5/A19, or with
 * DS4_METAL_DISABLE_METAL4=1), where the M5 flash keeps running. */
static int qwen35_attn_nax_env(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("DS4_QWEN35_ATTN_NAX");
        if (e && e[0] && strcmp(e, "0") != 0 && strcmp(e, "1") != 0)
            fprintf(stderr, "ds4: DS4_QWEN35_ATTN_NAX=%s not recognised (0 or 1); using off\n", e);
        on = (e && strcmp(e, "1") == 0) ? 1 : 0;
    }
    return on;
}
```

In `qwen35_graph_attend`, make the flash branch:

```c
        if (n > 8u && qwen35_attn_flash_env() && !qwen4_kv_mode(g)) {
            if (qwen35_attn_nax_env() && ds4_gpu_tensor_api_available()) {
                return ds4_gpu_qwen35_attn_flash_nax_tensor(g->attn_o, g->q, g->gate,
                            qwen4_kcache(g, il), qwen4_vcache(g, il), g->attn_flash_part, n,
                            DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, scale) != 0;
            }
            return ds4_gpu_qwen35_attn_flash_tensor(g->attn_o, g->q, g->gate,
                        qwen4_kcache(g, il), qwen4_vcache(g, il), g->attn_flash_part, n,
                        DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, scale) != 0;
        }
```

If Task 3 ledgered a minimum, add `#define QWEN35_ATTN_NAX_MIN_T <value>u` above `qwen35_attn_nax_env` with a comment citing `speed-bench/ornith/m6/BENCH.md`, and add `n >= QWEN35_ATTN_NAX_MIN_T &&` to the inner condition.

- [ ] **Step 2: Memory estimate**

In `ds4.c`, in the Ornith estimate after the `attn_flash_part` block (before `m.total_bytes = ...`):

```c
        /* M6: the accelerator flash's packed-query scratch (a ds4_metal.m
         * slot of Hkv x ceil8(T) x 8 x 256 halves, 16.8 MB at T = 2048),
         * counted whether or not the tensor API ends up available. */
        if (T > 8u) m.scratch_bytes += ((T + 7u) & ~7ull) * DS4_N_HEAD * DS4_N_HEAD_DIM * sizeof(uint16_t);
```

- [ ] **Step 3: Build and tests**

Run: `make ds4 ds4-server ds4_test` (warning-free), `make test-qwen35-kernels`, `./ds4_test --server`.
Expected: clean build, `qwen35 kernels: ok`, server tests pass.

- [ ] **Step 4: Commit**

`git commit -m "qwen35: DS4_QWEN35_ATTN_NAX selects the accelerator flash prefill (default off)"` with the trailer, files `ds4_qwen35moe.inc ds4.c` by path.

- [ ] **Step 5 (controller, GPU window): fallback identity**

Snapshots of develop `696d328` and of the Task 4 commit. For chunks 2048, 64, 65 run `python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg $c <dir>`:
- develop vs branch with defaults (knob off): every dump `cmp`-identical;
- develop vs branch, both with `DS4_METAL_DISABLE_METAL4=1`, the branch also with `DS4_QWEN35_ATTN_NAX=1`: every dump `cmp`-identical.

- [ ] **Step 6 (controller, same window): accelerator correctness**

With `DS4_QWEN35_ATTN_NAX=1`: gate 1 at chunks 2048/64/65 (pass/fail per `tests/ornith/tolerance.json`), `make test-qwen35-graph test-qwen35-mtp`, `tests/ornith/test_mtp_cli.py`, `./ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` with and without `DS4_TEST_GLM_MTP=1`, the `DS4_QWEN35_KV=q4` payload loop (must keep the old kernels), and `DS4_QWEN35_PROFILE=1 ./ds4 -m "$DS4_ORNITH_MODEL" --metal --raw --prompt-file <32K and 128K prompts> -c 262144 -n 8 --prefill-chunk 2048` (per-chunk attention ms vs M5's 2,423 ms at pos 30,720 and 8,887 ms at 122,880).

- [ ] **Step 7 (controller, same window): lever A/B**

`python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model "$DS4_ORNITH_MODEL" --out <dir> --base-env "DS4_QWEN35_ATTN_NAX=0 DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --lever-env "DS4_QWEN35_ATTN_NAX=1 DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --contexts 32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`; `speed-bench/qwen-regression/run.sh fast`. Write `speed-bench/ornith/m6/LEVERS.md` (fallback identity, correctness table, profile, A/B, Qwen fast gate).

- [ ] **Step 8: Default**

If Steps 5-7 pass and prefill improves: flip the default (`on = (e && strcmp(e, "0") == 0) ? 0 : 1;`, warning text `using on`, comment "on by default since the M6 A/B (<numbers>)") and commit `qwen35: the accelerator flash is the Ornith prefill default`. Otherwise leave it off and record why in `LEVERS.md` (and in the ledger).

---

### Task 5: Final gates, receipts and report

**Files:** `speed-bench/ornith/m6/{speed/GATE3.md,quality/GATE2.md,QWEN_GATE.md,REPORT.md}`, `docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md` (§4 attention, §7.3 knobs)

- [ ] **Step 1 (controller, GPU window): gate 3** — `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --out <dir> --ds4-env "DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1` on the final defaults; write `speed/GATE3.md` against spec §1 items 1-2 (prefill and TTFT vs the live oMLX; decode within 3% of M5's `speed-bench/ornith/m5/speed/GATE3.md` numbers).
- [ ] **Step 2 (controller, GPU window): gate 2** — the M4/M5 gate-2 scripts (eval, agentic matrix, probes) on 23G and 25G plus the eval against the live oMLX the same day; pass rule per spec §1 item 6. Write `quality/GATE2.md`.
- [ ] **Step 3 (controller, GPU window): Qwen full gate** — `speed-bench/qwen-regression/run.sh full`; write `QWEN_GATE.md`.
- [ ] **Step 4: parent spec** — §4 attention: prefill chunks with T > 8 run `kernel_qwen35_attn_flash_nax` when the tensor API is available (M5 simdgroup flash otherwise); §7.3: add `DS4_QWEN35_ATTN_NAX` with its default.
- [ ] **Step 5: report** — `REPORT.md`: M5 vs M6 per context against the live oMLX, the kernel benchmark, the stop-rule verdict and chosen variant/precision, the crossover, adoption verdict, gates 1/2/3 and Qwen results, open items.
- [ ] **Step 6: commit** the receipts, spec and report by path with the trailer.

---

## Self-Review

1. **Spec coverage.** §2.1 query pack: Task 1 (kernel, scratch slot). §2.2 accelerator flash: Task 1 (A1, masking, softmax, epilogue), Task 2 (A2), Task 3 (key split, merge3 reuse). §2.3 dispatch and knobs: Task 4 (knob, tensor-API condition, crossover minimum from Task 3, memory estimate), Task 3 (key-split scratch bound). §3 exactness: Task 1 tests (host reference, M5 flash, drift case), Task 4 Steps 5-6 (fallback identity, gate 1, MTP identity). §4 testing and the stop rule: Tasks 1-3 (tests, benchmark, Step 8 verdicts), 4-5 (model checks, gates). §1 success criteria: Task 5 (gates 2/3, Qwen full), Task 4 (criteria 3-5).
2. **Placeholders.** Task 2 is conditional and specified by a binding design plus Task 1's tests (it only runs if A1 misses); every other code step has its code. `<list>`, `<dir>` and the prompt files are the controller's existing M4/M5 window paths.
3. **Type consistency.** `ds4_gpu_qwen35_attn_flash_nax_tensor` has the M5 flash signature in Tasks 1, 3, 4, the tests and the bench; `ds4_metal_args_qwen35_attn_flash_nax` / `ds4_qwen35_attn_flash_nax_args` and `..._qpack` match field for field; enum names match their `qwen4_kernel_names` strings; `qwen35_attn_flash_merge3` is used with one parameter list in both wrappers.
4. **Review Focus.** Lines 1-2: Task 1 test (NaN past the fill, guard token). Line 3: Task 3 forced splits. Line 4: Task 1 drift case, Tasks 4-5 gates. Line 5: Task 4 Step 5 and both-Metal-settings test runs.
