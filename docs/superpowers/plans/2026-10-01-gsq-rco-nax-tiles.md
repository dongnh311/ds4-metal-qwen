# GSQ-RCO routed experts on the tensor-op tiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route GSQ-RCO routed experts in prefill to the Metal 4 tensor-op tiles
(`kernel_qwen4_moe_mm_{mid,down}_nax*`) that PROD's packs use on M5. On PROD's packs those tiles
added 29-51% prefill.

**Architecture:** The tensor-op tiles stage the weight operand through a register prefetch
(`qwen4_load_raw16` / `qwen4_dequant_raw16`). That prefetch knows only Q4_K, Q2_K, IQ2_XXS and
MXFP4. For every other type the tiles call `qwen4_mm_stage16` instead, which is the dequantizer the
simdgroup tiles already use for GSQ-RCO and which matches the CPU rows. Those types get no prefetch.
The cooperative matmul and the B staging are unchanged.

On M5 the pipelines are specialized per type (function constant 900), so PROD's four types compile
the same code as today. The host turns the tiles on for GSQ-RCO types at the default level 2, as for
the four types.

**Tech Stack:** Metal (metal/qwen4.metal), Objective-C (ds4_metal.m, ds4_gpu.h), C tests (tests/ds4_test.c).

**Spec:**
- `docs/superpowers/specs/2026-09-30-gsq-rco-types-design.md`: Engine item 3, prefill for routed types;
  "M5 tuning only if sub-project 5 shows speed is short".
- `speed-bench/nextgen-eval/results/2026-10-01-sp3-prefill-gemm.md`: "What is left", tensor-op tiles.

## Global Constraints

- PROD's types keep their kernels and numbers (byte-identical greedy output, speed within 2%).
- Mac only; no push, deploy or HF upload without the user.
- GPU model runs happen inside a gateway pause; never `kill -9` a Metal process.
- Code, docs and commits in English.

## Review Focus

1. The prefetch registers stay unused for GSQ-RCO types, and nothing reads past an expert's rows for them.
2. Down tiles over K = 640 (20 steps of 32) for IQ4_NL and Q2_0.
3. The half copy of `mid` that the down tiles read is still produced by the mid tiles for GSQ-RCO types.
4. The levels other than the default (`DS4_QWEN4_MOE_MM_NAX` 1/3/4/5/6) also run for GSQ-RCO types.
5. Devices without the tensor API keep the simdgroup tiles.

---

### Task 1: the tensor-op tiles stage GSQ-RCO experts

**Files:**
- Modify: `metal/qwen4.metal` (`kernel_qwen4_moe_mm_mid_nax_t`, `kernel_qwen4_moe_mm_down_nax_t`)
- Modify: `ds4_metal.m` (`qwen4_moe_mm_nax_level`, `qwen4_moe_mm_nax`, `ds4_gpu_qwen4_moe_mm_nax_width`), `ds4_gpu.h`
- Test: `tests/ds4_test.c` (`test_metal_qwen4_quant_moe_mm`)

**Interfaces:**
- Produces: `int ds4_gpu_qwen4_moe_mm_nax_width(uint32_t type)`. It returns the token width of the
  tensor-op tiles the MoE GEMM runs for `type` (64 or 32), or 0 for the simdgroup tiles.

- [ ] **Step 1: Write the failing test.** At the top of `test_metal_qwen4_quant_moe_mm`:

```c
    /* with the tensor API (IQ2_XXS takes the tensor-op tiles), the GSQ-RCO experts take them too;
     * the cases below then run those tiles */
    if (ds4_gpu_qwen4_moe_mm_nax_width(16) != 0) {
        static const uint32_t gsq[6] = { 17, 18, 20, 21, 22, 42 };
        for (int i = 0; i < 6; i++)
            TEST_ASSERT(ds4_gpu_qwen4_moe_mm_nax_width(gsq[i]) == ds4_gpu_qwen4_moe_mm_nax_width(16));
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test`
Expected: compile error, `ds4_gpu_qwen4_moe_mm_nax_width` undeclared. Once declared and returning
the old routing, the asserts fail on M5.

- [ ] **Step 3: Implement**

`metal/qwen4.metal`. In both tensor-op kernels, the weight staging becomes:

```metal
    /* the register prefetch covers Q4_K, Q2_K, IQ2_XXS and MXFP4; other types
     * (GSQ-RCO) stage straight from their rows */
    const bool raw = type == 12u || type == 10u || type == 16u || type == 39u;
```

The mid kernel:

```metal
        qwen4_raw16 rg{}, ru{};
        if (raw) { rg = qwen4_load_raw16(grow, 0, aq * 2, type); ru = qwen4_load_raw16(urow, 0, aq * 2, type); }
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dg = Ag + ar * NK + aq * 16;
                threadgroup half *du = Au + ar * NK + aq * 16;
                if (a_row) {
                    if (raw) {
                        qwen4_dequant_raw16(rg, kb, aq * 2, type, dg);
                        qwen4_dequant_raw16(ru, kb, aq * 2, type, du);
                    } else {
                        qwen4_mm_stage16(grow, kb, aq * 2, type, dg);
                        qwen4_mm_stage16(urow, kb, aq * 2, type, du);
                    }
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
                if (raw && kb + 1 < nk) { rg = qwen4_load_raw16(grow, kb + 1, aq * 2, type); ru = qwen4_load_raw16(urow, kb + 1, aq * 2, type); }
            }
```

The down kernel does the same with `rd`/`drow`/`dd`.

`ds4_metal.m`:
- `qwen4_mm_gsq_block` moves above `qwen4_moe_mm_nax_level`.
- In `qwen4_moe_mm_nax_level`, the unset default `(type == 12u || type == 39u || type == 16u || type == 10u) ? 2 : 0`
  becomes `(type == 12u || type == 39u || type == 16u || type == 10u || qwen4_mm_gsq_block(type)) ? 2 : 0`.
- In `qwen4_moe_mm_nax`, the type check gains `&& !qwen4_mm_gsq_block(type)`. The full line becomes
  `if (!(type == 12u || type == 39u || type == 16u || type == 10u) && !qwen4_mm_gsq_block(type)) return 0;`.
- After `qwen4_moe_mm_nax`, add:

```c
int ds4_gpu_qwen4_moe_mm_nax_width(uint32_t type) { return (int)qwen4_moe_mm_nax(type); }
```

`ds4_gpu.h`, after `ds4_gpu_qwen4_moe_mm_down_tensor`:
`/* token width of the tensor-op tiles the MoE GEMM runs for a type, 0 for the simdgroup tiles */`
`int ds4_gpu_qwen4_moe_mm_nax_width(uint32_t type);`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `make ds4_test && ./ds4_test --metal-kernels`
Expected: `metal-kernels: OK`. The five MoE GEMM cases now run the tensor-op tiles for the GSQ-RCO
types, within the same 1.5e-3 bound: the operands are the same halves, and only the accumulation
order differs.

- [ ] **Step 5: Commit** `metal: GSQ-RCO routed experts take the tensor-op tiles`

### Task 2: model-level checks (next GPU window)

1. **Correctness:** ISTA `--first-token-test` on the 181-token prompt at chunk 256, against this
   morning's 3.021 (simdgroup tiles) and Ivan's 4.249. The bar is the same as this morning's: 5.31.
2. **ISTA speed:** `ds4-bench`, 8K, 3 rounds: prefill against 361.3 t/s.
3. **Stage split:** `DS4_QWEN4_STAGE_TS_PROFILE=1` on one 8K ISTA prefill. It records how much is
   `moe` against the dense stages; that decides whether the dense GEMM needs tensor-op tiles too.
4. **PROD:** identity 3/3 and `ds4-bench` A/B, at least 0.98x.
5. **Results file:** append to `speed-bench/nextgen-eval/results/2026-10-01-sp5-gsq-verify.md`.
