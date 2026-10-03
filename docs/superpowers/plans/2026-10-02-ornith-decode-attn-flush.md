# Ornith decode Plan C: decode-attention prefetch (L5) and multi-flush (L1)

> **Status:** executed, measured, and not merged. The levers gave no decode gain (see
> `speed-bench/ornith/decode/REPORT.md`). Their code and env knobs exist only on the local branch
> `feature/ornith-decode` at `01d4bab0`, not on develop.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the rest of the +15% target after the two-row MoE (+10.6% at 2K, +7.6% at 32K
projected), losslessly.
- **At 32K:** decode attention runs at ~46% of its byte floor (7.37 ms of attention mixer per step).
- **At 2K:** the GPU idles ~1.5 ms per MTP cycle while the host encodes.

**Architecture:**
- **L5.** `kernel_qwen35_attn_decode3` stages 16-key K/V tiles into threadgroup memory behind a barrier,
  so every tile waits for its loads. A `PF` variant loads the next tile into registers while the
  current tile computes.
  - The threadgroup memory each tile sees is byte-identical, and the arithmetic is unchanged, so the
    outputs are bit-identical.
  - Selected by `DS4_QWEN35_ATTN_PREFETCH`.
- **L1.** Besides the existing flush after layer 2 (`DS4_QWEN35_FLUSH_LAYER`), `DS4_QWEN35_FLUSH_EVERY=N`
  submits the command buffer every N layers after it, so the GPU starts each chunk while the host
  encodes the rest. It changes submission only.

**Tech Stack:** Metal (`metal/qwen35.metal`), Objective-C (`ds4_metal.m`), C (`ds4.c`,
`ds4_qwen35moe.inc`), C tests (`tests/test_qwen35_kernels.c`, `tests/ds4_test.c`), the model-free
microbench `tests/bench_qwen35_attn`.

**Spec:** `docs/superpowers/specs/2026-10-02-ornith-decode-design.md` (levers L1 and L5).
Evidence: `speed-bench/ornith/decode/PROFILE.md`.

## Global Constraints

- **Lossless.**
  - The PF kernel's output (`out` and `part`) `memcmp`-equals the current decode3's for ROWS 1 and 2,
    at split and unsplit positions.
  - The flush change only moves `commit` calls.
  - Gate-1 dumps, `test_mtp_cli` and `test_qwen35_verify_batch` stay identical with both levers on.
- **Ornith-only.** A new kernel entry plus host selection in the Ornith decode3 wrapper. The flush
  change touches only `qwen35_graph_forward_tokens`.
- **Knobs** start default off and flip on after a window's identity check and an A/B gain of at
  least 3%, with no loss above 3%.
- **Timing measurements** (microbench, A/B) run only in my announced windows, never while another
  session measures or while DS41F's disk slot runs.

## Review Focus

1. **A tile at the end of a split** (fewer than 16 keys left, `present` false). The prefetch must load
   zeros exactly where the original did. Test: PF vs original at a position whose split leaves a
   partial last tile.
2. **ROWS == 2 with row 1 one key past row 0.** The prefetch uses `hi_max`, as the original staging
   does. Test: PF vs original, ROWS 2 at an odd split boundary.
3. **A split with a single tile, or zero tiles.** No prefetch beyond `hi_max`, and no read of
   uninitialised registers. Test: positions 0, 1, 15, 16 and 17.
4. **The flush schedule at the last layer.** Never flush after the final trunk layer (the head's
   commands follow). Test: `ds4_qwen35_flush_after` model-free cases.
5. **The flush env has a bad value.** Off, plus a warning, never a crash. Test: model-free cases for
   "0", "-1", "x" and "".

---

### Task 1: decode3 prefetch kernel (L5), exactness test, microbench arm

**Files:**
- Modify: `metal/qwen35.metal`. Add `template <uint ROWS, bool PF> kernel_qwen35_attn_decode3_pf`, a copy
  of `kernel_qwen35_attn_decode3` whose staging block becomes:

```metal
    constexpr uint UNITS = KT * 8u, UPT = (UNITS + NTHREADS - 1u) / NTHREADS;
    uint4 kreg[UPT][4], vreg[UPT][4];
    /* load tile tb's K/V units this thread stages into registers; a key at or
     * beyond hi_max loads as zero, exactly as the barrier-staged original */
#define QWEN35_D3_LOAD(tb) do { \
        for (uint u = 0; u < UPT; u++) { \
            const uint i = tid + u * NTHREADS; \
            if (i >= UNITS) break; \
            const uint key = i >> 3, seg = i & 7u, idx = (tb) + key; \
            const bool present = idx < hi_max; \
            const uint64_t row = ((uint64_t)(present ? idx : 0u) * Hkv + kvh) * D; \
            device const uint4 *kr = (device const uint4 *)(k_cache + row) + seg * 4; \
            device const uint4 *vr = (device const uint4 *)(v_cache + row) + seg * 4; \
            for (uint w = 0; w < 4; w++) { kreg[u][w] = present ? kr[w] : uint4(0u); vreg[u][w] = present ? vr[w] : uint4(0u); } \
        } \
    } while (0)
    if (lo_c < hi_max) QWEN35_D3_LOAD(lo_c);
    for (uint t0 = lo_c; t0 < hi_max; t0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = 0; u < UPT; u++) {
            const uint i = tid + u * NTHREADS;
            if (i >= UNITS) break;
            const uint key = i >> 3, seg = i & 7u;
            threadgroup uint4 *kd = (threadgroup uint4 *)(Ks + key * D) + seg * 4;
            threadgroup uint4 *vd = (threadgroup uint4 *)(Vs + key * D) + seg * 4;
            for (uint w = 0; w < 4; w++) { kd[w] = kreg[u][w]; vd[w] = vreg[u][w]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + KT < hi_max) QWEN35_D3_LOAD(t0 + KT);   /* next tile in flight during this one */
        /* ... the original tile body, unchanged ... */
    }
#undef QWEN35_D3_LOAD
```

  Instances: `kernel_qwen35_attn_decode3_pf_r1` and `_r2`.
- Modify: `ds4_metal.m`, the enum and name table after `QWEN4_K_QWEN35_ATTN_DECODE3_R2`. In
  `ds4_gpu_qwen35_attn_decode3_tensor`, pick the `_pf` pipeline when `qwen35_attn_prefetch_on()` holds
  (env `DS4_QWEN35_ATTN_PREFETCH`, default off).
- Test: `tests/test_qwen35_kernels.c`, new `test_attn_decode3_prefetch(a, pos0, split_keys)`. It runs
  decode3 with the env off then on, for ROWS 1 and 2, and `memcmp`s `out` and `part`. Positions are
  0, 1, 15, 16, 17, 1000, 4097 and 32767, with split_keys 0 (auto) and 512.
- Microbench: `tests/bench_qwen35_attn decode` gains a `decode3-pf` row (same setup, env on).

- [ ] **Step 1: RED.** The new test fails: the env does not change the kernel yet, so a check that the
  PF pipeline ran (a dispatch counter `ds4_gpu_qwen35_attn_pf_dispatches()`) fails.
- [ ] **Step 2:** Implement the kernel, the selection and the counter.
- [ ] **Step 3: GREEN.** `make tests/test_qwen35_kernels && ./tests/test_qwen35_kernels` prints
  `decode3 prefetch {...}: bit-identical` and passes. It is not timing, so it may run outside windows
  while nobody measures.
- [ ] **Step 4: Commit,** path-scoped.

### Task 2: Multi-flush (L1)

**Files:**
- Modify: `ds4.h` and `ds4.c`. Add the pure helper
  `bool ds4_qwen35_flush_after(uint32_t il, int flush_layer, uint32_t every, uint32_t n_trunk)`: true
  after layer `flush_layer - 1`; then, when `every > 0`, after each layer `flush_layer - 1 + k*every`;
  never after the last trunk layer.
- Modify: `ds4_qwen35moe.inc`. `qwen35_graph_forward_tokens` uses the helper with
  `DS4_QWEN35_FLUSH_EVERY` (parsed once; unset, 0 or invalid means off, with a warning for invalid).
- Test: `tests/ds4_test.c` entry `--qwen35-flush-schedule`. With n_trunk=40 and flush_layer 2:
  - every 0: true only at il 1;
  - every 8: true at il 1, 9, 17, 25, 33; il 39 false;
  - every 1: true at il 1-38, false at il 39.
- [ ] **Step 1:** RED (the helper is undeclared).
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** GREEN.
- [ ] **Step 4:** Commit.

### Task 3: Window, microbench, identity, A/B

- [ ] **Step 1: Microbench.** `tests/bench_qwen35_attn decode` at 2K, 32K and 128K positions: decode3
  against decode3-pf in ms, and the fraction of the floor.
- [ ] **Step 2: Identity.** With `DS4_QWEN35_ATTN_PREFETCH=1 DS4_QWEN35_FLUSH_EVERY=8` plus the
  two-row MoE:
  - gate 1 at chunks 64/512/2048 equals window M's dumps;
  - `test_mtp_cli` passes;
  - `test-qwen35-verify-batch` passes.
- [ ] **Step 3: A/B, one knob at a time.** `m4_ab.py --mode lever` at 2K, 32K and 128K. Base is
  `DS4_QWEN35_MOE_PAIR=1` and the other knob off; lever adds the knob. Try `DS4_QWEN35_FLUSH_EVERY` at 4
  and 8 if the window has time.
- [ ] **Step 4:** Flip each knob that passes to default on. Record the results in
  `speed-bench/ornith/decode/levers/`.
