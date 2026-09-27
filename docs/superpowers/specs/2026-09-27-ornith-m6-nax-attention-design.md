# Ornith M6: prefill attention on the M5 neural accelerators (design)

Milestone M6 of the Ornith-1.5-35B-A3B port (family `QWEN35_MOE`, parent spec
`docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`, previous milestone
`docs/superpowers/specs/2026-09-27-ornith-m5-attention-design.md`). Status: design approved in chat on 2026-09-27
(approach A, fused kernel); this file is the written spec for review. Branch `feature/ornith-m6` from develop
`696d328` (M5 merged).

## 1. Goal and success criteria

M5 left gate 3's prefill criterion open (`speed-bench/ornith/m5/REPORT.md`, same harness, same day):

| | ds4 after M5 | live oMLX |
|---|---|---|
| prefill, 32K / 128K | 983 / 354 t/s | 1677 / 739 t/s |
| cold ~31K first turn (TTFT) | 32.4 s | 20.9 s |

The gap is attention. From M5's in-model profile (`speed-bench/ornith/m5/LEVERS.md`), a 32K prompt spends about
20 of its ~33 s in prefill attention and a 128K prompt about 300 of ~370 s; the rest of prefill runs at roughly
2,400 t/s. `kernel_qwen35_attn_flash` is bound by simdgroup-matrix throughput (~4.8 TFLOP/s at 30K keys; TOK=4 and
key splits do not move it, `speed-bench/ornith/m5/BENCH.md`), so closing the gap needs more matrix throughput: the
M5 neural accelerators through the Metal 4 tensor API (`matmul2d`). The live oMLX runs MLX 0.32, whose metallib
contains a fused accelerator attention (`steel_attention_nax`) for head dims 64/80/128 only; Ornith's head dim is
256, so the live oMLX most likely runs attention unfused through accelerator GEMMs. Either way its attention
matrix work runs on the accelerators.

M6 succeeds when, measured with the M4 harness (`speed-bench/ornith/m4_ab.py`, A-B-B-A, one model process at a
time, fresh nonce per prompt, `cached_tokens == 0`, same day as the live oMLX):

1. ds4 prefill >= live oMLX at 32K and 128K, and the cold ~31K first-turn TTFT <= live oMLX (this needs prefill
   attention to fall from ~20 s to <= ~6.5 s at 32K and from ~300 s to <= ~110 s at 128K);
2. decode with `--mtp` does not regress: within 3% of M5 at 2K, 32K and 128K;
3. gate 1 (llama.cpp references, M1 tolerance) passes at prefill chunks 2048, 64 and 65;
4. `--mtp` greedy output stays byte-identical to plain decoding (M2 guarantee);
5. with `DS4_METAL_DISABLE_METAL4=1` the gate-1 dumps are byte-identical to develop `696d328` (the fallback path is
   today's M5 path, unchanged);
6. gate 2 at parity with the live oMLX on 23G and 25G (index >= oMLX index - 3.0, truncated = errored = 0, matrix
   without M6-specific quality failures, probes match; parent spec section 8 as updated by M5);
7. Qwen3.8 stays byte-identical (Qwen full gate PASS).

Out of scope: decode (`kernel_qwen35_attn_decode3` is bandwidth-bound and unchanged), the 2K/32K decode gap (its own
milestone), fp8/q4 K/V modes, Qwen3.8 attention, and non-M5 hardware (it keeps the M5 kernels).

## 2. Design

All new code is Ornith-only and additive: kernels in `metal/qwen35.metal` inside `#ifdef DS4_METAL_HAS_TENSOR`,
wrappers in `ds4_metal.m` (declared in `ds4_gpu.h`), dispatch in `ds4_qwen35moe.inc`, scratch accounting in `ds4.c`.
The shared qwen4 kernels, `metal/qwen4.metal` and every Qwen3.8 path are untouched.

Reference designs: MLX's fused accelerator attention (MIT licence, on this machine at
`~/.local/omlx-venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/`:
`kernels/steel_attention_nax.h`, `nax.h`) and this repo's accelerator MoE kernels
(`kernel_qwen35_moe_mm_mid_q5k_nax_t` / `_down_` in `metal/qwen35.metal`).

### 2.1 `kernel_qwen35_attn_qpack` — half-precision query pack

The query buffer is F32 laid out `[t][H][D]`, so the 16 rows of one 8-head x 2-token fragment are not evenly
strided and cannot feed an accelerator tile load directly. A small kernel writes `q * scale` as F16 into a scratch
laid out `[kvh][t][g][D]` (g = query head within the KV group, 0..7): the 8·T rows of one KV head are then
contiguous with row stride D, which is the shape MLX loads directly from device memory. The token count is
padded to a multiple of 8 and the padding tokens are written as zero, so the 8-token blocks need no row guards.
Scratch size `Hkv * ceil8(T) * 8 * 256 * 2` bytes (16.8 MB at T = 2048), held in a `ds4_metal.m` scratch slot
grown on first use, like the MoE accelerator kernels' half operands (`qwen4_nax_scratch`); one buffer serves every
layer.

### 2.2 `kernel_qwen35_attn_flash_nax` — prefill, T > 8

- **Work split.** One threadgroup covers one KV head and 8 query tokens: 64 query rows (8 tokens x the 8 query
  heads of that KV head), four 16-row fragments of 2 tokens x 8 heads each, so every K/V fragment read serves all 8
  heads. The grid is `Hkv x ceil(T/8) x Ks`.
- **Matrix work.** K and V are read directly from the F16 cache (row stride `Hkv*D`) in blocks of 32 keys.
  S = Q Kᵀ and O += P V use `matmul2d` through 16x32x16 fragments, as MLX does; S, O, the running max and the
  running sum are F32, and P is converted to F16 for the P V product (as the M5 flash stages it).
- **Softmax.** Online softmax in registers with `exp2` and the scale folded into the packed queries (log2e folded
  as MLX does). Causal mask: query token t (absolute position pos0 + t) attends to keys [0, pos0 + t]; only blocks
  that straddle the diagonal are masked, and blocks entirely past a fragment's last query are skipped.
- **Two variants, chosen by the micro-benchmark:**
  - **A1**: each simdgroup owns all 256 output dims of its 16 rows (128 F32 accumulator values per thread), as MLX
    does at head dim 128 with twice the width. Simplest; may spill registers.
  - **A2**: two simdgroups share 16 rows. Each computes S over its own 128-dim half, the halves are summed through
    threadgroup memory, both run the (identical) softmax, and each accumulates P V for its own 128 output dims
    (the D split M5's KT=8 flash instance uses in simdgroup form).
- **Short chunks: key split.** Keep M5's flash split rule (split when too few threadgroups fill the GPU) and M5's
  partial layout `(m, l, O)`, so `kernel_qwen35_attn_merge3` combines the partials unchanged. Neutral partials
  (no keys in a slice) write l = 0 as in M5.
- **Epilogue.** Normalise by the running sum, multiply by the sigmoid output gate and write `out[t][H*D]` F32,
  exactly the M5 flash contract.
- Supported: head dim 256, GQA group 8, F16 K/V. Anything else keeps today's kernels.

### 2.3 Dispatch and knobs

- A prefill chunk uses the accelerator path when T > 8, `DS4_QWEN35_ATTN_FLASH` is on, the K/V mode is F16,
  `ds4_gpu_tensor_api_available()` is true (false under `--quality`, before M5/A19 and with
  `DS4_METAL_DISABLE_METAL4=1`) and the accelerator pipelines exist. Otherwise the M5 simdgroup flash runs,
  unchanged.
- New knob `DS4_QWEN35_ATTN_NAX`: `0` selects the M5 simdgroup flash. Read once, like `DS4_QWEN35_ATTN_FLASH`; an
  unrecognised value warns and keeps the default. It starts off and flips to on after the adoption checks (M4
  rule: gate 1, the MTP identity tests and a lever A/B that shows a gain).
- If the micro-benchmark shows the accelerator kernel losing on short chunks, a minimum T is added to the dispatch
  rule with the measured crossover as its value (a constant, not a new knob).
- The packed-query scratch (2.1) is counted in the Ornith memory estimate in `ds4.c`. The key split keeps the
  graph's `attn_flash_part`; its bound `ds4_gpu_qwen35_attn_flash_part_floats` grows to cover 8-token blocks
  (worst case ~34 MB instead of ~17 MB). No silent fallback inside a forward: a failed dispatch returns false.

## 3. Data flow and exactness

- K/V are written by the attention prep exactly as today; the cache layout (`[pos][kv_head][256]` F16) and the
  disk-KV payload are unchanged, so rewind, snapshot and restore are unaffected.
- The accelerator kernel is not bit-identical to the simdgroup flash (different accumulation order inside the
  accelerator's products). `--mtp` and plain runs prefill identically (same chunking, same kernels), so prefill numerics only
  need gate 1's tolerance; decode exactness rests on `decode3`, which M6 does not touch.
- The MTP layer's own attention uses the same prefill kernels; it only shapes drafts, never output.
- Known risk: on M5 the `matmul2d` accumulate path has been observed to drift with context length (memory note
  "M5 tensor-drift landmine"; the Ornith MoE accelerator kernels already run in prefill and pass gate 1). The
  kernel test compares against the simdgroup flash at 30K keys, and gates 1 and 2 decide adoption.

## 4. Testing and measurement

- Kernel tests (`tests/test_qwen35_kernels.c`, model-free), `test_attn_flash_nax`:
  - vs a host double reference with the causal mask, within 1e-3 relative (the M5 flash measured <= 3.4e-4):
    (pos0, T) in {(0, 9), (0, 65), (37, 65), (37, 200), (4096, 64)}, T not a multiple of 8 included, plus forced
    key splits (4096, 64), (8000, 200), (1000, 1100);
  - vs the M5 simdgroup flash on the GPU at (30720, 2048), within 2e-3 relative (max absolute error over max
    absolute output), to catch long-context drift;
  - skipped with a printed notice when the tensor API is unavailable; `make test-qwen35-kernels` runs with and
    without `DS4_METAL_DISABLE_METAL4=1`.
- Micro-benchmark `tests/bench_qwen35_attn.c`: the accelerator kernel next to flash TOK=2 at pos 0 / 30720 /
  122880 with T = 2048, and at T = 128 for the short-chunk crossover.
- **Prototype stop rule (first task):** the unsplit A1 kernel must run at least 2.5x faster than flash TOK=2 at both
  30720 and 122880. If A1 misses, build A2; if A2 also misses, stop and report. The unfused approach (accelerator
  GEMMs + softmax, option B in the brainstorm) is then a separate decision for the user.
- Model checks (controller GPU windows): gate 1 at chunks 2048/64/65 with the accelerator path on; the fallback
  identity check (criterion 5); MTP identity (`test_qwen35_graph`, `test_qwen35_mtp`, `tests/ornith/test_mtp_cli.py`);
  payload/rewind tests.
- Macro: `DS4_QWEN35_PROFILE` before/after; lever A/B with `m4_ab.py --mode lever` at 32K/128K and cold ~31K;
  final gate 3 vs the live oMLX; gate 2 with the live oMLX on the same harness; Qwen full gate. Receipts in
  `speed-bench/ornith/m6/`.

## 5. Risks

- Register pressure at head dim 256 (A1): A2 halves the accumulator; the stop rule bounds the effort.
- `matmul2d` with F32 P against F16 V, fragment layouts and direct device loads: follow MLX's `nax.h` and the
  repo's accelerator MoE kernels; the kernel tests pin the numerics.
- Long-context drift of the accelerator accumulate path on M5: if gate 1 or gate 2 fails with the accelerator path
  on, `DS4_QWEN35_ATTN_NAX` defaults to off and the report says so.
- Even a fast enough attention kernel may leave prefill short of the live oMLX if the non-attention part (~13 s at
  32K) dominates; the report then states the remaining gap without widening scope.

## 6. Constraints (from the parent spec and M1-M5)

Qwen3.8 byte-identical and as fast; Metal only for Ornith; `DS4_QWEN35_*` knobs only; kernel changes additive in
`metal/qwen35.metal` (never `metal/qwen4.metal`); one model process at a time; never `kill -9` a Metal process;
GPU windows pause and restore the live stack; English code/docs/commits; no merge/push without the user's OK.
