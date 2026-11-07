# Qwen3.8 2-bit MoE kernel project - session report

Branch `feature/qwen38-2bit-mma` (worktree `qwen38-mma`), 13 commits on
top of `develop@3b6b3445`. All numbers: Apple M5 Pro, synthetic bench
`tests/bench_qwen4_moe_mm` (E=512, S=10, D=2560, F=640, O=2560),
ds4-server stopped during measurement.

## Headline results

- Clean baseline restored: mid 13.9 + down 7.4 = **21.3 ms** @T=1856
  (42.4 ms @T=4096). Earlier "13.7 ms" numbers were an artifact - see bug.
- Full-model MoE stage @T=1856 (48 layers, DS4_QWEN4_MOE_PROFILE):
  total ~910 ms; routed mid 427 (47%), routed down 253 (28%),
  reduce 79, router 76, shared expert 68, lists 9.
- Bench mid+down (~21.3 ms x 48 ~ 1.02 s) tracks the real routed stage
  (~680 ms) closely enough for A/B kernel work.

## Critical bug found and fixed

`DS4_QWEN4_MOE_MM_NOMMA` used `ds4_gpu_env_bool`, which returns -1 when
the variable is unset; cast to uint32 that became 0xFFFFFFFF and silently
disabled tensor MMA in EVERY benchmark after the diagnostic landed.
All post-diagnostic "fast" figures were the no-MMA path. Fixed to parse
the mask as an integer; normal path (unset/0) is bit-identical.

## True kernel decomposition (skip-mask, T=1856, mid/down ms)

    mask  mid   down   marginal
    0     13.9  7.4    baseline 21.3
    1(MMA) 7.2  6.4    MMA       ~6.7 / ~1.0
    2(W)  12.4  4.9    weights   ~1.5 / ~2.5
    4(B)  11.5  5.5    B stage   ~2.4 / ~1.9
    7(all)5.6  1.7    skeleton (barriers+loop+staging stores) ~7.3 total

=> barrier/sync skeleton ~34%, MMA ~36%, weight+B staging ~30%.
MMA is near tensor-core roofline for mid (~6.4 ms for ~120 GFLOP).

## Spikes tried (all measured, losers reverted)

| spike | result |
|---|---|
| NK=64 K-step | slower (24.9 vs 21.8 true-path) - reverted |
| fused stage16 dequant IQ2_XXS/IQ2_S | bit-identical, perf-neutral - kept |
| IQ2_S raw16 register prefetch | was missing on prod path; neutral - kept |
| raw16 prefetch depth-2 | slower - reverted |
| threadgroup super-block weight cache | +12KB tgmem hurt occupancy - reverted |
| ping-pong A/B K-loop (1 barrier/step) | byte-exact BUT ~3% slower (mm.run synchronous, staging section is critical path) - reverted |
| expert-major tile order | neutral, kept as env option |

## Direction (a) packed 2-bit dot: INFEASIBLE

IQ2_S/IQ2_XXS are vector-quantized (8-dim codebook lookup per group), so
the dot cannot collapse into counters; it reduces to dequant+FMA which
the simdgroup path already does and is ~3x slower than tensor cores.
No competitive scalar/LUT dot exists on this hardware for these types.

## Direction (b) weight repack: viable, bounded ~10-20%

Design (pure permutation, zero size overhead for superblock types):
`[row_block=64][block][row_in_block][block_bytes]` - e.g. IQ2_S gives a
contiguous 5248B region per (row-block, superblock) vs today's ~52KB
scattered window per K-step. Kernel row-block tiles (NR0=64) map 1:1.

Work items: stride-parameterized dequant (stride=0 keeps all current
callers), repacked row pointer + wstride in nax_t kernels, host flag +
expert_bytes, offline GGUF permutation tool (or free repack inside the
SSD-streaming path). Realistic ceiling ~100-200 ms of the ~910 ms MoE
stage - the barrier/dequant/MMA remainder is untouched.

## Recommendation

Cheap kernel levers are exhausted; every remaining option is a project:
1. weight repack (~10-20%, well understood, medium effort),
2. warp-specialized async tensor pipeline (large R&D, uncertain),
3. attack the other ~25% of MoE stage - router 76 + reduce 79 +
   shared-expert 68 ms - which no spike has touched yet.

## Addendum: untouched MoE components (reduce/router/shared)

kernel_qwen4_moe_reduce reads `part` = [T][S+1][dim] fp32 written by the
down kernel: ~208 MB per layer, ~10 GB per prefill chunk of pure
write+read traffic (~25-50 ms of the measured 79 ms reduce stage is
DRAM-bound floor). The kernel itself is already coalesced; the remaining
cost is the intermediate buffer. Candidate (unverified): store `part` as
fp16 + error compensation like the midh/midr path, or fuse the weighted
reduce into the down epilogue - both touch logits accuracy and need the
full correctness suite. Router (76 ms) and shared expert (68 ms) not yet
profiled at kernel level.

## Addendum 2: reduce kernel float4 variant (kept, small win)

`kernel_qwen4_moe_reduce_v4` (dim % 4 == 0) gives each thread four
consecutive dims via float4 loads/stores; per-element accumulation order
is unchanged so output is bit-identical. Scalar kernel remains for odd
dims and via DS4_QWEN4_MOE_REDUCE_SCALAR=1.

Measured with tests/bench_qwen4_moe_reduce (T=1856, dim=2560, S=10,
hc=4, shared slot): v4 ~1.68 ms vs scalar ~1.70 ms per call (~1-2%).
The kernel is DRAM-bound (~230 GB/s effective on the part buffer), so
the ceiling for load-width tuning is low - confirmed: hc=0/4/8 and
shared_src 1/2 all within noise-to-6%.

Full-model check (UD-IQ2_XS-mtp.gguf, T=1928, DS4_QWEN4_MOE_PROFILE,
47 logged layers): reduce stage 79.0 ms with v4 vs 81.0 ms scalar
(~2.4% of the stage, ~2 ms absolute). test_qwen4_kernels and
test_qwen35_kernels pass; reduce+combine output matches the double
reference at 2e-5.

Remaining reduce cost is the fp32 `part` intermediate itself (~208 MB/
layer write+read). Halving it needs fp16 part storage - a semantic
change to every down-kernel epilogue writer plus NLL validation; parked
behind the repack project which attacks 75% of the stage instead.

## Addendum 3: router stage decomposition (new bench + full-model)

`tests/bench_qwen4_router` times kernel_qwen4_router_topk alone:
~0.54 ms/call at T=1928, NE=512, NU=10, dim=2560, F32 gate row
(~25 ms over 47 layers).  The profiled "router" stage is ~78 ms, so
~53 ms is the router-logits GEMV, not top-k.

ffn_gate_inp.weight is F32 [512,2560] (5 MiB/layer, verified via
llama-gguf): qwen4_gemv_rows sends it to kernel_qwen4_dense_mm
(fp32 simdgroup_float8x8 tiles, ~4.6 TFLOPS effective, n_split=1 at
976 threadgroups).  F32 is the one dense type with no tensor-op NAX
variant (F16 exists but is env-gated off by default; GSQ types use
kernel_qwen4_dense_nax_*).  A tensor path for the router GEMM means
fp16 operands -> logit rounding ~1e-3 -> rare top-k flips: a semantic
change needing the NLL suite, same class as the fp16 `part` idea.

Shared expert (~73 ms: shared_mid 47.6 + shared_down 25.8) runs dense
Q8_0 projections visible in DS4_METAL_Q8_PREFILL_PROFILE output
(e.g. in=2560 out=640 ~0.6 ms, in=640 out=2560 ~0.5-1.0 ms per layer).

Updated MoE-stage map @T~1900 (47 layers):
  routed mid ~724 + routed down ~404 (repack target, ~75%)
  reduce ~79 (v4 done, ~77)
  router ~78 (GEMV ~53 + topk ~25)
  shared ~73, lists ~9

## Addendum 4: F32 router-logits GEMV on tensor tiles (implemented)

The ~53 ms router-logits GEMV (`ffn_gate_inp` F32 [512x2560]) was the last
dense type without a tensor-op variant.  Added `kernel_qwen4_dense_nax_f32`
(+_n64/_n128) in metal/dense.metal: F32 rows stage through `dequantize_f32`
into half4x4 threadgroup tiles while activations and the accumulator stay
float - the same weight-only fp16 rounding class as the existing NAX level-2
path.  `qwen4_dense_nax_kernel` maps type 0; `qwen4_dense_nax_ok` gates it
behind `DS4_QWEN4_DENSE_NAX_F32` (default on) plus the usual
`DS4_QWEN4_DENSE_NAX`, MPP, and shape checks (in%64, out%64, T%32).

Prefill lengths are arbitrary, so `ds4_gpu_qwen4_dense_mm_tensor` splits
F32 calls whose token count is not a multiple of 32: the aligned prefix runs
on the tensor kernel and the <32-token tail recurses through tensor views
into the existing scalar/k-split path (same pattern as the attention output
projection's matrix_rows/tail split).

Measured (UD-IQ2_XS-mtp.gguf, T=1928, DS4_QWEN4_MOE_PROFILE, 47 layers):
router stage 77.4 ms -> ~33 ms steady state (~0.70 ms/layer vs ~1.66;
the first layer pays ~106 ms one-time shader compile).  mid/down/reduce/
shared/lists unchanged within noise.  Roughly -45 ms per ~2k-token chunk.

Numerics: NAX vs scalar dense_mm agrees to max|d| ~1.5e-3 (fp16 weight
rounding; new test_dense_mm_f32_nax covers T=64 pure-tile and T=70 split).
Full-model --dump-logits A/B (F32 NAX on vs DS4_QWEN4_DENSE_NAX_F32=0,
all other NAX types identical - verified bit-identical baseline): final
logits max|d| 2.31, argmax unchanged (id 1596), top-20 overlap 18/20.
Perplexity on the 1870-token profile prompt: NLL 3739.96 (ppl 7.651) with
the F32 tensor path vs 3762.23 (ppl 7.744) scalar - no regression on this
sample.  Expert top-k flips are possible in principle (logit deltas ~1e-3
vs logit gaps); the NLL result says the aggregate effect is neutral-to-
positive here, but this is one text sample - treat as validated-for-prefill,
not proven for all inputs.

tests/test_qwen4_kernels and tests/test_qwen35_kernels pass.
