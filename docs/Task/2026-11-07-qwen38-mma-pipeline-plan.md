# Qwen3.8 routed-MoE: custom MMA pipeline project

Branch: `feature/qwen38-mma-pipeline` (worktree `qwen38-pipe`), based on
`feature/qwen38-2bit-mma` @ 0e562811.

## Why this project exists

`docs/Task/2026-11-06-qwen38-mma-report.md` established the ceiling of the
current `kernel_qwen4_moe_mm_{mid,down}_nax_t` pipeline (dequant -> threadgroup
-> `matmul2d` cooperative):

- Tensor MMA ~36% of kernel time, already ~19 TFLOPS effective on M5 Pro.
- Skeleton (barriers + K-loop) ~34%, staging ~30%.
- `matmul2d` is a cooperative blocking op: staging and MMA cannot be assigned
  to independent warps; the ping-pong K-loop spike proved `mm.run` serializes.
- Weight repack measured neutral; packed 2-bit dot is infeasible for IQ2_S.

The only remaining structural lever is to bypass `matmul2d` and drive
`simdgroup_matrix` ops directly, where the compiler sees plain per-threadgroup
code and may overlap dequant/staging with MMA across K-steps.

## Hypothesis

A hand-rolled pipeline can beat `matmul2d` if either:

1. Raw `simdgroup_matrix` multiply_accumulate reaches comparable TFLOPS while
   allowing register-resident double-buffered A/B fragments (no per-step
   threadgroup barrier for MMA operands), or
2. Dequant can feed `simdgroup_load` fragments straight from registers,
   skipping the fp16 threadgroup round-trip entirely.

Prior evidence against: the legacy `metal/moe.metal` simdgroup path measured
~3x slower than the tensor path — but that was the DeepSeek V4 MoE kernel with
a different tiling/staging structure, not a controlled comparison.

## Phase 1 — gating microbenchmark (cheap, do first)

Standalone `tests/bench_mma_pipeline.m`: compiles an inline `.metal` source at
runtime (no Makefile plumbing for the spike), allocates fp16 A/B/C buffers,
and times three variants on the routed-GEMM tile shapes:

- `mid`:  C[32 tokens][64 rows] += A[64][2560] * B[2560][32]  (per expert,
  one token tile; ~80 K-steps of NK=32)
- `down`: C[32][2560] += A[2560][640] * B[640][32]

Variants:

- V0 `matmul2d`: current structure — threadgroup stage +
  `matmul2d<descriptor(NR1,NR0,NK,false,true,false,multiply_accumulate),
  execution_simdgroups<4>>`, 2 barriers/K-step.
- V1 `simdgroup_matrix`: each simdgroup owns an 8x8 (or 16x16 via 4 sims)
  accumulator tile; A/B fragments loaded per K-step into registers;
  `threadgroup_barrier` only around staging, same as V0.
- V2 `simdgroup_matrix` + register double-buffer: dequant/stage K+1 into
  registers while MMA on K runs (only possible because there is no
  cooperative op forcing serialization).

Decision gate: if V1 cannot reach >=80% of V0's TFLOPS on pure fp16 data,
the project stops — `matmul2d`'s tensor scheduling is not beatable by hand.
If V1 ~ V0, V2 decides whether overlap buys the ~30% staging share.

## Phase 2 — fused dequant variant (only if Phase 1 passes)

Replace the fp16 A staging with IQ2_S decode straight into
`simdgroup_matrix` operand fragments. Correctness oracle:
`tests/test_qwen4_moe_mm_specialize` + `tests/test_qwen4_kernels` already
cover the packed formats bit-exactly vs CPU reference.

## Acceptance criteria

- Isolated mid+down bench (`tests/bench_qwen4_moe_mm`) improves >=10% at
  T=1856 vs the 21.3 ms baseline, bit-identical or NLL-validated outputs.
- Full-model MoE stage improves; decode unchanged (kernel is prefill-only).
- No new permanent semantic flags; diagnostic env knobs allowed during
  development.

## Risks

- `simdgroup_matrix` fragment layouts are fixed by Metal; dequant output must
  match `simdgroup_load` expectations — mismatched layouts silently corrupt.
- Register pressure: IQ2_S dequant needs the 82-byte superblock metadata;
  per-thread register caching was already measured occupancy-negative once.
- If Metal schedules `simdgroup_matrix` ops on the same tensor pipes as
  `matmul2d`, gains may cap at barrier removal only (~few %).
