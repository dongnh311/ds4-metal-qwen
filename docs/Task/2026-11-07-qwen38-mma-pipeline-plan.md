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

## Phase 1 — gating microbenchmark — DONE

`tests/bench_mma_pipeline.m` (standalone, compiles inline Metal source)
on the mid-GEMM tile shape, fp16 in / fp32 acc, all variants bit-exact
(`bad=0` on the all-0.5 input oracle):

| variant | T=256 | T=1024 |
|---|---|---|
| v0 matmul2d + tg staging (production structure) | 2.60-2.83 TF | 3.24-3.37 TF |
| v1 same staging + simdgroup_matrix mma | 1.97-2.46 TF | 2.95-3.10 TF |
| v2 simdgroup_matrix + direct device fragment loads, no staging | 4.31-5.05 TF | 5.91-5.93 TF |
| v3 double-buffered staging + pipelined simdgroup mma | 1.84-1.88 TF | 1.97-2.01 TF |
| v4 v1 with NK=64 K-steps | 1.88 TF | 1.97 TF |

Findings:

- v1 ~ v0 minus ~10%: `simdgroup_matrix` is not faster than `matmul2d`
  under identical staging — the cooperative op is not the problem.
- v3/v4 much slower: extra threadgroup memory halves occupancy; the
  barrier skeleton cannot be pipelined away because
  `threadgroup_barrier` is all-thread and `simdgroup_multiply_accumulate`
  does not stay in flight across it.
- v2 is the only winner (~1.8x) and it wins precisely by eliminating the
  stage->barrier->load chain: each simdgroup streams fragments straight
  from device memory.  It is unreachable for packed weights — dequant
  must materialize fp16 somewhere, and threadgroup staging (v1) costs the
  same barriers as v0 while device scratch costs more DRAM traffic than
  the MMA win (quantified in the expert-quant project).

**Verdict: no-go.** The staging barrier is structural for any packed
format; the only escape is fp16-resident operands, which the
quantization project shows loses on DRAM traffic. Both candidate projects
converge on the same wall from opposite directions.

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
