# Qwen3.8 expert quantization/layout project

Branch: `feature/qwen38-expert-quant` (worktree `qwen38-quant`), based on
`feature/qwen38-2bit-mma` @ 0e562811.

## Why this project exists

The routed-MoE GEMM is bound by the dequant -> threadgroup -> `matmul2d`
pipeline structure (~84% of the MoE stage). All in-kernel levers are measured
and exhausted (see `2026-11-06-qwen38-mma-report.md`). The remaining lever is
the weight *format itself*: choose a quantization/layout that is cheaper to
consume on both prefill (tensor-core GEMM) and decode (GEMV) paths.

Hard constraints:

- Decode must stay >=45 t/s on M5 Pro (Q2_K requant previously dropped
  decode 40 -> 22 t/s — rejected).
- Prefill target: MoE stage <=1000 ms @T~1856.
- Model must remain GGUF-loadable (or an explicit sidecar/repack format with
  the same mmap semantics); SSD streaming of experts must keep working.
- No C++; correctness before speed; NLL/full-model validation for any
  semantic change.

## Candidate directions (evaluate in order)

### Q1 — SoA scale/code layout for IQ2_S (no requantization)

IQ2_S stores per-256-block: 2 fp16 scales + sign/bit-plane data interleaved
in GGUF order. The NAX kernel's dequant is not ALU-bound, but the *layout*
forces each thread to gather bytes from scattered offsets per K-step.
Repack expert tensors at load time (or offline into a sidecar GGUF) into:

  [superblock][field] planes: all scales together, all code bytes together

so each K-step's loads become contiguous per plane. This is *not* the same
as the rejected row-block repack (which permuted row order and measured
neutral): here the goal is per-thread load coalescing within a K-step and
fewer distinct cache lines touched per dequant call. Cheap to test: the
bench already has the permutation harness pattern from the repack spike.

### Q2 — IQ2_S -> custom "IQ2_T" tensor-friendly variant

Design a variant of IQ2_S where the 8-dim codebook lookup produces values
already in a layout `simdgroup_load`/`matmul2d` staging can consume without
the per-element gather. E.g., store codes transposed so a warp's 32 lanes
read consecutive bytes. Requires a new GGUF quant type id or a sidecar
repack; decode path needs a matching GEMV kernel.

### Q3 — Requantize experts to Q4_K / Q5_K for tensor-friendly dequant

Higher bits = simpler dequant (integer + scale), but +2 bits/weight on the
largest tensors (~+25% model size, worse SSD streaming, likely decode loss).
Only viable if prefill win is large; measure decode first — the Q2_K failure
mode (decode collapse) is the known risk.

### Q4 — Mixed precision: keep IQ2_S codes, widen scales to fp16 blocks of 32

Finer-grained scales may allow dropping the codebook (plain 2-bit symmetric
quant) — i.e., a *new* format closer to Q2_K but tuned for this kernel.
This is real quantization research: needs a calibration pass over the model's
expert weights and a quality eval (perplexity on a fixed prompt set).

## Phase 0 — measurement harness (first deliverable)

- `tools/qwen4_expert_repack/` (or `misc/` spike first): reads the GGUF,
  extracts expert tensors for one layer, applies a candidate layout
  permutation, writes a sidecar file + a manifest describing the transform.
- Extend `tests/bench_qwen4_moe_mm` (already supports type override + has
  the repack-permutation precedent) to consume a repacked tensor.
- Baseline numbers to beat: mid+down = 21.3 ms @T=1856 (IQ2_S mid type 22,
  down type 20), full-model MoE ~1243 ms @T=1928.

## Phase 1 — Q1 spike (SoA repack) — DONE, neutral, reverted

Implemented and measured on `tests/bench_qwen4_moe_mm` @T=1856/4096,
IQ2_S mid (type 22) + Q2_0 down (type 20):

- **K-step-major 12B chunks** (`DS4_QWEN4_MOE_KSTEP`, diagnostic only):
  each 32-value chunk stored as `[grid4|sign4|qh|sc|d]` so `load_raw16`
  costs two vector loads instead of six scattered byte reads; dequant
  math unchanged (bit-exact by construction). Row bytes 820 -> 960 (+17%).
- Result: mid 12.65-12.69 ms vs 12.67-12.70 baseline @T=1856;
  25.12 vs 24.96-25.05 @T=4096 — **neutral at both sizes, reverted**.

Together with the earlier row-block repack (also neutral), this closes the
layout question analytically: the per-(expert,row-block) working set is
cache-resident, so no permutation of the same bytes reduces traffic, and
the measured bottleneck is the barrier/MMA skeleton — not operand fetch.
Load instruction count and coalescing within a cache line do not matter.

## Phase 1b — format-level conclusion

Cross-referenced with the mma-pipeline spike
(`feature/qwen38-mma-pipeline`, `tests/bench_mma_pipeline.m`): the only
configuration that beat the production pipeline was simdgroup_matrix
loading fragments **directly from device memory** (~1.8x). That requires
fp16-resident operands. For expert weights that means ~3.4 GB/layer of
fp16 scratch vs ~0.5 GB packed — at measured effective bandwidths the
extra DRAM traffic costs far more than the MMA win. A quantization format
cannot escape this: any format consumable without staging must already be
in tensor-fragment-compatible numeric form, i.e. fp16/bf16 density.

Remaining theoretical option is a *new trained quantization* whose codes
map to fp16 grid values cheaply enough that dequant+stage fits the MMA
shadow — but the barrier skeleton (~34% of kernel) is paid regardless of
format, capping any such win well below the ~1.8x the v2 spike showed.

**Recommendation: close the quantization/layout direction.** Three
independent repack variants measured neutral; the format that could win
is a different model, not a different layout.

## Phase 2 — format design doc + prototype

For whichever direction survives: write the format spec (bit layout,
scale placement, dequant pseudocode for both NAX tile staging and decode
GEMV), prototype the Metal dequant in `tests/` against CPU reference,
then a GGUF-side writer.

## Acceptance criteria

- Isolated bench >=10% on mid+down, or a documented reason the format cannot
  reach it.
- Decode >=45 t/s unchanged (GEMV path must be implemented and measured).
- Full-model: MoE stage improvement + NLL within noise on fixed prompt set.
- SSD streaming: expert tensors still loadable on demand (mmap or explicit
  reads); memory <=60 GiB @256k ctx.

## Risks

- Any requantization changes model outputs -> needs the NLL suite; the
  project may end as "analysis only" if no format preserves quality.
- GGUF type ids are a shared namespace; a new type needs upstream-compatible
  numbering or a ds4-private sidecar (preferred: sidecar, keeps GGUF valid).
- Decode GEMV for a new format is a second kernel to write and validate.
