# Qwen3.8 2-bit MoE kernel project — plan (2026-11-06)

Branch: `feature/qwen38-2bit-mma` · Worktree: `orca/workspaces/ds4-metal/qwen38-mma`
Source: `docs/QWEN38_KERNEL_ARCH_BACKLOG.md` — incremental tuning is exhausted;
MoE is ~75% of prefill time and compute/dequant bound.

## Goal

MoE stage ≤ 1000 ms at T=1856 (baseline 1438 ms), prefill ≥ 750 t/s @9.4k,
decode ≥ ~45 t/s, ≤ 60 GiB planned @256k ctx. No regression for other
qwen4/qwen35 models, SSD streaming, or CUDA paths.

## Phase 0 — measurement harness (before any kernel work)

- [ ] Reproduce the stage breakdown on this tree: `ple / hc_attn / gdn / attn /
      hc_ffn / moe` at T=116/296/956/1856 so every later A/B has a baseline.
- [ ] Kernel-level reference: extend `tests/test_qwen4_kernels.c` with a
      scalar reference for whatever new packed-dot path lands (compare vs the
      existing dequant→f16→dot result bit-tolerant).
- [ ] Bench scripts for MoE-only timing (prefill chunk sweep) committed under
      `speed-bench/` or `misc/` so A/B is one command.

## Phase 1 — spike: cheapest viable 2-bit consumption (days, not weeks)

Question to answer first: where exactly do cycles go inside the IQ2_S expert
kernels? Profile `kernel_qwen4_moe_mid`/`down` (and the GSQ `stage16` variants)
with Metal System Trace / Xcode GPU counter capture on the M5 Pro.

Candidate spike A — **in-register dequant + f16 MMA fragment** (backlog dir 3):
keep the existing lane structure but dequantise 2-bit → f16 inside the thread
and feed `simdgroup_matrix` ops directly, skipping the f16 tile materialisation
in threadgroup/global memory. Smallest change that attacks "expand bits into
f16 for tensor cores".

Candidate spike B — **integer dot + per-block scale** (backlog dir 1, dp4a
analogue): dot packed 2-bit codes against quantized activations, apply the
IQ2_S block scale/delta in f32. Only pursue if profile shows dequant (not MMA
issue) dominates.

Deliverable: one kernel variant behind the existing diagnostic-switch
mechanism, `tests/test_qwen4_kernels.c` reference passing, MoE timing at
T=1856 vs baseline. Keep or discard by measurement — same discipline as the
three reverted attempts.

## Phase 2 — whichever spike wins, widen it

- Wider k-tiles / deeper double buffering (backlog dir 2) on top of the new
  inner loop.
- If MoE is still bandwidth-bound after compute is cheap: expert hotlist /
  activation-frequency layout reusing the SSD-streaming machinery (dir 4) —
  separate follow-up, do not bundle.

## Phase 3 — hardening

- Deterministic suite: structured/toolcall/HE/GSM8K temp-0, plus logits-drift
  explanation if any.
- Memory gate: ≤60 GiB planned @256k.
- Regression sweep: other qwen4 quants (IQ2_XXS, Q2_K, Q2_0 share the path),
  qwen35 models, SSD streaming, CUDA — ask user before touching distributed.
- Rollback: single kernel dispatch switch; no format changes to GGUF.

## Explicit non-goals

- No GGUF format changes, no new tensor types.
- No permanent semantic flags — diagnostic switches only during dev.
- No CPU-path changes (reference/debug only).

## Findings so far (M5 Pro, synthetic bench T=1856, E=512 S=10)

- mid+down kernel pair: ~21.8 ms/layer-call at T=1856 (IQ2_S mid + IQ4_NL
  down); ~43.5 ms at T=4096; weak scaling with T → per-expert fixed cost
  dominates, not token-parallel work.
- NAX=1/2 tile knobs and mid/down tile counts: ±5% noise. Tensor-op
  64-token tiles are already the default winner.
- IQ2_S vs IQ2_XXS vs MXFP4 mid timings are within ~20% despite very
  different dequant cost → kernels are NOT dequant-bound; suspect
  MMA/barrier structure or weight-load latency.
- NK=64 K-step variant (built, env-gated): 24.9 ms — SLOWER, reverted.
  Larger tgmem (24KB) likely cut occupancy.
- Fused stage16 dequant for IQ types kept: bit-identical, perf-neutral.
- Next: Metal System Trace profile of the two kernels to split
  load/dequant/MMA/barrier time before choosing spike A/B.

## Profiling results (CORRECTED - see skip_mma fix)

- DS4_QWEN4_MOE_MM_NOMMA diagnostic: env_bool returns -1 when the var is
  unset, which cast to 0xFFFFFFFF and silently disabled MMA in all runs
  after the diagnostic landed. True T=1856 baseline: mid 13.7 + down
  7.4 = 21.1 ms. With MMA skipped: 13.7 ms total → MMA ~7.3 ms (~35%).
  Split per kernel: mid MMA ~6.4 ms (~47%, two matmul2d per K step),
  down MMA ~0.9 ms (~12%). Staging still dominates down (~88%) and is
  ~half of mid. Earlier "MMA ~5%" conclusion was an artifact of the bug.
- Clean baseline T=1856 (earlier 21.8 ms was contaminated by the running
  ds4-server stealing GPU): IQ2_S/IQ4_NL = 13.7 ms, IQ2_XXS/Q2_0 = 11.0,
  MXFP4 = 14.8, Q8_0 = 35. NAX=0 simdgroup = 45.7 (tensor path correct).
- tiles_per_launch / tile-count knobs: no effect (tiles~1 per expert at
  these shapes; weights are already read once).
- Added register prefetch (raw16) for IQ2_S type 22 — bit-exact
  (specialize suite all PASS) but perf-neutral: 13.80 vs 13.74 ms.
  One-step prefetch does not cover DRAM latency; the loads are also
  scattered across rows (uncoalesced) every K step.

## Diagnosis

Kernel is latency-bound in the staging pipeline, not MMA- or
dequant-bound: ~128 threads issue small scattered loads per K step,
2 barriers per step serialize, and only ~1 K-step of prefetch hides
~1us of memory latency. nk=80 steps for mid.

## Next spike (recommended)

Coalesced two-phase weight staging: per K-group, cooperatively copy the
expert row's packed bytes into threadgroup memory with consecutive
threads reading consecutive addresses (fully coalesced), then dequant
from tgmem into the f16 A slab. This removes scattered-load latency
from the critical path and is the concrete form of backlog dir 3
("expand in-register / avoid f16 materialization cost" being wrong —
the cost is the load pattern, not the dequant math).
Alternative cheaper probe: deepen raw16 prefetch to 2-4 K-steps and
measure whether latency hiding improves at all before committing to
the tgmem restructure.

## E-scaling probe (T=1856, IQ2_S/IQ4_NL, clean)

E=64: 8.4 ms | E=128: 12.7 | E=256: 14.7 | E=512: 15.9 — sub-linear,
saturating. Effective weight bandwidth ~35-70 GB/s, far below peak:
consistent with memory-level-parallelism/latency bound, not raw BW.

## Conclusion for spike selection

- MMA ~5%, dequant math cheap, prefetch distance 1 insufficient.
- The staging loop issues ~6 small scattered loads/thread/K-step with
  only one step of lookahead; DRAM latency ~1us >> per-step time.
- Winning move is restructuring weight staging for MLP: either
  (a) per-thread super-block register caching (82B/row/8 steps, needs
  careful byte extraction to avoid local-memory spills), or
  (b) cooperative coalesced tgmem staging of packed rows then dequant
  from tgmem (needs ~10KB extra tgmem; mid kernel at 16KB may fit in
  32KB budget).
- Both are real kernel projects; (a) is smaller and keeps the current
  tile shape. Recommend (a) first: load the 82-byte IQ2_S super-block
  (or equivalent per-format block) once per 8 K-steps into registers
  as uint4s, index bytes via switch/unrolled ib32.

## Spike: IQ2_S super-block window cache (reverted)

Built a threadgroup cache of packed IQ2_S super-blocks: per 8 K-steps,
each row's 82-byte block is fetched as 6 unaligned uint4 loads into
12 KB of extra threadgroup memory, then dequantized from tgmem.
Correct (specialize suite passed while enabled) but SLOWER:
mid 7.9 vs 7.3 ms — the +12 KB threadgroup budget likely cut occupancy
and per-step tgmem byte reads added LSU pressure. Reverted; env-gated
code removed entirely.

## Where this leaves the project

Every cheap staging variant is now measured and rejected: wider K
(NK64), deeper register prefetch, fused dequant, tgmem super-block
cache. The kernel's ~95% non-MMA time is bound by the serial
stage->barrier->MMA pipeline at only ~1-2 K-steps of memory-level
parallelism per thread, and occupancy is already tgmem-limited.

Remaining directions, in expected-value order:
1. Weight layout change (dir 4): repack expert rows so a K-step's
   bytes for all 64 rows are contiguous — makes staging loads truly
   coalesced. Requires GGUF-side or load-time repack; big but is the
   only change that attacks the actual bottleneck (scattered loads).
2. True 2-bit MMA (dir 1): bypass f16 staging entirely; large project,
   uncertain payoff given MMA is only ~5% today — the win would have
   to come from eliminating staging, not from faster MMA.
3. Full-model A/B: quantify how much of the 1438 ms MoE stage is these
   two kernels vs routing/shared/reduce before more kernel work.

## Full-model MoE breakdown (ds4-bench, T=1856, worktree build, no steering)

Per-layer "Qwen MoE ms" profile summed over 48 layers = 910 ms:
- mid (routed gate+up GEMM): 427 ms (47%)
- down (routed down proj):   253 ms (28%)
- reduce:                     79 ms (9%)
- router:                     76 ms (8%)
- shared_mid+shared_down:     68 ms (7%)
- lists:                       9 ms (1%)

Routed mid+down = 680 ms = 75% of the MoE stage, and the synthetic
bench (13.55 ms) matches the real per-layer cost (14.2 ms) closely —
the standalone harness is representative. Prefill measured 633 t/s.

Implication: kernel work on mid+down is correctly aimed (75% of stage),
but even a 30% kernel win yields only ~-200 ms of MoE. Reaching the
<=1000 ms target likely ALSO needs router+reduce (~155 ms combined)
attention, or the weight-layout repack that makes staging coalesced.
