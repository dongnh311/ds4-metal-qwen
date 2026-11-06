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
