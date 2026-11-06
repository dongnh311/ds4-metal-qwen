# Qwen3.8 kernel-architecture backlog (2026-11-06)

**Status: incremental tuning is exhausted. The next step is a separate kernel project,
not another one-off patch.**

Measured on the production box (Apple M5 Pro, 64 GB), Qwen3.8-Flash-Next UD-IQ2_XS
(50.3 GB, IQ2_S experts, hc/attn mixed, embedded MTP), ctx 262144, prefill-chunk 4096,
directional steering at ffn 0.5.

## Baseline (ds4-server, this tree)

Prefill stage ms at T=1856: ple 13 · hc_attn 175 · gdn 631 · attn 328 · hc_ffn 204 ·
**moe 1438** (≈2.9 s total). MoE scales with tokens: 404 ms at T=116, 536 at T=296,
930 at T=956, 1438 at T=1856. End-to-end: 645 t/s at 9.4k, 910 t/s at 31k, decode ~45 t/s.

## Incremental attempts, all measured and rejected

1. **Q2_K expert requantisation** (80 IQ2_S → Q2_K to reach the raw MM path):
   MoE 1437 vs 1440 ms — no change; decode collapsed 40-45 → 22 t/s. Reverted.
2. **Tiled Q8 dense MM beyond 32 rows** (`qwen4_gemv_rows`): gdn 645 → 1675 ms,
   decode 23.6 t/s. The per-token Q8 kernel is already better at large T; the batch
   kernel re-reads each matrix per 16-row tile. Reverted.
3. **IQ2_S raw-16 staging** in `qwen4_load_raw16`/`qwen4_dequant_raw16` (16 values per
   load, shared db for the even q0): MoE 1438 → 1690 ms, prefill 580 → 541. Packing into
   registers costs more than the saved header reads. Reverted.

None of these patches are in the tree. The one kept change from the same work is a
**GGUF compaction** (drops dead append regions: 0.73 GiB less resident), which is not a
kernel change.

## Why further gains need an architecture change

- MoE is ~75% T-dependent: it is compute/dequant bound, not dominated by the one-off
  weight read seen at small T.
- The GSQ `stage16` path is already the better trade for IQ2_S; forcing raw staging
  raises register pressure and predication.
- For Q8 dense, per-token beats the 16-row batch kernel at large T.
- The model is 2-bit MoE: dequant-to-f16 + f16 MMA spends most cycles expanding bits
  into a format the tensor cores can consume.

## Candidate directions (each project-sized)

1. **2-bit MMA / bit-sliced dot.** Consume packed 2-bit weights directly (dp4a-style
   integer dot or a simdgroup matrix path that unpacks 2-bit into the MMA fragment),
   avoiding a full f16 materialisation of expert tiles. Applies to IQ2_S, IQ2_XXS,
   Q2_K and Q2_0.
2. **Wider tiles / deeper double buffering.** Stage 2-4× more k per weight read, more
   rows per threadgroup, fewer barriers and partial-plane reductions.
3. **Remove dequant-to-f16 for experts.** Keep activations in f16 but dequantise into
   the MMA operand (or use integer dot + per-block scale) so weights are read once and
   expanded in-register.
4. **Bandwidth path.** If MoE stays bandwidth-bound after the above: persistent expert
   hotlist / layer streaming (SSD streaming already has the machinery) and an
   activation-frequency layout for the expert blocks.

## Acceptance criteria

- **Correctness:** the deterministic suite (structured/toolcall/HE/GSM8K, temp 0) plus a
  kernel-level reference test in `tests/test_qwen4_kernels.c` against the scalar
  dequantiser; any logits drift must be explained, not accepted.
- **Speed:** target MoE ≤ 1000 ms at T=1856 and prefill ≥ 750 t/s at 9.4k on the same
  box, with decode not below the current ~45 t/s.
- **Memory:** ≤ 60 GiB planned at 256k context; no regression for other qwen4/qwen35
  models or the SSD-streaming and CUDA paths.
- **Process:** own branch/worktree, benchmark harness and A/B scripts, rollback plan.
  This is a project, not a tuning patch.

Cross-reference: the AI-Gateway report `reports/qwen38-iq2-local-2026-11-06/RESULT.md`
holds the full measurements and the reverted patches (`q8-tiled-mm.diff`, Q2_K scripts).
