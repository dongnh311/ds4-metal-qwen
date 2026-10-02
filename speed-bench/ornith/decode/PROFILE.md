# Ornith decode Stage 0: where a decode step goes

Spec: `docs/superpowers/specs/2026-10-02-ornith-decode-design.md` §3. Plan A:
`docs/superpowers/plans/2026-10-02-ornith-decode-stage0.md`.

## Runs

- **Windows.** Window P (2026-10-02 22:10-22:18, `9c5175a6`) ran the identity check and the 2K runs.
  Window P2 (22:19-22:39, `8c2d96bb`) ran every context after a runner fix. P's first pass stopped on a
  bash 3.2 empty-array error.
- **CLI one-shots.** `./ds4 --metal -c 262144 --prefill-chunk 2048`, 256 tokens, chat mode on a
  `promessi_sposi.txt` prefix plus "Summarize the story so far", `--mtp` with the 64K draft vocabulary.
- **Raw logs** stay in the scratch windows. The parsed table is in `receipts/stage0/table.md`.

**Identity.** With every new env unset, the gate-1 dumps at chunk 2048 equal window M's develop dumps
(13/13). `test_mtp_cli` passes with `DS4_QWEN35_SPEC_STATS=1 DS4_QWEN35_SPEC_OVERLAP=1`.

## Method

- **Removing the profiler's sync cost.** The level-3 split syncs 81 times a step. The sync cost per mark
  is (profiled sum - unprofiled ms/step) / marks, and each bucket is corrected by its own marks.
  - The verify split also carries the overlap hook's extra sync in each MoE.
  - The cost comes out at 0.16-0.25 ms per mark.
  - Level 2 against level 3 at 2K gives the same order of cost: 0.15-0.21 ms per mark.
- **Cycle anatomy** comes from `DS4_QWEN35_SPEC_STATS` (wall time per cycle).
- **GPU busy** comes from `DS4_METAL_CB_TIMES`: decode buffers under 50 ms of GPU time, GPU time over GPU
  time plus gaps.

## Plain decode, ms per step, corrected

| context | rate | GDN mixer | GDN-layer MoE | attention mixer | attention-layer MoE | head | total |
|---|---|---|---|---|---|---|---|
| 2K | 70.0 t/s | 5.25 | 3.65 | 2.17 | 1.17 | 2.04 | 14.28 |
| 32K | 52.5 t/s | 4.83 | 3.22 | 7.37 | 1.23 | 2.40 | 19.07 |
| 128K | 34.1 t/s | 6.15 | 4.46 | 14.94 | 1.78 | 2.03 | 29.36 |

Against the byte floor at ~290 GB/s:
- MoE reads ~0.76 GB per step, a floor of 2.6 ms; at 2K it runs at ~55% of it.
- The GDN projections read ~1.07 GB, a floor of 3.7 ms; ~70%.
- The attention projections plus K/V read ~0.31 GB at 2K, a floor of 1.1 ms; ~50%. At 32K they read
  0.98 GB, a floor of 3.4 ms; ~46%.
- The head reads 0.54 GB, a floor of 1.9 ms; ~95%.

## MTP cycle (`--mtp`, temperature 0)

| context | rate | accept | tokens/cycle | cycle ms | verify | draft | host + outside | GPU busy | GPU idle ms/cycle |
|---|---|---|---|---|---|---|---|---|---|
| 2K | 84.0 t/s | 0.677 | 1.672 | 20.02 | 18.23 | 1.61 | 0.18 | 0.927 | 1.47 |
| 32K | 64.9 t/s | 0.709 | 1.703 | 26.29 | 24.08 | 2.09 | 0.12 | 0.932 | 1.78 |
| 128K | 41.2 t/s | 0.787 | 1.781 | 42.10 | 37.36 | 4.57 | 0.17 | 0.835 | 6.93 |

- **The verify costs 1.26-1.28 plain steps.** The corrected split shows where at 2K:
  - its MoE costs 7.55 ms against 4.82 ms for a plain step (x1.57), because each row reads its 8 experts
    and the shared expert again;
  - the GDN mixer costs x1.15, the attention mixer x1.20, the head x1.0 (M7's two-row matvec).
- **Overlap.** The two verify rows share 3.17-3.31 of their 8 routed experts per layer, the same at
  every context.
- **GPU idle.** The CB timeline shows the idle at 2K in three places:
  - before the draft (~0.5-0.9 ms): logits readback, CPU argmax, draft encode, commit latency;
  - before the next verify (~0.45 ms): the caller's sampling and encoding layers 0-1;
  - inside the verify (up to ~0.5 ms): encoding the other 38 layers takes longer than the GPU's 2 flushed
    layers.
- **Production sampling** (CLI `--temp 0.7 --seed 1`) shows two costs absent from the greedy bench:
  - the sampler: "outside" is 1.0-1.1 ms per cycle, against 0.04-0.06 at temperature 0;
  - 16 of 128 cycles fall back to plain decoding, because the sampled token is not the draft's parent.

  The 2K rate is 79.1 t/s against 84.0 greedy.

## Projected gains (lossless levers, spec §4)

| lever | projection at 2K | at 32K | basis |
|---|---|---|---|
| **L2 two-row MoE** | **+10.6%** | **+7.6%** | verify MoE ms x expert-byte saving, about 25%: (16 - overlap) routed + 1 shared read, against 16 + 2 |
| L1 cycle overhead | +2-3% | +2-3% | about a third of the measured GPU idle (multi-flush during the verify encode, a shorter readback before the draft); the rest needs the draft on the GPU |
| L3 launch and encode | not projected | not projected | the per-kernel inefficiency is real (MoE ~55%, attention ~50% of floor), but no lossless fusion is identified without a microbench |
| L4 MTP depth 2 | not built | not built | a 3-row verify adds ~+26% of a plain step per row; at 0.68-0.79 first-draft acceptance, the second draft needs well over 50% conditional acceptance to break even. Unmeasured, so not built (spec: build only on a measured projection of at least +5%) |
| L5 decode attention, 32K | up to +10% if it reached 70% of floor | same | the order-preserving form is unproven |

## Rulings

1. **L2 first.** It has the largest measured projection at both target contexts, and it is Ornith-only.
   Plan B: `docs/superpowers/plans/2026-10-02-ornith-decode-moe-pair.md`.
2. **L2 alone does not reach +15%** (+10.6% at 2K, +7.6% at 32K).
   - Next comes L1: multi-flush during the verify encode, and a shorter pre-draft path.
   - Then L5 at 32K, if an order-preserving improvement shows in a microbench.
3. **L4 is not built** (see the table).
4. **Production sampling cost is real but outside the greedy target.** It is ~1 ms per cycle in the
   sampler, plus 12.5% plain fallbacks at temperature 0.7. It is recorded here for a later lever and is
   not mixed into this target.
