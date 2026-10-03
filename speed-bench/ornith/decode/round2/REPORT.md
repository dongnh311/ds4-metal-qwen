# Ornith decode speed, round 2: report

Date 2026-10-03.
- **Branch.** `feature/ornith-decode-r2`, rebased onto develop `331cf3b2` (the PROD deployed today as
  `prod/q4k-down-trim-20261003`).
- **Spec.** `docs/superpowers/specs/2026-10-03-ornith-decode-round2-design.md`.
- **Plans.** `docs/superpowers/plans/2026-10-03-ornith-decode-r2-stageA.md` (measurement, roofline, anatomy)
  and `...-r2-merge-fold.md` (the lever).

## Result

Final paired A/B, window r2w4, 13:03-13:59:
- **Arms.** Base `DS4_QWEN35_ATTN_MERGE_FOLD=0` (develop's merge) against the new default.
- **Runs.** 3 blocks × 2 reps at 2K and 32K; 1 × 1 at 128K.
- **Identity.** 0 text mismatches over 13 paired samples (52 requests).
- **Conditions.** Stack paused, SSD quiet.

| context | base t/s | round 2 t/s | paired gain | n | sd | spec bar | met |
|---|---|---|---|---|---|---|---|
| 2K | 89.7 | 89.7 | +0.0% | 6 | 0.5% | +5% at 2K **or** 32K | yes, at 32K |
| 32K | 68.1 | 73.6 | **+8.1%** | 6 | 0.6% | (same) | yes |
| 128K | 44.6 | 47.2 | **+5.8%** | 1 | - | no more than 3% below base | yes |

- **Spec §1.2 is met.** The round is merge-ready, subject to the user's OK and the Qwen session's gate.
- **Round 1's +15% target is not met,** at 2K or at 32K. 2K is unchanged: with 32 splits per row, merge3's
  register spill costs little there.
- **The gains are not overstated.** In A-B-B-A the lever arm runs in the middle, and the middle runs read
  0.7% low in the A/A (`AA.md`). That bias works against the lever.
- **The lever's own A/B in r2w3 agrees:** 2K +0.9%, 32K +8.6%, 128K +5.2% (`levers/merge-fold.md`).

## What changed in the code

1. **`kernel_qwen35_attn_merge3_fold`** (`metal/qwen35.metal`) replaces `kernel_qwen35_attn_merge3` everywhere
   Ornith merges attention splits: decode3 (decode and verify) and the flash prefill key splits.
   - It folds the same binary tree over the same padded leaves, keeping one register slot per level. merge3's
     three 256-float private arrays spilled to memory.
   - The output is bit-identical.
2. **Selection** (`ds4_metal.m`). The fold is on by default; `DS4_QWEN35_ATTN_MERGE_FOLD=0` gives merge3.
   - The fold runs only after a once-per-process self-check has passed. The check compares it with merge3 on
     fixed partials, and is called from `qwen35_graph_alloc`, which the session and the one-shot CLI path
     both use.
   - A failed or skipped check means merge3.
3. **Measurement tools.**
   - `m4_ab.py --paired --blocks --reps`: the same nonces per block, a text SHA-256 per request, and a paired
     gain summary.
   - `tests/bench_qwen35_decode.c`: decode-kernel roofline.
   - The `decode3-fold` row in `tests/bench_qwen35_attn.c`.
   - `round2/timeline_anatomy.py`: per-encoder timeline anatomy.

**Rollback without a redeploy:** set `DS4_QWEN35_ATTN_MERGE_FOLD=0` in the Ornith row's environment.

## Exactness evidence (final code)

| check | result |
|---|---|
| `tests/test_qwen35_kernels` | `qwen35 kernels: ok`. Merge fold: memcmp-identical for decode3 rows 1 and 2 at 3/32/33/129/256 splits plus the one-split solo path, and for flash and accelerator flash at 2-8 splits. The fold does not run before the self-check. |
| gate-1 dumps, chunks 64/512/2048 | 39 of 39 identical to the develop references |
| `tests/ornith/test_mtp_cli.py` | PASS: 345 accepted / 92 rejected drafts, unchanged |
| verify-batch, `test_qwen35_mtp`, `test_qwen35_graph` | ok |
| paired A/B text hashes | 0 mismatches in r2w3 and r2w4 |

## How the round got there

1. **Measurement first** (`AA.md`).
   - Round 1's ±5% spread came from different prompts, not from speed.
   - Paired runs give 0.7% noise at 2K and 1.3% at 32K. Greedy decoding is identical across server processes.
2. **Roofline** (`KERNELS.md`).
   - The Q8 projections run at 88-90% of the practical peak (274.9 GB/s, the lm head); the MoE kernels at
     46-78%.
   - No single kernel family was worth 3% of the cycle.
   - The step model left 21% of the verify unexplained.
3. **Dispatch anatomy** (`ANATOMY.md`). No fusion candidate. merge3 at 34 µs per call, with its spill, was the
   one target.
4. **The lever** (`levers/merge-fold.md`): it passed spec §1.1 in r2w3 and was turned on by default.
5. **Whole-branch review** (one fresh reviewer, Opus). Verdict: "with fixes", with no Critical findings.
   - Fixed, each with a test that failed first:
     - the one-shot greedy CLI path ran the fold without the self-check, so the fold is now gated on a passed
       check, run at graph allocation;
     - the flash prefill splits had no bit-level test, now added (a mutant that pins merge3 fails it);
     - `.gitignore` lacked the new bench binary.
   - The gating fix also left the attention bench without a self-check call, so r2w4's two bench rows both ran
     merge3. The bench now calls the check and refuses to report a fold row in which the fold did not run.
     r2w3's bench is the in-window record.

## Left for later

- **Deferred minors from the review**, recorded in the ledger:
  - the self-check's neutral partials sit only on heads 7 and 15;
  - the fold's 9-level limit is not tied to the 256-split cap by a static assert;
  - `m4_ab` treats an all-missing text hash as identical, and a zero base rate would raise before writing its
    JSON;
  - the bench smoke does not clear outputs between rows;
  - one misplaced test comment.
- **Levers not built**, with their projections (spec §6 and `KERNELS.md`):
  - a four-kernel MoE retile, about 2-3%;
  - a draft parent on the GPU, about 1.5-2.5%;
  - plain fallbacks at temperature 0.7, about 1-2%;
  - MTP depth 2, which needs about 65% acceptance.
- **The largest remaining lever is lossy:** fp8/q4 K/V with an Ornith decode3 kernel, for 32K and up. It
  needs the user's explicit OK.

## Merge and deploy (waiting)

- **Merge into develop and push** wait for the user's OK. The Qwen session has been asked to run its gate on
  the branch. The shared files are touched only on Ornith paths: `ds4_metal.m`, `ds4.c`, `ds4_qwen35moe.inc`,
  `ds4_gpu.h`, the Makefile.
- **Deploy** would be `prod/ornith-decode-r2-YYYYMMDD`, cut from develop and installed with
  `deploy-ai-gateway.sh`, after the user's OK.
  - It changes the Ornith rows only (262K on :18087 and 512K on :18089).
  - Rollback is the env knob above, or the previous prod branch.
