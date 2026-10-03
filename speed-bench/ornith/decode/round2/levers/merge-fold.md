# Lever: attention merge folded in registers (round 2 Plan B)

- **What it is.** `kernel_qwen35_attn_merge3_fold` replaces `kernel_qwen35_attn_merge3` after decode3's
  partials.
  - merge3 keeps three private arrays of 256 floats per thread. They spill to device memory (`ANATOMY.md`:
    11 merges per 2K cycle, 34 µs each).
  - The fold walks the same binary tree over the same pow2-padded leaves. It keeps one register slot per
    tree level (at most 9) and carries finished nodes upward, so each node merges the same two operands in
    the same order.
  - It selects nothing and reorders nothing. The output is bit-identical by construction, and checked below.
- **Knob.** `DS4_QWEN35_ATTN_MERGE_FOLD`. A startup self-check compares the fold with merge3 on fixed
  partials (two rows, 256 and 37 splits, every eighth partial neutral). A mismatch pins merge3 for the
  process.
- **Window.** r2w3, 2026-10-03 11:32-12:25, stack paused, peers notified, DS41F SSD quiet. The code was
  rebased onto develop `331cf3b2`. Receipts: `receipts/r2w3-*` (progress, kernels, bench, both `m4_ab.json`).

## Kernel exactness (`tests/test_qwen35_kernels`)

decode3 with the fold against decode3 with merge3, `memcmp` on the output, rows 1 and 2:

| pos0 | split_keys | splits per row | rows 1 | rows 2 |
|---|---|---|---|---|
| 40 | 16 | 3 (pad 4) | identical | identical |
| 2046 | 64 | 32 (pow2) | identical | identical |
| 2100 | 64 | 33 (pad 64) | identical | identical |
| 8250 | 64 | 129 (pad 256) | identical | identical |
| 32766 | 64 | 256 (cap) | identical | identical |
| 63 | 64 | 1 / 2 | identical (no merge runs) | not reached in r2w3 |

- The run stopped at pos0=63 rows=1, on the test, not the kernel. That row has one split, so decode3
  writes it directly and no merge runs, but the test required the fold counter to move.
- The test now requires `ran == needs_merge` (`ec01c19f`). Its re-run is in the next window.

## Bench (`tests/bench_qwen35_attn decode`, ms per decode3 call, median)

| pos | rows | merge3 | fold | change |
|---|---|---|---|---|
| 2048 | 1 | 0.503 | 0.531 | +0.028 |
| 2048 | 2 | 0.638 | 0.646 | +0.008 |
| 32768 | 1 | 0.899 | 0.787 | -0.112 |
| 32768 | 2 | 1.127 | 0.888 | -0.239 |
| 131072 | 1 | 1.840 | 1.679 | -0.161 |
| 131072 | 2 | 1.937 | 1.739 | -0.198 |

- At 2K there are 32 splits per row; the fold's gain there is within the bench's noise.
- From 32K up there are 256 splits (the cap), where merge3's spill costs most.

## Identity (knob on)

- Gate-1 dumps at prefill chunks 64, 512 and 2048 against window M's references: 0 of 13 differ at each
  chunk (39 of 39 identical).
- `tests/ornith/test_mtp_cli.py`: PASS.
- `make test-qwen35-verify-batch`: `qwen35 verify batch: ok`.

## Paired A/B (`m4_ab.py --mode lever --paired`, base with the knob unset, which was merge3 then, against `=1`)

| context | blocks × reps | n | mean | sd | min | max | text mismatches |
|---|---|---|---|---|---|---|---|
| 2K | 2 × 2 | 4 | +0.9% | 0.9% | +0.1% | +1.8% | 0 |
| 32K | 2 × 2 | 4 | +8.6% | 1.4% | +7.0% | +10.1% | 0 |
| 128K | 1 × 1 | 1 | +5.2% | - | +5.2% | +5.2% | 0 |

- Mean decode rates, base → fold: 2K 89.2 → 90.0 t/s; 32K 68.3 → 74.2 t/s; 128K 45.5 → 47.8 t/s.
- Prefill is unchanged within noise (the fold only runs in decode3).
- Every repeated request decoded the same text in all four runs of its block (SHA-256 of reasoning and
  content), so this is also an identity check on the server path.

## §1 ruling

**The lever passes and turns on by default.** Spec §1.1 needs all three:

| condition | result |
|---|---|
| exact (§2) | yes: kernel `memcmp` on 11 row cases, 39 of 39 gate-1 dumps, `test_mtp_cli`, verify-batch, 0 text mismatches over 9 paired samples (36 requests) |
| mean gain ≥3% at 2K or 32K, and ≥2× the A/A noise there | 32K: +8.6% (min +7.0%), against 3% and 2 × 1.3% = 2.6% (`AA.md`) |
| no context loses more than 3% | 2K +0.9%, 32K +8.6%, 128K +5.2% |

- The 128K figure is a single pair. It only has to clear "no loss beyond 3%", and the bench (-0.2 ms per
  call) points the same way.
- **What changes.** `DS4_QWEN35_ATTN_MERGE_FOLD` unset now means the fold; `=0` gives merge3. The startup
  self-check still pins merge3 if the run-time compiled library disagrees with it.
- **Next.** The kernel tests and gate-1 identity on the flipped default, then spec §1.2's final paired A/B
  (3 blocks, base `=0` against unset). The round's merge-ready bar is +5% at 2K or 32K, with 128K no more
  than 3% below base. This lever alone shows +8.6% at 32K.
