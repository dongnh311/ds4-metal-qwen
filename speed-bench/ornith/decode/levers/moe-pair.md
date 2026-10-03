# Lever L2: two-row MoE in the MTP verify. Measured, not adopted

> The lever code measured here is not merged into develop. It lives on the local branch
> `feature/ornith-decode` at `01d4bab0`; see `../REPORT.md`.

Window L, 2026-10-02 22:46-23:18, `3022a4af`, `DS4_QWEN35_MOE_PAIR=0` against `=1`.

**Exactness holds.**
- Gate 1 at chunks 64/512/2048 equals window M's develop dumps.
- `test_mtp_cli` passes, with the run-time self-check passing in all 18 processes.
- `test-qwen35-verify-batch` passes: 150 cycles bit-identical to plain decoding.
- A forced self-check failure falls back to the per-row path and still passes.

**Speed regresses** (`m4_ab.py --mode lever`, A-B-B-A, 256 tokens, temperature 0, raw in
`ab-moe-pair.{txt,json}`):

| context | off | on | change |
|---|---|---|---|
| 2K | 91.9 | 73.4 | -20% |
| 32K | 68.9 | 57.9 | -16% |
| 128K | 44.3 | 38.6 | -13% |
| cold 31K | 61.0 | 53.3 | -13% |

**Why the projection missed.** Stage 0 assumed the per-row verify paid twice the DRAM traffic for an
expert both rows routed.
- Its two rows' threadgroups run concurrently, so the second read of a shared expert mostly hits the
  system cache. The verify MoE cost 1.57 plain MoEs, not 2.
- The pair kernel computes both rows' dot products one after the other in one simdgroup, while row 1's
  groups exit early. That doubles the critical path of every shared slot and halves the parallelism.

**Decision:** `DS4_QWEN35_MOE_PAIR` stays default off. A dual-accumulator variant could still help, by
reading each weight block once into registers and dequantizing it once for both rows. It needs the
per-type dot rewritten with two accumulators in the same order, and is left as a possible follow-up.
