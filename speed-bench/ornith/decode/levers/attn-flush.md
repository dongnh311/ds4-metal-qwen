# Levers L5 (decode3 K/V prefetch) and L1 (multi-flush): measured

> The lever code measured here is not merged into develop. It lives on the local branch
> `feature/ornith-decode` at `01d4bab0`; see `../REPORT.md`.

Window C, 2026-10-02 23:45 to 2026-10-03 00:24, `8cbfbd29`.

**Exactness.** With `DS4_QWEN35_ATTN_PREFETCH=1 DS4_QWEN35_FLUSH_EVERY=8 DS4_QWEN35_MOE_PAIR=1` all on:
- gate 1 at chunks 64/512/2048 equals window M's develop dumps (39/39);
- `test_mtp_cli` passes;
- `test-qwen35-verify-batch` passes.

The prefetch kernel test is also bit-identical: `out` and `part` for rows 1 and 2, at 11 positions.

**Microbench** (`bench_qwen35_attn decode`, ms per call, raw in `bench-decode3-prefetch.txt`):

| position | rows | decode3 | decode3-pf |
|---|---|---|---|
| 2K | 1 | 0.875 | 0.856 |
| 2K | 2 | 0.552 | 0.826 |
| 32K | 1 | 0.909 | 0.880 |
| 32K | 2 | 1.102 | 1.098 |
| 128K | 1 | 1.727 | 1.629 |
| 128K | 2 | 1.925 | 1.944 |

Each call costs ~0.85 ms even at 2K, where only 4 MB of K/V is read. Fixed per-dispatch cost and latency
dominate, not bandwidth, so prefetching the next tile can only move it a little.

**A/B** (`m4_ab.py --mode lever`, A-B-B-A, 2K and 32K, temperature 0, 256 tokens):

| knob | 2K base / lever | 32K base / lever | prefill change |
|---|---|---|---|
| `DS4_QWEN35_ATTN_PREFETCH=1` | 90.1 / 93.7 (+4.0%) | 67.5 / 68.7 (+1.8%) | +5% (prefill does not use decode3: noise) |
| `DS4_QWEN35_FLUSH_EVERY=8` | 91.2 / 94.7 (+3.8%) | 68.6 / 69.0 (+0.6%) | +0.6% |

**Noise.** The base arm's 2K decode across the three window-C runs was 90.1, 95.8 and 91.2 t/s, and a
decode-only lever moved prefill by 5%. Single-knob results near +4% are at the noise edge. The final
window measures both knobs together against all-off at 2K, 32K and 128K before either becomes the
default.

**Final, window F** (2026-10-03 00:44-01:15, `afd68849`). Both knobs on (prefetch plus flush every 8)
against every lever off, A-B-B-A:

| context | off | on | change |
|---|---|---|---|
| 2K | 92.9 | 93.1 | +0.2% |
| 32K | 69.8 | 67.5 | -3.3% |
| 128K | 45.1 | 44.7 | -0.9% |

Flush every 4 against every 8, at 2K with prefetch on: 92.0 against 92.6 (+0.7%).

**Decision:** both knobs stay default off. The window-C single-knob gains were within the run-to-run
noise, and together they gain nothing at 2K and lose a little at 32K.
