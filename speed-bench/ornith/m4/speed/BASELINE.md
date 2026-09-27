# Ornith M4 baseline: M1-resident ds4 vs live oMLX

- ds4 build: snapshot of `992fc53` (develop after M3 + the M4 harness; Q5_K experts still use the per-token row
  kernels, no M4 levers). Harness `speed-bench/ornith/m4_ab.py` at `1e1da5c` (oMLX lazy-load fix).
- oMLX: the live registry command (oMLX 0.6.4, Shiftedx Ornith abliterated MLX), started and stopped by the harness.
- Command (from the snapshot root, live stack paused, 2026-09-27 00:40-01:30):
  `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --out <dir>
   --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`
  (A-B-B-A: omlx, ds4, ds4, omlx; ds4 runs `ds4-server --metal -c 262144 --mtp`; every prompt starts with a fresh
  nonce and `cached_tokens == 0` is asserted.)
- Caveat: other agents compiled C code on the CPU during parts of this run; the decisive comparison is the
  final Task 12 run.

| context | ds4 decode t/s | oMLX decode t/s | ds4 prefill t/s | oMLX prefill t/s | ds4 TTFT s | oMLX TTFT s |
|---|---|---|---|---|---|---|
| 2K | 61.7 | 80.4 | 543 | 1684 | 3.98 | 1.22 |
| 32K | 41.7 | 64.5 | 358 | 1685 | 91.6 | 19.9 |
| 128K | 20.4 | 38.6 | 150 | 720 | 866.0 | 185.5 |
| cold ~31K | 39.9 | 64.3 | 336 | 1487 | 92.4 | 21.3 |

Gate 3 fails everywhere at this point, as expected for prefill (Q5_K row kernels; Task 4 adds the tiles).
Decode is also below oMLX and falls faster with context (61.7 -> 20.4 t/s from 2K to 128K vs 80.4 -> 38.6 for
oMLX), which points at attention decode over the F16 KV (and the 2-row MTP verify reading the KV once per row)
— the Task 5 profile attributes it, and Task 11's KV modes / shared-KV verify are triggered.
