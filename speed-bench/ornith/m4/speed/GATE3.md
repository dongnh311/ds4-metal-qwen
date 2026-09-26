# Ornith M4 gate 3 (speed) — final interleaved A/B against the live oMLX

- ds4 build: snapshot of `bbf5966` (M4 final code: Q5_K tiles; L1 flush, L8 Q5_K decode kernels and L12 shared-K/V
  decode on by default) with `DS4_QWEN35_MTP_DRAFT_VOCAB=speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt`;
  `ds4-server --metal -c 262144 --mtp`, F16 K/V, prefill chunk 2048.
- oMLX: the live registry command (oMLX 0.6.4, Shiftedx Ornith abliterated MLX), started/stopped by the harness.
- Command: `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --ds4-env
  "DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`
  (A-B-B-A omlx/ds4/ds4/omlx; fresh nonce at the start of every prompt; cached_tokens == 0 asserted; one model
  process at a time; live stack paused), 2026-09-27 04:40-05:20. Raw: `final.json`, `final.txt`.

| context | ds4 decode t/s | oMLX decode t/s | ds4 prefill t/s | oMLX prefill t/s | ds4 TTFT s | oMLX TTFT s |
|---|---|---|---|---|---|---|
| 2K | 65.0 | 81.2 | 1656 | 1737 | — | — |
| 32K | 47.1 | 64.4 | 650 | 1697 | — | — |
| 128K | 22.0 | 38.1 | 187 | 713 | — | — |
| cold ~31K | 43.0 | 63.9 | 626 | 1413 | 49.6 | 22.5 |

Verdict: **FAIL** — ds4 is below the live oMLX on decode at every context and on prefill at every context (2K
prefill is within 5%).

Against the M4 baseline (`BASELINE.md`, M1-resident ds4): decode +5% (2K), +13% (32K), +8% (128K); prefill 3.0x
(2K), 1.8x (32K), 1.25x (128K); cold ~31K first turn 92.4 s -> 49.6 s.

Where the gap is (`../profile/PROFILE.md`): attention. Prefill attention grows to 86% of a 32K and 96% of a 128K
prefill (per-chunk attention time rises linearly with position); decode attention costs ~32 ms/token at 128K, about
3x the K/V bandwidth floor. The MoE side is fixed (2K prefill at parity) and the GDN layers are flat with context.
The planned levers do not change attention efficiency, and the shared FP8/4-bit K/V attention paths are slower
than F16 for Ornith's 2-KV-head x 256-dim shape (`LEVERS.md`). Closing gate 3 needs an Ornith attention kernel
pair: a flash-style prefill kernel (reuse K/V tiles across query rows) and a decode kernel near the bandwidth floor.
