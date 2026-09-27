# Ornith M5 gate 3 (speed) — final interleaved A/B against the live oMLX

- ds4 build: snapshot of `f7f3e17` (M5 defaults: decode3 decode/verify attention, flash prefill attention, plus
  M4's defaults) with `DS4_QWEN35_MTP_DRAFT_VOCAB=speed-bench/ornith/m4/ornith-draft-vocab-vi-en-code-64k.txt`;
  `ds4-server --metal -c 262144 --mtp`, F16 K/V, prefill chunk 2048, 23G GGUF.
- oMLX: the live registry command (oMLX 0.6.4, Shiftedx Ornith abliterated MLX), started/stopped by the harness.
- Command: `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --ds4-env
  "DS4_QWEN35_MTP_DRAFT_VOCAB=<list>" --contexts 2048,32768,131072 --cold-tokens 31000 --max-tokens 256 --warmup 1`
  (A-B-B-A omlx/ds4/ds4/omlx; fresh nonce per prompt; cached_tokens == 0 asserted; one model process at a time;
  live stack paused), 2026-09-27 15:19-15:48. Raw: `final.json`, `final.txt`.

| context | ds4 decode t/s | oMLX decode t/s | ds4 prefill t/s | oMLX prefill t/s | ds4 TTFT s | oMLX TTFT s |
|---|---|---|---|---|---|---|
| 2K | 69.1 | 80.2 | **1738** | 1634 | — | — |
| 32K | 53.8 | 64.6 | 983 | 1677 | — | — |
| 128K | **39.3** | 38.9 | 354 | 739 | — | — |
| cold ~31K | 53.4 | 65.0 | 958 | 1515 | 32.4 | 20.9 |

Verdict against spec §1 items 1-2: **FAIL** — prefill is below the live oMLX at 32K (-41%) and 128K (-52%) and
the cold ~31K TTFT is 1.55x oMLX's; decode reaches oMLX at 128K (+1%) but stays below at 32K (-17%). The 2K
decode gap is out of M5's scope (reported, not judged); 2K prefill is above oMLX.

Against M4's gate 3 (same harness, same day, 04:40): decode 65.0 -> 69.1 (2K), 47.1 -> 53.8 (32K), 22.0 -> 39.3
(128K); prefill 1656 -> 1738, 650 -> 983, 187 -> 354; cold ~31K TTFT 49.6 -> 32.4 s. The live oMLX measured
within 5% of its M4 numbers.

Where the remaining gap is:
- Prefill: the flash kernel is bound by simdgroup-matrix throughput (~4.8 TFLOP/s of attention at 30K,
  `../BENCH.md`); a 32K prompt still spends most of its time in attention. oMLX's prefill rate implies more matrix
  throughput than the simdgroup path gives on this GPU, which points at the M5 neural accelerators (Metal 4
  tensor API, out of M5's scope; the repo already has a tensor-API attention kernel for DSV4.1 in
  `metal/dsv41.metal`).
- Decode at 2K/32K: attention is no longer the bottleneck (17.6 ms of a 128K step, ~1 ms per layer at 32K); the
  per-step GDN (~15 ms, flat in context), MoE and MTP overhead remain — the separate 2K-decode milestone.
