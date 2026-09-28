# Ornith M7: verify matvec microbench

`tests/bench_qwen35_verify` (model-free): Ornith's Q8_0 projection shapes, each timed call on a different copy of
the weights (>= 256 MB per shape, so no call finds its weights cached, as in real decoding). Apple M5 Pro, 64 GB.

## Task 1: existing paths (window `m7t1`, 2026-09-28 08:11, live stack paused, snapshot `3a9bd43`)

ms per call, median of 5 repetitions of 32 calls in one command buffer; ratio vs one T=1 call. Raw:
`bench/t1-run1.txt`, `bench/t1-run2.txt`.

| shape | t1 run 1 / 2 | t1x2 run 1 / 2 | rows_exact run 1 / 2 |
|---|---|---|---|
| 2048x8192 (lin_qkv, attn_q) | 0.130 / 0.073 | 0.139 / 0.108 | 0.108 / 0.081 |
| 2048x4096 (lin_gate) | 0.115 / 0.053 | 0.059 / 0.060 | 0.045 / 0.045 |
| 4096x2048 (lin_out, attn_output) | 0.083 / 0.112 | 0.060 / 0.064 | 0.043 / 0.044 |
| 2048x248320 (lm head) | **1.830 / 1.827** | **3.664 / 3.661** (x2.00) | **3.487 / 3.486 (x1.91)** |

**Verdict: `build-kernel`.** The existing exact two-row dispatch (`ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`,
grid (tiles, rows)) still streams the lm head twice: row 1's threadgroups re-read the 540 MB after row 0's pass
(x1.91, identical in both runs). One T=1 lm head call reads at ~296 GB/s. Task 2 builds the two-row kernel.

**Measurement caveat:** the small shapes are noisy (T=1 swings 0.07-0.13 ms between runs, and "two T=1 calls" is
sometimes faster than one): each measurement is only ~3 ms of GPU work right after an idle phase (the host fills the
next shape's weights), so GPU clock ramp-up and the host encoding time (included in wall time, since the command
buffer commits at the end) dominate. The lm head row (~60 ms per repetition) is stable. Task 2 fixes the method
(see below) before its stop rule reads the 2048x8192 row.
