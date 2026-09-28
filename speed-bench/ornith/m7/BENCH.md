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

## Task 2: the two-row kernel (window `m7t2`, 2026-09-28 08:14, snapshot `0e652aa`)

Method fixed: every method first runs untimed for 0.3 s (GPU clock ramp), each timed repetition reads >= 1 GiB of
weights (61 calls for 2048x8192, 121 for the 17.8/8.9 MB shapes, 32 for the lm head), and the host encoding time is
reported (0.7-3 us per call, negligible). Raw: `bench/t2-run1.txt`, `bench/t2-run2.txt`.

| shape | t1 | t1x2 (today's verify) | rows_exact | **rows2** |
|---|---|---|---|---|
| 2048x8192 | 0.066 / 0.085 | 0.106 / 0.107 | 0.078 / 0.078 | **0.067 / 0.067** |
| 2048x4096 | 0.035 / 0.035 | 0.055 / 0.055 | 0.040 / 0.040 | **0.036 / 0.036** |
| 4096x2048 | 0.041 / 0.036 | 0.055 / 0.056 | 0.038 / 0.039 | **0.036 / 0.036** |
| 2048x248320 (lm head) | 1.835 / 1.838 | 3.663 / 3.690 | 3.499 / 3.498 | **1.852 / 1.855** |

(ms per call, run 1 / run 2.) The two-row kernel costs the same as one T=1 call on every shape (x0.79-1.03; it
reads the weights at 246-292 GB/s), where today's two T=1 calls cost x1.3-2.0. T=1 on 2048x8192 still moves between
runs (0.066 / 0.085 ms) while rows2 does not, so the rows2 column is the reliable one.

**Stop rule: pass** (run 1: 2048x8192 x1.01, lm head x1.01; run 2: x0.79, x1.01; limit x1.50).
`make test-qwen4-kernels test-qwen4-q2`: pass. Qwen fast gate (`run.sh fast`): PASS.

Expected verify saving per token at 2K (from these numbers): lm head -1.8 ms, GDN projections 30 x ~0.07 ms
= -2.2 ms, attention q/output 10 x ~0.06 ms = -0.6 ms: ~4.6 ms of today's ~11.5 ms verify overhead.
