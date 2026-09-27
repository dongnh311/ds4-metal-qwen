# Ornith M5: attention micro-benchmark

`tests/bench_qwen35_attn` (model-free; Ornith shape: 16 query heads, 2 KV heads, head dim 256, F16 K/V).
One call = one attention layer (Ornith has 10). Median of 3 (prefill) or 9 (decode) timed calls, each ending
in a synchronize. Live stack paused (GPU window), Apple M5 Pro 64 GB, Metal 4 tensor API enabled.
"K/V read" is the K/V the kernel must read at least once; the floor assumes 250 GB/s. For the prefill rows the
GB/s column is that minimum divided by the time, not the traffic today's kernel really moves (it re-reads K/V
per query token).

## Baseline (commit f3636da, 2026-09-27)

| kernel | mode | pos | T | rows | ms / layer | floor ms | x floor |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen4_attn_mm | prefill | 0 | 2048 | - | 14.9 | 0.017 | - |
| qwen4_attn_mm | prefill | 30720 | 2048 | - | 431.7 | 0.268 | - |
| qwen4_attn_mm | prefill | 122880 | 2048 | - | 1715.9 | 1.023 | - |
| decode2 | decode | 2048 | 1 | 1 | 0.395 | 0.017 | 23 |
| decode2 | verify | 2048 | 2 | 2 | 0.607 | 0.017 | 36 |
| decode2 | decode | 32768 | 1 | 1 | 0.925 | 0.268 | 3.5 |
| decode2 | verify | 32768 | 2 | 2 | 1.779 | 0.268 | 6.6 |
| decode2 | decode | 131072 | 1 | 1 | 2.387 | 1.074 | 2.2 |
| decode2 | verify | 131072 | 2 | 2 | 5.235 | 1.074 | 4.9 |

A second run matched within 2% (decode 32K 0.939 / 1.746, 128K 2.354 / 5.255 ms).

Reading: prefill per 2048-token chunk is 10 x 432 ms = 4.3 s at 30K and 10 x 1716 ms = 17 s at 123K (M4's profile
measured 4.3 s and 19.7 s through the model), and it grows linearly with position because every query token
re-reads the K/V range. The 2-row verify costs about twice a plain decode step: decode2 does not share the K/V
read between its two rows in practice. At 128K a verify step spends ~52 ms in attention (10 layers).

## decode3 (commit a897b1d, 2026-09-27; Task 2 Step 7)

Same window, same run: decode2 and decode3 side by side (`./tests/bench_qwen35_attn decode`, two runs,
`DS4_QWEN35_ATTN_SPLIT_KEYS` default 64). ms per layer, run 1 / run 2.

| pos | rows | decode2 | decode3 | floor |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 0.728 / 0.729 | 0.597 / 0.632 | 0.017 |
| 2048 | 2 | 1.315 / 1.275 | 0.791 / 0.891 | 0.017 |
| 32768 | 1 | 1.502 / 1.294 | 1.101 / 0.886 | 0.268 |
| 32768 | 2 | 1.810 / 1.820 | 1.058 / 1.087 | 0.268 |
| 131072 | 1 | 2.460 / 2.459 | 1.706 / 1.718 | 1.074 |
| 131072 | 2 | 5.344 / 5.333 | 1.931 / 1.940 | 1.074 |

Split-keys sweep (one run each, decode3 rows 1 / 2): 32 -> 2048: 0.891 / 1.189, 32768: 0.893 / 1.053,
131072: 1.645 / 1.911; 128 -> 2048: 0.626 / 0.604, 32768: 0.917 / 1.058, 131072: 1.701 / 1.911. From 16K keys
on every value hits the 256-split cap, so only short contexts depend on it; 64 stays the default.

Reading: decode3 is faster than decode2 at every point. At 128K a plain step drops from 2.46 to 1.71 ms per layer
(1.6x the floor; the target was <= 1.7 ms) and the 2-row verify from 5.34 to 1.93 ms (the two rows now share each
K/V tile), i.e. ~19 ms instead of ~53 ms of attention per verify step over 10 layers. Sub-millisecond points move
by up to 20% between runs (GPU clock state: decode2 at 2048 read 0.395 ms in the baseline run, which ran the
prefill cases first); compare kernels within one run. Kernel tests in the same window: `qwen35 kernels: ok`.
