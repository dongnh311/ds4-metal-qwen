# Ornith M6: accelerator attention micro-benchmark

`tests/bench_qwen35_attn prefill` (model-free; Ornith shape: 16 query heads, 2 KV heads, head dim 256, F16 K/V).
One call = one attention layer (Ornith has 10). Median of 3 timed calls, each ending in a synchronize. Live stack
paused (controller GPU windows), Apple M5 Pro 64 GB, Metal 4 tensor API enabled. At 30,720 keys one 2048-token
layer is ~1.07 TFLOP (2048 tokens x 16 heads x ~31.7K keys x 1024 flop).

## Stop rule verdict

The plan's stop rule (accelerator kernel >= 2.5x `attn_flash_tok2` at 30,720 and 122,880 keys, T = 2048) is **met**
by the final kernel (A2 with fully unrolled fragment loops, commit `b4e8b7f`, measured again on `12c5b72`):

| pos | qwen4_attn_mm | attn_flash_tok2 (M5) | attn_flash_tok4 | **attn_flash_nax (M6)** | speedup vs tok2 | TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.7 | 7.89 / 7.97 | 8.05 | **2.99 / 2.90** | 2.7x | — |
| 30720 | 436.6 | 222.1 / 221.7 | 221.0 | **61.7 / 61.9** | 3.6x | ~17.3 |
| 122880 | 1724.4 | 882.2 / 877.5 | 946.0 | **249.5 / 243.5** | 3.6x | ~17.0 |

(ms per layer, run 1 / run 2, window `m6t3`, 2026-09-27 19:40-19:46.) Kernel tests in the same window, with and
without `DS4_METAL_DISABLE_METAL4=1`: `qwen35 kernels: ok`; the accelerator cases stay within 4.5e-4 relative of
both the host double reference and the M5 simdgroup flash.

## How we got there (windows m6t1, m6diag, m6t2, m6t2b)

1. **A1 as planned** (4 simdgroups, each owning 16 rows x all 256 dims; `2eb7a47`): 204-228 ms at 30,720 and
   796-927 ms at 122,880, i.e. 1.0-1.1x the simdgroup flash. `relaxed_precision = true` changed nothing
   (202 / 795 ms).
2. **Reference on the same GPU (MLX 0.32, oMLX's runtime):** f16 GEMM on the accelerators 16-22 TFLOP/s; MLX's fused
   accelerator attention at head dim 128 24.6 TFLOP/s; `mx.fast.scaled_dot_product_attention` at head dim 256
   13.7 TFLOP/s (32K keys) and 12.1 TFLOP/s (128K). So the hardware was not the limit.
3. **A1 variants at 30,720:** without the softmax 171.7 ms, with L1-resident K/V 187.7 ms, both 152.3 ms (~7
   TFLOP/s): the matrix loop itself was slow. Computing only 128 of the 256 dims took 90.3 ms (linear), so register
   pressure alone did not explain it. **A2** (dims split across simdgroup pairs, `b21cdea`) ran 208-212 / 817-819 ms,
   no better.
4. **Root cause:** `#pragma unroll` did not fully unroll the loops that index the fragment arrays, so the arrays
   lived in thread memory. With `_Pragma("clang loop unroll(full)")`:

   | variant | 30720 | 122880 |
   |---|---:|---:|
   | A1, unrolled | 199.3 | 837.8 |
   | **A2, unrolled** | **61.6** | **248.8** |
   | A2, unrolled + `max_total_threads_per_threadgroup(256)` | 60.0 | 280.2 |
   | A1 / A2 with only the threadgroup-size attribute | 230.4 / 209.7 | 936.7 / 850.8 |

   A1 fully unrolled holds 128 accumulators per lane and still spills; A2 holds 64. A2 with full unrolling became
   `kernel_qwen35_attn_flash_nax` (`b4e8b7f`); the threadgroup-size attribute was left out (no consistent effect).

## Short chunks (key split, default rule; window m6t3)

ms per layer, run 1 / run 2:

| pos | T | flash_tok2_split (M5) | flash_nax_split (M6) |
|---:|---:|---:|---:|
| 30720 | 16 | 2.34 / 2.35 | 1.24 / 1.15 |
| 30720 | 32 | 4.04 / 4.04 | 1.84 / 1.82 |
| 30720 | 64 | 7.57 / 7.58 | 2.65 / 2.64 |
| 30720 | 128 | 14.69 / 14.69 | 4.41 / 4.43 |
| 30720 | 256 | 28.80 / 28.83 | 8.12 / 8.07 |
| 122880 | 16 | 7.68 / 7.67 | 2.94 / 2.91 |
| 122880 | 32 | 14.69 / 14.70 | 4.55 / 4.66 |
| 122880 | 64 | 28.87 / 28.92 | 8.06 / 8.09 |
| 122880 | 128 | 57.34 / 58.51 | 15.35 / 15.20 |
| 122880 | 256 | 113.91 / 114.41 | 29.58 / 29.50 |

The accelerator kernel wins at every benched T at both positions, so the dispatch needs no minimum T.

## Open item

Other Metal kernels in this repo also use `#pragma unroll` over fragment or simdgroup-matrix arrays (the M5 flash,
`decode3`, MoE kernels); they may be losing the same way. Not measured here (out of M6's scope).
