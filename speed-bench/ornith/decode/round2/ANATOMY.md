# Round 2 window 2: the anatomy of a verify's dispatches (2K, `--mtp`)

- **Run.** Window r2w2, 2026-10-03 10:56-10:57, on `df20a5b6`.
- **Prompt.** The stage-0 2K prompt, CLI `--temp 0 --mtp -n 128`.
- **The two runs:**
  1. `DS4_METAL_ENCODER_TIMELINE`, which gives every dispatch group its own compute pass with GPU timestamps;
  2. the same run without it, as the reference.
- **Receipts.** `receipts/anatomy-2k.txt` (the parser output) and `receipts/anatomy-2k-runs.txt`. The raw
  timeline (6.6 MB) stays in the scratch window.

## Cycle shape

Each MTP cycle is three command buffers:
1. the verify's layers 0-1: 39 encoders, 0.9 ms;
2. the verify's layers 2-39 plus the head: 724 encoders, about 20 ms (timeline-inflated);
3. the draft: 22 encoders, 0.9 ms.

There were 78 cycles (one `mtp_concat` per draft). The CLI's "64 cycles" is the spec-stats print interval.

| per cycle | timeline run | reference run |
|---|---|---|
| wall ms (spec stats: target + draft) | 25.42 | 20.31 |
| buffer GPU ms | 21.33 | |
| encoder busy ms | 19.04 | |
| gaps inside buffers ms | 2.58 | |
| idle between buffers ms | 3.80 | (round 1: 1.47) |
| encoders | 772 | |
| encoders under 10 µs | 421 (busy 1.96 ms, their gaps 1.04 ms) | |

**What the timeline costs.** It adds 5.1 ms per cycle, about 6.5 µs per encoder. That covers its own pass
boundaries and its slower host encoding, which is what makes the idle between buffers 3.8 ms. So its gap and
idle figures are upper bounds, and only its per-kernel busy times are used below.

## Per-kernel busy time per cycle (largest first; `receipts/anatomy-2k.txt`)

| kernel | per cycle | µs each | ms per cycle | note |
|---|---|---|---|---|
| `kernel_qwen35_mv_q8_0_rows2` | 107 | 66 | 7.06 | two-row Q8_0 matvecs (bench: 88% of peak) |
| `kernel_qwen4_moe_mid_q4k_nr1` | 26 | 91 | 2.33 | bench: 71% |
| `kernel_qwen35_moe_mid_q5k_nr1` | 15 | 119 | 1.77 | bench: 62% |
| `kernel_qwen4_moe_down_q4k_nr4` | 26 | 53 | 1.35 | bench: 62% |
| `kernel_qwen35_moe_down_q5k_nr4` | 15 | 82 | 1.21 | bench: 46% |
| `kernel_mul_mv_q8_0_f32` | 5 | 140 | 0.76 | the draft's head |
| `kernel_mul_mv_f32_f32_4` | 79 | 9.0 | 0.71 | F32 router and `ssm_alpha`/`ssm_beta` |
| `kernel_qwen4_gdn_front` | 59 | 9.9 | 0.58 | |
| `kernel_qwen4_gdn_scan_r4` | 59 | 9.2 | 0.54 | |
| `kernel_mul_mv_f16_f32_4` | 40 | 10.6 | 0.42 | F16 `attn_k`/`attn_v` |
| `kernel_rms_norm_mul_f32_4` | 83 | 4.6 | 0.38 | |
| `kernel_qwen35_attn_decode3_r2` | 9 | 40 | 0.37 | the 2K decode attention, 32 splits per row |
| **`kernel_qwen35_attn_merge3`** | **11** | **34** | **0.36** | **merges the 32 split partials of 16 heads: as slow as the attention itself** |
| `kernel_qwen4_router_topk` | 41 | 7.1 | 0.29 | |
| `kernel_qwen35_gdn_out` | 59 | 3.1 | 0.18 | |
| `kernel_qwen4_moe_reduce` | 41 | 4.4 | 0.18 | |
| `kernel_add2_f32` | 81 | 2.1 | 0.17 | |
| `kernel_qwen4_attn_prep` | 11 | 14.7 | 0.16 | |

## Rulings

1. **There is no fusion candidate.**
   - The 421 small encoders per cycle carry 1.96 ms of busy time and 1.04 ms of gaps. Both figures are
     timeline-inflated, at about 6.5 µs of pass overhead each.
   - No single back-to-back chain removes 0.60 ms per cycle. The largest per-layer pairs (norm and add,
     gdn front/scan/out) are 0.2-0.6 ms per cycle in total, before the timeline's inflation is taken out.
   - Fusing elementwise kernels also invites a different FMA contraction under fast math, so it would need a
     run-time self-check.
   - Cost if wrong: maybe 0.3-0.5 ms per cycle left in dispatch overhead.
2. **`kernel_qwen35_attn_merge3` is a lossless candidate, pending its size at 32K.**
   - Each thread keeps three private arrays of `QWEN35_ATTN_MAX_SPLITS` = 256 floats (`mm`, `ll`, `oo`, 3 KB
     per thread) and folds them with a fixed binary tree. Private arrays that size live in device memory, not
     registers.
   - At 2K (32 splits) one merge costs as much as the attention it merges. With the default 64 keys per
     split, 32K and 128K both cap at 256 splits, which is 8 times the leaves and tree nodes.
   - The same tree can be folded in place with one register slot per tree level, at most 9 for 256 leaves,
     visiting the leaves in order and merging on carry. That gives the same pairs, the same operand order and
     the same per-node formula, so the outputs are bit-identical.
   - It runs in every decode, verify and draft attention layer, and in the flash prefill's key-split merge.
   - It is Ornith-only (`metal/qwen35.metal`, plus the wrapper in `ds4_metal.m`).
   - Plan B builds it behind a knob with a run-time self-check, then measures it in the next window with
     `bench_qwen35_attn decode` at 2K/32K/128K, the identity checks and the paired A/B.
   - Cost if wrong: one window, if the 32K merge turns out small.
