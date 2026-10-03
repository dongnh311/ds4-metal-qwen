# Round 2 window 1: decode-kernel roofline

- **Run.** `tests/bench_qwen35_decode` in window r2w1, 2026-10-03 09:51, stack paused.
- **Receipts.** `receipts/ksweep.txt` and `receipts/smoke.txt` (16/16 rows finite).
- **Peak reference.** The lm-head Q8_0 matvec at T=1, 274.9 GB/s, is taken as the practical peak (plan ruling).
  Its T=2 form reads 281.5 GB/s.

## Rows (GPU ms per call; % of the 274.9 GB/s peak)

| kernel (decode call) | T=1 ms | T=1 % | T=2 ms | T=2 % | per step |
|---|---|---|---|---|---|
| q8 2048x248320 lm head | 1.9659 | 100.0 | 1.9198 | 102.4 | 1 |
| q8 2048x8192 attn_qkv / attn_q | 0.0737 | 88.1 | 0.0739 | 87.8 | 40 |
| q8 2048x4096 attn_gate | 0.0360 | 90.0 | 0.0368 | 88.1 | 30 |
| q8 4096x2048 ssm_out / attn_output | 0.0410 | 79.1 | 0.0370 | 87.6 | 40 |
| MoE mid Q5_K + shared (`kernel_qwen35_moe_mid_q5k_nr*`) | 0.0662 | 75.6 | 0.1225 | 62.3 | 15 |
| MoE down Q5_K + shared (`kernel_qwen35_moe_down_q5k_nr*`) | 0.0442 | 56.7 | 0.0828 | 46.1 | 15 |
| MoE mid Q4_K + shared (qwen4 Q4_K mid) | 0.0546 | 77.8 | 0.0902 | 70.8 | 25 |
| MoE down Q4_K + shared (qwen4 Q4_K down) | 0.0304 | 69.9 | 0.0519 | 61.6 | 25 |

- **How T=2 bytes are counted.** They are the distinct bytes: row 1 shares 3 of row 0's 8 experts, as measured
  in `PROFILE.md`.
- **T=2 against T=1.** The MoE kernels cost ×1.65-×1.87 at T=2, for ×1.56 the distinct bytes. Counted on
  requested bytes (16 experts), the T=2 rate equals the T=1 rate. A repeated expert costs as much as a new one:
  the per-row kernels' rate, not DRAM, sets the pace. This matches round 1's finding that sharing the read
  (L2/L2b) did not pay.

## Family table (spec §4's rule: half the T=2 gap to peak, summed over the step, against 0.60 ms per cycle)

| family | half-gap ms/cycle | ruling |
|---|---|---|
| q8 lm head | 0.00 | not a candidate: at peak |
| q8 2048x8192 | 0.18 | not a candidate: 88% of peak |
| q8 2048x4096 | 0.07 | not a candidate |
| q8 4096x2048 | 0.09 | not a candidate |
| MoE mid Q5_K | 0.35 | not a candidate alone (under 0.60) |
| MoE down Q5_K | 0.34 | not a candidate alone. It is the least efficient kernel (46-57%): short 512-element rows, two Q5_K superblocks each |
| MoE mid Q4_K | 0.33 | not a candidate alone; a shared qwen4 kernel |
| MoE down Q4_K | 0.25 | not a candidate alone; a shared qwen4 kernel |

**No single kernel family passes.**
- All four MoE families together have a half-gap of 1.27 ms per cycle (about 6%).
- It is spread over four kernels, and two of them are shared qwen4 kernels. Each lever would face §1's 3% gate
  alone, and none projects 3% alone.
- Ruling: the MoE retile is not taken as a round-2 lever on these numbers. Cost if wrong: about 2-3% left in
  the MoE kernels, reachable only as a four-kernel project.

## Step model against the measured stage times (`PROFILE.md`, plain 2K, corrected)

The bench-covered work sums to 11.41 ms per plain step (T=1) and 14.09 ms per verify (T=2). The measured times
are 14.28 ms per plain step and 18.23 ms per verify.

| stage (plain 2K) | measured ms | bench-covered ms | residual ms | what the residual is |
|---|---|---|---|---|
| GDN mixer (30 layers) | 5.25 | 4.52 (qkv, gate, ssm_out) | 0.73 | conv, scan prep, scan, gated norm, `ssm_alpha`/`ssm_beta`: ~24 µs per layer |
| attention mixer (10 layers) | 2.17 | 1.15 (q, o) | 1.02 | F16 k/v projections, q/k norm, rope, decode attention, gate: ~100 µs per layer |
| MoE (40 layers) | 4.82 | 3.78 | 1.04 | router matvec, top-k, shared-expert gate, reduce: ~26 µs per layer |
| head | 2.04 | 1.97 | 0.07 | final norm, argmax |
| **total** | **14.28** | **11.41** | **2.87 (20%)** | |

**The verify's residual is 4.14 ms (18.23 - 14.09), 21% of the 20.02 ms cycle.** That is more than any kernel
family's gap.
- It is made of small kernels and the gaps between dependent dispatches.
- Plan A's ruling put it out of the bench, to be read "from the step model's gap". The gap is now the largest
  term, so the next measurement is its anatomy: per-dispatch GPU time and inter-dispatch gaps for one verify,
  from the existing `DS4_METAL_ENCODER_TIMELINE` diagnostic.
- This is round 1's lever L3 (round 1 spec §4): launch and encode cost, and fusing dispatches that run back to back without
  changing their arithmetic.
- **Ruling:** before applying §7's stopping rule, run that anatomy in window 2. A fusion candidate must remove
  at least 0.60 ms per cycle with the per-element code unchanged.
