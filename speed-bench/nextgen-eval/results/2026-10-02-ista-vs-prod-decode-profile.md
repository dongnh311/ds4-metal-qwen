# ISTA vs PROD decode profile: GDN dense rows are the gap, n-gram gather is not

**Question.** Where does ISTA hc-F16 lose decode against PROD? Does the n-gram (PLE sidecar) gather sit on the
critical path?

**Answer.**
- Per decode step, the GPU time is nearly the same for both: 39.7 vs 39.4 ms on ifeval, 41.3 vs 39.6 ms on the
  story prompt.
- ISTA loses 1.3-1.6 ms per step in the GDN stage. That stage holds the linear-attention dense projections
  (`attn_qkv`, `attn_gate`, `ssm_out`): IQ4_XS/IQ3_S/K-quants in ISTA, Q4_K in PROD. ISTA wins back
  0.3-0.6 ms in attention.
- About 0.5 ms more per step goes outside the GPU, and ISTA accepts slightly fewer tokens per step.
- **The n-gram gather is not the cause:**
  - reading the sidecar with pread instead of the mmap moved decode only as far as the GPU time moved;
  - everything outside the GPU is only 3-4 ms per step for both models.

## Setup

GPU window 09:18-09:31 on 2026-10-02, with the gateway paused through `hold-gateway.sh` (`backend_ok` again
09:33:48).
- **Binary:** `ds4-server`, rebuilt 09:17 from develop `260dc60a`.
- **Script:** `ds4-metal-data/sp3gemm/spike_profile.py`.
- **Each case:** a fresh server with no disk KV, one warm-up request, then one measured request at temperature 0:
  - ifeval key 13 (`max_tokens` 16384);
  - a story prompt (`max_tokens` 3000; both models hit the cap).
- **Flags:** `DS4_QWEN4_STAGE_TS_PROFILE=1` sums GPU timestamps per stage over every decode step, including the
  warm-up's few steps.
  - ISTA runs with the projection (FFN 0.5) and the full MTP head.
  - PROD runs with its registry flags (K=32 streaming, 6 GB expert cache, 64K draft vocab) and no projection,
    like the harness arms.

## Results

| case | tokens | decode t/s | steps | tokens/step | GPU ms/step | wall ms/step |
|---|---|---|---|---|---|---|
| ISTA, ifeval 13 | 5122 | 38.64 | 3058 | 1.675 | 39.74 | 43.4 |
| ISTA + PLE pread, ifeval 13 | 5122 | 36.56 | 3058 | 1.675 | 42.38 | 45.7 |
| ISTA, profiler off, ifeval 13 | 5122 | 36.63 | — | — | — | — |
| PROD, ifeval 13 | 4865 | 40.11 | 2860 | 1.701 | 39.41 | 42.4 |
| ISTA, story | 3000 | 40.15 | 1670 | 1.796 | 41.32 | 45.0 |
| ISTA + PLE pread, story | 3000 | 39.67 | 1670 | 1.796 | 42.11 | 45.6 |
| PROD, story | 3000 | 42.86 | 1650 | 1.818 | 39.59 | 42.8 |

GPU ms per step by stage:

| stage | ISTA ifeval | PROD ifeval | ISTA story | PROD story |
|---|---|---|---|---|
| ple | 0.30 | 0.19 | 0.30 | 0.19 |
| hc_attn | 3.24 | 3.34 | 3.34 | 3.35 |
| gdn | **8.98** | **7.72** | **9.44** | **7.84** |
| attn | 4.17 | 4.74 | 4.10 | 4.45 |
| hc_ffn | 3.85 | 3.77 | 4.02 | 3.79 |
| moe | 16.73 | 17.23 | 17.55 | 17.54 |
| output | 2.46 | 2.42 | 2.55 | 2.42 |

## Reading

- **The pread run** on ifeval overlapped the sha256 check of the 37 GB Q2_0 download (09:24:41-09:25:59).
  - Every stage's GPU time rose by about 7% in step. The pread change cannot do that; memory-bandwidth
    contention can.
  - On the story prompt, with no overlap, pread and mmap differ by 1%.
- **The same ISTA config read 38.6 and 36.6 t/s in two runs** (profiler on first, off later). This is
  run-to-run spread of about 5%; the profiler's own cost is below that.
- **The gap is GPU work in the GDN stage.**
  - In PROD, `attn_qkv`, `attn_gate` and `ssm_out` are Q4_K in all 36 GDN layers, with `ssm_alpha/beta` F32
    (read from its header).
  - In ISTA they are a per-layer mix of IQ4_XS, IQ3_S and K-quants, with `ssm_alpha/beta` BF16, read through
    the GSQ dequantizers.
  - Next check: which kernel each GDN projection runs on at decode, the GDN front's `qwen4_row_dot` or
    `kernel_qwen4_gsq_mv`, and its time per type. That is the next ISTA decode lever.
  - The Q2_0 tier has the same issue: its GDN projections are Q3_K, IQ4_XS and Q2_0.
