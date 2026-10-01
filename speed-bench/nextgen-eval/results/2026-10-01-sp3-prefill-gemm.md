# Sub-project 3 follow-up: ISTA prefill on the tiled GEMMs

ISTA's GSQ-RCO IQ3_XXS file now prefills at about 361 t/s instead of 59, six times faster, at 8K.
Its routed experts take the tiled MoE GEMM (`kernel_qwen4_moe_mm_mid/down`). Its dense GSQ-RCO and
BF16 tensors take the tiled dense GEMM (`kernel_qwen4_dense_mm`). Both stage weights through
`qwen4_mm_stage8`, which now dequantizes the nine GSQ-RCO types.

- Single-session decode and MTP verify keep the per-token kernels SP3 verified. A multi-session decode batch above 8 rows sends its GSQ-RCO and BF16 dense rows to the dense GEMM, the same batch policy F32/F16 rows already follow.
- PROD's output is byte-identical, and its speed is unchanged.
- PROD still prefills faster: about 537 t/s on the same bench. It runs the Metal 4 tensor-op tiles
  (`_nax`), which take only its four types.

Plan: `docs/superpowers/plans/2026-10-01-gsq-rco-prefill-gemm.md`. Commits `80aada8`, `2406dfb`, `a52466e`.
The GPU window ran 2026-10-01 06:46-07:13 with the gateway stack paused; the `vm_stat` swap-out
counter did not move.

## Kernel level (`./ds4_test --metal-kernels`)

- **MoE GEMM:** mid and down against `ds4_dequant_row` on real ISTA blocks. The test runs 3 experts and
  40 tokens × 2 slots; one expert gets two token tiles. Five cases: IQ2_XS/Q2_0, IQ2_S/IQ4_NL,
  IQ3_XXS/Q2_0, IQ3_S/IQ4_NL and IQ2_XXS/IQ4_NL. The bound is 1.5e-3 of the magnitude, because
  operands are staged as half.
- **Dense GEMM:** the nine GSQ-RCO types and BF16 on 37 rows. At 40 tokens there is one full and one
  partial token tile. At 9 tokens the k-split planes and their reduce run. The BF16 rows are 2568
  wide, so the last k tile ends mid-tile. The bound is fp32, 1e-5 of the magnitude.
- **Routing (`--quant-types`):**
  - GSQ-RCO experts count as tiled-GEMM types.
  - GSQ-RCO and BF16 dense rows take the dense GEMM above 8 rows, and only when the width is whole blocks.
  - Q8_0 and Q4_K keep their kernels.

## Model level

**Correctness:** `DS4_QWEN4_GPU=1 ./ds4 --first-token-test` on a 181-token prompt (Python code plus
prose), Metal against `qwen4_ref_forward_token`. With a chunk of 256, the whole prompt is one chunk,
so only its last position is compared:

| model | chunk | path | worst max\|gpu-cpu\| | top1 agree |
|---|---|---|---|---|
| ISTA | 64 | per-token experts (≤ 64 rows), dense rows on the new dense GEMM | 2.410 | 3/3 |
| ISTA | 256 | tiled GEMMs | 3.021 | 1/1 |
| Ivan IQ2 | 256 | PROD's tiled GEMMs (tensor-op tiles) | 4.249 | 0/1 |

The plan's bar is max(Ivan chunk 256, ISTA chunk 64) × 1.25 = 5.31, and 3.021 meets it. The half
staging moves ISTA further from the CPU than its per-token path does. It still stays inside the gap
that PROD's own tiled path shows on the same prompt.

This model-level check is thin evidence:
- the chunk-256 rows compare one position each;
- the bar comes partly from a different model;
- the chunk-64 control already runs the new dense GEMM.

The direct evidence is the kernel tests against the CPU rows, together with the unchanged coherence
and MTP acceptance below.

**Coherence:** `ds4 --metal -c 32768 --temp 0 -n 256 --mtp`, SP3's three prompts. All three answers are
coherent and correct: `is_prime`, Rayleigh scattering in Vietnamese, and TCP/UDP.

| prompt | MTP accepted (SP3) | MTP accepted (now) |
|---|---|---|
| Python prime | 97.4% | 99/101 (98.0%) |
| Vietnamese sky | 76.7% | 47/61 (77.0%) |
| TCP/UDP | 85.3% | 99/116 (85.3%) |

**ISTA speed:** `ds4-bench`, 8K story prompt, 128 tokens, resident, no MTP, 3 rounds:

| | SP3 (per-token prefill) | now | |
|---|---|---|---|
| prefill t/s (median) | 59.4 | 361.3 | × 6.1 |
| decode t/s (median) | 25.2 | 27.3 | decode code unchanged; last night's rounds spread 24.8-26.8 |
| steady decode t/s (median) | 26.1 | 27.4 | |

**PROD:** baseline is develop `6985bdb`, built from `git archive`; it is the commit before this
work. PROD's model, flags and environment.
- **Identity:** greedy, 128 tokens, `-c 262144`, 3 prompts. Identical 3/3, non-empty.
- **Speed A/B:** `ds4-bench`, 8K story prompt, 3 rounds each, alternating:

| | baseline | now | ratio |
|---|---|---|---|
| prefill t/s (median) | 537.98 | 536.94 | 0.998 |
| decode t/s (median) | 28.88 | 28.87 | 1.000 |
| steady decode t/s (median) | 30.00 | 29.98 | 0.999 |

## What is left for sub-project 5

- **Tensor-op tiles.** The `_nax` tiles (Metal 4 cooperative matmul) gave PROD's packs +29-51% prefill
  on M5. They read raw block words (`qwen4_load_raw16` / `qwen4_dequant_raw16`) for Q4_K, Q2_K, IQ2_XXS
  and MXFP4 only. Porting them to the GSQ-RCO types is the next prefill lever, and only worth it if
  ISTA's total answer time still loses to PROD in the harness.
- **Decode** is unchanged at 27 t/s without MTP; the one-row lane helpers are unoptimized (SP3 open item 4).
- **SSD streaming and the draft vocabulary** are unchanged (SP3 open items 2-3).
