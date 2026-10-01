# SP5 kernels: GSQ-RCO verify rows and tensor-op expert tiles

MTP now speeds ISTA up by ×1.21 to ×1.39 over its own one-token decode. With the old dense gemv the
gain was ×1.13 to ×1.29. ISTA decodes 15-18% faster with MTP and 9% faster without it. ISTA prefill at 8K went from
361 to 424 t/s. PROD output is byte-identical and its speed is unchanged.

- **Why MTP helped ISTA less than PROD:** the GSQ-RCO dense rows re-dequantized every weight for each
  verify token. A 3-token verify cost ×2.8 a decode step on those rows; PROD's Q4_K costs ×2.1.
  `kernel_qwen4_gsq_mv` dequantizes once per step and multiplies up to four tokens. Its verify
  columns equal the one-token rows bit for bit.
- **What is left of the gap:** PROD drafts against a 64K frequency-gathered vocabulary
  (`DS4_QWEN4_MTP_DRAFT_VOCAB`, vi-en-code list). ISTA's Q5_K head cannot use that loader, which takes
  Q8_0 heads only, so ISTA drafts against the full vocabulary. A 64K prefix
  (`DS4_QWEN4_MTP_DRAFT_ROWS=65536`) gives English prompts about +5% but drops the Vietnamese prompt's
  acceptance to 26%, so it is not an option.
- **Prefill:** the routed experts now run on the Metal 4 tensor-op tiles (+24% over the simdgroup
  tiles in the same session). The experts are no longer the bottleneck: the whole MoE is 24% of
  prefill time. The other 76% is attention, GDN, the hyper-connections and the dense projections, which
  still take the float `kernel_qwen4_dense_mm` tiles.

Plans: `docs/superpowers/plans/2026-10-01-gsq-rco-verify-rows.md` (commits `0851154`, `919a83e`) and
`docs/superpowers/plans/2026-10-01-gsq-rco-nax-tiles.md` (commit `2abfccd`). The GPU window ran
2026-10-01 08:42-08:58 with the gateway stack paused. The `vm_stat` swap-out counter did not move.

## Kernel level (`./ds4_test --metal-kernels`)

The kernel level is unchanged from the plans:
- `test_metal_qwen4_gsq_mv` checks the nine GSQ-RCO types plus BF16 against CPU rows. It also
  `memcmp`s 2, 3 and 6 tokens (R1 = 2, 3, 4 with a tail) against one-token rows.
- The MoE GEMM test asserts the tensor-op tile width for the GSQ-RCO types.
- `DS4_QWEN4_MOE_MM_NAX=1,3,4,5,6` each pass.

Dense gemv on a 2560×24576 matrix (ms at 1 / 2 / 3 tokens):

| type | old | new |
|---|---|---|
| Q6_K | 0.239 / 0.451 / 0.667 | 0.218 / 0.250 / 0.291 |
| Q5_K | 0.206 / 0.389 / 0.572 | 0.208 / 0.260 / 0.298 |
| IQ4_XS | 0.182 / 0.341 / 0.499 | 0.175 / 0.230 / 0.281 |
| IQ4_NL | 0.186 / 0.348 / 0.504 | 0.184 / 0.233 / 0.286 |
| IQ3_S | 0.162 / 0.300 / 0.431 | 0.142 / 0.200 / 0.254 |
| Q2_0 | 0.123 / 0.229 / 0.335 | 0.104 / 0.158 / 0.202 |

PROD's Q4_K kernel, for reference: 0.127 / 0.192 / 0.274.

## Model level

### Correctness

`DS4_QWEN4_GPU=1 ./ds4 --first-token-test`, Metal against `qwen4_ref_forward_token`:

| chunk | prompt | worst max\|gpu-cpu\| | top1 agree | before |
|---|---|---|---|---|
| 1 | "Viết một câu ngắn về Hà Nội." (31 tokens) | 3.098 | 30/31 | SP3: 3.098, 30/31 |
| 256 | 181-token code + prose prompt | 3.070 | 1/1 | prefill GEMM: 3.021 |

The chunk-256 drift moves by 0.05. That comes from the experts' tensor-op tiles, which stage operands
as half; it is the accepted drift class.

### MTP decode (ISTA)

`./ds4 -m ISTA --ple PLE --metal -c 32768 --nothink --temp 0 -n 256`, with `--mtp-timing` for the MTP
arms. Three prompts: Python prime check, Vietnamese "why is the sky blue", TCP vs UDP. Generation
t/s, with draft acceptance for MTP:

| arm | prompt 1 | prompt 2 (vi) | prompt 3 |
|---|---|---|---|
| no MTP, old gemv (`DS4_QWEN4_GSQ_MV_LEGACY=1`) | 28.27 | 28.14 | 28.29 |
| no MTP, new gemv | 30.80 | 30.77 | 30.65 |
| MTP, old gemv | 36.49 (98.0%) | 31.81 (77.0%) | 33.69 (85.3%) |
| MTP, new gemv, run a | 40.95 (98.0%) | 37.23 (77.3%) | 39.27 (85.3%) |
| MTP, new gemv, run b | 42.85 (98.0%) | 38.05 (77.3%) | 40.10 (85.3%) |
| MTP, new gemv, `DS4_QWEN4_MTP_DRAFT_ROWS=65536` | 43.77 (96.1%) | 28.24 (26.2%) | 41.51 (84.7%) |

MTP gain over the same kernel's no-MTP decode:

| kernel | prompt 1 | prompt 2 | prompt 3 |
|---|---|---|---|
| old gemv | ×1.29 | ×1.13 | ×1.19 |
| new gemv | ×1.33-1.39 | ×1.21-1.24 | ×1.28-1.31 |

Run b was meant as the no-MTP arm. Its command kept `--mtp-timing`, which also turns MTP on
(`ds4_cli.c:2033`), so it is a second MTP run. The real no-MTP rows ran later in the same gateway hold.

**MTP against no-MTP output:**
- Prompts 1 and 3 are byte-identical under both kernels.
- Prompt 2 diverges at byte 284 under both kernels. MTP picks "(như **màu xanh dương** và tím)" where
  no-MTP picks "(như **màu xanh dương**)", a near-tie.
- The pairs cross: new-MTP equals old-no-MTP, and old-MTP equals new-no-MTP. The tie therefore flips on
  numerics outside the gemv. The gemv's verify columns are bit-equal at kernel level, so the cause sits
  in another verify-path component (attention, GDN, MoE or the mixed Q4_K fused groups) and predates
  this work.
- Not investigated yet. PROD's MTP against no-MTP output is unmeasured.

### Speed (`ds4-bench`, 8K story prompt, 128 generated tokens, no MTP)

| model | arm | prefill t/s | decode t/s |
|---|---|---|---|
| ISTA | new, 3 runs | 417.76 / 424.04 / 425.22 | 29.56 / 29.79 / 29.90 |
| ISTA | `DS4_QWEN4_MOE_MM_NAX=0` | 341.60 | 29.92 |
| PROD | baseline `6985bdb`, 3 runs | 522.81 / 592.77 / 583.95 | 28.98 / 28.95 / 29.26 |
| PROD | new, 3 runs | 575.72 / 580.05 / 603.54 | 28.98 / 29.39 / 29.30 |

PROD medians: prefill 583.95 → 580.05 (×0.993), decode 28.98 → 29.30 (×1.01). PROD's output is
byte-identical on all three prompts, at 128 tokens and `-c 262144` with the registry flags.

### Where the time goes

**Prefill:** ISTA at 8192 tokens takes 19.5 s (`DS4_QWEN4_MOE_PROFILE=1`, which syncs per stage, so it
runs at 419 t/s). The MoE, summed over 48 layers:

| part | ms |
|---|---|
| routed mid (gate+up) | 2139 |
| routed down | 1111 |
| shared mid | 532 |
| shared down | 317 |
| reduce | 318 |
| router | 276 |
| lists | 11 |
| **MoE total** | **4705 (24%)** |

Matching PROD's ~14.1 s needs about 5.4 s off the other 14.8 s. The next prefill lever is therefore the
dense GSQ-RCO projections, which still take the float `kernel_qwen4_dense_mm` tiles, not the experts.

**Decode:** `DS4_QWEN4_STAGE_TS_PROFILE=1`, 128 steps, 33.4 ms per step with timestamps:

| stage | share |
|---|---|
| moe | 30.0% |
| gdn | 21.1% |
| hc_ffn | 16.5% |
| hc_attn | 15.0% |
| attn | 10.5% |
| output | 6.0% |
| ple | 0.9% |

The hyper-connection stages carry ISTA's F32 `hc_*_up` rows (PROD stores them F16).

## Open items

- **Draft-vocabulary gather for non-Q8_0 heads:** this would let ISTA use PROD's
  `vi-en-code-64k` list. The English prompts suggest about +5%; Vietnamese should keep its acceptance.
- **Dense GSQ-RCO projections on tensor-op tiles:** the prefill lever above.
- **The MTP/no-MTP near-tie divergence:** find the verify-path component, and measure PROD the same way.
- **`hc_*_up` in F16/BF16 instead of F32:** about 600 MiB per token less traffic. This changes the
  published GGUF.
