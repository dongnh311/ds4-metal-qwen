# ISTA speed: dense tensor-op tiles, draft-vocabulary gather, PROD think budget

ISTA prefill at 8K went from 423 to 737 t/s (×1.74, same session, `DS4_QWEN4_DENSE_NAX=0` vs default).
The MTP draft-vocabulary gather works for ISTA's Q5_K head but does not make it faster: the head is
now about 3% of a verify cycle. PROD output is byte-identical and its speed is unchanged.

The GPU window ran 2026-10-01 from 18:41 with the gateway stack paused (hold-gateway.sh). Branch
`feature/ista-speed`, cut from develop `f28e41d`. PROD baseline binary: develop `3a3bfa3`, the build
deployed as `prod/qwen-512k-20261001`.

## What changed

- **Dense tensor-op tiles** (`metal/qwen4.metal`, `ds4_metal.m`): the GSQ-RCO dense types (Q5_K, Q6_K,
  IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0) and BF16 now instantiate the existing Metal 4
  `kernel_mul_mm_mpp_direct_rhs` template with a GSQ dequantizer (`qwen4_dense_dequant`, built on
  `qwen4_gsq_deq8`). `ds4_gpu_qwen4_dense_mm_tensor` takes them when the batch is whole 32-token
  tiles (n ≥ 32, n % 32 == 0), the input width is a multiple of 64 and of the type's block, and the
  output rows are a multiple of 64. The tile width is 128, 64 or 32 tokens by divisibility. Decode and
  verify rows, Q4_K, Q8_0 and IQ2_XXS keep their kernels. `DS4_QWEN4_DENSE_NAX=0` turns the path off.
- **Draft-vocabulary gather for non-Q8_0 heads** (`ds4.c`, commit `16ddccfe`):
  `qwen4_mtp_draft_head_load` used to require a Q8_0 output head. It now takes Q8_0, IQ2_XXS and the
  GSQ-RCO types, and requantizes each gathered row to Q8_0 (`ds4_quant_row_to_q8_0`), so the draft
  still runs on `matmul_q8_0_weights`. Q8_0 heads (PROD) are copied unchanged.

## Kernel level (`./ds4_test --metal-kernels`)

OK. `test_metal_qwen4_quant_dense_mm` now:
- asserts the routing (`ds4_gpu_qwen4_dense_nax_selected`): Q5_K and BF16 at 64/128 tokens take the
  tensor-op tiles; 40 tokens, 8 tokens, 37 rows, Q4_K and Q8_0 do not;
- checks every GSQ-RCO fixture type and BF16 on 64 rows at 64, 128 and 96 tokens against CPU rows
  (tolerance 1.5e-3, since the tiles load the activations as half), next to the float tiles at 40
  and 9 tokens (1e-5).

The CPU unit tests for the gather (`test_quant_dequant`, `test_quant_types`) check the Q8_0
requantization of every fixture type against the bound
`0.5·d + 127·|d_fp16 − d| + 1e-6·amax`.

## ISTA correctness (GPU vs CPU, 181-token prompt)

| chunk | worst max\|diff\| | top-1 agree |
|---|---|---|
| 64 | 3.525 | 3/3 |
| 128 | 2.895 | 2/2 |

The bar is the one from the prefill GEMM plan: max(Ivan chunk 256, ISTA chunk 64) × 1.25 = 5.31.
Both pass. Chunk 64 rose from 2.410 (simdgroup tiles, this morning) to 3.525; the tensor-op tiles
round the activations to half.

## ISTA prefill (`ds4-bench`, 8K story prompt, 128 generated)

| run | prefill t/s | decode t/s |
|---|---|---|
| tensor-op dense, round 1 | 715.86 | 28.67 |
| tensor-op dense, round 2 | 738.85 | 29.91 |
| tensor-op dense, round 3 | 736.74 | 29.85 |
| `DS4_QWEN4_DENSE_NAX=0` | 423.09 | 29.86 |
| this morning (results 7feeefe) | 424.86 | 29.47 |

Decode does not move: one-token rows never take the tiles.

## ISTA MTP with the draft vocabulary (`-n 256`, temperature 0)

| prompt | MTP + draft vocab | MTP, full head | no MTP |
|---|---|---|---|
| prime function (EN code) | 42.26 t/s, 98.0% | 42.88, 98.0% | 30.23 |
| blue sky (VI) | 36.93, 77.3% | 38.03, 77.3% | 30.25 |
| TCP vs UDP (EN) | 39.39, 81.7% | 39.19, 85.3% | 29.25 |
| mean | 39.53 | 40.03 | 29.91 |

- The head loads as `65568 of 248320 vocabulary rows … (170 MiB, requantized to Q8_0)`.
- Output with the draft vocabulary equals output with the full head on all three prompts, and the
  full-head and no-MTP outputs equal this morning's.
- MTP vs no-MTP differs on the Vietnamese prompt (char 284). This is the same near-tie seen this
  morning, not a change from this branch.
- The gather costs acceptance where the true token is outside the 64K list (TCP/UDP: 85.3% → 81.7%)
  and saves about 250 MiB of head reads per draft, about 1.3 ms of a ~46 ms cycle. The net is within
  noise. Since the dense gemv fix (results 7feeefe), the full Q5_K head is not where ISTA's MTP time
  goes.
- Recommendation: keep `DS4_QWEN4_MTP_DRAFT_VOCAB` off for ISTA. The code stays: it is correct,
  tested, and a no-op for Q8_0 heads.

## PROD regression check

PROD's dense tensors are Q8_0 and Q4_K only (`token_embd` is BF16 but is a lookup), so no PROD
matmul takes the new tiles.

- Identity, 3 prompts × 128 tokens vs the `3a3bfa3` binary: 3/3 byte-identical.
- `ds4-bench` A/B (prefill / decode t/s):

| round | baseline `3a3bfa3` | this branch |
|---|---|---|
| 1 | 489.05 / 28.89 | 567.72 / 28.65 |
| 2 | 575.37 / 29.00 | 578.99 / 28.83 |

Round 1's baseline ran first after the ISTA model was unmapped and paid the page-in; round 2 is the
warm comparison (+0.6%, noise).

## PROD think budget (4096 vs 8192)

Keep 4096. On the 24 ifeval items most likely to need more thinking, 8192 passes the same 22 items
and takes 40% longer.

The items are the ones that hit the 4096 budget in the earlier PROD and ISTA arms
(`think-ab-ids.json`, first 24 of 48). Both arms ran PROD's binary (`3a3bfa3`) and the harness
config, at temperature 0, with only `--think-budget` changed (`prod-tb8192.json`). Runs:
`runs/think-ab-4096-20261001`, `runs/think-ab-8192-20261001`.

| | 4096 (PROD) | 8192 |
|---|---|---|
| passed | 22/24 | 22/24 |
| hit the budget | 18 | 11 |
| mean think tokens | 3619 | 5932 |
| mean generated tokens | 4956 | 6876 |
| mean seconds per item | 126.8 | 177.1 |
| `finish=length` | 1 | 1 |

- No item changed its verdict. The two failures (1300 letter frequency, 1883) fail in both arms. 18
  items thought past 4096 with the larger budget; 16 of them pass in both arms.
- 11 of the 18 ran into 8192 as well: the extra room goes to more thinking, not to an answer that
  passes where the shorter one failed.
- The earlier finding that budget-hit items pass 80-95% against 97-98% overall was a correlation:
  hard items think longer. Paired on the same items, the budget costs nothing here.
- Real traffic on the Qwen row is small (106 requests since 2026-09-15), so the budget mostly
  bounds worst-case latency.

The window's `vm_stat` swap-out counter rose 108k pages while the 47 GB ISTA model was made resident
(18:41-18:42) and 15k pages over the remaining 2h20m.
