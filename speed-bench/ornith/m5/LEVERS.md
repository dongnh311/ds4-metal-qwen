# Ornith M5: kernel adoption (Tasks 5-6)

One combined controller window (`m5t56`, 2026-09-27 13:18-15:16, live stack paused) on a snapshot of `efa0b1e`
(decode3 and flash dispatch, both still opt-in), 23G GGUF, F16 K/V, prefill chunk 2048. Adoption rule (M4):
gate 1 at prefill chunks 2048/64/65, the MTP identity tests, and a `m4_ab.py --mode lever` A/B that shows a gain.

## Correctness

| check | decode3 (`DS4_QWEN35_ATTN_DECODE=3`) | flash (`DS4_QWEN35_ATTN_FLASH=1`) | both |
|---|---|---|---|
| gate 1, chunks 2048 / 64 / 65 | PASS x3 | PASS x3 | PASS x3 |
| gate-1 dumps vs today's kernels | 39/39 differ (decode steps now stage Q/K/P as half) | 2/39 differ | — |
| `test_qwen35_graph` | ok | — | ok |
| `test_qwen35_mtp` | ok | — | ok |
| `tests/ornith/test_mtp_cli.py` (chunks 1/2/64 + long_copy) | PASS | — | PASS |
| `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` (f16/fp8/q4 payload loop) | — | — | ok, and ok with `DS4_TEST_GLM_MTP=1` |
| model-free kernel tests | ok | ok | ok |
| Qwen fast gate | — | — | PASS |

The two flash dumps that differ are `long_copy` (9,371 tokens) at chunks 64 and 65 only: short chunks past ~2K keys
take the key split, whose merge sums in a different order. Greedy tokens are identical (32/32); the top-1 logit
moves by <= 0.74, the same size as the gap between today's chunk-64 and chunk-2048 runs of that prompt (<= 0.83).
At chunk 2048 flash output is byte-identical to attn_mm (the TOK=2 kernel matches it bit for bit).

## Speed (lever A/B, A-B-B-A, fresh nonce per prompt, `cached_tokens == 0`)

Both arms run with the 64K MTP draft vocabulary (`DS4_QWEN35_MTP_DRAFT_VOCAB`), `ds4-server --mtp`.

decode3 (base `FLASH=1 DECODE=2`, lever `FLASH=1 DECODE=3`, `--cold-tokens 0`):

| context | decode t/s base -> lever | prefill t/s base -> lever |
|---|---|---|
| 32K | 46.8 -> 53.8 (+15%) | 992 -> 995 |
| 128K | 23.5 -> 43.2 (+84%) | 368 -> 368 |

flash (base `DECODE=3 FLASH=0`, lever `DECODE=3 FLASH=1`, `--cold-tokens 31000`):

| context | prefill t/s base -> lever | decode t/s base -> lever |
|---|---|---|
| 32K | 651 -> 995 (+53%) | 53.8 -> 53.1 |
| 128K | 201 -> 366 (+82%) | 41.8 -> 41.6 |
| cold ~31K | 648 -> 980 (+51%) | 50.9 -> 54.2 |

Raw: `levers/ab-decode.{txt,json}`, `levers/ab-flash.{txt,json}`.

## Profile (`DS4_QWEN35_PROFILE`, both kernels on; raw in `profile/`)

| | M4 | M5 |
|---|---|---|
| prefill attention, chunk at pos 30,720 (10 layers) | 4,305 ms | 2,423 ms |
| prefill attention, chunk at pos 122,880 | 19,700 ms | 8,887 ms |
| prefill t/s, 32K / 128K CLI one-shot | 652 / 186 | 993 / 373 |
| decode attention per step at 128K (plain, level 2) | 32.2 ms | 17.6 ms |

The plan's per-chunk attention targets (<= 0.6 s at 30,720, <= 2.5 s at 122,880) are missed; see `BENCH.md`
(the prefill kernel is bound by simdgroup-matrix throughput, ~4.8 TFLOP/s).

## Verdict

Both kernels pass the adoption rule and are now the defaults: `fcda818` (`DS4_QWEN35_ATTN_DECODE` defaults to 3; 2
restores decode2) and `b2e97a6` (`DS4_QWEN35_ATTN_FLASH` defaults to on; 0 restores attn_mm). fp8/q4 K/V and
`DS4_QWEN35_ATTN_DECODE2=0` keep today's kernels.
