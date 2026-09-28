# Ornith M7: batched verify, gates and lever A/B (Tasks 3-5)

`DS4_QWEN35_VERIFY_BATCH=1`: the 2-row MTP verify runs its Q8_0 projections (attention q/output, GDN
lin_qkv/lin_gate/lin_out, lm head) as one two-row matvec each (`kernel_qwen35_mv_q8_0_rows2`), weights read once,
per-row arithmetic of `kernel_mul_mv_q8_0_f32` kept. 23G GGUF, Apple M5 Pro 64 GB, live stack paused in every window.

## Correctness

Window `m7t34` (08:22-08:26, snapshots of each RED/GREEN commit) and `m7t5` (08:29-09:01, snapshot `a0577f7`;
`DS4_QWEN35_VERIFY_BATCH=1` unless noted). Raw: `tests/`.

| check | result |
|---|---|
| `test_qwen35_kernels` (model-free): two-row matvec vs T=1, memcmp, 5 shapes x3 | ok |
| `test_qwen35_verify_batch` at `fe0a8b5` (test only, RED) | FAIL as expected: 0 verifies counted |
| `test_qwen35_verify_batch` at `2774c34` (dense projections) | ok: 149 verifies x 21 two-row matvecs, logits bit-identical to plain every cycle |
| `test_qwen35_verify_batch` at `a699ea6` (expects GDN too, RED) | FAIL as expected: "cycle 1: 21 two-row matvecs, expected 0 or 111" |
| `test_qwen35_verify_batch` at `a0577f7` (GDN layers) | ok: 149 verifies x 111, bit-identical |
| `test_qwen35_mtp` knob on (`2774c34`, `a0577f7`) / knob off (`2774c34`) | ok / ok / ok (greedy 150 cycles 115 accepted, forced accepts, divergent prompt, context end: all bit-identical to plain) |
| `tests/ornith/test_mtp_cli.py` (chunks 1/2/64/0 + long_copy) | PASS: `--mtp` byte-identical to plain |
| `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind`, without / with `DS4_TEST_GLM_MTP=1` | ok / ok |
| `test_qwen35_graph` | ok |
| gate 1, chunks 2048 / 64 / 65 | PASS x3 (compared 104, checked 144) |
| gate 1 dumps (chunk 2048) vs develop `b764a66` | **0 of 13 differ** (plain decode unchanged) |
| `make test-qwen4-kernels test-qwen4-q2`, Qwen fast gate (`run.sh fast`) | pass, PASS (Task 2 window and `m7t5`) |

## Profile (`DS4_QWEN35_PROFILE=2`, ~2.3K-token prompt, 256 tokens; per-layer syncs inflate the absolutes)

| | ms |
|---|---:|
| plain T=1 step (no `--mtp`, mean of three 64-step averages: 20.9 / 22.5 / 23.5) | 22.3 |
| verify, knob off (median of 178) | 34.5 (**1.55x** a plain step) |
| verify, knob on (median of 178) | 26.9 (**1.21x**; spec target <= 1.35x) |

Raw: `profile/plain-2k.txt`, `profile/verify-2k-batch-{on,off}.txt`.

## Lever A/B (`m4_ab.py --mode lever`, A-B-B-A, `--mtp` + 64K draft vocabulary, 256 tokens)

| context | decode t/s base -> lever | prefill t/s base -> lever |
|---|---|---|
| 2K | 69.9 -> **91.3 (+30.6%)** | 1837 -> 1833 |
| 32K | 54.7 -> **68.1 (+24.5%)** | 1658 -> 1662 |
| 128K | 41.5 -> **50.2 (+21.0%)** | 896 -> 894 |
| cold ~31K | 54.6 -> **70.5 (+29.1%)** | 1534 -> 1532 |

Per repetition (decode t/s): base 67.1 / 72.8 (2K), 53.6 / 55.8 (32K), 42.0 / 41.1 (128K); lever 90.9 / 91.6,
66.8 / 69.4, 49.8 / 50.5. Raw: `levers/ab-verify-batch.{txt,json}`.

## Verdict

Adopt: every identity check holds (the verify is exact by construction and by test), plain decode is byte-unchanged,
the stop rule (>= +5% at 2K) passes with +30.6%, and no context regresses. For reference, M6's gate 3 measured the
live oMLX at 79.9 / 64.5 / 38.9 t/s decode (2K / 32K / 128K); the same-day comparison is Task 6's gate 3.

## Addendum: startup self-check and per-row fallback (final review fix)

Window `m7t7` (2026-09-28 10:13-10:22), RED `b563f6a` / GREEN `b923d2d`. The engine compiles a copy of
`metal/qwen35.metal` through `DS4_METAL_QWEN35_SOURCE` whose two-row kernel adds 1e-3 to every partial sum
(`--perturbed-kernel`) or is missing (`--broken-kernel`). Raw: `tests/m7t7-*.txt`.

| run | RED (no self-check) | GREEN |
|---|---|---|
| `--perturbed-kernel` | FAIL: "cycle 1: 111 two-row matvecs, expected 0" (the inexact kernel ran in the verify) | ok: "batched verify off: ... failed its startup check", 0 two-row matvecs, bit-identical to plain |
| `--broken-kernel` | FAIL: "Ornith mtp: verify failed" | ok: same fallback, bit-identical |
| default / knob on / knob off | - | ok: 149 x 111 / 149 x 111 / 0 (the self-check passes on this toolchain) |
| `test_qwen35_mtp`, `test_mtp_cli.py`, Qwen fast gate | - | ok, PASS, PASS |
