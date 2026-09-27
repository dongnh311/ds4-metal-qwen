# Ornith M5 report: attention kernels

Branch `feature/ornith-m5` (from develop `ffba1d1`), executed 2026-09-27 11:05-17:30, subagent-driven (sonnet
implementers and task reviewers, one opus whole-branch review, one fix wave). Every GPU measurement ran with the
live AI-Gateway stack paused, one model process at a time, no process SIGKILLed, and the stack restored
(`backend_ok`) after every window. Spec: `docs/superpowers/specs/2026-09-27-ornith-m5-attention-design.md`.

## Verdict against the spec's success criteria (§1)

| # | criterion | result |
|---|---|---|
| 1 | prefill >= live oMLX at 32K and 128K; cold ~31K TTFT <= oMLX | **FAIL**: 983 vs 1677, 354 vs 739 t/s; TTFT 32.4 vs 20.9 s |
| 2 | decode with `--mtp` >= live oMLX at 32K and 128K | **128K met** (39.3 vs 38.9 t/s), **32K not** (53.8 vs 64.6) |
| 3 | gate 1 at prefill chunks 2048 / 64 / 65 | PASS (every knob combination; final code byte-identical dumps) |
| 4 | `--mtp` byte-identical to plain decoding | PASS (`test_qwen35_graph`, `test_qwen35_mtp`, `test_mtp_cli.py`, live server test) |
| 5 | gate 2 at parity with the live oMLX (index >= oMLX - 3.0) | PASS at parity: 23G 83.2, 25G 81.7 vs bar 75.8 (see below for the matrix) |
| 6 | Qwen3.8 byte-identical | PASS (full gate: replies identical, needle hit, ~99% of PROD speed) |

Gate 3 fails on prefill (and on 32K decode); M5 still roughly doubles long-context prefill and nearly doubles
128K decode.

## Speed: M4 -> M5 vs the live oMLX (A-B-B-A, `m4_ab.py`, same harness and day; `speed/GATE3.md`)

| context | ds4 decode t/s (M4 -> M5) | oMLX decode | ds4 prefill t/s (M4 -> M5) | oMLX prefill |
|---|---|---|---|---|
| 2K | 65.0 -> 69.1 | 80.2 | 1656 -> **1738** | 1634 |
| 32K | 47.1 -> 53.8 | 64.6 | 650 -> 983 | 1677 |
| 128K | 22.0 -> **39.3** | 38.9 | 187 -> 354 | 739 |
| cold ~31K first turn | TTFT 49.6 -> 32.4 s | 20.9 s | 626 -> 958 | 1515 |

## What was built

- **`kernel_qwen35_attn_decode3`** (decode and the 2-row MTP verify): the 8 GQA query heads of each verify row as
  one simdgroup-matrix tile, K/V tiles of 16 keys shared by both rows, up to 256 key splits per row with a per-row
  split geometry, `kernel_qwen35_attn_merge3` for the partials. A rows==2 call equals two rows==1 calls bit for bit
  (memcmp-tested at split boundaries, tile boundaries and the split cap). One layer at 131,072 keys: 1.71 ms plain /
  1.93 ms verify, down from decode2's 2.46 / 5.34 ms (`BENCH.md`). The plan's per-key design was built first,
  measured 2.5x slower than decode2 and replaced (ruling R6).
- **`kernel_qwen35_attn_flash`** (prefill chunks with T > 8): two query tokens x 8 heads per threadgroup share every
  K/V tile, no padding rows; output bit-identical to `attn_mm`; per-chunk prefill attention at 30,720 / 122,880:
  2.4 / 8.9 s, down from 4.3 / 19.7 s. Short chunks at long context split the key range (-8..10% at T = 128).
- **Dispatch**: `DS4_QWEN35_ATTN_DECODE` (default 3; 2 = decode2) and `DS4_QWEN35_ATTN_FLASH` (default on; 0 =
  attn_mm), F16 K/V only; fp8/q4 K/V, T <= 8 tails and `DS4_QWEN35_ATTN_DECODE2=0` keep the older kernels.
- **Tests and tools**: `test_attn_decode3_rows` (8 cases), `test_attn_flash` (8 cases incl. forced splits and
  neutral partials), `tests/bench_qwen35_attn` (`make bench-qwen35-attn`).

Adoption (`LEVERS.md`): decode3 A/B +15% decode at 32K, +84% at 128K; flash A/B +53% prefill at 32K, +82% at 128K,
+51% cold ~31K. Both passed gate 1 and the MTP identity tests before their defaults flipped.

## Quality (gate 2, `quality/GATE2.md`)

| runtime | HumanEval-mini | GSM8K-mini | code_bench | review F1 | INDEX | truncated / errored |
|---|---|---|---|---|---|---|
| ds4 23G | 1.000 | 1.000 | 0.778 | 0.552 | 83.2 | 0 / 0 |
| ds4 25G | 1.000 | 1.000 | 0.667 | 0.600 | 81.7 | 0 / 0 |
| live oMLX (same day) | 0.882 | 1.000 | 0.555 | 0.714 | 78.8 | 0 / 0 |

Probes: 4/4 COMPLY, identical to the live oMLX. Matrix: 4/7 — `hard` and `bugfix` each hit the 480 s turn timeout
on their audit turn (speed; their tests pass), and `testgen` failed a hidden style check (bare-name calls). testgen
fails at the same rate with M4's kernels (M5 3/8, M4 3/5 over alternating reruns, same failure mode), so it is
sampling variance at temperature 0.7, not an M5 regression; the literal "no quality failure in one matrix run" is
not met for that reason.

## Why the prefill gap remains, and what would close it

The flash kernel removed the per-token K/V re-read and the padding rows; it is now bound by simdgroup-matrix
throughput (~4.8 TFLOP/s of attention at 30K; TOK=4 and key splits do not move it). A 32K prompt still spends most of
its time in attention (1.07 TFLOP per layer at 30K). The live oMLX's prefill rate implies more matrix throughput
than this path gives on the M5 Pro, which points at the M5 neural accelerators through the Metal 4 tensor API —
out of M5's scope (spec §1). The repo already has a tensor-API attention kernel (DSV4.1, `metal/dsv41.metal`) to
start from; it needs a non-M5 fallback (the simdgroup kernels here).

The 32K and 2K decode gaps are no longer attention: at 128K a step spends 17.6 ms in attention against ~15 ms in the
GDN layers, flat in context; at 2K/32K the GDN, MoE and MTP overhead dominate — the separate 2K-decode milestone.

## Recommended next milestones

1. **Tensor-API prefill attention** (Metal 4 matmul2d on the neural accelerators, simdgroup flash as the fallback):
   the only lever in sight for gate 3's prefill and TTFT criteria.
2. **Short/medium-context decode** (GDN per-step cost, MoE decode kernels, MTP verify overhead): the 2K/32K decode
   gap (69 vs 80 and 54 vs 65 t/s).

## Deploy configuration (not deployed)

`DS4_QWEN35_MTP_DRAFT_VOCAB=<path>/ornith-draft-vocab-vi-en-code-64k.txt ds4-server --metal -m <23G|25G GGUF>
-c 262144 --mtp --prefill-chunk 2048 --kv-disk-dir <dir>` — decode3 and flash are defaults (K/V F16); rollback
knobs `DS4_QWEN35_ATTN_DECODE=2`, `DS4_QWEN35_ATTN_FLASH=0`.

## Open items

- The matrix's audit turns time out at 480 s on `hard` and `bugfix` (speed).
- decode3 at 131,072 keys (verify) is 1.93 ms vs the 1.7 ms aim; the flash per-chunk targets (0.6 s / 2.5 s) were
  not reachable on the simdgroup path.
- The MTP byte-identity now rests on two separately compiled decode3 instantiations (rows 1 and rows 2) computing a
  row identically; `make test-qwen35-kernels` (with and without `DS4_METAL_DISABLE_METAL4=1`) pins it and should run
  after any Xcode / macOS toolchain update.
- Parked M3/M4 items are unchanged (MTP carry after a disk restore, trailing-whitespace cache cost).
