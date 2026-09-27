# Ornith M6: accelerator flash adoption (Task 4)

Controller window `m6t4` (2026-09-27 19:50-20:43, live stack paused): snapshots of develop `696d328` and of the
branch at `80db522` (`DS4_QWEN35_ATTN_NAX`, default off), 23G GGUF, F16 K/V, prefill chunk 2048. Follow-up debug
windows `m6dbg*` (20:44-21:35) isolated the one failing check. Adoption rule (M4): gate 1 at chunks 2048/64/65, the MTP
identity tests, the fallback identity check, and a lever A/B that shows a gain.

## Correctness

| check | result |
|---|---|
| model-free kernel tests (`test_qwen35_kernels`) | ok |
| gate 1, develop, chunks 2048 / 64 / 65 | PASS x3 |
| gate 1, branch defaults (knob off) | PASS x3; dumps vs develop: **0 of 39 differ** |
| gate 1, develop with `DS4_METAL_DISABLE_METAL4=1` | PASS x3 |
| gate 1, branch with `DS4_METAL_DISABLE_METAL4=1 DS4_QWEN35_ATTN_NAX=1` | PASS x3; dumps vs develop: **0 of 39 differ** (spec criterion 5) |
| gate 1, `DS4_QWEN35_ATTN_NAX=1`, chunks 2048 / 64 / 65 | **PASS x3** (compared 104, checked 144, same as today) |
| `test_qwen35_graph` / `test_qwen35_mtp` with the knob on | ok / ok |
| `tests/ornith/test_mtp_cli.py` (chunks 1/2/64 + long_copy) with the knob on | PASS (`--mtp` byte-identical to plain) |
| `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` with the knob on | ok |
| same with `DS4_TEST_GLM_MTP=1` | payloads OK, snapshot OK, **qwen35-rewind ERR** (see below) |
| Qwen fast gate | PASS |

Accelerator vs M5 dumps (knob on vs off): 36 of 39 differ; greedy tokens identical 1946 / 2016 (96.5%), top-1 logit
moves by up to 1.47. The M5 kernels' own chunk-64 vs chunk-2048 spread on the same prompts is 654 / 672 identical
and up to 1.25, so the accelerator path moves outputs about as much as a chunk-size change. At step 0 (pure prefill)
short prompts move by <= 1e-3 in log-probability (two Vietnamese prompts by 7e-3 and 2.2e-2), long prompts by
0.12-0.26 (M5's own chunk 64 vs 2048: up to 0.72).

## The qwen35-rewind (MTP) failure: MoE routing near-tie, not a kernel bug

The check: after three MTP cycles, the live session rewinds one token through the verify snapshot (its state was
built by the decode kernels) and must match a fresh session that prefills the same tokens within 2e-3 in
log-probability. With the knob on it misses by 0.16 / 0.12 at its two comparison points.

Isolation (debug build of `ds4_test` printing the values; nothing committed):

| prefill | decode | live vs fresh max abs(d logprob) |
|---|---|---|
| M5 flash | decode3 (default) | 1.1e-4 / 3.1e-4 |
| **accelerator** | **decode3** | **0.163 / 0.117** |
| accelerator | decode2 | 2.0e-4 / 2e-5 |
| accelerator | decode3 with key splits (`SPLIT_KEYS=8`) | 3.2e-4 |
| accelerator | qwen4 per-row decode (`DECODE2=0`) | 1.9e-4 |
| M5 flash | decode2 | 2.2e-4 / 8e-5 |

- The fresh session (pure prefill) does not depend on the decode kernel; the live one moves at the first plain
  decode step after the prefill, with or without MTP, with or without the second session.
- Overwriting the accelerator's prefill output with the M5 flash's (debug switch) makes decode3 match the flash run
  exactly: no side effect, only the accelerator's output values matter.
- decode3 and decode2 run on identical inputs differ by <= 3.7e-4 relative in every attention layer, in both the
  accelerator and the flash run: each kernel is accurate.
- **MoE routing at that decode step:** accelerator + decode3 selects a different expert at layer 13 (241 vs 138),
  then at layers 14, 19, 24, 35, 36, 38 and 39; accelerator + decode2 and flash + decode3 select identical experts
  in all 40 layers. A near-tie in layer 13's top-8 flips under a ~3e-4 difference.

So the check's 2e-3 bound holds only when the prefill and decode kernels share arithmetic (the M5 flash and decode3
are built on the same simdgroup tiles, so the router sees identical inputs). The accelerator kernel is within 4.5e-4
of the double reference, but its rounding differs, and a near-tie in the top-8 router can then flip. The same can
happen, in other states, to any prefill/decode pair that does not share arithmetic (M4's attn_mm + decode2 was such
a pair). The `--mtp` identity guarantee is unaffected (plain and `--mtp` runs prefill with the same kernel).

## Speed

Profile (`DS4_QWEN35_PROFILE=1`, knob on; attention stage = projections + prep + attention + output, 10 layers):

| | M5 | M6 |
|---|---:|---:|
| attention stage, chunk at pos 30,720 | 2,423 ms | 958 ms |
| attention stage, chunk at pos 122,880 | 8,916 ms | 3,303 ms |
| CLI one-shot prefill, ~31K / 128K tokens | 993 / 373 t/s | 1464 / 780 t/s |

Lever A/B (`m4_ab.py --mode lever`, A-B-B-A, `--mtp` with the 64K draft vocabulary, base knob off, lever knob on):

| context | prefill t/s base -> lever | decode t/s base -> lever |
|---|---|---|
| 32K | 994 -> **1666** (+68%) | 55.0 -> 54.2 |
| 128K | 345 -> **868** (+152%) | 37.5 -> 39.1 |
| cold ~31K | 916 -> **1421** (+55%) | 49.2 -> 48.8 |

Raw: `levers/ab-nax.{txt,json}`, `profile/`.

## Verdict

The accelerator flash passes gate 1, the MTP identity tests, the fallback identity and the A/B, but one existing
check fails with it on: `qwen35-rewind` under MTP (cross-path log-probability bound, MoE routing near-tie). Existing
tests are not edited without the user's decision, so **`DS4_QWEN35_ATTN_NAX` stays off by default**; the final gates
run with it on explicitly. Turning it on by default needs the user to accept changing that check (for example,
comparing the rewound session against a fresh one on the top-1 token only, or running the check with the knob off).

## Addendum (2026-09-28): default on

The user chose to turn the accelerator flash on by default. The `qwen35-rewind` check under MTP now builds its fresh
reference the way the rewound session was built (prompt prefill, then one decode step per generated token), so it
checks the snapshot rewind itself rather than prefill-vs-decode kernel parity: the old check fails with the new
default (14 assertions, RED), the new one passes with and without MTP together with `--session-snapshot` and
`--qwen35-payloads` (window `naxdef`, 2026-09-28 06:43-06:46).
