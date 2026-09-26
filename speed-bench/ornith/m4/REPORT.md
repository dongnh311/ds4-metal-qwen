# Ornith M4 report: acceptance (speed + quality)

Branch `feature/ornith-m4` (from develop `b9f1aca`), executed 2026-09-27 00:05-06:45, subagent-driven (sonnet
implementers and task reviewers, one opus whole-branch review + one fix wave). Every measurement ran with the live
AI-Gateway stack paused (watchdogs unloaded, oMLX stopped with SIGTERM), one model process at a time, no process
ever SIGKILLed, and the stack restored (`backend_ok`) after every window.

## Verdict
- **Gate 3 (speed): FAIL.** ds4 is below the live oMLX at every context on decode and on long-context prefill
  (`speed/GATE3.md`).
- **Gate 2 (quality): FAIL on the index** (23G 78.0, 25G 84.7 vs the 86.6 bar; truncated = errored = 0), with
  the live oMLX scoring 78.8 on the same harness today (quality parity; the 86.6 bar is from an older gateway run). Abliteration probes 4/4 COMPLY, identical to the live oMLX. Agentic matrix: 6/7 (the hard case hit the 480 s turn timeout on one turn; its tests pass).
- M4 made ds4 materially faster (below) and found where the remaining gap is: **attention**, prefill and decode.

## Speed: baseline -> final vs live oMLX (A-B-B-A, `m4_ab.py`)
| context | ds4 decode t/s (M1 -> M4) | oMLX decode | ds4 prefill t/s (M1 -> M4) | oMLX prefill |
|---|---|---|---|---|
| 2K | 61.7 -> 65.0 | 81.2 | 543 -> 1656 | 1737 |
| 32K | 41.7 -> 47.1 | 64.4 | 358 -> 650 | 1697 |
| 128K | 20.4 -> 22.0 | 38.1 | 150 -> 187 | 713 |
| cold ~31K first turn | TTFT 92.4 s -> 49.6 s | 22.5 s | 336 -> 626 | 1413 |

ds4 runs `--mtp` with the M4 defaults and the MTP draft vocabulary; oMLX is the live registry command.

## What changed and what it bought
- **Q5_K tiled prefill GEMM** (layers 0-14's experts; extractor-generated from the Q4_K tile templates): 2K prefill
  543 -> ~1700 t/s, now at oMLX parity; gate 1 passes at chunks 65/2048/64.
- **Stage profile** (`profile/PROFILE.md`): prefill attention grows linearly with position (per 2048-token chunk
  410 ms at pos 0, 4.3 s at 30K, 19.7 s at 123K; 96% of a 128K prefill); decode attention ~32 ms/token at 128K
  (~3x the K/V bandwidth floor); the exact 2-row MTP verify doubled it (63 ms) before L12.
- **Prefill chunk sweep** (`profile/CHUNK_SWEEP.md`): 2048/4096/8192 = 620/630/633 t/s at 32K; default 2048 kept.
- **Levers** (`speed/LEVERS.md`): kept L1 early flush (+5% decode at 2K), L8 Q5_K decode kernels (+4% at 2K, M5
  default), L12 shared-K/V decode (+5% at 32K; plain decode and verify rows share the kernel, so `--mtp` stays
  byte-identical), MTP draft vocabulary (+19% `--mtp` decode at 2K on Italian prose); rejected L2 (no gain), L4 (not
  exact), L5 (inapplicable with `--mtp`), K/V fp8 (-49% decode at 32K) and q4 (-3%).
- L12 without `--mtp` (plain CLI decode, DS4_QWEN35_ATTN_DECODE2=0 vs 1): 2K 65.6 -> 68.3 t/s, 32K 46.6 -> 47.3 t/s,
  so the default does not slow users who run without MTP.

## Quality (gate 2, `quality/GATE2.md`)
| runtime | HumanEval-mini | GSM8K-mini | code_bench | review F1 | INDEX | truncated / errored |
|---|---|---|---|---|---|---|
| ds4 23G | 1.000 | 1.000 | 0.667 | 0.455 | 78.0 | 0 / 0 |
| ds4 25G | 1.000 | 1.000 | 0.667 | 0.720 | 84.7 | 0 / 0 |
| live oMLX (same harness) | 0.882 | 1.000 | 0.555 | 0.714 | 78.8 | 0 / 0 |

ds4 23G's review score is lowered by one malformed JSON answer (a missing quote), not by missed bugs. Matrix 6/7:
the one failure is a turn timeout (speed). Probes: ds4 4/4 COMPLY, the same labels as the live oMLX. Against the
plan's literal rule gate 2 fails; against the live oMLX it is at parity. Details: `quality/GATE2.md`.

## Why gate 3 is out of reach with this plan, and what would close it
The MoE side is fixed (2K prefill at parity) and the GDN layers are flat with context. The remaining gap is the
attention kernels Ornith borrows from the qwen4 path: a prefill attention whose cost per chunk grows with position
without K/V tile reuse, and a decode attention ~3x off the bandwidth floor for Ornith's 2-KV-head x 256-dim shape
(the shared FP8/4-bit K/V paths are slower still). Next step (a new plan, not M4): an Ornith attention pair — a
flash-style prefill kernel and a decode kernel near the K/V bandwidth floor — then re-run gate 3 with the same
harness.

## Deploy configuration (for the later gateway-switch plan; not deployed)
`DS4_QWEN35_MTP_DRAFT_VOCAB=<path>/ornith-draft-vocab-vi-en-code-64k.txt ds4-server --metal -m <23G|25G GGUF>
-c 262144 --mtp --prefill-chunk 2048 --kv-disk-dir <dir>` (L1, L8 on M5 and L12 are defaults; K/V F16).

## Open items
- The MTP hidden-state carry after a disk-KV restore is still not pinned by a test (M3 parked item): a wrong carry
  costs at most one draft per restore, never output bytes; a logit-level pin needs an MTP test API.
- Ornith trims trailing whitespace like its template, which can cost a cache hit on think-off turns ending in
  whitespace (M3 parked item): not measured in M4.
- The KV-mode kernel test bounds fp8/q4 against the F16 path only (the host-dequant exactness check of the fix list
  was not implemented); fp8/q4 stay opt-in and rejected.
- Final code a9e4d8e (after the review fixes): `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` OK with
  and without MTP (f16/fp8/q4 round trips and refusals); `tests/ornith/test_server_kv.py` OK 0 WARN (an `--mtp` flip
  skips the other variant's checkpoint, stores its own, and the follow-up turn reuses it: 3702 cached);
  `tests/ornith/test_server_live.py` OK 0 WARN incl. the new `/v1/responses` tool round trip (594 cached).
- Qwen3.8 regression: full gate PASS on 46faf94 (replies byte-identical, paired decode 99.4% of PROD, wired 46.11 GiB,
  needle HIT; `QWEN_GATE.md`).
