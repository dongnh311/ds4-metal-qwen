# V4.1 Engram tables at 4 bits, converted in place — results, 2026-10-02

Spec: `docs/superpowers/specs/2026-10-02-v41-engram-q4-design.md`.
Plan: `docs/superpowers/plans/2026-10-02-v41-engram-q4.md`.
Code: branch `feature/v41-engram-q4`. Tool: `gguf-tools/deepseek41_engram_q4`.

**Outcome:**
- **Size:** the V4.1 Flash Q2 GGUF shrank from 365,713,686,528 B (340.60 GiB) to **267,406,761,552 B (249.04 GiB)**.
- **Encoding:** the Engram rows are now `lloyd4_e5m3_32_r136`.
- **Quality:** on the general set the converted file scores **byte-identically** to the `DS4_ENGRAM_SIM=v7` simulation that was
  scored against the FP8 original, so general NLL is −0.05 % against the original.
- **Speed:** see the speed section below; the warm-cache recheck is still pending.

## Files

| | Bytes | SHA-256 |
| --- | ---: | --- |
| Original (upstream `antirez/deepseek-v4.1-flash-gguf`) | 365,713,686,528 | `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` |
| Converted `DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf` | 267,406,761,552 | `311f35981bf14ef8e49968ef942e13263bf9011ffdc28c7b6d00a1de45d2f719` |
| Converted tail (table 1, zero gap, table 2) | 104,451,120,720 | `b3dbaf6d43f7e88cbec503e5d2c82980e1dbf0f1a85cb4e8f5259859edfa545c` |

**Layout after conversion:**
- Table 1 stays at abs 162,955,640,832.
- Table 2 moves from 264,333,279,232 to 215,180,492,800.
- Both tables are now [136, rows] I8.
- The original's 3,248 B of trailing alignment padding is dropped.

## Conversion (DISK slot 23:20:52–23:30:52, `disk.log`)

The gateway kept serving; nobody was measuring.

| Step | Time | Result |
| --- | ---: | --- |
| `--check` (encode all 768,022,850 rows) | 213 s | ok; tail SHA recorded in `eq4-check.txt` |
| `--sample` (20,000 rows per table) | 4 s | ok |
| `--convert` | 249 s | 267,406,761,552 B |
| `--verify` | 131 s | the tail SHA matches `--check` |
| `--verify-sample` | 3 s | 20,000 rows per table bit-identical through `ds4_engram_read` |
| `tests/test_deepseek41_gguf FILE` (loader + config validation) | <1 s | `V4.1 complete model layout: PASS` |

## Quality (GPU window 8, `window8.log`, `compare-v7-vs-eq4.txt`)

`score_official`, general set (100 cases, 2994 tokens, ctx 4096, `--ssd-streaming --ssd-streaming-cache-experts 24GB`):

| | NLL change | First-token matches | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| Converted file vs the v7 sim TSV | **0.000 %** | 81 / 81 | 0 / 0 / 100 |
| v7 sim vs the FP8 original (spike, 2026-10-02) | −0.045 % | 78 → 81 | — |

The TSV is byte-identical to `score-v7-general.tsv`.

The long-set floor still applies: v7 +0.94 %, 6-bit control +1.04 %, 5-bit +1.33 %. Any perturbation moves the 9-case long set
by about +1 %, so 4-bit is no worse than 5/6-bit there.

## Speed (windows 8-10)

`ds4-bench` switch, ctx 8192, auto cache (28.5 GiB, 8 slabs), gen 64, run as phase0 runs. No foreign ds4 process was present
in any run.

| Run | Prefill t/s | Steady decode t/s | Idle wired GiB |
| --- | ---: | ---: | ---: |
| Original file, earlier the same day: slab-fix A/B fix1 / fix2 | 252.9 / 256.9 | 9.32 / 9.35 | 3.00 / 2.91 |
| Original file, earlier: slab4096 / qgate0 / upstream (gen 16) | 239.2 / 257.4 / 246.7 | — | — |
| Converted, window 8 (23:3x): runs 1, 2 | 188.2, 214.1 | 9.09, 9.18 | 3.92, 3.94 |
| Converted, window 9 (01:1x), back to back: runs 3-6 | 217.6, 230.0, 229.5, 195.7 | 8.35, 9.04, 9.19, 8.79 | 3.73-4.25 |
| Converted, window 10, prefill profile (gen 16) | 218.7 | — | — |

**Steady decode** is within 2.6 % in 5 of the 6 runs; run 3 is 10 % low. **Prefill** is 15 % below the earlier runs on average
(212.5 vs 250.6) and never reaches the plan's 242 gate.

### What the prefill gap is not

- **Not the Engram format.** `DS4_METAL_GRAPH_PREFILL_PROFILE` on the converted file attributes 36.5 s of prefill:
  - map 1.25 s;
  - Engram 0.25 s, a single 247 ms wait at layer 1 for the second chunk;
  - encode 0.39 s;
  - drain 31.34 s;
  - seed 3.29 s.

  So Engram is at most 0.7 % of prefill time. A CPU microbenchmark (4M-row temp files, 16-reader `ds4_engram_read_batch`, one
  4096-token chunk of 98,304 rows) reads 136 B rows as fast as 264 B rows: 538-546 ms vs 541-568 ms.

  Every row read costs one SSD latency, about 45-63 us, even from a just-written file. `F_NOCACHE` keeps Engram rows out of the
  page cache in both encodings, so the early "cold Engram cache" hypothesis was wrong.
- **Not the SSD.** Random 9.49 MiB `F_NOCACHE` reads over the main-weight region run at 14.1 GB/s, steady over three reps
  (01:30).
- **Not the cache plan.** The cache plan, slab layout (8), expert-cache hits and misses (11562 / 6768 vs 11576 / 6754) and decode
  miss pread time are the same as before.

### What remains

The gap sits in drain (GPU and expert streaming), which the conversion does not change. The runs were not A/B'd under identical
conditions: the original file is gone, and re-downloading it (341 GiB) does not fit next to the converted file (205 GiB free).

The environment differed. Idle wired memory is about 1 GiB higher tonight, and the other sessions loaded and unloaded their
models between runs. **Status: unexplained, not attributable to the format; the gate is not met as measured.**
