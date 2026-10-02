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

## Speed (window 8, cold page cache)

`ds4-bench` switch, ctx 8192, auto cache (28.5 GiB, 8 slabs), gen 64:

| Run | Prefill t/s | Steady decode t/s | Expert miss pread |
| --- | ---: | ---: | --- |
| Slab-fix A/B 2026-10-02, original file (fix1 / fix2) | 252.9 / 256.9 | 9.32 / 9.35 | 35.08 GiB, 1.38–1.41 s |
| Converted, run 1 | 188.2 | 9.09 | 35.21 GiB, 1.42 s |
| Converted, run 2 | 214.1 | 9.18 | 35.21 GiB, 1.43 s |

**Steady decode** is within 2.6 %, which passes the 5 % gate.

**Prefill** misses the gate (≥ 242). What was ruled out:
- the cache plan (identical);
- the expert miss bytes and pread time (identical);
- the decode arithmetic (the 136 B path does fewer `ldexpf` calls per value and touches fewer pages).

Run 2 was faster than run 1.

Hypothesis: the page cache was cold.
- 136/264 B Engram rows are not page-aligned, so `F_NOCACHE` does not keep them out of the cache. Before the conversion, the rows
  this prompt touches were warm from the evening's many runs.
- The conversion's 97 GiB rewrite, plus another session's 52 GB write, evicted those pages and the resident weights.

**Pending:** a warm-cache recheck (several back-to-back runs) after the other sessions' GPU windows.
