# GLM-5.3 decode program: measurements

Model: GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf, `--ssd-streaming --power 100 -c 262144`,
M5 Pro 64 GB, greedy, 256 tokens, prompt: the ISO-8601 duration task.

## Phase 0 (env only)

2026-10-02 08:43-08:47, tree c09a1e38 with `DS4_GLM_STREAM_GATE=0` (the pre-SP1 drain path), PROD off.

| Run | rep 1 t/s | rep 2 t/s | Output vs base |
|---|---|---|---|
| base | 9.14 | 9.19 | - |
| readahead off (`DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1`) | 8.62 | 8.51 | IDENTICAL (both reps) |
| `--mtp` (vs `mtp-off` 9.42) | 7.55 | - | IDENTICAL |

- Expert readahead pays: turning it off costs ~7%. Keep it.
- MTP under streaming is byte-identical but 20% slower (7.55 vs 9.42 t/s): the 2-token verify drains
  and the draft's experts add misses. SP4 has to make the verify cheaper, not only correct.

Selected-load time split (from timing-summary.txt, timing run 9.30 t/s, 21420 selected calls):

| Part | avg per call |
|---|---|
| sync (wait for the GPU to reach the router) | 0.631 ms |
| bind (resolve, load, tables) | 0.583 ms |
| per load call (8315): prepare / pread / install | 0.871 / 1.075 / 0.001 ms |

Expert cache: 4546 slots (29.97 GiB), hit rate 0.769, 23825 misses of 6.75 MiB.

## SP1 (decode gates)

2026-10-02 08:47-09:05, tree c09a1e38, PROD off. drain = `DS4_GLM_STREAM_GATE=0`, gate1 = one-pass gates
(`DS4_GLM_STREAM_SPLIT=0`), split = the default.

| Check | Result | Pass |
|---|---|---|
| Identity, 16 tokens | gate1 and split IDENTICAL to drain | yes |
| Identity, 256 tokens | gate1 and split IDENTICAL to drain | yes |
| Identity, `--mtp` 256 | mtp-split IDENTICAL to mtp-drain (7.74 vs 7.60 t/s) | yes |
| Speed, 256 tokens, 2 interleaved reps | drain 9.34 / 9.33, gate1 9.68 / 9.69, **split 10.05 / 10.05** (+7.7%) | **no** (gate 12.5, stop rule 10.8) |
| GPU busy | drain 52.3 of 108.0 ms (48%); split 98.6 of 98.5 ms (100%, see below) | not meaningful |
| Split default | split faster than gate1 in both reps | split stays default |
| Server, decode (short2) | drain 9.31, split 10.22 t/s | yes |
| Server, prefill (9433-token doc) | drain 133.1, split 135.6 t/s | yes |
| Server, multi-turn | prompt tokens 22 -> 330 -> 623, requires_rebuild 0 (both) | yes |
| Injected timeout (gate 3000 held 900 ms) | request fails ("metal GLM decode failed"), log `poll timed out` + `gates are now off`, next request finishes with content | yes |
| Qwen gate full | stopped mid-run: the Qwen session owns Qwen testing (user, 2026-10-02) | n/a |

**Verdict: SP1 gate not met.** Gates are byte-identical and safe but give +7.7% (10.05 t/s), below the
10.8 t/s stop rule; the program stops for a re-evaluation.

Where the time goes (`DS4_GLM_STREAM_TIMING`, split, 256 tokens):

```
ds4: GLM stream gates 10500: avg service 1380.6 us, gpu arrival wait 935.3 us, split 77.4% with misses, fallback experts 0
```

- ~41 gates per token. Per gate the GPU runs ~0.94 ms (attention, router, shared expert) before the
  service thread sees the mailbox, then the service needs ~1.38 ms (cache lookup, victim choice,
  readahead and pread of ~1.7 misses, tables, release) while the routed MoE waits.
  41 x (0.94 + 1.38) = 95 ms, matching the measured 98.5 ms per token.
- The GPU busy profile counts the poll kernel's spin as busy, so 100% only says the GPU never sat idle
  between command buffers. Real compute stays ~52 ms per token, as with the drain.
- Split pass 1 overlaps little: 77% of gates have a miss, and the cached experts' share of the routed
  MoE is short compared with a miss's load.
- The spec expected part of the pread to hide behind GPU work. It cannot: a gate's experts are known
  only at its router, and the routed MoE needs them right after the short shared expert. Hiding the
  ~57 ms per token of service time needs either a much cheaper service (prepare alone averaged
  0.87 ms per load call in Phase 0) or experts loaded before their router runs (SP2 prefetch).

## Gate-service spike (2026-10-02 09:34-09:53)

Question: can the ~1.4 ms per-gate service time be cut on the host? Throwaway code (worker-side
readahead, service timers) was measured and deleted.

| Variant (split gates, 256 tokens) | t/s | Output vs drain |
|---|---|---|
| default | 9.93 / 9.94 | IDENTICAL |
| readahead off | 9.95 / 9.87 | IDENTICAL |
| readahead moved into the pread workers | 9.78 / 9.77 | IDENTICAL |
| pread threads 4 / 18 (default 9) | 9.81 / 9.83 | IDENTICAL |

Service split per gate (instrumented build, split): pre 23 us, `load_batch` 1291 us of which
**pread wall 1210 us**, tables 0.3 us, release 84 us, post-release prune ~0 us; 1.67 misses per gate.
The gate path never calls readahead (`load_batch` preads directly), so readahead is not a lever here.

SSD micro-benchmark (scratch file written with F_NOCACHE, 300 simulated gates of 5 random 2.25 MiB
reads, persistent pool): 2.25 MiB x 9 threads 1.215 ms per gate (9.7 GB/s); best chunking
(512 KiB x 16 threads) 1.157 ms (10.2 GB/s); smaller chunks or more threads are slower. The same
reads from the page cache take 0.17 ms (65-72 GB/s).

**Answer:** the gate service is SSD-bandwidth bound. Missed experts come from the SSD at its ~10 GB/s
ceiling, ~11 MiB per gate, exactly the measured 1.2 ms. Host-side changes can win at most ~5%.
Per token: ~41 x 1.2 ms of SSD reads (~49 ms) on the critical path plus ~52 ms of GPU work. Even
perfect overlap of the two caps single-token decode near 19 t/s at this hit rate (0.77); 20 t/s
needs fewer missed bytes. SP2 prefetch can only use the SSD's idle time during GPU work (~40%) and
pays for wrong guesses with the same bandwidth (V4.1 spike: ~3 reads per saved miss), so a realistic
SP2 result is ~11.5-12.5 t/s.

## Decision (2026-10-02): GLM-5.3 shelved

GLM-5.3-Flash scores higher on the Intelligence Index than DeepSeek V4.1 Flash, but on this M5 Pro
64 GB its decode speed is capped by hardware: the Q2 file (89.9 GiB) must stream, and the missed
experts (~470 MiB per token at a ~0.77 hit rate) come from the SSD at its ~10 GB/s ceiling.
Byte-identical work reached 10.05 t/s (SP1 decode gates, kept, `DS4_GLM_STREAM_GATE=0` restores the
drain); the realistic next step (SP2 prefetch) would reach ~12 t/s and perfect overlap ~19 t/s.
No further GLM speed work on 64 GB. GLM is not deployed. What would change the answer: a machine with
enough RAM to keep the experts resident (upstream ds4#1057: 22-25 t/s streamed on 128 GB), or a
much smaller expert set (e.g. REAP-style pruning to ~50 GiB, output not byte-identical, quality unknown).
