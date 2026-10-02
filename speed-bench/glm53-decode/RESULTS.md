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
