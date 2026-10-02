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
