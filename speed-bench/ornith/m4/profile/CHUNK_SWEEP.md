# Ornith M4 prefill chunk sweep

Build: snapshot of `646d159` (Q5_K tiles). `DS4_QWEN35_PREFILL_CHUNK=<chunk> ./ds4 --metal --raw -c 262144 -n 8`
on a ~32K-token prompt (112,000 chars of `speed-bench/promessi_sposi.txt`); raw lines in `chunk-sweep.txt`.

| chunk | prefill t/s at ~32K |
|---|---|
| 2048 | 620 |
| 4096 | 630 |
| 8192 | 633 |

The chunk barely matters (+2% at 8192): at 32K the prefill time is dominated by attention (PROFILE.md), whose cost
does not depend on the chunk. The 128K runs were stopped (SIGTERM) after the 32K result: at 128K attention is
96% of the prefill time, so the chunk cannot move it, and each run costs ~12 minutes of GPU time.

Decision: keep the default chunk 2048 (smallest scratch, admission unchanged at 262,144 context); larger chunks
buy ~2% prefill for 2-4x the prefill scratch. Deploy default: `--prefill-chunk 2048` (unchanged).
