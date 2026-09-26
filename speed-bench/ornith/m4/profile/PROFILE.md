# Ornith M4 stage profile (`DS4_QWEN35_PROFILE`)

Build: snapshot of `646d159` (Task 4 Q5_K tiles + Task 5 profiler; no decode levers). CLI one-shots
(`./ds4 --metal --raw -c 262144 --prefill-chunk 2048`) on `speed-bench/promessi_sposi.txt` prefixes, live stack
paused, 2026-09-27 01:32-02:09. The profiler syncs after every layer, so absolute ms include sync overhead; the
split between stages is what matters. Raw lines: `prefill-*.txt`, `decode-*.txt`, `mtp-verify-128k.txt`.

## Prefill (level 1, per 2048-token chunk)
| context | prefill t/s | gdn ms/chunk (30 layers) | attn ms/chunk (10 layers) |
|---|---|---|---|
| 2K (7,000 chars) | 1723 | 726 | 424 |
| 32K (112,000 chars) | 652 | ~700 | 410 at pos 0 -> 4305 at pos 30,720 |
| 128K (450,000 chars) | 186 | 700-790 | 410 at pos 0 -> 19,700 at pos 122,880 |

Attention prefill cost grows linearly with position (quadratic overall) and dominates long prompts: at 32K the
last chunk spends 86% of its time in the 10 attention layers, at 128K 96%. The Q5_K tiles (Task 4) fixed the
MoE side (2K prefill 543 -> 1723 t/s, above oMLX's 1684); long-context prefill now needs a faster prefill
attention kernel (flash-style, K/V tiles reused across query rows), which no M4 task covers.

## Decode (level 2, averaged over 64 steps)
| context | gdn ms/step | attn ms/step | head ms/step |
|---|---|---|---|
| 2K | 14.8 | 5.3 | 2.0 |
| 128K | 15.2 | 32.2 | 2.1 |

At 128K the 10 attention layers read ~2.7 GB of F16 K/V per token (2 KV heads x 256 dims x 2 x 2 bytes x
~131K positions x 10 layers); 32 ms is ~3x the bandwidth floor.

## MTP verify at 128K (`--mtp`, level 1, T=2 rows)
Each 2-row verify spends ~63 ms in attention (gdn ~24 ms, head ~4 ms), twice the T=1 decode attention: the
exact per-row verify reads the whole K/V once per row. With `--mtp` the 128K generation rate is 17.5 t/s versus
20.2 t/s plain. Review Focus 5 trigger: FIRED -> L12 (shared-K/V two-row verify) is in scope (Task 11).
