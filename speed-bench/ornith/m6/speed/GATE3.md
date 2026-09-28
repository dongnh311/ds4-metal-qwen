# Ornith M6 gate 3 (speed): final interleaved A/B against the live oMLX

- ds4 build: snapshot of `80db522` (code identical to the branch head) with `DS4_QWEN35_ATTN_NAX=1` set explicitly
  (the knob stays off by default, see `../LEVERS.md`) and the 64K MTP draft vocabulary; `ds4-server --metal
  -c 262144 --mtp`, F16 K/V, prefill chunk 2048, 23G GGUF.
- oMLX: the live registry command (oMLX 0.6.4), started and stopped by the harness.
- Command: `python3 speed-bench/ornith/m4_ab.py --mode baseline --ds4-model "$DS4_ORNITH_MODEL" --ds4-env
  "DS4_QWEN35_MTP_DRAFT_VOCAB=<list>,DS4_QWEN35_ATTN_NAX=1" --contexts 2048,32768,131072 --cold-tokens 31000
  --max-tokens 256 --warmup 1` (A-B-B-A omlx/ds4/ds4/omlx, fresh nonce per prompt, `cached_tokens == 0`, one model
  process at a time, live stack paused), 2026-09-27 21:38-21:58. Raw: `final.json`, `final.txt`.

| context | ds4 decode t/s | oMLX decode t/s | ds4 prefill t/s | oMLX prefill t/s | ds4 TTFT s | oMLX TTFT s |
|---|---|---|---|---|---|---|
| 2K | 69.5 | 79.9 | **1845** | 1745 | 1.17 | 1.18 |
| 32K | 53.1 | 64.5 | 1667 | 1701 | **19.65** | 19.67 |
| 128K | **40.1** | 38.9 | **888** | 727 | **146.2** | 183.6 |
| cold ~31K | 51.5 | 64.7 | **1497** | 1476 | **20.7** | 21.6 |

The two runtimes tokenize the harness prompts differently: oMLX counts 2.1% more prompt tokens at 32K (33,455 vs
32,766), 2.5% more at 128K and 2.1% more cold (31,710 vs 31,047); the prefill t/s columns are prompt tokens / TTFT
of each runtime's own count.

Verdict against spec §1:
1. Prefill vs the live oMLX: **met at 128K** (+22% t/s, TTFT 146 vs 184 s) and **cold ~31K** (TTFT 20.7 vs 21.6 s);
   **32K at parity but not above**: the same prompt takes the same time (TTFT 19.65 vs 19.67 s), while oMLX's t/s is
   2% higher because it counts 2% more tokens. The harness's t/s rule records this as FAIL.
2. Decode vs M5 (within 3%): 2K +0.6%, 32K -1.3%, 128K +2.0%, cold -3.6% (its two ds4 runs read 53.3 and 49.7; the
   Task 4 lever A/B measured no decode change from the knob: 49.2 -> 48.8, 55.0 -> 54.2, 37.5 -> 39.1).

Against M5's gate 3 (same harness, same day, 15:19): prefill 1738 -> 1845 (2K), 983 -> 1667 (32K), 354 -> 888
(128K), cold TTFT 32.4 -> 20.7 s. The live oMLX read within 1% of its M5 decode numbers and within 7% of its M5
prefill numbers (2K +7%, 32K +1%, 128K -2%, cold -3%).

What remains: 32K prefill per token is 2% behind; the 2K/32K decode gap (GDN, MoE and MTP overhead per step) is the
separate short-context decode milestone.
