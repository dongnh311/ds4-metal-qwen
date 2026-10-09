# Rewind point: re-send and compaction TTFT on Ornith (2026-10-09)

Setup:
- MacBook Pro, Apple M5 Pro, 64 GB.
- Model: Ornith-1.5-35B-A3B (`…MTPv2-23G-ICE.gguf`), run with the production 512K registry row's exact
  `process_command` (`-c 524288 --mtp`, YaRN 2, `--kv-cache-continued-interval-tokens 0`) on a scratch
  port and a scratch KV dir.
- Script: `bench_resend.py`. Thinking is off and `preserve_thinking` is on, as the gateway sends it.
  Temperature is 0.
- The gateway's ds4 slot was stopped, and the watchdog paused, for each run.

## Shape

The session grows the way production sessions do:
1. A ~16K cold first prompt. Its cold checkpoint is the only disk state, as in production, where
   continued checkpoints are off on the 512K slots.
2. Live continuations that each append the model's own reply plus ~6K tokens, up to ~61K.

Then four requests:
- `r1` is the next turn.
- `resend` and `resend2` send r1's prompt again with the reply dropped (the 16:13 production shape: a
  202 s stall).
- `compact` is r1's prompt plus a user "summarize" message, with the reply dropped (the 16:46 Claude
  Code compaction shape).

## Result (run 2, final binary `ba3f4acb`; rewind arm first, baseline second)

| request | baseline (PROD `8acc108`) | rewind point (`--rewind-point-min-tokens 16384`) |
|---|---|---|
| grow turns 21K→63K (TTFT) | 4.2 / 4.5 / 4.8 / 4.5 / 5.5 / 5.9 / 6.2 / 6.7 s | 4.4 / 4.4 / 4.6 / 4.2 / 5.2 / 5.9 / 6.3 / 6.6 s |
| r1 (normal turn) | 7.0 s | 7.0 s |
| **resend** | **48.7 s** | **0.5 s** |
| **resend2** | **48.5 s** | **0.5 s** |
| **compact** | **52.0 s** | **0.3 s** |
| reply identical r1 / resend / resend2 | yes | yes |
| live misses / rewind point hits | 3 / 0 | 0 / 3 |

- On normal turns the two arms are equal within noise (±0.3 s).
- The baseline falls back to the 16K cold checkpoint and re-prefills ~45K tokens on every re-send.
- The rewind arm restores the point (`pos=61286`) and evaluates 7 tokens.
- At production depth (120-230K) the baseline re-prefill would grow to 2-6 minutes, as the
  incidents showed. The rewind cost stays well under 1 s.

## Run 1 (before `ba3f4acb`): why the tail goes through decode evals

In run 1, the generation prompt after the cut (7 tokens) was its own prefill pass. On normal turns the
rewind arm was then 0.5-1.1 s slower, and the gap grew with depth. The run-1 rewind arm ran second,
and part of the gap was GPU heat: the first two chunks, before any new code ran, were ~10% slower too.
A timing harness, with production ctx 524288 and no gateway, measured the pass itself:

| depth | mark | 7-token prefill pass | 7 decode evals |
|---|---|---|---|
| 30K | 2-6 ms | 215-226 ms | 135 ms |
| 60K | 2-10 ms | 423-442 ms | 161 ms |
| 120K | 2-6 ms | 779-786 ms | 188 ms |

- The small prefill pass costs ~6.5 ms per 1K tokens of context, with or without the mark.
- Decode evals barely grow with depth.
- So `ba3f4acb` evaluates a tail of at most 16 tokens with `ds4_session_eval`, the path that already
  serves server-forced tokens, MTP row included.
- In run 2 the normal turns match the baseline. The mark itself costs 2-10 ms.
- Run 1's raw outputs are in `run1/`.

## Engine equivalence (Task 5)

`tests/test_qwen35_rewind_point` used the same GGUF:
1. Sync 6000 tokens, then mark.
2. Decode 300 tokens.
3. Rewind, then evaluate 64 rows against a session that was never rewound.

Results:
- max |dlogit| was **0** on both the first and the second rewind, with and without MTP.
- A rewind below the point drops it, and so does a payload load.

## Commands

```sh
python3 ~/Documents/GitHub/AI-Gateway-MLX/scripts/quiet-window.py --wait-s 7200   # then pause watchdog, stop slot
python3 bench_resend.py --bin ~/orca/workspaces/ds4-metal/rewind-point --label rewind --tokens 64000 --rewind 16384
python3 bench_resend.py --bin ~/.local/share/ai-gateway/ds4-metal --label base --tokens 64000
```
