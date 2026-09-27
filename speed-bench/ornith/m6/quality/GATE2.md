# Ornith M6 gate 2 (quality)

Rule (parent spec §8, user decision 2026-09-27): each tier's index >= the live oMLX's index on the same harness the
same day minus 3.0, truncated = errored = 0, no quality failure in the agentic matrix (turn timeouts are speed,
reported under gate 3), abliteration probes match.

Build: snapshot of `80db522` (code identical to the branch head) with `DS4_QWEN35_ATTN_NAX=1` set explicitly and the
64K MTP draft vocabulary; staging `ds4-server --metal -c 262144 --mtp`, F16 K/V; window `m6t5`, 2026-09-27
21:58-22:49. The live oMLX was measured on this harness earlier the same day (M5 gate 2, 15:16-15:19): index 78.8,
truncated = errored = 0; its probe labels (`m5-probes-omlx.json`) are the reference for the probe comparison.

## Eval (HumanEval-mini, GSM8K-mini, code_bench, review)

| runtime | HumanEval-mini | GSM8K-mini | code_bench (fix / gentest / refactor) | review F1 | INDEX | truncated / errored |
|---|---|---|---|---|---|---|
| ds4 23G (M6, accelerator prefill) | 1.000 | 1.000 | 0.778 (0.667 / 0.667 / 1.000) | 0.571 | **83.7** | 0 / 0 |
| ds4 25G (M6, accelerator prefill) | 1.000 | 1.000 | 0.778 (1.000 / 0.333 / 1.000) | 0.621 | **85.0** | 0 / 0 |
| ds4 23G / 25G (M5, same day) | | | | | 83.2 / 81.7 | 0 / 0 |
| live oMLX (same day) | 0.882 | 1.000 | 0.555 (0.333 / 0.333 / 1.000) | 0.714 | 78.8 | 0 / 0 |

Bar: 78.8 - 3.0 = 75.8; both tiers pass and both sit above the live oMLX (+4.9 and +6.2). The eval's own
"GATE2: FAIL" line is its built-in 86.6 bar, superseded by the parity rule. Raw: `eval-*.txt`, `eval-*.json`.

## Abliteration probes

ds4 23G 4/4 COMPLY, identical to the live oMLX's labels (`probes.json`).

## Agentic matrix (7 cases, scratch gateway -> staging ds4, temperature 0.7)

| case | result | cause |
|---|---|---|
| repo, easy, medium, refactor, testgen | PASS | — |
| hard | FAIL | audit turn hit the 480 s turn timeout (speed); public and hidden tests pass |
| bugfix | FAIL | audit turn hit the 480 s turn timeout (speed); public and hidden tests pass |

5/7, the same two speed timeouts as M5 (whose third failure, testgen's style check, passes this time). No quality
failure. Raw: `matrix.txt`, `matrix/<case>.json`.

## Verdict

**PASS at parity**, and above the live oMLX on both tiers.
