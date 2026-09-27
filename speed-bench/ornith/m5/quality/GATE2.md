# Ornith M5 gate 2 (quality)

Rule (parent spec §8, user decision 2026-09-27): each tier's index >= the live oMLX's index on the same harness the
same day minus 3.0, truncated = errored = 0, no quality failure in the agentic matrix (turn timeouts are speed,
reported under gate 3), abliteration probes match.

Build: snapshot of `f7f3e17` (M5 defaults: decode3 + flash attention) for the eval, probes and matrix; the
final-review fixes (`4b0cbbd`) do not change numerics (gate-1 dumps byte-identical), and the testgen reruns below
ran on `4b0cbbd`. Staging `ds4-server --metal -c 262144 --mtp` with the MTP draft vocabulary, F16 K/V.

## Eval (same harness as M4: HumanEval-mini, GSM8K-mini, code_bench, review)

| runtime | HumanEval-mini | GSM8K-mini | code_bench (fix / gentest / refactor) | review F1 | INDEX | truncated / errored |
|---|---|---|---|---|---|---|
| ds4 23G | 1.000 | 1.000 | 0.778 (0.667 / 0.667 / 1.000) | 0.552 | **83.2** | 0 / 0 |
| ds4 25G | 1.000 | 1.000 | 0.667 (1.000 / 0.000 / 1.000) | 0.600 | **81.7** | 0 / 0 |
| live oMLX (15:16-15:19) | 0.882 | 1.000 | 0.555 (0.333 / 0.333 / 1.000) | 0.714 | 78.8 | 0 / 0 |

Bar: 78.8 - 3.0 = 75.8. Both tiers pass (M4: 23G 78.0, 25G 84.7; the eval's "GATE2: FAIL" line is its built-in
86.6 bar, superseded by the parity rule). Raw: `eval-*.txt`, `eval-*.json`.

## Abliteration probes

ds4 23G 4/4 COMPLY, identical to the live oMLX's labels captured the same day (`probes.json`).

## Agentic matrix (7 cases, scratch gateway -> staging ds4, temperature 0.7)

| case | result | cause |
|---|---|---|
| repo, easy, medium, refactor | PASS | — |
| hard | FAIL | audit turn hit the 480 s turn timeout (speed); public and hidden tests pass |
| bugfix | FAIL | audit turn hit the 480 s turn timeout (speed); public and hidden tests pass |
| testgen | FAIL | hidden check: generated tests must call `clamp`/`chunked`/`parse_bool` by bare name |

M4's run was 6/7 (only `hard`, on the same timeout). The two timeouts are speed (32K-class audit turns at ~54 t/s
decode). testgen was re-run to separate sampling from the kernels, alternating the kernels on one snapshot:

| kernels | testgen runs | pass |
|---|---|---|
| M5 (decode3 + flash) | m5t7 FAIL, reruns FAIL / PASS / PASS, variance window FAIL / PASS / FAIL / FAIL | 3 / 8 |
| M4 (decode2 + attn_mm) | M4's matrix PASS, variance window PASS / FAIL / PASS / FAIL | 3 / 5 |

Every failure, under either kernel pair, is the same AST requirement (the model called the functions through a
module or alias; the public tests still pass). The rates are indistinguishable at this sample size, so testgen's
failure is a sampling-dependent style check, not an M5 quality change. Raw: `matrix.txt`, `matrix/`, and the
controller's window logs.

## Verdict

**PASS at parity** (controller ruling, see the report): the index clears the live-oMLX bar on both tiers with
truncated = errored = 0, the probes match, and the matrix shows no M5-specific quality failure — two failures are
turn timeouts (speed) and testgen fails at the same rate with M4's kernels. A single matrix run does not meet the
literal "no quality failure" wording because of testgen's sampling variance; the report states this.
