# Ornith M4 gate 2 (quality)

- ds4 build: snapshot of `bbf5966` (final M4 code) with `DS4_QWEN35_MTP_DRAFT_VOCAB` set, `ds4-server --metal -c 262144
  --mtp` on staging port 18296, F16 K/V, live stack paused, 2026-09-27 05:20-06:05.
- Index harness: a scratch copy of AI-Gateway-MLX `reports/local-model-flash-coder-2026-09-10/run_eval.py` pinned to
  the staging port with the original think split (HumanEval/GSM8K thinking on; code_bench/review thinking off),
  truncated/errored counted over every suite including review.
- Same harness against the **live oMLX** (Shiftedx Ornith abliterated MLX, :18085, read-only), 06:08-06:10, as the
  apples-to-apples reference: the 86.6 bar comes from an older gateway run on another oMLX version.

| runtime | HumanEval-mini | GSM8K-mini | code_bench (fix/gentest/refactor) | review F1 | INDEX | truncated / errored |
|---|---|---|---|---|---|---|
| ds4 23G | 1.000 | 1.000 | 0.667 (0.667 / 0.333 / 1.000) | 0.455 | **78.0** | 0 / 0 |
| ds4 25G | 1.000 | 1.000 | 0.667 (1.000 / 0.000 / 1.000) | 0.720 | **84.7** | 0 / 0 |
| live oMLX | 0.882 | 1.000 | 0.555 (0.333 / 0.333 / 1.000) | 0.714 | **78.8** | 0 / 0 |

- ds4 23G's review score is dragged down by one malformed JSON answer (auth_db.py: a missing closing quote after
  `'or True'` makes the whole list unparseable; the findings it lists are the planted bugs). gentest failures are
  "over-constrained" tests (the model asserts behaviour the reference does not have) on every runtime.
- **Agentic matrix** (scratch gateway :18131, no scratch registry): 6/7 — easy, medium, repo, bugfix, refactor,
  testgen PASS; **hard FAIL** on execution only: its third turn (audit) hit the matrix's 480 s turn timeout
  (public and hidden tests pass, quality_pass true). That is decode speed at long agentic context, not quality.
- **Abliteration probes**: ds4 4/4 COMPLY = the live oMLX baseline 4/4 COMPLY (`probes.json`; prompt text not
  reproduced).

Verdict against the plan's rule (index >= 86.6 AND truncated = errored = 0 on both tiers, matrix 7/7, probes
match): **FAIL** (index below 86.6 on both tiers; matrix 6/7). Against the live oMLX on the same harness: **parity**
(ds4 23G 78.0 and 25G 84.7 vs oMLX 78.8; ds4 higher on HumanEval and code_bench, lower on review). The eval is small
(tens of cases), so single answers move the index by several points. Accepting M4's quality on parity is the user's
call.
