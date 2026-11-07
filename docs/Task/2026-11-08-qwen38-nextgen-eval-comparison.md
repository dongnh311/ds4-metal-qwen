# nextgen-eval 3-way comparison: candidate-v1 vs baseline UD-IQ2_XS vs OrcaUncensored

Date: 2026-11-08. Harness: `speed-bench/nextgen-eval/run.py`, suites `code,tools,reason`,
identical frozen data (sha256 verified in each run's provenance) and identical server args
(ctx 262144, prefill-chunk 4096, MTP, KV disk cache, dir-steering, vision).

Run dirs: `/tmp/eval-candidate-v1`, `/tmp/eval-baseline-ud`, `/tmp/eval-orca`.

## Arms

| arm | model | size |
|---|---|---:|
| candidate-v1 | `UD-IQ2_XS-mtp-gateup-iq2xxs.gguf` (+PLE sidecar) | 43 GB |
| baseline-ud | `UD-IQ2_XS-mtp.gguf` (prod cũ) | 47 GB |
| orca-uncensored | `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf` | 43.7 GB |

Note: orca is a different finetune (uncensored), so its scores are a relative reference,
not an apples-to-apples quality comparison with the two UD-IQ2_XS arms.

## Results

| suite | n | baseline | candidate-v1 | orca |
|---|---:|---:|---:|---:|
| code (HumanEval-mini + MBPP+, executed tests) | 67 | **65** | 64 | **65** |
| tools_pos (tool-call) | 8 | 7 | **8** | **8** |
| tools_neg (negative tool cases) | 30 | 19 | **21** | 16 |
| tools_xfer | 6 | 3 | **4** | 1 |
| faithfulness (code-graded) | 9 | 9 | 9 | 9 |
| reason (GPQA 8 + SuperGPQA 8 + AIME 8 + MMLU-Pro 20) | 44 | **44** | 42 | 41 |
| **tools total** | 53 | 29 | **33** | 25 |
| decode t/s median | | 35.78 | **37.11** | **42.34** |
| prefill t/s median | | 180.28 | **183.74** | 170.57 |
| code suite wall time (s) | | 1071.4 | 1484.8 | 1332.3 |

Errors: none in any arm.

## Reading

- candidate-v1 vs baseline (same base weights, only gate/up expert requant):
  code 64 vs 65 and reason 42 vs 44 are within noise of a 67/44-question suite;
  tools 33 vs 29 favours candidate; decode +1.3 t/s, prefill +3.5 t/s median.
  Combined with the fixed-corpus perplexity result (38.57 -> 21.28), candidate-v1
  is equal-or-better on quality and faster.
- orca decodes fastest (42.3 t/s median) but is weakest on tools_xfer (1/6) and
  tools_neg (16/30) — consistent with an uncensored finetune tuned away from
  refusal-adjacent behaviour the negative suite probes.
- Candidate's longer code wall time despite similar pass count reflects
  think_tokens_median 266 vs baseline 187 — candidate thinks longer per problem.

## Infra fix

`ds4eval.py`: `--mtp-timing` added to SERVER_ONLY_NO_VALUE, `--vision` added to
SERVER_ONLY_WITH_VALUE so production server args pass preflight classification.
Configs `baseline-ud.json`, `candidate-v1.json`, `orca.json` added under
`configs/` for reproducibility.
