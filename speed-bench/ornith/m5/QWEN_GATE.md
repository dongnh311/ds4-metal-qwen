# Ornith M5: Qwen3.8 full gate

`speed-bench/qwen-regression/run.sh full` on a snapshot of `f7f3e17` (M5 kernels on by default), inside the final
GPU window (2026-09-27 16:39-16:56, live stack paused). Raw: `qwen-full/result.json`, `qwen-full/qwen-full.txt`.

- Model-free Qwen kernel suites (`test-qwen4-kernels`, `test-qwen4-q2`, MoE specialization, prefill pipe): PASS.
- Replies byte-identical to PROD (vi, code): PASS.
- Needle at 214,672 prompt tokens: hit.
- Interleaved A/B (PROD, branch, branch, PROD), decode t/s: PROD 39.6 / 38.6, branch 38.6 / 39.0 (branch ~99% of
  PROD; the bar is 97%).
- Wired memory steady 46.1 GiB, peak 47.4 GiB.

Verdict: `qwen_gate: PASS`. The Qwen fast gate also passed in the adoption window (`m5t56`, snapshot `efa0b1e`) and
is re-run on the final-review fixes (`4b0cbbd`, which touch the Ornith memory estimate in `ds4.c`).
