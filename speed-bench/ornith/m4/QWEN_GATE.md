# Qwen3.8 regression gate after Ornith M4

- Branch commit: 46faf94 (all M4 code incl. the final-review fixes; the later a9e4d8e only changes the Ornith-only
  DS4_QWEN35_KV parse, which Qwen3.8 never calls, and a test script).
- Command: `speed-bench/qwen-regression/run.sh full` (2026-09-27 06:15-06:33, live stack paused).
- Result: PASS; replies byte-identical yes; paired decode 99.4% of PROD (branch 39.68/38.74 vs PROD 39.94/38.95 t/s,
  run order prod-branch-branch-prod); wired steady 46.11 GiB (peak 47.37); needle HIT at 214672 prompt tokens.
- Every M4 task that touched a shared file also passed `run.sh fast` in its GPU window (Tasks 4, 7-9, 10-11 windows).
- Log tail:

```
./tests/test_qwen4_prefill_pipe
test_qwen4_prefill_pipe: PASS
qwen_gate: PASS
```
