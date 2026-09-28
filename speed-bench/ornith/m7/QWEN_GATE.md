# Ornith M7: Qwen3.8 gate

M7 touches shared files only additively (`ds4_metal.m`/`ds4_gpu.h`: a new entry point and counter;
`metal/qwen35.metal`: a new kernel; `Makefile`, `.gitignore`); `ds4.c` and the Qwen paths are unchanged.

| check | where | result |
|---|---|---|
| `make test-qwen4-kernels test-qwen4-q2` | window `m7t2`, snapshot `0e652aa` | pass |
| `run.sh fast` | windows `m7t2` (`0e652aa`), `m7t5` (`a0577f7`) | PASS, PASS |
| `run.sh full` | window `m7t6` (2026-09-28 09:42-09:57), snapshot `a820559` | **PASS**: replies byte-identical to PROD; interleaved A/B PROD 38.18 / 37.73 vs branch 37.86 / 37.74 t/s (~99.6%); steady wired 46.13 GiB (develop 46.13); needle hit at 214,672 tokens |

Raw: `tests/m7t6-qwen-full.txt`, `tests/m7t6-qwen-full-result.json`, `tests/m7t5-qwen-fast.txt`.
