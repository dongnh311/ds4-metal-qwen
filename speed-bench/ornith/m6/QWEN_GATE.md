# Ornith M6: Qwen3.8 full gate

`speed-bench/qwen-regression/run.sh full`, live stack paused. Raw: `qwen-full/`.

| run | build | replies vs PROD | needle (214,672 tokens) | decode A/B (PROD / branch, t/s) | steady wired | verdict |
|---|---|---|---|---|---|---|
| first (window `m6t5`, 22:50) | `80db522` | identical | hit | 38.4, 35.8 / 38.8, 37.6 | 46.48 GiB | FAIL (wired > 46.30) |
| control (window `m6h2h`, 23:18) | develop `696d328` | identical | hit | 38.7, 37.2 / 38.8, 38.1 | 46.13 GiB | PASS |
| rerun (window `m6h2h`, 23:37) | `0266bc6` (branch head incl. both final-review fixes) | identical | hit | 38.2, 37.4 / 37.4, 37.4 | 46.13 GiB | **PASS** |

The first run's wired reading came right after the agentic matrix and the oMLX / Ornith servers of the same window;
develop and the branch, measured back to back in a fresh window, read the same 46.13 GiB, so the M6 change does not
move Qwen3.8's memory (nothing in it runs on the Qwen3.8 path; the accelerator scratch exists only after an Ornith
accelerator call). Model-free Qwen kernel suites passed in every run.

Verdict: `qwen_gate: PASS` on the branch head.
