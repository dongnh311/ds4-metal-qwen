# Ornith M7 gate 3: ds4 vs the live oMLX (same day)

Window `m7t6` (2026-09-28 09:09-09:29), snapshot `a820559` (batched verify on by default), 23G GGUF, `--mtp` with the
64K draft vocabulary, `m4_ab.py --mode baseline` A-B-B-A (oMLX, ds4, ds4, oMLX), 256 tokens, fresh nonce per prompt.
Raw: `final.txt`, `final.json`.

| context | ds4 decode t/s | oMLX decode t/s | ds4 prefill t/s | oMLX prefill t/s | ds4 TTFT s | oMLX TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| 2K | **93.0** (94.0 / 92.1) | 80.7 (81.0 / 80.4) | 1842 | 1713 | 1.2 | 1.2 |
| 32K | **68.3** (68.2 / 68.4) | 64.8 (65.0 / 64.5) | 1668 | 1660 | 19.6 | 20.2 |
| 128K | **50.4** (50.2 / 50.6) | 39.9 (40.0 / 39.8) | 903 | 750 | 143.9 | 177.6 |
| cold ~31K | **68.5** (69.5 / 67.5) | 65.2 (65.2 / 65.2) | 1567 | 1527 | 19.8 | 20.8 |

Spec §1:
1. ds4 `--mtp` decode >= oMLX at 2K (93.0 vs 80.7, +15%) and at 32K (68.3 vs 64.8, +5%): **met**.
2. 128K decode no more than 3% below M6's 40.1: 50.4 (+26%): **met**.

M6's gate 3 for comparison (`../m6/speed/GATE3.md`): ds4 decode 69.5 / 53.1 / 40.1, oMLX 79.9 / 64.5 / 38.9.
