# Round 2 window 1: paired A/A noise

- **Run.** Window r2w1, 2026-10-03 09:51-10:15, on `edb4144`.
- **Command.** `m4_ab.py --mode lever --paired --blocks 2 --reps 2 --contexts 2048,32768 --cold-tokens 0`,
  with the same (empty) env on both arms.
- **What was measured.** 8 ds4-server runs of base. Each block sent the same 2 prompts per context to all 4 of
  its runs.
- **Receipts.** `receipts/aa.json` and `receipts/aa.txt`.

| context | n | mean | sd | min | max | text mismatches | noise = max(\|mean\|, sd) |
|---|---|---|---|---|---|---|---|
| 2K | 4 | -0.7% | 0.6% | -1.3% | +0.1% | 0 | **0.7%** |
| 32K | 4 | -0.7% | 1.3% | -2.2% | +0.4% | 0 | **1.3%** |

**What this shows**

- **Greedy decoding is deterministic across server processes.** All 16 samples hash the same in every run of
  their block. So the paired method's premise holds, and the text check is a free identity check for any
  lossless lever.
- **The same prompt reruns within about 1%.** For example, block 0's 2K rep 0 decoded at 93.2, 94.2, 93.2 and
  94.4 t/s across its four runs.
- **Different prompts move the rate by 5-8%.** Base at 2K ranged from 87.3 to 94.2 t/s, against round 1's
  unpaired base pair of 90.3 and 95.5 in window F. That spread was text, not speed.
- **The middle runs (the "lever" arm in A-B-B-A) read 0.7% low at both contexts.** This is within the noise,
  and it is also the sign a warm-up position effect would have. Lever A/Bs should keep `--blocks 2` so it
  averages out.

**Gate, from spec §1.** A lever must gain at least 3%, and at least twice this noise: 1.4% at 2K and 2.6% at
32K. The 3% floor binds at both contexts.
