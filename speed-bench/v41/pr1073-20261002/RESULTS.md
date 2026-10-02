# V4.1 PR #1073 under SSD streaming on M5 Pro 64 GB, 2026-10-02

Plan: `docs/superpowers/plans/2026-10-02-v41-pr1073-measure.md`.
Question: does antirez/ds4 PR #1073 ("Speed up DeepSeek V4.1 Flash on Metal") make V4.1 Q2 decode faster on this box, the way it does on an M3 Ultra with the model resident?

**Decision: no.** The PR is 13% slower than our `develop` at the shipped auto cache and 10% slower at a fixed 24 GB cache, with byte-identical output. It still beats plain upstream by 13% (P/M 1.127), but by the plan's rule that is too little to justify cherry-picking commits (threshold 1.15). Nothing from #1073 is merged.

## Setup

| Item | Value |
| --- | --- |
| Machine | M5 Pro, 64 GB, macOS 26, gateway paused, no other ds4 process |
| Model | `DeepSeek-V4.1-Flash-Q2.gguf`, 365,713,686,528 bytes, SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42` (verified by `download_model.sh`) |
| D (ours) | `ec931240` = `develop` `360b0e26` + antirez #1123/#1125/#1135; tree identical to `develop` `7df9bef9` |
| P (PR #1073) | `3d3c83bb` (kernelpool `ds41-perf`, base `0aaea5a`) |
| M (upstream) | `antirez/main` `0aaea5a` |
| Workload | `switch` prompt, ctx 8192, 512 tokens, `--teacher-forced-decode`, `--ssd-streaming` |
| Driver | `speed-bench/v41/ab.py --plain` (no diagnostics env on either side), order A, B, B, A, each run in its own tree (`5f93c532`) |

Commands, run from the `dugong` worktree (one per row below):

```sh
R=~/orca/workspaces/ds4-metal                     # arms: $R/v41-arm-{d,p,m}, each built in its own tree
M=~/orca/workspaces/ds4-metal-data/gguf/DeepSeek-V4.1-Flash-Q2.gguf
PROMPTS=~/orca/workspaces/ds4-metal-data/v41-phase0/20260924/prompts
O=~/orca/workspaces/ds4-metal-data/v41-pr1073/20261002/run2
AB="python3 speed-bench/v41/ab.py --model $M --prompts $PROMPTS --out $O --plain --ctx 8192 --gen 512"
$AB --label pd-auto --a-bin $R/v41-arm-d --b-bin $R/v41-arm-p --a-cache auto
$AB --label pm-auto --a-bin $R/v41-arm-m --b-bin $R/v41-arm-p --a-cache auto
$AB --label pd-24   --a-bin $R/v41-arm-d --b-bin $R/v41-arm-p --a-cache 24
```

## Decode

| A/B | A t/s | B t/s | B/A | Runs A | Runs B |
| --- | ---: | ---: | ---: | --- | --- |
| D vs P, auto cache | 11.72 | 10.15 | **0.866** | 11.95, 11.49 | 10.32, 9.98 |
| M vs P, auto cache | 9.19 | 10.36 | **1.127** | 9.17, 9.21 | 10.35, 10.36 |
| D vs P, 24 GB cache | 10.82 | 9.71 | **0.897** | 10.84, 10.80 | 9.74, 9.67 |

- **Same cache plan for every arm at auto.** All three builds planned 35.62 GiB = 7.12 GiB prefill headroom + 28.50 GiB dynamic cache (3075 experts). Steady wired memory was 41.3-41.4 GiB for every arm. The ratio is therefore runtime speed, not cache sizing.
- **No contaminated run.** Swap growth was 0 in all 12 runs.
- **The repeat rule did not fire.** The D vs P auto ratio, 0.866, is outside 1.00-1.10.
- **Why P loses (read from the source, not profiled):** #1073's speedups target resident decode on big machines. Under `--ssd-streaming` it turns off DSpark (`ds41_draft_init` returns false when `g->streaming`) and the concurrent shared/routed FFN (MXFP4 only, and off when streaming). Our streaming pipeline (queued layers under streaming, async miss loads, lookahead prefetch) is what D has and P lacks.

## Identity

`./ds4 -m Q2 --ssd-streaming -c 8192 --temp 0 -n 128 --nothink`, run from each arm's own tree, on a fixed prompt gave 463 bytes from every arm, **byte-identical across M, P and D**.

## Side finding: our prefill at the auto cache

| Arm | Prefill t/s, auto cache | Prefill t/s, 24 GB |
| --- | ---: | ---: |
| D (ours) | 114.2, 113.7 | 307.2, 298.6 |
| P | 263.5, 264.3, 268.0, 262.7 | 311.4, 310.3 |
| M | 260.6, 259.5 | (not run) |

- **With the identical cache plan, our runtime prefills the 8K prompt 2.3× slower than upstream:** D mean 113.9 t/s against P 264.6 (4 runs) and M 260.1 (2 runs). That is about 72 s instead of 31 s to first token at 8K.
- **At 24 GB the two are equal.**
- **The slowdown is not new.** Every auto-cache run since 2026-09-24 shows 98-145 t/s (raw rows in `~/orca/workspaces/ds4-metal-data/v41-pipeline/` and `v41-lookahead/`). Nobody had compared it with upstream before.
- **Not investigated here.** Next step if V4.1 work continues: bisect the auto-cache prefill gap between `0aaea5a` and `develop`.

## Open item

- **P is 12.7 % faster than plain upstream under streaming, and the source of that gain was not profiled.** Part of it may stack on top of D, if some of #1073's kernels are on the streaming path and not covered by our pipeline. It was not pursued because it is below the plan's 1.15 threshold for a commit-level bisect, and V4.1 stays frozen.

## A first attempt that was discarded

Upstream builds (M, P) read `metal/*.metal` relative to the current directory; ours resolve the shaders next to the executable. The first window ran every arm from our tree, so M and P compiled our shaders, failed (`undeclared identifier 'ds4_metal_kvalues_iq4nl'`) and exited. Commit `5f93c532` makes `phase0.run_one` run each binary from its own tree; all the numbers above come from the second attempt (`run2/`).

Raw output: `~/orca/workspaces/ds4-metal-data/v41-pr1073/20261002/run2/`.
