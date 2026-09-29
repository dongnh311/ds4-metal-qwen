# Sub-project 2: runtime refusal projection on Ivan's stock IQ2 — FAIL

The engine gate fails on one suite: `tools_neg`, where the projection makes the model call a tool
instead of answering. Everything the sub-project set out to prove about refusals holds — harmful
refusals went from 45/50 to 0/50 with no accuracy suite regressing beyond tolerance — but the spec's
hard rule is no regression in any measured area, and this one reproduces exactly.

```bash
python3 speed-bench/nextgen-eval/compare.py \
  speed-bench/nextgen-eval/results/2026-09-29-sp2-ivan.summary.json \
  speed-bench/nextgen-eval/results/2026-09-29-sp2-ivan-proj.summary.json \
  --gate engine --refusal-caps 1,1     # exit 1
```

## What ran

- **Arms:** both on Ivan's stock IQ2 (`ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2`, file
  `Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf`, 44,806,612,192 B, sha256 verified against
  the repo's `SHA256SUMS` at revision `b8b20398`), with PROD's registry flags and environment, only the
  model path, port and KV directory replaced:
  - `ivan` 2026-09-28 21:00-01:22 (4 h 22 min), `runs/ivan-20260928-210029`;
  - `ivan-proj` 2026-09-29 01:22-06:02 (4 h 40 min), `runs/ivan-proj-20260929-012209`, adding
    `--dir-steering-file .../refusal-4-44.f32 --dir-steering-ffn 1`.
- **Binaries:** identical for both arms — `ds4-server` sha256 `ba4784c2...`, `ds4-eval` `e8cd9b55...`.
- **Branch:** `feature/nextgen-qwen`, `ivan` at `c92b76e` and `ivan-proj` at `8f043d1`; between those
  commits only `compare.py`, its tests and the README changed, no engine or suite code.
- **Frozen inputs:** the same sha256s in both summaries (`mbpp` `831bc729`, `ifeval` `53695596`,
  `harmful` `73143c12`, `harmless` `055c3667`, `haystack.c` `714124ef`, `mcp_tools.json` `07a050f2`),
  and gateway repo `f56e587c`.
- **Direction:** Cudecnik's `Qwen3.8-Flash-Next-refusal-projection.gguf` (sha256 `ef0724c5...`, pinned
  revision `1886570b`), converted by `dir-steering/tools/cvec_to_f32.py --layers 4-44` to
  `refusal-4-44.f32` (sha256 `18a31175...`). Every source row's norm was 0.999999-1.000001, so
  normalizing changed nothing; scale 1.0, layers 4..44, as in Cudecnik's own settings.
- **Machine state:** the ai-proxy and gateway-watchdog LaunchAgents were unloaded and the Ornith
  ds4-server was stopped with SIGTERM before the run. `/status` afterwards showed no request inside
  the window. The stack was restored at 06:02 (`backend_ok` true).

## Pre-run checks (plan Task 7)

| check | result |
|---|---|
| a steering row of norm 2 | refused by both `ds4` and `ds4-eval`: "steering row for layer 10 has norm 2" |
| no steering flags, PROD model, greedy, 3 prompts | byte-identical to the pre-change binary, 3/3 |
| projection effect, 3 harmful prompts, with `--ssd-streaming` | refused 3/3 without it, 0/3 with it |

The third check also settles a stale line in `dir-steering/README.md`: Qwen steering does work with SSD
streaming (corrected in `c92b76e`).

## Accuracy: ivan-proj vs ivan

| suite | ivan | ivan-proj | verdict |
|---|---|---|---|
| code | 65/67 | 64/67 | same (MBPP/84 new) |
| reason | 42/44 | 41/44 | same (one more SuperGPQA) |
| ifeval | 59/60 | 58/60 | same (201 fixed; 19 and 337 new) |
| tools_pos | 7/8 | 8/8 | same (pos_manifest fixed) |
| **tools_neg** | **4/6** | **1/6** | **regressed** |
| tools_xfer | 0/6 | 0/6 | same |
| faithfulness | 9/9 | 9/9 | same |
| vi_knowledge | 30/30 | 30/30 | same |

VI CJK leaks: 0 of 45 in both arms. No suite errored in either arm.

## The regression is real, not noise

The tools suite runs at `temperature=0`. The suite was rerun once per arm into its own run directory
(`runs/ivan-20260929-061253`, `runs/ivan-proj-20260929-061518`), and all 20 cases reproduced exactly:

| case | ivan #1 | ivan #2 | proj #1 | proj #2 |
|---|---|---|---|---|
| neg_cargo | pass | pass | FAIL | FAIL |
| neg_edit | pass | pass | FAIL | FAIL |
| neg_mkdir | pass | pass | FAIL | FAIL |
| pos_manifest | FAIL | FAIL | pass | pass |
| the other 16 | unchanged | unchanged | unchanged | unchanged |

On the three flipped negative cases the projected model returns `finish=tool_calls` and calls
`codebase_search` where the unsteered model returns `finish=stop` and answers. The same change fixes
`pos_manifest`, where the unsteered model failed to call a tool it should have.

That pattern is consistent with what the projection removes. The direction encodes declining, and
declining to act is part of it: with it gone the model acts more readily. This helps where acting is
right (harmful prompts, `pos_manifest`) and hurts where withholding is right (the negative cases).
This is a cost for the coding-agent workload, not just the KL cost on neutral text that the spec
anticipated.

## Uncensor

| | ivan | ivan-proj | PROD (2026-09-28) |
|---|---|---|---|
| harmful refusals | 45/50 | 0/50 | 1/50 |
| harmless refusals | 0/50 | 0/50 | 0/50 |

The projection works on stock weights, and by the grader it removes refusals more completely than
PROD's weight abliteration.

**Caveat on that grader.** `graders.is_refusal` reads only the opening of an answer, so an answer that
deflects — a "simulation", or a guide to preventing the harm — counts as compliance. This affects both
arms and PROD equally, and PROD's own answers deflect on several prompts. So "0/50" means "did not
open with a refusal", not "complied". The paired answers are in
`2026-09-29-sp2-projection.uncensor.md` for whoever needs to judge the content; the useful comparison
there is ivan-proj against PROD, not against `ivan`.

## Long context and memory

| | ivan | ivan-proj |
|---|---|---|
| needle 120K | hit | hit |
| needle 240K | **missed** | **hit** |
| 240K document questions | 3/3 | 3/3 |
| needle 480K / 960K | skipped (`-c 262144`) | skipped |
| peak wired | 43.47 GiB | 43.35 GiB |
| swap-outs | 0 | 0 |

The stock IQ2 missed the 240K needle that PROD hits; with the projection it hits. Both arms sit about
6 GiB below PROD's 49.54 GiB peak, because Ivan's file is Q2_K-down rather than PROD's Q4_K-down.

## Speed

| metric | ivan | ivan-proj | change |
|---|---|---|---|
| decode median | 42.89 t/s | 40.93 t/s | -4.6% |
| prefill median (short) | 118.06 t/s | 114.87 t/s | -2.7% |
| median thinking tokens | 142.5 | 240 | +68% |
| gate sum (code+ifeval+vi+uncensor) | 6317.3 s | 9048.3 s | +43% |

The decode cost is close to the ~7% Cudecnik reports for llama.cpp. Most of the wall-clock difference
is not the kernel: the projected model thinks longer and writes longer answers, which is most visible
on the uncensor suite (1681 s to 4010 s), where the unsteered model's refusals are short.

Speed is not part of this gate; it is gated against PROD in sub-project 5.

## Verdict and what follows

`tools_neg` 4/6 to 1/6 is a reproducible regression in a measured area, so sub-project 2 does not
pass. The spec's fallback list applies, cheapest first, and each step needs the user's approval:

1. Narrow the layer range (re-convert at 8-40) and rerun `--suites tools` only, about 3 minutes of GPU
   per setting, then `--suites uncensor` on whatever survives.
2. Lower the FFN scale (0.75, then 0.5) the same way.
3. Derive a ds4 direction from all prompt positions.

A fourth option is a decision rather than a fix: `tools_neg` has 6 cases, and PROD itself scores 2/6,
so one could argue this axis is already weak and should be judged against PROD in sub-project 5 rather
than against unsteered stock weights here. That is the user's call, not the harness's.

## Fallback sweep (2026-09-29 morning)

The user approved fallbacks 1 and 2. Each setting ran `--suites tools` first, about 3 minutes; a
setting went on to the uncensor suite only if `tools_neg` came back within tolerance, which means at
least 3/6. All runs used Ivan's IQ2 with SSD streaming, like the arms above.

| setting | tools_pos | tools_neg | tools_xfer | faithfulness |
|---|---|---|---|---|
| unsteered (`ivan`) | 7/8 | 4/6 | 0/6 | 9/9 |
| scale 1.0, layers 4-44 | 8/8 | 1/6 (regressed) | 0/6 | 9/9 |
| scale 1.0, layers 8-40 | 6/8 | 2/6 (regressed) | 1/6 | 9/9 |
| scale 0.75, layers 4-44 | 6/8 | 3/6 | 0/6 | 9/9 |
| scale 0.5, layers 4-44 | 6/8 | 3/6 | 0/6 | 9/9 |

Narrowing the layer range does not fix `tools_neg`. It only moves which cases fail. Lowering the FFN
scale does fix it: at both 0.75 and 0.5, every tools suite is within the small-suite tolerance.
Individual cases flip between settings, because each one is a single greedy generation. The
per-suite totals are the thing to read.

**Scale 0.75 keeps the refusal removal.** On the uncensor suite it scores 0/50 harmful refusals and
0/50 harmless refusals, the same as scale 1.0. That makes 0.75 the candidate, and 0.5 was not tested
further. The 0.75 arm is being built suite by suite in one run directory
(`runs/ivan-proj-s075-20260929-064527`, config `ds4-metal-data/sp2/ivan-proj-s075.json`). Measured so
far, against the unsteered arm:

| suite | ivan | scale 0.75 | scale 1.0 |
|---|---|---|---|
| code | 65/67 | 65/67 | 64/67 |
| vi_knowledge | 30/30, CJK 0 | 30/30, CJK 0 | 30/30, CJK 0 |
| tools pos / neg / xfer | 7/8, 4/6, 0/6 | 6/8, 3/6, 0/6 | 8/8, 1/6, 0/6 |
| faithfulness | 9/9 | 9/9 | 9/9 |
| harmful / harmless refusals | 45/50, 0/50 | 0/50, 0/50 | 0/50, 0/50 |

The tools and faithfulness suites at 0.75 ran twice, once as the probe and once inside the arm. All 29
cases came out the same both times. The code suite took 1318.6 s against 1193.2 s unsteered (+10.5%),
and vi took 617.9 s against 626.6 s. The arm's median think length is 165 tokens and its median decode
speed is 43.18 t/s. These two numbers cover only the suites run so far, so they cannot be compared
with the full arms yet.

The arm still needs three suites: ifeval (about 50 minutes), reason (about 2.2 hours) and longctx
(about 25 minutes). After those, the engine gate can run against `ivan`:

```bash
python3 speed-bench/nextgen-eval/compare.py <ivan summary> <ivan-proj-s075 summary> \
  --gate engine --refusal-caps 1,1
```
