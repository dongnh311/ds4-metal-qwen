# DS4-IQ2 Fast Campaign — M5 Pro 64 GB (Qwen3.8-Flash-Next)

Protocol: `docs/Task/2026-09-15-ds4-iq2-m5pro-fast-campaign.md` (Task 0).
Machine: Apple M5 Pro (20-core GPU, 64 GB, 307 GB/s). Model:
`Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf`
(41.73 GiB resident weights + 30 GiB PLE Q4_1 demand-paged sidecar + 95 GiB
BF16 n-gram table read from disk). Production launch environment (draft vocab
64k rows, grow-on-demand KV, stream-full-layers 32, think-budget 4096,
refusal steering ffn=0.5) mirrored into every bench run from the live gateway
registry; receipts carry `model_sha256`, `prompt_sha256`, git commit and macOS build.

Numbers: `speed-bench/orca-iq2-baseline.csv` (one row per arm/rep/frontier),
per-run receipts under `speed-bench/receipts/`.

## Harness bugs found by Task 0 (both landed before any tuning)

1. `ds4-bench --mtp` did not exist: the flag appeared in the help text but was
   never parsed, so the campaign's own commands could not run (6ff16c4e).
2. Deeper: the bench generate loop was gated `speculative = cfg.dspark && …`,
   so even with the engine option wired, every `--mtp` arm executed the plain
   single-row eval loop. The first sweep (2026-10-29 16:19) produced four MTP
   arms indistinguishable from plain and zero verify cycles; discarded, kept at
   `speed-bench/receipts/orca-iq2-20261029-161812/`. Fixed in be8f37af.

## Measured MTP acceptance (greedy, code continuation, 2026-10-29)

Sanity run at ctx 4096, 64 greedy tokens, production draft vocab:

```
Qwen3.8 mtp:        31 verify cycles, 25 drafts accepted (80.6%)
Qwen3.8 mtp depth3: 16 cycles, 7 second-draft accepts (43.8%), 1.032 extra tokens per cycle
```

First-draft acceptance is high on code; the SECOND chained draft (depth-3)
accepts well below the `~0.95 deterministic continuation` figure cited in
`qwen4_spec_depth`. Implication for Task 4: the depth-3 engagement rule
(8/8 first-draft accepts engages it) measures the wrong signal — it should
watch accept2 itself, not infer depth-3 payoff from first-draft streaks.

First PRODUCTION telemetry (the deployed 16:16 slot, printed at session-free
during the 22:04 drain): `96 verify cycles, 80 drafts accepted (83.3%)`,
depth-3 `4 cycles, 0 second-draft accepts`. At the gateway's real temp 0.6
traffic acceptance is HIGHER than the greedy bench (83.3% vs 73.2%) — the
sampling verifier uses the probability ratio, not argmax equality, and the
profile's cold start pays off in speculation, not against it. Small sample;
the counters now print on every session-free, so this number only gets more
trustworthy with use.

## Baseline sweep (220K-target protocol)

Sweep 2026-10-29 16:50–18:04 (arms mtp/d2/d3/force, receipts
`speed-bench/receipts/orca-iq2-20261029-165008/`) plus the true-`plain`
re-run 22:06–22:23 (`…/orca-iq2-20261029-220605/`, after the flag-decoupling
fix — the 16:50 "plain" arm silently ran MTP because `--mtp-timing` implied
`--mtp`). Merged table: `speed-bench/orca-iq2-baseline.csv`. One rep per arm
(variance lesson below); gen = greedy continuation of a frozen 1.2 MB ds4.c
slice, production launch env mirrored from the live registry.

| arm (gen t/s) | 32K | 64K | 128K | 192K | 225K | draft-1 acc | depth-3 accept2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain (true, no MTP) | 37.9 | 37.7 | 37.4 | 28.5 | 31.4 | — | — |
| mtp (auto depth) | 44.7 | 41.3 | 41.7 | 39.5 | 40.6 | 73.2% | 68.8% (32/280 cycles engaged) |
| mtp-d2 (forced depth 2) | 48.3 | 41.8 | 40.8 | 40.4 | 40.6 | 74.2% | — (never engages) |
| mtp-d3 (forced depth 3) | 42.4 | 32.2 | 32.8 | 31.4 | 30.9 | 69.9% | 41.6% (99/238) |
| mtp-force (accept-all ceiling) | 38.2* | 46.6 | 45.7 | 42.7 | 44.5 | 100% (forced) | 100% |

*The 32K cell of mtp-force is a run outlier; the same config measured 46.6 at 64K.
†plain 192K/225K (and their prefill, 518/531 t/s vs ~640–670 elsewhere) look
depressed relative to the 32–128K plateau of the same run — a second rep is
running to separate real high-ctx degradation of the single-row path from a
one-run artifact. The 32–128K cells agree with the discarded bare-env sweep's
plain arm (37.8/37.8/36.9), so the low-context baseline is solid.

Readings that matter (corrected against the TRUE plain baseline):

1. **MTP is a real +10–18% on the main ladder, +29–39% at high ctx** (auto
   44.7 vs plain 37.9 at 32K; 40.6 vs 31.4 at 225K, pending the rep-2 check
   of the depressed plain cells). The earlier "MTP is neutral" reading was an
   artifact of the contaminated plain arm — it compared MTP against MTP.
2. **The ceiling is near the current auto result.** mtp-force commits 1.93
   extra tokens per cycle at PERFECT acceptance yet lands at 44.5–46.6,
   within ~5% of what auto already delivers (40.6–44.7): verification width
   on this model/quant eats most of the ideal gain. Acceptance tuning has
   little left to capture — MTP here is a real but sublinear lever, not the
   2× the plan hoped.
3. **Forced depth-3 costs ~25%.** `reject2_streak >= 2` is the exit that makes
   the auto policy safe: with it, engagement stayed at 11% of cycles and
   accept2 held 68.8%; without it, accept2 fell to 41.6% and decode collapsed
   below the plain baseline (32.2 vs 37.7 at 64K). Task 4 Step 2 verdict:
   **do not retune the engage thresholds** — the forced arms are the
   counterfactual that proves the current policy already measures the right
   thing on this workload.
4. **Rep variance is ±8%** (two identical auto-MTP configs measured 48.9 vs
   44.7 at 32K). One rep cannot decide kernel-task deltas; the plan's 3 reps
   are mandatory for anything that claims a percentage point.

## What this says about the ceiling

Hot small-context decode measured 47–52 t/s (server logs and analytics);
bandwidth roof on 307 GB/s with ~5–6 GB/token touched is ≈50–60 t/s
single-pass. Kernel-only work (Phase 1) therefore has roughly 5–10% headroom
before MTP becomes the only lever above roof — which is why Task 4 was
scheduled before Phase 1 kernel rewrites despite the plan's ordering, and why
the PLE/n-gram disk reads (~24 pread rows/token, sorted, batched; well under
SSD bandwidth) were confirmed NOT the bottleneck.

## Task 4 tuning candidates (from measured data, not guesses)

1. ~~Depth-3 engagement on observed accept2 rate~~ — measured, closed: the
   auto policy's reject2 exit keeps engagement at 11% of cycles with 68.8%
   accept2; forced d3 (the counterfactual) is 25% worse. Leave thresholds.
2. `--mtp-margin` (default 3) A/B at production temperature 0.6 — the greedy
   bench accepts 73%, production sampling traffic 83%; margin trades draft
   commitment against that distribution.
3. Draft-vocab row count: 65568/248320 rows today; the force-accept ceiling
   lands ~5% above auto, so row misses cannot be costing more than that —
   widen only if a cheap experiment shows >5% recoverable.
4. Keep 4-bit in-kernel KV + grow-on-demand KV as-is (cache_hit 0.75–0.93 in
   production analytics; prefix reuse already works).
5. NEW from this campaign: the plain (no-MTP) path degrades hard past 128K
   (37.4 → 28.5 t/s) while MTP holds ~40 — if the rep-2 confirms it, the
   single-row eval loop's high-context behavior is itself a kernel Task-1
   candidate, and `--mtp` is load-bearing at long ctx, not just a bonus.

## Production changes shipped alongside (2026-10-29)

- Gateway sampling profile for this model (temp 0.6 / top_p 0.95 / top_k 20;
  rep penalty documented as an engine no-op) — replaces Qwen3-Coder-30B globals.
- `escalate()` runtime-gated: ds4 route no longer sends penalty keys the
  engine ignores (AI-Gateway-MLX 0256e2ef).
- `--mtp-timing` live in the registry command: acceptance stats now printed per
  session-free in production logs.
- PROD binary = prod/qwen4fast0-20261029 (deploy refs/smoke byte-identical on
  all three ds4 rows).
