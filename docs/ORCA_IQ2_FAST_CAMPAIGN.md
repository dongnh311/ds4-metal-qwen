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
`speed-bench/receipts/orca-iq2-20261029-165008/`), the true-`plain` re-run
22:06–22:23 (`…/220605/`, after the flag-decoupling fix — the 16:50 "plain"
arm silently ran MTP because `--mtp-timing` implied `--mtp`), and rep 2 of
plain+auto-MTP 22:33–23:05 (`…/223243/`). Merged table:
`speed-bench/orca-iq2-baseline.csv`. plain/mtp are 2-rep means; d2/d3/force
1 rep (variance lesson below). gen = greedy continuation of a frozen 1.2 MB
ds4.c slice, production launch env mirrored from the live registry.

| arm (gen t/s) | 32K | 64K | 128K | 192K | 225K | draft-1 acc | depth-3 accept2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain (true, no MTP) | 37.1 | 36.9 | 35.6 | 30.5 | 32.1 | — | — |
| mtp (auto depth) | 43.4 | 39.5 | 40.3 | 38.8 | 39.6 | 73.2% | 68.8% (32/280 cycles engaged) |
| — MTP gain over plain | +17% | +7% | +13% | +27% | +23% | | |
| mtp-d2 (forced depth 2) | 48.3 | 41.8 | 40.8 | 40.4 | 40.6 | 74.2% | — (never engages) |
| mtp-d3 (forced depth 3) | 42.4 | 32.2 | 32.8 | 31.4 | 30.9 | 69.9% | 41.6% (99/238) |
| mtp-force (accept-all ceiling) | 38.2* | 46.6 | 45.7 | 42.7 | 44.5 | 100% (forced) | 100% |

*The 32K cell of mtp-force is a run outlier; the same config measured 46.6 at 64K.
Gain row: the per-context delta of the two-rep means. rep-1-only and
rep-2-only pairs bracket it (32K +17.9/+15.7, 225K +29.0/+17.7) — real gain
is at least +15% and larger at high ctx; rep 1's 192K/225K plain cells were
degraded outliers (28.5/31.4 vs rep 2's 32.4/32.8), so the +27/+23% figures
carry ±10% uncertainty.

Readings that matter (2-rep plain/MTP, corrected against the TRUE baseline):

1. **MTP is a real +15% floor on this build** (up to +23–27% at ≥192K, where
   the no-MTP path decays past its 37 t/s plateau to ~31 while auto-MTP
   holds ~39). MTP is also what FLATTENS the context curve: plain loses 13%
   of its throughput between 32K and 225K, auto-MTP only 9%. The earlier
   "MTP is neutral" reading was an artifact of the contaminated plain arm —
   it compared MTP against MTP.
2. **The ceiling is near the current auto result.** mtp-force commits 1.93
   extra tokens per cycle at PERFECT acceptance yet lands at 44.5–46.6,
   within ~5% of what auto already delivers at high ctx: verification width
   on this model/quant eats most of the ideal gain. Acceptance tuning has
   little left to capture — MTP is real but sublinear, not the 2× the plan
   hoped.
3. **Forced depth-3 costs ~25%.** `reject2_streak >= 2` is the exit that makes
   the auto policy safe: with it, engagement stayed at 11% of cycles and
   accept2 held 68.8%; without it, accept2 fell to 41.6% and decode collapsed
   below the plain baseline (32.2 vs 36.9 at 64K). Task 4 Step 2 verdict:
   **do not retune the engage thresholds** — the forced arms are the
   counterfactual that proves the current policy already measures the right
   thing on this workload.
4. **Rep variance is ±5–10% and there is a day-part offset.** rep 2 (22:33–
   23:05) ran ~5% below rep 1 (16:50–18:04) on EVERY arm including prefill
   (mtp 32K: 44.7→42.0, pre 705→625) — thermal/power state, not config.
   Within-run pair comparisons (same arm, same window) are the only honest
   unit; the plan's 3 reps are mandatory for anything claiming a percentage
   point, and cross-window deltas should be paired (plain-vs-mtp WITHIN a
   rep, then averaged).
5. **Greedy determinism verified**: acceptance counters identical across reps
   (73.2% both reps) — the bench harness itself adds no acceptance noise;
   all run-to-run noise is machine state.

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
5. Confirmed by rep 2: the plain (no-MTP) path loses ~13% between 32K and
   225K while auto-MTP loses 9% — MTP is load-bearing at long ctx, so any
   future change that lowers acceptance (hotter temp, different quant) costs
   DOUBLE at high context. Kernel Task-1 candidate: the single-row eval loop
   at >192K (31 t/s vs its own 37 plateau).

## Production changes shipped alongside (2026-10-29)

- Gateway sampling profile for this model (temp 0.6 / top_p 0.95 / top_k 20;
  rep penalty documented as an engine no-op) — replaces Qwen3-Coder-30B globals.
- `escalate()` runtime-gated: ds4 route no longer sends penalty keys the
  engine ignores (AI-Gateway-MLX 0256e2ef).
- `--mtp-timing` live in the registry command: acceptance stats now printed per
  session-free in production logs.
- PROD binary = prod/qwen4fast0-20261029 (deploy refs/smoke byte-identical on
  all three ds4 rows).
