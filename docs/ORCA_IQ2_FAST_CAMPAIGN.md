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

## Baseline sweep (220K-target protocol)

Sweep 2026-10-29 16:50–18:04, receipts `speed-bench/receipts/orca-iq2-20261029-165008/`,
merged table at `speed-bench/orca-iq2-baseline.csv`. One rep per arm
(variance lesson below); gen = greedy continuation of a frozen 1.2 MB ds4.c slice.

| arm | 32K | 64K | 128K | 192K | 225K | draft-1 acc | depth-3 accept2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain (true) | *rerun pending — first two "plain" arms measured MTP (see harness bugs)* | | | | | | |
| mtp (auto depth) | 44.7 | 41.3 | 41.7 | 39.5 | 40.6 | 73.2% | 68.8% (32/280 cycles engaged) |
| mtp-d2 (forced depth 2) | 48.3 | 41.8 | 40.8 | 40.4 | 40.6 | 74.2% | — (never engages) |
| mtp-d3 (forced depth 3) | 42.4 | 32.2 | 32.8 | 31.4 | 30.9 | 69.9% | 41.6% (99/238) |
| mtp-force (accept-all ceiling) | 38.2* | 46.6 | 45.7 | 42.7 | 44.5 | 100% (forced) | 100% |

*The 32K cell of mtp-force is a run outlier; the same config measured 46.6 at 64K.

Readings that matter:

1. **Forced depth-3 costs ~25%**. `reject2_streak >= 2` is the exit that makes
   the auto policy safe: with it, engagement stayed at 11% of cycles and
   accept2 held 68.8%; without it, accept2 fell to 41.6% and decode collapsed.
   Task 4 Step 2 verdict: **do not retune the engage thresholds** — the current
   policy already measures the right thing on this workload; the forced arms
   are the counterfactual that proves it.
2. **The ceiling is near the auto result.** mtp-force commits 1.93 extra tokens
   per cycle at perfect acceptance yet lands within a few percent of d2 —
   verification width on this model/quant eats most of the ideal gain. MTP is
   roughly neutral-to-small-positive here, not a 2× lever.
3. **Rep variance is ±8%** (two identical auto-MTP configs measured 48.9 vs
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

1. Depth-3 engagement on observed accept2 rate, not first-draft streaks.
2. `--mtp-margin` (default 3) A/B at production temperature 0.6.
3. Draft-vocab row count: 65568/248320 rows today; wider rows only if the
   forced-accept vs opportunistic gap (mtp-force arm) proves row misses cost
   more than verify width.
4. Keep 4-bit in-kernel KV + grow-on-demand KV as-is (cache_hit 0.75–0.93 in
   production analytics; prefix reuse already works).

## Production changes shipped alongside (2026-10-29)

- Gateway sampling profile for this model (temp 0.6 / top_p 0.95 / top_k 20;
  rep penalty documented as an engine no-op) — replaces Qwen3-Coder-30B globals.
- `escalate()` runtime-gated: ds4 route no longer sends penalty keys the
  engine ignores (AI-Gateway-MLX 0256e2ef).
- `--mtp-timing` live in the registry command: acceptance stats now printed per
  session-free in production logs.
- PROD binary = prod/qwen4fast0-20261029 (deploy refs/smoke byte-identical on
  all three ds4 rows).
