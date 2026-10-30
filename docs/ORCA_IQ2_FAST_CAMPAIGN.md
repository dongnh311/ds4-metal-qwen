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

## Task 2 — temperature A/B at the real sampling distribution (2026-10-30)

`speed-bench/run-mtp-accept-ab.py` (server-level, the EXACT registry command =
PROD binary + steering + draft vocab + `--mtp --mtp-timing`, only
port/kv-dir/lock swapped), 6 fixed seeded streaming `/v1/completions` prompts
(py, c, vi, json, math, agent), 300 tokens each, per-arm server SIGTERM drain
printing the acceptance counters. Raw rows:
`speed-bench/mtp-accept-task2.csv`; full server logs:
`speed-bench/receipts/accept-ab-20261030-084242/`.

| temp | cycles | draft-1 acc | committed tok/cycle | mean decode | mean TTFT |
|---:|---:|---:|---:|---:|---:|
| 0.0 (greedy) | 1073 | 65.9% | 1.678 | 48.0 t/s | 0.59 s |
| 0.3 | 969 | 65.9% | 1.759 | 47.1 t/s | 0.59 s |
| 0.6 (LIVE) | 968 | 68.5% | 1.860 | 46.4 t/s | 0.58 s |
| 1.0 | 858 | 71.2% | 2.008 | 44.0 t/s | 0.59 s |

Verdicts:

1. **`--mtp-margin` is RETRACTED as a Qwen3.8 lever.** Zero margin references
   in the qwen4 spec region; the `top0-top1 >= 3` gate (ds4.c:88926) belongs
   to the Ornith/dspark drafting loop. The gateway profile now says so
   (AI-Gateway commit ad4b1261).
2. **The cold-start reasoning was backwards.** Acceptance RISES with
   temperature (1.678 → 2.008 tokens/cycle): the probability verifier accepts
   near-miss drafts that strict argmax equality rejects. The measured decode
   cost of temperature (−3.3% at 0.6, −8.3% at 1.0) is the sampler building a
   248K-row distribution every cycle — not lost speculation.
3. **temp 0.6 stands, on an honest argument**: reliability (loop-prone family,
   agentic JSON) at a known ~3% price, speculation actually slightly better
   than greedy. The loop-counter half of the pre-registered A/B stays OPEN
   (needs a working day of `sampling_override` traffic per arm).
4. Per-request TTFT ~0.6 s at small prompts on a warm session confirms the
   prefill story below is about MISSES, not about the serving path.

## Task 1 — prefill/TTFT (2026-10-30)

`speed-bench/run-prefill-task1.sh`, receipts `prefill-20261030-085108/`,
table `speed-bench/prefill-task1.csv`. Morning window (08:52–10:04).

**P1 — rep 3 lands the Task-0 numbers at n=3 paired reps**
(`orca-iq2-baseline.csv`, all 3 reps of plain+mtp now):

| ctx | plain (3-rep) | mtp (3-rep) | paired gains per rep | MTP gain |
|---|---:|---:|---|---:|
| 32K | 37.1 | 43.5 | +18.0 / +15.9 / +18.6 | **+17.5%** |
| 64K | 36.7 | 40.1 | +9.7 / +4.4 / +14.2 | **+9.4%** |
| 128K | 36.0 | 40.5 | +11.6 / +15.1 / +11.1 | **+12.6%** |
| 192K | 31.5 | 39.3 | +38.7 / +17.4 / +20.3 | **+25.4%** |
| 225K | 32.9 | 40.0 | +29.2 / +17.7 / +18.4 | **+21.8%** |

The rep-1 192K outlier (+38.7%) shrinks to +17–20% with the 22:33 and 08:52
reps; the durable reading is **MTP = +9–25%, no ctx where it is negative**,
and the high-ctx half of the curve (192–225K, +22–25%) is where production
sessions actually live.

**P2 — the prefill depth curve at CONSTANT segment size.** Task-0's doubling
ladder confounds segment size (32K/64K/94K) with depth; the equal-32K-step
ladder removes it. Auto-MTP, chunk 4096, gen 128:

```
depth:     32K     64K     98K    131K    164K    197K    225K
t/s:    702.96  676.35  672.79  660.46  654.79  646.91  635.04
```

−9.7% from 32K to 225K at equal segments — prefill is NOT quadratically
collapsing on this model (4-bit in-kernel KV + chunked attention keep the
depth curve gentle). A cold 200K-token prompt costs ≈ 200K/660 ≈ 300 s of
prefill regardless of position; the earlier apparent 518 t/s 'collapse' in
the Task-0 plain rows was segment-order noise, not depth. **The real TTFT
lever is therefore cache hits (already 0.75–0.93 in production analytics),
not faster prefill math.**

**P3 — `--prefill-chunk` sweep, pure prefill (gen 0), same equal ladder:**

| chunk | 32K | 64K | 98K | 131K | 164K | 197K | 225K | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 650.6 | 631.2 | 629.1 | 618.8 | 614.5 | 607.9 | 599.7 | 621.7 |
| **4096 (prod)** | 702.4 | 680.5 | 673.7 | 660.3 | 654.2 | 644.9 | 636.4 | 664.6 |
| **8192** | 740.4 | 711.9 | 713.1 | 700.6 | 690.2 | 682.1 | 666.6 | **700.7** |
| 16384 | 720.6 | 660.5 | 636.6 | 627.7 | 631.5 | 616.0 | 451.7 | 620.4 |

**8192 beats production 4096 on EVERY frontier (+4.5–6.5%, mean +5.4%)**,
and it ran AFTER 4096 in the window — the day-part drift direction makes
that a floor, not a ceiling. 2048 is −6.5%, 16384 wins only at 32K then
falls off a cliff at 225K (451 — a residency/scratch cliff worth
understanding before anyone proposes bigger chunks again).

Decision gate before touching the production flag (window4): (V1) 8192 with
decode ON and an RSS/swap sampler — it must not push a 64 GB box toward
pressure; (V2) deploy-smoke BYTE IDENTITY vs the recorded 4096 refs — chunk
size changes batch shape, and batched reductions have flipped greedy ties
before in this engine. Only if BOTH pass does `--prefill-chunk` become 8192.

**Window4 — the two gates (10:18–10:37, receipts `prefill-20261030-101759/`):**

- V2 BYTE GATE **PASSED**: deploy-smoke against a registry COPY carrying 8192,
  same PROD binary, compared to the recorded greedy 4096 refs — all three
  outputs (vi 232 tok, code 300 tok, tool 68+102 tok) came back
  "same as ref". Chunk size does not flip a single token on this model/path.
- V1 decode/acceptance at 8192 with gen ON: prefill 801.7→668.5 t/s
  (32K→225K, still above every 4096 frontier), decode 48.2/43.3/…/39.8 t/s,
  acceptance 74.3% (n=495 cycles) — same distribution as the 4096 ladder.
  BUT the window's sampler showed machine swap climbing 983→2973 MB. On a
  64 GB box that is a stop-and-measure flag, not a footnote.

**Window5 — memory A/B on the REAL server command (10:46–10:53, receipts
`memab-8192-20261030/`):** exact registry `process_command` per arm (262K
ctx, steering, vision sidecar, KV disk), only `--prefill-chunk` and port/kv
swapped; 3 min idle sampling + 4 real chat requests per arm.

| arm | swap start | swap peak | server RSS peak | req walls |
|---|---:|---:|---:|---|
| 4096 | 2885 MB | 2885 MB | 4173 MB | 5/5/4/4 s |
| 8192 | 2885 MB | **2885 MB (delta 0)** | 7023 MB | 5/5/4/4 s |

The window4 swap climb was RESIDUAL from the earlier bench windows (both arms
started at 2885 MB and neither grew); 8192's real cost is RSS ≈ +2.9 GB of
logits scratch (chunk × vocab × 4 B) with zero swap movement, and identical
request walls. Pre-registered rule (switch only if swap delta ≤ +1.5 GB):
**PASSED at 0 MB.**

**DECISION 2026-10-30 11:33: production `--prefill-chunk` 4096 → 8192.**
Live registry edited (backup `runtime-registry.json.bak-prefillchunk8192-20261030`),
slot SIGTERMed on a quiet window, gateway relaunched it from the mtime-cached
registry; running process verified `--prefill-chunk 8192`, gateway `/status`
green, test request 200. Expected effect: +5.4–14% prefill on cold long
prompts (TTFT), byte-identical outputs. Rollback = restore the backup and
SIGTERM the slot.


1. ~~Depth-3 engagement on observed accept2 rate~~ — measured, closed: the
   auto policy's reject2 exit keeps engagement at 11% of cycles with 68.8%
   accept2; forced d3 (the counterfactual) is 25% worse. Leave thresholds.
2. ~~`--mtp-margin` A/B~~ — CLOSED 2026-10-30, twice: margin is not read by
   the qwen4 spec path at all (Ornith/dspark-only, ds4.c:88926), AND the
   temperature A/B that replaced it (Task 2 above) shows acceptance rises with
   temperature rather than falling, so the cold-start premise was inverted.
   0.6 stands on reliability grounds at a measured ~3% decode price.
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
