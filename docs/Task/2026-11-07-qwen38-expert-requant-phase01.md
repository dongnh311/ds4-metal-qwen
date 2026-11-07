# Qwen3.8 expert quantization — Phase 0/1 findings

Branch `feature/qwen38-expert-requant`. Continues
`2026-11-07-qwen38-expert-requant-plan.md`.

## Phase 0a — routing hotness (measured)

`DS4_QWEN4_ROUTE_COUNTS` instrumentation (resident prefill path, env-gated)
recorded **1,460,760 expert selections** across the `hard-smoke` eval suite
(12 cases, all prompt prefills, T>1 only; decode T==1 skipped by design).

Per-layer coverage of selections by top-k experts (512 experts/layer,
10 slots/token):

| top-k | mean coverage |
|---:|---:|
| 8 | 17.2% |
| 16 | 25.6% |
| 32 | 37.1% |
| 64 | 52.6% |
| 128 | 72.4% |
| 256 | 92.1% |

- All 512 experts per layer receive selections — no dead experts.
- Skew is moderate, not extreme: covering 70% of traffic needs ~125/512
  experts upgraded per layer (~25% of expert bytes).
- **Implication for mixed-precision-by-hotness:** upgrading hot experts to a
  *faster-dequant* format caps the win at the per-format delta (~20% of GEMM
  on ~70% of pairs ≈ ~8-10% of routed GEMM ≈ ~80 ms/chunk best case) and
  *downgrading* cold experts to buy memory costs quality on 30% of traffic.
  Upgrading hot experts to fp16 is dead on arrival: ~4x bytes on the hottest
  25% ≈ +30% model size (over the 60 GiB gate) AND decode reads exactly those
  experts every token → decode slows. **Direction A verdict: weak, park it.**

## Phase 0b — per-layer expert format audit (measured)

UD-IQ2_XS is a *dynamic* quant; the 48 layers do not share one format:

| mid (gate/up) | down | layers | count |
|---|---|---|---:|
| iq2_s | iq4_nl | 3,5,6,12,15-18,21,22,24-28,32,37-40,47 | 21 |
| iq2_s | q2_0 | 13,14,19,20,23,29-31,33-36,41-46 | 18 |
| iq2_xxs | q2_0 | 1,8-11 | 5 |
| iq2_xxs | iq4_nl | 0,2,7 | 3 |
| iq2_s | **q4_0** | **4** | **1** |
| q8_0 | q8_0 | 48 (MTP) | 1 |

`bench_qwen4_moe_mm` @T=1856, E=512 S=10 (mid+down ms, production tile path):

| mid\down | iq4_nl (20) | q2_0 (42) | q4_0 (2) |
|---|---:|---:|---:|
| iq2_xxs (16) | 20.14 | 21.38* | — |
| iq2_s (22) | 22.58 | 21.46 | **34.28** |

*mid timings vary ±1.5 ms run to run; down column is the stable signal.

### Findings

1. **Layer-4 down projection is a kernel-coverage gap, not a quant problem.**
   q4_0 (type 2) is the ONLY routed-expert type without a tensor-op (NAX/GSQ)
   path — it falls back to simdgroup tiles: **19.6 ms vs ~7 ms** for the same
   shape, ~+12 ms per chunk on that one layer. Fixing it is a *kernel* task
   (add type 2 to the GSQ staging set or instantiate the tensor tiles), zero
   quality risk. Estimated recoverable: **~12 ms/chunk ≈ 1% of MoE stage.**
   Hand off to the kernel backlog, not this project.
2. **q2_0 down is already the fastest AND smallest** of the three down
   formats (~6.8-7.1 ms vs iq4_nl ~7.4-7.8). Unsloth's dynamic assignment is
   already speed-sensible: q2_0 where quality allows, iq4_nl where it doesn't.
3. **iq2_s vs iq2_xxs mid** costs ~1.5-2 ms/layer (39 layers ≈ ~70 ms/chunk)
   but iq2_xxs is the *lower*-precision variant — Unsloth put iq2_s on
   sensitive layers deliberately. Harmonizing to iq2_xxs everywhere is a
   quality-for-speed trade that belongs to a calibrated requant, not a
   format swap. Same for iq4_nl→q2_0 down (~12 ms/chunk upside, real quality
   risk).
4. Current estimated routed-GEMM total ≈ 1037 ms/chunk @T≈1856, consistent
   with the measured ~1047 ms (mid 717 + down 330 @T≈1928).

## Phase 1 — quality baseline (done)

`ds4 --perplexity-file` teacher-forced NLL on a fixed 8160-token corpus
(story prompt + model docs, `/tmp/qwen38-ppl-corpus.txt`), UD-IQ2_XS-mtp
+ PLE sidecar, full Metal path:

- tokens=35218 scored=8160 **nll=29803.3632 avg_nll=3.65237 ppl=38.566**

This is the anchor any future requant A/B compares against. Caveat: the
corpus is small and domain-mismatched; treat deltas >~0.5% NLL as signal,
smaller as noise. A second seed/dataset would tighten it before any
production claim.

## Phase 1 constraint discovered

A same-weights higher-precision reference does NOT exist locally: no BF16
safetensors checkpoint (HF cache has only the index), and the two other local
GGUFs are different finetunes/quant recipes. True requant experiments
(direction C, quality recovery) require downloading the official BF16
checkpoint (~200+ GB) — a user decision on disk/time, flagged in the plan.

## Interim verdict

- Mixed precision by hotness: **no-go** (bounded upside, decode/memory
  conflicts).
- Format harmonization without recalibration: the dynamic quant is already
  near the speed/quality frontier *for the formats the kernel supports*; the
  only clear anomaly (layer-4 q4_0) was a kernel gap worth ~12 ms/chunk —
  resolved on this branch, see below.
- Remaining real lever: calibrated requant from BF16 source (direction C) —
  needs the checkpoint download; or accept current quality and stop.

## Update — layer-4 q4_0 anomaly resolved (kernel side)

The "kernel gap" flagged above is fixed on this branch (`9dd059c8`):
q4_0 (type 2) expert rows now take the tensor-tile NAX path like every
other routed format — `stage8` already dequantizes them into sequential
32-value chunks, so only the two host allowlists needed type 2.
Bench @T=1856 E=512 S=10: down type-2 **24.19 → 10.34 ms**; the layer-4
combo (IQ2_S mid + q4_0 down) **56.28 → 28.93 ms**. Correctness:
`test_qwen4_moe_mm_specialize` type=2 exact at all T. Production impact
≈10–14 ms per MoE pass (single layer). All routed expert formats in the
model now have tensor paths — format coverage is complete.
