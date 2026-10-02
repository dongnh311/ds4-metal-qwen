# PROD KV precision and the MTP confidence gate: neither changes PROD; ISTA parked

Two checks taken from the Strata engine (Niko1221/Strata) and run on M5 Pro 64 GB on 2026-10-02, in
two gateway pauses (`hold-gateway.sh`). Both found nothing to change in PROD.

1. **KV precision.** PROD's default in-kernel 4-bit KV costs +0.91% perplexity against an f16 cache.
   Strata measured +8-12% for 4-bit KV, which does not reproduce here. FP8 is +0.07%. The default stays
   4-bit.
2. **MTP confidence gate (`DS4_QWEN4_MTP_MIN_P`).** Strata drafts up to 3 tokens and keeps drafting only
   while the previous draft's probability is at least 0.5. In ds4 the gate cuts verify cycles by 7-34%,
   but tokens/s barely moves: ISTA +0.9 to +3.6%, PROD inside run noise. The spike branch was deleted
   unmerged.

Decision the same evening (user): **keep the current PROD**. ISTA is parked:
- the harness accuracy is a tie (paired 10 vs 14, p 0.54);
- ISTA decodes 5-9% slower;
- 512K is not proven for ISTA, which already wires 52.75 GiB at 240K.

The local ISTA and Ivan GGUFs were deleted. Both stay on HF:
`dongnhdev/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP` (e230cb9) and
`ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2`.

## 1. KV precision on PROD

Run in window 1, 18:16:50 to 18:51:19, with the gateway paused.
- **Binary:** develop `9695695c` plus the spike commit below, gate off. The spike touches only the MTP
  draft path, which `--perplexity-file` does not run.
- **Model and flags:** PROD with its registry streaming flags (`--ssd-streaming
  --ssd-streaming-cache-experts 6GB`, `DS4_QWEN4_STREAM_FULL_LAYERS=32`, `DS4_QWEN4_KV_GROW=1`).
- **Scoring:** teacher-forced, `--perplexity-file -c 4096 -n 2048`, on the six texts of
  `2026-10-01-512k-deploy-and-ista-arm.md`.

The three arms:
- **q4:** the default in-kernel 4-bit KV, with one scale per 64-value block;
- **fp8:** `DS4_QWEN4_KV_Q4=0 DS4_QWEN4_KV_FP8=1`;
- **f16:** `DS4_QWEN4_KV_Q4=0`, the reference.

Average NLL per token (lower is better); Δ is the perplexity change against f16:

| text | q4 | fp8 | f16 | q4 Δ | fp8 Δ |
|---|---|---|---|---|---|
| en | 0.7338 | 0.7188 | 0.7269 | +0.69% | −0.81% |
| code | 0.03772 | 0.03693 | 0.03643 | +0.13% | +0.05% |
| vi | 1.2579 | 1.2537 | 1.2586 | −0.06% | −0.49% |
| en-new | 2.1585 | 2.1409 | 2.1338 | +2.50% | +0.72% |
| code-new | 0.7860 | 0.7809 | 0.7725 | +1.36% | +0.84% |
| vi-new | 2.1363 | 2.1292 | 2.1278 | +0.85% | +0.15% |
| **mean** | | | | **+0.91%** | **+0.07%** |

- The worst text is en-new, at +2.5%. Strata's +8-12% came from its own 4-bit layout, which it measured
  even with a Hadamard rotation. It does not carry over to ds4's per-64-block scales.
- The long-context arm ran for q4 only: the 31K-token story at `-c 32768 -n 16384` gave avg_nll 3.6833.
  Each long arm takes about 10.5 min, so the window would have overrun the approved pause. The fp8 and
  f16 long arms were not run.
- FP8 would buy back almost all of the 0.9%. The gain is too small to change the PROD default without
  a harness run.

## 2. MTP confidence gate (spike)

Run in window 2, 18:57:41 to 19:03:11.
- **Branch:** `feature/qwen4-mtp-min-p` `46e098a3`, on develop `9695695c`. It was local only and was
  deleted unmerged.
- **What the gate does:**
  - `DS4_QWEN4_MTP_MIN_P=p` forces draft depth 3.
  - It keeps a draft only while the softmax probability of the drafted token, taken over the head rows
    (`qwen4_draft_prob`), is at least p.
  - It applies at all 9 draft sites: the session draft, plus 4 cycle sites, each with a first draft and a
    chain draft.

The arms, each run with `--mtp --mtp-timing --temp 0 -n 256 --nothink`:
- **base:** develop `9695695c`;
- **def:** the branch with the gate off. Its output must equal base;
- **d2:** `DS4_QWEN4_MTP_DEPTH=2`, which always drafts one token;
- **p50:** `DS4_QWEN4_MTP_MIN_P=0.5`.

The models:
- **ISTA:** the IQ3_XXS hc-F16 file, resident, `-c 32768`;
- **PROD:** registry streaming flags, `-c 262144`.

Generation t/s, with verify cycles and the accepted share of first drafts in brackets:

| model / prompt | base | def | d2 | p50 | p50 vs def |
|---|---|---|---|---|---|
| ISTA prime (code) | 45.93 | 46.28 (102, 96.1%) | 46.95 (102, 96.1%) | 47.96 (70, 97.1%) | +3.6% |
| ISTA blue sky (VI) | 41.31 | 41.64 (60, 80.0%) | 42.01 (66, 75.8%) | 42.34 (43, 93.0%) | +1.7% |
| ISTA TCP vs UDP | 42.05 | 43.40 (121, 84.3%) | 43.44 (138, 83.3%) | 43.78 (100, 87.0%) | +0.9% |
| PROD prime (code) | 44.22 | 52.24 (54, 94.4%) | 48.74 (72, 94.4%) | 50.04 (50, 96.0%) | −4.2% |
| PROD blue sky (VI) | 40.71 | 43.74 (85, 70.6%) | 44.17 (85, 70.6%) | 43.83 (56, 83.9%) | +0.2% |
| PROD TCP vs UDP | 43.76 | 44.38 (135, 81.5%) | 46.60 (140, 82.1%) | 44.05 (102, 86.3%) | −0.7% |

- **Gate off is byte-identical to develop:** def equals base on all 6 runs. On PROD, base is slower than
  def because it always runs first, on a colder expert cache, and the two have equal cycle counts.
- **Fewer cycles, the same speed.** p50 cuts verify cycles by 7-34%, but a 3-row verify plus the chained
  draft costs about what those cycles save. Single PROD streaming runs carry about ±5% noise, so the
  PROD rows are a tie.
- **Why this differs from Strata's 2.4-3.2 tokens per pass:** ds4's window policy already engages
  depth 3 on deterministic stretches. And on Metal, a wider verify grows the routed-expert union almost
  linearly.
- **Divergence at near-ties:** two runs differ from def — ISTA VI under d2 ("đỏ hay cam" vs "đỏ, cam")
  and PROD TCP under p50. Both stay correct and fluent. The T=2 vs T=3 verify numerics split a near-tie
  argmax. This predates the gate: d2 alone shows it.

## 3. Qwen gate for `fix/v41-qgate-slab` f51c9d86

DS41F's change limits the one-slab expert cache to qwen4 and GLM models. It is checked against develop
`2146f8db`, PROD only, from 21:24:44 to 21:29:55. **Verdict: PASS.**

- **Identity:** PROD greedy output on 3 prompts is identical to the saved `2146f8db` outputs. The base
  rebuild matches them too.
- **Logs:** the expert-cache lines (cache target, stream gates, memory plan) are identical between base
  and candidate.
- **Order of calls:** `ds4_gpu_set_qwen4_model()` runs before the streaming cache is configured
  (`ds4.c:74756`), so qwen4 keeps the one-slab layout.
- **Speed:** ds4-bench 8K/128, two rounds, order swapped. Steady decode: base 27.50 / 28.10, candidate
  28.52 / 27.72 t/s. Prefill: base 405 / 365, candidate 398 / 414 t/s. Both differences are noise.

Raw outputs are outside the repo:
- `ds4-metal-data/sp3gemm/kvq/` (KV);
- `ds4-metal-data/sp3gemm/minp/` (spike);
- `ds4-metal-data/merge-gate/out-f51c/` (gate).
