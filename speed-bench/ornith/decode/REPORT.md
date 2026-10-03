# Ornith decode speed (sub-project 2): report

Spec: `docs/superpowers/specs/2026-10-02-ornith-decode-design.md`. The work ran overnight 2026-10-02/03
under the user's delegated autonomy, sharing GPU windows with the Qwen and DS41F sessions.

**Merge scope, chosen by the user on 2026-10-03 ("A").** develop gets the measurement tools and the
records, not the lever code.
- Merged:
  - the env-gated instrumentation `DS4_QWEN35_SPEC_STATS`, `DS4_QWEN35_SPEC_OVERLAP` and
    `DS4_QWEN35_PROFILE=3`, all off and byte-identical unless set;
  - the Stage-0 runner and parser;
  - the spec, the plans, `PROFILE.md`, the lever records and this report.
- Not merged: the lever code. It stays on the local, unpushed branch `feature/ornith-decode` at
  `01d4bab0`:
  - two-row MoE `99ab7419` and `805868c9`;
  - dual accumulators `8cbfbd29`;
  - multi-flush `18b68818`;
  - decode3 prefetch `9f5c0915`.

  The env knobs those records name (`DS4_QWEN35_MOE_PAIR`, `DS4_QWEN35_ATTN_PREFETCH`,
  `DS4_QWEN35_FLUSH_EVERY`) exist only there.

## Outcome

**The +15% lossless target is not met.**
- Every lever built is exact: greedy output stays byte-identical, and the gate-1 dumps, `test_mtp_cli` and
  `test_qwen35_verify_batch` all pass with it on.
- None gives a measurable decode gain at 2K or 32K.
- The final A-B-B-A (window F) compared every lever off against prefetch plus multi-flush:

  | context | off | on | change |
  |---|---|---|---|
  | 2K | 92.9 | 93.1 | +0.2% |
  | 32K | 69.8 | 67.5 | -3.3% |
  | 128K | 45.1 | 44.7 | -0.9% |

- No lever is merged, so develop decodes exactly as before.
- No deploy, as the spec requires when the target is missed.

## What was learned (Stage 0, `PROFILE.md`)

- A plain 2K step takes 14.3 ms:
  - GDN mixer 5.3 ms, at ~70% of its byte floor;
  - MoE 4.8 ms, ~55%;
  - attention mixer 2.2 ms, ~50%;
  - head 2.0 ms, ~95%.
- The MTP verify costs 1.26 plain steps. Its MoE is x1.57, because each row reads its experts again.
  The two rows share 3.2 of 8 routed experts per layer.
- The GPU idles ~7% of an MTP cycle, between the verify, the draft and the next verify.
- Acceptance is 0.68 at 2K, 0.71 at 32K and 0.79 at 128K, with 1.67-1.78 tokens per cycle.

## Levers tried

| lever | what | exact | measured | decision |
|---|---|---|---|---|
| L2 two-row MoE | verify rows share an expert's weight read | yes (kernel memcmp, graph, gate 1) | -20% / -16% / -13% at 2K / 32K / 128K | off |
| L2b dual accumulators | L2 with one weight pass and two accumulators | yes | -20% / -13% | off |
| L5 decode3 prefetch | the next K/V tile loads during the current one | yes (11 positions, out + part) | single-knob +4.0% / +1.8%, combined in F: none | off |
| L1 multi-flush | commit the command buffer every 8 layers | yes (submission only) | single-knob +3.8% / +0.6%, combined in F: none; every 4 vs every 8: +0.7% | off |
| L4 MTP depth 2 | not built | | the projection needs >50% conditional second-draft acceptance | |

**Why L2 lost.**
- The per-row verify's two threadgroups already run concurrently, so the second read of a shared expert
  hits the system cache. The projected DRAM saving was not there.
- The pair kernels use the generic dot. The per-row path uses specialized Q4_K kernels (gate and up
  interleaved, multi-row down on float4 activations), so the pair kernels were slower on every slot.

**Measurement noise.** Single-knob A-B-B-A runs moved ±3-5%: the base 2K decode was 90.1, 95.8 and
91.2 t/s in three runs, and prefill moved 5% under a decode-only knob. Window F, with both knobs in one
run, shows the window-C "gains" were noise.

## What remains (not done; each would be its own decision)

- **Lossless, larger engineering.**
  - Raise the MoE and attention-projection kernels' bandwidth use (55% and 50% of floor) with new
    tilings that keep each row's per-lane order.
  - Overlap the MTP draft with the verify readback; it needs the draft's parent chosen on the GPU.
- **Not lossless (excluded by the user's choice).** An FP8 K/V cache would cut attention bytes in half
  from 16K up.

## Rulings made overnight

Listed in the final message; the plan ledgers under `.superpowers/sdd/2026-10-02-ornith-decode-*` hold
them all.

## Receipts

- `PROFILE.md` and `receipts/stage0/table.md`: Stage 0.
- `levers/moe-pair.md`, `levers/attn-flush.md` and `levers/ab-*`: the lever windows.
- Window F's final run: `levers/ab-final.{txt,json}`.
