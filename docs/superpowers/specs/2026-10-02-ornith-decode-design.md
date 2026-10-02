# Ornith decode speed (sub-project 2): design

Date: 2026-10-02. Branch: `feature/ornith-decode`, cut from develop `cd4a99ec` (Ornith 512K merged and
deployed). Sub-project 1 was the 512K context (`2026-10-02-ornith-512k-design.md`).

## 1. Goal and success

Raise Ornith's decode rate on ds4 with `--mtp` (the production setting), losslessly.

- **Target:** at least +15% decode t/s at 2K and at 32K. This is measured A-B-B-A in one window by
  `speed-bench/ornith/m4_ab.py --mode lever` (256 tokens, temperature 0, thinking on, a fresh nonce per
  prompt). The base arm is the same build with every new lever off, which is develop's behaviour.
  - M7's gate 3 for reference: 93.0 t/s at 2K, 68.3 at 32K, 50.4 at 128K.
  - So the target is about 107 t/s at 2K and 78.5 at 32K, measured against the same-day base arm.
- **Guard:** 128K decode no more than 3% below the base arm.
- **Production view:** recorded, not gated. Gateway analytics since 2026-09-27 show decode time spent
  mainly under 4K (345 requests, 99 t/s) and at 16-48K (159 requests, 71 t/s, 86% of the prompt cached).
  Production samples at temperature 0.7 with top-k 20, so acceptance there differs from the greedy
  bench; Stage 0 measures both.
- **If the target is met:** merge, push, and deploy through `prod/ornith-decode-YYYYMMDD`. The user chose
  "+15%, then ship and deploy" and asked for the work to be finished autonomously.
- **If it is not met** once the chosen levers are spent: no deploy. The report gives each lever's measured
  gain and what would come next.

## 2. Constraints

- **Lossless.** Greedy output is byte-identical to develop, with every lever on:
  - Ornith gate-1 dumps match develop's byte for byte at prefill chunks 64/512/2048, with and without
    `--mtp`;
  - `tests/ornith/test_mtp_cli.py`: `--mtp` output equals plain decoding;
  - `test_qwen35_verify_batch`, `test_qwen35_mtp` and `test_qwen35_graph` pass;
  - the `ds4_test` rewind, snapshot and payload entries pass.

  Anything that changes the arithmetic is out: a different reduction order, split count, accumulation
  type, or fused math that rounds differently. Moving work between command buffers, reading the same
  bytes fewer times, or computing the same values on the GPU instead of the host is in.
- **Qwen3.8 stays byte-identical.**
  - New kernels and paths are Ornith-only: `metal/qwen35.metal`, `ds4_qwen35moe.inc` and the Ornith
    branches of `ds4.c`.
  - A change to a shared file (`ds4_metal.m`, `ds4_gpu.h`, shared `metal/*.metal`, shared `ds4.c` code)
    is gated by the Qwen session before merge. That session owns the Qwen gate (standing rule); this
    work only tells it what changed.
- **Each lever sits behind an env knob** (`DS4_QWEN35_<NAME>`, `=0` restores the old path).
  - A knob turns on by default only after its identity tests pass and its A/B gains at least 3% at 2K or
    32K, with no loss above 3% elsewhere.
  - If a lever depends on the Metal compiler reproducing the same arithmetic in two kernels (as M7's
    two-row matvec did), it must also self-check at run time, the way M7's batched verify does.
- **GPU windows.**
  - One model process at a time.
  - The stack is paused and restored by the scratch `gpu/stack.sh` (zombie-aware).
  - Peers get a heads-up before and "done" after.
  - SIGTERM only. A Metal process is never `kill -9`ed.
- **No lossy levers.** FP8 K/V, approximate kernels and lower-precision accumulation are out of scope.
  The user chose lossless only.

## 3. Stage 0: profile (one GPU window, about 45 min)

The levers below are candidates. Stage 0 measures where a decode step's time goes, so the work goes
where the headroom is. It runs on develop's code plus additive instrumentation (env-gated, off by default,
byte-identical when off).

At 2K, 32K and 128K, each with plain decode and with `--mtp`, at temperature 0 and at production
sampling (0.7, top-k 20):
1. **Wall rate and GPU busy fraction:**
   - `DS4_METAL_GPU_BUSY_PROFILE` gives GPU busy time per committed token;
   - `DS4_METAL_CB_TIMES` gives the per-command-buffer driver, queue-wait and GPU time, and the idle gap
     between buffers.

   The gap between wall time and GPU-busy time is the host-side headroom: CPU argmax, the logits
   readback, encoding, and the wait between the verify and the draft.
2. **Stage split:** `DS4_QWEN35_PROFILE=2` for decode and `=1` for the T=2 verify. It reports GDN layers,
   attention layers and the head. The profiler syncs per layer, so the split is read as shares, not
   absolute times.
3. **Per-kernel cost against the byte floor** (model-free microbench, as in M7's
   `bench_qwen35_verify`):
   - the T=1 and T=2 routed-expert MoE for the Q5_K and Q4_K layers;
   - the GDN recurrence step;
   - the dense Q8_0 projections;
   - the lm head and the 64K draft head;
   - decode attention at 2K, 32K and 128K positions.

   Each is reported as ms and as the fraction of its byte floor (bytes read / ~280 GB/s).
4. **MTP cycle anatomy** (new instrumentation, `DS4_QWEN35_SPEC_STATS=1`). Over a run it prints:
   - acceptance;
   - mean time per cycle split into verify GPU, draft GPU and host;
   - the number of committed tokens per cycle.
5. **Two-row expert overlap:** for each verify, how many of row 1's 8 routed experts row 0 also routed, per
   layer, averaged. This bounds a two-row MoE's saving.
6. **Second-draft acceptance (measure only):** with `DS4_QWEN35_SPEC_PROBE2=1`, each cycle runs the MTP head
   once more on its own draft and records whether that second draft matches the token the target later
   commits. It does not change the output. This bounds MTP depth 2.

Stage 0 writes `speed-bench/ornith/decode/PROFILE.md`, with a table per context and a projected gain for
each lever in §4. The projected gain is time saved per committed token over the current time per
committed token, from measured quantities only.

## 4. Lever catalogue (all lossless)

| id | lever | mechanism | helps most |
|---|---|---|---|
| L1 | cycle overhead | fewer host round trips per MTP cycle: argmax on the GPU (same tie rule: lowest index) so only ids come back at temperature 0; encode the draft in the verify's command buffer where the parent is known on the GPU; avoid copying both verify rows' 248320 logits when one suffices | short contexts |
| L2 | two-row MoE in the verify | one Ornith-only kernel that serves both rows' routed experts. An expert both rows routed is read once. Each row's result `memcmp`-equals the T=1 kernel's (same per-row accumulation order), as M7 proved for the dense two-row matvec | 2K, 32K |
| L3 | launch and encode | merge adjacent per-layer dispatches that run back to back on the same data (for example norm then projection) only where the arithmetic stays the same kernel code; cut redundant barriers; tune `DS4_QWEN35_FLUSH_LAYER` | 2K |
| L4 | MTP depth 2 | chain the draft head once more; verify 3 rows with the GDN snapshot after rows 1 and 2; commit 1-3 tokens. Every row still equals plain decoding, so output is unchanged. Built only if Stage 0's second-draft acceptance projects at least +5% after the 3-row verify cost | 2K, 32K |
| L5 | decode attention, 16-48K | only changes that keep the per-split reduction order (memory layout, prefetch, occupancy without changing split boundaries) | 32K |

**Order.** After Stage 0, the levers are ranked by projected gain per unit of work. They are implemented
in that order until both targets are met or the list is spent. A lever whose projection is under 3% is
not built. The ranking and the cut are recorded as rulings in the plan's ledger.

**Planning in two parts.**
- Plan A covers Stage 0: the instrumentation, the microbench, the window and `PROFILE.md`.
- Plan B is written after Stage 0 and covers the chosen levers, with exact kernels and tests. It argues
  from this spec and `PROFILE.md`.

## 5. Verification per lever

- **RED first.** Each lever's test fails before the lever exists, as in M7:
  - kernel-level exactness (the new path's output `memcmp` equals the old path's, for the real shapes);
  - graph-level exactness (every cycle's logits equal a plain session's).
- **GPU identity check in a window:**
  - gate-1 dumps with the lever on equal develop's at chunks 64/512/2048;
  - `test_mtp_cli` passes;
  - `test_qwen35_verify_batch` passes.
- **Speed:** `m4_ab.py --mode lever` at 2K, 32K and 128K, base env = lever off and lever env = lever on.
  Raw output goes to `speed-bench/ornith/decode/levers/`.
- **Final acceptance:** one A-B-B-A at 2K, 32K and 128K, base = every lever off and lever = every lever on.
  The +15% target is read from this run.
- **Merge gates:**
  - a whole-branch review on the most capable model;
  - `make test` with Ornith, matching develop's known DeepSeek-only failures;
  - the Qwen session's gate if a shared file changed;
  - the identity checks re-run on the merge commit.

## 6. Deploy (only if §1's target is met)

- Cut `prod/ornith-decode-YYYYMMDD` from the pushed develop, then install it.
- Smoke all four ds4 rows against references recorded from the develop build.
- Run a live check through the gateway.
- The registry does not change, because the levers default on in code. If a lever needs a flag, the row
  edit goes through the gateway deploy owner, who re-stages with the user's go.
- Rollback: install `prod/ornith-512k-20261002`.

## 7. Out of scope

- Lossy levers.
- Prefill speed.
- The Qwen3.8 decode path.
- Vision.
- Multi-session batching.
- MTP drafts beyond depth 2.

## 8. Risks

- **The headroom may not be where M4 left it.** M5-M7 changed attention and the verify since then. Stage 0
  exists for this.
- **Two-row MoE exactness depends on the compiler keeping per-row accumulation identical.** M7 met the
  same risk for the dense matvec with a run-time self-check; L2 copies that guard.
- **MTP depth 2:** the head was trained for one step, so the second draft may be accepted rarely. Built
  only on a measured projection.
- **Production sampling:** at temperature 0.7 the draft parent mismatches more often, so production gains
  may be smaller than the greedy bench shows. Stage 0 measures both and the report states both.
