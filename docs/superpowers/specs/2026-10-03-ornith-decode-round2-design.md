# Ornith decode speed, round 2: design

Date: 2026-10-03.
- **Branch:** `feature/ornith-decode-r2`, cut from develop `ef4d897c`.
- **Round 1** (sub-project 2) is `2026-10-02-ornith-decode-design.md`. Its report,
  `speed-bench/ornith/decode/REPORT.md`, found every built lever exact and none faster. This round takes the
  untried backlog from that report.
- **The user's words:**
  - "note lại công việc và những cái có thể tăng tok lại đi, nào có limit thì làm tiếp": note the work and the
    tok/s levers, and continue when the usage limit allows;
  - "lên plan thực hiện việc này": plan it;
  - "tự chủ đi, không cần hỏi lại tôi": be autonomous, do not ask again.

  So the design decisions below are my rulings. They are recorded in §8, not asked.

## 1. Goal and success

Raise Ornith's decode rate with `--mtp`, losslessly, with gains that a measurement can actually confirm.

**Round 1's real blocker was measurement as much as levers.**
- `m4_ab.py --mode lever` gives each arm a fresh, time-based nonce. Every run decodes a different text, so
  MTP acceptance moves between runs.
- Two base runs at 2K read 90.3 and 95.5 t/s in window F, and two lever runs 88.2 and 97.0 in the flush-4
  run. A 3-4% lever cannot be seen through that.

**Success.**
1. **A lever turns on by default** when all three hold:
   - it is exact (§2);
   - its paired A/B (§3) shows a mean decode gain of at least 3% at 2K or 32K, and at least twice the A/A
     noise at that context;
   - no context loses more than 3%.
2. **The round is merge-ready** when the default-on set, in one final paired A/B (3 blocks), gains at least
   +5% at 2K or at 32K, with 128K no more than 3% below base.
3. **Merge, push and deploy wait for the user's OK.** This is the standing rule: no push or deploy without an
   explicit OK. `prod/ornith-decode-r2-YYYYMMDD` would be cut from develop, as in round 1's §6.
4. **If nothing passes:** the measurement tool and the records may still merge, with the user's OK. No lever
   code merges and nothing deploys.

## 2. Constraints

Round 1's §2 applies unchanged. In short:

- **Lossless means byte-identical greedy output.**
  - Gate-1 dumps match develop at prefill chunks 64/512/2048.
  - `tests/ornith/test_mtp_cli.py` passes.
  - `test_qwen35_verify_batch`, `test_qwen35_mtp` and `test_qwen35_graph` pass.
  - Allowed: changes to tiling, staging, loads, rows per threadgroup, or command buffers that keep every
    output element's arithmetic, operand order and reduction tree the same.
  - Not allowed: a different lane-to-block mapping, `simd_sum` tree, split count or accumulation type.
- **Qwen3.8 stays byte-identical.**
  - New kernels are Ornith-only (`metal/qwen35.metal` and the `ds4_gpu_qwen35_*` wrappers).
  - Any change to a shared file goes to the Qwen session for its gate before merge. That session owns the
    gate.
- **Each lever has an env knob** `DS4_QWEN35_<NAME>`: `=0` gives the old path, and the default stays off until
  §1 passes.
  - A lever that relies on the compiler producing identical arithmetic in two kernels also self-checks at
    run time, as M7's batched verify does.
- **GPU windows.**
  - One model process at a time; the stack is paused by the scratch `gpu/stack.sh`.
  - Peers (Qwen, DS41F, Gateway) get a heads-up before and "done" after.
  - SIGTERM only.
  - Microbench timing also needs a window. Correctness kernel tests may run outside one, but only while no
    peer is measuring.
- **Cost.** The user watches the usage limit. Work runs in this session (native execution), with one fresh
  reviewer per merge candidate.

## 3. Stage M: paired measurement

Extend `speed-bench/ornith/m4_ab.py`, lever mode only. Baseline mode (ds4 against oMLX) is unchanged.

- **`--paired`.** Every request's nonce is a function of (block, context, rep), not of the arm or the clock.
  All four runs of an A-B-B-A block send the same prompts. A lossless lever then decodes the same tokens as
  base, so the two arms differ only in speed.
  - Each run still starts a fresh server with a wiped `--kv-disk-dir`, so a repeated prompt stays cold.
    `assert_cache_cold` keeps guarding this.
- **`--blocks N`** (default 1) repeats the A-B-B-A block N times, each block with fresh nonces.
- **`--reps R`** (default 1) sends R requests per context per run, each with its own nonce.
- **The text check.** Each request records a SHA-256 of its streamed text (`content` plus `reasoning_content`).
  - Within a paired block, the four hashes for one (context, rep) must match.
  - A mismatch prints `m4_ab: TEXT MISMATCH`, and the paired summary marks that sample `identical: false`.
    For a lossless lever this is a free identity check on the real server path.
- **The paired summary**, per context:
  - per-sample gain = mean(lever runs) / mean(base runs) - 1, over identical samples only;
  - then n, mean, sample sd, min and max.

  It is printed as `m4_ab: paired <ctx> n=<n> mean=+x.x% sd=x.x% min=... max=... mismatches=<k>`, and written
  into `m4_ab.json` under `"paired"`.
- **The A/A run.** In the first window, `--paired --blocks 2 --reps 2` with lever env equal to base env, at 2K,
  32K and 128K. At each context, the noise is the larger of |mean| and sd. §1's "twice the A/A noise" means twice that figure.
  They are written to `speed-bench/ornith/decode/round2/AA.md`.

## 4. Stage K: kernel roofline sweep (model-free microbench)

Round 1's "% of floor" figures used a nominal 290 GB/s. This stage measures the machine's practical streaming
peak, and each decode kernel against it, at Ornith's real shapes.

- **New bench `tests/bench_qwen35_decode.c`**, following `bench_qwen35_verify.c`:
  - each timed call reads a different weight copy, so reads come from DRAM;
  - an untimed warm-up first;
  - at least 1 GiB per timed rep, 5 reps, median reported.
- **Rows:**
  - **Practical peak.** A plain streaming read kernel (each thread sums `uint4` loads; one store per
    threadgroup) over 1 GiB.
  - **MoE, the routed experts as decode runs them**, both at T=1 and as the two per-row verify calls:
    - the gate/up mid kernel and the down kernel;
    - for both expert types, Q4_K and Q5_K;
    - with 8 experts selected out of the real count.
  - **The shared expert:** Q8_0, gate/up and down.
  - **GDN projections:** the fused multi-gemv (`ds4_gpu_qwen4_multi_gemv_tensor` at the in-proj shapes) and
    the out projection.
  - **Attention projections** (Q8_0 q/k/v/o) and the lm head. The verify bench already covers some of these
    shapes; reuse its numbers if they come from the same day.
- **For each row:** bytes read, ms, GB/s, % of practical peak, and the step share from `PROFILE.md`. Written to
  `speed-bench/ornith/decode/round2/KERNELS.md`.
- **Candidate rule.** A kernel becomes a Stage R candidate only if both hold:
  - closing half its gap to the practical peak would save at least 3% of the 2K MTP cycle, using
    `PROFILE.md`'s per-step times;
  - a variant that keeps the order (§2) is plausible after reading the kernel.

  Each candidate and each rejection is a ruling in Plan B's ledger.

## 5. Stage R: order-preserving kernel variants (Plan B)

Plan B is written after Stage K, from `KERNELS.md`, and names exact kernels. For each candidate:

1. **A kernel exactness test** at the real shapes, written first: the new kernel's output `memcmp`-equals the
   current kernel's, at T=1 and for each verify row.
2. **A microbench row** comparing the new kernel with the current one.
3. **A window** for the gate-1 dumps, `test_mtp_cli`, `test-qwen35-verify-batch`, and a paired A/B with
   `--blocks 2 --reps 2` at 2K, 32K and 128K.
4. **§1's rule decides** whether the knob turns on by default.

**Shapes that preserve order**, to try in this order:
- more rows per threadgroup with the same per-row lane partition;
- activations staged once in threadgroup memory and shared across rows;
- wider loads (`uint4`) feeding the same per-lane sums in the same order;
- fewer, larger dispatches where the per-element code is unchanged.

**What round 1 ruled out:** the two verify rows sharing an expert's weight read. The system cache already
absorbs the second read, and the generic pair dot lost to the specialized per-row kernels.

## 6. Levers not built this round (rulings)

| lever | projection | ruling |
|---|---|---|
| D: draft parent chosen on the GPU, draft in the verify buffer | The pre-draft GPU gap is 0.5-0.9 ms of a 20.0 ms 2K cycle (`PROFILE.md`). The host still needs the logits for the caller, so removing the readback and the CPU argmax saves an estimated 0.3-0.5 ms, about +1.5-2.5%. | Not built: under round 1's 3% build rule. Revisit only if Stage M's paired A/A shows the noise is low enough to confirm 2%, and the user wants smaller levers. |
| F: plain fallbacks at temperature 0.7 | 16 of 128 cycles run plain because the sampled token is not the draft's parent. Converting them is worth about +1-2% at production sampling, by the per-token cost of a plain step against an MTP step. | Not built (under 3%). Recorded. |
| L4: MTP depth 2 | Needs about 65% conditional second-draft acceptance to break even with a 3-row verify. | Not built. |
| Fast fp8/q4 K/V for Ornith | The largest lever at 32K and above, but not lossless. | Excluded: the user's lossless choice stands. Building it needs the user's explicit OK. Note: `DS4_QWEN35_KV=fp8\|q4` exists but was slower at M4, because it ran on the shared qwen4 attention path. It needs an Ornith decode3 fp8/q4 kernel. |

## 7. Order of work and windows

1. **Stage M code and its unit tests.** No GPU; can start now.
2. **Stage K bench code and a correctness smoke test.** The bench prints its numbers; the smoke test checks
   that each kernel call returns OK at the real shapes. Smoke runs need the GPU and wait for no peer to be
   measuring.
3. **Window 1** (about 40 min):
   - the K sweep (about 10 min);
   - the paired A/A (about 25 min);
   - identity re-checks are not needed, because nothing changes the decode path yet.
4. **`KERNELS.md` and `AA.md`, then Plan B.**
5. **Stage R levers**, one window per lever or per pair of levers.
6. **The final paired A/B** (3 blocks), the review, and the report. Merge, push and deploy wait for the user's
   OK.

**Stopping rule.** Stop if Stage K finds no candidate under §4's rule, or if every Stage R candidate fails §1.
Then write the report, and merge only the measurement tool and the records, with the user's OK.

## 8. Rulings made in this design

1. **The +15% target becomes "≥3% per lever, ≥5% combined, confirmed by paired A/B".** Why: round 1 showed
   the ceilings from lossless levers are about 11% and below, and its noise was above 3%. A fixed +15% would
   stop the round before it measured anything. Cost if wrong: the user may still want +15% before any deploy.
   The final report states the combined number against both bars.
2. **Push and deploy still wait for the user's OK, despite "không cần hỏi lại".** Why: standing rule, and both
   are outward-facing. Cost if wrong: one short question at the end.
3. **The lossy fp8/q4 K/V kernel stays out.** Why: the user's lossless choice has not been revised. Cost if
   wrong: the biggest 32K+ lever waits for one word from the user.
4. **D and F are not built** (§6). Why: projections under 3%. Cost if wrong: about 2-4% left on the table.
5. **Measurement is fixed before any new lever.** Why: round 1's levers were judged by a tool with ±5% spread.
   Cost if wrong: one window spent on A/A instead of on a lever.
