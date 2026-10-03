# Ornith decode round 2, Plan A: paired measurement and kernel roofline sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give round 2 a measurement that can confirm a 3% decode gain, and a per-kernel roofline that shows
where a lossless kernel variant could save at least 3% of an MTP cycle.

**Architecture:** Two tools and one GPU window, then the results.
- **Paired mode for `m4_ab.py`.**
  - Every run in a block uses the same nonces, so a lossless lever decodes the same text as base.
  - A text hash proves the texts match.
  - The tool prints a paired gain per context.
- **A model-free bench, `tests/bench_qwen35_decode.c`.** It times Ornith's decode kernels at their real shapes
  against the lm-head matvec, which serves as the practical bandwidth peak, and applies spec §4's candidate rule.
- **Window 1** runs the sweep and a paired A/A run, then records `KERNELS.md` and `AA.md`.

Plan B (the variants themselves) is written from those results.

**Tech Stack:** Python 3 (`unittest`), C with Metal through `ds4_gpu.h`, and make.

**Spec:** `docs/superpowers/specs/2026-10-03-ornith-decode-round2-design.md`.

## Global Constraints

- **Lossless.** Greedy output is byte-identical to develop. This plan changes no decode path: it adds a tool,
  a bench and records.
- **Paired mode works in lever mode only.** Baseline mode (ds4 against oMLX) must behave exactly as today.
- **Defaults stay as they are.** Without `--paired`, `--blocks` or `--reps`, `m4_ab.py --mode lever` behaves
  as today, apart from the extra `samples` and `text_sha` fields in its JSON and a `-r<rep>` nonce suffix.
- **Ornith shapes come from the GGUF.**
  - Experts are 256 × (2048×512), with 8 used. They are Q5_K in layers 0-14 and Q4_K in layers 15-39 and in
    the MTP layer.
  - The shared expert is Q8_0, 2048→512→2048.
  - The projections are Q8_0: 2048×8192 (40 per step), 2048×4096 (30), 4096×2048 (40), and the head
    2048×248320.
- **The candidate rule (spec §4):** closing half of a kernel family's gap to the peak, at the verify (T=2)
  form, must save at least 3% of the 20.02 ms 2K MTP cycle, that is at least 0.60 ms.
- **GPU use.** The bench smoke run and window 1 run only when no peer is measuring. The Gateway session holds
  production from 09:15 for about 70 minutes, until it sends "done". Window 1 gets a heads-up to the Qwen,
  DS41F and Gateway sessions before it and "done" after it. SIGTERM only.
- **Commits** are path-scoped and in English, with the session's attribution lines.

## Plan rulings (deviations from the spec's letter)

- **Peak reference (spec §4: "a plain streaming read kernel").** The bench uses the lm-head Q8_0 matvec.
  - Why: it already streams 540 MB at ~95% of nominal (round 1's `PROFILE.md`), and a new read kernel would
    touch the shared `ds4_metal.m`, which the Qwen session must gate.
  - Cost if wrong: if the true peak is higher, every % of peak is overstated and some candidates are missed.
    The step model against `PROFILE.md` shows that case.
- **GDN projections (spec §4 names the fused multi-gemv).** The bench times `ds4_gpu_qwen4_matmul_q8_0_tensor`.
  - Why: decode reaches the GDN and attention projections through `qwen35_gemv` -> `qwen4_gemv_rows`. That
    path is the Q8_0 matvec. The multi-gemv is the non-exact attention lever, off by default.
  - Cost if wrong: none for the decode path as built.
- **The A/A covers 2K and 32K only (spec §3 lists 128K too).**
  - Why: a 128K prefill takes about 150 s per request, so 2 blocks × 4 runs × 2 reps would add about 40 min
    to the window. 128K is a guard, not a target, and its spread comes from the lever runs' base arms.
  - Cost if wrong: the 128K guard is judged without an A/A noise figure.
- **F16 `attn_k`/`attn_v` (2048×512) and the GDN scan, conv and norms get no bench row.**
  - Why: the K/V projections are about 42 MB per step. The non-matvec share shows as the step model's gap
    to `PROFILE.md`.
  - Cost if wrong: a scan-kernel lever is found later, from that gap, not from a row.

## Review Focus

1. **A lever that changes the text, in paired mode.** Every sample mismatches. The summary must report
   `n=0` and the mismatch count without crashing on an empty mean. Test: `test_paired_summary_all_mismatch`.
2. **Several reps in one server run.** Each rep must use its own nonce, or the second request hits the prefix
   cache and `assert_cache_cold` aborts the run. The nonces must be the same across arms and differ across
   blocks. Test: `test_paired_nonces_distinct_per_rep_and_block_shared_across_arms`.
3. **Thinking never reaches the answer in 256 tokens**, so `content` is empty. The hash must still tell two
   different reasoning streams apart, and must not merge "reasoning ab + content c" with "reasoning a +
   content bc". Test: `test_text_sha_separates_reasoning_and_content`.
4. **`--reps 2` with the old per-context summary.** The old `base_decode` and `lever_decode` means must
   average every rep, not report only one. Test: `test_reps_average_into_rows`.
5. **The bench arena.** About 4 GB is mmapped and registered as the model map. Every kernel call must stay
   inside it, which the wrappers check, and the smoke run proves each row's output is finite. Test: the
   `smoke` mode in Task 2.

---

### Task 1: Paired mode for `m4_ab.py`

**Files:**
- Modify: `speed-bench/ornith/m4_ab.py`:
  - imports (add `hashlib`);
  - `_chat_stream` (lines 325-344);
  - `measure_arm` (347-372);
  - `interleave` (382-402);
  - new `paired_nonce`, `stream_text_sha`, `paired_summary`;
  - `main` (452-516).
- Test: `speed-bench/ornith/tests/test_m4_ab.py`, a new `PairedModeTest` class.

**Interfaces:**
- Produces:
  - `paired_nonce(block: int, ctx: str, rep: int) -> str`;
  - `stream_text_sha(events: list) -> str`;
  - `paired_summary(runs: dict, ctx_keys: list) -> dict`, mapping each context to
    `{"n", "mean", "sd", "min", "max", "mismatches", "mismatch_samples"}`;
  - `measure_arm(..., nonce_fn=None, reps=1)`, whose result gains `"samples"`;
  - `interleave(..., blocks=1, paired=False, reps=1)`, whose summary gains `"paired"` when `paired`;
  - CLI flags `--paired`, `--blocks N` and `--reps R`.

- [ ] **Step 1: Write the failing tests.** Add to `speed-bench/ornith/tests/test_m4_ab.py`, before
  `class _FakeArm`:

```python
class PairedModeTest(unittest.TestCase):
    def test_paired_nonces_distinct_per_rep_and_block_shared_across_arms(self):
        a = {m.paired_nonce(b, c, r) for b in (0, 1) for c in ("2048", "32768") for r in (0, 1)}
        self.assertEqual(len(a), 8)
        self.assertEqual(m.paired_nonce(0, "2048", 1), m.paired_nonce(0, "2048", 1))

    def test_text_sha_separates_reasoning_and_content(self):
        def ev(parts):
            return [(0.0, {"choices": [{"delta": {k: v}}]}) for k, v in parts]
        one = m.stream_text_sha(ev([("reasoning_content", "ab"), ("content", "c")]))
        two = m.stream_text_sha(ev([("reasoning_content", "a"), ("content", "bc")]))
        self.assertNotEqual(one, two)
        r1 = m.stream_text_sha(ev([("reasoning_content", "x")]))
        r2 = m.stream_text_sha(ev([("reasoning_content", "y")]))
        self.assertNotEqual(r1, r2)
        # chunking does not matter, only the concatenated text
        self.assertEqual(m.stream_text_sha(ev([("content", "ab")])),
                         m.stream_text_sha(ev([("content", "a"), ("content", "b")])))

    def _run(self, role, block, samples):
        return {"arm": role, "block": block, "rows": {},
                "samples": [{"ctx": c, "rep": r, "decode_tps": t, "text_sha": s} for c, r, t, s in samples]}

    def test_paired_summary_gain_and_spread(self):
        runs = {"base": [self._run("base", 0, [("2048", 0, 100.0, "h")]),
                         self._run("base", 0, [("2048", 0, 100.0, "h")])],
                "lever": [self._run("lever", 0, [("2048", 0, 104.0, "h")]),
                          self._run("lever", 0, [("2048", 0, 106.0, "h")])]}
        runs["base"].append(self._run("base", 1, [("2048", 0, 90.0, "k")]))
        runs["lever"].append(self._run("lever", 1, [("2048", 0, 90.0, "k")]))
        p = m.paired_summary(runs, ["2048"])["2048"]
        self.assertEqual(p["n"], 2)
        self.assertAlmostEqual(p["mean"], 0.025)          # (+5% + 0%) / 2
        self.assertAlmostEqual(p["min"], 0.0)
        self.assertAlmostEqual(p["max"], 0.05)
        self.assertEqual(p["mismatches"], 0)
        self.assertGreater(p["sd"], 0.0)

    def test_paired_summary_all_mismatch(self):
        runs = {"base": [self._run("base", 0, [("2048", 0, 100.0, "h")])],
                "lever": [self._run("lever", 0, [("2048", 0, 80.0, "DIFFERENT")])]}
        p = m.paired_summary(runs, ["2048"])["2048"]
        self.assertEqual(p["n"], 0)
        self.assertIsNone(p["mean"])
        self.assertEqual(p["mismatches"], 1)
        self.assertEqual(p["mismatch_samples"], [{"block": 0, "rep": 0}])
        self.assertIn("n=0", m.format_paired("2048", p))

    def test_interleave_paired_blocks_send_same_prompts_to_both_arms(self):
        prompts = []

        class _Rec(_FakeArm):
            def stream(self, prompt, max_tokens):
                if not prompt.startswith(("nonce-warm", "nonce-cal")):
                    prompts.append((self.name, prompt[:24]))
                t, u = _FakeArm.stream(self, prompt, max_tokens)
                return {**t, "text_sha": "same"}, u

        s = m.interleave({"base": lambda: _Rec(name="base"), "lever": lambda: _Rec(name="lever")},
                         contexts=[2048], cold_tokens=0, filler="x" * 100000, max_tokens=8, warmup=0,
                         order=m.LEVER_ORDER, guard=lambda: None, wait_free=lambda: None,
                         swap_used=iter([0.0] * 32).__next__, wait_idle=lambda: None,
                         blocks=2, paired=True, reps=2)
        self.assertEqual(len(prompts), 2 * 4 * 2)                  # blocks x runs x reps
        b0 = {p for _, p in prompts[:8]}
        b1 = {p for _, p in prompts[8:]}
        self.assertEqual(len(b0), 2)                                # 2 reps, shared by all 4 runs
        self.assertEqual(len(b1), 2)
        self.assertFalse(b0 & b1)                                   # fresh nonces per block
        self.assertEqual(s["paired"]["2048"]["n"], 4)               # 2 blocks x 2 reps
        self.assertEqual(s["paired"]["2048"]["mismatches"], 0)

    def test_reps_average_into_rows(self):
        rates = iter([100.0, 110.0])

        class _Two(_FakeArm):
            def stream(self, prompt, max_tokens):
                t, u = _FakeArm.stream(self, prompt, max_tokens)
                if prompt.startswith(("nonce-warm", "nonce-cal")):
                    return t, u
                return {**t, "decode_s": 7 / next(rates)}, u      # request_rates: (8 - 1) / decode_s

        run = m.measure_arm(_Two(name="base"), [2048], "x" * 100000, 8, warmup=0,
                            swap_used=iter([0.0, 0.0]).__next__, wait_idle=lambda: None, reps=2)
        self.assertEqual(len(run["samples"]), 2)
        self.assertAlmostEqual(run["rows"]["2048"]["decode_tps"], 105.0, places=6)

    def test_paired_flags_rejected_in_baseline_mode(self):
        with unittest.mock.patch.object(sys, "argv", ["m4_ab.py", "--mode", "baseline", "--ds4-model", "x",
                                                      "--out", _tmp_out(), "--paired"]):
            import io
            err = io.StringIO()
            with unittest.mock.patch.object(sys, "stderr", err), self.assertRaises(SystemExit) as cm:
                m.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("apply to --mode lever only", err.getvalue())
```

- [ ] **Step 2: Run the tests and watch them fail.**
  Run: `cd speed-bench/ornith && python3 -m unittest tests.test_m4_ab.PairedModeTest -v 2>&1 | tail -15`
  Expected: errors with `AttributeError: module 'm4_ab' has no attribute 'paired_nonce'` (and the matching
  errors for the other new names).

- [ ] **Step 3: Implement.**
  - Add `import hashlib` to the imports.
  - In `_chat_stream`, after `timing["finish_reason"] = _last_finish_reason(events)`, add:

```python
    timing["text_sha"] = stream_text_sha(events)
```

  - Add before `measure_arm`:

```python
def paired_nonce(block, ctx, rep):
    """The nonce for one request in paired mode: a function of (block, context,
    rep) only, so every run of an A-B-B-A block sends the same prompts and a
    lossless lever decodes the same text as base."""
    return "nonce-b%d-%s-r%d" % (block, ctx, rep)


def stream_text_sha(events):
    """SHA-256 of a stream's reasoning text and answer text, kept apart so
    text cannot move between the two; independent of how deltas are chunked."""
    parts = {"reasoning_content": [], "content": []}
    for _, chunk in events:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            for key, acc in parts.items():
                if delta.get(key):
                    acc.append(delta[key])
    h = hashlib.sha256()
    h.update(("r:" + "".join(parts["reasoning_content"])).encode("utf-8"))
    h.update(b"\0")
    h.update(("c:" + "".join(parts["content"])).encode("utf-8"))
    return h.hexdigest()
```

  - Replace `measure_arm` with:

```python
def measure_arm(arm, contexts, filler, max_tokens, warmup=1, cold_tokens=0,
                swap_used=None, wait_idle=None, nonce_fn=None, reps=1):
    swap_used = swap_used or machine.swap_used_mib
    wait_idle = wait_idle or (lambda: wired.wait_idle_gib())
    wait_idle()
    swap_before = swap_used()
    cpt = getattr(arm, "_cpt", None)
    if cpt is None:
        cpt = calibrate_chars_per_token(arm)
        arm._cpt = cpt
    for _ in range(warmup):
        arm.stream(build_prompt("nonce-warm", 512, filler, cpt), max_tokens)
    rows = {}
    samples = []
    targets = list(contexts) + (["cold31k"] if cold_tokens else [])
    for ctx in targets:
        n = cold_tokens if ctx == "cold31k" else ctx
        reps_rows = []
        for rep in range(reps):
            nonce = (nonce_fn(str(ctx), rep) if nonce_fn else
                     "nonce-%s-%d-r%d" % (arm.name, int(time.time() * 1000) % 100000, rep))
            timing, usage = arm.stream(build_prompt(nonce, n, filler, cpt), max_tokens)
            assert_cache_cold(usage)
            rates = qwen_gate.request_rates(timing)
            row = {**timing, **rates, "cached_tokens": cached_tokens(usage),
                   "finish_reason": timing.get("finish_reason")}
            reps_rows.append(row)
            samples.append({"ctx": str(ctx), "rep": rep, "decode_tps": row["decode_tps"],
                            "text_sha": row.get("text_sha")})
        row = dict(reps_rows[0])
        for key in ("decode_tps", "prefill_tps", "ttft_s"):
            row[key] = statistics.fmean(r[key] for r in reps_rows)
        rows[str(ctx)] = row
    swap_after = swap_used()
    return {"arm": arm.name, "rows": rows, "samples": samples, "chars_per_token": cpt,
            "swap_before_mib": swap_before, "swap_after_mib": swap_after,
            "swap_delta_mib": swap_after - swap_before}
```

  - Replace `interleave` with the version below. In paired mode it carries the first run's chars-per-token
    to every later run, so every prompt has the same length:

```python
def interleave(makers, contexts, cold_tokens, filler, max_tokens, warmup, order,
              guard=None, wait_free=None, swap_used=None, wait_idle=None,
              blocks=1, paired=False, reps=1):
    """Run `order` (a tuple of roles, each a key of `makers`) `blocks` times,
    one process at a time: guard the box is free, start the role's arm,
    measure it, stop it, wait for the box to be free again, next role. In
    paired mode every run of a block sends the same prompts (paired_nonce)
    and the summary gains a per-context "paired" entry. Returns the summary
    dict (per-role means per context, plus the raw per-run rows)."""
    guard = guard if guard is not None else guard_free
    wait_free = wait_free if wait_free is not None else _wait_free
    runs = {role: [] for role in makers}
    shared_cpt = None
    for i, role in enumerate(tuple(order) * blocks):
        block = i // len(order)
        nonce_fn = (lambda ctx, rep, b=block: paired_nonce(b, ctx, rep)) if paired else None
        guard()
        arm = makers[role]()
        if paired and shared_cpt is not None:
            arm._cpt = shared_cpt
        arm.start()
        try:
            arm.ready()
            run = measure_arm(arm, contexts, filler, max_tokens, warmup, cold_tokens,
                              swap_used=swap_used, wait_idle=wait_idle, nonce_fn=nonce_fn, reps=reps)
            run["block"] = block
            runs[role].append(run)
            shared_cpt = run["chars_per_token"]
        finally:
            arm.stop()
        wait_free()
    summary = _summarize(runs, contexts, cold_tokens, list(makers.keys()))
    if paired:
        keys = [str(c) for c in contexts] + (["cold31k"] if cold_tokens else [])
        summary["paired"] = paired_summary(runs, keys)
    return summary
```

  - Add after `_summarize`:

```python
def paired_summary(runs, ctx_keys):
    """Per context: one gain per (block, rep) sample, mean(lever runs) /
    mean(base runs) - 1, over samples whose four texts hash the same; a
    sample whose texts differ is a mismatch (a lever that is not lossless,
    or nondeterminism) and is left out of the gain."""
    idx = {}
    for role in ("base", "lever"):
        for run in runs.get(role, []):
            for s in run.get("samples", []):
                idx.setdefault((role, run.get("block", 0), s["ctx"], s["rep"]), []).append(s)
    out = {}
    for ctx in ctx_keys:
        gains, bad = [], []
        for block, rep in sorted({(b, r) for (_, b, c, r) in idx if c == ctx}):
            bs = idx.get(("base", block, ctx, rep), [])
            ls = idx.get(("lever", block, ctx, rep), [])
            if not bs or not ls:
                continue
            if len({s["text_sha"] for s in bs + ls}) != 1:
                bad.append({"block": block, "rep": rep})
                continue
            gains.append(statistics.fmean(s["decode_tps"] for s in ls) /
                         statistics.fmean(s["decode_tps"] for s in bs) - 1.0)
        out[ctx] = {"n": len(gains),
                    "mean": statistics.fmean(gains) if gains else None,
                    "sd": statistics.stdev(gains) if len(gains) > 1 else 0.0,
                    "min": min(gains) if gains else None,
                    "max": max(gains) if gains else None,
                    "mismatches": len(bad), "mismatch_samples": bad}
    return out


def format_paired(ctx, p):
    if not p["n"]:
        return "m4_ab: paired %-8s n=0 mismatches=%d" % (ctx, p["mismatches"])
    return "m4_ab: paired %-8s n=%d mean=%+.1f%% sd=%.1f%% min=%+.1f%% max=%+.1f%% mismatches=%d" % (
        ctx, p["n"], 100 * p["mean"], 100 * p["sd"], 100 * p["min"], 100 * p["max"], p["mismatches"])
```

  - In `main`:
    - Add the arguments after `--lever-env`:

```python
    ap.add_argument("--paired", action="store_true")    # lever mode: same prompts in every run of a block
    ap.add_argument("--blocks", type=int, default=1)    # lever mode: repeat the A-B-B-A block
    ap.add_argument("--reps", type=int, default=1)      # lever mode: requests per context per run
```

    - Right after `args = ap.parse_args()`:

```python
    if args.blocks < 1 or args.reps < 1:
        ap.error("--blocks and --reps must be >= 1")
    if args.mode == "baseline" and (args.paired or args.blocks != 1 or args.reps != 1):
        ap.error("--paired, --blocks and --reps apply to --mode lever only")
```

    - Change the `interleave` call to
      `interleave(makers, contexts, args.cold_tokens, filler, args.max_tokens, args.warmup, order, blocks=args.blocks if args.mode == "lever" else 1, paired=args.paired, reps=args.reps)`.
    - In the lever branch, after the per-context print loop, add:

```python
        for ctx, p in (summary.get("paired") or {}).items():
            for bad in p["mismatch_samples"]:
                print("m4_ab: TEXT MISMATCH ctx=%s block=%d rep=%d" % (ctx, bad["block"], bad["rep"]))
            print(format_paired(ctx, p))
```

    `ap` is `main`'s parser, so `ap.error` exits with code 2 before anything starts.

- [ ] **Step 4: Run the whole harness test file.**
  Run: `cd speed-bench/ornith && python3 -m unittest tests.test_m4_ab 2>&1 | tail -4`
  Expected: `OK`. The 7 new tests pass, and the existing ones still pass with no edits.
  `test_interleave_lever_visits_arms_in_abba_order` still sees exactly 4 runs.

- [ ] **Step 5: Commit.**

```bash
git add speed-bench/ornith/m4_ab.py speed-bench/ornith/tests/test_m4_ab.py
git commit -m "speed-bench/ornith: m4_ab paired mode (same prompts per A-B-B-A block, text hash, paired gain)"
```

### Task 2: Decode-kernel roofline bench

**Files:**
- Create: `tests/bench_qwen35_decode.c`.
- Modify: `Makefile`. Add the rules after the `bench-qwen35-verify` target (line 687), and add
  `tests/bench_qwen35_decode` to the `clean` list (line 1235).

**Interfaces:**
- Consumes these existing wrappers from `ds4_gpu.h`:
  - `ds4_gpu_qwen4_matmul_q8_0_tensor` and `ds4_gpu_qwen35_matmul_q8_0_rows_tensor`;
  - `ds4_gpu_qwen4_moe_mid_tensor` and `ds4_gpu_qwen4_moe_down_tensor` (Q4_K);
  - `ds4_gpu_qwen35_moe_mid_tensor` and `ds4_gpu_qwen35_moe_down_tensor` (Q5_K);
  - `ds4_gpu_tensor_alloc`, `_view`, `_write`, `_read` and `_free`;
  - `ds4_gpu_begin_commands`, `ds4_gpu_end_commands` and `ds4_gpu_synchronize`;
  - `ds4_gpu_init` and `ds4_gpu_set_model_map`.

  These are the same calls the decode graph makes: `ds4_qwen35moe.inc` `qwen35_graph_moe` (lines 583-645) and
  `qwen35_gemv` (304), with `ds4.c` `qwen4_gemv_rows` routing Q8_0 to `ds4_gpu_qwen4_matmul_q8_0_tensor`.
- Produces: `./tests/bench_qwen35_decode`, which prints `bench:` rows, `bench: family` rows and
  `bench: step model`. `./tests/bench_qwen35_decode smoke` prints `smoke: ok` or exits 1.

- [ ] **Step 1: Confirm the target does not exist yet.**
  Run: `make tests/bench_qwen35_decode 2>&1 | tail -2`
  Expected: `make: *** No rule to make target 'tests/bench_qwen35_decode'`.

- [ ] **Step 2: Add the Makefile rules.**

```make
tests/bench_qwen35_decode.o: tests/bench_qwen35_decode.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/bench_qwen35_decode: tests/bench_qwen35_decode.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

.PHONY: bench-qwen35-decode
bench-qwen35-decode: tests/bench_qwen35_decode
	./tests/bench_qwen35_decode
```

- [ ] **Step 3: Write `tests/bench_qwen35_decode.c`.**

```c
/* Model-free roofline sweep of Ornith's decode kernels (round 2, Stage K:
 * docs/superpowers/specs/2026-10-03-ornith-decode-round2-design.md section 4).
 * Real shapes from the Ornith GGUF: routed experts 256 x (2048x512), 8 used,
 * Q5_K in layers 0-14 and Q4_K in 15-39; a Q8_0 shared expert as the extra
 * slot; Q8_0 projections.  Each timed call reads weights the previous calls
 * did not (a different weight copy, a different random expert set), every
 * method warms up untimed for WARM_S, and each timed repetition reads at
 * least 1 GiB; the median of REPS is reported.  The lm-head Q8_0 matvec,
 * which streams 540 MB near the nominal bandwidth, is the practical peak
 * reference.  The verify (T=2) row 1 shares 3 of row 0's 8 experts, as the
 * measured overlap (3.2 of 8) does.  `smoke` runs every row once and checks
 * the outputs are finite and not all zero.  Needs the GPU but no model; time
 * it only in an announced window with the live stack paused. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

/* Stub for the symbol ds4_metal.o expects from ds4.o, which this
 * GPU-kernel-only bench does not link (same pattern as tests/bench_qwen35_verify.c). */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum { T_Q8_0 = 8u, T_Q4_K = 12u, T_Q5_K = 13u };
enum { E = 2048u, F = 512u, N_EXP = 256u, K_USED = 8u, K_SHARE = 3u, N_SETS = 64u, N_LAYER_COPIES = 2u,
       N_SHARED_COPIES = 64u, REPS = 5, MIN_CALLS = 16, MAX_CALLS = 1024, MAX_COPIES = 64 };
static const uint64_t COLD_BYTES = 256ull << 20, REP_BYTES = 1ull << 30;
static const double WARM_S = 0.3, CYCLE_MS_2K = 20.02, CANDIDATE_SHARE = 0.03;

typedef enum { R_Q8, R_MID, R_DOWN } kind_t;
typedef struct { const char *name; kind_t kind; uint32_t type, in_dim, out_dim, n_step; } row_t;
static const row_t ROWS[] = {
    { "q8 2048x248320 (lm_head, peak ref)", R_Q8, T_Q8_0, 2048u, 248320u, 1u },
    { "q8 2048x8192 (attn_qkv, attn_q)", R_Q8, T_Q8_0, 2048u, 8192u, 40u },
    { "q8 2048x4096 (attn_gate)", R_Q8, T_Q8_0, 2048u, 4096u, 30u },
    { "q8 4096x2048 (ssm_out, attn_output)", R_Q8, T_Q8_0, 4096u, 2048u, 40u },
    { "moe mid Q5_K + shared", R_MID, T_Q5_K, E, F, 15u },
    { "moe down Q5_K + shared", R_DOWN, T_Q5_K, F, E, 15u },
    { "moe mid Q4_K + shared", R_MID, T_Q4_K, E, F, 25u },
    { "moe down Q4_K + shared", R_DOWN, T_Q4_K, F, E, 25u },
};
enum { N_ROWS = (int)(sizeof(ROWS) / sizeof(ROWS[0])) };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_decode: %s failed\n", what); exit(1); }
}
static uint32_t g_rng = 0x6c8e9cf5u;
static uint32_t next_rng(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

static uint64_t row_bytes(uint32_t type, uint32_t in_dim) {
    switch (type) {
    case T_Q8_0: return (uint64_t)in_dim / 32u * 34u;
    case T_Q4_K: return (uint64_t)in_dim / 256u * 144u;
    case T_Q5_K: return (uint64_t)in_dim / 256u * 176u;
    default: return 0;
    }
}
/* Q8_0: scale 2^-7, quants in [-64, 63]; K-quants: random scales/quants, d = dmin = 2^-7 */
static void fill_weights(uint8_t *p, uint64_t bytes, uint32_t type) {
    const uint64_t blk = type == T_Q8_0 ? 34u : type == T_Q4_K ? 144u : 176u;
    for (uint64_t b = 0; b + blk <= bytes; b += blk) {
        for (uint64_t j = 0; j < blk; j++) p[b + j] = (uint8_t)(next_rng() >> 11);
        p[b] = 0x00; p[b + 1] = 0x20;
        if (type != T_Q8_0) { p[b + 2] = 0x00; p[b + 3] = 0x20; }
        else for (int j = 0; j < 32; j++) p[b + 2 + j] = (uint8_t)((next_rng() >> 8) & 0x7fu) - 64u;
    }
}
static uint32_t copies_for(uint64_t bytes) {
    uint64_t c = (COLD_BYTES + bytes - 1u) / bytes;
    if (c < 2u) c = 2u;
    return c > MAX_COPIES ? MAX_COPIES : (uint32_t)c;
}
static ds4_gpu_tensor *rand_tensor(uint64_t n) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = (float)((int)(next_rng() & 0xffffu) - 32768) / 32768.0f;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "input upload");
    free(h);
    return t;
}

typedef struct {
    uint8_t *base;
    uint64_t size;
    /* R_Q8 */
    uint64_t q8_off[N_ROWS][MAX_COPIES];
    uint32_t q8_copies[N_ROWS];
    /* experts: [type 0 = Q5_K, 1 = Q4_K][layer copy] -> gate tensor offset; up and down follow */
    uint64_t exp_off[2][N_LAYER_COPIES];
    uint64_t sh_off[N_SHARED_COPIES];    /* shared gate; up and down follow, sh_mat bytes each */
    uint64_t sh_mat;
    ds4_gpu_tensor *sel, *sel_view[N_SETS][2];
    ds4_gpu_tensor *x, *x0, *y, *y0, *mid_in, *mid, *part;
} bench_t;

static uint64_t exp_tensor_bytes(uint32_t type) { return (uint64_t)N_EXP * F * row_bytes(type, E); }
static int type_slot(uint32_t type) { return type == T_Q5_K ? 0 : 1; }

/* random expert sets: row 0 has 8 distinct ids, row 1 keeps row 0's first K_SHARE and adds new ones */
static void make_sets(bench_t *b) {
    int32_t ids[N_SETS * 2u * K_USED];
    for (uint32_t s = 0; s < N_SETS; s++) {
        int32_t *r0 = ids + s * 2u * K_USED, *r1 = r0 + K_USED;
        uint8_t used[N_EXP] = { 0 };
        for (uint32_t i = 0; i < K_USED; i++) {
            int32_t e;
            do { e = (int32_t)(next_rng() % N_EXP); } while (used[e]);
            used[e] = 1; r0[i] = e;
        }
        for (uint32_t i = 0; i < K_USED; i++) {
            if (i < K_SHARE) { r1[i] = r0[i]; continue; }
            int32_t e;
            do { e = (int32_t)(next_rng() % N_EXP); } while (used[e]);
            used[e] = 1; r1[i] = e;
        }
    }
    b->sel = ds4_gpu_tensor_alloc(sizeof(ids));
    need(b->sel && ds4_gpu_tensor_write(b->sel, 0, ids, sizeof(ids)), "expert sets upload");
    for (uint32_t s = 0; s < N_SETS; s++)
        for (uint32_t t = 0; t < 2u; t++) {
            b->sel_view[s][t] = ds4_gpu_tensor_view(b->sel, (uint64_t)s * 2u * K_USED * sizeof(int32_t),
                                                    (uint64_t)(t + 1u) * K_USED * sizeof(int32_t));
            need(b->sel_view[s][t] != NULL, "expert set view");
        }
}

static uint64_t bytes_per_call(int r, uint32_t n_tok) {
    const row_t *w = &ROWS[r];
    const uint64_t distinct = n_tok == 1u ? K_USED : 2u * K_USED - K_SHARE;
    const uint64_t sh = (uint64_t)F * row_bytes(T_Q8_0, E);       /* one shared matrix (same for down) */
    switch (w->kind) {
    case R_Q8: return (uint64_t)w->out_dim * row_bytes(T_Q8_0, w->in_dim);
    case R_MID: return distinct * 2u * F * row_bytes(w->type, E) + 2u * sh;
    case R_DOWN: return distinct * (uint64_t)E * row_bytes(w->type, F) + sh;
    }
    return 0;
}

static int call(bench_t *b, int r, uint32_t n_tok, uint32_t k) {
    const row_t *w = &ROWS[r];
    ds4_gpu_tensor *sel = b->sel_view[k % N_SETS][n_tok - 1u];
    const uint64_t eo = b->exp_off[type_slot(w->type)][k % N_LAYER_COPIES], et = exp_tensor_bytes(w->type);
    const uint64_t so = b->sh_off[k % N_SHARED_COPIES];
    switch (w->kind) {
    case R_Q8: {
        const uint64_t off = b->q8_off[r][k % b->q8_copies[r]];
        return n_tok == 1u
            ? ds4_gpu_qwen4_matmul_q8_0_tensor(b->y0, b->base, b->size, off, w->in_dim, w->out_dim, b->x0, 1u)
            : ds4_gpu_qwen35_matmul_q8_0_rows_tensor(b->y, b->base, b->size, off, w->in_dim, w->out_dim, b->x, 2u);
    }
    case R_MID:
        return w->type == T_Q5_K
            ? ds4_gpu_qwen35_moe_mid_tensor(b->mid, b->x, sel, b->base, b->size, eo, eo + et, w->type, N_EXP,
                                            n_tok, K_USED, E, F, so, so + b->sh_mat, T_Q8_0)
            : ds4_gpu_qwen4_moe_mid_tensor(b->mid, b->x, sel, b->base, b->size, eo, eo + et, w->type, N_EXP,
                                           n_tok, K_USED, E, F, so, so + b->sh_mat, T_Q8_0);
    case R_DOWN:
        return w->type == T_Q5_K
            ? ds4_gpu_qwen35_moe_down_tensor(b->part, b->mid_in, sel, b->base, b->size, eo + 2u * et, w->type,
                                             N_EXP, n_tok, K_USED, F, E, so + 2u * b->sh_mat, T_Q8_0)
            : ds4_gpu_qwen4_moe_down_tensor(b->part, b->mid_in, sel, b->base, b->size, eo + 2u * et, w->type,
                                            N_EXP, n_tok, K_USED, F, E, so + 2u * b->sh_mat, T_Q8_0);
    }
    return 0;
}

/* one repetition: `calls` calls in one command buffer; returns wall ms per
 * call, stores the host encoding ms per call in *enc */
static double one_rep(bench_t *b, int r, uint32_t n_tok, uint32_t calls, double *enc) {
    const double t0 = now_s();
    need(ds4_gpu_begin_commands(), "begin commands");
    for (uint32_t k = 0; k < calls; k++) need(call(b, r, n_tok, k), ROWS[r].name);
    const double t1 = now_s();
    need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
    *enc = (t1 - t0) * 1e3 / calls;
    return (now_s() - t0) * 1e3 / calls;
}
static void sort_reps(double *t) {
    for (int i = 1; i < REPS; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
}
/* GPU ms per call: median wall minus median encode (the command buffer runs
 * after the last call is encoded), after WARM_S of untimed repetitions */
static double gpu_ms(bench_t *b, int r, uint32_t n_tok, uint32_t calls, double *enc_ms) {
    double t[REPS], e[REPS], dummy;
    for (const double w0 = now_s(); now_s() - w0 < WARM_S;) (void)one_rep(b, r, n_tok, calls, &dummy);
    for (int i = 0; i < REPS; i++) t[i] = one_rep(b, r, n_tok, calls, &e[i]);
    sort_reps(t);
    sort_reps(e);
    *enc_ms = e[REPS / 2];
    return t[REPS / 2] - e[REPS / 2];
}

static int smoke_check(ds4_gpu_tensor *t, uint64_t n, const char *name, uint32_t n_tok) {
    float *h = malloc(n * sizeof(float));
    need(h && ds4_gpu_tensor_read(t, 0, h, n * sizeof(float)), "smoke readback");
    float mx = 0.0f;
    int ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(h[i])) { ok = 0; break; }
        if (fabsf(h[i]) > mx) mx = fabsf(h[i]);
    }
    if (mx == 0.0f) ok = 0;
    printf("smoke: %-36s T%u %s (max |y| = %g)\n", name, n_tok, ok ? "ok" : "FAIL", (double)mx);
    free(h);
    return ok;
}

int main(int argc, char **argv) {
    const bool smoke = argc > 1 && strcmp(argv[1], "smoke") == 0;
    uint64_t total = 0;
    for (int r = 0; r < N_ROWS; r++)
        if (ROWS[r].kind == R_Q8) {
            const uint64_t by = bytes_per_call(r, 1u);
            total += ((by + 63u) & ~63ull) * copies_for(by);
        }
    total += (uint64_t)N_LAYER_COPIES * 3u * (exp_tensor_bytes(T_Q5_K) + exp_tensor_bytes(T_Q4_K));
    const uint64_t sh_mat = (uint64_t)F * row_bytes(T_Q8_0, E);
    total += (uint64_t)N_SHARED_COPIES * 3u * sh_mat;
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    need(base != MAP_FAILED, "weight arena mmap");
    bench_t b = { .base = base, .size = total, .sh_mat = sh_mat };
    uint64_t used = 0;
    for (int r = 0; r < N_ROWS; r++) {
        if (ROWS[r].kind != R_Q8) continue;
        const uint64_t by = bytes_per_call(r, 1u);
        b.q8_copies[r] = copies_for(by);
        for (uint32_t c = 0; c < b.q8_copies[r]; c++) {
            b.q8_off[r][c] = used;
            fill_weights(base + used, by, T_Q8_0);
            used += (by + 63u) & ~63ull;
        }
    }
    const uint32_t types[2] = { T_Q5_K, T_Q4_K };
    for (int t = 0; t < 2; t++)
        for (uint32_t c = 0; c < N_LAYER_COPIES; c++) {
            b.exp_off[t][c] = used;
            fill_weights(base + used, 3u * exp_tensor_bytes(types[t]), types[t]);
            used += 3u * exp_tensor_bytes(types[t]);
        }
    for (uint32_t c = 0; c < N_SHARED_COPIES; c++) {
        b.sh_off[c] = used;
        fill_weights(base + used, 3u * sh_mat, T_Q8_0);
        used += 3u * sh_mat;
    }
    need(used <= total, "arena layout");
    need(ds4_gpu_init(), "GPU initialization");
    need(ds4_gpu_set_model_map(base, total), "model map registration");
    make_sets(&b);
    const uint32_t max_in = 4096u, max_out = 248320u;
    b.x = rand_tensor(2ull * max_in);
    b.x0 = ds4_gpu_tensor_view(b.x, 0, (uint64_t)max_in * sizeof(float));
    b.y = ds4_gpu_tensor_alloc(2ull * max_out * sizeof(float));
    b.y0 = ds4_gpu_tensor_view(b.y, 0, (uint64_t)max_out * sizeof(float));
    b.mid_in = rand_tensor(2ull * (K_USED + 1u) * F);
    b.mid = ds4_gpu_tensor_alloc(2ull * (K_USED + 1u) * F * sizeof(float));
    b.part = ds4_gpu_tensor_alloc(2ull * (K_USED + 1u) * E * sizeof(float));
    need(b.x0 && b.y && b.y0 && b.mid && b.part, "work tensors");

    if (smoke) {
        int ok = 1;
        for (int r = 0; r < N_ROWS; r++)
            for (uint32_t n_tok = 1u; n_tok <= 2u; n_tok++) {
                need(ds4_gpu_begin_commands(), "begin commands");
                need(call(&b, r, n_tok, 0u), ROWS[r].name);
                need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
                const row_t *w = &ROWS[r];
                ds4_gpu_tensor *out = w->kind == R_Q8 ? b.y : w->kind == R_MID ? b.mid : b.part;
                const uint64_t n = w->kind == R_Q8 ? (uint64_t)n_tok * w->out_dim
                                                   : (uint64_t)n_tok * (K_USED + 1u) * w->out_dim;
                ok = smoke_check(out, n, w->name, n_tok) && ok;
            }
        printf("smoke: %s\n", ok ? "ok" : "FAIL");
        return ok ? 0 : 1;
    }

    double ms[N_ROWS][2], peak = 0.0;
    for (int r = 0; r < N_ROWS; r++)
        for (uint32_t n_tok = 1u; n_tok <= 2u; n_tok++) {
            const uint64_t by = bytes_per_call(r, n_tok);
            const uint64_t c = (REP_BYTES + by - 1u) / by;
            const uint32_t calls = c < MIN_CALLS ? MIN_CALLS : c > MAX_CALLS ? MAX_CALLS : (uint32_t)c;
            double enc = 0.0;
            const double g = gpu_ms(&b, r, n_tok, calls, &enc);
            ms[r][n_tok - 1u] = g;
            const double gbs = (double)by / (g * 1e6);
            if (r == 0 && n_tok == 1u) peak = gbs;
            printf("bench: %-36s T%u gpu %8.4f ms %7.1f GB/s %5.1f%% of peak (encode %.1f us/call, %u calls/rep)\n",
                   ROWS[r].name, n_tok, g, gbs, 100.0 * gbs / peak, enc * 1e3, calls);
        }
    printf("bench: peak-ref %.1f GB/s (lm_head T1)\n", peak);
    double step1 = 0.0, step2 = 0.0;
    const double bar = CANDIDATE_SHARE * CYCLE_MS_2K;
    for (int r = 0; r < N_ROWS; r++) {
        const double floor2 = (double)bytes_per_call(r, 2u) / (peak * 1e6);
        const double gap = ms[r][1] > floor2 ? ms[r][1] - floor2 : 0.0;
        const double save = 0.5 * gap * ROWS[r].n_step;
        step1 += ms[r][0] * ROWS[r].n_step;
        step2 += ms[r][1] * ROWS[r].n_step;
        printf("bench: family %-36s n=%2u T2 %.4f ms floor %.4f ms half-gap %.2f ms/cycle %s\n",
               ROWS[r].name, ROWS[r].n_step, ms[r][1], floor2, save, save >= bar ? "CANDIDATE" : "-");
    }
    printf("bench: step model T1 %.2f ms, T2 %.2f ms (projections + routed/shared experts only; bar %.2f ms)\n",
           step1, step2, bar);
    return 0;
}
```

- [ ] **Step 4: Build.**
  Run: `make tests/bench_qwen35_decode 2>&1 | grep -E "error|warning" | head; ls -la tests/bench_qwen35_decode`
  Expected: no errors and no warnings, and the binary exists.

- [ ] **Step 5: Smoke run.** This needs the GPU, and runs only once the Gateway session has sent "done" and
  no other peer is measuring. It takes seconds and loads a ~4 GB arena.
  Run: `./tests/bench_qwen35_decode smoke 2>&1 | tail -18`
  Expected: 16 lines `smoke: ... ok`, then `smoke: ok`, exit 0.
  - A `FAIL` on a finite-check means a row's arguments are wrong (offsets, slot counts) and need fixing.
  - It does not mean the kernel is wrong: these are production kernels.

- [ ] **Step 6: Commit.**

```bash
git add tests/bench_qwen35_decode.c Makefile
git commit -m "tests: bench_qwen35_decode, a model-free roofline sweep of Ornith's decode kernels"
```

### Task 3: Window 1 (the K sweep and the paired A/A), then the records

**Files:**
- Create: `speed-bench/ornith/decode/round2/KERNELS.md`, `speed-bench/ornith/decode/round2/AA.md` and
  `speed-bench/ornith/decode/round2/receipts/` (the bench output, plus the A/A `m4_ab.json` and `.txt`).
- Create (scratch, not committed): `$SCR/ornith-decode/steps-r2w1.sh`.

**Interfaces:**
- Consumes: Task 1's `--paired --blocks --reps` and `m4_ab: paired` lines; Task 2's binary and its
  `bench: family` lines.
- Produces:
  - `KERNELS.md`: the candidate list, which is Plan B's input;
  - `AA.md`: the noise figure per context (max(|mean|, sd)), which §1's gate uses.

- [ ] **Step 1: Write the steps file.** `$SCR` is the session scratchpad. Contents of
  `$SCR/ornith-decode/steps-r2w1.sh`:

```bash
# Ornith decode round 2, window 1: kernel roofline sweep, then the paired A/A at 2K and 32K.
export DS4_ORNITH_MODEL=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
export DS4_QWEN35_MTP_DRAFT_VOCAB=$HOME/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
ok=1
run_to ksweep 900 ./tests/bench_qwen35_decode || ok=0
grep "bench: family\|bench: step\|bench: peak" "$S/ksweep.log" >> "$S/progress"
rm -rf "$S/aa"
run_to aa 3000 python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model "$DS4_ORNITH_MODEL" --out "$S/aa" \
    --base-env "" --lever-env "" --contexts 2048,32768 --cold-tokens 0 --paired --blocks 2 --reps 2 || ok=0
grep "m4_ab:" "$S/aa.log" >> "$S/progress"
say "window r2w1 ok=$ok"
[ "$ok" = 1 ]
```

- [ ] **Step 2: Announce, then run.**
  - Send a heads-up to the Qwen, DS41F and Gateway sessions (SendMessage): "Ornith-decode window r2w1, about
    35 min: stack paused, K bench, paired A/A".
  - Run `bash $SCR/ornith512k/gpuwin.sh $SCR/ornith-decode/steps-r2w1.sh $SCR/ornith-decode/winR2W1` in the
    background, from the repo root.
  - When it ends, check that the stack is back (gateway `backend_ok`), then send "done".

  Expected:
  - `progress` ends with `window r2w1 ok=1`;
  - 8 `bench: family` lines;
  - the `m4_ab: paired 2048 n=4 ... mismatches=0` and `m4_ab: paired 32768 n=4 ... mismatches=0` lines.

  A mismatch in the A/A means base decoding is not deterministic across processes. In that case, stop and
  use superpowers:systematic-debugging before any lever work, because the paired method depends on it.

- [ ] **Step 3: Write `AA.md`.**
  - The A/A table: context, n, mean, sd, min, max, mismatches.
  - The noise figure per context, max(|mean|, sd).
  - The per-run decode rates from `m4_ab.json`.
  - The comparison with round 1's unpaired spread: 90.3 and 95.5 for base at 2K in window F.

  Copy the `aa/m4_ab.json` and the `aa.log` `m4_ab:` lines to `round2/receipts/aa.{json,txt}`.

- [ ] **Step 4: Write `KERNELS.md`.**
  - The bench rows, the peak reference, the family table and the step model, set against `PROFILE.md`'s
    measured plain 2K stage times. The difference is launch, encode and the non-matvec ops: the GDN scan,
    conv, norms, router, reduce and attention.
  - Then one ruling per family:
    - `CANDIDATE` (half-gap ≥ 0.60 ms per cycle): name the kernel source (`metal/*.metal` function), and say
      what an order-preserving variant could change, per spec §5's list.
    - Not a candidate: give the reason.

  Copy `ksweep.log` to `round2/receipts/ksweep.txt`.

- [ ] **Step 5: Commit the records.**

```bash
git add speed-bench/ornith/decode/round2
git commit -m "speed-bench/ornith/decode: round 2 window 1, kernel roofline and paired A/A noise"
```

- [ ] **Step 6: Branch.**
  - **With candidates:** write Plan B, `docs/superpowers/plans/2026-10-03-ornith-decode-r2-variants.md`, from
    `KERNELS.md`, using superpowers:writing-plans.
  - **With none:** apply spec §7's stopping rule. Write `speed-bench/ornith/decode/round2/REPORT.md` and go to
    the final review of the tool and the records.
