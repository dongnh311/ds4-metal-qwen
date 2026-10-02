# Ornith decode Stage 0 (profile) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure where an Ornith decode step and an MTP cycle spend their time at 2K, 32K and 128K, so
Plan B builds only the levers with measured headroom.

**Architecture:**
- Additive, env-gated instrumentation, byte-identical when unset:
  - `DS4_QWEN35_SPEC_STATS`: MTP cycle anatomy;
  - `DS4_QWEN35_SPEC_OVERLAP`: the experts both verify rows routed;
  - `DS4_QWEN35_PROFILE=3`: each layer split into mixer and MoE.
- Pure helpers in `ds4.c`, tested model-free.
- A runner (`stage0.sh`) and a parser (`parse_stage0.py`) turn one GPU window of CLI one-shots into
  `speed-bench/ornith/decode/PROFILE.md`.

**Tech Stack:** C (ds4.c, ds4_qwen35moe.inc), Metal (no kernel change), Python 3 (parser, unittest), bash.

**Spec:** `docs/superpowers/specs/2026-10-02-ornith-decode-design.md`.

## Global Constraints

- Lossless: with every new env unset, greedy output is byte-identical to develop. Gate-1 dumps at chunk
  2048 must equal the window-M `2146f8db` dumps (`$SCR/m1-merge-2048`; develop `cd4a99ec` differs from it
  only in docs).
- New code paths are Ornith-only. Qwen3.8 code is untouched; the new helpers are pure functions.
- Instrumentation never changes arithmetic. It may end and re-begin a command buffer (as
  `DS4_QWEN35_PROFILE` already does) only when its env is set.
- GPU window rules:
  - one model process at a time;
  - `gpuwin.sh` pauses and restores the stack (zombie-aware `stack.sh`);
  - peers get a heads-up and "done";
  - SIGTERM only, never `kill -9`.
- Chat in Vietnamese; code, comments and commits in English; path-scoped commits; no push in this plan.

## Review Focus

1. **The fast path does no extra work when the env is unset.** No sync and no read on the T=1 decode or
   the verify; each check is a cached `getenv`. Test: window P identity (gate 1 at 2048 equals
   `m1-merge-2048`) and `test_mtp_cli` PASS.
2. **The overlap hook fires only inside an MTP verify**, not in a 2-token prefill chunk or the MTP draft's
   T=1 MoE. Guard: `T == 2 && g->verify_rows_exact`. Test: in window P, the overlap count equals
   (verified cycles x 40 layers) at 2K.
3. **"Outside" time does not absorb gaps between requests.** A gap over 1 s counts 0. Test:
   `test_qwen35_decode_stats` adds outside_s = 2.0 and checks it is dropped.
4. **The split marks run only in the forward that set them.** The MTP draft and later forwards must not
   sync. Guard: `g_qwen35_prof_split` is cleared at the end of `qwen35_graph_forward_tokens`. Test: the
   level-3 run's generation output equals the level-0 run's for the same prompt (window P compares the
   decoded text).
5. **The parser survives a missing or failed run.** It reports "missing" and does not crash. Test:
   `test_parse_stage0.py::test_empty_log`.

---

### Task 1: Pure helpers and their model-free test

**Files:**
- Modify: `ds4.h` (after `ds4_qwen35_context_refusal`)
- Modify: `ds4.c` (after `ds4_qwen35_context_refusal`'s definition)
- Test: `tests/ds4_test.c` (new `test_qwen35_decode_stats` after `test_qwen35_context_policy`, and an
  entry after `--qwen35-context-policy`)

**Interfaces:**
- Produces:
  - `uint32_t ds4_qwen35_topk_overlap(const int32_t *a, const int32_t *b, uint32_t k)`;
  - the `ds4_qwen35_spec_stats` struct;
  - `void ds4_qwen35_spec_stats_add(ds4_qwen35_spec_stats *st, bool verified, int committed, double target_s, double draft_s, double host_s, double outside_s)`;
  - `int ds4_qwen35_spec_stats_format(const ds4_qwen35_spec_stats *st, char *buf, size_t n)`.

- [ ] **Step 1: Write the failing test** (in `tests/ds4_test.c`, after `test_qwen35_context_policy`)

```c
static void test_qwen35_decode_stats(void) {
    /* routed-expert overlap of two verify rows */
    const int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int32_t b[8] = {8, 7, 6, 5, 40, 41, 42, 43};
    const int32_t c[8] = {50, 51, 52, 53, 54, 55, 56, 57};
    TEST_ASSERT(ds4_qwen35_topk_overlap(a, b, 8) == 4);
    TEST_ASSERT(ds4_qwen35_topk_overlap(a, a, 8) == 8);
    TEST_ASSERT(ds4_qwen35_topk_overlap(a, c, 8) == 0);

    /* MTP cycle stats: an accepted verify, a rejected one, a plain step */
    ds4_qwen35_spec_stats st = {0};
    ds4_qwen35_spec_stats_add(&st, true, 2, 0.010, 0.002, 0.001, 0.0);
    ds4_qwen35_spec_stats_add(&st, true, 1, 0.010, 0.002, 0.001, 0.0);
    ds4_qwen35_spec_stats_add(&st, false, 1, 0.008, 0.002, 0.001, 2.0);   /* a gap > 1 s: a new request */
    TEST_ASSERT(st.cycles == 3 && st.plain_cycles == 1 && st.accepted == 1 && st.committed == 4);
    TEST_ASSERT(st.outside_s == 0.0);
    char line[320];
    TEST_ASSERT(ds4_qwen35_spec_stats_format(&st, line, sizeof(line)) > 0);
    TEST_ASSERT(strstr(line, "3 cycles (1 plain)") != NULL);
    TEST_ASSERT(strstr(line, "accept 0.500") != NULL);
    TEST_ASSERT(strstr(line, "1.333 tokens/cycle") != NULL);
    TEST_ASSERT(strstr(line, "ms/token 9.25") != NULL);   /* (0.028 + 0.006 + 0.003) / 4 s */
}
```

Entry, after the `--qwen35-context-policy` line:

```c
    {"--qwen35-decode-stats", "qwen35-decode-stats", "Ornith decode instrumentation: top-k overlap and MTP cycle stats (no model)", test_qwen35_decode_stats},
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test 2>&1 | grep -E " error" | head -3`
Expected: `call to undeclared function 'ds4_qwen35_topk_overlap'` (and an unknown type `ds4_qwen35_spec_stats`).

- [ ] **Step 3: Implement**

`ds4.h`, after the `ds4_qwen35_context_refusal` declaration:

```c
/* Ornith decode instrumentation (DS4_QWEN35_SPEC_STATS, DS4_QWEN35_SPEC_OVERLAP):
 * the routed experts two top-k lists share, and running MTP cycle totals.
 * A cycle is one target forward (a verify, or a plain step with no draft)
 * plus its draft; outside_s is the caller's time since the previous cycle,
 * dropped above 1 s (that is a new request, not decoding). */
uint32_t ds4_qwen35_topk_overlap(const int32_t *a, const int32_t *b, uint32_t k);
typedef struct {
    uint64_t cycles, plain_cycles, accepted, committed;
    double target_s, draft_s, host_s, outside_s;
} ds4_qwen35_spec_stats;
void ds4_qwen35_spec_stats_add(ds4_qwen35_spec_stats *st, bool verified, int committed,
                               double target_s, double draft_s, double host_s, double outside_s);
int ds4_qwen35_spec_stats_format(const ds4_qwen35_spec_stats *st, char *buf, size_t n);
```

`ds4.c`, after `ds4_qwen35_context_refusal`:

```c
uint32_t ds4_qwen35_topk_overlap(const int32_t *a, const int32_t *b, uint32_t k) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < k; i++) {
        for (uint32_t j = 0; j < k; j++) {
            if (a[i] == b[j]) {
                n++;
                break;
            }
        }
    }
    return n;
}

void ds4_qwen35_spec_stats_add(ds4_qwen35_spec_stats *st, bool verified, int committed,
                               double target_s, double draft_s, double host_s, double outside_s) {
    st->cycles++;
    if (!verified) st->plain_cycles++;
    else if (committed > 1) st->accepted++;
    if (committed > 0) st->committed += (uint64_t)committed;
    st->target_s += target_s;
    st->draft_s += draft_s;
    st->host_s += host_s;
    if (outside_s > 0.0 && outside_s <= 1.0) st->outside_s += outside_s;
}

int ds4_qwen35_spec_stats_format(const ds4_qwen35_spec_stats *st, char *buf, size_t n) {
    const uint64_t verified = st->cycles - st->plain_cycles;
    const double c = st->cycles ? (double)st->cycles : 1.0;
    const double tok = st->committed ? (double)st->committed : 1.0;
    return snprintf(buf, n, "Ornith spec stats: %llu cycles (%llu plain), accept %.3f, %.3f tokens/cycle, "
                    "ms/cycle target %.2f draft %.2f host %.2f outside %.2f, ms/token %.2f",
                    (unsigned long long)st->cycles, (unsigned long long)st->plain_cycles,
                    verified ? (double)st->accepted / (double)verified : 0.0, (double)st->committed / c,
                    1000.0 * st->target_s / c, 1000.0 * st->draft_s / c, 1000.0 * st->host_s / c,
                    1000.0 * st->outside_s / c,
                    1000.0 * (st->target_s + st->draft_s + st->host_s + st->outside_s) / tok);
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `make ds4_test 2>&1 | grep -E " error" | head -3; ./ds4_test --qwen35-decode-stats 2>&1 | tail -2`
Expected: no error; `qwen35-decode-stats: OK`, `ds4 tests: ok`.

- [ ] **Step 5: Commit**

```bash
git add ds4.h ds4.c tests/ds4_test.c
git commit -m "ds4: Ornith decode instrumentation helpers (top-k overlap, MTP cycle stats)" -- ds4.h ds4.c tests/ds4_test.c
```

---

### Task 2: The SPEC_STATS and SPEC_OVERLAP hooks

**Files:**
- Modify: `ds4.c`, inside the `#ifdef DS4_HAS_QWEN4_METAL` block with `qwen35_spec_trace` (~line 78111),
  and `ds4_session_qwen35_spec_cycle` (~78157)
- Modify: `ds4_qwen35moe.inc`, `qwen35_graph_moe` (~559), right after the router top-k

**Interfaces:**
- Consumes: Task 1's `ds4_qwen35_spec_stats_add`, `ds4_qwen35_spec_stats_format`, `ds4_qwen35_topk_overlap`.
- Produces:
  - stderr lines `ds4: Ornith spec stats: ...` (every 64 cycles, running totals);
  - stderr lines `ds4: Ornith verify expert overlap: X of 8 per layer over N layer-verifies` (every 1024
    layer-verifies, running mean).

- [ ] **Step 1: Add the stats hook** (ds4.c, after `qwen35_spec_force_accept`)

```c
/* DS4_QWEN35_SPEC_STATS=1: time each Ornith MTP cycle (the target forward,
 * the draft, the host work around them, and the caller's time since the
 * previous cycle) and print running totals every 64 cycles.  Timing only:
 * the cycle's work is unchanged. */
static bool qwen35_spec_stats_enabled(void) {
    static int v = -1;
    if (v < 0) v = getenv("DS4_QWEN35_SPEC_STATS") != NULL;
    return v != 0;
}

static ds4_qwen35_spec_stats g_qwen35_spec_stats;
static double g_qwen35_spec_last_end;

static void qwen35_spec_stats_note(bool verified, int committed, double t0, double target_s, double draft_s) {
    const double t1 = now_sec();
    const double outside = g_qwen35_spec_last_end > 0.0 ? t0 - g_qwen35_spec_last_end : 0.0;
    const double host = t1 - t0 - target_s - draft_s;
    ds4_qwen35_spec_stats_add(&g_qwen35_spec_stats, verified, committed, target_s, draft_s,
                              host > 0.0 ? host : 0.0, outside);
    g_qwen35_spec_last_end = t1;
    if (g_qwen35_spec_stats.cycles % 64u == 0u) {
        char line[320];
        ds4_qwen35_spec_stats_format(&g_qwen35_spec_stats, line, sizeof(line));
        fprintf(stderr, "ds4: %s\n", line);
    }
}
```

In `ds4_session_qwen35_spec_cycle`:
- right after the local declarations, add
  `const bool stats = qwen35_spec_stats_enabled(); const double t0 = stats ? now_sec() : 0.0; double tt = 0.0, td = 0.0, tm = 0.0;`
- time each target call: `tm = stats ? now_sec() : 0.0;` before, `if (stats) tt = now_sec() - tm;` after.
  The targets are `ds4_session_eval_internal(...)` in the plain branch and `qwen35_graph_forward_tokens(...)`
  in the verify.
- time each `qwen35_session_draft(...)` the same way into `td`.
- just before each successful `return 1;` / `return 2;`, add
  `if (stats) qwen35_spec_stats_note(<verified>, <committed>, t0, tt, td);`, where `<verified>` is false
  in the plain branch and true otherwise, and `<committed>` is the return value. The error returns
  (`-1`) record nothing.

- [ ] **Step 2: Add the overlap hook** (ds4_qwen35moe.inc, above `qwen35_graph_moe`)

```c
/* DS4_QWEN35_SPEC_OVERLAP=1: in each two-row MTP verify, count per layer the
 * routed experts both rows chose; this bounds what a two-row MoE could save.
 * Reading the ids ends the command buffer, so timings under it are not
 * representative; the arithmetic is unchanged. */
static bool qwen35_spec_overlap_enabled(void) {
    static int v = -1;
    if (v < 0) v = getenv("DS4_QWEN35_SPEC_OVERLAP") != NULL;
    return v != 0;
}

static uint64_t g_qwen35_overlap_sum, g_qwen35_overlap_n;

static bool qwen35_spec_overlap_note(ds4_qwen4_gpu_graph *g) {
    int32_t ids[2u * DS4_N_EXPERT_USED];
    if (!ds4_gpu_end_commands() || !ds4_gpu_tensor_read(g->selected, 0, ids, sizeof(ids))) return false;
    g_qwen35_overlap_sum += ds4_qwen35_topk_overlap(ids, ids + DS4_N_EXPERT_USED, DS4_N_EXPERT_USED);
    if (++g_qwen35_overlap_n % 1024u == 0u) {
        fprintf(stderr, "ds4: Ornith verify expert overlap: %.2f of %u per layer over %llu layer-verifies\n",
                (double)g_qwen35_overlap_sum / (double)g_qwen35_overlap_n, (unsigned)DS4_N_EXPERT_USED,
                (unsigned long long)g_qwen35_overlap_n);
    }
    return glm_graph_begin_commands_if_needed();
}
```

In `qwen35_graph_moe`, right after the `bool ok = qwen35_gemv(...) && ds4_gpu_qwen4_router_topk_tensor(...) != 0;`
statement:

```c
    if (ok && T == 2u && g->verify_rows_exact && qwen35_spec_overlap_enabled()) ok = qwen35_spec_overlap_note(g);
```

- [ ] **Step 3: Build and run the model-free suite**

Run: `make ds4_test ds4 ds4-server 2>&1 | grep -E " error|warning" | grep -v NaN | head -5; ./ds4_test --qwen35-decode-stats --qwen35-context-policy --qwen-yarn-policy 2>&1 | tail -1`
Expected: no error or new warning; `ds4 tests: ok`.

- [ ] **Step 4: Commit**

```bash
git commit -m "ds4: DS4_QWEN35_SPEC_STATS and DS4_QWEN35_SPEC_OVERLAP (Ornith MTP cycle anatomy, verify expert overlap)" -- ds4.c ds4_qwen35moe.inc
```

---

### Task 3: `DS4_QWEN35_PROFILE=3`, mixer and MoE split per layer

**Files:**
- Modify: `ds4_qwen35moe.inc`:
  - the new statics and `qwen35_prof_mark` above `qwen35_graph_layer` (~754);
  - one mark in `qwen35_graph_layer` before `qwen35_graph_moe`;
  - in `qwen35_graph_forward_tokens` (~783-890): `p[]` and `prof_last` become the statics; bucket
    choice; the level-3 print lines.

**Interfaces:**
- Produces stderr lines, at level 3 only (levels 1 and 2 print exactly as before):
  - `ds4: Ornith decode split ms/step (avg over 64 steps, pos=A..B): gdn-mix X gdn-moe X attn-mix X attn-moe X head X`
  - `ds4: Ornith split ms/chunk (pos=A T=N): gdn-mix X gdn-moe X attn-mix X attn-moe X head X` (T > 1,
    which includes the T=2 verify)

- [ ] **Step 1: Statics and mark** (ds4_qwen35moe.inc, above `qwen35_graph_layer`)

```c
/* DS4_QWEN35_PROFILE stage buckets, shared by qwen35_graph_forward_tokens
 * and qwen35_graph_layer.  1 gdn, 2 attn, 4 head; at level 3, 1 and 2 hold
 * the mixer half of each layer, 5 and 6 its MoE half (gdn, attn layers).
 * g_qwen35_prof_split is set only for the forward that asked for level 3
 * and cleared when it ends, so no other forward ever syncs on its account. */
static double g_qwen35_prof_p[8], g_qwen35_prof_last;
static bool g_qwen35_prof_split;

static void qwen35_prof_mark(int idx) {
    ds4_gpu_end_commands();
    const double n = now_sec();
    g_qwen35_prof_p[idx] += n - g_qwen35_prof_last;
    g_qwen35_prof_last = n;
    glm_graph_begin_commands_if_needed();
}
```

In `qwen35_graph_layer`, replace `if (ok) ok = qwen35_graph_moe(g, m, l, T);` with:

```c
    if (ok && g_qwen35_prof_split) qwen35_prof_mark(ds4_qwen35_layer_is_attention(il) ? 2 : 1);
    if (ok) ok = qwen35_graph_moe(g, m, l, T);
```

- [ ] **Step 2: Forward uses the statics** (`qwen35_graph_forward_tokens`)
- Replace `double p[5] = {0}, prof_last = prof ? now_sec() : 0.0;` with
  `memset(g_qwen35_prof_p, 0, sizeof(g_qwen35_prof_p)); g_qwen35_prof_last = prof ? now_sec() : 0.0; g_qwen35_prof_split = prof && prof_level >= 3;`
  and `#define p g_qwen35_prof_p` is NOT used: rename the uses of `p[...]` in this function to
  `g_qwen35_prof_p[...]` and `prof_last` to `g_qwen35_prof_last`, including inside `QWEN35_PROF`.
- The per-layer bucket becomes
  `QWEN35_PROF(g_qwen35_prof_split ? (ds4_qwen35_layer_is_attention(il) ? 6 : 5) : (ds4_qwen35_layer_is_attention(il) ? 2 : 1));`
- In the print block, at `prof_level >= 3` print the split formats from the Interfaces block. The T > 1
  line uses buckets 1, 5, 2, 6, 4. The decode accumulator `dprof[]` grows to 8 entries; on its 64-step
  line it prints the same five buckets divided by `dprof_n`. Level 2 keeps its exact old line.
- Directly before `if (!ds4_gpu_end_commands()) ok = false;`, add `g_qwen35_prof_split = false;`.

- [ ] **Step 3: Build and run the model-free suite**

Run: `make ds4_test ds4 ds4-server 2>&1 | grep -E " error|warning" | grep -v NaN | head -5; ./ds4_test --qwen35-decode-stats --qwen35-context-policy 2>&1 | tail -1`
Expected: no error or new warning; `ds4 tests: ok`.

- [ ] **Step 4: Commit**

```bash
git commit -m "ds4: DS4_QWEN35_PROFILE=3 splits each Ornith layer into mixer and MoE" -- ds4_qwen35moe.inc
```

---

### Task 4: Stage-0 runner and parser

**Files:**
- Create: `speed-bench/ornith/decode/stage0.sh` (sourced by the scratch `gpuwin.sh`; uses `say` and `run_to`)
- Create: `speed-bench/ornith/decode/parse_stage0.py`
- Test: `speed-bench/ornith/decode/test_parse_stage0.py`

**Interfaces:**
- Consumes the stderr lines of Tasks 2 and 3, plus the existing:
  - `ds4: cb <label>: driver X us, queue-wait X us, gpu X us, gap-from-prev-gpu-end X us` (`DS4_METAL_CB_TIMES`);
  - `... generation: X t/s` (CLI).
- Produces `parse_stage0.py DIR` → a markdown table on stdout, plus functions `gen_tps(text)`,
  `cb_decode(text)`, `last_line(text, prefix)`, `split_means(text)`.

- [ ] **Step 1: Write the failing parser tests**

```python
"""Unit tests for parse_stage0.py (python3 -m unittest, no model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_stage0 as p

LOG = """ds4: cb command batch: driver 30 us, queue-wait 5 us, gpu 812000 us, gap-from-prev-gpu-end 0 us
ds4: cb command batch: driver 20 us, queue-wait 4 us, gpu 9000 us, gap-from-prev-gpu-end 1000 us
ds4: cb command batch: driver 20 us, queue-wait 4 us, gpu 11000 us, gap-from-prev-gpu-end 3000 us
ds4: Ornith decode split ms/step (avg over 64 steps, pos=10..73): gdn-mix 4.00 gdn-moe 8.00 attn-mix 2.00 attn-moe 3.00 head 2.00
ds4: Ornith decode split ms/step (avg over 64 steps, pos=74..137): gdn-mix 6.00 gdn-moe 8.00 attn-mix 2.00 attn-moe 3.00 head 2.00
ds4: Ornith spec stats: 64 cycles (1 plain), accept 0.870, 1.850 tokens/cycle, ms/cycle target 14.00 draft 2.00 host 0.50 outside 0.20, ms/token 9.30
ds4: Ornith verify expert overlap: 3.10 of 8 per layer over 2048 layer-verifies
ds4: Ornith prefill: 1980.35 t/s, generation: 45.11 t/s
"""


class ParseTest(unittest.TestCase):
    def test_gen_tps(self):
        self.assertEqual(p.gen_tps(LOG), 45.11)

    def test_cb_decode_drops_prefill_buffers(self):
        d = p.cb_decode(LOG)        # the 812 ms prefill buffer is not decode
        self.assertEqual(d["n"], 2)
        self.assertAlmostEqual(d["gpu_ms"], 20.0)
        self.assertAlmostEqual(d["gap_ms"], 4.0)
        self.assertAlmostEqual(d["busy"], 20.0 / 24.0)

    def test_split_means(self):
        s = p.split_means(LOG, "decode split ms/step")
        self.assertAlmostEqual(s["gdn-mix"], 5.0)
        self.assertAlmostEqual(s["head"], 2.0)

    def test_last_line(self):
        self.assertIn("accept 0.870", p.last_line(LOG, "Ornith spec stats:"))
        self.assertIn("3.10 of 8", p.last_line(LOG, "Ornith verify expert overlap:"))

    def test_empty_log(self):
        self.assertIsNone(p.gen_tps(""))
        self.assertEqual(p.cb_decode("")["n"], 0)
        self.assertEqual(p.split_means("", "decode split ms/step"), {})
        self.assertIsNone(p.last_line("", "Ornith spec stats:"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest speed-bench/ornith/decode/test_parse_stage0.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'parse_stage0'`.

- [ ] **Step 3: Implement the parser**

```python
#!/usr/bin/env python3
"""Stage-0 log parser (spec 2026-10-02-ornith-decode-design.md section 3).

  parse_stage0.py DIR   -> one markdown row per <ctx>-<run>.log in DIR
"""
import glob
import os
import re
import sys

PREFILL_GPU_US = 50000      # a buffer with more GPU time than this is a prefill chunk, not a decode step
SPLIT_KEYS = ("gdn-mix", "gdn-moe", "attn-mix", "attn-moe", "head")


def gen_tps(text):
    m = re.findall(r"generation: ([\d.]+) t/s", text)
    return float(m[-1]) if m else None


def cb_decode(text):
    gpu = gap = 0.0
    n = 0
    for g, gp in re.findall(r"ds4: cb [^:]*: driver [\d.]+ us, queue-wait [\d.]+ us, gpu ([\d.]+) us, "
                            r"gap-from-prev-gpu-end ([\d.]+) us", text):
        g, gp = float(g), float(gp)
        if g > PREFILL_GPU_US:
            continue
        gpu += g / 1000.0
        gap += gp / 1000.0
        n += 1
    return {"n": n, "gpu_ms": gpu, "gap_ms": gap, "busy": gpu / (gpu + gap) if gpu + gap > 0 else None}


def split_means(text, what):
    rows = [ln for ln in text.splitlines() if what in ln]
    out = {}
    for k in SPLIT_KEYS:
        vals = [float(m.group(1)) for ln in rows for m in [re.search(r"\b%s ([\d.]+)" % re.escape(k), ln)] if m]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def last_line(text, prefix):
    rows = [ln[ln.index(prefix):] for ln in text.splitlines() if prefix in ln]
    return rows[-1] if rows else None


def fmt(v, f="%.2f"):
    return "missing" if v is None else f % v


def main():
    d = sys.argv[1]
    print("| run | gen t/s | decode CBs | GPU busy | split ms/step (gdn-mix/gdn-moe/attn-mix/attn-moe/head) "
          "| verify split ms/chunk T=2 | spec stats | overlap |")
    print("|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(d, "*.log"))):
        text = open(path, errors="replace").read()
        cb = cb_decode(text)
        sd = split_means(text, "decode split ms/step")
        sv = split_means(text, "T=2): gdn-mix")
        print("| %s | %s | %d | %s | %s | %s | %s | %s |" % (
            os.path.basename(path)[:-4], fmt(gen_tps(text)), cb["n"], fmt(cb["busy"], "%.3f"),
            "/".join(fmt(sd.get(k)) for k in SPLIT_KEYS) if sd else "missing",
            "/".join(fmt(sv.get(k)) for k in SPLIT_KEYS) if sv else "missing",
            last_line(text, "Ornith spec stats:") or "missing",
            last_line(text, "Ornith verify expert overlap:") or "missing"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest speed-bench/ornith/decode/test_parse_stage0.py 2>&1 | tail -3`
Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Write the runner** (`speed-bench/ornith/decode/stage0.sh`)

```bash
# Ornith decode Stage 0 (spec section 3).  Sourced by the scratch gpuwin.sh: $S (out dir), say, run_to;
# cwd = the repo.  CLI one-shots on promessi_sposi prefixes plus a summary request, 256 tokens.
ORNITH=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
export DS4_QWEN35_MTP_DRAFT_VOCAB=$HOME/.local/share/ai-gateway/ds4-models/Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt
unset DS4_QWEN4_YARN_FACTOR DS4_QWEN35_PROFILE DS4_QWEN35_SPEC_STATS DS4_QWEN35_SPEC_OVERLAP DS4_METAL_CB_TIMES
ok=1
cli() {   # cli NAME PROMPTFILE ENV... -- ARGS...
    local name=$1 pf=$2; shift 2
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    run_to "$name" 1800 env "${envs[@]}" ./ds4 -m "$ORNITH" --metal -c 262144 --prefill-chunk 2048 \
        --prompt-file "$pf" -n 256 "$@" || ok=0
}
for c in 2k:7000 32k:112000 128k:450000; do
    ctx=${c%%:*}; chars=${c#*:}
    pf="$S/prompt-$ctx.txt"
    { head -c "$chars" speed-bench/promessi_sposi.txt; printf '\n\nSummarize the story so far in a few sentences.\n'; } > "$pf"
    cli "$ctx-plain-cb" "$pf" DS4_METAL_CB_TIMES=1 -- --temp 0
    cli "$ctx-mtp-cb" "$pf" DS4_METAL_CB_TIMES=1 DS4_QWEN35_SPEC_STATS=1 -- --temp 0 --mtp
    cli "$ctx-plain-split" "$pf" DS4_QWEN35_PROFILE=3 -- --temp 0
    cli "$ctx-mtp-split" "$pf" DS4_QWEN35_PROFILE=3 DS4_QWEN35_SPEC_OVERLAP=1 -- --temp 0 --mtp
    if [ "$ctx" != 128k ]; then
        cli "$ctx-mtp-t07" "$pf" DS4_QWEN35_SPEC_STATS=1 -- --temp 0.7 --seed 1 --mtp
        cli "$ctx-mtp-plain" "$pf" -- --temp 0 --mtp
    fi
done
cli "2k-plain-l2" "$S/prompt-2k.txt" DS4_QWEN35_PROFILE=2 -- --temp 0
say "stage0 ok=$ok"
[ "$ok" = 1 ]
```

Run: `bash -n speed-bench/ornith/decode/stage0.sh && echo ok`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git commit -m "speed-bench/ornith/decode: Stage-0 runner and log parser" -- speed-bench/ornith/decode/stage0.sh speed-bench/ornith/decode/parse_stage0.py speed-bench/ornith/decode/test_parse_stage0.py
```

---

### Task 5: GPU window P, then PROFILE.md and the lever ranking

**Files:**
- Create (scratch): `$SCR/ornith-decode/steps-p.sh`
- Create: `speed-bench/ornith/decode/PROFILE.md`, `speed-bench/ornith/decode/receipts/stage0/` (parsed table + key log lines)

- [ ] **Step 1: Steps file** (`$SCR/ornith-decode/steps-p.sh`, sourced by the scratch `gpuwin.sh` with `$S`)

```bash
SCR=/private/tmp/claude-501/-Users-dongnh-orca-workspaces-ds4-metal-foxface/a18593e5-414b-4337-9d32-8629ac82447a/scratchpad/ornith512k
export DS4_ORNITH_MODEL=$HOME/.local/share/ai-gateway/ds4-models/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf
unset DS4_QWEN4_YARN_FACTOR
pok=1
# identity first: every new env unset must leave the decode byte-identical to develop (window M dumps)
rm -rf "$S/g1-2048"
run_to g1-2048 3600 python3 tests/ornith/gate1.py --ds4-arg --prefill-chunk --ds4-arg 2048 "$S/g1-2048" || pok=0
n=0; for f in "$SCR/m1-merge-2048"/ds4/*.json; do cmp -s "$f" "$S/g1-2048/ds4/$(basename "$f")" || n=$((n + 1)); done
say "identity chunk 2048 vs window M: $n differ"; [ "$n" = 0 ] || pok=0
run_to mtp-cli 7200 env DS4_QWEN35_SPEC_STATS=1 DS4_QWEN35_SPEC_OVERLAP=1 python3 tests/ornith/test_mtp_cli.py "$S/mtp-cli" || pok=0
source speed-bench/ornith/decode/stage0.sh || pok=0
say "window P ok=$pok"
[ "$pok" = 1 ]
```

- [ ] **Step 2: Run the window** (after a heads-up to Mac16-Ai-Gateway, DS41F and Qwen; one model process at a time)

Run: `caffeinate -i -s bash $SCR/gpuwin.sh $SCR/ornith-decode/steps-p.sh $SCR/ornith-decode/winP` in the
background, with a Monitor on `winP/progress`.
Expected:
- `identity chunk 2048 vs window M: 0 differ`;
- `test_mtp_cli` PASS;
- `stage0 ok=1`;
- `window P ok=1`;
- the stack is restored (`backend_ok True`).

Also check by reading:
- In `2k-plain-split.log` and `2k-mtp-plain.log`, the generated text equals that of `2k-plain-cb.log` and
  `2k-mtp-cb.log` respectively (Review Focus 4).
- The overlap count in `2k-mtp-split.log` equals verified cycles x 40, where the verified cycles come from
  the matching spec-stats line in `2k-mtp-cb.log` (Review Focus 2).

- [ ] **Step 3: Parse and write PROFILE.md**

Run: `python3 speed-bench/ornith/decode/parse_stage0.py $SCR/ornith-decode/winP > speed-bench/ornith/decode/receipts/stage0/table.md`

`PROFILE.md` holds:
1. The table.
2. Per context: ms per committed token with `--mtp`, split into target, draft, host and outside (from
   spec stats). Plain decode ms/step is split by the level-3 profile, scaled to the unprofiled
   generation rate. The scale factor is unprofiled ms/step over profiled ms/step; level 2 against
   level 3 at 2K shows what the extra marks cost.
3. GPU busy fraction during decode, from the CB times.
4. Expert overlap per verify layer, with the two-row MoE saving bound. The saving is
   (overlap + 1 shared) / 18 of the verify's MoE expert reads, applied to the verify's measured MoE ms.
5. A projected gain per lever (L1-L3, L5), in % of ms per committed token at 2K and 32K, each with its
   formula. L4 stays unprojected until its probe exists (Plan B, if needed).
6. The ranking and the cut, as ledger rulings (spec section 4: build in order, skip any under 3%).

- [ ] **Step 4: Commit**

```bash
git add speed-bench/ornith/decode/PROFILE.md speed-bench/ornith/decode/receipts/stage0
git commit -m "speed-bench/ornith/decode: Stage 0 profile at 2K/32K/128K and the lever ranking" -- speed-bench/ornith/decode/PROFILE.md speed-bench/ornith/decode/receipts/stage0
```
