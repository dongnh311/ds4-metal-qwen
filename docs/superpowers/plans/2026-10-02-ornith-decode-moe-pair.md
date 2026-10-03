# Ornith decode Plan B: two-row MoE in the MTP verify (lever L2)

> **Status:** executed, measured, and not merged. The levers gave no decode gain (see
> `speed-bench/ornith/decode/REPORT.md`). Their code and env knobs exist only on the local branch
> `feature/ornith-decode` at `01d4bab0`, not on develop.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read each routed expert that both rows of a 2-row MTP verify route to once instead of twice,
losslessly, for about +10% decode at 2K and about +7% at 32K.

**Architecture:**
- **Kernels.** Two new Ornith-only kernels in `metal/qwen35.metal`, `kernel_qwen35_moe_mid_pair<NR>` and
  `kernel_qwen35_moe_down_pair<NR>`, for Q5_K and Q4_K.
  - Row 0's threadgroup for a slot finds the row-1 slot that routed the same expert, and computes that
    slot's output too. For the shared expert it always computes both rows.
  - Row 1's threadgroup skips those slots.
  - Each output is one `qwen35_row_dot` per row, the same per-lane order as the per-row kernels.
- **Host wrappers.** `ds4_gpu_qwen35_moe_pair_mid_tensor` and `ds4_gpu_qwen35_moe_pair_down_tensor`.
- **Use.** `qwen35_graph_moe` uses them in the verify (`T == 2 && verify_rows_exact`) when
  `DS4_QWEN35_MOE_PAIR` is on.
- **Self-check.** On first use in a process, a run-time check compares them with the per-row path. A
  mismatch turns them off.

**Tech Stack:** Metal, Objective-C (`ds4_metal.m`), C (`ds4_qwen35moe.inc`), C tests (`tests/test_qwen35_kernels.c`).

**Spec:** `docs/superpowers/specs/2026-10-02-ornith-decode-design.md` (lever L2, §4). Stage 0 evidence:
`speed-bench/ornith/decode/PROFILE.md`.
- Verify MoE is about 7.7 ms against about 4.8 ms for a plain step at 2K.
- On average 3.2 of 8 experts overlap per layer.
- The projected saving is about 25% of the verify's expert bytes, about 1.9 ms per cycle.

## Global Constraints

- **Lossless.** Every pair output must `memcmp`-equal the per-row path's output for the same row:
  - the Q5_K kernels in use (`kernel_qwen35_moe_{mid,down}_q5k_nr*`);
  - the Q4_K kernels in use (`kernel_qwen4_moe_mid_q4k*`, the qwen4 down kernels).

  Graph level: `test_qwen35_verify_batch`, `test_mtp_cli` and the gate-1 dumps at chunks 64/512/2048
  stay byte-identical with the lever on.
- **Ornith-only.** The new kernels are new entry points in `metal/qwen35.metal`. Qwen3.8 kernels and
  dispatch are untouched. `ds4_metal.m` gains wrappers only; no existing function changes.
- **Knob.** `DS4_QWEN35_MOE_PAIR` starts default off. It flips to on only after window L shows identity
  and at least +3% at 2K or 32K (spec §2).
- **GPU kernel tests** run outside other sessions' measuring windows: the coding block, or my own windows.

## Review Focus

1. **A row-1 expert that row 0 also routed, in a different slot.** The partner search must match by
   expert id, not by slot index. Test: the kernel test's selections overlap at different slots.
2. **No overlap at all, and every expert overlapping.** Each must still write every (row, slot) output
   exactly once. Test: the kernel test adds a disjoint case and an identical case, with sentinel
   checks.
3. **The shared expert slot.** Row 0's group writes both rows, and row 1's group must not write it again
   (a race would still be identical, but double work). Test: covered by the identity cases, and the
   kernel only writes row 1's shared output from row 0's group.
4. **The self-check fails.** The process falls back to the per-row path, says so on stderr, and still
   produces correct results for that verify. Test: `DS4_QWEN35_MOE_PAIR_SELFTEST_FAIL=1` forces a
   mismatch, and `test_qwen35_verify_batch` still passes.
5. **Other T=2 forwards.** A 2-token prefill chunk or the MTP draft must never take the pair path. Guard:
   `g->verify_rows_exact`. Test: gate-1 prefill at chunk 64 with the lever on stays identical (window L).

---

### Task 1: Pair kernels, wrappers and the kernel-level exactness test

**Files:**
- Modify: `metal/qwen35.metal` (append after the `kernel_qwen35_moe_down_q5k_nr` instances)
- Modify: `ds4_metal.m`:
  - the kernel enum, after `QWEN4_K_QWEN35_MOE_DOWN_Q5K_NR4`;
  - the matching name table entries;
  - new wrappers after `ds4_gpu_qwen35_moe_down_tensor`.
- Modify: `ds4_gpu.h` (declare the wrappers next to `ds4_gpu_qwen35_moe_down_tensor`)
- Test: `tests/test_qwen35_kernels.c`, new `arena_q4_K` and `test_moe_pair`, called from `main`

**Interfaces:**
- Produces:
  - `int ds4_gpu_qwen35_moe_pair_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint32_t weight_type, uint32_t n_total_expert, uint32_t n_slots, uint32_t in_dim, uint32_t ff_dim, uint64_t shared_gate_offset, uint64_t shared_up_offset, uint32_t shared_type)`
  - `int ds4_gpu_qwen35_moe_pair_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected, const void *model_map, uint64_t model_size, uint64_t down_offset, uint32_t weight_type, uint32_t n_total_expert, uint32_t n_slots, uint32_t ff_dim, uint32_t out_dim, uint64_t shared_down_offset, uint32_t shared_type)`

  Both are always T = 2 and return 0 on a bad argument, as their siblings do.

- [ ] **Step 1: Failing test.**
  - `arena_q4_K(a, rows, cols, scale)`: random 144-byte Q4_K blocks with finite small half d and dmin.
  - `test_moe_pair(a, type)` for type 13 (Q5_K) and 12 (Q4_K):
    - NE=16, slots=8, E=512, F=256, T=2, shared Q8_0;
    - three selection cases: overlap at different slots (row0 0..7, row1 {5,2,9,10,11,7,12,13}),
      disjoint, identical;
    - reference: `ds4_gpu_qwen35_moe_mid_tensor` (Q5_K) or `ds4_gpu_qwen4_moe_mid_tensor` (Q4_K) with T=2,
      and the matching down wrapper;
    - pair outputs come from the new wrappers, each into a sentinel-filled buffer;
    - every element must `memcmp`-equal the reference.
- [ ] **Step 2: RED.** `make tests/test_qwen35_kernels` fails on the undeclared wrapper.
- [ ] **Step 3: Implement the kernels.**

```metal
/* Decode lever L2: the two-row MTP verify reads an expert once when both rows
 * routed it.  Row 0's group for a slot also writes row 1's output for the same
 * expert (and both rows of the shared expert); row 1's group skips those slots.
 * Every output is one qwen35_row_dot call per row, the per-row kernels' lane
 * order, so each value is bit-identical to them. */
static inline int qwen35_pair_partner(device const int32_t *selected, uint n_slots, uint slot) {
    const int32_t e = selected[slot];
    for (uint j = 0; j < n_slots; j++) if (selected[n_slots + j] == e) return (int)j;
    return -1;
}

static inline bool qwen35_pair_row0_has(device const int32_t *selected, uint n_slots, uint slot1) {
    const int32_t e = selected[n_slots + slot1];
    for (uint j = 0; j < n_slots; j++) if (selected[j] == e) return true;
    return false;
}

template <uint NR>
kernel void kernel_qwen35_moe_mid_pair(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *selected, device const float *x, device float *mid,
        device const char *sh_gate, device const char *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= 2u) return;
    const bool shared = slot == args.n_slots;
    int other = -1;
    if (tok == 0u) other = shared ? (int)args.n_slots : qwen35_pair_partner(selected, args.n_slots, slot);
    else if (shared || qwen35_pair_row0_has(selected, args.n_slots, slot)) return;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    device const float *x1 = x + (uint64_t)args.in_dim;
    const uint64_t mb = ((uint64_t)tok * n_out + slot) * args.out_rows;
    const uint64_t mb1 = ((uint64_t)n_out + (uint)max(other, 0)) * args.out_rows;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * rb;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[mb + r] = qwen4_silu(g) * u;
        if (other >= 0) {
            const float g1 = qwen35_row_dot(gb + off, x1, type, args.in_dim, tiisg);
            const float u1 = qwen35_row_dot(ub + off, x1, type, args.in_dim, tiisg);
            if (tiisg == 0) mid[mb1 + r] = qwen4_silu(g1) * u1;
        }
    }
}

template <uint NR>
kernel void kernel_qwen35_moe_down_pair(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base, device const int32_t *selected,
        device const float *mid, device float *part, device const char *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= 2u) return;
    const bool shared = slot == args.n_slots;
    int other = -1;
    if (tok == 0u) other = shared ? (int)args.n_slots : qwen35_pair_partner(selected, args.n_slots, slot);
    else if (shared || qwen35_pair_row0_has(selected, args.n_slots, slot)) return;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t pair1 = (uint64_t)n_out + (uint)max(other, 0);
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    device const float *m1 = mid + pair1 * args.in_dim;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        device const char *row = db + ebase + (uint64_t)r * rb;
        const float v = qwen35_row_dot(row, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
        if (other >= 0) {
            const float v1 = qwen35_row_dot(row, m1, type, args.in_dim, tiisg);
            if (tiisg == 0) part[pair1 * args.out_rows + r] = v1;
        }
    }
}
```

  Instances: `kernel_qwen35_moe_mid_pair_nr1/nr2/nr4` and `kernel_qwen35_moe_down_pair_nr1/nr2/nr4`, in
  the existing `template [[host_name(...)]]` style.

- [ ] **Step 4: Implement the wrappers.**
  - Copy `ds4_gpu_qwen35_moe_{mid,down}_tensor` with `n_tokens = 2`, and keep its argument checks and
    binds.
  - Pick NR and groups with `qwen35_moe_mr(down, ...)` (nr 0 maps to 2) so the threadgroup shape
    matches the per-row path, and dispatch the pair kernel with grid
    `((rows + nr*groups - 1)/(nr*groups), n_out, 2)`.
  - Count dispatches in `g_qwen35_moe_pair_dispatches` and expose it as
    `uint64_t ds4_gpu_qwen35_moe_pair_dispatches(void)` (test and graph use).
- [ ] **Step 5: GREEN.** Build `tests/test_qwen35_kernels` and run it. Expected: `moe pair q5_K/q4_K
  {overlap, disjoint, identical}: bit-identical` and every earlier line still passing. It runs on the
  GPU without a model; run it outside other sessions' measuring windows.
- [ ] **Step 6: Commit.** Commit `metal/qwen35.metal`, `ds4_metal.m`, `ds4_gpu.h` and
  `tests/test_qwen35_kernels.c`, path-scoped.

---

### Task 2: Use the pair kernels in the verify, with the run-time self-check

**Files:**
- Modify: `ds4_qwen35moe.inc`, `qwen35_graph_moe` (the non-mm `else if (ok)` branch)

**Interfaces:**
- Consumes the Task 1 wrappers.
- Produces:
  - the env `DS4_QWEN35_MOE_PAIR` (default off until window L), and its test hook
    `DS4_QWEN35_MOE_PAIR_SELFTEST_FAIL`;
  - stderr lines `ds4: Ornith two-row MoE self-check passed` and
    `ds4: Ornith two-row MoE self-check FAILED; per-row MoE for this process`.

- [ ] **Step 1: Behaviour.** In `qwen35_graph_moe`, compute
  `const bool pair = T == 2u && g->verify_rows_exact && qwen35_moe_pair_on();`. When `pair` holds and
  the path is the per-row one (not `mm`):
  - **State 1 (checked ok):** call the two pair wrappers instead of the per-row mid/down calls, for both
    Q5_K and Q4_K layers.
  - **State 0 (first use):** do the self-check.
    1. Run the per-row mid and down as today, `ds4_gpu_end_commands()`, and read `g->mid` and `g->part`
       (2 x n_out x F and 2 x n_out x E floats) to host buffers.
    2. Run the pair mid and down, end, read them, and `memcmp`. With
       `DS4_QWEN35_MOE_PAIR_SELFTEST_FAIL` set, the comparison is forced to fail.
    3. On a match, set state 1 and print "passed".
    4. On a mismatch, set state -1, print "FAILED", and re-run the per-row mid and down so `g->part`
       holds the per-row values.
    5. `glm_graph_begin_commands_if_needed()` before the reduce.
  - **State -1:** the per-row path, unchanged.

  `qwen35_moe_pair_on()` reads `DS4_QWEN35_MOE_PAIR` once, like the other knobs: unset or "0" means off,
  "1" means on.
- [ ] **Step 2: Model-free check.** `make ds4 ds4-server ds4_test` builds with no new warning, and the
  `./ds4_test` model-free entries pass. The model-backed checks are window L.
- [ ] **Step 3: Commit.** Commit `ds4_qwen35moe.inc`, path-scoped.

---

### Task 3: Window L, identity and A/B

**Files:**
- Create (scratch): `$SCR/ornith-decode/steps-l.sh`
- Create: `speed-bench/ornith/decode/levers/moe-pair.md` and raw `ab-moe-pair.{txt,json}`

- [ ] **Step 1: Steps** (sourced by `gpuwin.sh`), at its slot in the overnight schedule:
  1. `make test-qwen35-kernels` (the pair test included).
  2. Gate 1 at chunks 64/512/2048 with `DS4_QWEN35_MOE_PAIR=1`. The dumps must equal window M's
     `m1-merge-*`.
  3. `DS4_QWEN35_MOE_PAIR=1 python3 tests/ornith/test_mtp_cli.py`.
  4. `test_qwen35_verify_batch` with the lever on, and again with
     `DS4_QWEN35_MOE_PAIR_SELFTEST_FAIL=1` (it must still pass, with the "FAILED" line).
  5. `python3 speed-bench/ornith/m4_ab.py --mode lever --ds4-model $ORNITH --out ... --base-env DS4_QWEN35_MOE_PAIR=0 --lever-env DS4_QWEN35_MOE_PAIR=1 --contexts 2048,32768,131072`
     with `--ds4-args` carrying `--mtp` and the draft vocab env, matching the registry command. The
     contexts are A-B-B-A.
- [ ] **Step 2: Expected.**
  - Identity: 0 dumps differ.
  - Every test PASS.
  - A/B: lever/base at least 1.03 at 2K or 32K, and 128K no worse than 0.97.
- [ ] **Step 3: Record.** Write `levers/moe-pair.md`. If the expectations hold, flip the default: make
  `qwen35_moe_pair_on()` default to on when the env is unset, keep `=0` as the off switch, and commit.
  Otherwise ledger a ruling and leave it off.
