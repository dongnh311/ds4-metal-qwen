# Ornith M7: exact batched MTP verify

Status: design approved in conversation 2026-09-28; this document awaits the user's review.
Parent spec: `2026-09-25-ornith-qwen35moe-design.md`. Branch: `feature/ornith-m7`, cut from develop `b764a66`
(the M6 merge).

## 1. Goal

Raise Ornith's short-context decode rate on ds4 to at least the live oMLX's, so ds4 can replace Ornith on oMLX.
M6 left ds4 ahead at long context, in memory and in quality, and behind in short-context decode (gate 3, A-B-B-A,
`--mtp` with the 64K draft vocabulary):

| context | ds4 decode t/s | oMLX decode t/s |
|---|---:|---:|
| 2K | 69.5 | 79.9 |
| 32K | 53.1 | 64.5 |
| 128K | 40.1 | 38.9 |

**Done when**, on the gate 3 harness (`speed-bench/ornith/m4_ab.py`, A-B-B-A, same day as the oMLX arm):

1. ds4 `--mtp` decode >= oMLX decode at 2K and at 32K;
2. ds4 decode at 128K is no more than 3% below M6 (40.1 t/s);
3. `--mtp` output stays byte-identical to plain decoding (`tests/ornith/test_mtp_cli.py`);
4. gate 1 passes, gate 2 stays level with oMLX, and Qwen3.8 stays byte-identical (fast gate and full gate).

## 2. Decisions already taken

- **Keep the guarantee** (user, 2026-09-28): `--mtp` stays byte-identical to plain decoding. No lever here may
  trade exactness for speed.
- **Approach A** (user, 2026-09-28): make the 2-row MTP verify read each weight once for both rows while every
  row keeps the exact arithmetic of a plain 1-row step. Rejected alternatives: a non-exact batched verify
  (breaks the guarantee) and a faster plain T=1 path (decode is already weight-bandwidth-bound at ~175 GB/s).
- Qwen3.8 code paths are not changed. New code lives in Ornith files (`ds4_qwen35moe.inc`,
  `metal/qwen35.metal`) plus new, separately named entry points in `ds4_metal.m` / `ds4_gpu.h`.

## 3. Where the verify's time goes today

A plain decode step at 2K takes 15.5 ms (64.65 t/s without MTP) and reads ~2.7 GB of weights:

| part | bytes per token | type |
|---|---:|---|
| GDN projections (`lin_qkv` 2048->8192, `lin_gate` 2048->4096, `lin_out` 4096->2048) x 30 | 1.07 GB | Q8_0 |
| routed experts (8 of 256) + shared expert x 40 | ~0.80 GB | Q5_K / Q4_K, Q8_0 |
| lm head (2048->248,320) | 0.54 GB | Q8_0 |
| attention (`attn_q` 2048->8192, `attn_output` 4096->2048 Q8_0; `attn_k`/`attn_v` F16) x 10 | 0.31 GB | Q8_0, F16 |
| router (F32 2048->256) x 40, GDN alpha/beta (F32) x 30 | ~0.10 GB | F32 |

The verify of a 2-token draft runs a T=2 forward under `g->verify_rows_exact`, where every dense projection
(`qwen35_gemv`), every GDN layer (`qwen35_graph_linear_rows`) and the lm head run as one T=1 dispatch per row, so
each weight is streamed twice. Only attention shares work (`kernel_qwen35_attn_decode3` with rows=2 reads K/V once).
Result: a verify costs ~1.74x a plain step, and MTP (acceptance ~87%) adds only ~7% over plain decoding.

Every Q8_0 projection a plain `--mtp` step reaches goes through one kernel: `kernel_mul_mv_q8_0_f32`
(`metal/dense.metal`, NR0 = 2, NSG from `ds4_gpu_make_q8_0_mv_dispatch()`: 4 by default, 8 when out_dim > 65536,
i.e. the lm head). In an MTP run the fused `q8_pair` GDN projection is off (`g->mtp_R` is set), so `lin_qkv` and
`lin_gate` take that kernel too.

## 4. Design

### 4.1 Measure first (Task 1)

A model-free microbench (`tests/bench_qwen35_attn.c` style, new `tests/bench_qwen35_verify.c`) times, for each
Q8_0 shape above (2048->8192, 2048->4096, 4096->2048, 2048->248320):

- one T=1 call (`ds4_gpu_qwen4_matmul_q8_0_tensor`, n_tok = 1),
- two T=1 calls on two rows,
- the existing `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(..., n_rows = 2)` (same `mul_mv` pipeline, grid
  (tiles, rows): exact by construction, but row 1's threadgroups re-read the weights after row 0's),
- the new rows kernel (4.2) once it exists.

If `decode_rows_exact` already costs <= 1.2x one row for every shape including the lm head, the verify uses it and
4.2 is skipped. Otherwise 4.2 is built. Either way the bench is kept as the receipt.

### 4.2 Kernel: `kernel_qwen35_mv_q8_0_rows`

In `metal/qwen35.metal`: a copy of `kernel_mul_mv_q8_0_f32_impl` with an inner loop over `R` = 2 input rows. Per
threadgroup it keeps `yl[R][NQ]` and `sumf[R][NR0]`; each weight block (`qs`, `d`) is loaded once and multiplied
into both rows. For each row the sequence of multiply-adds, the `simd_sum`, the threadgroup-memory reduction and the
final `simd_sum` are exactly those of the 1-row kernel with the same NR0 and NSG, so row r of the output is bit
identical to a T=1 call on row r. Threadgroup memory is `R * NR0 * 32` floats.

Host entry point in `ds4_metal.m`, declared in `ds4_gpu.h`:

```c
int ds4_gpu_qwen35_matmul_q8_0_rows_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                           uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_rows);
```

It takes NSG and NR0 from `ds4_gpu_make_q8_0_mv_dispatch()` with the same `out_dim > 65536 -> nsg = 8` rule as the
T=1 path, so both always agree. `n_rows` is 2 (the only verify width Ornith supports today; anything else returns
0 and the caller falls back to per-row dispatch).

### 4.3 Dense projections in the verify

`qwen35_gemv` under `g->verify_rows_exact` with the knob on (4.6): a Q8_0 weight with T = 2 takes one batched call
(4.1's choice). F16 (`attn_k`, `attn_v`) and F32 weights (router, alpha/beta) keep the per-row loop: together they
are ~0.14 GB per token and their T=1 kernels differ from the batched ones. This covers `attn_q`, `attn_output` and
the verify's all-rows lm head (`qwen35_gemv(g, g->logits, ...)`) without further changes.

### 4.4 GDN layers in the verify: `qwen35_graph_linear_verify`

A new Ornith-local 2-row GDN layer in `ds4_qwen35moe.inc`, used by `qwen35_graph_linear_rows` when the knob is on:

1. `lin_qkv` and `lin_gate` for both rows in one batched call each (into `g->qkv`, `g->z`);
2. for row 0, then row 1: the T=1 front (`ds4_gpu_qwen4_gdn_front_tensor`, or alpha/beta GEMVs + conv stream + prep
   when `qwen4_graph_fused` is false), scan (`ds4_gpu_qwen4_gdn_scan_tensor`) and gated norm
   (`ds4_gpu_qwen35_gdn_out_tensor` / `ds4_gpu_qwen4_gdn_out_tensor`) on row views of `g->qkv`, `g->z`, `g->ga`,
   `g->gb`, `g->lin_o` and `g->mixed`, with the after-first-row snapshot pointers passed only to row 0's calls
   (as `qwen35_graph_linear_rows` does today);
3. `lin_out` for both rows in one batched call (into `g->blk`).

The recurrent state advances row by row exactly as in plain decoding; only the stateless projections are batched.
`qwen4_graph_linear` in `ds4.c` (shared with Qwen3.8) is not modified.

### 4.5 MoE (stage 2, conditional)

The routed experts and the shared expert (a slot inside the MoE row kernels for T <= 8) are not changed in stage 1.
Stage 2 is attempted only if stage 1 meets its stop rule (section 6) but not criterion 1. It starts by measuring how
many of the 8 experts the two verify rows share per layer (a debug env, `DS4_QWEN35_VERIFY_OVERLAP=1`, logs the
count). If the shared fraction is >= ~40%, a 2-row MoE kernel that loads each shared expert once gets its own short
design addendum to this spec, reviewed by the user before any code; below that, stage 2 is dropped (it could not
save enough reads).

### 4.6 Knob

`DS4_QWEN35_VERIFY_BATCH`, read once (static), gates 4.3 and 4.4. It starts default off; after section 5's checks and
section 6's A/B pass it turns default on in a separate commit, with `=0` kept as the fallback to today's per-row
verify.

## 5. Exactness checks

- **Kernel, model-free** (`tests/test_qwen35_kernels.c`): for each Ornith Q8_0 shape, plus an out_dim that is not a
  multiple of NR0 x NSG and the lm head's NSG = 8 case, with two different input rows: row r of the batched output
  `memcmp`-equals a T=1 call on row r. Failing this blocks the rest.
- **Plain decode unchanged**: plain T=1 steps keep `kernel_mul_mv_q8_0_f32`; gate 1 dumps with the knob on equal
  develop's (0 of 39 differ).
- **Fallback if exactness cannot be reached** in 4.2: plain Ornith decode moves to an R = 1 instance of the same
  template (Ornith-only), gate 1 and gate 2 re-run, and the choice is recorded as a ruling. The `--mtp` guarantee is
  kept either way.
- **Graph, with the knob on**: `test_qwen35_mtp`, `tests/ornith/test_mtp_cli.py` (chunks 1/2/64 + long_copy,
  `--mtp` byte-identical to plain), `ds4_test --session-snapshot --qwen35-payloads --qwen35-rewind` with and without
  `DS4_TEST_GLM_MTP=1`.
- **Gate 1** x3 (chunks 2048 / 64 / 65) with the knob on; **Qwen3.8** fast gate and full gate.

## 6. Measurement and stop rules

- Profile (`DS4_QWEN35_PROFILE`): verify time over plain-step time; the stage 1 target is <= 1.35x (today 1.74x).
- Lever A/B: `m4_ab.py --mode lever`, A-B-B-A, knob off -> on, at 2K, 32K and 128K.
- Gate 3 against the live oMLX, then a re-run of `speed-bench/ornith/m6/h2h.py` for the report table.

Stop rules:

- After Task 1: if neither `decode_rows_exact` nor a prototype of 4.2 brings two rows below 1.5x one row on the large
  shapes, stop and report that the lever does not exist on this hardware.
- After stage 1: if the lever A/B gains < 5% at 2K, stop; stage 2 is not attempted.
- Stage 2 only as in 4.5.

Estimate (not a criterion): stage 1 removes ~70% of the verify's duplicate weight reads, which puts a verify near
1.25-1.35x a plain step and `--mtp` decode near 85-90 t/s at 2K and ~70-75 t/s at 32K.

## 7. Out of scope

- Non-exact kernels, and any change to plain T=1 arithmetic other than the fallback in section 5.
- Drafts deeper than 2 tokens (the per-row verify already refuses the after-second-row snapshot).
- Prefill, attention kernels, the `DS4_QWEN35_ATTN_MULTI_GEMV` lever (it keeps its per-row verify loop).
- Qwen3.8 paths and `qwen4_graph_linear`.
- Deployment (a separate `prod/<feature>-YYYYMMDD` decision).

## 8. Risks

- **Compiler reassociation** could make the batched per-row sums differ from the 1-row kernel's; the `memcmp`
  kernel test catches it, and the section 5 fallback bounds the damage.
- **ALU pressure**: the rows kernel doubles the multiply-adds per weight byte; if it becomes compute-bound on the
  lm head, Task 1's numbers show it before integration.
- **Scratch aliasing** in 4.4: the row views must not overlap the other row's live data; the graph identity tests
  (byte-identical `--mtp`) catch any mistake.
- **MoE near-ties** (seen in M6) cannot arise from an exact change; they can only matter under the section 5
  fallback, where gate 1 and gate 2 re-run.

## 9. Files

- `metal/qwen35.metal`: `kernel_qwen35_mv_q8_0_rows` (only if Task 1 does not pick `decode_rows_exact`).
- `ds4_metal.m`, `ds4_gpu.h`: `ds4_gpu_qwen35_matmul_q8_0_rows_tensor` and its pipeline entry (same condition).
- `ds4_qwen35moe.inc`: knob, `qwen35_gemv` batched branch, `qwen35_graph_linear_verify`.
- `tests/test_qwen35_kernels.c`: exactness tests; `tests/bench_qwen35_verify.c`: microbench (+ Makefile target).
- `speed-bench/ornith/m7/`: receipts (bench, profile, A/B, gates, h2h, report).
