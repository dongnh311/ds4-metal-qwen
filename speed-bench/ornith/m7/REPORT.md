# Ornith M7 report: exact batched MTP verify

Branch `feature/ornith-m7` (from develop `b764a66`), 2026-09-28. Spec
`docs/superpowers/specs/2026-09-28-ornith-m7-verify-batch-design.md`, plan
`docs/superpowers/plans/2026-09-28-ornith-m7-verify-batch.md`.

## Result

ds4 now decodes Ornith faster than the live oMLX at every measured context, on top of M6's lead in prefill at long
context, memory and quality (gate 3, same day, A-B-B-A):

| context | ds4 decode t/s (M6 -> **M7**) | oMLX decode t/s | ds4 TTFT s | oMLX TTFT s |
|---|---|---:|---:|---:|
| 2K | 69.5 -> **93.0** | 80.7 | 1.2 | 1.2 |
| 32K | 53.1 -> **68.3** | 64.8 | 19.6 | 20.2 |
| 128K | 40.1 -> **50.4** | 39.9 | 143.9 | 177.6 |

Spec §1 criteria 1-4: met (details in `speed/GATE3.md`, `LEVERS.md`, `QWEN_GATE.md`).

Head-to-head sweep (`h2h/`, A-B order, oMLX first, one cold request per length, two at <= 32K; medians):

| prompt | ds4 TTFT | oMLX TTFT | ds4 decode | oMLX decode | ds4 memory* | oMLX memory |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 1.2 s | 1.2 s | **92.5** | 80.9 | 28.2 GiB | 26.0 GiB |
| 8K | 4.1 s | 4.1 s | **77.4** | 76.9 | 28.2 GiB | 28.0 GiB |
| 32K | 20.9 s | 20.2 s | **68.9** | 65.0 | 28.2 GiB | 32.0 GiB |
| 64K | 53.8 s | 54.5 s | **59.6** | 53.2 | 28.2 GiB | 37.0 GiB |
| 128K | **153.8 s** | 347.6 s | **49.3** | 39.5 | 28.2 GiB | 44.0 GiB |

\* process footprint + the 21.26 GiB model ds4 maps outside it (see `../m6/plot_h2h.py`). The 192K and 250K rows
were not re-run. M6 measured them (`../m6/h2h.txt`): oMLX aborted the 192K request, and ds4 decoded 31.8 / 28.6 t/s
there with M6's per-row verify. M7 should raise those too (the A/B gain was +21% at 128K), but that is an
expectation, not a measurement.

## What changed

The 2-row MTP verify used to stream every weight twice (one T=1 dispatch per row, to stay bit-identical to plain
decoding), costing ~1.55-1.74x a plain step. M7 adds `kernel_qwen35_mv_q8_0_rows2`: `kernel_mul_mv_q8_0_f32`'s K walk,
per-row multiply-adds and reduction tree for two input rows, each weight block loaded once. The verify runs its Q8_0
projections through it: attention q/output, the GDN `lin_qkv`/`lin_gate`/`lin_out` (a new Ornith-local GDN verify
layer keeps the recurrence one row at a time) and the lm head. F16/F32 projections and the experts stay per row.
`DS4_QWEN35_VERIFY_BATCH`, default on; `=0` restores the per-row verify. Because the shader library is compiled
at run time with fast math, the first MTP session re-checks the two-row matvec against two one-row dispatches on
the lm head and one GDN projection; a mismatch, or any two-row call that fails, turns the batched verify off for
the process (a stderr line says so), so `--mtp` output never depends on the compiler matching.

| | one T=1 call | today's two T=1 calls | two-row kernel |
|---|---:|---:|---:|
| lm head 2048x248320 | 1.84 ms | 3.66-3.69 ms | 1.85 ms |
| 2048x8192 | 0.066-0.085 ms | 0.106 ms | 0.067 ms |

Verify cost (profiled): 1.55x -> 1.21x a plain step. Lever A/B: decode +30.6% (2K), +24.5% (32K), +21% (128K),
prefill unchanged (`LEVERS.md`).

## Exactness

- Kernel: row r of the two-row matvec `memcmp`-equals a T=1 call on row r (5 shapes incl. NSG 8 and odd out_dim, x3).
- Graph: `test_qwen35_verify_batch` (new) checks every cycle's logits against a plain session byte for byte and counts
  exactly 111 two-row matvecs per verify (0 with `=0`); `test_qwen35_mtp`, `tests/ornith/test_mtp_cli.py` (`--mtp`
  byte-identical to plain), `ds4_test` rewind/snapshot/payloads with and without MTP, `test_qwen35_graph`: pass.
- Plain decode is byte-unchanged: gate 1 dumps equal develop's (0 of 13 differ); gate 1 x3 PASS.
- Gate 2: M7's output is byte-identical to M6's (plain decode unchanged, `--mtp` identical to plain), so M6's gate 2
  (`../m6/quality/GATE2.md`: eval 83.7 / 85.0 vs oMLX 78.8) stands.

## Stage 2 (MoE)

Not needed: criterion 1 is met by stage 1, so the conditional Task 7 (expert-overlap measurement, 2-row MoE
addendum) was not run. It remains the next decode lever if more speed is wanted (the verify's routed experts are
still read once per row).

## Decisions for the user

1. Merge `feature/ornith-m7` into develop (not pushed; the branch holds the spec, plan, code and receipts).
2. Deploy: Ornith on ds4 can now replace Ornith on oMLX on every axis measured here; a deploy goes through
   `prod/<feature>-YYYYMMDD` cut from develop with `deploy-ai-gateway.sh`, on the user's go-ahead.
