# ISTA hc-F16 overnight: harness arm vs PROD, F16 tiles dropped

With its hc mixer weights in F16, ISTA plus the refusal projection is now level with PROD on the
harness's accuracy suites. Run on the same items, ISTA passes 10 that PROD fails and PROD passes 14
that ISTA fails (McNemar p = 0.54). The reasoning gap from the first ISTA arm was mostly run-to-run
noise: two ISTA arms on near-identical weights differ by 10 items to 4.

PROD still decodes faster in the harness: 37.4 vs 34.1 t/s on ifeval. ds4 and ds4-bench show ×1.12
for the new file, but its harness decode did not move (×0.99 paired against the old file's arm). Why
is still open; see the end of this note.

GPU windows:
- 2026-10-01 23:15 to 2026-10-02 07:41, gateway stack paused (hold-gateway.sh, overnight autonomy);
- 07:44 to 07:49 for the server A/B.

The gateway was restored at 07:43:54 and 07:51:51 (backend_ok). Swap-outs stayed flat all night.

Binary: develop `d1b56556` plus the F16 tile opt-in below (local commit `73ac1415`, dropped
afterwards). ds4-eval was rebuilt from the same tree at 23:30.

## F16 dense prefill on the tensor-op tiles: no gain, dropped

`DS4_QWEN4_DENSE_NAX_F16=1` put F16 dense batches of whole 32-token tiles on the tensor-op tiles.
The kernel tests passed by default, with the opt-in, and with `DS4_QWEN4_DENSE_NAX=0`.

| ISTA hc-F16 file, ds4-bench 8K, 128 generated | opt-in off | opt-in on |
|---|---|---|
| prefill t/s, one 8K chunk (3 rounds) | 786.78 / 809.26 / 806.28 | 807.03 / 809.01 / 805.17 |
| prefill t/s, `--prefill-chunk 2048` | 695.87 | 696.15 |
| decode t/s (median) | 33.50 | 33.64 |

- **Greedy output:** 3/3 prompts identical with the opt-in off and on.
- **GPU vs CPU at chunk 64:** 3.393 with the opt-in. That is within the 5.31 bar, but above the 2.987
  measured without it.
- **PROD output:** identical on 3/3 prompts.
- **PROD prefill:** the off/on pairs read 539.6/574.3 and 567.0/600.6. That is not the opt-in:
  - this bench never reaches the F16 tile path (2048-token chunks of an 8192-token prompt);
  - "off" always ran first in each pair, and PROD streams experts from SSD, so the second run of each
    pair finds a warmer page cache.

Why there is no gain:
- `qwen4_gemv_rows` sends F16 batches to `ds4_gpu_qwen4_dense_mm_tensor` only from 4 to 64 tokens.
- Prefill batches go to `ds4_gpu_matmul_f16_tensor`, which was already fast.
- The 1.2 s per 8K forward in the narrow-output spike was the old file's F32 hc_up. That path is
  `dense_mm` above 8 tokens, and the F16 file had already removed it.

**Ruling:** dropped. The commit is kept on the local branch `wip/ista-f16-nax-optin` only. The
2026-10-01 note's "new lever" paragraph is corrected.

## MTP, warm, old vs new file

Each file ran as a block: one warm-up generation, then 3 prompts, `-n 256`, temperature 0.

| prompt | old file (F32 hc) | new file (F16 hc) |
|---|---|---|
| prime function (EN code) | 40.65 t/s, 98.0% | 46.17 t/s, 98.0% |
| blue sky (VI) | 37.22 t/s, 77.3% | 41.57 t/s, 80.0% |
| TCP vs UDP (EN) | 38.83 t/s, 85.3% | 42.80 t/s, 85.0% |
| mean | 38.90 | 43.51 |

The gain is ×1.12, the same as without MTP. The ×1.24 in the 2026-10-01 note came from old-file runs
that were slowed by page-in.

## Harness arm `ista-hcf16-proj-s050-20261001`

The arm is the new file plus the projection at FFN 0.5, with `DS4_QWEN4_MTP_DRAFT_VOCAB=""` (full head)
and the F16 opt-in on. The suites were ifeval, reason, code, longctx, tools and vi; uncensor was not
rerun, because the hc change cannot move it. The arm took 5.9 h.

| suite | ISTA hc-F16 + proj | ISTA (first arm) | Ivan + proj | PROD |
|---|---|---|---|---|
| ifeval (200) | 189 | 189 | 194 | 194 (`prod-ifeval200-20261001`) |
| reason (44) | 41 | 38 | 43 | 42 |
| code (67) | 65 | 64 | 66 | 65 |
| tools_neg (30) | 25 | 25 | 17 | - (6-item set: ISTA 3, PROD 2) |
| tools_pos / tools_xfer | 7/8, 4/6 | 7/8, 3/6 | 6/8, 0/6 | 8/8, 2/6 |
| vi_knowledge / faithfulness | 30/30, 9/9 | 30/30, 9/9 | 30/30, 9/9 | 30/30, 9/9 |
| longctx | needles 120K/240K hit, docqa 3/3, 52.75 GiB wired, 0 swap | docqa 2/3 | docqa 3/3 | docqa 3/3, 49.48 GiB |

### Paired comparison (same items, McNemar exact; A = ISTA hc-F16 + proj)

| B | A-only | B-only | p | per suite |
|---|---|---|---|---|
| PROD (ifeval-200 + baseline + tools_neg run) | 10 | 14 | 0.54 | ifeval 4 vs 9 (p 0.27); reason 1 vs 2; code 1 vs 1 |
| Ivan + proj | 18 | 13 | 0.47 | ifeval 3 vs 8; reason 0 vs 2; tools_neg 9 vs 1 (p 0.02) |
| ISTA first arm | 10 | 4 | 0.18 | reason 3 vs 0; ifeval 3 vs 3 |

On ifeval-200, PROD and Ivan + proj split 2 vs 2.

### Speed (medians per row)

| | ISTA hc-F16 + proj | PROD |
|---|---|---|
| ifeval decode t/s | 34.09 | 37.37 |
| ifeval prefill t/s | 141 | 81 |
| code / vi decode t/s | 35.3 / 35.2 | 39.9 / 39.4 (baseline arm) |

Paired against the first ISTA arm, ifeval decode is ×0.99. That ifeval rerun ran on kernels of the
same generation. The other suites read ×1.23-1.29, but they also gained the gemv fixes that the first
arm's code/vi/tools runs lacked.

## Open: the harness does not show the hc-F16 decode gain

`ds4-server` A/B (07:45):
- each case is a fresh server with the arm's flags, but no disk KV;
- one warm-up request, then a 1200-token thinking request, temperature 0;
- the projection is on in the "steer" cases.

| | new file | old file |
|---|---|---|
| steer | 43.62 t/s | 40.53 t/s |
| no steer | 43.24 t/s | 37.41 t/s |

So the server gains +8 to +16%, and the projection is not what hides it.

In the harness, the same ifeval item decodes at matching 50-token chunk speeds with either file:

| old file chunks | new file chunks |
|---|---|
| 36.3, 35.8, 33.1, 36.3, 37.5 | 39.6, 36.6, 39.5, 35.1, 36.5 |

The harness ratio stays at ×0.99 in every 25-item window of run order, so heat is not the cause.

What the harness does differently, any of which may cap decode near 34 t/s for both files:
- disk KV cache flags (`--kv-disk-dir`, `--kv-disk-space-mb 32768`, `--kv-cache-cold-max-tokens 262144`),
  which the gateway rows also use;
- `max_tokens` 16384;
- ifeval content.

Next step: the same A/B with the arm's exact argv (disk KV on) and an ifeval item. One variable at a
time.
