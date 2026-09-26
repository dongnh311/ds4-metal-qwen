# Ornith M4 decode levers

Rule (plan fix 17): a lever is *exact* only if the gate-1 dumps (`tests/ornith/gate1.py --dump-logprobs`, 13 prompts)
are byte-identical (`cmp`) to the Task 4 reference at prefill chunks 2048, 64 and 65, and `make test-qwen35-graph`,
`make test-qwen35-mtp` and `tests/ornith/test_mtp_cli.py` pass with the lever on. Exact levers are A/B-measured
ds4-vs-ds4 with `speed-bench/ornith/m4_ab.py --mode lever` (A-B-B-A, live stack paused, `--mtp`, 256 tokens).
L12 and the KV modes change numerics by design and are held to gate 1 (tolerance), the MTP byte-identity tests and
gate 2 instead. Builds: snapshots of `0eb56b2` (levers window) and `60bb430` (Tasks 10-11 window), 2026-09-27.

| lever | knob | exact | A/B (decode t/s, base -> lever) | decision |
|---|---|---|---|---|
| L1 early flush after layer 2 | `DS4_QWEN35_FLUSH_LAYER` (default 2; -1 off) | yes (0/39 dumps differ) | 2K 62.7 -> 65.9 (+5.1%), 32K 42.3 -> 42.7 | kept (default) |
| L2 add+RMSNorm fusion (T=1) | `DS4_QWEN35_FUSE_NORM` (default 0) | yes | 2K 62.0 -> 61.4, 32K 42.4 -> 41.5 | rejected |
| L4 q/k/v multi-GEMV | `DS4_QWEN35_ATTN_MULTI_GEMV` (default 0) | no (39/39 dumps differ; gate 1 still PASS) | not measured | rejected (fix 17) |
| L5 paired GDN | — | — | — | dropped (fix 18: needs `!mtp_R`, Ornith PROD runs `--mtp`) |
| L8 Q5_K 1-row mid / 4-row down decode kernels | `DS4_QWEN35_MOE_MR_MID/_DOWN` (M5 default 1/4) | yes (kernel test bit-identical for NR 0/1/2/4) | 2K 61.0 -> 63.3 (+3.8%), 32K 42.5 -> 42.3 | kept (M5 default) |
| L10 MTP draft vocabulary (64K ids, reused from Qwen3.8, same tokenizer) | `DS4_QWEN35_MTP_DRAFT_VOCAB=<file>` | output exact (verify uses the full head; MTP tests pass) | CLI `--mtp` 2K Italian prose: 49.1 -> 58.3 t/s (+19%), acceptance 56.8% -> 51.8%; lever A/B aborted by a harness guard false positive (fixed in 49c2f99) | kept (deploy env) |
| L12 shared-K/V 2-row verify decode | `DS4_QWEN35_ATTN_DECODE2` (default 1; 0 off) | by construction (memcmp 2-row vs 1-row calls); gate 1 PASS at 2048/64/65; MTP tests pass | 32K 43.1 -> 45.2 (+4.9%); 128K CLI with q4: 19.3 -> 20.6 | kept (default) |
| KV fp8 | `DS4_QWEN35_KV=fp8` | gate 1 PASS; MTP tests pass | 32K decode 43.2 -> 22.2, prefill 650 -> 273 | rejected |
| KV q4 | `DS4_QWEN35_KV=q4` | gate 1 PASS; MTP tests pass | 32K decode 43.5 -> 42.3, prefill 651 -> 623 | rejected |

Why so little: the stage profile (`../profile/PROFILE.md`) puts long-context time in attention — prefill attention
grows to 96% of a 128K prefill and decode attention is ~32 ms/token at 128K (3x over the K/V bandwidth floor). None
of the planned levers touch the attention kernels' efficiency; the shared qwen4 FP8/4-bit K/V attention paths are
slower than the F16 path for Ornith's 2-KV-head x 256-dim shape. A faster Ornith attention (flash-style prefill,
a decode kernel near the bandwidth floor) is the lever that remains.
