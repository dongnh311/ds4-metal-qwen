# ISTA GSQ-RCO Q2_0 tier in ds4 — design

Date: 2026-10-02. Status: written for the user's review. The user asked for it on 2026-10-02 ("tải Q2_0 trước
đi", then "bắt đầu luôn").

Related documents:
- Parent: `docs/superpowers/specs/2026-09-28-qwen-nextgen-design.md`, the candidate row "ISTA GSQ-RCO
  IQ2_XS / Q2_0, speed / 1M fallback".
- Builds on sub-project 3: `docs/superpowers/specs/2026-09-30-gsq-rco-types-design.md`.

## Goal

Make ds4 run ISTA-DASLab's GSQ-RCO **Q2_0** tier of Qwen3.8-Flash-Next with the MTP head grafted from Ivan's
GGUF, so that sub-project 5 can measure it against PROD.

Why:
- ISTA IQ3_XXS with the refusal projection is level with PROD on the harness accuracy suites.
- It loses on total answer time: ifeval takes 165 min against PROD's 140 min, at 34.1 vs 37.4 t/s decode.
- The Q2_0 tier is 8.8 GiB lighter, and all of its routed experts use a lookup-free type.

## Why Q2_0 and not IQ2_XS

| | Q2_0 tier | IQ2_XS tier |
|---|---|---|
| shard 1 | 35.0 GiB | 36.5 GiB |
| task average (AIME25/GPQA-D/LCB v6; BF16 93.12, IQ3_XXS 92.57) | 89.07 | 89.16 |
| ISTA card, llama.cpp (hardware not stated) | decode 93.8 t/s, prefill 367 t/s | decode 70 t/s, prefill 108 t/s |
| routed gate/up | Q2_0 in all 48 layers | IQ2_S 34, IQ2_XXS 11, IQ1_M 3 (no IQ2_XS tensor at all) |
| routed down | Q2_0 in all 48 layers | Q2_0 in all 48 layers |
| missing in ds4 | Q3_K and Q5_0, dense only | IQ1_M, routed experts (MoE id kernels) |

- The IQ2_XS tier's experts are codebook (lookup) types, which decode slowly on Metal.
- The Q2_0 tier's experts already have SP3 kernels. Its gaps are dense tensors, where ds4 already has a
  generic GSQ path to extend.
- Both tiers are smaller than Ivan's IQ2 (41.7 GiB) and PROD (51.6 GiB).

## The Q2_0 file

- **File:** `ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`.
  - Size: 37,623,740,192 bytes.
  - LFS sha256: `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`.
  - HF commit: `8f752f8a`. The download script checks the size and sha256 before it renames the file.
- **Shard 2 is not needed.** It holds the IQ4_NL n-gram table. ds4 keeps using our `PLE-Q4_1` sidecar,
  which is the same base table and the one every current run uses.
- **Tensor types by role**, from the release's `tensor-allocation/*Q2_0*.rco-allocation.txt`:

  | role | types (count) |
  |---|---|
  | `ffn_{gate,up,down}_exps` | Q2_0 (48 each) |
  | `ffn_down_shexp` | Q2_0 16, Q4_0 16, Q5_0 7, IQ4_NL 7, Q8_0 2 |
  | `ffn_gate_shexp`, `ffn_up_shexp` | Q2_0, Q3_K, IQ4_XS, K-quants |
  | `attn_q/k/v`, `attn_qkv`, `attn_gate`, `ssm_out` | Q3_K, IQ4_XS, Q2_0, K-quants |
  | `attn_output` | Q4_K, Q5_K, Q6_K |
  | `token_embd` / `output` | Q3_K / Q5_K |
  | `blk.1.ple_key` | Q2_0 |
  | `hc_*`, `ssm_alpha/beta`, `indexer.*_proj`, `ffn_gate_inp*`, `ple_value` | BF16 |
  | norms, `ssm_a`, `ssm_dt`, `ssm_conv1d` | F32 (`ple_conv1d` F16) |

- **New block layouts** (upstream `ggml-common.h`):
  - Q3_K, type 11: 256 weights in 110 bytes (32-byte hmask, 64 bytes of 2-bit qs, 12 bytes of 6-bit
    scales, f16 d), 3.44 bpw.
  - Q5_0, type 6: 32 weights in 22 bytes (f16 d, 4 bytes of high bits, 16 bytes of nibbles), 5.5 bpw.
- **Input widths fit the blocks:** every Q3_K tensor reads 2560 or 6144 (`ssm_out`), both multiples of 256.
  Q5_0 sits only on `ffn_down_shexp`, which reads 640 (a multiple of 32).

## What ds4 already has (develop `260dc60a`)

- **Type table:** `gguf_types` already sizes `q5_0` (32, 22) and `q3_k` (256, 110) correctly.
  `tools/gguf_lite.py` lacks both.
- **Every other type in the tier runs today:**
  - Q2_0 as routed experts and dense, from SP3;
  - IQ4_XS, IQ4_NL, Q5_K and Q6_K dense, from SP3;
  - Q4_K, Q4_0, Q8_0, BF16, F16 and F32.
- **The GSQ group:** `qwen4_type_is_gsq` gates the types that share one machinery:
  - CPU dequantizers `dq_*` in `ds4_quants.h`;
  - Metal `qwen4_gsq_deq8`, which feeds `kernel_qwen4_gsq_mv_r{1..4}` for decode and MTP verify;
  - the staged prefill tiles, through `qwen4_mm_gsq_block`;
  - the dense tensor-op tiles `kernel_qwen4_dense_nax_<t>{,_n64,_n128}`;
  - the row-bytes table in `ds4_metal.m`.
- **`token_embd` rows are gathered on the CPU** (`qwen4_ref_row`), so a Q3_K `token_embd` needs only the
  CPU dequantizer.
- **The repack tool needs no changes:** `tools/repack_ista_qwen4.py` is tier-agnostic. It copies ISTA's
  tensors byte for byte and grafts Ivan's `blk.48.*`. It converts `ffn_gate_inp*` from BF16 to F32. It
  converts the hc mixer weights from BF16 to F16 under the 1e-6 rule.

## Design

### Engine

1. **Types:** Q3_K and Q5_0 join the GSQ dense set, but not the routed-expert set.
   - Today `qwen4_graph_expert_ok` and `qwen4_expert_type_ok` accept every GSQ type as a routed expert.
     They get a predicate that excludes Q3_K and Q5_0, and the loader rejects a routed Q3_K or Q5_0
     tensor with a clear message.
   - Shared-expert tensors (`ffn_*_shexp`) carry both new types. The plan traces which predicate and which
     kernels the shared-expert path uses, and adds both types there.
2. **CPU:** `dq_q3_K` and `dq_q5_0` in `ds4_quants.h`.
   - They are ported from `ggml-quants.c` at the SP3 commit `931351ea`, keeping the MIT notice.
   - They are the reference for every test and the CPU path's way to read these tensors.
   - They also serve the CPU rows that call `qwen4_ref_row` / `qwen4_ref_matvec`.
3. **Metal, dense:** each piece below gains cases for type 11 and type 6.
   - `qwen4_gsq_deq8`;
   - the type switch in `kernel_qwen4_gsq_mv`;
   - the row-bytes table;
   - `qwen4_mm_gsq_block` (11 → 256, 6 → 32);
   - the dense tensor-op tile instances.

   The plan decides per kernel, as SP3 did. Where a kernel cannot take a layout, that tensor dispatches the
   generic GSQ path.
4. **Routed Q2_0 gate/up:** no new code is expected. SP3 tested Q2_0 only as `ffn_down_exps` (640 → 2560).
   The gate/up shape (2560 → 640) gets its own parity test on the decode (`mul_mv_id`) and prefill
   (MoE mm) paths.
5. **Default path:** pipelines are created only for types the model uses. PROD and ISTA IQ3_XXS run
   byte-identically, because every change is keyed on the tensor type.

### Repack

- **Run:** `tools/repack_ista_qwen4.py`, unchanged, on Q2_0 shard 1 plus Ivan's GGUF.
- **Output:** `ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-DS4-MTP.gguf`, about 36.5 GiB, with a
  manifest next to it.
- **Tool change:** `gguf_lite.BLOCK` gains types 6 and 11.
- **Verification**, as in SP3:
  - every copied tensor's sha256 equals its source's;
  - every converted tensor matches its source value for value, or within the 1e-6 hc rule;
  - ds4 parses the header.
- **HF:** after the exit check, a new dongnhdev repo for this file, with the user's OK at that time.

### Where the work happens

- **Branch:** `feature/ista-q2_0`, cut from develop `260dc60a`, in the `kv-grow` worktree (this session's
  own). No harness run currently uses its binaries.
- **Merge:** into develop with the user's approval.

## Testing

- **CPU dequantizers:**
  - `tools/gen_quant_fixtures.py` adds types 6 and 11, cutting two real blocks each from Q2_0 shard 1.
  - The expected values come from gguf-py's `quants.dequantize`, which covers both types.
  - ds4 must match exactly.
  - The type-size test pins both entries against upstream.
- **gguf_lite:** unit tests for the two new sizes.
- **Metal parity:** outputs against the CPU f32 reference on real rows, in `ds4_test --metal-kernels`,
  using SP3's per-path tolerances.
  - Q3_K and Q5_0 on every dense path they take: `gsq_mv` r1 to r4, the staged prefill tiles, and the
    tensor-op tiles at 32, 64 and 128 tokens.
  - Q2_0 on the routed gate/up shape.
- **Loader:** a header-only test that the repacked file passes the qwen4 type and layout checks.
- **No regression:**
  - The PROD identity check: greedy outputs byte-identical to the develop build.
  - The same check on ISTA IQ3_XXS hc-F16, 3 prompts.
  - The existing groups: `ds4_test --metal-kernels` and `python3 -m pytest -q tools`.
- **End to end:** needs a GPU window with the gateway paused, after GLM53F frees the GPU.
  - The repacked model loads and answers three prompts coherently.
  - Its MTP acceptance is logged.
  - ds4-bench records 8K decode and prefill.
  - Perplexity is measured on the same 6 texts as the 2026-10-01 comparison (PROD, Ivan, ISTA IQ3_XXS).
  - Wired RAM is reported.

## Milestones

| # | deliverable | GPU |
|---|---|---|
| M1 | Q3_K/Q5_0 in the GSQ dense set, routed-expert predicate, CPU dequantizers, fixtures, `gguf_lite` | no |
| M2 | repack on the real files, sha256 verified | no, after the download |
| M3 | Metal dense paths for Q3_K/Q5_0, Q2_0 gate/up parity | minutes |
| M4 | end to end, identity checks, speed and perplexity numbers | ~1 h, gateway paused |

M1 and M2 run while GLM53F holds the GPU. M3 and M4 wait for its "GPU free" and the user's go-ahead.

## Exit check

- Parity passes for Q3_K and Q5_0 on every dense path, and for Q2_0 gate/up.
- The repacked Q2_0 file loads, answers coherently and reports its MTP acceptance.
- The PROD and ISTA IQ3_XXS identity checks pass.
- Decode, prefill, perplexity and RAM are recorded.

## After the exit check

- **Next step if it passes:** a harness arm against PROD with the projection at FFN 0.5, the same flags as
  the IQ3_XXS arm. That is the sub-project 5 gate, and the user decides whether to run it.
- **Pre-filter before the ~6 h arm:** if the Q2_0 perplexity is worse than Ivan's IQ2 on most of the 6
  texts, report and ask first. Ivan's file uses PROD's quant recipe.

## Out of scope

- **Fused or M5-tuned kernels for Q2_0 experts.** This is the first follow-up if decode is short. Q2_0 is
  lookup-free, so a pair-swiglu fusion like the IQ2_XXS one is cheap.
- **Other lighter tiers and their types:** the IQ2_XS tier, IQ1_M, and Q3_K/Q5_0 as routed experts.
- **Other work:** prompt-lookup drafts (sub-project 6), CPU inference of the model, and the 512K check
  (sub-project 5).

## Risks

| risk | detection | fallback |
|---|---|---|
| Q2_0 quality below PROD (published 95.6% of BF16, against 99.4% for IQ3_XXS) | perplexity, then the harness | keep IQ3_XXS / PROD; the user's rule drops a dumber model |
| unfused Q2_0 expert kernels leave decode short | ds4-bench in M4 | fused Q2_0 gate/up (follow-up) |
| Ivan's MTP head accepts less on the lower-precision trunk | MTP acceptance in M4 | reported; prompt-lookup drafts (sub-project 6) |
| the shared-expert path validates or dispatches by another predicate than the dense path | loader test, M3 parity | the plan traces it and adds both types there |
| a dense tile kernel cannot take Q3_K or Q5_0 | M3 parity | that tensor dispatches the generic GSQ path |

## Constraints

- **Storage:** Mac only. Git holds source and small fixtures; GGUFs live in `ds4-metal-data` and on HF.
- **GPU runs:** no GPU work without the user's go-ahead and GLM53F's "GPU free". Pause the gateway stack for
  model runs and restore it afterwards.
- **Process handling:** never `kill -9` a Metal process.
- **Language:** code, docs and commits in English.
