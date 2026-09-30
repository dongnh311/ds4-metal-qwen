# Sub-project 3: GSQ-RCO tensor types and the ISTA IQ3_XXS repack — design

Date: 2026-09-30. Status: design approved in conversation (three sections, "ok ổn" each). Parent
design: `docs/superpowers/specs/2026-09-28-qwen-nextgen-design.md` (sub-project 3 row). Branch:
`feature/nextgen-sp3`, cut from `feature/nextgen-qwen`.

## Goal

Make ds4 run ISTA-DASLab's GSQ-RCO IQ3_XXS quant of Qwen3.8-Flash-Next with the MTP head grafted from
Ivan's GGUF, correct first and fast later (user decision 2026-09-30: "đúng trước, nhanh sau"), so that
sub-project 5 can measure it against PROD.

## The ISTA IQ3_XXS file (measured 2026-09-30)

- Shard 1: `ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`,
  47,039,860,096 bytes, sha256 `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`
  (equals the HF etag). Shard 2 is the n-gram PLE table at IQ4_NL; ds4 keeps using the existing
  `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` sidecar (the same base table), so shard 2 is not needed.
- Header: GGUF v3, `qwen4exp`, 1223 tensors, `block_count` 48, `expert_count` 512, 2560 embedding,
  expert and shared-expert FFN 640, no `blk.48` (no MTP), `split.count` 2.
- Tensor types by role (the release's `tensor-allocation/*.rco-allocation.txt`; count = layers):

  | role | types |
  |---|---|
  | `ffn_gate_exps`, `ffn_up_exps` (same type per layer) | IQ3_S 13, IQ2_XS 10, IQ2_S 10, IQ2_XXS 9, IQ3_XXS 6 |
  | `ffn_down_exps` (640 input rows, unpadded) | Q2_0 30, IQ4_NL 18 |
  | `ffn_down_shexp` | IQ4_NL 40, Q2_0 8 |
  | `ffn_gate_shexp`, `ffn_up_shexp` | IQ4_XS, IQ3_S, Q4_K, Q5_K, Q6_K |
  | `attn_q/k/v/output`, `attn_qkv`, `attn_gate`, `ssm_out` | Q6_K, IQ4_XS, Q4_K, Q5_K, IQ3_S |
  | `token_embd` / `output` | IQ3_S / Q5_K |
  | `hc_*`, `ssm_alpha/beta`, `indexer.*_proj`, `ffn_gate_inp`, `ffn_gate_inp_shexp` | BF16 |
  | norms, `ssm_a`, `ssm_dt`, `ssm_conv1d` | F32 |

- Type ids: IQ2_XS 17, IQ3_XXS 18, IQ4_NL 20, IQ3_S 21, IQ2_S 22, IQ4_XS 23, Q2_0 42 (upstream
  `block_q2_0`: 64 weights, one f16 scale and 16 bytes of 2-bit values, 18 bytes, 2.25 bpw).
- Bytes: routed experts 42.4 GiB, the other quantized tensors 2.1 GiB. Weight bytes read per decoded
  token (layers 0..47, token_embd excluded, 10 of 512 experts): PROD 4.85 GiB, ISTA with native kernels
  4.42 GiB, ISTA with its dense tensors transcoded to Q8_0 5.53 GiB. The dense tensors dominate, which
  is why every type gets a native kernel (user decision: "kernel gốc cho cả dense").
- Ivan's GGUF supplies what ISTA lacks: 32 `blk.48.*` tensors (1.40 GiB; F32, F16, Q8_0, Q4_K,
  MXFP4, all types ds4 already runs) and the keys `general.alignment`, `qwen4exp.nextn_predict_layers`,
  `qwen4exp.vocab_size`, `qwen4exp.ple.{row_count,row_dimension,seed,vocab_base,vocab_divisor}`,
  `ds4.qwen4.down.{logical_input,physical_input}` (its `ds4.iq2.imatrix_sha256` does not apply).

## What ds4 already has

- **The GGUF type table** (`gguf_types`, ds4.c) sizes every tensor. Entries 17, 18, 21, 22 and 23
  match upstream. Two are wrong: `iq4_nl` says 256 weights / 50 bytes (upstream: 32 / 18) and `iq1_s`
  says 110 bytes (upstream: 50). Q2_0 (42) is missing. With the table as it is, the ISTA file fails to
  load: its IQ4_NL down rows (640) are not a multiple of 256.
- **qwen4 type gates**: `tensor_type_is_qwen4_dense` and `qwen4_graph_dense_ok` (Q8_0, Q4_0, Q4_K,
  F16, BF16, F32), `qwen4_graph_expert_ok` (plus Q2_K and IQ2_XXS with `dim[0] % 256 == 0`, and MXFP4),
  `qwen4_expert_type_has_mm` (the tiled prefill GEMM), and `routed_expert_block_bytes`.
- **Fixed loader types**: norms, `ssm_a`, `ssm_dt`, `ssm_conv1d` and `ffn_gate_inp` must be F32
  (`tensor_expect_layout(..., DS4_TENSOR_F32, ...)`). ISTA stores `ffn_gate_inp` as BF16.
- **The down width**: the loader reads `ds4.qwen4.down.*` and pads 640 to 768 for Q2_K and Q4_K down.
- **Metal**: `metal/moe.metal` carries ggml's templates (`kernel_mul_mv_id<mmv_fn<...>>`,
  `kernel_mul_mm_id<..., block_T, QK_NL, dequantize_T, ...>`), with `mul_mm_id` instances for Q5_K and
  Q6_K already there, and M5-specialized fused IQ2_XXS and MXFP4 kernels (pair-swiglu, sum6, addr,
  cached). `metal/qwen4.metal` has dense kernels that switch on `weight_type` (f32, f16, q8_0, q2_K,
  q4_K, iq2_xxs, mxfp4). `ds4_metal.m` creates one pipeline per kernel name, eagerly.
- **CPU**: IQ2_XXS grid tables and matvec workers in ds4.c; no dequantizer for the new types.
- **Upstream (ggml-org/llama.cpp, MIT)**: `ggml/src/ggml-metal/kernels/{dequantize.h,mul_mv.metal,
  mul_mm.metal}` have `dequantize_*`, `kernel_mul_mv_*_f32` and `kernel_mul_mv_id_*_f32` for all nine
  types (IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0, Q5_K, Q6_K); `ggml/src/ggml-common.h`
  has the block structs and grids; `gguf-py/gguf/quants.py` dequantizes all of them except Q2_0.

## Design

### Engine

1. **Types.** Fix the `iq4_nl` and `iq1_s` entries, add `q2_0` (64, 18), add the enum values and the
   block structs, and port the grids (`iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`,
   `ksigns_iq2xs`, `kmask_iq2xs`, `kvalues_iq4nl`) from `ggml-common.h`, keeping the MIT notice.
2. **CPU.** One row dequantizer per new type, ported from `ggml-quants.c`: the reference for every
   test, and the CPU path's way to read these tensors. Full CPU inference of the ISTA model is not a
   goal.
3. **Metal, routed experts** (IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S for gate/up; IQ4_NL, Q2_0 for down): the
   ggml template instances `kernel_mul_mv_id_<t>_f32` (decode and MTP verify) and
   `kernel_mul_mm_id_<t>_{f32,f16}` (prefill), with the `dequantize_<t>` and `kernel_mul_mv_<t>_f32_impl`
   functions they need. Gate and up run as two products followed by the plain swiglu; no fused
   variants for the new types. The plan lists exactly which variants the qwen4 MoE graph calls for a
   type without fused kernels, and routes the new types there. The nine IQ2_XXS layers keep the
   existing fast kernels.
4. **Metal, dense** (IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q5_K, Q6_K): add the types to the qwen4 dense
   kernels' `weight_type` switch for decode and prefill, or, where a kernel cannot take a new block
   layout, dispatch the ggml `kernel_mul_mv_<t>_f32` / `kernel_mul_mm_<t>_f32` instead; the plan
   decides per kernel. `get_rows` gains IQ3_S for `token_embd`; `output` runs Q5_K.
5. **Pipelines.** New pipelines are created only when the loaded model uses the type, so the startup
   of today's models does not change.
6. **Graph wiring.** The type gates accept the new types; the expert block check becomes
   `dim[0] % block_elems == 0` (640 works for IQ4_NL and Q2_0 only, which is where ISTA uses it);
   `routed_expert_block_bytes` knows the new blocks; the down width takes
   `logical = physical = 640` from the file without padding.
7. **SSD streaming.** Expert bytes now differ per layer. The plan checks every place that sizes a
   streamed expert or a cache slot and makes it per-layer where it assumes one size.
8. **Default path.** Models without the new types run byte-identically: every change is behind the
   tensor type.

### Repack tool (`tools/repack_ista_qwen4.py`)

- Pure Python, streaming: reads both GGUF headers, writes the new header, then copies tensor data in
  chunks; never holds a tensor in memory beyond one chunk.
- Output tensors: ISTA's 1223 byte for byte, then Ivan's 32 `blk.48.*` byte for byte. The one
  conversion is `ffn_gate_inp` BF16 to F32 (exact), because the loader requires F32; the plan audits
  every fixed-type loader check against ISTA's types, and any other exact conversion it finds is added
  the same way.
- Metadata: ISTA's keys, minus `split.*`, with `block_count` 49, a 49-entry
  `attention.compress_ratios` (layer 48 = Ivan's value), and Ivan's `general.alignment`,
  `nextn_predict_layers`, `vocab_size`, the five `ple.*` keys, and `ds4.qwen4.down.logical_input` =
  `physical_input` = 640.
- A manifest JSON next to the output, shaped like Ivan's: sources with sha256, and per tensor its type,
  bytes, sha256 and whether it was converted.
- Output: `ds4-metal-data/gguf/ista/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-DS4-MTP.gguf`, about 45.3 GiB.
- Verification: every copied tensor's sha256 equals its source's; a converted tensor equals its
  source value for value; ds4 parses the header.
- HF: upload to dongnhdev after the exit check passes, with the user's OK at that time.

### Where the work happens

A new worktree `~/orca/workspaces/ds4-metal/sp3` on `feature/nextgen-sp3`. The kv-grow worktree's
binaries and Metal sources serve the sub-project 4 night run and must not change before it ends.
The branch merges back into `feature/nextgen-qwen` with the user's approval.

## Testing

- **CPU dequantizers**: fixtures of a few real blocks per type cut from the ISTA file, with the
  expected floats produced by an independent reference (`gguf-py`'s `quants.py`; for Q2_0, ggml's C
  `dequantize_row_q2_0` compiled standalone). The references run only when fixtures are generated.
  ds4's dequantizers must match exactly. The type table test pins the sizes of all entries against
  upstream, including the two fixed ones.
- **Metal parity**: for each type and each path it uses (`mul_mv`, `mul_mv_id`, `mul_mm`,
  `mul_mm_id`, `get_rows`), outputs against the CPU f32 reference on real ISTA rows, in `ds4_test
  --metal-kernels`, with per-path tolerances written in the plan.
- **Loader**: a header-only test that the repacked file passes the qwen4 type and layout checks.
- **Repack**: unit tests on small synthetic GGUFs (header rewrite, alignment, the BF16-to-F32
  conversion, manifest), and the sha256 verification on the real output.
- **No regression**: the identity check on PROD's model (outputs byte-identical to the base build),
  the existing CPU test groups, and the harness suite.
- **End to end (GPU, gateway paused)**: the repacked model loads, answers three prompts coherently,
  logs its MTP acceptance, and ds4-bench records decode and prefill tokens per second. These numbers
  are reported, not gated; sub-project 5 gates them.

## Milestones

| # | deliverable | GPU |
|---|---|---|
| M1 | type table fix, types, grids, CPU dequantizers, fixtures | no |
| M2 | repack tool and manifest, run on the real files, sha256 verified | no |
| M3 | Metal expert kernels for the six expert types, parity tests | minutes |
| M4 | Metal dense kernels, IQ3_S get_rows, Q5_K output, parity tests | minutes |
| M5 | graph wiring, streaming sizes, end-to-end run, PROD identity check | night, gateway paused |

GPU windows need the user's go-ahead. M3/M4 parity runs are short and small; M5 loads 45 GiB and runs
at night like sub-project 4.

## Exit check

Kernel parity passes for every new type on every path; the repacked ISTA IQ3_XXS loads, answers
coherently and reports its MTP acceptance; the PROD identity check passes; decode and prefill speed
are recorded.

## Out of scope

- Fused kernels and M5 tuning for the new types (only if sub-project 5 shows speed is short).
- Types used only by the other ISTA tiers (IQ1_M, Q3_K, Q5_0) and the Coder variant.
- CUDA and ROCm kernels; CPU inference of the ISTA model.
- Refusal behaviour (sub-project 2) and the comparison with PROD (sub-project 5).

## Risks

| risk | detection | fallback |
|---|---|---|
| IQ lookups make decode slower than PROD despite fewer bytes | ds4-bench in M5 | fused kernels and M5 tuning as a follow-up; lighter tiers in sub-project 5 |
| a qwen4 dense kernel cannot take a block-32/64 or IQ layout | M4 parity | dispatch ggml's generic `mul_mv`/`mul_mm` for that tensor |
| BF16 small tensors hit a kernel that only takes F16/F32 | M5 load or parity | exact BF16-to-F32 conversion in the repack (BF16-to-F16 only if exact) |
| Ivan's MTP head accepts less on the GSQ trunk | MTP acceptance in M5 | reported; the parent design's fallback (a head re-encoded from BF16) |
| more Metal kernels slow the runtime shader compile | startup time in M5 | pipelines only for types present; split the source if needed |

## Constraints

- Mac only. Git holds source and small fixtures; GGUFs live in `ds4-metal-data` and on HF.
- No GPU work without the user's go-ahead; pause the gateway stack for model runs and restore it.
- Never `kill -9` a Metal process. Code, docs and commits in English.
