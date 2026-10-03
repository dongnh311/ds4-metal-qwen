# Trimmed Q4_K down rows for unc48L

Date: 2026-10-03. Branch `feature/q4k-down-trim` from develop ef4d897c.
User: "lên plan đi, tự chủ làm cho xong hết chức năng". Autonomy covers the spec, the plan, the code and the
measurements. No push, merge, deploy or HF upload without the user's OK.

## Goal

PROD (unc48L) stores every routed expert's down projection as Q4_K `[768, 2560, 512]`, while the expert's real
input width is 640 (`qwen4exp.expert_feed_forward_length`). Q4_K needs whole 256-value super-blocks, so each row
is padded from 640 to 768 values. The kernels never use the padding.

This feature drops the padding's bytes from the file. The result is the same model, byte for byte in its
outputs, with fewer bytes to map, cache and stream.

| | PROD (padded) | trimmed |
|---|---|---|
| down row | 3 × 144 = 432 B | 2 × 144 + 80 = 368 B |
| down, one layer | 540.0 MiB | 460.0 MiB |
| one expert (gate + up + down) | 1,950,720 B (1.860 MiB) | 1,786,880 B (1.704 MiB), −8.4% |
| experts per 6 GB of stream cache | n | 1.092 n (+9.2%) |
| file | 48 layers × 80 MiB = 3.75 GiB smaller | |

## Success criteria

1. **Identity.** On the trimmed file, the new binary produces exactly the same output as the old binary on the
   PROD file:
   - greedy generation, which covers decode and MTP;
   - perplexity, which covers prefill;
   - with PROD's flags: SSD streaming, K=32, 6 GB cache, KV_GROW.
2. **No regression.** The new binary on the PROD file matches the old binary on the PROD file.
3. **Kernel tests.** Each down entry point gives bit-identical results on a trimmed fixture and on its padded
   twin.
4. **Speed.** A warm-controlled A/B, trimmed vs PROD, is measured and reported. The measurement itself is the
   deliverable; the feature does not have to win.
5. **Qwen gate.** The Qwen gate passes before any merge request.

## Format: Q4_K with a short final super-block

A trimmed row of n values (n % 64 == 0, n % 256 != 0) is:
- n / 256 whole Q4_K blocks (144 B each);
- then the first `16 + (n % 256) / 2` bytes of one more block: d, dmin, all 12 scale bytes, and the qs chunks
  that cover values n / 256 × 256 .. n − 1.

Q4_K qs chunk k (32 B) holds sub-blocks 2k (low nibbles) and 2k + 1 (high nibbles), so a cut at a 64-value
boundary keeps whole chunks. For n = 640 the last block keeps 80 B: header plus chunks 0-1, the real values
512..639.

- **Every byte a kernel reads sits at the same offset inside its block as before.** Only the row stride changes,
  from 432 to 368.
- **Alignment holds.** Rows stay 16-byte multiples (144, 48, 80 and 112 are all multiples of 16), so the
  kernels' uint2 and uint4 loads stay aligned.
- **In GGUF**, the tensor keeps type 12 (Q4_K) with its true shape `[640, 2560, 512]`. Standard GGUF readers
  reject a Q4_K row of 640, so no other tool can misread the file. ds4 accepts it only for the qwen4 routed down
  experts.

## Components

1. **Converter: `tools/qwen4_trim_down_pad.py`, using gguf_lite.**
   - It copies every tensor and all metadata unchanged, except `blk.*.ffn_down_exps.weight` of type Q4_K whose
     dim0 is `roundup256(F)`, where F is `qwen4exp.expert_feed_forward_length` with F % 64 == 0 and
     F % 256 != 0.
   - Each such row keeps its first `row_bytes(F)` bytes, and dim0 becomes F.
   - It refuses when:
     - no tensor qualifies;
     - F breaks the rule;
     - a down tensor has another shape.
   - It writes `OUT.json` with per-tensor sha256, like ista_hc_to_f16.
   - `gguf_lite.nbytes` learns the short-final-block rule for Q4_K.
2. **Loader (ds4.c).**
   - One helper, `q4k_row_bytes(n)`.
   - Used for the tensor parse size (`t->bytes`) and for `routed_expert_row_bytes`, so the stream planner and
     the cache stay in step.
   - The qwen4 validator accepts down `[F, E, NE]` Q4_K as trimmed. All routed layers must agree, trimmed or
     padded.
   - The qwen4 CPU reference row (`qwen4_ref_row`) reads the short block.
   - Builds without the qwen4 Metal backend refuse a trimmed file at load with a clear message.
3. **Metal host (ds4_metal.m).**
   - `ds4_gpu_qwen4_set_down_trimmed(bool)`, set by the loader on every qwen4 load, like `set_rope`.
   - One helper, `qwen4_down_row_bytes(type, ff_dim)`, replaces the five copies of the `roundup256` rule:
     `moe_down_tensor`, `moe_mm_down_tensor`, `moe_stream_layer`, `stage_union` and `stage_layer_pipe`.
   - `qwen4_expert_row_bytes` stays strict (whole blocks), so Q4_K gate/up checks do not loosen.
4. **Kernels: unchanged.** Every Q4_K down path already skips the values from F on, before loading their bytes:
   `qwen4_row_dot`, `qwen4_kdown_rows` (NR2/NR4 and addr), and the mm stage8/stage16/raw16 loaders, whose K loop
   ends at `in_dim / KS`. The kernel test proves this by comparing against the padded twin.

## Testing

- **Python:** converter and gguf_lite unit tests on synthetic GGUFs:
  - byte slices are exact;
  - dims and manifest are right;
  - other tensors stay identical;
  - refusals fire.
- **C, `tests/test_qwen4_down_trim.c`** (includes ds4.c, like test_qwen4_prefill_pipe):
  - `q4k_row_bytes` for n = 320, 640, 704 and 768;
  - the parse size;
  - `routed_expert_row_bytes`;
  - the reference row of a trimmed tensor equals the first n values of its padded twin.
- **Metal, `tests/test_qwen4_kernels.c`:** a padded Q4_K down arena and its trimmed copy, F = 640 and 320.
  `moe_down_tensor` (every NR/NSG geometry the padded test walks) and `moe_mm_down_tensor` must match bit for bit
  between padded with the flag off and trimmed with the flag on.
- **End to end**, in GPU windows shared with the peers:
  - old/new binary × PROD/trimmed file;
  - greedy `ds4 --temp 0` on 3 prompts, with outputs compared byte for byte;
  - perplexity on the six texts, with `avg_nll` strings compared exactly;
  - the streaming stage and pipe paths run here, since layers 32-47 stream.
- **Speed:** ds4-bench 8K, each arm run twice back to back, comparing the second runs in both orders. Also the
  stream-cache expert count from the startup log.
- **Qwen gate** (`speed-bench/qwen-regression`) on the trimmed file vs PROD.

## Rulings

- **R1: keep type Q4_K with the true shape, not a private type ID.**
  - Why: every kernel already dispatches on type 12. A private ID would touch every switch, and standard readers
    still reject the file.
  - Cost if wrong: a later private ID needs a converter rerun.
- **R2: pass the stored row width to Metal as a global flag set at each qwen4 load, not as a new parameter on
  five shared signatures.**
  - Why: `ds4_gpu.h` is shared with CUDA, and `set_rope` is the existing per-model pattern.
  - Cost if wrong: a refactor to explicit parameters later; behavior is the same.
- **R3: Q4_K only.** Q2_K keeps d and dmin at the end of the block, so it cannot be cut the same way. PROD does
  not use Q2_K down.
- **R4: Metal and the CPU reference only.** CUDA and ROCm refuse trimmed files. PROD is Metal.
- **R5: gateway pauses for this feature's GPU windows count as covered by the autonomy grant.** They are
  announced to the peers, kept at most 60 min, avoid other sessions' production windows, and end with the
  gateway restored.
- **R6: deploying the trimmed file, uploading it to HF dongnhdev and merging to develop wait for the user.**

## Out of scope

- Deploying (prod branch, registry path).
- Uploading the trimmed GGUF to HF.
- Q2_K trimming.
- CUDA kernels.
- Changing K=32 or the 6 GB cache size. The freed memory is reported, and any re-tuning is a later decision.
