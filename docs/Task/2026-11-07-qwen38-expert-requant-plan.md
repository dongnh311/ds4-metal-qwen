# Qwen3.8 expert requantization — research plan

Branch: `feature/qwen38-expert-requant` (worktree `qwen38-requant`), base `develop`.

## Why this project exists

The kernel-side campaign (`feature/qwen38-2bit-mma`, merged) proved the routed
MoE GEMM is at its architectural ceiling: tensor cores ~19 TFLOPS roofline,
DRAM ~36 GB/s, three independent repack/layout variants neutral, custom
`simdgroup_matrix` pipeline spike no-go (`feature/qwen38-mma-pipeline`), and a
third quant-layout spike neutral (`feature/qwen38-expert-quant`). The only
measured way to go meaningfully faster is **fp16-resident operands** (~1.8x in
`tests/bench_mma_pipeline.m` v2) — which packed 2-bit weights cannot provide
without reintroducing the staging barrier or DRAM traffic that makes it slow.

That leaves one lever: **change the expert weights themselves**. This project
asks whether a different quantization of the routed experts can buy prefill
speed, decode safety, or quality headroom that IQ2_S cannot.

## Hard constraints (from backlog acceptance criteria)

- Decode ≥ ~45 t/s (current ~45 t/s on UD-IQ2_XS).
- Memory ≤ 60 GiB planned at 256k ctx; model loading stays mmap-backed;
  SSD-streaming expert path must keep working (per-expert tensors, explicit
  buffers, fast reads).
- CUDA path must not break (new ggml types need CUDA dequant or a fallback).
- Correctness before speed: deterministic suite + explained logits drift only.
- No permanent semantic variants behind flags; diagnostics must validate the
  one release path.

## What already exists (do not rebuild)

- `gguf-tools/qwen4_iq2.py` — calibrated expert quantizer (IQ2_XXS gate/up,
  Q2_K down pad768) driven by `qwen4_pack.py` (`SourceDB`,
  `QwenGGUFImatrix`, `GGMLQuantizer` over `gguf-tools/quants.c`). Requires the
  official BF16 safetensors checkpoint (NOT present locally; HF cache has only
  `model.safetensors.index.json` — a multi-hundred-GB download decision).
- `gguf-tools/imatrix/` — calibration dataset + rendered prompts.
- Built into `ds4`/`ds4-eval`: `--perplexity-file` (teacher-forced NLL —
  Phase 1 quality metric), `--imatrix-dataset/--imatrix-out/
  --imatrix-min-expert-samples` (calibration collection), `--expert-profile`
  (Metal decode-time per-expert histograms + route weights + cache simulation,
  resident mode, eval rejects the flag but `ds4` accepts it).
- Local artifacts:
  - `~/.local/share/ai-gateway/ud/UD-IQ2_XS-mtp.gguf` (50.3 GB, production;
    expert types: 80× type22, 24× type20, 23× type42, 16× type16, 153× Q8_0 —
    Unsloth dynamic mix, NOT uniform IQ2_S).
  - `~/.local/share/ai-gateway/ud/Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf`
    (second shard of the same UD quant — not a quality reference).
  - `Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q2KDownPad768-DenseQ4Kselimat-MTP.gguf`
    (43.7 GB, different finetune — NOT a quality baseline for UD weights).
  - `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` (PLE n-gram sidecar, required via `--ple`).
- `ds4-eval` hard suite (50 cases) / hard-smoke (12) + `--trace`.
- Expert routing instrumentation added here: `DS4_QWEN4_ROUTE_COUNTS=<path>`
  dumps raw per-(layer, expert) selection counts. Two hooks: the streamed MoE
  path (`all_ids` before dedup, needs `--ssd-streaming` +
  `DS4_QWEN4_STREAM_GATE=0`) and the resident prefill path
  (`ds4_gpu_qwen4_route_counts_note_selected`, commits+waits the batch once per
  MoE layer per chunk — diagnostic-only slowdown). Decode T==1 is skipped.
  NOTE: `--ssd-streaming` rejects UD-IQ2_XS ("unsupported routed expert tensor
  type") — resident hook is the one that works for this model.

## Measured facts that constrain the design space

From `bench_qwen4_moe_mm` @T=1856 (mid+down, E=512 S=10 shape):

| expert format | mid+down ms | note |
|---|---:|---|
| IQ2_XXS | ~11.0 | fastest — simplest dequant |
| IQ2_S (current UD) | ~13.7 | grid+sign codebook |
| MXFP4 | ~14.8 | |
| Q8_0 | ~35.0 | dequant ALU dominates |

- Lower/cheaper dequant wins; the skeleton (barrier+loop) is ~34% and MMA
  ~36% regardless of format.
- Prior failed experiment: experts requantized IQ2_S→Q2_K for the "raw MM"
  path — MoE unchanged, decode collapsed 40→22 t/s (reverted; manifest
  `UD-IQ2_XS-mtp.gguf.q2k-revert.json`).
- fp16-resident operands ~1.8x faster in the MMA microbench, but full fp16
  experts are ~4x the 2-bit footprint — impossible under 60 GiB.

## Candidate directions (evaluate in order; each needs a go/no-go gate)

### A. Mixed precision by routing hotness
Hot experts (high selection frequency) get a faster/bigger format; cold
experts stay IQ2_S. Requires the route-counts data (Phase 0) to size the hot
set. Open questions: per-expert tensor splitting support in GGUF layout +
streaming cache (expert bytes are per-expert already — feasible); decode impact
(hot experts are also the decode-hot ones); quality neutral by construction
(if hot experts go *up* in precision).
Speed upside is bounded: prefill cost is per-(token,expert) pairs, so upgrading
hot experts to a *slower-dequant* format (Q4_K/Q8_0) likely loses; upgrading to
a *cheaper* format (IQ2_XXS-style) gains only the ~20% dequant delta seen in
the bench table. Verdict likely: small upside, real complexity.

### B. Custom 2-bit format designed for this kernel
A format cheaper to dequant than IQ2_XXS (e.g. uniform 2-bit + per-block scale,
no codebook) could shave the staging section. IQ2_XXS→ideal-trivial delta is at
most a few ms/layer of the ~14 ms mid+down — bounded by the same skeleton.
Quality risk is severe at 2-bit without codebooks; needs imatrix calibration
and full eval. Probably not worth it vs. IQ2_XXS which already exists.

### C. Quality recovery at iso-speed (the realistic prize)
If UD-IQ2_S has measurable quality loss vs. the BF16 source, a *better* 2-bit
expert quant (imatrix-calibrated IQ2_XXS/IQ2_S variants, or per-expert mixed
IQ2_XXS/IQ2_S by sensitivity) could recover quality at the same speed — the
kernel cost is set by format family, not by codebook contents. This uses the
existing `qwen4_iq2.py` pipeline and is the only direction that improves the
model without needing a faster kernel. Requires the BF16/source checkpoint and
imatrix (present in `gguf-tools/imatrix/`).

### D. fp16-fragment-friendly sub-format (long shot)
Store weights so a *custom* dequant kernel emits simdgroup fragment layout
directly, removing the tgmem staging barrier — the one structural cost proven
unavoidable for generic packed formats. This is kernel+format co-design; only
worth it if A–C show the quality budget allows a new type. Keep as research
note; do not start without Phase 0/1 data.

## Phases

- **Phase 0 — data collection (this branch, now).**
  - Route-counts run: `ds4-eval --suite hard-smoke` under `--ssd-streaming`
    with `DS4_QWEN4_STREAM_GATE=0 DS4_QWEN4_ROUTE_COUNTS=...` → per-layer
    expert popularity. Analysis: coverage curve (top-k experts → % of
    selections), per-layer skew. Output: `docs/Task/` note + counts file.
  - Quality baseline: `--trace` output of the same run = hard-smoke record
    for UD-IQ2_XS on this build.
- **Phase 1 — quality budget.** Establish UD-IQ2_XS perplexity/NLL on a fixed
  prompt set (eval trace + optionally `--dump-logits` on a fixed corpus).
  Compare against the best local higher-precision artifact *of the same base
  weights* if one exists; otherwise bound the budget by comparing IQ2_S vs
  IQ2_XXS expert dequant SNR on real tensors (offline, `gguf_lite`/`qwen4_pack`
  readers) — cheap proxy before committing to a full requant build.
- **Phase 2 — candidate spike.** Pick ONE direction from A–D based on Phase
  0/1 data. Likely C (quality) or A (hotness) — decide with numbers, not
  preference. Spike = build the artifact with existing tooling or a minimal
  kernel bench extension, measure speed (bench harness) AND quality (eval).
- **Phase 3 — decision gate.** Go/no-go vs. constraints above; document in
  `docs/Task/` report. A "no-go" with measurements is a valid result (the
  kernel campaign produced three).

## Explicit non-goals

- No new permanent ggml type unless Phase 2 proves a format win end-to-end
  (host loader + Metal + CUDA + tests + streaming all in scope then).
- No changes to dense/shared-expert/MTP quantization in this project.
- No quality claims without an eval trace on the same prompt set.
