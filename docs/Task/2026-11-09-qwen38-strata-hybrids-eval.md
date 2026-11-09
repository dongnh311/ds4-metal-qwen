# Qwen3.8-Flash-Next Strata hybrids — full eval comparison (2026-11-09)

All Strata rows: engine 0.1.58-metal, M5 Pro 64GB, ctx 32k, temp 0,
nextgen-eval harness (code = execute-graded HumanEval+MBPP+, tools =
tool-call suites, reason = ds4-eval replay of the candidate-v1 traces).

## Variants (all share: 144 Q2_0 type-42 routed experts + hc_* from IQ1_M
## mix + shared 28.8GB PLE sidecar; only dense source differs)

| dense source | main GGUF | total w/ PLE |
|---|---:|---:|
| UD-IQ1_M (q2mix) | 38.4GB (shard1 deleted) | ~67GB |
| UD-IQ2_XS (candv1) | 38.7GB | ~67.5GB — on HF public |
| UD-IQ4_XS (iq4xs) | 38.5GB | ~67.3GB |
| UD-IQ3_XXS (iq3xxs) | 38.5GB | ~67.3GB |

## Scores (/164 total)

| suite | n | IQ1_M | IQ2_XS | IQ4_XS | IQ3_XXS |
|---|---:|---:|---:|---:|---:|
| code | 67 | 63 | **64** | 60 | 60 |
| tools_pos | 8 | 6 | 6 | 6 | **7** |
| tools_neg | 30 | **20** | 19 | 14 | 19 |
| tools_xfer | 6 | 3 | 3 | 2 | 1 |
| faithfulness | 9 | 9 | 9 | 9 | 9 |
| tools total | 53 | **38** | 37 | 31 | 36 |
| GPQA Diamond | 8 | 7 | 7 | 6 | **7** |
| SuperGPQA | 8 | 7 | 7 | 6 | **7** |
| AIME2025 | 8 | **8** | 7 | **8** | 6 |
| MMLU-Pro | 20 | 18 | 16 | 16 | **19** |
| reason total | 44 | **40** | 37 | 36 | 39 |
| **TOTAL** | **164** | **141** | **138** | **127** | **135** |

## Speed (suite medians)

| | decode t/s | prefill t/s |
|---|---:|---:|
| IQ1_M | ~60 | ~284 |
| IQ2_XS | 55.7 | 286.3 |
| IQ4_XS | 56.9 | 283.9 |
| IQ3_XXS | 57.3 | 277.2 |

## ds4-side variant (UD-IQ3_XXS dense + Q2_0 experts + embedded MTP)

`ud/UD-IQ3_XXS-Q2_0-experts-mtp-strata48l.gguf` — 41.3GB, 1255 tensors,
block_count=49 + blk.48.* MTP weights from prod + nextn_predict_layers=1.
Loads on ds4-server --metal --mtp --ple (verified, ctx 32k):
decode ~37 t/s avg with MTP (~32.6 without). Cannot share one GGUF with
Strata: Strata requires 48-layer metadata, ds4 MTP requires 49.

## Conclusions

- Dense-tier ranking on this harness: IQ1_M >= IQ2_XS >> IQ4_XS ~= IQ3_XXS.
  Higher-bpw dense does NOT help — Unsloth's per-tier tuning matters more
  than nominal precision; IQ1_M dense remains the best measured.
- IQ3_XXS is the best reason scorer (39, incl. best-ever MMLU-Pro 19/20)
  but weaker on code/AIME — mixed profile, not a clear win.
- candv1 (IQ2_XS) is the practical choice: top code score, ~equal total,
  file intact, already on HF public repo
  `dongnhdev/Qwen3.8-Flash-Next-Strata-Q2mix`.

## Rebuild note (2026-11-09)

q2mix (IQ1_M dense + Q2_0 experts) was rebuilt from scratch after the
original shard1 was deleted: same rules-q2_0.json + same unsloth imatrix +
same Q2_0 encoder => deterministic expert payloads; dense tensors copied
verbatim from UD-IQ1_M shards 1+2 (shard3 = PLE sidecar, unchanged).
Artifacts:
- Strata: models/q2-ud-iq1m-mix/ shard1 rebuilt (38.8GB, pack rebuilt as
  packs/q2_0 with --compat-bf16, ple_conv1d.weight BF16->F16). Serving on
  :8080, coherent.
- ds4: ud/UD-IQ1_M-Q2_0-experts-mtp.gguf (41.6GB, 49 blocks, blk.48 MTP
  tensors from UD-IQ2_XS-mtp-gateup-iq2xxs.gguf, 96 hc_*_inject BF16->F32,
  ple_conv1d.weight BF16->F16). Verified on ds4-server --metal --mtp --ple,
  ctx 32k: decode ~42 t/s avg.
Expected quality = the q2mix row above (141/164) since weights are
recipe-identical; not re-evaluated.
