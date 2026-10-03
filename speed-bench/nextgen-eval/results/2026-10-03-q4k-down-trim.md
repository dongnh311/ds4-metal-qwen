# Trimmed Q4_K down rows for unc48L: same output, 3.75 GiB smaller

2026-10-03. Branch `feature/q4k-down-trim`.
- Spec: `docs/superpowers/specs/2026-10-03-qwen4-q4k-down-trim-design.md`.
- Plan: `docs/superpowers/plans/2026-10-03-qwen4-q4k-down-trim.md`.
- Scope: the user granted autonomy to build and measure the feature. No push, merge, deploy or HF upload was done.

## What changed

**Where the waste was.**
- PROD stores every routed expert's down projection as Q4_K `[768, 2560, 512]`.
- The expert's input width is 640, and Q4_K needs whole 256-value blocks, so each row carried 128 padding values.
- The padding sits in qs bytes 64-127 of each row's last block. The kernels never read those bytes.

**The trimmed format.**
- Each row keeps 2 whole blocks plus the last block's first 80 bytes: d, dmin, the 12 scale bytes and the qs chunks
  of values 512-639.
- Rows are 368 B instead of 432 B. Every byte a kernel reads sits at the same offset in its block, so only the
  row stride changes.
- In GGUF the tensor stays Q4_K with its true shape `[640, 2560, 512]`. Standard readers reject a Q4_K row of
  640, so no other tool can misread the file.

**Code:**

| part | what |
|---|---|
| `tools/qwen4_trim_down_pad.py` + `gguf_lite` | converter. 52 GB in, 48 GB out in 40 s, F_NOCACHE so the serving model keeps its pages. Writes a manifest with per-tensor sha256. |
| ds4.c loader | Q4_K short-final-block row size (parse, `routed_expert_row_bytes`); validator accepts trimmed down `[640, 2560, 512]` and refuses a trimmed/padded mix; CPU reference reads the short block; GPU graph accepts trimmed experts |
| ds4_metal.m | the five copies of the round-up rule (down MV, down GEMM, stream layer, stage union, stage pipe) now use one helper, `qwen4_down_row_bytes`; `ds4_gpu_qwen4_set_down_trimmed` is set on every qwen4 load |
| kernels | unchanged |
| qwen-regression | `--branch-model` / `BRANCH_MODEL`: branch servers load another GGUF, PROD keeps the registry model |

The file:
`/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownTrim-DenseQ4Kselimat-MTP.gguf`.

| | PROD | trimmed |
|---|---|---|
| file | 51.64 GiB | **47.89 GiB** (−3.75) |
| down tensor, one layer | 566,231,040 B | 482,344,960 B |
| one expert (gate + up + down) | 1.86 MiB | 1.70 MiB (−8.4%) |
| resident model at K=32 (startup plan) | 36.75 GiB | **34.25 GiB** (−2.50) |
| 6 GB stream cache: dynamic part | 4.14 GiB, 2278 experts | 4.30 GiB, **2581 experts** (+13.3%) |
| planned total at 262K | 52.52 GiB | 50.02 GiB |

The 6 GB cache gains more than 8.4%. Its prefill headroom is one expert per slot, so the headroom shrinks too and
leaves more room for the dynamic cache.

## Identity (window w1/w1b)

The arms:
- **base:** develop ef4d897c on PROD.
- **new:** this branch on PROD, then this branch on the trimmed file (binaries before the MTP-layer validator fix
  ad63c996, which does not touch the PROD or trimmed paths; the gate ran after it).

All arms used PROD's env and flags: SSD streaming, K=32, 6 GB cache, KV_GROW, and the 64K draft vocab for MTP.

| check | new × PROD | new × trimmed |
|---|---|---|
| greedy, 3 prompts, `--mtp`, 128 tokens | 3/3 byte-identical to base | 3/3 byte-identical to base |
| perplexity en / code / vi-new (`avg_nll`) | 0.733834646 / 0.037723827 / 2.136272103, equal | equal to base |
| kernel suite (`test_down_trim`: decode MR 0/2/4, generic and specialized, prefill GEMM; F = 640 and 320) | all qwen4 kernel tests passed | byte-exact vs padded |

**The first trimmed run failed.** The GPU graph's expert check (`qwen4_graph_expert_ok`) still required whole
256-value Q4_K rows.
- Fixed in 9d5b66f6, with a unit test.
- The trimmed arms were then rerun in w1b. The base and new×PROD outputs above are from w1.

## Speed (window w1b, SSD quiet, page-cache warmth controlled)

Method: ds4-bench at 8K with 128 generated tokens. Each file ran twice back to back, in both orders, and the
second runs are compared.

| run | prefill t/s | decode steady t/s | first token ms |
|---|---|---|---|
| PROD a1 / **a2** | 317.9 / **379.8** | 28.33 / **28.29** | 321 / 301 |
| trimmed a1 / **a2** | 409.2 / **504.3** | 29.94 / **29.77** | 249 / 180 |
| trimmed b1 / **b2** | 503.2 / **511.2** | 29.88 / **29.54** | 176 / 187 |
| PROD b1 / **b2** | 333.4 / **431.3** | 28.46 / **28.53** | 322 / 216 |

**Decode: +3.5% (order B) to +5.2% (order A), steady.** That matches fewer streamed bytes per missed expert plus a
cache that holds 13% more experts.

**Prefill: +19% (order B) to +33% (order A)**, but part of that is page-cache warmth, not trimming.
- PROD's second runs did not reach yesterday's warm 445-452 t/s (2026-10-02 Q4 note). With the other sessions'
  memory use today, a 52 GB file stays partly cold where a 48 GB file fits.
- The gain against PROD's best warm number on record, 452 t/s, is **+12-13%**.
- That smaller-file warmth is a user-facing effect too. It cannot be separated from the bytes saved without a
  machine that holds both files warm.

Cold start: a short chat prompt after reading 51 GB of other files, process wall time.

| | cold | warm |
|---|---|---|
| PROD | 9.9 s / 9.5 s | 4.4 s |
| trimmed | 8.1 s / 7.8 s | 2.7 s |

Cold start is −18%, and warm is −39%. All six short outputs had the same MD5.

## Qwen gate (window w2)

Run: `BRANCH_MODEL=<trimmed> speed-bench/qwen-regression/run.sh full`, with the branch binary on the trimmed file
against the registry PROD.

**PASS** (11:01-11:17, SSD quiet). The branch servers loaded the trimmed file and the PROD servers the registry
file, which the startup cache lines confirm: 2581 experts at 1.70 MiB against 2278 at 1.86 MiB.

| check | result |
|---|---|
| kernel + Q2 + prefill-pipe suites (run.sh make step) | all passed |
| fast tier: vi / code replies vs the recorded PROD baseline | byte-identical |
| decode, paired servers in the order PROD, branch, branch, PROD (median t/s per server) | PROD 32.54 / 29.13, trimmed **34.06 / 33.44** (floor: 97% of PROD) |
| steady wired | **44.55 GiB** vs the baseline's 46.09 (−1.54 GiB) |
| long-context needle (214,672 prompt tokens) | found |

The paired servers' decode gain, about +9% on average, is larger than ds4-bench's +3.5-5.2%. One of the PROD
servers drifted low (29.13). The gate's job is only the ≥97% floor, so the ds4-bench pairs above remain the speed
number.

**Final review** (fresh reviewer, whole branch): 0 Critical and 2 Important.
- The bound MTP (nextn) layers were left out of the down layout check and the trimmed/padded uniformity. Fixed in
  ad63c996, test first.
- The gate had not yet run. It has now passed.
- The six minors are deferred (see the branch ledger).

## Deploying (waits for the user)

1. Upload the trimmed GGUF and its `.json` manifest to the dongnhdev HF repo (model-sync rule).
2. Merge `feature/q4k-down-trim` into develop.
3. Cut `prod/q4k-down-trim-YYYYMMDD` from develop and deploy with `deploy-ai-gateway.sh`.
4. In the registry, change the Qwen rows' `-m` (262K and 512K) to the trimmed file. The rest of the command is
   unchanged.
5. Rollback: the registry's previous `-m`. The old binary cannot load the trimmed file; the new binary loads both.

The freed memory (2.5 GiB resident, smaller expert reserve) could later buy one more resident layer or a bigger
cache. That is a separate tuning decision.
