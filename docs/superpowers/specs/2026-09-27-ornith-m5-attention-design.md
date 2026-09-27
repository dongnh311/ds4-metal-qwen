# Ornith M5: attention kernels (design)

Milestone M5 of the Ornith-1.5-35B-A3B port (family `QWEN35_MOE`, parent spec
`docs/superpowers/specs/2026-09-25-ornith-qwen35moe-design.md`). Status: design approved in chat on 2026-09-27; this
file is the written spec for review.

## 1. Goal and success criteria

Close the long-context speed gap to the live oMLX that M4 measured and attributed to attention
(`speed-bench/ornith/m4/REPORT.md`, `speed-bench/ornith/m4/profile/PROFILE.md`):

| | ds4 after M4 | live oMLX |
|---|---|---|
| decode with `--mtp`, 32K / 128K | 47.1 / 22.0 t/s | 64.4 / 38.1 t/s |
| prefill, 32K / 128K | 650 / 187 t/s | 1697 / 713 t/s |
| cold ~31K first turn (TTFT) | 49.6 s | 22.5 s |

M5 succeeds when, measured with the M4 harness (`speed-bench/ornith/m4_ab.py`, A-B-B-A, one model process at a time,
fresh nonce per prompt, `cached_tokens == 0`):

1. ds4 prefill >= live oMLX at 32K and 128K, and the cold ~31K first-turn TTFT <= live oMLX;
2. ds4 decode with `--mtp` >= live oMLX at 32K and 128K;
3. gate 1 (llama.cpp references, M1 tolerance) passes at prefill chunks 2048, 64 and 65;
4. `--mtp` greedy output stays byte-identical to plain decoding (M2 guarantee);
5. gate 2 is re-run on 23G and 25G next to the live oMLX on the same harness the same day and stays at parity with it
   (index >= oMLX index - 3.0 on each tier; the rule the user set on 2026-09-27, see section 8);
6. Qwen3.8 stays byte-identical (Qwen full gate PASS).

Out of scope (decided with the user): the 2K decode gap (65.0 vs 81.2 t/s) — at 2K attention is ~5 of ~14 ms per
decode step, the rest is MoE/GDN/MTP overhead; it gets its own milestone. The Metal 4 tensor API (matmul2d) for
attention is a follow-up only if the simdgroup kernels below fall short at 128K. Compressed K/V modes stay rejected
(M4 measured fp8/q4 slower).

## 2. What limits attention today (from the M4 code read)

- **Prefill** uses the shared `kernel_qwen4_attn_mm` (metal/qwen4.metal, dispatched from
  `ds4_gpu_qwen4_attn_decode_tensor` for T > 8, head dim 256, group <= 16). It is tiled with online softmax, but one
  threadgroup computes **one query token** (the 8 query heads of one KV head) and streams the whole K/V range for it.
  A 2048-token chunk therefore reads the K/V cache 2048 times; per-chunk attention cost grows linearly with position
  (0.41 s at position 0, 4.3 s at 30K, 19.7 s at 123K).
- **Decode** (T=1 and the 2-row MTP verify, M4's L12 `kernel_qwen35_attn_decode2`) splits the keys into at most
  `QWEN4_ATTN_MAX_SPLITS = 64` splits x 2 KV heads, i.e. at most ~128 threadgroups for the whole GPU, reads K/V with
  scalar `half` loads, and merges the splits in a second kernel that loops serially over all splits. Decode
  attention costs ~32 ms/token at 128K, ~3x the K/V bandwidth floor (~2.7 GB/token at 128K for 10 layers).

## 3. Design

All new code is Ornith-only and additive: kernels in `metal/qwen35.metal`, wrappers in `ds4_metal.m` (declared in
`ds4_gpu.h`), dispatch in `ds4_qwen35moe.inc`. The shared qwen4 kernels and every Qwen3.8 path are untouched.

### 3.1 `kernel_qwen35_attn_flash` — prefill, T > 8

- One threadgroup computes a block of **Br query tokens x the 8 query heads of one KV head** (8·Br query rows) and
  streams K/V tiles of Bc keys; each K/V tile is loaded into threadgroup memory once and used by all 8·Br rows.
- QKᵀ and PV use simdgroup matrix multiply (`simdgroup_half8x8` / `simdgroup_float8x8`); running max, sum and the
  output accumulator are F32; Q/K/V are staged as F16. Online softmax exactly as `kernel_qwen4_attn_mm`.
- Causal mask inside the chunk: query token t (absolute position pos0 + t) attends to keys [0, pos0 + t]. Tiles
  entirely above the diagonal are skipped.
- Output: the sigmoid output gate is applied in the epilogue as today (q+gate interleaved per head in `attn_q`).
- **Key split (split-K) for long contexts:** when 2 x ceil(T/Br) threadgroups are too few to fill the GPU (long key
  ranges), the key range is also split into Ks parts; each part writes partial (max, sum, acc) rows and a merge kernel
  combines them (same arithmetic as the decode merge, 3.2). Ks is chosen from the key count.
- Parameters Br (candidates 2, 4, 8, 16), Bc and the split rule are chosen with the micro-benchmark (section 5),
  within the 32 KB threadgroup-memory limit and register budget for head dim 256.
- Supported: head dim 256, GQA group 8, F16 K/V. Anything else (and T <= 8 tails) uses today's path.

### 3.2 `kernel_qwen35_attn_decode3` — decode (T=1) and the 2-row MTP verify

- Successor of L12 with the same exactness contract: plain decode is a rows == 1 call of this kernel and the MTP
  verify is one rows == 2 call; each row computes its own split geometry from its own key count only, so verify row 0
  equals a plain decode step bit for bit.
- More parallelism: the split cap rises from 64 to a new Ornith constant (target 256, tuned by the benchmark), so
  128K keys get enough threadgroups; K/V are read with vector loads (`half4`/`half8`), each threadgroup serving all
  8 query heads of its KV head from one K/V read.
- Merge: a parallel reduction over splits inside one threadgroup per (row, head), with a fixed reduction order so the
  result does not depend on scheduling; a single split skips the merge.
- Scratch: the partial buffer is sized for the new cap (rows x heads x splits x (2 + 256) floats).

### 3.3 Dispatch and knobs

- `DS4_QWEN35_ATTN_FLASH` (0/1): prefill chunks with T > 8 use `attn_flash` when on.
- `DS4_QWEN35_ATTN_DECODE` (2/3): selects L12's `decode2` or the new `decode3` for plain decode and the verify; the
  M4 knob `DS4_QWEN35_ATTN_DECODE2=0` keeps meaning "per-row qwen4 decode kernel".
- Both start off/2; each flips to on/3 only after its gate 1, MTP identity and A/B checks pass (M4's rule).
- Fall back to today's kernels when a knob is off, for unsupported shapes, and for `DS4_QWEN35_KV=fp8|q4`
  (the new kernels read F16 K/V only). No silent fallback inside a forward: a failed dispatch returns false.

## 4. Data flow and exactness

- Prefill writes K/V in the attention prep exactly as today; the cache layout (`[pos][kv_head][256]` F16) and the
  disk-KV payload are unchanged, so rewind, snapshot and restore are unaffected.
- `--mtp` and plain runs prefill identically (same chunking, same kernels), so prefill numerics only need gate 1's
  tolerance; decode exactness (4 above) rests on `decode3`'s rows contract.
- The MTP layer's own attention (layer 40) uses the same kernels; it only shapes drafts, never output.

## 5. Testing and measurement

- Kernel tests (`tests/test_qwen35_kernels.c`, model-free):
  - `attn_flash` vs a host double reference with the causal mask at T in {9, 64, 65, 2048} and pos0 in
    {0, 100, >= 4096}, with and without key split; also vs `kernel_qwen4_attn_mm` within tolerance;
  - `decode3`: `memcmp` of one rows == 2 call vs two rows == 1 calls at split boundaries (including the cap), vs a
    host double reference, outputs sentinel-filled before each call.
- Micro-benchmark `tests/bench_qwen35_attn.c`: kernel time and effective K/V GB/s vs the bandwidth floor at pos 2K,
  32K, 128K for T = 1, 2 and 2048; used to pick Br, Bc, the split rules and the decode cap.
- Model checks (controller GPU windows): gate 1 at chunks 2048/64/65 with each knob on; MTP identity
  (`test_qwen35_graph`, `test_qwen35_mtp`, `tests/ornith/test_mtp_cli.py`); payload/rewind tests; live server tests.
- Macro: `DS4_QWEN35_PROFILE` before/after; lever A/Bs with `m4_ab.py --mode lever` at 32K/128K; final gate 3 vs the
  live oMLX; gate 2 re-run with the live oMLX on the same harness; Qwen full gate. Receipts in
  `speed-bench/ornith/m5/`.

## 6. Risks

- Head dim 256 puts pressure on registers and the 32 KB threadgroup memory; Br may have to stay small. Key split
  restores occupancy.
- If `attn_flash` + `decode3` do not reach oMLX at 128K, the next lever is the Metal 4 tensor API (out of scope here;
  it needs a non-M5 fallback).
- A faster decode may shift the 2K/32K balance so MoE/GDN dominate; that is the separate 2K milestone.

## 7. Constraints (from the parent spec and M1-M4)

Qwen3.8 byte-identical and as fast; Metal only for Ornith; `DS4_QWEN35_*` knobs only; kernel changes additive in
`metal/qwen35.metal` (never `metal/qwen4.metal`); one model process at a time; never `kill -9` a Metal process;
GPU windows pause and restore the live stack; English code/docs/commits; no merge/push without the user's OK.

## 8. Gate 2 rule update (user decision 2026-09-27)

M4 measured the live oMLX at 78.8 on the gate-2 harness, below the parent spec's fixed 86.6 bar, while ds4 scored
78.0 (23G) and 84.7 (25G). The user accepted parity: from now on gate 2 passes when ds4's index on each tier is at
at least the live oMLX's index minus 3.0 points on the same harness the same day (3 points is about one case of the
small eval: one code_bench case moves the index 2.8 points; truncated = errored = 0), the agentic matrix shows no quality failures (turn timeouts are speed, reported under gate 3), and the
abliteration probes match. The parent spec's gate 2 section is updated accordingly.
