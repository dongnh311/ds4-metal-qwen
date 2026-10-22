# Ornith-1.5-122B on ds4 (Metal, M5 Pro 64 GB)

Status: **running end to end** on the `feature/ornith-122b` engine branch.
This document is the operating manual + receipts for the qwq35moe 122B shape
(48 layers + MTP head, 3072 d, 32 H / 2 KV (ratio 16), 256 routed experts top
8, ff 1024) after the attention G-slicing widen.

## Model files

| File | Size | Source / verification |
| --- | --- | --- |
| `Ornith-1.5-122B-Q4_K_M.gguf` | 75,843,911,648 bytes | official `ornith-ai/Ornith-1.5-122B-GGUF` single file, sha256 `729fe18c2491aaa446074a923e12f184f9f2239ffd5ce3dae47b0ce9cf8df40c` (byte-identical to the HF LFS oid) |
| `Ornith-1.5-122B-IQ2.gguf` | 38,246,170,592 bytes (35.62 GiB) | phase-3 tier built from the official Q4_K_M by `gguf-tools/ornith_requant.py` (receipt `gguf-tools/ornith_requant.receipt.json`); dense path byte-identical, routed gate/up IQ2_XXS + down Q2_K |
| `mmproj-Ornith-1.5-122B-BF16.gguf` | 912,263,200 bytes | same repo, vision encoder for a later vision pass |

Both live in `~/.local/share/ai-gateway/ds4-models/ornith-122b/`. The
unsloth fork (`unsloth/Ornith-1.5-122B-GGUF`) publishes the UD quant ladder
(`UD-IQ1_M`, `UD-IQ2_XXS`, `UD-IQ3_XXS`, `UD-Q2_K_XL`, UD-`Q3..Q6_K_XL`) plus
`imatrix_unsloth.gguf_file` (calibration weights keyed to the 122B tensor
names) — the phase-3 IQ2 tier candidates.

## How to run

Standard smoke recipe (correctness first; a fresh engine lock keeps tests
independent of any server on the box):

```sh
export DS4_LOCK_FILE=/tmp/ds4-ornith-$(date +%s).lock   # fresh inode per instance
export DS4_METAL_NO_RESIDENCY=1                         # 75.8 GB does not fit 64 GB: skip the residency wire
./ds4 --metal -m ~/.local/share/ai-gateway/ds4-models/ornith-122b/Ornith-1.5-122B-Q4_K_M.gguf \
      -c 1024 -n 16 --temp 0 -p "The capital of France is"
```

The phase-3 **IQ2 tier** (`Ornith-1.5-122B-IQ2.gguf`, 35.62 GiB) is the
performance tier on this box: it plans as `resident model 35.61 GiB` and runs
compute-bound (prefill 22–94 t/s, generation 35–39 t/s). It is rebuilt from the
official file by `gguf-tools/ornith_requant.py`:

```sh
python3 gguf-tools/ornith_requant.py \
  --src ~/.local/share/ai-gateway/ds4-models/ornith-122b/Ornith-1.5-122B-Q4_K_M.gguf \
  --imatrix ~/.local/share/ai-gateway/ds4-models/ornith-122b/imatrix_unsloth.gguf_file \
  --out ~/.local/share/ai-gateway/ds4-models/ornith-122b/Ornith-1.5-122B-IQ2.gguf
# --layers 0,5,11 for a pilot; --down-type COPY keeps routed down verbatim (the
# conservative fallback if the Q2_K down ever measures worse).
python3 gguf-tools/ornith_pilot_qc.py          # byte-determinism + cosine QC
python3 gguf-tools/ornith_dequant_check.py     # dequants vs golden fixtures/C oracle
```

- `--temp 0` for deterministic receipts.
- NO `--ssd-streaming`: it is refused for `qwen35moe` today (engine streams
  the model through the page cache instead).
- `DS4_QWEN35_PREFILL_CHUNK` defaults to **2048** tokens; leave it unless a
  test needs a specific chunk.
- MTP: add `--mtp` (never `--mtp-exact-sampling` — refused by policy).
- The engine holds a single-instance `flock`; tests/servers that need to
  coexist each bind their own `DS4_LOCK_FILE` inode.

### Attention dispatch (ratio 16)

Every qwen35 attention kernel dispatches
`QWEN35_ATTN_GROUP` = 8 query heads per KV head. A ratio of 16 runs **two
sub-passes per KV head** (`head_stride = 16`, `head_sub = 0` / `8`):
partial slots interleave the passes inside each `[kvh][split]` block, and
one merge follows the last pass (`merge3` computes the group itself). A
ratio of 8 runs the single sub-pass `(8, 0)` — index arithmetic identical
to the pre-widen dispatch, which is why the 35B regression stayed
bit-identical. Head ratios the qwen4-family kernels can run (≤ 12) keep
their legacy dispatches; the qwen4 wrappers cannot run a ratio of 16, so
those rows go through the decode3 wrapper (`wide = (group % 8) == 0 &&
group > 8` in `ds4_qwen35moe.inc`).

## Receipts

### Kernel + regression safety net (35B stays bit-identical)

- `tests/test_qwen35_kernels` (rebuilt): green end to end including the
  new wide rows (`H=32/Hkv=2`) — wide decode3 rows=2 is memcmp-equal to two
  isolated rows=1 calls; wide flash and wide nax (forced 8-way key split =
  interleaved partials, one merge) match the host double reference at the
  same 3e-3 tolerance.
- 35B regression on the prod CyberTiel ICE model, all green both before
  and after the phase-2 smoke work: `graph` (bit-identical MTP graph
  logits, verify rows, post-verify prefill), `mtp`
  (150 cycles / 115 accepted, bit-identical), `verify_batch`
  (bit-identical to plain decode), `session` (divergent prompt
  bit-identical to a fresh session), `config_122b`.

### Commits

- `6378ec68` — the G-slicing widen (kernel sub-passes + host wrappers +
  `_hkv` partial sizing + dispatch fallbacks).
- `14b9cca5` — the wide-predicate fix in `qwen35_graph_attend`: the
  predicate was originally inverted (`(group % 8) != 0`), which marked
  ratio-16 rows narrow and sent the MTP draft's off-row attend (r0 = 1,
  n = 1) into the qwen4 wrapper, which refuses a ratio of 16 silently →
  "Ornith mtp: draft failed". The fast-branch arms masked it in plain
  decode (decode3 arm) and prefill (flash arm); only the row loop
  misfired.

### Text smoke (Q4_K_M, mmap-streamed, cold)

| Run | prompt | result |
| --- | --- | --- |
| plain `-c 1024 -n 16` | "The capital of France is" | `The capital of France is Paris.` |
| plain, larger budget | "…France is Paris. The capital of Germany is" `-n 96` | `Berlin.` |
| `--mtp -n 16` | "Name three colors: red," | `red, green, blue.` — draft cycles committed 2 tokens each (pos 187 → 189 → 191, drafts accepted) |

**The model reasons first.** Its chat template opens the assistant turn
with `ڮthink`; the first sampled tokens are a thinking preamble (e.g. "The
user is completing…"), and the answer lands after enough budget. A `-n
4/16` smoke shows the preamble, not a failure: the same prompts decode
correct answers at `-n 96` (and the MTP run's "red, green, blue." shows
the drafting head eats exactly the same preamble). Budget `n` accordingly
(short factual smokes: `--temp 0 -n 96`).

### Throughput on this box (Q4_K_M tier)

`prefill ≈ 4.7–5.5 tok/s`, `generation ≈ 0.04–0.06 tok/s` (cold page
cache; a "warm" re-run measures the same — the working set does not fit
RAM). This tier is **IO-bound**: the 48 × 256-expert blob alone is ~73 GB
(gate/up Q4_K + down Q6_K per layer), so every stream faults expert rows
from SSD and the page cache thrashes. It is a correctness tier on 64 GB.
The performance tier on this machine is the phase-3 IQ2 resident file
(~32–42 GB → compute-bound decode), bench-planned at 2 K / 8 K / 32 K.

### Max-context no-swap probe (target achieved at 131072)

`ctx 131072 -n 4` (183-token prefill, CLI, NO_RESIDENCY):

```
ds4: memory: KV 3.25 GiB (raw 3.25 + compressed 0.00) + buffers 1.74 GiB + resident model 70.62 GiB = 75.62 GiB planned
ds4: memory detail: ctx=131072 prefill_cap=2048 raw_kv_rows=131072 compressed_kv_rows=0 backend=metal
```

- KV rows are reserved for all **131,072** positions: 3.25 GiB
  (`26.0 KiB/token/48 layers`, per-64-block 4-bit in-kernel KV) + 1.74 GiB
  buffers, on top of the mapped 72.3 GiB view.
- `vm.swapusage: used = 0.00M` before, during and after (the model view is
  file-backed — macOS purges it instead of swapping; the 4.99 GiB of
  anonymous KV/buffers fit). Free pages dipped (proc mid-run low
  watermark ~98 MB) and recovered after exit.
- Decode behaved correctly at the deep ctx (`prefill 4.85 t/s,
  generation 0.06 t/s` in that run).
- `refuses`: ctx planning caps everything at the prefill-chunk size
  (2048) — peak activation buffers stay chunk-sized.
- Round 1 of the ladder also verified `-c 1024` shows `KV 0.03 GiB` — the
  KV curve is linear in ctx at the planner level (26.0 KiB/token).
- Beyond 131072: the Q4_K_M tier's shrink lever on 64 GB is not KV, it is
  the expert working set — that belongs to the IQ2 tier (phase 3), where
  the resident model is 32–42 GB and KV (4 GiB at 128 K) fits beside it
  with room for the expert cache.

### Known limitations / follow-ups

- `--ssd-streaming` refuses `qwen35moe` (engine streams-of-streams design
  item, not required for the resident IQ2 tier).
- Vision (`mmproj` BF16, 0.91 GB): the engine has the qwen35 M-RoPE /
  mmproj-merger path (phase 4, optional).
- Center-piece MTP: drafts accepted at depth 1 in the CLI smoke; a full
  `eval` + `bench` receipt (tokens/sec at 2 K/8 K/32 K) comes with the
  IQ2 tier bench.

## Phase 3 (IQ2 tier): requant build (in progress)

**Tier decision (changed from the original plan)**: instead of shipping unsloth's
`UD-IQ2_XXS` file, we requantize the **official Q4_K_M** ourselves:
`gguf-tools/ornith_requant.py` reads `Ornith-1.5-122B-Q4_K_M.gguf` and writes
`Ornith-1.5-122B-IQ2.gguf` with routed gate/up → **IQ2_XXS** and routed down →
**Q2_K**, per-expert, guided by the unsloth `imatrix_unsloth.gguf_file`
(sha256-verified `6e5ada71…dc103ab` on load). Every other tensor (attention, GDN,
shared experts, MTP head (its own Q4_K/Q6_K routed experts), norms, router,
embeddings, output) is **copied
verbatim**, so the dense path stays byte-identical to the official checkpoint and
all earlier Dense/MTP receipts carry over unchanged. Engine constraint verified in
`ds4.c` (~6201–6300): gate/up share one type per layer, routed superset
`{Q4_K,Q5_K,Q6_K,Q2_K,IQ2_XXS,Q8_0,MXFP4}` already widened (commit `9a7d2c12`).

**Checkpoint property (the "Q4_K_M mystery" resolved)**: the official Q4_K_M ships
*dead ff-rows* — expert blocks whose scale/min words are all zero (d=dmin=scales=0)
dequantizing to exact zeros. Rates fall with depth (L0 ≈ 79%, L21 ≈ 16%, deeper
shortly after; MTP unaffected). Position agreement between this file and unsloth's
independent UD quantization is **97.5%** (7985/8192 sampled blocks against ~66%
independent-set chance) — a real checkpoint property, not a quantizer bug; both
files are canonical and the engine reads them as-is (zeros contribute nothing).

**Dequant validation (gguf-tools/ornith_dequant_check.py + tests/q4k_dump.c)** —
before writing a single byte of output, the numpy dequants used by the requant tool
were pinned to three oracles:
- Q6_K (down): bit-match against the repo golden fixture `tests/quant_fixtures.h`
  (type 14, 420 B → 512 values) — caught that `d` is stored **LAST** in the
  canonical `block_q6_K` (`ql|qh|scales|d`).
- Q4_K (gate/up): byte-exact vs a C reference dump (`tests/q4k_dump.c`, built from
  the repo-verified `tests/test_q4k_dot.c` conventions) on live rows from the file;
  dead rows dequant to exact zeros.
- Q2_K / IQ2_XXS: canonical ggml math — note the q2_K scale layout reads **16
  consecutive bytes at one bit-shift** per group (`qs[32p+16i+l] >> 2j`) with a
  canonical **MINUS** min-term, and iq2_xxs grid/sign tables are parsed from
  upstream `ggml-common.h` (`gguf-tools/ggml_tables.py`, npz-cached).

**Engine lane verification (no fallback, and the tables match upstream)**: `metal/moe.metal`
has first-class `kernel_mul_mv_*_iq2_xxs_*` / `_q2_k_*` pipelines for the routed experts, and
the engine's own copies of the tables are **bit-identical** to upstream ggml
(`ds4_metal_iq2xxs_grid[256]`, `ds4_metal_ksigns_iq2xs[128]` compared against the parsed
upstream header — IDENTICAL). Its `dequantize_iq2_xxs` uses the same aux32 grid/sign decode,
and `dequantize_q2_K` uses `q + 32*(il/8) + 16*(il&1)`, per-16 masks 3/12/48/192 and
`dl*q - ml` — the same canonical MINUS convention the packer writes. This closes the old
Risk #1 ("IQ tiers must be self-checked before trusting speed numbers"): the 35–39 t/s decode
is real compute, not a generic dequant path.

**Scope note**: the MTP head `blk.48` is not a small Q8_0 blob — it has its own
256 routed experts (Q4_K gate/up, Q6_K down) plus the `nextn` draft head. The
requant tool deliberately leaves it verbatim (the draft path keeps the publisher's
precision); requanting it would save only ~0.8 GiB and is a phase-3.4 idea.

**Pilot QC (gguf-tools/ornith_pilot_qc.py, PASS)**: L0-only requant run
(`--layers 0`, 69.81 GiB written in 54 s) accepted by `./ds4 --inspect --metal`
(2×IQ2_XXS + 1×Q2_K in the dir, header/tensor count untouched). Proof chain:
- **byte determinism 90/90** — re-encoding the same source values through the same
  `libds4quants` packer with the same per-expert imatrix weights reproduces the
  pilot's stored bytes exactly, for gate/up/down alike;
- iq2_xxs golden fixture dequant match (inventory of the grid/sign tables);
- live-position cosine ≥ 0.94 everywhere; **zero-row agreement: 0 disagreements**
  (all-zero source rows stay all-zero).
  Honest note: whole-row rel-RMS is NOT a valid gate for this checkpoint — down
  rows are ~90% exact zeros (the dead ff-channels), and the coarse Q2_K grid maps
  those zeros to small min-term offsets while reconstructing live positions at
  cos ≈ 0.989. GPU perplexity remains the tier-quality arbiter.

**Full build (done)**: `--layers all` wrote `Ornith-1.5-122B-IQ2.gguf` —
**35.62 GiB** in 1744 s (8 threads), 144 routed tensors requantized
(65.39 GiB → 30.38 GiB), 899 tensors total, re-read clean. `./ds4 --inspect --metal`
accepts it: `iq2_xxs 96 tensors (18.56 GiB) + q2_k 48 (11.81 GiB) + q4_k 333 (3.30)
+ q6_k 52 (1.79) + f32 370 (0.15)`; the dense path is byte-identical to the
official file. Note the source is **mixed**: routed down is Q6_K on 25 layers and
Q4_K on 24 — the tool dispatches the dequant by actual tensor type (the first
pilot QC pass assumed Q6_K everywhere and tripped an `nearest_int` assertion in the
packer on absurd magnitudes; fixed by type-aware dispatch).

**QC across layers 0,6,11,21,24,35,47 (PASS)**: byte determinism **378/378**,
live-position cosine ≥ 0.94 for every sampled row, zero-row agreement 0
disagreements.

First residency/throughput probe (`-c 1024 -n 24 --temp 0`): planner
`KV 0.03 + buffers 1.06 + resident model 35.61 GiB = 36.69 GiB planned`, mapped
36463.91 MiB, `prefill 22.07 t/s, generation 35.09 t/s` — the IQ2 tier is
compute-bound on this box where the Q4_K_M tier was IO-bound (0.04–0.06 t/s).
The model reads through `ds4_vec_dot_iq2_xxs_f32/_q8_K` and the Q2_K dot kernels
(dedicated code paths, not a generic dequant fallback), and `--mtp` drafts
commit normally on the IQ2 file.

**Behaviour + quality battery (IQ2 tier, `-c 2048 -n 80 --temp 0`, warm page cache)**:

| prompt | output | prefill / generation |
| --- | --- | --- |
| "What is the capital of Japan?" | `Tokyo.` | 92.17 / 39.41 t/s |
| "What is 123 * 45?" | `123*40=4920, 123*5=615, Total = 5535` → `5535` | 93.75 / 39.15 t/s |
| "Who wrote the novel Dune?" | `Frank Herbert` | 93.68 / 39.50 t/s |
| `--mtp` "Name three colors: red," | `Red, blue, green.` (draft cycles committed) | — |

Perplexity (`/tmp/ppl_probe.txt`, `-c 1024`, 32-token prefix):
**matched 100-token window** — Q4_K_M `ppl 26.64` (avg_nll 3.283) vs **IQ2 `ppl 24.88`
(avg_nll 3.214)**: the IQ2 tier is 6.6% *lower* perplexity than the official Q4_K_M on
the identical window — the decision gate (≤20% worse) is passed with margin. The longer
992-token window scores `ppl 3.097` on the IQ2 tier (the full Q4_K_M equivalent needs
~5.5 h of IO-bound decode, so it stays deferred; the matched short window plus the QA
battery above is the quality evidence). The UD-IQ2_XXS file cannot serve as a reference:
the engine gate refuses its `iq1_s` routed gate/up ("expert type iq1_s not supported").
