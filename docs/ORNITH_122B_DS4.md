# Ornith-1.5-122B on ds4 (Metal, M5 Pro 64 GB)

Status: **running end to end** on the `feature/ornith-122b` engine branch.
This document is the operating manual + receipts for the qwq35moe 122B shape
(48 layers + MTP head, 3072 d, 32 H / 2 KV (ratio 16), 256 routed experts top
8, ff 1024) after the attention G-slicing widen.

## Model files

| File | Size | Source / verification |
| --- | --- | --- |
| `Ornith-1.5-122B-Q4_K_M.gguf` | 75,843,911,648 bytes | official `ornith-ai/Ornith-1.5-122B-GGUF` single file, sha256 `729fe18c2491aaa446074a923e12f184f9f2239ffd5ce3dae47b0ce9cf8df40c` (byte-identical to the HF LFS oid) |
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

## Phase 3 (IQ2 tier) plan

1. Artifact: `unsloth/Ornith-1.5-122B-GGUF` `Ornith-1.5-122B-UD-IQ2_XXS.gguf`
   (single file, 41,967,395,072 bytes) + `imatrix_unsloth.gguf_file`.
   Download, sha-verify against the HF listing, header-audit the expert
   quant layout (unsloth UD mixes per-tensor quant choices; engine must
   gate them).
2. Engine: widen `tensor_expect_qwen35_expert_layout` to accept the IQ
   column the artifact aetually uses (IQ2_XXS/Q2_K/Q8_0/MXFP4 via
   `qwen4_expert_row_bytes` + matching lane kernels); config test with the
   freshly built artifact; 35B regression re-run.
3. Bench Q4_K_M vs IQ2 tier at 2 K / 8 K / 32 K (CLI + `ds4 server`).
