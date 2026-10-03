# V4.1 Engram tables at 4 bits (Lloyd codebook), converted in place — design

Date: 2026-10-02. Status: approved in conversation (three sections), pending spec review.
Parent: `docs/V41_64GB_BUILD.md` (the V4.1 Flash Q2 GGUF on the M5 Pro 64 GB).
Evidence: the 2026-10-02 encoder spike. The code was throwaway: worktree `v41-engram-sim`, `DS4_ENGRAM_SIM`, plus scratchpad scripts. Scores are in
`~/orca/workspaces/ds4-metal-data/v41-engram-sim/20261002/`.

## Problem

`DeepSeek-V4.1-Flash-Q2.gguf` is 365 713 686 528 B (340.60 GiB), and the machine has 62 GiB free. The two Engram tables
are the last two tensors in the file. Together they hold 768 022 850 rows of 264 B, which is 202 758 032 400 B (188.8 GiB, 55 % of
the file):

| Tensor | dims | abs offset |
| --- | --- | ---: |
| `blk.1.engram_embd.weight` | [264, 384006168] I8 | 162 955 640 832 |
| `blk.14.engram_embd.weight` | [264, 384016682] I8 | 264 333 279 232 |

Each row holds 256 FP8 E4M3 values followed by 8 E8M0 scales, one per 32 values. The encoding string is
`deepseek41.engram.encoding = "e4m3_e8m0_32_row264"`. The runtime reads rows from disk with `pread`, one row per hashed n-gram,
and every read goes through `ds4_engram_read`. Rows are never mapped and never touched by the GPU.

Goal: a smaller file with no measurable quality loss.

## Evidence

The spike compared encodings at 136 B per row (256 values at 4 bits plus 8 scale bytes) by CPU reconstruction error (rel-L2
over 20K sampled rows). It then scored the closest candidates with `score_official`, which measures NLL against official-API
continuations. Scoring ran through `DS4_ENGRAM_SIM`, which re-encodes every row on read. Two sets were used:
- general: 100 cases, 2994 tokens, ctx 4096;
- long: 9 cases at 8K/16K/32K, 576 tokens, ctx 34816.

| Variant | Bits | B/row | File | rel-L2 | general NLL | long NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0: E2M1, power-of-two scale | 4 | 136 | 249 GiB | 0.121 | +0.12 % | +1.79 % |
| v4: E2M1, 4-significant-bit scale, best of {up, down} | 4 | 136 | 249 GiB | 0.095 | — | +3.66 % |
| **v7: Lloyd codebook, same scale** | 4 | 136 | **249 GiB** | **0.085** | **−0.05 %** | **+0.94 %** |
| b5: 16-level Lloyd, same scale | 5 | 168 | 272 GiB | 0.044 | — | +1.33 % |
| b6: 32-level Lloyd, same scale | 6 | 200 | 295 GiB | 0.021 | — | +1.04 % |

How to read the table:
- **The long set's offset is a noise floor, not a loss.** Even the 6-bit encoding (b6), with 16× less squared error than v7, shifts the long set by +1.04 %. The 5-bit encoding does worse than the 4-bit one. Nine cases are too few to tell 4, 5 and 6 bits apart.
- **On the general set, v7 matches the original or beats it.** NLL is −0.05 %, first-token matches go from 78 to 81, and greedy LCP goes from 9.50 to 9.76.
- **Decision: v7.** It meets the "no measurable loss" criterion at the smallest size: 249.04 GiB, which saves 91.56 GiB.

## 1. Format and runtime

**Encoding string.** `lloyd4_e5m3_32_r136`. It has the same 19 characters as the original, so the header can be patched in place.

**Row layout.** 136 B per row:
- **Bytes 0..127, codes.** Two codes per byte, with the even element in the low nibble. Each nibble is `sign << 3 | idx`, where `idx` (0..7) indexes the codebook.
- **Bytes 128..135, scales.** One scale byte per 32 values: `b = E << 3 | M`. The scale is `(8 + M) / 16 · 2^(E − 16)`, an unsigned float with 4 significant bits and frexp exponent `E − 16`, which covers −16..15.
- **Measured range.** On 100K sampled rows of each table, the frexp exponents fall in [−1, 3] (E = 15..19). There were no all-zero groups.

**Codebook.** `CB = {0, 0.095, 0.1901, 0.2944, 0.4126, 0.5586, 0.7484, 1.0}` as float constants, times the scale.
- It is a Lloyd-Max fit to |v| / amax on 20K rows of `blk.1`. The same codebook gives rel-L2 0.0850 on `blk.1` and 0.0851 on `blk.14`.
- It is hard-coded in `ds4_engram.c` and named by the encoding string. An in-place header patch cannot add metadata keys.

**Decode.** In `ds4_engram_read`, branched on the table's encoding: `v = copysignf(CB[idx] * scale, sign ? -1 : 1)`. The existing
BF16 rounding is then applied unchanged. The original encoding still decodes exactly as it does today.

**Encoder.** It lives in `ds4_engram.c` and is shared by the converter and the tests. Its arithmetic is exactly the spike's v7, so
the converted file decodes bit-identically to what was scored.

1. Decode the source row to float, as today: `ldexpf(e4m3(code), scale − 127)`.
2. For each group of 32, take `amax`. Then form two candidate scales:
   - `up = f8_up(amax)`, which rounds the frexp mantissa up to 4 significant bits;
   - `down` = one 4-significant-bit step below `up`. When the mantissa is already at 8/16, the step gives `15/16 · 2^(e−1)`.
3. For each candidate, quantise every value to the nearest codebook level of `|v| / s`. Search levels 1..7 with a strict `<`, so ties
   go to the lower index. Take the sign from the source value, so a value that rounds to 0 keeps its sign (−0).
4. Sum the float squared error in element order. Keep `down` only if its error is strictly smaller; ties keep `up`.
5. An all-zero group gets codes `sign << 3 | 0` and scale byte 0, so it decodes to ±0 exactly as the source does.
6. A scale candidate outside E = 0..31 is a conversion error. The converter's pre-scan finds it before anything is written.

**Runtime changes.**
- **`ds4_engram_table`** gains an `encoding` field, and `ds4_engram_table_open` takes it. The row size is derived from it: 264 or 136.
- **`ds4.c` loader.** These sites accept `e4m3_e8m0_32_row264` or `lloyd4_e5m3_32_r136`, and the tensor width must match (264 or 136):
  - `model_unmap_engram` (ds4.c:2905, 2916);
  - the config string table (ds4.c:7207);
  - `tensor_expect_layout` (ds4.c:7229).

  The parameter count at ds4.c:3330 subtracts `row_bytes − (encoding values)` per row, and gets this from the encoding.

  Every other string is refused, including the conversion-in-progress marker `conversion_underway` (section 2). The refusal
  message names the cause.
- **Unchanged:** hashes, row counts, primes, the Engram prefetch, all GPU code, and every other tensor.
- **Consequence:** only this fork reads the new file. Upstream ds4 (antirez/Ivan) refuses it. The original file keeps working here.

## 2. In-place converter

The tool is written in C: `gguf-tools/deepseek41_engram_q4.c`, built from the Makefile. It uses the encoder in `ds4_engram.c`
and only the CPU and disk, never the GPU.

**`--check FILE`** is read-only.
- It requires the original layout: encoding `e4m3_e8m0_32_row264`, the two Engram tables as the last two tensors, and the
  expected row counts.
- It encodes all 768M rows in RAM. It fails on:
  - any source code with `(code & 127) == 127`, or any source scale 255;
  - any scale candidate outside E = 0..31;
  - any non-finite value.
- It records the expected SHA-256 of the converted tail.
- Any failure stops it before the file is touched.

**`--sample FILE N OUT`** is read-only and runs before conversion, for the gate.
- It takes N rows per table from a fixed seed.
- It writes each row's v7 decode, after BF16 rounding, to OUT (about 40 MB for N = 20K).

**`--convert FILE`** has three steps.

1. **Block.** Replace the encoding string with `conversion_underway`, which also has 19 characters, and `fsync`. ds4 then
   refuses the file instead of reading half-converted rows.
2. **Compact the tail forward, chunk by chunk.**
   - Table 1 keeps its offset A. New row i goes to A + 136·i.
   - Table 2 moves to B' = align16384(A + 136·N1) = 215 180 492 800. Its old offset was 264 333 279 232.
   - For each chunk:
     1. read the source rows into RAM and encode them;
     2. write them to their new place and `fsync`;
     3. record progress in a small state file through an atomic rename.

   **The journal.** A chunk only ever overwrites bytes of rows that are already read, so no unread data is lost. Crash safety
   is the remaining risk: during a crash, the chunk being written may have overwritten its own source rows. Its source must
   therefore stay recoverable. This happens only for the first chunk or two of table 1, where 264·i0 < 136·i1. Those source
   bytes are copied to a small journal file (tens of MB) and `fsync`ed before the write.

   Table 2 never overlaps its own unread or in-flight source: B' sits about 49 GB below its old offset. It writes over table 1's
   already-converted source region.
3. **Finish.**
   1. Write the final header: encoding `lloyd4_e5m3_32_r136`, dims [136, rows] for both tables, and table 2's new offset. Every
      field is fixed width, so the header length does not change.
   2. `ftruncate` to 267 406 761 552 B, `fsync`, and delete the journal and state files.

**`--verify FILE`** runs after conversion. It hashes the tail and compares it with the expected SHA from `--check`. It also prints
the whole-file SHA-256 for the docs and HF.

**`--verify-sample FILE OUT`** runs after conversion, for the gate. It reads the same rows through `ds4_engram_read` and requires
bit-identical output to OUT.

**Errors and refusals.**
- After an I/O error, power loss or Ctrl-C, re-running `--convert` resumes from the state file. It restores the journaled source
  first when needed.
- The tool refuses a file that is neither the original nor a conversion in progress.
- The tool refuses to run while `lsof` shows a ds4 process with the file open.

**When to run.** Only after the CPU tests are green, and only on the user's explicit go. The FP8 original is lost locally once
conversion starts; it remains downloadable from its source repo. Never run it during another session's measurement window,
because the I/O load can skew their numbers.

## 3. Tests, gate, rollout

**Tests.** Written TDD and run on the CPU through `make test-engram` and a new converter test target.

- **`test_engram`, decode.**
  - Synthetic `r136` rows with known codes and scales decode to exactly the expected floats.
  - This covers −0, both codebook ends, scale exponents E = 0 and E = 31, and all-zero groups.
- **`test_engram`, encoder.**
  - Random FP8 source rows are encoded, then decoded.
  - The result must be bit-identical to an independent reference of the v7 arithmetic written inside the test. The spike's sim code
    is throwaway and is not reused.
  - Cases include values exactly halfway between two levels, and groups where the `up` and `down` errors tie.
- **Loader.**
  - Each encoding string selects its row size, and a width that does not match is refused.
  - Unknown strings and `conversion_underway` are refused.
  - The parameter count is right for both encodings.
- **Converter**, on a small synthetic GGUF with the same tail structure (Engram tables last, alignment 16384, unaligned table-1 end):
  - The run `--check` → `--convert` → `--verify` patches the header without changing its length, updates dims and offsets,
    truncates to the expected size, and the rows decode correctly.
  - An interruption at each phase, triggered by a test-only hook, followed by a resume gives a file byte-identical to an
    uninterrupted run.
  - If the pre-scan fails (for example, a source scale is 255), the file is left byte-identical.
  - Refusal cases: the wrong file, and a file that is open.

**Gate on the real file.**
1. **Before conversion, on the CPU:** run `--sample` on 20K rows per table and save the result in the scratchpad.
2. **After conversion, on the CPU:** run `--verify`. Then run `--verify-sample`, which must be bit-identical. This ties the real
   file to the encoding that was scored.
3. **GPU window, about 15 minutes with the gateway paused** (needs the user's OK and the usual coordination with the other
   sessions):
   - **Quality.** Run `score_official` on the general set against the converted file, without the sim. The TSV must equal
     `score-v7-general.tsv`. If the binary differences since 9695695c change the numerics, NLL must stay within ±0.1 % of it.
   - **Speed.** Run one `ds4-bench` (ctx 8192, auto cache, `switch`, gen 64). Prefill and steady decode must be within 5 % of the
     2026-10-02 slab-fix numbers (about 255 t/s and 9.3 t/s). With no original left to interleave with, this is the only speed
     check.

`ds4-eval core` is not run. It has 92 reasoning cases, which would be hours of gateway pause at V4.1 speed, and it is too noisy
to see a change of 0.1 % NLL.

**Rollout.**
- Code (runtime, converter, tests) goes on branch `feature/v41-engram-q4`, cut from develop 5c416255. Merge or push happens only
  on the user's request.
- The conversion runs after the CPU tests are green, and only on the user's go: it is irreversible.
- **HF.** Upload the converted GGUF (249 GiB) to a new dongnhdev repo, proposed name
  `dongnhdev/DeepSeek-V4.1-Flash-Q2-EngramQ4-GGUF`; the user confirms it at upload. The model card records:
  - the source repo and the original SHA;
  - the conversion command;
  - the new SHA;
  - the quality numbers;
  - that only this fork reads the file.
- **Docs and downloads.** Add a `ds41f-q2-eq4` target to `download_model.sh`, with a SHA check. Update `docs/V41_64GB_BUILD.md`.
- **Other sessions.** Engram is V4.1-only, so no Qwen gate is needed. The Qwen session is still told, because `ds4.c` is shared.
- **Cleanup.** Delete the throwaway `v41-engram-sim` worktree.

## Non-goals

- Changing rows, hashing, or the row count. Zeroing n-gram columns was measured in the spike on the general set:
  - drop3 (columns 8-23) is +62 % NLL and is rejected;
  - drop4 (columns 16-23) is +0.16 % and is not pursued here.
- 5- or 6-bit encodings. They were measured and are no better than v7 on these sets.
- Any GPU or Metal change.
- Keeping a local copy of the FP8 original. The user chose in-place conversion.

## Risks

- **Interrupted conversion.** Handled by the journal, the state file and resume. Covered by the interruption tests.
- **A scale out of range somewhere in the 768M rows the sample did not see.** `--check` encodes every row before any write.
- **Numerical drift between the spike binary (9695695c) and develop** could break an exact TSV match. The gate falls back to
  ±0.1 % NLL. `--verify-sample` remains the exact check of the format itself.
- **The file stops loading in upstream ds4.** Accepted and documented on the HF card.
