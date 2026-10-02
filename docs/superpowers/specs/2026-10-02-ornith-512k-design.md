# Ornith 512K context with YaRN: design

Date: 2026-10-02. Status: design approved in conversation ("ổn"). Branch: `feature/ornith-512k`,
cut from develop `5b36552a`.

This is the first of two sub-projects the user asked for: "đưa ornith lên 512k context và tối ưu
tok/s của nó tiếp". The second, decode speed, gets its own brainstorm and spec once this one is
approved. It starts by profiling where a decode step spends its time at 2K, 32K and 128K.

## Goal

Ornith-1.5-35B-A3B on ds4 serves prompts up to 512K tokens. The 262K default stays exactly as it is.

## User decisions

- **Separate entry, like Qwen.** The 512K Ornith is its own registry entry on its own port. The
  current 262K Ornith entry stays the default and does not change.
  - Reason: static YaRN costs some short-prompt quality. On Qwen, reason went from 43/44 to 41/44
    (`speed-bench/nextgen-eval/results/2026-09-30-sp4-yarn-vs-s050.md`).
- **Design approach A:** reuse the existing qwen4 YaRN machinery rather than writing an Ornith copy
  or reading YaRN keys from the GGUF. The gbuzhf GGUF has no rope-scaling keys.

## What exists today

- **Static YaRN for qwen4 (sub-project 4, merged).**
  - `qwen4_rope_configure` builds a 32-pair frequency table with `ds4_qwen4_rope_table`: HF
    correction range, beta_fast 32, beta_slow 1, cos/sin scaled by `0.1 ln f + 1`.
  - The factor comes from `-c` through `ds4_qwen4_yarn_factor`: the smallest power of two at least
    `context / native`, so 2 for 524288. `DS4_QWEN4_YARN_FACTOR` overrides it, and `=1` forces it
    off.
  - The table goes to Metal through `ds4_gpu_qwen4_set_rope`.
- **Server (sub-project 4).**
  - `ds4_engine_rope_yarn_factor` and `ds4_engine_native_context` drive two things: the disk KV
    directory `<kv-dir>/yarn-<f>`, and continued checkpoints thinned to `step × 2^k` past the native
    context (`continued_dense_max_tokens`).
  - Both functions return "no YaRN" for every family except qwen4 today.
- **Ornith shares the rope.**
  - Ornith has no rope code of its own: `metal/qwen35.metal` has none. Every Ornith rotation runs in
    the qwen4 prep kernels through `qwen4_rope_neox`, which reads `args.rope_freq` and
    `args.rope_mscale`. That covers prefill, decode and the MTP block.
  - `qwen4_rope_fill` (ds4_metal.m) fills those arguments from the table when one is set; otherwise
    it computes `powf(base, -2i/n_rot)`.
  - Ornith never sets the table today, so it runs on that fallback.
  - Ornith's rope parameters equal Qwen3.8's: 64 rotary dims, base 1e7, native context 262144. The
    YaRN table is therefore the same table.
- **The engine refuses Ornith above 262144:** "Ornith supports up to 262144 tokens of context (no
  YaRN)", in `ds4_engine_open`.
- **Precedent:** the Qwen 512K entry (`ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2-512K`, port 18088,
  `prod/qwen-512k-20261001`).
  - It is the 262K command with `-c 524288` and `--kv-cache-continued-interval-tokens 0`.
  - Its registry fields are `context_limit` 524288 and `recommended_context` 360448.

## Design

### Engine (`ds4.c`)

1. **Ornith takes YaRN from `-c`.**
   - `config_validate_qwen35moe_model` computes the factor with the existing
     `ds4_qwen4_yarn_factor(262144, context_size, getenv("DS4_QWEN4_YARN_FACTOR"))`.
   - When the factor is above 1 it builds the table with `ds4_qwen4_rope_table` and sets it, on Metal
     and on the CPU globals, exactly as `qwen4_rope_configure` does.
   - It logs one line naming the model, the factor and its source (`from -c` or the variable).
2. **No table at factor 1.**
   - With `-c` at or below 262144 and no variable, Ornith sets no table and keeps today's fallback.
   - Reason: the table is computed in double and the fallback in float, so they could differ in the
     last bit. Not setting it keeps the 262K path byte-identical by construction.
3. **Refuse unscaled long context.**
   - The "no YaRN" refusal becomes: refuse only when `-c` exceeds 262144 and the factor is 1
     (`DS4_QWEN4_YARN_FACTOR=1`).
   - Qwen warns and continues in that case. Ornith refuses, because a silent unscaled 512K Ornith
     has never been validated.
4. **Engine queries.** `ds4_engine_rope_yarn_factor` and `ds4_engine_native_context` also answer for
   Ornith (the factor, and 262144). The server's `yarn-<f>` cache directory and its checkpoint
   thinning then work for Ornith with no server change.
5. **Reuse the variable.** `DS4_QWEN4_YARN_FACTOR` also applies to Ornith, because the rope machinery
   is shared. `docs/` gets one line saying so. No new variable.
6. **CPU reference rope.**
   - `qwen4_ref_rope` reads `g_qwen4_rope_freq` directly, and those values stay zero unless
     `qwen4_rope_configure` ran.
   - The plan finds which CPU path Ornith's reference uses. If that path reads these globals, Ornith
     fills them at every factor, which leaves Metal untouched.

### Memory and index audit at 512K

At `-c 524288` the Ornith KV is about 11.0 GiB. The server line at 262144 reads "KV 5.50 GiB +
buffers 1.00 GiB + resident model 21.26 GiB". The plan audits every allocation and index that
scales with context:

- every buffer sized from the context: KV, MTP rows, attention split scratch for decode3, flash and
  NAX prefill scratch, disk-KV payload staging;
- every 32-bit offset or index that could overflow once one buffer passes 2 GiB or 4 GiB (KV is
  about 1.1 GiB per attention layer at 512K, and more if layers share a buffer);
- the fallback attention paths (`DS4_QWEN35_ATTN_FLASH=0`, `DS4_QWEN35_ATTN_DECODE=2`), checking
  whether any of them materializes a chunk × context score matrix.

Every unsafe site gets fixed, or refuses at startup with a message naming the knob and the limit.

The planned total at 512K is about 33-34 GiB, far below the Qwen 512K entry's 50.8 GiB. KV is
allocated for the full `-c` at startup, as today; KV grow-on-demand for Ornith is out of scope.

### Server

No new behavior is expected: Ornith gets the `yarn-2` directory and checkpoint thinning through the
engine queries. The plan verifies both from `server.log` and the KV directory.

### Gateway and deploy

The changes follow the Qwen 512K deploy and touch the same set of gateway files. The plan lists them
from that deploy's commits.

- **New registry entry** `Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx-512K`:
  - port 18089, pid file `ds4-ornith-512k.pid`;
  - the 262K Ornith command (MTP draft vocabulary, same GGUF) with `-c 524288` and
    `--kv-cache-continued-interval-tokens 0`;
  - same `--kv-disk-dir`: the server separates the cache into `yarn-2`;
  - `context_limit` 524288, `recommended_context` 360448.
- **The port goes into every list that names the ds4 ports:** `active-backend.py`, the healer and the
  dashboard.
- **The 262K Ornith entry does not change.** It stays the default model.
- **Deploy:**
  - `deploy-ai-gateway.sh cut ornith-512k` from develop after the merge, then
    `install prod/ornith-512k-YYYYMMDD`;
  - `smoke --model` for both Ornith entries and the two Qwen entries, each against a reference.
  - Ask the user first, and coordinate the window with the gateway deploy-owner session.

## Testing

### Offline (no GPU)

- **C unit tests** through the existing `tests/ds4_test.c` entry points:
  - the Ornith factor is 2 at 524288, 4 at 1048576 and 1 at 262144;
  - the variable overrides;
  - a forced-off variable above native is refused;
  - the table is set only when the factor is above 1.
- **Cache directory naming** already has tests. One more case asserts that Ornith's engine queries
  yield `yarn-2`.
- `make` builds every binary, `make test` passes, and so do the existing Ornith model-free tests.

### GPU (each window needs the user's go-ahead and a paused gateway stack)

1. **The default path does not change.**
   - Ornith gate 1 (`tests/ornith/gate1.py`, three chunk sizes) gives dumps byte-identical to develop
     at `-c 262144`.
   - The Qwen fast gate gives byte-identical replies, because the shared rope code is touched.
2. **YaRN is correct.**
   - Run llama.cpp (`/opt/homebrew/bin/llama-server`) on the same GGUF with
     `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144`, and ds4 at `-c 524288`.
   - The same short gate-1 prompts are compared the gate-1 way: probable-token compare with
     gate 1's tolerances.
   - Before ds4 is run, a check confirms that llama.cpp honours the flags for this architecture:
     its log names YaRN, and its output differs from the unscaled run.
3. **Long context** uses the nextgen-eval `longctx` suite.
   - The arm is the registry's Ornith entry plus `-c 524288` and
     `--kv-cache-continued-interval-tokens 0`.
   - The 120K, 240K and 480K needles are tested with real token counts. Docqa is run too.
   - Recorded for each tier: peak wired memory, swap-outs, prefill and decode t/s, and the MTP
     acceptance rate.
4. **Short-prompt quality is measured, not gated.**
   - The Ornith gate-2 eval (HumanEval-mini, GSM8K-mini, code_bench, review) and the four
     abliteration probes run on the 512K build and on the 262K build, the same day.
   - The 512K entry is separate, so a drop is recorded in the results and in the registry status
     text. It does not block.
5. **Server.**
   - `server.log` names the `yarn-2` directory.
   - A second request on a cached prompt above 262K restores from disk instead of a cold prefill.

## Exit check

Sub-project 1 passes when all of the following hold:

- check 1 is byte-identical (Ornith gate 1 and the Qwen fast gate);
- check 2 agrees with llama.cpp within gate 1's tolerances;
- the 480K needle hits with a real prompt within 5% of 480,000 tokens;
- the 512K run neither fails a startup memory check nor crashes.

Swap-outs past 400K are recorded. The user accepted them for Qwen, but Ornith's footprint is about
17 GiB smaller, so none are expected.

Results go to `speed-bench/ornith/512k/REPORT.md`.

## Estimates (not measured)

- **Decode at a full 512K:** about 20 t/s. Each step rereads about 11 GiB of KV. At 128K the measured
  decode is 50.4 t/s.
- **Cold prefill of a 512K prompt:** about 30 minutes, from M6's 511 s at 250K with quadratic
  attention. The disk KV cache makes later turns on the same prefix fast.

## Out of scope

- 1M context. The factor-4 path works by construction, but it is not validated or deployed.
- KV grow-on-demand for Ornith.
- Decode speed work (sub-project 2).
- Changes to the 262K entry.
