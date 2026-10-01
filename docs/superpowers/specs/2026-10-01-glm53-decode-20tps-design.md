# GLM-5.3 decode toward 20 t/s on a 64 GB Mac — program and sub-project 1 (decode gates) design

Date: 2026-10-01. Status: design approved section by section in conversation ("Ok đồng ý", "Ổn" x3).
Branch: `feature/glm53-decode-gates` (from origin/develop d1b56556).

## Goal

Raise single-stream decode of GLM-5.3-Flash DogContext Q2 (IQ2_XXS gate/up + Q2_K down, 288 experts,
8 routed + 1 shared) on the M5 Pro 64 GB from ~9.2-9.4 t/s to **at least 20 t/s**, with
`ds4-server --metal --ssd-streaming -c 262144`, without changing any generated token.

## Constraints (user-confirmed)

- **Byte-identical output is a hard requirement.** Greedy output must match the current decode path
  token for token. Only scheduling may change; kernels, their per-slot rows and the expert summation
  order stay as they are.
- **GLM and Metal only.** No behaviour change for Qwen3.8/qwen4 (PROD), Ornith, DeepSeek V4.1, ROCm or
  CUDA. The Qwen regression gate (speed-bench/qwen-regression, full tier) must pass for any change to
  shared code.
- **Memory:** stay within the current 256K plan (~48 GiB); no new multi-GiB buffers.
- **Phased with measurable gates**: every sub-project must show its own measured gain before the next
  one starts; a gate miss stops the program for a re-evaluation.
- Effort: about 1-3 weeks for sub-project 1, about 1-2 months for the whole program. Honest odds of
  reaching 20 t/s: ~30-40%. The realistic floor of value is sub-project 1 alone (~12.5-14 t/s).

Non-goals: prefill speed (already ~130 t/s after the 2026-10-01 work), ROCm/CUDA, two-Mac TP,
`--quality`, deployment of GLM to the gateway.

## What the measurements say (2026-10-01, develop 46f90a46)

- Wall time ~110 ms per decode token (9.0-9.4 t/s). GPU busy **54-56 ms per token** (env
  `DS4_METAL_GPU_BUSY_PROFILE`, 8 vs 136 generated tokens differenced), so the GPU idles ~50%.
- **~75.5 waited command buffers per token**: one `end_commands` per MoE layer inside
  `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` (ds4_metal.m ~18045), plus a flushed
  shared-expert buffer whenever the layer missed (~0.8 of layers).
- Expert misses: ~70 per token, ~470 MiB, read by `pread` at ~20 GB/s effective (mostly page cache),
  **~22 ms per token**, blocking. Per-miss `F_RDADVISE` fcntl calls (~210 per token) run on the main
  thread first.
- More expert cache does not help: 4546 -> 5443 slots raised the hit rate 0.769 -> 0.803 with no
  speed gain (the page cache was already a second-level cache). `-c 65536` gained ~2%.
- The async selected-load worker (V4.1 style) measured 4.8% slower (8.98 vs 9.43 t/s); Metal keeps it
  opt-in.
- Ceiling arithmetic: with zero idle the GPU-busy floor gives ~18 t/s. 20 t/s (50 ms/token) therefore
  also needs less GPU work per token and/or more than one token per forward (MTP).

## Program

| Step | Content | Gate to pass | Stop rule |
|---|---|---|---|
| Phase 0 | Measure only (env): split readahead/pread/install time; A/B disable expert readahead; greedy with and without `--mtp` (MTP byte-identity under streaming) | numbers recorded | none |
| **SP1** | **Decode gates**: no host wait inside a token (this document) | >= 12.5 t/s, byte-identical, GPU busy >= 70%, Qwen gate full PASS | < +15% (~10.8 t/s): stop and re-evaluate |
| SP2 | Expert prefetch: predict layer L+1 experts and load them while layer L runs; GPU-side hit/miss classification | recall spike >= ~0.6 first; then ~15-16 t/s | recall < 0.6: drop; < 14 t/s after SP2: re-evaluate 20 t/s |
| SP3 | Byte-identical decode kernel speed-ups (memory access, occupancy, specialization; same arithmetic, precedent: qwen4 M5 MoE specialization +20% identical) | GPU busy 55 -> <= 47 ms/token | no measurable kernel gain: skip |
| SP4 | MTP with streaming; verification must use arithmetic identical to single-token decode | >= 20 t/s, byte-identical | verify not byte-identical and not fixable: stop at SP3 result |

Each sub-project gets its own spec and plan; this document fixes the program and designs SP1.

## SP1: decode gates

### Approach

Port the qwen4 "stream gates" (qgate, ds4_metal.m ~52695-53570, "SCALE-3A", on by default for qwen4
PROD) to GLM decode. The per-layer drain (commit, wait, read ids, load, re-encode) is replaced by a
gate: the main thread encodes the whole token ahead and never waits; after each MoE layer's router a
publish kernel copies the selected ids to a mailbox and the batch is committed; the next batch opens
with `kernel_dsv4_tp_poll_release`; a service thread reads the mailbox, resolves the experts into
the cache, writes the gate's address table and releases the poll.

Rejected alternatives: incremental host-side fixes only (flush earlier, drop fcntl, cheaper victim
scans; ~8-20 ms/token, does not build the base SP2/SP4 need); GPU-side hit/miss classification now
(belongs to SP2).

### Components

1. **Shared stream-gate core.** Extract the model-neutral parts of qgate into an API used by both
   qwen4 and GLM: mailbox publish, poll regions and release, the gate ring of fresh address tables,
   the service thread and its queue, expert resolution into the stream expert cache, the gate-owned
   fallback buffer, the eviction guard for experts referenced by pending gates, timeouts, split
   (two-pass) gates and the lookahead hook. The extraction is its own first step and must keep qwen4
   byte-identical (Qwen gate full + existing qwen4 tests).
2. **GLM client (ds4.c, GLM decode FFN ~50520-50900).** For streamed IQ2 layers (generic routed MoE,
   `glm_stream_selected_expert_cache_supported`), when gates are enabled: encode router + select, the
   publish, then the **shared expert before the gate point** (GPU work that overlaps the service
   thread), commit without waiting, then the poll and the routed MoE.
3. **Routed kernels unchanged.** `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`,
   `kernel_mul_mv_addr_q2_K_f32` and `ds4_gpu_encode_moe_sum_experts` already index a device address
   table with the router's own ids (moe.metal ~4001, ~4143). In gate mode they read the gate's ring
   table instead of the per-layer table. Per-slot rows and the summation order are unchanged.
4. **Cache ownership.** While gates are pending only the service thread touches the expert cache.
   Every other user (prefill, MTP verify, session save/load/rewind, the drain path) starts with
   `end_commands`, which cannot return before all pending gates are released.
5. **Start condition.** Gates start once every cache slab exists (all slabs must be resident because
   ids are unknown at encode time), as qgate does; until then the drain path runs.
6. **Scope switches.** Metal only; streamed IQ2 GLM layers only; off under TP world 2 and `--quality`;
   supersedes the opt-in async worker; `DS4_GLM_STREAM_GATE=0` restores the drain path.

### Per-token data flow

Main thread, per MoE layer L: attention L, FFN norm, router and select (ids stay on the GPU) ->
publish 8 ids tagged with the gate sequence into mailbox slot r -> shared expert -> commit (no wait),
open a new batch -> poll on region r -> routed MoE reading ring table r -> expert sum and residual ->
layer L+1. After the last layer: output head and **one wait per token** for the logits/argmax.

Service thread, per gate r: spin on mailbox r -> look up each id in `g_stream_expert_cache[layer]` ->
hit: take its address; miss: pick a victim and pread into the slot; no room: gate fallback buffer ->
write the 8 addresses into ring table r, pin the entries for gate r -> release region r. **Split
gates**: release a first pass for the cached experts while the misses load, then a second pass; each
expert still writes its own row and the sum stays in slot order.

Expected idle accounting (55 ms/token today): submit/wake/buffer creation ~7-8 ms, main-thread encode
and bookkeeping ~10-15 ms, part of the pread hidden behind cached experts ~5-10 ms; ~20-25 ms of
un-hidden pread and resolution remain for SP2. Expected: ~75-85 ms/token, **~12-13.5 t/s**.

### Error handling

- **Poll timeout (~600 ms) or a fallback failure**: the poll kernel exits with a timeout status; the
  end-of-token wait sees it, **drops that token and fails the request** (no token computed from an
  unreleased gate is ever emitted), disables gates for the session (drain path from then on) and
  invalidates the session checkpoint so the next request rebuilds (the KDA recurrent state already
  absorbed the bad step).
- **Pinned experts**: entries referenced by a pending gate cannot be evicted or overwritten until the
  GPU has passed that gate (qgate guard); ring tables are reused under the same rule, so the GPU never
  reads stale cached lines of a table written after its command buffer started.
- Interactions: prefill, MTP and session operations drain first; TP and quality keep the drain path;
  ROCm/CUDA untouched; qwen4 keeps its current behaviour through the shared core.

### Invariants (each becomes a test)

1. The GPU reads ring table r only after gate r is released.
2. Gate mode and the drain path produce byte-identical outputs at every step.
3. An expert used by gate r is never evicted or overwritten before the GPU passes gate r.
4. A timed-out gate never yields an emitted token.

### Testing

- **Model-free, TDD:** qwen4 suite unchanged after the core extraction plus CPU unit tests for the
  ring/guard bookkeeping; a new Metal test (test_metal_ssd_experts style, synthetic GLM-like IQ2/Q2
  8-expert model) running many steps through gates and through the drain path, byte-identical, with
  evictions, misses, the fallback buffer and an injected timeout (must fail loudly).
- **Model, GPU window with PROD off:** greedy 16 and 256 tokens identical to `DS4_GLM_STREAM_GATE=0`;
  decode A/B (256 tokens, two interleaved reps) >= 12.5 t/s; GPU busy >= 70%; waited command buffers
  per token ~1-2; server smoke (prefill speed, multi-turn reuse, selected-chunk threshold) not worse;
  Qwen gate full PASS with Qwen decode >= 97% of PROD; injected timeout fails the request, disables
  gates and the next request succeeds.

### Risks and unknowns

- Poll-kernel spinning and the service thread compete with the GPU for memory bandwidth and the CPU;
  qwen4 shows it pays off, GLM's 8-expert layers and HC/KDA attention may differ.
- Slab preallocation: gates need all slabs; a cold start runs ~65 tokens on the drain path.
- The shared-core extraction touches PROD code; mitigated by doing it first and gating on Qwen full.
- The remaining pread (~20 ms/token) caps SP1 near 13-14 t/s; SP2 has to hide it.
