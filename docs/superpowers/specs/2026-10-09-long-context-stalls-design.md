# Long-context stalls: in-memory rewind point + a quiet-window guard

Date: 2026-10-09. Status: approved by the operator for autonomous execution ("Lên plan làm cả 2",
"Ok, tự chủ chạy đi"). Repos: ds4-metal (engine, this branch) and AI-Gateway-MLX (deploy tooling).

## 1. Problem, measured

The operator reports that sessions slow down badly past about 64K tokens, on Ornith and Qwen3.8, and
also on strata. Production data (analytics plus `gateway.log`, 10 days) splits this into three parts.

1. **Decode** slows gradually with depth: there is no cliff. Ornith-512K decode runs at 92 t/s
   below 8K, 68 at 48-64K, 64 at 64-80K, 56 at 96-128K, 49 at 128-192K, and 43.5 above 192K. The
   cause is f16 KV reads, about 24 KB per token per generated token. This is 85% of model time in
   the 2026-10-09 session, and it is **out of scope here** (physics, or lossy KV).
2. **Stalls of 2-3.5 minutes.** On the hybrid models (GDN recurrent layers plus attention), a
   request that is not an extension of the live state needs a disk checkpoint at or before the
   divergence point, because recurrent state cannot be rewound. The 512K slots run with
   `--kv-cache-continued-interval-tokens 0` (since the 09-30 180 GB write incident), so the
   fallback is the cold checkpoint at 32K. On 2026-10-09, session `695d68f3`:
   - **16:13:39**: `live=205288 prompt=171139 common=122454`. The checkpoint key texts show this
     request is the **16:09:54 request re-sent**: the same prompt up to the end of the same
     143-byte user message, with the 85-token `finish=stop` reply dropped. The fallback was the
     32768 checkpoint, so 138371 tokens were prefilled and the turn took **202 s**.
   - **16:46:49**: a **Claude Code compaction** request (`client-compact: summary` in the gateway
     log). It is the 16:45:10 prompt up to the end of its last `tool_response`, plus a 6369-byte
     user message, with the 2665-token reply dropped. A thinking-visible evict key happened to
     exist from the 16:37 re-stage, so 8.5K tokens were prefilled in 28.7 s. Without that key,
     the fallback was 32768, which means about 184K tokens, or 5+ minutes.
   - **15:41:36**: a re-stage (`f3473197`) cut an in-flight request at 15:38:06, then Tier L took
     the ds4 slot. The next turn was not an extension of any stored key, so 89687 tokens were
     prefilled and the turn took **116.5 s**.
   The render does **not** change at a new user turn. The gateway sends `preserve_thinking`, so
   every past assistant turn renders an empty `<think>\n\n</think>\n\n` block, and the 16:46
   prompt byte-extends the 16:37 key. The divergence is always "the client re-sends the
   previous request's prompt up to the end of its last message, then adds or repeats a user
   message": a retry, a re-send, or a compaction.
3. **Operational:** a re-stage plus Tier L during a live session. The idle check was 20 s of
   `/status` idle, and a Claude Code tool gap is longer than that.

Qwen3.8's renderer keeps reasoning in tool context too. Its known stall is the same compaction
shape (feature/prompt-end-checkpoint `bd55adcf`: 74-80K re-prefilled, 247-415 s per compaction).
Its decode is flat (4-bit KV).

## 2. Goals and non-goals

Goals:
- **G1.** A request that re-sends the previous request's prompt up to the end of its last message,
  then adds any suffix (retry, re-send, compaction), continues from the live session with **no
  re-prefill** of that prefix, on Ornith. Target: on the 16:13 shape, TTFT drops from 202 s to
  under 5 s; on the compaction shape it drops to the time to prefill the new user message.
- **G2.** No disk writes, no new disk budget, and no change to what the model sees (the KV and
  recurrent state after a rewind are exactly those a live prefill of the same tokens produced).
- **G3.** Re-stage, Tier L/A, and ds4 slot stops do not run while a real user session is active.
- **G4.** Qwen3.8 gets the compaction fix through the existing disk prompt-end checkpoint
  (`bd55adcf`), enabled only if a measured per-turn write cost is acceptable (§5.4).

Non-goals: decode speed at depth (lossy KV and kernels, a separate track); periodic continued
checkpoints on the 512K slots (no evidence they would beat G1, and their write cost caused the
09-30 incident); strata (a separate machine, the same engine, so it inherits the ds4 change when
it updates).

## 3. Design A: the rewind point (ds4-metal)

### 3.1 Idea

A hybrid session cannot rewind because its recurrent state at an earlier position is gone. The
attention KV, the MTP layer KV and the mrope rows are **append-only per position**, so their rows
`[0, P)` are still intact after the session moves past P. Only the fixed-size state is lost:
- the GDN states and conv histories (about 63 MB on Ornith);
- the MTP hidden-state carry and its row count;
- `pos` and `mrope_delta`;
- the logits row.

The design keeps **one in-memory copy of that fixed-size state at the prompt end of the latest
request** (the "rewind point"). Rewinding to P then means restoring that copy and setting the
session length to P. The rows `[0, P)` are reused untouched.

The prompt end is where bd55adcf already cuts prefill: the end of the last message, right before
the generation prompt, a clean token boundary opening with `<|im_start|>`. Each request
re-renders every past message byte-identically (preserve_thinking, thinking-visible contract). A
re-send, retry or compaction therefore begins with the previous request's prompt text up to that
cut.

### 3.2 Engine API (ds4.h)

```c
/* Snapshot the session's fixed-size state at its current position so that
 * ds4_session_rewind(s, ds4_session_rewind_point_pos(s)) can restore it exactly.
 * Ornith (qwen35) only; returns false (and keeps no point) otherwise or on
 * allocation failure.  One point per session: a new mark replaces the old. */
bool ds4_session_mark_rewind_point(ds4_session *s);
/* Position of the valid rewind point, or -1. */
int ds4_session_rewind_point_pos(const ds4_session *s);
```

`ds4_session_rewind(s, pos)` gains one case. For qwen35 with a valid point at exactly `pos`, it:
- restores the snapshot by copy, so the point stays valid for repeated re-sends;
- sets `checkpoint.len = pos`, `mtp_pos` and the MTP carry from the snapshot, and the logits;
- keeps `checkpoint_valid`, which is `state_ok`.

Every other rewind behaves as today.

**Invalidation.** The point is valid only while rows `[0, P)` are untouched. It is invalidated by:
- a rewind to a position below P;
- `qwen4_graph_reset`;
- `ds4_session_invalidate`;
- a payload load (a disk hit), which overwrites rows;
- any path that clears `checkpoint_valid`.

A rewind to a position at or above P that does not hit the point leaves it valid, because rows
`[0, P)` are untouched.

**Memory.** One snapshot set per session: the GDN state and history buffers, allocated lazily
through `qwen4_graph_ensure_snapshot`, plus the MTP carry (`DS4_N_EMBD` floats) and one logits row.

### 3.3 Server (ds4_server.c)

- **Merge `feature/prompt-end-checkpoint`** (bd55adcf, 429831ae, a84c5688) into this branch. The
  only conflict is in `ds4_kvstore.c`, where both sides add a function at the same place, so keep
  both. Its pure split plan and the "prefill to the cut, act, then evaluate the generation
  prompt" consumer are reused.
- **New option `--rewind-point-min-tokens N`** (default 0 = off; N must be ≥ 0). It does not need
  the disk cache. Eligibility (`prompt_end_text_len` in rewind mode):
  - an OpenAI-chat request with no images and not Responses or Anthropic;
  - Qwen syntax, Ornith **included**;
  - the prompt ends with `<|im_end|>\n` plus the generation prompt;
  - the prompt has at least N tokens and the cut lies past the tokens already reused.

  bd55adcf's disk gate (`--kv-cache-prompt-end-min-tokens`) keeps its own rules, Qwen3.8 only.
  When both apply, one cut serves both.
- **At the cut:**
  1. sync the prefix;
  2. `ds4_session_mark_rewind_point(session)`;
  3. remember `slot->rewind_point = {text = prompt_text[0, cut_bytes), live_tokens = cut}`;
  4. evaluate the generation prompt.

  A failed mark clears `slot->rewind_point`.
- **New reuse tier `REUSE_REWIND_POINT`** in `slot_probe_reuse_locked`. It is placed after
  `REUSE_THINKING_VISIBLE` and `REUSE_MEMORY_TEXT` and before falling through to disk. It is
  selected when all of these hold:
  - `slot->rewind_point.valid`;
  - `ds4_session_rewind_point_pos(session) == slot->rewind_point.live_tokens <= live_pos`;
  - the request's prompt text byte-prefix-matches the point's text and is longer than it;
  - the request has no images.

  It gives `reuse_tokens = live_tokens` and `suffix_off = text_len`. Routing shares the probe, so
  a re-send is routed back to its slot.
- **Materialization:**
  1. under `inference_mu`, call `ds4_session_rewind(session, P)`;
  2. verify `ds4_session_rewind_point_pos(session) == P` and `checkpoint_valid`;
  3. build `build_live_prompt_suffix(..., prompt_text + suffix_off)`;
  4. sync.

  Any failure becomes a clean miss to the disk path, as `REUSE_MEMORY_REWIND` already does. Log
  `ds4-server: rewind point hit pos=P live=L prompt=N` and, at the mark,
  `ds4-server: rewind point remembered pos=P text=B`.
- **Clearing:** `request_live_state_clear` and every path that clears `thinking_live` because the
  slot's state was replaced also clear `slot->rewind_point`.

### 3.4 Correctness evidence required

- **Engine equivalence (model-backed, Ornith GGUF, production slot stopped).**
  1. Prefill A (about 6K tokens).
  2. Mark.
  3. Eval B (about 300 tokens: a reply).
  4. Rewind to |A|.
  5. Eval C (about 200 tokens).

  Then compare against a fresh session that prefilled A+C:
  - greedy argmax is identical at every row of C;
  - max |Δlogit| ≤ 1e-2 at every row (same batch shapes, so in practice bit-equal);
  - a second rewind to the same point gives the same result.

  The same comparison runs with MTP on.
- **Server model-free tests:** tier selection (match, stale position, shorter or equal text, images,
  point cleared), option parsing, and the eligibility helper with Ornith admitted.
- **Server end-to-end (scratch port, Ornith).** A Claude-Code-shaped conversation at about 64K:
  - a re-send gives a `rewind point hit` with TTFT under 5 s, where the baseline binary re-prefills;
  - a compaction-shaped request prefills only the new message;
  - greedy output is identical between the rewind path and a cold prefill of the same prompt.

## 4. Design B: the quiet-window guard (AI-Gateway-MLX)

- **`scripts/quiet-window.py`**:
  - It is **quiet** when `/status` lists no `active` entry with `canary` false, **and** analytics has
    no event in the last `--quiet-s` seconds (default 600) with all of: `is_test = 0`,
    `is_canary = 0`, and a UA that does not start with `ai-test-suite`, `ai-journey-suite` or
    `ai-gateway-eval`.
  - A gateway that does not answer counts as quiet (there is nothing to protect, and a restore
    must never be blocked).
  - `--wait-s S` polls every `--poll-s` (default 15) until quiet or until S runs out.
  - It prints one line: `quiet: …` or `busy: last real request N s ago (door, conv) …`.
  - Exit codes: 0 quiet, 3 busy.
  - `AI_GW_IGNORE_QUIET=1` prints `quiet check BYPASSED` and exits 0.
- **`scripts/restage.sh`** is the supported re-stage entry point:
  1. `quiet-window.py --wait-s ${AI_GW_QUIET_WAIT_S:-1800}`;
  2. back up `~/.local/ai-gateway/gateway` to a timestamped directory;
  3. run `install-launchd.sh`;
  4. print the live `runtime_code_hash`.

  `install-launchd.sh` itself is unchanged, so restore and automation paths are unaffected.
- **`tests/run-success-gate.py`**: when the requested tiers include L or A, run `quiet-window.py`
  (wait `AI_GW_QUIET_WAIT_S`, default 0) after the live-target safety check. Busy means exit 2
  with summary `L-quiet`.
- **Docs:** repo CLAUDE.md rules #11 and #12 name `scripts/restage.sh` as the re-stage path and
  the quiet check before Tier L. The ds4 deploy procedure (`DEPLOY_AI_GATEWAY.md`) runs
  `quiet-window.py --wait-s` before stopping the ds4 slot.
- **Tests:** a hermetic `tests/test-suite-quiet-window.py` registered as `H.quiet_window`. It
  uses a temp analytics DB and a stub `/status` server, and covers:
  - test UAs, canaries and `is_test` ignored;
  - a real request inside or outside the window;
  - an in-flight non-canary request;
  - the gateway down;
  - wait then success;
  - wait then timeout;
  - the bypass.

## 5. Rollout

1. **B first:** branch, Tier H, merge, re-stage through `restage.sh` itself, then Tier L under the
   quiet check.
2. **A:** branch, model-free tests, model-backed tests in a quiet window with the slot stopped,
   then the E2E benchmark, baseline against new.
3. **Deploy A:**
   - merge to develop and push `origin`;
   - `deploy-ai-gateway.sh cut rewind-point`, then `install`, then `smoke --model` for every ds4
     row (the default flags are off, so nothing else changes);
   - back up the registry, then add `--rewind-point-min-tokens 16384` to both Ornith rows;
   - restart the slot when quiet.

   Rollback: `install` the previous `prod/` branch from `.deploy-history` and restore the registry
   backup.
4. **Qwen3.8 (G4):** on a scratch run, measure the prompt-end store size and save time per tool turn
   at about 64K and 128K. Enable `--kv-cache-prompt-end-min-tokens 65536` on the Qwen3.8 rows only
   if a store is ≤ 1.5 GB and ≤ 0.5 s at 128K. Otherwise record the numbers and leave it off.
5. **Verify in production:** on the next real session, `rewind point hit` lines appear on re-sends
   and compactions, and no stall over 30 s remains whose `common` lands at a previous prompt end.

## 6. Risks

- **A missed piece of state makes rewound output wrong.** The engine equivalence test (bit-level
  logits and greedy over 200 rows, with MTP) gates the merge. The point is Ornith-only. Qwen3.8 has
  an indexer key ring and block pooling and is excluded.
- **A stale point after a disk load or reset.** The engine invalidates on every such path, and the
  server double-checks `rewind_point_pos` before use.
- **Overhead per request:** one extra small batch (the generation prompt, 3-5 tokens) and a
  63 MB GPU copy, under 30 ms at depth.
- **The guard blocks a needed deploy:** `AI_GW_IGNORE_QUIET=1`, and `install-launchd.sh` and the
  restore paths are untouched.
