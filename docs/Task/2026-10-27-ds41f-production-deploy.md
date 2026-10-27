# DeepSeek-V4.1-Flash-1017 (ds41f) — production slot, 256k context

Status: **serving** on the ai-gateway as `local--DeepSeek-V4.1-Flash-1017-ds41f`
(gateway `:8090` → ds4 `:18090`), since 2026-10-27.

## 1. What is deployed

| Item | Value |
| --- | --- |
| GGUF | `~/.local/share/ai-gateway/ds4-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf` |
| Size / shape | 405,224,791,632 B (377 GiB), 1010 tensors, **130 KV** (merged 5-file ISTA GSQ-RCO package, engram tables grown in) |
| Quantization | gate/up `IQ2_XXS`, down `Q2_K`, dense `Q4_K`/`Q8_0`, `token_embd` F16, engram `lloyd4_e5m3_32_r136`, `hc_*_fn` F16, `ffn_gate_inp` F32 |
| Binary | `~/.local/share/ai-gateway/ds4-metal/ds4-server-ds41f` (built 2026-10-27 from the ds4-metal worktree; kept separate from the Ornith `ds4-server`) |
| Port | `127.0.0.1:18090` |
| Context | **`-c 262144`** (policy for all three local models; the 512k entry is disabled — it plans 53.99 GiB and starves the box) |
| Expert cache | **`--ssd-streaming-cache-experts 2600`** (24.10 GiB) in the final config — 41.92 GiB planned total. With 512k/4300 the fork plans 53.99 GiB and starves the rest of the stack |
| KV disk | **disabled** — see §4 |
| Gateway id | `local--DeepSeek-V4.1-Flash-1017-ds41f` (models.json + runtime-registry.json) |

Launch command (exactly what `runtime-registry.json` declares):

```sh
~/.local/share/ai-gateway/ds4-metal/ds4-server-ds41f \
  --metal -m ~/.local/share/ai-gateway/ds4-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-ds41.gguf \
  -c 524288 --ssd-streaming --ssd-streaming-cache-experts 4300 \
  --host 127.0.0.1 --port 18090
```

Startup memory plan (printed by the backend):

```
memory: KV 3.13 GiB (raw 0.01 + compressed 3.13) + buffers 10.08 GiB
        + resident model 7.45 GiB + expert cache 26.21 GiB
        + prefill expert reserve 7.12 GiB = 53.99 GiB planned
V4.1 SSD cache fitted from 4300 to 3595 experts for context/runtime headroom
→ dynamic cache 2827 experts (9.49 MiB/expert)
```

## 2. Measured performance (M5 Pro 64 GB)

Standalone CLI (no gateway, machine otherwise idle):

| Prompt tokens | Prefill | TTFT | Decode |
| --- | --- | --- | --- |
| 1,978 | 114 t/s | 17.3 s | 8.60 t/s |
| 7,522 | 275 t/s | 27.3 s | 8.47 t/s |
| 30,118 | 333 t/s | 90.4 s | 7.37 t/s |
| 60,190 | 401 t/s | 150 s | 7.01 t/s |
| 120,334 | 436 t/s | 276 s | 6.19 t/s |

Through the gateway with the full stack running (Orca/opencode/sidecars present):

| Request | Prefill | Decode |
| --- | --- | --- |
| 4,549-token prompt | **207 t/s** (22 s) | **4.77 t/s** |
| 454-token append | ~25 t/s when KV-disk was on, ~200 t/s without | 4.75 t/s |

The TTFT floor is ~12–15 s even for a short prompt: the first pass faults the
expert pages in and seeds the cache. The gateway adds ~0.3 s plus its own system
prompt (~193 tokens for a one-liner).

## 3. Switching / rollback

Only one ds4 model fits at a time (ds41f plans 53.99 GiB; Ornith plans 22.4 GiB).

```sh
# back to the Ornith prod slot (22.4 GiB, prefill ~1.4–2.1k t/s, decode ~70 t/s)
bash ~/.local/bin/server-power.sh on

# or explicitly
bash ~/.local/bin/proxy-stack.sh ensure Shiftedx--ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx

# stop ds41f by itself (ONE SIGTERM; the server persists its KV on the first signal)
kill -TERM "$(cat ~/.local/share/ai-gateway/ds4-ds41f.pid 2>/dev/null || pgrep -f ds4-server-ds41f)"
```

`proxy-stack.sh ensure <ds41f-id>` will not launch a ds4 backend by itself (the
launcher only owns MLX runtimes). Either start the backend with the command in
§1 first, or let `ai-watchdog.sh` `heal_runtime` start it from the registry
`process_command` — that path is exercised once the model is the active one.

## 4. Why KV-disk was removed (the 8× prefill finding)

The first deployment carried the Ornith-style KV persistence flags
(`--kv-disk-dir … --kv-disk-space-mb 32768 --kv-cache-cold-max-tokens 524288
--kv-cache-min-tokens 4096 --rewind-point-min-tokens 16384`). Measured on the
same 4,549-token prompt through the gateway:

| Config | Prefill | TTFT |
| --- | --- | --- |
| with `--kv-disk-*` | **24.5 t/s** (18.4 s for 453 tokens) | 49.3 s |
| without `--kv-disk-*` | **207 t/s** (21.9 s for 4,549 tokens) | 24.4 s |

The 512k `--kv-cache-cold-max-tokens` makes every new prefix pay a large KV
persist/restore cost; ds41f gains no useful prefix reuse from it (its prompt
prefix is short relative to the model), so the flags are excluded.

The remaining gap to standalone (decode 4.77 vs 8.6 t/s) is **memory pressure**:
the full stack leaves ~44 % RAM free, swap in use, and the expert cache cannot
stay fully wired. Stopping the vision/image sidecars or dropping `-c` to 262144
(frees ~6 GiB of context buffers and lets the cache grow) recovers most of it.

## 5. Known limits

- First token after a start is slow (12–15 s cold start) — send one warmup
  request after launch if the slot is user-facing.
- 512k context costs 13.22 GiB of context buffers and shrinks the expert cache
  4900 → 2827 experts; for interactive use prefer `-c 262144` (future slot).
- Prefill is the product of the streaming expert cache, not the gateway: the
  same model does 275–436 t/s in the CLI, so the server path is within ~1.3×.

## 6. Final prod config and the memory-pressure finding

The 512k/4300-slot plan (53.99 GiB) cannot coexist with the rest of the stack on
64 GB: free RAM dropped to 25 %, swap grew, and prefill for the same 4,549-token
prompt degraded from 22 s to 35 s while decode fell to 4.7 t/s.

Final deployed config — `-c 262144 --ssd-streaming-cache-experts 2600`:

| Metric | Value |
| --- | --- |
| Planned memory | KV 1.57 + buffers 8.80 + resident 7.45 + cache 24.10 = **41.92 GiB** |
| Free RAM after start | ~35 % (vs 25 % with the 512k plan) |
| Decode | **9.8 t/s** (as good as the standalone CLI 8.6–9.8) |
| Prefill, 4,549-token prompt | 22–40 s through the gateway → **115–207 t/s** (CLI: 275 t/s) |
| TTFT | ~24 s first request; repeat requests vary with expert locality |

Rule of thumb for this box: **ds41f gets the machine, or it gets a small cache.**
With a full stack (gateway + sidecars + browser/other apps) keep the cache at
~2600 and ctx at 262144; for long-context work stop the sidecars first and use
the 512k config. Never enable KV-disk persistence for this slot.

## 7. Vision for ds41f — encoder + router biases wired (2026-10-27)

What was done:

1. Downloaded the official V4.1 vision encoder (926 MB) via
   `./download_model.sh ds41f-vision` → `ds4-models/DeepSeek-V4.1-Flash-Vision.gguf`.
2. Fetched the 40 **visual router biases** (`layers.N.ffn.gate.bias_vl`, F32[384])
   from `deepseek-ai/DeepSeek-V4.1-Flash-1017` with HTTP range requests (only KBs;
   the safetensors index names the shard, then the header + tensor bytes).
3. Injected them into the **encoder** GGUF as `blk.N.exp_probs_b_vl.bias`
   (encoder header had 14 KB slack, so no data moved).
4. Patched `ds4.c` so the language model falls back to the encoder for those
   biases (they need the encoder's map; the fused router path is skipped when
   they come from the encoder), and so the encoder may carry extra trailing
   tensors (266 → 306).
5. Wired `--vision` into the ds41f registry entry, `capabilities.vision: true`.

Verified: the server logs `V4.1 visual router biases read from the vision encoder`
and an image request produces `vision embedding encoded; cache=3833597 bytes` +
`request images=1 prompt=259`.

**Still blocking end-to-end vision:** the language GGUF carries **no vision token
metadata** (no `deepseek41.vision.*` image-token ids), so the server cannot put
image placeholders into the prompt — the model then answers "no image". Next
step: add the image marker KVs/template (header slack in the language GGUF is
only 849 B, so they must come from a sidecar/flag rather than a header edit, or
one large KV must be freed), then re-test with the same PNG.

Also note: an image request costs 24–37 s on this slot (8–10 t/s decode).
