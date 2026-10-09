# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

The selected build chooses the backend. Pass `-m FILE` for an explicit model.
Use `--chdir /path/to/ds4` when starting outside the project directory so
relative runtime files such as Metal kernels can be found.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |

The Flash and PRO names accepted by the model endpoints are compatibility
aliases, not separate loaded models. The GGUF passed at startup selects the model.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. `reasoning_effort=max` selects Think
Max only with sufficient context; otherwise it falls back to normal thinking.
`xhigh` maps to normal thinking, not Think Max. Use `think:false`, a disabled
thinking object, or a non-thinking model alias for direct answers.

A hard reasoning cap is off unless the server flag or the request sets one.
`--think-budget N` caps every thinking request at N generated reasoning tokens:
the server inserts the `--think-budget-message` text (default: Qwen's
"Considering the limited time by the user, I have to give the solution based
on the thinking directly now."), then `</think>`, and the model answers in the
same generation. Chat, Responses, and Anthropic requests can also set or lower
the cap with `thinking.budget_tokens`, `thinking_budget`, or
`chat_template_kwargs.thinking_budget`; when both the flag and a request value
are set, the smaller one wins. `/v1/completions` uses only the server flag.
The close happens at the end of the decode block in which the budget runs out,
so reasoning can exceed N by a few tokens; while a tool call is open inside
the reasoning the close waits, up to 2N tokens. Forced tokens count toward
`max_tokens`. The cap is ignored with `--batched-session`.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
preallocates independent KV states and queues requests when all slots are busy.
Choose context and slot count together: a context that fits once may not fit
four times. Idle slots can be cached before reuse; active requests are not evicted.

Where the model supports it, the slots share one prefill workspace instead of
each keeping its own, so an extra slot costs only its caches. That matters most
for Qwen3.8 Flash Next, whose transients are sized by the prefill chunk rather
than by the context: at the default chunk they run to several GiB per session.
The startup line reports both figures.

| Backend/model | Decode execution |
| --- | --- |
| Metal, resident Flash | Native shared-expert/QKV batching where supported |
| Metal, resident V4.1 Flash | Native decoding for 2-8 sessions |
| Metal RDMA TP, V4.1 Flash | Native decoding for 3-8 sessions; ordered fallback for two |
| Metal SSD streaming, V4.1 Flash | Ordered fallback |
| Metal, GLM 5.2 | Ordered fallback |
| Metal, GLM 5.3 | Native batching through 2051 visible tokens; ordered fallback afterward |
| Metal, Qwen3.8 Flash Next | Native batching of the shared work; recurrent state, caches and PLE history stay per session |
| CUDA, supported multi-GPU Flash TP layout | Native grouped decode and mixed prefill/decode |
| Single-GPU CUDA, including Spark | Ordered fallback |

Fallback executes the rows separately. It provides concurrency and scheduling
fairness, not the aggregate speedup of native batching. Native grouping may
change floating-point reduction order slightly. V4.1 sessions containing images
use the ordered fallback.

Long prefills yield to active decoders in bounded intervals, normally 128
tokens. `--mixed-prefill-quantum N` changes that interval for testing.
Session-batched serving uses ordinary target decoding, except Qwen3.8 on
Metal, where `--mtp` also batches speculative decoding. Its
`--mtp-exact-sampling` mode uses ordinary batches for nonzero-temperature
requests. Other models do not use MTP/DSpark while session batching is active.
For the eight-L40S example, see [CUDA GPUs](CUDA_MULTI_GPU.md#serve-multiple-users).

## Images

Start with the matching language GGUF and `--vision FILE`; see
[model-specific instructions](MODELS.md#vision).

OpenAI chat and Responses accept inline PNG/JPEG data URIs. Anthropic accepts
base64 image sources. Remote URLs and server-side file paths are rejected.
Image blocks preserve their order in the request. The limit is 16 images and
a 64 MiB HTTP body.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

`--kv-cache-prompt-end-min-tokens N` (default 0, off) is for Qwen3.8 (not
Ornith) agent sessions that compact. A compaction request repeats the last tool
turn's prompt up to the end of its last message and then adds a user message,
but by then the live KV has moved on through the assistant reply and the
Qwen3.8 hybrid model cannot rewind. With this flag, every OpenAI-chat Qwen3.8
tool-turn prompt of at least N tokens stops prefill at the end of its last
message and stores a checkpoint there (one full-prefix write per such turn),
keyed by its visible text. The compaction request then loads it and prefills
only the new message. Only a later prompt-end checkpoint of the same
conversation marks an older one as the first to evict.

The flag needs the disk KV cache (`--kv-disk-dir`); without it nothing is
stored. A checkpoint is cut only when the end of the last message lies past
the tokens the request already reuses from the live or disk cache: a request
whose reused prefix reaches that point stores nothing new. N must be 0 or at
least `--kv-cache-min-tokens`, otherwise the server exits at startup. Ornith
is excluded because its template is not proven to replay a past turn as the
same bytes; DeepSeek and GLM are excluded too. The startup line
`kv cache prompt-end checkpoints for qwen tool-turn prompts >= N tokens` is
printed whenever the disk cache is on and N > 0, whatever the model, so on
Ornith, DeepSeek or GLM it announces a feature that never fires.

`--rewind-point-min-tokens N` (default 0, off) keeps an in-memory rewind point
on Ornith. For every OpenAI-chat request with tools whose prompt has at least
N tokens, prefill stops at the end of the last message (the boundary the
prompt-end checkpoint uses), and the session keeps its fixed-size state there:
GDN states and conv histories, the MTP carry, the position and the logits. A
later request whose text starts with that prompt up to the cut restores the
point and prefills only what follows. This covers a client re-sending or
retrying the last request, and a Claude Code compaction, which drops the last
reply and adds a user message. It writes nothing to disk and needs no disk
cache. The log shows `rewind point remembered` and `rewind point hit`. Other
models ignore the flag.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace /tmp/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../ds4_kvstore.h) and
[ds4_kvstore.c](../ds4_kvstore.c); model-specific payload handling is in
[ds4.c](../ds4.c).
