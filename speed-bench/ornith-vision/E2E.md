# Ornith native vision: end-to-end acceptance (2026-10-10)

Production Ornith-512K `process_command` from the gateway registry (`-c 524288 --mtp`, YaRN, KV disk cache), on a scratch
port and KV dir, with `--vision mmproj-Ornith-1.5-35B-A3B-f16.gguf` and `--rewind-point-min-tokens 256` so the short
prompts mark. The gateway's ds4 slot was held down for each run (quiet window checked first). Fixtures are synthetic
(`make_fixtures.py`). Script: `e2e_vision.py`; raw result: `e2e-2026-10-10.json`.

| check | what | result | time |
|---|---|---|---|
| C1 | newspaper: names the paper, 1969 and "MEN WALK ON MOON"; log `request images=1` | PASS | 1.8 s (480 prompt tokens, image included) |
| C2 | code screenshot: `parse_invoice_total` and `INV-2041` | PASS | 0.9 s |
| C3 | second turn of an image conversation (tools on) continues live: `cached_tokens` 1004 ≥ first prompt 804 | PASS | 0.37 s |
| C4 | re-send of C3: `rewind point hit`, same reply | PASS | 0.17 s |
| C5 | compaction of C3 (user "summarize" added): `rewind point hit` | PASS | 0.79 s |
| C6 | C1 without `--mtp`: identical reply | PASS | 1.6 s |
| C7 | Anthropic `/v1/messages`, image inside a `tool_result`: names `parse_invoice_total` | PASS | 0.5 s |

Server log over the run: `request images=` 7, `rewind point remembered` 3, `rewind point hit` 2,
`multimodal live kv hit` 3. The planned memory of the 512K slot with vision loaded is 33.26 GiB
(KV 11.01 + buffers 1.00 + model 21.26).

## What the first run found (fixed before this result)

The first run passed C1, C2, C6, C7 and failed C3-C5:
- `prompt_end_text_len_for` refused every request with images, so the rewind point was never cut for an image
  prompt. It now admits images on the rewind-point path only; the disk prompt-end checkpoint still refuses them.
- The memory-text tier compared the raw prompt text with the live text rendered from tokens. An image is a nonce
  marker in one and `<|vision_start|>` + N × `<|image_pad|>` + `<|vision_end|>` in the other, so every turn after
  an image was a full re-prefill (`live kv cache miss … reason=token-mismatch`). The gateway relays Claude Code as
  `/v1/chat/completions`, so this was the production path. The tier now compares a view of the request with each
  marker expanded to its block (block strings resolved at startup via `ds4_engine_vision_block_tokens`) and maps the
  match back to a raw suffix offset; the raw comparison stays as the fallback.
