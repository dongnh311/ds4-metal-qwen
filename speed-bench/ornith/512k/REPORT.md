# Ornith 512K context (YaRN): report

Spec: `docs/superpowers/specs/2026-10-02-ornith-512k-design.md`. Plan:
`docs/superpowers/plans/2026-10-02-ornith-512k.md`. Branch `feature/ornith-512k`.

## Memory and index audit at `-c 524288`

Read on the branch (Task 3). Nothing is unsafe at 512K; no code change was needed.

| item | sizing | at 524288 | safe |
|---|---|---|---|
| K/V per attention layer (10 + MTP block), `qwen35_graph_alloc` | `ctx_cap * 512 * 2` B, `uint64_t` | 512 MiB x 22 = 11.0 GiB | yes |
| `pos3` | `ctx_cap * 16` B | 8 MiB | yes |
| decode3 partials | 256 splits max, fixed | ~8.5 MB | yes |
| flash / NAX prefill partials | chunk-sized (T) | ~34 MB | yes |
| K/V row offsets in every attention kernel and the prep kernels' K/V writes | `((uint64_t)pos * Hkv + h) * D` | 64-bit | yes |
| disk payload size (`qwen35_payload_body_bytes`) | `uint64_t` | ~11 GB at 480K | yes |

Every other graph buffer is sized by the prefill chunk or fixed. Each layer's K and V are separate
buffers, so no offset crosses layers.

Planned total at 524288: K/V 11.0 + buffers ~1.0 + resident model 21.26 = ~33.3 GiB.

## GPU window A: identity and YaRN correctness

(Task 5)

## GPU window B: long context and server

(Task 6)

## GPU window C: short-prompt quality, measured

(Task 7)
