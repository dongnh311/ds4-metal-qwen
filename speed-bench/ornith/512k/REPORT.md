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

Run 2026-10-02 11:36-11:52 on `909aede0`, with the gateway stack paused. No request reached the
gateway during the window (`last_byte_ago_s` was 1053 at the restore). Receipts are in
`receipts/window-a/`.

| check | result |
|---|---|
| `make test-qwen35-kernels` | pass |
| `ds4_test --qwen35-yarn-engine` (model-backed) | OK; see the details below |
| `tests/ornith/test_loader.sh` | ok, including the new gate: `DS4_QWEN4_YARN_FACTOR=1 -c 262145` refused ("exceeds 262144 native tokens x YaRN factor 1") |
| 262K path vs develop `5b36552a`, gate-1 dumps at prefill chunks 64 / 512 / 2048 | 13 / 13 / 13 dumps byte-identical, 0 differ; gate 1 PASS on both builds |
| Qwen3.8 fast gate (`run.sh fast`) | PASS: vi/code replies byte-identical, registry command unchanged |
| llama.cpp YaRN references (0.5.0, build 11146, `--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144`) | 13 prompts recorded; all 13 differ from the unscaled references, and 7 change a greedy token |
| gate 1 at factor 2 (`DS4_QWEN4_YARN_FACTOR=2`) vs the YaRN references | PASS, compared 106, checked 150, max_delta 0.29 (long_copy), tol 1.48 |
| YaRN discrimination (see below) | 13/13 consistent |
| `test_mtp_cli.py` at factor 2 | PASS: `--mtp` greedy equals plain at every chunk, 369 accepted / 55 rejected drafts |

**Engine test details.** All five cases ran:
- `-c 524288` derived factor 2 "from -c" ("pairs 14..22 blended, mscale 1.0693"), with the table on
  Metal and "KV 11.01 GiB … = 33.26 GiB planned";
- `-c 262144` opened unscaled, and the earlier engine's table was cleared;
- the variable at 2 set the table on `-c 16384`;
- `-c 1048576` at 2 was refused;
- `-c 524288` at 1 was refused.

**YaRN discrimination.** On these short prompts the YaRN effect is far smaller than gate 1's
tolerance: llama.cpp's YaRN and unscaled references differ by at most 0.07 in step-0 top-1 logprob.
So gate 1 passing does not, alone, show that ds4 applied YaRN. The extra check compares the gate-1
dumps:
- the metric is the mean |Δ logprob| over the top-8 shared tokens, on the steps where ds4-YaRN,
  ds4-plain, llama-YaRN and llama-plain still select the same token;
- ds4-YaRN is closer to llama-YaRN than to llama-plain on 13/13 prompts (mean 0.071 vs 0.166);
- ds4-plain is closer to llama-plain than to llama-YaRN on 13/13 (0.074 vs 0.171).

ds4's YaRN therefore matches llama.cpp's. Per-prompt numbers are in `yarn-discrimination.txt`.

## GPU window B: long context and server

(Task 6)

## GPU window C: short-prompt quality, measured

(Task 7)
