# Ornith M4 run notes

## Preconditions (2026-09-27 00:05)
- Branch `feature/ornith-m4` cut from develop `b9f1aca` (M3 merge; M2 merge `e68e8b2` is an ancestor).
- Qwen3.8 gate: full gate PASS on the M3 branch code (`speed-bench/ornith/m3/QWEN_GATE.md`, paired decode 100.2% of PROD);
  fast gate PASS on the merged tree (`5843038` == `b9f1aca` tree).

## M2/M3 interfaces consumed by this plan (grep on b9f1aca)
| symbol | result |
|---|---|
| `ds4_session_qwen35_spec_cycle` (ds4.c) | present (3 hits) |
| `qwen35_graph_mtp` (ds4_qwen35moe.inc) | present |
| graph fields `mtp_h`, `mtp_h_pos0`, `mtp_h_rows` (ds4.c) | present (3/3) |
| `qwen35_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp)` | exact match |
| `qwen35_graph_forward_tokens(..., const int *tokens, uint32_t T, float *logits_out, bool all_rows)` | exact match |
| `DS4_QWEN35_SPEC_TRACE`, `DS4_QWEN35_SPEC_FORCE_ACCEPT` | present |
| `ds4_engine_mtp_draft_tokens` | 2 for Ornith with `--mtp` on Metal |
| `ds4_engine_is_qwen35moe` (ds4.h) | present |
| server ids `ornith-1.5-35b-a3b` | present |
| `DS4_QWEN35_PAYLOAD_TAG` | present |
| "milestone M3" refusals in ds4_server.c / ds4_agent.c | none (removed) |

`--mtp-timing` prints `ds4: Ornith mtp: N verify cycles, M drafts accepted (P%)` at session free.

## Staging server probe (port 18296, 00:10, live stack paused)
`./ds4-server --metal -m "$DS4_ORNITH_MODEL" -c 262144 --mtp --kv-disk-dir <tmp> --port 18296`: up in 4 s (warm page cache);
`/v1/models` = `ornith-1.5-35b-a3b`, `ornith-1.5-35b-a3b-chat`, `ornith-1.5-35b-a3b-reasoner` (context_length 262144);
a fresh-nonce chat returned `model` `ornith-1.5-35b-a3b`, `usage.prompt_tokens` 173, `usage.prompt_tokens_details.cached_tokens` 0.
Stopped with SIGTERM in 1 s.

## GGUF tiers
- 23G: `Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-23G-ICE.gguf` (22,836,518,208 bytes).
- 25G: download started 2026-09-27 00:02 with the user's OK (`hf download gbuzhf/...-ICE-GGUF ...-25G-ICE.gguf`);
  `DS4_ORNITH_MODEL_25G=$HOME/orca/workspaces/ds4-metal-data/gguf/ornith/Ornith-1.5-35B-A3B-Abliterated-CyberTiel_Calibrated-MTPv2-25G-ICE.gguf`
  (expected 24,849,784,096 bytes; the size is checked before gate 2).
