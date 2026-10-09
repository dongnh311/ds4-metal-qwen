# Deploy receipt: rewind point (2026-10-09)

| | |
|---|---|
| prod branch | `prod/rewind-point-20261009` @ `0f2cff2` (develop merge of `feature/rewind-point`), cut and pushed to `origin` (dongnh311/ds4-metal-qwen) |
| previous | `prod/kv-evict-minors-20261009@8acc108` (`.deploy-history`; rollback = `./deploy-ai-gateway.sh install prod/kv-evict-minors-20261009` + restore the registry backup below) |
| window | quiet-window `quiet:` → `/tmp/ds4.lock` held by a flock holder 22:30:20–22:31:47 (the gateway's one relaunch attempt refused) → install → 4 smokes with `DS4_LOCK_FILE=/tmp/ds4-smoke.lock` |
| smokes | Ornith 262K: vi 63.8 / code 76.3 t/s / tool ok · Ornith 512K: vi 64.8 / code 74.0 / tool ok · Qwen3.8 262K: vi 35.6 / code 36.5 / tool ok · Qwen3.8 512K: vi 35.1 / code 36.0 / tool ok |
| registry | `~/.local/ai-gateway/runtime-registry.json` (backup `.bak-rewind-20261009-223200`): `--rewind-point-min-tokens 16384` appended to both Ornith rows, atomic replace; nothing else changed. Repo declaration: AI-Gateway-MLX `b3c5d0ec` |
| live | 22:35:50 the 512K slot came up as `ds4-server … -c 524288 … --rewind-point-min-tokens 16384` and logged `rewind point at the prompt end of tool-chat prompts >= 16384 tokens` |
| Qwen3.8 | unchanged: the disk prompt-end checkpoint failed the spec gate (REPORT.md), so its flag stays off |

What to look for in the next real Ornith session (`gateway.log`):
- `rewind point remembered pos=… text=…` on every tool turn of at least 16K tokens.
- `rewind point hit pos=… live=… prompt=…` whenever Claude Code re-sends a turn or compacts.
- No `live kv cache miss … reason=token-mismatch` falling back to the 32K cold checkpoint for those shapes.
