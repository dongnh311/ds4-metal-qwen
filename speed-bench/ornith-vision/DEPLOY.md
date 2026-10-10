# Deploy receipt: Ornith native vision (2026-10-10)

- **Branch.** `prod/ornith-vision-20261010` at `f929011d` (develop merge of `feature/ornith-vision`), installed in the PROD
  checkout `~/.local/share/ai-gateway/ds4-metal`.
- **Previous.** `prod/rewind-point-20261009` (`0f2cff2`).
- **Window.** Quiet-window guard passed, `/tmp/ds4.lock` held 16:35:47-16:37:08, slot SIGTERMed, never SIGKILLed.
- **References.** Recorded from the develop build with the new registry command (`DS4_GATEWAY_REGISTRY` copy) for all four
  ds4 rows.
- **Smokes.** All four rows passed against those references: both Ornith rows with `--vision`, and both Qwen3.8 rows.
- **mmproj.** `~/.local/share/ai-gateway/ds4-models/mmproj-Ornith-1.5-35B-A3B-f16.gguf`, sha256 `815c9a67…3c7c`.
- **Registry.** Written atomically inside the same window. Both Ornith rows append `--vision <mmproj>` and declare
  `capabilities.vision: true`. Backup: `runtime-registry.json.bak-ornith-vision-20261010-163707`.
- **Live check, 17:06.** One image request through the gateway (`:8090`, marked as test traffic) to the Ornith route:
  - the slot started with `--vision`;
  - the reply named the headline and 1969 in 7.6 s, including the model switch;
  - ds4 logged `request images=1`, and no sidecar describe ran.

## Rollback

1. Restore the registry backup. The gateway reads it live, and the sidecar describes images again on the next image.
2. If the binary must go too, run `./deploy-ai-gateway.sh install prod/rewind-point-20261009` in a quiet window with the
   lock held (`docs/DEPLOY_AI_GATEWAY.md` rule 6).
