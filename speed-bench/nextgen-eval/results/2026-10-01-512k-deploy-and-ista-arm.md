# 512K PROD row deployed; ISTA arm completed

The 512K Qwen row is live:
- It runs on port 18088 from `prod/qwen-512k-20261001` (develop `3a3bfa3`).
- It uses PROD's 262K command at `-c 524288`, plus `--kv-cache-continued-interval-tokens 0`.
- Smoke tests passed for all three ds4 rows, with output identical to the reference.

ISTA + refusal projection (FFN 0.5) has now run every suite. Against Ivan + the same projection:
- On the capability suites it loses 14 items and wins 2 (pooled McNemar p ≈ 0.004).
- On refusing harmful tool calls it wins 8 and loses 0 (p = 0.008).
- ISTA decodes slower in the harness: 34.3 t/s on ifeval, against 39.0 (Ivan + projection) and 40.5 (PROD).

The GPU window ran 2026-10-01 10:19-17:24 with the gateway stack paused.

## 512K validation (develop `3a3bfa3`, PROD's unc48L command at `-c 524288`)

| run | 480K needle | prefill 480K | peak wired | swap-outs |
|---|---|---|---|---|
| N1, registry flags (continued disk-KV checkpoints every 10K tokens) | hit (`7314-QX`) | 187 t/s | 50.77 GiB | 5,620 |
| probe, `--kv-cache-continued-interval-tokens 0` | hit (`7314-QX`) | 299 t/s | 50.75 GiB | 33,744 |

In N1, the 120K and 240K needles hit and docqa scored 3/3. The deploy gate demanded zero swap, so it
blocked the deploy.

**Why the machine swaps at 480K:**
- My first hypothesis was the disk-KV snapshot writes. N1's two swap moments did coincide with them:
  - 40 pages during a 2.6 GB checkpoint at 194,560 tokens;
  - 5,580 pages during the 6.8 GB store at shutdown.

  The snapshots are also written twice: staged in `/tmp` (`ds4_session_stage_payload`), then copied.
- The probe falsified that. With the checkpoints off, it swapped 527 MB at ~125K tokens while nothing
  was being written, and its 6.8 GB shutdown store swapped nothing.
- What actually happens: at ~480K, PROD's process wires ~50.8 GiB, against 49.5 GiB at 240K. When
  free RAM reaches the floor, macOS swaps out 0.1-0.5 GB of cold memory. Prefill rate did not drop
  after the swap.
- SP4's zero-swap 480K run was on Ivan's lighter model (~45.5 GiB).

The user accepted the swap past ~400K and chose the row without continued checkpoints. Their only
effect at 480K was the prefill cost: 187 t/s with them, 299 t/s without.

**Deploy:**
- `deploy-ai-gateway.sh cut qwen-512k` ran at ~12:00, then `install prod/qwen-512k-20261001` at
  17:22, from `prod/ornith-ds4-20260928@1551baf`.
- The registry gained the row `ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2-512K`:
  - port 18088;
  - context_limit 524288;
  - recommended_context 360448;
  - pid file `ds4-iq2-512k.pid`.
- `~/.local/bin/active-backend.py` `_llm_ports()` gained 18088. That was a hand edit, because the
  AI-Gateway repo copy does not carry the port yet.
- Backups: `*.bak-qwen-512k-20261001`.
- Smoke (vi + code), each against a reference recorded from the `3a3bfa3` build: Ornith, Qwen 262K and
  Qwen 512K all "ok, same as ref".

## ISTA + projection: the three new suites

`ista-proj-s050-sp5-20261001` is last night's `ista-proj-s050` arm plus `--rerun ifeval,longctx,reason`
on feature/nextgen-next's kernels (`3a3bfa3`). The other suites ran last night on SP3's kernels.

| suite | ISTA + proj | Ivan + proj | PROD baseline |
|---|---|---|---|
| ifeval (200) | 189 | 194 | 60/60 on the older 60-item set |
| reason (44) | 38 | 43 | 42 |
| longctx needles 120K / 240K | hit / hit | hit / hit | hit / hit |
| longctx docqa (240K) | 2/3 (docqa-0 answered 56, expected 64) | 3/3 | 3/3 |
| longctx peak wired / swap | 53.86 GiB / 0 | | 49.48 GiB / 0 |
| ifeval decode median | 34.27 t/s | 38.95 t/s | 40.48 t/s |

**Paired comparison against Ivan + proj** (same items; McNemar exact):

| suite | ISTA only | Ivan only | p |
|---|---|---|---|
| ifeval | 2 | 7 | 0.18 |
| code | 0 | 2 | 0.50 |
| reason | 0 | 5 | 0.06 |
| ifeval + code + reason | 2 | 14 | ≈ 0.004 |
| tools_neg | 8 | 0 | 0.008 |

For scale: Ivan + proj with YaRN against Ivan + proj without it (same weights) splits ifeval 3/6
(p = 0.51).

Each capability suite alone is within noise. Together they point the same way: ISTA is slightly behind
on capability and ahead on refusing harmful tool calls.

Whether that comes from the weights or from this port is open. The GPU-vs-CPU checks only compare ds4
with its own CPU reference. The next check is a KLD or perplexity comparison of ISTA, Ivan and PROD on
the same text, and against ISTA's own llama.cpp build if one runs here.

## Open items

- KLD/perplexity: ISTA vs Ivan vs PROD, to separate weight quality from port numerics.
- ISTA decode in the harness (34.3 vs PROD's 40.5): the draft-vocabulary gather for its Q5_K head, the
  dense GSQ-RCO tensor-op tiles, and `hc_*_up` in F16.
- The AI-Gateway repo should carry the 512K row, the `--think-budget 4096` drift fix and LLM port 18088.
  The AI-Gateway owner was asked to do this.
