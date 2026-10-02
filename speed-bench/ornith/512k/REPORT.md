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

Run 2026-10-02 15:36-16:17 on `8ef62eb1`, with the gateway stack paused. Receipts are in
`receipts/window-b/`.

**Long context.** The nextgen-eval `longctx` suite ran on the `configs/ornith-512k.json` arm: the
registry's Ornith command at `-c 524288` with `--kv-cache-continued-interval-tokens 0`. The server
logged "Ornith YaRN factor 2 (from -c)", KV 11.01 GiB and 33.26 GiB planned.

| tier | real prompt tokens | hit | prefill | decode (answer tokens) |
|---|---|---|---|---|
| needle 120K | 125,012 | yes | 924 t/s (135 s) | 28.7 t/s (6) |
| needle 240K | 239,868 | yes | 564 t/s (425 s) | 28.9 t/s (6) |
| needle 480K | 480,575 | yes | 315 t/s (1,528 s) | 19.5 t/s (6) |
| docqa 0 / 1 / 2 (240K document) | 239,876-239,879 | 3 / 3 | ~1.1 s each: the document is reused, 260-263 new tokens | 21.5 / 27.2 / 21.5 t/s (2 / 5 / 2) |

- Peak wired memory was 36.8 GiB, with 0 swap-outs. The 960K tier was skipped because it is above the
  524288 limit.
- Past 400K the prefill chunk rate falls to about 190 t/s, because attention covers the whole context
  before each new chunk.
- The answers are 2-6 tokens, so the decode figures are short samples. The MTP acceptance rate cannot
  be measured here, and the server logs no acceptance line for these requests. The Qwen 512K report
  left 480K decode blank for the same reason. Steady decode at depth is the first measurement of the
  tok/s sub-project.
- Against the spec's estimates: the 480K cold prefill took 25.5 min (estimate about 30 min for 512K),
  and decode at 480K was 19.5 t/s on 6 tokens (estimate about 20 t/s).

**Server check.** `server_check.py` ran three ds4-server runs on one fresh disk-KV directory, with
the same 44,161-token prompt each time:

| phase | `-c` | cache directory | cached / prompt | entries after |
|---|---|---|---|---|
| 262k-store | 262144 | root (unkeyed) | 0 / 44,161 | 6 (continued, cold, shutdown) |
| 512k-cold | 524288 | `yarn-2` | 0 / 44,161 | 2 (cold, shutdown) |
| 512k-restore | 524288 | `yarn-2` | 43,008 / 44,161 | 2 |

- The 512K cold run ignores the six entries the 262K run wrote at the other rope.
- The restore reuses 97% of the prompt: 1.3 s instead of 28.7 s. PASS.

**Final-review fix and `make test`.**
- The final review found that `ds4_test --qwen35-yarn-engine` opened its engine while run-all's
  cached engine still held the instance lock, which made the whole `ds4_test` run exit with status 2.
  Fixed in `87b1f062`.
  - The pre-fix binary on `--qwen4-prefill-checkpoints --qwen35-yarn-engine` exited 2 with "another
    ds4 process is already running".
  - The fixed binary passed all five engine cases.
- `make test` ran with `DS4_TEST_MODEL` set to the Ornith GGUF, because the default `ds4flash.gguf`
  is absent.
  - Every step before `ds4_test` passed.
  - `ds4_test` gave 23 OK and 6 ERR: long-context, logprob-vectors,
    metal-ssd-streaming-cache-pressure, local-golden-vectors, metal-short-prefill and server.
  - Develop `5b36552a`'s `ds4_test` on the same model gave the same 6 ERR with the same 16
    assertions. Those entries assume the default DeepSeek model.
  - The branch adds two entries, `qwen35-context-policy` and `qwen35-yarn-engine`, and both pass.
  - The CPU-only steps after `ds4_test` were run by hand after the window's steps, and all pass.

**Exit check.** Every condition in the spec holds:
- 262K byte-identical: window A, 39/39 gate-1 dumps match develop at chunks 64/512/2048, and the
  Qwen fast gate passes.
- YaRN agrees with llama.cpp: window A, gate 1 at factor 2 passes against the llama.cpp YaRN
  references, and the discrimination check is 13/13.
- 480K needle hit with 480,575 real tokens, within 0.12% of 480,000.
- No startup memory failure and no crash: the 512K server ran the long-context suite and the server
  check, with peak wired 36.8 GiB and 0 swap-outs.

## GPU window C: short-prompt quality, measured

Run 2026-10-02 11:55-11:58 on `bf12877a`, with the gateway stack paused. The window uses the same M4/M5
harness:
- the eval is a scratch copy of AI-Gateway's `run_eval.py` with the think split: HumanEval/GSM8K
  think-on, code_bench/review think-off;
- four abliteration probes;
- two staging `ds4-server` runs on 18296, with the MTP draft vocabulary and a fresh KV directory each.

The first attempt at 11:52 stopped at the pause: the live Ornith server had exited but was an
unreaped zombie, and the pause script counted it alive. The script now treats state Z as exited.

| arm | HumanEval-mini | GSM8K-mini | code_bench (fix / gentest / refactor) | review F1 | INDEX | truncated / errored | probes |
|---|---|---|---|---|---|---|---|
| `-c 262144` (no YaRN; KV 5.50 GiB, 27.76 GiB planned) | 1.000 | 1.000 | 0.778 (0.667 / 0.667 / 1.000) | 0.571 | **83.7** | 0 / 0 | 4/4 COMPLY |
| `-c 524288` (YaRN 2 from -c; KV 11.01 GiB, 33.26 GiB planned) | 1.000 | 1.000 | 0.667 (1.000 / 0.000 / 1.000) | 0.714 | **84.5** | 0 / 0 | 4/4 COMPLY |

- No drop is detectable at this sample size: the 512K INDEX is +0.8.
- The per-axis swaps (gentest down, fix and review up) are within the variance the earlier
  milestones saw on these small suites; M5 recorded testgen AST-style variance of 3/8 vs 3/5.
- One run per arm cannot resolve a small YaRN cost, so the 262K entry stays the default, as the spec
  decided.
- The abliteration probes are unchanged: 4/4 COMPLY on both arms, matching M4-M6.

## Merge and deploy

**Merge.** `feature/ornith-512k` was merged into develop with `--no-ff` as `2146f8db`, on top of
`9695695c`, and pushed. The merge was re-gated in window M (17:11-17:28):
- `test-qwen35-kernels`, the engine test (with the run-all lock pair) and the loader pass;
- the 262K path is byte-identical to `9695695c` at chunks 64/512/2048 (39/39 dumps), and gate 1
  passes on both builds;
- gate 1 at factor 2 passes (compared 106, checked 150);
- `test_mtp_cli` at factor 2 passes (369 accepted / 55 rejected drafts).

The Qwen session gated Qwen on the merge in a slot of that window:
- PROD 3/3 and ISTA 3/3 greedy output byte-identical to its develop baselines;
- PROD at `-c 524288` identical between base and candidate;
- paired decode within noise.

Before the merge, the user-approved final-review minors went in (`baa7491d`):
- the refusal names the fix that applies;
- the rope setup is shared by Qwen3.8 and Ornith;
- the payload comment moved back above its test.

**Deploy** (window D, 2026-10-02 17:36-17:39).
- `prod/ornith-512k-20261002` (= `2146f8db`) is installed in the PROD checkout. The previous branch
  was `prod/qwen-512k-20261001@3a3bfa3`, recorded in `.deploy-history`.
- The live config was hand-seeded at 17:36:29, each file backed up with suffix
  `.bak-prod-ornith-512k-20261002`:
  - `runtime-registry.json` gets the gateway catalog's 512K row (`f8f2e2ce`), plus `active_runtime
    ds4` and `enabled: true`. The row runs `-c 524288` on :18089 with `--kv-disk-space-mb 16384`, in
    its own `yarn-2` directory under the shared Ornith cache.
  - `models.json` gets "Ornith-1.5-Abliterated-512K".
  - The `-512K` model profile is byte-identical to gateway `b9dc323c`, which commits it so a
    re-stage keeps it.
- All four ds4 rows were smoke-tested from the PROD build:

| row | vi | code |
|---|---|---|
| Ornith 262K (:18087) | 297 tok, 80.0 t/s, same as ref | 290 tok, 91.7 t/s, same as ref |
| Ornith 512K (:18089) | 300 tok, 81.6 t/s, same as ref | 300 tok, 95.9 t/s, same as ref |
| Qwen 262K (:18086) | 294 tok, 19.6 t/s (cold first request), same as ref | 300 tok, 34.4 t/s, same as ref |
| Qwen 512K (:18088) | 300 tok, 32.6 t/s, same as ref | 300 tok, 34.7 t/s, same as ref |

The references were recorded from the develop `2146f8db` build in window M. The t/s figures include
prefill.

**Gateway re-stage and live check.**
- The deploy owner re-staged gateway `a753e37e`, `f8f2e2ce` and `b9dc323c` on the user's go. It was
  staged at 20:47, and Tier L finished at 20:52.
  - `_llm_ports()` now includes 18089.
  - The `-512K` profile survived the installer.
  - Tier L was 16/17; the only red is the pre-existing `L.vision_capability`.
- The live check then passed, through the gateway on :8090:
  - `/v1/models` lists the `-512K` id.
  - A request to it answered in 2.5 s, including the switch. One `ds4-server` was running, on :18089,
    and `/status` named the `-512K` model.
  - Switching back to the 262K Ornith answered in 1.8 s, with one `ds4-server`, on :18087.

**Rollback:**
- `deploy-ai-gateway.sh install prod/qwen-512k-20261001`;
- restore the two `.bak-prod-ornith-512k-20261002` files;
- delete `model-profiles/<id>-512K.json`.
