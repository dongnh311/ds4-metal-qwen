# Sub-project 4: 512K context with YaRN — design

Date: 2026-09-30. Status: design approved in conversation ("ok duyệt"). Parent design:
`docs/superpowers/specs/2026-09-28-qwen-nextgen-design.md` (sub-project 4 row). Branch:
`feature/nextgen-qwen`.

## Goal

Serve 512K-token prompts with ds4 on the 64 GB Mac without losing accuracy on short prompts, and
record what 1M costs, so that sub-project 5 can choose the context of the next-gen PROD entry.

## What ds4 already has

- **Static YaRN for qwen4** (`qwen4_rope_configure`, ds4.c, from the model-support commit ccea768).
  `DS4_QWEN4_YARN_FACTOR=f` (f > 1) blends the rotary pairs between HF's correction range (beta_fast
  32, beta_slow 1) over the native `qwen4exp.context_length` (262144), and scales cos/sin by
  `0.1 ln f + 1`. The CPU path and the Metal path read the same table. At f = 2 the log reads
  "pairs 14..22 blended, mscale 1.0693", which matches the HF formula with base 1e7 and 64 rotary
  dims. Without the variable, a server with `-c` above 262144 warns once and runs unscaled.
- **The engine knows the context:** `ds4_engine_options.context_size` is set from `-c` by
  ds4-server, ds4 and ds4-eval before the model loads.
- **Disk KV cache:**
  - The key does not include the rope. Sub-project 2 added `<kv-dir>/steer-<sha8>` for steering
    (`kv_cache_steering_dir`).
  - It stores a "continued" checkpoint whenever the live context crosses a multiple of
    `--kv-cache-continued-interval-tokens` (default 10000, rounded up to the 2048 alignment: every
    10240 tokens). Each checkpoint is a full snapshot of the prefix.

## Probe (2026-09-30, Ivan's IQ2, gateway paused, throwaway scripts in `ds4-metal-data/sp4`)

The harness `longctx` suite ran with PROD's registry flags, Ivan's GGUF and the settings below.

| probe | settings | real prompt tokens | needle | prefill | swap-outs | peak wired |
|---|---|---|---|---|---|---|
| A | `-c 524288`, factor 2 | 124,858 / 239,714 / 415,227 | hit / hit / hit; docqa 3/3 | 537 / 462 / 296 t/s | 118,432 | 45.41 GiB |
| B | as A, continued checkpoints off | 469,432 | hit | 485 t/s | 0 | 45.56 GiB |
| C | `-c 1048576`, factor 4, checkpoints off | 899,537 | hit | 316 t/s (2845 s) | 29,000 | 49.61 GiB |

Findings:
1. **YaRN works as shipped.** Every needle hit at factor 2 up to 469K tokens and at factor 4 at 900K.
   Ivan's IQ2 without YaRN missed the 240K needle (`results/2026-09-29-sp2-projection.md`); with
   factor 2 it hits.
2. **Continued checkpoints break long prefills.**
   - Past 262K each snapshot is 3.5-5.4 GB. Probe A wrote about 180 GB for one 415K prompt, and the
     32 GB disk budget evicted each snapshot as soon as the next was stored.
   - The writes push the demand-paged PLE sidecar and streamed experts out of the page cache.
     Turning them off removed the swap-outs and raised prefill from 296 t/s at 415K to 485 t/s at
     469K.
   - Below 262K the same mechanism may slow PROD, but that is not measured yet.
3. **The harness undersizes the deep tiers.**
   - It sizes a tier as tokens × 3.03 characters, measured on the head of `ds4.c`. Deeper text
     tokenizes at about 3.5-3.7 characters per token.
   - The "480K" tier was therefore 415K tokens, and 1.086M nominal tokens were 900K.
4. **Memory:** KV grows about 2.1 GiB per 230K tokens on this model (43.47 GiB at 240K, 45.56 GiB at
   469K, 49.61 GiB at 900K, K=32 streaming with a 6 GB expert cache). Sub-project 3 uses this to size
   its tier.
5. **The disk cache is not keyed by the YaRN factor.** A cache written at one factor can be restored
   at another with the same `-c`.

## Design

Every change is inert for `-c` at or below the native context without `DS4_QWEN4_YARN_FACTOR`: that
path, which is PROD's, stays byte-identical.

### Engine

- **Factor from the context.** `qwen4_rope_configure` also takes the engine's `context_size`.
  - When `DS4_QWEN4_YARN_FACTOR` is unset and `context_size` exceeds the native context, the factor
    is the smallest power of two at least `context_size / native` (2 for 524288, 4 for 1048576).
  - The variable still overrides, and `DS4_QWEN4_YARN_FACTOR=1` turns YaRN off.
  - The existing log line names the factor and whether it came from `-c` or the environment. The
    "exceeds the native context" warning now fires only when YaRN was forced off.
- **Query.** `float ds4_engine_rope_yarn_factor(const ds4_engine *)` returns the active factor, or 1
  when YaRN is off. `uint32_t ds4_engine_native_context(const ds4_engine *)` returns the model's
  native context, or 0 for families without one.

### Server

- **Disk KV cache keyed by the rope.** With YaRN active the cache opens in `<dir>/yarn-<f>` (`%g`
  format, for example `yarn-2`).
  - Steering and YaRN together give `<dir>/steer-<sha8>-yarn-<f>`.
  - Without YaRN the directory is unchanged, as is the steering-only directory.
  - The server logs the directory when it differs from `--kv-disk-dir`.
- **Continued checkpoints past the native context.**
  - Up to the native context, checkpoints keep today's interval.
  - Past it, a continued checkpoint is stored only at positions `step × 2^k`: 327,680 and 655,360
    at the default step. For a 480K prompt that means one extra snapshot instead of 21.
  - The server passes the native context to the kv store as a new option,
    `continued_dense_max_tokens`. The default is 0, meaning no limit, which is today's behavior.
  - Final and shutdown stores do not change.
  - A PROD server (`-c 262144`) never crosses the native context, so its checkpoints do not change.

### Harness (`speed-bench/nextgen-eval`)

- **Tiers sized in real tokens.**
  - Each tier carries its character count as well as its token target.
  - The 120K and 240K counts stay exactly as they are (tokens × 3.03), so earlier rows stay
    comparable.
  - 480K and 960K use counts calibrated from the probe's measured tokenization of the frozen
    haystack (about 1.72M and 3.52M characters). The plan records the calibration.
  - The ctx check uses the token target.
- **Every needle row keeps `prompt_tokens`.** The summary gains `longctx.short_tiers`, the tiers whose
  real prompt fell more than 5% below target, and `compare.py` prints it.
- **`--kv-cache-continued-interval-tokens` is classified** for ds4-eval (server-only), as sub-project
  2 classified the steering flags. The reasoning suite passes `-c` to ds4-eval, so it runs with the
  same rope as the server. The plan checks the existing classification and adds what is missing.
- **Arm config:** `configs/ivan-proj-yarn.json` is `configs/ivan-proj.json` (arm `ivan-proj-s050`)
  plus `-c 524288`, with the factor coming from `-c`. The arm is named `ivan-proj-s050-yarn`.

## Testing

Offline (no GPU):
- rope table unit tests, run through the existing C test entry points:
  - the factor derived from `-c`;
  - override and forced off;
  - the table unchanged at or below the native context;
- the kv-store unit test for the checkpoint positions (dense below the limit, `step × 2^k` above it,
  0 = unchanged);
- the cache directory naming test (yarn, steer + yarn, neither);
- harness tests for the tier sizes, `short_tiers` and the arm config;
- `make` builds every binary.

GPU (each needs the user's go-ahead and a paused gateway stack):
1. **Default path unchanged:** greedy `ds4` output on the PROD model with `-c 262144`, three prompts,
   128 tokens: identical text from this branch and from `feature/nextgen-qwen` before this work.
2. **Directories:** the YaRN arm's `server.log` names `steer-<sha8>-yarn-2`.
3. **Measured, not gated:** the 240K needle on PROD's model with continued checkpoints at today's
   interval and off. It records whether PROD's own prefill pays the same cost, as input for
   sub-project 8.

## Exit check

One full harness arm, `ivan-proj-s050-yarn`: every suite, same binaries, frozen data, about 7 GPU
hours. Its baseline is the existing `ivan-proj-s050` run. Results go to
`speed-bench/nextgen-eval/results/<run date>-sp4-yarn.md`.

Sub-project 4 passes when all of the following hold:
- `compare.py ivan-proj-s050/summary.json ivan-proj-s050-yarn/summary.json --gate engine
  --refusal-caps 5,1` exits 0;
- the 480K needle hits with a real prompt within 5% of 480,000 tokens;
- the arm has zero swap-outs.

If only the short-prompt suites regress, sub-project 4 still delivers 512K as a separate registry
entry (`-c 524288`) beside the 262K default, which is the parent design's mitigation. The results
file records that the default stays at 262K. If a long-context check fails, sub-project 4 fails.

1M is not part of this exit check. Probe C shows it runs on Ivan's IQ2 (hit at 900K, 49.6 GiB).
Sub-project 5 decides on 1M with the candidate's weights.

## Out of scope

- Dynamic (position-dependent) YaRN: the KV cache needs one rope for the whole context.
- Changing PROD's checkpoint policy below the native context: GPU test 3 only measures it.
- The gateway's routing between a 262K and a 512K entry, which is sub-project 7.

## Constraints

As in the parent design:
- Mac only.
- No PROD change, merge or push without the user's approval.
- Model artifacts go to HF, never git.
- GPU runs wait for the user's go-ahead, pause the gateway stack and restore it.
- Never SIGKILL a Metal process.
- Code, docs and commits are in English.
