# Sub-project 4: 512K context with YaRN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve 512K-token prompts on the 64 GB Mac with the YaRN factor derived from `-c`, a disk KV
cache keyed by it, no full-prefix checkpoint storm past the native context, and a harness that sizes
its long-context tiers in real tokens.

**Architecture:**
- ds4 already has static qwen4 YaRN behind `DS4_QWEN4_YARN_FACTOR`. Task 1 turns its math into two
  pure, public functions (the factor rule and the rope table) and derives the factor from the
  engine's `context_size`.
- Tasks 2-3 teach the kv store and ds4-server about the rope and the native context.
- Tasks 4-5 fix the harness.
- Tasks 6-7 are GPU work that needs the user's go-ahead.

**Tech Stack:** C (ds4.c, ds4_kvstore.c, ds4_server.c, tests via `./ds4_test`), Python 3 stdlib (harness
tests via `python3 -m unittest`).

**Spec:** `docs/superpowers/specs/2026-09-30-yarn-context-design.md`

## Global Constraints

- Every change is inert for `-c` at or below the native context without `DS4_QWEN4_YARN_FACTOR`:
  that path, which is PROD's, stays byte-identical.
- Native context: `qwen4exp.context_length` = 262144 for every Qwen3.8-Flash-Next GGUF we run.
- Mac only. No PROD change, merge or push without the user's approval. Model artifacts go to HF,
  never git.
- GPU runs wait for the user's go-ahead, pause the gateway stack and restore it. Never SIGKILL a
  Metal process.
- Code, docs and commits are in English. Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn
  ```
- Worktree `/Users/dongnh/orca/workspaces/ds4-metal/kv-grow`, branch `feature/nextgen-qwen`. Never
  bare `git stash`.

**Spec correction carried by this plan:**
- The spec says the reasoning suite "passes `-c` to ds4-eval, so it runs with the same rope as the
  server". That is not true: `ds4eval.eval_argv` caps ds4-eval at `-c 65536` (`CTX_CAP`), below the
  native context, so ds4-eval would never derive YaRN from `-c`.
- Task 5 hands ds4-eval the server's factor through `DS4_QWEN4_YARN_FACTOR` instead, with the same
  rule as ds4.c.
- The server still derives its factor from `-c`, as the spec requires.

## Review Focus

1. A junk, empty or trailing-garbage `DS4_QWEN4_YARN_FACTOR` at `-c 524288` must not silently
   disable YaRN: the factor falls back to the `-c` rule. Task 1 tests it.
2. `-c 262144` exactly (PROD) must give no YaRN, the unscaled table, an unchanged cache directory and
   unchanged checkpoints. Tasks 1, 2 and 3 test it.
3. Caches written at factor 2 and at factor 4 must never share a directory, with or without
   steering. Task 3 tests it.
4. Checkpoints at the dense boundary, where the live context sits exactly on the limit or one step
   past it, must follow the rule: the limit is dense, and past it only step × 2^k. Task 2 tests it.
5. An explicit `DS4_QWEN4_YARN_FACTOR` in an arm's env, including `=1`, must reach ds4-eval unchanged.
   Task 5 tests it.

---

### Task 1: YaRN factor from the context (engine)

**Files:**
- Modify: `ds4.h` (declarations after `ds4_engine_prefill_chunk`, line ~249)
- Modify: `ds4.c:7207-7247` (`qwen4_rope_configure` and its globals), `ds4.c:~60007` (the
  native-context warning), `ds4.c:~73555` (`ds4_engine_open`, before `config_validate_model`), and
  the end of the `ds4_engine_prefill_chunk` block (line ~74845) for the two queries
- Modify: `docs/QWEN38_FLASH_NEXT.md:69-70`
- Test: `tests/ds4_test.c` (new `test_qwen_yarn_policy`, registered in `test_entries[]`)

**Interfaces:**
- Produces:
  - `double ds4_qwen4_yarn_factor(uint32_t native_ctx, uint32_t context_size, const char *env_value)`:
    the active factor, where 1.0 means YaRN is off.
  - `void ds4_qwen4_rope_table(uint32_t n_rot, double base, uint32_t native_ctx, double factor, float freq[32], float *mscale, double *low, double *high)`:
    the table. The `mscale`, `low` and `high` pointers may be NULL.
  - `float ds4_engine_rope_yarn_factor(const ds4_engine *e)`: returns 1.0f when YaRN is off or the
    model is not qwen4.
  - `uint32_t ds4_engine_native_context(const ds4_engine *e)`: returns 0 for non-qwen4 models.

- [ ] **Step 1: Snapshot the baseline binaries for Task 6**

The current binaries are HEAD 020ad01's C code: nothing in C has changed since sub-project 2's build.
Run:
```bash
cd /Users/dongnh/orca/workspaces/ds4-metal/kv-grow
make -j8 ds4 ds4-server ds4-eval 2>&1 | tail -1
mkdir -p ~/orca/workspaces/ds4-metal-data/sp4/baseline
cp ds4 ds4-server ds4-eval ~/orca/workspaces/ds4-metal-data/sp4/baseline/
cp -R metal ~/orca/workspaces/ds4-metal-data/sp4/baseline/
git rev-parse --short HEAD > ~/orca/workspaces/ds4-metal-data/sp4/baseline/HEAD
```
Expected: `make` reports nothing to rebuild, or only relinks. The baseline directory holds `ds4`,
`ds4-server`, `ds4-eval`, `metal/` and `HEAD` = `020ad01` or a later docs-only commit.

- [ ] **Step 2: Write the failing test**

In `tests/ds4_test.c`, directly after the closing brace of `test_qwen_kv_grow_policy`:
```c
static void test_qwen_yarn_policy(void) {
    /* The factor: the environment wins when it parses, otherwise -c decides. */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 262144, NULL) == 1.0);   /* PROD: off */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 8192, NULL) == 1.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 262145, NULL) == 2.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, NULL) == 2.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524289, NULL) == 4.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 1048576, NULL) == 4.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(0, 524288, NULL) == 1.0);        /* no native context */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "1") == 1.0);    /* forced off */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "0.5") == 1.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 8192, "4") == 4.0);      /* forced on */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "") == 2.0);     /* empty = unset */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "junk") == 2.0); /* unparsable = unset */
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "2x") == 2.0);
    TEST_ASSERT(ds4_qwen4_yarn_factor(262144, 524288, "0") == 2.0);    /* not positive = unset */

    /* The table: factor 1 is plain RoPE, factor 2 blends pairs 14..22 (HF, base 1e7, 64 dims). */
    float plain[32], yarn[32], ms = 0.0f;
    double low = -1.0, high = -1.0;
    ds4_qwen4_rope_table(64, 1e7, 262144, 1.0, plain, &ms, &low, &high);
    for (int i = 0; i < 32; i++) {
        TEST_ASSERT(plain[i] == (float)pow(1e7, -2.0 * (double)i / 64.0));
    }
    TEST_ASSERT(ms == 1.0f);
    TEST_ASSERT(low == 0.0 && high == 0.0);
    ds4_qwen4_rope_table(64, 1e7, 262144, 2.0, yarn, &ms, &low, &high);
    TEST_ASSERT(low == 14.0 && high == 22.0);
    TEST_ASSERT(ms == (float)(0.1 * log(2.0) + 1.0));
    for (int i = 0; i <= 14; i++) TEST_ASSERT(yarn[i] == plain[i]);             /* extrapolated */
    for (int i = 22; i < 32; i++) {
        TEST_ASSERT(yarn[i] == (float)(pow(1e7, -2.0 * (double)i / 64.0) / 2.0)); /* interpolated */
    }
    TEST_ASSERT(yarn[18] < plain[18] && yarn[18] > plain[18] / 2.0f);          /* blended */
    ds4_qwen4_rope_table(64, 1e7, 0, 2.0, yarn, NULL, NULL, NULL);             /* no native: plain */
    for (int i = 0; i < 32; i++) TEST_ASSERT(yarn[i] == plain[i]);
}
```
Add a row to `test_entries[]` directly after the `--qwen-kv-grow-policy` row:
```c
    {"--qwen-yarn-policy", "qwen-yarn-policy", "Qwen3.8 YaRN factor from -c and the rope table (no model)", test_qwen_yarn_policy},
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make ds4_test 2>&1 | grep -m3 -E "error|undefined"`
Expected: an implicit-declaration error for `ds4_qwen4_yarn_factor` (or undefined symbols at link).

- [ ] **Step 4: Implement**

`ds4.h`, after `uint32_t ds4_engine_prefill_chunk(ds4_engine *e);`:
```c
/* Qwen3.8 static YaRN. The factor is DS4_QWEN4_YARN_FACTOR when that parses
 * as a positive number (<= 1 turns YaRN off); otherwise the smallest power of
 * two covering context_size / native_ctx when the context exceeds the native
 * one, else 1 (off). */
double ds4_qwen4_yarn_factor(uint32_t native_ctx, uint32_t context_size, const char *env_value);
/* Rotary inverse frequencies of the n_rot/2 pairs (entries past n_rot/2 are 0),
 * with HF _compute_yarn_parameters blending (beta_fast 32, beta_slow 1) when
 * factor > 1 and native_ctx > 0. mscale, low and high may be NULL. */
void ds4_qwen4_rope_table(uint32_t n_rot, double base, uint32_t native_ctx, double factor,
                          float freq[32], float *mscale, double *low, double *high);
/* The loaded model's active YaRN factor (1 when off or not Qwen3.8) and its
 * native context (0 when the family has none). */
float ds4_engine_rope_yarn_factor(const ds4_engine *e);
uint32_t ds4_engine_native_context(const ds4_engine *e);
```

`ds4.c`: replace everything from `static float g_qwen4_rope_freq[32];` through the closing brace of
`qwen4_rope_configure` with:
```c
static float g_qwen4_rope_freq[32];
static float g_qwen4_rope_mscale = 1.0f;
static float g_qwen4_rope_factor = 1.0f;
static uint32_t g_qwen4_native_ctx = 0;
/* The engine's context_size, set by ds4_engine_open before the model validates. */
static uint32_t g_qwen4_rope_ctx_hint = 0;
static bool g_qwen4_rope_yarn = false;

double ds4_qwen4_yarn_factor(uint32_t native_ctx, uint32_t context_size, const char *env_value) {
    if (env_value && env_value[0]) {
        char *end = NULL;
        const double f = strtod(env_value, &end);
        if (end != env_value && *end == '\0' && f > 0.0) return f > 1.0 ? f : 1.0;
    }
    if (native_ctx == 0 || context_size <= native_ctx) return 1.0;
    double f = 2.0;
    while ((double)native_ctx * f < (double)context_size) f *= 2.0;
    return f;
}

void ds4_qwen4_rope_table(uint32_t n_rot, double base, uint32_t native_ctx, double factor,
                          float freq[32], float *mscale, double *low_out, double *high_out) {
    const uint32_t half = n_rot / 2u;
    const bool yarn = factor > 1.0 && native_ctx > 0;
    double low = 0.0, high = 0.0;
    if (yarn) {
        low = fmax(0.0, floor(n_rot * log(native_ctx / (32.0 * 2.0 * M_PI)) / (2.0 * log(base))));
        high = fmin((double)n_rot - 1.0, ceil(n_rot * log(native_ctx / (2.0 * M_PI)) / (2.0 * log(base))));
        if (high == low) high += 0.001;
    }
    for (uint32_t i = 0; i < 32u; i++) {
        double f = i < half ? pow(base, -2.0 * (double)i / (double)n_rot) : 0.0;
        if (yarn && i < half) {
            const double extrap = 1.0 - fmin(1.0, fmax(0.0, ((double)i - low) / (high - low)));
            f = (f / factor) * (1.0 - extrap) + f * extrap;
        }
        freq[i] = (float)f;
    }
    if (mscale) *mscale = yarn ? (float)(0.1 * log(factor) + 1.0) : 1.0f;
    if (low_out) *low_out = low;
    if (high_out) *high_out = high;
}

/* Static YaRN over the native context, the model card's recipe for prompts
 * beyond 262k tokens. It is on when -c exceeds the native context (factor
 * from ds4_qwen4_yarn_factor) or when DS4_QWEN4_YARN_FACTOR asks for it; it
 * costs some quality on short text, so a context within the native one stays
 * unscaled. */
static void qwen4_rope_configure(uint32_t native_ctx) {
    const char *env = getenv("DS4_QWEN4_YARN_FACTOR");
    const double factor = ds4_qwen4_yarn_factor(native_ctx, g_qwen4_rope_ctx_hint, env);
    const bool yarn = factor > 1.0 && native_ctx > 0;
    double low = 0.0, high = 0.0;
    ds4_qwen4_rope_table(DS4_N_ROT, DS4_ROPE_FREQ_BASE, native_ctx, factor,
                         g_qwen4_rope_freq, &g_qwen4_rope_mscale, &low, &high);
    g_qwen4_native_ctx = native_ctx;
    g_qwen4_rope_yarn = yarn;
    g_qwen4_rope_factor = yarn ? (float)factor : 1.0f;
    if (yarn) {
        const bool from_env = ds4_qwen4_yarn_factor(native_ctx, 0, env) > 1.0;
        fprintf(stderr, "ds4: Qwen3.8 YaRN factor %g (%s) over %u native tokens (pairs %g..%g blended, mscale %.4f)\n",
                factor, from_env ? "DS4_QWEN4_YARN_FACTOR" : "from -c", native_ctx, low, high,
                g_qwen4_rope_mscale);
    }
#ifdef DS4_HAS_QWEN4_GPU
    ds4_gpu_qwen4_set_rope(g_qwen4_rope_freq, DS4_N_ROT / 2u, g_qwen4_rope_mscale);
#endif
}
```

In `ds4_engine_open`, replace the line `    config_validate_model(&e->model);` (line ~73555, the one directly
after the V4.1 vision check) with:
```c
    g_qwen4_rope_ctx_hint = opt->context_size > 0 ? (uint32_t)opt->context_size : 0u;
    config_validate_model(&e->model);
```

The native-context warning (line ~60007) now fires only when YaRN was forced off. Replace its
`fprintf` with:
```c
        fprintf(stderr, "ds4: context %u exceeds the native %u tokens and YaRN is off "
                "(DS4_QWEN4_YARN_FACTOR <= 1); prompts past %u tokens will degrade\n",
                ctx_cap, g_qwen4_native_ctx, g_qwen4_native_ctx);
```

Directly after the closing brace of `ds4_engine_prefill_chunk`:
```c
float ds4_engine_rope_yarn_factor(const ds4_engine *e) {
    (void)e;
    return ds4_model_is_qwen4() && g_qwen4_rope_yarn ? g_qwen4_rope_factor : 1.0f;
}

uint32_t ds4_engine_native_context(const ds4_engine *e) {
    (void)e;
    return ds4_model_is_qwen4() ? g_qwen4_native_ctx : 0u;
}
```

`docs/QWEN38_FLASH_NEXT.md`, replace the two lines "262144 tokens; `DS4_QWEN4_YARN_FACTOR=2` or `=4`
enables static YaRN for / longer contexts, with a possible quality cost on shorter prompts." with:
```
262144 tokens. A larger `-c` turns on static YaRN with the smallest power-of-two
factor that covers it (2 for 524288, 4 for 1048576); `DS4_QWEN4_YARN_FACTOR=f`
overrides the factor and `=1` turns YaRN off. YaRN may cost some quality on
shorter prompts, so `-c 262144` stays unscaled.
```

- [ ] **Step 5: Run it to verify it passes**

Run: `make ds4_test 2>&1 | grep -E "error|warning: .*yarn" ; ./ds4_test --qwen-yarn-policy 2>&1 | tail -1`
Expected: no errors; `ds4 tests: ok`.

- [ ] **Step 6: Build everything and run the existing groups**

Run: `make -j8 ds4 ds4-server ds4-eval ds4_test 2>&1 | grep -E "error" ; ./ds4_test --qwen-kv-grow-policy 2>&1 | tail -1 ; ./ds4_test --server 2>&1 | tail -1`
Expected: no errors; both print `ds4 tests: ok`.

- [ ] **Step 7: Commit**

```bash
git add ds4.h ds4.c tests/ds4_test.c docs/QWEN38_FLASH_NEXT.md
git commit -m "qwen4: derive the YaRN factor from -c

The factor rule and the rope table become public pure functions
(ds4_qwen4_yarn_factor, ds4_qwen4_rope_table) with a no-model test.
A context above the native 262144 turns on static YaRN with the
smallest power-of-two factor; DS4_QWEN4_YARN_FACTOR still overrides,
=1 turns it off, and an unparsable value no longer disables it.
ds4_engine_rope_yarn_factor and ds4_engine_native_context expose the
result. -c 262144 stays unscaled.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

### Task 2: Continued checkpoints sparse past the native context (kv store)

**Files:**
- Modify: `ds4_kvstore.h:64-70` (`ds4_kvstore_options`), `ds4_kvstore.c:164-172` (defaults),
  `ds4_kvstore.c:760-767` (`ds4_kvstore_continued_store_target`)
- Test: `ds4_server.c` (new `test_kv_cache_continued_sparse_past_dense_max` after
  `test_kv_cache_continued_uses_aligned_frontiers`, called from `ds4_server_unit_tests_run` next to
  it, line ~24717)

**Interfaces:**
- Produces: `int ds4_kvstore_options.continued_dense_max_tokens`, where 0 means no limit (today's
  behavior).

- [ ] **Step 1: Write the failing test**

In `ds4_server.c`, directly after the closing brace of `test_kv_cache_continued_uses_aligned_frontiers`:
```c
static void test_kv_cache_continued_sparse_past_dense_max(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();          /* step 10240 after the 2048 alignment */
    TEST_ASSERT(kc.opt.continued_dense_max_tokens == 0);
    kc.continued_last_store_tokens = 0;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 266240) == 266240);   /* no limit: dense */

    kc.opt.continued_dense_max_tokens = 262144;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 256000) == 256000);   /* 25 steps, inside */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 266240) == 0);        /* 26 steps, past */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 317440) == 0);        /* 31 steps */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 327680) == 327680);   /* 32 = 2^5 steps */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 337920) == 0);        /* 33 steps */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 655360) == 655360);   /* 64 steps */
    kc.continued_last_store_tokens = 655360;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 655360) == 0);        /* already stored */

    kc.continued_last_store_tokens = 0;
    kc.opt.continued_dense_max_tokens = 266240;                            /* the limit itself is dense */
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 266240) == 266240);
}
```
In `ds4_server_unit_tests_run`, directly after `test_kv_cache_continued_uses_aligned_frontiers();`:
```c
    test_kv_cache_continued_sparse_past_dense_max();
```

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test 2>&1 | grep -m2 -E "error"`
Expected: `no member named 'continued_dense_max_tokens'`.

- [ ] **Step 3: Implement**

`ds4_kvstore.h`, in `ds4_kvstore_options` after `int boundary_align_tokens;`:
```c
    /* Past this many live tokens a continued checkpoint is stored only at
     * interval x 2^k: each one is a full-prefix snapshot, gigabytes deep into a
     * long context. 0 = no limit. ds4-server sets it to the native context. */
    int continued_dense_max_tokens;
```
`ds4_kvstore.c`, in `ds4_kvstore_default_options` after `.boundary_align_tokens = ...,`:
```c
        .continued_dense_max_tokens = 0,
```
`ds4_kvstore.c`, in `ds4_kvstore_continued_store_target`, replace
```c
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
```
with
```c
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    if (kc->opt.continued_dense_max_tokens > 0 && live_tokens > kc->opt.continued_dense_max_tokens) {
        const int q = live_tokens / step;
        if (q & (q - 1)) return 0;   /* past the dense range: interval x 2^k only */
    }
    return live_tokens;
```

- [ ] **Step 4: Run it to verify it passes**

Run: `make ds4_test 2>&1 | grep -E "error" ; ./ds4_test --server 2>&1 | tail -1`
Expected: no errors; `ds4 tests: ok`.

- [ ] **Step 5: Commit**

```bash
git add ds4_kvstore.h ds4_kvstore.c ds4_server.c
git commit -m "kvstore: continued checkpoints only at interval x 2^k past a limit

A continued checkpoint is a full-prefix snapshot; past 262K tokens each
is 3.5-5.4 GB, and one 415K prompt wrote about 180 GB, swapped and cut
prefill from 485 to 296 t/s. continued_dense_max_tokens (0 = no limit,
the default) keeps the interval up to the limit and stores only at
interval x 2^k past it.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

### Task 3: Server keys the disk cache by the rope and sets the dense limit

**Files:**
- Modify: `ds4_server.c:11838-11872` (`kv_cache_steering_dir` becomes `kv_cache_variant_dir`),
  `ds4_server.c:~16818-16833` (call site, log line, dense limit),
  `ds4_server.c:24534-24555` (`test_kv_cache_steering_dir` becomes `test_kv_cache_variant_dir`)
- Test: the renamed test in `ds4_server.c`

**Interfaces:**
- Consumes: `float ds4_engine_rope_yarn_factor(const ds4_engine *e)` and
  `uint32_t ds4_engine_native_context(const ds4_engine *e)` from Task 1;
  `ds4_kvstore_options.continued_dense_max_tokens` from Task 2.
- Produces: `static bool kv_cache_variant_dir(const char *dir, const char *steer_file, float attn_scale, float ffn_scale, double yarn_factor, char *out, size_t outlen)`.

- [ ] **Step 1: Write the failing test**

Replace the whole `test_kv_cache_steering_dir` function with:
```c
static void test_kv_cache_variant_dir(void) {
    char out[4096], other[4096];
    TEST_ASSERT(kv_cache_variant_dir("/kv", NULL, 0.0f, 1.0f, 1.0, out, sizeof(out)));
    TEST_ASSERT(!strcmp(out, "/kv"));
    TEST_ASSERT(kv_cache_variant_dir("/kv", NULL, 0.0f, 1.0f, 2.0, out, sizeof(out)));
    TEST_ASSERT(!strcmp(out, "/kv/yarn-2"));                   /* the rope is part of the key */
    TEST_ASSERT(kv_cache_variant_dir("/kv", NULL, 0.0f, 1.0f, 4.0, other, sizeof(other)));
    TEST_ASSERT(!strcmp(other, "/kv/yarn-4"));
    char path[] = "/tmp/ds4-steer-key-XXXXXX";
    const int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) return;
    TEST_ASSERT(write(fd, "abcdefgh", 8) == 8);
    close(fd);
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.0f, 0.0f, 1.0, out, sizeof(out)));
    TEST_ASSERT(!strcmp(out, "/kv"));                          /* zero scales steer nothing */
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.0f, 1.0f, 1.0, out, sizeof(out)));
    TEST_ASSERT(!strcmp(out, "/kv/steer-f113dc51"));           /* sha1("abcdefgh" "0,1") */
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.0f, 1.5f, 1.0, other, sizeof(other)));
    TEST_ASSERT(!strcmp(other, "/kv/steer-9cdec1ad"));         /* the scale is part of the key */
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.5f, 0.0f, 1.0, other, sizeof(other)));
    TEST_ASSERT(!strcmp(other, "/kv/steer-4bfb0ebf"));         /* attention-only steering is steering */
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.0f, 1.0f, 2.0, out, sizeof(out)));
    TEST_ASSERT(!strcmp(out, "/kv/steer-f113dc51-yarn-2"));    /* steering and rope together */
    TEST_ASSERT(kv_cache_variant_dir("/kv", path, 0.0f, 1.0f, 4.0, other, sizeof(other)));
    TEST_ASSERT(strcmp(out, other) != 0);
    unlink(path);
    TEST_ASSERT(!kv_cache_variant_dir("/kv", path, 0.0f, 1.0f, 1.0, out, sizeof(out)));
}
```
In `ds4_server_unit_tests_run`, replace `test_kv_cache_steering_dir();` with
`test_kv_cache_variant_dir();`.

- [ ] **Step 2: Run it to verify it fails**

Run: `make ds4_test 2>&1 | grep -m2 -E "error"`
Expected: an implicit-declaration error for `kv_cache_variant_dir`.

- [ ] **Step 3: Implement**

Replace the comment and function `kv_cache_steering_dir` (ds4_server.c:11838-11872) with:
```c
/* Cached KV encodes the residual stream and the rope it was computed with: it
 * must never be restored under other steering (direction or scale) or another
 * YaRN factor. The disk cache therefore lives in
 *   <dir>                      plain,
 *   <dir>/steer-<sha8>         steered (sha1 of the direction bytes + "attn,ffn"),
 *   <dir>/yarn-<f>             YaRN factor f > 1,
 *   <dir>/steer-<sha8>-yarn-<f> both.
 * Returns false when the direction file cannot be read or the path does not
 * fit. */
static bool kv_cache_variant_dir(const char *dir, const char *steer_file,
                                 float attn_scale, float ffn_scale, double yarn_factor,
                                 char *out, size_t outlen) {
    char steer[16] = "";
    if (steer_file && steer_file[0] && (attn_scale != 0.0f || ffn_scale != 0.0f)) {
        FILE *fp = fopen(steer_file, "rb");
        if (!fp) return false;
        long size = -1;
        if (fseek(fp, 0, SEEK_END) == 0) size = ftell(fp);
        if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
            fclose(fp);
            return false;
        }
        char *buf = xmalloc((size_t)size + 64);
        const bool read_ok = fread(buf, 1, (size_t)size, fp) == (size_t)size;
        fclose(fp);
        if (!read_ok) {
            free(buf);
            return false;
        }
        size_t len = (size_t)size;
        len += (size_t)snprintf(buf + len, 64, "%g,%g", (double)attn_scale, (double)ffn_scale);
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(buf, len, sha);
        free(buf);
        snprintf(steer, sizeof(steer), "steer-%.8s", sha);
    }
    char rope[32] = "";
    if (yarn_factor > 1.0) snprintf(rope, sizeof(rope), "yarn-%g", yarn_factor);
    int n;
    if (steer[0] && rope[0]) n = snprintf(out, outlen, "%s/%s-%s", dir, steer, rope);
    else if (steer[0]) n = snprintf(out, outlen, "%s/%s", dir, steer);
    else if (rope[0]) n = snprintf(out, outlen, "%s/%s", dir, rope);
    else n = snprintf(out, outlen, "%s", dir);
    return n >= 0 && n < (int)outlen;
}
```
At the call site (ds4_server.c:~16818), replace
```c
        if (!kv_cache_steering_dir(cfg.kv_disk_dir, cfg.engine.directional_steering_file,
                                   cfg.engine.directional_steering_attn,
                                   cfg.engine.directional_steering_ffn,
                                   kv_dir, sizeof(kv_dir))) {
```
with
```c
        if (!kv_cache_variant_dir(cfg.kv_disk_dir, cfg.engine.directional_steering_file,
                                  cfg.engine.directional_steering_attn,
                                  cfg.engine.directional_steering_ffn,
                                  (double)ds4_engine_rope_yarn_factor(engine),
                                  kv_dir, sizeof(kv_dir))) {
```
and replace
```c
        if (strcmp(kv_dir, cfg.kv_disk_dir) != 0) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: steered kv cache directory %s", kv_dir);
        }
        kv_cache_open(&s.kv, kv_dir, cfg.kv_disk_space_mb,
```
with
```c
        if (strcmp(kv_dir, cfg.kv_disk_dir) != 0) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: keyed kv cache directory %s", kv_dir);
        }
        cfg.kv_cache.continued_dense_max_tokens = (int)ds4_engine_native_context(engine);
        kv_cache_open(&s.kv, kv_dir, cfg.kv_disk_space_mb,
```

- [ ] **Step 4: Run it to verify it passes**

Run: `make -j8 ds4-server ds4_test 2>&1 | grep -E "error" ; ./ds4_test --server 2>&1 | tail -1 ; grep -rn "steered kv cache" speed-bench docs/superpowers/plans/2026-09-30-yarn-context.md | grep -v "keyed kv cache" | head -3`
Expected:
- no build errors;
- `ds4 tests: ok`;
- the grep finds nothing in `speed-bench`. No harness code matches on the old log text, and this
  plan's own mentions are prose.

- [ ] **Step 5: Commit**

```bash
git add ds4_server.c
git commit -m "ds4-server: key the disk KV cache by the YaRN factor

kv_cache_steering_dir becomes kv_cache_variant_dir: a cache computed at
one YaRN factor must never be restored at another, so the directory
gains yarn-<f> (steer-<sha8>-yarn-<f> with steering). The server also
passes the model's native context to the kv store as the dense limit
for continued checkpoints. -c 262144 without steering keeps <dir>.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

### Task 4: Long-context tiers sized in real tokens (harness)

**Files:**
- Create: `speed-bench/nextgen-eval/calibrate_tiers.py`
- Modify: `speed-bench/nextgen-eval/eval_suites.py:33-35` (TIERS), `eval_suites.py:205-214`
  (`haystack`), `eval_suites.py:218-240` (`run_longctx`)
- Modify: `speed-bench/nextgen-eval/run.py:55-79` (`summarize`: `short_tiers`)
- Modify: `speed-bench/nextgen-eval/compare.py:~134-150` (print `short_tiers`)
- Modify: `speed-bench/nextgen-eval/README.md` (tier sizing)
- Test: `test_eval_suites.py`, `test_run.py`, `test_compare.py`, new `test_calibrate_tiers.py`

**Interfaces:**
- Produces:
  - `eval_suites.TIERS: list[tuple[str, int, int]]`, holding (label, target tokens, haystack
    characters);
  - `eval_suites.haystack(path, chars, depth=0.5)`;
  - needle rows gain `target_tokens`;
  - `summary["longctx"]["short_tiers"]: list[str]`;
  - `calibrate_tiers.search(count_tokens, target, lo, hi, tol) -> int`.

- [ ] **Step 1: Write the failing tests**

`speed-bench/nextgen-eval/test_calibrate_tiers.py`:
```python
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_tiers  # noqa: E402


class Search(unittest.TestCase):
    def test_finds_chars_for_a_target_token_count(self):
        count = lambda chars: int(chars / 3.5)  # noqa: E731  deep ds4.c tokenizes at ~3.5 chars/token
        chars = calibrate_tiers.search(count, 480000, lo=1_000_000, hi=2_000_000, tol=1000)
        self.assertLessEqual(abs(count(chars) - 480000), 1000)

    def test_refuses_a_target_outside_the_bracket(self):
        with self.assertRaises(ValueError):
            calibrate_tiers.search(lambda c: c // 4, 960000, lo=1_000_000, hi=2_000_000, tol=1000)

    def test_counts_the_first_line_of_a_token_dump(self):
        self.assertEqual(calibrate_tiers.count_dump("[1, 2, 3]\n     1  a\n"), 3)


if __name__ == "__main__":
    unittest.main()
```
In `test_eval_suites.py`, add to class `Suites`:
```python
    def test_tiers_keep_the_old_sizes_and_calibrate_the_deep_ones(self):
        tiers = {label: (tokens, chars) for label, tokens, chars in eval_suites.TIERS}
        self.assertEqual(tiers["120k"], (120000, 363600))   # tokens x 3.03, unchanged
        self.assertEqual(tiers["240k"], (240000, 727200))
        self.assertEqual(tiers["480k"][0], 480000)
        self.assertEqual(tiers["960k"][0], 960000)
        self.assertTrue(1_650_000 <= tiers["480k"][1] <= 1_800_000)  # ~3.6 chars/token deep in ds4.c
        self.assertTrue(3_400_000 <= tiers["960k"][1] <= 3_700_000)

    def test_needle_rows_carry_their_target(self):
        src = self.data / eval_suites.HAYSTACK
        src.write_text("x = 1;\n" * 700000)
        rows = list(eval_suites.run_longctx(FakeCtx(self.data, lambda p: "7314-QX", ctx_limit=262144)))
        needles = [r for r in rows if r["id"].startswith("needle-") and r.get("passed") is not None]
        self.assertEqual([(r["id"], r["target_tokens"]) for r in needles],
                         [("needle-120k", 120000), ("needle-240k", 240000)])
```
`test_haystack_puts_needle_mid_document` calls `eval_suites.haystack(path, 10000)`. The 10000 now
means characters, and its mid-document assertion still holds, so it needs no change.

In `test_run.py`, add to the class that tests `summarize` (the one holding `test_rerun_provenance`
is `Rerun`; `summarize` tests live in `Summarize`):
```python
    def test_short_tiers_flag_needles_more_than_5_percent_under_target(self):
        rows = [{"suite": "longctx", "id": "needle-240k", "passed": True, "prompt_tokens": 239714,
                 "target_tokens": 240000},
                {"suite": "longctx", "id": "needle-480k", "passed": True, "prompt_tokens": 415227,
                 "target_tokens": 480000},
                {"suite": "longctx", "id": "needle-960k", "passed": None, "skipped": "ctx limit 524288"}]
        self.assertEqual(run.summarize(rows)["longctx"]["short_tiers"], ["480k"])
```
In `test_compare.py`, next to `test_markdown_has_verdict_and_rows` (same class):
```python
    def test_markdown_lists_short_tiers(self):
        c = candidate()
        c["longctx"]["short_tiers"] = ["480k"]
        md = compare.render_markdown(BASE, c, compare.gate(BASE, c))
        self.assertIn("| short tiers | none | 480k |", md)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/nextgen-eval -p 'test_*.py' 2>&1 | tail -4`
Expected: FAILED. There are errors for the missing `calibrate_tiers` module, TIERS unpacking and
`target_tokens`, and failures for `short_tiers` and the compare output.

- [ ] **Step 3: Implement calibrate_tiers.py**

```python
"""Calibrate the long-context tiers in real tokens (stdlib only; CPU, no GPU).

python3 speed-bench/nextgen-eval/calibrate_tiers.py --model GGUF [--targets 480000,960000]

The needle prompt is the frozen haystack's first N characters plus the question, and the harness
used to size N as tokens x 3.03, measured on the head of ds4.c. Deeper text tokenizes at about
3.5-3.7 characters per token, so the deep tiers came out short (the "480K" prompt was 415K tokens).
This tool finds N for each target with `ds4 --dump-tokens` (the chat template, like the server) and
prints the TIERS entries to paste into eval_suites.py.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import eval_suites  # noqa: E402
import fetch_data  # noqa: E402


def count_dump(text):
    """ds4 --dump-tokens prints the token ids as a JSON array on its first line."""
    return len(json.loads(text.splitlines()[0]))


def search(count_tokens, target, lo, hi, tol):
    """Smallest-effort bisection on the character count: returns chars whose count is within tol."""
    if not (count_tokens(lo) <= target <= count_tokens(hi)):
        raise ValueError("target %d outside [%d, %d] tokens" % (target, count_tokens(lo), count_tokens(hi)))
    while True:
        mid = (lo + hi) // 2
        n = count_tokens(mid)
        if abs(n - target) <= tol or hi - lo <= 1:
            return mid
        if n < target:
            lo = mid
        else:
            hi = mid


def ds4_counter(model, source):
    def count(chars):
        doc = "Here is a C source file.\n\n" + eval_suites.haystack(source, chars) + "\n\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(doc + eval_suites.NEEDLE_Q)
        r = subprocess.run([str(ROOT / "ds4"), "-m", str(model), "--nothink", "--dump-tokens",
                            "--prompt-file", f.name], capture_output=True, text=True, check=True)
        pathlib.Path(f.name).unlink()
        n = count_dump(r.stdout)
        print("  %9d chars -> %7d tokens" % (chars, n), flush=True)
        return n
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="480000,960000")
    args = ap.parse_args()
    source = fetch_data.DATA / eval_suites.HAYSTACK
    count = ds4_counter(args.model, source)
    size = len(source.read_text(errors="replace"))
    for target in (int(t) for t in args.targets.split(",")):
        chars = search(count, target, lo=int(target * 3.0), hi=min(size - 1000, int(target * 4.2)),
                       tol=max(500, target // 1000))
        print('    ("%dk", %d, %d),' % (target // 1000, target, chars))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the calibration (CPU only, the gateway can stay up)**

`ds4 --dump-tokens` only maps the GGUF and loads the vocabulary. It does not touch Metal.
Run:
```bash
python3 speed-bench/nextgen-eval/calibrate_tiers.py \
  --model ~/orca/workspaces/ds4-metal-data/gguf/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \
  2>&1 | tail -3
```
Expected: two lines like `    ("480k", 480000, 17xxxxx),` and `    ("960k", 960000, 35xxxxx),`. The probe
predicts about 1.72M and 3.52M characters. Record both numbers in the ledger. If `--prompt-file` is
not the flag's name, read `./ds4 --help | grep -i prompt`, use the flag it lists, and ledger a
ruling.

- [ ] **Step 5: Implement the harness changes**

`eval_suites.py`: replace
```python
TIERS = [("120k", 120000), ("240k", 240000), ("480k", 480000), ("960k", 960000)]
CHARS_PER_TOKEN = 3.03  # measured 2026-09-23: 685K characters of ds4.c = 226,292 tokens
```
with the list below, using the two character counts from Step 4 for 480k and 960k:
```python
# (label, target tokens, haystack characters). 120k/240k keep tokens x 3.03 (measured 2026-09-23 on
# the head of ds4.c), so earlier rows stay comparable. Deeper text tokenizes at ~3.5-3.7 chars/token,
# so 480k/960k are calibrated by calibrate_tiers.py (2026-09-30, the chat template of `ds4 --dump-tokens`).
TIERS = [("120k", 120000, 363600), ("240k", 240000, 727200),
         ("480k", 480000, CHARS_480K_FROM_STEP_4), ("960k", 960000, CHARS_960K_FROM_STEP_4)]
SHORT_TIER = 0.95  # a needle prompt more than 5% under its target is reported as short
```
Here `CHARS_480K_FROM_STEP_4` and `CHARS_960K_FROM_STEP_4` stand for the integers Step 4 printed.
Write the integers themselves. Keep `CHARS_PER_TOKEN = 3.03` and its comment, because
`test_eval_suites.py:158` reads it.

Replace `haystack`:
```python
def haystack(path, chars, depth=0.5):
    """The first `chars` characters of the frozen ds4.c, with the needle at `depth`."""
    src = pathlib.Path(path).read_text(errors="replace")
    if chars > len(src):
        raise ValueError("%s has %d characters, need %d" % (path, len(src), chars))
    text = src[:chars]
    cut = text.rfind("\n", 0, int(chars * depth)) + 1
    return text[:cut] + NEEDLE + text[cut:]
```
In `run_longctx`, the loop becomes `for label, tokens, chars in TIERS:`, the document is built from
`haystack(source, chars)`, and the needle row gains `target_tokens=tokens`. For example:
```python
            yield (dict(_speed(r), suite="longctx", id="needle-" + label,
                             passed=graders.needle_hit(r["content"], NEEDLE_VALUE),
                             prompt_tokens=r["usage"].get("prompt_tokens"), target_tokens=tokens,
                             answer=r["content"][:200]))
```
`run.py` `summarize`, next to `needle = {...}`:
```python
    short_tiers = sorted(r["id"][len("needle-"):] for r in rows
                         if r["suite"] == "longctx" and r["id"].startswith("needle-")
                         and r.get("prompt_tokens") and r.get("target_tokens")
                         and r["prompt_tokens"] < eval_suites.SHORT_TIER * r["target_tokens"])
```
Add `"short_tiers": short_tiers` to the `"longctx"` dict. `run.py` already imports `eval_suites`.

`compare.py` `render_markdown`: directly after the line
`lines.append("| swap-outs | %s | %s |" % (bl.get("swapouts"), cl.get("swapouts")))` add:
```python
    lines.append("| short tiers | %s | %s |" % (", ".join(bl.get("short_tiers") or []) or "none",
                                                ", ".join(cl.get("short_tiers") or []) or "none"))
```

`README.md`: in the suite table's `longctx` row, after "needle at ~120K/240K/480K/960K tokens", add:
"(the deep tiers are calibrated in real tokens by `calibrate_tiers.py`; a needle more than 5% under
its target is listed as a short tier)".

- [ ] **Step 6: Run the whole harness suite**

Run: `python3 -m unittest discover -s speed-bench/nextgen-eval -p 'test_*.py' 2>&1 | tail -3`
Expected: `OK`, with the test count up from 129.

- [ ] **Step 7: Commit**

```bash
git add speed-bench/nextgen-eval/calibrate_tiers.py speed-bench/nextgen-eval/test_calibrate_tiers.py \
  speed-bench/nextgen-eval/eval_suites.py speed-bench/nextgen-eval/test_eval_suites.py \
  speed-bench/nextgen-eval/run.py speed-bench/nextgen-eval/test_run.py \
  speed-bench/nextgen-eval/compare.py speed-bench/nextgen-eval/test_compare.py \
  speed-bench/nextgen-eval/README.md
git commit -m "nextgen-eval: size the long-context tiers in real tokens

The harness sized a tier as tokens x 3.03 characters, measured on the
head of ds4.c. Deeper text tokenizes at ~3.6, so the 480K needle was
415K tokens. calibrate_tiers.py finds the character count per target
with ds4 --dump-tokens; 120k/240k keep their old sizes. Needle rows
carry target_tokens and the summary lists short tiers.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

### Task 5: ds4-eval gets the server's rope; the YaRN arm config (harness)

**Files:**
- Modify: `speed-bench/nextgen-eval/ds4eval.py` (`SERVER_ONLY_WITH_VALUE`, new `eval_env`,
  `run_reason`)
- Create: `speed-bench/nextgen-eval/configs/ivan-proj-yarn.json`
- Test: `test_ds4eval.py`, `test_run.py`

**Interfaces:**
- Consumes: the factor rule of Task 1 (`ds4_qwen4_yarn_factor`), mirrored in Python.
- Produces: `ds4eval.eval_env(env: dict, server_argv: list) -> dict`; `ds4eval.QWEN4_NATIVE_CTX = 262144`.

- [ ] **Step 1: Write the failing tests**

In `test_ds4eval.py`, add to the class holding `test_eval_argv_whitelists_and_caps_ctx`:
```python
    def test_eval_env_carries_the_server_rope(self):
        # ds4-eval runs at -c <= 65536, below the native context, so it never derives YaRN from -c:
        # the harness hands it the factor the server derives from its own -c.
        self.assertNotIn("DS4_QWEN4_YARN_FACTOR", ds4eval.eval_env({}, ["ds4-server", "-c", "262144"]))
        self.assertEqual(ds4eval.eval_env({}, ["ds4-server", "-c", "524288"])["DS4_QWEN4_YARN_FACTOR"], "2")
        self.assertEqual(ds4eval.eval_env({}, ["ds4-server", "-c", "1048576"])["DS4_QWEN4_YARN_FACTOR"], "4")
        self.assertEqual(ds4eval.eval_env({"DS4_QWEN4_YARN_FACTOR": "1"}, ["ds4-server", "-c", "524288"]),
                         {"DS4_QWEN4_YARN_FACTOR": "1"})   # an explicit override is kept
        self.assertEqual(ds4eval.eval_env({"A": "b"}, ["ds4-server"]), {"A": "b"})

    def test_continued_interval_is_server_only(self):
        argv = ds4eval.eval_argv(["ds4-server", "--kv-cache-continued-interval-tokens", "0"],
                                 "/r", "core", "-", 1, "/t")
        self.assertNotIn("--kv-cache-continued-interval-tokens", argv)
```
Check that `run_reason` passes the env through `eval_env`. The existing `Run` tests patch
`subprocess.Popen` and never look at the env, so add this test to class `Run`. It patches `ds4eval._run`,
which `run_reason` calls as `_run(argv, cwd, env)`. An empty report then fails loudly, after the env
was recorded:
```python
    def test_run_reason_hands_ds4_eval_the_rope(self):
        seen = []

        def fake_run(argv, cwd, env):
            seen.append(env.get("DS4_QWEN4_YARN_FACTOR"))
            return 1, b"", b""

        with mock.patch.object(ds4eval, "_run", fake_run), tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                list(ds4eval.run_reason({}, ["ds4-server", "-c", "524288"], "/r", d))
        self.assertEqual(seen[:1], ["2"])
```
(`test_ds4eval.py` already imports `tempfile` and `mock`.) In `test_run.py`, class `Summarize`, next to
`test_ivan_configs_differ_only_by_the_projection`:
```python
    def test_yarn_config_is_the_projection_arm_at_512k(self):
        proj = json.loads((HERE / "configs" / "ivan-proj.json").read_text())
        yarn = json.loads((HERE / "configs" / "ivan-proj-yarn.json").read_text())
        self.assertEqual(yarn["name"], "ivan-proj-s050-yarn")
        self.assertEqual(yarn["args_add"], proj["args_add"] + ["-c", "524288"])
        self.assertEqual(dict(yarn, name=proj["name"], args_add=proj["args_add"]), proj)
        self.assertNotIn("DS4_QWEN4_YARN_FACTOR", yarn["env"])   # the server derives it from -c
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python3 -m unittest discover -s speed-bench/nextgen-eval -p 'test_*.py' 2>&1 | tail -4`
Expected: FAILED. There are errors for `eval_env` and the missing config file, and a failure for the
server-only flag (a `ValueError` from `eval_argv`).

- [ ] **Step 3: Implement**

`ds4eval.py`:
- add `"--kv-cache-continued-interval-tokens"` to `SERVER_ONLY_WITH_VALUE`;
- after `CTX_CAP = 65536` add:
```python
QWEN4_NATIVE_CTX = 262144  # qwen4exp.context_length of every Qwen3.8-Flash-Next GGUF this harness runs


def eval_env(env, server_argv):
    """The arm's env for ds4-eval. ds4-eval runs at -c <= CTX_CAP, below the native context, so it
    would never derive YaRN from -c the way the server does: hand it the server's factor, by ds4.c's
    rule (the smallest power of two covering -c / native). An explicit DS4_QWEN4_YARN_FACTOR wins."""
    out = dict(env)
    if out.get("DS4_QWEN4_YARN_FACTOR"):
        return out
    ctx = int(server.argv_value(server_argv, "-c") or 0)
    if ctx <= QWEN4_NATIVE_CTX:
        return out
    factor = 2
    while QWEN4_NATIVE_CTX * factor < ctx:
        factor *= 2
    out["DS4_QWEN4_YARN_FACTOR"] = str(factor)
    return out
```
- in `run_reason`, replace `{**os.environ, **env}` with `{**os.environ, **eval_env(env, server_argv)}`.

`configs/ivan-proj-yarn.json`:
```json
{"name": "ivan-proj-s050-yarn", "base": "registry", "registry_model": "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2",
 "model": "/Users/dongnh/orca/workspaces/ds4-metal-data/gguf/ivan/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf",
 "args_add": ["--dir-steering-file", "/Users/dongnh/orca/workspaces/ds4-metal-data/steering/refusal-4-44.f32",
              "--dir-steering-ffn", "0.5", "-c", "524288"],
 "args_remove": [], "env": {}}
```
README, after the sentence that names `ivan-proj.json` and scale 0.5, add: "`ivan-proj-yarn.json` is
the same arm at `-c 524288`, with YaRN factor 2 derived from `-c` (sub-project 4). ds4-eval gets the
factor through `DS4_QWEN4_YARN_FACTOR`, because its `-c` is capped below the native context."

- [ ] **Step 4: Run the whole harness suite**

Run: `python3 -m unittest discover -s speed-bench/nextgen-eval -p 'test_*.py' 2>&1 | tail -3`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add speed-bench/nextgen-eval/ds4eval.py speed-bench/nextgen-eval/test_ds4eval.py \
  speed-bench/nextgen-eval/configs/ivan-proj-yarn.json speed-bench/nextgen-eval/test_run.py \
  speed-bench/nextgen-eval/README.md
git commit -m "nextgen-eval: ds4-eval runs with the server's YaRN factor; 512K arm config

ds4-eval is capped at -c 65536, so it would never derive YaRN from -c;
eval_env hands it the server's factor (an explicit override wins).
--kv-cache-continued-interval-tokens is server-only. The arm
ivan-proj-s050-yarn is ivan-proj at -c 524288.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```

### Task 6: GPU pre-run checks (needs the user's go-ahead and a paused stack)

**Files:**
- Create (not git): `~/orca/workspaces/ds4-metal-data/sp4/checks.py`,
  `~/orca/workspaces/ds4-metal-data/sp4/prod-240k-cont.json`,
  `~/orca/workspaces/ds4-metal-data/sp4/prod-240k-nocont.json`
- Results (git, Task 7): noted in the ledger now, written up in Task 7

- [ ] **Step 1: Write the checks script**

`~/orca/workspaces/ds4-metal-data/sp4/checks.py`:
```python
"""Sub-project 4 GPU checks (not part of the harness): the default path is unchanged."""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("/Users/dongnh/orca/workspaces/ds4-metal/kv-grow")
OUT = pathlib.Path.home() / "orca/workspaces/ds4-metal-data/sp4"
MODELS = pathlib.Path.home() / ".local/share/ai-gateway/ds4-models"
PROD = MODELS / "Qwen3.8-Flash-Next-OrcaUncensored-IQ2XXS-Q4KDownPad768-DenseQ4Kselimat-MTP.gguf"
PLE = MODELS / "Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
ENV = dict(os.environ, DS4_QWEN4_STREAM_FULL_LAYERS="32", DS4_QWEN4_PLE_PREFETCH_FULL="0",
           DS4_QWEN4_KV_GROW="1",
           DS4_QWEN4_MTP_DRAFT_VOCAB=str(MODELS / "Qwen3.8-Flash-Next-draft-vocab-vi-en-code-64k.txt"))
ENV.pop("DS4_QWEN4_YARN_FACTOR", None)
FLAGS = ["--metal", "--ple", str(PLE), "-c", "262144", "--prefill-chunk", "2048", "--ssd-streaming",
         "--ssd-streaming-cache-experts", "6GB", "--nothink", "--temp", "0"]
PROMPTS = ["Write a Python function that checks whether a number is prime.",
           "Giải thích ngắn gọn vì sao bầu trời có màu xanh.",
           "List three differences between TCP and UDP."]


def run(binary, prompt):
    r = subprocess.run([str(binary), "-m", str(PROD)] + FLAGS + ["-n", "128", "-p", prompt],
                       cwd=str(REPO), env=ENV, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def identity():
    same = True
    for i, p in enumerate(PROMPTS):
        outs = []
        for tag, binary in (("baseline", OUT / "baseline/ds4"), ("branch", REPO / "ds4")):
            code, out, err = run(binary, p)
            (OUT / ("identity-%d-%s.txt" % (i, tag))).write_text(out)
            (OUT / ("identity-%d-%s.log" % (i, tag))).write_text(err)
            if code != 0:
                print("prompt %d %s exit %d" % (i, tag, code))
                same = False
            if "YaRN" in err:
                print("prompt %d %s logged YaRN at -c 262144" % (i, tag))
                same = False
            outs.append(out)
        print("prompt %d: %s" % (i, "identical" if outs[0] == outs[1] else "DIFFERENT"))
        same = same and outs[0] == outs[1]
    print("identity: %s" % ("PASS" if same else "FAIL"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit({"identity": identity}[sys.argv[1]]())
```
Both 240K configs start from the registry's PROD Qwen row (`configs/prod.json`). The first adds
nothing, and the second adds `--kv-cache-continued-interval-tokens 0`:
```json
{"name": "prod-240k-cont", "base": "registry", "registry_model": "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2",
 "model": null, "args_add": [], "args_remove": [], "env": {}}
```
```json
{"name": "prod-240k-nocont", "base": "registry", "registry_model": "ivanfioravanti--Qwen3.8-Flash-Next-DS4-IQ2",
 "model": null, "args_add": ["--kv-cache-continued-interval-tokens", "0"], "args_remove": [], "env": {}}
```

- [ ] **Step 2: Get the user's go-ahead, then pause the gateway stack**

Ask first. Then pause:
1. Confirm `curl -s http://127.0.0.1:8090/status` shows `"active":[]`.
2. `launchctl unload ~/Library/LaunchAgents/dev.dongnh.gateway-watchdog.plist` and
   `launchctl unload ~/Library/LaunchAgents/dev.dongnh.ai-proxy.plist`.
3. SIGTERM the pid in `~/.local/share/ai-gateway/ds4-ornith.pid` after checking that it names a
   ds4-server.
4. SIGTERM the gateway `server.py`, found with `lsof -nP -iTCP:8090 -sTCP:LISTEN -t`. It survives the
   unload.
5. Confirm that `pgrep -fl ds4-server` is empty.

- [ ] **Step 3: Default path unchanged**

Run: `python3 ~/orca/workspaces/ds4-metal-data/sp4/checks.py identity`
Expected: `prompt 0/1/2: identical`, `identity: PASS`.

- [ ] **Step 4: 240K on PROD's model, continued checkpoints at today's interval vs off (measured, not gated)**

Run one after the other (each takes about 15 minutes):
```bash
python3 speed-bench/nextgen-eval/run.py --config ~/orca/workspaces/ds4-metal-data/sp4/prod-240k-cont.json --suites longctx > ~/orca/workspaces/ds4-metal-data/sp4/prod-240k-cont.out 2>&1
python3 speed-bench/nextgen-eval/run.py --config ~/orca/workspaces/ds4-metal-data/sp4/prod-240k-nocont.json --suites longctx > ~/orca/workspaces/ds4-metal-data/sp4/prod-240k-nocont.out 2>&1
```
Expected: both exit 0. Ledger each arm's `needle-240k` prefill t/s, total seconds, peak wired and
swap-outs.

- [ ] **Step 5: Restore the gateway**

Restore the stack:
1. `launchctl load` both plists.
2. `launchctl kickstart gui/$(id -u)/dev.dongnh.gateway-watchdog`.
3. Wait until `curl -s http://127.0.0.1:8090/status` shows `"backend_ok":true`.

Leave the stack paused instead only if the user has approved running Task 7 straight after.

### Task 7: The exit arm and the results (needs the user's go-ahead; about 7 GPU hours)

**Files:**
- Create: `speed-bench/nextgen-eval/results/<run date>-sp4-yarn.md`,
  `results/<run date>-sp4-ivan-proj-s050-yarn.summary.json`,
  `results/<run date>-sp4-yarn-vs-s050.md`

- [ ] **Step 1: Run the arm (stack paused, as in Task 6 Step 2)**

Run in the background:
```bash
python3 speed-bench/nextgen-eval/run.py --config speed-bench/nextgen-eval/configs/ivan-proj-yarn.json \
  > ~/orca/workspaces/ds4-metal-data/sp4/arm-yarn.out 2>&1
```
Expected:
- `server.log` shows "YaRN factor 2 (from -c)" and "keyed kv cache directory …/steer-19516f04-yarn-2";
- the reasoning suite's `ds4-eval-*.log` files show "YaRN factor 2 (DS4_QWEN4_YARN_FACTOR)";
- every suite finishes.

Restore the stack afterwards, as in Task 6 Step 5.

- [ ] **Step 2: Gate**

Run:
```bash
cd speed-bench/nextgen-eval
python3 compare.py ~/orca/workspaces/ds4-metal-data/evals/nextgen/runs/ivan-proj-s050-20260929-063533/summary.json \
  RUNS/ivan-proj-s050-yarn-<stamp>/summary.json --gate engine --refusal-caps 5,1 \
  --out results/<run date>-sp4-yarn-vs-s050.md; echo "exit $?"
```
Expected: the exit code, the per-suite verdicts, and the long-context table:
- needles 120k/240k/480k;
- `short tiers: none`;
- swap-outs 0.

Sub-project 4 passes on exit 0, with the 480K needle hit, a real prompt within 5% of 480,000 tokens,
and zero swap-outs. If only short-prompt suites regress, the spec's rule applies: 512K ships as a
separate registry entry, and the default stays at 262K.

- [ ] **Step 3: Write the results and commit**

Copy the arm's `summary.json` into `results/<run date>-sp4-ivan-proj-s050-yarn.summary.json`. Then
write `results/<run date>-sp4-yarn.md` with these sections:
- what ran;
- the gate table;
- long context: real tokens per tier, prefill t/s, peak wired and swap-outs;
- Task 6's identity and 240K A/B numbers;
- the verdict: pass, pass as a separate entry, or fail, with the reason;
- the probe numbers from the spec, for reference.

Commit:
```bash
git add speed-bench/nextgen-eval/results/*sp4*
git commit -m "nextgen-eval: sub-project 4 exit arm, <verdict in a few words>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017f1tsCGFkVQ3wrmPsJ7FBn"
```
