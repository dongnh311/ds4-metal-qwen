#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * Stream gates against the per-layer drain on a synthetic expert file
 * (IQ2_XXS gate/up, Q2_K down; GLM-5.3 routes 8 of 288 experts). Every step
 * encodes LAYERS streamed layers in one batch, gated first so its layers
 * miss, then drained; the outputs must be byte-identical while a cache of
 * BUDGET experts evicts on every step.
 */
enum { D = 256, H = 512, E = 288, N = 8, LAYERS = 4, STEPS = 24, BUDGET = 24 };
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;

static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void *model;
static size_t model_bytes;
static FILE *model_file;
static uint64_t row_bytes, down_row_bytes, expert_bytes, down_expert_bytes, tensor_bytes;
static ds4_gpu_tensor *xt, *wt, *gate, *up, *mid, *down, *shared_t;
static ds4_gpu_tensor *ids_t[LAYERS], *out_t[LAYERS];
static float ref[LAYERS][N * D], got[LAYERS][N * D];
/* Elements each layer writes: qwen4 partials are N x D, GLM's summed output D. */
static int out_elems = N * D;
static int drain_layer = -1;   /* this layer never publishes (mixed tokens) */

static int setup(void) {
    row_bytes = D / 256 * sizeof(iq2_block);
    down_row_bytes = H / 256 * sizeof(q2_block);
    expert_bytes = (uint64_t)H * row_bytes;
    down_expert_bytes = (uint64_t)D * down_row_bytes;
    tensor_bytes = (uint64_t)E * expert_bytes;
    model_bytes = 2 * tensor_bytes + (uint64_t)E * down_expert_bytes;
    model_file = tmpfile();
    if (!model_file || ftruncate(fileno(model_file), (off_t)model_bytes)) return 0;
    model = mmap(NULL, model_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(model_file), 0);
    if (model == MAP_FAILED) return 0;
    iq2_block *g = model;
    for (uint64_t i = 0; i < 2 * tensor_bytes / sizeof(iq2_block); i++) {
        g[i].d = 0x1400;
        for (size_t j = 0; j < sizeof(g[i].qs); j++) g[i].qs[j] = (uint8_t)random_u32();
    }
    q2_block *q = (q2_block *)((char *)model + 2 * tensor_bytes);
    for (uint64_t i = 0; i < (uint64_t)E * down_expert_bytes / sizeof(q2_block); i++) {
        q[i].d = 0x1400;
        q[i].dmin = 0x1000;
        for (size_t j = 0; j < sizeof(q[i].scales); j++) q[i].scales[j] = (uint8_t)random_u32();
        for (size_t j = 0; j < sizeof(q[i].qs); j++) q[i].qs[j] = (uint8_t)random_u32();
    }
    if (msync(model, model_bytes, MS_SYNC) || !ds4_gpu_init()) return 0;
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * expert_bytes + down_expert_bytes);
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_set_model_fd(fileno(model_file))) return 0;
    float x[D], w[N];
    for (int i = 0; i < D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int i = 0; i < N; i++) w[i] = (i + 1) / 36.0f;
    xt = ds4_gpu_tensor_alloc(sizeof(x));
    wt = ds4_gpu_tensor_alloc(sizeof(w));
    gate = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    up = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    mid = ds4_gpu_tensor_alloc((uint64_t)N * H * sizeof(float));
    down = ds4_gpu_tensor_alloc((uint64_t)N * D * sizeof(float));
    shared_t = ds4_gpu_tensor_alloc(sizeof(x));
    int ok = xt && wt && gate && up && mid && down && shared_t &&
             ds4_gpu_tensor_write(xt, 0, x, sizeof(x)) &&
             ds4_gpu_tensor_write(wt, 0, w, sizeof(w));
    for (int l = 0; ok && l < LAYERS; l++) {
        ids_t[l] = ds4_gpu_tensor_alloc(N * sizeof(int32_t));
        out_t[l] = ds4_gpu_tensor_alloc((uint64_t)N * D * sizeof(float));
        ok = ids_t[l] && out_t[l];
    }
    return ok;
}

/* Four hot experts per layer and four that rotate; the slot order moves so a
 * miss lands in every slot position. */
static void route(int step, int layer, int32_t ids[N]) {
    for (int i = 0; i < N; i++) {
        const int slot = (i + step + layer) % N;
        ids[slot] = i < 4 ? layer * 4 + i : 64 + (step * 4 + i + layer * 37) % (E - 64);
    }
}

typedef int (*layer_fn)(int layer, int gated);

/* One step: LAYERS layers in one batch, then every layer's output. */
static int run_step(layer_fn fn, int gated, float out[LAYERS][N * D]) {
    for (int l = 0; l < LAYERS; l++)
        if (!ds4_gpu_tensor_fill_f32(out_t[l], NAN, (uint64_t)N * D)) return 0;
    if (!ds4_gpu_begin_commands()) return 0;
    int ok = 1;
    for (int l = 0; ok && l < LAYERS; l++) ok = fn(l, gated);
    ok = ds4_gpu_end_commands() && ok;
    for (int l = 0; ok && l < LAYERS; l++)
        ok = ds4_gpu_tensor_read(out_t[l], 0, out[l], sizeof(out[l]));
    return ok;
}

static int write_routes(int step) {
    for (int l = 0; l < LAYERS; l++) {
        int32_t ids[N];
        route(step, l, ids);
        if (!ds4_gpu_tensor_write(ids_t[l], 0, ids, sizeof(ids))) return 0;
    }
    return 1;
}

static int same_outputs(const char *name, int step) {
    for (int l = 0; l < LAYERS; l++) {
        for (int i = 0; i < out_elems; i++) {
            if (!isfinite(ref[l][i]) || memcmp(&ref[l][i], &got[l][i], sizeof(float))) {
                fprintf(stderr, "%s: step %d layer %d element %d: drain %g gated %g\n",
                        name, step, l, i, ref[l][i], got[l][i]);
                return 0;
            }
        }
    }
    return 1;
}

/* STEPS steps, gated then drained; counts the gates committed. */
static int compare_modes(const char *name, layer_fn fn, int split, uint64_t *gated_layers,
                         uint64_t *split_gates, uint64_t *fallback) {
    uint64_t c0 = 0, s0 = 0, f0 = 0, c1 = 0, s1 = 0, f1 = 0;
    int failed = 0;
    ds4_gpu_stream_gate_stats(&c0, &s0, &f0, &failed);
    int ok = !failed;
    for (int step = 0; ok && step < STEPS; step++) {
        ok = write_routes(step);
        ds4_gpu_stream_gate_test_set_mode(1, split);
        ok = ok && run_step(fn, 1, got);
        ds4_gpu_stream_gate_test_set_mode(0, split);
        ok = ok && run_step(fn, 0, ref) && same_outputs(name, step);
    }
    ds4_gpu_stream_gate_stats(&c1, &s1, &f1, &failed);
    *gated_layers = c1 - c0;
    if (split_gates) *split_gates = s1 - s0;
    if (fallback) *fallback = f1 - f0;
    if (ok && failed) {
        fprintf(stderr, "%s: a gate failed\n", name);
        ok = 0;
    }
    /* Step 0 drains until the first miss allocates the cache slab. */
    const uint64_t want = (uint64_t)(STEPS - 1) * (drain_layer >= 0 ? LAYERS - 1 : LAYERS);
    if (ok && *gated_layers < want) {
        fprintf(stderr, "%s: only %llu gated layers\n", name, (unsigned long long)*gated_layers);
        ok = 0;
    }
    fprintf(stderr, "%s: %llu gated layers: %s\n", name, (unsigned long long)*gated_layers,
            ok ? "PASS" : "FAIL");
    return ok;
}

/* end_commands after a step whose last gate keeps the service thread busy
 * past its release: qwen4 returns at once (PROD unchanged), GLM waits so the
 * cache is the main thread's again. */
static int check_end_wait(const char *name, layer_fn fn, int split, int want_wait) {
    ds4_gpu_stream_gate_test_set_mode(1, split);
    int ok = write_routes(0) && run_step(fn, 1, got);
    ds4_gpu_stream_gate_test_stall_after_release(LAYERS, 300);
    ok = ok && write_routes(1) && run_step(fn, 1, got);
    const int busy = ds4_gpu_stream_gate_test_service_busy();
    ds4_gpu_stream_gate_test_stall_after_release(0, 0);
    int failed = 0;
    ds4_gpu_stream_gate_stats(NULL, NULL, NULL, &failed);   /* waits for the service */
    ds4_gpu_stream_gate_test_set_mode(0, split);
    if (ok && (failed || busy == want_wait)) {
        fprintf(stderr, "%s: end_commands %s the busy service thread (failed %d)\n", name,
                want_wait ? "did not wait for" : "waited for", failed);
        ok = 0;
    }
    fprintf(stderr, "%s: %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

static int qwen4_layer(int layer, int gated) {
    (void)gated;   /* ds4_gpu_stream_gate_test_set_mode picks the path */
    return ds4_gpu_qwen4_moe_stream_layer(
               mid, out_t[layer], xt, ids_t[layer], model, model_bytes, 3u + (uint32_t)layer,
               0, tensor_bytes, 2 * tensor_bytes, 16u, 10u,
               E, 1u, N, D, H, D, 0, 0, 0, UINT32_MAX, UINT32_MAX) != 0;
}

static int qwen4_suite(void) {
    uint64_t gated = 0, split = 0;
    ds4_gpu_set_glm_model(false);
    int ok = compare_modes("qwen4 one pass", qwen4_layer, 0, &gated, &split, NULL);
    ok = ok && compare_modes("qwen4 split", qwen4_layer, 1, &gated, &split, NULL);
    if (ok && split == 0) {
        fprintf(stderr, "qwen4 split: no split gate ran\n");
        ok = 0;
    }
    ok = ok && check_end_wait("qwen4 end without service wait", qwen4_layer, 1, 0);
    return ok;
}

static int glm_layer(int layer, int gated) {
    const ds4_gpu_stream_expert_table table = {
        .model_map = model, .model_size = model_bytes, .layer = 3u + (uint32_t)layer,
        .n_total_expert = E, .gate_offset = 0, .up_offset = tensor_bytes,
        .down_offset = 2 * tensor_bytes, .gate_expert_bytes = expert_bytes,
        .down_expert_bytes = down_expert_bytes,
    };
    const int published = gated && layer != drain_layer &&
                          ds4_gpu_glm_stream_gate_publish(&table, ids_t[layer], N);
    /* GPU work between publish and commit, where ds4.c encodes the shared expert. */
    if (!ds4_gpu_add_tensor(shared_t, xt, xt, D)) return 0;
    if (published && !ds4_gpu_glm_stream_gate_commit()) return 0;
    return ds4_gpu_routed_moe_one_tensor(
               out_t[layer], gate, up, mid, down, model, model_bytes,
               0, tensor_bytes, 2 * tensor_bytes, 16u, 10u,
               expert_bytes, row_bytes, down_expert_bytes, down_row_bytes, D, H, D,
               ids_t[layer], wt, E, N, 7.0f, xt, NULL, 3u + (uint32_t)layer, false) != 0;
}

/* A publish whose layer fails before commit must not leak into the next batch. */
static int check_uncommitted_publish(int split) {
    const ds4_gpu_stream_expert_table table = {
        .model_map = model, .model_size = model_bytes, .layer = 3u,
        .n_total_expert = E, .gate_offset = 0, .up_offset = tensor_bytes,
        .down_offset = 2 * tensor_bytes, .gate_expert_bytes = expert_bytes,
        .down_expert_bytes = down_expert_bytes,
    };
    ds4_gpu_stream_gate_test_set_mode(1, split);
    int ok = write_routes(STEPS) && ds4_gpu_begin_commands();
    const int published = ok && ds4_gpu_glm_stream_gate_publish(&table, ids_t[0], N);
    ok = ds4_gpu_end_commands() && ok && published;
    ok = ok && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_set_mode(0, split);
    ok = ok && run_step(glm_layer, 0, ref) && same_outputs("glm uncommitted publish", STEPS);
    fprintf(stderr, "glm uncommitted publish: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* Switches that keep the routed MoE off the gate's tables (--quality, the
 * fusion and address-table disables, clamped activations) must refuse the
 * publish: a committed gate nobody takes reads experts for nothing and holds
 * off the next layer's gate. */
static int check_routed_switches(int split) {
    static const char *const envs[] = {
        "DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE", "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
        "DS4_METAL_MOE_WRITE_CLAMPED_ACT", NULL /* --quality */,
    };
    int ok = 1;
    for (int i = 0; ok && i < 4; i++) {
        const char *what = envs[i] ? envs[i] : "quality";
        uint64_t c0 = 0, c1 = 0;
        int failed = 0;
        if (envs[i]) setenv(envs[i], "1", 1);
        else ds4_gpu_set_quality(true);
        ds4_gpu_stream_gate_stats(&c0, NULL, NULL, &failed);
        ok = write_routes(STEPS + 1);
        ds4_gpu_stream_gate_test_set_mode(1, split);
        ok = ok && run_step(glm_layer, 1, got);
        ds4_gpu_stream_gate_test_set_mode(0, split);
        ok = ok && run_step(glm_layer, 0, ref) && same_outputs(what, STEPS + 1);
        ds4_gpu_stream_gate_stats(&c1, NULL, NULL, &failed);
        if (envs[i]) unsetenv(envs[i]);
        else ds4_gpu_set_quality(false);
        if (ok && (c1 != c0 || failed)) {
            fprintf(stderr, "glm %s: %llu gates committed (failed %d)\n", what,
                    (unsigned long long)(c1 - c0), failed);
            ok = 0;
        }
    }
    fprintf(stderr, "glm routed switches: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int glm_suite(int split) {
    const char *tag = split ? "glm split" : "glm one pass";
    char name[64];
    uint64_t gated = 0, splits = 0, fallback = 0;
    ds4_gpu_set_glm_model(true);
    out_elems = D;
    int ok = compare_modes(tag, glm_layer, split, &gated, &splits, &fallback);
    if (ok && split && splits == 0) {
        fprintf(stderr, "%s: no split gate ran\n", tag);
        ok = 0;
    }
    /* Mixed token: layer 2 drains between gated layers. */
    snprintf(name, sizeof(name), "%s mixed", tag);
    drain_layer = 2;
    uint64_t mixed = 0;
    ok = ok && compare_modes(name, glm_layer, split, &mixed, NULL, NULL);
    drain_layer = -1;
    /* Fallback buffer: every miss (and, split, every hit) treated as uncached. */
    snprintf(name, sizeof(name), "%s fallback", tag);
    ds4_gpu_stream_gate_test_force_fallback(1);
    ok = ok && compare_modes(name, glm_layer, split, &gated, NULL, &fallback);
    ds4_gpu_stream_gate_test_force_fallback(0);
    if (ok && fallback == 0) {
        fprintf(stderr, "%s: the fallback buffer was never used\n", name);
        ok = 0;
    }
    ok = ok && check_uncommitted_publish(split);
    ok = ok && check_routed_switches(split);
    ok = ok && check_end_wait(split ? "glm split end waits for the service" :
                                      "glm end waits for the service", glm_layer, split, 1);
    /* A larger budget adds a slab: gates pause until it exists, then resume. */
    snprintf(name, sizeof(name), "%s budget growth", tag);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET + 16);
    ok = ok && compare_modes(name, glm_layer, split, &gated, NULL, NULL);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    return ok;
}

/* A gate held past the poll's ~600 ms window must fail its batch, latch
 * gates off, and leave later steps exact on the drain path. */
static int check_timeout(void) {
    uint64_t c0 = 0, c1 = 0;
    int failed = 0;
    ds4_gpu_set_glm_model(true);
    out_elems = D;
    ds4_gpu_stream_gate_test_set_mode(1, 0);
    int ok = write_routes(0) && run_step(glm_layer, 1, got);   /* warm: slabs and gates up */
    ds4_gpu_stream_gate_test_stall(2, 900);
    const int stalled_ok = ok && write_routes(1) && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_stall(0, 0);
    ds4_gpu_stream_gate_stats(&c0, NULL, NULL, &failed);
    if (ok && (stalled_ok || !failed)) {
        fprintf(stderr, "glm timeout: the held gate was not reported (step ok %d, failed %d)\n",
                stalled_ok, failed);
        ok = 0;
    }
    ok = ok && write_routes(2) && run_step(glm_layer, 1, got);
    ds4_gpu_stream_gate_test_set_mode(0, 0);
    ok = ok && run_step(glm_layer, 0, ref) && same_outputs("glm after timeout", 2);
    ds4_gpu_stream_gate_stats(&c1, NULL, NULL, &failed);
    if (ok && c1 != c0) {
        fprintf(stderr, "glm timeout: %llu gates ran after the failure\n",
                (unsigned long long)(c1 - c0));
        ok = 0;
    }
    ds4_gpu_stream_gate_test_clear_failure();
    fprintf(stderr, "glm timeout: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* With gates requested, only the models whose gates need every slab (qwen4,
 * GLM) put the whole cache budget in one slab; any other model keeps the slab
 * target (here 4 slots). One step of 4 layers x 8 routes fills the budget. */
static int check_slabs(const char *name, bool glm, bool qwen4, uint32_t want) {
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t slot = (2 * expert_bytes + down_expert_bytes + page - 1) / page * page;
    ds4_gpu_set_glm_model(glm);
    ds4_gpu_set_qwen4_model(qwen4);
    ds4_gpu_stream_expert_slab_test_set_target(4 * slot);
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);   /* empties the cache and its slabs */
    ds4_gpu_stream_gate_test_set_mode(1, 0);
    int ok = write_routes(0) && run_step(glm_layer, 0, ref);
    ds4_gpu_stream_gate_test_set_mode(0, 0);
    const uint32_t slabs = ds4_gpu_stream_expert_slab_test_count();
    ds4_gpu_stream_expert_slab_test_set_target(0);
    if (ok && slabs != want) {
        fprintf(stderr, "%s: %u slabs, want %u\n", name, slabs, want);
        ok = 0;
    }
    fprintf(stderr, "%s: %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

static int slab_suite(void) {
    out_elems = D;
    int ok = check_slabs("deepseek keeps the slab target", false, false, BUDGET / 4);
    ok = ok && check_slabs("qwen4 one slab", false, true, 1);
    ok = ok && check_slabs("glm one slab", true, false, 1);
    ds4_gpu_set_qwen4_model(false);
    ds4_gpu_set_glm_model(false);
    return ok;
}

int main(int argc, char **argv) {
    const char *mode = argc == 2 ? argv[1] : "";
    if (strcmp(mode, "--qwen4") && strcmp(mode, "--glm") && strcmp(mode, "--glm-timeout") &&
        strcmp(mode, "--glm-split") && strcmp(mode, "--slabs")) {
        fprintf(stderr, "usage: %s --qwen4 | --glm | --glm-timeout | --glm-split | --slabs\n",
                argv[0]);
        return 1;
    }
    int ok = setup();
    if (ok && !strcmp(mode, "--qwen4")) ok = qwen4_suite();
    if (ok && !strcmp(mode, "--glm")) ok = glm_suite(0);
    if (ok && !strcmp(mode, "--glm-timeout")) ok = check_timeout();
    if (ok && !strcmp(mode, "--glm-split")) ok = glm_suite(1);
    if (ok && !strcmp(mode, "--slabs")) ok = slab_suite();
    ds4_gpu_stream_gate_test_set_mode(-1, -1);
    ds4_gpu_cleanup();
    if (model && model != MAP_FAILED) munmap(model, model_bytes);
    if (model_file) fclose(model_file);
    fprintf(stderr, "Metal stream gate %s: %s\n", mode + 2, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
