/* Model-free attention micro-benchmark for Ornith's shape (16 query heads,
 * 2 KV heads, head dim 256, F16 K/V).  Times each attention kernel at long
 * key counts and reports the K/V bytes it must read against a bandwidth
 * floor, so tile parameters can be chosen by measurement.  Needs the GPU but
 * no model; run it with the live stack paused for stable numbers. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4.h"
#include "ds4_gpu.h"

/* Stub for the symbol ds4_metal.o expects from ds4.o, which this
 * GPU-kernel-only test does not link (same pattern as tests/test_qwen35_kernels.c). */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum { H = 16, HKV = 2, D = 256 };
static const double FLOOR_GBS = 250.0;   /* assumed sustained read for the M5 Pro (~273 GB/s peak); the ratio is indicative */

static uint32_t g_rng = 0x2545f491u;
static float frand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xffffffu) / 16777216.0f * 2.0f - 1.0f;
}
static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t s = (v.u >> 16) & 0x8000u;
    int e = (int)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t m = v.u & 0x7fffffu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_attn: %s failed\n", what); exit(1); }
}
static ds4_gpu_tensor *rand_f32(uint64_t n, float scale) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = frand() * scale;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "f32 upload");
    free(h);
    return t;
}
static ds4_gpu_tensor *rand_f16(uint64_t n) {
    uint16_t *h = malloc(n * sizeof(uint16_t));
    for (uint64_t i = 0; i < n; i++) h[i] = f32_to_f16(frand());
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(uint16_t));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(uint16_t)), "f16 upload");
    free(h);
    return t;
}

/* median of `reps` timed calls after `warm` warm-up calls; each call ends in a
 * synchronize so the wall time is the kernel time plus one submit */
typedef int (*bench_fn)(void *ctx);
static double time_ms(bench_fn fn, void *ctx, int warm, int reps) {
    double t[16];
    for (int i = 0; i < warm; i++) need(fn(ctx) && ds4_gpu_synchronize(), "warm-up call");
    for (int i = 0; i < reps; i++) {
        const double t0 = now_s();
        need(fn(ctx) && ds4_gpu_synchronize(), "timed call");
        t[i] = (now_s() - t0) * 1e3;
    }
    for (int i = 1; i < reps; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[reps / 2];
}

typedef struct {
    ds4_gpu_tensor *q, *gate, *out, *part, *part3, *part_flash, *kc, *vc;
    uint32_t pos0, T, rows;
    float scale;
} bench_ctx;

static int run_prefill_mm(void *p) {       /* today's prefill: kernel_qwen4_attn_mm via the qwen4 wrapper */
    bench_ctx *c = p;
    return ds4_gpu_qwen4_attn_decode_tensor(c->out, c->q, c->gate, c->kc, c->vc, NULL, NULL, NULL,
                                            c->T, H, HKV, D, c->pos0, 0u, 0u, c->scale,
                                            NULL, NULL, NULL, NULL, 0u);
}
static int run_flash(void *p) {            /* M5 successor for T>8 prefill: flash (query-token tiles) */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_tensor(c->out, c->q, c->gate, c->kc, c->vc, NULL,
                                            c->T, H, HKV, D, c->pos0, c->scale);
}
static int run_flash_split(void *p) {      /* same, with the key split engaged via `part` */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part_flash,
                                            c->T, H, HKV, D, c->pos0, c->scale);
}
static int run_flash_nax(void *p) {        /* M6: flash on the neural accelerators (tensor API) */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_nax_tensor(c->out, c->q, c->gate, c->kc, c->vc, NULL,
                                                c->T, H, HKV, D, c->pos0, c->scale);
}
static int run_flash_nax_split(void *p) {  /* same, with the key split engaged via `part` */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_flash_nax_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part_flash,
                                                c->T, H, HKV, D, c->pos0, c->scale);
}
static int run_decode2(void *p) {          /* today's decode / verify: L12 */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_decode2_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part,
                                              H, HKV, D, c->pos0, c->rows, c->scale, NULL, NULL, NULL, NULL, 0u);
}
static int run_decode3(void *p) {          /* M5 successor: decode3 */
    bench_ctx *c = p;
    return ds4_gpu_qwen35_attn_decode3_tensor(c->out, c->q, c->gate, c->kc, c->vc, c->part3,
                                              H, HKV, D, c->pos0, c->rows, c->scale);
}

/* K/V bytes a kernel must read at least once: every key <= the last query's
 * position, K and V, 2 KV heads x 256 halves each */
static double kv_gb(uint32_t pos0, uint32_t T) {
    return (double)(pos0 + T) * HKV * D * 2.0 * 2.0 / 1e9;
}
static void report(const char *kernel, const char *mode, uint32_t pos0, uint32_t T, uint32_t rows, double ms) {
    const double gb = kv_gb(pos0, T);
    printf("bench %-14s %-7s pos=%-6u T=%-4u rows=%u: %8.3f ms, K/V read %.3f GB -> %6.1f GB/s (floor %.3f ms)\n",
           kernel, mode, pos0, T, rows, ms, gb, gb / (ms * 1e-3), gb / FLOOR_GBS * 1e3);
}

/* Puts DS4_QWEN35_ATTN_FLASH_MIN_TG back to what the caller had before this
 * bench started forcing its own values, so DS4_QWEN35_ATTN_FLASH_MIN_TG=64
 * ./tests/bench_qwen35_attn applies to every default-rule row and not just
 * the ones this file never overrides. */
static void restore_min_tg(int had_value, const char *saved) {
    if (had_value) setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", saved, 1);
    else unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
}

int main(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    const int do_prefill = !strcmp(what, "all") || !strcmp(what, "prefill");
    const int do_decode = !strcmp(what, "all") || !strcmp(what, "decode");
    need(ds4_gpu_init(), "GPU initialization");
    const char *min_tg_env = getenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
    const int have_min_tg = min_tg_env != NULL;
    char min_tg_saved[32] = {0};
    if (have_min_tg) strncpy(min_tg_saved, min_tg_env, sizeof(min_tg_saved) - 1);
    const uint32_t cap = 131072u + 2048u + 8u;
    bench_ctx c = {0};
    c.scale = 1.0f / sqrtf((float)D);
    c.kc = rand_f16((uint64_t)cap * HKV * D);
    c.vc = rand_f16((uint64_t)cap * HKV * D);
    c.q = rand_f32(2048ull * H * D, 1.0f);
    c.gate = rand_f32(2048ull * H * D, 1.0f);
    c.out = ds4_gpu_tensor_alloc(2048ull * H * D * sizeof(float));
    c.part = ds4_gpu_tensor_alloc(ds4_gpu_qwen4_attn_part_floats(2048u, H, D) * sizeof(float));
    c.part3 = ds4_gpu_tensor_alloc(ds4_gpu_qwen35_attn_part3_floats(2u, H, D) * sizeof(float));
    {
        /* Size for the largest MIN_TG this run can ever see: the caller's own
         * value (if set, e.g. DS4_QWEN35_ATTN_FLASH_MIN_TG=64 ./tests/bench...),
         * the 4096 this bench forces for the T=2048 split case below, and the
         * unset default (256) the default-rule rows use -- each checked at
         * every chunk size benched (T = 16..256 and 2048). */
        uint64_t pf_max = 0;
        const char *tg_settings[3];
        int n_settings = 0;
        if (have_min_tg) tg_settings[n_settings++] = min_tg_saved;
        tg_settings[n_settings++] = "4096";
        tg_settings[n_settings++] = NULL;   /* unset: the function's own default (256) */
        for (int i = 0; i < n_settings; i++) {
            if (tg_settings[i]) setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", tg_settings[i], 1);
            else unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
            const uint32_t benched_t[6] = { 16u, 32u, 64u, 128u, 256u, 2048u };
            for (int j = 0; j < 6; j++) {
                const uint64_t pf = ds4_gpu_qwen35_attn_flash_part_floats(benched_t[j], H, D);
                if (pf > pf_max) pf_max = pf;
            }
        }
        restore_min_tg(have_min_tg, min_tg_saved);
        c.part_flash = ds4_gpu_tensor_alloc(pf_max * sizeof(float));
    }
    need(c.out && c.part && c.part3 && c.part_flash, "output buffers");

    if (do_prefill) {
        const uint32_t pos[3] = { 0u, 30720u, 122880u };
        for (int i = 0; i < 3; i++) {
            c.pos0 = pos[i]; c.T = 2048u; c.rows = 0;
            report("qwen4_attn_mm", "prefill", c.pos0, c.T, 0, time_ms(run_prefill_mm, &c, 1, 3));
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "2", 1);
            report("attn_flash_tok2", "prefill", c.pos0, c.T, 0, time_ms(run_flash, &c, 1, 3));
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "4", 1);
            report("attn_flash_tok4", "prefill", c.pos0, c.T, 0, time_ms(run_flash, &c, 1, 3));
            if (ds4_gpu_tensor_api_available())
                report("attn_flash_nax", "prefill", c.pos0, c.T, 0, time_ms(run_flash_nax, &c, 1, 3));
        }
        unsetenv("DS4_QWEN35_ATTN_FLASH_TOK");

        /* key split at long context, T=2048 forced into the split path
         * (a 2048-token chunk already gives plenty of threadgroups, so the
         * split rule naturally picks Ks=1 here -- MIN_TG=4096 forces a split
         * anyway, to see its overhead/benefit at the chunk size this project
         * actually runs prefill at), and T=128 short chunks at the same long
         * positions, where the split rule engages on its own (default
         * min_tg). */
        const uint32_t split_pos[2] = { 30720u, 122880u };
        for (int i = 0; i < 2; i++) {
            c.pos0 = split_pos[i]; c.T = 2048u; c.rows = 0;
            setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", "4096", 1);
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "2", 1);
            report("flash_tok2_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "4", 1);
            report("flash_tok4_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
            restore_min_tg(have_min_tg, min_tg_saved);
        }
        unsetenv("DS4_QWEN35_ATTN_FLASH_TOK");

        for (int i = 0; i < 2; i++) {
            c.pos0 = split_pos[i]; c.T = 128u; c.rows = 0;
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "2", 1);
            report("flash_tok2", "prefill", c.pos0, c.T, 0, time_ms(run_flash, &c, 1, 3));
            report("flash_tok2_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "4", 1);
            report("flash_tok4", "prefill", c.pos0, c.T, 0, time_ms(run_flash, &c, 1, 3));
            report("flash_tok4_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
        }
        unsetenv("DS4_QWEN35_ATTN_FLASH_TOK");
        if (ds4_gpu_tensor_api_available()) {
            /* short chunks at long context, default split rule: where the
             * accelerator kernel starts to beat the simdgroup flash */
            const uint32_t ts[5] = { 16u, 32u, 64u, 128u, 256u };
            setenv("DS4_QWEN35_ATTN_FLASH_TOK", "2", 1);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 5; j++) {
                    c.pos0 = split_pos[i]; c.T = ts[j]; c.rows = 0;
                    report("flash_tok2_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_split, &c, 1, 3));
                    report("flash_nax_split", "prefill", c.pos0, c.T, 0, time_ms(run_flash_nax_split, &c, 1, 3));
                }
            unsetenv("DS4_QWEN35_ATTN_FLASH_TOK");
        }
    }
    if (do_decode) {
        const uint32_t pos[3] = { 2048u, 32768u, 131072u };
        for (int i = 0; i < 3; i++)
            for (uint32_t rows = 1; rows <= 2; rows++) {
                c.pos0 = pos[i]; c.T = rows; c.rows = rows;
                report("decode2", "decode", c.pos0, c.T, rows, time_ms(run_decode2, &c, 3, 9));
                report("decode3", "decode", c.pos0, c.T, rows, time_ms(run_decode3, &c, 3, 9));
            }
    }
    ds4_gpu_tensor_free(c.kc); ds4_gpu_tensor_free(c.vc); ds4_gpu_tensor_free(c.q);
    ds4_gpu_tensor_free(c.gate); ds4_gpu_tensor_free(c.out); ds4_gpu_tensor_free(c.part);
    ds4_gpu_tensor_free(c.part3); ds4_gpu_tensor_free(c.part_flash);
    return 0;
}
