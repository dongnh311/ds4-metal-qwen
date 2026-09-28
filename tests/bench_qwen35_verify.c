/* Model-free micro-benchmark for the M7 verify (Ornith's Q8_0 projection
 * shapes): one T=1 matvec, two T=1 matvecs (today's per-row verify), the
 * existing exact two-row dispatch (ds4_gpu_matmul_q8_0_decode_rows_exact_tensor)
 * and, from M7 Task 2, the two-row kernel.  Every timed call reads a
 * different copy of the weights (>= 256 MB per shape, above the system
 * cache): a decode step streams ~2.7 GB and never finds its weights cached.
 * Needs the GPU but no model; run it with the live stack paused. */
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
 * GPU-kernel-only bench does not link (same pattern as tests/bench_qwen35_attn.c). */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

typedef struct { const char *name; uint32_t in_dim, out_dim; } shape_t;
static const shape_t SHAPES[] = {
    { "2048x8192(lin_qkv,attn_q)", 2048u, 8192u },
    { "2048x4096(lin_gate)", 2048u, 4096u },
    { "4096x2048(lin_out,attn_output)", 4096u, 2048u },
    { "2048x248320(lm_head)", 2048u, 248320u },
};
enum { N_SHAPES = 4, CALLS = 32, REPS = 5, MAX_COPIES = 64 };
static const uint64_t COLD_BYTES = 256ull << 20;

typedef enum { M_T1, M_T1X2, M_ROWS_EXACT, N_METHODS } method_t;
static const char *METHOD_NAMES[N_METHODS] = { "t1", "t1x2", "rows_exact" };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_verify: %s failed\n", what); exit(1); }
}

static uint32_t g_rng = 0x6c8e9cf5u;
static uint32_t next_rng(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t off[MAX_COPIES];
    uint32_t copies, in_dim, out_dim;
    uint64_t bytes;
    ds4_gpu_tensor *x, *x0, *x1, *o, *o0, *o1;
} ctx_t;

static uint32_t copies_for(uint64_t bytes) {
    uint64_t c = (COLD_BYTES + bytes - 1u) / bytes;
    if (c < 2u) c = 2u;
    return c > MAX_COPIES ? MAX_COPIES : (uint32_t)c;
}

/* Q8_0 blocks: scale 2^-7 (half 0x2000), quants uniform in [-64, 63] */
static void fill_q8(uint8_t *p, uint64_t bytes) {
    for (uint64_t b = 0; b + 34u <= bytes; b += 34u) {
        p[b] = 0x00; p[b + 1] = 0x20;
        for (int j = 0; j < 32; j++) p[b + 2 + j] = (uint8_t)((next_rng() >> 8) & 0x7fu) - 64u;
    }
}

static int call(ctx_t *c, method_t m, uint32_t k) {
    const uint64_t off = c->off[k % c->copies];
    switch (m) {
    case M_T1:
        return ds4_gpu_qwen4_matmul_q8_0_tensor(c->o0, c->base, c->size, off, c->in_dim, c->out_dim, c->x0, 1u);
    case M_T1X2:
        return ds4_gpu_qwen4_matmul_q8_0_tensor(c->o0, c->base, c->size, off, c->in_dim, c->out_dim, c->x0, 1u) &&
               ds4_gpu_qwen4_matmul_q8_0_tensor(c->o1, c->base, c->size, off, c->in_dim, c->out_dim, c->x1, 1u);
    case M_ROWS_EXACT:
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(c->o, c->base, c->size, off, c->in_dim, c->out_dim,
                                                            c->x, 2u);
    default:
        return 0;
    }
}

/* median over REPS of (CALLS calls encoded into one command buffer) / CALLS,
 * after one untimed repetition */
static double time_ms(ctx_t *c, method_t m) {
    double t[REPS];
    for (int r = -1; r < REPS; r++) {
        const double t0 = now_s();
        need(ds4_gpu_begin_commands(), "begin commands");
        for (uint32_t k = 0; k < CALLS; k++) need(call(c, m, k), METHOD_NAMES[m]);
        need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
        if (r >= 0) t[r] = (now_s() - t0) * 1e3 / CALLS;
    }
    for (int i = 1; i < REPS; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[REPS / 2];
}

static ds4_gpu_tensor *rand_rows(uint64_t n) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = (float)((int)(next_rng() & 0xffffu) - 32768) / 32768.0f;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "input upload");
    free(h);
    return t;
}

int main(void) {
    uint64_t total = 0;
    for (int s = 0; s < N_SHAPES; s++) {
        const uint64_t bytes = (uint64_t)SHAPES[s].out_dim * (SHAPES[s].in_dim / 32u) * 34u;
        total += ((bytes + 63u) & ~63ull) * copies_for(bytes);
    }
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    need(base != MAP_FAILED, "weight arena mmap");
    need(ds4_gpu_init(), "GPU initialization");
    need(ds4_gpu_set_model_map(base, total), "model map registration");
    uint64_t used = 0;
    double ratio_exact[N_SHAPES];
    for (int s = 0; s < N_SHAPES; s++) {
        ctx_t c = { .base = base, .size = total, .in_dim = SHAPES[s].in_dim, .out_dim = SHAPES[s].out_dim };
        c.bytes = (uint64_t)c.out_dim * (c.in_dim / 32u) * 34u;
        c.copies = copies_for(c.bytes);
        for (uint32_t k = 0; k < c.copies; k++) {
            c.off[k] = used;
            fill_q8(base + used, c.bytes);
            used += (c.bytes + 63u) & ~63ull;
        }
        c.x = rand_rows(2ull * c.in_dim);
        c.x0 = ds4_gpu_tensor_view(c.x, 0, (uint64_t)c.in_dim * sizeof(float));
        c.x1 = ds4_gpu_tensor_view(c.x, (uint64_t)c.in_dim * sizeof(float), (uint64_t)c.in_dim * sizeof(float));
        c.o = ds4_gpu_tensor_alloc(2ull * c.out_dim * sizeof(float));
        c.o0 = ds4_gpu_tensor_view(c.o, 0, (uint64_t)c.out_dim * sizeof(float));
        c.o1 = ds4_gpu_tensor_view(c.o, (uint64_t)c.out_dim * sizeof(float), (uint64_t)c.out_dim * sizeof(float));
        need(c.x0 && c.x1 && c.o && c.o0 && c.o1, "tensor views");
        double t1 = 0.0;
        for (int m = 0; m < N_METHODS; m++) {
            const double ms = time_ms(&c, (method_t)m);
            if (m == M_T1) t1 = ms;
            if (m == M_ROWS_EXACT) ratio_exact[s] = ms / t1;
            printf("bench: %-32s %-10s %8.4f ms  x%.2f  %6.1f GB/s\n", SHAPES[s].name, METHOD_NAMES[m], ms,
                   ms / t1, (double)c.bytes / (ms * 1e6));
        }
        ds4_gpu_tensor_free(c.x0); ds4_gpu_tensor_free(c.x1); ds4_gpu_tensor_free(c.x);
        ds4_gpu_tensor_free(c.o0); ds4_gpu_tensor_free(c.o1); ds4_gpu_tensor_free(c.o);
    }
    bool use = true;
    for (int s = 0; s < N_SHAPES; s++) use = use && ratio_exact[s] <= 1.2;
    printf("bench: verdict decode_rows_exact %s\n", use ? "use" : "build-kernel");
    return 0;
}
