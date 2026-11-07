/* Standalone MoE router top-k bench: times kernel_qwen4_router_topk at
 * Qwen3.8-Flash-Next prefill shapes (T tokens, 512 experts, top-10,
 * in_dim 2560, F32 shared-gate row).  Timing only.
 *
 * usage: bench_qwen4_router [--tokens N] [--experts N] [--used N]
 *        [--dim N] [--iters N] [--gate 0|1] */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static uint32_t rng = 4242;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float random_f32(void) { return ((int)(random_u32() % 2049) - 1024) / 512.0f; }

int main(int argc, char **argv) {
    uint32_t T = 1928, NE = 512, NU = 10, dim = 2560, iters = 30, gate = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--tokens")) T = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--experts")) NE = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--used")) NU = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--dim")) dim = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--gate")) gate = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t gate_bytes = (uint64_t)dim * 4u;
    void *map = NULL;
    if (posix_memalign(&map, page, gate_bytes) != 0) return 1;
    float *hg = (float *)map;
    for (uint64_t i = 0; i < (uint64_t)dim; i++) hg[i] = random_f32();

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(map, gate_bytes)) {
        fprintf(stderr, "gpu init/map failed\n");
        return 1;
    }
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)T * NE * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)T * NU * sizeof(int32_t));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc((uint64_t)T * NU * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)T * dim * sizeof(float));
    ds4_gpu_tensor *sg = ds4_gpu_tensor_alloc(T * sizeof(float));
    float *hl = malloc((uint64_t)T * NE * sizeof(float));
    float *hx = malloc((uint64_t)T * dim * sizeof(float));
    float *hs = malloc(T * sizeof(float));
    int ok = logits && sel && w && x && sg && hl && hx && hs;
    for (uint64_t i = 0; ok && i < (uint64_t)T * NE; i++) hl[i] = random_f32() * 4.0f;
    for (uint64_t i = 0; ok && i < (uint64_t)T * dim; i++) hx[i] = random_f32();
    for (uint32_t i = 0; ok && i < T; i++) hs[i] = random_f32();
    ok = ok && ds4_gpu_tensor_write(logits, 0, hl, (uint64_t)T * NE * sizeof(float)) &&
         ds4_gpu_tensor_write(x, 0, hx, (uint64_t)T * dim * sizeof(float)) &&
         ds4_gpu_tensor_write(sg, 0, hs, (uint64_t)T * sizeof(float));

    double ms = 0;
    int good = 0;
    for (uint32_t it = 0; it < iters + 2 && ok; it++) {
        double a = now_ms();
        ok = ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_router_topk_tensor(sel, w, logits, x, map, gate_bytes,
                                              0u, 0u, gate ? dim : 0u, sg, T, NE, NU) &&
             ds4_gpu_end_commands();
        double b = now_ms();
        if (!ok) break;
        if (it >= 2) { ms += b - a; good++; }
    }
    if (good)
        printf("router_topk T=%u NE=%u NU=%u dim=%u gate=%u iters=%u: %.3f ms\n",
               T, NE, NU, dim, gate, good, ms / good);
    else
        fprintf(stderr, "bench failed (ok=%d)\n", ok);

    free(hl); free(hx); free(hs);
    ds4_gpu_tensor_free(logits); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(sg);
    ds4_gpu_cleanup(); free(map);
    return ok ? 0 : 1;
}
