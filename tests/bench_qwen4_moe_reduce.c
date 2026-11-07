/* Standalone MoE reduce bench: times ds4_gpu_qwen4_moe_reduce_tensor at
 * Qwen3.8-Flash-Next prefill shapes (T=1856, dim=2560, 10 slots + shared
 * slot, hc=4) on synthetic data.  Timing only; correctness lives in
 * test_qwen4_kernels / test_qwen35_kernels.
 *
 * usage: bench_qwen4_moe_reduce [--tokens N] [--dim N] [--slots N] [--iters N]
 *        [--hc N] [--shared 0|1|2]
 * env: DS4_QWEN4_MOE_REDUCE_SCALAR=1 forces the scalar kernel (A/B). */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static uint32_t rng = 999;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static float random_f32(void) { return ((int)(random_u32() % 2049) - 1024) / 512.0f; }

int main(int argc, char **argv) {
    uint32_t T = 1856, dim = 2560, S = 10, iters = 20, hc = 4, shared = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--tokens")) T = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--dim")) dim = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--slots")) S = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--hc")) hc = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--shared")) shared = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!ds4_gpu_init()) { fprintf(stderr, "gpu init failed\n"); return 1; }
    const uint32_t stride = S + 1;   /* part holds the shared slot too */
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc((uint64_t)T * stride * dim * sizeof(float));
    ds4_gpu_tensor *w = ds4_gpu_tensor_alloc((uint64_t)T * S * sizeof(float));
    ds4_gpu_tensor *sg = ds4_gpu_tensor_alloc(T * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)T * dim * sizeof(float));
    ds4_gpu_tensor *R = hc ? ds4_gpu_tensor_alloc((uint64_t)T * dim * hc * sizeof(float)) : NULL;
    ds4_gpu_tensor *inj = hc ? ds4_gpu_tensor_alloc((uint64_t)T * hc * 8 * hc * sizeof(float)) : NULL;
    ds4_gpu_tensor *sh = (shared == 2) ? ds4_gpu_tensor_alloc((uint64_t)T * dim * sizeof(float)) : NULL;
    float *hp = malloc((uint64_t)T * stride * dim * sizeof(float));
    float *hw = malloc((uint64_t)T * S * sizeof(float));
    float *hs = malloc((uint64_t)T * dim * sizeof(float));
    float *hg = malloc(T * sizeof(float));
    float *hi = hc ? malloc((uint64_t)T * hc * 8 * hc * sizeof(float)) : NULL;
    int ok = part && w && sg && out && (!hc || (R && inj)) && (!sh || sh) && hp && hw && hs && hg && (!hc || hi);
    for (uint64_t i = 0; ok && i < (uint64_t)T * stride * dim; i++) hp[i] = random_f32();
    for (uint64_t i = 0; ok && i < (uint64_t)T * S; i++) hw[i] = random_f32();
    for (uint64_t i = 0; ok && i < (uint64_t)T * dim; i++) hs[i] = random_f32();
    for (uint32_t i = 0; ok && i < T; i++) hg[i] = random_f32();
    if (hi) for (uint64_t i = 0; ok && i < (uint64_t)T * hc * 8 * hc; i++) hi[i] = random_f32();
    ok = ok && ds4_gpu_tensor_write(part, 0, hp, (uint64_t)T * stride * dim * sizeof(float)) &&
         ds4_gpu_tensor_write(w, 0, hw, (uint64_t)T * S * sizeof(float)) &&
         ds4_gpu_tensor_write(sg, 0, hg, (uint64_t)T * sizeof(float)) &&
         ds4_gpu_tensor_write(sh ? sh : out, 0, hs, (uint64_t)T * dim * sizeof(float)) &&
         (!inj || ds4_gpu_tensor_write(inj, 0, hi, (uint64_t)T * hc * 8 * hc * sizeof(float)));

    double ms = 0;
    int good = 0;
    for (uint32_t it = 0; it < iters + 2 && ok; it++) {
        double a = now_ms();
        ok = ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_moe_reduce_tensor(out, part, w, sg, sh, R, inj, T, S, stride, dim, hc) &&
             ds4_gpu_end_commands();
        double b = now_ms();
        if (!ok) break;
        if (it >= 2) { ms += b - a; good++; }
    }
    if (good)
        printf("moe_reduce T=%u dim=%u S=%u hc=%u shared=%u iters=%u: %.3f ms\n",
               T, dim, S, hc, shared, good, ms / good);
    else
        fprintf(stderr, "bench failed (ok=%d)\n", ok);

    free(hp); free(hw); free(hs); free(hg); free(hi);
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(sg);
    ds4_gpu_tensor_free(out); if (R) ds4_gpu_tensor_free(R); if (inj) ds4_gpu_tensor_free(inj);
    if (sh) ds4_gpu_tensor_free(sh);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
