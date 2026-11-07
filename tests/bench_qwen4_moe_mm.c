/* Standalone MoE prefill GEMM bench: times kernel_qwen4_moe_mm_mid/down at
 * Qwen3.8-Flash-Next shapes (E=512 experts, 10 used slots, embd 2560,
 * ff 640) on synthetic packed weights in a fake model map, so kernel A/B
 * runs do not need the 50 GB GGUF.  Correctness lives in
 * test_qwen4_moe_mm_specialize; this file only times.
 *
 * usage: bench_qwen4_moe_mm [--mid-type T] [--down-type T] [--tokens N]
 *        [--iters N] [--experts N] [--slots N]
 * env knobs are the production ones (DS4_QWEN4_MOE_MM_NAX, *_NT, ...). */
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

static uint32_t rng = 12345;
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}
static uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/* row bytes mirror ds4_metal.m's qwen4_expert_row_bytes/qwen4_down_row_bytes
 * for the types this bench supports (all %32 or %64 aligned dims). */
static uint32_t row_bytes(uint32_t type, uint32_t in_dim) {
    switch (type) {
    case 8:  return (in_dim / 32u) * 34u;
    case 16: return (in_dim / 256u) * 66u;
    case 22: return (in_dim / 256u) * 82u;
    case 17: return (in_dim / 256u) * 74u;
    case 20: return (in_dim / 32u) * 18u;
    case 42: return (in_dim / 64u) * 18u;
    case 2:  return (in_dim / 32u) * 18u;
    case 12: return (in_dim / 256u) * 144u;
    case 10: return ((in_dim + 255u) / 256u) * 84u;
    case 39: return (in_dim / 32u) * 17u;
    default: return 0;
    }
}

int main(int argc, char **argv) {
    uint32_t mid_type = 22, down_type = 20;   /* UD-IQ2_XS layer-0 types */
    uint32_t T = 1856, iters = 10, E = 512, S = 10;
    const uint32_t D = 2560, F = 640, O = 2560;   /* embd, ff_exp, out */
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--mid-type")) mid_type = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--down-type")) down_type = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--tokens")) T = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--experts")) E = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else if (!strcmp(argv[i], "--slots")) S = (uint32_t)strtoul(argv[i + 1], 0, 0);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    const uint32_t mid_rb = row_bytes(mid_type, D), down_rb = row_bytes(down_type, F);
    if (!mid_rb || !down_rb || (D % 64) || (F % 64)) {
        fprintf(stderr, "bad types/dims: mid_rb=%u down_rb=%u\n", mid_rb, down_rb);
        return 2;
    }
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t gu = (uint64_t)E * F * mid_rb;          /* one of gate/up */
    const uint64_t dw = (uint64_t)E * O * down_rb;
    const uint64_t bytes = align_up(align_up(align_up(gu, page) + gu, page) + dw, page);
    void *map = NULL;
    if (posix_memalign(&map, page, bytes) != 0) return 1;
    memset(map, 0, bytes);
    /* random-ish packed bytes; timing does not depend on values */
    uint8_t *p = map;
    for (uint64_t i = 0; i < bytes; i += 4096) {
        uint32_t w = random_u32();
        for (uint32_t j = 0; j < 4096 && i + j < bytes; j++) p[i + j] = (uint8_t)(w >> ((j & 3) * 8));
    }
    const uint64_t gate_off = 0, up_off = align_up(gu, page), down_off = align_up(up_off + gu, page);

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(map, bytes)) {
        fprintf(stderr, "gpu init/map failed\n");
        return 1;
    }
    const uint32_t list_cap = T;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)T * D * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc((uint64_t)T * S * sizeof(int32_t));
    ds4_gpu_tensor *lists = ds4_gpu_tensor_alloc((uint64_t)E * list_cap * sizeof(int32_t));
    ds4_gpu_tensor *counts = ds4_gpu_tensor_alloc(E * sizeof(int32_t));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)T * S * F * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc((uint64_t)T * S * O * sizeof(float));
    float *hx = malloc((uint64_t)T * D * sizeof(float));
    int32_t *hid = malloc((uint64_t)T * S * sizeof(int32_t));
    int ok = x && ids && lists && counts && mid && part && hx && hid;
    for (uint64_t i = 0; ok && i < (uint64_t)T * D; i++)
        hx[i] = ((int)(random_u32() % 257) - 128) / 1024.0f;
    for (uint64_t i = 0; ok && i < (uint64_t)T * S; i++) hid[i] = (int32_t)(random_u32() % E);
    ok = ok && ds4_gpu_tensor_write(x, 0, hx, (uint64_t)T * D * sizeof(float)) &&
         ds4_gpu_tensor_write(ids, 0, hid, (uint64_t)T * S * sizeof(int32_t)) &&
         ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, ids, T, S, E, list_cap);

    double mid_ms = 0, down_ms = 0;
    int good = 0;
    for (uint32_t it = 0; it < iters + 2 && ok; it++) {
        double t0 = now_ms(), t1, t2;
        ok = ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, map, bytes, gate_off, up_off,
                                             mid_type, E, T, S, S, D, F, list_cap);
        t1 = now_ms();
        ok = ok && ds4_gpu_qwen4_moe_mm_down_tensor(part, mid, lists, counts, map, bytes, down_off,
                                                    down_type, E, T, S, S, F, O, list_cap);
        t2 = now_ms();
        ok = ok && ds4_gpu_end_commands();
        if (!ok) break;
        /* end_commands waits for the whole batch; t1/t2 bracket CPU submit
         * times, not GPU stage times, so time each stage in its own batch. */
        (void)t0; (void)t1; (void)t2;
        good++;
        if (it >= 2) {
            /* re-time with one stage per command batch for a clean split */
        }
    }
    /* Clean per-stage timing: separate command batches. */
    mid_ms = down_ms = 0;
    good = 0;
    for (uint32_t it = 0; it < iters + 2 && ok; it++) {
        double a = now_ms();
        ok = ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts, map, bytes, gate_off, up_off,
                                             mid_type, E, T, S, S, D, F, list_cap) &&
             ds4_gpu_end_commands();
        double b = now_ms();
        ok = ok && ds4_gpu_begin_commands() &&
             ds4_gpu_qwen4_moe_mm_down_tensor(part, mid, lists, counts, map, bytes, down_off,
                                              down_type, E, T, S, S, F, O, list_cap) &&
             ds4_gpu_end_commands();
        double c = now_ms();
        if (!ok) break;
        if (it >= 2) { mid_ms += b - a; down_ms += c - b; }
        good++;
    }
    if (good > 2)
        printf("moe_mm mid_type=%u down_type=%u T=%u E=%u S=%u iters=%u: mid=%.2f ms down=%.2f ms total=%.2f ms\n",
               mid_type, down_type, T, E, S, iters, mid_ms / iters, down_ms / iters,
               (mid_ms + down_ms) / iters);
    else
        fprintf(stderr, "bench failed (ok=%d)\n", ok);

    free(hx); free(hid);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(lists);
    ds4_gpu_tensor_free(counts); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(part);
    ds4_gpu_cleanup(); free(map);
    return ok ? 0 : 1;
}
