/* GPU kernel tests for Ornith-1.5-35B-A3B (qwen35moe): the kernels in
 * metal/qwen35.metal against double-precision references.
 * Build: make test-qwen35-kernels */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x9e3779b9u;

static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "%s: non-finite value at %llu\n", what, (unsigned long long)i);
            exit(1);
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "%s: max|d| %.3e (rel %.3e) at %llu: got %.6f ref %.6f\n",
                what, worst, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        exit(1);
    }
    printf("  %-44s ok  max|d|=%.2e (scale %.2e)\n", what, worst, scale);
}

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        exp = 127 - 15 + 1;
        while (!(mant & 0x400u)) { mant <<= 1; exp--; }
        mant &= 0x3ffu;
        v.u = sign | (exp << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31) { v.u = sign | 0x7f800000u | (mant << 13); return v.f; }
    v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return v.f;
}

static double sigmoid_d(double x) { return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x)); }
static double silu_d(double x) { return x * sigmoid_d(x); }

/* ---- weight arena ---- */

/* anonymous mmap standing in for the model map; weights are appended, never freed */
typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t used;
} arena_t;

static uint64_t arena_alloc(arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->used + 63u) & ~63ull;
    if (off + bytes > a->size) {
        fprintf(stderr, "arena exhausted\n");
        exit(1);
    }
    a->used = off + bytes;
    return off;
}

/* f32 weights with a double shadow for the reference */
static uint64_t arena_f32(arena_t *a, uint64_t n, double **shadow, float lo, float hi) {
    const uint64_t off = arena_alloc(a, n * sizeof(float));
    float *w = (float *)(a->base + off);
    *shadow = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
        w[i] = lo + (hi - lo) * (0.5f * frand() + 0.5f);
        (*shadow)[i] = w[i];
    }
    return off;
}

/* q8_0 rows of `cols` elements (cols % 32 == 0), `rows` rows */
static uint64_t arena_q8_0(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 32;
    const uint64_t off = arena_alloc(a, rows * blocks * 34u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = scale * frand();
                if (fabsf(vals[j]) > amax) amax = fabsf(vals[j]);
            }
            const float d = amax / 127.0f;
            const uint16_t dh = f32_to_f16(d);
            const float dq = f16_to_f32(dh);
            uint8_t *blk = w + (r * blocks + b) * 34u;
            memcpy(blk, &dh, 2);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(d > 0 ? vals[j] / d : 0.0f);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                ((int8_t *)blk)[2 + j] = (int8_t)q;
                (*shadow)[r * cols + b * 32 + j] = (double)dq * q;
            }
        }
    }
    return off;
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    else require_ok(ds4_gpu_tensor_fill_f32(t, 0.0f, n), "tensor fill");
    return t;
}

static float *download(const ds4_gpu_tensor *t, uint64_t n) {
    float *out = malloc(n * sizeof(float));
    require_ok(ds4_gpu_tensor_read(t, 0, out, n * sizeof(float)), "tensor read");
    return out;
}

static float *rand_vec(uint64_t n, float scale) {
    float *v = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) v[i] = scale * frand();
    return v;
}

/* q5_K rows: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit
 * scale/min bytes, 32 high-bit bytes, 128 nibble bytes; llama.cpp layout).
 * Element j of 32-group g stores its low nibble in qs[(g/2)*32 + j] (low
 * half for even g) and its fifth bit in bit g of qh[j].  The shadow holds the
 * dequantized weights. */
static uint64_t arena_q5_K(arena_t *a, uint64_t rows, uint64_t cols, double **shadow, float scale) {
    const uint64_t blocks = cols / 256;
    const uint64_t off = arena_alloc(a, rows * blocks * 176u);
    uint8_t *w = a->base + off;
    *shadow = malloc(rows * cols * sizeof(double));
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t b = 0; b < blocks; b++) {
            uint8_t *blk = w + (r * blocks + b) * 176u;
            const float d = scale / 63.0f / 31.0f, dmin = d;
            const uint16_t dh = f32_to_f16(d), mh = f32_to_f16(dmin);
            const float dq = f16_to_f32(dh), mq = f16_to_f32(mh);
            memcpy(blk, &dh, 2);
            memcpy(blk + 2, &mh, 2);
            uint8_t sc[8], mn[8];
            for (int g = 0; g < 8; g++) {
                sc[g] = (uint8_t)(1 + (int)(62.0f * (0.5f * frand() + 0.5f)));
                mn[g] = (uint8_t)((int)(63.0f * (0.5f * frand() + 0.5f)));
            }
            uint8_t *s = blk + 4;
            memset(s, 0, 12);
            for (int g = 0; g < 4; g++) { s[g] = sc[g] & 63; s[g + 4] = mn[g] & 63; }
            for (int g = 4; g < 8; g++) {
                s[g + 4] = (uint8_t)((sc[g] & 0xF) | ((mn[g] & 0xF) << 4));
                s[g - 4] |= (uint8_t)((sc[g] >> 4) << 6);
                s[g] |= (uint8_t)((mn[g] >> 4) << 6);
            }
            memset(blk + 16, 0, 160);
            for (int g = 0; g < 8; g++) {
                for (int j = 0; j < 32; j++) {
                    const int q = (int)(31.0f * (0.5f * frand() + 0.5f));
                    blk[48 + (g >> 1) * 32 + j] |= (uint8_t)((q & 15) << ((g & 1) * 4));
                    if (q & 16) blk[16 + j] |= (uint8_t)(1u << g);
                    (*shadow)[r * cols + b * 256 + g * 32 + j] = (double)dq * sc[g] * q - (double)mq * mn[g];
                }
            }
        }
    }
    return off;
}

/* Q5_K routed experts with a Q8_0 shared expert slot: mid and down
 * against the double reference, for decode (T=1), a verify (T=2) and a
 * small prefill batch (T=5). */
static void test_moe_q5k(arena_t *a, uint32_t NE, uint32_t slots, uint32_t E, uint32_t F, uint32_t T) {
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    const uint64_t sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
    const uint64_t su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
    const uint64_t sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    const uint32_t n_out = slots + 1;
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
    double *mid = malloc((uint64_t)T * n_out * F * sizeof(double));
    double *part = malloc((uint64_t)T * n_out * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < n_out; s++) {
            const bool shared = s == slots;
            const uint32_t e = shared ? 0 : (uint32_t)sel[t * slots + s];
            const double *gw = shared ? sg_w : gate_w + (uint64_t)e * F * E;
            const double *uw = shared ? su_w : up_w + (uint64_t)e * F * E;
            const double *dw = shared ? sd_w : down_w + (uint64_t)e * E * F;
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t i = 0; i < E; i++) {
                    g += gw[(uint64_t)f * E + i] * x[t * E + i];
                    u += uw[(uint64_t)f * E + i] * x[t * E + i];
                }
                mid[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            for (uint32_t d = 0; d < E; d++) {
                double acc = 0.0;
                for (uint32_t f = 0; f < F; f++) acc += dw[(uint64_t)d * F + f] * mid[((uint64_t)t * n_out + s) * F + f];
                part[((uint64_t)t * n_out + s) * E + d] = acc;
            }
        }
    }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "sel write");
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gpart = upload(NULL, (uint64_t)T * n_out * E);
    require_ok(ds4_gpu_qwen35_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u, NE, T, slots,
                                             E, F, sg_off, su_off, 8u), "qwen35 moe mid");
    /* The down pass reads the reference mid, so a mid error cannot mask a down error. */
    float *mid_f = malloc((uint64_t)T * n_out * F * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)T * n_out * F; i++) mid_f[i] = (float)mid[i];
    ds4_gpu_tensor *gmid_ref = upload(mid_f, (uint64_t)T * n_out * F);
    require_ok(ds4_gpu_qwen35_moe_down_tensor(gpart, gmid_ref, gsel, a->base, a->size, down_off, 13u, NE, T, slots,
                                              F, E, sd_off, 8u), "qwen35 moe down");
    char what[64];
    snprintf(what, sizeof(what), "q5_K moe mid T=%u", T);
    {
        float *got = download(gmid, (uint64_t)T * n_out * F);
        check_close(what, got, mid, (uint64_t)T * n_out * F, 2e-4);
        free(got);
    }
    snprintf(what, sizeof(what), "q5_K moe down T=%u", T);
    {
        float *got = download(gpart, (uint64_t)T * n_out * E);
        check_close(what, got, part, (uint64_t)T * n_out * E, 2e-4);
        free(got);
    }
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gmid);
    ds4_gpu_tensor_free(gmid_ref); ds4_gpu_tensor_free(gpart);
    free(x); free(sel); free(mid); free(part); free(mid_f);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
}

/* GDN output: per-head RMSNorm of the scan output times ssm_norm, gated by
 * silu(z) (Qwen3.5 RMSNormGated). */
static void test_gdn_out_silu(arena_t *a, uint32_t H, uint32_t D, uint32_t T) {
    double *w;
    const uint64_t w_off = arena_f32(a, D, &w, 0.5f, 1.5f);
    const uint64_t n = (uint64_t)T * H * D;
    float *o = rand_vec(n, 1.0f), *z = rand_vec(n, 2.0f);
    double *ref = malloc(n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const uint64_t base = ((uint64_t)t * H + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)o[base + i] * o[base + i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) ref[base + i] = o[base + i] * r * w[i] * silu_d(z[base + i]);
        }
    }
    ds4_gpu_tensor *go = upload(o, n), *gz = upload(z, n);
    require_ok(ds4_gpu_qwen35_gdn_out_tensor(go, gz, a->base, a->size, w_off, T, H, D, 1e-6f), "qwen35 gdn out");
    float *got = download(go, n);
    check_close("gdn out silu", got, ref, n, 1e-5);
    free(got); free(o); free(z); free(ref); free(w);
    ds4_gpu_tensor_free(go); ds4_gpu_tensor_free(gz);
}

/* NEOX partial RoPE on the first n_rot dims (pairs i, i + n_rot/2). */
static void rope_ref(double *x, uint32_t n_rot, uint32_t pos, double base) {
    const uint32_t nh = n_rot / 2;
    for (uint32_t i = 0; i < nh; i++) {
        const double th = (double)pos * pow(base, -2.0 * i / n_rot);
        const double c = cos(th), s = sin(th), x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* Attention prep without the indexer: q (from the interleaved [q | gate]
 * rows) and k get RMSNorm, their weight and RoPE; the gate passes through; k
 * and v are appended to the F16 caches at pos0.. . */
static void test_attn_prep_noindexer(arena_t *a) {
    const uint32_t T = 3, H = 16, Hkv = 2, D = 256, n_rot = 64, pos0 = 5, cap = 16;
    const double base = 1.0e7;
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *qg = rand_vec((uint64_t)T * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)T * Hkv * D, 1.0f), *vp = rand_vec((uint64_t)T * Hkv * D, 1.0f);
    uint32_t pos3[16 * 4];
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    double *q_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *g_ref = malloc((uint64_t)T * H * D * sizeof(double));
    double *k_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    double row[256];
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t h = 0; h < H; h++) {
            const float *src = qg + ((uint64_t)t * H + h) * 2 * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gq[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) {
                q_ref[((uint64_t)t * H + h) * D + i] = row[i];
                g_ref[((uint64_t)t * H + h) * D + i] = src[D + i];
            }
        }
        for (uint32_t h = 0; h < Hkv; h++) {
            const float *src = kp + ((uint64_t)t * Hkv + h) * D;
            double ss = 0.0;
            for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
            const double r = 1.0 / sqrt(ss / D + 1e-6);
            for (uint32_t i = 0; i < D; i++) row[i] = src[i] * r * gk[i];
            rope_ref(row, n_rot, pos0 + t, base);
            for (uint32_t i = 0; i < D; i++) k_ref[((uint64_t)t * Hkv + h) * D + i] = row[i];
        }
    }
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)T * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)T * Hkv * D), *gvp = upload(vp, (uint64_t)T * Hkv * D);
    ds4_gpu_tensor *gq_out = upload(NULL, (uint64_t)T * H * D), *ggate = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc(sizeof(pos3));
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, sizeof(pos3)), "prep buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(gq_out, ggate, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                                               gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap,
                                               (float)base, 1e-6f, kc, vc, kc, vc, 0u), "qwen35 attn prep");
    float *got = download(gq_out, (uint64_t)T * H * D);
    check_close("attn prep q", got, q_ref, (uint64_t)T * H * D, 2e-5);
    free(got);
    got = download(ggate, (uint64_t)T * H * D);
    check_close("attn prep gate", got, g_ref, (uint64_t)T * H * D, 0.0);
    free(got);
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u), "k cache read");
    require_ok(ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "v cache read");
    float *kf = malloc((uint64_t)T * Hkv * D * sizeof(float)), *vf = malloc((uint64_t)T * Hkv * D * sizeof(float));
    double *v_ref = malloc((uint64_t)T * Hkv * D * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)T * Hkv * D; i++) {
        kf[i] = f16_to_f32(kh[(uint64_t)pos0 * Hkv * D + i]);
        vf[i] = f16_to_f32(vh[(uint64_t)pos0 * Hkv * D + i]);
        v_ref[i] = vp[i];
    }
    check_close("attn prep k cache (f16)", kf, k_ref, (uint64_t)T * Hkv * D, 2e-3);
    check_close("attn prep v cache (f16)", vf, v_ref, (uint64_t)T * Hkv * D, 2e-3);
    free(kh); free(vh); free(kf); free(vf); free(v_ref);
    free(qg); free(kp); free(vp); free(q_ref); free(g_ref); free(k_ref); free(gq); free(gk);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp);
    ds4_gpu_tensor_free(gq_out); ds4_gpu_tensor_free(ggate); ds4_gpu_tensor_free(kc);
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
}

/* MoE reduce without the residual combine (R = NULL, n_hc = 0): the shared
 * expert comes from a part slot (single rows) or a separate buffer (batches). */
static void test_reduce_nohc(uint32_t T, uint32_t K, uint32_t E, bool shared_slot) {
    const uint32_t stride = K + (shared_slot ? 1u : 0u);
    float *part = rand_vec((uint64_t)T * stride * E, 1.0f);
    float *w = rand_vec((uint64_t)T * K, 1.0f);
    float *sg = rand_vec(T, 2.0f);
    float *sh = rand_vec((uint64_t)T * E, 1.0f);
    double *ref = malloc((uint64_t)T * E * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t d = 0; d < E; d++) {
            double acc = 0.0;
            for (uint32_t s = 0; s < K; s++) acc += (double)w[t * K + s] * part[((uint64_t)t * stride + s) * E + d];
            const double shared = shared_slot ? part[((uint64_t)t * stride + K) * E + d] : sh[(uint64_t)t * E + d];
            ref[(uint64_t)t * E + d] = acc + sigmoid_d(sg[t]) * shared;
        }
    }
    ds4_gpu_tensor *gp = upload(part, (uint64_t)T * stride * E), *gw = upload(w, (uint64_t)T * K);
    ds4_gpu_tensor *gsg = upload(sg, T), *gsh = upload(sh, (uint64_t)T * E), *gout = upload(NULL, (uint64_t)T * E);
    require_ok(ds4_gpu_qwen4_moe_reduce_tensor(gout, gp, gw, gsg, shared_slot ? NULL : gsh, NULL, NULL,
                                               T, K, stride, E, 0u), "moe reduce (no hc)");
    float *got = download(gout, (uint64_t)T * E);
    check_close(shared_slot ? "moe reduce, shared slot" : "moe reduce, shared buffer", got, ref, (uint64_t)T * E, 1e-5);
    free(got); free(part); free(w); free(sg); free(sh); free(ref);
    ds4_gpu_tensor_free(gp); ds4_gpu_tensor_free(gw); ds4_gpu_tensor_free(gsg);
    ds4_gpu_tensor_free(gsh); ds4_gpu_tensor_free(gout);
}

/* Ornith MTP input: cat[t] = [RMSNorm(e_t) * enorm | RMSNorm(h_t) * hnorm],
 * embedding half first (llama.cpp qwen35moe graph_mtp). */
static void test_mtp_concat(arena_t *a, uint32_t E, uint32_t T) {
    double *we, *wh;
    const uint64_t e_off = arena_f32(a, E, &we, 0.5f, 1.5f);
    const uint64_t h_off = arena_f32(a, E, &wh, 0.5f, 1.5f);
    const uint64_t n = (uint64_t)T * E;
    float *e = rand_vec(n, 1.0f), *h = rand_vec(n, 3.0f);
    double *ref = malloc(2u * n * sizeof(double));
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t half = 0; half < 2u; half++) {
            const float *x = (half ? h : e) + (uint64_t)t * E;
            const double *w = half ? wh : we;
            double ss = 0.0;
            for (uint32_t i = 0; i < E; i++) ss += (double)x[i] * x[i];
            const double r = 1.0 / sqrt(ss / E + 1e-6);
            for (uint32_t i = 0; i < E; i++) ref[(uint64_t)t * 2u * E + (uint64_t)half * E + i] = x[i] * r * w[i];
        }
    }
    ds4_gpu_tensor *ge = upload(e, n), *gh = upload(h, n), *gc = upload(NULL, 2u * n);
    require_ok(ds4_gpu_qwen35_mtp_concat_tensor(gc, ge, gh, a->base, a->size, e_off, h_off, E, T, 1e-6f),
               "qwen35 mtp concat");
    float *got = download(gc, 2u * n);
    check_close("mtp concat", got, ref, 2u * n, 1e-5);
    free(got); free(e); free(h); free(ref); free(we); free(wh);
    ds4_gpu_tensor_free(ge); ds4_gpu_tensor_free(gh); ds4_gpu_tensor_free(gc);
}

/* Q5_K routed experts through the tiled GEMM: the simdgroup tiles
 * (DS4_QWEN35_MOE_MM_NAX=0) are the exact reference for the tensor tiles, and
 * both are bounded against the double reference and against the M1 per-token
 * row kernels.  T=65 exercises the first tile tail, T=641 the multi-tile,
 * partial-expert and empty-expert cases (cf. test_moe_mm_tiles_exact). */
static void test_moe_mm_q5k(arena_t *a, uint32_t T) {
    const uint32_t E = 256, F = 256, NE = 4, slots = 2, n_out = slots, list_cap = T + 7, guard = 16;
    const uint64_t mid_n = (uint64_t)T * n_out * F, part_n = (uint64_t)T * n_out * E;
    const char *nax_env = "DS4_QWEN35_MOE_MM_NAX";
    const char *saved_nax_v = getenv(nax_env);
    char *saved_nax = saved_nax_v ? strdup(saved_nax_v) : NULL;
    double *gate_w, *up_w, *down_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    float *x = rand_vec((uint64_t)T * E, 2.0f);
    int32_t *sel = malloc((uint64_t)T * slots * sizeof(int32_t));
    for (uint32_t t = 0; t < T; t++) { sel[t * slots] = 0; sel[t * slots + 1] = (int32_t)(1u + t % 2u); }
    double *mid_ex = malloc(mid_n * sizeof(double)), *part_ex = malloc(part_n * sizeof(double));
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) {
            const uint32_t e = (uint32_t)sel[t * slots + s];
            for (uint32_t f = 0; f < F; f++) {
                double g = 0.0, u = 0.0;
                for (uint32_t k = 0; k < E; k++) {
                    g += gate_w[((uint64_t)e * F + f) * E + k] * (double)x[(uint64_t)t * E + k];
                    u += up_w[((uint64_t)e * F + f) * E + k] * (double)x[(uint64_t)t * E + k];
                }
                mid_ex[((uint64_t)t * n_out + s) * F + f] = silu_d(g) * u;
            }
            for (uint32_t d = 0; d < E; d++) {
                double p = 0.0;
                for (uint32_t k = 0; k < F; k++) p += down_w[((uint64_t)e * E + d) * F + k] * mid_ex[((uint64_t)t * n_out + s) * F + k];
                part_ex[((uint64_t)t * n_out + s) * E + d] = p;
            }
        }
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    ds4_gpu_tensor *glists = ds4_gpu_tensor_alloc((uint64_t)NE * list_cap * 4);
    ds4_gpu_tensor *gcounts = ds4_gpu_tensor_alloc(NE * 4);
    require_ok(gsel && glists && gcounts && ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "q5k sel");
    require_ok(ds4_gpu_qwen4_moe_build_lists_tensor(glists, gcounts, gsel, T, slots, NE, list_cap), "q5k lists");
    ds4_gpu_tensor *gmid = upload(NULL, mid_n + guard), *gpart = upload(NULL, part_n + guard);
    const uint32_t levels[] = {0u, 64u};   /* 0 = simdgroup tiles; 64 = tensor tiles */
    float *ref_mid = NULL, *ref_part = NULL;
    const float sentinel = -1234.5f;
    for (uint32_t li = 0; li < 2u; li++) {
        if (levels[li] && !ds4_gpu_tensor_api_available()) {
            printf("  q5k tiles nax=%u skipped (tensor API unavailable)\n", levels[li]);
            continue;
        }
        char lv[4]; snprintf(lv, sizeof(lv), "%u", levels[li] ? 2u : 0u);   /* NAX level 2 => 64-tok tiles */
        setenv(nax_env, lv, 1);
        require_ok(ds4_gpu_tensor_fill_f32(gmid, sentinel, mid_n + guard) &&
                   ds4_gpu_tensor_fill_f32(gpart, sentinel, part_n + guard), "q5k sentinels");
        require_ok(ds4_gpu_qwen35_moe_mm_mid_tensor(gmid, gx, glists, gcounts, a->base, a->size,
                                gate_off, up_off, 13u, NE, T, slots, n_out, E, F, list_cap), "q5k mm mid");
        require_ok(ds4_gpu_qwen35_moe_mm_down_tensor(gpart, gmid, glists, gcounts, a->base, a->size,
                                down_off, 13u, NE, T, slots, n_out, F, E, list_cap), "q5k mm down");
        float *gm = download(gmid, mid_n + guard), *gp = download(gpart, part_n + guard);
        for (uint64_t i = mid_n; i < mid_n + guard; i++) require_ok(gm[i] == sentinel, "q5k mid tail guard");
        for (uint64_t i = part_n; i < part_n + guard; i++) require_ok(gp[i] == sentinel, "q5k down tail guard");
        double ew = 0.0, sc = 1e-6;
        for (uint64_t i = 0; i < part_n; i++) { double d = fabs((double)gp[i] - part_ex[i]); if (d > ew) ew = d; if (fabs(part_ex[i]) > sc) sc = fabs(part_ex[i]); }
        char what[64]; snprintf(what, sizeof(what), "q5k tiles nax=%u down vs exact T=%u", levels[li], T);
        printf("  %-40s max|d|=%.3e (rel %.3e)\n", what, ew, ew / sc);
        require_ok(ew <= 3e-3 * sc, what);
        if (levels[li] == 0u) { ref_mid = gm; ref_part = gp; }   /* simdgroup = exact reference */
        else {
            double wm = 0.0, sm = 1e-6;
            for (uint64_t i = 0; i < mid_n; i++) { if (ref_mid[i] == sentinel) continue; double d = fabs((double)gm[i] - ref_mid[i]); if (d > wm) wm = d; if (fabs(ref_mid[i]) > sm) sm = fabs(ref_mid[i]); }
            require_ok(wm <= 2e-3 * sm, "q5k tensor mid within 2e-3 of simdgroup");
            free(gm); free(gp);
        }
    }
    /* cross-check the simdgroup tiles against the M1 per-token row kernels (no shared slot). */
    setenv(nax_env, "0", 1);
    ds4_gpu_tensor *rmid = upload(NULL, mid_n), *rpart = upload(NULL, part_n);
    require_ok(ds4_gpu_qwen35_moe_mid_tensor(rmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u,
                                             NE, T, slots, E, F, 0, 0, UINT32_MAX), "q5k row mid");
    require_ok(ds4_gpu_qwen35_moe_down_tensor(rpart, rmid, gsel, a->base, a->size, down_off, 13u,
                                              NE, T, slots, F, E, 0, UINT32_MAX), "q5k row down");
    float *rp = download(rpart, part_n);
    double rw = 0.0, rs = 1e-6;
    for (uint64_t i = 0; i < part_n; i++) { double d = fabs((double)rp[i] - ref_part[i]); if (d > rw) rw = d; if (fabs(ref_part[i]) > rs) rs = fabs(ref_part[i]); }
    printf("  q5k tiles vs row kernels down T=%u: max|d|=%.3e (rel %.3e)\n", T, rw, rw / rs);
    require_ok(rw <= 2e-3 * rs, "q5k tiles match row kernels");
    if (saved_nax) { setenv(nax_env, saved_nax, 1); free(saved_nax); } else unsetenv(nax_env);
    free(x); free(sel); free(mid_ex); free(part_ex); free(gate_w); free(up_w); free(down_w);
    free(ref_mid); free(ref_part); free(rp);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(glists); ds4_gpu_tensor_free(gcounts);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gpart); ds4_gpu_tensor_free(rmid); ds4_gpu_tensor_free(rpart);
}

/* L8: the Q5_K decode kernels at each NR are bit-identical (each row's
 * per-lane dot order is unchanged; only the row-per-simdgroup split changes). */
static void test_moe_q5k_rows(arena_t *a) {
    const uint32_t NE = 16, slots = 8, E = 512, F = 256, T = 2, n_out = slots + 1;
    double *gate_w, *up_w, *down_w, *sg_w, *su_w, *sd_w;
    const uint64_t gate_off = arena_q5_K(a, (uint64_t)NE * F, E, &gate_w, 0.05f);
    const uint64_t up_off = arena_q5_K(a, (uint64_t)NE * F, E, &up_w, 0.05f);
    const uint64_t down_off = arena_q5_K(a, (uint64_t)NE * E, F, &down_w, 0.05f);
    const uint64_t sg_off = arena_q8_0(a, F, E, &sg_w, 0.05f);
    const uint64_t su_off = arena_q8_0(a, F, E, &su_w, 0.05f);
    const uint64_t sd_off = arena_q8_0(a, E, F, &sd_w, 0.05f);
    float *x = rand_vec((uint64_t)T * E, 1.0f);
    float *mid_in = rand_vec((uint64_t)T * n_out * F, 1.0f);
    int32_t *sel = malloc((uint64_t)T * slots * 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < slots; s++) sel[t * slots + s] = (int32_t)((t * 7u + s * 3u) % NE);
    ds4_gpu_tensor *gx = upload(x, (uint64_t)T * E);
    ds4_gpu_tensor *gsel = ds4_gpu_tensor_alloc((uint64_t)T * slots * 4);
    require_ok(ds4_gpu_tensor_write(gsel, 0, sel, (uint64_t)T * slots * 4), "l8 sel");
    ds4_gpu_tensor *gmidin = upload(mid_in, (uint64_t)T * n_out * F);
    ds4_gpu_tensor *gmid = upload(NULL, (uint64_t)T * n_out * F), *gpart = upload(NULL, (uint64_t)T * n_out * E);
    const char *mk = "DS4_QWEN35_MOE_MR_MID", *dk = "DS4_QWEN35_MOE_MR_DOWN";
    char *sm = getenv(mk) ? strdup(getenv(mk)) : NULL, *sd = getenv(dk) ? strdup(getenv(dk)) : NULL;
    float *ref_mid = NULL, *ref_part = NULL;
    const char *mrv[] = {"0", "1", "2", "4"}, *drv[] = {"0", "1", "2", "4"};
    for (uint32_t mi = 0; mi < 4u; mi++) {
        setenv(mk, mrv[mi], 1);
        require_ok(ds4_gpu_tensor_fill_f32(gmid, -1234.5f, (uint64_t)T * n_out * F), "l8 mid sentinel");
        require_ok(ds4_gpu_qwen35_moe_mid_tensor(gmid, gx, gsel, a->base, a->size, gate_off, up_off, 13u,
                       NE, T, slots, E, F, sg_off, su_off, 8u), "l8 mid");
        float *got = download(gmid, (uint64_t)T * n_out * F);
        if (ref_mid) { for (uint64_t i = 0; i < (uint64_t)T * n_out * F; i++) require_ok(got[i] == ref_mid[i], "l8 mid NR bit-identical"); free(got); }
        else ref_mid = got;
    }
    for (uint32_t di = 0; di < 4u; di++) {
        setenv(dk, drv[di], 1);
        require_ok(ds4_gpu_tensor_fill_f32(gpart, -1234.5f, (uint64_t)T * n_out * E), "l8 down sentinel");
        require_ok(ds4_gpu_qwen35_moe_down_tensor(gpart, gmidin, gsel, a->base, a->size, down_off, 13u,
                       NE, T, slots, F, E, sd_off, 8u), "l8 down");
        float *got = download(gpart, (uint64_t)T * n_out * E);
        if (ref_part) { for (uint64_t i = 0; i < (uint64_t)T * n_out * E; i++) require_ok(got[i] == ref_part[i], "l8 down NR bit-identical"); free(got); }
        else ref_part = got;
    }
    printf("  q5_K decode NR {mid 0/1/2/4, down 0/1/2/4}: bit-identical\n");
    if (sm) { setenv(mk, sm, 1); free(sm); } else unsetenv(mk);
    if (sd) { setenv(dk, sd, 1); free(sd); } else unsetenv(dk);
    free(x); free(mid_in); free(sel); free(ref_mid); free(ref_part);
    free(gate_w); free(up_w); free(down_w); free(sg_w); free(su_w); free(sd_w);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gsel); ds4_gpu_tensor_free(gmidin);
    ds4_gpu_tensor_free(gmid); ds4_gpu_tensor_free(gpart);
}

/* Ornith attention prep in an FP8/4-bit KV mode: the packed K/V a decode
 * reads.  Prep + decode in mode m must match the F16 prep + decode within the
 * quant's tolerance; the shared decode kernel is the reader, so no per-block
 * packing is reproduced here.  Tolerances (1e-1 fp8, 2.5e-1 q4) are calibrated
 * against a host dequantization of the packed cache (exact to fp8 rel 1.5e-7,
 * q4 rel 5.2e-4); the intrinsic quantization error vs the F16 reference
 * itself measures fp8 rel ~4.4e-2, q4 rel ~1.1e-1. */
static void test_attn_prep_kv_modes(arena_t *a, uint32_t mode) {
    const uint32_t T = 4, H = 16, Hkv = 2, D = 256, n_rot = 64, pos0 = 5, cap = 16;
    const double base = 1.0e7;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *qg = rand_vec((uint64_t)T * H * 2 * D, 1.0f);
    float *kp = rand_vec((uint64_t)T * Hkv * D, 1.0f), *vp = rand_vec((uint64_t)T * Hkv * D, 1.0f);
    uint32_t pos3[16 * 4];
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(qg, (uint64_t)T * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(kp, (uint64_t)T * Hkv * D), *gvp = upload(vp, (uint64_t)T * Hkv * D);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc(sizeof(pos3));
    require_ok(gpos && ds4_gpu_tensor_write(gpos, 0, pos3, sizeof(pos3)), "kv-mode pos");
    /* F16 reference: prep into F16 caches, decode into ref */
    ds4_gpu_tensor *gq0 = upload(NULL, (uint64_t)T * H * D), *ggate0 = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *o_ref = upload(NULL, (uint64_t)T * H * D);
    require_ok(kc && vc && ds4_gpu_qwen35_attn_prep_tensor(gq0, ggate0, kc, vc, gqg, gkp, gvp, gpos,
                   a->base, a->size, gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap, (float)base, 1e-6f,
                   kc, vc, kc, vc, 0u), "prep f16");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(o_ref, gq0, ggate0, kc, vc, NULL, NULL, NULL, T,
                   H, Hkv, D, pos0, 0u, 0u, scale, NULL, NULL, NULL, NULL, 0u), "decode f16");
    float *ref = download(o_ref, (uint64_t)T * H * D);
    /* mode m: prep into packed caches + scales, decode with the same query */
    const uint64_t kv_elems = (uint64_t)cap * Hkv * D;
    ds4_gpu_tensor *gq1 = upload(NULL, (uint64_t)T * H * D), *ggate1 = upload(NULL, (uint64_t)T * H * D);
    ds4_gpu_tensor *kf = ds4_gpu_tensor_alloc(mode == 2u ? kv_elems / 2u : kv_elems);
    ds4_gpu_tensor *vf = ds4_gpu_tensor_alloc(mode == 2u ? kv_elems / 2u : kv_elems);
    ds4_gpu_tensor *ksb = ds4_gpu_tensor_alloc((uint64_t)cap * (Hkv * D / 64u) * 2u);
    ds4_gpu_tensor *vsb = ds4_gpu_tensor_alloc((uint64_t)cap * (Hkv * D / 64u) * 2u);
    ds4_gpu_tensor *o_m = upload(NULL, (uint64_t)T * H * D);
    require_ok(kf && vf && ksb && vsb && ds4_gpu_qwen35_attn_prep_tensor(gq1, ggate1, kf, vf, gqg, gkp, gvp,
                   gpos, a->base, a->size, gq_off, gk_off, T, H, Hkv, D, n_rot, pos0, cap, (float)base, 1e-6f,
                   kf, vf, ksb, vsb, mode), "prep kv mode");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(o_m, gq1, ggate1, kf, vf, NULL, NULL, NULL, T,
                   H, Hkv, D, pos0, 0u, 0u, scale, kf, vf, ksb, vsb, mode), "decode kv mode");
    float *got = download(o_m, (uint64_t)T * H * D);
    double worst = 0.0, sc = 1e-6;
    for (uint64_t i = 0; i < (uint64_t)T * H * D; i++) {
        if (fabs(ref[i]) > sc) sc = fabs(ref[i]);
        if (fabs((double)got[i] - ref[i]) > worst) worst = fabs((double)got[i] - ref[i]);
    }
    printf("  attn prep+decode kv mode %u: attn out max|d| %.3e (rel %.3e)\n", mode, worst, worst / sc);
    require_ok(worst <= (mode == 2u ? 2.5e-1 : 1e-1) * sc, "kv-mode attention within tolerance");
    free(ref); free(got); free(qg); free(kp); free(vp); free(gq); free(gk);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq0); ds4_gpu_tensor_free(ggate0); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc);
    ds4_gpu_tensor_free(o_ref); ds4_gpu_tensor_free(gq1); ds4_gpu_tensor_free(ggate1); ds4_gpu_tensor_free(kf);
    ds4_gpu_tensor_free(vf); ds4_gpu_tensor_free(ksb); ds4_gpu_tensor_free(vsb); ds4_gpu_tensor_free(o_m);
}

/* L12: kernel_qwen35_attn_decode2's shared-KV verify must reproduce, bit for
 * bit, what the SAME kernel gives for each row decoded alone (rows==1) --
 * that is the "exact by construction" bar (Task 11 fix 6).  Comparing
 * against the unrelated per-row kernel_qwen4_attn_decode is not a fair
 * memcmp (a different kernel's instruction scheduling can 1-ulp diverge even
 * for identical math -- that is what sank the dry run's version of this
 * test).  A host double reference separately bounds the two-row call within
 * FP noise of the true softmax.  pos0=31 gives row 0 (32 keys) one split and
 * row 1 (33 keys) two: the mismatched-geometry case fix 6 exists for. */
static void test_attn_decode2_matches_perrow(arena_t *a, uint32_t pos0) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, cap = 320, fill = pos0 + 2u;
    const double base = 1.0e7;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "decode2 fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, (float)base, 1e-6f, kc, vc, kc, vc, 0u),
               "decode2 fill");

    float *q2 = rand_vec(2ull * H * D, 1.0f), *g2 = rand_vec(2ull * H * D, 1.0f);
    ds4_gpu_tensor *gq2 = upload(q2, 2ull * H * D), *gg2 = upload(g2, 2ull * H * D);
    ds4_gpu_tensor *o2 = upload(NULL, 2ull * H * D);
    ds4_gpu_tensor *part2 = upload(NULL, ds4_gpu_qwen4_attn_part_floats(2u, H, D));
    require_ok(part2 && ds4_gpu_qwen35_attn_decode2_tensor(o2, gq2, gg2, kc, vc, part2, H, Hkv, D, pos0, 2u, scale,
                   NULL, NULL, NULL, NULL, 0u), "decode2 rows=2");
    float *shared = download(o2, 2ull * H * D);

    /* the same kernel, called once per row (rows==1) */
    float *solo = malloc(2ull * H * D * sizeof(float));
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(gq2, (uint64_t)r * H * D * sizeof(float),
                                                 (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *gr = ds4_gpu_tensor_view(gg2, (uint64_t)r * H * D * sizeof(float),
                                                 (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *orow = upload(NULL, (uint64_t)H * D);
        ds4_gpu_tensor *part1 = upload(NULL, ds4_gpu_qwen4_attn_part_floats(1u, H, D));
        require_ok(qr && gr && orow && part1 &&
                   ds4_gpu_qwen35_attn_decode2_tensor(orow, qr, gr, kc, vc, part1, H, Hkv, D, pos0 + r, 1u, scale,
                       NULL, NULL, NULL, NULL, 0u), "decode2 rows=1 reference");
        float *rr = download(orow, (uint64_t)H * D);
        memcpy(solo + (uint64_t)r * H * D, rr, (uint64_t)H * D * sizeof(float));
        free(rr);
        ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(gr); ds4_gpu_tensor_free(orow); ds4_gpu_tensor_free(part1);
    }
    const int bytes_equal = memcmp(shared, solo, 2ull * H * D * sizeof(float)) == 0;
    printf("  attn decode2 pos0=%u: rows=2 vs two rows=1 calls memcmp %s\n",
           pos0, bytes_equal ? "== 0" : "DIFFERS");
    require_ok(bytes_equal, "decode2 rows=2 matches two rows=1 calls of the same kernel");

    /* host double reference: exact softmax over the raw K/V this test wrote.
     * kc/vc are the F16 packed cache (2 bytes/elem), not float32. */
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * sizeof(uint16_t));
    uint16_t *vh = malloc((uint64_t)cap * Hkv * D * sizeof(uint16_t));
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * sizeof(uint16_t)), "decode2 k cache read");
    require_ok(ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * sizeof(uint16_t)), "decode2 v cache read");
    float *kraw = malloc((uint64_t)cap * Hkv * D * sizeof(float));
    float *vraw = malloc((uint64_t)cap * Hkv * D * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)cap * Hkv * D; i++) { kraw[i] = f16_to_f32(kh[i]); vraw[i] = f16_to_f32(vh[i]); }
    free(kh); free(vh);
    float *qraw = download(gq2, 2ull * H * D);
    float *graw = download(gg2, 2ull * H * D);
    double worst = 0.0, sc = 1e-6;
    for (uint32_t r = 0; r < 2u; r++) {
        const uint32_t n_keys = pos0 + r + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double acc[256], m = -1e300, l = 0.0;
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            for (uint32_t idx = 0; idx < n_keys; idx++) {
                double s = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    s += (double)qraw[((uint64_t)r * H + h) * D + d] * scale *
                         (double)kraw[((uint64_t)idx * Hkv + kvh) * D + d];
                const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                l = l * corr + w;
                for (uint32_t d = 0; d < D; d++)
                    acc[d] = acc[d] * corr + w * (double)vraw[((uint64_t)idx * Hkv + kvh) * D + d];
                m = mn;
            }
            const double inv = l > 0.0 ? 1.0 / l : 0.0;
            for (uint32_t d = 0; d < D; d++) {
                const double sig = 1.0 / (1.0 + exp(-(double)graw[((uint64_t)r * H + h) * D + d]));
                const double refv = acc[d] * inv * sig;
                const double got = (double)shared[((uint64_t)r * H + h) * D + d];
                if (fabs(refv) > sc) sc = fabs(refv);
                if (fabs(got - refv) > worst) worst = fabs(got - refv);
            }
        }
    }
    printf("  attn decode2 pos0=%u: vs host double ref max|d| %.3e (rel %.3e)\n", pos0, worst, worst / sc);
    require_ok(worst <= 1e-5 * sc, "decode2 within FP noise of the host double reference");

    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(q2); free(g2);
    free(shared); free(solo); free(kraw); free(vraw); free(qraw); free(graw);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq2); ds4_gpu_tensor_free(gg2); ds4_gpu_tensor_free(o2); ds4_gpu_tensor_free(part2);
}

/* decode3 (M5): the rows==2 verify call equals two rows==1 calls of the same
 * kernel bit for bit, and both equal a host double reference within FP noise.
 * `split_keys` forces small splits so short caches exercise many splits and
 * the split-count boundaries (ns0 != ns1, kps0 != kps1). Outputs are
 * sentinel-filled before every call so a kernel that writes nothing fails. */
static void test_attn_decode3_rows(arena_t *a, uint32_t pos0, uint32_t split_keys) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, fill = pos0 + 2u;
    const uint32_t cap = fill + 8u;
    const float scale = 1.0f / sqrtf((float)D);
    char sk[16];
    snprintf(sk, sizeof(sk), "%u", split_keys);
    setenv("DS4_QWEN35_ATTN_SPLIT_KEYS", sk, 1);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "decode3 fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, 1.0e7f, 1e-6f, kc, vc, kc, vc, 0u),
               "decode3 fill");

    float *q2 = rand_vec(2ull * H * D, 1.0f), *g2 = rand_vec(2ull * H * D, 1.0f);
    ds4_gpu_tensor *gq2 = upload(q2, 2ull * H * D), *gg2 = upload(g2, 2ull * H * D);
    ds4_gpu_tensor *o2 = upload(NULL, 2ull * H * D);
    ds4_gpu_tensor *part2 = upload(NULL, ds4_gpu_qwen35_attn_part3_floats(2u, H, D));
    require_ok(o2 && part2 && ds4_gpu_tensor_fill_f32(o2, -1234.5f, 2ull * H * D), "decode3 sentinel rows=2");
    require_ok(ds4_gpu_qwen35_attn_decode3_tensor(o2, gq2, gg2, kc, vc, part2, H, Hkv, D, pos0, 2u, scale),
               "decode3 rows=2");
    float *shared = download(o2, 2ull * H * D);

    float *solo = malloc(2ull * H * D * sizeof(float));
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(gq2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *gr = ds4_gpu_tensor_view(gg2, (uint64_t)r * H * D * sizeof(float), (uint64_t)H * D * sizeof(float));
        ds4_gpu_tensor *orow = upload(NULL, (uint64_t)H * D);
        ds4_gpu_tensor *part1 = upload(NULL, ds4_gpu_qwen35_attn_part3_floats(1u, H, D));
        require_ok(qr && gr && orow && part1 && ds4_gpu_tensor_fill_f32(orow, -1234.5f, (uint64_t)H * D),
                   "decode3 sentinel rows=1");
        require_ok(ds4_gpu_qwen35_attn_decode3_tensor(orow, qr, gr, kc, vc, part1, H, Hkv, D, pos0 + r, 1u, scale),
                   "decode3 rows=1");
        float *rr = download(orow, (uint64_t)H * D);
        memcpy(solo + (uint64_t)r * H * D, rr, (uint64_t)H * D * sizeof(float));
        free(rr);
        ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(gr); ds4_gpu_tensor_free(orow); ds4_gpu_tensor_free(part1);
    }
    const int same = memcmp(shared, solo, 2ull * H * D * sizeof(float)) == 0;
    printf("  attn decode3 pos0=%u split_keys=%u: rows=2 vs two rows=1 memcmp %s\n",
           pos0, split_keys, same ? "== 0" : "DIFFERS");
    require_ok(same, "decode3 rows=2 matches two rows=1 calls of the same kernel");

    /* host double reference over the packed F16 cache */
    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u) &&
               ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "decode3 cache read");
    float *qraw = download(gq2, 2ull * H * D), *graw = download(gg2, 2ull * H * D);
    double worst = 0.0, sc = 1e-6;
    for (uint32_t r = 0; r < 2u; r++) {
        const uint32_t n_keys = pos0 + r + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double acc[256], m = -1e300, l = 0.0;
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            for (uint32_t idx = 0; idx < n_keys; idx++) {
                double s = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    s += (double)qraw[((uint64_t)r * H + h) * D + d] * scale *
                         (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                l = l * corr + w;
                for (uint32_t d = 0; d < D; d++)
                    acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                m = mn;
            }
            for (uint32_t d = 0; d < D; d++) {
                const double sig = 1.0 / (1.0 + exp(-(double)graw[((uint64_t)r * H + h) * D + d]));
                const double refv = acc[d] / l * sig;
                const double got = (double)shared[((uint64_t)r * H + h) * D + d];
                if (fabs(refv) > sc) sc = fabs(refv);
                if (fabs(got - refv) > worst) worst = fabs(got - refv);
            }
        }
    }
    printf("  attn decode3 pos0=%u: vs host double ref max|d| %.3e (rel %.3e)\n", pos0, worst, worst / sc);
    /* queries, keys and probabilities are staged as half on the matrix path, like attn_mm */
    require_ok(worst <= 3e-3 * sc, "decode3 within half-precision staging of the host double reference");
    unsetenv("DS4_QWEN35_ATTN_SPLIT_KEYS");
    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(q2); free(g2);
    free(shared); free(solo); free(kh); free(vh); free(qraw); free(graw);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(gq2); ds4_gpu_tensor_free(gg2); ds4_gpu_tensor_free(o2); ds4_gpu_tensor_free(part2);
}

/* attn_flash (M5): causal prefill attention for T tokens at pos0 against a
 * host double reference and against today's kernel (qwen4 attn_mm via the
 * qwen4 wrapper). */
static void test_attn_flash(arena_t *a, uint32_t pos0, uint32_t T, int use_part) {
    const uint32_t H = 16, Hkv = 2, D = 256, n_rot = 64, fill = pos0 + T, cap = fill + 8u;
    const float scale = 1.0f / sqrtf((float)D);
    double *gq, *gk;
    const uint64_t gq_off = arena_f32(a, D, &gq, 0.5f, 1.5f);
    const uint64_t gk_off = arena_f32(a, D, &gk, 0.5f, 1.5f);
    float *pqg = rand_vec((uint64_t)fill * H * 2 * D, 1.0f);
    float *pkp = rand_vec((uint64_t)fill * Hkv * D, 1.0f), *pvp = rand_vec((uint64_t)fill * Hkv * D, 1.0f);
    uint32_t *pos3 = malloc((uint64_t)cap * 4u * sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) { pos3[p * 4] = pos3[p * 4 + 1] = pos3[p * 4 + 2] = p; pos3[p * 4 + 3] = 0; }
    ds4_gpu_tensor *gqg = upload(pqg, (uint64_t)fill * H * 2 * D);
    ds4_gpu_tensor *gkp = upload(pkp, (uint64_t)fill * Hkv * D), *gvp = upload(pvp, (uint64_t)fill * Hkv * D);
    ds4_gpu_tensor *pq = upload(NULL, (uint64_t)fill * H * D), *pg = upload(NULL, (uint64_t)fill * H * D);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc((uint64_t)cap * Hkv * D * 2u);
    ds4_gpu_tensor *gpos = ds4_gpu_tensor_alloc((uint64_t)cap * 16u);
    require_ok(kc && vc && gpos && ds4_gpu_tensor_write(gpos, 0, pos3, (uint64_t)cap * 16u), "flash fill buffers");
    require_ok(ds4_gpu_qwen35_attn_prep_tensor(pq, pg, kc, vc, gqg, gkp, gvp, gpos, a->base, a->size,
                   gq_off, gk_off, fill, H, Hkv, D, n_rot, 0u, cap, 1.0e7f, 1e-6f, kc, vc, kc, vc, 0u),
               "flash fill");
    /* the last T rows of the prepped q/gate are the queries */
    ds4_gpu_tensor *qT = ds4_gpu_tensor_view(pq, (uint64_t)pos0 * H * D * sizeof(float), (uint64_t)T * H * D * sizeof(float));
    ds4_gpu_tensor *gT = ds4_gpu_tensor_view(pg, (uint64_t)pos0 * H * D * sizeof(float), (uint64_t)T * H * D * sizeof(float));
    ds4_gpu_tensor *of = upload(NULL, (uint64_t)T * H * D), *om = upload(NULL, (uint64_t)T * H * D);
    /* key-split scratch only for the forced-split cases: its size follows the split rule */
    ds4_gpu_tensor *partf = use_part ? upload(NULL, ds4_gpu_qwen35_attn_flash_part_floats(T, H, D)) : NULL;
    require_ok(qT && gT && of && om && (!use_part || partf) &&
               ds4_gpu_tensor_fill_f32(of, -1234.5f, (uint64_t)T * H * D), "flash buffers");
    require_ok(ds4_gpu_qwen35_attn_flash_tensor(of, qT, gT, kc, vc, partf, T, H, Hkv, D, pos0, scale), "flash call");
    require_ok(ds4_gpu_qwen4_attn_decode_tensor(om, qT, gT, kc, vc, NULL, NULL, NULL, T, H, Hkv, D, pos0,
                   0u, 0u, scale, NULL, NULL, NULL, NULL, 0u), "qwen4 attn reference");
    float *gotf = download(of, (uint64_t)T * H * D), *gotm = download(om, (uint64_t)T * H * D);

    uint16_t *kh = malloc((uint64_t)cap * Hkv * D * 2u), *vh = malloc((uint64_t)cap * Hkv * D * 2u);
    require_ok(ds4_gpu_tensor_read(kc, 0, kh, (uint64_t)cap * Hkv * D * 2u) &&
               ds4_gpu_tensor_read(vc, 0, vh, (uint64_t)cap * Hkv * D * 2u), "flash cache read");
    float *qraw = download(qT, (uint64_t)T * H * D), *graw = download(gT, (uint64_t)T * H * D);
    double worst_ref = 0.0, worst_mm = 0.0, sc = 1e-6;
    for (uint32_t t = 0; t < T; t++) {
        const uint32_t n_keys = pos0 + t + 1u;
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / (H / Hkv);
            double acc[256], m = -1e300, l = 0.0;
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            for (uint32_t idx = 0; idx < n_keys; idx++) {
                double s = 0.0;
                for (uint32_t d = 0; d < D; d++)
                    s += (double)qraw[((uint64_t)t * H + h) * D + d] * scale *
                         (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                l = l * corr + w;
                for (uint32_t d = 0; d < D; d++)
                    acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                m = mn;
            }
            for (uint32_t d = 0; d < D; d++) {
                const uint64_t o = ((uint64_t)t * H + h) * D + d;
                const double refv = acc[d] / l / (1.0 + exp(-(double)graw[o]));
                if (fabs(refv) > sc) sc = fabs(refv);
                if (fabs((double)gotf[o] - refv) > worst_ref) worst_ref = fabs((double)gotf[o] - refv);
                if (fabs((double)gotf[o] - (double)gotm[o]) > worst_mm) worst_mm = fabs((double)gotf[o] - (double)gotm[o]);
            }
        }
    }
    printf("  attn flash pos0=%u T=%u: vs host double ref max|d| %.3e (rel %.3e), vs attn_mm %.3e\n",
           pos0, T, worst_ref, worst_ref / sc, worst_mm);
    /* queries and K/V are staged as half on the matrix path, like attn_mm */
    require_ok(worst_ref <= 3e-3 * sc, "flash within half-precision staging of the host double reference");
    require_ok(worst_mm <= 3e-3 * sc, "flash within half-precision staging of qwen4 attn_mm");
    free(gq); free(gk); free(pqg); free(pkp); free(pvp); free(pos3); free(gotf); free(gotm);
    free(kh); free(vh); free(qraw); free(graw);
    ds4_gpu_tensor_free(gqg); ds4_gpu_tensor_free(gkp); ds4_gpu_tensor_free(gvp); ds4_gpu_tensor_free(pq);
    ds4_gpu_tensor_free(pg); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(gpos);
    ds4_gpu_tensor_free(qT); ds4_gpu_tensor_free(gT); ds4_gpu_tensor_free(of); ds4_gpu_tensor_free(om);
    if (partf) ds4_gpu_tensor_free(partf);
}

/* flash_nax (M6): causal prefill attention on the accelerator path.  Every
 * K/V row past the fill is NaN (the cache is 40 rows longer than the fill,
 * so a 32-key block past it stays inside the buffer): a kernel that
 * multiplies such a row, even by a zero probability, turns the output
 * non-finite.  One guard token after the output must keep its sentinel (no
 * write past T).  Checked against a host double reference (host_ref) and
 * always against the M5 simdgroup flash.  Returns the Ks the accelerator
 * call used (0 when the tensor API is unavailable). */
static uint32_t test_attn_flash_nax(uint32_t pos0, uint32_t T, int use_part, int host_ref) {
    const uint32_t H = 16, Hkv = 2, D = 256, fill = pos0 + T, cap = fill + 40u;
    const uint64_t n_out = (uint64_t)T * H * D, row = (uint64_t)H * D, n_kv = (uint64_t)cap * Hkv * D;
    const float scale = 1.0f / sqrtf((float)D);
    if (!ds4_gpu_tensor_api_available()) {
        printf("  attn flash nax pos0=%u T=%u skipped (tensor API unavailable)\n", pos0, T);
        return 0;
    }
    float *qv = rand_vec(n_out, 3.0f), *gv = rand_vec(n_out, 1.0f);
    uint16_t *kh = malloc(n_kv * sizeof(uint16_t)), *vh = malloc(n_kv * sizeof(uint16_t));
    for (uint64_t i = 0; i < n_kv; i++) {
        const int past = i >= (uint64_t)fill * Hkv * D;
        kh[i] = past ? 0x7e00u : f32_to_f16(frand());   /* half NaN past the fill */
        vh[i] = past ? 0x7e00u : f32_to_f16(frand());
    }
    ds4_gpu_tensor *q = upload(qv, n_out), *gt = upload(gv, n_out);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc(n_kv * sizeof(uint16_t));
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc(n_kv * sizeof(uint16_t));
    require_ok(kc && vc && ds4_gpu_tensor_write(kc, 0, kh, n_kv * sizeof(uint16_t)) &&
               ds4_gpu_tensor_write(vc, 0, vh, n_kv * sizeof(uint16_t)), "nax cache upload");
    ds4_gpu_tensor *on = upload(NULL, n_out + row), *of = upload(NULL, n_out);
    ds4_gpu_tensor *part = use_part ? upload(NULL, ds4_gpu_qwen35_attn_flash_part_floats(T, H, D)) : NULL;
    require_ok(ds4_gpu_tensor_fill_f32(on, -1234.5f, n_out + row), "nax sentinel");
    require_ok(ds4_gpu_qwen35_attn_flash_nax_tensor(on, q, gt, kc, vc, part, T, H, Hkv, D, pos0, scale),
               "flash nax call");
    const uint32_t splits = ds4_gpu_qwen35_attn_flash_last_splits();
    require_ok(ds4_gpu_qwen35_attn_flash_tensor(of, q, gt, kc, vc, NULL, T, H, Hkv, D, pos0, scale),
               "flash reference");
    float *gotn = download(on, n_out + row), *gotf = download(of, n_out);

    uint64_t nonfinite = 0, guard_bad = 0;
    double worst_f = 0.0, scf = 1e-6, worst_ref = 0.0, sc = 1e-6;
    for (uint64_t o = 0; o < n_out; o++) {
        if (!isfinite(gotn[o])) nonfinite++;
        if (fabs((double)gotf[o]) > scf) scf = fabs((double)gotf[o]);
        if (fabs((double)gotn[o] - (double)gotf[o]) > worst_f) worst_f = fabs((double)gotn[o] - (double)gotf[o]);
    }
    for (uint64_t o = n_out; o < n_out + row; o++) guard_bad += gotn[o] != -1234.5f;
    if (host_ref) {
        for (uint32_t t = 0; t < T; t++) {
            const uint32_t n_keys = pos0 + t + 1u;
            for (uint32_t h = 0; h < H; h++) {
                const uint32_t kvh = h / (H / Hkv);
                double acc[256], m = -1e300, l = 0.0;
                for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
                for (uint32_t idx = 0; idx < n_keys; idx++) {
                    double s = 0.0;
                    for (uint32_t d = 0; d < D; d++)
                        s += (double)qv[((uint64_t)t * H + h) * D + d] * scale *
                             (double)f16_to_f32(kh[((uint64_t)idx * Hkv + kvh) * D + d]);
                    const double mn = s > m ? s : m, corr = exp(m - mn), w = exp(s - mn);
                    l = l * corr + w;
                    for (uint32_t d = 0; d < D; d++)
                        acc[d] = acc[d] * corr + w * (double)f16_to_f32(vh[((uint64_t)idx * Hkv + kvh) * D + d]);
                    m = mn;
                }
                for (uint32_t d = 0; d < D; d++) {
                    const uint64_t o = ((uint64_t)t * H + h) * D + d;
                    const double refv = acc[d] / l * sigmoid_d((double)gv[o]);
                    if (fabs(refv) > sc) sc = fabs(refv);
                    if (fabs((double)gotn[o] - refv) > worst_ref) worst_ref = fabs((double)gotn[o] - refv);
                }
            }
        }
    }
    printf("  attn flash nax pos0=%u T=%u splits=%u: vs M5 flash max|d| %.3e (rel %.3e)", pos0, T, splits,
           worst_f, worst_f / scf);
    if (host_ref) printf(", vs host double ref rel %.3e", worst_ref / sc);
    printf("\n");
    require_ok(nonfinite == 0, "flash nax output finite (no K/V row past the fill multiplied)");
    require_ok(guard_bad == 0, "flash nax writes nothing past token T");
    require_ok(worst_f <= 2e-3 * scf, "flash nax within 2e-3 of the M5 simdgroup flash");
    if (host_ref) require_ok(worst_ref <= 1e-3 * sc, "flash nax within 1e-3 of the host double reference");
    free(qv); free(gv); free(kh); free(vh); free(gotn); free(gotf);
    ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(gt); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(vc);
    ds4_gpu_tensor_free(on); ds4_gpu_tensor_free(of);
    if (part) ds4_gpu_tensor_free(part);
    return splits;
}

int main(void) {
    arena_t arena;
    arena.size = (uint64_t)512 << 20;
    arena.base = mmap(NULL, arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    arena.used = 0;
    if (arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(arena.base, arena.size), "model map registration");

    printf("qwen35 moe (q5_K experts, q8_0 shared slot)\n");
    test_moe_q5k(&arena, 16, 8, 512, 256, 1);
    test_moe_q5k(&arena, 16, 8, 512, 256, 2);
    test_moe_q5k(&arena, 16, 8, 512, 512, 5);
    printf("qwen35 q5_K decode row variants (L8)\n"); test_moe_q5k_rows(&arena);
    printf("qwen35 moe q5_K tiled GEMM (simdgroup + tensor tiles)\n");
    test_moe_mm_q5k(&arena, 65);
    test_moe_mm_q5k(&arena, 200);
    test_moe_mm_q5k(&arena, 641);
    printf("qwen35 gdn out (silu gate)\n");
    test_gdn_out_silu(&arena, 32, 128, 3);
    printf("qwen35 attention prep (no indexer)\n");
    test_attn_prep_noindexer(&arena);
    printf("qwen35 attention prep + decode (KV modes)\n");
    test_attn_prep_kv_modes(&arena, 1u);
    test_attn_prep_kv_modes(&arena, 2u);
    printf("qwen35 attention decode2 (L12 shared-KV verify)\n");
    test_attn_decode2_matches_perrow(&arena, 6u);
    test_attn_decode2_matches_perrow(&arena, 30u);
    test_attn_decode2_matches_perrow(&arena, 31u);
    test_attn_decode2_matches_perrow(&arena, 100u);
    printf("qwen35 attention decode3 (M5)\n");
    test_attn_decode3_rows(&arena, 6u, 64u);      /* one split per row */
    test_attn_decode3_rows(&arena, 62u, 64u);     /* n0 = 63 (1 split), n1 = 64 (1 split, full) */
    test_attn_decode3_rows(&arena, 63u, 64u);     /* n0 = 64 (1 split), n1 = 65 (2 splits): ns0 != ns1 */
    test_attn_decode3_rows(&arena, 200u, 16u);    /* shared dispatch, kps 16/16 */
    test_attn_decode3_rows(&arena, 194u, 16u);    /* same ns 13, kps0 = 15 != kps1 = 16 */
    test_attn_decode3_rows(&arena, 4200u, 16u);   /* at the 256-split cap: n / 16 > 256 */
    test_attn_decode3_rows(&arena, 64u, 64u);     /* shared dispatch (ns 2, kps 33): row 1 needs a third tile */
    test_attn_decode3_rows(&arena, 4112u, 16u);   /* near the cap (ns 242, kps 17), shared dispatch, extra tile */
    printf("qwen4 moe reduce without residual (as Ornith calls it)\n");
    test_reduce_nohc(1, 8, 2048, true);
    test_reduce_nohc(12, 8, 2048, false);
    printf("qwen35 mtp concat\n");
    test_mtp_concat(&arena, 2048, 1);
    test_mtp_concat(&arena, 2048, 7);
    test_mtp_concat(&arena, 256, 3);
    printf("qwen35 attention flash prefill (M5)\n");
    test_attn_flash(&arena, 0u, 9u, 0);
    test_attn_flash(&arena, 0u, 65u, 0);
    test_attn_flash(&arena, 37u, 65u, 0);
    test_attn_flash(&arena, 37u, 200u, 0);
    test_attn_flash(&arena, 4096u, 64u, 0);
    setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", "4096", 1);   /* force key splits on short caches */
    test_attn_flash(&arena, 4096u, 64u, 1);
    require_ok(ds4_gpu_qwen35_attn_flash_last_splits() > 1, "flash key split taken (T=64)");
    test_attn_flash(&arena, 8000u, 200u, 1);
    require_ok(ds4_gpu_qwen35_attn_flash_last_splits() > 1, "flash key split taken (T=200)");
    test_attn_flash(&arena, 1000u, 1100u, 1);
    require_ok(ds4_gpu_qwen35_attn_flash_last_splits() > 1, "flash key split taken (T=1100, neutral partials)");
    unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
    printf("qwen35 attention flash on the neural accelerators (M6)\n");
    test_attn_flash_nax(0u, 9u, 0, 1);
    test_attn_flash_nax(0u, 65u, 0, 1);
    test_attn_flash_nax(37u, 65u, 0, 1);
    test_attn_flash_nax(37u, 200u, 0, 1);
    test_attn_flash_nax(4096u, 64u, 0, 1);
    test_attn_flash_nax(30720u, 2048u, 0, 0);   /* long context: accelerator drift vs the simdgroup flash */
    setenv("DS4_QWEN35_ATTN_FLASH_MIN_TG", "4096", 1);   /* force key splits on short caches */
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(4096u, 64u, 1, 1) > 1,
               "flash nax key split taken (T=64)");
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(8000u, 200u, 1, 1) > 1,
               "flash nax key split taken (T=200)");
    require_ok(!ds4_gpu_tensor_api_available() || test_attn_flash_nax(1000u, 1100u, 1, 1) > 1,
               "flash nax key split taken (T=1100, neutral partials)");
    unsetenv("DS4_QWEN35_ATTN_FLASH_MIN_TG");
    printf("qwen35 kernels: ok\n");
    return 0;
}
