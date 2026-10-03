/* Model-free roofline sweep of Ornith's decode kernels (round 2, Stage K:
 * docs/superpowers/specs/2026-10-03-ornith-decode-round2-design.md section 4).
 * Real shapes from the Ornith GGUF: routed experts 256 x (2048x512), 8 used,
 * Q5_K in layers 0-14 and Q4_K in 15-39; a Q8_0 shared expert as the extra
 * slot; Q8_0 projections.  Each timed call reads weights the previous calls
 * did not (a different weight copy, a different random expert set), every
 * method warms up untimed for WARM_S, and each timed repetition reads at
 * least 1 GiB; the median of REPS is reported.  The lm-head Q8_0 matvec,
 * which streams 540 MB near the nominal bandwidth, is the practical peak
 * reference.  The verify (T=2) row 1 shares 3 of row 0's 8 experts, as the
 * measured overlap (3.2 of 8) does.  `smoke` runs every row once and checks
 * the outputs are finite and not all zero.  Needs the GPU but no model; time
 * it only in an announced window with the live stack paused. */
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
 * GPU-kernel-only bench does not link (same pattern as tests/bench_qwen35_verify.c). */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

enum { T_Q8_0 = 8u, T_Q4_K = 12u, T_Q5_K = 13u };
enum { E = 2048u, F = 512u, N_EXP = 256u, K_USED = 8u, K_SHARE = 3u, N_SETS = 64u, N_LAYER_COPIES = 2u,
       N_SHARED_COPIES = 64u, REPS = 5, MIN_CALLS = 16, MAX_CALLS = 1024, MAX_COPIES = 64 };
static const uint64_t COLD_BYTES = 256ull << 20, REP_BYTES = 1ull << 30;
static const double WARM_S = 0.3, CYCLE_MS_2K = 20.02, CANDIDATE_SHARE = 0.03;

typedef enum { R_Q8, R_MID, R_DOWN } kind_t;
typedef struct { const char *name; kind_t kind; uint32_t type, in_dim, out_dim, n_step; } row_t;
static const row_t ROWS[] = {
    { "q8 2048x248320 (lm_head, peak ref)", R_Q8, T_Q8_0, 2048u, 248320u, 1u },
    { "q8 2048x8192 (attn_qkv, attn_q)", R_Q8, T_Q8_0, 2048u, 8192u, 40u },
    { "q8 2048x4096 (attn_gate)", R_Q8, T_Q8_0, 2048u, 4096u, 30u },
    { "q8 4096x2048 (ssm_out, attn_output)", R_Q8, T_Q8_0, 4096u, 2048u, 40u },
    { "moe mid Q5_K + shared", R_MID, T_Q5_K, E, F, 15u },
    { "moe down Q5_K + shared", R_DOWN, T_Q5_K, F, E, 15u },
    { "moe mid Q4_K + shared", R_MID, T_Q4_K, E, F, 25u },
    { "moe down Q4_K + shared", R_DOWN, T_Q4_K, F, E, 25u },
};
enum { N_ROWS = (int)(sizeof(ROWS) / sizeof(ROWS[0])) };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "bench_qwen35_decode: %s failed\n", what); exit(1); }
}
static uint32_t g_rng = 0x6c8e9cf5u;
static uint32_t next_rng(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}

static uint64_t row_bytes(uint32_t type, uint32_t in_dim) {
    switch (type) {
    case T_Q8_0: return (uint64_t)in_dim / 32u * 34u;
    case T_Q4_K: return (uint64_t)in_dim / 256u * 144u;
    case T_Q5_K: return (uint64_t)in_dim / 256u * 176u;
    default: return 0;
    }
}
/* Q8_0: scale 2^-7, quants in [-64, 63]; K-quants: random scales/quants, d = dmin = 2^-7 */
static void fill_weights(uint8_t *p, uint64_t bytes, uint32_t type) {
    const uint64_t blk = type == T_Q8_0 ? 34u : type == T_Q4_K ? 144u : 176u;
    for (uint64_t b = 0; b + blk <= bytes; b += blk) {
        for (uint64_t j = 0; j < blk; j++) p[b + j] = (uint8_t)(next_rng() >> 11);
        p[b] = 0x00; p[b + 1] = 0x20;
        if (type != T_Q8_0) { p[b + 2] = 0x00; p[b + 3] = 0x20; }
        else for (int j = 0; j < 32; j++) p[b + 2 + j] = (uint8_t)((next_rng() >> 8) & 0x7fu) - 64u;
    }
}
static uint32_t copies_for(uint64_t bytes) {
    uint64_t c = (COLD_BYTES + bytes - 1u) / bytes;
    if (c < 2u) c = 2u;
    return c > MAX_COPIES ? MAX_COPIES : (uint32_t)c;
}
static ds4_gpu_tensor *rand_tensor(uint64_t n) {
    float *h = malloc(n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) h[i] = (float)((int)(next_rng() & 0xffffu) - 32768) / 32768.0f;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    need(t && ds4_gpu_tensor_write(t, 0, h, n * sizeof(float)), "input upload");
    free(h);
    return t;
}

typedef struct {
    uint8_t *base;
    uint64_t size;
    /* R_Q8 */
    uint64_t q8_off[N_ROWS][MAX_COPIES];
    uint32_t q8_copies[N_ROWS];
    /* experts: [type 0 = Q5_K, 1 = Q4_K][layer copy] -> gate tensor offset; up and down follow */
    uint64_t exp_off[2][N_LAYER_COPIES];
    uint64_t sh_off[N_SHARED_COPIES];    /* shared gate; up and down follow, sh_mat bytes each */
    uint64_t sh_mat;
    ds4_gpu_tensor *sel, *sel_view[N_SETS][2];
    ds4_gpu_tensor *x, *x0, *y, *y0, *mid_in, *mid, *part;
} bench_t;

static uint64_t exp_tensor_bytes(uint32_t type) { return (uint64_t)N_EXP * F * row_bytes(type, E); }
static int type_slot(uint32_t type) { return type == T_Q5_K ? 0 : 1; }

/* random expert sets: row 0 has 8 distinct ids, row 1 keeps row 0's first K_SHARE and adds new ones */
static void make_sets(bench_t *b) {
    int32_t ids[N_SETS * 2u * K_USED];
    for (uint32_t s = 0; s < N_SETS; s++) {
        int32_t *r0 = ids + s * 2u * K_USED, *r1 = r0 + K_USED;
        uint8_t used[N_EXP] = { 0 };
        for (uint32_t i = 0; i < K_USED; i++) {
            int32_t e;
            do { e = (int32_t)(next_rng() % N_EXP); } while (used[e]);
            used[e] = 1; r0[i] = e;
        }
        for (uint32_t i = 0; i < K_USED; i++) {
            if (i < K_SHARE) { r1[i] = r0[i]; continue; }
            int32_t e;
            do { e = (int32_t)(next_rng() % N_EXP); } while (used[e]);
            used[e] = 1; r1[i] = e;
        }
    }
    b->sel = ds4_gpu_tensor_alloc(sizeof(ids));
    need(b->sel && ds4_gpu_tensor_write(b->sel, 0, ids, sizeof(ids)), "expert sets upload");
    for (uint32_t s = 0; s < N_SETS; s++)
        for (uint32_t t = 0; t < 2u; t++) {
            b->sel_view[s][t] = ds4_gpu_tensor_view(b->sel, (uint64_t)s * 2u * K_USED * sizeof(int32_t),
                                                    (uint64_t)(t + 1u) * K_USED * sizeof(int32_t));
            need(b->sel_view[s][t] != NULL, "expert set view");
        }
}

static uint64_t bytes_per_call(int r, uint32_t n_tok) {
    const row_t *w = &ROWS[r];
    const uint64_t distinct = n_tok == 1u ? K_USED : 2u * K_USED - K_SHARE;
    const uint64_t sh = (uint64_t)F * row_bytes(T_Q8_0, E);       /* one shared matrix (same for down) */
    switch (w->kind) {
    case R_Q8: return (uint64_t)w->out_dim * row_bytes(T_Q8_0, w->in_dim);
    case R_MID: return distinct * 2u * F * row_bytes(w->type, E) + 2u * sh;
    case R_DOWN: return distinct * (uint64_t)E * row_bytes(w->type, F) + sh;
    }
    return 0;
}

static int call(bench_t *b, int r, uint32_t n_tok, uint32_t k) {
    const row_t *w = &ROWS[r];
    ds4_gpu_tensor *sel = b->sel_view[k % N_SETS][n_tok - 1u];
    const uint64_t eo = b->exp_off[type_slot(w->type)][k % N_LAYER_COPIES], et = exp_tensor_bytes(w->type);
    const uint64_t so = b->sh_off[k % N_SHARED_COPIES];
    switch (w->kind) {
    case R_Q8: {
        const uint64_t off = b->q8_off[r][k % b->q8_copies[r]];
        return n_tok == 1u
            ? ds4_gpu_qwen4_matmul_q8_0_tensor(b->y0, b->base, b->size, off, w->in_dim, w->out_dim, b->x0, 1u)
            : ds4_gpu_qwen35_matmul_q8_0_rows_tensor(b->y, b->base, b->size, off, w->in_dim, w->out_dim, b->x, 2u);
    }
    case R_MID:
        return w->type == T_Q5_K
            ? ds4_gpu_qwen35_moe_mid_tensor(b->mid, b->x, sel, b->base, b->size, eo, eo + et, w->type, N_EXP,
                                            n_tok, K_USED, E, F, so, so + b->sh_mat, T_Q8_0)
            : ds4_gpu_qwen4_moe_mid_tensor(b->mid, b->x, sel, b->base, b->size, eo, eo + et, w->type, N_EXP,
                                           n_tok, K_USED, E, F, so, so + b->sh_mat, T_Q8_0);
    case R_DOWN:
        return w->type == T_Q5_K
            ? ds4_gpu_qwen35_moe_down_tensor(b->part, b->mid_in, sel, b->base, b->size, eo + 2u * et, w->type,
                                             N_EXP, n_tok, K_USED, F, E, so + 2u * b->sh_mat, T_Q8_0)
            : ds4_gpu_qwen4_moe_down_tensor(b->part, b->mid_in, sel, b->base, b->size, eo + 2u * et, w->type,
                                            N_EXP, n_tok, K_USED, F, E, so + 2u * b->sh_mat, T_Q8_0);
    }
    return 0;
}

/* one repetition: `calls` calls in one command buffer; returns wall ms per
 * call, stores the host encoding ms per call in *enc */
static double one_rep(bench_t *b, int r, uint32_t n_tok, uint32_t calls, double *enc) {
    const double t0 = now_s();
    need(ds4_gpu_begin_commands(), "begin commands");
    for (uint32_t k = 0; k < calls; k++) need(call(b, r, n_tok, k), ROWS[r].name);
    const double t1 = now_s();
    need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
    *enc = (t1 - t0) * 1e3 / calls;
    return (now_s() - t0) * 1e3 / calls;
}
static void sort_reps(double *t) {
    for (int i = 1; i < REPS; i++)
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
}
/* GPU ms per call: median wall minus median encode (the command buffer runs
 * after the last call is encoded), after WARM_S of untimed repetitions */
static double gpu_ms(bench_t *b, int r, uint32_t n_tok, uint32_t calls, double *enc_ms) {
    double t[REPS], e[REPS], dummy;
    for (const double w0 = now_s(); now_s() - w0 < WARM_S;) (void)one_rep(b, r, n_tok, calls, &dummy);
    for (int i = 0; i < REPS; i++) t[i] = one_rep(b, r, n_tok, calls, &e[i]);
    sort_reps(t);
    sort_reps(e);
    *enc_ms = e[REPS / 2];
    return t[REPS / 2] - e[REPS / 2];
}

static int smoke_check(ds4_gpu_tensor *t, uint64_t n, const char *name, uint32_t n_tok) {
    float *h = malloc(n * sizeof(float));
    need(h && ds4_gpu_tensor_read(t, 0, h, n * sizeof(float)), "smoke readback");
    float mx = 0.0f;
    int ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(h[i])) { ok = 0; break; }
        if (fabsf(h[i]) > mx) mx = fabsf(h[i]);
    }
    if (mx == 0.0f) ok = 0;
    printf("smoke: %-36s T%u %s (max |y| = %g)\n", name, n_tok, ok ? "ok" : "FAIL", (double)mx);
    free(h);
    return ok;
}

int main(int argc, char **argv) {
    const bool smoke = argc > 1 && strcmp(argv[1], "smoke") == 0;
    uint64_t total = 0;
    for (int r = 0; r < N_ROWS; r++)
        if (ROWS[r].kind == R_Q8) {
            const uint64_t by = bytes_per_call(r, 1u);
            total += ((by + 63u) & ~63ull) * copies_for(by);
        }
    total += (uint64_t)N_LAYER_COPIES * 3u * (exp_tensor_bytes(T_Q5_K) + exp_tensor_bytes(T_Q4_K));
    const uint64_t sh_mat = (uint64_t)F * row_bytes(T_Q8_0, E);
    total += (uint64_t)N_SHARED_COPIES * 3u * sh_mat;
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    need(base != MAP_FAILED, "weight arena mmap");
    bench_t b = { .base = base, .size = total, .sh_mat = sh_mat };
    uint64_t used = 0;
    for (int r = 0; r < N_ROWS; r++) {
        if (ROWS[r].kind != R_Q8) continue;
        const uint64_t by = bytes_per_call(r, 1u);
        b.q8_copies[r] = copies_for(by);
        for (uint32_t c = 0; c < b.q8_copies[r]; c++) {
            b.q8_off[r][c] = used;
            fill_weights(base + used, by, T_Q8_0);
            used += (by + 63u) & ~63ull;
        }
    }
    const uint32_t types[2] = { T_Q5_K, T_Q4_K };
    for (int t = 0; t < 2; t++)
        for (uint32_t c = 0; c < N_LAYER_COPIES; c++) {
            b.exp_off[t][c] = used;
            fill_weights(base + used, 3u * exp_tensor_bytes(types[t]), types[t]);
            used += 3u * exp_tensor_bytes(types[t]);
        }
    for (uint32_t c = 0; c < N_SHARED_COPIES; c++) {
        b.sh_off[c] = used;
        fill_weights(base + used, 3u * sh_mat, T_Q8_0);
        used += 3u * sh_mat;
    }
    need(used <= total, "arena layout");
    need(ds4_gpu_init(), "GPU initialization");
    need(ds4_gpu_set_model_map(base, total), "model map registration");
    make_sets(&b);
    const uint32_t max_in = 4096u, max_out = 248320u;
    b.x = rand_tensor(2ull * max_in);
    b.x0 = ds4_gpu_tensor_view(b.x, 0, (uint64_t)max_in * sizeof(float));
    b.y = ds4_gpu_tensor_alloc(2ull * max_out * sizeof(float));
    b.y0 = ds4_gpu_tensor_view(b.y, 0, (uint64_t)max_out * sizeof(float));
    b.mid_in = rand_tensor(2ull * (K_USED + 1u) * F);
    b.mid = ds4_gpu_tensor_alloc(2ull * (K_USED + 1u) * F * sizeof(float));
    b.part = ds4_gpu_tensor_alloc(2ull * (K_USED + 1u) * E * sizeof(float));
    need(b.x0 && b.y && b.y0 && b.mid && b.part, "work tensors");

    if (smoke) {
        int ok = 1;
        for (int r = 0; r < N_ROWS; r++)
            for (uint32_t n_tok = 1u; n_tok <= 2u; n_tok++) {
                need(ds4_gpu_begin_commands(), "begin commands");
                need(call(&b, r, n_tok, 0u), ROWS[r].name);
                need(ds4_gpu_end_commands() && ds4_gpu_synchronize(), "end commands");
                const row_t *w = &ROWS[r];
                ds4_gpu_tensor *out = w->kind == R_Q8 ? b.y : w->kind == R_MID ? b.mid : b.part;
                const uint64_t n = w->kind == R_Q8 ? (uint64_t)n_tok * w->out_dim
                                                   : (uint64_t)n_tok * (K_USED + 1u) * w->out_dim;
                ok = smoke_check(out, n, w->name, n_tok) && ok;
            }
        printf("smoke: %s\n", ok ? "ok" : "FAIL");
        return ok ? 0 : 1;
    }

    double ms[N_ROWS][2], peak = 0.0;
    for (int r = 0; r < N_ROWS; r++)
        for (uint32_t n_tok = 1u; n_tok <= 2u; n_tok++) {
            const uint64_t by = bytes_per_call(r, n_tok);
            const uint64_t c = (REP_BYTES + by - 1u) / by;
            const uint32_t calls = c < MIN_CALLS ? MIN_CALLS : c > MAX_CALLS ? MAX_CALLS : (uint32_t)c;
            double enc = 0.0;
            const double g = gpu_ms(&b, r, n_tok, calls, &enc);
            ms[r][n_tok - 1u] = g;
            const double gbs = (double)by / (g * 1e6);
            if (r == 0 && n_tok == 1u) peak = gbs;
            printf("bench: %-36s T%u gpu %8.4f ms %7.1f GB/s %5.1f%% of peak (encode %.1f us/call, %u calls/rep)\n",
                   ROWS[r].name, n_tok, g, gbs, 100.0 * gbs / peak, enc * 1e3, calls);
        }
    printf("bench: peak-ref %.1f GB/s (lm_head T1)\n", peak);
    double step1 = 0.0, step2 = 0.0;
    const double bar = CANDIDATE_SHARE * CYCLE_MS_2K;
    for (int r = 0; r < N_ROWS; r++) {
        const double floor2 = (double)bytes_per_call(r, 2u) / (peak * 1e6);
        const double gap = ms[r][1] > floor2 ? ms[r][1] - floor2 : 0.0;
        const double save = 0.5 * gap * ROWS[r].n_step;
        step1 += ms[r][0] * ROWS[r].n_step;
        step2 += ms[r][1] * ROWS[r].n_step;
        printf("bench: family %-36s n=%2u T2 %.4f ms floor %.4f ms half-gap %.2f ms/cycle %s\n",
               ROWS[r].name, ROWS[r].n_step, ms[r][1], floor2, save, save >= bar ? "CANDIDATE" : "-");
    }
    printf("bench: step model T1 %.2f ms, T2 %.2f ms (projections + routed/shared experts only; bar %.2f ms)\n",
           step1, step2, bar);
    return 0;
}
