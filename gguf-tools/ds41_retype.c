// ds41_retype.c — retype/requantize GGUF tensor payloads for the ds4 V4.1
// loader.  Source formats come from the ISTA RSC package (iq1_m, iq2_xs,
// q5_K, bf16, f32); targets must be ds4-legal (f16, q8_0, iq2_xxs).
//
// Dequantizers other than iq1_m live in ds4_quants.h (ported ggml); the
// iq1_m path is a line-for-line port of llama.cpp's
// dequantize_row_iq1_m (ggml-quants.c) with iq1s_grid extracted from
// cuda/mmq/ggml-common.h.  Quantizers come from quants.c (ds4q_*).
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (!man) {
            bits = sign;
        } else {
            exp = 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    }
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

#include "ds4_iq_tables.h"
#include "ds41_iq2_tables.h"
#include "ds4_quants.h"
#include "quants.h"
#include "ds41_iq1s_grid.h"

#define IQ1S_DELTA 0.125f
#include <stdio.h>
static int ds41_dbg = 0;

/* Port of dequantize_row_iq1_m (llama.cpp ggml-quants.c). */
static void dq_iq1_m(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t blk = 0; blk < n / 256u; blk++, p += 56u) {
        if (ds41_dbg && blk >= 1990)
            fprintf(stderr, "[dq_iq1_m] blk %llu p=%p y=%p\n",
                    (unsigned long long)blk, (void *)p, (void *)y);
        const uint8_t *qs = p;
        const uint8_t *qh = p + 32u;
        uint16_t sc[4];
        memcpy(sc, p + 48u, 8);
        const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) |
                                       ((sc[2] >> 4) & 0x0f00u) |
                                       (sc[3] & 0xf000u));
        const float d = f16_to_f32(su);
        for (int ib = 0; ib < 8; ib++) {
            const float dl1 =
                d * (2.0f * (float)((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7u) + 1.0f);
            const float dl2 =
                d * (2.0f * (float)((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7u) + 1.0f);
            uint16_t idx[4];
            idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700u));
            idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700u));
            idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700u));
            idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700u));
            const float delta[4] = {
                (qh[0] & 0x08u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[0] & 0x80u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[1] & 0x08u) ? -IQ1S_DELTA : IQ1S_DELTA,
                (qh[1] & 0x80u) ? -IQ1S_DELTA : IQ1S_DELTA,
            };
            for (int l = 0; l < 2; l++) {
                const int8_t *grid = (const int8_t *)(ds41_iq1s_grid + idx[l]);
                for (int j = 0; j < 8; j++)
                    y[j] = dl1 * ((float)grid[j] + delta[l]);
                y += 8;
            }
            for (int l = 2; l < 4; l++) {
                const int8_t *grid = (const int8_t *)(ds41_iq1s_grid + idx[l]);
                for (int j = 0; j < 8; j++)
                    y[j] = dl2 * ((float)grid[j] + delta[l]);
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

/* ggml type codes (must match the merged GGUF's tensor types). */
enum {
    K_F32 = 0, K_F16 = 1, K_Q2K = 10, K_Q8_0 = 8, K_Q5K = 13, K_IQ2XXS = 16,
    K_IQ2XS = 17, K_IQ1M = 29, K_BF16 = 30,
};

int64_t ds41_type_size(int kind, int64_t ne) {
    switch (kind) {
    case K_F32:    return ne * 4;
    case K_F16:    return ne * 2;
    case K_BF16:   return ne * 2;
    case K_Q2K:    return (ne / 256) * 84;
    case K_Q8_0:   return (ne / 32) * 34;
    case K_Q5K:    return (ne / 256) * 176;
    case K_IQ2XXS: return (ne / 256) * 66;
    case K_IQ2XS:  return (ne / 256) * 74;
    case K_IQ1M:   return (ne / 256) * 56;
    }
    return -1;
}

static void dequant_any(const uint8_t *src, int kind, int64_t ne, float *y) {
    switch (kind) {
    case K_F32:  memcpy(y, src, (size_t)ne * 4);
                 return;
    case K_F16:  for (int64_t i = 0; i < ne; i++)
                     y[i] = f16_to_f32(((const uint16_t *)src)[i]);
                 return;
    case K_BF16: for (int64_t i = 0; i < ne; i++)
                     y[i] = ds4q_bf16_to_f32(((const uint16_t *)src)[i]);
                 return;
    case K_Q5K:  dq_q5_k(src, (uint64_t)ne, y);
                 return;
    case K_IQ2XS: dq_iq2_xs(src, (uint64_t)ne, y);
                 return;
    case K_IQ2XXS: dq_iq2_xxs(src, (uint64_t)ne, y);
                 return;
    case K_IQ1M: dq_iq1_m(src, (uint64_t)ne, y);
                 return;
    default:     break;
    }
}

/* Convert `ne` elements laid out as `rows` x `ncols` (ncols = ne0).
 * scratch must hold ne floats.  Returns 0 on success. */
int ds41_retype(const uint8_t *src, int src_kind, uint8_t *dst, int dst_kind,
                int64_t ne, int64_t ncols, float *scratch) {
    if (!src || !dst || !scratch || ne <= 0 || ncols <= 0 || ne % ncols)
        return -1;
    ds41_dbg = getenv("DS41_DBG") != NULL;
    if (ds41_dbg)
        fprintf(stderr, "[retype] kind %d->%d ne=%lld ncols=%lld\n",
                src_kind, dst_kind, (long long)ne, (long long)ncols);
    dequant_any(src, src_kind, ne, scratch);
    if (ds41_dbg) fprintf(stderr, "[retype] dequant done\n");
    switch (dst_kind) {
    case K_F32:
        memcpy(dst, scratch, (size_t)ne * 4);
        return 0;
    case K_F16:
        ds4q_f32_to_f16_row(scratch, (uint16_t *)dst, ne);
        return 0;
    case K_BF16:
        ds4q_f32_to_bf16_row(scratch, (uint16_t *)dst, ne);
        return 0;
    case K_Q2K:
        if (ncols % 256) return -2;
        ds4q_quantize_init(DS4Q_TYPE_Q2_K);
        ds4q_quantize_chunk(DS4Q_TYPE_Q2_K, scratch, dst, 0,
                            ne / ncols, ncols, NULL);
        return 0;
    case K_Q8_0:
        if (ncols % 32) return -2;
        ds4q_quantize_init(DS4Q_TYPE_Q8_0);
        ds4q_quantize_chunk(DS4Q_TYPE_Q8_0, scratch, dst, 0,
                            ne / ncols, ncols, NULL);
        return 0;
    case K_IQ2XXS: {
        if (ncols % 256) return -2;
        const int64_t rows = ne / ncols;
        float *imat = malloc((size_t)ncols * sizeof(float));
        if (!imat) return -3;
        for (int64_t c = 0; c < ncols; c++) imat[c] = 0.0f;
        for (int64_t r = 0; r < rows; r++) {
            const float *row = scratch + r * ncols;
            for (int64_t c = 0; c < ncols; c++) imat[c] += row[c] * row[c];
        }
        ds4q_quantize_init(DS4Q_TYPE_IQ2_XXS);
        ds4q_quantize_chunk(DS4Q_TYPE_IQ2_XXS, scratch, dst, 0, rows, ncols,
                            imat);
        free(imat);
        return 0;
    }
    default:
        break;
    }
    return -4;
}