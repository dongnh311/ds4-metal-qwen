/* Row dequantizers for the GSQ-RCO quant types, ported from ggml-quants.c
 * (ggml-org/llama.cpp@931351ea, Copyright (c) 2023-2026 The ggml authors,
 * MIT License). Included by ds4.c after the IQ2 tables (kmask_iq2xs,
 * ksigns_iq2xs, iq2xxs_grid), ds4_iq_tables.h and f16_to_f32. n is a whole
 * number of blocks; every read is a byte read, so rows need no alignment. */
#pragma once

static float dq_half(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return f16_to_f32(h);
}

static void dq_iq2_xxs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 66u) {
        const float d = dq_half(p);
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            uint16_t q2[4];
            memcpy(q2, p + 2 + ib32 * 8u, 8);
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            for (uint32_t j = 0; j < 4u; j++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + ((aux_g >> (8u * j)) & 0xFFu));
                const uint32_t signs = ksigns_iq2xs[(aux_s >> (7u * j)) & 127u];
                for (uint32_t i = 0; i < 8u; i++) *y++ = dl * (float)grid[i] * (((signs >> i) & 1u) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq2_xs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 74u) {
        const float d = dq_half(p);
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const uint8_t sc = p[66u + ib32];
            const float db[2] = { d * (0.5f + (float)(sc & 0xfu)) * 0.25f, d * (0.5f + (float)(sc >> 4)) * 0.25f };
            for (uint32_t l = 0; l < 4u; l++) {
                const uint32_t o = 2u + 2u * (4u * ib32 + l);
                const uint32_t q = (uint32_t)p[o] | ((uint32_t)p[o + 1u] << 8);
                const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q & 511u));
                const uint8_t signs = ksigns_iq2xs[q >> 9];
                for (uint32_t j = 0; j < 8u; j++) *y++ = db[l / 2u] * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq2_s(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 82u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *signs = p + 34u, *qh = p + 66u, *scales = p + 74u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const float db[2] = { d * (0.5f + (float)(scales[ib32] & 0xfu)) * 0.25f,
                                  d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f };
            for (uint32_t l = 0; l < 4u; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + (qs[4u * ib32 + l] | ((qh[ib32] << (8u - 2u * l)) & 0x300u)));
                const uint8_t s = signs[4u * ib32 + l];
                for (uint32_t j = 0; j < 8u; j++) *y++ = db[l / 2u] * (float)grid[j] * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
        }
    }
}

static void dq_iq3_xxs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 98u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *ss = p + 66u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, ss + 4u * ib32, 4);
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (uint32_t l = 0; l < 4u; l++, y += 8) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7u * l)) & 127u];
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[8u * ib32 + 2u * l]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[8u * ib32 + 2u * l + 1u]);
                for (uint32_t j = 0; j < 4u; j++) {
                    y[j] = db * (float)g1[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4u] = db * (float)g2[j] * ((signs & kmask_iq2xs[j + 4u]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

static void dq_iq3_s(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 110u) {
        const float d = dq_half(p);
        const uint8_t *qs = p + 2u, *qh = p + 66u, *signs = p + 74u, *scales = p + 106u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32++) {
            const float db = d * (float)(1 + 2 * (int)((scales[ib32 / 2u] >> (4u * (ib32 & 1u))) & 0xfu));
            for (uint32_t l = 0; l < 4u; l++, y += 8) {
                const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[8u * ib32 + 2u * l] | ((qh[ib32] << (8u - 2u * l)) & 256u)));
                const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[8u * ib32 + 2u * l + 1u] | ((qh[ib32] << (7u - 2u * l)) & 256u)));
                const uint8_t s = signs[4u * ib32 + l];
                for (uint32_t j = 0; j < 4u; j++) {
                    y[j] = db * (float)g1[j] * ((s & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4u] = db * (float)g2[j] * ((s & kmask_iq2xs[j + 4u]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

static void dq_iq4_nl(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 32u; b++, p += 18u, y += 32) {
        const float d = dq_half(p);
        for (uint32_t j = 0; j < 16u; j++) {
            y[j] = d * (float)kvalues_iq4nl[p[2u + j] & 0xfu];
            y[j + 16u] = d * (float)kvalues_iq4nl[p[2u + j] >> 4];
        }
    }
}

static void dq_iq4_xs(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 136u) {
        const float d = dq_half(p);
        const uint32_t scales_h = (uint32_t)p[2] | ((uint32_t)p[3] << 8);
        const uint8_t *qs = p + 8u;
        for (uint32_t ib = 0; ib < 8u; ib++, qs += 16, y += 32) {
            const int ls = (int)(((p[4u + ib / 2u] >> (4u * (ib % 2u))) & 0xfu) | (((scales_h >> (2u * ib)) & 3u) << 4));
            const float dl = d * (float)(ls - 32);
            for (uint32_t j = 0; j < 16u; j++) {
                y[j] = dl * (float)kvalues_iq4nl[qs[j] & 0xfu];
                y[j + 16u] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
        }
    }
}

static void dq_q2_0(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 64u; b++, p += 18u, y += 64) {
        const float d = dq_half(p);
        for (uint32_t j = 0; j < 64u; j++) {
            y[j] = (float)((int)((p[2u + j / 4u] >> ((j % 4u) * 2u)) & 3u) - 1) * d;
        }
    }
}

static void dq_q5_k(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 176u) {
        const float d = dq_half(p), dmin = dq_half(p + 2);
        const uint8_t *sc = p + 4u, *qh = p + 16u;
        for (uint32_t g = 0; g < 8u; g++) {
            uint32_t s, m;
            if (g < 4u) { s = sc[g] & 63u; m = sc[g + 4u] & 63u; }
            else { s = (sc[g + 4u] & 0xfu) | ((sc[g - 4u] >> 6) << 4); m = (sc[g + 4u] >> 4) | ((sc[g] >> 6) << 4); }
            const float dl = d * (float)s, ml = dmin * (float)m;
            const uint8_t *ql = p + 48u + 32u * (g / 2u);
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t lo = (g & 1u) ? (uint32_t)(ql[l] >> 4) : (uint32_t)(ql[l] & 0xfu);
                *y++ = dl * (float)(lo + (((qh[l] >> g) & 1u) ? 16u : 0u)) - ml;
            }
        }
    }
}

static void dq_q6_k(const uint8_t *p, uint64_t n, float *y) {
    for (uint64_t b = 0; b < n / 256u; b++, p += 210u, y += 256) {
        const float d = dq_half(p + 208u);
        for (uint32_t h = 0; h < 2u; h++) {
            const uint8_t *ql = p + 64u * h, *qh = p + 128u + 32u * h;
            const int8_t *sc = (const int8_t *)(p + 192u + 8u * h);
            float *yy = y + 128u * h;
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t is = l / 16u;
                const int q1 = (int)((ql[l] & 0xfu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int q2 = (int)((ql[l + 32u] & 0xfu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int q4 = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                yy[l] = d * (float)sc[is] * (float)q1;
                yy[l + 32u] = d * (float)sc[is + 2u] * (float)q2;
                yy[l + 64u] = d * (float)sc[is + 4u] * (float)q3;
                yy[l + 96u] = d * (float)sc[is + 6u] * (float)q4;
            }
        }
    }
}
