// ds41_q5k_dequant.c — reference ggml Q5_K -> F16 dequant (little endian),
// used to rebuild blk.*.engram_kv.weight from the ISTA RSC package's
// engram_wkv.weight (Q5_K) into the layout ds4's loader expects (F16).
//
// block_q5_K layout (ggml, 176 bytes / 256 values):
//   0  : fp16 d
//   2  : fp16 dmin
//   4  : uint8 scales[12]
//   16 : uint8 qh[32]
//   48 : uint8 qs[128]
#include <stdint.h>
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

static uint16_t f32_to_f16(float f) {
    uint32_t w;
    memcpy(&w, &f, 4);
    uint32_t sign = (w >> 16) & 0x8000u;
    int32_t exp = (w >> 23) & 0xFF;
    uint32_t man = w & 0x7FFFFFu;
    if (exp == 0xFF)
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    int32_t e = exp - 127 + 15;
    if (e >= 31)
        return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10)
            return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - e;
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u)))
            half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = ((uint32_t)e << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        half++;
    return (uint16_t)(sign | half);
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                             uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* src: n/256 blocks of 176 B; dst: n little-endian f16 values. */
int ds41_q5k_to_f16(const uint8_t *src, uint8_t *dst, uint64_t n_elements) {
    if (!src || !dst || (n_elements % 256u))
        return -1;
    const uint64_t nb = n_elements / 256u;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *b = src + i * 176u;
        uint16_t dh, dmh;
        memcpy(&dh, b, 2);
        memcpy(&dmh, b + 2, 2);
        const float d = f16_to_f32(dh);
        const float dmin = f16_to_f32(dmh);
        const uint8_t *sc = b + 4;
        const uint8_t *qh = b + 16;
        const uint8_t *ql = b + 48;
        uint16_t *out = (uint16_t *)(dst + i * 512u);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 4; j++) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4(is + 0, sc, &s1, &m1);
            get_scale_min_k4(is + 1, sc, &s2, &m2);
            const float d1 = d * (float)s1, dm1 = dmin * (float)m1;
            const float d2 = d * (float)s2, dm2 = dmin * (float)m2;
            for (int l = 0; l < 32; l++) {
                float v = d1 * (float)((ql[l] & 15u) +
                                       ((qh[l] & u1) ? 16u : 0u)) - dm1;
                out[64 * j + l] = f32_to_f16(v);
            }
            for (int l = 0; l < 32; l++) {
                float v = d2 * (float)((ql[l] >> 4) +
                                       ((qh[l] & u2) ? 16u : 0u)) - dm2;
                out[64 * j + 32 + l] = f32_to_f16(v);
            }
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
    return 0;
}

/* bf16 -> f32, n elements; dst little-endian f32. */
void ds41_bf16_to_f32(const uint8_t *src, uint8_t *dst, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[2 * i] << 16 | (uint32_t)src[2 * i + 1] << 24;
        memcpy(dst + 4 * i, &bits, 4);
    }
}