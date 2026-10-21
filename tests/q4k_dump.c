/*
 * q4_K reference dequant dump for the requant validation: reads q4_K block bytes on stdin
 * (multiples of 144), prints float32 dequanted values on stdout in canonical ggml order.
 * Shipped functions (block struct, f16_to_f32, q4_k_get_scale_min) come from the repo-verified
 * tests/test_q4k_dot.c so this is the same convention the engine lane and fixture use.
 * Build: cc -O2 -Itests -o tests/q4k_dump tests/q4k_dump.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define QK_K 256

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K_dump;

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    if ((h & 0x7C00) == 0) { /* zero or denormal: exact, 10-bit frac * 2^-24 */
        float v = ldexpf((float)(h & 0x03FF), -24);
        return sign ? -v : v;
    }
    if ((h & 0x7C00) == 0x7C00) { /* inf/nan */
        uint32_t f = sign | 0x7F800000u | ((uint32_t)(h & 0x03FF) << 13);
        float out;
        memcpy(&out, &f, sizeof(out));
        return out;
    }
    uint32_t f = sign | (((uint32_t)((h >> 10) & 0x1F) + 127 - 15) << 23) | ((uint32_t)(h & 0x03FF) << 13);
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

static inline void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

int main(void) {
    static block_q4_K_dump blocks[4096];
    size_t want = 144;
    size_t nb = 0;
    for (;;) {
        size_t r = fread((char *)&blocks[nb], 1, want, stdin);
        if (r == 0) break;
        if (r != want) { fprintf(stderr, "partial block read\n"); return 1; }
        nb++;
    }
    for (size_t i = 0; i < nb; i++) {
        const uint8_t *q = blocks[i].qs;
        const float d = f16_to_f32(blocks[i].d);
        const float min = f16_to_f32(blocks[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            q4_k_get_scale_min(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            q4_k_get_scale_min(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) { float v = d1 * (q[l] & 0xF) - m1; fwrite(&v, 4, 1, stdout); }
            for (int l = 0; l < 32; ++l) { float v = d2 * (q[l]  >> 4) - m2; fwrite(&v, 4, 1, stdout); }
            q += 32;
            is += 2;
        }
    }
    return 0;
}
