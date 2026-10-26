/*
 * deepseek41_ista_core.c — conversion kernel for gguf-tools/
 * deepseek41_ista_merge.py. It turns llama.cpp IQ4_NL Engram rows into
 * the ds4 fork's lloyd4 row136 rows using the fork's own encoder.
 *
 * Placement contract (mirrors ds4_engram.c):
 *   IQ4_NL row layout: 8 blocks of 18 bytes per 256-value row.
 *     block = [uint16 d (fp16, little-endian)][16 bytes nibbles]
 *     value[j] = d * kvalues_iq4nl[nibble], low nibble first.
 *   Intermediate e4m3_e8m0_32_row264 row:
 *     work[0..255] = e4m3 codes (bit 7 = sign); work[256+g] = e8m0
 *     scale byte for value group g (decoded value = e4m3(c)·2^(s-127)).
 *     The engine rejects codes with (code & 127) == 127 and byte 255,
 *     so the packer never emits those; zero groups keep code 0 and
 *     scale 0. Scale choice: 2^e2 with amax = m·2^e2, m in [0.5, 1)
 *     (frexp), leaving all codes below 0x7f by construction.
 *   Output: fork's lloyd4 row136 layout via ds4_engram_encode_row136()
 *     (linked from ds4_engram.c), so decoding stays engine-exact.
 *
 * Build:
 *   cc -O2 -fPIC -shared -o libds41ista.dylib deepseek41_ista_core.c \
 *        ../ds4_engram.c -lm
 */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4_engram.h"
#include "../ds4_iq_tables.h"

#define ROW144 144
#define ROW264 264

static float fp16_get(uint16_t h) {
    uint32_t sign = h >> 15, exp = (h >> 10) & 31, man = h & 1023;
    float v;
    if (exp == 31) v = man ? NAN : (float)INFINITY;
    else if (exp == 0) v = man ? ldexpf((float)man, -24) : 0.0f;
    else v = ldexpf((float)(1024 + man), (int)exp - 25);
    return sign ? -v : v;
}

/* decode |code| exactly like ds4_engram.c's e4m3(); only used to guard */
static float e4m3_decodeMag(uint8_t code) {
    int exp = (code >> 3) & 15, man = code & 7;
    return exp ? ldexpf((float)(8 + man), exp - 10) : ldexpf((float)man, -9);
}

/* encode |q| into an e4m3 code on the value ladder ... the ladder is:
 *   subnormals 0..7·2^-9 and the first normal band share the 2^-9 step;
 *   exp-field e0 covers (8..15)·2^(e0-10); rounding is RNE (rintf), ties
 *   at band boundaries carry upward. The emitted code never has
 *   (code & 127) == 127, which the row136 encoder rejects.            */
static uint8_t e4m3_encodeMag(float q) {
    if (!(q >= 0.0f) || q > 448.0f) return 0xff;   /* caller guards     */
    if (q == 0.0f) return 0;
    int e2;
    (void)frexpf(q, &e2);                          /* q = m·2^e2        */
    int e0 = e2 + 6;
    if (e0 < 1) {
        float k = rintf(q * 512.0f);               /* grid of 2^-9      */
        if (k <= 15.0f) return (uint8_t)(int)k;    /* 0..7 sub, 8..15 n */
        return (uint8_t)((2u << 3) | 0u);          /* 16·2^-9 next rung */
    }
    float qp = q * ldexpf(1.0f, 10 - e0);          /* qp in [8, 16)     */
    float man = rintf(qp) - 8.0f;
    if (man >= 8.0f) { man = 0.0f; e0 += 1; }      /* band carry        */
    return (uint8_t)((e0 << 3) | (int)man);
}

/* convert n IQ4_NL rows (n*144 bytes at src) into n lloyd4 rows136
 * (n*DS4_ENGRAM_Q4_ROW_BYTES at dst). Returns 0 on success, errno-style
 * negatives on failure (details printed by the caller).               */
int ds41_ista_convert_chunk(const uint8_t *src, uint8_t *dst, uint64_t n) {
    uint8_t work[ROW264];
    for (uint64_t r = 0; r < n; r++) {
        const uint8_t *s = src + r * ROW144;
        uint8_t *d = dst + r * (uint64_t)DS4_ENGRAM_Q4_ROW_BYTES;
        float v[256];
        for (int g = 0; g < 8; g++) {
            const uint8_t *p = s + 18 * g;
            float scale = fp16_get((uint16_t)(p[0] | (p[1] << 8)));
            if (!isfinite(scale)) return -(30 + r % 2);   /* NaN/inf fp16 */
            for (int j = 0; j < 16; j++) {
                v[32 * g + j]      = scale * (float)kvalues_iq4nl[p[2 + j] & 0xf];
                v[32 * g + 16 + j] = scale * (float)kvalues_iq4nl[p[2 + j] >> 4];
            }
        }
        memset(work, 0, ROW264);
        for (int g = 0; g < 8; g++) {
            const float *gv = v + 32 * g;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = fabsf(gv[j]);
                if (a > amax) amax = a;
            }
            uint8_t sc = 0;
            if (amax > 0.0f) {
                int e2;
                (void)frexpf(amax, &e2);
                if (e2 < -127) e2 = -127;
                if (e2 > 126) return -(40 + r % 9);
                sc = (uint8_t)(e2 + 127);
                float inv = ldexpf(1.0f, -e2);
                for (int j = 0; j < 32; j++) {
                    float q = fabsf(gv[j]) * inv;
                    uint8_t code = e4m3_encodeMag(q);
                    if ((code & 127) == 127 ||
                        !isfinite(e4m3_decodeMag(code)))
                        return -(50 + r % 9);
                    work[32 * g + j] =
                        (uint8_t)(code | (signbit(gv[j]) ? 128 : 0));
                }
            }
            work[256 + g] = sc;
        }
        if (!ds4_engram_encode_row136(work, d)) return -2;
    }
    return 0;
}
