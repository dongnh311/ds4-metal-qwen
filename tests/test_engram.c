#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "ds4_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ds4_engram_layout layout(void) {
    static uint32_t map[256];
    for (int i = 0; i < 256; i++) map[i] = i / 2;
    ds4_engram_layout l = {.token_map = map, .vocab_size = 256,
                           .compressed_vocab_size = 128, .pad_id = 1};
    for (int layer = 0; layer < 2; layer++) {
        for (int j = 0; j < 4; j++)
            l.multipliers[layer][j] = 35184372088831ull - 2 * (j + 4 * layer);
        for (int j = 0; j < 24; j++) {
            l.primes[layer][j] = 16000057;
            l.rows[layer] += l.primes[layer][j];
        }
    }
    return l;
}

/* Independent full-history formulation: no rolling state or reuse of hashes. */
static void reference(const ds4_engram_layout *l, const int *tokens,
                      const uint8_t *mask, int count, uint32_t *out) {
    for (int pos = 0; pos < count; pos++) {
        for (int layer = 0; layer < 2; layer++) {
            uint32_t offset = 0;
            for (int col = 0; col < 24; col++) {
                __uint128_t hash = 0;
                bool blocked = false;
                for (int shift = 0; shift < col / 8 + 2; shift++) {
                    int p = pos - shift;
                    blocked |= p < 0 || (mask && !mask[p]);
                    uint32_t id = blocked ? l->pad_id : l->token_map[tokens[p]];
                    hash ^= (__uint128_t)id * l->multipliers[layer][shift];
                }
                *out++ = (uint32_t)(hash % l->primes[layer][col]) + offset;
                offset += l->primes[layer][col];
            }
        }
    }
}

static void test_hash(void) {
    enum {N = 513, WIDTH = 48};
    ds4_engram_layout l = layout();
    assert(ds4_engram_layout_valid(&l));
    ds4_engram_layout bad = l;
    bad.multipliers[1][3] = UINT64_MAX;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.rows[0]--;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.primes[0][0] = 0;
    assert(!ds4_engram_layout_valid(&bad));
    int tokens[N];
    uint8_t mask[N];
    uint32_t expected[N * WIDTH], actual[N * WIDTH];
    for (int i = 0; i < N; i++) {
        tokens[i] = (i * 97 + i / 3) % 256;
        mask[i] = (i % 17 != 0 && (i < 125 || i > 131));
    }
    for (int masked = 0; masked < 2; masked++) {
        const uint8_t *m = masked ? mask : NULL;
        reference(&l, tokens, m, N, expected);
        for (int chunk = 1; chunk <= N; chunk++) {
            ds4_engram_history h;
            ds4_engram_history_reset(&h);
            for (int i = 0; i < N; i += chunk) {
                int n = N - i < chunk ? N - i : chunk;
                ds4_engram_history snapshot = h;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                ds4_engram_history after = h;
                h = snapshot;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                assert(memcmp(&h, &after, sizeof(h)) == 0);
            }
            assert(memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    ds4_engram_history h, before;
    ds4_engram_history_reset(&h);
    before = h;
    int invalid[] = {0, 256};
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(memcmp(&h, &before, sizeof(h)) == 0);
    invalid[1] = -1;
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(ds4_engram_hash(&l, &h, NULL, NULL, 0, NULL));
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, SIZE_MAX, actual));
    h.tail[0] = 128;
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, 1, actual));
}

static void test_rows(void) {
    char path[] = "/tmp/ds4-engram-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    /* Exercise offsets beyond 32 bits without allocating a large file. */
    const uint64_t offset = (1ull << 33) + 32;
    uint8_t raw[3][DS4_ENGRAM_ROW_BYTES];
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 256; i++) raw[r][i] = i;
        raw[r][127] = 0;
        raw[r][255] = 128;
        for (int i = 0; i < 8; i++) raw[r][256 + i] = 126 + r;
    }
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    ds4_engram_table t;
    assert(ds4_engram_table_open(&t, path, offset, 3, DS4_ENGRAM_ENC_E4M3_ROW264));
    assert(fcntl(t.fd, F_GETFD) & FD_CLOEXEC);
    uint32_t rows[] = {2, 0, 2, 1};
    float out[4 * 256];
    assert(ds4_engram_read(&t, rows, 4, out));
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 256; i++) {
            int code = raw[rows[r]][i], exponent = (code >> 3) & 15;
            double v = exponent ? (1 + (code & 7) / 8.) * pow(2., exponent - 7) :
                                  (code & 7) / 512.;
            if (code & 128) v = -v;
            v *= pow(2., (int)rows[r] - 1);
            assert(out[r * 256 + i] == (float)v);
            if (!v) assert(!!signbit(out[r * 256 + i]) == !!(code & 128));
        }
    }
    enum { TOKENS = 2051, STRIDE = 48, WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };
    uint32_t *batch_ids = malloc((size_t)TOKENS * STRIDE * sizeof(*batch_ids));
    float *batch = malloc(((size_t)TOKENS * WIDTH + 1) * sizeof(*batch));
    assert(batch_ids && batch);
    for (size_t i = 0; i < (size_t)TOKENS * STRIDE; i++) batch_ids[i] = UINT32_MAX;
    for (size_t i = 0; i < TOKENS; i++)
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++)
            batch_ids[i * STRIDE + j] = (i * 7 + j * 11) % 3;
    const size_t sizes[] = {1, 2, 31, 65, 257, 2047, 2048, 2049, TOKENS};
    float expected[3][DS4_ENGRAM_DIM];
    const uint32_t all_rows[] = {0, 1, 2};
    assert(ds4_engram_read(&t, all_rows, 3, &expected[0][0]));
    for (size_t n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        size_t count = sizes[n];
        batch[count * WIDTH] = 123456.0f;
        assert(ds4_engram_read_batch(&t, batch_ids, count, STRIDE, batch));
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
                assert(memcmp(batch + i * WIDTH + j * DS4_ENGRAM_DIM,
                    expected[batch_ids[i * STRIDE + j]], sizeof(expected[0])) == 0);
            }
        }
        assert(batch[count * WIDTH] == 123456.0f);
    }
    assert(ds4_engram_read_batch(&t, NULL, 0, 0, NULL));
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, 23, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, 2, SIZE_MAX, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, SIZE_MAX, STRIDE, batch) && errno == EINVAL);
    batch_ids[(TOKENS - 1) * STRIDE] = 3;
    batch[0] = 123456.0f;
    assert(!ds4_engram_read_batch(&t, batch_ids, TOKENS, STRIDE, batch) && errno == EINVAL);
    assert(batch[0] == 123456.0f);
    batch_ids[(TOKENS - 1) * STRIDE] = 0;
    uint32_t bad = 3;
    errno = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EINVAL);
    assert(ds4_engram_read(&t, NULL, 0, NULL));
    uint8_t nan = 127;
    assert(pwrite(fd, &nan, 1, offset) == 1);
    bad = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 31, STRIDE, batch) && errno == EDOM);
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    nan = 255;
    assert(pwrite(fd, &nan, 1, offset + 256) == 1);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(ftruncate(fd, offset + 260) == 0);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EIO);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EIO);
    assert(!ds4_engram_read_batch(&t, batch_ids, 31, STRIDE, batch) && errno == EIO);
    free(batch_ids);
    free(batch);
    ds4_engram_table_close(&t);
    ds4_engram_table_close(&t);
    assert(!ds4_engram_table_open(&t, path, offset, 1, DS4_ENGRAM_ENC_E4M3_ROW264));
    assert(!ds4_engram_table_open(&t, path, UINT64_MAX - 1, 3, DS4_ENGRAM_ENC_E4M3_ROW264));
    assert(!ds4_engram_table_open(&t, path, offset, 0, DS4_ENGRAM_ENC_E4M3_ROW264));
    close(fd);
    assert(unlink(path) == 0);
}

static void test_all_scaled_values(void) {
    char path[] = "/tmp/ds4-engram-values-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(unlink(path) == 0);
    ds4_engram_table table = {.fd = fd, .rows = 1};
    const uint32_t row = 0;
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    float out[DS4_ENGRAM_DIM];
    for (uint32_t code = 0; code < 256; code++) {
        memset(raw, code, DS4_ENGRAM_DIM);
        for (uint32_t scale = 0; scale < 256; scale++) {
            memset(raw + DS4_ENGRAM_DIM, scale, DS4_ENGRAM_ROW_BYTES - DS4_ENGRAM_DIM);
            assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
            const int exponent = (code >> 3) & 15;
            double value = exponent ? (1.0 + (code & 7u) / 8.0) * pow(2.0, exponent - 7) :
                                      (code & 7u) / 512.0;
            if (code & 128u) value = -value;
            float expected = (float)(value * pow(2.0, (int)scale - 127));
            uint32_t bits;
            memcpy(&bits, &expected, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&expected, &bits, sizeof(expected));
            const bool valid = (code & 127u) != 127u && scale != 255u && isfinite(expected);
            errno = 0;
            assert(ds4_engram_read(&table, &row, 1, out) == valid);
            if (valid) {
                for (uint32_t j = 0; j < DS4_ENGRAM_DIM; j++)
                    assert(!memcmp(out + j, &expected, sizeof(expected)));
            } else assert(errno == EDOM);
        }
    }
    close(fd);
}

/* Independent copy of the Lloyd 4-bit codebook (lloyd4_e5m3_32_r136). */
static const float test_cb[8] = {0.0f, 0.095f, 0.1901f, 0.2944f, 0.4126f, 0.5586f, 0.7484f, 1.0f};

static float bf16_round(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

/* code = sign << 3 | idx; scale byte = E << 3 | M, scale (8 + M) / 16 * 2^(E - 16). */
static float q4_expected(int code, int scale_byte) {
    const double s = (8 + (scale_byte & 7)) / 16.0 * pow(2.0, (scale_byte >> 3) - 16);
    float v = (float)((double)test_cb[code & 7] * s);
    if (code & 8) v = -v;
    return bf16_round(v);
}

static void q4_pack(uint8_t row[DS4_ENGRAM_Q4_ROW_BYTES], const int codes[DS4_ENGRAM_DIM],
                    const int scales[DS4_ENGRAM_DIM / 32]) {
    for (int k = 0; k < DS4_ENGRAM_DIM / 2; k++)
        row[k] = (uint8_t)(codes[2 * k] | codes[2 * k + 1] << 4);
    for (int g = 0; g < DS4_ENGRAM_DIM / 32; g++) row[DS4_ENGRAM_DIM / 2 + g] = (uint8_t)scales[g];
}

static void test_q4_rows(void) {
    char path[] = "/tmp/ds4-engram-q4-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    const uint64_t offset = (1ull << 33) + 48;
    int codes[3][DS4_ENGRAM_DIM], scales[3][DS4_ENGRAM_DIM / 32];
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
        codes[0][j] = ((j / 8) & 1) << 3 | (j % 8);   /* every level, both signs */
        codes[1][j] = (j * 5) & 15;
        codes[2][j] = 8;                               /* sign set, level 0: -0 */
    }
    for (int g = 0; g < DS4_ENGRAM_DIM / 32; g++) {
        scales[0][g] = 16 << 3;                        /* 0.5 */
        scales[1][g] = g & 1 ? 31 << 3 : 7;            /* both ends: 2^14, 15/16 * 2^-16 */
        scales[2][g] = 16 << 3;
    }
    uint8_t raw[3][DS4_ENGRAM_Q4_ROW_BYTES];
    for (int r = 0; r < 3; r++) q4_pack(raw[r], codes[r], scales[r]);
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    ds4_engram_table t;
    assert(!ds4_engram_table_open(&t, path, offset, 4, DS4_ENGRAM_ENC_LLOYD4_ROW136));
    assert(!ds4_engram_table_open(&t, path, offset, 3, (ds4_engram_encoding)7) && errno == EINVAL);
    assert(ds4_engram_table_open(&t, path, offset, 3, DS4_ENGRAM_ENC_LLOYD4_ROW136));
    const uint32_t rows[] = {2, 0, 1};
    float out[3 * DS4_ENGRAM_DIM];
    assert(ds4_engram_read(&t, rows, 3, out));
    for (int r = 0; r < 3; r++) {
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            const float want = q4_expected(codes[rows[r]][j], scales[rows[r]][j / 32]);
            assert(!memcmp(&out[r * DS4_ENGRAM_DIM + j], &want, sizeof(want)));
        }
    }
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) assert(out[j] == 0.0f && signbit(out[j]));
    ds4_engram_table_close(&t);
    close(fd);
    assert(unlink(path) == 0);
}

static void test_encodings(void) {
    ds4_engram_encoding e = (ds4_engram_encoding)5;
    assert(ds4_engram_encoding_from_name("e4m3_e8m0_32_row264", 19, &e) &&
           e == DS4_ENGRAM_ENC_E4M3_ROW264);
    assert(ds4_engram_encoding_from_name("lloyd4_e5m3_32_r136", 19, &e) &&
           e == DS4_ENGRAM_ENC_LLOYD4_ROW136);
    assert(!ds4_engram_encoding_from_name("conversion_underway", 19, &e));
    assert(!ds4_engram_encoding_from_name("", 0, &e));
    assert(!ds4_engram_encoding_from_name("e4m3_e8m0_32_row264x", 20, &e));
    assert(!ds4_engram_encoding_from_name("e4m3_e8m0_32_row264", 18, &e));
    assert(!strcmp(ds4_engram_encoding_name(DS4_ENGRAM_ENC_E4M3_ROW264), "e4m3_e8m0_32_row264"));
    assert(!strcmp(ds4_engram_encoding_name(DS4_ENGRAM_ENC_LLOYD4_ROW136), "lloyd4_e5m3_32_r136"));
    assert(ds4_engram_row_bytes(DS4_ENGRAM_ENC_E4M3_ROW264) == 264);
    assert(ds4_engram_row_bytes(DS4_ENGRAM_ENC_LLOYD4_ROW136) == 136);
}

/* The spike's v7 arithmetic, retyped independently of ds4_engram.c. */
static float ref_f8_up(float x) {
    int e;
    const float m = frexpf(x, &e);
    return ldexpf(ceilf(m * 16.0f) / 16.0f, e);
}

static float ref_f8_down(float s) {
    int e;
    const float m = frexpf(s, &e);
    const float n = roundf(m * 16.0f) - 1.0f;
    return n < 8.0f ? ldexpf(15.0f / 16.0f, e - 1) : ldexpf(n / 16.0f, e);
}

static float ref_group(const float *v, float *q, int *idx, float s) {
    float err = 0.0f;
    for (int j = 0; j < 32; j++) {
        const float x = fabsf(v[j]) / s;
        int best = 0;
        for (int k = 1; k < 8; k++)
            if (fabsf(test_cb[k] - x) < fabsf(test_cb[best] - x)) best = k;
        q[j] = copysignf(test_cb[best] * s, v[j]);
        idx[j] = best;
        const float d = q[j] - v[j];
        err += d * d;
    }
    return err;
}

static float ref_e4m3_value(uint8_t code, uint8_t scale) {
    const int exponent = (code >> 3) & 15;
    double value = exponent ? (1.0 + (code & 7u) / 8.0) * pow(2.0, exponent - 7) : (code & 7u) / 512.0;
    if (code & 128u) value = -value;
    return (float)(value * pow(2.0, (int)scale - 127));
}

/* Reference v7 of one source row: BF16 output values and the expected 136 bytes. */
static void ref_encode(const uint8_t src[DS4_ENGRAM_ROW_BYTES], float out[DS4_ENGRAM_DIM],
                       uint8_t row[DS4_ENGRAM_Q4_ROW_BYTES]) {
    float v[DS4_ENGRAM_DIM];
    int codes[DS4_ENGRAM_DIM], scales[DS4_ENGRAM_DIM / 32];
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) v[j] = ref_e4m3_value(src[j], src[DS4_ENGRAM_DIM + j / 32]);
    for (int g = 0; g < DS4_ENGRAM_DIM / 32; g++) {
        const float *gv = v + 32 * g;
        float amax = 0.0f, q[32];
        int idx[32] = {0};
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(gv[j]));
        float s = 0.0f;
        if (amax == 0.0f) {
            memcpy(q, gv, sizeof(q));
            scales[g] = 0;
        } else {
            const float up = ref_f8_up(amax / test_cb[7]), down = ref_f8_down(up);
            float qa[32], qb[32];
            int ia[32], ib[32];
            const float ea = ref_group(gv, qa, ia, up), eb = ref_group(gv, qb, ib, down);
            s = eb < ea ? down : up;
            memcpy(q, eb < ea ? qb : qa, sizeof(q));
            memcpy(idx, eb < ea ? ib : ia, sizeof(idx));
            int e;
            const float m = frexpf(s, &e);
            scales[g] = (e + 16) << 3 | (int)(m * 16.0f - 8.0f);
        }
        for (int j = 0; j < 32; j++) {
            out[32 * g + j] = bf16_round(q[j]);
            codes[32 * g + j] = (signbit(gv[j]) ? 8 : 0) | idx[j];
        }
    }
    q4_pack(row, codes, scales);
}

static uint32_t enc_rng = 2463534242u;
static uint32_t enc_next(void) {
    enc_rng ^= enc_rng << 13;
    enc_rng ^= enc_rng >> 17;
    enc_rng ^= enc_rng << 5;
    return enc_rng;
}

static void random_source_row(uint8_t src[DS4_ENGRAM_ROW_BYTES]) {
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
        uint8_t c;
        do c = (uint8_t)enc_next(); while ((c & 127) == 127);
        src[j] = c;
    }
    for (int g = 0; g < DS4_ENGRAM_DIM / 32; g++)
        src[DS4_ENGRAM_DIM + g] = (uint8_t)(enc_next() % 4 ? 117 + enc_next() % 6 : 112 + enc_next() % 16);
}

/* Encode a row, decode it through the 136-byte table path, compare with the reference. */
static void check_encoded(int fd, const uint8_t src[DS4_ENGRAM_ROW_BYTES]) {
    uint8_t row[DS4_ENGRAM_Q4_ROW_BYTES], want_row[DS4_ENGRAM_Q4_ROW_BYTES];
    float want[DS4_ENGRAM_DIM], got[DS4_ENGRAM_DIM];
    memset(row, 0xa5, sizeof(row));
    assert(ds4_engram_encode_row136(src, row));
    ref_encode(src, want, want_row);
    assert(!memcmp(row, want_row, sizeof(row)));
    assert(pwrite(fd, row, sizeof(row), 0) == sizeof(row));
    ds4_engram_table t = {.fd = fd, .rows = 1, .encoding = DS4_ENGRAM_ENC_LLOYD4_ROW136};
    const uint32_t id = 0;
    assert(ds4_engram_read(&t, &id, 1, got));
    assert(!memcmp(got, want, sizeof(got)));
}

static void test_encoder(void) {
    char path[] = "/tmp/ds4-engram-enc-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(unlink(path) == 0);
    uint8_t src[DS4_ENGRAM_ROW_BYTES];
    for (int r = 0; r < 20000; r++) {
        random_source_row(src);
        check_encoded(fd, src);
    }
    /* An all-zero group keeps the sign of each source zero. */
    random_source_row(src);
    for (int j = 32; j < 64; j++) src[j] = j & 1 ? 0x80 : 0x00;
    check_encoded(fd, src);
    /* amax exactly a power of two: the down step takes the 15/16 * 2^(e-1) branch. */
    random_source_row(src);
    for (int j = 64; j < 96; j++) src[j] = (uint8_t)(enc_next() % 48);
    src[70] = 56;                         /* 1.0: the group's largest magnitude */
    check_encoded(fd, src);
    /* Refused sources leave errno EDOM. */
    uint8_t row[DS4_ENGRAM_Q4_ROW_BYTES];
    random_source_row(src);
    src[5] = 0xff;
    errno = 0;
    assert(!ds4_engram_encode_row136(src, row) && errno == EDOM);
    random_source_row(src);
    src[DS4_ENGRAM_DIM + 3] = 255;
    errno = 0;
    assert(!ds4_engram_encode_row136(src, row) && errno == EDOM);
    random_source_row(src);                /* 448 * 2^13: up needs E > 31 */
    src[0] = 0x7e;
    src[DS4_ENGRAM_DIM] = 140;
    errno = 0;
    assert(!ds4_engram_encode_row136(src, row) && errno == EDOM);
    random_source_row(src);                /* every |v| = 2^-21: both scales need E < 0 */
    for (int j = 0; j < 32; j++) src[j] = j & 1 ? 0x08 : 0x88;
    src[DS4_ENGRAM_DIM] = 112;
    errno = 0;
    assert(!ds4_engram_encode_row136(src, row) && errno == EDOM);
    close(fd);
}

int main(void) {
    test_hash();
    test_rows();
    test_all_scaled_values();
    test_encodings();
    test_q4_rows();
    test_encoder();
    puts("Engram hashes, history and bounded disk rows: PASS");
    return 0;
}
