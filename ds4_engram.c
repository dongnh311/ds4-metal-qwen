#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <pthread.h>
#endif

bool ds4_engram_layout_valid(const ds4_engram_layout *l) {
    if (!l || !l->token_map || !l->vocab_size ||
        !l->compressed_vocab_size || l->compressed_vocab_size > INT32_MAX ||
        l->pad_id >= l->compressed_vocab_size) return false;
    for (uint32_t i = 0; i < l->vocab_size; i++)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
        for (int i = 0; i < DS4_ENGRAM_NGRAM; i++) {
            uint64_t m = l->multipliers[layer][i];
            if (!(m & 1) || m > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t total = 0;
        for (int i = 0; i < DS4_ENGRAM_COLS; i++) {
            if (l->primes[layer][i] < 2) return false;
            total += l->primes[layer][i];
        }
        if (total != l->rows[layer]) return false;
    }
    return true;
}

void ds4_engram_history_reset(ds4_engram_history *h) {
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) h->tail[i] = DS4_ENGRAM_DEAD;
}

bool ds4_engram_hash(const ds4_engram_layout *l, ds4_engram_history *h,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (DS4_ENGRAM_LAYERS * DS4_ENGRAM_COLS * sizeof(*rows)))
        return false;
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) {
        if (h->tail[i] < DS4_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size) return false;
    }
    for (size_t i = 0; i < count; i++) {
        int32_t current = mask && !mask[i] ? DS4_ENGRAM_DEAD :
                          (int32_t)l->token_map[tokens[i]];
        uint32_t ids[DS4_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < DS4_ENGRAM_NGRAM; j++) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == DS4_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int j = 1; j < DS4_ENGRAM_NGRAM; j++) {
                hash ^= (uint64_t)ids[j] * l->multipliers[layer][j];
                for (int head = 0; head < DS4_ENGRAM_HEADS; head++) {
                    int col = (j - 1) * DS4_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = DS4_ENGRAM_NGRAM - 2; j > 0; j--) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

const float ds4_engram_lloyd4[8] = {
    0.0f, 0.095f, 0.1901f, 0.2944f, 0.4126f, 0.5586f, 0.7484f, 1.0f};

static const char *const engram_encoding_names[] = {
    [DS4_ENGRAM_ENC_E4M3_ROW264] = "e4m3_e8m0_32_row264",
    [DS4_ENGRAM_ENC_LLOYD4_ROW136] = "lloyd4_e5m3_32_r136",
};

static bool engram_encoding_valid(ds4_engram_encoding e) {
    return e == DS4_ENGRAM_ENC_E4M3_ROW264 || e == DS4_ENGRAM_ENC_LLOYD4_ROW136;
}

uint32_t ds4_engram_row_bytes(ds4_engram_encoding e) {
    return e == DS4_ENGRAM_ENC_LLOYD4_ROW136 ? DS4_ENGRAM_Q4_ROW_BYTES :
           e == DS4_ENGRAM_ENC_E4M3_ROW264 ? DS4_ENGRAM_ROW_BYTES : 0;
}

bool ds4_engram_encoding_from_name(const char *name, size_t len,
                                   ds4_engram_encoding *e) {
    for (int i = 0; i < 2; i++) {
        const char *want = engram_encoding_names[i];
        if (name && len == strlen(want) && !memcmp(name, want, len)) {
            if (e) *e = (ds4_engram_encoding)i;
            return true;
        }
    }
    return false;
}

const char *ds4_engram_encoding_name(ds4_engram_encoding e) {
    return engram_encoding_valid(e) ? engram_encoding_names[e] : NULL;
}

bool ds4_engram_table_open(ds4_engram_table *t, const char *path,
                           uint64_t offset, uint32_t rows,
                           ds4_engram_encoding encoding) {
    if (!t) return false;
    *t = (ds4_engram_table){.fd = -1};
    uint64_t bytes = (uint64_t)rows * ds4_engram_row_bytes(encoding);
    if (!path || !rows || !engram_encoding_valid(encoding) ||
        offset > INT64_MAX || bytes > INT64_MAX - offset) {
        errno = EINVAL;
        return false;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto fail;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto fail;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#endif
    *t = (ds4_engram_table){.fd = fd, .offset = offset, .rows = rows,
                            .encoding = encoding};
    return true;
fail: {
        int saved = errno;
        close(fd);
        errno = saved;
        return false;
    }
}

void ds4_engram_table_close(ds4_engram_table *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static bool read_row(int fd, uint64_t offset, uint8_t *row, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, row + done, bytes - done,
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

static float engram_f8_up(float x) {
    int e;
    const float m = frexpf(x, &e);
    return ldexpf(ceilf(m * 16.0f) / 16.0f, e);
}

static float engram_f8_down(float s) {
    int e;
    const float m = frexpf(s, &e);
    const float n = roundf(m * 16.0f) - 1.0f;
    return n < 8.0f ? ldexpf(15.0f / 16.0f, e - 1) : ldexpf(n / 16.0f, e);
}

/* Nearest level of |v| / s, ties to the lower index; float squared error in order. */
static float engram_q4_group(const float *v, uint8_t *idx, float s) {
    float err = 0.0f;
    for (int j = 0; j < 32; j++) {
        const float x = fabsf(v[j]) / s;
        int best = 0;
        for (int k = 1; k < 8; k++)
            if (fabsf(ds4_engram_lloyd4[k] - x) < fabsf(ds4_engram_lloyd4[best] - x)) best = k;
        idx[j] = (uint8_t)best;
        const float d = copysignf(ds4_engram_lloyd4[best] * s, v[j]) - v[j];
        err += d * d;
    }
    return err;
}

static bool engram_scale_byte(float s, uint8_t *byte) {
    int e;
    const float m = frexpf(s, &e);
    const float steps = m * 16.0f - 8.0f;
    if (!(steps >= 0.0f && steps <= 7.0f) || steps != floorf(steps) ||
        e + 16 < 0 || e + 16 > 31) return false;
    *byte = (uint8_t)((e + 16) << 3 | (int)steps);
    return true;
}

bool ds4_engram_encode_row136(const uint8_t src[DS4_ENGRAM_ROW_BYTES],
                              uint8_t dst[DS4_ENGRAM_Q4_ROW_BYTES]) {
    float v[DS4_ENGRAM_DIM];
    for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
        const uint8_t code = src[j], scale = src[DS4_ENGRAM_DIM + j / 32];
        if ((code & 127) == 127 || scale == 255) {
            errno = EDOM;
            return false;
        }
        v[j] = ldexpf(e4m3(code), (int)scale - 127);
        if (!isfinite(v[j])) {
            errno = EDOM;
            return false;
        }
    }
    memset(dst, 0, DS4_ENGRAM_Q4_ROW_BYTES);
    for (int g = 0; g < DS4_ENGRAM_DIM / 32; g++) {
        const float *gv = v + 32 * g;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(gv[j]));
        uint8_t idx[32] = {0}, byte = 0;
        if (amax != 0.0f) {   /* an all-zero group keeps level 0 and its signs */
            const float up = engram_f8_up(amax / ds4_engram_lloyd4[7]);
            const float down = engram_f8_down(up);
            uint8_t up_byte, down_byte, idx_down[32];
            if (!engram_scale_byte(up, &up_byte) || !engram_scale_byte(down, &down_byte)) {
                errno = EDOM;
                return false;
            }
            const float err_up = engram_q4_group(gv, idx, up);
            const float err_down = engram_q4_group(gv, idx_down, down);
            byte = up_byte;
            if (err_down < err_up) {
                memcpy(idx, idx_down, sizeof(idx));
                byte = down_byte;
            }
        }
        for (int j = 0; j < 32; j++) {
            const int k = 32 * g + j;
            const uint8_t nibble = (uint8_t)((signbit(gv[j]) ? 8 : 0) | idx[j]);
            dst[k / 2] |= (uint8_t)(nibble << (4 * (k & 1)));
        }
        dst[DS4_ENGRAM_DIM / 2 + g] = byte;
    }
    return true;
}

bool ds4_engram_read(const ds4_engram_table *t, const uint32_t *rows,
                     size_t count, float *out) {
    if (!t || t->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (DS4_ENGRAM_DIM * sizeof(*out))) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (rows[i] >= t->rows) {
            errno = EINVAL;
            return false;
        }
    }
    if (!engram_encoding_valid(t->encoding)) {
        errno = EINVAL;
        return false;
    }
    const bool q4 = t->encoding == DS4_ENGRAM_ENC_LLOYD4_ROW136;
    const size_t row_bytes = ds4_engram_row_bytes(t->encoding);
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    for (size_t i = 0; i < count; i++) {
        if (!read_row(t->fd, t->offset + (uint64_t)rows[i] * row_bytes, raw, row_bytes))
            return false;
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            float value;
            if (q4) {
                const uint8_t code = (raw[j / 2] >> (4 * (j & 1))) & 15;
                const uint8_t scale = raw[DS4_ENGRAM_DIM / 2 + j / 32];
                const float s = ldexpf((float)(8 + (scale & 7)) / 16.0f, (scale >> 3) - 16);
                value = copysignf(ds4_engram_lloyd4[code & 7] * s, code & 8 ? -1.0f : 1.0f);
            } else {
                uint8_t code = raw[j], scale = raw[DS4_ENGRAM_DIM + j / 32];
                if ((code & 127) == 127 || scale == 255) {
                    errno = EDOM;
                    return false;
                }
                value = ldexpf(e4m3(code), (int)scale - 127);
            }
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) {
                errno = EDOM;
                return false;
            }
            out[i * DS4_ENGRAM_DIM + j] = value;
        }
    }
    return true;
}

typedef struct {
    uint32_t row, output;
} engram_request;

static int request_order(const void *a, const void *b) {
    const engram_request *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

enum { ENGRAM_READERS = 16 };
#ifdef __APPLE__
enum { ENGRAM_PARALLEL_MIN_ROWS = 8 };
#else
/* Unlike dispatch's shared pool, this path creates threads for each batch. */
enum { ENGRAM_PARALLEL_MIN_ROWS = 256 };
#endif

typedef struct {
    const ds4_engram_table *table;
    const engram_request *request;
    float *out;
    size_t count, readers;
    int error[ENGRAM_READERS];
} engram_batch;

static void read_batch_part(void *context, size_t part) {
    engram_batch *batch = context;
    const engram_request *request = batch->request;
    const size_t begin = batch->count * part / batch->readers;
    const size_t end = batch->count * (part + 1) / batch->readers;
    const float *previous = NULL;
    for (size_t i = begin; i < end; i++) {
        float *dst = batch->out + (size_t)request[i].output * DS4_ENGRAM_DIM;
        if (i > begin && request[i].row == request[i - 1].row) {
            memcpy(dst, previous, DS4_ENGRAM_DIM * sizeof(*dst));
        } else {
            if (!ds4_engram_read(batch->table, &request[i].row, 1, dst)) {
                batch->error[part] = errno ? errno : EIO;
                return;
            }
            previous = dst;
        }
    }
}

#ifndef __APPLE__
typedef struct {
    engram_batch *batch;
    size_t part;
} engram_reader;

static void *read_batch_thread(void *context) {
    engram_reader *reader = context;
    read_batch_part(reader->batch, reader->part);
    return NULL;
}
#endif

bool ds4_engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out) {
    if (!t || t->fd < 0 || (tokens && (!rows || !out || stride < DS4_ENGRAM_COLS)) ||
        tokens > SIZE_MAX / (DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(*out)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(*rows) - DS4_ENGRAM_COLS) / stride)) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < tokens; i++) {
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            if (rows[i * stride + j] >= t->rows) {
                errno = EINVAL;
                return false;
            }
        }
    }
    if (!tokens) return true;
    enum { BATCH_TOKENS = 2048 };
    const size_t cap = tokens < BATCH_TOKENS ? tokens : BATCH_TOKENS;
    engram_request *request = malloc(cap * DS4_ENGRAM_COLS * sizeof(*request));
    if (!request) return false;
    bool ok = true;
    for (size_t start = 0; ok && start < tokens; start += cap) {
        const size_t n = tokens - start < cap ? tokens - start : cap;
        const size_t count = n * DS4_ENGRAM_COLS;
        for (size_t i = 0; i < count; i++) {
            request[i] = (engram_request){
                rows[(start + i / DS4_ENGRAM_COLS) * stride + i % DS4_ENGRAM_COLS],
                (uint32_t)i
            };
        }
        qsort(request, count, sizeof(*request), request_order);
        engram_batch batch = {.table = t, .request = request, .count = count,
            .out = out + start * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM, .readers = 1};
        /* Fixed concurrency hides random-read latency without caching the table.
         * Each worker owns disjoint output rows; all finish before GPU use. */
        if (count >= ENGRAM_PARALLEL_MIN_ROWS) {
            batch.readers = ENGRAM_READERS;
#ifdef __APPLE__
            dispatch_apply_f(batch.readers,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &batch, read_batch_part);
#else
            pthread_t threads[ENGRAM_READERS - 1];
            engram_reader readers[ENGRAM_READERS - 1];
            size_t started = 0;
            for (size_t part = 1; part < batch.readers; part++) {
                readers[started] = (engram_reader){&batch, part};
                if (pthread_create(&threads[started], NULL, read_batch_thread,
                                   &readers[started])) break;
                started++;
            }
            read_batch_part(&batch, 0);
            /* Thread exhaustion only reduces concurrency, not correctness. */
            for (size_t part = started + 1; part < batch.readers; part++)
                read_batch_part(&batch, part);
            for (size_t part = 0; part < started; part++)
                if (pthread_join(threads[part], NULL)) abort();
#endif
        } else
        read_batch_part(&batch, 0);
        for (size_t i = 0; i < batch.readers; i++) {
            if (batch.error[i]) {
                errno = batch.error[i];
                ok = false;
                break;
            }
        }
    }
    int saved = errno;
    free(request);
    errno = saved;
    return ok;
}
