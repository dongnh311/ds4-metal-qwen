#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_ROW_BYTES = 264,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

/* Row encodings, named by deepseek41.engram.encoding:
 * e4m3_e8m0_32_row264: 256 E4M3 bytes, then 8 original E8M0 scales.
 * lloyd4_e5m3_32_r136: 128 bytes of 4-bit codes (even element in the low
 *   nibble; sign << 3 | index into ds4_engram_lloyd4), then 8 scale bytes
 *   E << 3 | M meaning (8 + M) / 16 * 2^(E - 16), one per 32 values. */
typedef enum {
    DS4_ENGRAM_ENC_E4M3_ROW264 = 0,
    DS4_ENGRAM_ENC_LLOYD4_ROW136 = 1
} ds4_engram_encoding;

enum { DS4_ENGRAM_Q4_ROW_BYTES = 136 };

extern const float ds4_engram_lloyd4[8];

uint32_t ds4_engram_row_bytes(ds4_engram_encoding encoding);
/* Exact match on one of the encoding names above. */
bool ds4_engram_encoding_from_name(const char *name, size_t len,
                                   ds4_engram_encoding *encoding);
const char *ds4_engram_encoding_name(ds4_engram_encoding encoding);

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
    ds4_engram_encoding encoding;
} ds4_engram_table;

/* Encode one e4m3_e8m0_32_row264 row as lloyd4_e5m3_32_r136. Per 32 values,
 * the scale is the better (by float squared error, ties to the larger) of
 * amax rounded up to 4 significant bits and one step below it; each value
 * takes the nearest codebook level (ties to the lower) and its source sign.
 * False with errno EDOM on a NaN code, scale byte 255, a non-finite value, or
 * a scale candidate outside E = 0..31. */
bool ds4_engram_encode_row136(const uint8_t src[DS4_ENGRAM_ROW_BYTES],
                              uint8_t dst[DS4_ENGRAM_Q4_ROW_BYTES]);

/* A separate uncached file descriptor, never an mmap or Metal model view. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows,
                           ds4_engram_encoding encoding);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage for the reference's BF16-rounded values. No whole
 * table allocation; caller owns count * DIM floats. Failure invalidates output. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. Temporary
 * storage is bounded to 384 KiB, independent of the table and prefix size.
 * On macOS, large batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);

#endif
