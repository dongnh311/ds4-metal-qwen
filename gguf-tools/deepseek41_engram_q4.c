#define _DARWIN_C_SOURCE
/*
 * deepseek41_engram_q4: re-encode the two Engram tables of a DeepSeek V4.1
 * GGUF from e4m3_e8m0_32_row264 to lloyd4_e5m3_32_r136 in place (264 -> 136
 * bytes per row; the Flash Q2 file goes from 340.60 to 249.04 GiB). CPU and
 * disk only.
 *
 *   --check FILE            encode every row in memory, write FILE.eq4-check
 *                           (layout fingerprint + SHA-256 of the new tail)
 *   --sample FILE N OUT     before converting: N rows per table and their
 *                           decode once converted
 *   --convert FILE          block the file, compact the tables forward chunk
 *                           by chunk (FILE.eq4-state, FILE.eq4-journal),
 *                           patch the header, truncate; rerun to resume
 *   --verify FILE           the converted tail against FILE.eq4-check; also
 *                           prints the whole-file SHA-256
 *   --verify-sample FILE OUT  the sampled rows through ds4_engram_read
 *   [--chunk-rows K]        rows per chunk (default 262144)
 *
 * Design: docs/superpowers/specs/2026-10-02-v41-engram-q4-design.md
 */
#include "ds4_engram.h"

#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TOOL "deepseek41_engram_q4"

enum { ENC_LEN = 19, SRC_ROW = DS4_ENGRAM_ROW_BYTES, DST_ROW = DS4_ENGRAM_Q4_ROW_BYTES,
       MAX_THREADS = 16, STATE_VERSION = 1 };
static const char ENC_ORIG[] = "e4m3_e8m0_32_row264";
static const char ENC_Q4[] = "lloyd4_e5m3_32_r136";
static const char ENC_BUSY[] = "conversion_underway";
static const char *const TABLE_NAMES[2] = {"blk.1.engram_embd.weight", "blk.14.engram_embd.weight"};

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, TOOL ": ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

/* ---- I/O ---- */

static void pread_full(int fd, void *p, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(fd, (uint8_t *)p + done, n - done, (off_t)(off + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) die("read at %" PRIu64 " failed: %s", off + done, r ? strerror(errno) : "end of file");
        done += (size_t)r;
    }
}

static void pwrite_full(int fd, const void *p, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pwrite(fd, (const uint8_t *)p + done, n - done, (off_t)(off + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) die("write at %" PRIu64 " failed: %s", off + done, strerror(errno));
        done += (size_t)r;
    }
}

/* fsync on macOS does not order writes against a power loss; F_FULLFSYNC
 * flushes the drive cache, so the journal and state files really precede the
 * writes they protect. */
static void sync_fd(int fd) {
    if (fcntl(fd, F_FULLFSYNC) == 0) return;
    if (fsync(fd) != 0) die("fsync failed: %s", strerror(errno));
}

static void sync_dir(const char *path) {
    char copy[4096];
    snprintf(copy, sizeof(copy), "%s", path);
    const int fd = open(dirname(copy), O_RDONLY | O_CLOEXEC);
    if (fd < 0) die("cannot open the directory of %s", path);
    sync_fd(fd);
    close(fd);
}

/* Write a side file through a temporary name, fsync, rename, fsync the directory. */
static void write_atomic(const char *path, const void *p, size_t n) {
    char tmp[4200];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) die("cannot create %s: %s", tmp, strerror(errno));
    pwrite_full(fd, p, n, 0);
    sync_fd(fd);
    close(fd);
    if (rename(tmp, path) != 0) die("cannot rename %s: %s", tmp, strerror(errno));
    sync_dir(path);
}

static int test_stop(const char *phase) {
    static int parsed;
    static char name[32];
    static long target, count;
    if (!parsed) {
        parsed = 1;
        const char *e = getenv("DS4_ENGRAM_Q4_TEST_STOP");
        const char *colon = e ? strchr(e, ':') : NULL;
        if (colon && (size_t)(colon - e) < sizeof(name)) {
            memcpy(name, e, (size_t)(colon - e));
            target = atol(colon + 1);
        }
    }
    if (target > 0 && !strcmp(name, phase) && ++count == target) _exit(3);
    return 0;
}

/* ---- GGUF header ---- */

typedef struct {
    uint64_t file_size, data_start, align, enc_pos;
    char enc[ENC_LEN + 1];
    int v41, have_enc;
    struct { uint64_t dim_pos, off_pos, width, rows, rel; int found; } t[2];
    uint64_t max_other_rel;
} gguf;

typedef struct { const uint8_t *p; size_t n, pos; } cursor;

static const uint8_t *need(cursor *c, size_t k) {
    if (k > c->n - c->pos) die("GGUF header is truncated or larger than the parse window");
    const uint8_t *p = c->p + c->pos;
    c->pos += k;
    return p;
}
static uint32_t rd32(cursor *c) { uint32_t v; memcpy(&v, need(c, 4), 4); return v; }
static uint64_t rd64(cursor *c) { uint64_t v; memcpy(&v, need(c, 8), 8); return v; }
static const char *rdstr(cursor *c, uint64_t *len) {
    *len = rd64(c);
    if (*len > c->n) die("GGUF string length out of range");
    return (const char *)need(c, (size_t)*len);
}
static int str_is(const char *s, uint64_t len, const char *want) {
    return len == strlen(want) && !memcmp(s, want, len);
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

static void skip_value(cursor *c, uint32_t type) {
    uint64_t len;
    if (type == 8) {
        (void)rdstr(c, &len);
    } else if (type == 9) {
        const uint32_t et = rd32(c);
        const uint64_t count = rd64(c);
        if (et == 8) {
            for (uint64_t i = 0; i < count; i++) (void)rdstr(c, &len);
        } else {
            const size_t es = scalar_size(et);
            if (!es || count > (c->n - c->pos) / es) die("bad GGUF array");
            (void)need(c, (size_t)(count * es));
        }
    } else {
        const size_t es = scalar_size(type);
        if (!es) die("unknown GGUF value type %u", type);
        (void)need(c, es);
    }
}

static void parse(int fd, gguf *g) {
    memset(g, 0, sizeof(*g));
    struct stat st;
    if (fstat(fd, &st) != 0) die("stat failed: %s", strerror(errno));
    g->file_size = (uint64_t)st.st_size;
    const size_t window = g->file_size < (64u << 20) ? (size_t)g->file_size : (64u << 20);
    uint8_t *head = malloc(window ? window : 1);
    if (!head) die("out of memory");
    pread_full(fd, head, window, 0);
    cursor c = {head, window, 0};
    if (rd32(&c) != 0x46554747u) die("not a GGUF file");
    if (rd32(&c) != 3) die("GGUF version 3 required");
    const uint64_t n_tensors = rd64(&c), n_kv = rd64(&c);
    g->align = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen, vlen;
        const char *key = rdstr(&c, &klen);
        const uint32_t type = rd32(&c);
        if (type == 8 && str_is(key, klen, "general.architecture")) {
            const char *v = rdstr(&c, &vlen);
            g->v41 = str_is(v, vlen, "deepseek41");
        } else if (type == 8 && str_is(key, klen, "deepseek41.engram.encoding")) {
            vlen = rd64(&c);
            if (vlen != ENC_LEN) die("unexpected Engram encoding length %" PRIu64, vlen);
            g->enc_pos = c.pos;
            memcpy(g->enc, need(&c, ENC_LEN), ENC_LEN);
            g->have_enc = 1;
        } else if (type == 4 && str_is(key, klen, "general.alignment")) {
            g->align = rd32(&c);
            if (!g->align || (g->align & (g->align - 1))) die("bad general.alignment");
        } else {
            skip_value(&c, type);
        }
    }
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        const char *name = rdstr(&c, &nlen);
        const uint32_t nd = rd32(&c);
        if (nd == 0 || nd > 8) die("bad tensor rank");
        const uint64_t dim_pos = c.pos;
        uint64_t dims[8];
        for (uint32_t d = 0; d < nd; d++) dims[d] = rd64(&c);
        const uint32_t type = rd32(&c);
        const uint64_t off_pos = c.pos, rel = rd64(&c);
        const int idx = str_is(name, nlen, TABLE_NAMES[0]) ? 0 : str_is(name, nlen, TABLE_NAMES[1]) ? 1 : -1;
        if (idx < 0) {
            if (rel > g->max_other_rel) g->max_other_rel = rel;
            continue;
        }
        if (g->t[idx].found) die("duplicate %s", TABLE_NAMES[idx]);
        if (nd != 2 || type != 24) die("%s is not a 2-D I8 tensor", TABLE_NAMES[idx]);
        g->t[idx].found = 1;
        g->t[idx].dim_pos = dim_pos;
        g->t[idx].off_pos = off_pos;
        g->t[idx].width = dims[0];
        g->t[idx].rows = dims[1];
        g->t[idx].rel = rel;
    }
    g->data_start = align_up(c.pos, g->align);
    free(head);
}

/* The geometry both layouts derive from: what --check records. */
typedef struct { uint64_t file_size, data_start, align, rel0, rows0, rel1, rows1; } layout;

static uint64_t new_rel1(const layout *l) { return align_up(l->rel0 + l->rows0 * DST_ROW, l->align); }
static uint64_t new_end(const layout *l) { return l->data_start + new_rel1(l) + l->rows1 * DST_ROW; }

/* The header and size of an untouched original file; returns its layout. */
static layout original_layout(const gguf *g) {
    if (!g->v41) die("not a DeepSeek V4.1 GGUF");
    if (!g->have_enc) die("no deepseek41.engram.encoding");
    if (!g->t[0].found || !g->t[1].found) die("Engram tables missing");
    layout l = {g->file_size, g->data_start, g->align, g->t[0].rel, g->t[0].rows, g->t[1].rel, g->t[1].rows};
    for (int t = 0; t < 2; t++)
        if (g->t[t].width != SRC_ROW || !g->t[t].rows || g->t[t].rows > UINT32_MAX)
            die("%s is not [264, rows]", TABLE_NAMES[t]);
    const uint64_t end = l.data_start + l.rel1 + l.rows1 * SRC_ROW;
    if (g->max_other_rel >= l.rel0 || l.rel0 % l.align ||
        l.rel1 != align_up(l.rel0 + l.rows0 * SRC_ROW, l.align) ||
        l.file_size < end || l.file_size - end >= l.align)
        die("the Engram tables are not the last two tensors of the file");
    return l;
}

/* ---- encoding pool ---- */

typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    uint64_t lo, hi, bad;
} enc_job;

static void *enc_thread(void *arg) {
    enc_job *j = arg;
    j->bad = UINT64_MAX;
    for (uint64_t r = j->lo; r < j->hi; r++) {
        if (!ds4_engram_encode_row136(j->src + r * SRC_ROW, j->dst + r * DST_ROW)) {
            j->bad = r;
            break;
        }
    }
    return NULL;
}

/* UINT64_MAX when every row encodes, else the first row that cannot. */
static uint64_t encode_rows(const uint8_t *src, uint8_t *dst, uint64_t n) {
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int threads = cpus < 1 ? 1 : cpus > MAX_THREADS ? MAX_THREADS : (int)cpus;
    if ((uint64_t)threads > n) threads = (int)(n ? n : 1);
    pthread_t tid[MAX_THREADS];
    enc_job jobs[MAX_THREADS];
    for (int i = 0; i < threads; i++) {
        jobs[i] = (enc_job){src, dst, n * i / threads, n * (i + 1) / threads, UINT64_MAX};
        if (pthread_create(&tid[i], NULL, enc_thread, &jobs[i]) != 0) die("pthread_create failed");
    }
    uint64_t bad = UINT64_MAX;
    for (int i = 0; i < threads; i++) {
        pthread_join(tid[i], NULL);
        if (jobs[i].bad < bad) bad = jobs[i].bad;
    }
    return bad;
}

/* ---- side files ---- */

static void side(char *out, size_t cap, const char *path, const char *suffix) {
    if ((size_t)snprintf(out, cap, "%s%s", path, suffix) >= cap) die("path too long");
}

static void hex(const uint8_t d[CC_SHA256_DIGEST_LENGTH], char out[2 * CC_SHA256_DIGEST_LENGTH + 1]) {
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) sprintf(out + 2 * i, "%02x", d[i]);
}

/* Which file --check read, and that it has not been written since. */
typedef struct { uint64_t dev, ino, size, mtime_sec, mtime_nsec; } identity;

static identity file_identity(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) die("stat failed: %s", strerror(errno));
    return (identity){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                      (uint64_t)st.st_mtimespec.tv_sec, (uint64_t)st.st_mtimespec.tv_nsec};
}

static void write_check(const char *path, const layout *l, const identity *id, const char *sha) {
    char file[4200], text[640];
    side(file, sizeof(file), path, ".eq4-check");
    const int n = snprintf(text, sizeof(text),
        TOOL " check 1\nfingerprint %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
        " %" PRIu64 " %" PRIu64 "\nidentity %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
        "\nsha256 %s\n",
        l->file_size, l->data_start, l->align, l->rel0, l->rows0, l->rel1, l->rows1,
        id->dev, id->ino, id->size, id->mtime_sec, id->mtime_nsec, sha);
    write_atomic(file, text, (size_t)n);
}

/* 0 when there is no check file. */
static int read_check(const char *path, layout *l, identity *id, char sha[65]) {
    char file[4200];
    side(file, sizeof(file), path, ".eq4-check");
    FILE *fp = fopen(file, "r");
    if (!fp) return 0;
    int version = 0;
    identity unused;
    if (!id) id = &unused;
    const int got = fscanf(fp, TOOL " check %d fingerprint %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64
                           " %" SCNu64 " %" SCNu64 " %" SCNu64 " identity %" SCNu64 " %" SCNu64 " %" SCNu64
                           " %" SCNu64 " %" SCNu64 " sha256 %64s", &version,
                           &l->file_size, &l->data_start, &l->align, &l->rel0, &l->rows0, &l->rel1,
                           &l->rows1, &id->dev, &id->ino, &id->size, &id->mtime_sec, &id->mtime_nsec, sha);
    fclose(fp);
    if (got != 14 || version != 1 || strlen(sha) != 64) die("%s is malformed", file);
    return 1;
}

typedef struct {
    char magic[8];
    uint32_t version, phase, table, pad;
    uint64_t next;
    layout l;
} conv_state;

static int read_state(const char *path, conv_state *s) {
    char file[4200];
    side(file, sizeof(file), path, ".eq4-state");
    const int fd = open(file, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    const ssize_t n = read(fd, s, sizeof(*s));
    close(fd);
    if (n != (ssize_t)sizeof(*s) || memcmp(s->magic, "EQ4STATE", 8) || s->version != STATE_VERSION ||
        s->phase < 1 || s->phase > 3 || s->table > 2)
        die("%s is malformed", file);
    return 1;
}

static void write_state(const char *path, uint32_t phase, uint32_t table, uint64_t next, const layout *l) {
    char file[4200];
    side(file, sizeof(file), path, ".eq4-state");
    conv_state s;
    memset(&s, 0, sizeof(s));
    memcpy(s.magic, "EQ4STATE", 8);
    s.version = STATE_VERSION;
    s.phase = phase;
    s.table = table;
    s.next = next;
    s.l = *l;
    write_atomic(file, &s, sizeof(s));
}

typedef struct {
    char magic[8];
    uint32_t table, pad;
    uint64_t first, count;
} journal_head;

static void write_journal(const char *path, uint32_t table, uint64_t first, uint64_t count,
                          const uint8_t *src) {
    char file[4200];
    side(file, sizeof(file), path, ".eq4-journal");
    const size_t bytes = sizeof(journal_head) + (size_t)count * SRC_ROW;
    uint8_t *p = malloc(bytes);
    if (!p) die("out of memory");
    journal_head h = {{0}, table, 0, first, count};
    memcpy(h.magic, "EQ4JRNL1", 8);
    memcpy(p, &h, sizeof(h));
    memcpy(p + sizeof(h), src, (size_t)count * SRC_ROW);
    write_atomic(file, p, bytes);
    free(p);
}

/* A journal for the chunk that starts at (table, first): its row count, else 0. */
static uint64_t read_journal(const char *path, uint32_t table, uint64_t first, uint64_t max_rows,
                             uint8_t *src) {
    char file[4200];
    side(file, sizeof(file), path, ".eq4-journal");
    const int fd = open(file, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    journal_head h;
    uint64_t count = 0;
    if (read(fd, &h, sizeof(h)) == (ssize_t)sizeof(h) && !memcmp(h.magic, "EQ4JRNL1", 8) &&
        h.table == table && h.first == first && h.count) {
        if (h.count > max_rows)
            die("the journal holds %" PRIu64 " rows; resume with --chunk-rows %" PRIu64, h.count, h.count);
        pread_full(fd, src, (size_t)h.count * SRC_ROW, sizeof(h));
        count = h.count;
    }
    close(fd);
    return count;
}

static void remove_side(const char *path, const char *suffix) {
    char file[4200];
    side(file, sizeof(file), path, suffix);
    if (unlink(file) != 0 && errno != ENOENT) die("cannot remove %s: %s", file, strerror(errno));
}

/* Refuse when a ds4 process (or score_official) holds the file open. Other
 * holders, such as a scanner that opens new files briefly, get ~5 s to go. */
static void refuse_if_open(const char *path) {
    if (strchr(path, '\'')) die("path must not contain a single quote");
    char cmd[4300];
    snprintf(cmd, sizeof(cmd), "lsof -F pc -- '%s' 2>/dev/null", path);
    for (int attempt = 0;; attempt++) {
        FILE *p = popen(cmd, "r");
        if (!p) die("cannot run lsof");
        char line[512], name[256] = "";
        long pid = 0, other = 0;
        while (fgets(line, sizeof(line), p)) {
            line[strcspn(line, "\n")] = 0;
            if (line[0] == 'p') pid = atol(line + 1);
            else if (line[0] == 'c' && pid != (long)getpid() && !other) {
                other = pid;
                snprintf(name, sizeof(name), "%s", line + 1);
            }
        }
        pclose(p);
        if (!other) return;
        if (!strncmp(name, "ds4", 3) || !strcmp(name, "score_official") || attempt >= 10)
            die("process %ld (%s) has %s open; stop it first", other, name, path);
        usleep(500000);
    }
}

static int open_file(const char *path, int flags) {
    /* O_CLOEXEC: the lsof shell must not inherit the file and look like a holder. */
    const int fd = open(path, flags | O_CLOEXEC);
    if (fd < 0) die("cannot open %s: %s", path, strerror(errno));
    (void)fcntl(fd, F_NOCACHE, 1);
    return fd;
}

/* ---- commands ---- */

static void cmd_check(const char *path, uint64_t chunk) {
    const int fd = open_file(path, O_RDONLY);
    gguf g;
    parse(fd, &g);
    if (!g.have_enc || memcmp(g.enc, ENC_ORIG, ENC_LEN))
        die("%s is not an original e4m3_e8m0_32_row264 file", path);
    const layout l = original_layout(&g);
    const identity id = file_identity(fd);
    remove_side(path, ".eq4-check");   /* a failed check leaves none behind */
    uint8_t *src = malloc((size_t)chunk * SRC_ROW), *dst = malloc((size_t)chunk * DST_ROW);
    if (!src || !dst) die("out of memory");
    CC_SHA256_CTX sha;
    CC_SHA256_Init(&sha);
    const uint64_t rel[2] = {l.rel0, l.rel1}, rows[2] = {l.rows0, l.rows1};
    for (int t = 0; t < 2; t++) {
        for (uint64_t r = 0; r < rows[t]; r += chunk) {
            const uint64_t n = rows[t] - r < chunk ? rows[t] - r : chunk;
            pread_full(fd, src, (size_t)n * SRC_ROW, l.data_start + rel[t] + r * SRC_ROW);
            const uint64_t bad = encode_rows(src, dst, n);
            if (bad != UINT64_MAX)
                die("%s row %" PRIu64 " cannot be encoded (NaN code, scale 255 or a scale out of range)",
                    TABLE_NAMES[t], r + bad);
            CC_SHA256_Update(&sha, dst, (CC_LONG)(n * DST_ROW));
        }
        if (t == 0) {
            uint64_t pad = new_rel1(&l) - (l.rel0 + l.rows0 * DST_ROW);
            memset(dst, 0, (size_t)chunk * DST_ROW);
            while (pad) {
                const uint64_t k = pad < chunk * DST_ROW ? pad : chunk * DST_ROW;
                CC_SHA256_Update(&sha, dst, (CC_LONG)k);
                pad -= k;
            }
        }
    }
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    char text[65];
    CC_SHA256_Final(digest, &sha);
    hex(digest, text);
    write_check(path, &l, &id, text);
    printf("check ok: %" PRIu64 " + %" PRIu64 " rows; converted size %" PRIu64 " bytes; tail sha256 %s\n",
           l.rows0, l.rows1, new_end(&l), text);
    free(src);
    free(dst);
    close(fd);
}

static void cmd_convert(const char *path, uint64_t chunk) {
    const int fd = open_file(path, O_RDWR);
    gguf g;
    parse(fd, &g);
    if (!g.have_enc || !g.t[0].found || !g.t[1].found) die("%s is not a V4.1 Engram GGUF", path);
    layout l;
    char sha[65];
    conv_state s;
    const int resume = read_state(path, &s);
    if (!resume) {
        if (!memcmp(g.enc, ENC_Q4, ENC_LEN)) die("%s is already converted", path);
        if (!memcmp(g.enc, ENC_BUSY, ENC_LEN)) die("%s is mid-conversion but its state file is gone", path);
        if (memcmp(g.enc, ENC_ORIG, ENC_LEN)) die("unexpected Engram encoding in %s", path);
        const layout here = original_layout(&g);
        identity checked;
        if (!read_check(path, &l, &checked, sha)) die("run --check on %s first", path);
        const identity now = file_identity(fd);
        if (memcmp(&here, &l, sizeof(l)) || memcmp(&now, &checked, sizeof(now)))
            die("%s changed since --check; rerun --check", path);
    } else {
        if (!read_check(path, &l, NULL, sha)) die("the check file of %s is gone", path);
        if (memcmp(&s.l, &l, sizeof(l))) die("state and check files of %s disagree", path);
        const int busy = !memcmp(g.enc, ENC_BUSY, ENC_LEN), orig = !memcmp(g.enc, ENC_ORIG, ENC_LEN),
                  done = !memcmp(g.enc, ENC_Q4, ENC_LEN);
        if ((s.phase == 1 && !busy && !orig) || (s.phase == 2 && !busy) || (s.phase == 3 && !busy && !done))
            die("the header of %s does not match its state file", path);
        printf("resuming phase %u at table %u row %" PRIu64 "\n", s.phase, s.table, s.next);
    }
    refuse_if_open(path);
    if (!resume) {
        remove_side(path, ".eq4-journal");   /* never feed a fresh start from an abandoned run */
        write_state(path, 1, 0, 0, &l);
        s.phase = 1;
        s.table = 0;
        s.next = 0;
    }
    if (s.phase == 1) {
        pwrite_full(fd, ENC_BUSY, ENC_LEN, g.enc_pos);
        sync_fd(fd);
        test_stop("block");
        write_state(path, 2, 0, 0, &l);
        s.phase = 2;
        s.table = 0;
        s.next = 0;
    }
    const uint64_t abs_old[2] = {l.data_start + l.rel0, l.data_start + l.rel1};
    const uint64_t abs_new[2] = {l.data_start + l.rel0, l.data_start + new_rel1(&l)};
    const uint64_t rows[2] = {l.rows0, l.rows1};
    if (s.phase == 2) {
        uint8_t *src = malloc((size_t)chunk * SRC_ROW), *dst = malloc((size_t)chunk * DST_ROW);
        if (!src || !dst) die("out of memory");
        for (uint32_t t = s.table; t < 2; t++) {
            uint64_t r = t == s.table ? s.next : 0;
            if (t == 1 && r == 0) {   /* zero the gap before table 2 (idempotent) */
                const uint64_t lo = abs_new[0] + l.rows0 * DST_ROW, hi = abs_new[1];
                memset(dst, 0, (size_t)chunk * DST_ROW);
                for (uint64_t o = lo; o < hi; o += chunk * DST_ROW) {
                    const uint64_t k = hi - o < chunk * DST_ROW ? hi - o : chunk * DST_ROW;
                    pwrite_full(fd, dst, (size_t)k, o);
                }
                sync_fd(fd);
            }
            while (r < rows[t]) {
                uint64_t n = rows[t] - r < chunk ? rows[t] - r : chunk;
                const uint64_t journaled = read_journal(path, t, r, chunk, src);
                if (journaled) n = journaled;
                else pread_full(fd, src, (size_t)n * SRC_ROW, abs_old[t] + r * SRC_ROW);
                const uint64_t src_lo = abs_old[t] + r * SRC_ROW, src_hi = src_lo + n * SRC_ROW;
                const uint64_t dst_lo = abs_new[t] + r * DST_ROW, dst_hi = dst_lo + n * DST_ROW;
                const uint64_t bad = encode_rows(src, dst, n);
                if (bad != UINT64_MAX)
                    die("%s row %" PRIu64 " cannot be encoded although --check passed", TABLE_NAMES[t], r + bad);
                if (!journaled && dst_lo < src_hi && src_lo < dst_hi) {
                    write_journal(path, t, r, n, src);
                    test_stop("journal");
                }
                pwrite_full(fd, dst, (size_t)n * DST_ROW, dst_lo);
                sync_fd(fd);
                test_stop("write");
                r += n;
                if (r == rows[t]) write_state(path, 2, t + 1, 0, &l);
                else write_state(path, 2, t, r, &l);
                test_stop("chunk");
                remove_side(path, ".eq4-journal");
            }
        }
        free(src);
        free(dst);
        write_state(path, 3, 2, 0, &l);
        s.phase = 3;
    }
    /* phase 3: the final header, then the new size. */
    const uint64_t width = DST_ROW, rel1 = new_rel1(&l);
    pwrite_full(fd, &width, 8, g.t[0].dim_pos);
    pwrite_full(fd, &width, 8, g.t[1].dim_pos);
    pwrite_full(fd, &l.rel0, 8, g.t[0].off_pos);
    pwrite_full(fd, &rel1, 8, g.t[1].off_pos);
    sync_fd(fd);
    test_stop("geometry");
    /* Only now the encoding: ds4 must never see the new name over old geometry. */
    pwrite_full(fd, ENC_Q4, ENC_LEN, g.enc_pos);
    sync_fd(fd);
    test_stop("header");
    if (ftruncate(fd, (off_t)new_end(&l)) != 0) die("ftruncate failed: %s", strerror(errno));
    sync_fd(fd);
    close(fd);
    remove_side(path, ".eq4-journal");
    remove_side(path, ".eq4-state");
    sync_dir(path);
    printf("converted: %s is now %" PRIu64 " bytes (lloyd4_e5m3_32_r136)\n", path, new_end(&l));
}

/* Header and size of a converted file match the layout from its check file. */
static layout converted_layout(const char *path, const gguf *g) {
    layout l;
    char sha[65];
    if (!read_check(path, &l, NULL, sha)) die("no check file for %s", path);
    if (!g->have_enc || memcmp(g->enc, ENC_Q4, ENC_LEN)) die("%s is not converted", path);
    if (!g->t[0].found || !g->t[1].found || g->t[0].width != DST_ROW || g->t[1].width != DST_ROW ||
        g->t[0].rows != l.rows0 || g->t[1].rows != l.rows1 || g->t[0].rel != l.rel0 ||
        g->t[1].rel != new_rel1(&l) || g->data_start != l.data_start || g->file_size != new_end(&l))
        die("the converted header or size of %s does not match its check file", path);
    return l;
}

static void cmd_verify(const char *path, uint64_t chunk) {
    char state[4200];
    side(state, sizeof(state), path, ".eq4-state");
    if (access(state, F_OK) == 0) die("%s is mid-conversion; finish --convert first", path);
    const int fd = open_file(path, O_RDONLY);
    gguf g;
    parse(fd, &g);
    const layout l = converted_layout(path, &g);
    layout dummy;
    char want[65];
    (void)read_check(path, &dummy, NULL, want);
    const size_t bytes = (size_t)chunk * DST_ROW;
    uint8_t *p = malloc(bytes);
    if (!p) die("out of memory");
    CC_SHA256_CTX tail, whole;
    CC_SHA256_Init(&tail);
    CC_SHA256_Init(&whole);
    const uint64_t start = l.data_start + l.rel0, end = new_end(&l);
    for (uint64_t o = 0, k; o < start; o += k) {
        k = start - o < bytes ? start - o : bytes;
        pread_full(fd, p, (size_t)k, o);
        CC_SHA256_Update(&whole, p, (CC_LONG)k);
    }
    for (uint64_t o = start, k; o < end; o += k) {
        k = end - o < bytes ? end - o : bytes;
        pread_full(fd, p, (size_t)k, o);
        CC_SHA256_Update(&whole, p, (CC_LONG)k);
        CC_SHA256_Update(&tail, p, (CC_LONG)k);
    }
    uint8_t d[CC_SHA256_DIGEST_LENGTH];
    char got[65], file[65];
    CC_SHA256_Final(d, &tail);
    hex(d, got);
    CC_SHA256_Final(d, &whole);
    hex(d, file);
    free(p);
    close(fd);
    if (strcmp(got, want)) die("tail sha256 %s does not match the check file (%s)", got, want);
    printf("verify ok: tail sha256 %s\nfile sha256 %s  %s\n", got, file, path);
}

static uint64_t xorshift64(uint64_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

static void cmd_sample(const char *path, uint32_t n, const char *out) {
    const int fd = open_file(path, O_RDONLY);
    gguf g;
    parse(fd, &g);
    if (memcmp(g.enc, ENC_ORIG, ENC_LEN)) die("--sample runs on the original file");
    const layout l = original_layout(&g);
    char tmp[] = "/tmp/" TOOL "-sample-XXXXXX";
    const int tfd = mkstemp(tmp);
    if (tfd < 0) die("mkstemp failed");
    unlink(tmp);
    uint32_t *ids = malloc((size_t)n * 4), *seq = malloc((size_t)n * 4);
    float *vals = malloc((size_t)n * DS4_ENGRAM_DIM * sizeof(float));
    if (!ids || !seq || !vals) die("out of memory");
    FILE *fp = fopen(out, "wb");
    if (!fp) die("cannot create %s", out);
    fwrite("EQ4SMPL1", 1, 8, fp);
    fwrite(&n, 4, 1, fp);
    const uint64_t rel[2] = {l.rel0, l.rel1}, rows[2] = {l.rows0, l.rows1};
    for (int t = 0; t < 2; t++) {
        uint64_t state = 0x9e3779b97f4a7c15ull ^ (uint64_t)(t + 1);
        for (uint32_t i = 0; i < n; i++) {
            ids[i] = (uint32_t)(xorshift64(&state) % rows[t]);
            seq[i] = i;
            uint8_t src[SRC_ROW], dst[DST_ROW];
            pread_full(fd, src, SRC_ROW, l.data_start + rel[t] + (uint64_t)ids[i] * SRC_ROW);
            if (!ds4_engram_encode_row136(src, dst)) die("%s row %u cannot be encoded", TABLE_NAMES[t], ids[i]);
            pwrite_full(tfd, dst, DST_ROW, (uint64_t)i * DST_ROW);
        }
        const ds4_engram_table table = {.fd = tfd, .offset = 0, .rows = n,
                                        .encoding = DS4_ENGRAM_ENC_LLOYD4_ROW136};
        if (!ds4_engram_read(&table, seq, n, vals)) die("decode of the sampled rows failed");
        fwrite(ids, 4, n, fp);
        fwrite(vals, sizeof(float), (size_t)n * DS4_ENGRAM_DIM, fp);
    }
    if (fclose(fp) != 0) die("cannot write %s", out);
    close(tfd);
    close(fd);
    free(ids);
    free(seq);
    free(vals);
    printf("sample ok: %u rows per table -> %s\n", n, out);
}

static void cmd_verify_sample(const char *path, const char *out) {
    const int fd = open_file(path, O_RDONLY);
    gguf g;
    parse(fd, &g);
    close(fd);
    if (!g.have_enc || memcmp(g.enc, ENC_Q4, ENC_LEN) || !g.t[0].found || !g.t[1].found ||
        g.t[0].width != DST_ROW || g.t[1].width != DST_ROW)
        die("%s is not converted", path);
    FILE *fp = fopen(out, "rb");
    if (!fp) die("cannot open %s", out);
    char magic[8];
    uint32_t n = 0;
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "EQ4SMPL1", 8) || fread(&n, 4, 1, fp) != 1 || !n)
        die("%s is not a sample file", out);
    uint32_t *ids = malloc((size_t)n * 4);
    float *want = malloc((size_t)n * DS4_ENGRAM_DIM * sizeof(float));
    float *got = malloc((size_t)n * DS4_ENGRAM_DIM * sizeof(float));
    if (!ids || !want || !got) die("out of memory");
    uint64_t mismatched = 0;
    for (int t = 0; t < 2; t++) {
        if (fread(ids, 4, n, fp) != n ||
            fread(want, sizeof(float), (size_t)n * DS4_ENGRAM_DIM, fp) != (size_t)n * DS4_ENGRAM_DIM)
            die("%s is truncated", out);
        ds4_engram_table table;
        if (!ds4_engram_table_open(&table, path, g.data_start + g.t[t].rel, (uint32_t)g.t[t].rows,
                                   DS4_ENGRAM_ENC_LLOYD4_ROW136))
            die("cannot open %s: %s", TABLE_NAMES[t], strerror(errno));
        if (!ds4_engram_read(&table, ids, n, got)) die("reading %s failed: %s", TABLE_NAMES[t], strerror(errno));
        ds4_engram_table_close(&table);
        for (uint32_t i = 0; i < n; i++)
            if (memcmp(got + (size_t)i * DS4_ENGRAM_DIM, want + (size_t)i * DS4_ENGRAM_DIM,
                       DS4_ENGRAM_DIM * sizeof(float)))
                mismatched++;
    }
    fclose(fp);
    free(ids);
    free(want);
    free(got);
    if (mismatched) die("%" PRIu64 " sampled rows differ", mismatched);
    printf("verify-sample ok: %u rows per table are bit-identical\n", n);
}

static void usage(void) {
    die("usage: " TOOL " --check FILE | --sample FILE N OUT | --convert FILE | --verify FILE |"
        " --verify-sample FILE OUT  [--chunk-rows K]");
}

int main(int argc, char **argv) {
    if (argc < 3) usage();
    const char *mode = argv[1], *path = argv[2];
    const char *args[2] = {NULL, NULL};
    int nargs = 0;
    uint64_t chunk = 262144;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--chunk-rows") && i + 1 < argc) {
            chunk = strtoull(argv[++i], NULL, 10);
            if (!chunk || chunk > (1u << 24)) die("bad --chunk-rows");
        } else if (nargs < 2) {
            args[nargs++] = argv[i];
        } else {
            usage();
        }
    }
    if (!strcmp(mode, "--check") && nargs == 0) cmd_check(path, chunk);
    else if (!strcmp(mode, "--convert") && nargs == 0) cmd_convert(path, chunk);
    else if (!strcmp(mode, "--verify") && nargs == 0) cmd_verify(path, chunk);
    else if (!strcmp(mode, "--sample") && nargs == 2) {
        const unsigned long n = strtoul(args[0], NULL, 10);
        if (!n || n > 1000000) die("bad sample count");
        cmd_sample(path, (uint32_t)n, args[1]);
    } else if (!strcmp(mode, "--verify-sample") && nargs == 1) cmd_verify_sample(path, args[0]);
    else usage();
    return 0;
}
