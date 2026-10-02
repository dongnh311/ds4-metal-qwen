#define _DARWIN_C_SOURCE
/*
 * deepseek41_engram_q4 on a small V4.1-shaped GGUF: the two Engram tables
 * last, alignment 16384, table 1 ending unaligned and the file padded to the
 * alignment like the real one. Every run is compared with the file built
 * here byte for byte: the header patched in place and the tables re-encoded
 * with ds4_engram_encode_row136 and compacted forward.
 */
#include "ds4_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

enum { ALIGN = 16384, ROWS1 = 3001, ROWS2 = 2999 };

static const char *tool;
static uint32_t rng = 88172645u;

static uint32_t next_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

typedef struct {
    uint8_t *bytes;
    size_t size, cap;
} buf;

static void put(buf *b, const void *p, size_t n) {
    if (b->size + n > b->cap) {
        b->cap = (b->size + n) * 2;
        b->bytes = realloc(b->bytes, b->cap);
        assert(b->bytes);
    }
    memcpy(b->bytes + b->size, p, n);
    b->size += n;
}
static void put32(buf *b, uint32_t v) { put(b, &v, 4); }
static void put64(buf *b, uint64_t v) { put(b, &v, 8); }
static void putstr(buf *b, const char *s) { put64(b, strlen(s)); put(b, s, strlen(s)); }
static uint64_t align_up(uint64_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }

typedef struct {
    buf src;             /* the original file */
    buf want;            /* the file after conversion */
    size_t enc_pos, dim_pos[2], off_pos[2];
} fixture;

/* bad_scale: one source scale byte of table 2 is 255, which --check refuses. */
static void build(fixture *f, int bad_scale, const char *arch) {
    memset(f, 0, sizeof(*f));
    buf *h = &f->src;
    put32(h, 0x46554747u); put32(h, 3); put64(h, 3); put64(h, 5);
    putstr(h, "general.architecture"); put32(h, 8); putstr(h, arch);
    putstr(h, "deepseek41.engram.rows"); put32(h, 9); put32(h, 4); put64(h, 2);
    put32(h, ROWS1); put32(h, ROWS2);
    putstr(h, "test.names"); put32(h, 9); put32(h, 8); put64(h, 2);
    putstr(h, "alpha"); putstr(h, "beta");
    putstr(h, "deepseek41.engram.encoding"); put32(h, 8); put64(h, 19);
    f->enc_pos = h->size;
    put(h, "e4m3_e8m0_32_row264", 19);
    putstr(h, "general.alignment"); put32(h, 4); put32(h, ALIGN);
    const uint64_t rel1 = ALIGN, rel2 = align_up(rel1 + (uint64_t)ROWS1 * 264);
    const char *names[2] = {"blk.1.engram_embd.weight", "blk.14.engram_embd.weight"};
    const uint64_t rows[2] = {ROWS1, ROWS2}, rel[2] = {rel1, rel2};
    putstr(h, "test.weight"); put32(h, 2); put64(h, 16); put64(h, 1); put32(h, 0); put64(h, 0);
    for (int t = 0; t < 2; t++) {
        putstr(h, names[t]); put32(h, 2);
        f->dim_pos[t] = h->size;
        put64(h, 264); put64(h, rows[t]); put32(h, 24);
        f->off_pos[t] = h->size;
        put64(h, rel[t]);
    }
    const size_t data_start = align_up(h->size);
    assert(data_start == ALIGN);
    const uint64_t end2 = data_start + rel2 + (uint64_t)ROWS2 * 264;
    const size_t size = align_up(end2);
    h->bytes = realloc(h->bytes, size);
    assert(h->bytes);
    memset(h->bytes + h->size, 0, size - h->size);
    for (size_t i = data_start; i < data_start + 64; i++) h->bytes[i] = (uint8_t)next_u32();
    h->size = size;
    h->cap = size;
    for (int t = 0; t < 2; t++) {
        uint8_t *row = h->bytes + data_start + rel[t];
        for (uint64_t r = 0; r < rows[t]; r++, row += 264) {
            for (int j = 0; j < 256; j++) {
                uint8_t c;
                do c = (uint8_t)next_u32(); while ((c & 127) == 127);
                row[j] = c;
            }
            for (int g = 0; g < 8; g++) row[256 + g] = (uint8_t)(117 + next_u32() % 6);
        }
    }
    if (bad_scale) h->bytes[data_start + rel2 + 1234 * 264 + 260] = 255;
    /* The converted file. */
    const uint64_t new_rel2 = align_up(rel1 + (uint64_t)ROWS1 * 136);
    const size_t new_size = data_start + new_rel2 + (size_t)ROWS2 * 136;
    f->want.bytes = calloc(1, new_size);
    assert(f->want.bytes);
    f->want.size = new_size;
    memcpy(f->want.bytes, h->bytes, data_start + rel1);
    memcpy(f->want.bytes + f->enc_pos, "lloyd4_e5m3_32_r136", 19);
    const uint64_t new_rel[2] = {rel1, new_rel2}, w = 136;
    for (int t = 0; t < 2; t++) {
        memcpy(f->want.bytes + f->dim_pos[t], &w, 8);
        memcpy(f->want.bytes + f->off_pos[t], &new_rel[t], 8);
        for (uint64_t r = 0; r < rows[t]; r++) {
            const uint8_t *s = h->bytes + data_start + rel[t] + r * 264;
            uint8_t *d = f->want.bytes + data_start + new_rel[t] + r * 136;
            if (!ds4_engram_encode_row136(s, d)) assert(bad_scale);
        }
    }
}

static void write_file(const char *path, const buf *b) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    assert(fd >= 0);
    assert(write(fd, b->bytes, b->size) == (ssize_t)b->size);
    assert(close(fd) == 0);
}

static int same_file(const char *path, const buf *b) {
    const int fd = open(path, O_RDONLY);
    assert(fd >= 0);
    struct stat st;
    assert(fstat(fd, &st) == 0);
    int same = (size_t)st.st_size == b->size;
    uint8_t *got = malloc(b->size ? b->size : 1);
    assert(got);
    if (same) same = read(fd, got, b->size) == (ssize_t)b->size && !memcmp(got, b->bytes, b->size);
    free(got);
    close(fd);
    return same;
}

static void side_paths(const char *path, char check[512], char state[512], char journal[512]) {
    snprintf(check, 512, "%s.eq4-check", path);
    snprintf(state, 512, "%s.eq4-state", path);
    snprintf(journal, 512, "%s.eq4-journal", path);
}

static void clean(const char *path) {
    char check[512], state[512], journal[512];
    side_paths(path, check, state, journal);
    unlink(path);
    unlink(check);
    unlink(state);
    unlink(journal);
}

/* Run the tool; stop is DS4_ENGRAM_Q4_TEST_STOP or NULL. Returns the exit code. */
static int run(const char *stop, const char *a1, const char *a2, const char *a3, const char *a4,
               const char *a5) {
    const pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        if (stop) setenv("DS4_ENGRAM_Q4_TEST_STOP", stop, 1);
        else unsetenv("DS4_ENGRAM_Q4_TEST_STOP");
        const int log = open("/tmp/ds4-eq4-test.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (log >= 0) {
            dup2(log, 1);
            dup2(log, 2);
        }
        execl(tool, tool, a1, a2, a3, a4, a5, (char *)NULL);
        _exit(127);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

static void test_plain(const fixture *f, const char *path) {
    char check[512], state[512], journal[512];
    side_paths(path, check, state, journal);
    clean(path);
    write_file(path, &f->src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    assert(access(check, F_OK) == 0);
    assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 0);
    assert(same_file(path, &f->want));
    assert(access(state, F_OK) != 0 && access(journal, F_OK) != 0);
    assert(run(NULL, "--verify", path, NULL, NULL, NULL) == 0);
    /* Already converted: refused, file unchanged. */
    assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 1);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 1);
    assert(same_file(path, &f->want));
    puts("engram q4 convert: check, convert, verify: PASS");
}

static void test_resume(const fixture *f, const char *path) {
    /* write:1 and write:2 stop after a chunk overwrote part of its own source:
     * only the journal can restore it. */
    static const char *stops[] = {"block:1", "journal:1", "journal:2", "write:1", "write:2",
                                  "write:3", "chunk:1", "chunk:5", "chunk:60", "geometry:1",
                                  "header:1"};
    for (size_t i = 0; i < sizeof(stops) / sizeof(stops[0]); i++) {
        clean(path);
        write_file(path, &f->src);
        assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
        assert(run(stops[i], "--convert", path, "--chunk-rows", "64", NULL) == 3);
        assert(!same_file(path, &f->want));
        assert(run(NULL, "--convert", path, "--chunk-rows", "64", NULL) == 0);
        if (!same_file(path, &f->want)) {
            fprintf(stderr, "resume after %s: file differs\n", stops[i]);
            assert(0);
        }
        assert(run(NULL, "--verify", path, NULL, NULL, NULL) == 0);
    }
    /* A journal left by an abandoned run must not feed a fresh start. */
    fixture other;
    build(&other, 0, "deepseek41");
    char check[512], state[512], journal[512];
    side_paths(path, check, state, journal);
    clean(path);
    write_file(path, &f->src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    assert(run("chunk:1", "--convert", path, "--chunk-rows", "64", NULL) == 3);
    assert(access(journal, F_OK) == 0);
    write_file(path, &other.src);
    unlink(state);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    assert(run(NULL, "--convert", path, "--chunk-rows", "64", NULL) == 0);
    assert(same_file(path, &other.want));
    free(other.src.bytes);
    free(other.want.bytes);
    puts("engram q4 convert: resume after every phase: PASS");
}

static void test_refusals(const fixture *f, const char *path) {
    fixture bad;
    build(&bad, 1, "deepseek41");
    clean(path);
    write_file(path, &bad.src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 1);
    assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 1);
    assert(same_file(path, &bad.src));
    free(bad.src.bytes);
    free(bad.want.bytes);
    fixture other;
    build(&other, 0, "llama");
    clean(path);
    write_file(path, &other.src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 1);
    assert(same_file(path, &other.src));
    free(other.src.bytes);
    free(other.want.bytes);
    /* The file changed after --check: --convert refuses before writing. */
    clean(path);
    write_file(path, &f->src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    {
        const int fd = open(path, O_WRONLY);
        assert(fd >= 0);
        const uint64_t rel2 = align_up(ALIGN + (uint64_t)ROWS1 * 264);
        const uint8_t bad = 255;
        assert(pwrite(fd, &bad, 1, (off_t)(ALIGN + rel2 + 1234 * 264 + 260)) == 1);
        close(fd);
        buf changed = {malloc(f->src.size), f->src.size, f->src.size};
        assert(changed.bytes);
        memcpy(changed.bytes, f->src.bytes, f->src.size);
        changed.bytes[ALIGN + rel2 + 1234 * 264 + 260] = 255;
        assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 1);
        assert(same_file(path, &changed));
        free(changed.bytes);
    }
    /* Another process holds the file open. */
    clean(path);
    write_file(path, &f->src);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    int ready[2];
    assert(pipe(ready) == 0);
    const pid_t holder = fork();
    assert(holder >= 0);
    if (holder == 0) {
        const int fd = open(path, O_RDONLY);
        char c = fd >= 0 ? 'y' : 'n';
        (void)!write(ready[1], &c, 1);
        pause();
        _exit(0);
    }
    char c = 0;
    assert(read(ready[0], &c, 1) == 1 && c == 'y');
    assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 1);
    assert(same_file(path, &f->src));
    kill(holder, SIGTERM);
    int status;
    assert(waitpid(holder, &status, 0) == holder);
    close(ready[0]);
    close(ready[1]);
    puts("engram q4 convert: refusals leave the file untouched: PASS");
}

static void test_sample(const fixture *f, const char *path) {
    char out[512];
    snprintf(out, sizeof(out), "%s.sample", path);
    clean(path);
    unlink(out);
    write_file(path, &f->src);
    assert(run(NULL, "--sample", path, "50", out, NULL) == 0);
    assert(run(NULL, "--check", path, NULL, NULL, NULL) == 0);
    assert(run(NULL, "--convert", path, NULL, NULL, NULL) == 0);
    assert(run(NULL, "--verify-sample", path, out, NULL, NULL) == 0);
    const int fd = open(out, O_RDWR);
    assert(fd >= 0);
    struct stat st;
    assert(fstat(fd, &st) == 0 && st.st_size > 4096);
    uint8_t byte;
    assert(pread(fd, &byte, 1, st.st_size - 3) == 1);
    byte ^= 0x40;
    assert(pwrite(fd, &byte, 1, st.st_size - 3) == 1);
    close(fd);
    assert(run(NULL, "--verify-sample", path, out, NULL, NULL) == 1);
    unlink(out);
    puts("engram q4 convert: sample before, verify-sample after: PASS");
}

int main(int argc, char **argv) {
    assert(argc == 2);
    tool = argv[1];
    unlink("/tmp/ds4-eq4-test.log");
    char path[] = "/tmp/ds4-eq4-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    fixture f;
    build(&f, 0, "deepseek41");
    test_plain(&f, path);
    test_resume(&f, path);
    test_refusals(&f, path);
    test_sample(&f, path);
    clean(path);
    free(f.src.bytes);
    free(f.want.bytes);
    puts("Engram 4-bit in-place conversion: PASS");
    return 0;
}
