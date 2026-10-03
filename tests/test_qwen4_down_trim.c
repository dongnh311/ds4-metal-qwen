/* Model-free checks for trimmed Q4_K down rows
 * (docs/superpowers/specs/2026-10-03-qwen4-q4k-down-trim-design.md). */
#include "../ds4.c"
#include <sys/wait.h>

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

static void check_row_bytes(void) {
    CHECK(q4k_row_bytes(768) == 432);
    CHECK(q4k_row_bytes(640) == 368);
    CHECK(q4k_row_bytes(320) == 192);
    CHECK(q4k_row_bytes(704) == 400);
    CHECK(q4k_row_bytes(672) == 0);
    uint64_t b = 0;
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q4_K, 640, 640ull * 2560 * 512, &b) && b == 368ull * 2560 * 512);
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q4_K, 768, 768ull * 2560 * 512, &b) && b == 432ull * 2560 * 512);
    CHECK(!tensor_nbytes_dims(DS4_TENSOR_Q4_K, 672, 672ull * 4, &b));
    CHECK(tensor_nbytes_dims(DS4_TENSOR_Q8_0, 640, 640ull * 2560, &b) && b == 20ull * 34 * 2560);
    ds4_tensor t = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(routed_expert_row_bytes(&t) == 368);
    t.dim[0] = 768;
    CHECK(routed_expert_row_bytes(&t) == 432);
}

static void check_trimmed_layouts(void) {
    g_ds4_shape.n_ff_exp = 640;
    ds4_tensor t = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(qwen4_down_is_trimmed(&t));
    t.dim[0] = 768;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* padded */
    t.dim[0] = 640; t.type = DS4_TENSOR_Q8_0;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* MTP Q8_0 down: not a Q4_K row */
    g_ds4_shape.n_ff_exp = 512; t.type = DS4_TENSOR_Q4_K; t.dim[0] = 512;
    CHECK(!qwen4_down_is_trimmed(&t));                 /* whole blocks: nothing to trim */
    g_ds4_shape.n_ff_exp = 640;

#ifdef DS4_HAS_QWEN4_GPU
    /* the Metal graph runs trimmed down rows; a Q4_K row of 640 that is not the ff width stays refused */
    ds4_tensor down = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(qwen4_graph_expert_ok(&down));
    ds4_tensor odd = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    g_ds4_shape.n_ff_exp = 512;
    CHECK(!qwen4_graph_expert_ok(&odd));
    g_ds4_shape.n_ff_exp = 640;
    ds4_tensor gate = { .ndim = 3, .dim = {2560, 640, 512}, .type = DS4_TENSOR_Q4_K };
    CHECK(qwen4_graph_expert_ok(&gate));
#endif

    /* a Q8_0 MTP down beside trimmed Q4_K layers is accepted (the call returns) */
    const bool tr[3] = { true, true, false };
    const uint32_t ty[3] = { DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K, DS4_TENSOR_Q8_0 };
    qwen4_check_down_layouts(tr, ty, 3);
    /* trimmed and padded Q4_K in one model is refused with exit status 1 */
    fflush(NULL);
    const pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(2);
        const bool mix[2] = { true, false };
        const uint32_t mt[2] = { DS4_TENSOR_Q4_K, DS4_TENSOR_Q4_K };
        qwen4_check_down_layouts(mix, mt, 2);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1);
}

/* the bound nextn (MTP) layers after the executable range count toward the down layout, so a
 * padded Q4_K MTP down beside trimmed main layers is refused instead of read at the wrong stride */
static void check_collect_includes_nextn(void) {
    static ds4_weights w;
    const uint32_t saved_layers = g_ds4_shape.n_layer;
    g_ds4_shape.n_layer = 4;
    g_ds4_shape.n_ff_exp = 640;
    static ds4_tensor main_down = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    static ds4_tensor mtp_pad = { .ndim = 3, .dim = {768, 2560, 512}, .type = DS4_TENSOR_Q4_K };
    static ds4_tensor mtp_q8 = { .ndim = 3, .dim = {640, 2560, 512}, .type = DS4_TENSOR_Q8_0 };
    for (uint32_t il = 0; il < 3; il++) w.layer[il].ffn_down_exps = &main_down;
    bool tr[DS4_MAX_LAYER];
    uint32_t ty[DS4_MAX_LAYER];
    /* layers 0..2 executable, layer 3 the nextn block */
    w.layer[3].ffn_down_exps = &mtp_pad;
    CHECK(qwen4_collect_down_layouts(&w, 0, tr, ty) == 4);
    CHECK(tr[0] && tr[2] && !tr[3] && ty[3] == DS4_TENSOR_Q4_K);
    w.layer[3].ffn_down_exps = &mtp_q8;
    CHECK(qwen4_collect_down_layouts(&w, 0, tr, ty) == 4);
    CHECK(!tr[3] && ty[3] == DS4_TENSOR_Q8_0);
    /* a slice that did not bind the nextn block: only the bound layers count */
    w.layer[3].ffn_down_exps = NULL;
    CHECK(qwen4_collect_down_layouts(&w, 0, tr, ty) == 3);
    g_ds4_shape.n_layer = saved_layers;
}

/* the CPU reference reads a trimmed row exactly like the first n values of its padded twin */
static void check_ref_row(void) {
    enum { ROWS = 3 };
    static uint8_t pad[ROWS * 432], trim[ROWS * 368];
    for (uint32_t i = 0; i < sizeof(pad); i++) pad[i] = (uint8_t)(i * 37u + 11u);
    for (uint32_t r = 0; r < ROWS; r++) {
        /* d and dmin = f16 0x2000 (2^-7) in every block header, so every value is finite */
        for (uint32_t b = 0; b < 3; b++) {
            uint8_t *h = pad + r * 432 + b * 144;
            h[0] = 0x00; h[1] = 0x20; h[2] = 0x00; h[3] = 0x20;
        }
        memcpy(trim + r * 368, pad + r * 432, 368);
    }
    float a[768], b[640];
    for (uint32_t r = 0; r < ROWS; r++) {
        qwen4_ref_q4k_row(pad, 768, r, a);
        qwen4_ref_q4k_row(trim, 640, r, b);
        CHECK(memcmp(a, b, sizeof(b)) == 0);
        bool nonzero = false;
        for (uint32_t i = 512; i < 640; i++) nonzero |= b[i] != 0.0f;
        CHECK(nonzero);                                /* the short block's real values were read */
    }
}

int main(void) {
    check_row_bytes();
    check_trimmed_layouts();
    check_collect_includes_nextn();
    check_ref_row();
    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_qwen4_down_trim: ok\n");
    return 0;
}
