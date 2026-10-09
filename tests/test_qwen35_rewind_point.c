/* Model-backed oracle for the Ornith rewind point (ds4_session_mark_rewind_point).
 *
 * Run with the production ds4 slot stopped (one model process on 64 GB):
 *   DS4_TEST_MODEL=/path/to/Ornith.gguf make test-qwen35-rewind-point
 *   DS4_TEST_MTP=1 also opens the embedded MTP block (ds4-server --mtp).
 *
 * A = about 6K prompt tokens, B = 300 greedy reply tokens, C = 64 new tokens.
 * Session X: sync A, mark, decode B, rewind to |A|, eval C one token at a time.
 * Session Y: sync A, eval C one token at a time (never rewound).
 * Every row of C must match: same argmax, max |dlogit| <= 1e-2.  X is then
 * rewound to |A| a second time and must match again (restore is a copy).
 * A rewind below the point drops it, and so does a payload load. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CTX 16384
#define A_TOKENS 6000
#define B_TOKENS 300
#define C_TOKENS 64

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) fail("open tests/long_context_story_prompt.txt");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc((size_t)n + 1);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) fail("read prompt file");
    p[n] = 0;
    fclose(f);
    return p;
}

static void eval_compare(ds4_session *x, ds4_session *y, const ds4_tokens *c, int vocab,
                         float *lx, float *ly, const char *pass) {
    char err[256];
    float worst = 0.0f;
    for (int i = 0; i < c->len; i++) {
        if (ds4_session_eval(x, c->v[i], err, sizeof(err))) fail(err);
        if (ds4_session_eval(y, c->v[i], err, sizeof(err))) fail(err);
        if (ds4_session_copy_logits(x, lx, vocab) != vocab ||
            ds4_session_copy_logits(y, ly, vocab) != vocab) fail("copy logits");
        if (ds4_session_argmax(x) != ds4_session_argmax(y)) {
            fprintf(stderr, "FAIL: %s argmax differs at row %d\n", pass, i);
            exit(1);
        }
        for (int k = 0; k < vocab; k++) {
            float d = fabsf(lx[k] - ly[k]);
            if (d > worst) worst = d;
        }
    }
    printf("%s: %d rows, max |dlogit| %.6g\n", pass, c->len, worst);
    if (!(worst <= 1e-2f)) fail("logits differ after rewind");
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) fail("set DS4_TEST_MODEL to the Ornith GGUF");
    const char *mtp_env = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = CTX,
        .glm_mtp = mtp_env && mtp_env[0] == '1',
        .mtp_draft_tokens = 1,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int vocab = ds4_engine_vocab_size(e);

    char *text = read_file("tests/long_context_story_prompt.txt");
    ds4_tokens all = {0}, a = {0}, c = {0};
    ds4_tokenize_text(e, text, &all);
    if (all.len < A_TOKENS + C_TOKENS) fail("prompt file too short");
    for (int i = 0; i < A_TOKENS; i++) ds4_tokens_push(&a, all.v[i]);
    for (int i = 0; i < C_TOKENS; i++) ds4_tokens_push(&c, all.v[all.len - C_TOKENS + i]);

    ds4_session *x = NULL, *y = NULL;
    char err[256];
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync(x, &a, err, sizeof(err))) fail(err);
    if (ds4_session_sync(y, &a, err, sizeof(err))) fail(err);

    if (!ds4_session_mark_rewind_point(x)) fail("mark rewind point");
    if (ds4_session_rewind_point_pos(x) != A_TOKENS) fail("rewind point position");

    for (int i = 0; i < B_TOKENS; i++)
        if (ds4_session_eval(x, ds4_session_argmax(x), err, sizeof(err))) fail(err);

    ds4_session_rewind(x, A_TOKENS);
    if (!ds4_session_checkpoint_valid(x) || ds4_session_pos(x) != A_TOKENS)
        fail("rewind to the point did not keep a valid checkpoint");
    float *lx = malloc((size_t)vocab * sizeof(float));
    float *ly = malloc((size_t)vocab * sizeof(float));
    if (!lx || !ly) fail("alloc logits");
    eval_compare(x, y, &c, vocab, lx, ly, "first rewind");

    /* second rewind: X back to |A|, Y re-synced to A from scratch */
    ds4_session_rewind(x, A_TOKENS);
    if (!ds4_session_checkpoint_valid(x)) fail("second rewind lost the checkpoint");
    ds4_session_invalidate(y);
    if (ds4_session_sync(y, &a, err, sizeof(err))) fail(err);
    eval_compare(x, y, &c, vocab, lx, ly, "second rewind");

    /* rewinding below the point drops it */
    ds4_session_rewind(x, A_TOKENS - 1);
    if (ds4_session_rewind_point_pos(x) != -1) fail("rewind below the point kept it");

    /* stale after load: a payload loaded into a session drops its point */
    ds4_session_invalidate(x);
    if (ds4_session_sync(x, &a, err, sizeof(err))) fail(err);
    if (!ds4_session_mark_rewind_point(x)) fail("re-mark");
    ds4_session_snapshot snap = {0};
    if (ds4_session_save_snapshot(y, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_load_snapshot(x, &snap, err, sizeof(err))) fail(err);
    if (ds4_session_rewind_point_pos(x) != -1) fail("stale after load: point survived a payload load");
    ds4_session_snapshot_free(&snap);

    printf("PASS: rewind point (mtp=%d)\n", opt.glm_mtp ? 1 : 0);
    free(lx);
    free(ly);
    free(text);
    ds4_tokens_free(&all);
    ds4_tokens_free(&a);
    ds4_tokens_free(&c);
    ds4_session_free(x);
    ds4_session_free(y);
    ds4_engine_close(e);
    return 0;
}
