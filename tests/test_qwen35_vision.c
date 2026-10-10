/* Model-backed checks for Ornith native vision.
 *
 * Run with the production ds4 slot held down (one model process on 64 GB):
 *   DS4_TEST_MODEL=<Ornith.gguf> DS4_TEST_MMPROJ=<mmproj> DS4_TEST_IMAGE=<jpeg> \
 *     make test-qwen35-vision
 * DS4_TEST_MTP=1 also opens the embedded MTP block.  Without DS4_TEST_MMPROJ
 * only the text-only checksum case runs (the Makefile compares the two). */
#include "ds4.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CTX 8192

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_engine *open_engine(bool with_vision, bool mtp) {
    const char *model = getenv("DS4_TEST_MODEL");
    const char *mmproj = getenv("DS4_TEST_MMPROJ");
    if (!model || !model[0]) fail("set DS4_TEST_MODEL to the Ornith GGUF");
    ds4_engine_options opt = {
        .model_path = model,
        .vision_path = with_vision ? mmproj : NULL,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = CTX,
        .prefill_chunk = 64,          /* chunk boundaries fall inside images */
        .glm_mtp = mtp,
        .mtp_draft_tokens = 1,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    return e;
}

static void case_encode(ds4_engine *e) {
    const char *image = getenv("DS4_TEST_IMAGE");
    if (!image || !image[0]) fail("set DS4_TEST_IMAGE to a JPEG or PNG");
    if (!ds4_engine_has_vision(e)) fail("engine has no vision after --vision");
    ds4_vision_embedding emb = {0};
    char err[256] = {0};
    if (!ds4_engine_vision_encode_file(e, image, &emb, err, sizeof(err))) fail(err);
    if (emb.token_count == 0 || emb.grid_width * emb.grid_height != emb.token_count)
        fail("image grid does not cover its rows");
    printf("encode: %u rows (%u x %u)\n", emb.token_count, emb.grid_height, emb.grid_width);
    ds4_vision_embedding_free(&emb);
}

/* Deterministic stand-in for an encoded image: grid gh x gw rows of n_embd. */
static ds4_vision_embedding fake_image(int n_embd, uint32_t gh, uint32_t gw, uint8_t seed) {
    ds4_vision_embedding e = {0};
    e.token_count = gh * gw;
    e.grid_height = gh;
    e.grid_width = gw;
    e.data = malloc((size_t)e.token_count * (size_t)n_embd * sizeof(float));
    if (!e.data) fail("alloc image rows");
    for (size_t i = 0; i < (size_t)e.token_count * (size_t)n_embd; i++)
        e.data[i] = 0.02f * sinf(0.37f * (float)i + (float)seed);
    memset(e.fingerprint, seed, sizeof(e.fingerprint));
    return e;
}

static void push_text(ds4_engine *e, ds4_tokens *out, const char *text) {
    ds4_tokens t = {0};
    ds4_tokenize_text(e, text, &t);
    for (int i = 0; i < t.len; i++) ds4_tokens_push(out, t.v[i]);
    ds4_tokens_free(&t);
}

/* text1 [image A 6x8] text2 [image B 5x4] text3: image A straddles the 64-token
 * prefill chunk boundary.  Spans own the image rows. */
static void build_prompt(ds4_engine *e, ds4_tokens *out, ds4_vision_span spans[2]) {
    char err[256];
    const int n = ds4_engine_embd_dim(e);
    push_text(e, out, "<|im_start|>user\nLook at the first picture carefully. ");
    ds4_vision_embedding a = fake_image(n, 6, 8, 11);
    if (!ds4_prompt_append_vision(e, out, &spans[0], &a, err, sizeof(err))) fail(err);
    push_text(e, out, " and now the second one, which is smaller: ");
    ds4_vision_embedding b = fake_image(n, 5, 4, 23);
    if (!ds4_prompt_append_vision(e, out, &spans[1], &b, err, sizeof(err))) fail(err);
    push_text(e, out, " What differs?<|im_end|>\n<|im_start|>assistant\n");
}

/* HF get_rope_index, written independently of qwen4_mrope_pos: text takes
 * (n, n, n) and advances n by one; an image takes (n, n+r, n+c) and then
 * advances n by max(rows, cols). */
static void expected_pos3(const ds4_vision_span *sp, size_t n_sp, int len, uint32_t (*out)[3]) {
    uint32_t next = 0;
    size_t k = 0;
    for (int p = 0; p < len;) {
        if (k < n_sp && (uint32_t)p == sp[k].token_start) {
            const uint32_t gh = sp[k].embedding.grid_height, gw = sp[k].embedding.grid_width;
            for (uint32_t r = 0; r < gh; r++)
                for (uint32_t c = 0; c < gw; c++, p++) {
                    out[p][0] = next;
                    out[p][1] = next + r;
                    out[p][2] = next + c;
                }
            next += gh > gw ? gh : gw;
            k++;
        } else {
            out[p][0] = out[p][1] = out[p][2] = next++;
            p++;
        }
    }
}

static void check_pos3(ds4_session *s, const uint32_t (*want)[3], int from, int to, const char *what) {
    for (int p = from; p < to; p++) {
        uint32_t got[3];
        if (!ds4_session_test_read_pos3(s, p, got)) fail("read pos3");
        if (memcmp(got, want[p], sizeof(got)) != 0) {
            fprintf(stderr, "FAIL: %s pos %d got (%u,%u,%u) want (%u,%u,%u)\n", what, p,
                    got[0], got[1], got[2], want[p][0], want[p][1], want[p][2]);
            exit(1);
        }
    }
}

/* positions after a cold multimodal prefill, then 8 decode evals; the same
 * prompt reached by a continuation (prefix up to the middle text, then the
 * rest) gives the same positions and the same last-row argmax */
static void case_positions(ds4_engine *e) {
    char err[256];
    ds4_tokens prompt = {0};
    ds4_vision_span spans[2];
    build_prompt(e, &prompt, spans);
    const int len = prompt.len, extra = 8;
    uint32_t (*want)[3] = malloc((size_t)(len + extra) * sizeof(*want));
    if (!want) fail("alloc positions");
    ds4_tokens full = {0};
    for (int i = 0; i < len; i++) ds4_tokens_push(&full, prompt.v[i]);
    for (int i = 0; i < extra; i++) ds4_tokens_push(&full, prompt.v[i % 8]); /* any text ids */
    expected_pos3(spans, 2, len + extra, want);

    ds4_session *x = NULL, *y = NULL;
    if (ds4_session_create(&x, e, CTX) || ds4_session_create(&y, e, CTX)) fail("session create");
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    check_pos3(x, (const uint32_t (*)[3])want, 0, len, "prefill");
    printf("positions: %d prefill rows match get_rope_index\n", len);
    for (int i = 0; i < extra; i++)
        if (ds4_session_eval(x, full.v[len + i], err, sizeof(err))) fail(err);
    check_pos3(x, (const uint32_t (*)[3])want, len, len + extra, "decode");
    printf("decode: %d eval rows continue the offset\n", extra);

    const int cut = (int)(spans[0].token_start + spans[0].embedding.token_count) + 4;
    ds4_tokens head = {0};
    for (int i = 0; i < cut; i++) ds4_tokens_push(&head, prompt.v[i]);
    if (ds4_session_sync_multimodal(y, &head, spans, 1, err, sizeof(err))) fail(err);
    if (ds4_session_sync_multimodal(y, &prompt, spans, 2, err, sizeof(err))) fail(err);
    check_pos3(y, (const uint32_t (*)[3])want, 0, len, "continuation");
    ds4_session_invalidate(x);
    if (ds4_session_sync_multimodal(x, &prompt, spans, 2, err, sizeof(err))) fail(err);
    if (ds4_session_argmax(x) != ds4_session_argmax(y)) fail("continuation changes the next token");
    printf("continuation: offset carried across syncs\n");
    ds4_session_free(x);
    ds4_session_free(y);
    free(want);
    ds4_tokens_free(&head);
    ds4_tokens_free(&full);
    ds4_tokens_free(&prompt);
}

/* text-only prompt: positions are (p, p, p) and the logits checksum must not
 * depend on whether --vision was loaded (the Makefile compares two runs) */
static void case_text_only(ds4_engine *e) {
    char err[256];
    ds4_tokens t = {0};
    ds4_tokenize_text(e, "<|im_start|>user\nName three prime numbers.<|im_end|>\n<|im_start|>assistant\n", &t);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, CTX)) fail("session create");
    if (ds4_session_sync(s, &t, err, sizeof(err))) fail(err);
    for (int p = 0; p < t.len; p++) {
        uint32_t got[3];
        if (!ds4_session_test_read_pos3(s, p, got)) fail("read pos3");
        if (got[0] != (uint32_t)p || got[1] != (uint32_t)p || got[2] != (uint32_t)p) fail("text row is not (p,p,p)");
    }
    const int vocab = ds4_engine_vocab_size(e);
    float *l = malloc((size_t)vocab * sizeof(float));
    if (!l || ds4_session_copy_logits(s, l, vocab) != vocab) fail("copy logits");
    uint32_t h = 2166136261u;
    for (int k = 0; k < vocab; k++) {
        uint32_t bits;
        memcpy(&bits, &l[k], sizeof(bits));
        h = (h ^ bits) * 16777619u;
    }
    printf("TEXT-CHECKSUM %08x argmax %d\n", h, ds4_session_argmax(s));
    free(l);
    ds4_session_free(s);
    ds4_tokens_free(&t);
}

int main(void) {
    const char *mmproj = getenv("DS4_TEST_MMPROJ");
    const char *mtp = getenv("DS4_TEST_MTP");
    const bool vision = mmproj && mmproj[0];
    ds4_engine *e = open_engine(vision, mtp && mtp[0] == '1');
    case_text_only(e);
    if (vision) {
        case_encode(e);
        case_positions(e);
    }
    printf("PASS: ornith vision (vision=%d mtp=%d)\n", vision ? 1 : 0, mtp && mtp[0] == '1');
    ds4_engine_close(e);
    return 0;
}
