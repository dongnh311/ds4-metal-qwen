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

int main(void) {
    const char *mmproj = getenv("DS4_TEST_MMPROJ");
    const char *mtp = getenv("DS4_TEST_MTP");
    const bool vision = mmproj && mmproj[0];
    ds4_engine *e = open_engine(vision, mtp && mtp[0] == '1');
    if (vision) case_encode(e);
    printf("PASS: ornith vision (vision=%d mtp=%d)\n", vision ? 1 : 0, mtp && mtp[0] == '1');
    ds4_engine_close(e);
    return 0;
}
