/* Real-model check of the M7 batched verify.
 * Usage: test_qwen35_verify_batch MODEL
 *            [--default | --knob-off | --perturbed-kernel | --broken-kernel]
 * Sets DS4_QWEN35_VERIFY_BATCH=1 before the engine opens (the knob is read
 * once); --default leaves it unset, --knob-off sets it to 0 and then expects
 * no two-row matvec in any verify.  --perturbed-kernel and --broken-kernel
 * leave the knob at its default and make the engine compile a copy of
 * metal/qwen35.metal (DS4_METAL_QWEN35_SOURCE) whose two-row kernel adds
 * 1e-3 to every partial sum, or is missing: the startup self-check must turn
 * the batched verify off, so no verify dispatches it and every cycle stays
 * bit-identical to plain decoding.  150 greedy speculative cycles from a 1500-token prompt: after
 * every cycle the logits are bit-identical to a plain session fed the
 * committed tokens, and every verify issues exactly VERIFY_ROWS two-row
 * Q8_0 matvecs (ds4_gpu_qwen35_q8_rows_dispatches): attn_q and attn_output
 * in each of Ornith's 10 attention layers, lin_qkv, lin_gate and lin_out in
 * each of its 30 GDN layers, plus the lm head. */
#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { CTX = 4160, CHUNK = 512, CYCLES = 150, VERIFY_ROWS = 10 * 2 + 30 * 3 + 1 };

static int g_vocab;
static float *g_a, *g_b;
static char g_metal_copy[512];

/* Writes metal/qwen35.metal with one substitution in the two-row kernel to a
 * temporary file and points DS4_METAL_QWEN35_SOURCE at it. */
static void use_metal_copy(bool broken) {
    FILE *fp = fopen("metal/qwen35.metal", "rb");
    assert(fp);
    assert(fseek(fp, 0, SEEK_END) == 0);
    const long n = ftell(fp);
    assert(n > 0 && fseek(fp, 0, SEEK_SET) == 0);
    char *src = calloc((size_t)n + 1, 1);
    assert(src && fread(src, 1, (size_t)n, fp) == (size_t)n);
    fclose(fp);
    const char *old = broken ? "[[host_name(\"kernel_qwen35_mv_q8_0_rows2\")]]" : "sumf[r][row] += sumq*d;";
    const char *new = broken ? "[[host_name(\"kernel_qwen35_mv_q8_0_rows2_absent\")]]"
                             : "sumf[r][row] += sumq*d + 1e-3f;";
    char *at = strstr(src, old);
    assert(at);
    const char *tmp = getenv("TMPDIR");
    snprintf(g_metal_copy, sizeof(g_metal_copy), "%s/ds4-m7-qwen35-%d.metal", tmp && tmp[0] ? tmp : "/tmp",
             (int)getpid());
    FILE *out = fopen(g_metal_copy, "wb");
    assert(out);
    fwrite(src, 1, (size_t)(at - src), out);
    fputs(new, out);
    fputs(at + strlen(old), out);
    assert(fclose(out) == 0);
    free(src);
    setenv("DS4_METAL_QWEN35_SOURCE", g_metal_copy, 1);
}

static void sync_len(ds4_session *s, const ds4_tokens *tokens, int n) {
    ds4_tokens prefix = *tokens;
    prefix.len = n;
    char err[256] = {0};
    const int rc = ds4_session_sync(s, &prefix, err, sizeof(err));
    if (rc) fprintf(stderr, "sync %d: %s\n", n, err);
    assert(rc == 0 && ds4_session_pos(s) == n);
}

static void same_logits(ds4_session *spec, ds4_session *plain) {
    assert(ds4_session_copy_logits(spec, g_a, g_vocab) == g_vocab);
    assert(ds4_session_copy_logits(plain, g_b, g_vocab) == g_vocab);
    assert(memcmp(g_a, g_b, (size_t)g_vocab * sizeof(float)) == 0);
    assert(ds4_session_pos(spec) == ds4_session_pos(plain));
}

int main(int argc, char **argv) {
    const char *mode = argc == 3 ? argv[2] : "";
    const bool knob_off = strcmp(mode, "--knob-off") == 0;
    const bool perturbed = strcmp(mode, "--perturbed-kernel") == 0;
    const bool broken = strcmp(mode, "--broken-kernel") == 0;
    if (argc < 2 || argc > 3 ||
        (argc == 3 && !knob_off && !perturbed && !broken && strcmp(mode, "--default") != 0)) {
        fprintf(stderr, "usage: %s MODEL [--default | --knob-off | --perturbed-kernel | --broken-kernel]\n",
                argv[0]);
        return 1;
    }
    if (knob_off) setenv("DS4_QWEN35_VERIFY_BATCH", "0", 1);
    else if (argc == 3) unsetenv("DS4_QWEN35_VERIFY_BATCH");
    else setenv("DS4_QWEN35_VERIFY_BATCH", "1", 1);
    if (perturbed || broken) use_metal_copy(broken);
    const bool batch_off = knob_off || perturbed || broken;
    const uint64_t expect = batch_off ? 0u : (uint64_t)VERIFY_ROWS;
    unsetenv("DS4_QWEN35_SPEC_FORCE_ACCEPT");
    ds4_engine_options opt = {.model_path = argv[1], .context_size = CTX, .prefill_chunk = CHUNK,
                              .backend = DS4_BACKEND_METAL, .glm_mtp = true};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_qwen35moe(engine));
    const int eos = ds4_token_eos(engine);
    FILE *fp = fopen("speed-bench/promessi_sposi.txt", "rb");
    assert(fp);
    char *text = calloc(40001, 1);
    assert(fread(text, 1, 40000, fp) > 0);
    fclose(fp);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    assert(tokens.len >= 1500);
    g_vocab = ds4_engine_vocab_size(engine);
    g_a = malloc((size_t)g_vocab * sizeof(float));
    g_b = malloc((size_t)g_vocab * sizeof(float));
    ds4_session *spec = NULL, *plain = NULL;
    assert(ds4_session_create(&spec, engine, CTX) == 0);
    assert(ds4_session_create(&plain, engine, CTX) == 0);
    char err[256] = {0};
    sync_len(spec, &tokens, 1500);
    sync_len(plain, &tokens, 1500);
    same_logits(spec, plain);
    int verifies = 0, accepts = 0;
    for (int c = 0; c < CYCLES; c++) {
        const uint64_t before = ds4_gpu_qwen35_q8_rows_dispatches();
        const int first = ds4_session_argmax(spec);
        int acc[17];
        const int n = ds4_session_eval_speculative_argmax(spec, first, 16, eos, acc, 17, err, sizeof(err));
        if (n < 0) fprintf(stderr, "cycle %d: %s\n", c, err);
        assert(n >= 1 && n <= 2 && acc[0] == first);
        const uint64_t d = ds4_gpu_qwen35_q8_rows_dispatches() - before;
        if (d != 0u && d != expect) {
            fprintf(stderr, "cycle %d: %llu two-row matvecs, expected 0 or %llu\n", c, (unsigned long long)d,
                    (unsigned long long)expect);
            return 1;
        }
        verifies += expect && d == expect;
        accepts += n == 2;
        for (int i = 0; i < n; i++) {
            assert(ds4_session_argmax(plain) == acc[i]);
            char perr[256] = {0};
            if (ds4_session_eval(plain, acc[i], perr, sizeof(perr)) != 0) {
                fprintf(stderr, "plain eval: %s\n", perr);
                return 1;
            }
        }
        same_logits(spec, plain);
    }
    printf("  %d cycles, %d verifies with %llu two-row matvecs each, %d accepted, bit-identical to plain\n",
           CYCLES, verifies, (unsigned long long)expect, accepts);
    assert((batch_off ? verifies == 0 : verifies >= CYCLES - 5) && accepts >= 10);
    ds4_session_free(plain);
    ds4_session_free(spec);
    ds4_tokens_free(&tokens);
    free(g_a);
    free(g_b);
    ds4_engine_close(engine);
    if (g_metal_copy[0]) unlink(g_metal_copy);
    printf("qwen35 verify batch: ok\n");
    return 0;
}
