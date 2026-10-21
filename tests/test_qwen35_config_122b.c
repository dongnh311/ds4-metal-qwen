/* The qwen35moe config dispatch (ds4.c): embedding_length 2048 selects the
 * Ornith-1.5-35B-A3B profile, 3072 selects the Ornith-1.5-122B profile, and
 * any other length dies.  A value that disagrees with the selected profile
 * dies naming the key, so a 122B embedding with a 35B block_count must be
 * refused.  Q6_K joined the dense gate: llama.cpp Q4_K_M files carry
 * Q6_K embed/output rows.  Header-only synthetic GGUFs: no weights, no
 * tokenizer, no GPU, no Metal. */
#define DS4_NO_GPU
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

static void w_u32(FILE *f, uint32_t n) { assert(fwrite(&n, 4, 1, f) == 1); }
static void w_u64(FILE *f, uint64_t n) { assert(fwrite(&n, 8, 1, f) == 1); }
static void w_f32(FILE *f, float v)    { assert(fwrite(&v, 4, 1, f) == 1); }
static void w_str(FILE *f, const char *s) {
    w_u64(f, (uint64_t)strlen(s));
    assert(fwrite(s, strlen(s), 1, f) == 1);
}

typedef struct {
    uint32_t n_embd, n_layer, nextn, ctx, head, head_kv, key_len, value_len, rot;
    float freq_base, eps;
    uint32_t expert, expert_used, ff, sff, conv, state, group, tsr, inner, interval;
} q35_dims;

static const q35_dims Q35_35B = {
    .n_embd = 2048, .n_layer = 41, .nextn = 1, .ctx = 262144,
    .head = 16, .head_kv = 2, .key_len = 256, .value_len = 256, .rot = 64,
    .freq_base = 1.0e7f, .eps = 1.0e-6f,
    .expert = 256, .expert_used = 8, .ff = 512, .sff = 512,
    .conv = 4, .state = 128, .group = 16, .tsr = 32, .inner = 4096, .interval = 4,
};

static const q35_dims Q35_122B = {
    .n_embd = 3072, .n_layer = 49, .nextn = 1, .ctx = 262144,
    .head = 32, .head_kv = 2, .key_len = 256, .value_len = 256, .rot = 64,
    .freq_base = 1.0e7f, .eps = 1.0e-6f,
    .expert = 256, .expert_used = 8, .ff = 1024, .sff = 1024,
    .conv = 4, .state = 128, .group = 16, .tsr = 64, .inner = 8192, .interval = 4,
};

/* GGUF v3 header with only the qwen35moe metadata the validator reads.
 * 0 overrides keep the profile's own value. */
static void fixture(const char *path, const q35_dims *d,
                    uint32_t embd_override, uint32_t block_override) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 1, 4, f) == 4);
    w_u32(f, 3);   /* version */
    w_u64(f, 0);   /* tensors */
    w_u64(f, 23);  /* metadata pairs */
    w_str(f, "general.architecture");  w_u32(f, 8); w_str(f, "qwen35moe");
    w_str(f, "general.quantization_version"); w_u32(f, 4); w_u32(f, 2);
    w_str(f, "qwen35moe.embedding_length");   w_u32(f, 4);
    w_u32(f, embd_override ? embd_override : d->n_embd);
    w_str(f, "qwen35moe.block_count");        w_u32(f, 4);
    w_u32(f, block_override ? block_override : d->n_layer);
    w_str(f, "qwen35moe.nextn_predict_layers"); w_u32(f, 4); w_u32(f, d->nextn);
    w_str(f, "qwen35moe.context_length");       w_u32(f, 4); w_u32(f, d->ctx);
    w_str(f, "qwen35moe.attention.head_count"); w_u32(f, 4); w_u32(f, d->head);
    w_str(f, "qwen35moe.attention.head_count_kv"); w_u32(f, 4); w_u32(f, d->head_kv);
    w_str(f, "qwen35moe.attention.key_length"); w_u32(f, 4); w_u32(f, d->key_len);
    w_str(f, "qwen35moe.attention.value_length"); w_u32(f, 4); w_u32(f, d->value_len);
    w_str(f, "qwen35moe.rope.dimension_count"); w_u32(f, 4); w_u32(f, d->rot);
    w_str(f, "qwen35moe.rope.freq_base");       w_u32(f, 6); w_f32(f, d->freq_base);
    w_str(f, "qwen35moe.attention.layer_norm_rms_epsilon"); w_u32(f, 6); w_f32(f, d->eps);
    w_str(f, "qwen35moe.expert_count");         w_u32(f, 4); w_u32(f, d->expert);
    w_str(f, "qwen35moe.expert_used_count");    w_u32(f, 4); w_u32(f, d->expert_used);
    w_str(f, "qwen35moe.expert_feed_forward_length"); w_u32(f, 4); w_u32(f, d->ff);
    w_str(f, "qwen35moe.expert_shared_feed_forward_length"); w_u32(f, 4); w_u32(f, d->sff);
    w_str(f, "qwen35moe.ssm.conv_kernel");      w_u32(f, 4); w_u32(f, d->conv);
    w_str(f, "qwen35moe.ssm.state_size");       w_u32(f, 4); w_u32(f, d->state);
    w_str(f, "qwen35moe.ssm.group_count");      w_u32(f, 4); w_u32(f, d->group);
    w_str(f, "qwen35moe.ssm.time_step_rank");   w_u32(f, 4); w_u32(f, d->tsr);
    w_str(f, "qwen35moe.ssm.inner_size");       w_u32(f, 4); w_u32(f, d->inner);
    w_str(f, "qwen35moe.full_attention_interval"); w_u32(f, 4); w_u32(f, d->interval);
    fclose(f);
}

static void validate_file(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    model_close(&m);
}

/* Forks a child that runs the dying validation; returns the raw wait status. */
static int death_status(const char *path) {
    pid_t pid = fork();
    assert(pid != -1);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        validate_file(path);
        _exit(0); /* reaching this means the validator did NOT refuse */
    }
    int st = 0;
    assert(waitpid(pid, &st, 0) == pid);
    return st;
}

static bool exited_1(int st) {
    return WIFEXITED(st) && WEXITSTATUS(st) == 1;
}

int main(void) {
    char dir[] = "/tmp/ds4_qwen35_config_XXXXXX";
    const char *tmp = mkdtemp(dir);
    assert(tmp);
    char path[256];
    snprintf(path, sizeof(path), "%s/35b.gguf", tmp);

    /* 35B profile stays selected and intact. */
    fixture(path, &Q35_35B, 0, 0);
    {
        ds4_model m;
        model_open(&m, path, false, false);
        config_validate_model(&m);
        CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35_MOE);
        CHECK(DS4_N_LAYER == 41);
        CHECK(DS4_N_EMBD == 2048);
        CHECK(DS4_N_HEAD == 16);
        CHECK(DS4_N_HEAD_KV == 2);
        CHECK(DS4_N_HEAD_DIM == 256);
        CHECK(DS4_N_EXPERT == 256);
        CHECK(DS4_N_EXPERT_USED == 8);
        CHECK(DS4_N_FF_EXP == 512);
        CHECK(DS4_N_LIN_K_HEAD == 16);
        CHECK(DS4_N_LIN_V_HEAD == 32);
        CHECK(DS4_N_LIN_HEAD_DIM == 128);
        CHECK(DS4_N_FULL_ATTN_INTERVAL == 4);
        CHECK(DS4_N_NEXTN_PREDICT == 1);
        CHECK(strcmp(DS4_MODEL_SHAPE_NAME, "Ornith-1.5-35B-A3B") == 0);
        model_close(&m);
    }

    /* 122B profile: the new shape. */
    snprintf(path, sizeof(path), "%s/122b.gguf", tmp);
    fixture(path, &Q35_122B, 0, 0);
    {
        ds4_model m;
        model_open(&m, path, false, false);
        config_validate_model(&m);
        CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_QWEN35_MOE);
        CHECK(DS4_N_LAYER == 49);
        CHECK(DS4_N_EMBD == 3072);
        CHECK(DS4_N_VOCAB == 248320);
        CHECK(DS4_N_HEAD == 32);
        CHECK(DS4_N_HEAD_KV == 2);
        CHECK(DS4_N_HEAD_DIM == 256);
        CHECK(DS4_N_VALUE_DIM == 256);
        CHECK(DS4_N_ROT == 64);
        CHECK(DS4_N_EXPERT == 256);
        CHECK(DS4_N_EXPERT_USED == 8);
        CHECK(DS4_N_EXPERT_SHARED == 1);
        CHECK(DS4_N_FF_EXP == 1024);
        CHECK(DS4_N_LIN_K_HEAD == 16);
        CHECK(DS4_N_LIN_V_HEAD == 64);
        CHECK(DS4_N_LIN_HEAD_DIM == 128);
        CHECK(DS4_N_LIN_CONV == 4);
        CHECK(DS4_N_FULL_ATTN_INTERVAL == 4);
        CHECK(DS4_N_NEXTN_PREDICT == 1);
        CHECK(DS4_ROPE_ORIG_CTX == 262144);
        CHECK(abs(DS4_RMS_EPS - 1.0e-6f) <= 1.1e-11f);
        CHECK(strcmp(DS4_MODEL_SHAPE_NAME, "Ornith-1.5-122B") == 0);
        model_close(&m);
    }

    /* 122B embedding with a 35B layer count dies naming the key. */
    snprintf(path, sizeof(path), "%s/mixed.gguf", tmp);
    fixture(path, &Q35_122B, 0, 41);
    CHECK(exited_1(death_status(path)));

    /* An embedding length that matches no profile dies. */
    snprintf(path, sizeof(path), "%s/other.gguf", tmp);
    fixture(path, &Q35_122B, 2560, 0);
    CHECK(exited_1(death_status(path)));

    /* Q6_K (and the IQ-family GSQ ids) already satisfy the dense gate; keep
     * the contract pinned so a future gate tightening cannot strand Q6_K
     * embed/output rows from llama.cpp Q4_K_M files. */
    CHECK(tensor_type_is_qwen4_dense(DS4_TENSOR_Q6_K));

    printf("qwen35 config dispatch: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
