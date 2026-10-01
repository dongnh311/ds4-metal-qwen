/* Model-free GLM checks: the standard Q2 layout (IQ2_XXS gate/up, Q2_K down,
 * eight routed experts) streams through the selected-expert cache, and
 * live-prefix rewinds are only promised where the session keeps its state. */
#include "../ds4.c"

static int failures;
#define CHECK(cond) do { \
        if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
    } while (0)

static ds4_tensor gate_t, up_t, down_t;
static ds4_layer_weights layer;

static void set_layout(uint32_t gate, uint32_t down) {
    memset(&layer, 0, sizeof(layer));
    gate_t.type = gate;
    up_t.type = gate;
    down_t.type = down;
    layer.ffn_gate_exps = &gate_t;
    layer.ffn_up_exps = &up_t;
    layer.ffn_down_exps = &down_t;
}

static void check_glm_iq2_q2(const ds4_shape *shape) {
    g_ds4_shape = *shape;
    CHECK(DS4_N_EXPERT_USED == 8);
    set_layout(DS4_TENSOR_IQ2_XXS, DS4_TENSOR_Q2_K);
    CHECK(glm_stream_selected_expert_cache_supported(&layer, 10));
    /* Dense leading layers never stream. */
    CHECK(!glm_stream_selected_expert_cache_supported(&layer, 0));
    /* The per-expert address path can still be switched off. */
    setenv("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE", "1", 1);
    setenv("DS4_ROCM_DISABLE_IQ2_STREAM_ADDR_TABLE", "1", 1);
#ifdef __APPLE__
    CHECK(!glm_stream_selected_expert_cache_supported(&layer, 10));
#endif
    unsetenv("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE");
    unsetenv("DS4_ROCM_DISABLE_IQ2_STREAM_ADDR_TABLE");
    /* IQ2/IQ2 stays supported; Q4 down on IQ2 gate is not a streamed layout. */
    set_layout(DS4_TENSOR_IQ2_XXS, DS4_TENSOR_IQ2_XXS);
    CHECK(glm_stream_selected_expert_cache_supported(&layer, 10));
    set_layout(DS4_TENSOR_IQ2_XXS, DS4_TENSOR_Q4_K);
    CHECK(!glm_stream_selected_expert_cache_supported(&layer, 10));
#ifdef __APPLE__
    /* Quality mode and two-rank TP have no IQ2 streamed-expert kernels, so
     * those runs keep IQ2 layers mapped for the resident fused path. */
    static const uint32_t downs[] = {DS4_TENSOR_Q2_K, DS4_TENSOR_IQ2_XXS};
    for (size_t i = 0; i < sizeof(downs) / sizeof(*downs); i++) {
        set_layout(DS4_TENSOR_IQ2_XXS, downs[i]);
        glm_stream_configure_iq2_cache(true, false);
        CHECK(!glm_stream_selected_expert_cache_supported(&layer, 10));
        glm_stream_configure_iq2_cache(false, true);
        CHECK(!glm_stream_selected_expert_cache_supported(&layer, 10));
        glm_stream_configure_iq2_cache(false, false);
        CHECK(glm_stream_selected_expert_cache_supported(&layer, 10));
    }
#endif
}

static ds4_engine engine;
static ds4_session session;

/* GLM-5.2's DSA cache is positional; GLM-5.3's recurrent KDA state only rolls
 * back inside the MTP two-token window. */
static void check_glm_session_can_rewind(void) {
    g_ds4_shape = DS4_SHAPE_GLM53;
    memset(&session, 0, sizeof(session));
    session.engine = &engine;
    session.checkpoint_valid = true;
    session.checkpoint.len = 174;
    session.glm_graph.glm53 = true;
    CHECK(!ds4_session_glm_can_rewind(&session, 45));
    session.glm_mtp_rollback_valid = true;
    session.glm_mtp_rollback_pos = 172;
    CHECK(ds4_session_glm_can_rewind(&session, 172));
    CHECK(ds4_session_glm_can_rewind(&session, 173));
    CHECK(!ds4_session_glm_can_rewind(&session, 171));
    CHECK(!ds4_session_glm_can_rewind(&session, 174));
    session.checkpoint.len = 175;
    CHECK(!ds4_session_glm_can_rewind(&session, 173));
    session.checkpoint.len = 174;
    session.checkpoint_valid = false;
    CHECK(!ds4_session_glm_can_rewind(&session, 173));
    session.checkpoint_valid = true;
    session.glm_graph.glm53 = false;
    CHECK(ds4_session_glm_can_rewind(&session, 45));
    CHECK(!ds4_session_glm_can_rewind(&session, 174));
    CHECK(!ds4_session_glm_can_rewind(NULL, 45));
}

int main(void) {
    /* Start from the default routed kernels whatever the caller exported. */
    static const char *const disables[] = {
        "DS4_METAL_MOE_WRITE_CLAMPED_ACT", "DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
        "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
        "DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
        "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE",
        "DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
        "DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE", "DS4_ROCM_DISABLE_IQ2_STREAM_ADDR_TABLE",
        "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
        "DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
    };
    for (size_t i = 0; i < sizeof(disables) / sizeof(*disables); i++) unsetenv(disables[i]);
    check_glm_iq2_q2(&DS4_SHAPE_GLM53);
    check_glm_iq2_q2(&DS4_SHAPE_GLM52);
    check_glm_session_can_rewind();
    if (failures) {
        fprintf(stderr, "test_glm53_stream_layout: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_glm53_stream_layout: PASS\n");
    return 0;
}
