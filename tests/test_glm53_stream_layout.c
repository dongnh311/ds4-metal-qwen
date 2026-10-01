/* Model-free checks that GLM's standard Q2 layout (IQ2_XXS gate/up, Q2_K
 * down, eight routed experts) is served by the selected-expert streaming
 * cache instead of the per-layer mapped fallback. */
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

static ds4_glm_gpu_graph async_graph;

/* Metal keeps the synchronous selected-expert load by default: the async
 * worker measured 4.8% slower GLM decode on an M5 Pro. The opt-in switch
 * enables it; the disable switches win. */
static void check_glm_streaming_async_load_default(void) {
    memset(&async_graph, 0, sizeof(async_graph));
    CHECK(!glm_graph_use_streaming_selected_async_load(&async_graph));
    async_graph.ssd_streaming = true;
#if defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    CHECK(!glm_graph_use_streaming_selected_async_load(&async_graph));
    setenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD", "1", 1);
#endif
    CHECK(glm_graph_use_streaming_selected_async_load(&async_graph));
    setenv("DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD", "1", 1);
    setenv("DS4_ROCM_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD", "1", 1);
    CHECK(!glm_graph_use_streaming_selected_async_load(&async_graph));
    unsetenv("DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD");
    unsetenv("DS4_ROCM_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD");
    unsetenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD");
    CHECK(!glm_graph_use_streaming_selected_async_load(NULL));
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
        "DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD",
        "DS4_ROCM_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD",
        "DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD",
        "DS4_ROCM_DISABLE_STREAMING_SELECTED_ASYNC_LOAD",
    };
    for (size_t i = 0; i < sizeof(disables) / sizeof(*disables); i++) unsetenv(disables[i]);
    check_glm_iq2_q2(&DS4_SHAPE_GLM53);
    check_glm_iq2_q2(&DS4_SHAPE_GLM52);
    check_glm_streaming_async_load_default();
    if (failures) {
        fprintf(stderr, "test_glm53_stream_layout: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_glm53_stream_layout: PASS\n");
    return 0;
}
