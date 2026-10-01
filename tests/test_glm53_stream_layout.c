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
    /* GLM-5.3 routed shapes: 288 experts, hidden 4096, expert width 2048. */
    gate_t.dim[0] = up_t.dim[0] = 4096;
    gate_t.dim[1] = up_t.dim[1] = 2048;
    down_t.dim[0] = 2048;
    down_t.dim[1] = 4096;
    gate_t.dim[2] = up_t.dim[2] = down_t.dim[2] = 288;
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

static ds4_weights weights;
static ds4_glm_gpu_graph graph;

static bool full_layer(uint32_t pos0, uint32_t n_tokens) {
    return glm_graph_indexed_prefill_full_layer(&graph, &weights, pos0, n_tokens, true);
}

/* Indexed prefill decides per chunk whether routed layers are read whole
 * through the pread prepare or paged in through the mapped views. */
static void check_glm53_prefill_full_layer(void) {
    g_ds4_shape = DS4_SHAPE_GLM53;
    memset(&graph, 0, sizeof(graph));
    graph.glm53 = true;
    graph.ssd_streaming = true;
    graph.tp_world = 1;
    graph.layer_count = glm_graph_normal_layer_count();
    memset(&weights, 0, sizeof(weights));
    set_layout(DS4_TENSOR_IQ2_XXS, DS4_TENSOR_Q2_K);
    weights.layer[DS4_N_LEADING_DENSE] = layer;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    /* Without an expert cache IQ2/Q2 has no selected-expert batch prefill, so
     * every chunk would page in whole layers anyway: read them with pread. */
    CHECK(full_layer(0, 2048));
    CHECK(full_layer(2048, 2048));
    CHECK(full_layer(0, 28));
    CHECK(full_layer(4096, 40));
    /* With a cache that holds a whole layer, chunks under 256 tokens load only
     * their selected experts through it; larger and single-token chunks still
     * read whole layers. */
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_streaming_expert_cache_budget(4546);
    const ds4_layer_weights *routed = &weights.layer[DS4_N_LEADING_DENSE];
    CHECK(glm_graph_stream_prefill_expert_addr_supported(&graph, &weights, routed,
                                                         DS4_N_LEADING_DENSE, 28));
    CHECK(!full_layer(0, 28));
    CHECK(!full_layer(4096, 40));
    CHECK(!full_layer(0, 255));
    CHECK(full_layer(0, 256));
    CHECK(full_layer(2048, 2048));
    /* A one-token tail chunk (prompt length 2048k+1, or an append) loads its
     * eight experts instead of reading every routed layer. */
    CHECK(glm_graph_stream_prefill_expert_addr_supported(&graph, &weights, routed,
                                                         DS4_N_LEADING_DENSE, 1));
    CHECK(!full_layer(4096, 1));
    ds4_gpu_set_streaming_expert_cache_budget(100);
    CHECK(full_layer(0, 28));
    ds4_gpu_set_streaming_expert_cache_budget(4546);
    /* Metal skips the cached batch whenever full-layer prefill is forced, so
     * the forced chunk must read whole layers on the ds4.c side too. */
    setenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER", "1", 1);
    CHECK(!glm_graph_stream_prefill_expert_addr_supported(&graph, &weights, routed,
                                                          DS4_N_LEADING_DENSE, 28));
    CHECK(full_layer(0, 28));
    unsetenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER");
    /* The selected-expert chunk limit is tunable below the Metal bound. */
    setenv("DS4_METAL_GLM_STREAMING_PREFILL_SELECTED_MAX_TOKENS", "64", 1);
    CHECK(!full_layer(0, 63));
    CHECK(full_layer(0, 64));
    setenv("DS4_METAL_GLM_STREAMING_PREFILL_SELECTED_MAX_TOKENS", "4096", 1);
    CHECK(!full_layer(0, 255));
    CHECK(full_layer(0, 256));
    unsetenv("DS4_METAL_GLM_STREAMING_PREFILL_SELECTED_MAX_TOKENS");
    /* Only the final chunk seeds the decode expert cache from a full layer. */
    CHECK(glm_graph_full_layer_prefill_seeds_cache(true, true));
    CHECK(!glm_graph_full_layer_prefill_seeds_cache(true, false));
    CHECK(!glm_graph_full_layer_prefill_seeds_cache(false, true));
    CHECK(!glm_graph_indexed_prefill_full_layer(&graph, &weights, 2048, 2048, false));
    graph.quality = true;
    CHECK(!full_layer(2048, 2048));
    graph.quality = false;
    graph.tp_world = 2;
    CHECK(!full_layer(2048, 2048));
    graph.tp_world = 1;
    setenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER", "1", 1);
    CHECK(!full_layer(2048, 2048));
    CHECK(!full_layer(0, 28));
    unsetenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER");
    /* Typed uniform Q2_K layers keep selected-expert reuse after the first
     * chunk and stay on that path for short chunks. */
    set_layout(DS4_TENSOR_Q2_K, DS4_TENSOR_Q2_K);
    weights.layer[DS4_N_LEADING_DENSE] = layer;
    CHECK(full_layer(0, 2048));
    CHECK(!full_layer(2048, 2048));
    CHECK(!full_layer(0, 28));
#endif
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
    check_glm53_prefill_full_layer();
    if (failures) {
        fprintf(stderr, "test_glm53_stream_layout: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_glm53_stream_layout: PASS\n");
    return 0;
}
