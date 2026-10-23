/* ========================================================================== *
 * ds4_dflash.inc — DFlash2 external block-diffusion drafter (W-D3).          *
 *                                                                            *
 * Included by ds4.c immediately before ds4_qwen35moe.inc: the Ornith graph   *
 * hooks (qwen35_graph_forward_tokens tap capture, qwen35_graph_reset) reach  *
 * this code, while this code reaches ds4.c primitives (qwen4_gemv,           *
 * qwen4_ref_row, f16_to_f32, tensor_data, the ds4_gpu_tensor arena).         *
 *                                                                            *
 * Reference semantics locked to llama.cpp src/models/dflash.cpp +            *
 * common/speculative.cpp (draft-dflash), the port the released incoai        *
 * drafter targets:                                                           *
 *   - block = n_draft+1 rows at positions pos0..pos0+n_draft; row 0 re-stages
 *     the anchor token's embedding at pos0 (the position of the last         *
 *     committed token), rows 1..n_draft are mask rows; attention over the    *
 *     ctx + the whole block with no causal cap; +-2048 sliding band on the   *
 *     injected ctx rows (dflash.attention.sliding_window).                    *
 *   - ctx K/V: per committed trunk row, the per-layer hidden taps are fused  *
 *     fc -> RMS(enc.output_norm) -> per-layer k/v projections -> per-head    *
 *     k_norm + RoPE (V raw), cached post-rope at the row's absolute position.*
 *   - layer = RMS(attn_norm) -> block-internal kernel-2 conv (per-tap        *
 *     weights = conv_proj coefficient + conv_base, groups of 16 channels,    *
 *     zero left padding inside the block) -> q/k/v -> per-head q_norm/k_norm *
 *     -> RoPE -> attention -> attn_output -> side-1 conv -> residual; the    *
 *     ffn half mirrors with ffn_norm / ffn_conv / silu(gate)*up / down.      *
 *   - final RMS(output_norm) -> the TRUNK's output head (Q6_K in the released
 *     IQ2 tier) -> top-16 candidates per row; the greedy lattice walk runs   *
 *     per row over candidates c of that row:                                 *
 *       score[c] = unary[c] + sum_r succ[id_c][r] * pred[pred_id][r] *       *
 *                  gate_row[r],                                             *
 *     pred_id = the anchor token for row 1, the carried candidate of row     *
 *     i-1 for rows 2..n_draft; draft_i = cand_i[argmax c] — the same read    *
 *     llama.cpp's draft-dflash2 performs on its packed lattice rows.         *
 *                                                                            *
 * Stage-A (this build): the drafter runs in PROBE mode only                  *
 * (DS4_DFLASH_PROBE=1): plain decode is untouched; after every commit the    *
 * ctx cache gets the new rows' features and one draft block is scored        *
 * against the continuing stream. All matmuls (fc, projections, head, gate)   *
 * run on the GPU through qwen4_gemv; norms, convs, RoPE, attention and the   *
 * walk run on the host in fp32 (Accelerate sgemm for attention). The block   *
 * forward doubles as the numerical oracle for the W-D4 Metal msa-attention   *
 * kernel.                                                                    *
 *                                                                            *
 * Tap plumbing: the trunk's per-layer residual (g->R) must be captured       *
 * inside the forward's command batch (it is overwritten by the next          *
 * layer), so qwen35_graph_forward_tokens blits each tap layer's rows into    *
 * this module's GPU ring ([tap plane][slot][E] f32, slot = pos % ring_cap)   *
 * via ds4_gpu_tensor_copy; the injection later reads the committed ring.     *
 * ========================================================================== */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#define DS4_DFLASH_WINDOW_DEFAULT 2048u
#define DS4_DFLASH_INJ_CAP        256u
#define DS4_DFL_TSTAT_SLOTS       24u
#define DS4_DFLASH_MAX_DRAFT      15u    /* trained block_size - 1          */

typedef struct ds4_dflash_gpu {
    /* configuration (copied from the bound weights at attach)              */
    uint32_t n_layer, E, ff, heads, kv_heads, hd, block_size, top_k, rank;
    uint32_t n_groups, group, kernel;
    uint32_t n_taps, window, ring_cap;
    uint32_t tap_layers[DS4_DFLASH_MAX_LAYERS];
    float    rope_base, eps;
    int32_t  mask_id;
    uint32_t cap_rows;                      /* = block_size                    */
    bool     probe;                         /* DS4_DFLASH_PROBE                */
    bool     wired;
    bool     tstat;                         /* DS4_DFLASH_TSTAT=1              */
    double   tstat_ms[DS4_DFL_TSTAT_SLOTS];          /* cumulative wall ms      */
    uint64_t tstat_n[DS4_DFL_TSTAT_SLOTS];             /* region occurrences     */

    /* model handles: dm = drafter support file, tm/tw = trunk              */
    ds4_model        *dm;
    ds4_model        *tm;
    const ds4_weights *tw;
    ds4_dflash_weights dw;                  /* by-value bind copy              */
    float   *mask_rowf;                     /* trained mask embedding, E       */
    int      dbg;                           /* DS4_DFLASH_DEBUG: per-call dump */
    int      gpu_fc;                        /* DS4_DFLASH_GPU_FC=1: try GPU fc */
    uint32_t inj_chunk;                     /* rows per inject gemv batch      */
    float   *fc_w;                          /* host fp32 fc [E][taps*E]        */
    int32_t  dbg_rc[4];                     /* dbg: prev cycle row0 top1+drafts */

    /* GPU arena                                                            */
    ds4_gpu_tensor *gxn;                    /* gemv input, cap_rows*E          */
    ds4_gpu_tensor *gq, *gattn;             /* cap_rows*heads*hd               */
    ds4_gpu_tensor *gk, *gv;                /* cap_rows*kv_heads*hd            */
    ds4_gpu_tensor *gwo;                    /* cap_rows*E (wo / down out)      */
    ds4_gpu_tensor *gconv, *gconv2;         /* cap_rows*2k*n_groups            */
    ds4_gpu_tensor *gffn1, *gffn2;          /* cap_rows*ff (gate / up out)     */
    ds4_gpu_tensor *gmid;                   /* cap_rows*ff (silu*up in)        */
    ds4_gpu_tensor *ggate;                  /* cap_rows*rank                   */
    ds4_gpu_tensor *glogits;                /* cap_rows*trunk vocab            */
    ds4_gpu_tensor *gtap;                   /* inj_cap rows of n_taps*E        */
    ds4_gpu_tensor *ginj;                   /* inj_cap*E (k/v gemv input)      */
    ds4_gpu_tensor *gfc;                    /* inj_cap*E (fc out)              */
    ds4_gpu_tensor *gk_inj, *gv_inj;        /* inj_cap*kv_heads*hd             */
    ds4_gpu_tensor *gtap_ring;              /* n_taps planes * ring_cap*E      */

    /* drafter ctx KV cache [layer][head][slot][hd] fp32; slot = pos%ring_cap */
    float   *ck, *cv;
    uint32_t ring_base;                     /* oldest valid ring position      */
    uint32_t ring_valid;                    /* tap rows held                   */
    uint32_t inj_next;                      /* first position not yet injected */

    /* host scratch                                                         */
    float *h;                               /* cap_rows*E residual stream      */
    float *xn;                              /* cap_rows*E normed rows          */
    float *cr1, *cr2;                       /* cap_rows*E conv results         */
    float *q;                               /* cap_rows*heads*hd               */
    float *kb, *vb;                         /* cap_rows*kv_heads*hd            */
    float *o;                               /* cap_rows*heads*hd attention out */
    float *ho;                              /* cap_rows*E gemv readback        */
    float *dyn, *dyn2;                      /* cap_rows*2k*n_groups            */
    float *gw, *gu, *gm;                    /* cap_rows*ff gate/up/silu*up     */
    float *stage;                           /* inj_cap*n_taps*E fused input    */
    float *enc, *ik, *iv;                   /* inj rows: fc out, k, v          */
    float *gate;                            /* cap_rows*rank                   */
    float *logits;                          /* cap_rows*trunk vocab            */
    float *score, *score_f;                 /* cap_rows*ring_cap, cap_rows^2   */
    int32_t *cand;                          /* (cap_rows)*top_k ids            */
    float  *uv;                             /* (cap_rows)*top_k logits         */
    float  *succ, *pv;                      /* walk: top_k*rank, rank          */

    /* probe receipts                                                       */
    uint64_t probe_cycles, probe_tokens;
    uint32_t probe_hist[2 * DS4_DFLASH_MAX_DRAFT + 2];
    uint32_t probe_seq;
    int32_t  pd_drafts[DS4_DFLASH_MAX_DRAFT];
    uint32_t pd_pos0, pd_len, pd_matched;
    bool     pd_valid;
    /* W-D4 verify receipts (cycles committing matched+1 tokens each)       */
    uint64_t verify_cycles, verify_tokens;
    uint64_t verify_swaps, verify_fallbacks;
    uint32_t verify_hist[DS4_DFLASH_MAX_DRAFT + 2];
} ds4_dflash_gpu;

/* ------------------------------------------------------------------ */
/* W-D4b TSTAT: fine wall decomposition of inject+draft (default off)   */
/*                                                                      */
/* The stage picture (inject+draft ≈ 25-40 ms/cycle whole) hides where   */
/* the time actually sits: sync round trips per gemv batch, the host    */
/* fc sgemm, cblas attention, or the host conv/rope work.  TSTAT scopes */
/* every inject/draft sub-region and prints the cumulative mean per     */
/* occurrence alongside the verify receipts.                            */
/* ------------------------------------------------------------------ */
static const char *ds4_dfl_tstat_name[DS4_DFL_TSTAT_SLOTS] = {
    "inj_taps", "inj_w",  "inj_fc",  "inj_rms", "inj_kv",  "inj_kvhost",
    "d_rmsw",   "d_cproj", "d_conv0", "d_qkv",  "d_rope",  "d_attn",
    "d_wo",     "d_hup",   "d_fproj", "d_fconv", "d_gate",  "d_up",
    "d_silu",   "d_down",  "d_fup2",  "d_final", "d_walk",  "spare22"
};

#define DS4_DFL_TEND(df, slot_, t0_) do { \
    if ((df)->tstat && (t0_) > 0.0) { \
        (df)->tstat_ms[slot_] += (now_sec() - (t0_)) * 1000.0; \
        (df)->tstat_n[slot_]++; \
    } \
} while (0)

/* always releases the inject k/v plane views (NULL-safe elements)      */
#define DFLASH_KV_VIEW_FREE(kvw_, vvw_) do { \
    for (int _il = 0; _il < (int)DS4_DFLASH_MAX_LAYERS; _il++) { \
        ds4_gpu_tensor_free((kvw_)[_il]); ds4_gpu_tensor_free((vvw_)[_il]); \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* host math helpers                                                    */
/* ------------------------------------------------------------------ */

/* llama.cpp build_norm(RMS): x * w * (1/sqrt(mean(x^2)+eps))            */
static void ds4_dfl_rms(const float *x, const float *w, uint32_t n, float eps, float *out) {
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) acc += (double)x[i] * (double)x[i];
    const float s = 1.0f / (float)sqrt(acc / (double)n + (double)eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

/* NeoX RoPE: pairs (i, i+hd/2), inv_freq_i = base^(-2i/hd), i < hd/2.   */
static void ds4_dfl_rope(float *x, uint32_t hd, double pos, float base) {
    const uint32_t half = hd / 2u;
    for (uint32_t i = 0; i < half; i++) {
        const float theta = (float)(pos * pow(base, -(double)(2u * i) / (double)hd));
        const float c = cosf(theta), s = sinf(theta);
        const float a = x[i], b = x[i + half];
        x[i]        = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

/* per-row per-head RMS with a hd-sized weight; rows [n][heads*hd]       */
static void ds4_dfl_head_rms(float *rows, uint32_t n, uint32_t heads, uint32_t hd,
                             const float *w, float eps) {
    for (uint32_t t = 0; t < n; t++)
        for (uint32_t h = 0; h < heads; h++) {
            float *r = rows + ((size_t)t * heads + h) * hd;
            ds4_dfl_rms(r, w, hd, eps, r);
        }
}

static const float *ds4_dfl_f32(const ds4_model *m, const ds4_tensor *t) {
    if (!t || t->type != DS4_TENSOR_F32) {
        if (t) fprintf(stderr, "ds4: dflash: tensor '%.*s' is not F32 (%u)\n",
                       (int)t->name.len, t->name.ptr, t->type);
        return NULL;
    }
    return (const float *)tensor_data(m, t);
}

/* dequant one row of any drafter codebook into fp32                      */
static void ds4_dfl_row(const ds4_model *m, const ds4_tensor *t, uint64_t row, float *out) {
    qwen4_ref_row(m, t, row, out);
}

/* raw conv (llama.cpp build_dflash2_conv, one block, side split):        *
 * out[t][e] = sum_k (dyn[t][side,k,g] + base[k][side][e]) *              *
 *             in[t - k (block rows only, zero left pad)][e]              *
 * dyn layout (conv_proj rows, ggml ne [g][k][side] per token) =          *
 *              [t][side*(kernel*ng) + k*ng + g]                          *
 * base layout = [side][kernel][E] as (e + k*E + side*E*kernel)            */
static void ds4_dfl_conv(uint32_t T, uint32_t bs, uint32_t E, uint32_t group, uint32_t kernel,
                         const float *dyn, int side, const float *base,
                         const float *in, float *out) {
    const uint32_t ng = E / group;
    for (uint32_t t = 0; t < T; t++) {
        float *o = out + (size_t)t * E;
        const int32_t b = (int32_t)(t % bs);
        const float *drow = dyn + (size_t)t * (2u * kernel * ng) + (size_t)side * (kernel * ng);
        for (uint32_t e = 0; e < E; e++) {
            float acc = 0.0f;
            const uint32_t g = e / group;
            for (uint32_t k = 0; k < kernel; k++) {
                if (b - (int32_t)k < 0) break;   /* taps read left padding */
                const float inrow  = in[(size_t)(t - (uint32_t)k) * E + e];
                const float coeff  = drow[(size_t)k * ng + g];
                const float basew  = base[(size_t)e + (size_t)k * E + (size_t)side * ((size_t)E * kernel)];
                acc += (coeff + basew) * inrow;
            }
            o[e] = acc;
        }
    }
}

/* attention of T block rows against the ctx slot ring + the block rows   *
 * (non-causal + sliding band: every query sees every valid key).         *
 * ckl/cvl = this layer's cache base ([head][slot][hd], slot = pos%cap).  */
static void ds4_dfl_attention(ds4_dflash_gpu *df, const float *q, const float *ckl,
                              const float *cvl, uint32_t n_ctx, const float *kb,
                              const float *vb, uint32_t T, float *o) {
    const uint32_t heads = df->heads, kvh = df->kv_heads, hd = df->hd;
    const uint32_t qph = heads / kvh;
    const float scale = 1.0f / sqrtf((float)hd);
    const int a_ctx = (int)(n_ctx ? n_ctx : 1);
    for (uint32_t h = 0; h < kvh; h++) {
        const float *K = ckl + (size_t)h * df->ring_cap * hd;
        const float *V = cvl + (size_t)h * df->ring_cap * hd;
        for (uint32_t qi = 0; qi < qph; qi++) {
            const uint32_t head = h * qph + qi;
            const float *Qh = q + (size_t)head * hd;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)T, a_ctx, (int)hd, scale,
                        Qh, (int)(heads * hd), K, (int)hd,
                        0.0f, df->score, (int)df->ring_cap);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)T, (int)T, (int)hd, scale,
                        Qh, (int)(heads * hd), kb + (size_t)h * hd, (int)(kvh * hd),
                        0.0f, df->score_f, (int)T);
            for (uint32_t t = 0; t < T; t++) {
                float *sp = df->score + (size_t)t * df->ring_cap;
                float *sf = df->score_f + (size_t)t * T;
                float mx = -1e30f;
                for (uint32_t i = 0; i < n_ctx; i++) if (sp[i] > mx) mx = sp[i];
                for (uint32_t i = 0; i < T; i++) if (sf[i] > mx) mx = sf[i];
                float zs = 0.0f;
                for (uint32_t i = 0; i < n_ctx; i++) {
                    const float e = expf(sp[i] - mx);
                    sp[i] = e;
                    zs += e;
                }
                float zf = 0.0f;
                for (uint32_t i = 0; i < T; i++) {
                    const float e = expf(sf[i] - mx);
                    sf[i] = e;
                    zf += e;
                }
                const float zi = 1.0f / (zs + zf);
                float *od = o + ((size_t)t * heads + head) * hd;
                if (n_ctx)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                1, (int)hd, (int)n_ctx, 1.0f,
                                sp, (int)df->ring_cap, V, (int)hd,
                                0.0f, od, (int)(heads * hd));
                else
                    for (uint32_t i = 0; i < hd; i++) od[i] = 0.0f;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            1, (int)hd, (int)T, 1.0f,
                            sf, (int)T, vb + (size_t)h * hd, (int)(kvh * hd),
                            1.0f, od, (int)(heads * hd));
                for (uint32_t i = 0; i < hd; i++) od[i] *= zi;
            }
        }
    }
}

/* top-k scan of one trunk-vocab row (ids into cand, logits into uv);     *
 * earlier ids win ties, matching sample_argmax.                          */
static void ds4_dfl_topk(const float *x, uint32_t n, uint32_t k, int32_t *cand, float *val) {
    for (uint32_t i = 0; i < k; i++) { cand[i] = -1; val[i] = -1e30f; }
    for (uint32_t i = 0; i < n; i++) {
        if (cand[k - 1] >= 0 && val[k - 1] >= x[i]) continue;
        uint32_t pos = k - 1;
        while (pos > 0 && val[pos - 1] < x[i]) {
            val[pos]  = val[pos - 1];
            cand[pos] = cand[pos - 1];
            pos--;
        }
        val[pos]  = x[i];
        cand[pos] = (int32_t)i;
    }
}

/* ================================================================== */
/* lifecycle                                                           */
/* ================================================================== */

static void ds4_dflash_probe_flush(ds4_dflash_gpu *df) {
    if (!df || !df->probe_cycles) return;
    const double mean = (double)df->probe_tokens / (double)df->probe_cycles;
    fprintf(stderr, "ds4: dflash probe: %llu cycles, mean %.3f tok/cycle, hist",
            (unsigned long long)df->probe_cycles, mean);
    for (uint32_t i = 0; i < df->block_size; i++)
        fprintf(stderr, " %u", df->probe_hist[i]);
    fprintf(stderr, " | first-accept %.1f%%\n",
            100.0 * (double)(df->probe_cycles - df->probe_hist[0]) / (double)df->probe_cycles);
    fflush(stderr);
}

/* W-D4 verify receipt line (final print is the authoritative run receipt) */
static void ds4_dflash_verify_flush(ds4_dflash_gpu *df) {
    if (!df || !df->verify_cycles) return;
    const double mean = (double)df->verify_tokens / (double)df->verify_cycles;
    fprintf(stderr, "ds4: dflash verify: %llu cycles, mean %.3f tok/cycle, hist",
            (unsigned long long)df->verify_cycles, mean);
    for (uint32_t i = 1; i <= df->block_size && i <= DS4_DFLASH_MAX_DRAFT + 1u; i++)
        fprintf(stderr, " %u", df->verify_hist[i]);
    fprintf(stderr, " | swaps %llu fallbacks %llu\n",
            (unsigned long long)df->verify_swaps, (unsigned long long)df->verify_fallbacks);
    fflush(stderr);
    if (df->tstat) {
        fprintf(stderr, "ds4: dfl tstat ms/call:");
        for (int s = 0; s < DS4_DFL_TSTAT_SLOTS; s++)
            if (df->tstat_n[s])
                fprintf(stderr, " %s %.2f", ds4_dfl_tstat_name[s],
                        df->tstat_ms[s] / (double)df->tstat_n[s]);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
}

static void ds4_dflash_free(struct ds4_dflash_gpu *df) {
    if (!df) return;
    if (df->dbg)
        fprintf(stderr, "ds4: dfl dbg free: cycles=%llu tokens=%llu "
                "hist[0]=%u pd_valid=%d wired=%d\n",
                (unsigned long long)df->probe_cycles,
                (unsigned long long)df->probe_tokens,
                df->probe_hist[0], df->pd_valid, df->wired);
    ds4_dflash_probe_flush(df);
    ds4_dflash_verify_flush(df);
    ds4_gpu_tensor_free(df->gxn);      ds4_gpu_tensor_free(df->gq);
    ds4_gpu_tensor_free(df->gattn);    ds4_gpu_tensor_free(df->gk);
    ds4_gpu_tensor_free(df->gv);       ds4_gpu_tensor_free(df->gwo);
    ds4_gpu_tensor_free(df->gconv);    ds4_gpu_tensor_free(df->gconv2);
    ds4_gpu_tensor_free(df->gffn1);    ds4_gpu_tensor_free(df->gffn2);
    ds4_gpu_tensor_free(df->gmid);     ds4_gpu_tensor_free(df->ggate);
    ds4_gpu_tensor_free(df->glogits);  ds4_gpu_tensor_free(df->gtap);
    ds4_gpu_tensor_free(df->ginj);     ds4_gpu_tensor_free(df->gfc);
    ds4_gpu_tensor_free(df->gk_inj);   ds4_gpu_tensor_free(df->gv_inj);
    ds4_gpu_tensor_free(df->gtap_ring);
    free(df->mask_rowf); free(df->fc_w);
    free(df->ck); free(df->cv);
    free(df->h);  free(df->xn); free(df->cr1); free(df->cr2); free(df->q);
    free(df->kb); free(df->vb); free(df->o);  free(df->ho); free(df->dyn);
    free(df->dyn2); free(df->gw); free(df->gu); free(df->gm); free(df->stage);
    free(df->enc); free(df->ik); free(df->iv); free(df->gate); free(df->logits);
    free(df->score); free(df->score_f); free(df->cand); free(df->uv);
    free(df->succ); free(df->pv);
    memset(df, 0, sizeof(*df));
    free(df);
}

static void ds4_dflash_reset(ds4_qwen4_gpu_graph *g) {
    ds4_dflash_gpu *df = (ds4_dflash_gpu *)g->dflash_gpu;
    if (!df) return;
    df->ring_base = 0;
    df->ring_valid = 0;
    df->inj_next = 0;
    df->pd_valid = false;
    df->pd_len = 0;
    df->pd_matched = 0;
}

/* attach: validate and allocate per-session drafter state.  Called once per
 * graph right after qwen35_graph_alloc when e->dflash; the drafter was
 * already bound and validated at engine open (W-D2), and the trained mask
 * row sidecar is mandatory there. */
static bool ds4_dflash_attach(const ds4_dflash_weights *dw, const float *mask_row,
                              ds4_model *dm, ds4_model *tm, const ds4_weights *tw,
                              ds4_qwen4_gpu_graph *g) {
    if (g->dflash_gpu) return true;
    if (!dw->block_count || !tw || !tw->token_embd || !tw->output) {
        fprintf(stderr, "ds4: dflash: attach requires bound drafter and trunk weights\n");
        return false;
    }
    if (!mask_row) {
        fprintf(stderr, "ds4: dflash: mask-row sidecar missing\n");
        return false;
    }
    if (dw->embedding_length != DS4_N_EMBD || tw->token_embd->dim[1] != DS4_N_VOCAB ||
        tw->output->dim[1] != DS4_N_VOCAB) {
        fprintf(stderr, "ds4: dflash: drafter E %u must match the trunk and the "
                        "head rows must cover the trunk vocabulary\n",
                dw->embedding_length);
        return false;
    }
    ds4_dflash_gpu *df = xcalloc(1, sizeof(*df));
    df->n_layer = dw->block_count;
    df->E       = dw->embedding_length;
    df->ff      = dw->feed_forward_length;
    df->heads   = dw->head_count;
    df->kv_heads = dw->head_count_kv;
    df->hd      = dw->key_length;
    df->block_size = dw->block_size;
    df->top_k   = dw->selector_top_k;
    df->rank    = dw->selector_rank;
    df->kernel  = dw->conv_kernel_size;
    df->group   = dw->conv_group_size;
    df->n_groups = df->E / df->group;
    df->n_taps  = dw->n_taps;
    df->window  = dw->sliding_window ? dw->sliding_window : DS4_DFLASH_WINDOW_DEFAULT;
    df->ring_cap = df->window;
    df->rope_base = dw->rope_freq_base > 0.0f ? dw->rope_freq_base : 10000000.0f;
    df->eps     = 1.0e-6f;
    {
        float eps = 0.0f;
        if (model_get_f32_compat(dm, "dflash.attention.layer_norm_rms_epsilon", &eps) && eps > 0.0f)
            df->eps = eps;
    }
    df->mask_id = dw->mask_token_id;
    df->cap_rows = dw->block_size;
    memcpy(df->tap_layers, dw->target_layers, sizeof(uint32_t) * dw->n_target_layers);
    df->dm = dm;
    df->tm = tm;
    df->tw = tw;
    df->dw = *dw;

    { const char *pv = getenv("DS4_DFLASH_PROBE");
      df->probe = pv && pv[0] && pv[0] != '0'; }
    /* W-D4: the verify cycle replaces the probe's stream-comparison decode;
     * both envs together would double-draft, so verify wins. */
    if (g->dflash_verify_rows > 1u && df->probe) {
        df->probe = false;
        fprintf(stderr, "ds4: dflash: DS4_DFLASH_VERIFY is set, probe stays off\n");
    }
    { const char *dv = getenv("DS4_DFLASH_DEBUG");
      df->dbg = dv && dv[0] && dv[0] != '0'; }
    /* W-D4b receipts (dfl_b4fc*.txt, 2026-10-23): the fc Q4_K gemv with
     * in-dim taps*E = 24576 runs on the multi-gemv kernel fine — the old
     * "no GPU kernel path" note was wrong — and the 302 MB host sgemm
     * (DRAM-bound ~6-8 ms per chunk, re-read every inject) drops to
     * ~0.6 ms of GPU time, so the GPU path is the default now; an explicit
     * DS4_DFLASH_GPU_FC=0 keeps the host sgemm for empirical A/B.        */
    { const char *gv = getenv("DS4_DFLASH_GPU_FC");
      df->gpu_fc = !gv || !gv[0] || gv[0] != '0'; }
    { const char *tv = getenv("DS4_DFLASH_TSTAT");
      df->tstat = tv && tv[0] && tv[0] != '0'; }
    df->inj_chunk = DS4_DFLASH_INJ_CAP;
    {
        const char *cv = getenv("DS4_DFLASH_INJ_CHUNK");
        if (cv && cv[0]) {
            const long v = strtol(cv, NULL, 10);
            if (v >= 1 && (unsigned long)v <= (unsigned long)DS4_DFLASH_INJ_CAP)
                df->inj_chunk = (uint32_t)v;
        }
    }

    bool ok = true;
#define DF_ALLOC(field_, floats_) do { df->field_ = qwen4_graph_alloc_f32(floats_); \
        ok = ok && df->field_; } while (0)
    DF_ALLOC(gxn,      (uint64_t)df->cap_rows * df->E);
    DF_ALLOC(gq,       (uint64_t)df->cap_rows * df->heads * df->hd);
    DF_ALLOC(gattn,    (uint64_t)df->cap_rows * df->heads * df->hd);
    DF_ALLOC(gk,       (uint64_t)df->cap_rows * df->kv_heads * df->hd);
    DF_ALLOC(gv,       (uint64_t)df->cap_rows * df->kv_heads * df->hd);
    DF_ALLOC(gwo,      (uint64_t)df->cap_rows * df->E);
    DF_ALLOC(gconv,    (uint64_t)df->cap_rows * 2u * df->kernel * df->n_groups);
    DF_ALLOC(gconv2,   (uint64_t)df->cap_rows * 2u * df->kernel * df->n_groups);
    DF_ALLOC(gffn1,    (uint64_t)df->cap_rows * df->ff);
    DF_ALLOC(gffn2,    (uint64_t)df->cap_rows * df->ff);
    DF_ALLOC(gmid,     (uint64_t)df->cap_rows * df->ff);
    DF_ALLOC(ggate,    (uint64_t)df->cap_rows * df->rank);
    DF_ALLOC(glogits,  (uint64_t)df->cap_rows * DS4_N_VOCAB);
    DF_ALLOC(gtap,     (uint64_t)DS4_DFLASH_INJ_CAP * df->n_taps * df->E);
    DF_ALLOC(ginj,     (uint64_t)DS4_DFLASH_INJ_CAP * df->E);
    DF_ALLOC(gfc,      (uint64_t)DS4_DFLASH_INJ_CAP * df->E);
    DF_ALLOC(gk_inj,   (uint64_t)df->n_layer * DS4_DFLASH_INJ_CAP * df->kv_heads * df->hd);
    DF_ALLOC(gv_inj,   (uint64_t)df->n_layer * DS4_DFLASH_INJ_CAP * df->kv_heads * df->hd);
    DF_ALLOC(gtap_ring,
             (uint64_t)df->n_taps * df->ring_cap * df->E);
#undef DF_ALLOC
#define DF_MALLOC(field_, bytes_) do { df->field_ = xmalloc(bytes_); ok = ok && df->field_; } while (0)
    DF_MALLOC(mask_rowf, (size_t)df->E * sizeof(float));
    DF_MALLOC(ck, (size_t)df->n_layer * df->kv_heads * df->ring_cap * df->hd * sizeof(float));
    DF_MALLOC(cv, (size_t)df->n_layer * df->kv_heads * df->ring_cap * df->hd * sizeof(float));
    DF_MALLOC(h,   (size_t)df->cap_rows * df->E * sizeof(float));
    DF_MALLOC(xn,  (size_t)df->cap_rows * df->E * sizeof(float));
    DF_MALLOC(cr1, (size_t)df->cap_rows * df->E * sizeof(float));
    DF_MALLOC(cr2, (size_t)df->cap_rows * df->E * sizeof(float));
    DF_MALLOC(q,  (size_t)df->cap_rows * df->heads * df->hd * sizeof(float));
    DF_MALLOC(kb, (size_t)df->cap_rows * df->kv_heads * df->hd * sizeof(float));
    DF_MALLOC(vb, (size_t)df->cap_rows * df->kv_heads * df->hd * sizeof(float));
    DF_MALLOC(o,  (size_t)df->cap_rows * df->heads * df->hd * sizeof(float));
    DF_MALLOC(ho, (size_t)df->cap_rows * df->E * sizeof(float));
    DF_MALLOC(dyn, (size_t)df->cap_rows * 2u * df->kernel * df->n_groups * sizeof(float));
    DF_MALLOC(dyn2, (size_t)df->cap_rows * 2u * df->kernel * df->n_groups * sizeof(float));
    DF_MALLOC(gw, (size_t)df->cap_rows * df->ff * sizeof(float));
    DF_MALLOC(gu, (size_t)df->cap_rows * df->ff * sizeof(float));
    DF_MALLOC(gm, (size_t)df->cap_rows * df->ff * sizeof(float));
    DF_MALLOC(stage, (size_t)DS4_DFLASH_INJ_CAP * df->n_taps * df->E * sizeof(float));
    DF_MALLOC(enc, (size_t)DS4_DFLASH_INJ_CAP * df->E * sizeof(float));
    DF_MALLOC(ik, (size_t)DS4_DFLASH_INJ_CAP * df->kv_heads * df->hd * sizeof(float));
    DF_MALLOC(iv, (size_t)DS4_DFLASH_INJ_CAP * df->kv_heads * df->hd * sizeof(float));
    DF_MALLOC(gate, (size_t)df->cap_rows * df->rank * sizeof(float));
    DF_MALLOC(logits, (size_t)df->cap_rows * DS4_N_VOCAB * sizeof(float));
    DF_MALLOC(score, (size_t)df->cap_rows * (df->ring_cap + 1u) * sizeof(float));
    DF_MALLOC(score_f, (size_t)df->cap_rows * df->cap_rows * sizeof(float));
    DF_MALLOC(cand, sizeof(int32_t) * (size_t)df->cap_rows * df->top_k);
    DF_MALLOC(uv, sizeof(float) * (size_t)df->cap_rows * df->top_k);
    DF_MALLOC(succ, (size_t)df->top_k * df->rank * sizeof(float));
    DF_MALLOC(pv, (size_t)df->rank * sizeof(float));
    DF_MALLOC(fc_w, (size_t)df->E * (size_t)df->n_taps * df->E * sizeof(float));
#undef DF_MALLOC
    if (ok) {
        /* host fc fallback (DS4_DFLASH_GPU_FC=0): the drafter's only tensor
         * with in-dim taps*E = 24576 — W-D4b receipts show the GPU multi-gemv
         * handles that in-dim, so this host dequant + sgemm only runs on
         * request (DRAM-bound ~6-8 ms per inject chunk).                    */
        for (uint32_t e = 0; ok && e < df->E; e++)
            ds4_dfl_row(df->dm, dw->fc, e, df->fc_w + (size_t)e * df->n_taps * df->E);
    }
    if (ok) {
        memcpy(df->mask_rowf, mask_row, (size_t)df->E * sizeof(float));
        df->wired = true;
        g->dflash_gpu = df;
        fprintf(stderr, "ds4: dflash: drafter attached (%u layers, window %u, block %u, "
                        "taps %u, probe=%d)\n",
                df->n_layer, df->window, df->block_size, df->n_taps, df->probe);
        if (df->dbg)
            fprintf(stderr, "ds4: dfl dbg offsets: sizeof=%zu gtap_ring+%zu gxn+%zu "
                    "gfc+%zu stage+%zu ck+%zu\n",
                    sizeof(*df),
                    (size_t)((char *)&df->gtap_ring - (char *)df),
                    (size_t)((char *)&df->gxn - (char *)df),
                    (size_t)((char *)&df->gfc - (char *)df),
                    (size_t)((char *)&df->stage - (char *)df),
                    (size_t)((char *)&df->ck - (char *)df));
        return true;
    }
    fprintf(stderr, "ds4: dflash: attach failed (allocation)\n");
    ds4_dflash_free(df);
    return false;
}

/* ================================================================== */
/* tap capture — called from qwen35_graph_forward_tokens' layer loop    */
/* while the forward's command batch is open: every tap layer's residual*/
/* rows are blitted into this module's GPU ring (slot = pos % ring_cap) */
/* ================================================================== */
static void ds4_dflash_tap_hook(ds4_qwen4_gpu_graph *g, uint32_t il, uint32_t pos0, uint32_t T) {
    ds4_dflash_gpu *df = (ds4_dflash_gpu *)g->dflash_gpu;
    if (!df || !df->wired || pos0 >= g->ctx_cap) return;
    uint32_t tslot = UINT32_MAX;
    for (uint32_t i = 0; i < df->n_taps; i++)
        if (df->tap_layers[i] == il) { tslot = i; break; }
    if (tslot == UINT32_MAX) return;
    const uint32_t end0 = pos0 + T;
    const uint32_t keep_from = end0 > df->ring_cap ? end0 - df->ring_cap : 0;
    const uint32_t row_bytes = (uint64_t)df->E * sizeof(float);
    const uint64_t plane_off = (uint64_t)tslot * df->ring_cap * df->E * sizeof(float);
    uint32_t pos = pos0;
    while (pos < end0 && pos < g->ctx_cap) {
        uint32_t run = end0 - pos;
        if (run > df->ring_cap - (pos % df->ring_cap)) run = df->ring_cap - (pos % df->ring_cap);
        if (pos + run > g->ctx_cap) run = g->ctx_cap - pos;
        if (pos < keep_from) {                     /* rows the ring will drop */
            pos += run;
            continue;
        }
        const uint64_t dst = plane_off + (uint64_t)(pos % df->ring_cap) * row_bytes;
        const uint64_t src = (uint64_t)(pos - pos0) * row_bytes;
        if (ds4_gpu_tensor_copy(df->gtap_ring, dst, g->R, src,
                                (uint64_t)run * row_bytes) == 0) {
            fprintf(stderr, "ds4: dflash: tap ring copy failed (pos %u layer %u)\n", pos, il);
            df->wired = false;
            return;
        }
        pos += run;
    }
    /* host ring bookkeeping */
    if (df->ring_valid < df->ring_cap) {
        df->ring_valid = end0 > df->ring_cap ? df->ring_cap : end0;
        if (df->ring_valid >= df->ring_cap)
            df->ring_base = end0 - df->ring_cap;
    } else if (end0 > df->ring_base + df->ring_cap) {
        df->ring_base = end0 - df->ring_cap;
    }
}

/* ================================================================== */
/* inject: append committed rows' features to the drafter ctx cache      */
/* ================================================================== */
static void ds4_dflash_inject(ds4_qwen4_gpu_graph *g, uint32_t last) {
    ds4_dflash_gpu *df = (ds4_dflash_gpu *)g->dflash_gpu;
    if (!df || !df->wired) return;
    const ds4_dflash_weights *dw = &df->dw;
    const uint32_t KV = df->kv_heads * df->hd;
    const size_t  row_bytes = (size_t)df->n_taps * df->E * sizeof(float);
    const uint64_t erow = (uint64_t)df->E * sizeof(float);

    while (df->inj_next <= last && df->wired) {
        uint32_t from = df->inj_next;
        if (df->ring_valid >= df->ring_cap && from < df->ring_base)
            from = df->ring_base;                     /* tap window trim */
        if (from > last) { df->inj_next = last + 1u; break; }
        uint32_t n = last - from + 1u;
        if (n > df->inj_chunk) n = df->inj_chunk;

        /* 1. fc input: tap planes -> fused rows (row = n_taps concat)      */
        if (df->dbg && df->probe_seq == 0)
            fprintf(stderr, "ds4: dfl dbg inject: last=%u from=%u n=%u chunk=%u ring(base=%u valid=%u "
                    "ringb=%llu gfc=%llu gk=%llu gv=%llu stage=%p)\n",
                    last, from, n, df->inj_chunk, df->ring_base, df->ring_valid,
                    (unsigned long long)ds4_gpu_tensor_bytes(df->gtap_ring),
                    (unsigned long long)ds4_gpu_tensor_bytes(df->gfc),
                    (unsigned long long)ds4_gpu_tensor_bytes(df->gk_inj),
                    (unsigned long long)ds4_gpu_tensor_bytes(df->gv_inj),
                    (void *)df->stage);
        const double ts = df->tstat ? now_sec() : 0.0;           /* TSTAT inj_taps */
        for (uint32_t j = 0; j < n; j++) {
            const uint32_t pos = from + j;
            const uint32_t slot = pos % df->ring_cap;
            for (uint32_t tp = 0; tp < df->n_taps; tp++) {
                const uint64_t off = ((uint64_t)tp * df->ring_cap + slot) * erow;
                /* stage rows are float-indexed: row j starts at j*(n_taps*E) */
                if (ds4_gpu_tensor_read(df->gtap_ring, off,
                                        df->stage + ((size_t)j * df->n_taps + tp) * df->E,
                                        erow) != 1) {
                    fprintf(stderr, "ds4: dflash: inject ring read failed (tp=%u slot=%u j=%u n=%u)\n",
                            tp, slot, j, n);
                    df->wired = false;
                    return;
                }
            }
        }
        DS4_DFL_TEND(df, 0, ts);
        const double tw = df->tstat ? now_sec() : 0.0;            /* TSTAT inj_w   */
        if (ds4_gpu_tensor_write(df->gtap, 0, df->stage, (size_t)n * row_bytes) != 1) {
            fprintf(stderr, "ds4: dflash: inject gtap write failed (n=%u)\n", n);
            df->wired = false;
            return;
        }
        DS4_DFL_TEND(df, 1, tw);
        const double tfc = df->tstat ? now_sec() : 0.0;           /* TSTAT inj_fc  */
        if (df->gpu_fc) {
            if (!glm_graph_begin_commands_if_needed()) { df->wired = false; return; }
            if (!qwen4_gemv(df->gfc, df->dm, dw->fc, df->gtap, n)) {
                df->wired = false;
                return;
            }
            if (!ds4_gpu_end_commands()) { df->wired = false; return; }
            if (ds4_gpu_tensor_read(df->gfc, 0, df->enc, (size_t)n * erow) != 1) {
                fprintf(stderr, "ds4: dflash: inject enc read failed\n");
                df->wired = false;
                return;
            }
        } else {
            /* C[n][E] = stage[n][taps*E] * fc_w[E][taps*E]^T (row-major host) */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        (int)n, (int)df->E, (int)(df->n_taps * df->E),
                        1.0f, df->stage, (int)(df->n_taps * df->E),
                        df->fc_w, (int)(df->n_taps * df->E),
                        0.0f, df->enc, (int)df->E);
        }
        DS4_DFL_TEND(df, 2, tfc);
        const double trm = df->tstat ? now_sec() : 0.0;           /* TSTAT inj_rms */
        {
            const float *ew = ds4_dfl_f32(df->dm, dw->enc_output_norm);
            if (!ew) { df->wired = false; return; }
            for (uint32_t j = 0; j < n; j++)
                ds4_dfl_rms(df->enc + (size_t)j * df->E, ew, df->E, df->eps,
                            df->enc + (size_t)j * df->E);
        }
        DS4_DFL_TEND(df, 3, trm);

        /* 2. all layers: k/v projections share the same enc rows — one     *
         * command batch for the whole stack (per-layer planes inside       *
         * gk_inj/gv_inj), one sync, then per-layer read + cache append     */
        const double tkv = df->tstat ? now_sec() : 0.0;           /* TSTAT inj_kv  */
        ds4_gpu_tensor *kvw[DS4_DFLASH_MAX_LAYERS] = {0};
        ds4_gpu_tensor *vvw[DS4_DFLASH_MAX_LAYERS] = {0};
        bool kok = true, opened = false;
        for (uint32_t il = 0; kok && il < df->n_layer; il++) {
            const uint64_t plane = (uint64_t)il * DS4_DFLASH_INJ_CAP * KV * sizeof(float);
            const uint64_t bytes = (uint64_t)DS4_DFLASH_INJ_CAP * KV * sizeof(float);
            kvw[il] = ds4_gpu_tensor_view(df->gk_inj, plane, bytes);
            vvw[il] = ds4_gpu_tensor_view(df->gv_inj, plane, bytes);
            kok = kvw[il] && vvw[il];
        }
        if (kok && ds4_gpu_tensor_write(df->ginj, 0, df->enc, (size_t)n * erow) != 1) {
            fprintf(stderr, "ds4: dflash: inject ginj write failed\n");
            kok = false;
        }
        /* the probe runs outside forwards: open the batch ourselves so
         * gemv dispatch + end_commands pair up (a bare end_commands
         * fails when no batch is open, and the raw gemvs would run
         * owned-synchronous and leave the readbacks racing)          */
        if (kok) opened = glm_graph_begin_commands_if_needed();
        kok = kok && opened;
        for (uint32_t il = 0; kok && il < df->n_layer; il++) {
            if (!qwen4_gemv(kvw[il], df->dm, dw->block[il].attn_k, df->ginj, n) ||
                !qwen4_gemv(vvw[il], df->dm, dw->block[il].attn_v, df->ginj, n)) {
                fprintf(stderr, "ds4: dflash: inject k/v gemv failed (layer %u n=%u)\n", il, n);
                kok = false;
            }
        }
        if (opened && !ds4_gpu_end_commands()) kok = false;
        DS4_DFL_TEND(df, 4, tkv);
        for (uint32_t il = 0; kok && il < df->n_layer; il++) {
            const double tkh = df->tstat ? now_sec() : 0.0;       /* TSTAT inj_kvhost */
            if (ds4_gpu_tensor_read(kvw[il], 0, df->ik, (size_t)n * KV * sizeof(float)) != 1 ||
                ds4_gpu_tensor_read(vvw[il], 0, df->iv, (size_t)n * KV * sizeof(float)) != 1) {
                fprintf(stderr, "ds4: dflash: inject kv read failed (layer %u)\n", il);
                kok = false;
            }
            if (kok) {
                const float *knw = ds4_dfl_f32(df->dm, dw->block[il].attn_k_norm);
                if (!knw) { kok = false; }
                else {
                    const size_t lbase = (size_t)il * df->kv_heads * df->ring_cap * df->hd;
                    for (uint32_t j = 0; j < n; j++) {
                        const uint32_t pos = from + j;
                        const size_t dst = lbase + (size_t)(pos % df->ring_cap) * df->hd;
                        ds4_dfl_head_rms(df->ik + (size_t)j * KV, 1u, df->kv_heads, df->hd, knw, df->eps);
                        for (uint32_t h = 0; h < df->kv_heads; h++) {
                            float *kh = df->ik + (size_t)j * KV + (size_t)h * df->hd;
                            ds4_dfl_rope(kh, df->hd, (double)pos, df->rope_base);
                            memcpy(df->ck + dst + (size_t)h * df->ring_cap * df->hd,
                                   kh, (size_t)df->hd * sizeof(float));
                            memcpy(df->cv + dst + (size_t)h * df->ring_cap * df->hd,
                                   df->iv + ((size_t)j * KV) + (size_t)h * df->hd,
                                   (size_t)df->hd * sizeof(float));
                        }
                    }
                }
            }
            DS4_DFL_TEND(df, 5, tkh);
        }
        DFLASH_KV_VIEW_FREE(kvw, vvw);              /* always releases views */
        if (!kok) { df->wired = false; return; }
        df->inj_next = from + n;
    }
}

/* number of valid ctx rows the attention may read (slot ring [0..n))     */
static uint32_t ds4_dflash_n_ctx(const ds4_dflash_gpu *df) {
    if (!df->wired || df->inj_next == 0) return 0;
    const uint32_t vstart = df->ring_valid >= df->ring_cap ? df->ring_base : 0u;
    if (df->inj_next <= vstart) return 0;
    uint32_t n = df->inj_next - vstart;
    return n > df->ring_cap ? df->ring_cap : n;
}

/* ================================================================== */
/* draft block cycle: anchor + mask rows through the drafter layers,    *
 * trunk head, selector walk.  Returns n_draft tokens.                  */
/* ================================================================== */
static bool ds4_dflash_draft(ds4_qwen4_gpu_graph *g, int32_t anchor, uint32_t pos0,
                             uint32_t n_draft, int32_t *drafts, uint32_t *n_out) {
    ds4_dflash_gpu *df = (ds4_dflash_gpu *)g->dflash_gpu;
    const ds4_dflash_weights *dw = &df->dw;
    const uint32_t T = n_draft + 1u;
    const uint32_t E = df->E, KV = df->kv_heads * df->hd, QD = df->heads * df->hd;
    const uint64_t erow = (uint64_t)E * sizeof(float);
    const size_t ffrow = (size_t)df->ff * sizeof(float);
    const uint32_t dk = 2u * df->kernel * df->n_groups;
    const uint32_t n_ctx = ds4_dflash_n_ctx(df);

    *n_out = 0;
    if (anchor < 0 || anchor >= (int32_t)DS4_N_VOCAB || !df->wired) return false;
    if (T < 2u || T > df->cap_rows) return false;

    /* 1. block embeddings: the anchor row, then the trained mask row       */
    qwen4_ref_row(df->tm, df->tw->token_embd, (uint64_t)anchor, df->h);
    for (uint32_t i = 1; i < T; i++)
        memcpy(df->h + (size_t)i * E, df->mask_rowf, (size_t)E * sizeof(float));

    /* 2. the drafter layers                                                */
    for (uint32_t il = 0; il < df->n_layer; il++) {
        const size_t lbase = (size_t)il * df->kv_heads * df->ring_cap * df->hd;
        /* --- attention half                                               */
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_rmsw */
            const float *aw = ds4_dfl_f32(df->dm, dw->block[il].attn_norm);
            if (!aw) return false;
            for (uint32_t t = 0; t < T; t++)
                ds4_dfl_rms(df->h + (size_t)t * E, aw, E, df->eps, df->xn + (size_t)t * E);
            if (ds4_gpu_tensor_write(df->gxn, 0, df->xn, (size_t)T * erow) != 1) return false;
            DS4_DFL_TEND(df, 6, t0);
            const double t1 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_cproj */
            if (!glm_graph_begin_commands_if_needed()) return false;
            if (!qwen4_gemv(df->gconv, df->dm, dw->block[il].attn_conv_proj, df->gxn, T))
                return false;
            if (!ds4_gpu_end_commands()) return false;
            if (ds4_gpu_tensor_read(df->gconv, 0, df->dyn, (size_t)T * dk * sizeof(float)) != 1)
                return false;
            DS4_DFL_TEND(df, 7, t1);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_conv0 */
            const float *basep = ds4_dfl_f32(df->dm, dw->block[il].attn_conv_base);
            if (!basep) return false;
            ds4_dfl_conv(T, df->block_size, E, df->group, df->kernel,
                         df->dyn, 0, basep, df->xn, df->cr1);
            if (ds4_gpu_tensor_write(df->gxn, 0, df->cr1, (size_t)T * erow) != 1) return false;
            DS4_DFL_TEND(df, 8, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_qkv   */
            if (!glm_graph_begin_commands_if_needed()) return false;
            if (!qwen4_gemv(df->gq, df->dm, dw->block[il].attn_q, df->gxn, T) ||
                !qwen4_gemv(df->gk, df->dm, dw->block[il].attn_k, df->gxn, T) ||
                !qwen4_gemv(df->gv, df->dm, dw->block[il].attn_v, df->gxn, T) ||
                !ds4_gpu_end_commands() ||
                ds4_gpu_tensor_read(df->gq, 0, df->q, (size_t)T * QD * sizeof(float)) != 1 ||
                ds4_gpu_tensor_read(df->gk, 0, df->kb, (size_t)T * KV * sizeof(float)) != 1 ||
                ds4_gpu_tensor_read(df->gv, 0, df->vb, (size_t)T * KV * sizeof(float)) != 1)
                return false;
            DS4_DFL_TEND(df, 9, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_rope  */
            const float *qw = ds4_dfl_f32(df->dm, dw->block[il].attn_q_norm);
            const float *kw = ds4_dfl_f32(df->dm, dw->block[il].attn_k_norm);
            if (!qw || !kw) return false;
            ds4_dfl_head_rms(df->q, T, df->heads, df->hd, qw, df->eps);
            ds4_dfl_head_rms(df->kb, T, df->kv_heads, df->hd, kw, df->eps);
            for (uint32_t t = 0; t < T; t++) {
                for (uint32_t h = 0; h < df->heads; h++)
                    ds4_dfl_rope(df->q + (size_t)t * QD + (size_t)h * df->hd, df->hd,
                                 (double)(pos0 + t), df->rope_base);
                for (uint32_t h = 0; h < df->kv_heads; h++)
                    ds4_dfl_rope(df->kb + (size_t)t * KV + (size_t)h * df->hd, df->hd,
                                 (double)(pos0 + t), df->rope_base);
            }
            DS4_DFL_TEND(df, 10, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_attn  */
            ds4_dfl_attention(df, df->q, df->ck + lbase, df->cv + lbase, n_ctx,
                              df->kb, df->vb, T, df->o);
            DS4_DFL_TEND(df, 11, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_wo    */
            if (ds4_gpu_tensor_write(df->gattn, 0, df->o, (size_t)T * QD * sizeof(float)) != 1)
                return false;
            if (!glm_graph_begin_commands_if_needed()) return false;
            if (!qwen4_gemv(df->gwo, df->dm, dw->block[il].attn_output, df->gattn, T) ||
                !ds4_gpu_end_commands() ||
                ds4_gpu_tensor_read(df->gwo, 0, df->ho, (size_t)T * erow) != 1)
                return false;
            DS4_DFL_TEND(df, 12, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_hup   */
            const float *basep = ds4_dfl_f32(df->dm, dw->block[il].attn_conv_base);
            if (!basep) return false;
            ds4_dfl_conv(T, df->block_size, E, df->group, df->kernel,
                         df->dyn, 1, basep, df->ho, df->cr2);
            for (size_t i = 0; i < (size_t)T * E; i++) df->h[i] += df->cr2[i];
            DS4_DFL_TEND(df, 13, t0);
        }

        /* --- ffn half                                                     */
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_fproj */
            const float *fwn = ds4_dfl_f32(df->dm, dw->block[il].ffn_norm);
            if (!fwn) return false;
            for (uint32_t t = 0; t < T; t++)
                ds4_dfl_rms(df->h + (size_t)t * E, fwn, E, df->eps, df->xn + (size_t)t * E);
            if (ds4_gpu_tensor_write(df->gxn, 0, df->xn, (size_t)T * erow) != 1) return false;
            if (!glm_graph_begin_commands_if_needed()) return false;
            if (!qwen4_gemv(df->gconv2, df->dm, dw->block[il].ffn_conv_proj, df->gxn, T) ||
                !ds4_gpu_end_commands() ||
                ds4_gpu_tensor_read(df->gconv2, 0, df->dyn2, (size_t)T * dk * sizeof(float)) != 1)
                return false;
            DS4_DFL_TEND(df, 14, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_fconv */
            const float *basep = ds4_dfl_f32(df->dm, dw->block[il].ffn_conv_base);
            if (!basep) return false;
            ds4_dfl_conv(T, df->block_size, E, df->group, df->kernel,
                         df->dyn2, 0, basep, df->xn, df->cr1);
            if (ds4_gpu_tensor_write(df->gxn, 0, df->cr1, (size_t)T * erow) != 1) return false;
            DS4_DFL_TEND(df, 15, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_gate: merged gate+up batch (W-D4b) */
            if (!glm_graph_begin_commands_if_needed()) return false;
            /* the gate and up projections share xn and carry no host step
             * between: one command batch, one sync (the two readbacks ride
             * the same fence)                                             */
            if (!qwen4_gemv(df->gffn1, df->dm, dw->block[il].ffn_gate, df->gxn, T) ||
                !qwen4_gemv(df->gffn2, df->dm, dw->block[il].ffn_up, df->gxn, T) ||
                !ds4_gpu_end_commands() ||
                ds4_gpu_tensor_read(df->gffn1, 0, df->gw, (size_t)T * ffrow) != 1 ||
                ds4_gpu_tensor_read(df->gffn2, 0, df->gu, (size_t)T * ffrow) != 1)
                return false;
            DS4_DFL_TEND(df, 16, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_silu  */
            for (size_t i = 0; i < (size_t)T * df->ff; i++) {
                const float v = df->gw[i];
                df->gm[i] = v / (1.0f + expf(-v)) * df->gu[i];
            }
            if (ds4_gpu_tensor_write(df->gmid, 0, df->gm, (size_t)T * ffrow) != 1) return false;
            DS4_DFL_TEND(df, 18, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_down  */
            if (!glm_graph_begin_commands_if_needed()) return false;
            if (!qwen4_gemv(df->gwo, df->dm, dw->block[il].ffn_down, df->gmid, T) ||
                !ds4_gpu_end_commands() ||
                ds4_gpu_tensor_read(df->gwo, 0, df->ho, (size_t)T * erow) != 1)
                return false;
            DS4_DFL_TEND(df, 19, t0);
        }
        {
            const double t0 = df->tstat ? now_sec() : 0.0;        /* TSTAT d_fup2  */
            const float *basep = ds4_dfl_f32(df->dm, dw->block[il].ffn_conv_base);
            if (!basep) return false;
            ds4_dfl_conv(T, df->block_size, E, df->group, df->kernel,
                         df->dyn2, 1, basep, df->ho, df->cr2);
            for (size_t i = 0; i < (size_t)T * E; i++) df->h[i] += df->cr2[i];
            DS4_DFL_TEND(df, 20, t0);
        }
    }

    /* 3. final norm + the trunk's output head + selector gate              */
    {
        const double t0 = df->tstat ? now_sec() : 0.0;            /* TSTAT d_final */
        const float *ow = ds4_dfl_f32(df->dm, dw->output_norm);
        if (!ow) return false;
        for (uint32_t t = 0; t < T; t++)
            ds4_dfl_rms(df->h + (size_t)t * E, ow, E, df->eps, df->xn + (size_t)t * E);
        if (ds4_gpu_tensor_write(df->gxn, 0, df->xn, (size_t)T * erow) != 1) return false;
        if (!glm_graph_begin_commands_if_needed()) return false;
        if (!qwen4_gemv(df->glogits, df->tm, df->tw->output, df->gxn, T) ||
            !qwen4_gemv(df->ggate, df->dm, dw->sel_hidden, df->gxn, T) ||
            !ds4_gpu_end_commands() ||
            ds4_gpu_tensor_read(df->glogits, 0, df->logits,
                                (size_t)T * DS4_N_VOCAB * sizeof(float)) != 1 ||
            ds4_gpu_tensor_read(df->ggate, 0, df->gate,
                                (size_t)T * df->rank * sizeof(float)) != 1)
            return false;
        DS4_DFL_TEND(df, 21, t0);
    }

    /* 4. candidates then the greedy lattice walk                           */
    const double dwt = df->tstat ? now_sec() : 0.0;              /* TSTAT d_walk  */
    for (uint32_t i = 1; i <= n_draft; i++)
    ds4_dfl_topk(df->logits + (size_t)i * DS4_N_VOCAB, DS4_N_VOCAB, df->top_k,
                 df->cand + (size_t)i * df->top_k, df->uv + (size_t)i * df->top_k);
    int32_t carried = 0;
    for (uint32_t i = 1; i <= n_draft; i++) {
        const uint32_t K = df->top_k, R = df->rank;
        const float *gr = df->gate + (size_t)i * R;
        const int32_t pred_id = i == 1 ? anchor
            : df->cand[(size_t)(i - 1) * K + carried];
        if (pred_id < 0 || pred_id >= (int32_t)DS4_N_VOCAB) return false;
        ds4_dfl_row(df->dm, dw->sel_predecessor, (uint64_t)pred_id, df->pv);
        for (uint32_t c = 0; c < K; c++)
            ds4_dfl_row(df->dm, dw->sel_successor,
                        (uint64_t)df->cand[(size_t)i * K + c],
                        df->succ + (size_t)c * R);
        uint32_t best = 0;
        float best_s = -1e30f;
        for (uint32_t c = 0; c < K; c++) {
            float s = df->uv[(size_t)i * K + c];
            const float *sv = df->succ + (size_t)c * R;
            for (uint32_t r = 0; r < R; r++)
                s += sv[r] * (df->pv[r] * gr[r]);
            if (s > best_s) { best_s = s; best = c; }
        }
        carried = (int32_t)best;
        drafts[i - 1] = df->cand[(size_t)i * K + best];
    }
    DS4_DFL_TEND(df, 22, dwt);
    if (df->dbg) {                             /* row-0 argmax: the drafter's
                                                * own next-token prediction */
        int32_t b0 = 0;
        float bv = -1e30f;
        const float *r0 = df->logits;
        for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
            if (r0[v] > bv) { bv = r0[v]; b0 = (int32_t)v; }
        df->dbg_rc[0] = b0;
        df->dbg_rc[1] = drafts[0];
        df->dbg_rc[2] = n_draft > 1 ? drafts[1] : -1;
        df->dbg_rc[3] = n_draft > 2 ? drafts[2] : -1;
    }
    *n_out = n_draft;
    return true;
}

/* ================================================================== */
/* probe hook: ctx injection, one draft cycle, stream comparison        */
/* ================================================================== */
static void ds4_dflash_probe_step(ds4_qwen4_gpu_graph *g, int32_t token) {
    ds4_dflash_gpu *df = (ds4_dflash_gpu *)g->dflash_gpu;
    if (!df || !df->wired || !df->probe) {
        if (df && df->dbg && df->probe_seq < 2u)
            fprintf(stderr, "ds4: dfl dbg step skip: wired=%d probe=%d\n",
                    df ? df->wired : 0, df ? df->probe : 0);
        return;
    }
    if (df->dbg && df->probe_seq < 3u)
        fprintf(stderr, "ds4: dfl dbg step pos0=%u tok=%d seq=%llu\n",
                g->pos, token, (unsigned long long)df->probe_seq);
    if (df->dbg && df->probe_seq < 17u)
        fprintf(stderr, "ds4: dfl dbg t: pos0=%u tok=%d prev(row0=%d d0=%d d1=%d d2=%d) "
                "want0=%d\n",
                g->pos, token, df->dbg_rc[0], df->dbg_rc[1], df->dbg_rc[2], df->dbg_rc[3],
                df->pd_valid ? (int32_t)df->pd_drafts[0] : -1);
    const uint32_t pos0 = g->pos;                 /* the anchor's position  */
    ds4_dflash_inject(g, pos0 - 1u);
    if (!df->wired) return;

    /* score the previous cycle's drafts against this commit                */
    if (df->pd_valid) {
        const uint32_t want = df->pd_pos0 + df->pd_matched + 1u;
        if (pos0 != want) {
            df->probe_hist[df->pd_matched]++;
            df->probe_tokens += df->pd_matched + 1u;
            df->probe_cycles++;
            df->pd_valid = false;
        } else if (token == df->pd_drafts[df->pd_matched]) {
            df->pd_matched++;
            if (df->pd_matched >= df->pd_len) {
                df->probe_hist[df->pd_len]++;
                df->probe_tokens += df->pd_len + 1u;
                df->probe_cycles++;
                df->pd_valid = false;
            }
        } else {
            df->probe_hist[df->pd_matched]++;
            df->probe_tokens += df->pd_matched + 1u;
            df->probe_cycles++;
            df->pd_valid = false;
        }
    }

    /* a fresh block is drafted only when no cycle is pending: a pending    *
     * cycle extends across steps as its drafts verify one-by-one, closed   *
     * by the scorer above on mismatch or full-block consumption.           *
     * (Redrafting every step resets the streak before it can grow.)        */
    if (!df->pd_valid) {
        uint32_t nd = df->block_size - 1u;
        {
            const char *e = getenv("DS4_DFLASH_NDRAFT");
            if (e && e[0]) {
                const long v = strtol(e, NULL, 10);
                if (v >= 1 && v <= (long)(df->block_size - 1u)) nd = (uint32_t)v;
            }
        }
        int32_t *drafts = xmalloc(sizeof(int32_t) * nd);
        if (!drafts) return;
        uint32_t got = 0;
        const bool ok = ds4_dflash_draft(g, token, pos0, nd, drafts, &got);
        if (ok && got == nd) {
            memcpy(df->pd_drafts, drafts, sizeof(int32_t) * nd);
            df->pd_pos0 = pos0;
            df->pd_len = nd;
            df->pd_matched = 0;
            df->pd_valid = true;
        } else {
            /* a failed cycle disables further probing; decode is never touched */
            df->probe = false;
            fprintf(stderr, "ds4: dflash probe: draft cycle failed/probe off at pos %u\n", pos0);
        }
        free(drafts);
    }
    df->probe_seq++;
    if ((df->probe_seq % 256u) == 0u) ds4_dflash_probe_flush(df);
}
