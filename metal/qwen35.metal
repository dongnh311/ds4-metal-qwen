// Ornith-1.5-35B-A3B (llama.cpp qwen35moe) kernels.  This file is appended
// after qwen4.metal in the single Metal library, so the qwen4 helpers and
// argument structs (qwen4_row_dot, qwen4_silu, qwen4_sigmoid,
// ds4_metal_args_qwen4_moe, ds4_metal_args_qwen4_gdn_out) are in scope.
// Everything here is a new entry point: the Qwen3.8 kernels are not touched.

/* --- Q5_K routed experts ------------------------------------------------ */

/* q5_K: 176-byte super-blocks of 256 (d, dmin, 12 packed 6-bit scale/min
 * pairs, 32 high-bit bytes, 128 nibble bytes).  The lane mapping and the
 * accumulation order follow the q4_K case of qwen4_row_dot: group = lane/4,
 * l = (lane%4)*8, eight consecutive elements per lane per block.  Element j
 * of group g takes its fifth bit from bit g of qh[j]. */
static inline float qwen35_row_dot(device const char *row, device const float *x,
                                   uint weight_type, uint in_dim, ushort tiisg) {
    if (weight_type != 13u) return qwen4_row_dot(row, x, weight_type, in_dim, tiisg);
    float acc = 0.0f;
    const uint nb = in_dim / 256u;
    const uint group = tiisg / 4, l = (tiisg % 4) * 8;
    for (uint ib = 0; ib < nb; ib++) {
        device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 176);
        const float d = (float)(*(device const half *)blk);
        const float dmin = (float)(*(device const half *)(blk + 2));
        device const uchar *sc = blk + 4;
        uint s, mn;
        if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
        else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
        const float ds = d * (float)s, dm = dmin * (float)mn;
        device const uchar *qh = blk + 16 + l;
        device const uchar *qs = blk + 48 + (group >> 1) * 32 + l;
        const uint shift = (group & 1u) * 4u;
        device const float *y = x + ib * 256 + group * 32 + l;
        for (uint i = 0; i < 8; i++) {
            const uint q = ((qs[i] >> shift) & 0xFu) | (((qh[i] >> group) & 1u) << 4);
            acc += (ds * (float)q - dm) * y[i];
        }
    }
    return simd_sum(acc);
}

/* kernel_qwen4_moe_mid with qwen35_row_dot: mid[t][s][r] =
 * silu(gate_row . x) * (up_row . x); slot n_slots (when has_shared) is the
 * shared expert from its own bases.  Two rows per SIMD group. */
kernel void kernel_qwen35_moe_mid(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_gate,
        device const char    *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * row_bytes;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[((uint64_t)tok * n_out + slot) * args.out_rows + r] = qwen4_silu(g) * u;
    }
}

/* kernel_qwen4_moe_down with qwen35_row_dot. */
kernel void kernel_qwen35_moe_down(
        constant ds4_metal_args_qwen4_moe & args,
        device const char    *down_base,
        device const int32_t *selected,   /* [T][n_slots] */
        device const float   *mid,        /* [T][n_slots+has_shared][in_dim] */
        device float         *part,       /* [T][n_slots+has_shared][out_rows] */
        device const char    *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint nr = 2u;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * nr;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint row_bytes = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + nr && r < args.out_rows; r++) {
        const float v = qwen35_row_dot(db + ebase + (uint64_t)r * row_bytes, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}

/* Q5_K decode: NR rows per SIMD group, each row's per-lane dot order identical
 * to the M1 2-row kernel, so output is bit-for-bit the same. */
template <uint NR>
kernel void kernel_qwen35_moe_mid_q5k_nr(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *gate_base, device const char *up_base,
        device const int32_t *selected, device const float *x, device float *mid,
        device const char *sh_gate, device const char *sh_up,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *gb = shared ? sh_gate : gate_base;
    device const char *ub = shared ? sh_up : up_base;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *xt = x + (uint64_t)tok * args.in_dim;
    const uint64_t mb = ((uint64_t)tok * n_out + slot) * args.out_rows;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        const uint64_t off = ebase + (uint64_t)r * rb;
        const float g = qwen35_row_dot(gb + off, xt, type, args.in_dim, tiisg);
        const float u = qwen35_row_dot(ub + off, xt, type, args.in_dim, tiisg);
        if (tiisg == 0) mid[mb + r] = qwen4_silu(g) * u;
    }
}
template <uint NR>
kernel void kernel_qwen35_moe_down_q5k_nr(
        constant ds4_metal_args_qwen4_moe & args,
        device const char *down_base, device const int32_t *selected,
        device const float *mid, device float *part, device const char *sh_down,
        uint3 tgpig [[threadgroup_position_in_grid]], ushort3 ntg [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y, tok = tgpig.z;
    const uint n_out = args.n_slots + args.has_shared;
    const uint row0 = (tgpig.x * (ntg.x / 32u) + (uint)sgitg) * NR;
    if (row0 >= args.out_rows || slot >= n_out || tok >= args.n_tokens) return;
    const bool shared = slot == args.n_slots;
    const uint type = shared ? args.shared_type : args.weight_type;
    const uint rb = shared ? args.shared_row_bytes : args.row_bytes;
    device const char *db = shared ? sh_down : down_base;
    const uint64_t pair = (uint64_t)tok * n_out + slot;
    const uint64_t ebase = shared ? 0 : (uint64_t)(uint)selected[(uint64_t)tok * args.n_slots + slot] * args.expert_bytes;
    device const float *m = mid + pair * args.in_dim;
    for (uint r = row0; r < row0 + NR && r < args.out_rows; r++) {
        const float v = qwen35_row_dot(db + ebase + (uint64_t)r * rb, m, type, args.in_dim, tiisg);
        if (tiisg == 0) part[pair * args.out_rows + r] = v;
    }
}
template [[host_name("kernel_qwen35_moe_mid_q5k_nr1")]] kernel void kernel_qwen35_moe_mid_q5k_nr<1>(constant ds4_metal_args_qwen4_moe &, device const char *, device const char *, device const int32_t *, device const float *, device float *, device const char *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mid_q5k_nr4")]] kernel void kernel_qwen35_moe_mid_q5k_nr<4>(constant ds4_metal_args_qwen4_moe &, device const char *, device const char *, device const int32_t *, device const float *, device float *, device const char *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_down_q5k_nr1")]] kernel void kernel_qwen35_moe_down_q5k_nr<1>(constant ds4_metal_args_qwen4_moe &, device const char *, device const int32_t *, device const float *, device float *, device const char *, uint3, ushort3, ushort, ushort);
template [[host_name("kernel_qwen35_moe_down_q5k_nr4")]] kernel void kernel_qwen35_moe_down_q5k_nr<4>(constant ds4_metal_args_qwen4_moe &, device const char *, device const int32_t *, device const float *, device float *, device const char *, uint3, ushort3, ushort, ushort);

/* --- Q5_K tiled prefill GEMM (additive; qwen4.metal untouched) ----------- */

/* Stage 16 consecutive Q5_K values (quarters q0 and q0+1 of 32-block b, q0
 * even) as halves.  Same coordinate as qwen4_mm_stage16's Q4_K branch:
 * sb = b/8, group = b%8, l = q0*8.  Q5_K super-block is 176 bytes: d, dmin,
 * 12 packed 6-bit scale/min bytes, 32 high-bit bytes, 128 nibble bytes.  The
 * 6-bit scale/min unpack is identical to Q4_K; the value is the low nibble at
 * qs plus the group-th bit of qh as the fifth bit. */
template <typename D>
static inline void qwen35_mm_stage16_q5k(device const char *row, uint b, uint q0, uint type, threadgroup D *dst) {
    (void)type;
    const uint sb = b / 8, group = b % 8;
    device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 176);
    const float d = (float)(*(device const half *)blk);
    const float dmin = (float)(*(device const half *)(blk + 2));
    device const uchar *sc = blk + 4;
    uint s, mn;
    if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
    else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
    const float ds = d * (float)s, dm = dmin * (float)mn;
    const uint4 v = *(device const uint4 *)(blk + 48 + (group >> 1) * 32 + q0 * 8);   /* 16 nibble bytes */
    const uint4 h = *(device const uint4 *)(blk + 16 + q0 * 8);                        /* 16 high-bit bytes */
    const uint shift = (group & 1u) * 4u;
    for (uint i = 0; i < 16; i++) {
        const uint lo = (v[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu;
        const uint hi = (h[i >> 2] >> (8u * (i & 3u) + group)) & 1u;
        dst[i] = (D)(ds * (float)(lo | (hi << 4)) - dm);
    }
}

/* Register prefetch for the tensor-op tiles: 16-byte header (d|dmin|scales),
 * the 16 nibble bytes and the 16 high-bit bytes of one 32-block quarter-pair. */
struct qwen35_raw_q5k { uint4 hdr; uint4 q; uint4 qh; };
static inline qwen35_raw_q5k qwen35_load_raw_q5k(device const char *row, uint b, uint q0, uint type) {
    (void)type;
    const uint sb = b / 8, group = b % 8;
    device const uchar *blk = (device const uchar *)(row + (uint64_t)sb * 176);
    qwen35_raw_q5k r;
    r.hdr = *(device const uint4 *)blk;
    r.q   = *(device const uint4 *)(blk + 48 + (group >> 1) * 32 + q0 * 8);
    r.qh  = *(device const uint4 *)(blk + 16 + q0 * 8);
    return r;
}
static inline void qwen35_dequant_raw_q5k(qwen35_raw_q5k r, uint b, uint q0, uint type, threadgroup half *dst) {
    (void)q0; (void)type;
    const uint group = b % 8;
    const float d = (float)as_type<half>((ushort)(r.hdr.x & 0xFFFFu));
    const float dmin = (float)as_type<half>((ushort)(r.hdr.x >> 16));
    const uint scw[3] = { r.hdr.y, r.hdr.z, r.hdr.w };
#define QWEN35_SCB(i) ((scw[(i) >> 2] >> (8u * ((i) & 3u))) & 0xFFu)
    uint s, mn;
    if (group < 4) { s = QWEN35_SCB(group) & 63u; mn = QWEN35_SCB(group + 4) & 63u; }
    else { s = (QWEN35_SCB(group + 4) & 0xFu) | ((QWEN35_SCB(group - 4) & 0xC0u) >> 2); mn = (QWEN35_SCB(group + 4) >> 4) | ((QWEN35_SCB(group) & 0xC0u) >> 2); }
#undef QWEN35_SCB
    const float ds = d * (float)s, dm = dmin * (float)mn;
    const uint shift = (group & 1u) * 4u;
    for (uint i = 0; i < 16; i++) {
        const uint lo = (r.q[i >> 2] >> (8u * (i & 3u) + shift)) & 0xFu;
        const uint hi = (r.qh[i >> 2] >> (8u * (i & 3u) + group)) & 1u;
        dst[i] = (half)(ds * (float)(lo | (hi << 4)) - dm);
    }
}

/* --- Gated DeltaNet output ---------------------------------------------- */

/* Qwen3.5 RMSNormGated: per-head RMSNorm of the scan output, scaled by
 * ssm_norm and gated by silu(z).  Qwen3.8 gates with sigmoid
 * (kernel_qwen4_gdn_out); the arithmetic is otherwise the same. */
kernel void kernel_qwen35_gdn_out(
        constant ds4_metal_args_qwen4_gdn_out & args,
        device float       *o,        /* [T][H*D], in place */
        device const float *z,        /* [T][H*D] */
        device const float *weight,   /* [D] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint D = args.head_dim;
    const uint npt = D / 32;
    const uint64_t base = ((uint64_t)tok * args.n_head + h) * D + tiisg * npt;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += o[base + i] * o[base + i];
    ss = simd_sum(ss);
    const float r = rsqrt(ss / (float)D + args.eps);
    for (uint i = 0; i < npt; i++) {
        o[base + i] = o[base + i] * r * weight[tiisg * npt + i] * qwen4_silu(z[base + i]);
    }
}

/* --- MTP input ------------------------------------------------------------ */

struct ds4_metal_args_qwen35_mtp_concat {
    uint32_t n_embd;
    uint32_t n_tokens;
    float    eps;
};

/* The MTP block's input row: [RMSNorm(e) * enorm | RMSNorm(h) * hnorm],
 * embedding half first, as llama.cpp's qwen35moe graph_mtp concatenates
 * them.  One threadgroup of 256 threads per (half, token). */
kernel void kernel_qwen35_mtp_concat(
        constant ds4_metal_args_qwen35_mtp_concat & args,
        device float       *cat,     /* [T][2E] */
        device const float *e,       /* [T][E] token embeddings */
        device const float *h,       /* [T][E] trunk hidden, post output_norm */
        device const float *enorm,   /* [E] */
        device const float *hnorm,   /* [E] */
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint half_ = tgpig.x;
    const uint tok = tgpig.y;
    if (half_ > 1u || tok >= args.n_tokens) return;
    const uint E = args.n_embd;
    device const float *x = (half_ ? h : e) + (uint64_t)tok * E;
    device const float *w = half_ ? hnorm : enorm;
    device float *y = cat + (uint64_t)tok * 2u * E + half_ * E;
    threadgroup float partial[8];
    float ss = 0.0f;
    for (uint i = tid; i < E; i += 256u) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (tiisg == 0) partial[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8u; s++) total += partial[s];
    const float r = rsqrt(total / (float)E + args.eps);
    for (uint i = tid; i < E; i += 256u) y[i] = x[i] * r * w[i];
}

// BEGIN GENERATED q5_K tiles (extract_q5k_tiles.py) — do not edit by hand
/* mid[t][slot][r] = silu(gate . x) * (up . x) for every (token, slot) routed
 * to expert e, as 32-row x (8*NT)-token tiles: A (weights, half) and B
 * (activations, half) are staged in threadgroup memory per 64-wide K step
 * and multiplied with simdgroup matrices into float accumulators, so each
 * expert row is read once per token tile. */
template <uint NT>
kernel void kernel_qwen35_moe_mm_mid_q5k(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const float   *x,          /* [T][in_dim] */
        device float         *mid,        /* [T][n_out][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TT = QWEN4_MM_TOKS * NT;
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    threadgroup half Ag[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Au[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Bs[QWEN4_MM_KS * TT];
    threadgroup float Cs[4][2][64];
    device const char *gbase = gate_base + (uint64_t)e * args.expert_bytes;
    device const char *ubase = up_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / QWEN4_MM_KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 Cg[NT], Cu[NT];
        for (uint nt = 0; nt < NT; nt++) {
            Cg[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            Cu[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        const uint my_tok = tid % TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint my_t = my_pair >= 0 ? (uint)my_pair / args.n_slots : 0;
        for (uint kb = 0; kb < nk; kb++) {
            /* A: 32 rows x 64 k; thread = (row, 16-wide slice) */
            {
                const uint r = tid / 4, q = tid % 4;   /* q: 16-value half of one of the two 32-blocks */
                threadgroup half *dg = Ag + r * QWEN4_MM_KS + q * 16;
                threadgroup half *du = Au + r * QWEN4_MM_KS + q * 16;
                if (row0 + r < args.out_rows) {
                    device const char *grow = gbase + (uint64_t)(row0 + r) * args.row_bytes;
                    device const char *urow = ubase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * 2 + (q >> 1), quarter0 = (q & 1) * 2;
                    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
                    qwen35_mm_stage16_q5k(grow, b, quarter0, type, dg);
                    qwen35_mm_stage16_q5k(urow, b, quarter0, type, du);
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
            }
            /* B: 64 k x TT tokens; distribute all K values over 128 threads. */
            {
                const uint tok = tid % TT, kq = tid / TT;
                constexpr uint K_PER_THREAD = QWEN4_MM_KS * TT / 128;
                device const float *xr = x + (uint64_t)my_t * args.in_dim + kb * QWEN4_MM_KS + kq * K_PER_THREAD;
                for (uint j = 0; j < K_PER_THREAD; j += 4) {
                    const float4 v = my_pair >= 0 ? *(device const float4 *)(xr + j) : float4(0.0f);
                    Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = (half)v.x;
                    Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = (half)v.y;
                    Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = (half)v.z;
                    Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = (half)v.w;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < QWEN4_MM_KS / 8; sub++) {
                simdgroup_half8x8 ag, au, b;
                simdgroup_load(ag, Ag + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                simdgroup_load(au, Au + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * TT + nt * 8, TT, 0, false);
                    simdgroup_multiply_accumulate(Cg[nt], ag, b, Cg[nt]);
                    simdgroup_multiply_accumulate(Cu[nt], au, b, Cu[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(Cg[nt], Cs[sgitg][0], 8, 0, false);
            simdgroup_store(Cu[nt], Cs[sgitg][1], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 4 * 64; idx += 128) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                const float g = Cs[sg][0][el], u = Cs[sg][1][el];
                mid[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = qwen4_silu(g) * u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nt1")]]
kernel void kernel_qwen35_moe_mm_mid_q5k<1>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nt2")]]
kernel void kernel_qwen35_moe_mm_mid_q5k<2>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen35_moe_mm_mid_q5k")]]
kernel void kernel_qwen35_moe_mm_mid_q5k<4>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

/* 64-token tiles: each decoded weight tile serves twice the tokens (8 KB of
 * activations staged per K step); remainders take the 8/16/32-token kernels. */
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nt8")]]
kernel void kernel_qwen35_moe_mm_mid_q5k<8>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

/* part[t][slot][r] = down . mid[t][slot], same tiling with mid as B */
template <uint NT>
kernel void kernel_qwen35_moe_mm_down_q5k(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *down_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const float   *midv,       /* [T][n_out][in_dim] */
        device float         *part,       /* [T][n_out][out_rows] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TT = QWEN4_MM_TOKS * NT;
    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 8u ? 8u : remainder <= 16u ? 16u : remainder <= 32u ? 32u : 64u;
        if (TT < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != TT) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < TT) {
            work_count = count - remainder;
        }
    }
    threadgroup half As[QWEN4_MM_ROWS * QWEN4_MM_KS];
    threadgroup half Bs[QWEN4_MM_KS * TT];
    threadgroup float Cs[4][64];
    device const char *dbase = down_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * QWEN4_MM_ROWS;
    const uint nk = args.in_dim / QWEN4_MM_KS;
    for (uint tile = block.y; tile * TT < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * TT;
        const uint n_tile = min((uint)TT, work_count - tile * TT);
        simdgroup_float8x8 C[NT];
        for (uint nt = 0; nt < NT; nt++) C[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        const uint my_tok = tid % TT;
        const int my_pair = my_tok < n_tile ? list[t0 + my_tok] : -1;
        const uint64_t my_row = my_pair >= 0 ?
            ((uint64_t)((uint)my_pair / args.n_slots) * args.n_out + (uint)my_pair % args.n_slots) : 0;
        for (uint kb = 0; kb < nk; kb++) {
            {
                const uint r = tid / 4, q = tid % 4;
                threadgroup half *dd = As + r * QWEN4_MM_KS + q * 16;
                if (row0 + r < args.out_rows) {
                    device const char *drow = dbase + (uint64_t)(row0 + r) * args.row_bytes;
                    const uint b = kb * 2 + (q >> 1), quarter0 = (q & 1) * 2;
                    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
                    qwen35_mm_stage16_q5k(drow, b, quarter0, type, dd);
                } else {
                    for (uint i = 0; i < 16; i++) dd[i] = 0.0h;
                }
            }
            {
                const uint tok = tid % TT, kq = tid / TT;
                constexpr uint K_PER_THREAD = QWEN4_MM_KS * TT / 128;
                device const float *mr = midv + my_row * args.in_dim + kb * QWEN4_MM_KS + kq * K_PER_THREAD;
                for (uint j = 0; j < K_PER_THREAD; j += 4) {
                    const float4 v = my_pair >= 0 ? *(device const float4 *)(mr + j) : float4(0.0f);
                    Bs[(kq * K_PER_THREAD + j + 0) * TT + tok] = (half)v.x;
                    Bs[(kq * K_PER_THREAD + j + 1) * TT + tok] = (half)v.y;
                    Bs[(kq * K_PER_THREAD + j + 2) * TT + tok] = (half)v.z;
                    Bs[(kq * K_PER_THREAD + j + 3) * TT + tok] = (half)v.w;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint sub = 0; sub < QWEN4_MM_KS / 8; sub++) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, As + (sgitg * 8) * QWEN4_MM_KS + sub * 8, QWEN4_MM_KS, 0, false);
                for (uint nt = 0; nt < NT; nt++) {
                    simdgroup_load(b, Bs + sub * 8 * TT + nt * 8, TT, 0, false);
                    simdgroup_multiply_accumulate(C[nt], a, b, C[nt]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint nt = 0; nt < NT; nt++) {
            simdgroup_store(C[nt], Cs[sgitg], 8, 0, false);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 4 * 64; idx += 128) {
                const uint sg = idx / 64, el = idx % 64, r = el / 8, tok = nt * 8 + el % 8;
                const uint row = row0 + sg * 8 + r;
                if (tok >= n_tile || row >= args.out_rows) continue;
                const int pair = list[t0 + tok];
                const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
                part[((uint64_t)t * args.n_out + slot) * args.out_rows + row] = Cs[sg][el];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template [[host_name("kernel_qwen35_moe_mm_down_q5k_nt1")]]
kernel void kernel_qwen35_moe_mm_down_q5k<1>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen35_moe_mm_down_q5k_nt2")]]
kernel void kernel_qwen35_moe_mm_down_q5k<2>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen35_moe_mm_down_q5k")]]
kernel void kernel_qwen35_moe_mm_down_q5k<4>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

template [[host_name("kernel_qwen35_moe_mm_down_q5k_nt8")]]
kernel void kernel_qwen35_moe_mm_down_q5k<8>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const float *, device float *, uint3, ushort, ushort);

#ifdef DS4_METAL_HAS_TENSOR
template <int NR1, typename XT, bool COMP>
kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *gate_base,
        device const char    *up_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const XT      *x,          /* [T][in_dim] */
        device float         *mid,
        device half          *midh,       /* [T][n_out][out_rows], the down tiles' operand */
        device half          *midr,       /* [T][n_out][out_rows] residual of midh (COMP) */
        threadgroup char     *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NK = 32;
    constexpr int NB = NR1 * 4 / 128;   /* B staging items per thread (token, 8-wide k slice) */
    using BT = typename qwen4_nax_btype<COMP, XT>::type;
    uint rb, tile0;
    if (args.expert_major) { const uint n_rb = (args.out_rows + NR0 - 1u) / NR0; rb = tgpig.x % n_rb; tile0 = tgpig.x / n_rb; }
    else { rb = tgpig.x; tile0 = tgpig.z; }
    const uint e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    /* tails: with tail_base 64 the 64-token tiles keep the full tiles and the
     * 32-token kernel takes a remainder of at most 32 tokens */
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 32u ? 32u : 64u;
        if ((uint)NR1 < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != (uint)NR1) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < (uint)NR1) {
            work_count = count - remainder;
        }
    }
    threadgroup half *Ag = (threadgroup half *)shmem;                 /* [64][32] */
    threadgroup half *Au = (threadgroup half *)(shmem + 4096);        /* [64][32] */
    threadgroup BT *Bs = (threadgroup BT *)(shmem + 8192);            /* [NR1][32] */
    threadgroup half *Br = (threadgroup half *)(shmem + 8192 + NR1 * 64); /* [NR1][32] residual (COMP) */
    threadgroup float *Cs = (threadgroup float *)shmem;               /* [NR1 tok][64 row] after the K loop */
    device const char *gbase = gate_base + (uint64_t)e * args.expert_bytes;
    device const char *ubase = up_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
    auto tA_g = tensor(Ag, dextents<int32_t, 2>(NK, NR0));
    auto tA_u = tensor(Au, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));   /* left operand: k contiguous, one token per column */
    auto tBr = tensor(Br, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;       /* A staging: (row, 16-wide half of the 32-block) */
    for (uint tile = tile0; tile * NR1 < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * NR1;
        const uint n_tile = min((uint)NR1, work_count - tile * NR1);
        auto cG = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_g), float>();
        auto cU = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA_u), float>();
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = 0.0f; }
#pragma unroll
        for (uint16_t i = 0; i < cU.get_capacity(); ++i) { if (cU.is_valid_element(i)) cU[i] = 0.0f; }
        device const XT *xr[NB];
        threadgroup BT *bdst[NB];
        threadgroup half *rdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            xr[b] = pair >= 0 ? x + (uint64_t)((uint)pair / args.n_slots) * args.in_dim + kq * 8u : (device const XT *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
            if constexpr (COMP) rdst[b] = Br + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   /* previous tile's C tile consumed */
        const bool a_row = row0 + ar < args.out_rows;
        device const char *grow = gbase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        device const char *urow = ubase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qwen35_raw_q5k rg = qwen35_load_raw_q5k(grow, 0, aq * 2, type), ru = qwen35_load_raw_q5k(urow, 0, aq * 2, type);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dg = Ag + ar * NK + aq * 16;
                threadgroup half *du = Au + ar * NK + aq * 16;
                if (a_row) {
                    qwen35_dequant_raw_q5k(rg, kb, aq * 2, type, dg);
                    qwen35_dequant_raw_q5k(ru, kb, aq * 2, type, du);
                } else {
                    for (uint i = 0; i < 16; i++) { dg[i] = 0.0h; du[i] = 0.0h; }
                }
                if (kb + 1 < nk) { rg = qwen35_load_raw_q5k(grow, kb + 1, aq * 2, type); ru = qwen35_load_raw_q5k(urow, kb + 1, aq * 2, type); }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if constexpr (COMP) {
                    /* stage xh and the residual xr: x = xh + xr to ~2^-22 relative */
                    if (xr[b]) {
                        const float4 v0 = *(device const float4 *)(xr[b] + kb * NK);
                        const float4 v1 = *(device const float4 *)(xr[b] + kb * NK + 4);
                        const half4 h0 = half4(v0), h1 = half4(v1);
                        *(threadgroup uint2 *)bdst[b] = as_type<uint2>(h0);
                        *(threadgroup uint2 *)(bdst[b] + 4) = as_type<uint2>(h1);
                        *(threadgroup uint2 *)rdst[b] = as_type<uint2>(half4(v0 - float4(h0)));
                        *(threadgroup uint2 *)(rdst[b] + 4) = as_type<uint2>(half4(v1 - float4(h1)));
                    } else {
                        *(threadgroup uint2 *)bdst[b] = uint2(0u);
                        *(threadgroup uint2 *)(bdst[b] + 4) = uint2(0u);
                        *(threadgroup uint2 *)rdst[b] = uint2(0u);
                        *(threadgroup uint2 *)(rdst[b] + 4) = uint2(0u);
                    }
                } else {
                    qwen4_nax_stage8(bdst[b], xr[b] ? xr[b] + kb * NK : (device const XT *)0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mAg = tA_g.slice(0, 0);
                auto mAu = tA_u.slice(0, 0);
                mm.run(mB, mAg, cG);
                mm.run(mB, mAu, cU);
                if constexpr (COMP) {
                    auto mBr = tBr.slice(0, 0);
                    mm.run(mBr, mAg, cG);
                    mm.run(mBr, mAu, cU);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
#pragma unroll
        for (uint16_t i = 0; i < cG.get_capacity(); ++i) { if (cG.is_valid_element(i)) cG[i] = qwen4_silu(cG[i]) * cU[i]; }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cG.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            const int pair = list[t0 + j];
            const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
            device float *out = mid + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
            device half *outh = nullptr, *outr = nullptr;
            for (uint i = tiisg; i < NR0 && row0 + i < args.out_rows; i += 32) {
                const float v = Cs[j * NR0 + i];
                out[i] = v;
                if constexpr (is_same<XT, half>::value || COMP) {
                    outh = midh + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
                    if constexpr (COMP) outr = midr + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
                }
                if (outh) {
                    const half h = (half)v;
                    outh[i] = h;
                    if constexpr (COMP) outr[i] = (half)(v - (float)h);
                }
            }
        }
    }
}

template <int NR1, typename XT, bool COMP>
kernel void kernel_qwen35_moe_mm_down_q5k_nax_t(
        constant ds4_metal_args_qwen4_moe_mm & args,
        device const char    *down_base,
        device const int32_t *lists,
        device const int32_t *counts,
        device const XT      *midv,       /* [T][n_out][in_dim] */
        device float         *part,
        device const half    *midr,       /* [T][n_out][in_dim] residual of midh (COMP) */
        threadgroup char     *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NK = 32;
    constexpr int NB = NR1 * 4 / 128;
    uint rb, tile0;
    if (args.expert_major) { const uint n_rb = (args.out_rows + NR0 - 1u) / NR0; rb = tgpig.x % n_rb; tile0 = tgpig.x / n_rb; }
    else { rb = tgpig.x; tile0 = tgpig.z; }
    const uint e = tgpig.y;
    if (e >= args.n_expert) return;
    const uint count = (uint)counts[e];
    /* tails: with tail_base 64 the 64-token tiles keep the full tiles and the
     * 32-token kernel takes a remainder of at most 32 tokens */
    uint work_count = count, work_start = 0;
    if (qwen4_moe_tail_base) {
        const uint remainder = count % qwen4_moe_tail_base;
        const uint tail_tt = remainder <= 32u ? 32u : 64u;
        if ((uint)NR1 < qwen4_moe_tail_base) {
            if (!remainder || tail_tt != (uint)NR1) return;
            work_start = count - remainder;
            work_count = remainder;
        } else if (remainder && tail_tt < (uint)NR1) {
            work_count = count - remainder;
        }
    }
    threadgroup half *As = (threadgroup half *)shmem;                 /* [64][32] */
    threadgroup XT *Bs = (threadgroup XT *)(shmem + 4096);            /* [NR1][32] */
    threadgroup half *Br = (threadgroup half *)(shmem + 4096 + NR1 * 64); /* [NR1][32] residual (COMP) */
    threadgroup float *Cs = (threadgroup float *)shmem;               /* [NR1 tok][64 row] after the K loop */
    device const char *dbase = down_base + (uint64_t)e * args.expert_bytes;
    device const int32_t *list = lists + (uint64_t)e * args.list_cap;
    const uint row0 = rb * NR0;
    const uint nk = args.in_dim / NK;
    const uint type = qwen4_moe_weight_type ? qwen4_moe_weight_type : args.weight_type;
    auto tA = tensor(As, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));   /* left operand: k contiguous, one token per column */
    auto tBr = tensor(Br, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    const uint ar = tid / 2, aq = tid % 2;
    for (uint tile = tile0; tile * NR1 < work_count; tile += args.tiles_per_launch) {
        const uint t0 = work_start + tile * NR1;
        const uint n_tile = min((uint)NR1, work_count - tile * NR1);
        auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
        device const XT *mr[NB];
        device const half *rr[NB];
        threadgroup XT *bdst[NB];
        threadgroup half *rdst[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
            const uint item = (uint)tid + (uint)b * 128u, tok = item / 4u, kq = item % 4u;
            const int pair = tok < n_tile ? list[t0 + tok] : -1;
            mr[b] = pair >= 0 ? midv + ((uint64_t)((uint)pair / args.n_slots) * args.n_out + (uint)pair % args.n_slots) * args.in_dim + kq * 8u
                              : (device const XT *)0;
            rr[b] = pair >= 0 ? midr + ((uint64_t)((uint)pair / args.n_slots) * args.n_out + (uint)pair % args.n_slots) * args.in_dim + kq * 8u
                              : (device const half *)0;
            bdst[b] = Bs + tok * NK + kq * 8u;
            if constexpr (COMP) rdst[b] = Br + tok * NK + kq * 8u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool a_row = row0 + ar < args.out_rows;
        device const char *drow = dbase + (uint64_t)(row0 + min(ar, args.out_rows - 1u)) * args.row_bytes;
        qwen35_raw_q5k rd = qwen35_load_raw_q5k(drow, 0, aq * 2, type);
        for (uint kb = 0; kb < nk; kb++) {
            {
                threadgroup half *dd = As + ar * NK + aq * 16;
                if (a_row) qwen35_dequant_raw_q5k(rd, kb, aq * 2, type, dd);
                else for (uint i = 0; i < 16; i++) dd[i] = 0.0h;
                if (kb + 1 < nk) rd = qwen35_load_raw_q5k(drow, kb + 1, aq * 2, type);
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if constexpr (COMP) {
                    /* stage the half operand and its residual separately */
                    *(threadgroup uint4 *)bdst[b] = mr[b] ? *(device const uint4 *)(mr[b] + kb * NK) : uint4(0u);
                    *(threadgroup uint4 *)rdst[b] = rr[b] ? *(device const uint4 *)(rr[b] + kb * NK) : uint4(0u);
                } else {
                    qwen4_nax_stage8(bdst[b], mr[b] ? mr[b] + kb * NK : (device const XT *)0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                auto mB = tB.slice(0, 0);
                auto mA = tA.slice(0, 0);
                mm.run(mB, mA, cT);
                if constexpr (COMP) {
                    auto mBr = tBr.slice(0, 0);
                    mm.run(mBr, mA, cT);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
            auto mC = tC.slice(0, 0);
            cT.store(mC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sgitg; j < n_tile; j += 4) {
            const int pair = list[t0 + j];
            const uint t = (uint)pair / args.n_slots, slot = (uint)pair % args.n_slots;
            device float *out = part + ((uint64_t)t * args.n_out + slot) * args.out_rows + row0;
            for (uint i = tiisg; i < NR0 && row0 + i < args.out_rows; i += 32) out[i] = Cs[j * NR0 + i];
        }
    }
}


template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_mid_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_mid_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device half *, device half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax")]]   kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<32, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
template [[host_name("kernel_qwen35_moe_mm_down_q5k_nax64")]] kernel void kernel_qwen35_moe_mm_down_q5k_nax_t<64, half, false>(constant ds4_metal_args_qwen4_moe_mm &, device const char *, device const int32_t *, device const int32_t *, device const half *, device float *, device const half *, threadgroup char *, uint3, ushort, ushort, ushort);
#endif
// END GENERATED q5_K tiles
