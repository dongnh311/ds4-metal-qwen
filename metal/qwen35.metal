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

/* --- L12: shared-KV decode (DS4_QWEN35_ATTN_DECODE2) --------------------- */

struct ds4_metal_args_qwen35_attn_decode2 {
    uint32_t n_head, n_head_kv, head_dim, pos0;
    uint32_t rows, split_keys, ns0, ns1;
    uint32_t fp8;
    float scale;
};

/* One threadgroup per (key split, kv head) serves every row in [0, args.rows)
 * -- 1 for a lone decode, 2 for the MTP verify's two rows (pos0, pos0+1) --
 * sharing each key's K/V load across the rows that both need it instead of
 * reading it once per row.  Row r's own split count (ns0/ns1, from
 * qwen4_attn_row_splits on that row's OWN key count, host side) and its
 * [lo,hi) boundary within a split never depend on the other row, so decode2
 * called with rows==1 for a given pos0 visits exactly the same keys in the
 * same order, with the same online-softmax arithmetic, as row 0 of a
 * decode2(rows==2) call at that same pos0: bit-identical by construction.
 * Mode 0 F16, 1 E4M3, 2 4-bit, exactly as qwen4_attn_decode_tile.  A row
 * whose own split count is 1 writes its gated output directly; a row with
 * more splits writes partials into its own compact block of `part`
 * ([Hkv][own n_splits][group][2+D], row 0's block first), consumed by a
 * plain single-row kernel_qwen4_attn_merge call (that row's own n_splits,
 * tok=0). */
template <uint NPT>
kernel void kernel_qwen35_attn_decode2(
        constant ds4_metal_args_qwen35_attn_decode2 & args,
        device const float   *q,          /* [rows][H*D] */
        device const float   *gate,       /* [rows][H*D] */
        device const half    *k_cache,
        device const half    *v_cache,
        device float         *out,        /* [rows][H*D] */
        device float         *part,       /* row0 block, then row1 block if rows==2 */
        device const uchar   *k_cache_fp8,
        device const uchar   *v_cache_fp8,
        device const half    *k_scale,
        device const half    *v_scale,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x, kvh = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    if (kvh >= Hkv) return;
    constexpr uint D = NPT * 32;
    const uint group = H / Hkv;
    const uint hps = (group + QWEN4_ATTN_NSG - 1) / QWEN4_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint rows = args.rows;
    const uint n0 = args.pos0 + 1u, n1 = args.pos0 + 2u;
    const uint ns0 = args.ns0, ns1 = args.ns1;
    const uint kps0 = (n0 + ns0 - 1u) / ns0;
    const uint kps1 = rows > 1u ? (n1 + ns1 - 1u) / ns1 : 0u;
    const bool v0 = split < ns0;
    const bool v1 = rows > 1u && split < ns1;
    if (!v0 && !v1) return;
    const uint lo0 = split * kps0, hi0 = v0 ? min(n0, lo0 + kps0) : lo0;
    const uint lo1 = split * kps1, hi1 = v1 ? min(n1, lo1 + kps1) : lo1;
    uint k0, k1;
    if (v0 && v1) { k0 = min(lo0, lo1); k1 = max(hi0, hi1); }
    else if (v0)  { k0 = lo0; k1 = hi0; }
    else          { k0 = lo1; k1 = hi1; }
    const uint fp8 = args.fp8;

    float qv[2][QWEN4_ATTN_HPS][NPT], m[2][QWEN4_ATTN_HPS], l[2][QWEN4_ATTN_HPS], acc[2][QWEN4_ATTN_HPS][NPT];
    for (uint r = 0; r < rows; r++) {
#pragma unroll
        for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
            const uint h = kvh * group + g0 + min(g, ng - 1u);
            device const float *qh = q + ((uint64_t)r * H + h) * D + tiisg * NPT;
#pragma unroll
            for (uint i = 0; i < NPT; i++) qv[r][g][i] = qh[i] * args.scale;
            m[r][g] = -3.0e38f;
            l[r][g] = 0.0f;
#pragma unroll
            for (uint i = 0; i < NPT; i++) acc[r][g][i] = 0.0f;
        }
    }
    for (uint idx = k0; idx < k1; idx++) {
        const uint64_t kvbase = ((uint64_t)idx * Hkv + kvh) * D + tiisg * NPT;
        float kv[NPT], vv[NPT];
        if (fp8 == 2u && NPT == 8u) {
            const uint64_t sidx = ((uint64_t)idx * Hkv + kvh) * (D / 64u) + (tiisg * NPT) / 64u;
            const float ksc = (float)k_scale[sidx], vsc = (float)v_scale[sidx];
            const uint kw = reinterpret_cast<device const uint *>(k_cache_fp8)[kvbase >> 3];
            const uint vw = reinterpret_cast<device const uint *>(v_cache_fp8)[kvbase >> 3];
#pragma unroll
            for (uint i = 0; i < NPT; i++) {
                kv[i] = (float)(half)((float)((int)((kw >> (4u * i)) & 15u) - 8) * ksc);
                vv[i] = (float)(half)((float)((int)((vw >> (4u * i)) & 15u) - 8) * vsc);
            }
        } else if (fp8) {
            const uint64_t sidx = ((uint64_t)idx * Hkv + kvh) * (D / 64u) + (tiisg * NPT) / 64u;
            const float ksc = (float)k_scale[sidx], vsc = (float)v_scale[sidx];
#pragma unroll
            for (uint i = 0; i < NPT; i++) {
                kv[i] = (float)(half)(dsv4_e4m3fn_decode(k_cache_fp8[kvbase + i]) * ksc);
                vv[i] = (float)(half)(dsv4_e4m3fn_decode(v_cache_fp8[kvbase + i]) * vsc);
            }
        } else {
            device const half *kr = k_cache + kvbase, *vr = v_cache + kvbase;
#pragma unroll
            for (uint i = 0; i < NPT; i++) { kv[i] = (float)kr[i]; vv[i] = (float)vr[i]; }
        }
        const bool in0 = v0 && idx >= lo0 && idx < hi0;
        const bool in1 = v1 && idx >= lo1 && idx < hi1;
        for (uint r = 0; r < rows; r++) {
            if ((r == 0u && !in0) || (r == 1u && !in1)) continue;
#pragma unroll
            for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
                if (g < ng) {
                    float s = 0.0f;
#pragma unroll
                    for (uint i = 0; i < NPT; i++) s += qv[r][g][i] * kv[i];
                    s = simd_sum(s);
                    const float mn = max(m[r][g], s), corr = exp(m[r][g] - mn), w = exp(s - mn);
                    l[r][g] = l[r][g] * corr + w;
#pragma unroll
                    for (uint i = 0; i < NPT; i++) acc[r][g][i] = acc[r][g][i] * corr + w * vv[i];
                    m[r][g] = mn;
                }
            }
        }
    }
    for (uint g = 0; g < QWEN4_ATTN_HPS; g++) {
        if (g >= ng) break;
        const uint h = kvh * group + g0 + g;
        if (v0) {
            if (ns0 == 1u) {
                device float *dst = out + h * D + tiisg * NPT;
                device const float *gt = gate + h * D + tiisg * NPT;
                const float inv = l[0][g] > 0.0f ? 1.0f / l[0][g] : 0.0f;
#pragma unroll
                for (uint i = 0; i < NPT; i++) dst[i] = acc[0][g][i] * inv * qwen4_sigmoid(gt[i]);
            } else {
                device float *dst = part + (((uint64_t)kvh * ns0 + split) * group + g0 + g) * (2u + D);
                if (tiisg == 0) { dst[0] = m[0][g]; dst[1] = l[0][g]; }
#pragma unroll
                for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = acc[0][g][i];
            }
        }
        if (v1) {
            if (ns1 == 1u) {
                device float *dst = out + ((uint64_t)1u * H + h) * D + tiisg * NPT;
                device const float *gt = gate + ((uint64_t)1u * H + h) * D + tiisg * NPT;
                const float inv = l[1][g] > 0.0f ? 1.0f / l[1][g] : 0.0f;
#pragma unroll
                for (uint i = 0; i < NPT; i++) dst[i] = acc[1][g][i] * inv * qwen4_sigmoid(gt[i]);
            } else {
                device float *dst = part + ((uint64_t)ns0 * Hkv * group +
                                            (((uint64_t)kvh * ns1 + split) * group + g0 + g)) * (2u + D);
                if (tiisg == 0) { dst[0] = m[1][g]; dst[1] = l[1][g]; }
#pragma unroll
                for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = acc[1][g][i];
            }
        }
    }
}
template [[host_name("kernel_qwen35_attn_decode2_npt8")]]
kernel void kernel_qwen35_attn_decode2<8>(
        constant ds4_metal_args_qwen35_attn_decode2 &, device const float *, device const float *,
        device const half *, device const half *, device float *, device float *,
        device const uchar *, device const uchar *, device const half *, device const half *,
        uint3, ushort, ushort);

/* --- M5: decode3, decode2's successor (more splits, keys across simdgroups,
 * parallel merge) ---------------------------------------------------------- */

/* Must match QWEN35_ATTN_MAX_SPLITS in ds4_metal.m (the host cap on a row's
 * own split count) and QWEN4_ATTN_NSG (the simdgroups-per-threadgroup used
 * for the dispatch's threadgroup size). */
#define QWEN35_ATTN_MAX_SPLITS 256u
#define QWEN35_ATTN_GROUP_MAX 8u   /* Ornith: 16 query heads / 2 KV heads */

struct ds4_metal_args_qwen35_attn_decode3 {
    uint32_t n_head, n_head_kv, head_dim, pos0;
    uint32_t rows, ns0, ns1;
    float scale;
};

/* decode2's successor: instead of dividing the KV head's query-head group
 * across the threadgroup's 4 simdgroups (decode2), every simdgroup here
 * serves the WHOLE group (up to 8 heads) and instead divides the row's OWN
 * key range across the 4 simdgroups, so a 128-thread threadgroup reads keys
 * in parallel rather than scanning them serially.  Row r's own [lo,hi) split
 * geometry (from its own key count via qwen35_attn_row_splits3, host side)
 * and its key-to-simdgroup assignment (lo_r + sg + NSG*j, strided off THE
 * ROW'S OWN lo_r) depend only on that row's own key count, so a rows==2 call
 * visits exactly the keys, in exactly the order, a lone rows==1 call would
 * visit for each row: bit-identical by construction (both rows run this
 * exact same per-row code path, one after the other, so there is nothing
 * for the other row to perturb).  The NSG simdgroup partial states of a row
 * (for this split) are combined in threadgroup memory in the fixed order
 * sg=0,1,2,3 (the standard online-softmax merge), GCHUNK query heads at a
 * time to keep the threadgroup memory footprint under the 32KB per-
 * threadgroup limit.  A row whose own split count is 1 then writes its
 * gated output directly; otherwise the split's combined (m,l,acc) is written
 * to `part` ([Hkv][own n_splits][group][2+D], row 0's block first, row 1's
 * block starting after it) for kernel_qwen35_attn_merge3. */
template <uint NPT>
kernel void kernel_qwen35_attn_decode3(
        constant ds4_metal_args_qwen35_attn_decode3 & args,
        device const float   *q,          /* [rows][H*D] */
        device const float   *gate,       /* [rows][H*D] */
        device const half    *k_cache,
        device const half    *v_cache,
        device float         *out,        /* [rows][H*D] */
        device float         *part,       /* row0 block, then row1 block if rows==2 */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x, kvh = tgpig.y;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    if (kvh >= Hkv) return;
    constexpr uint D = NPT * 32;
    constexpr uint NSG = 4;
    constexpr uint GMAX = QWEN35_ATTN_GROUP_MAX;
    constexpr uint GCHUNK = 4u;
    const uint group = H / Hkv;
    const uint rows = args.rows;
    const uint n0 = args.pos0 + 1u, n1 = args.pos0 + 2u;
    const uint ns0 = args.ns0, ns1 = args.ns1;
    const uint kps0 = (n0 + ns0 - 1u) / ns0;
    const uint kps1 = rows > 1u ? (n1 + ns1 - 1u) / ns1 : 0u;
    const bool v0 = split < ns0;
    const bool v1 = rows > 1u && split < ns1;
    if (!v0 && !v1) return;
    const uint lo0 = split * kps0, hi0 = v0 ? min(n0, lo0 + kps0) : lo0;
    const uint lo1 = split * kps1, hi1 = v1 ? min(n1, lo1 + kps1) : lo1;

    /* One thread's live state for a single row/head-chunk pass: keeping this
     * to one row at a time (instead of tracking both rows' [8] head arrays
     * live throughout the whole key scan) roughly halves live registers,
     * which is what actually bounded occupancy here -- not the merge. */
    threadgroup float tg_m[NSG][GCHUNK];
    threadgroup float tg_l[NSG][GCHUNK];
    threadgroup float tg_acc[NSG][GCHUNK][D];
    for (uint r = 0; r < rows; r++) {
        if ((r == 0u && !v0) || (r == 1u && !v1)) continue;
        const uint ns_r = r == 0u ? ns0 : ns1;
        const uint lo = r == 0u ? lo0 : lo1, hi = r == 0u ? hi0 : hi1;

        float qv[GMAX][NPT], m[GMAX], l[GMAX], acc[GMAX][NPT];
#pragma unroll
        for (uint g = 0; g < GMAX; g++) {
            if (g >= group) break;
            const uint h = kvh * group + g;
            device const float *qh = q + ((uint64_t)r * H + h) * D + tiisg * NPT;
#pragma unroll
            for (uint i = 0; i < NPT; i++) qv[g][i] = qh[i] * args.scale;
            m[g] = -3.0e38f;
            l[g] = 0.0f;
#pragma unroll
            for (uint i = 0; i < NPT; i++) acc[g][i] = 0.0f;
        }
        for (uint idx = lo + sgitg; idx < hi; idx += NSG) {
            const uint64_t kvbase = ((uint64_t)idx * Hkv + kvh) * D + tiisg * NPT;
            device const half *kr = k_cache + kvbase, *vr = v_cache + kvbase;
            float kv[NPT], vv[NPT];
#pragma unroll
            for (uint i = 0; i < NPT; i++) { kv[i] = (float)kr[i]; vv[i] = (float)vr[i]; }
#pragma unroll
            for (uint g = 0; g < GMAX; g++) {
                if (g >= group) break;
                float s = 0.0f;
#pragma unroll
                for (uint i = 0; i < NPT; i++) s += qv[g][i] * kv[i];
                s = simd_sum(s);   /* each thread holds NPT of the 256 dims: sum across the simdgroup's 32 lanes */
                const float mn = max(m[g], s), corr = exp(m[g] - mn), w = exp(s - mn);
                l[g] = l[g] * corr + w;
#pragma unroll
                for (uint i = 0; i < NPT; i++) acc[g][i] = acc[g][i] * corr + w * vv[i];
                m[g] = mn;
            }
        }

        /* One barrier round per chunk of GCHUNK heads (not per head): every
         * simdgroup's state for the chunk goes to threadgroup memory in one
         * shot, then the NSG partials are combined per head -- spread across
         * all NSG simdgroups (simdgroup sg merges chunk-relative heads
         * sg, sg+NSG, ...) instead of leaving 3 of 4 simdgroups idle. */
        for (uint g0 = 0; g0 < group; g0 += GCHUNK) {
            const uint gn = min(GCHUNK, group - g0);
            for (uint gc = 0; gc < gn; gc++) {
                const uint g = g0 + gc;
                if (tiisg == 0) { tg_m[sgitg][gc] = m[g]; tg_l[sgitg][gc] = l[g]; }
#pragma unroll
                for (uint i = 0; i < NPT; i++) tg_acc[sgitg][gc][tiisg * NPT + i] = acc[g][i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint gc = sgitg; gc < gn; gc += NSG) {
                const uint g = g0 + gc;
                float mm = -3.0e38f;
                for (uint s = 0; s < NSG; s++) mm = max(mm, tg_m[s][gc]);
                float ll = 0.0f, o[NPT];
#pragma unroll
                for (uint i = 0; i < NPT; i++) o[i] = 0.0f;
                for (uint s = 0; s < NSG; s++) {
                    const float c = tg_l[s][gc] > 0.0f ? exp(tg_m[s][gc] - mm) : 0.0f;
                    ll += tg_l[s][gc] * c;
#pragma unroll
                    for (uint i = 0; i < NPT; i++) o[i] += tg_acc[s][gc][tiisg * NPT + i] * c;
                }
                const uint h = kvh * group + g;
                if (ns_r == 1u) {
                    device float *dst = out + ((uint64_t)r * H + h) * D + tiisg * NPT;
                    device const float *gt = gate + ((uint64_t)r * H + h) * D + tiisg * NPT;
                    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
#pragma unroll
                    for (uint i = 0; i < NPT; i++) dst[i] = o[i] * inv * qwen4_sigmoid(gt[i]);
                } else {
                    const uint64_t row_base = (r == 0u ? 0u : (uint64_t)ns0 + (uint64_t)(r - 1u) * ns1) *
                                               H * (2u + D);
                    device float *dst = part + row_base + (((uint64_t)kvh * ns_r + split) * group + g) * (2u + D);
                    if (tiisg == 0) { dst[0] = mm; dst[1] = ll; }
#pragma unroll
                    for (uint i = 0; i < NPT; i++) dst[2u + tiisg * NPT + i] = o[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
template [[host_name("kernel_qwen35_attn_decode3_npt8")]]
kernel void kernel_qwen35_attn_decode3<8>(
        constant ds4_metal_args_qwen35_attn_decode3 &, device const float *, device const float *,
        device const half *, device const half *, device float *, device float *,
        uint3, ushort, ushort);

/* Parallel merge of decode3's split partials: one thread per head dim (256
 * threads, head_dim is always 256 here), one threadgroup per (head, row).
 * Folds the row's own ns_r partials with a fixed binary tree (pairs
 * (0,1),(2,3),... then pairs of pairs, padded with neutral elements up to
 * the next power of two): the shape depends only on ns_r, never on how many
 * were live at the source split's dispatch, so it matches decode3's own
 * direct-write path (ns_r==1) and any other ns_r bit for bit.  A neutral pad
 * slot has l==0 (m finite, no infinities -- this file builds under fast
 * math) so it contributes exactly zero to both l and the accumulator, ready
 * for a future caller that fills a real "no keys landed here" partial the
 * same way.  Rows whose own split count is 1 were written directly by
 * decode3 and are skipped here; the host dispatches this kernel only when at
 * least one row needs it. */
kernel void kernel_qwen35_attn_merge3(
        constant ds4_metal_args_qwen35_attn_decode3 & args,
        device const float *part,
        device const float *gate,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgpig.x, r = tgpig.y;
    if (h >= args.n_head || r >= args.rows) return;
    const uint ns = r == 0u ? args.ns0 : args.ns1;
    if (ns <= 1u) return;
    constexpr uint D = 256u;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t row_base = (r == 0u ? 0u : (uint64_t)args.ns0 + (uint64_t)(r - 1u) * args.ns1) * H * (2u + D);
    const uint64_t stride = (uint64_t)group * (2u + D);
    device const float *base = part + row_base + ((uint64_t)kvh * ns * group + g) * (2u + D);

    float mm[QWEN35_ATTN_MAX_SPLITS];
    float ll[QWEN35_ATTN_MAX_SPLITS];
    float oo[QWEN35_ATTN_MAX_SPLITS];
    uint p = 1u;
    while (p < ns) p <<= 1u;
    for (uint s = 0; s < p; s++) {
        if (s < ns) {
            device const float *ps = base + s * stride;
            mm[s] = ps[0]; ll[s] = ps[1]; oo[s] = ps[2u + tid];
        } else {
            mm[s] = -3.0e38f; ll[s] = 0.0f; oo[s] = 0.0f;   /* neutral pad */
        }
    }
    uint n = p;
    while (n > 1u) {
        const uint half_n = n >> 1u;
        for (uint i = 0; i < half_n; i++) {
            const uint s0 = 2u * i, s1 = 2u * i + 1u;
            const float m0 = mm[s0], m1 = mm[s1], l0 = ll[s0], l1 = ll[s1], o0 = oo[s0], o1 = oo[s1];
            const float nm = max(m0, m1);
            const float c0 = l0 > 0.0f ? exp(m0 - nm) : 0.0f;
            const float c1 = l1 > 0.0f ? exp(m1 - nm) : 0.0f;
            mm[i] = nm; ll[i] = l0 * c0 + l1 * c1; oo[i] = o0 * c0 + o1 * c1;
        }
        n = half_n;
    }
    const float inv = ll[0] > 0.0f ? 1.0f / ll[0] : 0.0f;
    const uint64_t at = ((uint64_t)r * H + h) * D + tid;
    out[at] = oo[0] * inv * qwen4_sigmoid(gate[at]);
}
