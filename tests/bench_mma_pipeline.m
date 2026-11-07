/* Spike for docs/Task/2026-11-07-qwen38-mma-pipeline-plan.md Phase 1.
 *
 * Standalone Metal microbenchmark comparing three ways to evaluate the
 * routed-MoE mid GEMM tile (C[32 tokens][64 rows] += W[64][2560] * X[2560][32],
 * K-loop in 32-wide steps):
 *
 *   v0  production-style: cooperative threadgroup staging + matmul2d
 *   v1  same staging, but simdgroup_matrix multiply_accumulate instead of
 *       matmul2d (isolates the MMA primitive)
 *   v2  simdgroup_matrix loading fragments straight from device memory,
 *       no threadgroup staging at all (upper bound of the no-barrier idea)
 *
 * fp16 operands, fp32 accumulate, matching the NAX tensor path's types.
 * Not wired into the Makefile: build with
 *   clang -O2 -fobjc-arc -framework Metal -framework Foundation \
 *       -o /tmp/bench_mma_pipeline tests/bench_mma_pipeline.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *kSource = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_simdgroup_matrix>
using namespace metal;
using namespace mpp::tensor_ops;

/* Problem: per expert, W[640 rows][2560 k] fp16, X[2560 k][T tokens] fp16.
 * Tile: 64 rows x 32 tokens, K stepped by NK=32 (80 steps).
 * Grid: x = row-blocks(10) * token-tiles, y = expert.
 * Buffers: W [E][640][2560], X [E][T][2560] (token-major for coalesced x),
 *          C [E][T][640] fp32. */

#define NR0 64
#define NR1 32
#define NK  32

/* ---- v0: threadgroup staging + matmul2d (production structure) ---- */
kernel void mma_v0(device const half *W, device const half *X, device float *C,
                   constant uint &Kdim, constant uint &Tdim,
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]]) {
    const uint e = tgpig.y;
    const uint nrb = 640 / NR0;
    const uint tt = Tdim / NR1;
    const uint rb = tgpig.x % nrb, tile = tgpig.x / nrb;
    if (e >= 8 || tile >= tt) return;
    threadgroup half Ag[NR0 * NK];
    threadgroup half Bs[NR1 * NK];   /* [token][k] physical */
    auto tA = tensor(Ag, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false,
                matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    device const half *w = W + ((uint64_t)e * 640 + rb * NR0) * Kdim;
    device const half *x = X + ((uint64_t)e * Tdim + tile * NR1) * Kdim;
    for (uint kb = 0; kb < Kdim / NK; kb++) {
        /* stage: 128 threads, Ag 64*32=2048, Bs 32*32=1024 halves */
        for (uint i = tid; i < NR0 * NK; i += 128) {
            const uint r = i / NK, k = i % NK;
            Ag[i] = w[(uint64_t)r * Kdim + kb * NK + k];
        }
        for (uint i = tid; i < NR1 * NK; i += 128) {
            const uint t = i / NK, k = i % NK;
            Bs[i] = x[(uint64_t)t * Kdim + kb * NK + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mB = tB.slice(0, 0);
        auto mA = tA.slice(0, 0);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float Cs[NR0 * NR1];
    {
        auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
        auto mC = tC.slice(0, 0);
        cT.store(mC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *c = C + ((uint64_t)e * Tdim + tile * NR1) * 640 + rb * NR0;
    for (uint i = tid; i < NR0 * NR1; i += 128) {
        const uint lt = i / NR0, r = i % NR0;
        c[(uint64_t)lt * 640 + r] = Cs[i];
    }
}

/* ---- v1: same staging, simdgroup_matrix mma ----
 * C[token][row]; per sim s: rows s*16..s*16+16 (2 row-tiles), all 32 tokens
 * (4 token-tiles) -> 8 accumulators.  Bs physical [token][k] is already the
 * left-operand layout [t][k]; Ag [row][k] loads transposed for [k][row]. */
kernel void mma_v1(device const half *W, device const half *X, device float *C,
                   constant uint &Kdim, constant uint &Tdim,
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint e = tgpig.y;
    const uint nrb = 640 / NR0;
    const uint tt = Tdim / NR1;
    const uint rb = tgpig.x % nrb, tile = tgpig.x / nrb;
    if (e >= 8 || tile >= tt) return;
    threadgroup half Ag[NR0 * NK];
    threadgroup half Bs[NR1 * NK];
    device const half *w = W + ((uint64_t)e * 640 + rb * NR0) * Kdim;
    device const half *x = X + ((uint64_t)e * Tdim) * Kdim;
    const uint t0 = tile * NR1;
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0; kb < Kdim / NK; kb++) {
        for (uint i = tid; i < NR0 * NK; i += 128) {
            const uint r = i / NK, k = i % NK;
            Ag[i] = w[(uint64_t)r * Kdim + kb * NK + k];
        }
        for (uint i = tid; i < NR1 * NK; i += 128) {
            const uint t = i / NK, k = i % NK;
            Bs[i] = x[(uint64_t)(t0 + t) * Kdim + kb * NK + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 4) {
            const uint r0 = sgitg * 16;
            simdgroup_half8x8 a[4], b[2];
            for (uint kk = 0; kk < NK / 8; kk++) {
                for (uint ct = 0; ct < 4; ct++)
                    simdgroup_load(a[ct], Bs + (ct * 8) * NK + kk * 8, NK, ulong2(0), false);
                for (uint rt = 0; rt < 2; rt++)
                    simdgroup_load(b[rt], Ag + (r0 + rt * 8) * NK + kk * 8, NK, ulong2(0), true);
                for (uint rt = 0; rt < 2; rt++)
                    for (uint ct = 0; ct < 4; ct++)
                        simdgroup_multiply_accumulate(mc[rt * 4 + ct], a[ct], b[rt], mc[rt * 4 + ct]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* store: mc[rt*4+ct] is [token 8][row 8] at (t0+ct*8, r0+rt*8) */
    threadgroup float Cs[NR0 * NR1];
    if (sgitg < 4) {
        const uint r0 = sgitg * 16;
        for (uint rt = 0; rt < 2; rt++)
            for (uint ct = 0; ct < 4; ct++)
                simdgroup_store(mc[rt * 4 + ct],
                                Cs + (ct * 8) * NR0 + r0 + rt * 8, NR0, ulong2(0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *c = C + ((uint64_t)e * Tdim + t0) * 640 + rb * NR0;
    for (uint i = tid; i < NR0 * NR1; i += 128) {
        const uint lt = i / NR0, r = i % NR0;
        c[(uint64_t)lt * 640 + r] = Cs[i];
    }
}

/* ---- v3: double-buffered staging + simdgroup mma, software pipeline ----
 * Same math as v1 but two Ag/Bs buffer sets.  Each K-step: load fragments
 * and issue mma for the staged step, then cooperatively stage the NEXT step
 * into the other buffer, one barrier per step.  The staging ALU work
 * overlaps the tensor-pipe mma that mm.run could never overlap. */
kernel void mma_v3(device const half *W, device const half *X, device float *C,
                   constant uint &Kdim, constant uint &Tdim,
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint e = tgpig.y;
    const uint nrb = 640 / NR0;
    const uint tt = Tdim / NR1;
    const uint rb = tgpig.x % nrb, tile = tgpig.x / nrb;
    if (e >= 8 || tile >= tt) return;
    threadgroup half Ag0[NR0 * NK], Ag1[NR0 * NK];
    threadgroup half Bs0[NR1 * NK], Bs1[NR1 * NK];
    device const half *w = W + ((uint64_t)e * 640 + rb * NR0) * Kdim;
    device const half *x = X + (uint64_t)e * Tdim * Kdim;
    const uint t0 = tile * NR1;
    const uint nksteps = Kdim / NK;

    /* prologue: stage step 0 into buffer set 0 */
    {
        for (uint i = tid; i < NR0 * NK; i += 128) {
            const uint r = i / NK, k = i % NK;
            Ag0[i] = w[(uint64_t)r * Kdim + k];
        }
        for (uint i = tid; i < NR1 * NK; i += 128) {
            const uint t = i / NK, k = i % NK;
            Bs0[i] = x[(uint64_t)(t0 + t) * Kdim + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0; kb < nksteps; kb++) {
        const uint cur = kb & 1;
        threadgroup half *Ag = cur ? Ag1 : Ag0;
        threadgroup half *Bs = cur ? Bs1 : Bs0;
        /* mma on the staged step (fragments -> registers) */
        if (sgitg < 4) {
            const uint r0 = sgitg * 16;
            simdgroup_half8x8 a[4], b[2];
            for (uint kk = 0; kk < NK / 8; kk++) {
                for (uint ct = 0; ct < 4; ct++)
                    simdgroup_load(a[ct], Bs + (ct * 8) * NK + kk * 8, NK, ulong2(0), false);
                for (uint rt = 0; rt < 2; rt++)
                    simdgroup_load(b[rt], Ag + (r0 + rt * 8) * NK + kk * 8, NK, ulong2(0), true);
                for (uint rt = 0; rt < 2; rt++)
                    for (uint ct = 0; ct < 4; ct++)
                        simdgroup_multiply_accumulate(mc[rt * 4 + ct], a[ct], b[rt], mc[rt * 4 + ct]);
            }
        }
        /* stage step kb+1 into the other buffer while mma is in flight */
        if (kb + 1 < nksteps) {
            const uint nb = kb + 1, dst = nb & 1;
            threadgroup half *dAg = dst ? Ag1 : Ag0;
            threadgroup half *dBs = dst ? Bs1 : Bs0;
            const uint kbase = nb * NK;
            for (uint i = tid; i < NR0 * NK; i += 128) {
                const uint r = i / NK, k = i % NK;
                dAg[i] = w[(uint64_t)r * Kdim + kbase + k];
            }
            for (uint i = tid; i < NR1 * NK; i += 128) {
                const uint t = i / NK, k = i % NK;
                dBs[i] = x[(uint64_t)(t0 + t) * Kdim + kbase + k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float Cs[NR0 * NR1];
    if (sgitg < 4) {
        const uint r0 = sgitg * 16;
        for (uint rt = 0; rt < 2; rt++)
            for (uint ct = 0; ct < 4; ct++)
                simdgroup_store(mc[rt * 4 + ct],
                                Cs + (ct * 8) * NR0 + r0 + rt * 8, NR0, ulong2(0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *c = C + ((uint64_t)e * Tdim + t0) * 640 + rb * NR0;
    for (uint i = tid; i < NR0 * NR1; i += 128) {
        const uint lt = i / NR0, r = i % NR0;
        c[(uint64_t)lt * 640 + r] = Cs[i];
    }
}

/* ---- v4: simdgroup mma, NK=64 K-steps (half the barriers of v1) ---- */
kernel void mma_v4(device const half *W, device const half *X, device float *C,
                   constant uint &Kdim, constant uint &Tdim,
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NK2 = 64;
    const uint e = tgpig.y;
    const uint nrb = 640 / NR0;
    const uint tt = Tdim / NR1;
    const uint rb = tgpig.x % nrb, tile = tgpig.x / nrb;
    if (e >= 8 || tile >= tt) return;
    threadgroup half Ag[NR0 * NK2];
    threadgroup half Bs[NR1 * NK2];
    device const half *w = W + ((uint64_t)e * 640 + rb * NR0) * Kdim;
    device const half *x = X + (uint64_t)e * Tdim * Kdim;
    const uint t0 = tile * NR1;
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0; kb < Kdim / NK2; kb++) {
        for (uint i = tid; i < NR0 * NK2; i += 128) {
            const uint r = i / NK2, k = i % NK2;
            Ag[i] = w[(uint64_t)r * Kdim + kb * NK2 + k];
        }
        for (uint i = tid; i < NR1 * NK2; i += 128) {
            const uint t = i / NK2, k = i % NK2;
            Bs[i] = x[(uint64_t)(t0 + t) * Kdim + kb * NK2 + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 4) {
            const uint r0 = sgitg * 16;
            simdgroup_half8x8 a[8], b[4];
            for (uint kk = 0; kk < NK2 / 8; kk++) {
                for (uint ct = 0; ct < 4; ct++)
                    simdgroup_load(a[ct], Bs + (ct * 8) * NK2 + kk * 8, NK2, ulong2(0), false);
                for (uint rt = 0; rt < 2; rt++)
                    simdgroup_load(b[rt], Ag + (r0 + rt * 8) * NK2 + kk * 8, NK2, ulong2(0), true);
                for (uint rt = 0; rt < 2; rt++)
                    for (uint ct = 0; ct < 4; ct++)
                        simdgroup_multiply_accumulate(mc[rt * 4 + ct], a[ct], b[rt], mc[rt * 4 + ct]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float Cs[NR0 * NR1];
    if (sgitg < 4) {
        const uint r0 = sgitg * 16;
        for (uint rt = 0; rt < 2; rt++)
            for (uint ct = 0; ct < 4; ct++)
                simdgroup_store(mc[rt * 4 + ct],
                                Cs + (ct * 8) * NR0 + r0 + rt * 8, NR0, ulong2(0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *c = C + ((uint64_t)e * Tdim + t0) * 640 + rb * NR0;
    for (uint i = tid; i < NR0 * NR1; i += 128) {
        const uint lt = i / NR0, r = i % NR0;
        c[(uint64_t)lt * 640 + r] = Cs[i];
    }
}

/* ---- v2: simdgroup_matrix, fragments straight from device memory ----
 * No threadgroup at all.  X is [token][k] so the left operand loads
 * directly; W is [row][k] so the right operand uses transpose loads.
 * Each simdgroup still owns rows sgitg*16..+16, all 32 tokens. */
kernel void mma_v2(device const half *W, device const half *X, device float *C,
                   constant uint &Kdim, constant uint &Tdim,
                   uint3 tgpig [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]],
                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint e = tgpig.y;
    const uint nrb = 640 / NR0;
    const uint tt = Tdim / NR1;
    const uint rb = tgpig.x % nrb, tile = tgpig.x / nrb;
    if (e >= 8 || tile >= tt) return;
    const uint t0 = tile * NR1;
    device const half *w = W + ((uint64_t)e * 640 + rb * NR0) * Kdim;
    device const half *x = X + (uint64_t)e * Tdim * Kdim;
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint kb = 0; kb < Kdim / NK; kb++) {
        if (sgitg < 4) {
            const uint r0 = sgitg * 16;
            simdgroup_half8x8 a[4], b[2];
            for (uint kk = 0; kk < NK / 8; kk++) {
                const uint k0 = kb * NK + kk * 8;
                for (uint ct = 0; ct < 4; ct++)
                    simdgroup_load(a[ct], x + (uint64_t)(t0 + ct * 8) * Kdim + k0,
                                   Kdim, ulong2(0), false);
                for (uint rt = 0; rt < 2; rt++)
                    simdgroup_load(b[rt], w + (uint64_t)(r0 + rt * 8) * Kdim + k0,
                                   Kdim, ulong2(0), true);
                for (uint rt = 0; rt < 2; rt++)
                    for (uint ct = 0; ct < 4; ct++)
                        simdgroup_multiply_accumulate(mc[rt * 4 + ct], a[ct], b[rt], mc[rt * 4 + ct]);
            }
        }
    }
    device float *c = C + ((uint64_t)e * Tdim + t0) * 640 + rb * NR0;
    if (sgitg < 4) {
        const uint r0 = sgitg * 16;
        for (uint rt = 0; rt < 2; rt++)
            for (uint ct = 0; ct < 4; ct++)
                simdgroup_store(mc[rt * 4 + ct],
                                c + (uint64_t)(ct * 8) * 640 + r0 + rt * 8, 640, ulong2(0), false);
    }
}
)MSL";

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char **argv) {
    int T = 256, iters = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--tokens")) T = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
    }
    const int E = 8, K = 2560, F = 640;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    NSError *err = nil;
    MTLCompileOptions *mopts = [MTLCompileOptions new];
    mopts.preprocessorMacros = @{@"DS4_METAL_HAS_TENSOR": @"1"};
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource)
                                           options:mopts
                                             error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    const uint64_t wN = (uint64_t)E * F * K, xN = (uint64_t)E * T * K, cN = (uint64_t)E * T * F;
    id<MTLBuffer> W = [dev newBufferWithLength:wN * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> X = [dev newBufferWithLength:xN * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [dev newBufferWithLength:cN * sizeof(float) options:MTLResourceStorageModeShared];
    uint16_t *hw = (uint16_t *)W.contents, *hx = (uint16_t *)X.contents;
    float *hc = (float *)C.contents;
    /* fp16 ~ small values; exact values irrelevant for timing */
    for (uint64_t i = 0; i < wN; i++) hw[i] = 0x3800;           /* 0.5 */
    for (uint64_t i = 0; i < xN; i++) hx[i] = 0x3800;
    memset(hc, 0, cN * sizeof(float));

    const int nrb = F / 64, ntt = T / 32;
    MTLSize grid = MTLSizeMake(nrb * ntt, E, 1);
    const char *names[] = {"mma_v0", "mma_v1", "mma_v2", "mma_v3", "mma_v4"};
    for (int v = 0; v < 5; v++) {
        id<MTLFunction> fn = [lib newFunctionWithName:@(names[v])];
        if (!fn) { fprintf(stderr, "missing %s\n", names[v]); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "pso %s: %s\n", names[v], [[err localizedDescription] UTF8String]); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        /* warmup + correctness on first variant only */
        for (int it = 0; it < iters + 2; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:W offset:0 atIndex:0];
            [enc setBuffer:X offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            uint32_t kd = K, td = T;
            [enc setBytes:&kd length:4 atIndex:3];
            [enc setBytes:&td length:4 atIndex:4];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) { fprintf(stderr, "%s error: %s\n", names[v], [[cb.error localizedDescription] UTF8String]); return 1; }
        }
        /* timed run: single commit, GPU-side timing via MTLCounterSample? keep CPU wall of N iters */
        double t0 = now_ms();
        for (int it = 0; it < iters; it++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:W offset:0 atIndex:0];
            [enc setBuffer:X offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            uint32_t kd = K, td = T;
            [enc setBytes:&kd length:4 atIndex:3];
            [enc setBytes:&td length:4 atIndex:4];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double dt = (now_ms() - t0) / iters;
        /* every input is 0.5 -> every output must be K*0.25 = 640 */
        double bad = 0;
        for (uint64_t i = 0; i < cN; i += 4096)
            if (fabs(hc[i] - K * 0.25) > 1.0) bad++;
        /* GFLOP per launch: 2 * E * F * T * K */
        double gflop = 2.0 * E * F * T * K / 1e9;
        printf("%s: %.3f ms/launch  %.2f TFLOPS  (grid %ux%u)  bad=%g\n",
               names[v], dt, gflop / (dt / 1e3) / 1e3,
               (unsigned)grid.width, (unsigned)grid.height, bad);
    }
    /* sanity: C[0][0][0] should be K*0.25 = 640 */
    printf("C[0][0][0]=%.1f (expect ~%.1f)\n", hc[0], K * 0.25);
    return 0;
}
