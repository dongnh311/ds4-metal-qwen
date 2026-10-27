// Layout probe: does the IQ2_XXS block layout (8 B per lane, 66 B block stride)
// cost throughput versus a planar layout where a warp reads 256 B contiguous?
// Both kernels move the same number of useful bytes from the mmap'd model file.
// build: clang -O2 -fobjc-arc -framework Foundation -framework Metal -o /tmp/layoutprobe /tmp/layoutprobe.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static const char *kSrcUnused =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"// A: current layout. One warp covers 4 x 66 B blocks (32 lanes x 8 B).\n"
"kernel void probe_a(device const uchar * src [[buffer(0)]],\n"
"                    device float * out [[buffer(1)]],\n"
"                    constant uint64_t & rows [[buffer(2)]],\n"
"                    constant uint64_t & row_stride [[buffer(3)]],\n"
"                    uint tg [[threadgroup_position_in_grid]],\n"
"                    ushort ti [[thread_index_in_threadgroup]],\n"
"                    ushort nt [[threads_per_threadgroup]]) {\n"
"    uint lane = ti % 32;\n"
"    uint64_t ntg = (uint64_t)tg * nt + ti;\n"
"    uint64_t stride_total = (uint64_t)get_num_threadgroups_dummy(nt);\n" // replaced below
"}\n";

// The two probes are written explicitly (no shared helper) for clarity.
static const char *kSrc2 =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint TGS [[function_constant(0)]];\n"
"kernel void probe_a(device const uchar * src [[buffer(0)]],\n"
"                    device float * out [[buffer(1)]],\n"
"                    constant uint64_t & rows [[buffer(2)]],\n"
"                    uint tg [[threadgroup_position_in_grid]],\n"
"                    ushort ti [[thread_index_in_threadgroup]],\n"
"                    ushort nt [[threads_per_threadgroup]]) {\n"
"    uint lane = ti % 32;\n"
"    uint warp = ti / 32;\n"
"    uint warp_id = tg * (nt / 32) + warp;\n"
"    uint64_t warps = (uint64_t)TGS * (nt / 32);\n"
"    float acc = 0.0f;\n"
"    for (uint64_t r = warp_id; r < rows; r += warps) {\n"
"        device const uchar * row = src + r * 1320ul;\n"
"        for (int k = 0; k < 5; ++k) {\n"
"            device const uchar * p = row + k * 264ul + (lane / 8) * 66ul + (lane % 8) * 8ul;\n"
"            acc += (float)p[0] + (float)p[7];\n"
"        }\n"
"    }\n"
"    if (ti == 0) out[tg] = acc;\n"
"}\n"
"kernel void probe_b(device const uchar * src [[buffer(0)]],\n"
"                    device float * out [[buffer(1)]],\n"
"                    constant uint64_t & rows [[buffer(2)]],\n"
"                    uint tg [[threadgroup_position_in_grid]],\n"
"                    ushort ti [[thread_index_in_threadgroup]],\n"
"                    ushort nt [[threads_per_threadgroup]]) {\n"
"    uint lane = ti % 32;\n"
"    uint warp = ti / 32;\n"
"    uint warp_id = tg * (nt / 32) + warp;\n"
"    uint64_t warps = (uint64_t)TGS * (nt / 32);\n"
"    float acc = 0.0f;\n"
"    for (uint64_t r = warp_id; r < rows; r += warps) {\n"
"        device const uchar * row = src + r * 1280ul;\n"
"        for (int k = 0; k < 5; ++k) {\n"
"            device const uchar * p = row + k * 256ul + lane * 8ul;\n"
"            acc += (float)p[0] + (float)p[7];\n"
"        }\n"
"    }\n"
"    if (ti == 0) out[tg] = acc;\n"
"}\n";

static double now(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e9; }

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *path = getenv("LP_FILE");
        if (!path) { printf("set LP_FILE\n"); return 1; }
        size_t bytes = (size_t)atoll(getenv("LP_MB") ? getenv("LP_MB") : "2048") << 20;
        double touch_frac = getenv("LP_TOUCH") ? atof(getenv("LP_TOUCH")) : 0.02;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc2]
                                              options:nil error:&err];
        if (!lib) { printf("compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        int fd = open(path, O_RDONLY);
        if (fd < 0) { printf("open fail\n"); return 1; }
        void *m = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { printf("mmap fail\n"); return 1; }
        size_t step = (size_t)(4096.0 / touch_frac);
        if (step < 4096) step = 4096;
        for (size_t i = 0; i < bytes; i += step) (void)*(volatile char *)((char *)m + i);

        id<MTLBuffer> src = [dev newBufferWithBytesNoCopy:m length:bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        id<MTLBuffer> dst = [dev newBufferWithLength:65536 * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        uint64_t rows_a = bytes / 1320, rows_b = bytes / 1280;
        unsigned nt = 64;
        printf("%-10s %8s %10s %10s\n", "layout", "TG", "useful GB/s", "ms");
        for (int v = 0; v < 2; v++) {
            unsigned tgs[] = {288, 1728, 3456};
            for (int g = 0; g < 3; g++) {
                unsigned ntgs = tgs[g];
                MTLFunctionConstantValues *cv = [[MTLFunctionConstantValues alloc] init];
                [cv setConstantValue:&ntgs type:MTLDataTypeUInt atIndex:0];
                err = nil;
                id<MTLFunction> fn = [lib newFunctionWithName:(v == 0 ? @"probe_a" : @"probe_b")
                                              constantValues:cv error:&err];
                id<MTLComputePipelineState> pso = fn ?
                    [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
                if (!pso) { printf("pso: %s\n", [[err localizedDescription] UTF8String]); return 1; }
                uint64_t rows = v == 0 ? rows_a : rows_b;
                uint64_t useful = rows * 20 * 64;   // 20 blocks x 64 B of qs per row
                int reps = 3;
                double t0 = now();
                for (int i = 0; i < reps; i++) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                    [e setComputePipelineState:pso];
                    [e setBuffer:src offset:0 atIndex:0];
                    [e setBuffer:dst offset:0 atIndex:1];
                    [e setBytes:&rows length:8 atIndex:2];
                    [e dispatchThreadgroups:MTLSizeMake(ntgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];
                    [e endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }
                double dt = (now() - t0) / reps;
                printf("%-10s %8u %10.1f %10.3f\n", v == 0 ? "blocked66" : "planar128", ntgs,
                       (double)useful / dt / 1e9, dt * 1e3);
            }
        }
    }
    return 0;
}