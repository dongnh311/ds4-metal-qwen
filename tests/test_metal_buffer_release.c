/* A freed tensor must give its Metal memory back, whatever touched it first.
 * C callers have no autorelease pool, so an accessor that leaves the
 * MTLBuffer autoreleased pins it for the life of the thread, and a later
 * ds4_gpu_tensor_free releases only the handle.  On the Qwen3.8 512K slot
 * that kept every KV set freed by a grow/shrink (qwen4_graph_resize_ctx)
 * resident: 60 GB of IOAccelerator memory after 2 h 40 min of agent turns.
 *
 * The KV disk cache touches the cache tensors from the CPU: a store reads
 * them (payload_write_tensor_span -> ds4_gpu_tensor_read), a load writes
 * them (payload_read_tensor_span -> ds4_gpu_tensor_write), and a fresh
 * cache set is zero-filled (qwen4_graph_alloc_f32 -> ds4_gpu_tensor_fill_f32).
 *
 * Measured on the process's own phys_footprint (task_info), which is what
 * `footprint <pid>` reports as IOAccelerator (graphics).
 *
 *   resize     copy old -> new in a command batch, free old
 *   write      a disk-cache load into the tensor, then free it
 *   read       a disk-cache store out of the tensor, then free it
 *   fill       zero-fill a fresh tensor, then free it
 *   encode     a kernel binds the tensor in a command batch, then free it
 *   fill-pool  fill, with the caller holding an autorelease pool per round
 *              (diagnostic: shows whether a pending autorelease is what
 *              keeps a freed buffer alive)
 */
#include "ds4_gpu.h"

#include <mach/mach.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MIB (1024ull * 1024ull)
#define ROUNDS 16
#define BIG (192ull * MIB)
/* One leaked set is BIG; allow a little driver slack, never a whole set. */
#define SLACK (BIG / 2u)

/* libobjc's C entry points, so this C test can play an Objective-C caller. */
void *objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void *pool);

typedef enum { MODE_RESIZE, MODE_WRITE, MODE_READ, MODE_FILL, MODE_ENCODE, MODE_FILL_POOL } mode_t_;
static const char *const mode_names[] = {"resize", "write", "read", "fill", "encode", "fill-pool"};
#define N_MODES 6

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t phys_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return info.phys_footprint;
}

/* Sizes alternate like the observed 32768 -> N -> 2N -> 32768 cycle, so a
 * size-keyed reuse pool cannot hide a leak by handing the same buffer back. */
static uint64_t round_bytes(int i) {
    return BIG + (uint64_t)(i % 4) * 4u * MIB;
}

/* Dirty every page of t on the GPU, inside the library's own pooled copy,
 * so the round's footprint reflects the buffer without a CPU access. */
static bool gpu_dirty(ds4_gpu_tensor *t, ds4_gpu_tensor *src) {
    const uint64_t n = ds4_gpu_tensor_bytes(t) < ds4_gpu_tensor_bytes(src)
        ? ds4_gpu_tensor_bytes(t) : ds4_gpu_tensor_bytes(src);
    return ds4_gpu_begin_commands() &&
           ds4_gpu_tensor_copy(t, 0, src, 0, n) &&
           ds4_gpu_end_commands();
}

static int run(mode_t_ mode) {
    const char *name = mode_names[mode];
    ds4_gpu_tensor *cur = NULL, *next = NULL, *src = NULL;
    static float chunk[16384];
    int rc = 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "%s: Metal init failed\n", name);
        return 1;
    }
    /* src is the largest round size so a full-size copy is always possible. */
    src = ds4_gpu_tensor_alloc(round_bytes(3));
    cur = ds4_gpu_tensor_alloc(round_bytes(0));
    if (!src || !cur) goto done;
    if (!gpu_dirty(src, src) || !gpu_dirty(cur, src)) goto done;

    uint64_t baseline = 0;
    for (int i = 1; i <= ROUNDS; i++) {
        void *pool = mode == MODE_FILL_POOL ? objc_autoreleasePoolPush() : NULL;
        const uint64_t bytes = round_bytes(i);
        next = ds4_gpu_tensor_alloc(bytes);
        bool ok = next != NULL;
        switch (mode) {
        case MODE_RESIZE:
            ok = ok && gpu_dirty(next, cur);
            break;
        case MODE_WRITE:
            ok = ok && gpu_dirty(next, src) &&
                 ds4_gpu_tensor_write(next, 0, chunk, sizeof(chunk));
            break;
        case MODE_READ:
            ok = ok && gpu_dirty(next, src) &&
                 ds4_gpu_tensor_read(next, 0, chunk, sizeof(chunk));
            break;
        case MODE_ENCODE:
            ok = ok && gpu_dirty(next, src) &&
                 ds4_gpu_begin_commands() &&
                 ds4_gpu_add_tensor(next, src, src, (uint32_t)(sizeof(chunk) / sizeof(float))) &&
                 ds4_gpu_end_commands();
            break;
        case MODE_FILL:
        case MODE_FILL_POOL:
            ok = ok && ds4_gpu_tensor_fill_f32(next, 0.0f, bytes / sizeof(float));
            break;
        }
        if (pool) objc_autoreleasePoolPop(pool);
        if (!ok) goto done;
        ds4_gpu_tensor_free(cur);
        cur = next;
        next = NULL;
        const uint64_t fp = phys_footprint();
        if (i == 2) baseline = fp;   /* after warmup: live sets + driver caches */
        fprintf(stderr, "%s round %2d: phys_footprint %.0f MiB\n", name, i, (double)fp / MIB);
    }
    const uint64_t last = phys_footprint();
    const double grew = ((double)last - (double)baseline) / MIB;
    if (last > baseline + SLACK) {
        fprintf(stderr, "%s: FAIL footprint grew %.0f MiB over %d rounds (%.0f MiB per freed tensor of ~%.0f MiB)\n",
                name, grew, ROUNDS - 2, grew / (ROUNDS - 2), (double)BIG / MIB);
        goto done;
    }
    printf("%s: freed tensors return their Metal memory (grew %.0f MiB over %d rounds): PASS\n",
           name, grew, ROUNDS - 2);
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(cur);
    ds4_gpu_tensor_free(src);
    ds4_gpu_cleanup();
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        for (int m = 0; m < N_MODES; m++)
            if (!strcmp(argv[1], mode_names[m])) return run((mode_t_)m);
    }
    fprintf(stderr, "usage: %s resize|write|read|fill|encode|fill-pool\n", argv[0]);
    return 2;
}
