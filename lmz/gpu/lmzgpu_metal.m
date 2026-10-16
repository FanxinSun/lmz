/*
 * lmz rANS decode on Apple Silicon Metal -- the library form.
 *
 * Implements the exact same C ABI as CUDA (lmzgpu.cu):
 * - lmz_gpu_abi_version
 * - lmz_gpu_last_error
 * - lmz_gpu_grain
 * - lmz_gpu_header_bytes
 * - lmz_gpu_pad_bytes
 * - lmz_gpu_device_name
 * - lmz_gpu_validate
 * - lmz_gpu_decode_batch_dev
 * - lmz_gpu_decode_batch
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __cplusplus
#define LMZ_GPU_API extern "C" __attribute__((visibility("default")))
#else
#define LMZ_GPU_API __attribute__((visibility("default")))
#endif

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8
#define HEADER 516                 /* magic(2) + reserved(2) + 256 * uint16 */
#define STAGE_ITERS 16             /* 16 iters x 8 bytes = 128 B per group */
#define GRAIN (NST * STAGE_ITERS)  /* bytes a group retires per outer step */
#define BUFB 512                   /* prefetch ring / pad bytes */

#define LMZ_GPU_OK            0
#define LMZ_GPU_ENODEV       -1    /* no Metal device */
#define LMZ_GPU_EUNSUPPORTED -2    /* a shape this kernel does not decode */
#define LMZ_GPU_EBADSTREAM   -3    /* magic or frequency sum is wrong */
#define LMZ_GPU_ECUDA        -4    /* API call or launch failed */
#define LMZ_GPU_EMISMATCH    -5

#define ABI_VERSION 1

struct Params {
    uint32_t nstr;
    uint32_t plane;
};

static __thread char g_err[256];

static int fail_msg(int rc, const char *what)
{
    snprintf(g_err, sizeof(g_err), "%s", what);
    return rc;
}

LMZ_GPU_API int lmz_gpu_abi_version(void) { return ABI_VERSION; }
LMZ_GPU_API const char *lmz_gpu_last_error(void) { return g_err; }
LMZ_GPU_API int lmz_gpu_grain(void) { return GRAIN; }
LMZ_GPU_API int lmz_gpu_header_bytes(void) { return HEADER; }
LMZ_GPU_API int lmz_gpu_pad_bytes(void) { return BUFB + 64; }

LMZ_GPU_API int lmz_gpu_device_name(char *buf, int cap)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return fail_msg(LMZ_GPU_ENODEV, "no Metal device");
        snprintf(buf, (size_t)cap, "%s (Metal unified=%d)",
                 [[dev name] UTF8String], (int)[dev hasUnifiedMemory]);
        return LMZ_GPU_OK;
    }
}

LMZ_GPU_API int lmz_gpu_validate(const void *streams_v, const void *off_v, unsigned nstr)
{
    const uint8_t *streams = (const uint8_t *)streams_v;
    const uint64_t *off = (const uint64_t *)off_v;
    for (unsigned i = 0; i < nstr; i++) {
        const uint8_t *p = streams + off[2 * i];
        if (p[0] != 'R' || p[1] != '1') return LMZ_GPU_EBADSTREAM;
        uint32_t sum = 0;
        for (int s = 0; s < 256; s++)
            sum += (uint32_t)p[4 + 2 * s] | ((uint32_t)p[5 + 2 * s] << 8);
        if (sum != PROB_SCALE) return LMZ_GPU_EBADSTREAM;
    }
    return LMZ_GPU_OK;
}

// ------------------------------------------------------------- runtime cache
static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pso_shared = nil;
static id<MTLComputePipelineState> g_pso_perstream = nil;
static int g_init_rc = LMZ_GPU_OK;
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

static void init_metal_once(void)
{
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) {
            g_init_rc = fail_msg(LMZ_GPU_ENODEV, "no Metal device");
            return;
        }
        g_queue = [g_dev newCommandQueue];
        if (!g_queue) {
            g_init_rc = fail_msg(LMZ_GPU_ECUDA, "failed to create Metal command queue");
            return;
        }

        // Locate lmz_rans.metal next to this dylib
        Dl_info dli;
        NSString *shader_path = nil;
        if (dladdr((void *)lmz_gpu_abi_version, &dli) && dli.dli_fname) {
            NSString *dir = [[NSString stringWithUTF8String:dli.dli_fname] stringByDeletingLastPathComponent];
            NSString *candidate = [dir stringByAppendingPathComponent:@"lmz_rans.metal"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                shader_path = candidate;
            }
        }

        NSError *err = nil;
        NSString *source = nil;
        if (shader_path) {
            source = [NSString stringWithContentsOfFile:shader_path encoding:NSUTF8StringEncoding error:&err];
        }

        if (!source) {
            g_init_rc = fail_msg(LMZ_GPU_ECUDA, "could not load lmz_rans.metal shader source");
            return;
        }

        id<MTLLibrary> lib = [g_dev newLibraryWithSource:source options:nil error:&err];
        if (!lib) {
            snprintf(g_err, sizeof(g_err), "Metal compile error: %s", [[err localizedDescription] UTF8String]);
            g_init_rc = LMZ_GPU_ECUDA;
            return;
        }

        id<MTLFunction> fn_shared = [lib newFunctionWithName:@"lmz_decode_plane_direct"];
        if (!fn_shared) {
            g_init_rc = fail_msg(LMZ_GPU_ECUDA, "Metal function lmz_decode_plane_direct not found");
            return;
        }
        g_pso_shared = [g_dev newComputePipelineStateWithFunction:fn_shared error:&err];
        if (!g_pso_shared) {
            snprintf(g_err, sizeof(g_err), "PSO shared error: %s", [[err localizedDescription] UTF8String]);
            g_init_rc = LMZ_GPU_ECUDA;
            return;
        }

        id<MTLFunction> fn_perstream = [lib newFunctionWithName:@"lmz_decode_perstream"];
        if (!fn_perstream) {
            g_init_rc = fail_msg(LMZ_GPU_ECUDA, "Metal function lmz_decode_perstream not found");
            return;
        }
        g_pso_perstream = [g_dev newComputePipelineStateWithFunction:fn_perstream error:&err];
        if (!g_pso_perstream) {
            snprintf(g_err, sizeof(g_err), "PSO perstream error: %s", [[err localizedDescription] UTF8String]);
            g_init_rc = LMZ_GPU_ECUDA;
            return;
        }
        g_init_rc = LMZ_GPU_OK;
    }
}

static int ensure_init(void)
{
    pthread_once(&g_init_once, init_metal_once);
    return g_init_rc;
}

// Wrap a pointer into an MTLBuffer using unified memory if page-aligned,
// or fallback to allocated buffer.
static id<MTLBuffer> make_buffer_for_ptr(id<MTLDevice> dev, const void *ptr, size_t length, size_t *out_offset)
{
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t page_addr = addr & ~(page_size - 1);
    size_t page_offset = addr - page_addr;
    if (page_offset % 4 == 0) {
        size_t alloc_len = (length + page_offset + page_size - 1) & ~(page_size - 1);
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:(void *)page_addr
                                                   length:alloc_len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (buf) {
            *out_offset = page_offset;
            return buf;
        }
    }
    *out_offset = 0;
    return [dev newBufferWithBytes:ptr length:(length > 1 ? length : 1) options:MTLResourceStorageModeShared];
}

LMZ_GPU_API int lmz_gpu_decode_batch_dev(const void *d_hdr, const void *d_streams,
                                         const void *d_off, unsigned nstr,
                                         unsigned plane, void *d_out,
                                         void *metal_queue, unsigned tpb)
{
    (void)tpb;
    if (nstr == 0) return LMZ_GPU_OK;
    if (plane == 0 || plane % GRAIN != 0)
        return fail_msg(LMZ_GPU_EUNSUPPORTED, "plane is not a multiple of the grain");

    int init_rc = ensure_init();
    if (init_rc != LMZ_GPU_OK) return init_rc;

    @autoreleasepool {
        id<MTLCommandQueue> queue = metal_queue ? (__bridge id<MTLCommandQueue>)metal_queue : g_queue;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate command buffer");

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate compute encoder");

        size_t off_s = 0, off_o = 0, off_out = 0, off_h = 0;
        size_t nbytes_off = (size_t)nstr * 16;
        size_t nbytes_out = (size_t)nstr * plane;

        id<MTLBuffer> b_streams = make_buffer_for_ptr(g_dev, d_streams, (size_t)GRAIN * nstr + lmz_gpu_pad_bytes(), &off_s);
        id<MTLBuffer> b_off = make_buffer_for_ptr(g_dev, d_off, nbytes_off, &off_o);
        id<MTLBuffer> b_out = make_buffer_for_ptr(g_dev, d_out, nbytes_out, &off_out);
        id<MTLBuffer> b_hdr = nil;
        if (d_hdr != NULL) {
            b_hdr = make_buffer_for_ptr(g_dev, d_hdr, HEADER, &off_h);
        }

        struct Params P = { nstr, plane };

        if (d_hdr != NULL) {
            [enc setComputePipelineState:g_pso_shared];
            [enc setBuffer:b_streams offset:off_s atIndex:0];
            [enc setBuffer:b_off offset:off_o atIndex:1];
            [enc setBuffer:b_out offset:off_out atIndex:2];
            [enc setBuffer:b_hdr offset:off_h atIndex:3];
            [enc setBytes:&P length:sizeof(P) atIndex:4];
            uint32_t tot = nstr * 8;
            uint32_t tgs = (tot + 127) / 128;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:g_pso_perstream];
            [enc setBuffer:b_streams offset:off_s atIndex:0];
            [enc setBuffer:b_off offset:off_o atIndex:1];
            [enc setBuffer:b_out offset:off_out atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            uint32_t tot = nstr * 8;
            uint32_t tgs = (tot + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        // If caller passed custom queue, do not synchronously wait unless needed
        if (!metal_queue) {
            [cb waitUntilCompleted];
            if ([cb error]) {
                snprintf(g_err, sizeof(g_err), "Metal execution error: %s",
                         [[[cb error] localizedDescription] UTF8String]);
                return LMZ_GPU_ECUDA;
            }
        }
        return LMZ_GPU_OK;
    }
}

LMZ_GPU_API int lmz_gpu_decode_batch(const void *hdr, const void *streams, size_t nbytes,
                                     const void *off, unsigned nstr, unsigned plane,
                                     void *out)
{
    if (nstr == 0) return LMZ_GPU_OK;
    if (plane == 0 || plane % GRAIN != 0)
        return fail_msg(LMZ_GPU_EUNSUPPORTED, "plane is not a multiple of the grain");

    int init_rc = ensure_init();
    if (init_rc != LMZ_GPU_OK) return init_rc;

    if (hdr == NULL) {
        int rc = lmz_gpu_validate(streams, off, nstr);
        if (rc != LMZ_GPU_OK) return rc;
    }

    @autoreleasepool {
        size_t pad = (size_t)lmz_gpu_pad_bytes();
        size_t obytes = (size_t)nstr * plane;

        // Allocate shared buffers with readable slack
        NSMutableData *streams_padded = [NSMutableData dataWithLength:nbytes + pad];
        if (!streams_padded) return fail_msg(LMZ_GPU_ECUDA, "memory allocation failed");
        memcpy([streams_padded mutableBytes], streams, nbytes);

        id<MTLBuffer> b_streams = [g_dev newBufferWithBytes:[streams_padded bytes]
                                                     length:[streams_padded length]
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_off = [g_dev newBufferWithBytes:off
                                                 length:(size_t)nstr * 16
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [g_dev newBufferWithLength:obytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_hdr = nil;
        if (hdr != NULL) {
            b_hdr = [g_dev newBufferWithBytes:hdr length:HEADER options:MTLResourceStorageModeShared];
        }

        if (!b_streams || !b_off || !b_out || (hdr && !b_hdr)) {
            return fail_msg(LMZ_GPU_ECUDA, "failed to allocate Metal buffers");
        }

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate Metal encoder");

        struct Params P = { nstr, plane };

        if (hdr != NULL) {
            [enc setComputePipelineState:g_pso_shared];
            [enc setBuffer:b_streams offset:0 atIndex:0];
            [enc setBuffer:b_off offset:0 atIndex:1];
            [enc setBuffer:b_out offset:0 atIndex:2];
            [enc setBuffer:b_hdr offset:0 atIndex:3];
            [enc setBytes:&P length:sizeof(P) atIndex:4];
            uint32_t tot = nstr * 8;
            uint32_t tgs = (tot + 127) / 128;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:g_pso_perstream];
            [enc setBuffer:b_streams offset:0 atIndex:0];
            [enc setBuffer:b_off offset:0 atIndex:1];
            [enc setBuffer:b_out offset:0 atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            uint32_t tot = nstr * 8;
            uint32_t tgs = (tot + 31) / 32;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if ([cb error]) {
            snprintf(g_err, sizeof(g_err), "Metal execution error: %s",
                     [[[cb error] localizedDescription] UTF8String]);
            return LMZ_GPU_ECUDA;
        }

        memcpy(out, [b_out contents], obytes);
        return LMZ_GPU_OK;
    }
}
