#if !__has_feature(objc_arc)
#error Metal host requires ARC for retained buffer/view lifetimes
#endif
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
#import <IOKit/IOKitLib.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdatomic.h>
#include <float.h>
#include <time.h>

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
static __thread double g_timing[9];

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
static id<MTLComputePipelineState> g_pso_bf16 = nil;
static id<MTLComputePipelineState> g_pso_perstream = nil;
static id<MTLComputePipelineState> g_pso_dequant_q8 = nil;
static id<MTLComputePipelineState> g_pso_project_fp8 = nil;
static id<MTLComputePipelineState> g_pso_perstream_old = nil;
static _Atomic unsigned g_perstream_tgs = 0;
static _Atomic int g_stride_verified = 1;
static pthread_mutex_t g_tune_lock = PTHREAD_MUTEX_INITIALIZER;
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

        g_pso_perstream_old = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lmz_decode_perstream"] error:&err];
        id<MTLFunction> fn_perstream = [lib newFunctionWithName:@"lmz_decode_perstream_stride"];
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

        g_pso_bf16 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lmz_decode_bf16"] error:&err];

        id<MTLFunction> fn_dequant = [lib newFunctionWithName:@"lmz_dequant_q8_0_to_fp16"];
        if (fn_dequant) {
            g_pso_dequant_q8 = [g_dev newComputePipelineStateWithFunction:fn_dequant error:nil];
        }
        id<MTLFunction> fn_fp8 = [lib newFunctionWithName:@"lmz_project_fp8_e4m3_to_bf16"];
        if (fn_fp8) {
            g_pso_project_fp8 = [g_dev newComputePipelineStateWithFunction:fn_fp8 error:nil];
        }
        g_init_rc = LMZ_GPU_OK;
    }
}

static int ensure_init(void)
{
    pthread_once(&g_init_once, init_metal_once);
    return g_init_rc;
}


// Override is read at dispatch, so callers can compare geometry without rebuilding.
static unsigned perstream_grid(unsigned nstr) {
    unsigned full = (nstr + 3) / 4;
    if (!atomic_load(&g_stride_verified)) return full;
    const char *env = getenv("LMZ_METAL_PERSTREAM_TGS");
    if (env && *env) {
        char *end = NULL; unsigned long x = strtoul(env, &end, 10);
        if (*end == 0 && x > 0 && x <= UINT32_MAX) return (unsigned)(x < full ? x : full);
    }
    unsigned tuned = atomic_load(&g_perstream_tgs);
    return tuned && tuned < full ? tuned : full;
}

// Tuning costs about twenty times one batch's decode (4.8 s on a 936 MB per-chunk
// batch on an Apple M4) and pays back only over many calls: the tuned geometry was
// 3.6x faster, so a single batch of this size never recovers it. It is therefore
// EXPLICIT: prepare_metal_batch() asks for it (that API exists for repeated decodes),
// and LMZ_METAL_PERSTREAM_TGS=auto asks for it from the environment. A plain
// decode_batch() keeps the shipped geometry and pays nothing. A number in the
// variable forces that geometry and never tunes.
static __thread int t_tune_requested = 0;

// Tune on the first caller's full ordinary batch, rather than inventing a
// distribution or changing the traffic. Verify in bounded oracle chunks. Small calls do not pay the once-per-process cost.
// Device-pointer asynchronous calls only consume the cache; they never tune.
static void tune_perstream(id<MTLBuffer> streams, id<MTLBuffer> off,
                           id<MTLBuffer> out, unsigned nstr, unsigned plane) {
    if (nstr < 1024 || atomic_load(&g_perstream_tgs)) return;
    const char *want = getenv("LMZ_METAL_PERSTREAM_TGS");
    if (want && *want && strcmp(want, "auto") != 0) return;          // a number: the caller chose
    if (!t_tune_requested && !(want && strcmp(want, "auto") == 0)) return;   // not asked: no hidden cost
    pthread_mutex_lock(&g_tune_lock);
    if (!atomic_load(&g_perstream_tgs)) {
        unsigned cores = 0;
        io_iterator_t iterator = IO_OBJECT_NULL;
        if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS) {
            io_service_t service;
            while ((service = IOIteratorNext(iterator))) {
                CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
                if (value && CFGetTypeID(value) == CFNumberGetTypeID()) CFNumberGetValue(value, kCFNumberIntType, &cores);
                if (value) CFRelease(value);
                IOObjectRelease(service);
                if (cores) break;
            }
            IOObjectRelease(iterator);
        }
        if (!cores || cores > 1024) cores = 8; // generic candidate scale, still measured
        unsigned count = nstr < 4096 ? nstr : 4096;
        size_t max_count = ((size_t)256 << 20) / plane;
        if (count > max_count) count = (unsigned)max_count;
        count &= ~3u;
        if (!count) { pthread_mutex_unlock(&g_tune_lock); return; }
        unsigned full = (nstr + 3) / 4;
        unsigned factors[] = {4, 8, 10, 16};
        struct Params P = {nstr, plane};
        id<MTLBuffer> oracle = [g_dev newBufferWithLength:(size_t)count*plane options:MTLResourceStorageModeShared];
        double best = DBL_MAX; unsigned winner = full;
        for (unsigned ci = 0; ci < 5; ci++) {
            unsigned grid = ci == 4 ? full : cores * factors[ci];
            if (grid > full) grid = full;
            double times[3] = {0}; BOOL valid = oracle != nil;
            for (unsigned run = 0; valid && run < 4; run++) {
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:g_pso_perstream];
                [enc setBuffer:streams offset:0 atIndex:0]; [enc setBuffer:off offset:0 atIndex:1];
                [enc setBuffer:out offset:0 atIndex:2]; [enc setBytes:&P length:sizeof P atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                valid = !cb.error;
                if (run) times[run-1] = cb.GPUEndTime - cb.GPUStartTime;
            }
            // Full output identity without a second full-size allocation.
            for (unsigned first=0; valid && first<nstr; first+=count) {
                unsigned batch=nstr-first<count?nstr-first:count;
                struct Params Q={batch,plane};
                id<MTLCommandBuffer> cb=[g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:g_pso_perstream_old];
                [enc setBuffer:streams offset:0 atIndex:0]; [enc setBuffer:off offset:(size_t)first*16 atIndex:1];
                [enc setBuffer:oracle offset:0 atIndex:2]; [enc setBytes:&Q length:sizeof Q atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((batch+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                valid = !cb.error && memcmp((const uint8_t *)out.contents+(size_t)first*plane,
                                            oracle.contents,(size_t)batch*plane)==0;
            }
            if (valid) {
                double a=times[0], b=times[1], c=times[2];
                double median = a > b ? (b > c ? b : (a > c ? c : a)) : (a > c ? a : (b > c ? c : b));
                if (median > 0 && median < best) { best = median; winner = grid; }
            }
        }
        if (best == DBL_MAX) atomic_store(&g_stride_verified, 0);
        atomic_store(&g_perstream_tgs, winner);
    }
    pthread_mutex_unlock(&g_tune_lock);
}
LMZ_GPU_API unsigned lmz_gpu_metal_perstream_tgs(void) { return atomic_load(&g_perstream_tgs); }

// Wrap a pointer into an MTLBuffer using unified memory if page-aligned,
// or fallback to allocated buffer.
static id<MTLBuffer> make_buffer_for_ptr(id<MTLDevice> dev, const void *ptr, size_t length, size_t *out_offset)
{
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    *out_offset = 0;
    if (length > dev.maxBufferLength) return nil;
    // Do not round arbitrary objects outwards into pages the caller did not
    // export. Only a complete, aligned page range is eligible for no-copy.
    if (ptr && length && ((uintptr_t)ptr % page_size) == 0 && length % page_size == 0) {
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:(void *)ptr length:length
                            options:MTLResourceStorageModeShared deallocator:nil];
        if (buf) return buf;
    }
    *out_offset = 0;
    return [dev newBufferWithBytes:ptr length:(length > 1 ? length : 1) options:MTLResourceStorageModeShared];
}


// B1: retain the HEAD refill instructions. Ordinary valid rANS streams need
// readable slack for speculative/unconditional refill loads. Only share an
// aligned coded region when that slack fits within its mapped final page.
// The Python exporter/raw device caller keeps these pages alive and unchanged.
static id<MTLBuffer> make_coded_buffer(id<MTLDevice> dev,const void *ptr,size_t capacity,
                                     const uint64_t *off,unsigned nstr) {
    size_t extent=0, page=sysconf(_SC_PAGESIZE), pad=lmz_gpu_pad_bytes();
    for (unsigned i=0;i<nstr;i++) {
        if (off[2*i] > SIZE_MAX-off[2*i+1]) return nil;
        size_t end=off[2*i]+off[2*i+1]; if (end>extent) extent=end;
    }
    if (extent>capacity || extent>SIZE_MAX-page || extent>dev.maxBufferLength-pad) return nil;
    size_t mapped=(extent+page-1)&~(page-1);
    if ((uintptr_t)ptr%page==0 && mapped-extent>=pad) {
        // A page-aligned exporter/caller owns the mapped tail of its last page.
        // Never round an output or an arbitrary unaligned Python object outwards.
        id<MTLBuffer> b=[dev newBufferWithBytesNoCopy:(void *)ptr length:mapped
                             options:MTLResourceStorageModeShared deallocator:nil];
        if (b) return b;
    }
    id<MTLBuffer> b=[dev newBufferWithLength:extent+pad options:MTLResourceStorageModeShared];
    if (b) { memcpy(b.contents,ptr,extent); memset((uint8_t *)b.contents+extent,0,pad); }
    return b;
}

LMZ_GPU_API void *lmz_gpu_make_mtl_buffer(const void *ptr, size_t length, size_t *out_offset)
{
    int init_rc = ensure_init();
    if (init_rc != LMZ_GPU_OK) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = make_buffer_for_ptr(g_dev, ptr, length, out_offset);
        if (!buf) return NULL;
        return (__bridge_retained void *)buf;
    }
}

LMZ_GPU_API void lmz_gpu_release_mtl_buffer(void *buf_handle)
{
    if (buf_handle) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_handle;
        (void)buf;
    }
}

LMZ_GPU_API int lmz_gpu_metal_decode_batch_dev_sized(const void *d_hdr, const void *d_streams,
                                         const void *d_off, unsigned nstr,
                                         unsigned plane, void *d_out,
                                         void *metal_queue, unsigned tpb, size_t streams_capacity, size_t output_capacity)
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

        size_t extent = 0;
        const uint64_t *entries = d_off;
        for (unsigned i=0; i<nstr; i++) {
            if (entries[2*i+1] < NST*4 + (d_hdr ? 0 : HEADER)) return fail_msg(LMZ_GPU_EBADSTREAM,"stream too short");
            if (entries[2*i] > SIZE_MAX - entries[2*i+1]) return fail_msg(LMZ_GPU_EBADSTREAM, "offset overflow");
            size_t end = entries[2*i] + entries[2*i+1]; if (end > extent) extent = end;
        }
        if (streams_capacity && streams_capacity < extent) return fail_msg(LMZ_GPU_EBADSTREAM,"coded capacity too small");
        if (output_capacity && output_capacity < nbytes_out) return fail_msg(LMZ_GPU_EUNSUPPORTED,"output capacity too small");
        id<MTLBuffer> b_streams = make_coded_buffer(g_dev, d_streams, streams_capacity ? streams_capacity : extent, entries, nstr);
        id<MTLBuffer> b_off = make_buffer_for_ptr(g_dev, d_off, nbytes_off, &off_o);
        id<MTLBuffer> b_out = make_buffer_for_ptr(g_dev, d_out, output_capacity ? output_capacity : nbytes_out, &off_out);
        id<MTLBuffer> b_hdr = nil;
        if (d_hdr != NULL) {
            b_hdr = make_buffer_for_ptr(g_dev, d_hdr, HEADER, &off_h);
        }

        if (!b_streams || !b_off || !b_out || (d_hdr && !b_hdr)) {
            [enc endEncoding]; return fail_msg(LMZ_GPU_ECUDA,"Metal buffer allocation failed");
        }
        if (!d_hdr && lmz_gpu_validate(d_streams,d_off,nstr)) {
            [enc endEncoding]; return fail_msg(LMZ_GPU_EBADSTREAM,"invalid ordinary stream header");
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
            [enc setComputePipelineState:atomic_load(&g_stride_verified)?g_pso_perstream:g_pso_perstream_old];
            [enc setBuffer:b_streams offset:off_s atIndex:0];
            [enc setBuffer:b_off offset:off_o atIndex:1];
            [enc setBuffer:b_out offset:off_out atIndex:2];
            [enc setBytes:&P length:sizeof(P) atIndex:3];
            uint32_t tot = nstr * 8;
            uint32_t tgs = perstream_grid(nstr);
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        [enc endEncoding];
        if (metal_queue && b_out.contents != d_out) {
            [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                if (!done.error) memcpy(d_out, b_out.contents, nbytes_out);
            }];
        }
        [cb commit];
        // If caller passed custom queue, do not synchronously wait unless needed
        if (!metal_queue) {
            [cb waitUntilCompleted];
            g_timing[3]=cb.GPUEndTime-cb.GPUStartTime;
            if (b_out.contents != d_out) memcpy(d_out, b_out.contents, nbytes_out);
            if ([cb error]) {
                snprintf(g_err, sizeof(g_err), "Metal execution error: %s",
                         [[[cb error] localizedDescription] UTF8String]);
                return LMZ_GPU_ECUDA;
            }
        }
        return LMZ_GPU_OK;
    }
}


LMZ_GPU_API int lmz_gpu_decode_batch_dev(const void *hdr,const void *streams,const void *off,unsigned nstr,unsigned plane,void *out,void *queue,unsigned tpb) {
    return lmz_gpu_metal_decode_batch_dev_sized(hdr,streams,off,nstr,plane,out,queue,tpb,0,0);
}

static double seconds_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
LMZ_GPU_API const double *lmz_gpu_metal_last_timing(void) { return g_timing; }
LMZ_GPU_API void *lmz_gpu_mtl_contents(void *handle) { return [(__bridge id<MTLBuffer>)handle contents]; }

// Sized host ABI. A NULL output returns a retained shared MTLBuffer; the
// caller releases it after the exported memoryview is no longer referenced.
LMZ_GPU_API int lmz_gpu_metal_decode(const void *hdr, const void *streams, size_t nbytes,
                    const void *off, unsigned nstr, unsigned plane,
                    void *out, size_t out_capacity, void **owned, void **contents) {
    if (owned) *owned = NULL;
    if (contents) *contents = NULL;
    if (!nstr) return LMZ_GPU_OK;
    if (!plane || plane % GRAIN) return fail_msg(LMZ_GPU_EUNSUPPORTED, "invalid plane grain");
    int rc = ensure_init(); if (rc) return rc;
    size_t obytes = (size_t)nstr * plane;
    if (obytes > g_dev.maxBufferLength) return fail_msg(LMZ_GPU_EUNSUPPORTED,"batch output exceeds Metal buffer limit");
    if (out && out_capacity < obytes) return fail_msg(LMZ_GPU_EUNSUPPORTED, "output is too small");
    const uint64_t *entries = off;
    for (unsigned i=0; i<nstr; i++) {
        size_t o=entries[2*i], n=entries[2*i+1];
        if (o > nbytes || n > nbytes-o || n < NST*4 + (hdr ? 0 : HEADER))
            return fail_msg(LMZ_GPU_EBADSTREAM, "stream offset/length out of range");
    }
    if (hdr) {
        const uint8_t *h=hdr; unsigned sum=0;
        for (int i=0;i<256;i++) sum += h[4+2*i] | (h[5+2*i]<<8);
        if (h[0]!='R' || h[1]!='1' || sum!=PROB_SCALE) return fail_msg(LMZ_GPU_EBADSTREAM,"invalid shared header");
    } else { rc=lmz_gpu_validate(streams,off,nstr); if (rc) return rc; }
    @autoreleasepool {
        memset(g_timing,0,sizeof g_timing); double begin=seconds_now(), t=begin; size_t ignored=0;
        id<MTLBuffer> bs=make_coded_buffer(g_dev,streams,nbytes,off,nstr);
        g_timing[0]=seconds_now()-t; g_timing[7]=bs.contents==streams; t=seconds_now();
        id<MTLBuffer> bo=make_buffer_for_ptr(g_dev,off,(size_t)nstr*16,&ignored);
        id<MTLBuffer> bh=hdr ? make_buffer_for_ptr(g_dev,hdr,HEADER,&ignored) : nil;
        g_timing[1]=seconds_now()-t; t=seconds_now();
        BOOL aligned_out=out && (uintptr_t)out % sysconf(_SC_PAGESIZE)==0 && out_capacity % sysconf(_SC_PAGESIZE)==0;
        id<MTLBuffer> dst=aligned_out ? make_buffer_for_ptr(g_dev,out,out_capacity,&ignored) :
            [g_dev newBufferWithLength:obytes options:MTLResourceStorageModeShared];
        if (!bs || !bo || !dst || (hdr && !bh)) return fail_msg(LMZ_GPU_ECUDA,"Metal allocation failed");
        g_timing[2]=seconds_now()-t; g_timing[8]=!out || dst.contents==out;
        if (!hdr) tune_perstream(bs,bo,dst,nstr,plane);
        struct Params P={nstr,plane};
        id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:hdr ? g_pso_shared : (atomic_load(&g_stride_verified)?g_pso_perstream:g_pso_perstream_old)];
        [enc setBuffer:bs offset:0 atIndex:0]; [enc setBuffer:bo offset:0 atIndex:1]; [enc setBuffer:dst offset:0 atIndex:2];
        if (hdr) { [enc setBuffer:bh offset:0 atIndex:3]; [enc setBytes:&P length:sizeof P atIndex:4]; }
        else [enc setBytes:&P length:sizeof P atIndex:3];
        unsigned grid=hdr ? (nstr+15)/16 : perstream_grid(nstr);
        [enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(hdr?128:32,1,1)];
        [enc endEncoding]; t=seconds_now(); [cb commit]; [cb waitUntilCompleted];
        g_timing[4]=seconds_now()-t; g_timing[3]=cb.GPUEndTime-cb.GPUStartTime;
        if (cb.error) return fail_msg(LMZ_GPU_ECUDA,cb.error.localizedDescription.UTF8String);
        t=seconds_now();
        if (out && dst.contents!=out) memcpy(out,dst.contents,obytes);
        g_timing[5]=seconds_now()-t;
        if (!out) {
            if (!owned || !contents) return fail_msg(LMZ_GPU_EUNSUPPORTED,"owned output needs handles");
            *contents=dst.contents; *owned=(__bridge_retained void *)dst;
        } else if (contents) *contents=out;
        g_timing[6]=seconds_now()-begin;
        return LMZ_GPU_OK;
    }
}
LMZ_GPU_API int lmz_gpu_metal_decode_bf16(const void *hdr, const void *streams, size_t nbytes,
                    const void *off, unsigned nstr, unsigned plane,
                    const void *sm, size_t sm_bytes, void *out, size_t out_capacity, void **owned, void **contents) {
    if (owned) *owned = NULL;
    if (contents) *contents = NULL;
    if (!hdr) return fail_msg(LMZ_GPU_EUNSUPPORTED,"fused decode requires a shared header");
    if (sm_bytes < (size_t)nstr*plane) return fail_msg(LMZ_GPU_EBADSTREAM,"sign/mantissa plane too short");
    if (!nstr) return LMZ_GPU_OK;
    if (!plane || plane % GRAIN) return fail_msg(LMZ_GPU_EUNSUPPORTED, "invalid plane grain");
    int rc = ensure_init(); if (rc) return rc;
    size_t obytes = (size_t)nstr * plane * 2;
    if (obytes > g_dev.maxBufferLength) return fail_msg(LMZ_GPU_EUNSUPPORTED,"batch output exceeds Metal buffer limit");
    if (out && out_capacity < obytes) return fail_msg(LMZ_GPU_EUNSUPPORTED, "output is too small");
    const uint64_t *entries = off;
    for (unsigned i=0; i<nstr; i++) {
        size_t o=entries[2*i], n=entries[2*i+1];
        if (o > nbytes || n > nbytes-o || n < NST*4 + (hdr ? 0 : HEADER))
            return fail_msg(LMZ_GPU_EBADSTREAM, "stream offset/length out of range");
    }
    if (hdr) {
        const uint8_t *h=hdr; unsigned sum=0;
        for (int i=0;i<256;i++) sum += h[4+2*i] | (h[5+2*i]<<8);
        if (h[0]!='R' || h[1]!='1' || sum!=PROB_SCALE) return fail_msg(LMZ_GPU_EBADSTREAM,"invalid shared header");
    } else { rc=lmz_gpu_validate(streams,off,nstr); if (rc) return rc; }
    @autoreleasepool {
        memset(g_timing,0,sizeof g_timing); double begin=seconds_now(), t=begin; size_t ignored=0;
        id<MTLBuffer> bs=make_coded_buffer(g_dev,streams,nbytes,off,nstr);
        g_timing[0]=seconds_now()-t; g_timing[7]=bs.contents==streams; t=seconds_now();
        id<MTLBuffer> bm=make_buffer_for_ptr(g_dev,sm,sm_bytes,&ignored);
        id<MTLBuffer> bo=make_buffer_for_ptr(g_dev,off,(size_t)nstr*16,&ignored);
        id<MTLBuffer> bh=hdr ? make_buffer_for_ptr(g_dev,hdr,HEADER,&ignored) : nil;
        g_timing[1]=seconds_now()-t; t=seconds_now();
        BOOL aligned_out=out && (uintptr_t)out % sysconf(_SC_PAGESIZE)==0 && out_capacity % sysconf(_SC_PAGESIZE)==0;
        id<MTLBuffer> dst=aligned_out ? make_buffer_for_ptr(g_dev,out,out_capacity,&ignored) :
            [g_dev newBufferWithLength:obytes options:MTLResourceStorageModeShared];
        if (!bm || !g_pso_bf16 || !bs || !bo || !dst || (hdr && !bh)) return fail_msg(LMZ_GPU_ECUDA,"Metal allocation failed");
        g_timing[2]=seconds_now()-t; g_timing[8]=!out || dst.contents==out;
        if (!hdr) tune_perstream(bs,bo,dst,nstr,plane);
        struct Params P={nstr,plane};
        id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso_bf16];
        [enc setBuffer:bs offset:0 atIndex:0]; [enc setBuffer:bo offset:0 atIndex:1];
        [enc setBuffer:bm offset:0 atIndex:2]; [enc setBuffer:dst offset:0 atIndex:3];
        [enc setBuffer:bh offset:0 atIndex:4]; [enc setBytes:&P length:sizeof P atIndex:5];
        unsigned grid=hdr ? (nstr+15)/16 : perstream_grid(nstr);
        [enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(hdr?128:32,1,1)];
        [enc endEncoding]; t=seconds_now(); [cb commit]; [cb waitUntilCompleted];
        g_timing[4]=seconds_now()-t; g_timing[3]=cb.GPUEndTime-cb.GPUStartTime;
        if (cb.error) return fail_msg(LMZ_GPU_ECUDA,cb.error.localizedDescription.UTF8String);
        t=seconds_now();
        if (out && dst.contents!=out) memcpy(out,dst.contents,obytes);
        g_timing[5]=seconds_now()-t;
        if (!out) {
            if (!owned || !contents) return fail_msg(LMZ_GPU_EUNSUPPORTED,"owned output needs handles");
            *contents=dst.contents; *owned=(__bridge_retained void *)dst;
        } else if (contents) *contents=out;
        g_timing[6]=seconds_now()-begin;
        return LMZ_GPU_OK;
    }
}
LMZ_GPU_API int lmz_gpu_decode_batch(const void *hdr,const void *streams,size_t nbytes,
                    const void *off,unsigned nstr,unsigned plane,void *out) {
    return lmz_gpu_metal_decode(hdr,streams,nbytes,off,nstr,plane,out,(size_t)nstr*plane,NULL,NULL);
}


// Prepared synchronous host bindings retain buffer objects between calls.
// Preparation performs one validated decode, then reuses its output allocation.
LMZ_GPU_API void *lmz_gpu_metal_prepare(const void *hdr,const void *streams,size_t nbytes,
 const void *off,unsigned nstr,unsigned plane,const void *sm,size_t sm_bytes,void *out,size_t capacity) {
    if (!nstr) return NULL;
    size_t page=sysconf(_SC_PAGESIZE);
    if (out && ((uintptr_t)out%page || capacity%page)) { fail_msg(LMZ_GPU_EUNSUPPORTED,"prepared output must span whole aligned pages"); return NULL; }
    @autoreleasepool {
        void *owned=NULL,*ptr=NULL;
        t_tune_requested = 1;       // preparing a batch is the explicit request to tune it
        int rc=sm ? lmz_gpu_metal_decode_bf16(hdr,streams,nbytes,off,nstr,plane,sm,sm_bytes,out,capacity,&owned,&ptr) :
                    lmz_gpu_metal_decode(hdr,streams,nbytes,off,nstr,plane,out,capacity,&owned,&ptr);
        t_tune_requested = 0;
        if (rc) return NULL;
        size_t ignored=0;
        id<MTLBuffer> dst=owned ? (__bridge_transfer id<MTLBuffer>)owned : make_buffer_for_ptr(g_dev,out,capacity,&ignored);
        id<MTLBuffer> bs=make_coded_buffer(g_dev,streams,nbytes,off,nstr), bo=make_buffer_for_ptr(g_dev,off,(size_t)nstr*16,&ignored);
        id<MTLBuffer> bh=hdr?make_buffer_for_ptr(g_dev,hdr,HEADER,&ignored):nil;
        id<MTLBuffer> bm=sm?make_buffer_for_ptr(g_dev,sm,sm_bytes,&ignored):nil;
        if (!dst || !bs || !bo || (hdr && !bh) || (sm && !bm)) return NULL;
        NSDictionary *ctx=@{@"src":bs,@"off":bo,@"dst":dst,@"hdr":bh?:[NSNull null],@"sm":bm?:[NSNull null],@"n":@(nstr),@"plane":@(plane)};
        return (__bridge_retained void *)ctx;
    }
}
LMZ_GPU_API void lmz_gpu_metal_release_prepared(void *ctx) { if (ctx) { id obj=(__bridge_transfer id)ctx; (void)obj; } }
LMZ_GPU_API void *lmz_gpu_metal_prepared_contents(void *ctx) { return [((__bridge NSDictionary *)ctx)[@"dst"] contents]; }
LMZ_GPU_API int lmz_gpu_metal_run_prepared(void *handle) {
    @autoreleasepool {
        NSDictionary *ctx=(__bridge NSDictionary *)handle;
        id<MTLBuffer> bs=ctx[@"src"],bo=ctx[@"off"],dst=ctx[@"dst"];
        BOOL shared=ctx[@"hdr"]!=[NSNull null], fused=ctx[@"sm"]!=[NSNull null];
        struct Params P={[ctx[@"n"] unsignedIntValue],[ctx[@"plane"] unsignedIntValue]};
        memset(g_timing,0,sizeof g_timing); double begin=seconds_now();
        id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:fused?g_pso_bf16:(shared?g_pso_shared:(atomic_load(&g_stride_verified)?g_pso_perstream:g_pso_perstream_old))];
        [enc setBuffer:bs offset:0 atIndex:0]; [enc setBuffer:bo offset:0 atIndex:1];
        if (fused) {
            [enc setBuffer:ctx[@"sm"] offset:0 atIndex:2]; [enc setBuffer:dst offset:0 atIndex:3];
            [enc setBuffer:ctx[@"hdr"] offset:0 atIndex:4]; [enc setBytes:&P length:sizeof P atIndex:5];
        } else {
            [enc setBuffer:dst offset:0 atIndex:2];
            if (shared) { [enc setBuffer:ctx[@"hdr"] offset:0 atIndex:3]; [enc setBytes:&P length:sizeof P atIndex:4]; }
            else [enc setBytes:&P length:sizeof P atIndex:3];
        }
        unsigned grid=shared?(P.nstr+15)/16:perstream_grid(P.nstr);
        [enc dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(shared?128:32,1,1)];
        [enc endEncoding]; g_timing[1]=seconds_now()-begin;
        double wait_start=seconds_now(); [cb commit]; [cb waitUntilCompleted];
        g_timing[4]=seconds_now()-wait_start;
        g_timing[3]=cb.GPUEndTime-cb.GPUStartTime;g_timing[6]=seconds_now()-begin;g_timing[7]=g_timing[8]=1;
        return cb.error ? fail_msg(LMZ_GPU_ECUDA,cb.error.localizedDescription.UTF8String) : LMZ_GPU_OK;
    }
}

LMZ_GPU_API int lmz_gpu_dequant_q8_0_to_fp16(const void *blocks, void *out, unsigned nblocks)
{
    if (nblocks == 0) return LMZ_GPU_OK;
    int init_rc = ensure_init();
    if (init_rc != LMZ_GPU_OK) return init_rc;
    if (!g_pso_dequant_q8) return fail_msg(LMZ_GPU_EUNSUPPORTED, "dequant pipeline not available");

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (!cb) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate command buffer");
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate compute encoder");

        size_t off_b = 0, off_o = 0;
        id<MTLBuffer> b_in = make_buffer_for_ptr(g_dev, blocks, (size_t)nblocks * 34, &off_b);
        id<MTLBuffer> b_out = make_buffer_for_ptr(g_dev, out, (size_t)nblocks * 32 * sizeof(uint16_t), &off_o);

        [enc setComputePipelineState:g_pso_dequant_q8];
        [enc setBuffer:b_in offset:off_b atIndex:0];
        [enc setBuffer:b_out offset:off_o atIndex:1];
        [enc setBytes:&nblocks length:sizeof(nblocks) atIndex:2];

        uint32_t tg_size = 128;
        uint32_t tgs = (nblocks + tg_size - 1) / tg_size;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if ([cb error]) {
            snprintf(g_err, sizeof(g_err), "Metal dequant execution error: %s",
                     [[[cb error] localizedDescription] UTF8String]);
            return LMZ_GPU_ECUDA;
        }
        if ([b_out contents] != out) {
            memcpy(out, (const uint8_t *)[b_out contents] + off_o, (size_t)nblocks * 32 * sizeof(uint16_t));
        }
        return LMZ_GPU_OK;
    }
}

LMZ_GPU_API int lmz_gpu_project_fp8_to_bf16(const void *in, void *out, unsigned nelem)
{
    if (nelem == 0) return LMZ_GPU_OK;
    int init_rc = ensure_init();
    if (init_rc != LMZ_GPU_OK) return init_rc;
    if (!g_pso_project_fp8) return fail_msg(LMZ_GPU_EUNSUPPORTED, "project fp8 pipeline not available");

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (!cb) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate command buffer");
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return fail_msg(LMZ_GPU_ECUDA, "failed to allocate compute encoder");

        size_t off_i = 0, off_o = 0;
        id<MTLBuffer> b_in = make_buffer_for_ptr(g_dev, in, (size_t)nelem, &off_i);
        id<MTLBuffer> b_out = make_buffer_for_ptr(g_dev, out, (size_t)nelem * sizeof(uint16_t), &off_o);

        [enc setComputePipelineState:g_pso_project_fp8];
        [enc setBuffer:b_in offset:off_i atIndex:0];
        [enc setBuffer:b_out offset:off_o atIndex:1];
        [enc setBytes:&nelem length:sizeof(nelem) atIndex:2];

        uint32_t tg_size = 256;
        uint32_t tgs = (nelem + tg_size - 1) / tg_size;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
        if ([cb error]) {
            snprintf(g_err, sizeof(g_err), "Metal project fp8 execution error: %s",
                     [[[cb error] localizedDescription] UTF8String]);
            return LMZ_GPU_ECUDA;
        }
        if ([b_out contents] != out) {
            memcpy(out, (const uint8_t *)[b_out contents] + off_o, (size_t)nelem * sizeof(uint16_t));
        }
        return LMZ_GPU_OK;
    }
}

