/*
 * lmz rANS decode on the GPU -- the library form.
 *
 * This is `scratchpad/gpu/cuda/gpu_rans10.cu`'s final kernel promoted out of
 * its benchmark: no main(), no file paths, no timing, and an entry point that
 * takes an ordinary lmz stream rather than a purpose-built blob. The kernel
 * itself is unchanged in substance, because it was verified byte-identical to
 * lmz_rans_decode over 936 MB and 1.87 GB of real planes and there is no
 * reason to re-derive it.
 *
 * WHAT MADE IT FAST, and what a caller must not break: lmz's 8 interleaved
 * rANS states share one input cursor and refill in strict k order, which looks
 * serial. It is not -- whether a state needs a refill depends only on that
 * state, never on the cursor. So the 8 states map onto 8 LANES, and one
 * __ballot_sync plus a popc prefix sum reconstructs every lane's byte offset.
 * That is 7.3x and it needs no format change. Anything that changes the
 * interleave breaks it.
 *
 * TWO TABLE LAYOUTS, because lmz has two. An archive written today carries a
 * 516-byte frequency table per chunk; `hdr == NULL` selects the kernel that
 * builds one compact table per group. A stream set that shares one table
 * across chunks passes that table in `hdr` and gets the faster kernel, whose
 * single LUT lives in shared memory for the whole block. Both decode the same
 * coded bytes -- sharing a table is a question of where the 516 bytes are
 * stored, not of how a symbol is coded -- so this file does not care which
 * one the format ends up preferring.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#if defined(_WIN32)
#define LMZ_GPU_API extern "C" __declspec(dllexport)
#else
#define LMZ_GPU_API extern "C" __attribute__((visibility("default")))
#endif

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8
#define HEADER 516                 /* magic(2) + reserved(2) + 256 * uint16 */
#define STAGE_ITERS 16             /* 16 iters x 8 bytes = 128 B per group */
#define GRAIN (NST * STAGE_ITERS)  /* bytes a group retires per outer step */
#define BUFB 512                   /* cp.async ring, four Q=128 slots */

/* Return codes. Negative is failure; the Python layer falls back to the CPU
 * decoder on every one of them, so none of these is fatal to a decode. */
#define LMZ_GPU_OK            0
#define LMZ_GPU_ENODEV       -1    /* no CUDA device, or the runtime is absent */
#define LMZ_GPU_EUNSUPPORTED -2    /* a shape this kernel does not decode */
#define LMZ_GPU_EBADSTREAM   -3    /* magic or frequency sum is wrong */
#define LMZ_GPU_ECUDA        -4    /* an API call or the launch failed */
#define LMZ_GPU_EMISMATCH    -5    /* reserved: verification failed */

#define ABI_VERSION 1

/*
 * Why a decode declined, in words. Every failure here is one the Python layer
 * turns into "fall back to the CPU" silently, so without this the difference
 * between "no card", "not enough shared memory for this block size" and "the
 * launch was rejected" is invisible -- and those want very different answers.
 * `lmz doctor` prints it.
 */
static __thread char g_err[256];

static int fail_cuda(const char *what, cudaError_t e)
{
    snprintf(g_err, sizeof g_err, "%s: %s", what, cudaGetErrorString(e));
    return LMZ_GPU_ECUDA;
}

static int fail_msg(int rc, const char *what)
{
    snprintf(g_err, sizeof g_err, "%s", what);
    return rc;
}

#define CK(what, x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) return fail_cuda(what, e_); } while (0)

/* ------------------------------------------------------------------ tables */

/*
 * The wide table: one 32-bit entry per probability slot holding frequency,
 * cumulative start and symbol together, so a decode step is a single dependent
 * load. Identical in layout to the `lut` in lmz_rans_decode, frequency biased
 * by one for the same reason -- a stream of one repeated symbol has frequency
 * PROB_SCALE, which does not fit a 12-bit field unbiased.
 *
 * 16 KiB. One per block, so it is only affordable when the whole block shares
 * a table.
 */
__device__ __forceinline__ void build_lut(const uint8_t *src, uint32_t *lut,
                                          int nth, int me)
{
    __shared__ uint32_t starts[256];
    if (me == 0) {
        uint32_t s = 0;
        for (int i = 0; i < 256; i++) {
            starts[i] = s;
            s += (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
        }
    }
    __syncthreads();
    for (int i = me; i < 256; i += nth) {
        uint32_t f = (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
        uint32_t st = starts[i];
        uint32_t packed = ((f ? f - 1 : 0) << 20) | (st << 8) | (uint32_t)i;
        for (uint32_t j = 0; j < f; j++) lut[st + j] = packed;
    }
    __syncthreads();
}

/*
 * The narrow table, for per-chunk frequencies: slot -> symbol as one byte,
 * plus symbol -> (start, freq-1) as one word. 4 KiB + 1 KiB against 16 KiB,
 * which is the whole reason a per-chunk table fits in shared memory at all --
 * at 16 KiB a block affords six groups and 48 threads, at 5 KiB it affords
 * sixteen and 128.
 *
 * The cost is a second dependent shared load on the critical path. That is
 * the trade this kernel exists to make: shared-memory latency for the ability
 * to decode an archive lmz already writes.
 *
 * Built by the group's own 8 lanes. The start offsets need a serial scan, so
 * lane 0 walks the 256 frequencies and the group fills the slots.
 */
__device__ __forceinline__ void build_lut_narrow(const uint8_t *src, uint8_t *sym,
                                                 uint32_t *sf, uint32_t k)
{
    uint32_t gmask = 0xffu << (threadIdx.x & 24u);
    if (k == 0) {
        uint32_t s = 0;
        for (int i = 0; i < 256; i++) {
            uint32_t f = (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
            sf[i] = ((f ? f - 1 : 0) << 12) | s;
            s += f;
        }
    }
    __syncwarp(gmask);
    for (uint32_t i = k; i < 256; i += NST) {
        uint32_t f = (uint32_t)src[4 + 2 * i] | ((uint32_t)src[5 + 2 * i] << 8);
        uint32_t st = sf[i] & 0xfffu;
        for (uint32_t j = 0; j < f; j++) sym[st + j] = (uint8_t)i;
    }
    __syncwarp(gmask);
}

/* ------------------------------------------------------- the shared-table kernel */

/*
 * Four cp.async slots, two copies in flight. With only two the consumer can
 * reach bytes whose copy has not landed, and the failure does not show on the
 * exponent plane -- it takes the sign+mantissa plane's 3x higher refill rate
 * to reach the boundary. This is the exact shape of a bug that passes its
 * test; do not reduce the slot count.
 */
__global__ void k_shared(const uint8_t *__restrict__ streams,
                         const uint64_t *__restrict__ off,
                         uint32_t nstr, uint32_t plane, uint8_t *__restrict__ out,
                         const uint8_t *__restrict__ shdr)
{
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    uint32_t ngrp = blockDim.x >> 3;
    uint8_t *inbuf = stage + ngrp * GRAIN;
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3, k = tid & 7;
    uint32_t lane = threadIdx.x & 31, gbase = lane & ~7u;
    uint32_t wmask = 0xffu << gbase;
    uint32_t grp = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp * GRAIN;
    uint8_t *mybuf = inbuf + grp * BUFB;
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];   /* table is in shdr */
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    const uint32_t Q = BUFB / 4;
    const uint32_t ROUNDS = Q / (NST * 16);
    /* A 16-byte cp.async needs a 16-byte-aligned SOURCE, and a stream begins
     * wherever the archive put it. Drop the cursor to the boundary below and
     * start the byte count there: ring slot is source offset mod BUFB either
     * way, so the shift costs one add at setup and nothing in the loop. */
    uint32_t skew = (uint32_t)((uintptr_t)ptr & 15u);
    ptr -= skew;
    uint32_t filled = 0, consumed = skew;

#define ISSUE(OFFS) do { \
        _Pragma("unroll") \
        for (uint32_t u = 0; u < ROUNDS; u++) { \
            uint32_t o2 = (OFFS) + u * (NST * 16) + k * 16; \
            __pipeline_memcpy_async(mybuf + (o2 & (BUFB - 1)), ptr + o2, 16); \
        } \
        __pipeline_commit(); } while (0)

    ISSUE(0); ISSUE(Q); ISSUE(2 * Q);
    filled = 3 * Q;
    __pipeline_wait_prior(0);
    __syncwarp(wmask);

    for (uint32_t base = 0; base < plane; base += GRAIN) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            if (filled - consumed <= 2 * Q) {
                /* MUST be wait_prior(0), not (1): with (1) the batch about to
                 * be read may still be outstanding. Overlap survives anyway,
                 * because the copy issued here gets a full 2Q of consumption
                 * before the next wait. */
                __pipeline_wait_prior(0);
                __syncwarp(wmask);
                ISSUE(filled);
                filled += Q;
            }
            uint32_t e = lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uint8_t)(e & 0xff);
            uint32_t sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint32_t x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            uint32_t need = (x < RANS_L);
            uint32_t gm = (__ballot_sync(0xffffffffu, need) >> gbase) & 0xffu;
            uint32_t o = consumed + 2 * __popc(gm & ((1u << k) - 1));
            uint32_t w = (uint32_t)mybuf[o & (BUFB - 1)] |
                         ((uint32_t)mybuf[(o + 1) & (BUFB - 1)] << 8);
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * __popc(gm);
        }
        __syncwarp();
        if (gid < nstr) {
            uint4 vv = *(const uint4 *)(mystage + k * 16);
            *(uint4 *)(dst + base + k * 16) = vv;
        }
        __syncwarp();
    }
#undef ISSUE
}

/* --------------------------------------------------- the per-chunk-table kernel */

/*
 * Same lane trick, same cp.async ring. The differences are the narrow table,
 * built per group from the stream's own header, and the cursor starting
 * HEADER bytes further in.
 */
__global__ void k_perstream(const uint8_t *__restrict__ streams,
                            const uint64_t *__restrict__ off,
                            uint32_t nstr, uint32_t plane, uint8_t *__restrict__ out)
{
    extern __shared__ uint32_t smem[];
    uint32_t ngrp = blockDim.x >> 3;
    uint32_t grp = threadIdx.x >> 3;
    uint8_t *symtab = (uint8_t *)smem;                       /* ngrp * 4096 */
    uint32_t *sftab = (uint32_t *)(symtab + ngrp * PROB_SCALE);  /* ngrp * 256 */
    uint8_t *stage = (uint8_t *)(sftab + ngrp * 256);        /* ngrp * GRAIN */
    uint8_t *inbuf = stage + ngrp * GRAIN;                   /* ngrp * BUFB */

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3, k = tid & 7;
    uint32_t lane = threadIdx.x & 31, gbase = lane & ~7u;
    uint32_t wmask = 0xffu << gbase;
    uint8_t *mysym = symtab + grp * PROB_SCALE;
    uint32_t *mysf = sftab + grp * 256;
    uint8_t *mystage = stage + grp * GRAIN;
    uint8_t *mybuf = inbuf + grp * BUFB;
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *base_ptr = streams + off[2 * src_id];
    build_lut_narrow(base_ptr, mysym, mysf, k);

    const uint8_t *ptr = base_ptr + HEADER;
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    /* See k_shared. Here it is not optional even for a 16-byte-aligned stream:
     * the coded bytes start HEADER + 32 = 548 in, which never is. */
    uint32_t skew = (uint32_t)((uintptr_t)ptr & 15u);
    ptr -= skew;

    const uint32_t Q = BUFB / 4;
    const uint32_t ROUNDS = Q / (NST * 16);
    uint32_t filled = 0, consumed = skew;

#define ISSUE(OFFS) do { \
        _Pragma("unroll") \
        for (uint32_t u = 0; u < ROUNDS; u++) { \
            uint32_t o2 = (OFFS) + u * (NST * 16) + k * 16; \
            __pipeline_memcpy_async(mybuf + (o2 & (BUFB - 1)), ptr + o2, 16); \
        } \
        __pipeline_commit(); } while (0)

    ISSUE(0); ISSUE(Q); ISSUE(2 * Q);
    filled = 3 * Q;
    __pipeline_wait_prior(0);
    __syncwarp(wmask);

    for (uint32_t bpos = 0; bpos < plane; bpos += GRAIN) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            if (filled - consumed <= 2 * Q) {
                __pipeline_wait_prior(0);
                __syncwarp(wmask);
                ISSUE(filled);
                filled += Q;
            }
            uint32_t slot = st & (PROB_SCALE - 1);
            uint32_t sym = mysym[slot];
            uint32_t p = mysf[sym];
            mystage[s * NST + k] = (uint8_t)sym;
            uint32_t sv = p & 0xfffu, fq = (p >> 12) + 1;
            uint32_t x = fq * (st >> PROB_BITS) + slot - sv;
            uint32_t need = (x < RANS_L);
            uint32_t gm = (__ballot_sync(0xffffffffu, need) >> gbase) & 0xffu;
            uint32_t o = consumed + 2 * __popc(gm & ((1u << k) - 1));
            uint32_t w = (uint32_t)mybuf[o & (BUFB - 1)] |
                         ((uint32_t)mybuf[(o + 1) & (BUFB - 1)] << 8);
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * __popc(gm);
        }
        __syncwarp();
        if (gid < nstr) {
            uint4 vv = *(const uint4 *)(mystage + k * 16);
            *(uint4 *)(dst + bpos + k * 16) = vv;
        }
        __syncwarp();
    }
#undef ISSUE
}

/* ------------------------------------------------------------------- host side */

LMZ_GPU_API int lmz_gpu_abi_version(void) { return ABI_VERSION; }

/* Why the last call on this thread declined. Empty if it did not. */
LMZ_GPU_API const char *lmz_gpu_last_error(void) { return g_err; }

/* The shape rules a caller has to satisfy, exported rather than duplicated in
 * Python: `plane` must be a multiple of the grain, and a per-chunk-table
 * stream begins with `header` bytes of frequencies. */
LMZ_GPU_API int lmz_gpu_grain(void) { return GRAIN; }
LMZ_GPU_API int lmz_gpu_header_bytes(void) { return HEADER; }
LMZ_GPU_API int lmz_gpu_pad_bytes(void) { return BUFB + 64; }

LMZ_GPU_API int lmz_gpu_device_name(char *buf, int cap)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1)
        return fail_msg(LMZ_GPU_ENODEV, "no CUDA device");
    cudaDeviceProp prop;
    CK("cudaGetDeviceProperties", cudaGetDeviceProperties(&prop, 0));
    snprintf(buf, (size_t)cap, "%s sm_%d%d %d SMs", prop.name, prop.major,
             prop.minor, prop.multiProcessorCount);
    return LMZ_GPU_OK;
}

/*
 * Is every stream in the batch one this kernel can decode? Checked on the host
 * because the alternative is a device-side early return, and a group that
 * leaves the warp early makes the full-warp __ballot_sync in the decode loop
 * undefined for the groups that stayed.
 */
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

/* Largest thread count whose shared-memory footprint the device will grant,
 * walking down from `want` in whole warps. */
static int pick_tpb(size_t per_group, size_t fixed, int want, size_t optin, size_t *shm_out)
{
    for (int tpb = want; tpb >= 32; tpb -= 32) {
        size_t shm = fixed + (size_t)(tpb / NST) * per_group;
        if (shm <= optin) { *shm_out = shm; return tpb; }
    }
    return 0;
}

/*
 * Decode `nstr` streams that are already in device memory.
 *
 * `d_hdr` is a 516-byte shared frequency table, or NULL when each stream
 * carries its own. `d_off` is 2*nstr uint64 (byte offset, byte length) into
 * `d_streams`; only the offset is read here. Every stream must decode to
 * exactly `plane` bytes and `plane` must be a multiple of lmz_gpu_grain().
 *
 * The caller owns the padding: the kernel prefetches up to lmz_gpu_pad_bytes()
 * past the cursor, so `d_streams` must have that much readable slack after the
 * last stream. Those bytes are never used, only fetched.
 *
 * `tpb` of 0 picks a block size; `cuda_stream` of NULL uses the default stream.
 * This does not synchronise -- the caller decides when the result is needed.
 */
LMZ_GPU_API int lmz_gpu_decode_batch_dev(const void *d_hdr, const void *d_streams,
                                         const void *d_off, unsigned nstr,
                                         unsigned plane, void *d_out,
                                         void *cuda_stream, unsigned tpb)
{
    if (nstr == 0) return LMZ_GPU_OK;
    if (plane == 0 || plane % GRAIN != 0)
        return fail_msg(LMZ_GPU_EUNSUPPORTED, "plane is not a multiple of the grain");

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
        return fail_msg(LMZ_GPU_ENODEV, "no CUDA device");
    cudaDeviceProp prop;
    CK("cudaGetDeviceProperties", cudaGetDeviceProperties(&prop, 0));
    size_t optin = (size_t)prop.sharedMemPerBlockOptin;
    if (optin == 0) optin = (size_t)prop.sharedMemPerBlock;

    cudaStream_t cs = (cudaStream_t)cuda_stream;
    size_t shm = 0;
    int threads;

    if (d_hdr != NULL) {
        /* 384 measured best on an RTX 5080 over 936 MB, 414 GB/s against 403
         * at 256 and 325 at 128. pick_tpb walks down from here in whole warps
         * until the shared-memory request is one the device will grant. */
        threads = pick_tpb(GRAIN + BUFB, (size_t)PROB_SCALE * 4,
                           tpb ? (int)tpb : 384, optin, &shm);
        if (!threads)
            return fail_msg(LMZ_GPU_EUNSUPPORTED, "shared table needs more shared memory than the device grants");
        CK("cudaFuncSetAttribute(k_shared)",
           cudaFuncSetAttribute(k_shared, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)shm));
        unsigned blocks = (unsigned)(((size_t)nstr * NST + threads - 1) / threads);
        k_shared<<<blocks, threads, shm, cs>>>(
            (const uint8_t *)d_streams, (const uint64_t *)d_off, nstr, plane,
            (uint8_t *)d_out, (const uint8_t *)d_hdr);
    } else {
        /* 128 is as far up as a 5.6 KiB-per-group table goes inside the
         * 99 KiB a block may opt into, and it measured 110 GB/s against
         * 111 at 64 -- flat, so take the wider block. */
        threads = pick_tpb((size_t)PROB_SCALE + 256 * 4 + GRAIN + BUFB, 0,
                           tpb ? (int)tpb : 128, optin, &shm);
        if (!threads)
            return fail_msg(LMZ_GPU_EUNSUPPORTED, "per-chunk tables need more shared memory than the device grants");
        CK("cudaFuncSetAttribute(k_perstream)",
           cudaFuncSetAttribute(k_perstream, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)shm));
        unsigned blocks = (unsigned)(((size_t)nstr * NST + threads - 1) / threads);
        k_perstream<<<blocks, threads, shm, cs>>>(
            (const uint8_t *)d_streams, (const uint64_t *)d_off, nstr, plane,
            (uint8_t *)d_out);
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return fail_cuda("launch", le);
    return LMZ_GPU_OK;
}

/*
 * The same decode for host memory: allocate, copy in, decode, copy out.
 *
 * This is the convenience form, and it is bandwidth-bound on PCIe rather than
 * on the kernel -- it moves the coded bytes up and the plain bytes back, so it
 * can never beat the link. It exists so a CPU-side caller and the test suite
 * have one call to make. Anything that wants the decoder's real rate keeps the
 * bytes resident and uses lmz_gpu_decode_batch_dev.
 */
LMZ_GPU_API int lmz_gpu_decode_batch(const void *hdr, const void *streams, size_t nbytes,
                                     const void *off, unsigned nstr, unsigned plane,
                                     void *out)
{
    if (nstr == 0) return LMZ_GPU_OK;
    if (plane == 0 || plane % GRAIN != 0)
        return fail_msg(LMZ_GPU_EUNSUPPORTED, "plane is not a multiple of the grain");

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
        return fail_msg(LMZ_GPU_ENODEV, "no CUDA device");
    if (hdr == NULL) {
        int rc = lmz_gpu_validate(streams, off, nstr);
        if (rc != LMZ_GPU_OK) return rc;
    }

    size_t pad = (size_t)BUFB + 64;
    size_t obytes = (size_t)nstr * plane;
    uint8_t *d_streams = NULL, *d_out = NULL, *d_hdr = NULL;
    uint64_t *d_off = NULL;
    int rc = LMZ_GPU_ECUDA;

#define TRY(what, call) do { cudaError_t e_ = (call); \
        if (e_ != cudaSuccess) { rc = fail_cuda(what, e_); goto done; } } while (0)

    TRY("cudaMalloc(streams)", cudaMalloc(&d_streams, nbytes + pad));
    TRY("cudaMalloc(offsets)", cudaMalloc(&d_off, (size_t)nstr * 16));
    TRY("cudaMalloc(out)", cudaMalloc(&d_out, obytes));
    if (hdr != NULL) TRY("cudaMalloc(table)", cudaMalloc(&d_hdr, HEADER));

    TRY("H2D streams", cudaMemcpy(d_streams, streams, nbytes, cudaMemcpyHostToDevice));
    TRY("memset pad", cudaMemset(d_streams + nbytes, 0, pad));
    TRY("H2D offsets", cudaMemcpy(d_off, off, (size_t)nstr * 16, cudaMemcpyHostToDevice));
    if (hdr != NULL)
        TRY("H2D table", cudaMemcpy(d_hdr, hdr, HEADER, cudaMemcpyHostToDevice));

    rc = lmz_gpu_decode_batch_dev(d_hdr, d_streams, d_off, nstr, plane, d_out, NULL, 0);
    if (rc != LMZ_GPU_OK) goto done;
    TRY("cudaDeviceSynchronize", cudaDeviceSynchronize());
    TRY("D2H out", cudaMemcpy(out, d_out, obytes, cudaMemcpyDeviceToHost));
    rc = LMZ_GPU_OK;
#undef TRY

done:
    if (d_streams) cudaFree(d_streams);
    if (d_off) cudaFree(d_off);
    if (d_out) cudaFree(d_out);
    if (d_hdr) cudaFree(d_hdr);
    return rc;
}
