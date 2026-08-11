// lmz rANS decode on the GPU, optimised.
//
// The naive port gives one thread a whole stream. That is wrong twice over:
// lmz's 8 interleaved states are independent, so a thread serialises work that
// wants to be parallel; and every access is uncoalesced, because thread t
// touches byte t*PLANE + i while its warp neighbours are 32 KiB away.
//
// The fix maps the 8 states onto 8 LANES. The obstacle is that all 8 states
// share ONE input cursor and refill in strict k order, so lane k must know how
// many bytes lanes 0..k-1 consumed. But whether a state needs a refill depends
// only on its own arithmetic, never on the cursor -- so all 8 lanes can decide
// independently, ballot, and take an exclusive prefix sum with popc. One
// warp instruction replaces the serial cursor walk.
//
//   need_k   = (x_k < L)                       -- independent per lane
//   gmask    = ballot(need) for this lane's group of 8
//   before_k = popc(gmask & ((1<<k)-1))        -- bytes taken by lanes 0..k-1
//   my word  = *(ptr + 2*before_k)
//   ptr     += 2*popc(gmask)                   -- same for all 8 lanes
//
// Lanes 0..7 then write output bytes i..i+7, which is contiguous.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8
#define HEADER 516
#define STAGE_ITERS 16                     // 16 iters x 8 bytes = 128 B per group

#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); exit(1);} } while (0)

__device__ __forceinline__ void build_lut(const uint8_t *src, uint32_t *lut, int nth, int me)
{
    // Cooperative build: each thread takes a slice of the 256 symbols.
    // start offsets need a serial scan, so thread 0 does the scan cheaply
    // into shared and everyone fills.
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

// ---- V0 reference: one thread per stream, shared LUT (the previous best) ---
__global__ void k_thread_per_stream(const uint8_t *streams, const uint64_t *off,
                                    uint32_t nstr, uint32_t plane, uint8_t *out,
                                    const uint8_t *shdr)
{
    extern __shared__ uint32_t lut[];
    build_lut(shdr, lut, blockDim.x, threadIdx.x);
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nstr) return;
    const uint8_t *ptr = streams + off[2 * t];
    uint8_t *dst = out + (size_t)t * plane;
    uint32_t st[NST];
#pragma unroll
    for (int k = 0; k < NST; k++) {
        st[k] = (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) |
                ((uint32_t)ptr[2] << 16) | ((uint32_t)ptr[3] << 24);
        ptr += 4;
    }
    for (uint32_t i = 0; i < plane; i += NST) {
        uint32_t e[NST];
#pragma unroll
        for (int k = 0; k < NST; k++) e[k] = lut[st[k] & (PROB_SCALE - 1)];
#pragma unroll
        for (int k = 0; k < NST; k++) dst[i + k] = (uint8_t)(e[k] & 0xff);
#pragma unroll
        for (int k = 0; k < NST; k++) {
            uint32_t sv = (e[k] >> 8) & 0xfff, fq = (e[k] >> 20) + 1;
            uint32_t x = fq * (st[k] >> PROB_BITS) + (st[k] & (PROB_SCALE - 1)) - sv;
            uint32_t w = (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8);
            uint32_t need = (x < RANS_L);
            x = need ? ((x << 16) | w) : x;
            ptr += (need << 1);
            st[k] = x;
        }
    }
}

// ---- V1: 8 lanes per stream, ballot cursor, direct byte stores ------------
__global__ void k_lane_per_state(const uint8_t *streams, const uint64_t *off,
                                 uint32_t nstr, uint32_t plane, uint8_t *out,
                                 const uint8_t *shdr)
{
    extern __shared__ uint32_t lut[];
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3;                 // one group of 8 lanes per stream
    uint32_t k = tid & 7;
    uint32_t lane = threadIdx.x & 31;
    uint32_t shift = (lane & ~7u);           // group's byte within the ballot
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    for (uint32_t i = 0; i < plane; i += NST) {
        uint32_t e = lut[st & (PROB_SCALE - 1)];
        dst[i + k] = (uint8_t)(e & 0xff);
        uint32_t sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
        uint32_t x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
        uint32_t need = (x < RANS_L);
        uint32_t gmask = (__ballot_sync(0xffffffffu, need) >> shift) & 0xffu;
        const uint8_t *p = ptr + 2 * __popc(gmask & ((1u << k) - 1));
        uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
        st = need ? ((x << 16) | w) : x;
        ptr += 2 * __popc(gmask);
    }
}

// ---- V2: V1 + output staged in shared memory, flushed as uint4 ------------
__global__ void k_lane_staged(const uint8_t *streams, const uint64_t *off,
                              uint32_t nstr, uint32_t plane, uint8_t *out,
                              const uint8_t *shdr)
{
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3;
    uint32_t k = tid & 7;
    uint32_t lane = threadIdx.x & 31;
    uint32_t shift = (lane & ~7u);
    uint32_t grp_in_blk = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp_in_blk * (NST * STAGE_ITERS);
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            uint32_t e = lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uint8_t)(e & 0xff);
            uint32_t sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint32_t x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            uint32_t need = (x < RANS_L);
            uint32_t gmask = (__ballot_sync(0xffffffffu, need) >> shift) & 0xffu;
            const uint8_t *p = ptr + 2 * __popc(gmask & ((1u << k) - 1));
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
            st = need ? ((x << 16) | w) : x;
            ptr += 2 * __popc(gmask);
        }
        __syncwarp();
        // 8 lanes x 16 B = one contiguous 128 B store per group
        uint4 v = *(const uint4 *)(mystage + k * 16);
        *(uint4 *)(dst + base + k * 16) = v;
        __syncwarp();
    }
}


// ---- V3: V2 + coalesced refill loads, distributed by shuffle -------------
// The 8 words the group may need live at ptr+0,2,..,14 -- contiguous. So lane
// k loads the word at ptr+2k (16 coalesced bytes across the group) and every
// lane then shuffles in the one that belongs to its own prefix position.
// Replaces 8 scattered 2-byte loads with one coalesced 16-byte segment.
__global__ void k_lane_shuffled(const uint8_t *streams, const uint64_t *off,
                                uint32_t nstr, uint32_t plane, uint8_t *out,
                                const uint8_t *shdr)
{
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3;
    uint32_t k = tid & 7;
    uint32_t lane = threadIdx.x & 31;
    uint32_t gbase = lane & ~7u;
    uint32_t grp_in_blk = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp_in_blk * (NST * STAGE_ITERS);
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            uint32_t e = lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uint8_t)(e & 0xff);
            uint32_t sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint32_t x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            uint32_t need = (x < RANS_L);
            uint32_t gmask = (__ballot_sync(0xffffffffu, need) >> gbase) & 0xffu;
            // coalesced: the group reads ptr[0..15] as 8 adjacent 2-byte words
            const uint8_t *q = ptr + 2 * k;
            uint32_t mine = (uint32_t)q[0] | ((uint32_t)q[1] << 8);
            uint32_t before = __popc(gmask & ((1u << k) - 1));
            uint32_t w = __shfl_sync(0xffffffffu, mine, gbase + before);
            st = need ? ((x << 16) | w) : x;
            ptr += 2 * __popc(gmask);
        }
        __syncwarp();
        uint4 v = *(const uint4 *)(mystage + k * 16);
        *(uint4 *)(dst + base + k * 16) = v;
        __syncwarp();
    }
}


// ---- ablations: WRONG OUTPUT, timing only. Each removes one memory path
// from V2 so the remaining cost can be attributed.
template <int MODE>   // 1 = no LUT read, 2 = no input read, 3 = neither
__global__ void k_ablate(const uint8_t *streams, const uint64_t *off,
                         uint32_t nstr, uint32_t plane, uint8_t *out,
                         const uint8_t *shdr)
{
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    build_lut(shdr, lut, blockDim.x, threadIdx.x);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3, k = tid & 7;
    uint32_t lane = threadIdx.x & 31, gbase = lane & ~7u;
    uint8_t *mystage = stage + (threadIdx.x >> 3) * (NST * STAGE_ITERS);
    uint32_t src_id = gid < nstr ? gid : 0;
    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;
    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            uint32_t e = (MODE & 1) ? (0x00301000u | (st & 0xff)) : lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uint8_t)(e & 0xff);
            uint32_t sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint32_t x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            uint32_t need = (x < RANS_L);
            uint32_t gmask = (__ballot_sync(0xffffffffu, need) >> gbase) & 0xffu;
            uint32_t w;
            if (MODE & 2) { w = 0xABCDu; }
            else { const uint8_t *p = ptr + 2 * __popc(gmask & ((1u << k) - 1));
                   w = (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
            st = need ? ((x << 16) | w) : x;
            ptr += 2 * __popc(gmask);
        }
        __syncwarp();
        uint4 v = *(const uint4 *)(mystage + k * 16);
        *(uint4 *)(dst + base + k * 16) = v;
        __syncwarp();
    }
}


// ---- V4: V2 + the stream prefetched into shared memory --------------------
// The ablation says the dependent 2-byte global load is 64% of the runtime,
// and coalescing it (V3) changed nothing -- so it is latency on the critical
// path, not transactions. Bulk-load BUFB bytes per group cooperatively (8
// lanes x uint4, fully coalesced, independent of the state chain) and let the
// refill read from shared. Circular, so the half being refilled is always the
// half already consumed.
template <int BUFB>
__global__ void k_lane_prefetch(const uint8_t *streams, const uint64_t *off,
                                uint32_t nstr, uint32_t plane, uint8_t *out,
                                const uint8_t *shdr)
{
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    uint32_t ngrp = blockDim.x >> 3;
    uint8_t *inbuf = stage + ngrp * (NST * STAGE_ITERS);
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3, k = tid & 7;
    uint32_t lane = threadIdx.x & 31, gbase = lane & ~7u;
    uint32_t gmask_warp = 0xffu << gbase;
    uint32_t grp = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp * (NST * STAGE_ITERS);
    uint8_t *mybuf = inbuf + grp * BUFB;
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;                       // 16-byte aligned: streams are padded

    const uint32_t HALF = BUFB / 2;
    uint32_t filled = 0, consumed = 0;
    // prime both halves
#pragma unroll
    for (int h = 0; h < 2; h++) {
#pragma unroll
        for (uint32_t u = 0; u < HALF / (NST * 16); u++) {
            uint32_t o2 = filled + u * (NST * 16) + k * 16;
            *(uint4 *)(mybuf + (o2 & (BUFB - 1))) = *(const uint4 *)(ptr + o2);
        }
        filled += HALF;
    }
    __syncwarp(gmask_warp);

    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            if (filled - consumed < HALF) {          // uniform across the group
                __syncwarp(gmask_warp);
#pragma unroll
                for (uint32_t u = 0; u < HALF / (NST * 16); u++) {
                    uint32_t o2 = filled + u * (NST * 16) + k * 16;
                    *(uint4 *)(mybuf + (o2 & (BUFB - 1))) = *(const uint4 *)(ptr + o2);
                }
                filled += HALF;
                __syncwarp(gmask_warp);
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
        uint4 vv = *(const uint4 *)(mystage + k * 16);
        *(uint4 *)(dst + base + k * 16) = vv;
        __syncwarp();
    }
}


// ---- V5: V4 with the prefetch made ASYNCHRONOUS (cp.async) ----------------
// V4 still stalls the group while its bulk load lands. cp.async issues the
// copy and returns, so the copy for half N+1 is in flight during the whole of
// half N's decoding -- a full HALF bytes of work to cover the latency. The
// group only ever waits at a half boundary, and by then the data has had
// ~HALF/16 iterations to arrive.
template <int BUFB>
__global__ void k_lane_async(const uint8_t *streams, const uint64_t *off,
                             uint32_t nstr, uint32_t plane, uint8_t *out,
                             const uint8_t *shdr)
{
    // FOUR slots, TWO copies in flight. With only two slots the consumer can
    // reach bytes whose copy has not landed: after waiting, ready == filled,
    // and the refill trigger lets `consumed` run all the way to `filled`, so a
    // word straddling the boundary reads in-flight memory. Four slots give a
    // whole quarter of slack on both the read edge and the overwrite edge:
    //   ready    = filled - Q          (one issue may still be outstanding)
    //   trigger  at consumed >= filled - 2Q  = ready - Q   -> read margin Q
    //   overwrite [filled-4Q, filled-3Q)                   -> consumed is past it
    extern __shared__ uint32_t smem[];
    uint32_t *lut = smem;
    uint8_t *stage = (uint8_t *)(smem + PROB_SCALE);
    uint32_t ngrp = blockDim.x >> 3;
    uint8_t *inbuf = stage + ngrp * (NST * STAGE_ITERS);
    build_lut(shdr, lut, blockDim.x, threadIdx.x);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gid = tid >> 3, k = tid & 7;
    uint32_t lane = threadIdx.x & 31, gbase = lane & ~7u;
    uint32_t wmask = 0xffu << gbase;
    uint32_t grp = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp * (NST * STAGE_ITERS);
    uint8_t *mybuf = inbuf + grp * BUFB;
    uint32_t src_id = gid < nstr ? gid : 0;

    const uint8_t *ptr = streams + off[2 * src_id];
    uint8_t *dst = out + (size_t)gid * plane;
    uint32_t st = (uint32_t)ptr[4 * k] | ((uint32_t)ptr[4 * k + 1] << 8) |
                  ((uint32_t)ptr[4 * k + 2] << 16) | ((uint32_t)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    const uint32_t Q = BUFB / 4;
    const uint32_t ROUNDS = Q / (NST * 16);
    uint32_t filled = 0, consumed = 0;

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

    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            if (filled - consumed <= 2 * Q) {
                // MUST be wait_prior(0), not (1). With (1) the batch we are
                // about to read may still be outstanding -- ready is only
                // filled-2Q, not filled-Q -- and a word straddling the slot
                // boundary reads memory in flight. That raced on the
                // sign+mantissa plane (~8 bits/symbol, so ~50% of symbols
                // refill) while passing on the exponent plane (~2.8 bits,
                // ~17%). Overlap is preserved anyway: the copy issued here
                // gets a full 2Q of consumption before the next wait.
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
        uint4 vv = *(const uint4 *)(mystage + k * 16);
        *(uint4 *)(dst + base + k * 16) = vv;
        __syncwarp();
    }
#undef ISSUE
}

static std::vector<uint8_t> slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f); *n = sz; return v;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    char pS[512], pR[512];
    snprintf(pS, sizeof pS, "%s/streams.bin", dir);
    snprintf(pR, sizeof pR, "%s/ref.bin", dir);
    size_t ns, nr;
    std::vector<uint8_t> blob = slurp(pS, &ns);
    std::vector<uint8_t> ref = slurp(pR, &nr);
    uint32_t nstr, plane;
    memcpy(&nstr, blob.data(), 4);
    memcpy(&plane, blob.data() + 4, 4);
    const uint8_t *shdr = blob.data() + 8;                       // 516-byte shared table
    const uint64_t *offs = (const uint64_t *)(blob.data() + 8 + 516);
    const uint8_t *sdata = blob.data() + 8 + 516 + (size_t)nstr * 16;
    size_t sbytes = ns - (8 + 516 + (size_t)nstr * 16);

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s  sm_%d%d  %d SMs  shared/block %zu KiB\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.sharedMemPerBlockOptin / 1024);

    std::vector<uint64_t> rep(offs, offs + (size_t)nstr * 2);   // real, distinct
    size_t padded = ((size_t)nstr + 31) / 32 * 32;      // whole blocks of 32 groups
    size_t obytes = padded * plane;
    const std::vector<uint8_t> &refrep = ref;
    printf("workload %u DISTINCT streams x %u B = %.1f MB out, %.1f MB in\n"
           "memory-bandwidth floor for %.2f GB of traffic: %.2f ms\n\n", nstr, plane,
           (double)nstr*plane/1e6, (double)sbytes/1e6,
           ((double)nstr*plane + sbytes)/1e9,
           ((double)nstr*plane + sbytes)/960e9*1e3);

    uint8_t *d_streams, *d_out, *d_hdr; uint64_t *d_off;
    CK(cudaMalloc(&d_streams, sbytes + 64));            // slack: the fast path
    CK(cudaMemset(d_streams + sbytes, 0, 64));          // reads 2 B past the end
    CK(cudaMalloc(&d_off, (size_t)nstr * 16));
    CK(cudaMalloc(&d_out, obytes));
    CK(cudaMemcpy(d_streams, sdata, sbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hdr, 516));
    CK(cudaMemcpy(d_hdr, shdr, 516, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off, rep.data(), (size_t)nstr * 16, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<uint8_t> host((size_t)nstr * plane);
    double useful = (double)nstr * plane;

    auto run = [&](const char *name, int kind, int tpb) {
        size_t shm = PROB_SCALE * 4;
        if (kind >= 2) shm += (tpb / 8) * NST * STAGE_ITERS;
        if (kind == 7) shm += (tpb / 8) * 256;
        if (kind == 8) shm += (tpb / 8) * 512;
        if (kind == 9) shm += (tpb / 8) * 1024;
        if (kind == 10) shm += (tpb / 8) * 512;
        if (kind == 11) shm += (tpb / 8) * 1024;
        bool ablation = kind >= 4 && kind <= 6;
        int groups_per_blk = (kind == 0) ? tpb : tpb / 8;
        int blocks = (int)((padded + groups_per_blk - 1) / groups_per_blk);
        if (kind == 0) blocks = (int)((nstr + tpb - 1) / tpb);
        auto launch = [&]() {
            if (kind == 0) k_thread_per_stream<<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 1) k_lane_per_state<<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 2) k_lane_staged<<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 3) k_lane_shuffled<<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 4) k_ablate<1><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 5) k_ablate<2><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 6) k_ablate<3><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 7) k_lane_prefetch<256><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 8) k_lane_prefetch<512><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 9) k_lane_prefetch<1024><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else if (kind == 10) k_lane_async<512><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
            else k_lane_async<1024><<<blocks, tpb, shm>>>(d_streams, d_off, nstr, plane, d_out, d_hdr);
        };
        CK(cudaMemset(d_out, 0, obytes));
        launch(); CK(cudaDeviceSynchronize());
        cudaError_t le = cudaGetLastError();
        if (le) { printf("%-34s tpb %3d  LAUNCH FAIL %s\n", name, tpb, cudaGetErrorString(le)); return; }
        float best = 1e30f;
        for (int r = 0; r < 7; r++) {
            CK(cudaEventRecord(e0)); launch(); CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        CK(cudaMemcpy(host.data(), d_out, (size_t)nstr * plane, cudaMemcpyDeviceToHost));
        bool ok = memcmp(host.data(), refrep.data(), (size_t)nstr * plane) == 0;
        printf("%-34s tpb %3d  %7.2f ms  %7.1f GB/s  %s\n", name, tpb, best,
               useful / (best * 1e-3) / 1e9,
               ablation ? "(ablation: output invalid by design)"
                        : (ok ? "byte-identical" : "*** MISMATCH ***"));
    };

    printf("%-34s %7s  %7s  %9s\n", "kernel", "tpb", "time", "decode");
    for (int tpb : {128, 256}) run("V0 thread/stream (previous best)", 0, tpb);
    for (int tpb : {128, 256, 512}) run("V1 8 lanes/stream, ballot cursor", 1, tpb);
    for (int tpb : {128, 256, 512}) run("V2 V1 + staged uint4 stores", 2, tpb);
    for (int tpb : {128, 256, 512}) run("V3 V2 + coalesced+shuffled refill", 3, tpb);
    printf("\n-- ablations on V2 @ tpb 256 (attribute the remaining time) --\n");
    run("  V2 without the LUT read", 4, 256);
    run("  V2 without the input read", 5, 256);
    run("  V2 with neither (stores only)", 6, 256);
    printf("\n-- V4: stream prefetched into shared memory --\n");
    for (int tpb : {128, 256}) run("V4 prefetch, 256 B/group", 7, tpb);
    for (int tpb : {128, 256}) run("V4 prefetch, 512 B/group", 8, tpb);
    for (int tpb : {128, 256}) run("V4 prefetch, 1024 B/group", 9, tpb);
    printf("\n-- V5: asynchronous prefetch (cp.async) --\n");
    for (int tpb : {128, 256, 384}) run("V5 cp.async 4-slot, 512 B/grp", 10, tpb);
    for (int tpb : {128, 256}) run("V5 cp.async 4-slot, 1024 B/grp", 11, tpb);
    return 0;
}
