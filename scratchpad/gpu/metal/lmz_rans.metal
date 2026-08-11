// lmz's rANS decoder for Apple GPUs, ported from the CUDA kernel that was
// verified byte-identical on an RTX 5080 (gpu_rans10.cu / gpu_fused.cu).
//
// The idea that made the CUDA version fast transfers unchanged: lmz's 8
// interleaved rANS states share ONE input cursor and refill in strict k order,
// but `need = (x < L)` depends only on the state, never on the cursor. So map
// the 8 states onto 8 LANES, ballot the refill flags, and take an exclusive
// prefix sum with popcount to get each lane's byte offset. One SIMD
// instruction replaces the serial cursor walk.
//
//   CUDA                        Metal
//   __ballot_sync(full, p)      simd_ballot(p)
//   __popc                      popcount
//   __shfl_sync                 simd_shuffle
//   __syncwarp(mask)            simdgroup_barrier(mem_flags::mem_threadgroup)
//   __shared__                  threadgroup
//   cp.async                    (none on M1 -- PREFETCH below is synchronous)
//
// Threadgroup memory on Apple GPUs is 32 KiB, so TG is fixed at 128 threads:
//   LUT 16384 + starts 1024 + stage 2048 + inbuf 4096 = 23.5 KiB.
// A 256-thread group would need 44 KiB and will not launch.
#include <metal_stdlib>
using namespace metal;

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8                 // rANS states, and lanes per stream
#define TG 128                // threads per threadgroup
#define GROUPS (TG / NST)     // streams in flight per threadgroup
#define STAGE_ITERS 16        // 16 iters x 8 bytes = 128 B staged per group
#define BUFB 256              // circular input buffer per group
#define HALFB (BUFB / 2)

struct Params {
    uint nstr;
    uint plane;
};

// Build the shared decode table cooperatively. Frequencies come from the
// 516-byte lmz header; the packed entry is ((freq-1)<<20)|(start<<8)|symbol,
// so a decode step is one dependent load.
static void build_lut(device const uchar *hdr,
                      threadgroup uint *lut,
                      threadgroup uint *starts,
                      uint tid)
{
    if (tid == 0) {
        uint s = 0;
        for (uint i = 0; i < 256; i++) {
            starts[i] = s;
            s += (uint)hdr[4 + 2 * i] | ((uint)hdr[5 + 2 * i] << 8);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 256; i += TG) {
        uint f = (uint)hdr[4 + 2 * i] | ((uint)hdr[5 + 2 * i] << 8);
        uint st = starts[i];
        uint packed = ((f ? f - 1u : 0u) << 20) | (st << 8) | i;
        for (uint j = 0; j < f; j++) lut[st + j] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------- plane only
// PREFETCH == 0 reads refill bytes straight from device memory. No divergent
// barrier anywhere, so it is correct by construction -- use it as the oracle.
// PREFETCH == 1 stages the stream in threadgroup memory first, which was worth
// 1.44x on the RTX. It relies on simdgroup_barrier being a memory fence over a
// SIMD-group that already runs in lockstep; if Apple's scheduler ever diverges
// a group, that assumption breaks, which is exactly what the verify step in
// the host harness is there to catch.
template <bool PREFETCH>
static void decode_plane(device const uchar *streams,
                         device const ulong *off,
                         device uchar *out,
                         threadgroup uint *lut,
                         threadgroup uchar *stage,
                         threadgroup uchar *inbuf,
                         uint gid, uint k, uint lane, uint grp,
                         uint plane)
{
    device const uchar *ptr = streams + off[2 * gid];
    device uchar *dst = out + (ulong)gid * plane;
    threadgroup uchar *mystage = stage + grp * (NST * STAGE_ITERS);
    threadgroup uchar *mybuf = inbuf + grp * BUFB;
    uint gbase = lane & ~7u;

    uint st = (uint)ptr[4 * k] | ((uint)ptr[4 * k + 1] << 8) |
              ((uint)ptr[4 * k + 2] << 16) | ((uint)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    uint filled = 0, consumed = 0;
    if (PREFETCH) {
        // prime both halves: 8 lanes x 16 B = 128 B = HALFB per round
        for (uint h = 0; h < 2; h++) {
            *(threadgroup packed_uint4 *)(mybuf + ((filled + k * 16) & (BUFB - 1))) =
                *(device const packed_uint4 *)(ptr + filled + k * 16);
            filled += HALFB;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint base = 0; base < plane; base += NST * STAGE_ITERS) {
        for (uint s = 0; s < STAGE_ITERS; s++) {
            if (PREFETCH && (filled - consumed < HALFB)) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
                *(threadgroup packed_uint4 *)(mybuf + ((filled + k * 16) & (BUFB - 1))) =
                    *(device const packed_uint4 *)(ptr + filled + k * 16);
                filled += HALFB;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            uint e = lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uchar)(e & 0xff);
            uint sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            bool need = (x < RANS_L);
            uint ball = (uint)((simd_vote::vote_t)simd_ballot(need));
            uint gm = (ball >> gbase) & 0xffu;
            uint before = popcount(gm & ((1u << k) - 1u));
            uint w;
            if (PREFETCH) {
                uint o = consumed + 2 * before;
                w = (uint)mybuf[o & (BUFB - 1)] |
                    ((uint)mybuf[(o + 1) & (BUFB - 1)] << 8);
            } else {
                device const uchar *p = ptr + consumed + 2 * before;
                w = (uint)p[0] | ((uint)p[1] << 8);
            }
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * popcount(gm);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // 8 lanes x 16 B = one contiguous 128 B store per group
        *(device packed_uint4 *)(dst + base + k * 16) =
            *(threadgroup const packed_uint4 *)(mystage + k * 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void lmz_decode_plane_direct(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device uchar *out             [[buffer(2)]],
    device const uchar *hdr       [[buffer(3)]],
    constant Params &P            [[buffer(4)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    threadgroup uint lut[PROB_SCALE];
    threadgroup uint starts[256];
    threadgroup uchar stage[GROUPS * NST * STAGE_ITERS];
    threadgroup uchar inbuf[GROUPS * BUFB];
    build_lut(hdr, lut, starts, tid_t);
    uint gid = tid_g >> 3;
    if (gid >= P.nstr) return;
    decode_plane<false>(streams, off, out, lut, stage, inbuf,
                        gid, tid_g & 7, lane, tid_t >> 3, P.plane);
}

kernel void lmz_decode_plane_prefetch(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device uchar *out             [[buffer(2)]],
    device const uchar *hdr       [[buffer(3)]],
    constant Params &P            [[buffer(4)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    threadgroup uint lut[PROB_SCALE];
    threadgroup uint starts[256];
    threadgroup uchar stage[GROUPS * NST * STAGE_ITERS];
    threadgroup uchar inbuf[GROUPS * BUFB];
    build_lut(hdr, lut, starts, tid_t);
    uint gid = tid_g >> 3;
    if (gid >= P.nstr) return;
    decode_plane<true>(streams, off, out, lut, stage, inbuf,
                       gid, tid_g & 7, lane, tid_t >> 3, P.plane);
}

// -------------------------------------------------------------- fused BF16
// Decode the exponent plane, read the (incompressible, stored-raw) sign+
// mantissa plane, and write bit-interleaved BF16 using lmz's own merge:
//     w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F)
kernel void lmz_decode_bf16(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device const uchar *smplane   [[buffer(2)]],
    device uchar *out             [[buffer(3)]],
    device const uchar *hdr       [[buffer(4)]],
    constant Params &P            [[buffer(5)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    threadgroup uint lut[PROB_SCALE];
    threadgroup uint starts[256];
    threadgroup uchar stage[GROUPS * NST * STAGE_ITERS];
    build_lut(hdr, lut, starts, tid_t);

    uint gid = tid_g >> 3;
    if (gid >= P.nstr) return;
    uint k = tid_g & 7, grp = tid_t >> 3, gbase = lane & ~7u;
    threadgroup uchar *mystage = stage + grp * (NST * STAGE_ITERS);

    device const uchar *ptr = streams + off[2 * gid];
    ulong ebase = (ulong)gid * P.plane;
    uint st = (uint)ptr[4 * k] | ((uint)ptr[4 * k + 1] << 8) |
              ((uint)ptr[4 * k + 2] << 16) | ((uint)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;
    uint consumed = 0;

    for (uint base = 0; base < P.plane; base += NST * STAGE_ITERS) {
        for (uint s = 0; s < STAGE_ITERS; s++) {
            uint e = lut[st & (PROB_SCALE - 1)];
            mystage[s * NST + k] = (uchar)(e & 0xff);
            uint sv = (e >> 8) & 0xfff, fq = (e >> 20) + 1;
            uint x = fq * (st >> PROB_BITS) + (st & (PROB_SCALE - 1)) - sv;
            bool need = (x < RANS_L);
            uint ball = (uint)((simd_vote::vote_t)simd_ballot(need));
            uint gm = (ball >> gbase) & 0xffu;
            device const uchar *p = ptr + consumed +
                                    2 * popcount(gm & ((1u << k) - 1u));
            uint w = (uint)p[0] | ((uint)p[1] << 8);
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * popcount(gm);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        ulong eo = ebase + base + k * 16;
        threadgroup const uchar *ep = mystage + k * 16;
        device const uchar *sp = smplane + eo;
        ushort wv[16];
        for (uint j = 0; j < 16; j++)
            wv[j] = (ushort)((((uint)sp[j] & 0x80u) << 8) |
                             ((uint)ep[j] << 7) | ((uint)sp[j] & 0x7Fu));
        device packed_uint4 *d4 = (device packed_uint4 *)(out + 2 * eo);
        thread const packed_uint4 *s4 = (thread const packed_uint4 *)wv;
        d4[0] = s4[0];
        d4[1] = s4[1];
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
