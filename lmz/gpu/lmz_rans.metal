// lmz's rANS decoder for Apple GPUs (Metal Shading Language).
//
// Supports both shared-frequency-table batch decode and per-stream-table decode.
// The 8 interleaved rANS states map onto 8 lanes, using simd_ballot and popcount
// to reconstruct refills in parallel without serial cursor walks.

#include <metal_stdlib>
using namespace metal;

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8
#define TG_SHARED 128
#define GROUPS_SHARED (TG_SHARED / NST)
#define STAGE_ITERS 16
#define GRAIN (NST * STAGE_ITERS)
#define BUFB 256
#define HALFB (BUFB / 2)
#define HEADER 516

struct Params {
    uint nstr;
    uint plane;
};

// ------------------------------------------------------------- shared table
static void build_lut_shared(device const uchar *hdr,
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
    for (uint i = tid; i < 256; i += TG_SHARED) {
        uint f = (uint)hdr[4 + 2 * i] | ((uint)hdr[5 + 2 * i] << 8);
        uint st = starts[i];
        uint packed = ((f ? f - 1u : 0u) << 20) | (st << 8) | i;
        for (uint j = 0; j < f; j++) lut[st + j] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <bool PREFETCH>
static void decode_plane_shared(device const uchar *streams,
                                device const ulong *off,
                                device uchar *out,
                                threadgroup uint *lut,
                                threadgroup uchar *stage,
                                threadgroup uchar *inbuf,
                                uint gid, uint k, uint lane, uint grp,
                                uint plane, uint nstr)
{
    uint src_id = gid < nstr ? gid : 0;
    device const uchar *ptr = streams + off[2 * src_id];
    device uchar *dst = out + (ulong)gid * plane;
    threadgroup uchar *mystage = stage + grp * GRAIN;
    threadgroup uchar *mybuf = inbuf + grp * BUFB;
    uint gbase = lane & ~7u;

    uint st = (uint)ptr[4 * k] | ((uint)ptr[4 * k + 1] << 8) |
              ((uint)ptr[4 * k + 2] << 16) | ((uint)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    uint filled = 0, consumed = 0;
    if (PREFETCH) {
        for (uint h = 0; h < 2; h++) {
            *(threadgroup packed_uint4 *)(mybuf + ((filled + k * 16) & (BUFB - 1))) =
                *(device const packed_uint4 *)(ptr + filled + k * 16);
            filled += HALFB;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint base = 0; base < plane; base += GRAIN) {
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
                w = (uint)mybuf[o & (BUFB - 1)] | ((uint)mybuf[(o + 1) & (BUFB - 1)] << 8);
            } else {
                device const uchar *p = ptr + consumed + 2 * before;
                w = (uint)p[0] | ((uint)p[1] << 8);
            }
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * popcount(gm);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (gid < nstr) {
            *(device packed_uint4 *)(dst + base + k * 16) =
                *(threadgroup const packed_uint4 *)(mystage + k * 16);
        }
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
    threadgroup uchar stage[GROUPS_SHARED * GRAIN];
    threadgroup uchar inbuf[GROUPS_SHARED * BUFB];
    build_lut_shared(hdr, lut, starts, tid_t);
    uint gid = tid_g >> 3;
    decode_plane_shared<false>(streams, off, out, lut, stage, inbuf,
                               gid, tid_g & 7, lane, tid_t >> 3, P.plane, P.nstr);
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
    threadgroup uchar stage[GROUPS_SHARED * GRAIN];
    threadgroup uchar inbuf[GROUPS_SHARED * BUFB];
    build_lut_shared(hdr, lut, starts, tid_t);
    uint gid = tid_g >> 3;
    decode_plane_shared<true>(streams, off, out, lut, stage, inbuf,
                              gid, tid_g & 7, lane, tid_t >> 3, P.plane, P.nstr);
}

// --------------------------------------------------------- per-stream table
kernel void lmz_decode_perstream(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device uchar *out             [[buffer(2)]],
    constant Params &P            [[buffer(3)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    threadgroup uchar symtab[4 * PROB_SCALE];
    threadgroup uint sftab[4 * 256];
    threadgroup uchar stage[4 * GRAIN];

    uint gid = tid_g >> 3;
    uint k = tid_g & 7;
    uint grp = tid_t >> 3;
    uint gbase = lane & ~7u;

    threadgroup uchar *mysym = symtab + grp * PROB_SCALE;
    threadgroup uint *mysf = sftab + grp * 256;
    threadgroup uchar *mystage = stage + grp * GRAIN;

    uint src_id = gid < P.nstr ? gid : 0;
    device const uchar *base_ptr = streams + off[2 * src_id];

    if (k == 0) {
        uint s = 0;
        for (uint i = 0; i < 256; i++) {
            uint f = (uint)base_ptr[4 + 2 * i] | ((uint)base_ptr[5 + 2 * i] << 8);
            mysf[i] = ((f ? f - 1u : 0u) << 12) | s;
            s += f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = k; i < 256; i += NST) {
        uint f = (uint)base_ptr[4 + 2 * i] | ((uint)base_ptr[5 + 2 * i] << 8);
        uint st = mysf[i] & 0xfffu;
        for (uint j = 0; j < f; j++) mysym[st + j] = (uchar)i;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const uchar *ptr = base_ptr + HEADER;
    device uchar *dst = out + (ulong)gid * P.plane;
    uint st = (uint)ptr[4 * k] | ((uint)ptr[4 * k + 1] << 8) |
              ((uint)ptr[4 * k + 2] << 16) | ((uint)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;

    uint consumed = 0;
    for (uint bpos = 0; bpos < P.plane; bpos += GRAIN) {
        for (uint s = 0; s < STAGE_ITERS; s++) {
            uint slot = st & (PROB_SCALE - 1u);
            uint sym = mysym[slot];
            uint p = mysf[sym];
            mystage[s * NST + k] = (uchar)sym;
            uint sv = p & 0xfffu, fq = (p >> 12) + 1u;
            uint x = fq * (st >> PROB_BITS) + slot - sv;
            bool need = (x < RANS_L);
            uint ball = (uint)((simd_vote::vote_t)simd_ballot(need));
            uint gm = (ball >> gbase) & 0xffu;
            uint before = popcount(gm & ((1u << k) - 1u));
            device const uchar *pbyte = ptr + consumed + 2 * before;
            uint w = (uint)pbyte[0] | ((uint)pbyte[1] << 8);
            st = need ? ((x << 16) | w) : x;
            consumed += 2 * popcount(gm);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (gid < P.nstr) {
            *(device packed_uint4 *)(dst + bpos + k * 16) =
                *(threadgroup const packed_uint4 *)(mystage + k * 16);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// -------------------------------------------------------------- fused BF16
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
    threadgroup uchar stage[GROUPS_SHARED * GRAIN];
    build_lut_shared(hdr, lut, starts, tid_t);

    uint gid = tid_g >> 3;
    uint k = tid_g & 7, grp = tid_t >> 3, gbase = lane & ~7u;
    threadgroup uchar *mystage = stage + grp * GRAIN;

    uint src_id = gid < P.nstr ? gid : 0;
    device const uchar *ptr = streams + off[2 * src_id];
    ulong ebase = (ulong)gid * P.plane;
    uint st = (uint)ptr[4 * k] | ((uint)ptr[4 * k + 1] << 8) |
              ((uint)ptr[4 * k + 2] << 16) | ((uint)ptr[4 * k + 3] << 24);
    ptr += 4 * NST;
    uint consumed = 0;

    for (uint base = 0; base < P.plane; base += GRAIN) {
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
        if (gid < P.nstr) {
            ulong eo = ebase + base + k * 16;
            threadgroup const uchar *ep = mystage + k * 16;
            device const uchar *sp = smplane + eo;
            ushort wv[16];
            for (uint j = 0; j < 16; j++) {
                wv[j] = (ushort)((((uint)sp[j] & 0x80u) << 8) |
                                 ((uint)ep[j] << 7) | ((uint)sp[j] & 0x7Fu));
            }
            device packed_uint4 *d4 = (device packed_uint4 *)(out + 2 * eo);
            thread const packed_uint4 *s4 = (thread const packed_uint4 *)wv;
            d4[0] = s4[0];
            d4[1] = s4[1];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ------------------------------------------------------------- dequant & projection
kernel void lmz_dequant_q8_0_to_fp16(
    device const uchar *q8_blocks [[buffer(0)]],
    device half *out_fp16         [[buffer(1)]],
    constant uint &nblocks        [[buffer(2)]],
    uint bid                      [[thread_position_in_grid]])
{
    if (bid >= nblocks) return;
    device const uchar *b = q8_blocks + bid * 34;
    half d = *(device const half *)b;
    device const char *qs = (device const char *)(b + 2);
    device half *out = out_fp16 + bid * 32;
    for (uint i = 0; i < 32; i++) {
        out[i] = d * (half)qs[i];
    }
}

kernel void lmz_project_fp8_e4m3_to_bf16(
    device const uchar *fp8_in    [[buffer(0)]],
    device ushort *bf16_out       [[buffer(1)]],
    constant uint &nelem          [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]])
{
    if (gid >= nelem) return;
    uchar v = fp8_in[gid];
    uint sign = (v >> 7) & 1u;
    uint exp = (v >> 3) & 0x0Fu;
    uint mant = v & 0x07u;
    ushort res;
    if (exp == 0) {
        if (mant == 0) {
            res = (ushort)(sign << 15);
        } else {
            uint shift = clz((ushort)mant) - 13;
            uint new_exp = 127 - 7 + 1 - shift;
            uint new_mant = (mant << (shift + 4)) & 0x7F;
            res = (ushort)((sign << 15) | (new_exp << 7) | new_mant);
        }
    } else if (exp == 15 && mant == 7) {
        res = (ushort)((sign << 15) | (0xFF << 7) | 0x40);
    } else {
        uint new_exp = exp + 120;
        uint new_mant = mant << 4;
        res = (ushort)((sign << 15) | (new_exp << 7) | new_mant);
    }
    bf16_out[gid] = res;
}
