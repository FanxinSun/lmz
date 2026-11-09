// Generated from runtime shipped source by build_phase4_wrappers.py. Wrapper bodies only.

kernel void lmz_decode_perstream_stride(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device uchar *out             [[buffer(2)]],
    constant Params &P            [[buffer(3)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint grids [[threadgroups_per_grid]])
{
    threadgroup uchar symtab[4 * PROB_SCALE];
    threadgroup uint sftab[4 * 256];
    threadgroup uchar stage[4 * GRAIN];

    for (uint gid = tid_g >> 3; gid < P.nstr; gid += grids * 4) {
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
}

kernel void lmz_decode_bf16_stride(
    device const uchar *streams   [[buffer(0)]],
    device const ulong *off       [[buffer(1)]],
    device const uchar *smplane   [[buffer(2)]],
    device uchar *out             [[buffer(3)]],
    device const uchar *hdr       [[buffer(4)]],
    constant Params &P            [[buffer(5)]],
    uint tid_g                    [[thread_position_in_grid]],
    uint tid_t                    [[thread_position_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint grids [[threadgroups_per_grid]])
{
    threadgroup uint lut[PROB_SCALE];
    threadgroup uint starts[256];
    threadgroup uchar stage[GROUPS_SHARED * GRAIN];
    build_lut_shared(hdr, lut, starts, tid_t);

    for (uint gid = tid_g >> 3; gid < P.nstr; gid += grids * 16) {
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
}
