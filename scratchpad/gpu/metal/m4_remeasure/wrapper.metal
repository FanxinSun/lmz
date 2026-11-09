// Appended to the unchanged shipped source at runtime. Only grid stride is new.
kernel void lmz_decode_plane_stride(
    device const uchar *streams [[buffer(0)]],
    device const ulong *off [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    device const uchar *hdr [[buffer(3)]],
    constant Params &P [[buffer(4)]],
    uint tid_g [[thread_position_in_grid]],
    uint tid_t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint groups [[threadgroups_per_grid]])
{
    threadgroup uint lut[PROB_SCALE];
    threadgroup uint starts[256];
    threadgroup uchar stage[GROUPS_SHARED * GRAIN];
    threadgroup uchar inbuf[GROUPS_SHARED * BUFB];
    build_lut_shared(hdr,lut,starts,tid_t);
    for (uint gid=tid_g>>3; gid<P.nstr; gid+=groups*GROUPS_SHARED) {
        decode_plane_shared<false>(streams,off,out,lut,stage,inbuf,
            gid,tid_g&7,lane,tid_t>>3,P.plane,P.nstr);
    }
}
kernel void lmz_empty(device uint *out [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
    if (tid==0) out[0]=0;
}
