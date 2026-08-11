// Whole-BF16 decode on the GPU, in one kernel.
//
// A BF16 lmz chunk is two planes. The exponent plane is rANS-coded (~2.8
// bits/symbol). The sign+mantissa plane is NOT coded at all -- measured, it
// expands to 100.12% of raw, so lmz's _est_stream stores it verbatim. So the
// fused kernel decodes one plane, reads the other, and writes bit-interleaved
// BF16 using lmz's own merge:
//     w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F)
// which is merge_bf16_scalar in lmzcore.c, byte for byte.
//
// Decode uses the V5 arrangement: 8 lanes per stream, ballot prefix-sum for
// the shared cursor, cp.async 4-slot prefetch, output staged in shared.
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8
#define STAGE_ITERS 16
#define CK(x) do { cudaError_t e_ = (x); if (e_) { \
    fprintf(stderr,"%s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); \
    exit(1);} } while (0)

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

template <int BUFB>
__global__ void k_fused_bf16(const uint8_t *__restrict__ streams,
                             const uint64_t *__restrict__ off,
                             const uint8_t *__restrict__ smplane,
                             uint32_t nstr, uint32_t plane,
                             uint8_t *__restrict__ out,
                             const uint8_t *__restrict__ shdr)
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
    uint32_t wmask = 0xffu << gbase;
    uint32_t grp = threadIdx.x >> 3;
    uint8_t *mystage = stage + grp * (NST * STAGE_ITERS);
    uint8_t *mybuf = inbuf + grp * BUFB;
    if (gid >= nstr) return;

    const uint8_t *ptr = streams + off[2 * gid];
    size_t ebase = (size_t)gid * plane;              // element index of stream
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
        } __pipeline_commit(); } while (0)
    ISSUE(0); ISSUE(Q); ISSUE(2 * Q);
    filled = 3 * Q;
    __pipeline_wait_prior(0);
    __syncwarp(wmask);

    for (uint32_t base = 0; base < plane; base += NST * STAGE_ITERS) {
#pragma unroll
        for (uint32_t s = 0; s < STAGE_ITERS; s++) {
            if (filled - consumed <= 2 * Q) {
                __pipeline_wait_prior(0);      // (0), not (1) -- see gpu_rans10
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
        // merge: 16 exponent bytes from shared + 16 sm bytes from global
        // -> 32 bytes of BF16. Across the group that is 256 contiguous bytes.
        size_t eo = ebase + base + k * 16;
        uint4 E = *(const uint4 *)(mystage + k * 16);
        uint4 S = *(const uint4 *)(smplane + eo);
        const uint8_t *ep = (const uint8_t *)&E;
        const uint8_t *sp = (const uint8_t *)&S;
        uint16_t wv[16];
#pragma unroll
        for (int j = 0; j < 16; j++)
            wv[j] = (uint16_t)((((uint32_t)sp[j] & 0x80u) << 8) |
                               ((uint32_t)ep[j] << 7) | ((uint32_t)sp[j] & 0x7Fu));
        *(uint4 *)(out + 2 * eo) = *(const uint4 *)(wv);
        *(uint4 *)(out + 2 * eo + 16) = *(const uint4 *)(wv + 8);
        __syncwarp();
    }
#undef ISSUE
}

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static std::vector<uint8_t> slurp(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(sz);
    if (fread(v.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr,"short read\n"); exit(1);}
    fclose(f); return v;
}

// O_DIRECT read straight into pinned memory, then H2D. Returns seconds.
static double read_to_device(const char *path, uint8_t *dev, void *pin,
                             size_t stage, size_t *nread)
{
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "O_DIRECT open %s failed\n", path); exit(1); }
    double t0 = now_s();
    size_t off = 0;
    for (;;) {
        ssize_t got = pread(fd, pin, stage, off);
        if (got <= 0) break;
        CK(cudaMemcpy(dev + off, pin, got, cudaMemcpyHostToDevice));
        off += got;
    }
    CK(cudaDeviceSynchronize());
    double dt = now_s() - t0;
    close(fd);
    *nread = off;
    return dt;
}

int main(int argc, char **argv)
{
    const char *dirE = "/home/rog/.cache/lmz-gpu-shared";
    const char *smpath = "/home/rog/.cache/lmz-gpu-sm/ref.bin";
    char pS[512], pB[512];
    snprintf(pS, sizeof pS, "%s/streams.bin", dirE);
    snprintf(pB, sizeof pB, "%s/bf16.bin", dirE);

    std::vector<uint8_t> blob = slurp(pS);
    uint32_t nstr, plane;
    memcpy(&nstr, blob.data(), 4);
    memcpy(&plane, blob.data() + 4, 4);
    const uint8_t *shdr = blob.data() + 8;
    const uint64_t *offs = (const uint64_t *)(blob.data() + 8 + 516);
    const uint8_t *sdata = blob.data() + 8 + 516 + (size_t)nstr * 16;
    size_t sbytes = blob.size() - (8 + 516 + (size_t)nstr * 16);
    size_t elems = (size_t)nstr * plane;
    size_t obytes = elems * 2;

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s  sm_%d%d  %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("model  %.1f MB BF16 (%zu M elements)\n", obytes / 1e6, elems / 1000000);
    printf("       exponent streams %.1f MB + sm plane raw %.1f MB = %.1f MB "
           "(%.2f%% saved)\n\n", sbytes / 1e6, elems / 1e6,
           (sbytes + elems) / 1e6, 100.0 * (1.0 - (double)(sbytes + elems) / obytes));

    uint8_t *d_streams, *d_sm, *d_out, *d_hdr; uint64_t *d_off;
    CK(cudaMalloc(&d_streams, ((sbytes + 4095) / 4096) * 4096 + 64));
    CK(cudaMemset(d_streams + sbytes, 0, 64));
    CK(cudaMalloc(&d_sm, elems));
    CK(cudaMalloc(&d_off, (size_t)nstr * 16));
    CK(cudaMalloc(&d_hdr, 516));
    CK(cudaMalloc(&d_out, obytes));
    CK(cudaMemcpy(d_streams, sdata, sbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_off, offs, (size_t)nstr * 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_hdr, shdr, 516, cudaMemcpyHostToDevice));
    {
        std::vector<uint8_t> sm = slurp(smpath);
        CK(cudaMemcpy(d_sm, sm.data(), elems, cudaMemcpyHostToDevice));
    }

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<uint8_t> ref = slurp(pB);
    std::vector<uint8_t> host(obytes);

    printf("%-30s %5s %9s %11s %12s\n", "fused decode", "tpb", "time", "BF16 out", "verdict");
    float best_ms = 1e30f; int best_tpb = 0;
    for (int tpb : {128, 256, 384}) {
        size_t shm = PROB_SCALE * 4 + (tpb / 8) * (NST * STAGE_ITERS) + (tpb / 8) * 512;
        CK(cudaFuncSetAttribute(k_fused_bf16<512>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
        int blocks = (int)(((size_t)nstr * 8 + tpb - 1) / tpb);
        CK(cudaMemset(d_out, 0, obytes));
        float best = 1e30f;
        for (int r = 0; r < 7; r++) {
            CK(cudaEventRecord(e0));
            k_fused_bf16<512><<<blocks, tpb, shm>>>(d_streams, d_off, d_sm, nstr,
                                                    plane, d_out, d_hdr);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            cudaError_t le = cudaGetLastError();
            if (le) { printf("  launch fail: %s\n", cudaGetErrorString(le)); break; }
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        CK(cudaMemcpy(host.data(), d_out, obytes, cudaMemcpyDeviceToHost));
        bool ok = memcmp(host.data(), ref.data(), obytes) == 0;
        printf("%-30s %5d %7.2f ms %9.1f GB/s %12s\n", "  exp rANS + sm raw + merge",
               tpb, best, obytes / (best * 1e-3) / 1e9,
               ok ? "byte-identical" : "*** MISMATCH ***");
        if (ok && best < best_ms) { best_ms = best; best_tpb = tpb; }
    }

    // ---------------- end to end, straight off the disk ---------------------
    printf("\n-- end to end: cold disk (O_DIRECT) -> VRAM, BF16 resident --\n");
    void *pin; CK(cudaHostAlloc(&pin, 64 << 20, 0));
    size_t got;
    uint8_t *d_plain;
    CK(cudaMalloc(&d_plain, obytes));

    double bp = 1e30, bl = 1e30;
    for (int r = 0; r < 3; r++) {
        double t = read_to_device(pB, d_plain, pin, 64 << 20, &got);
        if (t < bp) bp = t;
    }
    printf("plain safetensors bytes   read %.1f MB   %6.3f s   %5.2f GB/s\n",
           got / 1e6, bp, got / bp / 1e9);

    for (int r = 0; r < 3; r++) {
        size_t g1, g2;
        double t = read_to_device(smpath, d_sm, pin, 64 << 20, &g2);
        char pP[512]; snprintf(pP, sizeof pP, "%s/exp_payload.bin", dirE);
        t += read_to_device(pP, d_streams, pin, 64 << 20, &g1);
        double t0 = now_s();
        size_t shm = PROB_SCALE * 4 + (best_tpb / 8) * (NST * STAGE_ITERS)
                     + (best_tpb / 8) * 512;
        int blocks = (int)(((size_t)nstr * 8 + best_tpb - 1) / best_tpb);
        k_fused_bf16<512><<<blocks, best_tpb, shm>>>(d_streams, d_off, d_sm, nstr,
                                                     plane, d_out, d_hdr);
        CK(cudaDeviceSynchronize());
        t += now_s() - t0;
        if (t < bl) bl = t;
    }
    printf("lmz (exp coded + sm raw)  read %.1f MB   %6.3f s   %5.2f GB/s of BF16\n",
           (sbytes + elems) / 1e6, bl, obytes / bl / 1e9);
    printf("\n%-24s %6.3f s\n%-24s %6.3f s   -> %.2fx %s\n", "plain:", bp, "lmz:", bl,
           bp > bl ? bp / bl : bl / bp, bp > bl ? "FASTER" : "slower");
    printf("compression ratio is %.3fx, so lmz converts %.0f%% of its ratio "
           "into load speed\n", (double)obytes / (sbytes + elems),
           100.0 * (bp / bl) / ((double)obytes / (sbytes + elems)));
    // NOTE: the streams.bin read includes its offset table, which a real
    // archive would fold into the chunk index -- a few MB, counted against lmz.
    return 0;
}
