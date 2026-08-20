/*
 * What the shipped decoder reaches once the bytes are already in VRAM.
 *
 * This includes lmz/gpu/lmzgpu.cu rather than linking it, so what is timed is
 * exactly the library the package builds -- no second copy of the kernel to
 * drift. It is here and not in the package because a benchmark is not a
 * library: it owns file paths and timing, which lmz_gpu_decode_batch_dev must
 * not.
 *
 *   nvcc -O3 -std=c++17 -arch=sm_120 -o libbench libbench.cu
 *   ./libbench perstream   # ordinary lmz streams, a table per chunk
 *   ./libbench shared      # one table for the batch
 */
#include "../../../lmz/gpu/lmzgpu.cu"
#include <cstdlib>
#include <vector>

static std::vector<uint8_t> slurp(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v((size_t)n);
    if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f); return v;
}

#define MUST(x) do { cudaError_t e_=(x); if (e_) { \
    fprintf(stderr, "%d: %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

int main(int argc, char **argv)
{
    bool shared = argc > 1 && !strcmp(argv[1], "shared");
    const char *dir = shared ? "/home/rog/.cache/lmz-gpu-shared" : "/home/rog/.cache/lmz-gpu";
    char ps[512], pr[512];
    snprintf(ps, sizeof ps, "%s/streams.bin", dir);
    snprintf(pr, sizeof pr, "%s/ref.bin", dir);

    std::vector<uint8_t> blob = slurp(ps);
    uint32_t nstr, plane;
    memcpy(&nstr, blob.data(), 4);
    memcpy(&plane, blob.data() + 4, 4);
    const uint8_t *hdr = shared ? blob.data() + 8 : NULL;
    size_t obase = 8 + (shared ? HEADER : 0);
    const uint64_t *offs = (const uint64_t *)(blob.data() + obase);
    const uint8_t *data = blob.data() + obase + (size_t)nstr * 16;
    size_t nbytes = blob.size() - (obase + (size_t)nstr * 16);
    size_t obytes = (size_t)nstr * plane;

    cudaDeviceProp prop; MUST(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s  sm_%d%d  %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("%s tables: %u streams x %u B = %.1f MB plain from %.1f MB coded\n\n",
           shared ? "one shared" : "one per chunk", nstr, plane, obytes / 1e6, nbytes / 1e6);

    uint8_t *d_streams, *d_out, *d_hdr = NULL; uint64_t *d_off;
    size_t pad = (size_t)lmz_gpu_pad_bytes();
    MUST(cudaMalloc(&d_streams, nbytes + pad));
    MUST(cudaMemcpy(d_streams, data, nbytes, cudaMemcpyHostToDevice));
    MUST(cudaMemset(d_streams + nbytes, 0, pad));
    MUST(cudaMalloc(&d_off, (size_t)nstr * 16));
    MUST(cudaMemcpy(d_off, offs, (size_t)nstr * 16, cudaMemcpyHostToDevice));
    MUST(cudaMalloc(&d_out, obytes));
    if (hdr) { MUST(cudaMalloc(&d_hdr, HEADER));
               MUST(cudaMemcpy(d_hdr, hdr, HEADER, cudaMemcpyHostToDevice)); }

    std::vector<uint8_t> ref = slurp(pr);
    std::vector<uint8_t> host(obytes);
    cudaEvent_t e0, e1; MUST(cudaEventCreate(&e0)); MUST(cudaEventCreate(&e1));

    printf("%-6s %9s %11s %14s\n", "tpb", "time", "decode", "verdict");
    for (unsigned tpb : {64u, 96u, 128u, 160u, 192u, 256u, 384u}) {
        MUST(cudaMemset(d_out, 0, obytes));
        float best = 1e30f;
        int rc = LMZ_GPU_OK;
        for (int r = 0; r < 5; r++) {
            MUST(cudaEventRecord(e0));
            rc = lmz_gpu_decode_batch_dev(d_hdr, d_streams, d_off, nstr, plane,
                                          d_out, NULL, tpb);
            MUST(cudaEventRecord(e1)); MUST(cudaEventSynchronize(e1));
            if (rc != LMZ_GPU_OK) break;
            float ms; MUST(cudaEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms;
        }
        if (rc != LMZ_GPU_OK) { printf("%-6u %9s %11s %14s\n", tpb, "-", "-",
                                       lmz_gpu_last_error()); continue; }
        MUST(cudaMemcpy(host.data(), d_out, obytes, cudaMemcpyDeviceToHost));
        bool ok = memcmp(host.data(), ref.data(), obytes) == 0;
        printf("%-6u %7.2f ms %8.1f GB/s %14s\n", tpb, best,
               obytes / (best * 1e-3) / 1e9, ok ? "byte-identical" : "*** MISMATCH ***");
    }
    return 0;
}
