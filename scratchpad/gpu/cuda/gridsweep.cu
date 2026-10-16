/*
 * Does the cost model predict the kernel outside the bandwidth-bound corner?
 *
 * THE INSTRUMENT MUST BE THE SHIPPED KERNEL. The body below is extracted from
 * lmzgpu.cu verbatim, with a grid-stride loop wrapped around the per-stream
 * setup and nothing else changed. A first attempt hand-rewrote the body and
 * measured 2.4x slower at the same occupancy -- it was a different kernel, and
 * every constant derived from it was wrong. The check that catches this is
 * cheap: at a full grid this must reproduce the shipped kernel's rate (414.6
 * against 409.8-418 here). If it does not, the instrument has drifted and
 * nothing measured with it means anything.
 *
 * The 5080 saturates on bandwidth at and above 192 threads, so nothing
 * measured at full size can tell the compute term from the memory term. This
 * shrinks the GRID at fixed byte traffic: the same streams are decoded either
 * way, but with fewer blocks resident the kernel has to leave the
 * bandwidth-bound regime and enter the one the projections live in.
 *
 * The shipped kernel maps one group to one stream, so a smaller grid would
 * decode fewer streams. This wraps it in a grid-stride loop instead -- same
 * decode, same bytes, fewer resident blocks -- which is the only change.
 */
#include "/home/rog/business/YFCE/lmz/lmz/gpu/lmzgpu.cu"
#include <vector>

static std::vector<uint8_t> slurp(const char *p){
  FILE *f=fopen(p,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",p);exit(1);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<uint8_t> v((size_t)n);
  if(fread(v.data(),1,(size_t)n,f)!=(size_t)n){fprintf(stderr,"short read\n");exit(1);}
  fclose(f); return v;
}
#define MUST(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

/* The shipped k_shared body VERBATIM, with only a grid-stride loop added.
 * Extracted from lmzgpu.cu so the instrument cannot drift from the kernel. */
__global__ void k_shared_gs(const uint8_t *__restrict__ streams,
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
    uint32_t k = tid & 7;
    uint32_t gs_first = tid >> 3, gs_step = (gridDim.x * blockDim.x) >> 3;
    for (uint32_t gid = gs_first; gid < nstr; gid += gs_step) {
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
}

int main(int argc, char **argv){
  const char *dir = argc>1 ? argv[1] : "/home/rog/.cache/lmz-gpu-shared";
  char ps[512], pr[512];
  snprintf(ps,sizeof ps,"%s/streams.bin",dir); snprintf(pr,sizeof pr,"%s/ref.bin",dir);
  std::vector<uint8_t> blob=slurp(ps);
  uint32_t nstr,plane; memcpy(&nstr,blob.data(),4); memcpy(&plane,blob.data()+4,4);
  const uint8_t *hdr=blob.data()+8; size_t obase=8+HEADER;
  const uint64_t *offs=(const uint64_t*)(blob.data()+obase);
  const uint8_t *data=blob.data()+obase+(size_t)nstr*16;
  size_t nbytes=blob.size()-(obase+(size_t)nstr*16), obytes=(size_t)nstr*plane;
  cudaDeviceProp p; MUST(cudaGetDeviceProperties(&p,0));
  printf("GPU %s  sm_%d%d  %d SMs  pool %zu\n", p.name,p.major,p.minor,
         p.multiProcessorCount,(size_t)p.sharedMemPerMultiprocessor);
  printf("%u streams x %u B = %.1f MB plain from %.1f MB coded\n\n",
         nstr,plane,obytes/1e6,nbytes/1e6);
  uint8_t *d_s,*d_o2,*d_out,*d_hdr; uint64_t *d_off;
  size_t pad=(size_t)lmz_gpu_pad_bytes();
  MUST(cudaMalloc(&d_s,nbytes+pad)); MUST(cudaMemcpy(d_s,data,nbytes,cudaMemcpyHostToDevice));
  MUST(cudaMemset(d_s+nbytes,0,pad));
  MUST(cudaMalloc(&d_off,(size_t)nstr*16)); MUST(cudaMemcpy(d_off,offs,(size_t)nstr*16,cudaMemcpyHostToDevice));
  MUST(cudaMalloc(&d_out,obytes));
  MUST(cudaMalloc(&d_hdr,HEADER)); MUST(cudaMemcpy(d_hdr,hdr,HEADER,cudaMemcpyHostToDevice));
  std::vector<uint8_t> ref=slurp(pr), host(obytes);
  cudaEvent_t e0,e1; MUST(cudaEventCreate(&e0)); MUST(cudaEventCreate(&e1));
  unsigned tpb = argc>2 ? (unsigned)atoi(argv[2]) : 192;
  size_t shm = (size_t)PROB_SCALE*4 + (tpb/NST)*(GRAIN+BUFB);
  MUST(cudaFuncSetAttribute(k_shared_gs,
       cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm));
  printf("tpb %u, shm %zu B, %zu blocks fit a unit\n\n", tpb, shm,
         (size_t)p.sharedMemPerMultiprocessor/shm);
  printf("%-8s %10s %12s %10s\n","blocks","time","decode","verdict");
  int grids[]={1,2,4,8,16,21,32,42,64,84,128,168,252,1191};
  for(int gi=0; gi<14; gi++){
    unsigned nb=grids[gi];
    MUST(cudaMemset(d_out,0,obytes));
    float best=1e30f;
    for(int r=0;r<3;r++){
      MUST(cudaEventRecord(e0));
      k_shared_gs<<<nb,tpb,shm>>>(d_s,d_off,nstr,plane,d_out,d_hdr);
      MUST(cudaEventRecord(e1)); MUST(cudaEventSynchronize(e1));
      cudaError_t le=cudaGetLastError(); if(le){printf("launch: %s\n",cudaGetErrorString(le)); return 1;}
      float ms; MUST(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best)best=ms;
    }
    MUST(cudaMemcpy(host.data(),d_out,obytes,cudaMemcpyDeviceToHost));
    bool ok = memcmp(host.data(),ref.data(),obytes)==0;
    printf("%-8u %7.2f ms %8.1f GB/s %10s\n", nb, best,
           obytes/(best*1e-3)/1e9, ok?"identical":"*** BAD ***");
  }
  return 0;
}
