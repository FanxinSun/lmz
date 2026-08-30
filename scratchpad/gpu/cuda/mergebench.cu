/*
 * What the plane merge costs, and against what floor.
 *
 * lmz's GPU decoder returns byte planes plane-major; the plaintext is the
 * transpose. The question is whether that transpose is expensive enough to be
 * worth fusing into the decode kernel, and the honest way to ask it is against
 * a pure copy of the same bytes -- if the merge matches the copy, the scatter
 * is free and the only thing fusing buys is the DRAM round trip.
 *
 * It does match: 400.3 against 403.5 GB/s on an RTX 5080 over 936 MB of
 * 4-byte elements. See docs/gpu-residency-handover.md 3c for what follows from
 * that, including why fusing is worth least on the small devices it was
 * proposed for.
 *
 * ONE TERM THIS DOES NOT MEASURE, and it matters on the devices the question
 * is aimed at. Here the merge has VRAM to itself, so its cost is device
 * bandwidth. On a unified-memory part the round trip shares one bus with the
 * host and with whatever is fetching the next chunk, so it costs more than
 * this benchmark or the model in 3c will tell you -- a downstream consumer's
 * plan was 29% wrong on exactly this, treating overlapped stages as free while
 * they contended for one bus. Re-running this on an integrated GPU measures
 * the uncontended floor, not what a real pipeline pays.
 *
 *   nvcc -O3 -arch=native -o mergebench mergebench.cu && ./mergebench
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("%d: %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

/* The loaned shape: one thread per element, gather nplanes bytes. */
__global__ void merge4(const uint8_t*__restrict__ p, uint8_t*__restrict__ o, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; if(i>=n) return;
  uchar4 v; v.x=p[i]; v.y=p[n+i]; v.z=p[2*n+i]; v.w=p[3*n+i];
  reinterpret_cast<uchar4*>(o)[i]=v;
}
/* The floor: a pure copy of the same bytes, nothing rearranged. */
__global__ void copyk(const uint8_t*__restrict__ p, uint8_t*__restrict__ o, size_t nb){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t n4 = nb/16; if(i>=n4) return;
  reinterpret_cast<uint4*>(o)[i] = reinterpret_cast<const uint4*>(p)[i];
}
int main(){
  size_t nelem = 234ull<<20;        /* 936 MB of 4-byte elements */
  size_t nb = nelem*4;
  uint8_t *p,*o; CK(cudaMalloc(&p,nb)); CK(cudaMalloc(&o,nb)); CK(cudaMemset(p,7,nb));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best;
  best=1e30f;
  for(int r=0;r<5;r++){ CK(cudaEventRecord(a));
    merge4<<<(nelem+255)/256,256>>>(p,o,nelem);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best)best=ms; }
  printf("merge  (plane-major -> interleaved) %7.2f ms  %6.1f GB/s of output\n",
         best, nb/(best*1e-3)/1e9);
  best=1e30f;
  for(int r=0;r<5;r++){ CK(cudaEventRecord(a));
    copyk<<<(nb/16+255)/256,256>>>(p,o,nb);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best)best=ms; }
  printf("copy   (same bytes, no rearrange)   %7.2f ms  %6.1f GB/s of output\n",
         best, nb/(best*1e-3)/1e9);
  return 0;
}
