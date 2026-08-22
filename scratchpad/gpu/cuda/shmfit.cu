/*
 * Which block size the shipped picker lands on, per architecture.
 *
 * `pick_tpb` returning 0 is how the decoder declines a device whose shared
 * memory its tables will not fit, so "does lmz launch on a T4" is answerable
 * without a T4: it is arithmetic over one number, sharedMemPerBlockOptin.
 * This includes the kernel rather than restating its constants, and it
 * re-derives the local card's row from the driver, so the table is checked
 * against real hardware at least once instead of resting on my recollection.
 *
 *   nvcc -O3 -std=c++17 -o shmfit scratchpad/gpu/cuda/shmfit.cu && ./shmfit
 */
#include "../../../lmz/gpu/lmzgpu.cu"

int main(void)
{
    /* Documented maximum dynamic shared memory per block, by architecture. */
    struct Card { const char *name; size_t optin; };
    Card cards[] = {
        {"sm_75  T4, RTX 2080",      64 * 1024},
        {"sm_80  A100",             163 * 1024},
        {"sm_86  A10G, RTX 3090",    99 * 1024},
        {"sm_89  L4, RTX 4090",      99 * 1024},
        {"sm_90  H100",             227 * 1024},
        {"sm_100 B200",             227 * 1024},
        {"sm_120 RTX 5080",          99 * 1024},
    };

    size_t chunk_per = (size_t)PROB_SCALE + 256 * 4 + GRAIN + BUFB;
    size_t shared_per = GRAIN + BUFB, shared_fixed = (size_t)PROB_SCALE * 4;

    printf("per-chunk table: %zu B a group.  shared table: %zu B fixed + %zu B a group.\n",
           chunk_per, shared_fixed, shared_per);
    printf("A group is %d lanes, so threads/%d groups share one block.\n\n", NST, NST);
    printf("%-22s %7s   %-16s %-16s\n", "device", "optin", "per-chunk", "shared");
    for (const Card &c : cards) {
        size_t a = 0, b = 0;
        int ta = pick_tpb(chunk_per, 0, 128, c.optin, &a);
        int tb = pick_tpb(shared_per, shared_fixed, 384, c.optin, &b);
        printf("%-22s %5zu K   ", c.name, c.optin / 1024);
        if (ta) printf("%3d thr %5zu K   ", ta, a / 1024); else printf("%-16s ", "DECLINES");
        if (tb) printf("%3d thr %5zu K\n", tb, b / 1024); else printf("%-16s\n", "DECLINES");
    }

    cudaDeviceProp p;
    int n = 0;
    if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 &&
        cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        size_t optin = (size_t)p.sharedMemPerBlockOptin;
        printf("\nlocal %s is sm_%d%d and reports optin %zu K", p.name, p.major,
               p.minor, optin / 1024);
        for (const Card &c : cards)
            if (atoi(c.name + 3) == p.major * 10 + p.minor)
                printf(", table says %zu K -- %s", c.optin / 1024,
                       c.optin == optin ? "agrees" : "*** TABLE IS WRONG ***");
        printf("\n");
    }
    return 0;
}
