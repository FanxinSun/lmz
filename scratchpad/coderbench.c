/*
 * coderbench.c -- what lmz's rANS coder costs on this machine.
 *
 * Self-contained on purpose: no dependencies, no input files, no libm, and
 * the planes are generated from a fixed seed so two machines measure the same
 * bytes. It prints a digest of them; if that differs, nothing below is
 * comparable and the numbers should be thrown away.
 *
 *     cc -O3 -o coderbench coderbench.c && ./coderbench
 *     ./coderbench 4.4        # pass the core clock in GHz for cycles/byte
 *
 * It answers three questions that the x86 box cannot answer for arm64:
 *
 *   1. What do the encoder and decoder cost per byte here? lmz decodes far
 *      faster on arm64 runners than on x86 ones while encoding slower, and
 *      nothing so far explains it.
 *
 *   2. Is the decoder latency-bound here too? Two independent streams stepped
 *      in one loop did twice the work in 1.03x the time on x86 -- the machine
 *      was idle, waiting. If that holds here, widening the format is worth the
 *      same on both, and an AVX2/NEON decoder over eight lanes is worth little.
 *
 *   3. Does deciding the refills before their addresses help here? That change
 *      shipped on x86 evidence alone (1.16x on an exponent plane, neutral on a
 *      near-uniform one), and it should not be quietly costing arm64 anything.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define HDR 516
#define XSHIFT 20
#define NPLANE (16u << 20)

/* ----------------------------------------------------------------- input */

static uint64_t rng_state;
static uint32_t rnd(void)
{
    /* xorshift64*, so the planes are the same bytes on every machine. */
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

/*
 * A BF16 checkpoint's two byte planes, built to the statistics the real ones
 * measure rather than by rounding real floats: the exponent plane peaked over
 * a few dozen values, the sign+mantissa plane close to uniform. Those are the
 * two shapes the coder actually meets, and they behave very differently -- the
 * peaked one renormalises about a sixth of the time, the flat one about half.
 */
static void make_planes(uint8_t *expp, uint8_t *smp, size_t n)
{
    rng_state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) {
        /* A geometric exponent under a uniform sign: peaked, with a thin tail
         * running out to a few dozen values. Measured at 3.00 bits per byte
         * against the 2.77 a real Llama exponent plane holds, which puts the
         * renormalisation rate -- the thing the decoder's behaviour turns on --
         * within a symbol of the real one. */
        uint32_t r = rnd();
        int e = r ? __builtin_ctz(r) : 31;
        uint32_t sign = rnd() >> 31;
        expp[i] = (uint8_t)((sign << 7) | ((124 + e) & 0x7F));
        smp[i] = (uint8_t)(rnd() >> 24);
    }
}

static uint32_t digest(const uint8_t *p, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

/* ------------------------------------------------------------- the coder */

static void hist(const uint8_t *p, size_t n, uint64_t *h)
{
    uint64_t a[256]={0}, b[256]={0}, c[256]={0}, d[256]={0};
    size_t i = 0;
    for (; i+4<=n; i+=4){ a[p[i]]++; b[p[i+1]]++; c[p[i+2]]++; d[p[i+3]]++; }
    for (; i<n; i++) a[p[i]]++;
    for (int j=0;j<256;j++) h[j]=a[j]+b[j]+c[j]+d[j];
}

static void normalize(const uint64_t *counts, uint16_t *freqs)
{
    uint64_t total = 0;
    for (int i=0;i<256;i++) total += counts[i];
    uint32_t sum = 0; int big = 0; uint64_t bmax = 0;
    for (int i=0;i<256;i++){
        if (!counts[i]) { freqs[i]=0; continue; }
        uint64_t f = (counts[i]*PROB_SCALE)/total;
        if (!f) f = 1;
        if (f > PROB_SCALE) f = PROB_SCALE;
        freqs[i]=(uint16_t)f; sum += (uint32_t)f;
        if (counts[i]>bmax){ bmax=counts[i]; big=i; }
    }
    if (sum<PROB_SCALE) freqs[big]=(uint16_t)(freqs[big]+(PROB_SCALE-sum));
    else { uint32_t e=sum-PROB_SCALE;
           while(e) for(int i=0;i<256&&e;i++) if(freqs[i]>1){freqs[i]--;e--;} }
}

typedef struct { uint32_t freq, bias, cmpl; } Sym;

static void syms_of(const uint16_t *freqs, Sym *sy)
{
    uint32_t start = 0;
    for (int s=0;s<256;s++){
        sy[s].freq = freqs[s] ? freqs[s] : 1;
        sy[s].cmpl = PROB_SCALE - freqs[s];
        sy[s].bias = start;
        start += freqs[s];
    }
}

/* The encoder as it ships: branchless emit, hardware divide, NS interleaved
 * states. NS is what the format fixes at eight; sixteen is here to price
 * widening it, which is the open decision. */
#define MAKE_ENC(NS)                                                          \
static long encode##NS(const uint8_t *src, size_t n, uint8_t *dst, size_t cap,\
                       const uint16_t *freqs)                                 \
{                                                                             \
    Sym sy[256]; syms_of(freqs, sy);                                          \
    uint8_t *end = dst+cap, *ptr = end;                                       \
    uint32_t st[NS]; for (int k=0;k<NS;k++) st[k]=RANS_L;                     \
    size_t i = n;                                                             \
    while (i>0 && (i&(NS-1))) { i--;                                          \
        uint32_t x=st[i&(NS-1)]; const Sym *s=&sy[src[i]];                    \
        uint32_t need=(x >= ((uint64_t)s->freq<<XSHIFT));                     \
        ptr[-2]=(uint8_t)x; ptr[-1]=(uint8_t)(x>>8); ptr -= need<<1;          \
        x = need ? (x>>16) : x;                                               \
        st[i&(NS-1)] = x + s->bias + (x/s->freq)*s->cmpl; }                   \
    while (i>=NS) { i-=NS;                                                    \
        for (int k=NS-1;k>=0;k--) {                                           \
            uint32_t x=st[k]; const Sym *s=&sy[src[i+k]];                     \
            uint32_t need=(x >= ((uint64_t)s->freq<<XSHIFT));                 \
            ptr[-2]=(uint8_t)x; ptr[-1]=(uint8_t)(x>>8); ptr -= need<<1;      \
            x = need ? (x>>16) : x;                                           \
            st[k] = x + s->bias + (x/s->freq)*s->cmpl; } }                    \
    for (int k=NS-1;k>=0;k--){ uint32_t v=st[k]; ptr-=4; memcpy(ptr,&v,4); }  \
    size_t coded=(size_t)(end-ptr);                                           \
    dst[0]='R'; dst[1]='1'; dst[2]=0; dst[3]=0;                               \
    for (int s=0;s<256;s++){ dst[4+2*s]=(uint8_t)(freqs[s]&0xff);             \
        dst[5+2*s]=(uint8_t)(freqs[s]>>8); }                                  \
    memmove(dst+HDR, ptr, coded);                                             \
    return (long)(HDR+coded);                                                 \
}
MAKE_ENC(8)
MAKE_ENC(16)

static void build_lut(const uint16_t *freqs, uint32_t *lut)
{
    uint32_t start = 0;
    for (int s=0;s<256;s++){
        uint32_t f = freqs[s];
        if (!f) continue;
        uint32_t packed = ((f-1)<<20) | (start<<8) | (uint32_t)s;
        for (uint32_t j=0;j<f;j++) lut[start+j]=packed;
        start += f;
    }
}

#define DEC_INIT                                                              \
    const uint8_t *ptr=src+HDR, *limit=src+src_len;                           \
    uint32_t state[8];                                                        \
    for (int k=0;k<8;k++){ state[k]=(uint32_t)ptr[0]|((uint32_t)ptr[1]<<8)    \
        |((uint32_t)ptr[2]<<16)|((uint32_t)ptr[3]<<24); ptr+=4; }

#define DEC_TAIL                                                              \
    for (; i<n; i++){                                                         \
        uint32_t k=(uint32_t)(i&7);                                           \
        uint32_t e=lut[state[k]&(PROB_SCALE-1)];                              \
        dst[i]=(uint8_t)(e&0xff);                                             \
        uint32_t x=state[k], st2=(e>>8)&0xfff, fq=(e>>20)+1;                  \
        x = fq*(x>>PROB_BITS) + (x&(PROB_SCALE-1)) - st2;                     \
        if (x<RANS_L){ if(ptr+2>limit) return -1;                             \
            x=(x<<16)|(uint32_t)ptr[0]|((uint32_t)ptr[1]<<8); ptr+=2; }       \
        state[k]=x; }                                                         \
    return 0;

/* The decoder before this branch: one cursor, advanced lane by lane, so each
 * refill load waits on the one before it just to learn its address. */
static int dec_shared(const uint8_t *src, size_t src_len, uint8_t *dst,
                      size_t n, const uint32_t *lut)
{
    DEC_INIT
    size_t i = 0;
    for (; i+8<=n && ptr+16<=limit; i+=8){
        uint32_t e[8];
        for (int k=0;k<8;k++) e[k]=lut[state[k]&(PROB_SCALE-1)];
        for (int k=0;k<8;k++) dst[i+k]=(uint8_t)(e[k]&0xff);
        for (int k=0;k<8;k++){
            uint32_t x=state[k];
            x = ((e[k]>>20)+1)*(x>>PROB_BITS) + (x&(PROB_SCALE-1)) - ((e[k]>>8)&0xfff);
            const uint8_t *p=ptr;
            uint32_t word=(uint32_t)p[0]|((uint32_t)p[1]<<8);
            uint32_t need=(x<RANS_L);
            x = need ? ((x<<16)|word) : x;
            ptr = p + (need<<1);
            state[k]=x;
        }
    }
    DEC_TAIL
}

/* The decoder as it ships now: all eight refill decisions first, so the eight
 * loads are independent. */
static int dec_offsets(const uint8_t *src, size_t src_len, uint8_t *dst,
                       size_t n, const uint32_t *lut)
{
    DEC_INIT
    size_t i = 0;
    for (; i+8<=n && ptr+16<=limit; i+=8){
        uint32_t e[8], x[8], need[8], off[8];
        for (int k=0;k<8;k++) e[k]=lut[state[k]&(PROB_SCALE-1)];
        for (int k=0;k<8;k++) dst[i+k]=(uint8_t)(e[k]&0xff);
        for (int k=0;k<8;k++)
            x[k]=((e[k]>>20)+1)*(state[k]>>PROB_BITS)
                 + (state[k]&(PROB_SCALE-1)) - ((e[k]>>8)&0xfff);
        for (int k=0;k<8;k++) need[k]=(x[k]<RANS_L);
        off[0]=0;
        for (int k=1;k<8;k++) off[k]=off[k-1]+(need[k-1]<<1);
        for (int k=0;k<8;k++){
            const uint8_t *p=ptr+off[k];
            uint32_t word=(uint32_t)p[0]|((uint32_t)p[1]<<8);
            state[k]= need[k] ? ((x[k]<<16)|word) : x[k];
        }
        ptr += off[7] + (need[7]<<1);
    }
    DEC_TAIL
}

/* Two chunks decoded together: different bytes, different coded streams,
 * and a frequency table each, because that is what two chunks are. Decoding
 * one stream twice instead would let the second ride on cache lines the first
 * had just pulled in, and reports a speedup that does not exist. */
static int dec_two(const uint8_t *sa, size_t la, const uint32_t *lua, uint8_t *d0,
                   const uint8_t *sb, size_t lb, const uint32_t *lub, uint8_t *d1,
                   size_t n)
{
    const uint8_t *p0=sa+HDR, *p1=sb+HDR, *enda=sa+la, *endb=sb+lb;
    uint32_t s0[8], s1[8];
    for (int k=0;k<8;k++){ s0[k]=(uint32_t)p0[0]|((uint32_t)p0[1]<<8)
        |((uint32_t)p0[2]<<16)|((uint32_t)p0[3]<<24); p0+=4; }
    for (int k=0;k<8;k++){ s1[k]=(uint32_t)p1[0]|((uint32_t)p1[1]<<8)
        |((uint32_t)p1[2]<<16)|((uint32_t)p1[3]<<24); p1+=4; }
    size_t i=0;
    for (; i+8<=n && p0+16<=enda && p1+16<=endb; i+=8){
        uint32_t e0[8], e1[8];
        for (int k=0;k<8;k++) e0[k]=lua[s0[k]&(PROB_SCALE-1)];
        for (int k=0;k<8;k++) e1[k]=lub[s1[k]&(PROB_SCALE-1)];
        for (int k=0;k<8;k++) d0[i+k]=(uint8_t)(e0[k]&0xff);
        for (int k=0;k<8;k++) d1[i+k]=(uint8_t)(e1[k]&0xff);
        for (int k=0;k<8;k++){
            uint32_t x=s0[k];
            x=((e0[k]>>20)+1)*(x>>PROB_BITS)+(x&(PROB_SCALE-1))-((e0[k]>>8)&0xfff);
            const uint8_t *p=p0; uint32_t w=(uint32_t)p[0]|((uint32_t)p[1]<<8);
            uint32_t nd=(x<RANS_L); x=nd?((x<<16)|w):x; p0=p+(nd<<1); s0[k]=x;
        }
        for (int k=0;k<8;k++){
            uint32_t x=s1[k];
            x=((e1[k]>>20)+1)*(x>>PROB_BITS)+(x&(PROB_SCALE-1))-((e1[k]>>8)&0xfff);
            const uint8_t *p=p1; uint32_t w=(uint32_t)p[0]|((uint32_t)p[1]<<8);
            uint32_t nd=(x<RANS_L); x=nd?((x<<16)|w):x; p1=p+(nd<<1); s1[k]=x;
        }
    }
    return (int)i;
}

/* Sixteen interleaved states in one stream: the format change that a "more
 * chains would help" reading of the two-chunk figure appears to argue for.
 * It does not survive being measured. */
static int dec16(const uint8_t *src, size_t src_len, uint8_t *dst, size_t n,
                 const uint32_t *lut)
{
    const uint8_t *ptr=src+HDR, *limit=src+src_len;
    uint32_t state[16];
    for (int k=0;k<16;k++){ state[k]=(uint32_t)ptr[0]|((uint32_t)ptr[1]<<8)
        |((uint32_t)ptr[2]<<16)|((uint32_t)ptr[3]<<24); ptr+=4; }
    size_t i=0;
    for (; i+16<=n && ptr+32<=limit; i+=16){
        uint32_t e[16];
        for (int k=0;k<16;k++) e[k]=lut[state[k]&(PROB_SCALE-1)];
        for (int k=0;k<16;k++) dst[i+k]=(uint8_t)(e[k]&0xff);
        for (int k=0;k<16;k++){
            uint32_t x=state[k];
            x=((e[k]>>20)+1)*(x>>PROB_BITS)+(x&(PROB_SCALE-1))-((e[k]>>8)&0xfff);
            const uint8_t *p=ptr; uint32_t w=(uint32_t)p[0]|((uint32_t)p[1]<<8);
            uint32_t nd=(x<RANS_L); x=nd?((x<<16)|w):x; ptr=p+(nd<<1); state[k]=x;
        }
    }
    for (; i<n; i++){
        uint32_t k=(uint32_t)(i&15);
        uint32_t e=lut[state[k]&(PROB_SCALE-1)];
        dst[i]=(uint8_t)(e&0xff);
        uint32_t x=state[k], st2=(e>>8)&0xfff, fq=(e>>20)+1;
        x=fq*(x>>PROB_BITS)+(x&(PROB_SCALE-1))-st2;
        if (x<RANS_L){ if(ptr+2>limit) return -1;
            x=(x<<16)|(uint32_t)ptr[0]|((uint32_t)ptr[1]<<8); ptr+=2; }
        state[k]=x;
    }
    return 0;
}

/* ------------------------------------------------------------------ main */

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double ghz = 0.0;

static void row(const char *name, double best, size_t bytes, double base,
                const char *note)
{
    printf("    %-26s %8.2f ms  %7.0f MiB/s", name, best*1e3,
           bytes/best/1048576.0);
    if (ghz > 0) printf("  %5.2f cyc/B", best*ghz*1e9/bytes);
    if (base > 0) printf("  %5.2fx", base/best);
    printf("  %s\n", note ? note : "");
}

int main(int argc, char **argv)
{
    if (argc > 1) ghz = atof(argv[1]);
    size_t n = NPLANE;
    uint8_t *expp = malloc(n), *smp = malloc(n);
    make_planes(expp, smp, n);
    printf("coderbench: %zu MiB per plane, digests %08x %08x\n",
           (size_t)(n >> 20), digest(expp, n), digest(smp, n));
    if (ghz <= 0)
        printf("(pass the core clock in GHz as an argument for cycles/byte)\n");

    const char *names[2] = {"exponent plane", "sign+mantissa plane"};
    uint8_t *planes[2] = {expp, smp};

    for (int f = 0; f < 2; f++) {
        uint8_t *buf = planes[f];
        uint64_t counts[256]; hist(buf, n, counts);
        uint16_t freqs[256]; normalize(counts, freqs);
        int distinct = 0; uint32_t maxf = 0;
        for (int s=0;s<256;s++){ if(freqs[s]) distinct++;
                                 if(freqs[s]>maxf) maxf=freqs[s]; }
        size_t cap = HDR + 2*n + 4*16 + 64;
        uint8_t *enc = malloc(cap), *enc16 = malloc(cap);
        long l8 = encode8(buf, n, enc, cap, freqs);
        long l16 = encode16(buf, n, enc16, cap, freqs);
        uint32_t *lut = malloc(PROB_SCALE*4); build_lut(freqs, lut);
        uint8_t *o0 = malloc(n), *o1 = malloc(n);

        printf("\n  %s: %.3fx coded, %d symbols, top frequency %u/4096\n",
               names[f], (double)n/l8, distinct, maxf);

        double b, base;
        b = 1e9;
        for (int r=0;r<7;r++){ double s=now(); encode8(buf,n,enc,cap,freqs);
                               double e=now()-s; if(e<b)b=e; }
        base = b;
        row("encode, 8 states", b, n, 0, "as it ships");
        b = 1e9;
        for (int r=0;r<7;r++){ double s=now(); encode16(buf,n,enc16,cap,freqs);
                               double e=now()-s; if(e<b)b=e; }
        row("encode, 16 states", b, n, base, "a format change");

        b = 1e9;
        int rc = 0;
        for (int r=0;r<7;r++){ memset(o0,0,n); double s=now();
            rc=dec_shared(enc,(size_t)l8,o0,n,lut); double e=now()-s; if(e<b)b=e; }
        base = b;
        row("decode, shared cursor", b, n, 0,
            (rc==0 && !memcmp(o0,buf,n)) ? "before this branch" : "!! WRONG");
        b = 1e9;
        for (int r=0;r<7;r++){ memset(o0,0,n); double s=now();
            rc=dec_offsets(enc,(size_t)l8,o0,n,lut); double e=now()-s; if(e<b)b=e; }
        row("decode, offsets up front", b, n, base,
            (rc==0 && !memcmp(o0,buf,n)) ? "as it ships" : "!! WRONG");
        b = 1e9;
        for (int r=0;r<7;r++){ memset(o0,0,n); double s=now();
            rc=dec16(enc16,(size_t)l16,o0,n,lut); double e=now()-s; if(e<b)b=e; }
        row("decode, 16 states", b, n, base,
            (rc==0 && !memcmp(o0,buf,n)) ? "the format change" : "!! WRONG");

        /* Two chunks together. Each half of the plane is its own stream with
         * its own table, which is what pairing chunks would really mean. */
        size_t h = n / 2;
        uint64_t ca[256], cb[256]; uint16_t fa[256], fb[256];
        hist(buf, h, ca); normalize(ca, fa);
        hist(buf + h, h, cb); normalize(cb, fb);
        uint8_t *ea = malloc(cap), *eb = malloc(cap);
        long la = encode8(buf, h, ea, cap, fa);
        long lb = encode8(buf + h, h, eb, cap, fb);
        uint32_t *lua = malloc(PROB_SCALE*4), *lub = malloc(PROB_SCALE*4);
        build_lut(fa, lua); build_lut(fb, lub);
        b = 1e9;
        for (int r=0;r<7;r++){ double s=now();
            dec_shared(ea,(size_t)la,o0,h,lua);
            dec_shared(eb,(size_t)lb,o1,h,lub);
            double e=now()-s; if(e<b)b=e; }
        double apart = b;
        row("two chunks, one at a time", b, n, base, "");
        b = 1e9;
        for (int r=0;r<7;r++){ double s=now();
            dec_two(ea,(size_t)la,lua,o0,eb,(size_t)lb,lub,o1,h);
            double e=now()-s; if(e<b)b=e; }
        row("two chunks, interleaved", b, n, apart, "against the line above");
        free(ea); free(eb); free(lua); free(lub);

        free(enc); free(enc16); free(lut); free(o0); free(o1);
        (void)l16;
    }
    printf("\nWhat the last three lines are for. Sixteen states is the format\n"
           "change; if it is below 1.00x the register file, not the dependency\n"
           "chains, is what the decoder is short of. Interleaving two chunks is\n"
           "the same idea without a format change, and the honest version of it:\n"
           "two different streams with a table each, not one stream decoded\n"
           "twice, which only measures the second one reusing the first's cache.\n");
    free(expp); free(smp);
    return 0;
}
