/*
 * encbench.c -- does the vector encoder write the same bytes, and is it faster?
 *
 * It compiles the shipped kernel itself, not a copy of it:
 *
 *     cc -O3 -o encbench scratchpad/encbench.c && ./encbench
 *     ./encbench 4.4          # pass the core clock in GHz for cycles/byte
 *
 * On x86 that measures the AVX2 body and on arm64 the NEON one. A machine
 * that is neither, or that has no vector body for its ISA, still runs the
 * comparison -- both paths are then the scalar loop, the bytes still have to
 * match, and the ratio prints 1.00x.
 *
 * The arm64 body can also be checked from anywhere:
 *
 *     cc -O2 -DLMZ_FORCE_NEON -Iscratchpad/neonshim -o encbench-neon \
 *         scratchpad/encbench.c && ./encbench-neon
 *
 * which stands scratchpad/neonshim/arm_neon.h in front of the kernel's include
 * and runs the NEON path in plain C. Its timings are meaningless and it says
 * so; its byte comparison is the real thing, because the lane order, the
 * emission mask and the compaction are all decided by the source under test.
 *
 * Two properties are being checked, and only the first of them is negotiable:
 *
 *   1. the vector path writes byte for byte what lmz_rans_encode_portable
 *      writes. If it does not, an archive depends on the machine that made it,
 *      and this is a format change wearing a speed change's clothes;
 *
 *   2. it is faster. If it is not, on this machine, the dispatch in
 *      rans_encode_impl should not be sending work to it.
 */
/* clock_gettime, under -std=c11 as well as the default. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../lmz/native/lmzcore.c"

#if LMZ_X86
#define VECTOR_BODY "avx2"
#elif LMZ_NEON_ENC
#define VECTOR_BODY "neon"
#else
#define VECTOR_BODY "none"
#endif

#define NPLANE (16u << 20)

/* ----------------------------------------------------------------- input */

static uint64_t rng_state;
static uint32_t rnd(void)
{
    /* xorshift64*, so the planes are the same bytes on every machine --
     * the same generator coderbench.c uses, and the same seed, so the two
     * tools measure the same planes. */
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

/*
 * A BF16 checkpoint's two byte planes: the exponent plane peaked over a few
 * dozen values, the sign+mantissa plane close to uniform. They renormalise at
 * very different rates -- about a sixth of the time against about half -- and
 * the renormalisation rate is what the vector encoder's emission path costs
 * its time in, so both are timed.
 */
static void make_planes(uint8_t *expp, uint8_t *smp, size_t n)
{
    rng_state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) {
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

/* ----------------------------------------------------------- the comparison */

static int failures = 0;
static long checked = 0, vectored = 0;

/*
 * Encode `n` bytes both ways and hold the two streams against each other.
 *
 * It also asks whether this input would reach the vector body at all -- the
 * same two questions rans_encode_impl asks -- and counts the answer. A pair of
 * encoders that agree because both of them ran the scalar loop is not evidence
 * of anything, and that is a quiet way for a test like this to pass forever.
 */
static void check(const char *what, const uint8_t *src, size_t n)
{
    uint64_t counts[256];
    uint16_t freqs[256];
    uint32_t maxf = 0;
    lmz_hist(src, n, counts);
    if (lmz_rans_freqs(counts, freqs) == 0)
        for (int s = 0; s < 256; s++) if (freqs[s] > maxf) maxf = freqs[s];
    checked++;
    vectored += (n >= RANS_SIMD_MIN && maxf && maxf <= RANS_SIMD_MAX_FREQ);

    size_t cap = lmz_rans_bound(n);
    uint8_t *va = malloc(cap), *vb = malloc(cap), *back = malloc(n + 1);
    long a = lmz_rans_encode(src, n, va, cap);
    long b = lmz_rans_encode_portable(src, n, vb, cap, NULL);
    const char *why = 0;
    if (a != b) why = "different lengths";
    else if (a < 0) why = "encode refused";
    else if (memcmp(va, vb, (size_t)a) != 0) why = "different bytes";
    else if (lmz_rans_decode(va, (size_t)a, back, n) != 0) why = "decode failed";
    else if (memcmp(back, src, n) != 0) why = "decode differs";
    if (why) {
        size_t at = 0;
        if (a == b && a > 0) while (at < (size_t)a && va[at] == vb[at]) at++;
        printf("    !! %-22s n=%zu: %s (%ld vs %ld, first at %zu)\n",
               what, n, why, a, b, at);
        failures++;
    }
    free(va); free(vb); free(back);
}

/* The five shapes the Python suite uses, over the lengths that straddle both
 * the group boundary and the length the vector path starts at. */
static void fixed_shapes(void)
{
    static const size_t sizes[] = {1, 7, 8, 9, 4088, 4095, 4096, 4097,
                                   4103, 4104, 4111, 8192, 65537};
    uint8_t *buf = malloc(70000);
    for (unsigned si = 0; si < sizeof sizes / sizeof *sizes; si++) {
        size_t n = sizes[si];
        for (int shape = 0; shape < 5; shape++) {
            const char *name;
            rng_state = 0xD1B54A32D192ED03ULL + n;
            for (size_t i = 0; i < n; i++) {
                switch (shape) {
                case 0: { uint32_t r = rnd(); int e = r ? __builtin_ctz(r) : 31;
                          buf[i] = (uint8_t)((rnd() >> 31 << 7) | ((124 + e) & 0x7F));
                          break; }
                case 1: buf[i] = (uint8_t)(rnd() >> 24); break;
                case 2: buf[i] = (uint8_t)((i * i) % 7); break;
                /* Drives one frequency past half the scale, which is where the
                 * reciprocal stops being exact and the vector path must
                 * decline. The two encoders then have to agree by taking the
                 * same road, which is worth checking too. */
                case 3: buf[i] = (uint8_t)(i % 8 ? 5 : (i & 0xFF)); break;
                default: buf[i] = 200; break;
                }
            }
            name = (const char *[]){"exponent", "near-uniform", "few symbols",
                                    "one dominant", "single symbol"}[shape];
            check(name, buf, n);
        }
    }
    free(buf);
}

/*
 * Random alphabets at random lengths. The fixed shapes cover the boundaries;
 * this covers the emission patterns, which is where a vector encoder actually
 * goes wrong -- the mask, the compaction and the cursor step are the only
 * parts with no scalar counterpart, and they are exercised by whatever mix of
 * lanes happens to renormalise together.
 */
static void random_alphabets(int rounds)
{
    uint8_t *buf = malloc(40001);
    uint32_t w[256];
    for (int r = 0; r < rounds; r++) {
        rng_state = 0x243F6A8885A308D3ULL + (uint64_t)r * 0x9E3779B97F4A7C15ULL;
        size_t n = 1 + rnd() % 40000;
        int distinct = 1 + (int)(rnd() % 40);
        uint32_t total = 0;
        for (int s = 0; s < 256; s++) w[s] = 0;
        for (int k = 0; k < distinct; k++) {
            uint32_t weight = 1 + rnd() % (1u << (rnd() % 12));
            w[rnd() & 0xFF] += weight;
            total += weight;
        }
        if (!total) { w[0] = 1; total = 1; }
        for (size_t i = 0; i < n; i++) {
            uint32_t pick = rnd() % total, acc = 0;
            int s = 0;
            for (; s < 256; s++) { acc += w[s]; if (pick < acc) break; }
            buf[i] = (uint8_t)(s & 0xFF);
        }
        check("random alphabet", buf, n);
    }
    free(buf);
}

/* ------------------------------------------------------------------ timing */

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double ghz = 0.0;

/* Timing a shim build measures the shim. The pass still runs, because it is
 * also the only place a plane-sized input goes through the vector body and
 * back, but it runs once. */
#ifdef LMZ_FORCE_NEON
#define REPS 1
#else
#define REPS 7
#endif

static void row(const char *name, double best, size_t bytes, double base,
                const char *note)
{
    printf("    %-24s %8.2f ms  %7.0f MiB/s", name, best * 1e3,
           bytes / best / 1048576.0);
    if (ghz > 0) printf("  %5.2f cyc/B", best * ghz * 1e9 / bytes);
    if (base > 0) printf("  %5.2fx", base / best);
    printf("  %s\n", note ? note : "");
}

int main(int argc, char **argv)
{
    if (argc > 1) ghz = atof(argv[1]);
    size_t n = NPLANE;
    uint8_t *expp = malloc(n), *smp = malloc(n);
    make_planes(expp, smp, n);
    printf("encbench: kernel isa %s, vector encoder %s\n", lmz_isa(), VECTOR_BODY);
#ifdef LMZ_FORCE_NEON
    printf("  (built against the NEON shim: the bytes are real, "
           "the times are not)\n");
#endif
    printf("  %zu MiB per plane, digests %08x %08x\n",
           (size_t)(n >> 20), digest(expp, n), digest(smp, n));
    if (ghz <= 0)
        printf("  (pass the core clock in GHz as an argument for cycles/byte)\n");

    printf("\n  same bytes as the portable loop\n");
    fixed_shapes();
    random_alphabets(300);
    printf("    %s\n", failures ? "!! SOME STREAMS DIFFER -- do not ship this"
                                : "every stream identical, and every one "
                                  "decoded back to its input");
    printf("    %ld of %ld inputs were eligible for the vector body%s\n",
           vectored, checked,
           strcmp(VECTOR_BODY, "none") == 0 ? ", which is not built here" : "");
    if (vectored < checked / 4 && strcmp(VECTOR_BODY, "none") != 0) {
        printf("    !! too few reached it: this comparison is mostly the "
               "scalar loop against itself\n");
        failures++;
    }

    const char *names[2] = {"exponent plane", "sign+mantissa plane"};
    uint8_t *planes[2] = {expp, smp};
    size_t cap = lmz_rans_bound(n);
    uint8_t *dst = malloc(cap), *back = malloc(n);

    for (int f = 0; f < 2; f++) {
        uint8_t *buf = planes[f];
        uint64_t counts[256];
        lmz_hist(buf, n, counts);
        long len = lmz_rans_encode_portable(buf, n, dst, cap, counts);
        uint32_t maxf = 0;
        uint16_t freqs[256];
        lmz_rans_freqs(counts, freqs);
        for (int s = 0; s < 256; s++) if (freqs[s] > maxf) maxf = freqs[s];
        printf("\n  %s: %.3fx coded, top frequency %u/4096%s\n",
               names[f], (double)n / len, maxf,
               maxf > RANS_SIMD_MAX_FREQ ? " -- above the guard, scalar only"
                                         : "");

        double b = 1e9, base;
        for (int r = 0; r < REPS; r++) {
            double s = now();
            lmz_rans_encode_portable(buf, n, dst, cap, counts);
            double e = now() - s;
            if (e < b) b = e;
        }
        base = b;
        row("encode, portable", b, n, 0, "the scalar loop");

        b = 1e9;
        long vlen = 0;
        for (int r = 0; r < REPS; r++) {
            double s = now();
            vlen = lmz_rans_encode_h(buf, n, dst, cap, counts);
            double e = now() - s;
            if (e < b) b = e;
        }
        int ok = vlen == len && lmz_rans_decode(dst, (size_t)vlen, back, n) == 0
                 && memcmp(back, buf, n) == 0;
        row("encode, vector", b, n, base, ok ? "" : "!! WRONG");
        if (!ok) failures++;
    }

    printf("\n");
    return failures ? 1 : 0;
}
