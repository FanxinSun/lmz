/* lmz rANS encoding against ONE table shared by every stream.
 *
 * Identical coder to lmz/native/lmzcore.c -- same normalisation, same
 * backwards 8-state walk, same flush -- with the 516-byte per-stream header
 * lifted out and stored once for the whole file. A stream is then just
 * [8 initial states][coded bytes], and a decoder that is handed the table
 * needs no change at all: prepending the shared header to any stream makes it
 * a valid lmz stream again, which is how the round-trip is verified.
 *
 * A global histogram gives every symbol that occurs anywhere a non-zero
 * frequency, so the shared table can always code every stream -- the
 * "occurs implies representable" invariant lmz asserts still holds.
 */
#include <stdint.h>
#include <string.h>

#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define NST 8

typedef struct { uint32_t freq, bias, cmpl_freq; } EncSym;

static void put(uint32_t *r, uint8_t **pptr, const EncSym *s)
{
    uint32_t x = *r;
    uint64_t x_max = ((uint64_t)(RANS_L >> PROB_BITS) << 16) * s->freq;
    if (x >= x_max) {
        uint8_t *p = *pptr - 2;
        p[0] = (uint8_t)(x & 0xff);
        p[1] = (uint8_t)((x >> 8) & 0xff);
        *pptr = p;
        x >>= 16;
    }
    uint32_t q = x / s->freq;                 /* exact, as in lmzcore.c */
    *r = x + s->bias + q * s->cmpl_freq;
}

static void flush(uint32_t *r, uint8_t **pptr)
{
    uint32_t x = *r;
    uint8_t *p = *pptr - 4;
    p[0] = (uint8_t)(x >> 0); p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16); p[3] = (uint8_t)(x >> 24);
    *pptr = p;
}

/* lmzcore.c's rans_normalize, verbatim in behaviour. */
int lmzx_normalize(const uint64_t *counts, uint16_t *freqs)
{
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) total += counts[i];
    if (total == 0) return -1;
    uint32_t sum = 0;
    int biggest = -1;
    uint64_t biggest_count = 0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) { freqs[i] = 0; continue; }
        uint64_t f = (counts[i] * PROB_SCALE) / total;
        if (f == 0) f = 1;
        if (f > PROB_SCALE) f = PROB_SCALE;
        freqs[i] = (uint16_t)f;
        sum += (uint32_t)f;
        if (counts[i] > biggest_count) { biggest_count = counts[i]; biggest = i; }
    }
    if (biggest < 0) return -1;
    if (sum < PROB_SCALE) {
        freqs[biggest] = (uint16_t)(freqs[biggest] + (PROB_SCALE - sum));
    } else {
        uint32_t excess = sum - PROB_SCALE;
        while (excess > 0) {
            int bi = -1; uint32_t bf = 1;
            for (int i = 0; i < 256; i++) if (freqs[i] > bf) { bf = freqs[i]; bi = i; }
            if (bi < 0) return -1;
            uint32_t take = (bf - 1 < excess) ? (bf - 1) : excess;
            freqs[bi] = (uint16_t)(freqs[bi] - take);
            excess -= take;
        }
    }
    for (int i = 0; i < 256; i++)
        if ((counts[i] != 0) != (freqs[i] != 0)) return -1;
    return 0;
}

void lmzx_hist(const uint8_t *p, size_t n, uint64_t *hist)
{
    for (size_t i = 0; i < n; i++) hist[p[i]]++;
}

/* Encode with a caller-supplied table. Writes [8 states][coded] into dst and
 * returns its length, or -1. */
long lmzx_encode_shared(const uint8_t *src, size_t n, const uint16_t *freqs,
                        uint8_t *dst, size_t cap)
{
    EncSym syms[256];
    uint32_t start = 0;
    for (int s = 0; s < 256; s++) {
        if (freqs[s]) {
            syms[s].freq = freqs[s];
            syms[s].cmpl_freq = PROB_SCALE - freqs[s];
            syms[s].bias = start;
            start += freqs[s];
        } else {
            syms[s].freq = 0;
        }
    }
    if (start != PROB_SCALE) return -2;
    for (size_t i = 0; i < n; i++) if (!syms[src[i]].freq) return -3;

    uint8_t *end = dst + cap;
    uint8_t *ptr = end;
    uint32_t state[NST];
    for (int k = 0; k < NST; k++) state[k] = RANS_L;
    size_t i = n;
    while (i > 0 && (i & (NST - 1))) { i--; put(&state[i & (NST - 1)], &ptr, &syms[src[i]]); }
    while (i >= NST) {
        i -= NST;
        for (int k = NST - 1; k >= 0; k--) put(&state[k], &ptr, &syms[src[i + k]]);
    }
    for (int k = NST - 1; k >= 0; k--) flush(&state[k], &ptr);
    size_t coded = (size_t)(end - ptr);
    memmove(dst, ptr, coded);
    return (long)coded;
}
