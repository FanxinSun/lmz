/* lmzcore.c -- byte-plane split/merge kernels for lmz.
 *
 * Built as a plain shared library and loaded through ctypes, so it needs no
 * Python headers and is not tied to any particular interpreter build.
 *
 * The central operation is a radix-2 byte deinterleave: given an array of
 * fixed-size elements, produce one contiguous "plane" per byte position.
 * For BF16 that separates the sign+exponent byte (a handful of distinct
 * values) from the mantissa byte (near-uniform noise), which is what makes
 * the entropy coder effective.
 *
 * Element sizes above 2 are handled by repeated 2-byte deinterleaving:
 * splitting 4-byte elements once yields two arrays of 2-byte elements
 * holding the even and odd byte positions, and splitting those again
 * yields the four planes. So one well-optimised 2-byte kernel covers
 * every power-of-two element size.
 *
 * Build: gcc -O3 -fPIC -shared -o lmzcore.so lmzcore.c
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#define LMZ_X86 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#define LMZ_NEON 1
#include <arm_neon.h>
#endif

#define LMZ_API __attribute__((visibility("default")))

#define LMZ_MAX_ESIZE 8

/* ---------------------------------------------------------------- scalar */

static void split2_scalar(const uint8_t *s, uint8_t *p0, uint8_t *p1, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        p0[i] = s[2 * i];
        p1[i] = s[2 * i + 1];
    }
}

static void merge2_scalar(const uint8_t *p0, const uint8_t *p1, uint8_t *d, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        d[2 * i] = p0[i];
        d[2 * i + 1] = p1[i];
    }
}

/* ------------------------------------------------------------------- x86 */

#if LMZ_X86

static void split2_sse2(const uint8_t *s, uint8_t *p0, uint8_t *p1, size_t n)
{
    const __m128i m = _mm_set1_epi16(0x00FF);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(s + 2 * i));
        __m128i b = _mm_loadu_si128((const __m128i *)(s + 2 * i + 16));
        _mm_storeu_si128((__m128i *)(p0 + i),
                         _mm_packus_epi16(_mm_and_si128(a, m), _mm_and_si128(b, m)));
        _mm_storeu_si128((__m128i *)(p1 + i),
                         _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8)));
    }
    if (i < n) split2_scalar(s + 2 * i, p0 + i, p1 + i, n - i);
}

static void merge2_sse2(const uint8_t *p0, const uint8_t *p1, uint8_t *d, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i l = _mm_loadu_si128((const __m128i *)(p0 + i));
        __m128i h = _mm_loadu_si128((const __m128i *)(p1 + i));
        _mm_storeu_si128((__m128i *)(d + 2 * i), _mm_unpacklo_epi8(l, h));
        _mm_storeu_si128((__m128i *)(d + 2 * i + 16), _mm_unpackhi_epi8(l, h));
    }
    if (i < n) merge2_scalar(p0 + i, p1 + i, d + 2 * i, n - i);
}

__attribute__((target("avx2")))
static void split2_avx2(const uint8_t *s, uint8_t *p0, uint8_t *p1, size_t n)
{
    const __m256i m = _mm256_set1_epi16(0x00FF);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(s + 2 * i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(s + 2 * i + 32));
        __m256i lo = _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
        __m256i hi = _mm256_packus_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
        /* packus works per 128-bit lane; restore linear order. */
        lo = _mm256_permute4x64_epi64(lo, 0xD8);
        hi = _mm256_permute4x64_epi64(hi, 0xD8);
        _mm256_storeu_si256((__m256i *)(p0 + i), lo);
        _mm256_storeu_si256((__m256i *)(p1 + i), hi);
    }
    if (i < n) split2_sse2(s + 2 * i, p0 + i, p1 + i, n - i);
}

__attribute__((target("avx2")))
static void merge2_avx2(const uint8_t *p0, const uint8_t *p1, uint8_t *d, size_t n)
{
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i l = _mm256_loadu_si256((const __m256i *)(p0 + i));
        __m256i h = _mm256_loadu_si256((const __m256i *)(p1 + i));
        /* Pre-permute so the per-lane unpacks land in linear order. */
        l = _mm256_permute4x64_epi64(l, 0xD8);
        h = _mm256_permute4x64_epi64(h, 0xD8);
        _mm256_storeu_si256((__m256i *)(d + 2 * i), _mm256_unpacklo_epi8(l, h));
        _mm256_storeu_si256((__m256i *)(d + 2 * i + 32), _mm256_unpackhi_epi8(l, h));
    }
    if (i < n) merge2_sse2(p0 + i, p1 + i, d + 2 * i, n - i);
}

static int lmz_have_avx2(void)
{
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    }
    return cached;
}

#endif /* LMZ_X86 */

/* ------------------------------------------------------------------ NEON */

#if LMZ_NEON

static void split2_neon(const uint8_t *s, uint8_t *p0, uint8_t *p1, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16x2_t v = vld2q_u8(s + 2 * i);
        vst1q_u8(p0 + i, v.val[0]);
        vst1q_u8(p1 + i, v.val[1]);
    }
    if (i < n) split2_scalar(s + 2 * i, p0 + i, p1 + i, n - i);
}

static void merge2_neon(const uint8_t *p0, const uint8_t *p1, uint8_t *d, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16x2_t v;
        v.val[0] = vld1q_u8(p0 + i);
        v.val[1] = vld1q_u8(p1 + i);
        vst2q_u8(d + 2 * i, v);
    }
    if (i < n) merge2_scalar(p0 + i, p1 + i, d + 2 * i, n - i);
}

#endif /* LMZ_NEON */

/* -------------------------------------------------------------- dispatch */

static void split2(const uint8_t *s, uint8_t *p0, uint8_t *p1, size_t n)
{
#if LMZ_X86
    if (lmz_have_avx2()) split2_avx2(s, p0, p1, n);
    else split2_sse2(s, p0, p1, n);
#elif LMZ_NEON
    split2_neon(s, p0, p1, n);
#else
    split2_scalar(s, p0, p1, n);
#endif
}

static void merge2(const uint8_t *p0, const uint8_t *p1, uint8_t *d, size_t n)
{
#if LMZ_X86
    if (lmz_have_avx2()) merge2_avx2(p0, p1, d, n);
    else merge2_sse2(p0, p1, d, n);
#elif LMZ_NEON
    merge2_neon(p0, p1, d, n);
#else
    merge2_scalar(p0, p1, d, n);
#endif
}

LMZ_API const char *lmz_isa(void)
{
#if LMZ_X86
    return lmz_have_avx2() ? "avx2" : "sse2";
#elif LMZ_NEON
    return "neon";
#else
    return "scalar";
#endif
}

LMZ_API int lmz_abi_version(void) { return 9; }

/*
 * out = a ^ b, the whole of a delta chunk's arithmetic.
 *
 * A training run rewrites every weight each checkpoint but moves each one very
 * little, so nothing dedups while the XOR against the previous checkpoint is
 * almost all zero bytes -- which the plane split and rANS then code for
 * nearly nothing. Word-at-a-time is enough here; the loop is memory bound and
 * the compiler vectorises it.
 */
LMZ_API int lmz_xor(const uint8_t *a, const uint8_t *b, size_t n, uint8_t *out)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t x, y;
        memcpy(&x, a + i, 8);
        memcpy(&y, b + i, 8);
        x ^= y;
        memcpy(out + i, &x, 8);
    }
    for (; i < n; i++) out[i] = (uint8_t)(a[i] ^ b[i]);
    return 0;
}

/* ------------------------------------------------------- recursive split */

/*
 * pos[j] is the byte position, within the original element, of sub-byte j of
 * the array currently being processed. Splitting deals the even-indexed
 * sub-bytes to one child and the odd-indexed ones to the other, so the
 * mapping stays correct at every depth without any index arithmetic tricks.
 *
 * scratch must have room for 2 * nelem * esize bytes: each level consumes
 * nelem*esize for its two halves, and the levels below it need half that
 * again, summing to strictly less than twice the top-level size. Siblings
 * run one after the other and write their results straight to `out`, so
 * they can safely share the same scratch region.
 */
static void split_rec(const uint8_t *src, size_t nelem, size_t esize,
                      uint8_t *const *outp, const size_t *pos, uint8_t *scratch)
{
    if (esize == 1) {
        memcpy(outp[pos[0]], src, nelem);
        return;
    }
    if (esize == 2) {
        split2(src, outp[pos[0]], outp[pos[1]], nelem);
        return;
    }

    size_t half_units = nelem * (esize / 2); /* 2-byte units in each half */
    uint8_t *a = scratch;
    uint8_t *b = scratch + half_units;
    uint8_t *next = scratch + nelem * esize;

    split2(src, a, b, half_units);

    size_t pa[LMZ_MAX_ESIZE], pb[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize / 2; j++) {
        pa[j] = pos[2 * j];
        pb[j] = pos[2 * j + 1];
    }
    split_rec(a, nelem, esize / 2, outp, pa, next);
    split_rec(b, nelem, esize / 2, outp, pb, next);
}

static void merge_rec(uint8_t *dst, size_t nelem, size_t esize,
                      const uint8_t *const *inp, const size_t *pos, uint8_t *scratch)
{
    if (esize == 1) {
        memcpy(dst, inp[pos[0]], nelem);
        return;
    }
    if (esize == 2) {
        merge2(inp[pos[0]], inp[pos[1]], dst, nelem);
        return;
    }

    size_t half_units = nelem * (esize / 2);
    uint8_t *a = scratch;
    uint8_t *b = scratch + half_units;
    uint8_t *next = scratch + nelem * esize;

    size_t pa[LMZ_MAX_ESIZE], pb[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize / 2; j++) {
        pa[j] = pos[2 * j];
        pb[j] = pos[2 * j + 1];
    }
    merge_rec(a, nelem, esize / 2, inp, pa, next);
    merge_rec(b, nelem, esize / 2, inp, pb, next);

    merge2(a, b, dst, half_units);
}

static int check_esize(size_t esize)
{
    return !(esize == 0 || esize > LMZ_MAX_ESIZE || (esize & (esize - 1)) != 0);
}

/*
 * Scatter/gather forms. Taking one pointer per plane lets the decoder feed
 * buffers that the entropy coder produced separately, instead of copying
 * them into one contiguous block first.
 */
LMZ_API int lmz_split_planes(const uint8_t *src, uint8_t *const *planes,
                             size_t nelem, size_t esize, uint8_t *scratch)
{
    if (!check_esize(esize)) return -1;
    if (nelem == 0) return 0;
    size_t pos[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize; j++) pos[j] = j;
    split_rec(src, nelem, esize, planes, pos, scratch);
    return 0;
}

LMZ_API int lmz_merge_planes(const uint8_t *const *planes, uint8_t *dst,
                             size_t nelem, size_t esize, uint8_t *scratch)
{
    if (!check_esize(esize)) return -1;
    if (nelem == 0) return 0;
    size_t pos[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize; j++) pos[j] = j;
    merge_rec(dst, nelem, esize, planes, pos, scratch);
    return 0;
}

/*
 * Split `nelem` elements of `esize` bytes into `esize` planes of `nelem`
 * bytes, laid out back to back in `out` (plane k at out + k*nelem).
 * Returns 0 on success, -1 for an unsupported element size.
 */
LMZ_API int lmz_split(const uint8_t *src, uint8_t *out, size_t nelem,
                      size_t esize, uint8_t *scratch)
{
    if (!check_esize(esize)) return -1;
    uint8_t *planes[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize; j++) planes[j] = out + j * nelem;
    return lmz_split_planes(src, planes, nelem, esize, scratch);
}

LMZ_API int lmz_merge(const uint8_t *in, uint8_t *dst, size_t nelem,
                      size_t esize, uint8_t *scratch)
{
    if (!check_esize(esize)) return -1;
    const uint8_t *planes[LMZ_MAX_ESIZE];
    for (size_t j = 0; j < esize; j++) planes[j] = in + j * nelem;
    return lmz_merge_planes(planes, dst, nelem, esize, scratch);
}

/* Scratch bytes required by lmz_split / lmz_merge for a given chunk. */
LMZ_API size_t lmz_scratch_size(size_t nelem, size_t esize)
{
    if (esize <= 2) return 0;
    return 2 * nelem * esize;
}

/* ----------------------------------------------------------- entropy aid */

/*
 * Byte histogram, four accumulators deep so that runs of a repeated value
 * do not serialise on store-to-load forwarding.
 */
LMZ_API void lmz_hist(const uint8_t *p, size_t n, uint64_t *hist)
{
    uint64_t h0[256] = {0}, h1[256] = {0}, h2[256] = {0}, h3[256] = {0};
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        h0[p[i]]++;
        h1[p[i + 1]]++;
        h2[p[i + 2]]++;
        h3[p[i + 3]]++;
    }
    for (; i < n; i++) h0[p[i]]++;
    for (int k = 0; k < 256; k++) hist[k] = h0[k] + h1[k] + h2[k] + h3[k];
}

/*
 * Count distinct byte values in a sample. A plane whose bytes take only a
 * few values is worth entropy coding; one that uses most of the alphabet
 * uniformly is not.
 */
LMZ_API int lmz_distinct(const uint8_t *p, size_t n)
{
    uint8_t seen[256] = {0};
    int c = 0;
    for (size_t i = 0; i < n; i++) {
        if (!seen[p[i]]) { seen[p[i]] = 1; c++; }
    }
    return c;
}

/* ------------------------------------------------------------- bf16 split */

/*
 * BF16 laid out on its own field boundaries rather than on byte boundaries.
 *
 * A plain byte split cuts a bfloat16 through the middle of its exponent: the
 * top byte gets the sign and seven exponent bits, the bottom byte gets the
 * eighth exponent bit welded to the mantissa. Those two fragments of one
 * exponent are correlated, and coding them as separate alphabets pays for
 * that correlation twice -- 0.065 bits per element on real weights.
 *
 * Regrouping costs nothing: plane A takes the whole 8-bit exponent, plane B
 * takes the sign in its top bit and the 7 mantissa bits below. Still two
 * bytes out per element, but now each plane is a field the model actually
 * has structure in.
 */
static void split_bf16_scalar(const uint8_t *s, uint8_t *pa, uint8_t *pb, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t w = (uint32_t)s[2 * i] | ((uint32_t)s[2 * i + 1] << 8);
        pa[i] = (uint8_t)((w >> 7) & 0xFF);
        pb[i] = (uint8_t)(((w >> 8) & 0x80) | (w & 0x7F));
    }
}

static void merge_bf16_scalar(const uint8_t *pa, const uint8_t *pb, uint8_t *d, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t a = pa[i], b = pb[i];
        uint32_t w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F);
        d[2 * i] = (uint8_t)(w & 0xFF);
        d[2 * i + 1] = (uint8_t)(w >> 8);
    }
}

LMZ_API int lmz_split_bf16(const uint8_t *src, uint8_t *pa, uint8_t *pb, size_t nelem)
{
    split_bf16_scalar(src, pa, pb, nelem);
    return 0;
}

LMZ_API int lmz_merge_bf16(const uint8_t *pa, const uint8_t *pb, uint8_t *dst,
                           size_t nelem)
{
    merge_bf16_scalar(pa, pb, dst, nelem);
    return 0;
}

/* -------------------------------------------------------- strided planes */

/*
 * Deinterleave fixed-period records into one plane per byte position, for
 * periods the power-of-two kernel cannot express. Built for GGUF's
 * block-quantised types: a Q8_0 block is 34 bytes -- a 2-byte fp16 scale
 * and 32 int8 quants -- and coding scale bytes and quant bytes in one
 * alphabet was measured to waste 2% of the payload on real weights.
 * The k-quants go wider: a Q6_K block is 210 bytes, which is where the
 * period ceiling comes from.
 *
 * One sequential read against `period` write cursors; the cursor array
 * lives in L1 and the loop runs at memory speed for the periods used here.
 */

#define LMZ_MAX_PERIOD 256

LMZ_API int lmz_split_stride(const uint8_t *src, uint8_t *out,
                             size_t nblocks, size_t period)
{
    if (period == 0 || period > LMZ_MAX_PERIOD) return -1;
    uint8_t *cur[LMZ_MAX_PERIOD];
    for (size_t k = 0; k < period; k++) cur[k] = out + k * nblocks;
    const uint8_t *p = src;
    for (size_t i = 0; i < nblocks; i++)
        for (size_t k = 0; k < period; k++)
            *cur[k]++ = *p++;
    return 0;
}

/* Inverse, planes given one pointer per byte position. */
LMZ_API int lmz_merge_stride(const uint8_t *const *planes, uint8_t *dst,
                             size_t nblocks, size_t period)
{
    if (period == 0 || period > LMZ_MAX_PERIOD) return -1;
    const uint8_t *cur[LMZ_MAX_PERIOD];
    for (size_t k = 0; k < period; k++) cur[k] = planes[k];
    uint8_t *p = dst;
    for (size_t i = 0; i < nblocks; i++)
        for (size_t k = 0; k < period; k++)
            *p++ = *cur[k]++;
    return 0;
}

/* ------------------------------------------------- bucketed conditioning */

/*
 * The sign+mantissa plane is not quite independent of the exponent: on real
 * Llama weights the joint 16-bit entropy sits 0.075 bits per element below
 * the sum of the two planes' own entropies. Grouping elements by their
 * exponent and giving each group its own frequency table collects that gap,
 * and eight equal-mass groups were measured to collect essentially all of it
 * (7.8407 bits conditional versus 7.8407 with the full 256-way context).
 *
 * The bucket map is a pure function of the exponent histogram, so it is
 * never stored: the decoder recovers the exponent plane first and rebuilds
 * the identical map from the identical bytes.
 */

#define LMZ_MAX_BUCKETS 32

/* Contiguous equal-mass buckets over the symbol alphabet. Deterministic:
 * both sides derive it from the same histogram with the same integers. */
LMZ_API int lmz_bucket_lut(const uint64_t *hist, size_t k, uint8_t *lut)
{
    if (k == 0 || k > LMZ_MAX_BUCKETS) return -1;
    uint64_t total = 0;
    for (int s = 0; s < 256; s++) total += hist[s];
    if (total == 0) {
        memset(lut, 0, 256);
        return 0;
    }
    uint64_t acc = 0;
    uint64_t b = 0;
    for (int s = 0; s < 256; s++) {
        lut[s] = (uint8_t)b;
        acc += hist[s];
        /* Move on once this bucket holds its share of the mass. */
        while (b + 1 < k && acc * k >= (b + 1) * total) b++;
    }
    return 0;
}

/* Deal val[i] into per-bucket segments of `out` by ctx[i]'s bucket, keeping
 * order within each bucket. counts[k] receives the segment lengths. */
LMZ_API int lmz_bucket_partition(const uint8_t *ctx, const uint8_t *val,
                                 size_t n, const uint8_t *lut, size_t k,
                                 uint8_t *out, uint64_t *counts)
{
    if (k == 0 || k > LMZ_MAX_BUCKETS) return -1;
    uint64_t hist[256];
    lmz_hist(ctx, n, hist);
    for (size_t b = 0; b < k; b++) counts[b] = 0;
    for (int s = 0; s < 256; s++) counts[lut[s]] += hist[s];
    uint8_t *cur[LMZ_MAX_BUCKETS];
    uint8_t *p = out;
    for (size_t b = 0; b < k; b++) {
        cur[b] = p;
        p += counts[b];
    }
    for (size_t i = 0; i < n; i++) *cur[lut[ctx[i]]]++ = val[i];
    return 0;
}

/* Inverse of the partition: streams[b] holds bucket b's segment. */
LMZ_API int lmz_bucket_unpartition(const uint8_t *ctx,
                                   const uint8_t *const *streams, size_t n,
                                   const uint8_t *lut, size_t k,
                                   uint8_t *val_out)
{
    if (k == 0 || k > LMZ_MAX_BUCKETS) return -1;
    const uint8_t *cur[LMZ_MAX_BUCKETS];
    for (size_t b = 0; b < k; b++) cur[b] = streams[b];
    for (size_t i = 0; i < n; i++) val_out[i] = *cur[lut[ctx[i]]]++;
    return 0;
}

/* ------------------------------------------------- k-quant sub-block work */

/*
 * A Q4_K super-block holds 256 weights in eight sub-blocks, each with its own
 * 6-bit scale and 6-bit min; the quant a sub-block stores is `d*q - m`. So a
 * nibble's distribution depends on which sub-block it belongs to, and the
 * sub-block's parameters are decoded before its quants -- a context that
 * costs nothing to transmit. Conditioning on it was measured at 9.7 bits per
 * block on real Llama Q4_K weights, against 0.02 bits for every context that
 * can be built inside a Q8_0 block.
 *
 * Two obstacles, one per function below. The sub-block parameters are packed
 * six bits at a time across twelve bytes, four of each straddling a byte
 * boundary; and the quants themselves interleave two sub-blocks per byte, so
 * a byte-plane split leaves every byte holding two different alphabets.
 */

/*
 * ggml's get_scale_min_k4, applied plane-wise across a whole chunk. `sp` is
 * twelve planes of `nblocks` bytes; `sc` and `mn` receive eight each.
 *
 * Only ever used to derive a context. The twelve bytes are still coded as
 * their own field, so an unpacking that disagreed with ggml would cost ratio
 * and could not cost correctness.
 */
LMZ_API int lmz_k4_scales(const uint8_t *sp, size_t nblocks,
                          uint8_t *sc, uint8_t *mn)
{
    for (size_t j = 0; j < 4; j++) {
        const uint8_t *a = sp + j * nblocks;
        const uint8_t *b = sp + (j + 4) * nblocks;
        uint8_t *ds = sc + j * nblocks;
        uint8_t *dm = mn + j * nblocks;
        for (size_t i = 0; i < nblocks; i++) {
            ds[i] = a[i] & 63;
            dm[i] = b[i] & 63;
        }
    }
    for (size_t j = 4; j < 8; j++) {
        const uint8_t *lo = sp + (j + 4) * nblocks;
        const uint8_t *hs = sp + (j - 4) * nblocks;
        const uint8_t *hm = sp + j * nblocks;
        uint8_t *ds = sc + j * nblocks;
        uint8_t *dm = mn + j * nblocks;
        for (size_t i = 0; i < nblocks; i++) {
            ds[i] = (uint8_t)((lo[i] & 0x0F) | ((hs[i] >> 6) << 4));
            dm[i] = (uint8_t)((lo[i] >> 4) | ((hm[i] >> 6) << 4));
        }
    }
    return 0;
}

/*
 * Regroup 128 quant planes so each output plane belongs to one sub-block.
 *
 * ggml stores sub-block 2g in the low nibbles of quant bytes [32g, 32g+32)
 * and sub-block 2g+1 in their high nibbles. Pairing two nibbles of the *same*
 * sub-block back into a byte keeps one symbol per two quants -- so the coder
 * still runs at byte rate over a 256-entry table -- while giving every byte a
 * single context. Adjacent quants measured independent, so the pairing costs
 * nothing: 979.53 bits per block packed against 979.90 coded as loose
 * nibbles, and the packed form needs half the symbols.
 */
LMZ_API int lmz_k4_pack(const uint8_t *q, size_t nblocks, uint8_t *packed)
{
    for (size_t s = 0; s < 8; s++) {
        const size_t g = s >> 1, sh = (s & 1) ? 4 : 0;
        for (size_t i = 0; i < 16; i++) {
            const uint8_t *a = q + (32 * g + 2 * i) * nblocks;
            const uint8_t *b = q + (32 * g + 2 * i + 1) * nblocks;
            uint8_t *d = packed + (16 * s + i) * nblocks;
            for (size_t n = 0; n < nblocks; n++)
                d[n] = (uint8_t)(((a[n] >> sh) & 0x0F)
                                 | (((b[n] >> sh) & 0x0F) << 4));
        }
    }
    return 0;
}

/* Inverse of lmz_k4_pack. */
LMZ_API int lmz_k4_unpack(const uint8_t *packed, size_t nblocks, uint8_t *q)
{
    for (size_t c = 0; c < 128; c++) {
        const size_t g = c >> 5, t = c & 31, sh = 4 * (t & 1);
        const uint8_t *lo = packed + (16 * (2 * g) + (t >> 1)) * nblocks;
        const uint8_t *hi = packed + (16 * (2 * g + 1) + (t >> 1)) * nblocks;
        uint8_t *d = q + c * nblocks;
        for (size_t n = 0; n < nblocks; n++)
            d[n] = (uint8_t)(((lo[n] >> sh) & 0x0F)
                             | (((hi[n] >> sh) & 0x0F) << 4));
    }
    return 0;
}

/* ------------------------------------------------------------------ rANS */

/*
 * Static order-0 range Asymmetric Numeral Systems, after Duda; the byte-wise
 * renormalising variant and the reciprocal-multiply encoder are Giesen's.
 *
 * This exists because a general-purpose compressor cannot reach the entropy
 * of this data. An exponent plane is an order-0 symbol stream of about 32
 * values with no useful repeats, and zstd's match finder both wastes time on
 * it and dilutes its own Huffman stage -- measured at 11% above the order-0
 * bound on real BF16 weights. Huffman alone still gives up 1.6%, because it
 * must round every symbol to a whole bit. rANS pays fractional bits, so it
 * lands within a fraction of a percent of entropy, which is what turns a
 * 31.6% saving into a 34%+ one.
 *
 * Four states are interleaved so the decoder has four independent dependency
 * chains in flight; a single-state decoder stalls on its own renormalisation.
 */

#define RANS_PROB_BITS 12
#define RANS_PROB_SCALE (1u << RANS_PROB_BITS)
/*
 * Renormalisation floor. Refilling 16 bits at a time against 12-bit
 * probabilities means a step can never need more than one refill, so the
 * refill loop collapses to a single branch. Refilling a byte at a time
 * instead needs a loop running zero, one or two times, and at these symbol
 * frequencies that branch is close to unpredictable -- it measured as the
 * dominant cost, at roughly 13 cycles per symbol.
 */
#define RANS_L (1u << 16)
#define RANS_STREAMS 8
#define RANS_HEADER 516 /* magic(2) + reserved(2) + 256 * uint16 frequency */
#define RANS_MAGIC0 'R'
#define RANS_MAGIC1 '1'

typedef struct {
    uint32_t freq, bias, cmpl_freq;
} RansEncSym;

static void rans_enc_sym_init(RansEncSym *s, uint32_t start, uint32_t freq)
{
    s->freq = freq;
    s->cmpl_freq = RANS_PROB_SCALE - freq;
    s->bias = start;
}

static inline void rans_enc_put(uint32_t *r, uint8_t **pptr, const RansEncSym *sym)
{
    uint32_t x = *r;
    /* 64-bit: for a symbol occupying the whole probability range this is
     * exactly 2^32, and truncating it to 32 bits would wrap to zero and
     * renormalise on every symbol. */
    uint64_t x_max = ((uint64_t)(RANS_L >> RANS_PROB_BITS) << 16) * sym->freq;
    if (x >= x_max) {
        uint8_t *p = *pptr - 2;
        p[0] = (uint8_t)(x & 0xff);
        p[1] = (uint8_t)((x >> 8) & 0xff);
        *pptr = p;
        x >>= 16;
    }
    /* The quotient must be floor(x/freq) EXACTLY for every state the
     * renormalisation allows, which here reaches (1<<20)*freq -- past 2^31
     * whenever freq exceeds 2048. The fixed-point reciprocal this used to
     * use (after Giesen) is only exact below 2^31: it was built for a coder
     * whose renormalisation keeps states under that line, and this one does
     * not. Above it, a majority symbol's quotient could come out one high,
     * landing the state in a neighbouring symbol's slot -- corruption that
     * only strikes planes where one byte value passes 50% frequency, which
     * float planes never produce but quantised-weight planes do. Hardware
     * division is exact everywhere, and the encoder stays memory-bound at
     * the thread counts that matter.
     */
    uint32_t q = x / sym->freq;
    *r = x + sym->bias + q * sym->cmpl_freq;
}

static inline void rans_enc_flush(uint32_t *r, uint8_t **pptr)
{
    uint32_t x = *r;
    uint8_t *p = *pptr - 4;
    p[0] = (uint8_t)(x >> 0);
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
    *pptr = p;
}

static inline void rans_dec_init(uint32_t *r, const uint8_t **pptr)
{
    const uint8_t *p = *pptr;
    *r = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
    *pptr = p + 4;
}

/*
 * Branchless step. Whether a symbol needs a refill depends on how many bits
 * it consumed, which for a near-uniform plane is close to a coin flip -- as
 * a branch it mispredicts about half the time and dominates the decode. So
 * the refill word is always loaded and then applied conditionally, which
 * compiles to a conditional move and a pointer add.
 *
 * The unconditional load reads up to two bytes beyond the cursor, so callers
 * must only use this while a margin of readable input remains.
 */
static inline uint32_t rans_dec_advance_fast(uint32_t x, const uint8_t **pptr,
                                             uint32_t start, uint32_t freq)
{
    x = freq * (x >> RANS_PROB_BITS) + (x & (RANS_PROB_SCALE - 1)) - start;
    const uint8_t *p = *pptr;
    uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
    uint32_t need = (x < RANS_L);
    x = need ? ((x << 16) | word) : x;
    *pptr = p + (need << 1);
    return x;
}

/* Bounds-checked equivalent, for the last few symbols. */
static inline int rans_dec_advance_safe(uint32_t *r, const uint8_t **pptr,
                                        const uint8_t *limit,
                                        uint32_t start, uint32_t freq)
{
    uint32_t x = *r;
    x = freq * (x >> RANS_PROB_BITS) + (x & (RANS_PROB_SCALE - 1)) - start;
    if (x < RANS_L) {
        const uint8_t *p = *pptr;
        if (p + 2 > limit) return -1;
        x = (x << 16) | (uint32_t)p[0] | ((uint32_t)p[1] << 8);
        *pptr = p + 2;
    }
    *r = x;
    return 0;
}

/*
 * Scale counts so they sum to exactly RANS_PROB_SCALE, never letting a symbol
 * that occurs reach frequency zero -- the coder cannot represent a symbol it
 * believes impossible, and the histogram came from this very buffer.
 */
static int rans_normalize(const uint64_t *counts, uint16_t *freqs)
{
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) total += counts[i];
    if (total == 0) return -1;

    uint32_t sum = 0;
    int biggest = -1;
    uint64_t biggest_count = 0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) {
            freqs[i] = 0;
            continue;
        }
        uint64_t f = (counts[i] * RANS_PROB_SCALE) / total;
        if (f == 0) f = 1;
        if (f > RANS_PROB_SCALE) f = RANS_PROB_SCALE;
        freqs[i] = (uint16_t)f;
        sum += (uint32_t)f;
        if (counts[i] > biggest_count) {
            biggest_count = counts[i];
            biggest = i;
        }
    }
    if (biggest < 0) return -1;

    if (sum < RANS_PROB_SCALE) {
        freqs[biggest] = (uint16_t)(freqs[biggest] + (RANS_PROB_SCALE - sum));
    } else {
        /* Rounding up the rare symbols overshot; claw the excess back from
         * the most frequent ones, which lose the least by giving it up. */
        uint32_t excess = sum - RANS_PROB_SCALE;
        while (excess > 0) {
            int bi = -1;
            uint32_t bf = 1;
            for (int i = 0; i < 256; i++) {
                if (freqs[i] > bf) { bf = freqs[i]; bi = i; }
            }
            if (bi < 0) return -1;
            uint32_t take = (bf - 1 < excess) ? (bf - 1) : excess;
            freqs[bi] = (uint16_t)(freqs[bi] - take);
            excess -= take;
        }
    }

    for (int i = 0; i < 256; i++) {
        if ((counts[i] != 0) != (freqs[i] != 0)) return -1;
    }
    return 0;
}

LMZ_API size_t lmz_rans_bound(size_t n)
{
    /* Worst case is one 16-bit renormalisation per symbol, plus the states
     * flushed at the end. */
    return RANS_HEADER + 2 * n + 4 * RANS_STREAMS + 64;
}

/*
 * Encode `n` bytes. Returns the stream length, or -1 if it does not fit.
 * The layout is [header][4 initial states][coded bytes].
 */
LMZ_API long lmz_rans_encode(const uint8_t *src, size_t n, uint8_t *dst, size_t cap)
{
    if (n == 0) return -1;
    if (cap < lmz_rans_bound(n)) return -1;

    uint64_t counts[256];
    lmz_hist(src, n, counts);
    uint16_t freqs[256];
    if (rans_normalize(counts, freqs) != 0) return -1;

    RansEncSym syms[256];
    uint32_t start = 0;
    for (int s = 0; s < 256; s++) {
        if (freqs[s]) {
            rans_enc_sym_init(&syms[s], start, freqs[s]);
            start += freqs[s];
        }
    }

    /* rANS is last-in-first-out, so encoding runs backwards from the end of
     * the buffer and the result is moved down once its size is known. */
    uint8_t *end = dst + cap;
    uint8_t *ptr = end;
    uint32_t state[RANS_STREAMS];
    for (int k = 0; k < RANS_STREAMS; k++) state[k] = RANS_L;

    /* Symbol j is encoded with state[j & (STREAMS-1)]. Encoding walks
     * backwards, so the ragged tail is dealt with first. */
    size_t i = n;
    while (i > 0 && (i & (RANS_STREAMS - 1))) {
        i--;
        rans_enc_put(&state[i & (RANS_STREAMS - 1)], &ptr, &syms[src[i]]);
    }
    while (i >= RANS_STREAMS) {
        i -= RANS_STREAMS;
        for (int k = RANS_STREAMS - 1; k >= 0; k--)
            rans_enc_put(&state[k], &ptr, &syms[src[i + k]]);
    }
    for (int k = RANS_STREAMS - 1; k >= 0; k--) rans_enc_flush(&state[k], &ptr);

    size_t coded = (size_t)(end - ptr);
    size_t total = RANS_HEADER + coded;
    if (total > cap) return -1;

    dst[0] = RANS_MAGIC0;
    dst[1] = RANS_MAGIC1;
    dst[2] = 0;
    dst[3] = 0;
    for (int s = 0; s < 256; s++) {
        dst[4 + 2 * s] = (uint8_t)(freqs[s] & 0xff);
        dst[5 + 2 * s] = (uint8_t)(freqs[s] >> 8);
    }
    memmove(dst + RANS_HEADER, ptr, coded);
    return (long)total;
}

/* Decode exactly `n` bytes. Returns 0 on success, -1 on malformed input. */
LMZ_API int lmz_rans_decode(const uint8_t *src, size_t src_len, uint8_t *dst, size_t n)
{
    if (n == 0) return src_len == 0 ? 0 : -1;
    if (src_len < RANS_HEADER + 4 * RANS_STREAMS) return -1;
    if (src[0] != RANS_MAGIC0 || src[1] != RANS_MAGIC1) return -1;

    uint16_t freqs[256];
    uint32_t sum = 0;
    for (int s = 0; s < 256; s++) {
        freqs[s] = (uint16_t)(src[4 + 2 * s] | ((uint16_t)src[5 + 2 * s] << 8));
        sum += freqs[s];
    }
    if (sum != RANS_PROB_SCALE) return -1;

    /* One 32-bit entry per probability slot: frequency, cumulative start and
     * symbol together, so a decode step is a single dependent load. 16 KiB,
     * kept on the stack because a per-call heap allocation here would sit in
     * the hot path of every plane of every chunk.
     *
     * Frequency is stored biased by one. It ranges 1..RANS_PROB_SCALE, and
     * the full-scale case -- a stream of a single repeated symbol -- would
     * otherwise overflow its 12-bit field and decode as frequency zero.
     */
    uint32_t lut[RANS_PROB_SCALE];
    uint32_t start = 0;
    for (int s = 0; s < 256; s++) {
        uint32_t f = freqs[s];
        if (!f) continue;
        uint32_t packed = ((f - 1) << 20) | (start << 8) | (uint32_t)s;
        for (uint32_t j = 0; j < f; j++) lut[start + j] = packed;
        start += f;
    }
    if (start != RANS_PROB_SCALE) return -1;

    const uint8_t *ptr = src + RANS_HEADER;
    const uint8_t *limit = src + src_len;
    uint32_t state[RANS_STREAMS];
    for (int k = 0; k < RANS_STREAMS; k++) rans_dec_init(&state[k], &ptr);

    /* Symbol j was encoded with state[j & 3]. Decoding runs forward from
     * zero, so the aligned block comes first and the ragged tail last --
     * the reverse of the encoder, whose backwards walk meets the ragged
     * part first.
     */
    size_t i = 0;
    /* Fast path, while enough readable input remains for the block's worth
     * of unconditional refill loads. It always stops on a multiple of
     * RANS_STREAMS, so the tail below resumes on the right state. */
    for (; i + RANS_STREAMS <= n && ptr + 2 * RANS_STREAMS <= limit;
         i += RANS_STREAMS) {
        /* Every table lookup is issued before any state advances, so the
         * loads overlap instead of serialising behind one another. */
        uint32_t e[RANS_STREAMS];
        for (int k = 0; k < RANS_STREAMS; k++)
            e[k] = lut[state[k] & (RANS_PROB_SCALE - 1)];
        for (int k = 0; k < RANS_STREAMS; k++)
            dst[i + k] = (uint8_t)(e[k] & 0xff);
        for (int k = 0; k < RANS_STREAMS; k++)
            state[k] = rans_dec_advance_fast(state[k], &ptr,
                                             (e[k] >> 8) & 0xfff, (e[k] >> 20) + 1);
    }
    for (; i < n; i++) {
        uint32_t k = (uint32_t)(i & (RANS_STREAMS - 1));
        uint32_t e = lut[state[k] & (RANS_PROB_SCALE - 1)];
        dst[i] = (uint8_t)(e & 0xff);
        if (rans_dec_advance_safe(&state[k], &ptr, limit,
                                  (e >> 8) & 0xfff, (e >> 20) + 1) != 0)
            return -1;
    }
    return 0;
}
