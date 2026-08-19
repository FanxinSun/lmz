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

/*
 * LMZ_FORCE_NEON is how the arm64 kernels get checked on a machine that is not
 * arm64: a harness puts its own arm_neon.h in front of this include and the
 * shipped source, unmodified, compiles down the arm64 path. Correctness is
 * then testable everywhere and only the timings need real hardware --
 * scratchpad/neonshim is that harness. Nothing defines this in a normal build.
 */
#if defined(LMZ_FORCE_NEON)
#define LMZ_NEON 1
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(__i386__)
#define LMZ_X86 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#define LMZ_NEON 1
#include <arm_neon.h>
#endif

/*
 * The vector encoder needs what aarch64 added to NEON and 32-bit ARM never
 * had: a sixteen-byte table lookup, an across-vector add, and the paired
 * unzips. Big-endian would also lay the emitted words down in the wrong
 * order. Both cases fall through to the scalar body, which is correct
 * everywhere.
 */
#if LMZ_NEON && (defined(__aarch64__) || defined(LMZ_FORCE_NEON)) \
    && !defined(__ARM_BIG_ENDIAN)
#define LMZ_NEON_ENC 1
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

LMZ_API int lmz_abi_version(void) { return 11; }

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
 * order within each bucket. counts[k] receives the segment lengths.
 *
 * `ctx_hist` is the context plane's histogram when the caller has one. The lut
 * is built from that same histogram, so the caller always does -- and counting
 * it again here is a second pass over the context plane. NULL to count it
 * here; counts that do not sum to `n` are not this plane's and are ignored. */
LMZ_API int lmz_bucket_partition(const uint8_t *ctx, const uint8_t *val,
                                 size_t n, const uint8_t *lut, size_t k,
                                 uint8_t *out, uint64_t *counts,
                                 const uint64_t *ctx_hist)
{
    if (k == 0 || k > LMZ_MAX_BUCKETS) return -1;
    uint64_t own[256];
    if (ctx_hist) {
        uint64_t total = 0;
        for (int s = 0; s < 256; s++) total += ctx_hist[s];
        if (total != n) ctx_hist = NULL;
    }
    if (!ctx_hist) {
        lmz_hist(ctx, n, own);
        ctx_hist = own;
    }
    for (size_t b = 0; b < k; b++) counts[b] = 0;
    for (int s = 0; s < 256; s++) counts[lut[s]] += ctx_hist[s];
    uint8_t *cur[LMZ_MAX_BUCKETS];
    uint8_t *p = out;
    for (size_t b = 0; b < k; b++) {
        cur[b] = p;
        p += counts[b];
    }
    /*
     * Four at a time. Written one at a time, each element has to load the
     * cursor the one before it just stored, and neighbouring context bytes
     * land in the same bucket constantly -- so the loop spends its time
     * waiting for stores to forward rather than moving bytes.
     *
     * Taking four cursors before any of them is written breaks that. An
     * element's cursor is then corrected by how many earlier elements of its
     * group share its bucket, which is a compare and an add and needs no
     * load; the write-backs follow in group order, so the last one for a
     * bucket leaves exactly what a serial loop would have left. Measured
     * 1.22x on a real exponent plane. Groups of five and up lose it again:
     * the corrections grow as the square of the group size.
     */
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        unsigned b0 = lut[ctx[i]], b1 = lut[ctx[i + 1]];
        unsigned b2 = lut[ctx[i + 2]], b3 = lut[ctx[i + 3]];
        uint8_t *p0 = cur[b0];
        uint8_t *p1 = cur[b1] + (b0 == b1);
        uint8_t *p2 = cur[b2] + (b0 == b2) + (b1 == b2);
        uint8_t *p3 = cur[b3] + (b0 == b3) + (b1 == b3) + (b2 == b3);
        *p0 = val[i];
        *p1 = val[i + 1];
        *p2 = val[i + 2];
        *p3 = val[i + 3];
        cur[b0] = p0 + 1;
        cur[b1] = p1 + 1;
        cur[b2] = p2 + 1;
        cur[b3] = p3 + 1;
    }
    for (; i < n; i++) *cur[lut[ctx[i]]]++ = val[i];
    return 0;
}

/* Inverse of the partition: streams[b] holds bucket b's segment. The cursors
 * are read from rather than written to, but they are the same chain and take
 * the same grouping; see above. */
LMZ_API int lmz_bucket_unpartition(const uint8_t *ctx,
                                   const uint8_t *const *streams, size_t n,
                                   const uint8_t *lut, size_t k,
                                   uint8_t *val_out)
{
    if (k == 0 || k > LMZ_MAX_BUCKETS) return -1;
    const uint8_t *cur[LMZ_MAX_BUCKETS];
    for (size_t b = 0; b < k; b++) cur[b] = streams[b];
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        unsigned b0 = lut[ctx[i]], b1 = lut[ctx[i + 1]];
        unsigned b2 = lut[ctx[i + 2]], b3 = lut[ctx[i + 3]];
        const uint8_t *p0 = cur[b0];
        const uint8_t *p1 = cur[b1] + (b0 == b1);
        const uint8_t *p2 = cur[b2] + (b0 == b2) + (b1 == b2);
        const uint8_t *p3 = cur[b3] + (b0 == b3) + (b1 == b3) + (b2 == b3);
        val_out[i] = *p0;
        val_out[i + 1] = *p1;
        val_out[i + 2] = *p2;
        val_out[i + 3] = *p3;
        cur[b0] = p0 + 1;
        cur[b1] = p1 + 1;
        cur[b2] = p2 + 1;
        cur[b3] = p3 + 1;
    }
    for (; i < n; i++) val_out[i] = *cur[lut[ctx[i]]]++;
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

/*
 * x_max is freq << RANS_XSHIFT: the same quantity the step below builds as
 * ((RANS_L >> RANS_PROB_BITS) << 16) * freq, in a form that compares in 32
 * bits so the vector path does not need a 64-bit multiply per symbol.
 */
#define RANS_XSHIFT 20
_Static_assert(((RANS_L >> RANS_PROB_BITS) << 16) == (1u << RANS_XSHIFT),
               "RANS_XSHIFT must match the renormalisation bound");

/*
 * Above this frequency the vector path declines and the scalar loop runs.
 * Its fixed-point reciprocal is exact only while states stay below 2^31, and
 * states reach freq << 20, so the two meet at half the probability scale. No
 * float plane produces a symbol that common; quantised-weight planes do, and
 * those get the hardware divide, which is exact everywhere. This is the same
 * boundary rans_enc_put's comment describes.
 */
#define RANS_SIMD_MAX_FREQ (1u << (31 - RANS_XSHIFT))

/* Below this the vector path's own tables cost more than stepping eight at a
 * time saves. Small streams are all setup. */
#define RANS_SIMD_MIN 4096

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
    /* Branchless, for the same reason the decoder is: whether a symbol needs
     * to emit is close to a coin flip on a near-uniform plane, and as a branch
     * it mispredicts about half the time. Encoding runs backwards, so the two
     * bytes below the cursor are always scratch -- write them every time and
     * advance the cursor only when the emission was real. The write that was
     * not needed is overwritten by the next one. */
    uint32_t need = (x >= x_max);
    uint8_t *p = *pptr;
    p[-2] = (uint8_t)(x & 0xff);
    p[-1] = (uint8_t)((x >> 8) & 0xff);
    *pptr = p - (need << 1);
    x = need ? (x >> 16) : x;
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

/*
 * The frequencies the encoder will use for this histogram, so a caller can
 * price a stream before coding it.
 *
 * The order-0 entropy of the raw counts is not that price. Frequencies are
 * quantised to twelve bits, and close to a uniform alphabet -- which is
 * exactly where the decision is marginal -- that rounding costs a few tenths
 * of a percent, enough to turn a stream that looks worth coding into one that
 * is stored after the work is done. Charging against these numbers instead
 * answers the real question for the price of 256 logarithms.
 */
LMZ_API int lmz_rans_freqs(const uint64_t *counts, uint16_t *freqs)
{
    return rans_normalize(counts, freqs);
}

LMZ_API size_t lmz_rans_bound(size_t n)
{
    /* Worst case is one 16-bit renormalisation per symbol, plus the states
     * flushed at the end. */
    return RANS_HEADER + 2 * n + 4 * RANS_STREAMS + 64;
}

/* Encode the body backwards from `end`, returning where the coded bytes
 * start. Split out so the vector path below can be swapped in whole. */
static uint8_t *rans_enc_body(const uint8_t *src, size_t n, uint8_t *end,
                              const RansEncSym *syms)
{
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
    return ptr;
}

/*
 * Tables the vector encoders share.
 *
 * Both ISAs want the same three numbers per symbol and the same shuffle per
 * emission pattern; what differs is only how they are laid out in a register.
 * So the parts that are easy to get wrong -- the fixed-point reciprocal, and
 * what a frequency below two does to it -- are written once and read twice.
 */
#if LMZ_X86 || LMZ_NEON_ENC

/*
 * freq, bias and the reciprocal's shift share one word, so a lane needs two
 * loads rather than four; the complement is 4096 - freq and is computed rather
 * than stored.
 */
static void rans_simd_tables(const uint16_t *freqs, uint32_t *packed,
                             uint32_t *rcp)
{
    uint32_t start = 0;
    for (int s = 0; s < 256; s++) {
        uint32_t f = freqs[s], bias = start, r, sh;
        if (f < 2) {
            /* A quotient of x itself is not something the reciprocal can
             * express, so it comes out one low and the bias makes up the
             * difference. That keeps the lane step free of any special case. */
            r = ~0u;
            sh = 0;
            bias = start + RANS_PROB_SCALE - 1;
        } else {
            sh = 0;
            while (f > (1u << sh)) sh++;
            r = (uint32_t)((((uint64_t)1 << (sh + 31)) + f - 1) / f);
            sh -= 1;
        }
        packed[s] = (f ? f : 1u) | (bias << 12) | (sh << 28);
        rcp[s] = r;
        start += f;
    }
}

/*
 * One shuffle control per emission pattern. Lane j's two bytes are sent to
 * slot 8 - popcount(m) + (how many lower lanes emitted), which leaves the
 * words that really emitted contiguous, right-aligned in sixteen bytes and in
 * lane order -- the order the scalar loop writes them in. 0x80 selects a zero
 * byte on both ISAs, x86 because the high bit is set and NEON because the
 * index is outside the table, and those bytes land where nothing is kept.
 *
 * Built per call for the same reason the decoder builds its table per call: a
 * heap allocation would sit in the hot path of every plane of every chunk, and
 * this is 4 KiB of stack against a plane measured in megabytes.
 */
static void rans_compact_table(uint8_t compact[256][16])
{
    for (unsigned m = 0; m < 256; m++) {
        int slot = 8 - __builtin_popcount(m);
        for (int b = 0; b < 16; b++) compact[m][b] = 0x80;
        for (int j = 0; j < 8; j++) {
            if (!(m & (1u << j))) continue;
            compact[m][2 * slot] = (uint8_t)(2 * j);
            compact[m][2 * slot + 1] = (uint8_t)(2 * j + 1);
            slot++;
        }
    }
}

#endif /* LMZ_X86 || LMZ_NEON_ENC */

#if LMZ_X86

__attribute__((target("avx2")))
static inline __m256i rans_mulhi_epu32(__m256i a, __m256i b)
{
    /* mul_epu32 multiplies the low half of each 64-bit lane, so the even and
     * odd dwords are done separately and blended back into place. */
    __m256i lo = _mm256_mul_epu32(a, b);
    __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(a, 32),
                                  _mm256_srli_epi64(b, 32));
    return _mm256_blend_epi32(_mm256_srli_epi64(lo, 32), hi, 0xAA);
}

/*
 * The same eight states, stepped eight at a time.
 *
 * Eight 32-bit states are exactly one AVX2 register, and eight is what this
 * format already interleaves -- so this is not another stream layout, it is
 * the same arithmetic in parallel, and it writes the same bytes. Two things in
 * the scalar step have no vector instruction:
 *
 *   the divide, which becomes a fixed-point reciprocal after Giesen. That is
 *   exact only over the range RANS_SIMD_MAX_FREQ describes, which is why the
 *   caller checks the frequencies before choosing this path;
 *
 *   the emission, because lanes renormalise independently and their words have
 *   to be made contiguous before they can be written. They are packed down,
 *   compacted to the right by a shuffle chosen from the emission mask, and
 *   stored unconditionally sixteen bytes below the cursor -- which is scratch,
 *   since encoding runs backwards. Only the cursor moves by the number of
 *   lanes that really emitted, so the lanes that did not are overwritten by
 *   the next group.
 *
 * Measured 1.94x over the scalar loop on real BF16 planes, and within 3% of
 * that loop with its table lookups removed altogether -- so what remains is
 * the latency of one dependency chain, not anything left on the table. Only
 * more interleaved states would shorten it, and the format fixes those at
 * eight.
 */
__attribute__((target("avx2")))
static uint8_t *rans_enc_body_avx2(const uint8_t *src, size_t n, uint8_t *end,
                                   const RansEncSym *syms, const uint16_t *freqs)
{
    uint32_t packed[256], rcp[256];
    rans_simd_tables(freqs, packed, rcp);
    uint8_t compact[256][16];
    rans_compact_table(compact);

    /* The low half of each 32-bit lane, four to a 128-bit half. */
    static const uint8_t pack_ctl[32] = {
        0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    };

    uint8_t *ptr = end;
    uint32_t state[RANS_STREAMS];
    for (int k = 0; k < RANS_STREAMS; k++) state[k] = RANS_L;

    /* The ragged tail, exactly where the scalar loop meets it. */
    size_t i = n;
    while (i > 0 && (i & (RANS_STREAMS - 1))) {
        i--;
        rans_enc_put(&state[i & (RANS_STREAMS - 1)], &ptr, &syms[src[i]]);
    }

    __m256i x = _mm256_loadu_si256((const __m256i *)state);
    const __m256i m12 = _mm256_set1_epi32(0xFFF);
    const __m256i m16 = _mm256_set1_epi32(0xFFFF);
    const __m256i ones = _mm256_set1_epi32(-1);
    const __m256i scale = _mm256_set1_epi32(RANS_PROB_SCALE);
    const __m256i pk = _mm256_loadu_si256((const __m256i *)pack_ctl);
    uint32_t pa[8], pb[8];

    while (i >= RANS_STREAMS) {
        i -= RANS_STREAMS;
        /* Eight ordinary loads, not a gather: vpgatherdd measured 20% slower
         * here than filling the vectors by hand. */
        for (int j = 0; j < 8; j++) {
            unsigned s = src[i + j];
            pa[j] = packed[s];
            pb[j] = rcp[s];
        }
        __m256i pkd = _mm256_loadu_si256((const __m256i *)pa);
        __m256i rc = _mm256_loadu_si256((const __m256i *)pb);
        __m256i freq = _mm256_and_si256(pkd, m12);
        __m256i bias = _mm256_and_si256(_mm256_srli_epi32(pkd, 12), m16);
        __m256i sh = _mm256_srli_epi32(pkd, 28);
        __m256i cmpl = _mm256_sub_epi32(scale, freq);

        /* x >= freq << 20 is exactly (x >> 20) >= freq, and both sides then
         * fit in twelve bits, so the signed compare is enough. */
        __m256i need = _mm256_xor_si256(
            _mm256_cmpgt_epi32(freq, _mm256_srli_epi32(x, RANS_XSHIFT)), ones);
        unsigned m = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(need));

        __m256i w = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(x, pk), 0xD8);
        _mm_storeu_si128((__m128i *)(ptr - 16),
                         _mm_shuffle_epi8(_mm256_castsi256_si128(w),
                             _mm_loadu_si128((const __m128i *)compact[m])));
        ptr -= 2 * __builtin_popcount(m);

        x = _mm256_blendv_epi8(x, _mm256_srli_epi32(x, 16), need);
        __m256i q = _mm256_srlv_epi32(rans_mulhi_epu32(x, rc), sh);
        x = _mm256_add_epi32(x,
                             _mm256_add_epi32(bias, _mm256_mullo_epi32(q, cmpl)));
    }

    _mm256_storeu_si256((__m256i *)state, x);
    for (int k = RANS_STREAMS - 1; k >= 0; k--) rans_enc_flush(&state[k], &ptr);
    return ptr;
}

#endif /* LMZ_X86 */

#if LMZ_NEON_ENC

static inline uint32x4_t rans_mulhi_u32(uint32x4_t a, uint32x4_t b)
{
    /* Two widening multiplies, and then the odd 32-bit elements of a pair of
     * 64-bit products are exactly their high halves. */
    uint64x2_t lo = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    uint64x2_t hi = vmull_u32(vget_high_u32(a), vget_high_u32(b));
    return vuzp2q_u32(vreinterpretq_u32_u64(lo), vreinterpretq_u32_u64(hi));
}

/* A symbol's whole table entry, both words, in one d-register load. */
static inline uint32x4_t rans_pair(const uint64_t *tab, unsigned a, unsigned b)
{
    return vreinterpretq_u32_u64(vcombine_u64(vld1_u64(tab + a),
                                              vld1_u64(tab + b)));
}

/*
 * The same eight states, stepped eight at a time: the AVX2 body above, on
 * arm64, writing byte for byte what the scalar loop writes. That is what makes
 * either of them a speed change rather than a format change, and arm64 is
 * where models are actually loaded.
 *
 * Eight 32-bit states are two 128-bit registers rather than one, so the lane
 * step is written twice over independent halves. That costs issue slots, not
 * latency -- the two halves are two chains, and the machine has room for both.
 *
 * Three things have no NEON instruction, and only one of them is the problem
 * x86 also had:
 *
 *   the divide, which becomes the same fixed-point reciprocal, exact over the
 *   same range, declined by the same check in the caller;
 *
 *   the emission mask, because NEON has no movemask. The lane masks are
 *   unzipped down to one 16-bit element each and summed against a weight
 *   vector, which is a single across-vector add -- and the weights carry 256
 *   as well as 1 << j, so one sum holds the shuffle's index in its low byte
 *   and the count of lanes that emitted above it. The cursor waits on the
 *   second and the store waits on the first, and neither waits on a second
 *   reduction;
 *
 *   the variable shift, which is a left shift by a negative count.
 *
 * Two things are cheaper here than on x86. Packing the eight low halves into
 * one register is a single unzip, where AVX2 needs a shuffle and a cross-lane
 * permute; and one table of 64-bit entries means a group is eight loads that
 * land in a register directly, where AVX2 fills its two vectors by writing
 * sixteen words to the stack and reading them back.
 */
static uint8_t *rans_enc_body_neon(const uint8_t *src, size_t n, uint8_t *end,
                                   const RansEncSym *syms, const uint16_t *freqs)
{
    uint64_t tab[256];
    {
        /* One table of 64-bit entries rather than two of 32-bit: a symbol's
         * whole entry then arrives in one load, so a group of eight costs
         * eight of them instead of sixteen. */
        uint32_t packed[256], rcp[256];
        rans_simd_tables(freqs, packed, rcp);
        for (int s = 0; s < 256; s++)
            tab[s] = (uint64_t)packed[s] | ((uint64_t)rcp[s] << 32);
    }
    uint8_t compact[256][16];
    rans_compact_table(compact);

    /* 1 << j for the emission mask, 256 for the count of lanes that emitted.
     * The mask cannot reach 256 and eight counts cannot reach 65536, so one
     * 16-bit sum carries both without them meeting. */
    static const uint16_t weights[8] = {257, 258, 260, 264, 272, 288, 320, 384};

    uint8_t *ptr = end;
    uint32_t state[RANS_STREAMS];
    for (int k = 0; k < RANS_STREAMS; k++) state[k] = RANS_L;

    /* The ragged tail, exactly where the scalar loop meets it. */
    size_t i = n;
    while (i > 0 && (i & (RANS_STREAMS - 1))) {
        i--;
        rans_enc_put(&state[i & (RANS_STREAMS - 1)], &ptr, &syms[src[i]]);
    }

    uint32x4_t xa = vld1q_u32(state), xb = vld1q_u32(state + 4);
    const uint32x4_t m12 = vdupq_n_u32(0xFFF);
    const uint32x4_t m16 = vdupq_n_u32(0xFFFF);
    const uint32x4_t scale = vdupq_n_u32(RANS_PROB_SCALE);
    const uint16x8_t wt = vld1q_u16(weights);

    while (i >= RANS_STREAMS) {
        i -= RANS_STREAMS;
        const uint8_t *sy = src + i;
        uint32x4_t t0 = rans_pair(tab, sy[0], sy[1]);
        uint32x4_t t1 = rans_pair(tab, sy[2], sy[3]);
        uint32x4_t t2 = rans_pair(tab, sy[4], sy[5]);
        uint32x4_t t3 = rans_pair(tab, sy[6], sy[7]);
        uint32x4_t pka = vuzp1q_u32(t0, t1), rca = vuzp2q_u32(t0, t1);
        uint32x4_t pkb = vuzp1q_u32(t2, t3), rcb = vuzp2q_u32(t2, t3);

        uint32x4_t fa = vandq_u32(pka, m12), fb = vandq_u32(pkb, m12);
        uint32x4_t ba = vandq_u32(vshrq_n_u32(pka, 12), m16);
        uint32x4_t bb = vandq_u32(vshrq_n_u32(pkb, 12), m16);
        int32x4_t sa = vnegq_s32(vreinterpretq_s32_u32(vshrq_n_u32(pka, 28)));
        int32x4_t sb = vnegq_s32(vreinterpretq_s32_u32(vshrq_n_u32(pkb, 28)));
        uint32x4_t ca = vsubq_u32(scale, fa), cb = vsubq_u32(scale, fb);

        /* x >= freq << 20 is exactly (x >> 20) >= freq. */
        uint32x4_t na = vcgeq_u32(vshrq_n_u32(xa, RANS_XSHIFT), fa);
        uint32x4_t nb = vcgeq_u32(vshrq_n_u32(xb, RANS_XSHIFT), fb);

        /* Every lane's mask narrowed to one 16-bit element, then weighed. */
        uint16x8_t nm = vuzp1q_u16(vreinterpretq_u16_u32(na),
                                   vreinterpretq_u16_u32(nb));
        unsigned sum = vaddvq_u16(vandq_u16(nm, wt));

        /* The same unzip on the states themselves is the pack: the low half
         * of all eight, in lane order, in one register. */
        uint8x16_t w = vreinterpretq_u8_u16(
            vuzp1q_u16(vreinterpretq_u16_u32(xa), vreinterpretq_u16_u32(xb)));
        vst1q_u8(ptr - 16, vqtbl1q_u8(w, vld1q_u8(compact[sum & 0xFF])));
        ptr -= 2 * (sum >> 8);

        xa = vbslq_u32(na, vshrq_n_u32(xa, 16), xa);
        xb = vbslq_u32(nb, vshrq_n_u32(xb, 16), xb);
        uint32x4_t qa = vshlq_u32(rans_mulhi_u32(xa, rca), sa);
        uint32x4_t qb = vshlq_u32(rans_mulhi_u32(xb, rcb), sb);
        /* x + bias + q * cmpl, arranged so only the multiply-accumulate is
         * on the chain: x + bias is ready long before q is, and folding the
         * bias into the accumulator leaves one instruction between the
         * quotient and the next state instead of three. */
        xa = vmlaq_u32(vaddq_u32(xa, ba), qa, ca);
        xb = vmlaq_u32(vaddq_u32(xb, bb), qb, cb);
    }

    vst1q_u32(state, xa);
    vst1q_u32(state + 4, xb);
    for (int k = RANS_STREAMS - 1; k >= 0; k--) rans_enc_flush(&state[k], &ptr);
    return ptr;
}

#endif /* LMZ_NEON_ENC */

/*
 * Encode `n` bytes. Returns the stream length, or -1 if it does not fit.
 * The layout is [header][4 initial states][coded bytes].
 *
 * `counts` is this buffer's byte histogram when the caller already has one --
 * the encoder is chosen from a histogram, so by the time it runs the counts
 * have usually just been taken, and computing them again is a second pass over
 * the stream for nothing. Pass NULL to have it counted here.
 *
 * The contract is that `counts` describes exactly these bytes; a histogram of
 * anything else can give an occurring symbol frequency zero, which the coder
 * cannot represent. Counts that do not sum to `n` are the form that mistake
 * actually takes -- a stale slice, the wrong segment -- so they are rejected
 * and the histogram is taken here instead.
 */
static long rans_encode_impl(const uint8_t *src, size_t n, uint8_t *dst,
                             size_t cap, const uint64_t *counts, int allow_simd)
{
    if (n == 0) return -1;
    if (cap < lmz_rans_bound(n)) return -1;

    uint64_t own[256];
    if (counts) {
        uint64_t total = 0;
        for (int s = 0; s < 256; s++) total += counts[s];
        if (total != n) counts = NULL;
    }
    if (!counts) {
        lmz_hist(src, n, own);
        counts = own;
    }
    uint16_t freqs[256];
    if (rans_normalize(counts, freqs) != 0) return -1;

    RansEncSym syms[256];
    uint32_t start = 0, maxf = 0;
    for (int s = 0; s < 256; s++) {
        if (freqs[s]) {
            rans_enc_sym_init(&syms[s], start, freqs[s]);
            if (freqs[s] > maxf) maxf = freqs[s];
            start += freqs[s];
        }
    }

    /* rANS is last-in-first-out, so encoding runs backwards from the end of
     * the buffer and the result is moved down once its size is known. */
    uint8_t *end = dst + cap;
    uint8_t *ptr;
#if LMZ_X86
    if (allow_simd && lmz_have_avx2() && maxf <= RANS_SIMD_MAX_FREQ
            && n >= RANS_SIMD_MIN)
        ptr = rans_enc_body_avx2(src, n, end, syms, freqs);
    else
#elif LMZ_NEON_ENC
    /* No runtime check: NEON is not optional on arm64, it is the baseline. */
    if (allow_simd && maxf <= RANS_SIMD_MAX_FREQ && n >= RANS_SIMD_MIN)
        ptr = rans_enc_body_neon(src, n, end, syms, freqs);
    else
#endif
    {
        (void)allow_simd;
        (void)maxf;
        ptr = rans_enc_body(src, n, end, syms);
    }

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

LMZ_API long lmz_rans_encode_h(const uint8_t *src, size_t n, uint8_t *dst,
                               size_t cap, const uint64_t *counts)
{
    return rans_encode_impl(src, n, dst, cap, counts, 1);
}

/* The portable path on its own, so the two can be held side by side. The
 * vector path must write the same bytes as this one -- that is the whole
 * basis for it being a speed change rather than a format change, and it is
 * what the tests check. */
LMZ_API long lmz_rans_encode_portable(const uint8_t *src, size_t n, uint8_t *dst,
                                      size_t cap, const uint64_t *counts)
{
    return rans_encode_impl(src, n, dst, cap, counts, 0);
}

/* The same, counting the histogram here. Kept as its own symbol because it is
 * the whole interface anything outside this package uses. */
LMZ_API long lmz_rans_encode(const uint8_t *src, size_t n, uint8_t *dst, size_t cap)
{
    return rans_encode_impl(src, n, dst, cap, NULL, 1);
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

        /*
         * The refills go two ways, and which is faster is a property of the
         * machine rather than of the code. Both are branchless: the refill
         * word is always loaded, because whether a symbol needs one is close
         * to a coin flip on a near-uniform plane and mispredicts about half
         * the time as a branch. They differ in what they do about the cursor.
         *
         * Written one lane at a time, a lane cannot compute its own address
         * until the lane before it has decided whether it consumed a word, so
         * the eight loads run one behind another. But whether a lane refills
         * depends on how many bits its own symbol consumed and on nothing
         * else, so all eight decisions can be made first and the cursor
         * becomes a prefix sum of small integers, leaving the loads
         * independent. That trades a chain of loads for two dozen more
         * arithmetic operations.
         *
         * Measured per plane, decoding: +11% on Zen 5, +4% on a Neoverse
         * runner, -12% on Apple silicon, where the serial version is the
         * fastest of the three machines outright and both planes decode at
         * the same rate whatever their refill rate -- that core hides the
         * chain completely, so the extra arithmetic is pure loss. arm64
         * cannot tell those two parts apart at build time (a VM reports no
         * implementer at all), so it takes the shape that does not cost 12%
         * on the hardware people run models on.
         *
         * Either way the loads read up to two bytes past their own offset,
         * the furthest fifteen bytes beyond the cursor, which the loop guard
         * has established is readable.
         */
        uint32_t x[RANS_STREAMS], need[RANS_STREAMS];
        for (int k = 0; k < RANS_STREAMS; k++)
            x[k] = ((e[k] >> 20) + 1) * (state[k] >> RANS_PROB_BITS)
                 + (state[k] & (RANS_PROB_SCALE - 1)) - ((e[k] >> 8) & 0xfff);
        for (int k = 0; k < RANS_STREAMS; k++) need[k] = (x[k] < RANS_L);
#if LMZ_X86
        uint32_t off[RANS_STREAMS];
        off[0] = 0;
        for (int k = 1; k < RANS_STREAMS; k++)
            off[k] = off[k - 1] + (need[k - 1] << 1);
        for (int k = 0; k < RANS_STREAMS; k++) {
            const uint8_t *p = ptr + off[k];
            uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
            state[k] = need[k] ? ((x[k] << 16) | word) : x[k];
        }
        ptr += off[RANS_STREAMS - 1] + (need[RANS_STREAMS - 1] << 1);
#else
        for (int k = 0; k < RANS_STREAMS; k++) {
            const uint8_t *p = ptr;
            uint32_t word = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
            state[k] = need[k] ? ((x[k] << 16) | word) : x[k];
            ptr = p + (need[k] << 1);
        }
#endif
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

/* ------------------------------------------------- whole-chunk plane decode */

/*
 * Decode every plane of a split chunk and merge them, in one call.
 *
 * The Python path crosses into the kernel once per plane and once more to
 * merge, so a 64 KiB block hands the GIL back and forth three times inside
 * 40 us of work. Alone that is only ~9% of the time, but under threads it is
 * ruinous: measured 0.81 GB/s on one thread, 1.60 on two and 0.70 on four --
 * more threads made it *slower*, because the interpreter spent its time
 * passing the GIL around rather than decoding. Holding one call for the whole
 * block leaves the GIL released from start to finish.
 *
 * `methods` packs two bits per plane, matching the chunk record's flags: 0 is
 * stored, 3 is rANS. A plane coded any other way returns -2 so the caller can
 * fall back; a malformed payload returns -1 and the caller falls back too, so
 * the Python path stays the one place that decides what a bad chunk means.
 *
 * `scratch` holds the decoded planes followed by whatever lmz_merge_planes
 * needs, so nothing is allocated here.
 */
LMZ_API int lmz_decode_planes(const uint8_t *payload, size_t plen,
                              size_t nplanes, size_t nelem, uint32_t methods,
                              int bf16, uint8_t *dst, uint8_t *scratch,
                              size_t scratch_len)
{
    if (nplanes == 0 || nplanes > 16 || nelem == 0) return -1;
    if (bf16 && nplanes != 2) return -1;

    const size_t hdr = 4 * nplanes;
    if (plen < hdr) return -1;
    if (scratch_len < nplanes * nelem) return -1;

    uint32_t lens[16];
    size_t total = hdr;
    for (size_t k = 0; k < nplanes; k++) {
        const uint8_t *h = payload + 4 * k;
        lens[k] = (uint32_t)h[0] | ((uint32_t)h[1] << 8)
                | ((uint32_t)h[2] << 16) | ((uint32_t)h[3] << 24);
        total += lens[k];
        if (total > plen) return -1;   /* truncated */
    }

    const uint8_t *p = payload + hdr;
    const uint8_t *planes[16];
    for (size_t k = 0; k < nplanes; k++) {
        const uint32_t m = (methods >> (2 * k)) & 3u;
        if (m == 0) {                              /* stored */
            if (lens[k] != nelem) return -1;
            planes[k] = p;
        } else if (m == 3) {                       /* rANS */
            uint8_t *d = scratch + k * nelem;
            if (lmz_rans_decode(p, lens[k], d, nelem) != 0) return -1;
            planes[k] = d;
        } else {
            return -2;                             /* zstd/deflate: not here */
        }
        p += lens[k];
    }

    if (bf16) return lmz_merge_bf16(planes[0], planes[1], dst, nelem);

    uint8_t *mscratch = scratch + nplanes * nelem;
    const size_t need = lmz_scratch_size(nelem, nplanes);
    if (scratch_len < nplanes * nelem + need) return -1;
    return lmz_merge_planes(planes, dst, nelem, nplanes, mscratch);
}
