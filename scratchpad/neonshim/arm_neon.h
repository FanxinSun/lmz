/*
 * arm_neon.h -- a portable stand-in, so lmzcore.c's arm64 kernels can be
 * compiled and checked byte for byte on a machine that is not arm64.
 *
 *     cc -O2 -DLMZ_FORCE_NEON -Iscratchpad/neonshim scratchpad/encbench.c
 *
 * LMZ_FORCE_NEON makes lmzcore.c take its NEON branch and include <arm_neon.h>;
 * -I puts this file where that include looks. The kernel source is not
 * modified and not copied: what runs is the shipped file, down the shipped
 * path, with each intrinsic replaced by the plainest C that has the same
 * definition.
 *
 * What this proves and what it does not. It proves the arithmetic, the lane
 * order, the emission mask and the shuffle -- everything that decides whether
 * the vector encoder writes the same bytes as the scalar one, which is the
 * property the format depends on. It proves nothing at all about speed, and it
 * is not a NEON emulator: only the intrinsics lmzcore.c actually names are
 * here, with the semantics the ARM intrinsics reference gives them, and each
 * one is written to be read rather than to be fast.
 *
 * Little-endian hosts only, which is every machine this would be run on. The
 * reinterprets are memcpy of the same sixteen bytes, and that is only the
 * identity NEON gives them where the byte order agrees.
 */
#ifndef LMZ_NEON_SHIM_H
#define LMZ_NEON_SHIM_H

#include <stdint.h>
#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the NEON shim assumes a little-endian host"
#endif

/* Structs, not GCC vector types, so that a lane-width mistake is a compile
 * error here exactly as it would be on the real header. */
typedef struct { uint8_t  v[16]; } uint8x16_t;
typedef struct { uint16_t v[8];  } uint16x8_t;
typedef struct { uint32_t v[4];  } uint32x4_t;
typedef struct { int32_t  v[4];  } int32x4_t;
typedef struct { uint64_t v[2];  } uint64x2_t;
typedef struct { uint8_t  v[8];  } uint8x8_t;
typedef struct { uint16_t v[4];  } uint16x4_t;
typedef struct { uint32_t v[2];  } uint32x2_t;
typedef struct { uint64_t v[1];  } uint64x1_t;
typedef struct { uint8x16_t val[2]; } uint8x16x2_t;

#define LMZ_SHIM static inline

/* --------------------------------------------------------------- load/store */

LMZ_SHIM uint8x16_t vld1q_u8(const uint8_t *p)
{ uint8x16_t r; memcpy(r.v, p, 16); return r; }
LMZ_SHIM uint16x8_t vld1q_u16(const uint16_t *p)
{ uint16x8_t r; memcpy(r.v, p, 16); return r; }
LMZ_SHIM uint32x4_t vld1q_u32(const uint32_t *p)
{ uint32x4_t r; memcpy(r.v, p, 16); return r; }
LMZ_SHIM uint64x1_t vld1_u64(const uint64_t *p)
{ uint64x1_t r; memcpy(r.v, p, 8); return r; }

LMZ_SHIM void vst1q_u8(uint8_t *p, uint8x16_t a)   { memcpy(p, a.v, 16); }
LMZ_SHIM void vst1q_u32(uint32_t *p, uint32x4_t a) { memcpy(p, a.v, 16); }

/* De-interleave and interleave, the byte split and merge. */
LMZ_SHIM uint8x16x2_t vld2q_u8(const uint8_t *p)
{
    uint8x16x2_t r;
    for (int i = 0; i < 16; i++) { r.val[0].v[i] = p[2 * i];
                                   r.val[1].v[i] = p[2 * i + 1]; }
    return r;
}
LMZ_SHIM void vst2q_u8(uint8_t *p, uint8x16x2_t a)
{
    for (int i = 0; i < 16; i++) { p[2 * i] = a.val[0].v[i];
                                   p[2 * i + 1] = a.val[1].v[i]; }
}

/* ------------------------------------------------------------- rearrangement */

LMZ_SHIM uint32x2_t vget_low_u32(uint32x4_t a)
{ uint32x2_t r; r.v[0] = a.v[0]; r.v[1] = a.v[1]; return r; }
LMZ_SHIM uint32x2_t vget_high_u32(uint32x4_t a)
{ uint32x2_t r; r.v[0] = a.v[2]; r.v[1] = a.v[3]; return r; }
LMZ_SHIM uint64x2_t vcombine_u64(uint64x1_t a, uint64x1_t b)
{ uint64x2_t r; r.v[0] = a.v[0]; r.v[1] = b.v[0]; return r; }

/* uzp1 takes the even elements of the two vectors concatenated, uzp2 the odd
 * ones -- the halves of a de-interleave. */
LMZ_SHIM uint16x8_t vuzp1q_u16(uint16x8_t a, uint16x8_t b)
{
    uint16x8_t r;
    for (int i = 0; i < 4; i++) { r.v[i] = a.v[2 * i]; r.v[4 + i] = b.v[2 * i]; }
    return r;
}
LMZ_SHIM uint32x4_t vuzp1q_u32(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    for (int i = 0; i < 2; i++) { r.v[i] = a.v[2 * i]; r.v[2 + i] = b.v[2 * i]; }
    return r;
}
LMZ_SHIM uint32x4_t vuzp2q_u32(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    for (int i = 0; i < 2; i++) { r.v[i] = a.v[2 * i + 1];
                                  r.v[2 + i] = b.v[2 * i + 1]; }
    return r;
}

/* An index outside the table selects a zero byte -- the property the encoder's
 * 0x80 fill relies on. */
LMZ_SHIM uint8x16_t vqtbl1q_u8(uint8x16_t t, uint8x16_t idx)
{
    uint8x16_t r;
    for (int i = 0; i < 16; i++) r.v[i] = idx.v[i] < 16 ? t.v[idx.v[i]] : 0;
    return r;
}

/* ---------------------------------------------------------------- arithmetic */

LMZ_SHIM uint32x4_t vdupq_n_u32(uint32_t a)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a; return r; }

LMZ_SHIM uint16x8_t vandq_u16(uint16x8_t a, uint16x8_t b)
{ uint16x8_t r; for (int i = 0; i < 8; i++) r.v[i] = a.v[i] & b.v[i]; return r; }
LMZ_SHIM uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] & b.v[i]; return r; }
LMZ_SHIM uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] + b.v[i]; return r; }
LMZ_SHIM uint32x4_t vsubq_u32(uint32x4_t a, uint32x4_t b)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] - b.v[i]; return r; }
LMZ_SHIM uint32x4_t vmulq_u32(uint32x4_t a, uint32x4_t b)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] * b.v[i]; return r; }
LMZ_SHIM uint32x4_t vmlaq_u32(uint32x4_t a, uint32x4_t b, uint32x4_t c)
{ uint32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] + b.v[i] * c.v[i]; return r; }
LMZ_SHIM int32x4_t vnegq_s32(int32x4_t a)
{ int32x4_t r; for (int i = 0; i < 4; i++) r.v[i] = (int32_t)(0u - (uint32_t)a.v[i]); return r; }

/* The widening multiply: the whole 64-bit product of the low two lanes. */
LMZ_SHIM uint64x2_t vmull_u32(uint32x2_t a, uint32x2_t b)
{
    uint64x2_t r;
    for (int i = 0; i < 2; i++) r.v[i] = (uint64_t)a.v[i] * (uint64_t)b.v[i];
    return r;
}

/* A shift by an immediate is always in range; a shift by a vector is not, and
 * NEON's answer to an out-of-range count is zero rather than the undefined
 * behaviour C has. Negative counts shift right, which is the only reason the
 * encoder can shift each lane by its own amount at all. */
#define vshrq_n_u32(a, n) lmz_shim_shrq_u32((a), (n))
LMZ_SHIM uint32x4_t lmz_shim_shrq_u32(uint32x4_t a, int n)
{
    uint32x4_t r;
    for (int i = 0; i < 4; i++) r.v[i] = n >= 32 ? 0 : a.v[i] >> n;
    return r;
}
LMZ_SHIM uint32x4_t vshlq_u32(uint32x4_t a, int32x4_t b)
{
    uint32x4_t r;
    for (int i = 0; i < 4; i++) {
        int32_t k = b.v[i];
        if (k >= 32 || k <= -32) r.v[i] = 0;
        else if (k >= 0)         r.v[i] = a.v[i] << k;
        else                     r.v[i] = a.v[i] >> -k;
    }
    return r;
}

LMZ_SHIM uint32x4_t vcgeq_u32(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    for (int i = 0; i < 4; i++) r.v[i] = a.v[i] >= b.v[i] ? 0xFFFFFFFFu : 0u;
    return r;
}
LMZ_SHIM uint32x4_t vbslq_u32(uint32x4_t m, uint32x4_t a, uint32x4_t b)
{
    uint32x4_t r;
    for (int i = 0; i < 4; i++) r.v[i] = (a.v[i] & m.v[i]) | (b.v[i] & ~m.v[i]);
    return r;
}

/* Across-vector add. It sums into one element of the vector's own width, so
 * the total wraps at 16 bits -- which the encoder's weights are chosen to
 * stay inside. */
LMZ_SHIM uint16_t vaddvq_u16(uint16x8_t a)
{
    uint16_t s = 0;
    for (int i = 0; i < 8; i++) s = (uint16_t)(s + a.v[i]);
    return s;
}

/* --------------------------------------------------------------- reinterpret */

#define LMZ_SHIM_CAST(sto, sfrom, to, from)                                   \
    LMZ_SHIM to vreinterpretq_##sto##_##sfrom(from a)                         \
    { to r; memcpy(&r, &a, 16); return r; }

LMZ_SHIM_CAST(u8,  u16, uint8x16_t, uint16x8_t)
LMZ_SHIM_CAST(u8,  u32, uint8x16_t, uint32x4_t)
LMZ_SHIM_CAST(u16, u32, uint16x8_t, uint32x4_t)
LMZ_SHIM_CAST(u32, u16, uint32x4_t, uint16x8_t)
LMZ_SHIM_CAST(u32, u64, uint32x4_t, uint64x2_t)
LMZ_SHIM_CAST(u32, s32, uint32x4_t, int32x4_t)
LMZ_SHIM_CAST(s32, u32, int32x4_t,  uint32x4_t)
LMZ_SHIM_CAST(u64, u32, uint64x2_t, uint32x4_t)

#undef LMZ_SHIM_CAST
#undef LMZ_SHIM
#endif /* LMZ_NEON_SHIM_H */
