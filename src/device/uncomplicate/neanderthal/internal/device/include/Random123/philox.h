#ifndef _philox_dot_h_
#define _philox_dot_h_

#include "features/compilerfeatures.h"
#include "array.h"

#define _mulhilo_dword_tpl(W, Word, Dword)                              \
R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
    Dword product = ((Dword)a)*((Dword)b);                              \
    *hip = product>>W;                                                  \
    return (Word)product;                                               \
}

#ifdef __powerpc__
#define _mulhilo_asm_tpl(W, Word, INSN)                         \
R123_STATIC_INLINE Word mulhilo##W(Word ax, Word b, Word *hip){ \
    Word dx = 0;                                                \
    __asm__("\n\t"                                              \
        INSN " %0,%1,%2\n\t"                                    \
        : "=r"(dx)                                              \
        : "r"(b), "r"(ax)                                       \
        );                                                      \
    *hip = dx;                                                  \
    return ax*b;                                                \
}
#else
#define _mulhilo_asm_tpl(W, Word, INSN)                         \
R123_STATIC_INLINE Word mulhilo##W(Word ax, Word b, Word *hip){      \
    Word dx;                                                    \
    __asm__("\n\t"                                              \
        INSN " %2\n\t"                                          \
        : "=a"(ax), "=d"(dx)                                    \
        : "r"(b), "0"(ax)                                       \
        );                                                      \
    *hip = dx;                                                  \
    return ax;                                                  \
}
#endif

#define _mulhilo_msvc_intrin_tpl(W, Word, INTRIN)               \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){       \
    return INTRIN(a, b, hip);                                   \
}

#define _mulhilo_cuda_intrin_tpl(W, Word, INTRIN)                       \
R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
    *hip = INTRIN(a, b);                                                \
    return a*b;                                                         \
}

#define _mulhilo_c99_tpl(W, Word) \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word *hip){ \
    const unsigned WHALF = W/2;                                    \
    const Word LOMASK = ((((Word)1)<<WHALF)-1);                    \
    Word lo = a*b;                          \
    Word ahi = a>>WHALF;                                           \
    Word alo = a& LOMASK;                                          \
    Word bhi = b>>WHALF;                                           \
    Word blo = b& LOMASK;                                          \
                                                                   \
    Word ahbl = ahi*blo;                                           \
    Word albh = alo*bhi;                                           \
                                                                   \
    Word ahbl_albh = ((ahbl&LOMASK) + (albh&LOMASK));                   \
    Word hi = ahi*bhi + (ahbl>>WHALF) +  (albh>>WHALF);                 \
    hi += ahbl_albh >> WHALF; \
    hi += ((lo >> WHALF) < (ahbl_albh&LOMASK));                         \
    *hip = hi;                                                          \
    return lo;                                                          \
}

#define _mulhilo_fail_tpl(W, Word)                                      \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word *hip){               \
    R123_STATIC_ASSERT(0, "mulhilo" #W " is not implemented on this machine\n"); \
}

#if R123_USE_MULHILO32_ASM
#ifdef __powerpc__
_mulhilo_asm_tpl(32, uint32_t, "mulhwu")
#else
_mulhilo_asm_tpl(32, uint32_t, "mull")
#endif
#else
_mulhilo_dword_tpl(32, uint32_t, uint64_t)
#endif

#if R123_USE_PHILOX_64BIT
#if R123_USE_MULHILO64_ASM
#ifdef __powerpc64__
_mulhilo_asm_tpl(64, uint64_t, "mulhdu")
#else
_mulhilo_asm_tpl(64, uint64_t, "mulq")
#endif
#elif R123_USE_MULHILO64_MSVC_INTRIN
_mulhilo_msvc_intrin_tpl(64, uint64_t, _umul128)
#elif R123_USE_MULHILO64_CUDA_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, __umul64hi)
#elif R123_USE_MULHILO64_OPENCL_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, mul_hi)
#elif R123_USE_MULHILO64_MULHI_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, R123_MULHILO64_MULHI_INTRIN)
#elif R123_USE_GNU_UINT128
_mulhilo_dword_tpl(64, uint64_t, __uint128_t)
#elif R123_USE_MULHILO64_C99
_mulhilo_c99_tpl(64, uint64_t)
#else
_mulhilo_fail_tpl(64, uint64_t)
#endif
#endif

#ifndef PHILOX_M2x64_0
#define PHILOX_M2x64_0 R123_64BIT(0xD2B74407B1CE6E93)
#endif

#ifndef PHILOX_M4x64_0
#define PHILOX_M4x64_0 R123_64BIT(0xD2E7470EE14C6C93)
#endif

#ifndef PHILOX_M4x64_1
#define PHILOX_M4x64_1 R123_64BIT(0xCA5A826395121157)
#endif

#ifndef PHILOX_M2x32_0
#define PHILOX_M2x32_0 ((uint32_t)0xd256d193)
#endif

#ifndef PHILOX_M4x32_0
#define PHILOX_M4x32_0 ((uint32_t)0xD2511F53)
#endif
#ifndef PHILOX_M4x32_1
#define PHILOX_M4x32_1 ((uint32_t)0xCD9E8D57)
#endif

#ifndef PHILOX_W64_0
#define PHILOX_W64_0 R123_64BIT(0x9E3779B97F4A7C15)
#endif
#ifndef PHILOX_W64_1
#define PHILOX_W64_1 R123_64BIT(0xBB67AE8584CAA73B)
#endif

#ifndef PHILOX_W32_0
#define PHILOX_W32_0 ((uint32_t)0x9E3779B9)
#endif
#ifndef PHILOX_W32_1
#define PHILOX_W32_1 ((uint32_t)0xBB67AE85)
#endif

#ifndef PHILOX2x32_DEFAULT_ROUNDS
#define PHILOX2x32_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX2x64_DEFAULT_ROUNDS
#define PHILOX2x64_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX4x32_DEFAULT_ROUNDS
#define PHILOX4x32_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX4x64_DEFAULT_ROUNDS
#define PHILOX4x64_DEFAULT_ROUNDS 10
#endif

#define _philox2xWround_tpl(W, T)                                       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key){ \
    T hi;                                                               \
    T lo = mulhilo##W(PHILOX_M2x##W##_0, ctr.v[0], &hi);                \
    struct r123array2x##W out = {{hi^key.v[0]^ctr.v[1], lo}};               \
    return out;                                                         \
}
#define _philox2xWbumpkey_tpl(W)                                        \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array1x##W _philox2x##W##bumpkey( struct r123array1x##W key) { \
    key.v[0] += PHILOX_W##W##_0;                                        \
    return key;                                                         \
}

#define _philox4xWround_tpl(W, T)                                       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key){ \
    T hi0;                                                              \
    T hi1;                                                              \
    T lo0 = mulhilo##W(PHILOX_M4x##W##_0, ctr.v[0], &hi0);              \
    T lo1 = mulhilo##W(PHILOX_M4x##W##_1, ctr.v[2], &hi1);              \
    struct r123array4x##W out = {{hi1^ctr.v[1]^key.v[0], lo1,               \
                              hi0^ctr.v[3]^key.v[1], lo0}};             \
    return out;                                                         \
}

#define _philox4xWbumpkey_tpl(W)                                        \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox4x##W##bumpkey( struct r123array2x##W key) { \
    key.v[0] += PHILOX_W##W##_0;                                        \
    key.v[1] += PHILOX_W##W##_1;                                        \
    return key;                                                         \
}

#define _philoxNxW_tpl(N, Nhalf, W, T)                         \
enum r123_enum_philox##N##x##W { philox##N##x##W##_rounds = PHILOX##N##x##W##_DEFAULT_ROUNDS }; \
typedef struct r123array##N##x##W philox##N##x##W##_ctr_t;                  \
typedef struct r123array##Nhalf##x##W philox##N##x##W##_key_t;              \
typedef struct r123array##Nhalf##x##W philox##N##x##W##_ukey_t;              \
R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_key_t philox##N##x##W##keyinit(philox##N##x##W##_ukey_t uk) { return uk; } \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key) { \
    R123_ASSERT(R<=16);                                                 \
    if(R>0){                                       ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>1){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>2){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>3){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>4){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>5){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>6){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>7){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>8){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>9){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>10){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>11){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>12){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>13){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>14){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>15){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    return ctr;                                                         \
}

_philox2xWbumpkey_tpl(32)
_philox4xWbumpkey_tpl(32)
_philox2xWround_tpl(32, uint32_t)
_philox4xWround_tpl(32, uint32_t)

_philoxNxW_tpl(2, 1, 32, uint32_t)
_philoxNxW_tpl(4, 2, 32, uint32_t)
#if R123_USE_PHILOX_64BIT

_philox2xWbumpkey_tpl(64)
_philox4xWbumpkey_tpl(64)
_philox2xWround_tpl(64, uint64_t)
_philox4xWround_tpl(64, uint64_t)

_philoxNxW_tpl(2, 1, 64, uint64_t)
_philoxNxW_tpl(4, 2, 64, uint64_t)
#endif

#define philox2x32(c,k) philox2x32_R(philox2x32_rounds, c, k)
#define philox4x32(c,k) philox4x32_R(philox4x32_rounds, c, k)
#if R123_USE_PHILOX_64BIT
#define philox2x64(c,k) philox2x64_R(philox2x64_rounds, c, k)
#define philox4x64(c,k) philox4x64_R(philox4x64_rounds, c, k)
#endif

#endif
