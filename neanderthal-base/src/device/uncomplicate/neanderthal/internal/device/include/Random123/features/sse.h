#ifndef _Random123_sse_dot_h__
#define _Random123_sse_dot_h__

#if R123_USE_SSE

#if R123_USE_X86INTRIN_H
#include <x86intrin.h>
#endif
#if R123_USE_IA32INTRIN_H
#include <ia32intrin.h>
#endif
#if R123_USE_XMMINTRIN_H
#include <xmmintrin.h>
#endif
#if R123_USE_EMMINTRIN_H
#include <emmintrin.h>
#endif
#if R123_USE_SMMINTRIN_H
#include <smmintrin.h>
#endif
#if R123_USE_WMMINTRIN_H
#include <wmmintrin.h>
#endif
#if R123_USE_INTRIN_H
#include <intrin.h>
#endif

#if R123_USE_ASM_GNU

R123_STATIC_INLINE int haveAESNI(){
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__ ("cpuid": "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx) :
                      "a" (1));
    return (ecx>>25) & 1;
}
#elif R123_USE_CPUID_MSVC
R123_STATIC_INLINE int haveAESNI(){
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[2]>>25)&1;
}
#else
#warning "No R123_USE_CPUID_XXX method chosen.  haveAESNI will always return false"
R123_STATIC_INLINE int haveAESNI(){
    return 0;
}
#endif


#if (defined(__ICC) && __ICC<1210) || (defined(_MSC_VER) && !defined(_WIN64))

R123_STATIC_INLINE __m128i _mm_set_epi64x(uint64_t v1, uint64_t v0){
    union{
        uint64_t u64;
        uint32_t u32[2];
    } u1, u0;
    u1.u64 = v1;
    u0.u64 = v0;
    return _mm_set_epi32(u1.u32[1], u1.u32[0], u0.u32[1], u0.u32[0]);
}
#endif

#if !defined(__x86_64__) || defined(_MSC_VER) || defined(__OPEN64__)
R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    union{
        uint64_t u64[2];
        __m128i m;
    }u;
    _mm_store_si128(&u.m, si);
    return u.u64[0];
}
#elif defined(__llvm__) || defined(__ICC)
R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    return (uint64_t)_mm_cvtsi128_si64(si);
}
#else

R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    return (uint64_t)_mm_cvtsi128_si64x(si);
}
#endif
#if defined(__GNUC__) && __GNUC__ < 4

R123_STATIC_INLINE __m128 _mm_castsi128_ps(__m128i si){
    return (__m128)si;
}
#endif

#else
R123_STATIC_INLINE int haveAESNI(){
    return 0;
}
#endif

#endif
