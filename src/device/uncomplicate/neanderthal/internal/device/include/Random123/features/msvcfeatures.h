#ifndef __msvcfeatures_dot_hpp
#define __msvcfeatures_dot_hpp

#if !defined(_M_IX86) && !defined(_M_X64)
#  error "This code has only been tested on x86 platforms."
{
#endif

#ifndef R123_STATIC_INLINE
#define R123_STATIC_INLINE static __inline
#endif

#ifndef R123_FORCE_INLINE
#define R123_FORCE_INLINE(decl) _forceinline decl
#endif

#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE
#endif

#ifndef R123_ASSERT
#include <assert.h>
#define R123_ASSERT(x) assert(x)
#endif

#ifndef R123_BUILTIN_EXPECT
#define R123_BUILTIN_EXPECT(expr,likely) expr
#endif

#ifndef R123_USE_AES_NI
#if defined(_M_X64)
#define R123_USE_AES_NI 1
#else
#define R123_USE_AES_NI 0
#endif
#endif

#ifndef R123_USE_SSE4_2
#if defined(_M_X64)
#define R123_USE_SSE4_2 1
#else
#define R123_USE_SSE4_2 0
#endif
#endif

#ifndef R123_USE_SSE4_1
#if defined(_M_X64)
#define R123_USE_SSE4_1 1
#else
#define R123_USE_SSE4_1 0
#endif
#endif

#ifndef R123_USE_SSE
#define R123_USE_SSE 1
#endif

#ifndef R123_USE_AES_OPENSSL
#define R123_USE_AES_OPENSSL 0
#endif

#ifndef R123_USE_GNU_UINT128
#define R123_USE_GNU_UINT128 0
#endif

#ifndef R123_USE_ASM_GNU
#define R123_USE_ASM_GNU 0
#endif

#ifndef R123_USE_CPUID_MSVC
#define R123_USE_CPUID_MSVC 1
#endif

#ifndef R123_USE_X86INTRIN_H
#define R123_USE_X86INTRIN_H 0
#endif

#ifndef R123_USE_IA32INTRIN_H
#define R123_USE_IA32INTRIN_H 0
#endif

#ifndef R123_USE_XMMINTRIN_H
#define R123_USE_XMMINTRIN_H 0
#endif

#ifndef R123_USE_EMMINTRIN_H
#define R123_USE_EMMINTRIN_H 1
#endif

#ifndef R123_USE_SMMINTRIN_H
#define R123_USE_SMMINTRIN_H 1
#endif

#ifndef R123_USE_WMMINTRIN_H
#define R123_USE_WMMINTRIN_H 1
#endif

#ifndef R123_USE_INTRIN_H
#define R123_USE_INTRIN_H 1
#endif

#ifndef R123_USE_MULHILO16_ASM
#define R123_USE_MULHILO16_ASM 0
#endif

#ifndef R123_USE_MULHILO32_ASM
#define R123_USE_MULHILO32_ASM 0
#endif

#ifndef R123_USE_MULHILO64_ASM
#define R123_USE_MULHILO64_ASM 0
#endif

#ifndef R123_USE_MULHILO64_MSVC_INTRIN
#if defined(_M_X64)
#define R123_USE_MULHILO64_MSVC_INTRIN 1
#else
#define R123_USE_MULHILO64_MSVC_INTRIN 0
#endif
#endif

#ifndef R123_USE_MULHILO64_CUDA_INTRIN
#define R123_USE_MULHILO64_CUDA_INTRIN 0
#endif

#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
#define R123_USE_MULHILO64_OPENCL_INTRIN 0
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <stdint.h>
#ifndef UINT64_C
#error UINT64_C not defined.  You must define __STDC_CONSTANT_MACROS before you #include <stdint.h>
#endif

#pragma warning(disable:4244)
#pragma warning(disable:4996)

#endif
