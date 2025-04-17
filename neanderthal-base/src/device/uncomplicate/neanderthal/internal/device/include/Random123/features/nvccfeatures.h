#ifndef __r123_nvcc_features_dot_h__
#define __r123_nvcc_features_dot_h__

#include <stdint.h>//Dragan

#if !defined(CUDART_VERSION)
#error "why are we in nvccfeatures.h if CUDART_VERSION is not defined"
#endif

#if CUDART_VERSION < 4010
#error "CUDA versions earlier than 4.1 produce incorrect results for some templated functions in namespaces.  Random123 isunsupported.  See comments in nvccfeatures.h"
#endif

#ifndef R123_STATIC_INLINE //Dragan
#define R123_STATIC_INLINE inline
#endif

#ifndef R123_FORCE_INLINE //Dragan
#define R123_FORCE_INLINE(decl) decl// __attribute__((always_inline))
#endif

#define UINT64_C(x) ((unsigned long)(x##UL))

#ifdef  __CUDA_ARCH__
#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE __device__
#endif

#ifndef R123_USE_MULHILO64_CUDA_INTRIN
#define R123_USE_MULHILO64_CUDA_INTRIN 1
#endif

#ifndef R123_THROW
#define R123_THROW(x)    R123_ASSERT(0)
#endif

#ifndef R123_ASSERT
#define R123_ASSERT(x) if((x)) ; else asm("trap;")
#endif

#else
#ifndef R123_USE_MULHILO64_ASM
#define R123_USE_MULHILO64_ASM 1
#endif

#endif

#ifndef R123_BUILTIN_EXPECT
#define R123_BUILTIN_EXPECT(expr,likely) expr
#endif

#ifndef R123_USE_AES_NI
#define R123_USE_AES_NI 0
#endif

#ifndef R123_USE_SSE4_2
#define R123_USE_SSE4_2 0
#endif

#ifndef R123_USE_SSE4_1
#define R123_USE_SSE4_1 0
#endif

#ifndef R123_USE_SSE
#define R123_USE_SSE 0
#endif

#ifndef R123_USE_GNU_UINT128
#define R123_USE_GNU_UINT128 0
#endif

#ifndef R123_ULONG_LONG
#define R123_ULONG_LONG unsigned long long
#endif

#if defined(__GNUC__)
#include "gccfeatures.h"
#elif defined(_MSC_FULL_VER)
#include "msvcfeatures.h"
#endif

#endif
