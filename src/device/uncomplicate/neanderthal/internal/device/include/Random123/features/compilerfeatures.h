#if defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ > 0
#include "openclfeatures.h"
#elif defined(__CUDACC__)
#include "nvccfeatures.h"
#elif defined(__ICC)
#include "iccfeatures.h"
#elif defined(__xlC__)
#include "xlcfeatures.h"
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#include "sunprofeatures.h"
#elif defined(__OPEN64__)
#include "open64features.h"
#elif defined(__clang__)
#include "clangfeatures.h"
#elif defined(__GNUC__)
#include "gccfeatures.h"
#elif defined(__PGI)
#include "pgccfeatures.h"
#elif defined(_MSC_FULL_VER)
#include "msvcfeatures.h"
#else
#error "Can't identify compiler.  You'll need to add a new xxfeatures.hpp"
{
#endif

#ifndef R123_USE_CXX11
#define R123_USE_CXX11 (__cplusplus >= 201103L)
#endif

#ifndef R123_USE_CXX11_UNRESTRICTED_UNIONS
#define R123_USE_CXX11_UNRESTRICTED_UNIONS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_STATIC_ASSERT
#define R123_USE_CXX11_STATIC_ASSERT R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_CONSTEXPR
#define R123_USE_CXX11_CONSTEXPR R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_EXPLICIT_CONVERSIONS
#define R123_USE_CXX11_EXPLICIT_CONVERSIONS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_RANDOM
#define R123_USE_CXX11_RANDOM R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_TYPE_TRAITS
#define R123_USE_CXX11_TYPE_TRAITS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_LONG_LONG
#define R123_USE_CXX11_LONG_LONG R123_USE_CXX11
#endif

#ifndef R123_USE_MULHILO64_C99
#define R123_USE_MULHILO64_C99 0
#endif

#ifndef R123_USE_MULHILO64_MULHI_INTRIN
#define R123_USE_MULHILO64_MULHI_INTRIN 0
#endif

#ifndef R123_USE_MULHILO32_MULHI_INTRIN
#define R123_USE_MULHILO32_MULHI_INTRIN 0
#endif

#ifndef R123_STATIC_ASSERT
#if R123_USE_CXX11_STATIC_ASSERT
#define R123_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#else

#define R123_STATIC_ASSERT(expr, msg) typedef char static_assertion[(!!(expr))*2-1]
#endif
#endif

#ifndef R123_CONSTEXPR
#if R123_USE_CXX11_CONSTEXPR
#define R123_CONSTEXPR constexpr
#else
#define R123_CONSTEXPR
#endif
#endif

#ifndef R123_USE_PHILOX_64BIT
#define R123_USE_PHILOX_64BIT (R123_USE_MULHILO64_ASM || R123_USE_MULHILO64_MSVC_INTRIN || R123_USE_MULHILO64_CUDA_INTRIN || R123_USE_GNU_UINT128 || R123_USE_MULHILO64_C99 || R123_USE_MULHILO64_OPENCL_INTRIN || R123_USE_MULHILO64_MULHI_INTRIN)
#endif

#ifndef R123_ULONG_LONG
#if defined(__cplusplus) && !R123_USE_CXX11_LONG_LONG

#define R123_ULONG_LONG uint64_t
#else
#define R123_ULONG_LONG unsigned long long
#endif
#endif

#ifndef R123_64BIT
#define R123_64BIT(x) UINT64_C(x)
#endif

#ifndef R123_THROW
#define R123_THROW(x)    throw (x)
#endif

#define R123_NO_MACRO_SUBST
