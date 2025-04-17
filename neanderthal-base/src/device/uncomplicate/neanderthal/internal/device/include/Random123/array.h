#ifndef _r123array_dot_h__
#define _r123array_dot_h__
#include "features/compilerfeatures.h"
//#include "features/sse.h"

#define _r123array_tpl(_N, W, T)                   \
struct r123array##_N##x##W{                         \
 T v[_N];                                       \
};                                              \

_r123array_tpl(1, 32, uint32_t)
_r123array_tpl(2, 32, uint32_t)
_r123array_tpl(4, 32, uint32_t)
_r123array_tpl(8, 32, uint32_t)

_r123array_tpl(1, 64, uint64_t)
_r123array_tpl(2, 64, uint64_t)
_r123array_tpl(4, 64, uint64_t)

_r123array_tpl(16, 8, uint8_t)

#if R123_USE_SSE
_r123array_tpl(1, m128i, r123m128i)
#endif

#define R123_W(a)   (8*sizeof(((a *)0)->v[0]))
#endif
