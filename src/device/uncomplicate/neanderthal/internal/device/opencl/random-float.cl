#include "Random123/philox.h"

#ifndef M_2PI_FLOAT
#define M_2PI_FLOAT 6.2831855f
#endif

#ifndef NATIVE
#define NATIVE(fun) native_ ## fun
#endif

inline float u01_float(const uint32_t i) {
    return (0.5f + (i >> 9)) * 0x1.0p-23f;
}

inline philox4x32_ctr_t rand_arr_32 (const uint64_t seed) {
    const philox4x32_key_t key = {{seed, 0xdecafaaa}};
    const philox4x32_ctr_t cnt = {{get_global_id(0), get_global_id(1),
                                   get_global_id(2), 0xbeeff00d}};
    const philox4x32_ctr_t rand = philox4x32(cnt, key);
    return rand;
}

inline void box_muller_float(const uint32_t* i, float* g) {
    g[0] = NATIVE(sin)(M_2PI_FLOAT * u01_float(i[0]))
        * sqrt(-2.0f * NATIVE(log)(u01_float(i[1])));
    g[1] = NATIVE(cos)(M_2PI_FLOAT * u01_float(i[0]))
        * sqrt(-2.0f * NATIVE(log)(u01_float(i[1])));
    g[2] = NATIVE(sin)(M_2PI_FLOAT * u01_float(i[2]))
        * sqrt(-2.0f * NATIVE(log)(u01_float(i[3])));
    g[3] = NATIVE(cos)(M_2PI_FLOAT * u01_float(i[2]))
        * sqrt(-2.0f * NATIVE(log)(u01_float(i[3])));
}

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_uniform_float (const uint n, const ulong seed,
                                    const float lower, const float upper,
                                    __global float* x, const uint offset_x, const uint stride_x) {

    const uint i = get_global_id(0) * 4;

    const philox4x32_ctr_t rand = rand_arr_32(seed);
    const float low = lower;
    const float upplow = upper - low;

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        x[offset_x + ((i + j) * stride_x)] = u01_float(rand.v[j]) * upplow + low;
    }
}

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_normal_float (const uint n, const ulong seed,
                                   const float mu, const float sigma,
                                   __global float* x, const uint offset_x, const uint stride_x) {

    const uint i = get_global_id(0) * 4;

    const philox4x32_ctr_t rand = rand_arr_32(seed);
    float g[4];
    box_muller_float(rand.v, g);
    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        x[offset_x + ((i + j) * stride_x)] = g[j] * sigma + mu;
    }
}

__kernel void ge_uniform_float (const uint n, const ulong seed,
                                const float lower, const float upper,
                                __global float* a, const uint offset_a, const uint ld_a) {

    const uint i = get_global_id(0) * 4;

    const philox4x32_ctr_t rand = rand_arr_32(seed);
    const float low = lower;
    const float upplow = upper - low;

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        a[offset_a + i + j + get_global_id(1) * ld_a] = u01_float(rand.v[j]) * upplow + low;
    }
}

__kernel void ge_normal_float (const uint n, const ulong seed,
                               const float mu, const float sigma,
                               __global float* a, const uint offset_a, const uint ld_a) {

    const uint i = get_global_id(0) * 4;

    const philox4x32_ctr_t rand = rand_arr_32(seed);
    float g[4];
    box_muller_float(rand.v, g);

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        a[offset_a + i + j + get_global_id(1) * ld_a] = g[j] * sigma + mu;
    }
}