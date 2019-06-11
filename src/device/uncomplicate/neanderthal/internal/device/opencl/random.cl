#include "Random123/philox.h"

#ifndef M_2PI_FLOAT
#define M_2PI_FLOAT 6.2831855f
#endif

#ifndef M_2PI_DOUBLE
#define M_2PI_DOUBLE 6.283185307179586
#endif

inline float u01_float(const uint32_t i) {
    return (0.5f + (i >> 9)) * 0x1.0p-23f;
}

inline double u01_double(const uint64_t i) {
    return (0.5 + (i >> 12)) * 0x1.0p-52;
}

inline philox4x32_ctr_t rand_arr_32 (const uint64_t seed) {
    const philox4x32_key_t key = {{seed, 0xdecafaaa}};
    const philox4x32_ctr_t cnt = {{get_global_id(0), get_global_id(1),
                                   get_global_id(2), 0xbeeff00d}};
    const philox4x32_ctr_t rand = philox4x32(cnt, key);
    return rand;
}

inline philox4x64_ctr_t rand_arr_64 (const uint64_t seed) {
    const philox4x64_key_t key = {{seed, 0xdecafaaa}};
    const philox4x64_ctr_t cnt = {{get_global_id(0), get_global_id(1),
                                   get_global_id(2), 0xbeeff00d}};
    const philox4x64_ctr_t rand = philox4x64(cnt, key);
    return rand;
}

inline void box_muller_float(const uint32_t* i, float* g) {
    g[0] = native_sin(M_2PI_FLOAT * u01_float(i[0]))
        * sqrt(-2.0f * native_log(u01_float(i[1])));
    g[1] = native_cos(M_2PI_FLOAT * u01_float(i[0]))
        * sqrt(-2.0f * native_log(u01_float(i[1])));
    g[2] = native_sin(M_2PI_FLOAT * u01_float(i[2]))
        * sqrt(-2.0f * native_log(u01_float(i[3])));
    g[3] = native_cos(M_2PI_FLOAT * u01_float(i[2]))
        * sqrt(-2.0f * native_log(u01_float(i[3])));
}

inline void box_muller_double(const uint64_t* i, double* g) {
    g[0] = native_sin(M_2PI_DOUBLE * u01_double(i[0]))
        * sqrt(-2.0f * native_log(u01_double(i[1])));
    g[1] = native_cos(M_2PI_DOUBLE * u01_double(i[0]))
        * sqrt(-2.0f * native_log(u01_double(i[1])));
    g[2] = native_sin(M_2PI_DOUBLE * u01_double(i[2]))
        * sqrt(-2.0f * native_log(u01_double(i[3])));
    g[3] = native_cos(M_2PI_DOUBLE * u01_double(i[2]))
        * sqrt(-2.0f * native_log(u01_double(i[3])));
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
__kernel void vector_uniform_double (const uint n, const ulong seed,
                                    const double lower, const double upper,
                                    __global double* x, const uint offset_x, const uint stride_x) {

    const uint i = get_global_id(0) * 4;

    const philox4x64_ctr_t rand = rand_arr_64(seed);
    const double low = lower;
    const double upplow = upper - low;

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        x[offset_x + ((i + j) * stride_x)] = u01_double(rand.v[j]) * upplow + low;
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

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_normal_double (const uint n, const ulong seed,
                                    const double mu, const double sigma,
                                    __global double* x, const uint offset_x, const uint stride_x) {

    const uint i = get_global_id(0) * 4;

    const philox4x64_ctr_t rand = rand_arr_64(seed);
    double g[4];
    box_muller_double(rand.v, g);
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

__kernel void ge_uniform_double (const uint n, const ulong seed,
                                 const double lower, const double upper,
                                 __global double* a, const uint offset_a, const uint ld_a) {

    const uint i = get_global_id(0) * 4;

    const philox4x64_ctr_t rand = rand_arr_64(seed);
    const double low = lower;
    const double upplow = upper - low;

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        a[offset_a + i + j + get_global_id(1) * ld_a] = u01_double(rand.v[j]) * upplow + low;
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

__kernel void ge_normal_double (const uint n, const ulong seed, const double mu, const double sigma,
                                __global double* a, const uint offset_a, const uint ld_a) {

    const uint i = get_global_id(0) * 4;

    const philox4x64_ctr_t rand = rand_arr_64(seed);
    double g[4];
    box_muller_double(rand.v, g);

    const uint limit = (i + 3) < n ? 4 : n - i;
    for (uint j = 0; j < limit; j++) {
        a[offset_a + i + j + get_global_id(1) * ld_a] = g[j] * sigma + mu;
    }
}
