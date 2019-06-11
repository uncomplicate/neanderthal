extern "C" {
    
#include "Random123/philox.h"
#include <stdint.h>
#include <float.h>
    
#ifndef M_2PI_FLOAT
#define M_2PI_FLOAT 6.2831855f
#endif

#ifndef M_2PI_DOUBLE
#define M_2PI_DOUBLE 6.283185307179586
#endif

    inline float u01_float(const uint32_t i) {
        return (0.5f + (i >> 9)) * FLT_EPSILON;
    }

    inline double u01_double(const uint64_t i) {
        return (0.5 + (i >> 12)) * DBL_EPSILON;
    }

    inline philox4x32_ctr_t rand_arr_32 (const uint64_t seed) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t gid_2 = blockIdx.z * blockDim.z + threadIdx.z;
        philox4x32_key_t key;
        uint32_t* key_v = key.v;
        key_v[0] = seed;
        key_v[1] = 0xdecafaaa;
        philox4x32_ctr_t cnt;
        uint32_t* cnt_v = cnt.v;
        cnt_v[0] = gid_0;
        cnt_v[1] = gid_1;
        cnt_v[2] = gid_2;
        cnt_v[3] = 0xbeeff00d;
        const philox4x32_ctr_t rand = philox4x32(cnt, key);
        return rand;
    }

    inline philox4x64_ctr_t rand_arr_64 (const uint64_t seed) {
        const uint64_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint64_t gid_2 = blockIdx.z * blockDim.z + threadIdx.z;
        philox4x64_key_t key;
        uint64_t* key_v = key.v;
        key_v[0] = seed;
        key_v[1] = 0xdecafaaa;
        philox4x64_ctr_t cnt;
        uint64_t* cnt_v = cnt.v;
        cnt_v[0] = gid_0;
        cnt_v[1] = gid_1;
        cnt_v[2] = gid_2;
        cnt_v[3] = 0xbeeff00d;
        const philox4x64_ctr_t rand = philox4x64(cnt, key);
        return rand;
    }

    inline void box_muller_float(const uint32_t* i, float* g) {
        g[0] = sinf(M_2PI_FLOAT * u01_float(i[0]))
            * sqrtf(-2.0f * logf(u01_float(i[1])));
        g[1] = cosf(M_2PI_FLOAT * u01_float(i[0]))
            * sqrtf(-2.0f * logf(u01_float(i[1])));
        g[2] = sinf(M_2PI_FLOAT * u01_float(i[2]))
            * sqrtf(-2.0f * logf(u01_float(i[3])));
        g[3] = cosf(M_2PI_FLOAT * u01_float(i[2]))
            * sqrtf(-2.0f * logf(u01_float(i[3])));
    }

    inline void box_muller_double(const uint64_t* i, double* g) {
        g[0] = sin(M_2PI_DOUBLE * u01_double(i[0]))
            * sqrt(-2.0f * log(u01_double(i[1])));
        g[1] = cos(M_2PI_DOUBLE * u01_double(i[0]))
            * sqrt(-2.0f * log(u01_double(i[1])));
        g[2] = sin(M_2PI_DOUBLE * u01_double(i[2]))
            * sqrt(-2.0f * log(u01_double(i[3])));
        g[3] = cos(M_2PI_DOUBLE * u01_double(i[2]))
            * sqrt(-2.0f * log(u01_double(i[3])));
    }
    
    __global__ void vector_uniform_float (const int n, const uint64_t seed,
                                          const float lower, const float upper,
                                          float* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = gid * 4;

        const philox4x32_ctr_t rand = rand_arr_32(seed);
        const float low = lower;
        const float upplow = upper - low;

        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = u01_float(rand.v[j]) * upplow + low;
        }
    }

    __global__ void vector_uniform_double (const int n, const uint64_t seed,
                                           const double lower, const double upper,
                                           double* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = gid * 4;

        const philox4x64_ctr_t rand = rand_arr_64(seed);
        const double low = lower;
        const double upplow = upper - low;

        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = u01_double(rand.v[j]) * upplow + low;
        }
    }

    __global__ void vector_normal_float (const int n, const uint64_t seed,
                                         const float mu, const float sigma,
                                         float* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = gid * 4;
        const philox4x32_ctr_t rand = rand_arr_32(seed);
        float g[4];
        box_muller_float(rand.v, g);
        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = g[j] * sigma + mu;
        }
    }

    __global__ void vector_normal_double (const int n, const uint64_t seed,
                                          const double mu, const double sigma,
                                          double* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int i = gid * 4;
        const philox4x64_ctr_t rand = rand_arr_64(seed);
        double g[4];
        box_muller_double(rand.v, g);
        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = g[j] * sigma + mu;
        }
    }

    __global__ void ge_uniform_float (const int sd, const int fd, const uint64_t seed,
                                      const float lower, const float upper,
                                      float* a, const int offset_a, const int ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = gid_0 * 4;
        if (gid_1 < fd) {
            const philox4x32_ctr_t rand = rand_arr_32(seed);
            const float low = lower;
            const float upplow = upper - low;
        
            const int limit = (i + 3) < sd ? 4 : sd - i;
            for (int j = 0; j < limit; j++) {
                a[offset_a + i + j + gid_1 * ld_a] = u01_float(rand.v[j]) * upplow + low;
            }
        }
    }

    __global__ void ge_uniform_double (const int sd, const int fd, const uint64_t seed,
                                       const double lower, const double upper,
                                       double* a, const int offset_a, const int ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = gid_0 * 4;
        if (gid_1 < fd) {
            const philox4x64_ctr_t rand = rand_arr_64(seed);
            const double low = lower;
            const double upplow = upper - low;
            const int limit = (i + 3) < sd ? 4 : sd - i;
            for (int j = 0; j < limit; j++) {
                a[offset_a + i + j + gid_1 * ld_a] = u01_double(rand.v[j]) * upplow + low;
            }
        }
    }

    __global__ void ge_normal_float (const int sd, const int fd, const uint64_t seed,
                                     const float mu, const float sigma,
                                     float* a, const int offset_a, const int ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = gid_0 * 4;
        if (gid_1 < fd) {    
            const philox4x32_ctr_t rand = rand_arr_32(seed);
            float g[4];
            box_muller_float(rand.v, g);

            const int limit = (i + 3) < sd ? 4 : sd - i;
            for (int j = 0; j < limit; j++) {
                a[offset_a + i + j + gid_1 * ld_a] = g[j] * sigma + mu;
            }
        }
    }

    __global__ void ge_normal_double (const int sd, const int fd, const uint64_t seed,
                                      const double mu, const double sigma,
                                      double* a, const int offset_a, const int ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = gid_0 * 4;
        if (gid_1 < fd) {
            const philox4x64_ctr_t rand = rand_arr_64(seed);
            double g[4];
            box_muller_double(rand.v, g);
            const int limit = (i + 3) < sd ? 4 : sd - i;
            for (int j = 0; j < limit; j++) {
                a[offset_a + i + j + gid_1 * ld_a] = g[j] * sigma + mu;
            }
        }
    }
    
}
