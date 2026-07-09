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

__global__ void vector_uniform_float (const uint32_t n, const uint64_t seed,
                                          const float lower, const float upper,
                                          float* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t i = gid * 4;

        const philox4x32_ctr_t rand = rand_arr_32(seed);
        const float low = lower;
        const float upplow = upper - low;

        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = u01_float(rand.v[j]) * upplow + low;
        }
    }

    __global__ void vector_uniform_double (const uint32_t n, const uint64_t seed,
                                           const double lower, const double upper,
                                           double* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t i = gid * 4;

        const philox4x64_ctr_t rand = rand_arr_64(seed);
        const double low = lower;
        const double upplow = upper - low;

        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = u01_double(rand.v[j]) * upplow + low;
        }
    }

    __global__ void vector_normal_float (const uint32_t n, const uint64_t seed,
                                         const float mu, const float sigma,
                                         float* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t i = gid * 4;
        const philox4x32_ctr_t rand = rand_arr_32(seed);
        float g[4];
        box_muller_float(rand.v, g);
        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = g[j] * sigma + mu;
        }
    }

    __global__ void vector_normal_double (const uint32_t n, const uint64_t seed,
                                          const double mu, const double sigma,
                                          double* x, const uint32_t offset_x, const uint32_t stride_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t i = gid * 4;
        const philox4x64_ctr_t rand = rand_arr_64(seed);
        double g[4];
        box_muller_double(rand.v, g);
        const int limit = (i + 3) < n ? 4 : n - i;
        for (int j = 0; j < limit; j++) {
            x[offset_x + ((i + j) * stride_x)] = g[j] * sigma + mu;
        }
    }

    __global__ void ge_uniform_float (const uint32_t sd, const uint32_t fd, const uint64_t seed,
                                      const float lower, const float upper,
                                      float* a, const uint32_t offset_a, const uint32_t ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = gid_0 * 4;
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

    __global__ void ge_uniform_double (const uint32_t sd, const uint32_t fd, const uint64_t seed,
                                       const double lower, const double upper,
                                       double* a, const uint32_t offset_a, const uint32_t ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = gid_0 * 4;
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

    __global__ void ge_normal_float (const uint32_t sd, const uint32_t fd, const uint64_t seed,
                                     const float mu, const float sigma,
                                     float* a, const uint32_t offset_a, const uint32_t ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = gid_0 * 4;
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

    __global__ void ge_normal_double (const uint32_t sd, const uint32_t fd, const uint64_t seed,
                                      const double mu, const double sigma,
                                      double* a, const uint32_t offset_a, const uint32_t ld_a) {

        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = gid_0 * 4;
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
