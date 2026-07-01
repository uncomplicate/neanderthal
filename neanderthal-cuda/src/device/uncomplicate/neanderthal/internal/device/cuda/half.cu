#include <cuda_fp16.h>

extern "C" {

    __global__ void vector_scal (const int n, const half alpha,
                                 half* x, const int offset_x, const int stride_x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset_x + gid * stride_x] = __hmul(alpha, x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_axpby (const int n,
                                  const half alpha, const half* x, const int offset_x, const int stride_x,
                                  const half beta, half* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy]  = __hfma(alpha, x[ix], __hmul(beta, y[iy]));
        }
    }

    __global__ void vector_sum (const int n,
                                const half* x, const int offset_x, const int stride_x,
                                float* acc) {

        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        float sum = block_reduction_sum((gid < n) ? __half2float(x[offset_x + gid * stride_x]) : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }

    }

    __global__ void vector_asum (const int n,
                                 const half* x, const int offset_x, const int stride_x,
                                 float* acc) {

        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        float sum = block_reduction_sum((gid < n) ? fabsf(__half2float(x[offset_x + gid * stride_x])) : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }

    }

    __global__ void ge_scal (const int sd, const int fd,
                             const half alpha, half* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            a[offset_a + gid_0 + gid_1 * ld_a] *= __hmul(alpha, a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_sum (const int sd, const int fd,
                            float* acc,
                            const half* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const float sum = block_reduction_sum_2((valid) ? __half2float(a[i]) : 0.0);
        if (threadIdx.y == 0) {
            const float sum2 = block_reduction_sum((valid) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

    __global__ void ge_asum (const int sd, const int fd,
                             float* acc,
                             const half* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const float sum = block_reduction_sum_2( (valid) ? abs(__half2float(a[i])) : 0.0);
        if (threadIdx.y == 0) {
            const float sum2 = block_reduction_sum((valid) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

}
