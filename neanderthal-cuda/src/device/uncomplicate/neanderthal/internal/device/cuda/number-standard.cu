extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif

#ifndef ACCUMULATOR
#define ACCUMULATOR float
#endif

    __global__ void vector_scal (const int n, const NUMBER alpha,
                                 NUMBER* x, const int offset_x, const int stride_x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset_x + gid * stride_x] *= alpha;
        }
    }

    __global__ void vector_sum (const int n,
                                const NUMBER* x, const int offset_x, const int stride_x,
                                ACCUMULATOR* acc) {

        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        ACCUMULATOR sum = block_reduction_sum( (gid < n) ? x[offset_x + gid * stride_x] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }

    }

    __global__ void vector_axpby (const int n,
                                  const NUMBER alpha, const NUMBER* x, const int offset_x, const int stride_x,
                                  const NUMBER beta, NUMBER* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = alpha * x[ix] + beta * y [iy];
        }
    }

    __global__ void ge_scal (const int sd, const int fd,
                             const NUMBER alpha, NUMBER* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            a[offset_a + gid_0 + gid_1 * ld_a] *= alpha;
        }
    }

    __global__ void ge_sum (const int sd, const int fd,
                            ACCUMULATOR* acc,
                            const NUMBER* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const ACCUMULATOR sum = block_reduction_sum_2((valid) ? a[i] : 0.0);
        if (threadIdx.y == 0) {
            const ACCUMULATOR sum2 = block_reduction_sum((valid) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

}
