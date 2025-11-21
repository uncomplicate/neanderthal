extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif

#ifndef ACCUMULATOR
#define ACCUMULATOR float
#endif

    __global__ void vector_equals (const int n,
                                   const NUMBER* x, const int offset_x, const int stride_x,
                                   const NUMBER* y, const int offset_y, const int stride_y,
                                   int* eq_flag) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            if (x[ix] != y[iy]) {
                eq_flag[0]++;
            }
        }
    }

    __global__ void vector_copy (const int n,
                                 const NUMBER* x, const int offset_x, const int stride_x,
                                 NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = x[ix];
        }
    }

    __global__ void vector_swap (const int n,
                                 NUMBER* x, const int offset_x, const int stride_x,
                                 NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            const NUMBER val = y[ix];
            y[iy] = x[ix];
            x[ix] = val;
        }
    }

    __global__ void vector_set (const int n, const NUMBER val,
                                NUMBER* x, const int offset_x, const int stride_x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset_x + gid * stride_x] = val;
        }
    }

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

    __global__ void ge_copy_no_transp (const int sd, const int fd,
                                       NUMBER* a, const int offset_a, const int ld_a,
                                       NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            b[ib] = a[ia];
        }
    }

    __global__ void ge_copy_transp (const int sd, const int fd,
                                    NUMBER* a, const int offset_a, const int ld_a,
                                    NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            b[ib] = a[ia];
        }
    }

    __global__ void ge_swap_no_transp (const int sd, const int fd,
                                       NUMBER* a, const int offset_a, const int ld_a,
                                       NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            const NUMBER c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void ge_swap_transp (const int sd, const int fd,
                                    NUMBER* a, const int offset_a, const int ld_a,
                                    NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            const NUMBER c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void ge_equals_no_transp (const int sd, const int fd,
                                         const NUMBER* a, const int offset_a, const int ld_a,
                                         const NUMBER* b, const int offset_b, const int ld_b,
                                         int* eq_flag) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            if (a[ia] != b[ib]){
                eq_flag[0]++;
            }
        }
    }

    __global__ void ge_equals_transp (const int sd, const int fd,
                                      const NUMBER* a, const int offset_a, const int ld_a,
                                      const NUMBER* b, const int offset_b, const int ld_b,
                                      int* eq_flag) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            if (a[ia] != b[ib]){
                eq_flag[0]++;
            }
        }
    }

    __global__ void ge_set (const int sd, const int fd,
                            const NUMBER val, NUMBER* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            a[offset_a + gid_0 + gid_1 * ld_a] = val;
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
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? a[i] : 0.0);
        if (threadIdx.y == 0) {
            const ACCUMULATOR sum2 = block_reduction_sum((valid) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

}
