extern "C" {
    
#ifndef REAL
#define REAL float
#endif

    __global__ void vector_equals (const int n,
                                   const REAL* x, const int offset_x, const int stride_x,
                                   const REAL* y, const int offset_y, const int stride_y,
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
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = x[ix]; 
        }
    }

    
    __global__ void vector_sum (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                ACCUMULATOR* acc) {

        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        ACCUMULATOR sum = block_reduction_sum( (gid < n) ? x[offset_x + gid * stride_x] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
        
    }
    
    __global__ void vector_set (const int n, const REAL val,
                                REAL* x, const int offset_x, const int stride_x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset_x + gid * stride_x] = val;
        }
    }

    __global__ void vector_axpby (const int n,
                                  const REAL alpha, const REAL* x, const int offset_x, const int stride_x,
                                  const REAL beta, REAL* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = alpha * x[ix] + beta * y [iy]; 
        }
    }

    __global__ void ge_equals_no_transp (const int sd, const int fd,
                                         const REAL* a, const int offset_a, const int ld_a,
                                         const REAL* b, const int offset_b, const int ld_b,
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
                                      const REAL* a, const int offset_a, const int ld_a,
                                      const REAL* b, const int offset_b, const int ld_b,
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

    __global__ void ge_swap_no_transp (const int sd, const int fd,
                                       REAL* a, const int offset_a, const int ld_a,
                                       REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            const REAL c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void ge_swap_transp (const int sd, const int fd,
                                    REAL* a, const int offset_a, const int ld_a,
                                    REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            const REAL c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void ge_set (const int sd, const int fd,
                            const REAL val, REAL* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            a[offset_a + gid_0 + gid_1 * ld_a] = val;
        }
    }

    __global__ void uplo_equals_no_transp (const int sd, const int unit, const int bottom,
                                           const REAL* a, const int offset_a, const int ld_a,
                                           const REAL* b, const int offset_b, const int ld_b,
                                           int* eq_flag) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            if (a[ia] != b[ib]){
                eq_flag[0]++;
            }
        }
    }

    __global__ void uplo_equals_transp (const int sd, const int unit, const int bottom,
                                        const REAL* a, const int offset_a, const int ld_a,
                                        const REAL* b, const int offset_b, const int ld_b,
                                        int* eq_flag) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            if (a[ia] != b[ib]){
                eq_flag[0]++;
            }
        }
    }

    __global__ void uplo_copy_no_transp (const int sd, const int unit, const int bottom,
                                         const REAL* a, const int offset_a, const int ld_a,
                                         REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            b[ib] = a[ia];
        }
    }

    __global__ void uplo_copy_transp (const int sd, const int unit, const int bottom,
                                      const REAL* a, const int offset_a, const int ld_a,
                                      REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            b[ib] = a[ia];
        }
    }

    __global__ void uplo_swap_no_transp (const int sd, const int unit, const int bottom,
                                         REAL* a, const int offset_a, const int ld_a,
                                         REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            const REAL c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void uplo_swap_transp (const int sd, const int unit, const int bottom,
                                      REAL* a, const int offset_a, const int ld_a,
                                      REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            const REAL c = b[ib];
            b[ib] = a[ia];
            a[ia] = c;
        }
    }

    __global__ void uplo_axpby_no_transp (const int sd, const int unit, const int bottom,
                                          const REAL alpha, const REAL* a, const int offset_a, const int ld_a,
                                          const REAL beta, REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_0 + gid_1 * ld_b;
            b[ib] = alpha * a[ia] + beta * b[ib];
        }
    }

    __global__ void uplo_axpby_transp (const int sd, const int unit, const int bottom,
                                       const REAL alpha, const REAL* a, const int offset_a, const int ld_a,
                                       const REAL beta, REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid && (unit == 132)
            ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
        if (check) {
            const int ia = offset_a + gid_0 + gid_1 * ld_a;
            const int ib = offset_b + gid_1 + gid_0 * ld_b;
            b[ib] = alpha * a[ia] + beta * b[ib];
        }
    }

    __global__ void uplo_scal (const int sd, const int unit, const int bottom,
                               const REAL alpha, REAL* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            a[offset_a + gid_0 + gid_1 * ld_a] *= alpha;
        }
    }

    __global__ void uplo_set (const int sd, const int unit, const int bottom,
                              const REAL alpha, REAL* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            a[offset_a + gid_0 + gid_1 * ld_a] = alpha;
        }
    }
    
}
