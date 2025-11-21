extern "C" {

#ifndef REAL
#define REAL float
#endif

#ifndef ACCUMULATOR
#define ACCUMULATOR float
#endif

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

    __global__ void tr_sum (const int sd, const int fd, const int unit, const int bottom,
                            ACCUMULATOR* acc,
                            const REAL* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const ACCUMULATOR sum = block_reduction_sum_2( (check) ? a[i] : 0.0);
        if (threadIdx.y == 0) {
            const ACCUMULATOR sum2 = block_reduction_sum((check) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

    __global__ void sy_sum (const int sd, const int fd, const int unit, const int bottom,
                            ACCUMULATOR* acc,
                            const REAL* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        //const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const bool check = bottom * gid_0 >= bottom * gid_1;
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const REAL val = (check) ? a[i] : 0.0;
        const ACCUMULATOR sum = block_reduction_sum_2( (gid_0 == gid_1) ? val : 2.0 * val);
        if (threadIdx.y == 0) {
            const ACCUMULATOR sum2 = block_reduction_sum((check) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

}
