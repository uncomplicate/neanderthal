#ifndef REAL
define REAL float
#endif

#ifndef WGS
#define WGS 256
#endif

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_equals (__global const REAL* x, const uint offset_x, const uint stride_x,
                             __global const REAL* y, const uint offset_y, const uint stride_y,
                             __global uint* eq_flag) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    const uint iy = offset_y + get_global_id(0) * stride_y;
    if (x[ix] != y[iy]){
        eq_flag[0]++;
    }
}

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_axpby (REAL alpha, __global const REAL* x, const uint offset_x, const uint stride_x,
                            REAL beta, __global REAL* y, const uint offset_y, const uint stride_y) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    const uint iy = offset_y + get_global_id(0) * stride_y;
    y[iy] = alpha * x[ix] + beta * y[iy];
}

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void vector_set (const REAL val, __global REAL* x, const uint offset_x, const uint stride_x) {
    x[offset_x + get_global_id(0) * stride_x] = val;
}

__kernel void ge_equals_no_transp (__global const REAL* a, const uint offset_a, const uint ld_a,
                                   __global const REAL* b, const uint offset_b, const uint ld_b,
                                   __global uint* eq_flag) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    if (a[ia] != b[ib]){
        eq_flag[0]++;
    }
}

__kernel void ge_equals_transp (__global const REAL* a, const uint offset_a, const uint ld_a,
                                __global const REAL* b, const uint offset_b, const uint ld_b,
                                __global uint* eq_flag) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
    if (a[ia] != b[ib]){
        eq_flag[0]++;
    }
}

__kernel void ge_swap_no_transp (__global REAL* a, const uint offset_a, const uint ld_a,
                                 __global REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    const REAL c = b[ib];
    b[ib] = a[ia];
    a[ia] = c;
}

__kernel void ge_swap_transp (__global REAL* a, const uint offset_a, const uint ld_a,
                              __global REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
    const REAL c = b[ib];
    b[ib] = a[ia];
    a[ia] = c;
}


__kernel void ge_set (const REAL val, __global REAL* a, const uint offset_a, const uint ld_a) {
    a[offset_a + get_global_id(0) + get_global_id(1) * ld_a] = val;
}

__kernel void ge_axpby_no_transp (REAL alpha, __global const REAL* a, const uint offset_a, const uint ld_a,
                                  REAL beta, __global REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    b[ib] = alpha * a[ia] + beta * b[ib];
}

__kernel void ge_axpby_transp (REAL alpha, __global const REAL* a, const uint offset_a, const uint ld_a,
                               REAL beta, __global REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
    b[ib] = alpha * a[ia] + beta * b[ib];
}

__kernel void uplo_equals_no_transp (const uint unit, const int bottom,
                                     __global const REAL* a, const uint offset_a, const uint ld_a,
                                     __global const REAL* b, const uint offset_b, const uint ld_b,
                                     __global uint* eq_flag) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_0 + gid_1 * ld_b;
        if (a[ia] != b[ib]){
            eq_flag[0]++;
        }
    }
}

__kernel void uplo_equals_transp (const uint unit, const int bottom,
                                  __global const REAL* a, const uint offset_a, const uint ld_a,
                                  __global const REAL* b, const uint offset_b, const uint ld_b,
                                  __global uint* eq_flag) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_1 + gid_0 * ld_b;
        if (a[ia] != b[ib]){
            eq_flag[0]++;
        }
    }
}

__kernel void uplo_swap_no_transp (const uint unit, const int bottom,
                                   __global REAL* a, const uint offset_a, const uint ld_a,
                                   __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_0 + gid_1 * ld_b;
        const REAL c = b[ib];
        b[ib] = a[ia];
        a[ia] = c;
    }
}

__kernel void uplo_swap_transp (const uint unit, const int bottom,
                                __global REAL* a, const uint offset_a, const uint ld_a,
                                __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_1 + gid_0 * ld_b;
        const REAL c = b[ib];
        b[ib] = a[ia];
        a[ia] = c;
    }
}

__kernel void uplo_copy_no_transp (const uint unit, const int bottom,
                                   __global const REAL* a, const uint offset_a, const uint ld_a,
                                   __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_0 + gid_1 * ld_b;
        b[ib] = a[ia];
    }
}

__kernel void uplo_copy_transp (const uint unit, const int bottom,
                                __global const REAL* a, const uint offset_a, const uint ld_a,
                                __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_1 + gid_0 * ld_b;
        b[ib] = a[ia];
    }
}

__kernel void uplo_axpby_no_transp (const uint unit, const int bottom, const REAL alpha,
                                    __global const REAL* a, const uint offset_a, const uint ld_a,
                                    const REAL beta, __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_0 + gid_1 * ld_b;
        b[ib] = alpha * a[ia] + beta * b[ib];
    }
}

__kernel void uplo_axpby_transp (const uint unit, const int bottom, const REAL alpha,
                                 __global const REAL* a, const uint offset_a, const uint ld_a,
                                 const REAL beta, __global REAL* b, const uint offset_b, const uint ld_b) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        const uint ia = offset_a + gid_0 + gid_1 * ld_a;
        const uint ib = offset_b + gid_1 + gid_0 * ld_b;
        b[ib] = alpha * a[ia] + beta * b[ib];
    }
}

__kernel void uplo_scal (const uint unit, const int bottom,
                         const REAL alpha, __global REAL* a, const uint offset_a, const uint ld_a) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        a[offset_a + gid_0 + gid_1 * ld_a] *= alpha;
    }
}

__kernel void uplo_set (const uint unit, const int bottom,
                        const REAL alpha, __global REAL* a, const uint offset_a, const uint ld_a) {
    const int gid_0 = get_global_id(0);
    const int gid_1 = get_global_id(1);
    const bool check = (unit == 132)
        ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1;
    if (check) {
        a[offset_a + gid_0 + gid_1 * ld_a] = alpha;
    }
}
