#ifndef REAL
#define REAL float
#endif

__kernel void vector_equals (__global uint* eq_flag,
                             __global const REAL* x, const uint offset_x, const uint stride_x,
                             __global const REAL* y, const uint offset_y, const uint stride_y) {
    const uint ix = offset_x + get_global_id(0) * stride_x;
    const uint iy = offset_y + get_global_id(0) * stride_y;
    if (x[ix] != y[iy]){
        eq_flag[0]++;
    }
}

__kernel void ge_equals_no_transp (__global uint* eq_flag,
                                   __global const REAL* a, const uint offset_a, const uint ld_a,
                                   __global const REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    if (a[ia] != b[ib]){
        eq_flag[0]++;
    }
}

__kernel void ge_equals_transp (__global uint* eq_flag,
                                __global const REAL* a, const uint offset_a, const uint ld_a,
                                __global const REAL* b, const uint offset_b, const uint ld_b) {
    const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
    if (a[ia] != b[ib]){
        eq_flag[0]++;
    }
}

__kernel void tr_equals_bottom (__global uint* eq_flag, const uint unit,
                                __global const REAL* a, const uint offset_a, const uint ld_a,
                                __global const REAL* b, const uint offset_b, const uint ld_b) {
    const bool check = (unit == 132)
        ? get_global_id(0) > get_global_id(1) : get_global_id(0) >= get_global_id(1);
    if (check) {
        const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
        const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
        if (a[ia] != b[ib]){
            eq_flag[0]++;
        }
    }
}

__kernel void tr_equals_top (__global uint* eq_flag, const uint unit,
                             __global const REAL* a, const uint offset_a, const uint ld_a,
                             __global const REAL* b, const uint offset_b, const uint ld_b) {
    const bool check = (unit == 132)
        ? get_global_id(0) < get_global_id(1) : get_global_id(0) <= get_global_id(1);
    if (check) {
        const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
        const uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
        if ((a[ia] != b[ib])){
            eq_flag[0]++;
        }
    }
}

__kernel void tr_equals_bottom_transp (__global uint* eq_flag, const uint unit,
                                       __global const REAL* a, const uint offset_a, const uint ld_a,
                                       __global const REAL* b, const uint offset_b, const uint ld_b) {
    const bool check = (unit == 132)
        ? get_global_id(0) > get_global_id(1) : get_global_id(0) >= get_global_id(1);
    if (check) {
        const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
        const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
        if (a[ia] != b[ib]){
            eq_flag[0]++;
        }
    }
}

__kernel void tr_equals_top_transp (__global uint* eq_flag, const uint unit,
                                    __global const REAL* a, const uint offset_a, const uint ld_a,
                                    __global const REAL* b, const uint offset_b, const uint ld_b) {
    const bool check = (unit == 132)
        ? get_global_id(0) < get_global_id(1) : get_global_id(0) <= get_global_id(1);
    if (check) {
        const uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
        const uint ib = offset_b + get_global_id(1) + get_global_id(0) * ld_b;
        if (a[ia] != b[ib]){
            eq_flag[0]++;
        }
    }
}

__kernel void vector_set (const REAL val, __global REAL* x, const uint offset_x, const uint stride_x) {
    x[offset_x + get_global_id(0) * stride_x] = val;
}

__kernel void ge_set (const REAL val, __global REAL* a, const uint offset_a, const uint ld_a) {
    a[offset_a + get_global_id(0) + get_global_id(1) * ld_a] = val;
}
