#ifndef REAL
#define REAL float
#endif

__kernel void equals_vector (__global uint* eq_flag,
                             __global const REAL* x, const uint offset_x,
                             const uint stride_x,
                             __global const REAL* y, const uint offset_y,
                             const uint stride_y) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    uint iy = offset_y + get_global_id(0) * stride_y;
    if ((x[ix] != y[iy])){
        eq_flag[0]++;
    }
}

__kernel void equals_matrix (__global uint* eq_flag,
                             __global const REAL* a, const uint offset_a,
                             const uint ld_a,
                             __global const REAL* b, const uint offset_b,
                             const uint ld_b) {
    uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    if ((a[ia] != b[ib])){
        eq_flag[0]++;
    }
}
