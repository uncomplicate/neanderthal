__kernel void swp_kernel(__global float* x, __global float* y) {
    int gid = get_global_id(0);
    float temp = x[gid];
    x[gid] = y[gid];
    y[gid] = temp;
}
