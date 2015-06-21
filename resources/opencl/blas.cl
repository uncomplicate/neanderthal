__kernel void swp_kernel(__global float* x, __global float* y) {
    int gid = get_global_id(0);
    float temp = x[gid];
    x[gid] = y[gid];
    y[gid] = temp;
}

__kernel void scal_kernel(float alpha, __global float* x) {
    int gid = get_global_id(0);
    x[gid] = alpha * x[gid];
}

__kernel void axpy_kernel(float alpha, __global float* x, __global float* y) {
    int gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}
