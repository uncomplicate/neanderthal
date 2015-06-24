__kernel void swp (__global float* x, __global float* y) {
    int gid = get_global_id(0);
    float temp = x[gid];
    x[gid] = y[gid];
    y[gid] = temp;
}

__kernel void scal (__private float alpha, __global float* x) {
    int gid = get_global_id(0);
    x[gid] = alpha * x[gid];
}

__kernel void axpy (__private float alpha, __global float* x, __global float* y) {
    int gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}

// ================= Reduction ==============================================

__kernel void reduce (__global float* acc, __local float* lacc) {

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    float pacc = 0;
    pacc = acc[gid];
    lacc[lid] = pacc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size >> 1; i > 0; i >>= 1) {
        if (lid < i) {
            pacc += lacc[lid + i];
            lacc[lid] = pacc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

// ================== Dot product ===========================================

__kernel void dot_reduce (__global float* acc, __local float* lacc,
                          __global float* x, __global float* y) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    float pacc = 0;
    pacc = x[gid] * y[gid];
    lacc[lid] = pacc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size >> 1; i > 0; i >>= 1) {
        if (lid < i) {
            pacc += lacc[lid + i];
            lacc[lid] = pacc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

//==================
