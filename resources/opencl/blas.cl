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

// ================= Summing reduction ==============================================

__kernel void sum_reduction (__local float* lacc, __global float* acc) {

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    float pacc;
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

__kernel void dot_reduce (__local float* lacc, __global float* acc,
                          __global float* x, __global float* y) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    float pacc;
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

// ================== asum =================================================

__kernel void asum_reduce (__local float* lacc, __global float* acc,
                           __global float* x) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    float pacc;
    pacc = fabs(x[gid]);
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

// ================ Max reduction =======================================

__kernel void max_reduction (__local int* liacc, __local float* lvacc,
                             __global int* iacc, __global float* vacc) {

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int index = iacc[gid];
    float value = vacc[gid];
    liacc[lid] = index;
    lvacc[lid] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size >> 1; i > 0; i >>= 1) {
        if (lid < i) {
            float other_value = lvacc[lid + i];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i];
                lvacc[lid] = value;
                liacc[lid] = index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        int group_id = get_group_id(0);
        iacc[group_id] = index;
        vacc[group_id] = value;
    }
}

// ================== iamax reduce  =================================================

__kernel void iamax_reduce (__local int* liacc, __local float* lvacc,
                            __global int* iacc, __global float* vacc,
                            __global float* x) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    float value = fabs(x[gid]);
    int index = gid;
    liacc[lid] = gid;
    lvacc[lid] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size >> 1; i > 0; i >>= 1) {
        if (lid < i) {
            float other_value = lvacc[lid + i];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i];
                lvacc[lid] = value;
                liacc[lid] = index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        int group_id = get_group_id(0);
        iacc[group_id] = index;
        vacc[group_id] = value;
    }

}
