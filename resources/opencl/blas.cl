__kernel void swp (__global float* x, __global float* y) {
    uint gid = get_global_id(0);
    float temp = x[gid];
    x[gid] = y[gid];
    y[gid] = temp;
}

__kernel void scal (__private float alpha, __global float* x) {
    uint gid = get_global_id(0);
    x[gid] = alpha * x[gid];
}

__kernel void axpy (__private float alpha, __global float* x, __global float* y) {
    uint gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}

// ================= Summing reduction =====================================

__kernel void sum_reduction (__global float* acc) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    float pacc = work_group_reduce_add(acc[gid]);

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
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (lid == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pacc += lacc[lid + i + 1];
        }
        if (lid < i) {
            pacc += lacc[lid + i];
            lacc[lid] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
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
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (lid == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pacc += lacc[lid + i + 1];
        }
        if (lid < i) {
            pacc += lacc[lid + i];
            lacc[lid] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

// ================ Max reduction =======================================

__kernel void imax_reduction (__local int* liacc, __local float* lvacc,
                              __global int* iacc, __global float* vacc) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    uint index = iacc[gid];
    float value = vacc[gid];
    liacc[lid] = index;
    lvacc[lid] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (lid == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            float other_value = lvacc[lid + i + 1];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i + 1];
            }
        }
        if (lid < i) {
            float other_value = lvacc[lid + i];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i];
                lvacc[lid] = value;
                liacc[lid] = index;
            }
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        int group_id = get_group_id(0);
        iacc[group_id] = index;
        vacc[group_id] = value;
    }
}

// ================== iamax reduce  ============================================

__kernel void iamax_reduce (__local int* liacc, __local float* lvacc,
                               __global int* iacc, __global float* vacc,
                               __global float* x) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    float value = fabs(x[gid]);
    uint index = gid;
    liacc[lid] = gid;
    lvacc[lid] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (lid == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            float other_value = lvacc[lid + i + 1];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i + 1];
            }
        }
        if (lid < i) {
            float other_value = lvacc[lid + i];
            if (other_value > value) {
                value = other_value;
                index = liacc[lid + i];
                lvacc[lid] = value;
                liacc[lid] = index;
            }
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        int group_id = get_group_id(0);
        iacc[group_id] = index;
        vacc[group_id] = value;
    }

}

// ======================== BLAS 2 ============================================

/*__kernel void gemv_reduce (__local float* lacc, __global float* acc,
                           __global float* x, __global float* y) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    float pacc;
    pacc = x[gid] * y[gid];
    lacc[lid] = pacc;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = local_size >> 1; i > 0; i >>= 1) {
        if (lid < i) {
            pacc += lacc[lid + i];
            lacc[lid] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
    }*/


/*__kernel void gemv(__local int* lacc,
                   const float alpha, const float beta,
                   const __global float* restrict A,
                   const __global float* restrict X,
                   __global float* Y) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    int local_col_size = get_global_id(0);
    int local_row_size = get_global_id(1);



}
*/
