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

__kernel void axpy (__private float alpha,
                    __global float* x, __global float* y) {
    uint gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}

// ================= Sum reduction =====================================

inline float work_group_reduction_sum (__local float* lacc,
                                       uint local_id, float value) {

    lacc[local_id] = value;
    uint local_size = get_local_size(0);
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float result = value;
    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            result += lacc[local_id + i + 1];
        }
        if (local_id < i) {
            result += lacc[local_id + i];
            lacc[local_id] = result;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
    return result;
}

__kernel void sum_reduction (__local float* lacc, __global float* acc) {

    uint lid = get_local_id(0);

    float pacc = work_group_reduction_sum(lacc, lid, acc[get_global_id(0)]);

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

// ================== Dot product ===========================================

__kernel void dot_reduce (__local float* lacc, __global float* acc,
                          __global float* x, __global float* y) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    float pacc = work_group_reduction_sum(lacc, lid, x[gid] * y[gid]);

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

// ================== asum =================================================

__kernel void asum_reduce (__local float* lacc, __global float* acc,
                           __global float* x) {

    uint lid = get_local_id(0);

    float pacc = work_group_reduction_sum(lacc, lid, fabs(x[get_global_id(0)]));

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }

}

// ================== nrm2 =================================================

__kernel void nrm2_reduce (__local float* lacc, __global float* acc,
                           __global float* x) {

    uint lid = get_local_id(0);

    float pacc = work_group_reduction_sum(lacc, lid,
                                          pown(x[get_global_id(0)], 2));

    if(lid == 0) {
        acc[get_group_id(0)] = pacc;
    }

}

// ================ Max reduction =======================================

inline void work_group_reduction_imax (__local uint* liacc,
                                       __local float* lvacc,
                                       __global uint* iacc,
                                       __global float* vacc,
                                       uint local_id,
                                       uint ind, float val) {
    liacc[local_id] = ind;
    lvacc[local_id] = val;
    uint local_size = get_local_size(0);
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint index = ind;
    float value = val;

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            float other_value = lvacc[local_id + i + 1];
            if (other_value > value) {
                value = other_value;
                index = liacc[local_id + i + 1];
                lvacc[local_id] = value;
                liacc[local_id] = index;
            }
        }
        if (local_id < i) {
            float other_value = lvacc[local_id + i];
            if (other_value > value) {
                value = other_value;
                index = liacc[local_id + i];
                lvacc[local_id] = value;
                liacc[local_id] = index;
            }
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0) {
        uint group_id = get_group_id(0);
        iacc[group_id] = index;
        vacc[group_id] = value;
    }

}

__kernel void imax_reduction (__local uint* liacc, __local float* lvacc,
                              __global uint* iacc, __global float* vacc) {

    uint gid = get_global_id(0);

    work_group_reduction_imax(liacc, lvacc, iacc, vacc, get_local_id(0),
                              iacc[gid], vacc[gid]);

}

// ================== iamax reduce  ============================================

__kernel void iamax_reduce (__local uint* liacc, __local float* lvacc,
                               __global uint* iacc, __global float* vacc,
                               __global float* x) {
    uint gid = get_global_id(0);

    work_group_reduction_imax(liacc, lvacc, iacc, vacc, get_local_id(0),
                              gid, fabs(x[gid]));

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
