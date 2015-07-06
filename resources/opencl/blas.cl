#ifndef REAL
    #define REAL float
#endif

#ifndef WGS
    #define WGS 256
#endif

//|||||||||||||||||       BLAS 1       |||||||||||||||||||||||||||||||||||||||||

// ================ Embarassingly parallel kernels =============================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void swp (__global REAL* x, __global REAL* y) {
    uint gid = get_global_id(0);
    REAL temp = x[gid];
    x[gid] = y[gid];
    y[gid] = temp;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void scal (__private const REAL alpha, __global REAL* x) {
    uint gid = get_global_id(0);
    x[gid] = alpha * x[gid];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void axpy (__private const REAL alpha,
                    __global const REAL* x, __global REAL* y) {
    uint gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}

// ================= Sum reduction =============================================

inline void work_group_reduction_sum (__global double* acc, const double value) {

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    __local double lacc[WGS];
    lacc[local_id] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    double pacc = value;
    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pacc += lacc[local_id + i + 1];
        }
        if (local_id < i) {
            pacc += lacc[local_id + i];
            lacc[local_id] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global double* acc) {
    work_group_reduction_sum(acc, acc[get_global_id(0)]);
}

// ================== Dot product ==============================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void dot_reduce (__global double* acc,
                          __global const REAL* x, __global const REAL* y) {
    uint gid = get_global_id(0);
    work_group_reduction_sum(acc, (double)(x[gid] * y[gid]));
}

// ================== asum =====================================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void asum_reduce (__global double* acc, __global const REAL* x) {
    work_group_reduction_sum(acc, (double)fabs(x[get_global_id(0)]));
}

// ================== nrm2 =====================================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void nrm2_reduce (__global double* acc, __global const REAL* x) {
    work_group_reduction_sum(acc, (double)pown(x[get_global_id(0)], 2));
}

// ================ Max reduction ==============================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
inline void work_group_reduction_imax (__global uint* iacc,
                                       __global double* vacc,
                                       uint const ind, const double val) {

    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);

    __local uint liacc[WGS];
    __local double lvacc[WGS];
    liacc[local_id] = ind;
    lvacc[local_id] = val;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint index = ind;
    double value = val;

    uint i = local_size;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            double other_value = lvacc[local_id + i + 1];
            if (other_value > value) {
                value = other_value;
                index = liacc[local_id + i + 1];
                lvacc[local_id] = value;
                liacc[local_id] = index;
            }
        }
        if (local_id < i) {
            double other_value = lvacc[local_id + i];
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

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void imax_reduction (__global uint* iacc, __global double* vacc) {
    uint gid = get_global_id(0);
    work_group_reduction_imax(iacc, vacc, iacc[gid], (double)(vacc[gid]));
}

// ================== iamax reduce  ============================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void iamax_reduce (__global uint* iacc, __global REAL* vacc,
                            __global const REAL* x) {
    uint gid = get_global_id(0);
    work_group_reduction_imax(iacc, vacc, gid, (double)(fabs(x[gid])));
}

// ||||||||||||||||       BLAS 2      ||||||||||||||||||||||||||||||||||||||||||

/*__kernel void gemv_reduce (__local REAL* lacc, __global REAL* acc,
                           __global REAL* x, __global REAL* y) {

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    REAL pacc;
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
                   const REAL alpha, const REAL beta,
                   const __global REAL* restrict A,
                   const __global REAL* restrict X,
                   __global REAL* Y) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    int local_col_size = get_global_id(0);
    int local_row_size = get_global_id(1);



}
*/
