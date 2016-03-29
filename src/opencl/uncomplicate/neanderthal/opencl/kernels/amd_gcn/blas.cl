#ifndef REAL
#define REAL float
#endif

#ifndef WGS
#define WGS 256
#endif

#ifndef WGSm
#define WGSm 16
#endif

#ifndef WGSn
#define WGSn 16
#endif

#ifndef TS
#define TS 32
#endif

#ifndef WPT
#define WPT 4
#endif

#define RTS (TS/WPT)

//||||||||||||||||| CLBuffer |||||||||||||||||||||||||||||||||||||||||||||||||||

__attribute__((reqd_work_group_size(WGS, 1, 1)))
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

//|||||||||||||||||       BLAS 1       |||||||||||||||||||||||||||||||||||||||||

// ================ Embarassingly parallel kernels =============================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void swap (__global REAL* x, const uint offset_x, const uint stride_x,
                    __global REAL* y, const uint offset_y, const uint stride_y) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    uint iy = offset_y + get_global_id(0) * stride_y;
    REAL temp = x[ix];
    x[ix] = y[iy];
    y[iy] = temp;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void copy (__global const REAL* x,
                    const uint offset_x, const uint stride_x,
                    __global REAL* y, const uint offset_y, const uint stride_y) {
    y[offset_y + get_global_id(0) * stride_y] =
        x[offset_x + get_global_id(0) * stride_x];
}

__kernel void copy_matrix (__global const REAL* a, const uint offset_a,
                           const uint ld_a,
                           __global REAL* b, const uint offset_b,
                           const uint ld_b) {
    uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    b[ib] = a[ia];
}

__kernel void swap_matrix (__global REAL* a, const uint offset_a,
                           const uint ld_a,
                           __global REAL* b, const uint offset_b,
                           const uint ld_b) {
    uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    uint ib = offset_b + get_global_id(0) + get_global_id(1) * ld_b;
    REAL temp = a[ia];
    a[ia] = b[ia];
    b[ib] = temp;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void scal (const REAL alpha, __global REAL* x,
                    const uint offset_x, const uint stride_x) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    x[ix] = alpha * x[ix];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void axpy (const REAL alpha, __global const REAL* x,
                    const uint offset_x, const uint stride_x,
                    __global REAL* y,
                    const uint offset_y, const uint stride_y) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    uint iy = offset_y + get_global_id(0) * stride_y;
    y[iy] = alpha * x[ix] + y[iy];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void axpby (const REAL alpha, __global const REAL* x,
                     uint offset_x, const uint stride_x,
                     const REAL beta, __global REAL* y,
                     const uint offset_y, const uint stride_y) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    uint iy = offset_y + get_global_id(0) * stride_y;
    y[iy] = alpha * x[ix] + beta * y[iy];
}

// ================= Reduction kernels ========================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global double* acc) {
    double sum = work_group_reduction_sum(acc[get_global_id(0)]);
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

// ================== Dot product ==============================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void dot_reduce (__global double* acc, __global const REAL* x,
                          const uint offset_x, const uint stride_x,
                          __global const REAL* y,
                          const uint offset_y, const uint stride_y) {
    REAL px = x[offset_x + get_global_id(0) * stride_x];
    REAL py = y[offset_y + get_global_id(0) * stride_y];
    double sum = work_group_reduction_sum((double)(px * py));
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

// ================== asum =====================================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void asum_reduce (__global double* acc, __global const REAL* x,
                           const uint offset_x, const uint stride_x) {
    REAL px = x[offset_x + get_global_id(0) * stride_x];
    double sum = work_group_reduction_sum((double)fabs(px));
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

// ================== sum =====================================================
__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduce (__global double* acc, __global const REAL* x,
                          const uint offset_x, const uint stride_x) {
    REAL px = x[offset_x + get_global_id(0) * stride_x];
    double sum = work_group_reduction_sum((double)px);
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

// ================== nrm2 =====================================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void nrm2_reduce (__global double* acc, __global const REAL* x,
                           const uint offset_x, const uint stride_x) {
    REAL px = x[offset_x + get_global_id(0) * stride_x];
    double sum = work_group_reduction_sum((double)(px * px));
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

// ================ Max reduction ==============================================
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
__kernel void iamax_reduce (__global uint* iacc, __global double* vacc,
                            __global const REAL* x,
                            const uint offset_x, const uint stride_x) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    work_group_reduction_imax(iacc, vacc, ix, (double)(fabs(x[ix])));
}

// ================== imax reduce  ============================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void imax_reduce (__global uint* iacc, __global double* vacc,
                           __global const REAL* x,
                           const uint offset_x, const uint stride_x) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    work_group_reduction_imax(iacc, vacc, ix, (double)(x[ix]));
}

// ================== imin reduce  ============================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void imin_reduce (__global uint* iacc, __global double* vacc,
                           __global const REAL* x,
                           const uint offset_x, const uint stride_x) {
    uint ix = offset_x + get_global_id(0) * stride_x;
    work_group_reduction_imax(iacc, vacc, ix, (double)(-x[ix]));
}

// ||||||||||||||||       BLAS 2      ||||||||||||||||||||||||||||||||||||||||||

// ================== GEMV =====================================================

__attribute__((reqd_work_group_size(WGSm, WGSn, 1)))
__kernel void sum_reduction_horizontal (__global REAL* acc) {

    uint ia = get_global_size(0) * get_global_id(1) + get_global_id(0);
    REAL sum = work_group_reduction_sum_horizontal(acc[ia]);
    if (get_local_id(1) == 0) {
        acc[get_global_size(0) * get_group_id(1) + get_global_id(0)] = sum;
    }
}

// ========================= gemv ==============================================

__attribute__((reqd_work_group_size(WGSm, WGSn, 1)))
__kernel void gemv_reduce (__global REAL* acc,
                           const REAL alpha, __global const REAL* a,
                           const uint offset_a, const uint ld_a,
                           __global const REAL* x, const uint offset_x,
                           const uint stride_x) {

    uint ia = offset_a + ld_a * get_global_id(1) + get_local_id(0);
    uint ix = offset_x + get_global_id(1) * stride_x;
    REAL sum = work_group_reduction_sum_horizontal(alpha * a[ia] * x[ix]);
    if (get_local_id(1) == 0) {
        acc[get_global_size(0) * get_group_id(1) + get_global_id(0)] = sum;
    }
}

// ========================== gerk =============================================

__attribute__((reqd_work_group_size(WGSm, WGSn, 1)))
__kernel void gerk (const REAL alpha, __global const REAL* x,
                    const uint offset_x, const uint stride_x,
                    __global const REAL* y,
                    const uint offset_y, const uint stride_y,
                    __global REAL* a, const uint offset_a, const uint ld_a) {

    uint ix = offset_x + get_global_id(0) * stride_x;
    uint iy = offset_y + get_global_id(1) * stride_y;
    uint ia = offset_a + get_global_id(0) + get_global_id(1) * ld_a;
    a[ia] += alpha * x[ix] * y[iy];
}


// ||||||||||||||||       BLAS 3      ||||||||||||||||||||||||||||||||||||||||||

// ================== GEMM =====================================================

inline uint globalize (const uint tile_size, const uint tile_id,
                       const uint id){
    return tile_id * tile_size + id;
}

// ========================= gemm ==============================================

__attribute__((reqd_work_group_size(TS, RTS, 1)))
__kernel void gemm_tiled (const REAL alpha, __global const REAL* a,
                          const uint offset_a, const uint ld_a,
                          __global const REAL* b,
                          const uint offset_b, const uint ld_b,
                          const REAL beta, __global REAL* c,
                          const uint offset_c, const uint ld_c,
                          const uint m, const uint k, const uint n) {

    const uint row = get_local_id(0);
    const uint col = get_local_id(1);
    const uint c_row = globalize(TS, get_group_id(0), row);
    const uint c_col = globalize(TS, get_group_id(1), col);

    // Local tiles of matrices A and B
    __local REAL a_tile[TS][TS];
    __local REAL b_tile[TS][TS];

    REAL acc[WPT];

    // Elements that are in partial m-tiles and n-tiles need to be
    // loaded, but only if they exist.
    const bool load_row = c_row < m;
    bool load_col[WPT];

    #pragma unroll
    for (uint w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
        load_col[w] = c_col + w * RTS < n;
    }

    // Compute full k-tiles
    const uint tc = k / TS;
    for (uint t = 0; t < tc; t++) {

        const uint tile_row = TS * t + row;
        const uint tile_col = TS * t + col;

        for (uint w = 0; w < WPT; w++) {
            const uint ia = offset_a + (tile_col + w * RTS) * ld_a + c_row;
            const uint ib = offset_b + (c_col + w * RTS) * ld_b + tile_row;
            a_tile[col + w * RTS][row] = load_row ? a[ia] : 0.0f;
            b_tile[col + w * RTS][row] = load_col[w] ? b[ib] : 0.0f;
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(uint i = 0; i < TS; i++) {

            #pragma unroll
            for(uint w = 0; w < WPT; w++) {
                acc[w] += a_tile[i][row] * b_tile[col + w * RTS][i];
            }
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Compute partial k-tiles.
    const uint rest_k = k - tc * TS;
    if (0 < rest_k) {

        const uint tile_row = TS * tc + row;
        const uint tile_col = TS * tc + col;

        for (uint w = 0; w < WPT; w++) {
            const uint ia = offset_a + (tile_col + w * RTS) * ld_a + c_row;
            const uint ib = offset_b + (c_col + w * RTS) * ld_b + tile_row;
            a_tile[col + w * RTS][row] = load_row ? a[ia] : 0.0f;
            b_tile[col + w * RTS][row] = load_col[w] ? b[ib] : 0.0f;
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(uint i = 0; i < rest_k; i++) {

            #pragma unroll
            for(uint w = 0; w < WPT; w++) {
                acc[w] += a_tile[i][row] * b_tile[col + w * RTS][i];
            }
        }

    }

    //Only the elements that exist in partial c-tiles should be stored.
#pragma unroll
    for (uint w = 0; w < WPT; w++) {
        const bool store = load_row && load_col[w];
        if (store) {
            const uint ic = offset_c + (c_col + w * RTS) * ld_c + c_row;
            c[ic] = alpha * acc[w] + beta * c[ic];
        }
    }

}

// Simpler version that requires dimensions that fit tiles

__attribute__((reqd_work_group_size(TS, RTS, 1)))
__kernel void gemm_tiled_fit (const REAL alpha, __global const REAL* a,
                              const uint offset_a, const uint ld_a,
                              __global const REAL* b,
                              const uint offset_b, const uint ld_b,
                              const REAL beta, __global REAL* c,
                              const uint offset_c, const uint ld_c,
                              const uint m, const uint k, const uint n) {

    const uint row = get_local_id(0);
    const uint col = get_local_id(1);
    const uint c_row = globalize(TS, get_group_id(0), row);
    const uint c_col = globalize(TS, get_group_id(1), col);

    // Local tiles of matrices A and B
    __local REAL a_tile[TS][TS];
    __local REAL b_tile[TS][TS];

    REAL acc[WPT];

    #pragma unroll
    for (uint w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Compute full k-tiles
    const uint tc = k / TS;
    for (uint t = 0; t < tc; t++) {
        for (uint w = 0; w < WPT; w++) {
            const uint tile_row = TS * t + row;
            const uint tile_col = TS * t + col;
            const uint ia = offset_a + (tile_col + w * RTS) * ld_a + c_row;
            const uint ib = offset_b + (c_col + w * RTS) * ld_b + tile_row;
            a_tile[col + w * RTS][row] = a[ia];
            b_tile[col + w * RTS][row] = b[ib];
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(uint i = 0; i < TS; i++) {

            #pragma unroll
            for(uint w = 0; w < WPT; w++) {
                acc[w] += a_tile[i][row] * b_tile[col + w * RTS][i];
            }
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint w = 0; w < WPT; w++) {
        const uint ic = offset_c + (c_col + w * RTS) * ld_c + c_row;
        c[ic] = alpha * acc[w] + beta * c[ic];
    }

}
