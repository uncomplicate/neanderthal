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
    #define TS 16
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
__kernel void axpy (__private const REAL alpha, __global const REAL* x,
                     __global REAL* y) {
    uint gid = get_global_id(0);
    y[gid] = alpha * x[gid] + y[gid];
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void xpby (__global const REAL* x,
                    const REAL beta, __global REAL* y) {
    uint gid = get_global_id(0);
    y[gid] = x[gid] + beta * y[gid];
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
__kernel void iamax_reduce (__global uint* iacc, __global double* vacc,
                            __global const REAL* x) {
    uint gid = get_global_id(0);
    work_group_reduction_imax(iacc, vacc, gid, (double)(fabs(x[gid])));
}

// ||||||||||||||||       BLAS 2      ||||||||||||||||||||||||||||||||||||||||||

// ================== GEMV =====================================================

inline void work_group_reduction_sum_horizontal
(__global REAL* acc, const REAL value) {

    uint global_size_m = get_global_size(0);
    uint group_id_m = get_group_id(0);
    uint group_id_n = get_group_id(1);

    uint local_m = get_local_size(0);
    uint local_n = get_local_size(1);
    uint local_row = get_local_id(0);
    uint local_col = get_local_id(1);

    __local REAL lacc[WGSm][WGSn];
    lacc[local_row][local_col] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    REAL pacc = value;
    uint i = local_n;
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pacc += lacc[local_row][local_col + i + 1];
        }
        if (local_col < i) {
            pacc += lacc[local_row][local_col + i];
            lacc[local_row][local_col] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_col == 0) {
        acc[(global_size_m * group_id_n)
            + (group_id_m  * WGSm)
            + (global_size_m * local_col) + local_row] = pacc;
    }
}

__attribute__((reqd_work_group_size(WGSm, WGSn, 1)))
__kernel void sum_reduction_horizontal (__global REAL* acc) {

    uint global_size_m = get_global_size(0);
    uint group_id_m = get_group_id(0);
    uint group_id_n = get_group_id(1);
    uint local_row = get_local_id(0);
    uint local_col = get_local_id(1);

    uint a_id = (global_size_m * WGSn * group_id_n)
        + (group_id_m  * WGSm)
        + (global_size_m * local_col) + local_row;

    work_group_reduction_sum_horizontal(acc, acc[a_id]);
}

// ========================= gemv ==============================================

__attribute__((reqd_work_group_size(WGSm, WGSn, 1)))
__kernel void gemv_reduce (__global REAL* acc,
                           const REAL alpha, __global const REAL* a,
                           __global const REAL* x) {

    uint global_size_m = get_global_size(0);
    uint group_id_m = get_group_id(0);
    uint group_id_n = get_group_id(1);
    uint local_row = get_local_id(0);
    uint local_col = get_local_id(1);

    uint a_id = (global_size_m * WGSn * group_id_n)
        + (group_id_m  * WGSm)
        + (global_size_m * local_col) + local_row;

    uint x_id = WGSn * group_id_n + local_col;

    work_group_reduction_sum_horizontal(acc, alpha * a[a_id] * x[x_id]);
}

// ||||||||||||||||       BLAS 3      ||||||||||||||||||||||||||||||||||||||||||

// ================== GEMM =====================================================

inline uint index (const uint m, const uint row, const uint col) {
    return row + m * col;
}

inline uint globalize (const uint tile_size, const uint tile_id,
                       const uint id){
    return tile_id * tile_size + id;
}

// ========================= gemm ==============================================
/* Squared version
__attribute__((reqd_work_group_size(TS, TS, 1)))
__kernel void gemm_tiled (const REAL alpha, __global const REAL* a,
                          __global const REAL* b,
                          const REAL beta, __global REAL* c,
                          const uint m, const uint k, const uint n) {

    uint global_row = get_global_id(0);
    uint global_col = get_global_id(1);

    uint tile_i = get_group_id(0);
    uint tile_j = get_group_id(1);

    uint local_row = get_local_id(0);
    uint local_col = get_local_id(1);

    uint c_row = globalize(WGSm, tile_i, local_row);
    uint c_col = globalize(WGSn, tile_j, local_col);

    uint c_id = index(m, c_row, c_col);

    __local REAL c_tile[TS][TS];
    __local REAL a_tile[TS][TS];
    __local REAL b_tile[TS][TS];

    REAL cacc = 0.0f;

    uint tc = k / TS;


    // Compute full k-tiles
    for (uint t = 0; t < tc; t++) {

        a_tile[local_row][local_col] =
            a[index(m, c_row, globalize(TS, t, local_col))];

        b_tile[local_col][local_row] =
            b[index(k, globalize(TS, t, local_row), c_col)];

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(uint i = 0; i < TS; i++) {
            cacc += a_tile[local_row][i] * b_tile[local_col][i];
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }


        c[c_id] = alpha * cacc + beta * c[c_id];


        }*/


__attribute__((reqd_work_group_size(TS, TS, 1)))
__kernel void gemm_tiled (const REAL alpha, __global const REAL* a,
                          __global const REAL* b,
                          const REAL beta, __global REAL* c,
                          const uint m, const uint k, const uint n) {

    const uint row = get_local_id(0);
    const uint col = get_local_id(1);
    const uint c_row = globalize(TS, get_group_id(0), row);
    const uint c_col = globalize(TS, get_group_id(1), col);

    __local REAL a_tile[TS][TS];
    __local REAL b_tile[TS][TS];

    REAL acc = 0.0f;

    // Elements that are in partial m-tiles and n-tiles need to be
    // loaded, but only if they exist.
    const bool load_row = c_row < m;
    const bool load_col = c_col < n;

    // Compute full k-tiles
    const uint tc = k / TS;
    for (uint t = 0; t < tc; t++) {

        const uint tile_row = TS * t + row;
        const uint tile_col = TS * t + col;
        a_tile[col][row] = load_row ? a[tile_col * m + c_row] : 0.0f;
        b_tile[col][row] = load_col ? b[c_col * k + tile_row] : 0.0f;

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for(uint i = 0; i < TS; i++) {
            acc += a_tile[i][row] * b_tile[col][i];
        }

        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Compute partial k-tiles.
    const uint rest_k = k - tc * TS;
    if (0 < rest_k) {

        const uint tile_row = TS * tc + row;
        const uint tile_col = TS * tc + col;
        a_tile[col][row] = load_row ? a[tile_col * m + c_row] : 0.0f;
        b_tile[col][row] = load_col ? b[c_col * k + tile_row] : 0.0f;

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        for(uint i = 0; i < rest_k; i++) {
            acc += a_tile[i][row] * b_tile[col][i];
        }

    }

    //Only the elements that exist in partial c-tiles should be stored.
    const bool store = load_row && load_col;

    const uint c_id = index(m, c_row, c_col);
    if (store) {
        c[c_id] = alpha * acc + beta * c[c_id];
    }

}

// Cedric's comparative example
__attribute__((reqd_work_group_size(TS, TS, 1)))
__kernel void ce_gemm_tiled (const REAL alpha, __global const REAL* A,
                          __global const REAL* B,
                          const REAL beta, __global REAL* C,
                          const uint M, const uint K, const uint N) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation register
    float acc = 0.0f;

    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];

        // Synchronise to make sure the tile is loaded
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[globalCol*M + globalRow] = alpha * acc + beta * C[globalCol*M + globalRow];
}
/**/
