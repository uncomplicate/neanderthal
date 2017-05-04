extern "C" {
    
#ifndef REAL
#define REAL float
#endif

    __global__ void vector_equals (const int n, unsigned int* eq_flag,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 const REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            if (x[ix] != y[iy]) {
                eq_flag[0]++;
            }
        }
    }


    __global__ void vector_copy (const int n,
                               const REAL* x, const int offset_x, const int stride_x,
                               REAL* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = x[ix]; 
        }
    }

    
    __global__ void vector_sum (const int n,
                              const REAL* x, const int offset_x, const int stride_x,
                              ACCUMULATOR* acc) {

        int gid = offset_x + (blockIdx.x * blockDim.x + threadIdx.x) * stride_x;
        
        ACCUMULATOR sum = block_reduction_sum( (gid < n) ? x[gid] : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }
        
    }
    
    __global__ void vector_set (const int n, const REAL val,
                              REAL* x, const int offset_x, const int stride_x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset_x + gid * stride_x] = val;
        }
    }

    __global__ void vector_axpby (const int n,
                                  const REAL alpha, const REAL* x, const int offset_x, const int stride_x,
                                  const REAL beta, REAL* y, int offset_y, int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            y[iy] = alpha * x[ix] + beta * y [iy]; 
        }
    }

    
}
