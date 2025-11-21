extern "C" {

#ifndef INTEGER
#define INTEGER int
#endif

#ifndef ACCUMULATOR
#define ACCUMULATOR int
#endif

        __global__ void vector_asum (const int n,
                                const INTEGER* x, const int offset_x, const int stride_x,
                                ACCUMULATOR* acc) {

        int gid = (blockIdx.x * blockDim.x + threadIdx.x);
        ACCUMULATOR sum = block_reduction_sum( (gid < n) ? abs(x[offset_x + gid * stride_x]) : 0.0);
        if (threadIdx.x == 0) {
            acc[blockIdx.x] = sum;
        }

    }


    __global__ void ge_asum (const int sd, const int fd,
                             ACCUMULATOR* acc,
                             const INTEGER* a, const int offset_a, const int ld_a) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        const int i = offset_a + gid_0 + gid_1 * ld_a;
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? abs(a[i]) : 0.0);
        if (threadIdx.y == 0) {
            const ACCUMULATOR sum2 = block_reduction_sum((valid) ? sum : 0.0);
            if (threadIdx.x == 0) {
                acc[gridDim.x * blockIdx.y + blockIdx.x] = sum2;
            }
        }
    }

}
