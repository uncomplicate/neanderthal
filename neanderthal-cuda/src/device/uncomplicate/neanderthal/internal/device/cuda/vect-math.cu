extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif

    __global__ void vector_sqr (const int n,
                                const NUMBER* x, const int offset_x, const int stride_x,
                                NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const NUMBER xval = x[offset_x + gid * stride_x];
            y[offset_y + gid * stride_y] = xval * xval;
        }
    }

    __global__ void vector_mul (const int n,
                                const NUMBER* x, const int offset_x, const int stride_x,
                                const NUMBER* y, const int offset_y, const int stride_y,
                                NUMBER* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] = x[offset_x + gid * stride_x] * y[offset_y + gid * stride_y];
        }
    }

    __global__ void vector_div (const int n,
                                const NUMBER* x, const int offset_x, const int stride_x,
                                const NUMBER* y, const int offset_y, const int stride_y,
                                NUMBER* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] = x[offset_x + gid * stride_x] / y[offset_y + gid * stride_y];
        }
    }

    __global__ void vector_inv (const int n,
                                const NUMBER* x, const int offset_x, const int stride_x,
                                NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = (NUMBER)1 / x[offset_x + gid * stride_x];
        }
    }

    __global__ void vector_scale_shift (const int n,
                                        const NUMBER* x, const int offset_x, const int stride_x,
                                        const NUMBER scalea, const NUMBER shifta, const NUMBER scaleb, const NUMBER shiftb,
                                        NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = scalea * x[offset_x + gid * stride_x] + shifta;

        }
    }

    __global__ void vector_relu (const int n,
                                 const NUMBER* alpha, const int offset_alpha, const int stride_alpha,
                                 const NUMBER* x, const int offset_x, const int stride_x,
                                 NUMBER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const NUMBER val = x[offset_x + gid * stride_x];
            y[offset_y + gid * stride_y]
                = ((NUMBER)0.0 < val) ? val : alpha[offset_alpha + gid * stride_alpha] * val;
        }
    }

    __global__ void ge_sqr (const int sd, const int fd,
                            const NUMBER* a, const int offset_a, const int ld_a,
                            NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const NUMBER aval = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = aval * aval;
        }
    }

    __global__ void ge_mul (const int sd, const int fd,
                            const NUMBER* a, const int offset_a, const int ld_a,
                            const NUMBER* b, const int offset_b, const int ld_b,
                            NUMBER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] * b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void ge_div (const int sd, const int fd,
                            const NUMBER* a, const int offset_a, const int ld_a,
                            const NUMBER* b, const int offset_b, const int ld_b,
                            NUMBER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] / b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void ge_inv (const int sd, const int fd,
                            const NUMBER* a, const int offset_a, const int ld_a,
                            NUMBER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = (NUMBER)1.0 / a[offset_a + gid_0 + gid_1 * ld_a];
        }
    }

    __global__ void ge_scale_shift (const int sd, const int fd,
                                    const NUMBER* a, const int offset_a, const int ld_a,
                                    const NUMBER scalea, const NUMBER shifta, const NUMBER scaleb, const NUMBER shiftb,
                                    NUMBER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] = scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta;

        }
    }

}
