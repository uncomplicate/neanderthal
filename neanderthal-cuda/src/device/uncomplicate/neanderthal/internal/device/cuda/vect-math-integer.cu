extern "C" {

#ifndef INTEGER
#define INTEGER float
#endif

    __global__ void vector_abs (const int n,
                                const INTEGER* x, const int offset_x, const int stride_x,
                                INTEGER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = abs(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_lmax (const int n,
                                 const INTEGER* x, const int offset_x, const int stride_x,
                                 const INTEGER* y, const int offset_y, const int stride_y,
                                 INTEGER* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                max(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_lmin (const int n,
                                 const INTEGER* x, const int offset_x, const int stride_x,
                                 const INTEGER* y, const int offset_y, const int stride_y,
                                 INTEGER* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                min(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_ramp (const int n,
                                 const INTEGER* x, const int offset_x, const int stride_x,
                                 INTEGER* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = max((INTEGER)0, x[offset_x + gid * stride_x]);
        }
    }

    __global__ void ge_abs (const int sd, const int fd,
                            const INTEGER* a, const int offset_a, const int ld_a,
                            INTEGER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = abs(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_linear_frac (const int sd, const int fd,
                                    const INTEGER* a, const int offset_a, const int ld_a,
                                    const INTEGER* b, const int offset_b, const int ld_b,
                                    const INTEGER scalea, const INTEGER shifta, const INTEGER scaleb, const INTEGER shiftb,
                                    INTEGER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                (scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta) /
                (scaleb * b[offset_b + gid_0 + gid_1 * ld_b] + shiftb);
        }
    }

    __global__ void ge_lmax (const int sd, const int fd,
                             const INTEGER* a, const int offset_a, const int ld_a,
                             const INTEGER* b, const int offset_b, const int ld_b,
                             INTEGER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                max(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_lmin (const int sd, const int fd,
                             const INTEGER* a, const int offset_a, const int ld_a,
                             const INTEGER* b, const int offset_b, const int ld_b,
                             INTEGER* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                min(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_ramp (const int sd, const int fd,
                             const INTEGER* a, const int offset_a, const int ld_a,
                             INTEGER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = max(a[offset_a + gid_0 + gid_1 * ld_a], (INTEGER)0.0);
        }
    }

    __global__ void ge_relu (const int sd, const int fd,
                             const INTEGER* alpha, const int offset_alpha, const int ld_alpha,
                             const INTEGER* a, const int offset_a, const int ld_a,
                             INTEGER* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const INTEGER val = a[offset_a + gid_0 + gid_1 * ld_a];
            const INTEGER alpha_val = alpha[offset_alpha + gid_0 + gid_1 * ld_alpha];
            b[offset_b + gid_0 + gid_1 * ld_b] = ((INTEGER)0 < val) ? val : (alpha_val * val);
        }
    }

}
