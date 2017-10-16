extern "C" {
    
#ifndef REAL
#define REAL float
#endif

#ifndef CAST
#define CAST(fun) fun ## f
#endif
    
#ifndef REAL2o3
#define REAL2o3 (REAL)0.6666666666666666
#endif

#ifndef REAL3o2
#define REAL3o2 (REAL)1.5
#endif

    __global__ void vector_sqr (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const REAL xval = x[offset_x + gid * stride_x];
            y[offset_y + gid * stride_y] = xval * xval;
        }
    }

    __global__ void vector_mul (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                const REAL* y, const int offset_y, const int stride_y,
                                REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] = x[offset_x + gid * stride_x] * y[offset_y + gid * stride_y];
        }
    }

    __global__ void vector_div (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                const REAL* y, const int offset_y, const int stride_y,
                                REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] = x[offset_x + gid * stride_x] / y[offset_y + gid * stride_y];
        }
    }

    __global__ void vector_inv (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = (REAL)1.0 / x[offset_x + gid * stride_x];
        }
    }

    __global__ void vector_abs (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(fabs)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_linear_frac (const int n,
                                        const REAL* x, const int offset_x, const int stride_x,
                                        const REAL* y, const int offset_y, const int stride_y,
                                        const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                        REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                (scalea * x[offset_x + gid * stride_x] + shifta) /
                (scaleb * y[offset_y + gid * stride_y] + shiftb);
        }
    }

    __global__ void vector_scale_shift (const int n,
                                        const REAL* x, const int offset_x, const int stride_x,
                                        const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                        REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = scalea * x[offset_x + gid * stride_x] + shifta;
                
        }
    }

    __global__ void vector_fmod (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 const REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] = CAST(fmod)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_frem (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 const REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(remainder)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_sqrt (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(sqrt)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_inv_sqrt (const int n,
                                     const REAL* x, const int offset_x, const int stride_x,
                                     REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(rsqrt)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_cbrt (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(cbrt)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_inv_cbrt (const int n,
                                     const REAL* x, const int offset_x, const int stride_x,
                                     REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(rcbrt)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_pow2o3 (const int n,
                                   const REAL* x, const int offset_x, const int stride_x,
                                   REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(pow)(x[offset_x + gid * stride_x], REAL2o3);
        }
    }

    __global__ void vector_pow3o2 (const int n,
                                   const REAL* x, const int offset_x, const int stride_x,
                                   REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(pow)(x[offset_x + gid * stride_x], REAL3o2);
        }
    }

    __global__ void vector_pow (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                const REAL* y, const int offset_y, const int stride_y,
                                REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(pow)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_powx (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 const REAL b,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(pow)(x[offset_x + gid * stride_x], b);
        }
    }

    __global__ void vector_hypot (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  const REAL* y, const int offset_y, const int stride_y,
                                  REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(hypot)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }
    
    __global__ void vector_exp (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(exp)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_expm1 (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(expm1)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_log (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(log)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_log10 (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(log10)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_sin (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(sin)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_cos (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(cos)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_tan (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(tan)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_sincos (const int n,
                                   const REAL* x, const int offset_x, const int stride_x,
                                   REAL* y, const int offset_y, const int stride_y,
                                   REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            CAST(sincos)(x[offset_x + gid * stride_x], &y[offset_y + gid * stride_y], &z[offset_z + gid * stride_z]);
        }
    }

    __global__ void vector_asin (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(asin)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_acos (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(acos)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_atan (const int n,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(atan)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_atan2 (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  const REAL* y, const int offset_y, const int stride_y,
                                  REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(atan2)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_sinh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(sinh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_cosh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(cosh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_tanh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(tanh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_asinh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(asinh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_acosh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(acosh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_atanh (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(atanh)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_erf (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(erf)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_erf_inv (const int n,
                                    const REAL* x, const int offset_x, const int stride_x,
                                    REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(erfinv)(x[offset_x + gid * stride_x]);
        }
    }
    
    __global__ void vector_erfc (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(erfc)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_erfc_inv (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(erfcinv)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_cdf_norm (const int n,
                                     const REAL* x, const int offset_x, const int stride_x,
                                     REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(normcdf)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_cdf_norm_inv (const int n,
                                         const REAL* x, const int offset_x, const int stride_x,
                                         REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(normcdfinv)(x[offset_x + gid * stride_x]);
        }
    }

    __global__ void vector_gamma (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(tgamma)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_lgamma (const int n,
                                   const REAL* x, const int offset_x, const int stride_x,
                                   REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(lgamma)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_floor (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(floor)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_ceil (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(ceil)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_trunc (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(trunc)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_round (const int n,
                                  const REAL* x, const int offset_x, const int stride_x,
                                  REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = (REAL)CAST(lrint)(x[offset_x + gid * stride_x]);
        }
    }    

    __global__ void vector_modf (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(modf)(x[offset_x + gid * stride_x], &y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_frac (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            REAL dummy;
            y[offset_y + gid * stride_y] = CAST(modf)(x[offset_x + gid * stride_x], &dummy);
        }
    }

    __global__ void vector_fmax (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(fmax)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_fmin (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(fmin)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    
    


}
