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
                                 const REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(fmax)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_fmin (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 const REAL* y, const int offset_y, const int stride_y,
                                 REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(fmin)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_copysign (const int n,
                                     const REAL* x, const int offset_x, const int stride_x,
                                     const REAL* y, const int offset_y, const int stride_y,
                                     REAL* z, const int offset_z, const int stride_z) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            z[offset_z + gid * stride_z] =
                CAST(copysign)(x[offset_x + gid * stride_x], y[offset_y + gid * stride_y]);
        }
    }

    __global__ void vector_sigmoid (const int n,
                                    const REAL* x, const int offset_x, const int stride_x,
                                    REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] =
                CAST(tanh)((REAL)0.5 * x[offset_x + gid * stride_x]) * (REAL)0.5 + (REAL)0.5;
        }
    }

    __global__ void vector_ramp (const int n,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            y[offset_y + gid * stride_y] = CAST(fmax)(x[offset_x + gid * stride_x], (REAL)0.0);
        }
    }

    __global__ void vector_relu (const int n, const REAL alpha,
                                 const REAL* x, const int offset_x, const int stride_x,
                                 REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const REAL val = x[offset_x + gid * stride_x];
            y[offset_y + gid * stride_y] = CAST(fmax)(val, alpha * val);
        }
    }

    __global__ void vector_elu (const int n, const REAL alpha,
                                const REAL* x, const int offset_x, const int stride_x,
                                REAL* y, const int offset_y, const int stride_y) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const REAL val = x[offset_x + gid * stride_x];
            y[offset_y + gid * stride_y] = CAST(fmax)(val, alpha * expm1(val));
        }
    }


    __global__ void ge_sqr (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const REAL aval = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = aval * aval;
        }
    }
    
    __global__ void ge_mul (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            const REAL* b, const int offset_b, const int ld_b,
                            REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] * b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void ge_div (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            const REAL* b, const int offset_b, const int ld_b,
                            REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] / b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void ge_inv (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = (REAL)1.0 / a[offset_a + gid_0 + gid_1 * ld_a];
        }
    }

    __global__ void ge_abs (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fabs)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_linear_frac (const int sd, const int fd,
                                    const REAL* a, const int offset_a, const int ld_a,
                                    const REAL* b, const int offset_b, const int ld_b,
                                    const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                    REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                (scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta) /
                (scaleb * b[offset_b + gid_0 + gid_1 * ld_b] + shiftb);
        }
    }

    __global__ void ge_scale_shift (const int sd, const int fd,
                                    const REAL* a, const int offset_a, const int ld_a,
                                    const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                    REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] = scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta;
                
        }
    }

    __global__ void ge_fmod (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             const REAL* b, const int offset_b, const int ld_b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmod)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_frem (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             const REAL* b, const int offset_b, const int ld_b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(remainder)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_sqrt (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sqrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_inv_sqrt (const int sd, const int fd,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(rsqrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_cbrt (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cbrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_inv_cbrt (const int sd, const int fd,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(rcbrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_pow2o3 (const int sd, const int fd,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], REAL2o3);
        }
    }

    __global__ void ge_pow3o2 (const int sd, const int fd,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], REAL3o2);
        }
    }

    __global__ void ge_pow (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            const REAL* b, const int offset_b, const int ld_b,
                            REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_powx (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             const REAL b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], b);
        }
    }

    __global__ void ge_hypot (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              const REAL* b, const int offset_b, const int ld_b,
                              REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(hypot)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_exp (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(exp)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_expm1 (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(expm1)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_log (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(log)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_log10 (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(log10)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_sin (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sin)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_cos (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cos)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_tan (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tan)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_sincos (const int sd, const int fd,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) { 
            CAST(sincos)(a[offset_a + gid_0 + gid_1 * ld_a],
                         &b[offset_b + gid_0 + gid_1 * ld_b], &c[offset_c + gid_0 + gid_1 * ld_c]);
        }
    }

    __global__ void ge_asin (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(asin)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_acos (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(acos)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_atan (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(atan)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_atan2 (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              const REAL* b, const int offset_b, const int ld_b,
                              REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(atan2)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_sinh (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sinh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_cosh (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cosh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_tanh (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tanh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_asinh (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(asinh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_acosh (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(acosh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_atanh (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(atanh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_erf (const int sd, const int fd,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erf)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_erf_inv (const int sd, const int fd,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_erfc (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfc)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_erfc_inv (const int sd, const int fd,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfcinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_cdf_norm (const int sd, const int fd,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(normcdf)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_cdf_norm_inv (const int sd, const int fd,
                                     const REAL* a, const int offset_a, const int ld_a,
                                     REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(normcdfinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_gamma (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tgamma)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_lgamma (const int sd, const int fd,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(lgamma)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_floor (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(floor)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_ceil (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(ceil)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_trunc (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(trunc)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_round (const int sd, const int fd,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(round)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void ge_modf (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(modf)(a[offset_a + gid_0 + gid_1 * ld_a], &b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_frac (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            REAL dummy;
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(modf)(a[offset_a + gid_0 + gid_1 * ld_a], &dummy);
        }
    }

    __global__ void ge_fmax (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             const REAL* b, const int offset_b, const int ld_b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmax)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_fmin (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             const REAL* b, const int offset_b, const int ld_b,
                             REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmin)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_copysign (const int sd, const int fd,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 const REAL* b, const int offset_b, const int ld_b,
                                 REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(copysign)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void ge_sigmoid (const int sd, const int fd,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] =
                CAST(tanh)((REAL)0.5 * a[offset_a + gid_0 + gid_1 * ld_a]) * (REAL)0.5 + (REAL)0.5;
        }
    }

    __global__ void ge_ramp (const int sd, const int fd,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(a[offset_a + gid_0 + gid_1 * ld_a], (REAL)0.0);
        }
    }

    __global__ void ge_relu (const int sd, const int fd,
                             const REAL alpha,
                             const REAL* a, const int offset_a, const int ld_a,
                             REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const REAL val = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(val, alpha * val);
        }
    }

    __global__ void ge_elu (const int sd, const int fd,
                            const REAL alpha,
                            const REAL* a, const int offset_a, const int ld_a,
                            REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < fd);
        if (valid) {
            const REAL val = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(val, alpha * expm1(val));
        }
    }

    __global__ void uplo_sqr (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const REAL aval = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = aval * aval;
        }
    }

    __global__ void uplo_mul (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              const REAL* b, const int offset_b, const int ld_b,
                              REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] * b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void uplo_div (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              const REAL* b, const int offset_b, const int ld_b,
                              REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                a[offset_a + gid_0 + gid_1 * ld_a] / b[offset_b + gid_0 + gid_1 * ld_b];
        }
    }

    __global__ void uplo_inv (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = (REAL)1.0 / a[offset_a + gid_0 + gid_1 * ld_a];
        }
    }

    __global__ void uplo_abs (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fabs)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }
    
    __global__ void uplo_linear_frac (const int sd, const int unit, const int bottom,
                                      const REAL* a, const int offset_a, const int ld_a,
                                      const REAL* b, const int offset_b, const int ld_b,
                                      const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                      REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                (scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta) /
                (scaleb * b[offset_b + gid_0 + gid_1 * ld_b] + shiftb);
        }
    }

    __global__ void uplo_scale_shift (const int sd, const int unit, const int bottom,
                                      const REAL* a, const int offset_a, const int ld_a,
                                      const REAL scalea, const REAL shifta, const REAL scaleb, const REAL shiftb,
                                      REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] = scalea * a[offset_a + gid_0 + gid_1 * ld_a] + shifta;
                
        }
    }

    __global__ void uplo_fmod (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               const REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmod)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_frem (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               const REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(remainder)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_sqrt (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sqrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_inv_sqrt (const int sd, const int unit, const int bottom,
                                   const REAL* a, const int offset_a, const int ld_a,
                                   REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(rsqrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_cbrt (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cbrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_inv_cbrt (const int sd, const int unit, const int bottom,
                                   const REAL* a, const int offset_a, const int ld_a,
                                   REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(rcbrt)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_pow2o3 (const int sd, const int unit, const int bottom,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], REAL2o3);
        }
    }

    __global__ void uplo_pow3o2 (const int sd, const int unit, const int bottom,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], REAL3o2);
        }
    }

    __global__ void uplo_pow (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              const REAL* b, const int offset_b, const int ld_b,
                              REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_powx (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               const REAL b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] = CAST(pow)(a[offset_a + gid_0 + gid_1 * ld_a], b);
        }
    }

    __global__ void uplo_hypot (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                const REAL* b, const int offset_b, const int ld_b,
                                REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(hypot)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_exp (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(exp)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_expm1 (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(expm1)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_log (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(log)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_log10 (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(log10)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_sin (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sin)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_cos (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cos)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_tan (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tan)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_sincos (const int sd, const int unit, const int bottom,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b,
                                 REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            CAST(sincos)(a[offset_a + gid_0 + gid_1 * ld_a],
                         &b[offset_b + gid_0 + gid_1 * ld_b], &c[offset_c + gid_0 + gid_1 * ld_c]);
        }
    }

    __global__ void uplo_asin (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(asin)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_acos (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(acos)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_atan (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(atan)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_atan2 (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                const REAL* b, const int offset_b, const int ld_b,
                                REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(atan2)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_sinh (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(sinh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_cosh (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(cosh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_tanh (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tanh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_asinh (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(asinh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_acosh (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(acosh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_atanh (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(atanh)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_erf (const int sd, const int unit, const int bottom,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erf)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_erf_inv (const int sd, const int unit, const int bottom,
                                  const REAL* a, const int offset_a, const int ld_a,
                                  REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_erfc (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfc)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_erfc_inv (const int sd, const int unit, const int bottom,
                                   const REAL* a, const int offset_a, const int ld_a,
                                   REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(erfcinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_cdf_norm (const int sd, const int unit, const int bottom,
                                   const REAL* a, const int offset_a, const int ld_a,
                                   REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(normcdf)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_cdf_norm_inv (const int sd, const int unit, const int bottom,
                                       const REAL* a, const int offset_a, const int ld_a,
                                       REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(normcdfinv)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_gamma (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(tgamma)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_lgamma (const int sd, const int unit, const int bottom,
                                 const REAL* a, const int offset_a, const int ld_a,
                                 REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(lgamma)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_floor (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(floor)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_ceil (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(ceil)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_trunc (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(trunc)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_round (const int sd, const int unit, const int bottom,
                                const REAL* a, const int offset_a, const int ld_a,
                                REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(round)(a[offset_a + gid_0 + gid_1 * ld_a]);
        }
    }

    __global__ void uplo_modf (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(modf)(a[offset_a + gid_0 + gid_1 * ld_a], &b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_frac (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            REAL dummy;
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(modf)(a[offset_a + gid_0 + gid_1 * ld_a], &dummy);
        }
    }

    __global__ void uplo_fmax (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               const REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmax)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_fmin (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               const REAL* b, const int offset_b, const int ld_b,
                               REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(fmin)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_copysign (const int sd, const int unit, const int bottom,
                                   const REAL* a, const int offset_a, const int ld_a,
                                   const REAL* b, const int offset_b, const int ld_b,
                                   REAL* c, const int offset_c, const int ld_c) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            c[offset_c + gid_0 + gid_1 * ld_c] =
                CAST(copysign)(a[offset_a + gid_0 + gid_1 * ld_a], b[offset_b + gid_0 + gid_1 * ld_b]);
        }
    }

    __global__ void uplo_sigmoid (const int sd, const int unit, const int bottom,
                                  const REAL* a, const int offset_a, const int ld_a,
                                  REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] =
                CAST(tanh)((REAL)0.5 * a[offset_a + gid_0 + gid_1 * ld_a]) * (REAL)0.5 + (REAL)0.5;
        }
    }

    __global__ void uplo_ramp (const int sd, const int unit, const int bottom,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(a[offset_a + gid_0 + gid_1 * ld_a], (REAL)0.0);
        }
    }

    __global__ void uplo_relu (const int sd, const int unit, const int bottom,
                               const REAL alpha,
                               const REAL* a, const int offset_a, const int ld_a,
                               REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const REAL val = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(val, alpha * val);
        }
    }

    __global__ void uplo_elu (const int sd, const int unit, const int bottom,
                              const REAL alpha,
                              const REAL* a, const int offset_a, const int ld_a,
                              REAL* b, const int offset_b, const int ld_b) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < sd) && (gid_1 < sd);
        const bool check = valid &&
            ((unit == 132) ? bottom * gid_0 > bottom * gid_1 : bottom * gid_0 >= bottom * gid_1);
        if (check) {
            const REAL val = a[offset_a + gid_0 + gid_1 * ld_a];
            b[offset_b + gid_0 + gid_1 * ld_b] = CAST(fmax)(val, alpha * expm1(val));
        }
    }

}
