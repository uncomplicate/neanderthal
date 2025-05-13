;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release view]]
             [utils :refer [dragan-says-ex with-check generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap! extract]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [pointer long-ptr long-pointer float-pointer double-pointer byte-pointer]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols matrix-type entry symmetric?]]
             [real :as real]
             [integer :as integer]
             [math :refer [f=] :as math]
             [block :refer [stride contiguous?]]]
            [uncomplicate.neanderthal.internal
             [constants :refer :all]
             [api :refer :all]
             [navigation :refer [full-storage accu-layout dostripe-layout]]
             [common :refer [check-stride check-eq-navigators flip-uplo real-accessor]]]
            [uncomplicate.neanderthal.internal.cpp
             [common :refer :all]
             [structures :refer :all]
             [blas :refer :all]
             [lapack :refer :all]
             [factory :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.mkl
             [constants :refer [mkl-sparse-request]]
             [core :refer [malloc! free! mkl-sparse sparse-matrix mkl-sparse-copy sparse-error]]
             [structures :refer [ge-csr-matrix spmat descr sparse-transpose sparse-layout csr-ge-sp2m]]])
  (:import java.nio.ByteBuffer
           [uncomplicate.neanderthal.internal.api DataAccessor Vector LayoutNavigator Region
            GEMatrix UploMatrix DenseStorage]
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer BytePointer]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$VSLStreamStatePtr]))

;; =============== Factories ==================================================

(defn math
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (math "v" type name)))

;; ================= MKL specific blas and lapack functions =====================

(defmacro mkl-gd-sv [mkl vdiv-method sv-method ptr a b]
  `(let [n-a# (ncols ~a)
         n-b# (ncols ~b)
         nav-b# (navigator ~b)
         stor-b# (storage ~b)
         buff-a# (~ptr ~a)
         buff-b# (~ptr ~b 0)
         strd-b# (stride ~b)]
     (if (.isColumnMajor nav-b#)
       (dotimes [j# n-b#]
         (. ~mkl ~vdiv-method n-a# (.position buff-b# (.index nav-b# stor-b# 0 j#)) buff-a#
            (.position buff-b# (.index nav-b# stor-b# 0 j#))))
       (dotimes [j# n-b#]
         (. ~mkl ~sv-method ~(:row blas-layout) ~(:lower blas-uplo) ~(:no-trans blas-transpose)
            ~(:non-unit blas-diag) n-a# 0 buff-a# 1
            (.position buff-b# (.index nav-b# stor-b# 0 j#)) strd-b#)))
     ~b))

;; ================= Real Vector Engines ========================================

(defmacro vector-math
  ([method ptr a y]
   `(do
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (stride ~a) (~ptr ~y) (stride ~y))
      ~y))
  ([method ptr a b y]
   `(do
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) (~ptr ~y) (stride ~y))
      ~y)))

(defmacro vector-linear-frac [method ptr a b scalea shifta scaleb shiftb y]
  `(do
     (. mkl_rt ~method (dim ~a) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) ~scalea ~shifta ~scaleb ~shiftb (~ptr ~y) (stride ~y))
     ~y))

(defmacro vector-powx [method ptr a b y]
  `(do
     (. mkl_rt ~method (dim ~a) (~ptr ~a) (stride ~a) ~b (~ptr ~y) (stride ~y))
     ~y))

(defmacro vector-ramp [t ptr a zero y]
  `(. mkl_rt ~(math t 'FmaxI) (dim ~a) (~ptr ~a) (stride ~a) (~ptr ~zero) 0 (~ptr ~y) (stride ~y)))

(defmacro vector-relu [t ptr alpha a zero y]
  `(if (identical? ~a ~y)
     (dragan-says-ex "MKL ReLU requires distinct arguments a and y.")
     (with-release [temp# (raw ~y)]
       (let [a# (~ptr ~a)
             n# (dim ~a)
             strd-a# (stride ~a)
             y# (~ptr ~y)
             strd-y# (stride ~y)
             zero# (~ptr ~zero)]
         (. mkl_rt ~(math t 'FminI) n# a# strd-a# zero# 0 y# strd-y#)
         (. mkl_rt ~(math t 'MulI) n# (~ptr ~alpha) (stride ~alpha) y# strd-y# y# strd-y#)
         (. mkl_rt ~(math t 'FmaxI) n# a# strd-a# zero# 0 (~ptr temp#) 1)
         (. mkl_rt ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
         ~y))))

(defmacro vector-elu
  ([t ptr alpha a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "MKL ELU requires distinct arguments a and y.")
      (with-release [temp# (raw ~a)]
        (let [a# (~ptr ~a)
              n# (dim ~a)
              strd-a# (stride ~a)
              y# (~ptr ~y)
              strd-y# (stride ~y)
              zero# (~ptr ~zero)]
          (. mkl_rt ~(math t 'FminI) n# a# strd-a# zero# 0 y# strd-y#)
          (. mkl_rt ~(math t 'Expm1I) n# y# strd-y# y# strd-y#)
          (. mkl_rt ~(math t 'MulI) n# (~ptr ~alpha) (stride ~alpha) y# strd-y# y# strd-y#)
          (. mkl_rt ~(math t 'FmaxI) n# a# strd-a# zero# 0 (~ptr temp#) 1)
          (. mkl_rt ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
          ~y))))
  ([t ptr a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "MKL ELU requires distinct arguments a and y.")
      (with-release [temp# (raw ~a)]
        (let [a# (~ptr ~a)
              n# (dim ~a)
              strd-a# (stride ~a)
              y# (~ptr ~y)
              strd-y# (stride ~y)
              zero# (~ptr ~zero)]
          (. mkl_rt ~(math t 'FminI) n# a# strd-a# zero# 0 y# strd-y#)
          (. mkl_rt ~(math t 'Expm1I) n# y# strd-y# y# strd-y#)
          (. mkl_rt ~(math t 'FmaxI) n# a# strd-a# zero# 0 (~ptr temp#) 1)
          (. mkl_rt ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
          ~y)))))

;; ============ Delegate math functions  ============================================

(defn sigmoid-over-tanh [eng a y]
  (when-not (identical? a y) (copy eng a y))
  (linear-frac eng (tanh eng (scal eng 0.5 y) y) a 0.5 0.5 0.0 1.0 y))

(defmacro with-mkl-check [expr res]
  `(let [err# ~expr]
     (if (zero? err#)
       ~res
       (throw (ex-info "MKL error." {:error-code err#})))))

(defn create-stream-ars5 ^mkl_rt$VSLStreamStatePtr [seed]
  (let-release [stream (mkl_rt$VSLStreamStatePtr. (long-pointer 1))]
    (with-mkl-check (mkl_rt/vslNewStream stream mkl_rt/VSL_BRNG_ARS5 seed)
      stream)))

(def ^:private default-rng-stream (create-stream-ars5 (generate-seed)))

(defmacro real-math* [name t ptr cast
                      vector-math vector-linear-frac vector-powx
                      vector-ramp vector-relu vector-elu zero]
  `(extend-type ~name
     VectorMath
     (sqr [_# a# y#]
       (~vector-math ~(math t 'SqrI) ~ptr a# y#))
     (mul [_# a# b# y#]
       (~vector-math ~(math t 'MulI) ~ptr a# b# y#))
     (div [_# a# b# y#]
       (~vector-math ~(math t 'DivI) ~ptr a# b# y#))
     (inv [_# a# y#]
       (~vector-math ~(math t 'InvI) ~ptr a# y#))
     (abs [_# a# y#]
       (~vector-math ~(math t 'AbsI) ~ptr a# y#))
     (linear-frac [_# a# b# scalea# shifta# scaleb# shiftb# y#]
       (~vector-linear-frac ~(math t 'LinearFracI) ~ptr a# b#
        (~cast scalea#) (~cast shifta#) (~cast scaleb#) (~cast shiftb#) y#))
     (fmod [_# a# b# y#]
       (~vector-math ~(math t 'FmodI) ~ptr a# b# y#))
     (frem [_# a# b# y#]
       (~vector-math  ~(math t 'RemainderI) ~ptr a# b# y#))
     (sqrt [_# a# y#]
       (~vector-math ~(math t 'SqrtI) ~ptr a# y#))
     (inv-sqrt [_# a# y#]
       (~vector-math ~(math t 'InvSqrtI) ~ptr a# y#))
     (cbrt [_# a# y#]
       (~vector-math ~(math t 'CbrtI) ~ptr a# y#))
     (inv-cbrt [_# a# y#]
       (~vector-math ~(math t 'InvCbrtI) ~ptr a# y#))
     (pow2o3 [_# a# y#]
       (~vector-math ~(math t 'Pow2o3I) ~ptr a# y#))
     (pow3o2 [_# a# y#]
       (~vector-math ~(math t 'Pow3o2I) ~ptr a# y#))
     (pow [_# a# b# y#]
       (~vector-math ~(math t 'PowI) ~ptr a# b# y#))
     (powx [_# a# b# y#]
       (~vector-powx ~(math t 'PowxI) ~ptr a# (~cast b#) y#))
     (hypot [_# a# b# y#]
       (~vector-math ~(math t 'HypotI) ~ptr a# b# y#))
     (exp [_# a# y#]
       (~vector-math ~(math t 'ExpI) ~ptr a# y#))
     (exp2 [_# a# y#]
       (~vector-math ~(math t 'Exp2I) ~ptr a# y#))
     (exp10 [_# a# y#]
       (~vector-math ~(math t 'Exp10I) ~ptr a# y#))
     (expm1 [_# a# y#]
       (~vector-math ~(math t 'Expm1I) ~ptr a# y#))
     (log [_# a# y#]
       (~vector-math ~(math t 'LnI) ~ptr a# y#))
     (log2 [_# a# y#]
       (~vector-math ~(math t 'Log2I) ~ptr a# y#))
     (log10 [_# a# y#]
       (~vector-math ~(math t 'Log10I) ~ptr a# y#))
     (log1p [_# a# y#]
       (~vector-math ~(math t 'Log1pI) ~ptr a# y#))
     (sin [_# a# y#]
       (~vector-math ~(math t 'SinI) ~ptr a# y#))
     (cos [_# a# y#]
       (~vector-math ~(math t 'CosI) ~ptr a# y#))
     (tan [_# a# y#]
       (~vector-math ~(math t 'TanI) ~ptr a# y#))
     (sincos [_# a# y# z#]
       (~vector-math ~(math t 'SinCosI) ~ptr a# y# z#))
     (asin [_# a# y#]
       (~vector-math ~(math t 'AsinI) ~ptr a# y#))
     (acos [_# a# y#]
       (~vector-math ~(math t 'AcosI) ~ptr a# y#))
     (atan [_# a# y#]
       (~vector-math ~(math t 'AtanI) ~ptr a# y#))
     (atan2 [_# a# b# y#]
       (~vector-math ~(math t 'Atan2I) ~ptr a# b# y#))
     (sinh [_# a# y#]
       (~vector-math ~(math t 'SinhI) ~ptr a# y#))
     (cosh [_# a# y#]
       (~vector-math ~(math t 'CoshI) ~ptr a# y#))
     (tanh [_# a# y#]
       (~vector-math ~(math t 'TanhI) ~ptr a# y#))
     (asinh [_# a# y#]
       (~vector-math ~(math t 'AsinhI) ~ptr a# y#))
     (acosh [_# a# y#]
       (~vector-math ~(math t 'AcoshI) ~ptr a# y#))
     (atanh [_# a# y#]
       (~vector-math ~(math t 'AtanhI) ~ptr a# y#))
     (erf [_# a# y#]
       (~vector-math ~(math t 'ErfI) ~ptr a# y#))
     (erfc [_# a# y#]
       (~vector-math ~(math t 'ErfcI) ~ptr a# y#))
     (erf-inv [_# a# y#]
       (~vector-math ~(math t 'ErfInvI) ~ptr a# y#))
     (erfc-inv [_# a# y#]
       (~vector-math ~(math t 'ErfcInvI) ~ptr a# y#))
     (cdf-norm [_# a# y#]
       (~vector-math ~(math t 'CdfNormI) ~ptr a# y#))
     (cdf-norm-inv [_# a# y#]
       (~vector-math ~(math t 'CdfNormInvI) ~ptr a# y#))
     (gamma [_# a# y#]
       (~vector-math ~(math t 'TGammaI) ~ptr a# y#))
     (lgamma [_# a# y#]
       (~vector-math ~(math t 'LGammaI) ~ptr a# y#))
     (expint1 [_# a# y#]
       (~vector-math ~(math t 'ExpInt1I) ~ptr a# y#))
     (floor [_# a# y#]
       (~vector-math ~(math t 'FloorI) ~ptr a# y#))
     (fceil [_# a# y#]
       (~vector-math ~(math t 'CeilI) ~ptr a# y#))
     (trunc [_# a# y#]
       (~vector-math ~(math t 'TruncI) ~ptr a# y#))
     (round [_# a# y#]
       (~vector-math ~(math t 'RoundI) ~ptr a# y#))
     (modf [_# a# y# z#]
       (~vector-math ~(math t 'ModfI) ~ptr a# y# z#))
     (frac [_# a# y#]
       (~vector-math ~(math t 'FracI) ~ptr a# y#))
     (fmin [_# a# b# y#]
       (~vector-math ~(math t 'FminI) ~ptr a# b# y#))
     (fmax [_# a# b# y#]
       (~vector-math ~(math t 'FmaxI) ~ptr a# b# y#))
     (copy-sign [_# a# b# y#]
       (~vector-math ~(math t 'CopySignI) ~ptr a# b# y#))
     (sigmoid [this# a# y#]
       (sigmoid-over-tanh this# a# y#))
     (ramp [this# a# y#]
       (~vector-ramp ~t ~ptr a# ~zero y#))
     (relu [this# alpha# a# y#]
       (~vector-relu ~t ~ptr alpha# a# ~zero y#))
     (elu
       ([this# alpha# a# y#]
        (~vector-elu ~t ~ptr alpha# a# ~zero y#))
       ([this# a# y#]
        (~vector-elu ~t ~ptr a# ~zero y#)))))

(defmacro real-vector-math* [name t ptr cast zero]
  `(real-math* ~name ~t ~ptr ~cast vector-math vector-linear-frac vector-powx vector-ramp vector-relu vector-elu ~zero))

(defn cast-stream ^mkl_rt$VSLStreamStatePtr [stream]
  (or stream default-rng-stream))

(defmacro mkl-real-vector-rng* [name t ptr cast]
  `(extend-type ~name
     RandomNumberGenerator
     (rand-uniform [_# rng-stream# lower# upper# x#]
       (with-rng-check x#
         (. mkl_rt ~(cblas "v" t 'RngUniform) mkl_rt/VSL_RNG_METHOD_UNIFORM_STD
            (cast-stream rng-stream#) (dim x#) (~ptr x#) (~cast lower#) (~cast upper#)))
       x#)
     (rand-normal [_# rng-stream# mu# sigma# x#]
       (with-rng-check x#
         (. mkl_rt ~(cblas "v" t 'RngGaussian) mkl_rt/VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
            (cast-stream rng-stream#) (dim x#) (~ptr x#) (~cast mu#) (~cast sigma#)))
       x#)))

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt)
(real-vector-blas-plus* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(real-vector-lapack* FloatVectorEngine "s" float-ptr float mkl_rt)
(real-vector-math* FloatVectorEngine "s" float-ptr float zero-float)
(mkl-real-vector-rng* FloatVectorEngine "s" float-ptr float)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt)
(real-vector-blas-plus* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(real-vector-lapack* DoubleVectorEngine "d" double-ptr double mkl_rt)
(real-vector-math* DoubleVectorEngine "d" double-ptr double zero-double)
(mkl-real-vector-rng* DoubleVectorEngine "d" double-ptr double)

(deftype LongVectorEngine [])
(integer-vector-blas* LongVectorEngine "d" double-ptr mkl_rt 1)
(integer-vector-blas-plus* LongVectorEngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntVectorEngine [])
(integer-vector-blas* IntVectorEngine "s" float-ptr mkl_rt 1)
(integer-vector-blas-plus* IntVectorEngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortVectorEngine [])
(integer-vector-blas* ShortVectorEngine "s" float-ptr mkl_rt 2)
(integer-vector-blas-plus* ShortVectorEngine "s" float-ptr short-float mkl_rt mkl_rt 2)

(deftype ByteVectorEngine [])
(integer-vector-blas* ByteVectorEngine "s" float-ptr mkl_rt 4)
(integer-vector-blas-plus* ByteVectorEngine "s" float-ptr byte-float mkl_rt mkl_rt 4)

;; ================= Integer GE Engine ========================================

(defmacro mkl-patch-ge-copy [chunk blas method ptr a b]
  (if (= 1 chunk)
    `(when (< 0 (dim ~a))
       (let [stor-b# (full-storage ~b)
             no-trans# (= (navigator ~a) (navigator ~b))]
         (. ~blas ~method ~(byte (int \C)) (byte (int (if no-trans# \N \T)))
            (if no-trans# (.sd stor-b#) (.fd stor-b#)) (if no-trans# (.fd stor-b#) (.sd stor-b#))
            1.0 (~ptr ~a) (stride ~a) (~ptr ~b) (.ld stor-b#))))
    `(if (or (and (.isGapless (storage ~a)) (= 0 (rem (dim ~a)) ~chunk))
             (and (= 0 (rem (mrows ~a) ~chunk)) (= 0 (rem (ncols ~a) ~chunk))))
       (let [stor-b# (full-storage ~b)
             no-trans# (= (navigator ~a) (navigator ~b))]
         (. ~blas ~method ~(byte (int \C)) (byte (int (if no-trans# \N \T)))
            (quot (if no-trans# (.sd stor-b#) (.fd stor-b#)) ~chunk)
            (quot (if no-trans# (.fd stor-b#) (.sd stor-b#)) ~chunk)
            1.0 (~ptr ~a) (quot (stride ~a) ~chunk) (~ptr ~b) (quot (.ld stor-b#) ~chunk)))
       (dragan-says-ex SHORT_UNSUPPORTED_MSG {:mrows (mrows ~a) :ncols (ncols ~a)}))))

(defmacro mkl-integer-ge-blas* [name t ptr blas chunk]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (mkl-patch-ge-copy ~chunk ~blas ~(cblas "mkl_" t 'omatcopy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm1 [_# a#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm2 [_# a#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrmi [_# a#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (asum [_# a#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (scal [_# alpha# a#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (axpy [_# alpha# a# b#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (mv
       ([_# alpha# a# x# beta# y#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
       ([_# a# _#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))
     (rk
       ([_# alpha# x# y# a#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
       ([_# _# _# a#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))
     (mm
       ([_# alpha# a# b# _#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
       ([_# alpha# a# b# beta# c# _#]
        (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))))

;; ================= Real GE Engine ========================================

(defmacro mkl-ge-axpby [blas method ptr alpha a beta b]
  `(if (< 0 (dim ~a))
     (let [nav-b# (navigator ~b)]
       (. ~blas ~method (byte (int (if (.isColumnMajor nav-b#) \C \R)))
          (byte (int (if (= (navigator ~a) nav-b#) \n \t))) ~(byte (int \n)) (mrows ~b) (ncols ~b)
          ~alpha (~ptr ~a) (stride ~a) ~beta (~ptr ~b) (stride ~b)
          (~ptr ~b) (stride ~b))
       ~b)
     ~b))

(defmacro mkl-real-ge-blas* [name t ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (when (< 0 (dim a#))
         (let [stor-b# (full-storage b#)
               no-trans# (= (navigator a#) (navigator b#))]
           (. ~blas ~(cblas "mkl_" t 'omatcopy) ~(byte (int \C)) (byte (int (if no-trans# \N \T)))
              (if no-trans# (.sd stor-b#) (.fd stor-b#)) (if no-trans# (.fd stor-b#) (.sd stor-b#))
              1.0 (~ptr a#) (stride a#) (~ptr b#) (.ld stor-b#))))
       b#)
     (dot [_# a# b#]
       (ge-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr \O a#))
     (nrm2 [_# a#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr \F a#))
     (nrmi [_# a#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr \I a#))
     (asum [_# a#]
       (ge-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (when (< 0 (dim a#))
         (let [stor# (full-storage a#)]
           (. ~blas ~(cblas "mkl_" t 'imatcopy) ~(byte (int \c)) ~(byte (int \n))
              (.sd stor#) (.fd stor#) (~cast alpha#) (~ptr a#) (.ld stor#) (.ld stor#))))
       a#)
     (axpy [_# alpha# a# b#]
       (mkl-ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# 1.0 b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (. ~blas ~(cblas t 'gemv) (.layout (navigator a#)) ~(:no-trans blas-transpose) (mrows a#) (ncols a#)
           (~cast alpha#) (~ptr a#) (stride a#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
        y#)
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for GE matrices." {:a (info a#)})))
     (rk
       ([_# alpha# x# y# a#]
        (. ~blas ~(cblas t 'ger) (.layout (navigator a#)) (mrows a#) (ncols a#)
           (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr a#) (stride a#))
        a#)
       ([_# _# _# a#]
        (dragan-says-ex "In-place rk! is not supported for GE matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# _#]
        (if-not (instance? GEMatrix b#)
          (mm (engine b#) alpha# b# a# false)
          (dragan-says-ex "In-place mm! is not supported for GE matrices. Use QR factorization."
                          {:a (info a#) :b (info b#)} )))
       ([_# alpha# a# b# beta# c# _#]
        (if (instance? GEMatrix b#)
          (let [nav# (navigator c#)]
            (. ~blas ~(cblas t 'gemm) (.layout nav#)
               (if (= nav# (navigator a#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
               (if (= nav# (navigator b#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
               (mrows a#) (ncols b#) (ncols a#) (~cast alpha#) (~ptr a#) (stride a#)
               (~ptr b#) (stride b#) (~cast beta#) (~ptr c#) (stride c#))
            c#)
          (mm (engine b#) (~cast alpha#) b# a# (~cast beta#) c# false))))))

(defmacro mkl-real-ge-blas-plus* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (ge-lan ~lapack ~(lapacke t 'lange) ~ptr \M a#))
     (sum [_# a#]
       (ge-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (with-lapack-check "laset"
         (. ~lapack ~(lapacke t 'laset) (.layout (navigator a#)) ~(byte (int \g))
            (mrows a#) (ncols a#) (~cast alpha#) (~cast alpha#) (~ptr a#) (stride a#)))
       a#)
     (axpby [_# alpha# a# beta# b#]
       (mkl-ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# (~cast beta#) b#)
       b#)
     (trans [_# a#]
       (when (< 0 (dim a#))
         (let [stor# (full-storage a#)]
           (if (.isGapless stor#)
             (. ~blas ~(cblas "mkl_" t 'imatcopy) ~(byte (int \c)) ~(byte (int \t))
                (.sd stor#) (.fd stor#) (~cast 1.0) (~ptr a#) (.ld stor#) (.fd stor#))
             (dragan-says-ex "You can not hard-transpose the content of a matrix with a gap in memory. Sorry."
                             {:a (info a#)}))))
       a#)))

(defmacro matrix-math
  ([method ptr a y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~y))]
          (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                            (. mkl_rt ~method surface# buff-a# 1 buff-y# 1)
                            (. mkl_rt ~method len# buff-a# ld-a# buff-y# 1))))
      ~y))
  ([method ptr a b y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-b# (~ptr ~b 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~y))]
          (full-storage-map ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
                            (. mkl_rt ~method surface# buff-a# 1 buff-b# 1 buff-y# 1)
                            (. mkl_rt ~method len# buff-a# ld-a# buff-b# ld-b# buff-y# 1))))
      ~y)))

(defmacro matrix-powx [method ptr a b y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             surface# (.surface (region ~y))]
         (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                           (. mkl_rt ~method surface# buff-a# 1 ~b buff-y# 1)
                           (. mkl_rt ~method len# buff-a# ld-a# ~b buff-y# 1))))
     ~y))

(defmacro matrix-linear-frac [method ptr a b scalea shifta scaleb shiftb y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)
             buff-y# (~ptr ~y 0)
             surface# (.surface (region ~y))]
         (full-storage-map
          ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
          (. mkl_rt ~method surface# buff-a# 1 buff-b# 1 ~scalea ~shifta ~scaleb ~shiftb buff-y# 1)
          (. mkl_rt ~method len# buff-a# ld-a# buff-b# ld-b# ~scalea ~shifta ~scaleb ~shiftb buff-y# 1))))
     ~y))

(defmacro matrix-ramp [t ptr a zero y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             zero# (~ptr ~zero)
             surface# (.surface (region ~y))]
         (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                           (. mkl_rt ~(math t 'FmaxI) surface# buff-a# 1 zero# 0 buff-y# 1)
                           (. mkl_rt ~(math t 'FmaxI) len# buff-a# ld-a# zero# 0 buff-y# 1))))
     ~y))

(defmacro matrix-relu [t ptr alpha a zero y]
  `(if (identical? ~a ~y)
     (dragan-says-ex "MKL ReLU requires distinct arguments a and y.")
     (do
       (when (< 0 (dim ~a))
         (with-release [temp-stripe# (delay (raw (.stripe (navigator ~y) ~y 0)))]
           (let [buff-a# (~ptr ~a 0)
                 buff-y# (~ptr ~y 0)
                 buff-alpha# (~ptr ~alpha ~0)
                 zero# (~ptr ~zero)
                 surface# (.surface (region ~y))]
             (full-storage-map ~alpha ~a ~y len# buff-alpha# buff-a# buff-y# ld-alpha# ld-a#
                               (with-release [temp# (raw ~y)]
                                 (. mkl_rt ~(math t 'FminI) surface# buff-a# 1 zero# 0 buff-y# 1)
                                 (. mkl_rt ~(math t 'MulI) surface# buff-alpha# 1 buff-y# 1 buff-y# 1)
                                 (. mkl_rt ~(math t 'FmaxI) surface# buff-a# 1 zero# 0 (~ptr temp#) 1)
                                 (. mkl_rt ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                               (let [temp# (~ptr (deref temp-stripe#))]
                                 (. mkl_rt ~(math t 'FminI) len# buff-a# ld-a# zero# 0 buff-y# 1)
                                 (. mkl_rt ~(math t 'MulI) len# buff-alpha# ld-alpha# buff-y# 1 buff-y# 1)
                                 (. mkl_rt ~(math t 'FmaxI) len# buff-a# ld-a# zero# 0 temp# 1)
                                 (. mkl_rt ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1))))))
       ~y)))

(defmacro matrix-elu
  ([t ptr alpha a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "MKL ELU requires distinct arguments a and y.")
      (do
        (when (< 0 (dim ~a))
          (with-release [temp-stripe# (delay (raw (.stripe (navigator ~y) ~y 0)))]
            (let [buff-a# (~ptr ~a 0)
                  buff-y# (~ptr ~y 0)
                  buff-alpha# (~ptr ~alpha ~0)
                  zero# (~ptr ~zero)
                  surface# (.surface (region ~y))]
              (full-storage-map ~alpha ~a ~y len# buff-alpha# buff-a# buff-y# ld-alpha# ld-a#
                                (with-release [temp# (raw ~y)]
                                  (. mkl_rt ~(math t 'FminI) surface# buff-a# 1 zero# 0 buff-y# 1)
                                  (. mkl_rt ~(math t 'Expm1I) surface# buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'MulI) surface# buff-alpha# 1 buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'FmaxI) surface# buff-a# 1 zero# 0 (~ptr temp#) 1)
                                  (. mkl_rt ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                                (let [temp# (~ptr (deref temp-stripe#))]
                                  (. mkl_rt ~(math t 'FminI) len# buff-a# ld-a# zero# 0 buff-y# 1)
                                  (. mkl_rt ~(math t 'Expm1I) len# buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'MulI) len# buff-alpha# ld-alpha# buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'FmaxI) len# buff-a# ld-a# zero# 0 temp# 1)
                                  (. mkl_rt ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1))))))
        ~y)))
  ([t ptr a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "MKL ELU requires distinct arguments a and y.")
      (do
        (when (< 0 (dim ~a))
          (with-release [temp-stripe# (delay (raw (.stripe (navigator ~y) ~y 0)))]
            (let [buff-a# (~ptr ~a 0)
                  buff-y# (~ptr ~y 0)
                  zero# (~ptr ~zero)
                  surface# (.surface (region ~y))]
              (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                                (with-release [temp# (raw ~y)]
                                  (. mkl_rt ~(math t 'FminI) surface# buff-a# 1 zero# 0 buff-y# 1)
                                  (. mkl_rt ~(math t 'Expm1I) surface# buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'FmaxI) surface# buff-a# 1 zero# 0 (~ptr temp#) 1)
                                  (. mkl_rt ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                                (let [temp# (~ptr (deref temp-stripe#))]
                                  (. mkl_rt ~(math t 'FminI) len# buff-a# ld-a# zero# 0 buff-y# 1)
                                  (. mkl_rt ~(math t 'Expm1I) len# buff-y# 1 buff-y# 1)
                                  (. mkl_rt ~(math t 'FmaxI) len# buff-a# ld-a# zero# 0 temp# 1)
                                  (. mkl_rt ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1))))))
        ~y))))

(defmacro real-matrix-math* [name t ptr cast zero]
  `(real-math* ~name ~t ~ptr ~cast matrix-math matrix-linear-frac matrix-powx matrix-ramp matrix-relu matrix-elu ~zero))

(defmacro matrix-rng [method ptr rng-method rng-stream a par1 par2]
  `(do
     (when (< 0 (dim ~a))
       (if (.isGapless (storage ~a))
         (with-mkl-check
           (. mkl_rt ~method ~rng-method ~rng-stream (dim ~a) (~ptr ~a) ~par1 ~par2)
           ~a)
         (let [buff# (~ptr ~a 0)]
           (dostripe-layout ~a len# idx#
                            (with-rng-check ~a
                              (. mkl_rt ~method ~rng-method ~rng-stream
                                 len# (.position buff# idx#) ~par1 ~par2))))))
     ~a))

(defmacro mkl-real-ge-rng* [name t ptr cast]
  `(extend-type ~name
     RandomNumberGenerator
     (rand-uniform [_# rng-stream# lower# upper# a#]
       (matrix-rng ~(cblas "v" t 'RngUniform) ~ptr mkl_rt/VSL_RNG_METHOD_UNIFORM_STD
                   (cast-stream rng-stream#) a# (~cast lower#) (~cast upper#)))
     (rand-normal [_# rng-stream# mu# sigma# a#]
       (matrix-rng ~(cblas "v" t 'RngGaussian) ~ptr mkl_rt/VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
                   (cast-stream rng-stream#) a# (~cast mu#) (~cast sigma#)))))

(deftype FloatGEEngine [])
(mkl-real-ge-blas* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt)
(mkl-real-ge-blas-plus* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt ge-zero-float)
(real-matrix-math* FloatGEEngine "s" float-ptr float zero-float)
(mkl-real-ge-rng* FloatGEEngine "s" float-ptr float)

(deftype DoubleGEEngine [])
(mkl-real-ge-blas* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt)
(mkl-real-ge-blas-plus* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt ge-zero-double)
(real-matrix-math* DoubleGEEngine "d" double-ptr double zero-double)
(mkl-real-ge-rng* DoubleGEEngine "d" double-ptr double)

;;TODO
(deftype LongGEEngine [])
(mkl-integer-ge-blas* LongGEEngine "d" double-ptr mkl_rt 1)

(deftype IntGEEngine [])
(mkl-integer-ge-blas* IntGEEngine "s" float-ptr mkl_rt 1)

(deftype ShortGEEngine []) ;; TODO

(deftype ByteGEEngine []) ;; TODO

;; ========================= TR matrix engines ===============================================

(deftype FloatTREngine [])
(real-tr-blas* FloatTREngine "s" float-ptr float mkl_rt mkl_rt)
(real-tr-blas-plus* FloatTREngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(real-tr-lapack* FloatTREngine "s" float-ptr cpp/float-pointer float mkl_rt mkl_rt)
(real-matrix-math* FloatTREngine "s" float-ptr float ge-zero-float)

(deftype DoubleTREngine [])
(real-tr-blas* DoubleTREngine "d" double-ptr double mkl_rt mkl_rt)
(real-tr-blas-plus* DoubleTREngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby"  ones-double)
(real-tr-lapack* DoubleTREngine "d" double-ptr cpp/double-pointer double mkl_rt mkl_rt)
(real-matrix-math* DoubleTREngine "d" double-ptr double zero-double)

(deftype LongTREngine [])
;;(integer-tr-blas* LongTREngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntTREngine [])
;;(integer-tr-blas* IntTREngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortTREngine []) ;; TODO

(deftype ByteTREngine []) ;; TODO

;; ========================= SY matrix engines ===============================================

(deftype FloatSYEngine [])
(real-sy-blas* FloatSYEngine "s" float-ptr float mkl_rt mkl_rt)
(real-sy-blas-plus* FloatSYEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(real-sy-lapack* FloatSYEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt ge-zero-float)
(real-matrix-math* FloatSYEngine "s" float-ptr float zero-float)

(deftype DoubleSYEngine [])
(real-sy-blas* DoubleSYEngine "d" double-ptr double mkl_rt mkl_rt)
(real-sy-blas-plus* DoubleSYEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(real-sy-lapack* DoubleSYEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt ge-zero-double)
(real-matrix-math* DoubleSYEngine "d" double-ptr double zero-double)

;;TODO
(deftype LongSYEngine [])
;;(integer-tr-blas* LongSYEngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntSYEngine [])
;;(integer-tr-blas* IntSYEngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortSYEngine []);; TODO

(deftype ByteSYEngine []);; TODO

;; ============================ GB matrix engines ==================================================

(defmacro mkl-real-gb-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (gb-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (gb-map ~lapack ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (gb-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (gb-lan ~lapack ~(lapacke "" t 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (gb-lan ~lapack ~(lapacke "" t 'langb_64) ~(cblas t 'nrm2) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (gb-lan ~lapack ~(lapacke "" t 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \I a#))
     (asum [_# a#]
       (gb-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (gb-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (gb-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (gb-mv ~blas ~(cblas t 'gbmv) ~ptr (~cast alpha#) a# x# (~cast beta#) y#))
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for GB matrices." {:a (info a#)})))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for GB matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for GB matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# _#]
        (gb-mm a#))
       ([_# alpha# a# b# beta# c# left#]
        (gb-mm ~blas ~(cblas t 'gbmv) ~ptr (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro mkl-real-gb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (gb-lan ~lapack ~(lapacke t "" 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (gb-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (gb-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (gb-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for GB matrices." {:a (info a#)}))))

(deftype FloatGBEngine [])
(mkl-real-gb-blas* FloatGBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(mkl-real-gb-blas-plus* FloatGBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-gb-lapack* FloatGBEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatGBEngine "s" float-ptr float zero-float)

(deftype DoubleGBEngine [])
(mkl-real-gb-blas* DoubleGBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(mkl-real-gb-blas-plus* DoubleGBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-gb-lapack* DoubleGBEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleGBEngine "d" double-ptr double zero-double)

(deftype LongGBEngine [])
(deftype IntGBEngine [])
(deftype ShortGBEngine [])
(deftype ByteGBEngine [])

;; ============================ SB matrix engines ==================================================

(defmacro mkl-real-sb-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (gb-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (gb-map ~lapack ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (gb-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (sb-lan ~lapack ~(lapacke "" t 'lansb_64) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (sb-lan ~lapack ~(lapacke "" t 'lansb_64) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (sb-lan ~lapack ~(lapacke "" t 'lansb_64) ~ptr ~cpp-ptr \I a#))
     (asum [_# a#]
       (gb-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (gb-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (gb-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (sb-mv ~blas ~(cblas t 'sbmv) ~ptr (~cast alpha#) a# x# (~cast beta#) y#))
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for SB matrices." {:a (info a#)})))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for SB matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for SB matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# _#]
        (sb-mm a#))
       ([_# alpha# a# b# beta# c# left#]
        (sb-mm ~blas ~(cblas t 'sbmv) ~ptr (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro mkl-real-sb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (sb-lan ~lapack ~(lapacke "" t 'lansb_64) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (gb-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (gb-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (gb-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for SB matrices." {:a (info a#)}))))

(deftype FloatSBEngine [])
(mkl-real-sb-blas* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(mkl-real-sb-blas-plus* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-sb-lapack* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatSBEngine "s" float-ptr float zero-float)

(deftype DoubleSBEngine [])
(mkl-real-sb-blas* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(mkl-real-sb-blas-plus* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-sb-lapack* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleSBEngine "d" double-ptr double zero-double)

(deftype LongSBEngine [])
(deftype IntSBEngine [])
(deftype ShortSBEngine [])
(deftype ByteSBEngine [])

;; ============================ TB matrix engines ====================================================

(defmacro mkl-real-tb-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (gb-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (gb-map ~lapack ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (gb-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (tb-lan ~lapack ~(lapacke "" t 'lantb_64) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (tb-lan ~lapack ~(lapacke "" t 'lantb_64) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (tb-lan ~lapack ~(lapacke "" t 'lantb_64) ~ptr ~cpp-ptr \I a#))
     (asum [_# a#]
       (gb-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (gb-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (gb-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# _# a# _# _# _#]
        (tb-mv a#))
       ([_# a# x#]
        (tb-mv ~blas ~(cblas t 'tbmv) ~ptr a# x#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for TB matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for TB matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# left#]
        (tb-mm ~blas ~(cblas t 'tbmv) ~ptr (~cast alpha#) a# b# left#))
       ([_# _# a# _# _# _# _#]
        (tb-mm a#)))))

(defmacro mkl-real-tb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (tb-lan ~lapack ~(lapacke "" t 'lantb_64) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (gb-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (gb-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (gb-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for TB matrices." {:a (info a#)}))))

(deftype FloatTBEngine [])
(mkl-real-tb-blas* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(mkl-real-tb-blas-plus* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-tb-lapack* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-matrix-math* FloatTBEngine "s" float-ptr float zero-float)

(deftype DoubleTBEngine [])
(mkl-real-tb-blas* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(mkl-real-tb-blas-plus* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-tb-lapack* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-matrix-math* DoubleTBEngine "d" double-ptr double zero-double)

(deftype LongTBEngine [])
(deftype IntTBEngine [])
(deftype ShortTBEngine [])
(deftype ByteTBEngine [])

;; ============================ TP matrix engines ====================================================

(defmacro mkl-real-tp-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (packed-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (packed-map ~lapack ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (tp-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (tp-lan ~lapack ~(lapacke "" t 'lantp_64) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (tp-lan ~lapack ~(lapacke "" t 'lantp_64) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (tp-lan ~lapack ~(lapacke "" t 'lantp_64) ~ptr ~cpp-ptr \I a#))
     (asum [_# a#]
       (tp-sum ~blas ~(cblas t 'asum) ~ptr Math/abs a#))
     (scal [_# alpha# a#]
       (packed-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (packed-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# _# a# _# _# _#]
        (dragan-says-ex "Out-of-place mv! is not supported by TP matrices." {:a (info a#)}))
       ([_# a# x#]
        (let [reg# (region a#)]
          (. ~blas ~(cblas t 'tpmv) (.layout (navigator a#)) (.uplo reg#) ~(:no-trans blas-transpose)
             (.diag reg#) (ncols a#) (~ptr a#) (~ptr x#) (stride x#))
          x#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for packed matrices." {:a (info a#)})) ;;TODO extract method
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for packed matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# left#]
        (tp-mm ~blas ~(cblas t 'tpmv) ~ptr (~cast alpha#) a# b# left#))
       ([_# _# a# _# _# _# _#]
        (tp-mm a#)))))

(defmacro mkl-real-tp-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (tp-lan ~lapack ~(lapacke "" t 'lantp_64) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (tp-sum ~blas ~(cblas t 'dot) ~ptr Math/abs a# ~ones))
     (set-all [_# alpha# a#]
       (packed-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (packed-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for TP matrices." {:a (info a#)}))))

(deftype FloatTPEngine [])
(mkl-real-tp-blas* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(mkl-real-tp-blas-plus* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-tp-lapack* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatTPEngine "s" float-ptr float zero-float)

(deftype DoubleTPEngine [])
(mkl-real-tp-blas* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(mkl-real-tp-blas-plus* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-tp-lapack* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleTPEngine "d" double-ptr double zero-double)

(deftype LongTPEngine [])
(deftype IntTPEngine [])
(deftype ShortTPEngine [])
(deftype ByteTPEngine [])

;; ============================ SP matrix engines ====================================================

(defmacro mkl-real-sp-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (packed-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (packed-map ~lapack ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (sp-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (sp-lan ~lapack ~(lapacke "" t 'lansp_64) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (sp-lan ~lapack ~(lapacke "" t 'lansp_64) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (sp-lan ~lapack ~(lapacke "" t 'lansp_64) ~ptr ~cpp-ptr \I a#))
     (asum [_# a#]
       (sp-sum ~blas ~(cblas t 'asum) ~ptr Math/abs a#))
     (scal [_# alpha# a#]
       (packed-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (packed-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (let [reg# (region a#)]
          (. ~blas ~(cblas t 'spmv) (.layout (navigator a#)) (.uplo reg#) (ncols a#)
             (~cast alpha#) (~ptr a#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
          y#))
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported by SP matrices." {:a (info a#)})))
     (rk
       ([_# alpha# x# y# a#]
        (. ~blas ~(cblas t 'spr2) (.layout (navigator a#)) (.uplo (region a#)) (mrows a#)
           (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr a#))
        a#)
       ([_# alpha# x# a#]
        (. ~blas ~(cblas t 'spr) (.layout (navigator a#)) (.uplo (region a#)) (mrows a#)
           (~cast alpha#) (~ptr x#) (stride x#) (~ptr a#))
        a#))
     (mm
       ([_# alpha# a# b# left#]
        (sp-mm a#))
       ([_# alpha# a# b# beta# c# left#]
        (sp-mm ~blas ~(cblas t 'spmv) ~ptr (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro mkl-real-sp-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (sp-lan ~lapack ~(lapacke "" t 'lansp_64) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (sp-sum ~blas ~(cblas t 'dot) ~ptr ~cast a# ~ones))
     (set-all [_# alpha# a#]
       (packed-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (packed-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for SP matrices." {:a (info a#)}))))

(deftype FloatSPEngine [])
(mkl-real-sp-blas* FloatSPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(mkl-real-sp-blas-plus* FloatSPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-sp-lapack* FloatSPEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatSPEngine "s" float-ptr float zero-float)

(deftype DoubleSPEngine [])
(mkl-real-sp-blas* DoubleSPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(mkl-real-sp-blas-plus* DoubleSPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-sp-lapack* DoubleSPEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleSPEngine "d" double-ptr double zero-double)

(deftype LongSPEngine [])
(deftype IntSPEngine [])
(deftype ShortSPEngine [])
(deftype ByteSPEngine [])

;; ============================ GD matrix engines ==================================================

(defmacro mkl-real-gd-lapack* [name t ptr cpp-ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (laswp [_# _# _# _# _#]
       (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
     (tri [_# a#]
       (let [buff-a# (~ptr a#)]
         (. ~mkl ~(math t 'Inv) (ncols a#) buff-a# buff-a#)))
     (trs [_# a# b#]
       (gd-trs ~mkl ~(lapacke t 'tbtrs) ~ptr a# b#))
     (sv [_# a# b# _#]
       (mkl-gd-sv ~mkl ~(math t 'Div) ~(cblas t 'tbsv) ~ptr a# b#))
     (con [_# a# nrm1?#]
       (gd-con ~mkl ~(lapacke t 'tbcon) ~ptr ~cpp-ptr a# nrm1?#))))

(deftype FloatGDEngine [])
(real-gd-blas* FloatGDEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-diagonal-blas-plus* FloatGDEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(mkl-real-gd-lapack* FloatGDEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatGDEngine "s" float-ptr float zero-float)

(deftype DoubleGDEngine [])
(real-gd-blas* DoubleGDEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-diagonal-blas-plus* DoubleGDEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(mkl-real-gd-lapack* DoubleGDEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleGDEngine "d" double-ptr double zero-double)

(deftype LongGDEngine [])
(deftype IntGDEngine [])
(deftype ShortGDEngine [])
(deftype ByteGDEngine [])

(defmacro mkl-tridiagonal-lan [lapack method ptr norm a]
 `(if (< 0 (dim ~a))
    (let [n# (mrows ~a)
          n1# (if (< 0 n#) (dec n#) 0)
          du# (~ptr ~a n#)
          dl# (if (symmetric? ~a) du# (~ptr ~a (+ n# n1#)))]
      (with-release [norm# (byte-pointer (pointer ~norm))
                     n# (long-ptr (pointer (mrows ~a)))]
        (. ~lapack ~method norm# n# dl# (~ptr ~a) du#)))
    0.0))

(defmacro mkl-real-tridiagonal-blas* [name t ptr cpp-ptr cast mkl]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (diagonal-method ~mkl ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (diagonal-method ~mkl ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (diagonal-method ~mkl ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (mkl-tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \O a#))
     (nrm2 [_# a#]
       (diagonal-method ~mkl ~(cblas t 'nrm2) ~ptr a#))
     (nrmi [_# a#]
       (mkl-tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \I a#))
     (asum [_# a#]
       (diagonal-method ~mkl ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~mkl ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~mkl ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr long-pointer
                        (~cast alpha#) a# x# (~cast beta#) y#))
       ([_# a# x#]
        (tridiagonal-mv a#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for GT matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for GT matrices." {:a (info a#)})))
     (mm
       ([_# _# a# _# _#]
        (tridiagonal-mm a#))
       ([_# alpha# a# b# beta# c# left#]
        (tridiagonal-mm ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr long-pointer
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(deftype FloatGTEngine [])
(mkl-real-tridiagonal-blas* FloatGTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-diagonal-blas-plus* FloatGTEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(real-gt-lapack* FloatGTEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatGTEngine "s" float-ptr float zero-float)

(deftype DoubleGTEngine [])
(mkl-real-tridiagonal-blas* DoubleGTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-diagonal-blas-plus* DoubleGTEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(real-gt-lapack* DoubleGTEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleGTEngine "d" double-ptr double zero-double)

(deftype LongGTEngine [])
(deftype IntGTEngine [])
(deftype ShortGTEngine [])
(deftype ByteGTEngine [])

(defmacro mkl-real-dt-lapack* [name t ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# _# _#]
        (dragan-says-ex "Pivoted factorization is not available for DT matrices."))
       ([_# a#]
        (dt-trf ~mkl ~(lapacke "" t 'dttrfb_64) ~ptr a#)))
     (tri [_# _#]
       (dragan-says-ex "Inverse is not available for DT matrices."))
     (trs [_# lu# b#]
       (dt-trs ~mkl ~(lapacke "" t 'dttrsb_64) ~ptr lu# b#))
     (sv [_# a# b# pure#]
       (dt-sv ~mkl ~(lapacke "" t 'dtsvb_64) ~ptr a# b# pure#))
     (con [_# _# _# _# _#]
       (dragan-says-ex "Condition number is not available for DT matrices."))))

(deftype FloatDTEngine [])
(mkl-real-tridiagonal-blas* FloatDTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-diagonal-blas-plus* FloatDTEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(mkl-real-dt-lapack* FloatDTEngine "s" float-ptr float mkl_rt)
(real-matrix-math* FloatDTEngine "s" float-ptr float zero-float)

(deftype DoubleDTEngine [])
(mkl-real-tridiagonal-blas* DoubleDTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-diagonal-blas-plus* DoubleDTEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(mkl-real-dt-lapack* DoubleDTEngine "d" double-ptr double mkl_rt)
(real-matrix-math* DoubleDTEngine "d" double-ptr double zero-double)

(deftype LongDTEngine [])
(deftype IntDTEngine [])
(deftype ShortDTEngine [])
(deftype ByteDTEngine [])

(defmacro mkl-real-st-blas* [name t ptr cpp-ptr cast mkl]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (diagonal-method ~mkl ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (diagonal-method ~mkl ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (st-dot ~mkl ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (mkl-tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \O a#))
     (nrm2 [_# a#]
       (mkl-tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \F a#))
     (nrmi [_# a#]
       (mkl-tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \I a#))
     (asum [_# a#]
       (st-asum ~mkl ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~mkl ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~mkl ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr long-pointer
                        (~cast alpha#) a# x# (~cast beta#) y#))
       ([_# a# x#]
        (tridiagonal-mv a#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for ST matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for ST matrices." {:a (info a#)})))
     (mm
       ([_# _# a# _# _#]
        (tridiagonal-mm a#))
       ([_# alpha# a# b# beta# c# left#]
        (tridiagonal-mm ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr long-pointer
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(deftype FloatSTEngine [])
(mkl-real-st-blas* FloatSTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-st-blas-plus* FloatSTEngine "s" float-ptr float mkl_rt mkl_rt "cblas_saxpby" ones-float)
(real-st-lapack* FloatSTEngine "s" float-ptr float mkl_rt)
(real-matrix-math* FloatSTEngine "s" float-ptr float ge-zero-float)

(deftype DoubleSTEngine [])
(mkl-real-st-blas* DoubleSTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-st-blas-plus* DoubleSTEngine "d" double-ptr double mkl_rt mkl_rt "cblas_daxpby" ones-double)
(real-st-lapack* DoubleSTEngine "d" double-ptr double mkl_rt)
(real-matrix-math* DoubleSTEngine "d" double-ptr double zero-double)

(deftype LongSTEngine [])
(deftype IntSTEngine [])
(deftype ShortSTEngine [])
(deftype ByteSTEngine [])

;; ========================= Sparse Vector engines ============================================

(def ^{:no-doc true :const true} MIXED_UNSUPPORTED_MSG
  "This operation is not supported on mixed sparse and dense vectors.")

(def ^{:no-doc true :const true} SPARSE_UNSUPPORTED_MSG
  "This operation is not supported on sparse.")

(defmacro real-cs-vector-blas* [name t ptr idx-ptr cast blas ones]
  `(extend-type ~name
     Blas
     (swap [this# x# y#]
       (if (indices y#)
         (swap (engine (entries x#)) (entries x#) (entries y#))
         (throw (UnsupportedOperationException. MIXED_UNSUPPORTED_MSG)))
       y#)
     (copy [this# x# y#]
       (if (indices y#)
         (copy (engine (entries x#)) (entries x#) (entries y#))
         (throw (UnsupportedOperationException. MIXED_UNSUPPORTED_MSG)))
       y#)
     (dot [this# x# y#]
       (if (indices y#)
         (dot (engine (entries x#)) (entries x#) (entries y#))
         (. ~blas ~(cblas t 'doti) (dim (entries x#)) (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#))))
     (nrm1 [this# x#]
       (asum this# x#))
     (nrm2 [this# x#]
       (nrm2 (engine (entries x#)) (entries x#)))
     (nrmi [this# x#]
       (amax this# x#))
     (asum [this# x#]
       (asum (engine (entries x#)) (entries x#)))
     (iamax [this# x#]
       (entry (indices x#) (iamax (engine (entries x#)) (entries x#))))
     (iamin [this# x#]
       (entry (indices x#) (iamin (engine (entries x#)) (entries x#))))
     (rot [this# x# y# c# s#]
       (if (indices y#)
         (rot (engine (entries x#)) (entries x#) (entries y#) c# s#)
         (. ~blas ~(cblas t 'roti) (dim (entries x#))
            (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#) (~cast c#) (~cast s#))))
     (rotg [this# abcs#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (rotm [this# x# y# param#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (rotmg [this# d1d2xy# param#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (scal [this# alpha# x#]
       (scal (engine (entries x#)) alpha# (entries x#))
       x#)
     (axpy [this# alpha# x# y#]
       (if (indices y#)
         (axpy (engine (entries x#)) alpha# (entries x#) (entries y#))
         (. ~blas ~(cblas t 'axpyi) (dim (entries x#))
            (~cast alpha#) (~ptr x#) (~idx-ptr (indices x#)) (~ptr y#)))
       y#)))

(defmacro real-cs-blas-plus* [name t ptr cast]
  `(extend-type ~name
     BlasPlus
     (amax [this# x#]
       (amax (engine (entries x#)) (entries x#)))
     (subcopy [this# x# y# kx# lx# ky#]
       (throw (UnsupportedOperationException. SPARSE_UNSUPPORTED_MSG)))
     (sum [this# x#]
       (sum (engine (entries x#)) (entries x#)))
     (imax [this# x#]
       (imax (engine (entries x#)) (entries x#)))
     (imin [this# x#]
       (imin (engine (entries x#)) (entries x#)))
     (set-all [this# alpha# x#]
       (set-all (engine (entries x#)) alpha# (entries x#))
       x#)
     (axpby [this# alpha# x# beta# y#]
       (if (= 1.0 beta#)
         (axpy this# alpha# x# y#)
         (if (indices y#)
           (axpby (engine (entries x#)) alpha# (entries x#) beta# (entries y#))
           (do (scal (engine y#) beta# (entries y#))
               (axpy this# alpha# x# y#))))
       y#)))

(defmacro real-cs-vector-sparse-blas* [name t ptr idx-ptr blas]
  `(extend-type ~name
     SparseBlas
     (gthr [this# y# x#]
       (. ~blas ~(cblas t 'gthr) (dim (entries x#)) (~ptr y#) (~ptr x#) (~idx-ptr (indices x#)))
       x#)))

(deftype FloatCSVectorEngine [])
(real-cs-vector-blas* FloatCSVectorEngine "s" float-ptr int-ptr float mkl_rt ones-float)
(real-cs-blas-plus* FloatCSVectorEngine "s" float-ptr float)
(real-vector-math* FloatCSVectorEngine "s" float-ptr float ge-zero-float)
(real-cs-vector-sparse-blas* FloatCSVectorEngine "s" float-ptr int-ptr mkl_rt)

(deftype DoubleCSVectorEngine [])
(real-cs-vector-blas* DoubleCSVectorEngine "d" double-ptr int-ptr double mkl_rt ones-double)
(real-cs-blas-plus* DoubleCSVectorEngine "d" double-ptr double)
(real-vector-math* DoubleCSVectorEngine "d" double-ptr double ge-zero-double)
(real-cs-vector-sparse-blas* DoubleCSVectorEngine "d" double-ptr int-ptr mkl_rt)

;; =================== Sparse Matrix engines ======================================

(defmacro real-csr-blas* [name t ptr cast]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (with-release [tmp# (sparse-matrix)]
         (mkl-sparse-copy (~ptr b#) tmp#)
         (mkl-sparse-copy (~ptr a#) (~ptr b#))
         (mkl-sparse-copy tmp# (~ptr a#)))
       a#)
     (copy [_# a# b#]
       (mkl-sparse-copy (~ptr a#) (~ptr b#))
       b#)
     (nrm1 [this# a#]
       (asum this# a#))
     (nrm2 [this# a#]
       (nrm2 (engine (entries a#)) (entries a#)))
     (nrmi [this# a#]
       (amax this# a#))
     (asum [this# a#]
       (asum (engine (entries a#)) (entries a#)))
     (scal [this# alpha# a#]
       (scal (engine (entries a#)) alpha# (entries a#))
       a#)
     (mv
       ([_# alpha# a# x# beta# y#]
        (. mkl_rt ~(mkl-sparse t 'mv) (sparse-transpose a#) (~cast alpha#) (spmat a#) (descr a#)
           (~ptr x#) (~cast beta#) (~ptr y#))
        y#)
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for sparse matrices." {:a (info a#)})))
     (mm
       ;; TODO ([this# alpha# a# b#]) instead of mm!
       ([this# alpha# a# b# _#]
        (let-release [c# (sparse-matrix)
                      res# (with-check sparse-error ;;TODO perhaps use mkl_sparse_sp2m?!
                             (mkl_rt/mkl_sparse_spmm (sparse-transpose a#) (spmat a#) (spmat b#) c#)
                             (ge-csr-matrix (factory a#) c# (.isColumnMajor (navigator a#))))]
          (if (f= 1.0 (~cast alpha#))
            res#
            (scal this# alpha# res#))))
       ([this# alpha# a# b# beta# c# _#]
        (cond
          (instance? GEMatrix b#)
          (with-check sparse-error
            (. mkl_rt ~(mkl-sparse t 'mm) (sparse-transpose a#) (~cast alpha#) (spmat a#) (descr a#)
               (sparse-layout b#) (~ptr b#) (ncols c#) (stride b#) (~cast beta#) (~ptr c#) (stride c#))
            c#)
          (instance? GEMatrix c#)
          (if (f= 0.0 (~cast beta#))
            (with-check sparse-error
              (. mkl_rt ~(mkl-sparse t 'spmmd) (sparse-transpose a#) (spmat a#) (spmat b#)
                 (sparse-layout c#) (~ptr c#) (stride c#))
              (if (f= 1.0 (~cast alpha#))
                c#
                (scal this# alpha# c#)))
            (dragan-says-ex "Beta parameter not supported for sparse matrix multiplication." {:beta beta#}))
          :default
          (do
            (if (f= 0.0 (~cast beta#))
              (csr-ge-sp2m a# b# c# :finalize)
              (dragan-says-ex "Beta parameter not supported for sparse matrix multiplication." {:beta beta#}))
            (if (f= 1.0 (~cast alpha#))
              c#
              (scal this# alpha# c#))))))))

(deftype DoubleCSREngine [])
(real-csr-blas* DoubleCSREngine "d" double-ptr double)
(real-cs-blas-plus* DoubleCSREngine "d" double-ptr double)

(deftype FloatCSREngine [])
(real-csr-blas* FloatCSREngine "s" float-ptr float)
(real-cs-blas-plus* FloatCSREngine "s" float-ptr float)

;; ================================================================================

(deftype MKLRealFactory [index-fact ^DataAccessor da
                         vector-eng ge-eng tr-eng sy-eng gb-eng sb-eng tb-eng
                         sp-eng tp-eng gd-eng gt-eng dt-eng st-eng
                         cs-vector-eng csr-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  (device [_]
    :cpu)
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  UnsafeFactory
  (create-vector* [this master buf-ptr n strd]
    (real-block-vector* this master buf-ptr n strd))
  Factory
  (create-vector [this n init]
    (let-release [res (real-block-vector this n)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-vector [this master buf-ptr n ofst strd]
    (real-block-vector this master buf-ptr n ofst strd))
  (create-ge [this m n column? init]
    (let-release [res (real-ge-matrix this m n column?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (real-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (real-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (real-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-banded [this m n kl ku matrix-type column? init]
    (let-release [res (real-banded-matrix this m n kl ku column? matrix-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-gb [this m n kl ku lower? init]
    (let-release [res (real-banded-matrix this m n kl ku lower?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-tb [this n k column? lower? diag-unit? init]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (let-release [res (real-tb-matrix this n k column? lower? diag-unit?)]
        (when init
          (.initialize da (extract res)))
        res)
      (dragan-says-ex "TB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-sb [this n k column? lower? init]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (let-release [res (real-sb-matrix this n k column? lower?)]
        (when init
          (.initialize da (extract res)))
        res)
      (dragan-says-ex "SB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-packed [this n matrix-type column? lower? diag-unit? init]
    (let-release [res (real-packed-matrix this n column? lower? diag-unit? matrix-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-tp [this n column? lower? diag-unit? init]
    (let-release [res (real-packed-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-sp [this n column? lower? init]
    (let-release [res (real-packed-matrix this n column? lower?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-diagonal [this n matrix-type init]
    (let-release [res (real-diagonal-matrix this n matrix-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng)
  (gb-engine [_]
    gb-eng)
  (sb-engine [_]
    sb-eng)
  (tb-engine [_]
    tb-eng)
  (tp-engine [_]
    tp-eng)
  (sp-engine [_]
    sp-eng)
  (gd-engine [_]
    gd-eng)
  (gt-engine [_]
    gt-eng)
  (dt-engine [_]
    dt-eng)
  (st-engine [_]
    st-eng)
  SparseFactory
  (create-ge-csr [this m n idx idx-b idx-e column? init]
    (ge-csr-matrix this m n idx idx-b (view idx-e) column? init))
  (create-ge-csr [this a b indices?]
    (csr-ge-sp2m a b (if indices? :full-no-val :count)))
  (cs-vector-engine [_]
    cs-vector-eng)
  (csr-engine [_]
    csr-eng))

(deftype MKLIntegerFactory [index-fact ^DataAccessor da
                            vector-eng ge-eng tr-eng sy-eng gb-eng sb-eng tb-eng
                            sp-eng tp-eng gd-eng gt-eng dt-eng st-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  (device [_]
    :cpu)
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  UnsafeFactory
  (create-vector* [this master buf-ptr n strd]
    (integer-block-vector* this master buf-ptr n strd))
  Factory
  (create-vector [this n init]
    (let-release [res (integer-block-vector this n)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-vector [this master buf-ptr n ofst strd]
    (integer-block-vector this master buf-ptr n ofst strd))
  (create-ge [this m n column? init]
    (let-release [res (integer-ge-matrix this m n column?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (integer-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (integer-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (integer-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (extract res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng)
  (gb-engine [_]
    gb-eng)
  (sb-engine [_]
    sb-eng)
  (tb-engine [_]
    tb-eng)
  (tp-engine [_]
    tp-eng)
  (sp-engine [_]
    sp-eng)
  (gd-engine [_]
    gd-eng)
  (gt-engine [_]
    gt-eng)
  (dt-engine [_]
    dt-eng)
  (st-engine [_]
    st-eng))

(def mkl-float-accessor (->FloatPointerAccessor malloc! free!))
(def mkl-double-accessor (->DoublePointerAccessor malloc! free!))
(def mkl-int-accessor (->IntPointerAccessor malloc! free!))
(def mkl-long-accessor (->LongPointerAccessor malloc! free!))
(def mkl-short-accessor (->ShortPointerAccessor malloc! free!))
(def mkl-byte-accessor (->BytePointerAccessor malloc! free!))

(def mkl-int (->MKLIntegerFactory mkl-int mkl-int-accessor (->IntVectorEngine) (->IntGEEngine)
                                  (->IntTREngine) (->IntSYEngine)
                                  (->IntGBEngine) (->IntSBEngine) (->IntTBEngine)
                                  (->IntSPEngine) (->IntTPEngine) (->IntGDEngine)
                                  (->IntGTEngine) (->IntDTEngine) (->IntSTEngine)))

(def mkl-long (->MKLIntegerFactory mkl-int mkl-long-accessor (->LongVectorEngine) (->LongGEEngine)
                                   (->LongTREngine) (->LongSYEngine)
                                   (->LongGBEngine) (->LongSBEngine) (->LongTBEngine)
                                   (->LongSPEngine) (->LongTPEngine) (->LongGDEngine)
                                   (->LongGTEngine) (->LongDTEngine) (->LongSTEngine)))

(def mkl-short (->MKLIntegerFactory mkl-int mkl-short-accessor (->ShortVectorEngine) (->ShortGEEngine)
                                    (->ShortTREngine) (->ShortSYEngine)
                                    (->ShortGBEngine) (->ShortSBEngine) (->ShortTBEngine)
                                    (->ShortSPEngine) (->ShortTPEngine) (->ShortGDEngine)
                                    (->ShortGTEngine) (->ShortDTEngine) (->ShortSTEngine)))

(def mkl-byte (->MKLIntegerFactory mkl-int mkl-byte-accessor (->ByteVectorEngine) (->ByteGEEngine)
                                   (->ByteTREngine) (->ByteSYEngine)
                                   (->ByteGBEngine) (->ByteSBEngine) (->ByteTBEngine)
                                   (->ByteSPEngine) (->ByteTPEngine) (->ByteGDEngine)
                                   (->ByteGTEngine) (->ByteDTEngine) (->ByteSTEngine)))

(def mkl-float (->MKLRealFactory mkl-int mkl-float-accessor (->FloatVectorEngine) (->FloatGEEngine)
                                 (->FloatTREngine) (->FloatSYEngine)
                                 (->FloatGBEngine) (->FloatSBEngine) (->FloatTBEngine)
                                 (->FloatSPEngine) (->FloatTPEngine) (->FloatGDEngine)
                                 (->FloatGTEngine) (->FloatDTEngine) (->FloatSTEngine)
                                 (->FloatCSVectorEngine) (->FloatCSREngine)))

(def mkl-double (->MKLRealFactory mkl-int mkl-double-accessor (->DoubleVectorEngine) (->DoubleGEEngine)
                                  (->DoubleTREngine) (->DoubleSYEngine)
                                  (->DoubleGBEngine) (->DoubleSBEngine) (->DoubleTBEngine)
                                  (->DoubleSPEngine) (->DoubleTPEngine) (->DoubleGDEngine)
                                  (->DoubleGTEngine) (->DoubleDTEngine) (->DoubleSTEngine)
                                  (->DoubleCSVectorEngine)  (->DoubleCSREngine)))
