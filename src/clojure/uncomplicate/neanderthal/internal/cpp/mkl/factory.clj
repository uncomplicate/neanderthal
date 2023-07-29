;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.factory
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release view]]
             [utils :refer [dragan-says-ex with-check generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojure-cpp :as cpp :refer [long-pointer float-pointer double-pointer]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols matrix-type] :as core]
             [real :as real]
             [integer :as integer]
             [math :refer [f=] :as math]
             [block :refer [create-data-source initialize offset stride contiguous?]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage accu-layout dostripe-layout]]
             [common :refer [check-stride check-eq-navigators flip-uplo real-accessor]]]
            [uncomplicate.neanderthal.internal.cpp
             [structures :refer :all]
             [lapack :refer :all]
             [blas :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.mkl
             [constants :refer [mkl-sparse-request]]
             [core :refer [malloc! free! mkl-sparse sparse-matrix mkl-sparse-copy sparse-error]]
             [structures :refer [ge-csr-matrix spmat descr sparse-transpose sparse-layout csr-ge-sp2m]]])
  (:import java.nio.ByteBuffer
           [uncomplicate.neanderthal.internal.api DataAccessor Block Vector LayoutNavigator Region
            GEMatrix UploMatrix DenseStorage]
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer BytePointer]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$VSLStreamStatePtr]))

;; =============== Factories ==================================================

(declare mkl-int)

(defn cblas
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (cblas "cblas_" type name)))

(defn lapacke
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (lapacke "LAPACKE_" type name)))

(defn math
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (math "v" type name)))

;; ================= MKL specific blas and lapack functions =====================

(defmacro gd-sv [mkl vdiv-method sv-method ptr a b]
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
       (dotimes [j# (ncols ~b)]
         (. ~mkl ~sv-method ~(:row blas-layout) ~(:lower blas-uplo) ~(:no-trans blas-transpose)
            ~(:non-unit blas-diag) n-a# 0 buff-a# 1
            (.position buff-b# (.index nav-b# stor-b# 0 j#)) strd-b#)))
     ~b))

;; ================= Integer Vector Engines =====================================

(def ^{:no-doc true :const true} INTEGER_UNSUPPORTED_MSG
  "Integer BLAS operations are not supported. Please transform data to float or double.")

(def ^{:no-doc true :const true} SHORT_UNSUPPORTED_MSG
  "BLAS operation on short integers are supported only on dimensions divisible by 2 (short) or 4 (byte).")

(defmacro patch-vector-method [chunk blas method ptr x y]
  (if (= 1 chunk)
    `(. ~blas ~method (dim ~x) (~ptr ~x 0) (stride ~x) (~ptr ~y 0) (stride ~y))
    `(if (= 0 (rem (dim ~x) ~chunk))
       (. ~blas ~method (quot (dim ~x) ~chunk) (~ptr ~x 0) (stride ~x) (~ptr ~y 0) (stride ~y))
       (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim ~x)}))))

(defmacro patch-subcopy [chunk blas method ptr x y kx lx ky]
  (if (= 1 chunk)
    `(. ~blas ~method (int lx#) (~ptr ~x ~kx) (stride ~x) (~ptr ~y ~ky) (stride ~y))
    `(do
       (check-stride ~x ~y)
       (if (= 0 (rem ~kx ~chunk) (rem ~lx ~chunk) (rem ~ky ~chunk))
         (. ~blas ~method (quot ~lx ~chunk) (~ptr ~x ~kx) (stride ~x) (~ptr ~y ~ky) (stride ~y))
         (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim ~x)})))))

(defmacro patch-vector-laset [chunk lapack method ptr alpha x]
  (if (= 1 chunk)
    `(with-lapack-check "laset"
       (. ~lapack ~method ~(int (:row blas-layout)) ~(byte (int \g))
          (dim ~x) 1 ~alpha ~alpha (~ptr ~x 0) (stride ~x)))
    `(do
       (check-stride ~x)
       (if (= 0 (rem (dim ~x) ~chunk))
         (with-lapack-check "laset"
           (. ~lapack ~method ~(int (:row blas-layout)) ~(byte (int \g))
              (quot (dim ~x) ~chunk) 1 ~alpha ~alpha (~ptr ~x 0) (stride ~x)))
         (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim ~x)})))))

(defmacro integer-vector-blas* [name t ptr blas chunk]
  `(extend-type ~name
     Blas
     (swap [_# x# y#]
       (patch-vector-method ~chunk ~blas ~(cblas t 'swap) ~ptr x# y#)
       x#)
     (copy [_# x# y#]
       (patch-vector-method ~chunk ~blas ~(cblas t 'copy) ~ptr x# y#)
       y#)
     (dot [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm1 [_# x#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrm2 [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (nrmi [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (asum [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (iamax [this# x#]
       (vector-iaopt < x# integer/entry))
     (iamin [this# x#]
       (vector-iaopt > x# integer/entry))
     (rot [_# _# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotg [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotm [_# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (rotmg [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (scal [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (axpy [_# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     Lapack
     (srt [_# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))))

(defmacro integer-vector-blas-plus* [name t ptr cast blas lapack chunk]
  `(extend-type ~name
     BlasPlus
     (amax [this# x#]
       (if (< 0 (dim x#))
         (Math/abs (integer/entry x# (iamax this# x#)))
         0))
     (subcopy [_# x# y# kx# lx# ky#]
       (patch-subcopy (long ~chunk) ~blas ~(cblas t 'copy) ~ptr x# y# (long kx#) (long lx#) (long ky#))
       y#)
     (sum [_# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
     (imax [this# x#]
       (vector-iopt < x# integer/entry))
     (imin [this# x#]
       (vector-iopt > x# integer/entry))
     (set-all [_# alpha# x#]
       (patch-vector-laset (long ~chunk) ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) x#)
       x#)
     (axpby [_# _# _# _# _#]
       (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))))

;; ================= Real Vector Engines ========================================

(defmacro real-vector-blas* [name t ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [this# x# y#]
       (. ~blas ~(cblas t 'swap) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       x#)
     (copy [this# x# y#]
       (. ~blas ~(cblas t 'copy) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     (dot [this# x# y#]
       (. ~blas ~(cblas t 'dot) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#)))
     (nrm1 [this# x#]
       (asum this# x#))
     (nrm2 [this# x#]
       (. ~blas ~(cblas t 'nrm2) (dim x#) (~ptr x#) (stride x#)))
     (nrmi [this# x#]
       (amax this# x#))
     (asum [this# x#]
       (. ~blas ~(cblas t 'asum) (dim x#) (~ptr x#) (stride x#)))
     (iamax [this# x#]
       (. ~blas ~(cblas 'cblas_i t 'amax) (dim x#) (~ptr x#) (stride x#)))
     (iamin [this# x#]
       (. ~blas ~(cblas 'cblas_i t 'amin) (dim x#) (~ptr x#) (stride x#)))
     (rot [this# x# y# c# s#]
       (. ~blas ~(cblas t 'rot) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#)
          (~cast c#) (~cast s#))
       x#)
     (rotg [this# abcs#]
       (let [stride# (stride abcs#)]
         (. ~blas ~(cblas t 'rotg) (~ptr abcs#) (~ptr abcs# stride#) (~ptr abcs# (* 2 stride#))
            (~ptr abcs# (* 3 stride#)))
         abcs#))
     (rotm [this# x# y# param#]
       (check-stride param#)
       (. ~blas ~(cblas t 'rotm) (dim x#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr param#))
       x#)
     (rotmg [this# d1d2xy# param#]
       (check-stride d1d2xy# param#)
       (. ~blas ~(cblas t 'rotmg) (~ptr d1d2xy#) (~ptr d1d2xy# 1) (~ptr d1d2xy# 2)
          (~cast (real/entry d1d2xy# 3)) (~ptr param#))
       param#)
     (scal [this# alpha# x#]
       (. ~blas ~(cblas t 'scal) (dim x#) (~cast alpha#) (~ptr x#) (stride x#))
       x#)
     (axpy [this# alpha# x# y#]
       (. ~blas ~(cblas t 'axpy) (dim x#) (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#))
       y#)
     Lapack
     (srt [this# x# increasing#]
       (if (= 1 (stride x#))
         (with-lapack-check "lasrt"
           (. ~lapack ~(lapacke t 'lasrt) (byte (int (if increasing# \I \D))) (dim x#) (~ptr x#)))
         (dragan-says-ex "You cannot sort a vector with stride." {"stride" (stride x#)}))
       x#)))

(defmacro real-vector-blas-plus* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [this# x#]
       (if (< 0 (dim x#))
         (Math/abs (real/entry x# (iamax this# x#)))
         0.0))
     (subcopy [this# x# y# kx# lx# ky#]
       (. ~blas ~(cblas t 'copy) (int lx#) (~ptr x# kx#) (stride x#) (~ptr y# ky#) (stride y#))
       y#)
     (sum [this# x#]
       (. ~blas ~(cblas t 'dot) (dim x#) (~ptr x#) (stride x#) (~ptr ~ones) 0))
     (imax [this# x#]
       (vector-iopt < x# real/entry))
     (imin [this# x#]
       (vector-iopt > x# real/entry))
     (set-all [this# alpha# x#]
       (with-lapack-check "laset"
         (. ~lapack ~(lapacke t 'laset) ~(int (:row blas-layout)) ~(byte (int \g)) (dim x#) 1
            (~cast alpha#) (~cast alpha#) (~ptr x#) (stride x#)))
       x#)
     (axpby [this# alpha# x# beta# y#]
       (. ~blas ~(cblas t 'axpby) (dim x#)
          (~cast alpha#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
       y#)))

(defmacro vector-math
  ([method ptr a y]
   `(do
      (check-stride ~a ~y)
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (~ptr ~y))
      ~y))
  ([method ptr a b y]
   `(do
      (check-stride ~a ~b ~y)
      (. mkl_rt ~method (dim ~a) (~ptr ~a) (~ptr ~b) (~ptr ~y))
      ~y)))

(defmacro vector-linear-frac [method ptr a b scalea shifta scaleb shiftb y]
  `(do
     (check-stride ~a ~b ~y)
     (. mkl_rt ~method (dim ~a) (~ptr ~a) (~ptr ~b) ~scalea ~shifta ~scaleb ~shiftb (~ptr ~y))
     ~y))

(defmacro vector-powx [method ptr a b y]
  `(do
     (check-stride ~a ~y)
     (. mkl_rt ~method (dim ~a) (~ptr ~a) ~b (~ptr ~y))
     ~y))

;; ============ Delegate math functions  ============================================

(defn sigmoid-over-tanh [eng a y]
  (when-not (identical? a y) (copy eng a y))
  (linear-frac eng (tanh eng (scal eng 0.5 y) y) a 0.5 0.5 0.0 1.0 y))

(defn vector-ramp [eng a y]
  (cond (identical? a y) (fmap! math/ramp y)
        (and (contiguous? a) (contiguous? y)) (fmax eng a (set-all eng 0.0 y) y)
        :else (fmap! math/ramp (copy eng a y))))

(defn vector-relu [eng alpha a y]
  (cond (identical? a y) (fmap! (math/relu alpha) y)
        (and (contiguous? a) (contiguous? y)) (fmax eng a (axpby eng alpha a 0.0 y) y)
        :else (fmap! (math/relu alpha) (copy eng a y))))

(defn vector-elu [eng alpha a y]
  (cond (identical? a y) (fmap! (math/elu alpha) y)
        (and (contiguous? a) (contiguous? y))
        (fmax eng a (scal eng alpha (expm1 eng (copy eng a y) y)) y)
        :else (fmap! (math/elu alpha) (copy eng a y))))

(defmacro with-rng-check [x expr] ;;TODO maybe avoid special method for this.
  `(if (< 0 (dim ~x))
     (if (= 1 (stride ~x))
       (let [err# ~expr]
         (if (= 0 err#)
           ~x
           (throw (ex-info "MKL error." {:error-code err#}))))
       (dragan-says-ex "This engine cannot generate random entries in host vectors with stride. Sorry."
                       {:v (info ~x)}))
     ~x))

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

(defmacro real-math* [name t ptr cast vector-math vector-linear-frac vector-powx]
  `(extend-type ~name
     VectorMath
     (sqr [_# a# y#]
       (~vector-math ~(math t 'Sqr) ~ptr a# y#))
     (mul [_# a# b# y#]
       (~vector-math ~(math t 'Mul) ~ptr a# b# y#))
     (div [_# a# b# y#]
       (~vector-math ~(math t 'Div) ~ptr a# b# y#))
     (inv [_# a# y#]
       (~vector-math ~(math t 'Inv) ~ptr a# y#))
     (abs [_# a# y#]
       (~vector-math ~(math t 'Abs) ~ptr a# y#))
     (linear-frac [_# a# b# scalea# shifta# scaleb# shiftb# y#]
       (~vector-linear-frac ~(math t 'LinearFrac) ~ptr a# b#
        (~cast scalea#) (~cast shifta#) (~cast scaleb#) (~cast shiftb#) y#))
     (fmod [_# a# b# y#]
       (~vector-math ~(math t 'Fmod) ~ptr a# b# y#))
     (frem [_# a# b# y#]
       (~vector-math  ~(math t 'Remainder) ~ptr a# b# y#))
     (sqrt [_# a# y#]
       (~vector-math ~(math t 'Sqrt) ~ptr a# y#))
     (inv-sqrt [_# a# y#]
       (~vector-math ~(math t 'InvSqrt) ~ptr a# y#))
     (cbrt [_# a# y#]
       (~vector-math ~(math t 'Cbrt) ~ptr a# y#))
     (inv-cbrt [_# a# y#]
       (~vector-math ~(math t 'InvCbrt) ~ptr a# y#))
     (pow2o3 [_# a# y#]
       (~vector-math ~(math t 'Pow2o3) ~ptr a# y#))
     (pow3o2 [_# a# y#]
       (~vector-math ~(math t 'Pow3o2) ~ptr a# y#))
     (pow [_# a# b# y#]
       (~vector-math ~(math t 'Pow) ~ptr a# b# y#))
     (powx [_# a# b# y#]
       (~vector-powx ~(math t 'Powx) ~ptr a# (~cast b#) y#))
     (hypot [_# a# b# y#]
       (~vector-math ~(math t 'Hypot) ~ptr a# b# y#))
     (exp [_# a# y#]
       (~vector-math ~(math t 'Exp) ~ptr a# y#))
     (exp2 [_# a# y#]
       (~vector-math ~(math t 'Exp2) ~ptr a# y#))
     (exp10 [_# a# y#]
       (~vector-math ~(math t 'Exp10) ~ptr a# y#))
     (expm1 [_# a# y#]
       (~vector-math ~(math t 'Expm1) ~ptr a# y#))
     (log [_# a# y#]
       (~vector-math ~(math t 'Ln) ~ptr a# y#))
     (log2 [_# a# y#]
       (~vector-math ~(math t 'Log2) ~ptr a# y#))
     (log10 [_# a# y#]
       (~vector-math ~(math t 'Log10) ~ptr a# y#))
     (log1p [_# a# y#]
       (~vector-math ~(math t 'Log1p) ~ptr a# y#))
     (sin [_# a# y#]
       (~vector-math ~(math t 'Sin) ~ptr a# y#))
     (cos [_# a# y#]
       (~vector-math ~(math t 'Cos) ~ptr a# y#))
     (tan [_# a# y#]
       (~vector-math ~(math t 'Tan) ~ptr a# y#))
     (sincos [_# a# y# z#]
       (~vector-math ~(math t 'SinCos) ~ptr a# y# z#))
     (asin [_# a# y#]
       (~vector-math ~(math t 'Asin) ~ptr a# y#))
     (acos [_# a# y#]
       (~vector-math ~(math t 'Acos) ~ptr a# y#))
     (atan [_# a# y#]
       (~vector-math ~(math t 'Atan) ~ptr a# y#))
     (atan2 [_# a# b# y#]
       (~vector-math ~(math t 'Atan2) ~ptr a# b# y#))
     (sinh [_# a# y#]
       (~vector-math ~(math t 'Sinh) ~ptr a# y#))
     (cosh [_# a# y#]
       (~vector-math ~(math t 'Cosh) ~ptr a# y#))
     (tanh [_# a# y#]
       (~vector-math ~(math t 'Tanh) ~ptr a# y#))
     (asinh [_# a# y#]
       (~vector-math ~(math t 'Asinh) ~ptr a# y#))
     (acosh [_# a# y#]
       (~vector-math ~(math t 'Acosh) ~ptr a# y#))
     (atanh [_# a# y#]
       (~vector-math ~(math t 'Atanh) ~ptr a# y#))
     (erf [_# a# y#]
       (~vector-math ~(math t 'Erf) ~ptr a# y#))
     (erfc [_# a# y#]
       (~vector-math ~(math t 'Erfc) ~ptr a# y#))
     (erf-inv [_# a# y#]
       (~vector-math ~(math t 'ErfInv) ~ptr a# y#))
     (erfc-inv [_# a# y#]
       (~vector-math ~(math t 'ErfcInv) ~ptr a# y#))
     (cdf-norm [_# a# y#]
       (~vector-math ~(math t 'CdfNorm) ~ptr a# y#))
     (cdf-norm-inv [_# a# y#]
       (~vector-math ~(math t 'CdfNormInv) ~ptr a# y#))
     (gamma [_# a# y#]
       (~vector-math ~(math t 'TGamma) ~ptr a# y#))
     (lgamma [_# a# y#]
       (~vector-math ~(math t 'LGamma) ~ptr a# y#))
     (expint1 [_# a# y#]
       (~vector-math ~(math t 'ExpInt1) ~ptr a# y#))
     (floor [_# a# y#]
       (~vector-math ~(math t 'Floor) ~ptr a# y#))
     (fceil [_# a# y#]
       (~vector-math ~(math t 'Ceil) ~ptr a# y#))
     (trunc [_# a# y#]
       (~vector-math ~(math t 'Trunc) ~ptr a# y#))
     (round [_# a# y#]
       (~vector-math ~(math t 'Round) ~ptr a# y#))
     (modf [_# a# y# z#]
       (~vector-math ~(math t 'Modf) ~ptr a# y# z#))
     (frac [_# a# y#]
       (~vector-math ~(math t 'Frac) ~ptr a# y#))
     (fmin [_# a# b# y#]
       (~vector-math ~(math t 'Fmin) ~ptr a# b# y#))
     (fmax [_# a# b# y#]
       (~vector-math ~(math t 'Fmax) ~ptr a# b# y#))
     (copy-sign [_# a# b# y#]
       (~vector-math ~(math t 'CopySign) ~ptr a# b# y#))
     (sigmoid [this# a# y#]
       (sigmoid-over-tanh this# a# y#))
     (ramp [this# a# y#]
       (~vector-ramp this# a# y#))
     (relu [this# alpha# a# y#]
       (~vector-relu this# alpha# a# y#))
     (elu [this# alpha# a# y#]
       (~vector-elu this# alpha# a# y#))))

(defmacro real-vector-math* [name t ptr cast]
  `(real-math* ~name ~t ~ptr ~cast vector-math vector-linear-frac vector-powx))

(def ^:private ones-float (->RealBlockVector nil nil nil true (float-pointer [1.0]) 1 0))
(def ^:private ones-double (->RealBlockVector nil nil nil true (double-pointer [1.0]) 1 0))
(def ^:private zero-float (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                          nil nil nil nil true (float-pointer [0.0]) 0 0))
(def ^:private zero-double (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                           nil nil nil nil true (double-pointer [0.0]) 0 0))

(defn cast-stream ^mkl_rt$VSLStreamStatePtr [stream]
  (or stream default-rng-stream))

(defmacro real-vector-rng* [name t ptr cast]
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

(defmacro byte-float [x]
  `(let [b# (ByteBuffer/allocate Float/BYTES)
         x# (byte ~x)]
     (.put b# 0 x#)
     (.put b# 1 x#)
     (.put b# 2 x#)
     (.put b# 3 x#)
     (.getFloat b# 0)))

(defmacro short-float [x]
  `(let [b# (ByteBuffer/allocate Float/BYTES)
         x# (short ~x)]
     (.putShort b# 0 x#)
     (.putShort b# 1 x#)
     (.getFloat b# 0)))

(defmacro long-double [x]
  `(Double/longBitsToDouble ~x))

(defmacro int-float [x]
  `(Float/intBitsToFloat ~x))

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt)
(real-vector-blas-plus* FloatVectorEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-vector-math* FloatVectorEngine "s" float-ptr float)
(real-vector-rng* FloatVectorEngine "s" float-ptr float)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt)
(real-vector-blas-plus* DoubleVectorEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-vector-math* DoubleVectorEngine "d" double-ptr double)
(real-vector-rng* DoubleVectorEngine "d" double-ptr double)

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

(defmacro patch-ge-copy [chunk blas method ptr a b]
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

(defmacro integer-ge-blas* [name t ptr cast blas lapack chunk]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (patch-ge-copy ~chunk ~blas ~(cblas "mkl_" t 'omatcopy) ~ptr a# b#)
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

(defmacro ge-axpby [blas method ptr alpha a beta b]
  `(if (< 0 (dim ~a))
     (let [nav-b# (navigator ~b)]
       (. ~blas ~method (byte (int (if (.isColumnMajor nav-b#) \C \R)))
          (byte (int (if (= (navigator ~a) nav-b#) \n \t))) ~(byte (int \n)) (mrows ~b) (ncols ~b)
          ~alpha (~ptr ~a) (stride ~a) ~beta (~ptr ~b) (stride ~b)
          (~ptr ~b) (stride ~b))
       ~b)
     ~b))

(defmacro real-ge-blas* [name t ptr cast blas lapack]
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
       (ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# 1.0 b#))
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

(defmacro real-ge-blas-plus* [name t ptr cast blas lapack ones]
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
       (ge-axpby ~blas ~(cblas "mkl_" t 'omatadd) ~ptr (~cast alpha#) a# (~cast beta#) b#)
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

(defmacro real-ge-lapack* [name t ptr cpp-ptr idx-ptr cast lapack zero-matrix]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~lapack ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (laswp [_# a# ipiv# k1# k2#]
       (ge-laswp ~lapack ~(lapacke t 'laswp) ~ptr ~idx-ptr a# ipiv# k1# k2#))
     (lapmr [_# a# k# forward#]
       (ge-lapm ~lapack ~(lapacke t 'lapmr) ~ptr ~idx-ptr a# k# forward#))
     (lapmt [_# a# k# forward#]
       (ge-lapm ~lapack ~(lapacke t 'lapmt) ~ptr ~idx-ptr a# k# forward#))
     (trf
       ([_# a# ipiv#]
        (ge-trf ~lapack ~(lapacke t 'getrf) ~ptr ~idx-ptr a# ipiv#))
       ([_# _#]
        (dragan-says-ex "Pivotless factorization is not available for general matrices.")))
     (tri [_# lu# ipiv#]
       (ge-tri ~lapack ~(lapacke t 'getri) ~ptr ~idx-ptr lu# ipiv#))
     (trs [_# lu# b# ipiv#]
       (ge-trs ~lapack ~(lapacke t 'getrs) ~ptr ~idx-ptr lu# b# ipiv#))
     (sv [_# a# b# pure#]
       (ge-sv ~lapack ~(lapacke t 'gesv) ~ptr ~idx-ptr a# b# pure#))
     (con [_# lu# _# nrm# nrm1?#]
       (ge-con ~lapack ~(lapacke t 'gecon) ~ptr ~cpp-ptr lu# (~cast nrm#) nrm1?#))
     (qrf [_# a# tau#]
       (ge-lqrf ~lapack ~(lapacke t 'geqrf) ~ptr a# tau#))
     (qrfp [_# a# tau#]
       (ge-lqrf ~lapack ~(lapacke t 'geqrfp) ~ptr a# tau#))
     (qp3 [_# a# jpiv# tau#]
       (ge-qp3 ~lapack ~(lapacke t 'geqp3) ~ptr ~idx-ptr a# jpiv# tau#))
     (gqr [_# a# tau#]
       (or-glqr  ~lapack ~(lapacke t 'orgqr) ~ptr a# tau#))
     (mqr [_# a# tau# c# left#]
       (or-mlqr ~lapack ~(lapacke t 'ormqr) ~ptr a# tau# c# left#))
     (rqf [_# a# tau#]
       (ge-lqrf ~lapack ~(lapacke t 'gerqf) ~ptr a# tau#))
     (grq [_# a# tau#]
       (or-glqr ~lapack ~(lapacke t 'orgrq) ~ptr a# tau#))
     (mrq [_# a# tau# c# left#]
       (or-mlqr ~lapack ~(lapacke t 'ormrq) ~ptr a# tau# c# left#))
     (lqf [_# a# tau#]
       (ge-lqrf ~lapack ~(lapacke t 'gelqf) ~ptr a# tau#))
     (glq [_# a# tau#]
       (or-glqr ~lapack ~(lapacke t 'orglq) ~ptr a# tau#))
     (mlq [_# a# tau# c# left#]
       (or-mlqr ~lapack ~(lapacke t 'ormlq) ~ptr a# tau# c# left#))
     (qlf [_# a# tau#]
       (ge-lqrf ~lapack ~(lapacke t 'geqlf) ~ptr a# tau#))
     (gql [_# a# tau#]
       (or-glqr ~lapack ~(lapacke t 'orgql) ~ptr a# tau#))
     (mql [_# a# tau# c# left#]
       (or-mlqr ~lapack ~(lapacke t 'ormql) ~ptr a# tau# c# left#))
     (ls [_# a# b#]
       (ge-ls ~lapack ~(lapacke t 'gels) ~ptr a# b#))
     (lse [_# a# b# c# d# x#]
       (ge-lse ~lapack ~(lapacke t 'gglse) ~ptr a# b# c# d# x#))
     (gls [_# a# b# d# x# y#]
       (ge-gls ~lapack ~(lapacke t 'ggglm) ~ptr a# b# d# x# y#))
     (ev [_# a# w# vl# vr#]
       (let [vl# (or vl# ~zero-matrix)
             vr# (or vr# ~zero-matrix)]
         (ge-ev ~lapack ~(lapacke t 'geev) ~ptr a# w# vl# vr#)))
     (es [_# a# w# vs#]
       (let [vs# (or vs# ~zero-matrix)]
         (ge-es ~lapack ~(lapacke t 'gees) ~ptr a# w# vs#)))
     (svd
       ([_# a# sigma# u# vt# superb#]
        (let [u# (or u# ~zero-matrix)
              vt# (or vt# ~zero-matrix)]
          (ge-svd ~lapack ~(lapacke t 'gesvd) ~ptr a# sigma# u# vt# superb#)))
       ([_# a# sigma# superb#]
        (ge-svd ~lapack ~(lapacke t 'gesvd) ~ptr a# sigma# ~zero-matrix ~zero-matrix superb#)))
     (sdd
       ([_# a# sigma# u# vt#]
        (let [u# (or u# ~zero-matrix)
              vt# (or vt# ~zero-matrix)]
          (ge-sdd ~lapack ~(lapacke t 'gesdd) ~ptr a# sigma# u# vt#)))
       ([_# a# sigma#]
        (ge-sdd ~lapack ~(lapacke t 'gesdd) ~ptr a# sigma# ~zero-matrix ~zero-matrix)))))

(defmacro matrix-math
  ([method ptr a y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~a))]
          (full-matching-map ~a ~y len# buff-a# buff-y#
                             (. mkl_rt ~method surface# buff-a# buff-y#)
                             (. mkl_rt ~method len# buff-a# buff-y#))))
      ~y))
  ([method ptr a b y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-b# (~ptr ~b 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~a))]
          (full-matching-map ~a ~b ~y len# buff-a# buff-b# buff-y#
                             (. mkl_rt ~method surface# buff-a# buff-b# buff-y#)
                             (. mkl_rt ~method len# buff-a# buff-b# buff-y#))))
      ~y)))

(defmacro matrix-powx [method ptr a b y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             surface# (.surface (region ~a))]
         (full-matching-map ~a ~y len# buff-a# buff-y#
                            (. mkl_rt ~method surface# buff-a# ~b buff-y#)
                            (. mkl_rt ~method len# buff-a# ~b buff-y#))))
     ~y))

(defmacro matrix-linear-frac [method ptr a b scalea shifta scaleb shiftb y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)
             buff-y# (~ptr ~y 0)
             surface# (.surface (region ~a))]
         (full-matching-map
          ~a ~b ~y len# buff-a# buff-b# buff-y#
          (. mkl_rt ~method surface# buff-a# buff-b# ~scalea ~shifta ~scaleb ~shiftb buff-y#)
          (. mkl_rt ~method len# buff-a# buff-b# ~scalea ~shifta ~scaleb ~shiftb buff-y# ))))
     ~y))

(defmacro real-matrix-math* [name t ptr cast]
  `(real-math* ~name ~t ~ptr ~cast matrix-math matrix-linear-frac matrix-powx))

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

(defmacro real-ge-rng* [name t ptr cast]
  `(extend-type ~name
     RandomNumberGenerator
     (rand-uniform [_# rng-stream# lower# upper# a#]
       (matrix-rng ~(cblas "v" t 'RngUniform) ~ptr mkl_rt/VSL_RNG_METHOD_UNIFORM_STD
                   (cast-stream rng-stream#) a# (~cast lower#) (~cast upper#)))
     (rand-normal [_# rng-stream# mu# sigma# a#]
       (matrix-rng ~(cblas "v" t 'RngGaussian) ~ptr mkl_rt/VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
                   (cast-stream rng-stream#) a# (~cast mu#) (~cast sigma#)))))

(deftype FloatGEEngine [])
(real-ge-blas* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt)
(real-ge-blas-plus* FloatGEEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt zero-float)
(real-matrix-math* FloatGEEngine "s" float-ptr float)
(real-ge-rng* FloatGEEngine "s" float-ptr float)

(deftype DoubleGEEngine [])
(real-ge-blas* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt)
(real-ge-blas-plus* DoubleGEEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt zero-double)
(real-matrix-math* DoubleGEEngine "d" double-ptr double)
(real-ge-rng* DoubleGEEngine "d" double-ptr double)

;;TODO
(deftype LongGEEngine [])
(integer-ge-blas* LongGEEngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntGEEngine [])
(integer-ge-blas* IntGEEngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortGEEngine []) ;; TODO

(deftype ByteGEEngine []) ;; TODO

;; ========================= TR matrix engines ===============================================

(defmacro real-tr-blas* [name t ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (tr-lacpy ~lapack ~(lapacke t 'lacpy) ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (tr-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (tr-lan ~lapack ~(lapacke t 'lantr) ~ptr \O a#))
     (nrm2 [_# a#]
       (tr-lan ~lapack ~(lapacke t 'lantr) ~ptr \F a#))
     (nrmi [_# a#]
       (tr-lan ~lapack ~(lapacke t 'lantr) ~ptr \I a#))
     (asum [_# a#]
       (tr-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (uplo-lascl ~lapack ~(lapacke t 'lascl) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (matrix-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# _# a# _# _# _#]
        (dragan-says-ex "Out-of-place mv! is not supported for TR matrices." {:a (info a#)}))
       ([_# a# x#]
        (let [reg# (region a#)]
          (. ~blas ~(cblas t 'trmv) (.layout (navigator a#)) (.uplo reg#) ~(:no-trans blas-transpose)
             (.diag reg#) (ncols a#) (~ptr a#) (stride a#) (~ptr x#) (stride x#))
          x#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for TR matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for TR matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# left#]
        (let [nav# (navigator b#)
              reg# (region a#)
              nav-eq# (= nav# (navigator a#))]
          (. ~blas ~(cblas t 'trmm) (.layout nav#)
             (if left# ~(:left blas-side) ~(:right blas-side))
             (int (if nav-eq# (.uplo reg#) (flip-uplo (.uplo reg#))))
             (if nav-eq# ~(:no-trans blas-transpose) ~(:trans blas-transpose))
             (.diag reg#) (mrows b#) (ncols b#) (~cast alpha#) (~ptr a#) (stride a#) (~ptr b#) (stride b#))
          b#))
       ([_# _# a# _# _# _# _#]
        (dragan-says-ex "Out-of-place mm! is not supported for TR matrices." {:a (info a#)})))))

(defmacro real-tr-blas-plus* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (tr-lan ~lapack ~(lapacke t 'lantr) ~ptr \M a#))
     (sum [_# a#]
       (tr-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (uplo-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (matrix-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for TR matrices." {:a (info a#)}))))

(defmacro real-tr-lapack* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~lapack ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (laswp [_# a# ipiv# k1# k2#]
       (dragan-says-ex "There is no use for pivots when working with TR matrices."))
     (lapmr [_# a# k# forward#]
       (dragan-says-ex "Not implemented for TR matrices."))
     (lapmt [_# a# k# forward#]
       (dragan-says-ex "Not implemented for TR matrices."))
     (trf
       ([_# a# ipiv#]
        (dragan-says-ex "Not implemented for TR matrices."))
       ([_# _#]
        (dragan-says-ex "Not implemented for TR matrices.")))
     (tri [_# a#]
       (tr-tri ~lapack ~(lapacke t 'trtri) ~ptr a#))
     (trs [_# a# b#]
       (tr-trs ~lapack ~(lapacke t 'trtrs) ~ptr a# b#))
     (sv [_# a# b# _#]
       (tr-sv ~lapack ~(cblas t 'trsm) ~ptr a# b#))
     (con [_# a# nrm1?#]
       (tr-con ~lapack ~(lapacke t 'trcon) ~ptr ~cpp-ptr a# nrm1?#))))

(deftype FloatTREngine [])
(real-tr-blas* FloatTREngine "s" float-ptr float mkl_rt mkl_rt)
(real-tr-blas-plus* FloatTREngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-tr-lapack* FloatTREngine "s" float-ptr cpp/float-pointer float mkl_rt mkl_rt)
(real-matrix-math* FloatTREngine "s" float-ptr float)

(deftype DoubleTREngine [])
(real-tr-blas* DoubleTREngine "d" double-ptr double mkl_rt mkl_rt)
(real-tr-blas-plus* DoubleTREngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-tr-lapack* DoubleTREngine "d" double-ptr cpp/double-pointer double mkl_rt mkl_rt)
(real-matrix-math* DoubleTREngine "d" double-ptr double)

(deftype LongTREngine [])
;;(integer-tr-blas* LongTREngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntTREngine [])
;;(integer-tr-blas* IntTREngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortTREngine []) ;; TODO

(deftype ByteTREngine []) ;; TODO

;; ========================= SY matrix engines ===============================================

(defmacro real-sy-blas* [name t ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (sy-lacpy ~lapack ~(lapacke t 'lacpy) ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (sy-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (sy-lan ~lapack ~(lapacke t 'lansy) ~ptr \O a#))
     (nrm2 [_# a#]
       (sy-lan ~lapack ~(lapacke t 'lansy) ~ptr \F a#))
     (nrmi [_# a#]
       (sy-lan ~lapack ~(lapacke t 'lansy) ~ptr \I a#))
     (asum [_# a#]
       (sy-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (uplo-lascl ~lapack ~(lapacke t 'lascl) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (matrix-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (. ~blas ~(cblas t 'symv) (.layout (navigator a#)) (.uplo (region a#)) (mrows a#)
           (~cast alpha#) (~ptr a#) (stride a#) (~ptr x#) (stride x#) (~cast beta#) (~ptr y#) (stride y#))
        y#)
       ([_# a# _#]
        (dragan-says-ex "In-place mv! is not supported for SY matrices." {:a (info a#)})))
     (rk
       ([_# alpha# x# y# a#]
        (. ~blas ~(cblas t 'syr2) (.layout (navigator a#)) (.uplo (region a#)) (mrows a#)
           (~cast alpha#) (~ptr x#) (stride x#) (~ptr y#) (stride y#) (~ptr a#) (stride a#))
        a#)
       ([_# alpha# x# a#]
        (. ~blas ~(cblas t 'syr) (.layout (navigator a#)) (.uplo (region a#)) (mrows a#)
           (~cast alpha#) (~ptr x#) (stride x#) (~ptr a#) (stride a#))
        a#))
     (srk [_# alpha# a# beta# c#]
       (if (and (= :ge (matrix-type a#)) (= :sy (matrix-type c#)))
         (let [nav# (navigator c#)]
           (.~blas ~(cblas t 'syrk) (.layout nav#) (.uplo (region c#))
                   (if (= nav# (navigator a#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
                   (mrows c#) (ncols a#)
                   (~cast alpha#) (~ptr a#) (stride a#) (~cast beta#) (~ptr c#) (stride c#))
           c#)
         (throw (ex-info "sy-rk is only available for symmetric matrices." {:c (info c#)})))
       c#)
     (mm
       ([_# alpha# a# b# _#]
        (dragan-says-ex "In-place mv! is not supported for SY matrices." {:a (info a#)}))
       ([_# alpha# a# b# beta# c# left#]
        (let [nav-c# (navigator c#)
              uplo# (int (if (= nav-c# (navigator a#)) (.uplo (region a#)) (flip-uplo (.uplo (region a#)))))]
          (if (= nav-c# (navigator b#))
            (. ~blas ~(cblas t 'symm) (.layout nav-c#) (if left# ~(:left blas-side) ~(:right blas-side))
               uplo# (mrows c#) (ncols c#)
               (~cast alpha#) (~ptr a#) (stride a#) (~ptr b#) (stride b#)
               (~cast beta#) (~ptr c#) (stride c#))
            (dragan-says-ex "Both GE matrices in symmetric multiplication must have the same orientation."
                            {:b (info b#) :c (info c#)}))
          c#)))))

(defmacro real-sy-blas-plus* [name t ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (sy-lan ~lapack ~(lapacke t 'lansy) ~ptr \M a#))
     (sum [_# a#]
       (sy-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (uplo-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (matrix-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for SY matrices." {:a (info a#)}))))

(defmacro real-sy-lapack* [name t ptr cpp-ptr idx-ptr cast mkl zero-matrix]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# a# ipiv#]
        (sy-trx ~mkl ~(lapacke t 'sytrf) ~ptr ~idx-ptr a# ipiv#))
       ([_# a#]
        (sy-trx ~mkl ~(lapacke t 'potrf) ~ptr a#)))
     (trfx [_# a#]
       (sy-trfx ~mkl ~(lapacke t 'potrf) ~ptr a#))
     (tri
       ([_# ldl# ipiv#]
        (sy-trx ~mkl ~(lapacke t 'sytri) ~ptr ~idx-ptr ldl# ipiv#))
       ([_# gg#]
        (sy-trx ~mkl ~(lapacke t 'potri) ~ptr gg#)))
     (trs
       ([_# ldl# b# ipiv#]
        (sy-trs ~mkl ~(lapacke t 'sytrs) ~ptr ~idx-ptr ldl# b# ipiv#))
       ([_# gg# b# ]
        (sy-trs ~mkl ~(lapacke t 'potrs) ~ptr gg# b#)))
     (sv
       ([_# a# b# pure#]
        (sy-sv ~mkl ~(lapacke t 'posv) ~(lapacke t 'sysv) ~ptr ~idx-ptr a# b# pure#))
       ([_# a# b#]
        (sy-sv ~mkl ~(lapacke t 'posv) ~ptr a# b#)))
     (con
       ([_# ldl# ipiv# nrm# _#]
        (sy-con ~mkl ~(lapacke t 'sycon) ~ptr ~cpp-ptr ~idx-ptr ldl# ipiv# (~cast nrm#)))
       ([_# gg# nrm# _#]
        (sy-con ~mkl ~(lapacke t 'pocon) ~ptr ~cpp-ptr gg# (~cast nrm#))))
     (ev [_# a# w# vl# vr#]
       (let [v# (or vl# vr# ~zero-matrix)]
         (sy-ev ~mkl ~(lapacke t 'syevd) ~(lapacke t 'syevr) ~ptr a# w# v#)))))

(deftype FloatSYEngine [])
(real-sy-blas* FloatSYEngine "s" float-ptr float mkl_rt mkl_rt)
(real-sy-blas-plus* FloatSYEngine "s" float-ptr float mkl_rt mkl_rt ones-float)
(real-sy-lapack* FloatSYEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt zero-float)
(real-matrix-math* FloatSYEngine "s" float-ptr float)

(deftype DoubleSYEngine [])
(real-sy-blas* DoubleSYEngine "d" double-ptr double mkl_rt mkl_rt)
(real-sy-blas-plus* DoubleSYEngine "d" double-ptr double mkl_rt mkl_rt ones-double)
(real-sy-lapack* DoubleSYEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt zero-double)
(real-matrix-math* DoubleSYEngine "d" double-ptr double)

;;TODO
(deftype LongSYEngine [])
;;(integer-tr-blas* LongSYEngine "d" double-ptr long-double mkl_rt mkl_rt 1)

(deftype IntSYEngine [])
;;(integer-tr-blas* IntSYEngine "s" float-ptr int-float mkl_rt mkl_rt 1)

(deftype ShortSYEngine []);; TODO

(deftype ByteSYEngine []);; TODO

;; ============================ GB matrix engines ==================================================

(defmacro real-gb-blas* [name t ptr cpp-ptr cast blas lapack]
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
       (gb-lan ~lapack ~(lapacke t "" 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \O a#))
     (nrm2 [_# a#]
       (gb-lan ~lapack ~(lapacke t "" 'langb_64) ~(cblas t 'nrm2) ~ptr ~cpp-ptr \F a#))
     (nrmi [_# a#]
       (gb-lan ~lapack ~(lapacke t "" 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \I a#))
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

(defmacro real-gb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (gb-lan ~lapack ~(lapacke "" t 'langb_64) ~(cblas 'cblas_i t 'amax) ~ptr ~cpp-ptr \M a#))
     (sum [_# a#]
       (gb-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (gb-laset ~lapack ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (gb-axpby ~blas ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for GB matrices." {:a (info a#)}))))

(defmacro real-gb-lapack* [name t ptr cpp-ptr idx-ptr cast lapack]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~lapack ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# a# ipiv#]
        (gb-trf ~lapack ~(lapacke t 'gbtrf) ~ptr ~idx-ptr a# ipiv#))
       ([_# _#]
        (dragan-says-ex "Pivotless factorization is not available for GB matrices.")))
     (tri [_# _#]
       (dragan-says-ex "Inverse is not available for banded matrices."))
     (trs [_# lu# b# ipiv#]
       (gb-trs ~lapack ~(lapacke t 'gbtrs) ~ptr ~idx-ptr lu# b# ipiv#))
     (sv [_# a# b# pure#]
       (gb-sv ~lapack ~(lapacke t 'gbsv) ~ptr ~idx-ptr a# b# pure#))
     (con [_# ldl# ipiv# nrm# nrm1?#]
       (gb-con ~lapack ~(lapacke t 'gbcon) ~ptr ~cpp-ptr ~idx-ptr ldl# ipiv# (~cast nrm#) nrm1?#))))

(deftype FloatGBEngine [])
(real-gb-blas* FloatGBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-gb-blas-plus* FloatGBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-gb-lapack* FloatGBEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatGBEngine "s" float-ptr float)

(deftype DoubleGBEngine [])
(real-gb-blas* DoubleGBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-gb-blas-plus* DoubleGBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-gb-lapack* DoubleGBEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleGBEngine "d" double-ptr double)

(deftype LongGBEngine [])
(deftype IntGBEngine [])
(deftype ShortGBEngine [])
(deftype ByteGBEngine [])

;; ============================ SB matrix engines ==================================================

(defmacro real-sb-blas* [name t ptr cpp-ptr cast blas lapack]
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

(defmacro real-sb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
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

(defmacro real-sb-lapack* [name t ptr cpp-ptr cast lapack]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~lapack ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# _# _#]
        (dragan-says-ex "Pivots are not supported in SB trf."))
       ([_# a#]
        (sb-trf ~lapack ~(lapacke t 'pbtrf) ~ptr a#)))
     (tri
       ([_# _#]
        (dragan-says-ex "Inverse is not available for banded matrices."))
       ([_# _# _#]
        (dragan-says-ex "Inverse is not available for banded matrices.")))
     (trs [_# gg# b#]
       (sb-trs ~lapack ~(lapacke t 'pbtrs) ~ptr gg# b#))
     (sv
       ([_# a# b# pure#]
        (sb-sv ~lapack ~(lapacke t 'pbsv) ~ptr a# b# pure#))
       ([_# a# b#]
        (sb-sv ~lapack ~(lapacke t 'pbsv) ~ptr a# b# false)))
     (con [_# gg# nrm# _#]
       (sb-con ~lapack ~(lapacke t 'pbcon) ~ptr ~cpp-ptr gg# (~cast nrm#)))))

(deftype FloatSBEngine [])
(real-sb-blas* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-sb-blas-plus* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-sb-lapack* FloatSBEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatSBEngine "s" float-ptr float)

(deftype DoubleSBEngine [])
(real-sb-blas* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-sb-blas-plus* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-sb-lapack* DoubleSBEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleSBEngine "d" double-ptr double)

(deftype LongSBEngine [])
(deftype IntSBEngine [])
(deftype ShortSBEngine [])
(deftype ByteSBEngine [])

;; ============================ TB matrix engines ====================================================

(defmacro real-tb-blas* [name t ptr cpp-ptr cast blas lapack]
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

(defmacro real-tb-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
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

(defmacro real-tb-lapack* [name t ptr cpp-ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (tri
       ([_# _#]
        (dragan-says-ex "Inverse is not available for banded matrices."))
       ([_# _# _#]
        (dragan-says-ex "Inverse is not available for banded matrices.")))
     (trs [_# a# b#]
       (tb-trs ~mkl ~(lapacke t 'tbtrs) ~ptr a# b#))
     (sv [_# a# b# _#]
       (tb-sv ~mkl ~(cblas t 'tbsv) ~ptr a# b#))
     (con [_# a# nrm1?#]
       (tb-con ~mkl ~(lapacke t 'tbcon) ~ptr ~cpp-ptr a# nrm1?#))))

(deftype FloatTBEngine [])
(real-tb-blas* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-tb-blas-plus* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-tb-lapack* FloatTBEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatTBEngine "s" float-ptr float)

(deftype DoubleTBEngine [])
(real-tb-blas* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-tb-blas-plus* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-tb-lapack* DoubleTBEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleTBEngine "d" double-ptr double)

(deftype LongTBEngine [])
(deftype IntTBEngine [])
(deftype ShortTBEngine [])
(deftype ByteTBEngine [])

;; ============================ TP matrix engines ====================================================

(defmacro real-tp-blas* [name t ptr cpp-ptr cast blas lapack]
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

(defmacro real-tp-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
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

(defmacro real-tp-lapack* [name t ptr cpp-ptr cast lapack]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (packed-lasrt ~lapack ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (laswp [_# a# ipiv# k1# k2#]
       (dragan-says-ex "There is no use for pivots when working with TP matrices."))
     (tri [_# a#]
       (tp-tri ~lapack ~(lapacke t 'tptri) ~ptr a#))
     (trs [_# a# b#]
       (tp-trs ~lapack ~(lapacke t 'tptrs) ~ptr a# b#))
     (sv [_# a# b# _#]
       (tp-sv ~lapack ~(cblas t 'tpsv) ~ptr a# b#))
     (con [_# a# nrm1?#]
       (tp-con ~lapack ~(lapacke t 'tpcon) ~ptr ~cpp-ptr a# nrm1?#))))

(deftype FloatTPEngine [])
(real-tp-blas* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-tp-blas-plus* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-tp-lapack* FloatTPEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatTPEngine "s" float-ptr float)

(deftype DoubleTPEngine [])
(real-tp-blas* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-tp-blas-plus* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-tp-lapack* DoubleTPEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleTPEngine "d" double-ptr double)

(deftype LongTPEngine [])
(deftype IntTPEngine [])
(deftype ShortTPEngine [])
(deftype ByteTPEngine [])

;; ============================ SP matrix engines ====================================================

(defmacro real-sp-blas* [name t ptr cpp-ptr cast blas lapack]
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

(defmacro real-sp-blas-plus* [name t ptr cpp-ptr cast blas lapack ones]
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

(defmacro real-sp-lapack* [name t ptr cpp-ptr idx-ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (packed-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# a# ipiv#]
        (sp-trx ~mkl ~(lapacke t 'sptrf) ~ptr ~idx-ptr a# ipiv#))
       ([_# a#]
        (sp-trx ~mkl ~(lapacke t 'pptrf) ~ptr a#)))
     (trfx [_# a#]
       (sp-trfx ~mkl ~(lapacke t 'pptrf) ~ptr a#))
     (tri
       ([_# ldl# ipiv#]
        (sp-trx ~mkl ~(lapacke t 'sptri) ~ptr ~idx-ptr ldl# ipiv#))
       ([_# gg#]
        (sp-trx ~mkl ~(lapacke t 'pptri) ~ptr gg#)))
     (trs
       ([_# ldl# b# ipiv#]
        (sp-trs ~mkl ~(lapacke t 'sptrs) ~ptr ~idx-ptr ldl# b# ipiv#))
       ([_# gg# b# ]
        (sp-trs ~mkl ~(lapacke t 'pptrs) ~ptr gg# b#)))
     (sv
       ([_# a# b# pure#]
        (sp-sv ~mkl ~(lapacke t 'ppsv) ~(lapacke t 'spsv) ~ptr ~idx-ptr a# b# pure#))
       ([_# a# b#]
        (sp-sv ~mkl ~(lapacke t 'ppsv) ~ptr a# b#)))
     (con
       ([_# ldl# ipiv# nrm# _#]
        (sp-con ~mkl ~(lapacke t 'spcon) ~ptr ~cpp-ptr ~idx-ptr ldl# ipiv# (~cast nrm#)))
       ([_# gg# nrm# _#]
        (sp-con ~mkl ~(lapacke t 'ppcon) ~ptr ~cpp-ptr gg# (~cast nrm#))))))

(deftype FloatSPEngine [])
(real-sp-blas* FloatSPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt)
(real-sp-blas-plus* FloatSPEngine "s" float-ptr cpp/float-ptr float mkl_rt mkl_rt ones-float)
(real-sp-lapack* FloatSPEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatSPEngine "s" float-ptr float)

(deftype DoubleSPEngine [])
(real-sp-blas* DoubleSPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt)
(real-sp-blas-plus* DoubleSPEngine "d" double-ptr cpp/double-ptr double mkl_rt mkl_rt ones-double)
(real-sp-lapack* DoubleSPEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleSPEngine "d" double-ptr double)

(deftype LongSPEngine [])
(deftype IntSPEngine [])
(deftype ShortSPEngine [])
(deftype ByteSPEngine [])

;; ============================ GD matrix engines ==================================================

(defmacro real-gd-blas* [name t ptr cpp-ptr cast mkl]
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
       (diagonal-amax ~mkl ~(cblas 'cblas_i t 'amax) ~ptr a#))
     (nrm2 [_# a#]
       (diagonal-method ~mkl ~(cblas t 'nrm2) ~ptr a#))
     (nrmi [_# a#]
       (diagonal-amax ~mkl ~(cblas 'cblas_i t 'amax) ~ptr a#))
     (asum [_# a#]
       (diagonal-method ~mkl ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~mkl ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~mkl ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (gd-mv ~mkl ~(cblas t 'sbmv) ~ptr (~cast alpha#) a# x# (~cast beta#) y#))
       ([_# a# x#]
        (gd-mv ~mkl ~(lapacke "" t 'lascl2_64) ~ptr a# x#)))
     (rk
       ([_# _# _# _# a#]
        (dragan-says-ex "rk! is not supported for GD matrices." {:a (info a#)}))
       ([_# _# _# a#]
        (dragan-says-ex "rk! is not supported for GD matrices." {:a (info a#)})))
     (mm
       ([_# alpha# a# b# left#]
        (gd-mm ~mkl ~(lapacke "" t 'lascl2_64) ~(cblas t 'tbmv) ~ptr (~cast alpha#) a# b# left#))
       ([_# alpha# a# b# beta# c# left#]
        (gd-mm ~mkl ~(cblas t 'sbmv) ~ptr (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro real-diagonal-blas-plus* [name t ptr cast mkl ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (diagonal-amax ~mkl ~(cblas 'cblas_i t 'amax) ~ptr a#))
     (sum [_# a#]
       (diagonal-method ~mkl ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (diagonal-laset ~mkl ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (diagonal-axpby ~mkl ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for diagonal matrices." {:a (info a#)}))))

(defmacro real-gd-lapack* [name t ptr cpp-ptr cast mkl]
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
       (gd-sv ~mkl ~(math t 'Div) ~(cblas t 'tbsv) ~ptr a# b#))
     (con [_# a# nrm1?#]
       (gd-con ~mkl ~(lapacke t 'tbcon) ~ptr ~cpp-ptr a# nrm1?#))))

(deftype FloatGDEngine [])
(real-gd-blas* FloatGDEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-diagonal-blas-plus* FloatGDEngine "s" float-ptr float  mkl_rt ones-float)
(real-gd-lapack* FloatGDEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-matrix-math* FloatGDEngine "s" float-ptr float)

(deftype DoubleGDEngine [])
(real-gd-blas* DoubleGDEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-diagonal-blas-plus* DoubleGDEngine "d" double-ptr double mkl_rt ones-double)
(real-gd-lapack* DoubleGDEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-matrix-math* DoubleGDEngine "d" double-ptr double)

(deftype LongGDEngine [])
(deftype IntGDEngine [])
(deftype ShortGDEngine [])
(deftype ByteGDEngine [])

(defmacro real-tridiagonal-blas* [name t ptr cpp-ptr cast mkl]
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
       (tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \O a#))
     (nrm2 [_# a#]
       (diagonal-method ~mkl ~(cblas t 'nrm2) ~ptr a#))
     (nrmi [_# a#]
       (tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \I a#))
     (asum [_# a#]
       (diagonal-method ~mkl ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~mkl ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~mkl ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr
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
        (tridiagonal-mm ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro real-gt-lapack* [name t ptr cpp-ptr idx-ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (diagonal-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# a# ipiv#]
        (gt-trf ~mkl ~(lapacke t 'gttrf) ~ptr ~idx-ptr a# ipiv#))
       ([_# _#]
        (dragan-says-ex "Pivotless factorization is not available for DT matrices.")))
     (tri [_# _#]
       (dragan-says-ex "Inverse is not available for GT matrices."))
     (trs [_# lu# b# ipiv#]
       (gt-trs ~mkl ~(lapacke t 'gttrs) ~ptr ~idx-ptr lu# b# ipiv#))
     (sv [_# a# b# pure#]
       (gt-sv ~mkl ~(lapacke t 'gtsv) ~ptr a# b# pure#))
     (con [_# ldl# ipiv# nrm# nrm1?#]
       (gt-con ~mkl ~(lapacke t 'gtcon) ~ptr ~cpp-ptr ~idx-ptr ldl# ipiv# (~cast nrm#) nrm1?#))))

(deftype FloatGTEngine [])
(real-tridiagonal-blas* FloatGTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-diagonal-blas-plus* FloatGTEngine "s" float-ptr float mkl_rt ones-float)
(real-gt-lapack* FloatGTEngine "s" float-ptr cpp/float-ptr int-ptr float mkl_rt)
(real-matrix-math* FloatGTEngine "s" float-ptr float)

(deftype DoubleGTEngine [])
(real-tridiagonal-blas* DoubleGTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-diagonal-blas-plus* DoubleGTEngine "d" double-ptr double mkl_rt ones-double)
(real-gt-lapack* DoubleGTEngine "d" double-ptr cpp/double-ptr int-ptr double mkl_rt)
(real-matrix-math* DoubleGTEngine "d" double-ptr double)

(deftype LongGTEngine [])
(deftype IntGTEngine [])
(deftype ShortGTEngine [])
(deftype ByteGTEngine [])

(defmacro real-dt-lapack* [name t ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (matrix-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (trf
       ([_# _# _#]
        (dragan-says-ex "Pivoted factorization is not available for GT matrices."))
       ([_# a#]
        (dt-trf ~mkl ~(lapacke "" t 'dttrfb_64) ~ptr a#)))
     (tri [_# _#]
       (dragan-says-ex "Inverse is not available for GT matrices."))
     (trs [_# lu# b#]
       (dt-trs ~mkl ~(lapacke "" t 'dttrsb_64) ~ptr lu# b#))
     (sv [_# a# b# pure#]
       (dt-sv ~mkl ~(lapacke "" t 'dtsvb_64) ~ptr a# b# pure#)) ;;TODO lapacke "" should be (mkl t 'etc)
     (con [_# _# _# _# _#]
       (dragan-says-ex "Condition number is not available for DT matrices."))))

(deftype FloatDTEngine [])
(real-tridiagonal-blas* FloatDTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-diagonal-blas-plus* FloatDTEngine "s" float-ptr float mkl_rt ones-float)
(real-dt-lapack* FloatDTEngine "s" float-ptr float mkl_rt)
(real-matrix-math* FloatDTEngine "s" float-ptr float)

(deftype DoubleDTEngine [])
(real-tridiagonal-blas* DoubleDTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-diagonal-blas-plus* DoubleDTEngine "d" double-ptr double mkl_rt ones-double)
(real-dt-lapack* DoubleDTEngine "d" double-ptr double mkl_rt)
(real-matrix-math* DoubleDTEngine "d" double-ptr double)

(deftype LongDTEngine [])
(deftype IntDTEngine [])
(deftype ShortDTEngine [])
(deftype ByteDTEngine [])

(defmacro real-st-blas* [name t ptr cpp-ptr cast mkl]
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
       (tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \O a#))
     (nrm2 [_# a#]
       (tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \F a#))
     (nrmi [_# a#]
       (tridiagonal-lan ~mkl ~(lapacke "" t 'langt_64) ~ptr \I a#))
     (asum [_# a#]
       (st-asum ~mkl ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~mkl ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~mkl ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr
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
        (tridiagonal-mm ~mkl ~(lapacke "" t 'lagtm_64) ~ptr ~cpp-ptr
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(defmacro real-st-blas-plus* [name t ptr cast mkl ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (diagonal-amax ~mkl ~(cblas 'cblas_i t 'amax) ~ptr a#))
     (sum [_# a#]
       (st-sum ~mkl ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (diagonal-laset ~mkl ~(lapacke t 'laset) ~ptr (~cast alpha#) a#))
     (axpby [_# alpha# a# beta# b#]
       (diagonal-axpby ~mkl ~(cblas t 'axpby) ~ptr (~cast alpha#) a# (~cast beta#) b#))
     (trans [_# a#]
       (dragan-says-ex "In-place transpose is not available for ST matrices." {:a (info a#)}))))

(defmacro real-st-lapack* [name t ptr cast mkl]
  `(extend-type ~name
     Lapack
     (srt [_# a# increasing#]
       (diagonal-lasrt ~mkl ~(lapacke t 'lasrt) ~ptr a# increasing#))
     (laswp [_# _# _# _# _#]
       (dragan-says-ex "Pivoted swap is not available for ST matrices."))
     (trf
       ([_# _# _#]
        (dragan-says-ex "Pivoted factorization is not available for ST matrices."))
       ([_# a#]
        (st-trf ~mkl ~(lapacke t 'pttrf) ~ptr a#)))
     (tri [_# _#]
       (dragan-says-ex "Inverse is not available for ST matrices."))
     (trs [_# lu# b#]
       (st-trs ~mkl ~(lapacke t 'pttrs) ~ptr lu# b#))
     (sv [_# a# b# pure#]
       (st-sv ~mkl ~(lapacke t 'ptsv) ~ptr a# b# pure#))
     (con [_# _# _# _# _#]
       (dragan-says-ex "Condition number is not available for ST matrices."))))

(deftype FloatSTEngine [])
(real-st-blas* FloatSTEngine "s" float-ptr cpp/float-ptr float mkl_rt)
(real-st-blas-plus* FloatSTEngine "s" float-ptr float mkl_rt ones-float)
(real-st-lapack* FloatSTEngine "s" float-ptr float mkl_rt)
(real-matrix-math* FloatSTEngine "s" float-ptr float)

(deftype DoubleSTEngine [])
(real-st-blas* DoubleSTEngine "d" double-ptr cpp/double-ptr double mkl_rt)
(real-st-blas-plus* DoubleSTEngine "d" double-ptr double mkl_rt ones-double)
(real-st-lapack* DoubleSTEngine "d" double-ptr double mkl_rt)
(real-matrix-math* DoubleSTEngine "d" double-ptr double)

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
       (real/entry (indices x#) (iamax (engine (entries x#)) (entries x#))))
     (iamin [this# x#]
       (real/entry (indices x#) (iamin (engine (entries x#)) (entries x#))))
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
(real-vector-math* FloatCSVectorEngine "s" float-ptr float)
(real-cs-vector-sparse-blas* FloatCSVectorEngine "s" float-ptr int-ptr mkl_rt)

(deftype DoubleCSVectorEngine [])
(real-cs-vector-blas* DoubleCSVectorEngine "d" double-ptr int-ptr double mkl_rt ones-double)
(real-cs-blas-plus* DoubleCSVectorEngine "d" double-ptr double)
(real-vector-math* DoubleCSVectorEngine "d" double-ptr double)
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
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  Factory
  (create-vector [this n init]
    (let-release [res (real-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-vector [this master buf-ptr n ofst strd]
    (real-block-vector this master buf-ptr n ofst strd))
  (create-ge [this m n column? init]
    (let-release [res (real-ge-matrix this m n column?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (real-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (real-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (real-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-banded [this m n kl ku matrix-type column? init]
    (let-release [res (real-banded-matrix this m n kl ku column? matrix-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-gb [this m n kl ku lower? init]
    (let-release [res (real-banded-matrix this m n kl ku lower?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tb [this n k column? lower? diag-unit? init]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (let-release [res (real-tb-matrix this n k column? lower? diag-unit?)]
        (when init
          (.initialize da (.buffer ^Block res)))
        res)
      (dragan-says-ex "TB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-sb [this n k column? lower? init]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (let-release [res (real-sb-matrix this n k column? lower?)]
        (when init
          (.initialize da (.buffer ^Block res)))
        res)
      (dragan-says-ex "SB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-packed [this n matrix-type column? lower? diag-unit? init]
    (let-release [res (real-packed-matrix this n column? lower? diag-unit? matrix-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tp [this n column? lower? diag-unit? init]
    (let-release [res (real-packed-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-sp [this n column? lower? init]
    (let-release [res (real-packed-matrix this n column? lower?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-diagonal [this n matrix-type init]
    (let-release [res (real-diagonal-matrix this n matrix-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
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
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  Factory
  (create-vector [this n init]
    (let-release [res (integer-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-vector [this master buf-ptr n ofst strd]
    (integer-block-vector this master buf-ptr n ofst strd))
  (create-ge [this m n column? init]
    (let-release [res (integer-ge-matrix this m n column?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (integer-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (integer-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (integer-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (.buffer ^Block res)))
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

(def float-accessor (->FloatPointerAccessor malloc! free!))
(def double-accessor (->DoublePointerAccessor malloc! free!))
(def int-accessor (->IntPointerAccessor malloc! free!))
(def long-accessor (->LongPointerAccessor malloc! free!))
(def short-accessor (->ShortPointerAccessor malloc! free!))
(def byte-accessor (->BytePointerAccessor malloc! free!))

(def mkl-int (->MKLIntegerFactory mkl-int int-accessor (->IntVectorEngine) (->IntGEEngine)
                                  (->IntTREngine) (->IntSYEngine)
                                  (->IntGBEngine) (->IntSBEngine) (->IntTBEngine)
                                  (->IntSPEngine) (->IntTPEngine) (->IntGDEngine)
                                  (->IntGTEngine) (->IntDTEngine) (->IntSTEngine)))

(def mkl-long (->MKLIntegerFactory mkl-int long-accessor (->LongVectorEngine) (->LongGEEngine)
                                   (->LongTREngine) (->LongSYEngine)
                                   (->LongGBEngine) (->LongSBEngine) (->LongTBEngine)
                                   (->LongSPEngine) (->LongTPEngine) (->LongGDEngine)
                                   (->LongGTEngine) (->LongDTEngine) (->LongSTEngine)))

(def mkl-short (->MKLIntegerFactory mkl-int short-accessor (->ShortVectorEngine) (->ShortGEEngine)
                                    (->ShortTREngine) (->ShortSYEngine)
                                    (->ShortGBEngine) (->ShortSBEngine) (->ShortTBEngine)
                                    (->ShortSPEngine) (->ShortTPEngine) (->ShortGDEngine)
                                    (->ShortGTEngine) (->ShortDTEngine) (->ShortSTEngine)))

(def mkl-byte (->MKLIntegerFactory mkl-int byte-accessor (->ByteVectorEngine) (->ByteGEEngine)
                                   (->ByteTREngine) (->ByteSYEngine)
                                   (->ByteGBEngine) (->ByteSBEngine) (->ByteTBEngine)
                                   (->ByteSPEngine) (->ByteTPEngine) (->ByteGDEngine)
                                   (->ByteGTEngine) (->ByteDTEngine) (->ByteSTEngine)))

(def mkl-float (->MKLRealFactory mkl-int float-accessor (->FloatVectorEngine) (->FloatGEEngine)
                                 (->FloatTREngine) (->FloatSYEngine)
                                 (->FloatGBEngine) (->FloatSBEngine) (->FloatTBEngine)
                                 (->FloatSPEngine) (->FloatTPEngine) (->FloatGDEngine)
                                 (->FloatGTEngine) (->FloatDTEngine) (->FloatSTEngine)
                                 (->FloatCSVectorEngine) (->FloatCSREngine)))

(def mkl-double (->MKLRealFactory mkl-int double-accessor (->DoubleVectorEngine) (->DoubleGEEngine)
                                  (->DoubleTREngine) (->DoubleSYEngine)
                                  (->DoubleGBEngine) (->DoubleSBEngine) (->DoubleTBEngine)
                                  (->DoubleSPEngine) (->DoubleTPEngine) (->DoubleGDEngine)
                                  (->DoubleGTEngine) (->DoubleDTEngine) (->DoubleSTEngine)
                                  (->DoubleCSVectorEngine)  (->DoubleCSREngine)))

(extend-pointer FloatPointer mkl-float)
(extend-pointer DoublePointer mkl-double)
(extend-pointer LongPointer mkl-long)
(extend-pointer IntPointer mkl-int)
(extend-pointer ShortPointer mkl-short)
(extend-pointer BytePointer mkl-byte)
