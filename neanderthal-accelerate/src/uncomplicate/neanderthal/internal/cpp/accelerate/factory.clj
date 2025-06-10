;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.accelerate.factory
  (:refer-clojure :exclude [abs])
  (:require [clojure.string :refer [lower-case]]
            [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release view]]
             [utils :refer [dragan-says-ex with-check generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap! extract]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [pointer long-pointer float-pointer double-pointer clong-pointer get-entry malloc! free!]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols cols rows matrix-type entry entry!]]
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
             [lapack :refer :all]
             [blas :refer :all]
             [factory :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.openblas.factory :as openblas])
  (:import java.nio.ByteBuffer
           [uncomplicate.neanderthal.internal.api DataAccessor Vector LayoutNavigator Region
            GEMatrix UploMatrix DenseStorage]
           [uncomplicate.javacpp.accelerate.global thread blas_new lapack_new vforce vdsp]))

(let [openblas-load (System/getProperty "org.bytedeco.openblas.load")
      library-path (System/getProperty "java.library.path") ]
  (try
    (System/setProperty "java.library.path" (str library-path ":/usr/lib"))
    (System/setProperty "org.bytedeco.openblas.load" "blas")
    (import 'org.bytedeco.openblas.global.openblas_full)
    (catch Exception e
      (System/setProperty "java.library.path" (str library-path))
      (System/setProperty "org.bytedeco.openblas.load" (str openblas-load)))))

(defn accelerate-threading? []
  (let [threading (thread/BLASGetThreading)]
    (cond
      (= thread/BLAS_THREADING_MULTI_THREADED threading) true
      (= thread/BLAS_THREADING_SINGLE_THREADED threading) false
      :default (throw (dragan-says-ex "Accelerate returned a threading option unknown to this library."
                                      {:threading threading
                                       :max-threading-options thread/BLAS_THREADING_MAX_OPTIONS})))))

(defn threading? []
  (and (accelerate-threading?)
       (openblas/threading?)))

(defn accelerate-threading! [multi-threading?]
  (let [threading (if multi-threading?
                    thread/BLAS_THREADING_MULTI_THREADED
                    thread/BLAS_THREADING_SINGLE_THREADED)]
    (if (= 0 (thread/BLASSetThreading threading))
      multi-threading?
      (throw (dragan-says-ex "The current platform does not support the requested threading model."
                             {:threading threading
                              :max-threading-options thread/BLAS_THREADING_MAX_OPTIONS})))))

(defn threading!
  ([param]
   (uncomplicate.neanderthal.internal.cpp.openblas.factory/threading! param)
   (cond
       (or (= false param) (= 1 param)) (accelerate-threading! false)
       (or (= true param) (< 1 param)) (accelerate-threading! true)
       :default (throw (dragan-says-ex "Threading model is not supported by Accelerate." {:threading param}))))
  ([]
   (openblas/threading!)
   (accelerate-threading! true)))

;; ============ Vector Engines ============================================

(defn vvforce [suffix name]
  (symbol (format "vv%s%s" name suffix)))

(defn vvdsp [suffix name]
  (symbol (format "vDSP_%s%s" name suffix)))

(defmacro vector-vforce
  ([method ptr a y]
   `(do
      (check-stride ~a ~y)
      (. vforce ~method (~ptr ~y) (~ptr ~a) (cpp/int-ptr (pointer (int (dim ~a)))))
      ~y))
  ([method ptr a b y]
   `(do
      (check-stride ~a ~b ~y)
      (. vforce ~method (~ptr ~y) (~ptr ~a) (~ptr ~b) (cpp/int-ptr (pointer (int (dim ~a)))))
      ~y)))

(defmacro vector-vsdsp [method ptr cpp-ptr a s y]
  `(do
     (. vdsp ~method (~ptr ~a) (stride ~a) (~cpp-ptr (pointer ~s)) (~ptr ~y) (stride ~y) (dim ~a))
     ~y))

(defmacro vector-vdsp
  ([method ptr a y]
   `(do
      (. vdsp ~method (~ptr ~a) (stride ~a) (~ptr ~y) (stride ~y) (dim ~a))
      ~y))
  ([method ptr a b y]
   `(do
      (. vdsp ~method (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) (~ptr ~y) (stride ~y) (dim ~a))
      ~y))
  ([method ptr a b zero y]
   `(let [zero# (~ptr ~zero)]
      (. vdsp ~method (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) zero# 0 zero# 0 (~ptr ~y) (stride ~y) (dim ~a))
      ~y)))

(defmacro vector-linear-frac [dsuffix ptr cpp-ptr cast a b scalea shifta scaleb shiftb y]
  `(let [a# (~ptr ~a)
         strd-a# (stride ~a)
         y# (~ptr ~y)
         strd-y# (stride ~y)
         n# (dim ~a)
         scalea# (~cpp-ptr (pointer (~cast ~scalea)))
         shifta# (~cpp-ptr (pointer (~cast ~shifta)))
         scaleb# (~cpp-ptr (pointer (~cast ~scaleb)))
         shiftb# (~cpp-ptr (pointer (~cast ~shiftb)))]
     (cond (and (f= 0.0 ~scalea) (f= 0.0 ~scaleb)) (entry! ~y (/ (~cast ~shifta) (~cast ~shiftb)))
           (and (f= 0.0 ~scaleb) (f= 0.0 ~shiftb)) (dragan-says-ex "Division by zero is not allowed.")
           (f= 0.0 ~scaleb) (do
                              (. vdsp ~(vvdsp dsuffix 'vsmul) a# strd-a# scalea# y# strd-y# n#)
                              (. vdsp ~(vvdsp dsuffix 'vsadd) y# strd-y# shifta# y# strd-y# n#)
                              (. vdsp ~(vvdsp dsuffix 'vsdiv) y# strd-y# shiftb# y# strd-y# n#))
           :default
           (with-release [temp# (raw ~a)]
             (. vdsp ~(vvdsp dsuffix 'vsmul) a# strd-a# scalea# (~ptr temp#) 1 n#)
             (. vdsp ~(vvdsp dsuffix 'vsadd) (~ptr temp#) 1 shifta# (~ptr temp#) 1 n#)
             (. vdsp ~(vvdsp dsuffix 'vsmul) (~ptr ~b) (stride ~b) scaleb# y# strd-y# n#)
             (. vdsp ~(vvdsp dsuffix 'vsadd) y# strd-y# shiftb# y# strd-y# n#)
             (. vdsp ~(vvdsp dsuffix 'vdiv) y# strd-y# (~ptr temp#) 1 y# strd-y# n#)))
     ~y))

(defmacro vector-sigmoid-over-tanh [vsuffix dsuffix ptr cpp-ptr cast a y]
  `(let [a# (~ptr ~a)
         strd-a# (stride ~a)
         y# (~ptr ~y)
         strd-y# (stride ~y)
         n# (dim ~a)
         p05# (~cpp-ptr (pointer (~cast 0.5)))]
     (. vdsp ~(vvdsp dsuffix 'vsmul) a# strd-a# p05# y# strd-y# n#)
     (. vforce ~(vvforce vsuffix 'tanh) y# y# (cpp/int-ptr (pointer (int n#))))
     (. vdsp ~(vvdsp dsuffix 'vsadd) y# strd-y# (~cpp-ptr (pointer (~cast 1.0))) y# strd-y# n#)
     (. vdsp ~(vvdsp dsuffix 'vsmul) y# strd-y# p05# y# strd-y# n#)
     ~y))

(defmacro vector-ramp [dsuffix ptr a zero y]
  `(do
     (. vdsp ~(vvdsp dsuffix 'vmax) (~ptr ~a) (stride ~a) (~ptr ~zero) 0 (~ptr ~y) (stride ~y) (dim ~a))
     ~y))

(defmacro vector-relu [t dsuffix ptr alpha a zero y]
  `(if (identical? ~a ~y)
     (dragan-says-ex "Accelerate ReLU requires distinct arguments a and y.")
     (with-release [temp# (raw ~y)]
       (let [a# (~ptr ~a)
             n# (dim ~a)
             strd-a# (stride ~a)
             y# (~ptr ~y)
             strd-y# (stride ~y)
             zero# (~ptr ~zero)]
         (. vdsp ~(vvdsp dsuffix 'vmin) zero# 0 a# strd-a# y# strd-y# n#)
         (. vdsp ~(vvdsp dsuffix 'vmul) (~ptr ~alpha) (stride ~alpha) y# strd-y# y# strd-y# n#)
         (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 a# strd-a# (~ptr temp#) 1 n#)
         (. blas_new ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
         ~y))))

(defmacro vector-elu
  ([t vsuffix dsuffix ptr alpha a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "Accelerate ReLU requires distinct arguments a and y.")
      (with-release [temp# (raw ~y)]
        (let [a# (~ptr ~a)
              n# (dim ~a)
              strd-a# (stride ~a)
              y# (~ptr ~y)
              strd-y# (stride ~y)
              zero# (~ptr ~zero)]
          (. vdsp ~(vvdsp dsuffix 'vmin) zero# 0 a# strd-a# y# strd-y# n#)
          (. vforce ~(vvforce vsuffix 'expm1) y# y# (cpp/int-ptr (pointer (int n#))))
          (. vdsp ~(vvdsp dsuffix 'vmul) y# strd-y# (~ptr ~alpha) (stride ~alpha) y# strd-y# n#)
          (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 a# strd-a# (~ptr temp#) 1 n#)
          (. blas_new ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
          ~y))))
  ([t vsuffix dsuffix ptr a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "Accelerate ReLU requires distinct arguments a and y.")
      (with-release [temp# (raw ~y)]
        (let [a# (~ptr ~a)
              n# (dim ~a)
              strd-a# (stride ~a)
              y# (~ptr ~y)
              strd-y# (stride ~y)
              zero# (~ptr ~zero)]
          (. vdsp ~(vvdsp dsuffix 'vmin) zero# 0 a# strd-a# y# strd-y#  n#)
          (. vforce ~(vvforce vsuffix 'expm1) y# y# (cpp/int-ptr (pointer (int n#))))
          (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 a# strd-a# (~ptr temp#) 1 n#)
          (. blas_new ~(cblas t 'axpy) n# 1.0 (~ptr temp#) 1 y# strd-y#)
          ~y)))))

(defmacro real-math* [name t vsuffix dsuffix ptr cpp-ptr cast
                      vector-vforce vector-vdsp
                      vector-linear-frac sigmoid-over-tanh
                      vector-ramp vector-relu vector-elu zero]
  `(extend-type ~name
     VectorMath
     (sqr [_# a# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vsq) ~ptr a# y#)
       y#)
     (mul [_# a# b# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vmul) ~ptr a# b# y#)
       y#)
     (div [_# a# b# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vdiv) ~ptr b# a# y#)
       y#)
     (inv [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'rec) ~ptr a# y#)
       y#)
     (abs [_# a# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vabs) ~ptr a# y#)
       y#)
     (linear-frac [_# a# b# scalea# shifta# scaleb# shiftb# y#]
       (~vector-linear-frac ~dsuffix ~ptr ~cpp-ptr ~cast a# b# scalea# shifta# scaleb# shiftb# y#)
       y#)
     (fmod [_# a# b# y#]
       (~vector-vforce ~(vvforce vsuffix 'fmod) ~ptr a# b# y#)
       y#)
     (frem [_# a# b# y#]
       (~vector-vforce  ~(vvforce vsuffix 'remainder) ~ptr a# b# y#)
       y#)
     (sqrt [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'sqrt) ~ptr a# y#)
       y#)
     (inv-sqrt [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'sqrt) ~ptr a# y#)
       (~vector-vforce ~(vvforce vsuffix 'rec) ~ptr y# y#)
       y#)
     (cbrt [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'cbrt) ~ptr a# y#)
       y#)
     (inv-cbrt [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'cbrt) ~ptr a# y#)
       y#
       (~vector-vforce ~(vvforce vsuffix 'rec) ~ptr y# y#)
       y#)
     (pow2o3 [this# a# y#]
       (set-all this# ~(/ 2 3) y#)
       (~vector-vforce ~(vvforce vsuffix 'pow) ~ptr y# a# y#)
       y#)
     (pow3o2 [this# a# y#]
       (set-all this# ~(/ 3 2) y#)
       (~vector-vforce ~(vvforce vsuffix 'pow) ~ptr y# a# y#)
       y#)
     (pow [_# a# b# y#]
       (~vector-vforce ~(vvforce vsuffix 'pow) ~ptr b# a# y#)
       y#)
     (powx [this# a# b# y#]
       (set-all this# b# y#)
       (~vector-vforce ~(vvforce vsuffix 'pow) ~ptr y# a# y#)
       y#)
     (hypot [_# a# b# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vpythg) ~ptr a# b# ~zero y#)
       y#)
     (exp [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'exp) ~ptr a# y#)
       y#)
     (exp2 [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'exp2) ~ptr a# y#)
       y#)
     (exp10 [this# a# y#]
       (set-all this# 10.0 y#)
       (~vector-vforce ~(vvforce vsuffix 'pow) ~ptr a# y# y#)
       y#)
     (expm1 [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'expm1) ~ptr a# y#)
       y#)
     (log [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'log) ~ptr a# y#)
       y#)
     (log2 [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'log2) ~ptr a# y#)
       y#)
     (log10 [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'log10) ~ptr a# y#)
       y#)
     (log1p [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'log1p) ~ptr a# y#)
       y#)
     (sin [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'sin) ~ptr a# y#)
       y#)
     (cos [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'cos) ~ptr a# y#)
       y#)
     (tan [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'tan) ~ptr a# y#)
       y#)
     (sincos [_# a# y# z#]
       (~vector-vforce ~(vvforce vsuffix 'sincos) ~ptr z# a# y#)
       z#)
     (asin [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'asin) ~ptr a# y#)
       y#)
     (acos [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'acos) ~ptr a# y#)
       y#)
     (atan [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'atan) ~ptr a# y#)
       y#)
     (atan2 [_# a# b# y#]
       (~vector-vforce ~(vvforce vsuffix 'atan2) ~ptr a# b# y#)
       y#)
     (sinh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'sinh) ~ptr a# y#)
       y#)
     (cosh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'cosh) ~ptr a# y#)
       y#)
     (tanh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'tanh) ~ptr a# y#)
       y#)
     (asinh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'asinh) ~ptr a# y#)
       y#)
     (acosh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'acosh) ~ptr a# y#)
       y#)
     (atanh [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'atanh) ~ptr a# y#)
       y#)
     (erf [this# a# y#]
       (fmap! math/erf (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (erfc [this# a# y#]
       (fmap! math/erfc (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (erf-inv [this# a# y#]
       (fmap! math/erf-inv (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (erfc-inv [this# a# y#]
       (fmap! math/erfc-inv (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (cdf-norm [this# a# y#]
       (fmap! math/cdf-norm (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (cdf-norm-inv [this# a# y#]
       (fmap! math/cdf-norm-inv (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (gamma [this# a# y#]
       (fmap! math/gamma (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (lgamma [this# a# y#]
       (fmap! math/lgamma (if (identical? a# y#) y# (copy this# a# y#)))
       y#)
     (expint1 [this# a# y#]
       "TODO")
     (floor [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'floor) ~ptr a# y#)
       y#)
     (fceil [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'ceil) ~ptr a# y#)
       y#)
     (trunc [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'int) ~ptr a# y#)
       y#)
     (round [_# a# y#]
       (~vector-vforce ~(vvforce vsuffix 'nint) ~ptr a# y#)
       y#)
     (modf [_# a# y# z#]
       (~vector-vforce ~(vvforce vsuffix 'int) ~ptr a# y#)
       (~vector-vdsp ~(vvdsp dsuffix 'vsub) ~ptr y# a# z#)
       z#)
     (frac [_# a# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vfrac) ~ptr a# y#)
       y#)
     (fmin [_# a# b# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vmin) ~ptr a# b# y#)
       y#)
     (fmax [_# a# b# y#]
       (~vector-vdsp ~(vvdsp dsuffix 'vmax) ~ptr a# b# y#)
       y#)
     (copy-sign [_# a# b# y#]
       (~vector-vforce ~(vvforce vsuffix 'copysign) ~ptr a# b# y#)
       y#)
     (sigmoid [_# a# y#]
       (~sigmoid-over-tanh ~vsuffix ~dsuffix ~ptr ~cpp-ptr ~cast a# y#)
       y#)
     (ramp [this# a# y#]
       (~vector-ramp ~dsuffix ~ptr a# ~zero y#)
       y#)
     (relu [this# alpha# a# y#]
       (~vector-relu ~t ~dsuffix ~ptr alpha# a# ~zero y#)
       y#)
     (elu
       ([this# alpha# a# y#]
        (~vector-elu ~t ~vsuffix ~dsuffix ~ptr alpha# a# ~zero y#)
        y#)
       ([this# a# y#]
        (~vector-elu ~t ~vsuffix ~dsuffix ~ptr a# ~zero y#)
        y#))))

(defmacro real-vector-math* [name t vsuffix dsuffix ptr cpp-ptr cast zero]
  `(real-math* ~name ~t ~vsuffix ~dsuffix ~ptr ~cpp-ptr ~cast
               vector-vforce vector-vdsp
               vector-linear-frac vector-sigmoid-over-tanh
               vector-ramp vector-relu vector-elu
               ~zero))

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float blas_new openblas_full)
(real-vector-blas-plus* FloatVectorEngine "s" float-ptr float blas_new openblas_full
                        "catlas_saxpby" ones-float)
(real-vector-lapack* FloatVectorEngine "s" float-ptr float openblas_full)
(real-vector-math* FloatVectorEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)
(real-vector-rng* FloatVectorEngine "s" float-ptr float blas_new openblas_full
                  "catlas_saxpby" ones-float)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double blas_new openblas_full)
(real-vector-blas-plus* DoubleVectorEngine "d" double-ptr double blas_new openblas_full
                        "catlas_daxpby" ones-double)
(real-vector-lapack* DoubleVectorEngine "d" double-ptr double openblas_full)
(real-vector-math* DoubleVectorEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)
(real-vector-rng* DoubleVectorEngine "d" double-ptr double blas_new openblas_full
                  "catlas_daxpby" ones-double)

(deftype LongVectorEngine [])
(integer-vector-blas* LongVectorEngine "d" double-ptr blas_new 1)
(integer-vector-blas-plus* LongVectorEngine "d" double-ptr long-double blas_new openblas_full 1)

(deftype IntVectorEngine [])
(integer-vector-blas* IntVectorEngine "s" float-ptr blas_new 1)
(integer-vector-blas-plus* IntVectorEngine "s" float-ptr int-float blas_new openblas_full 1)

(deftype ShortVectorEngine [])
(integer-vector-blas* ShortVectorEngine "s" float-ptr blas_new 2)
(integer-vector-blas-plus* ShortVectorEngine "s" float-ptr short-float blas_new openblas_full 2)

(deftype ByteVectorEngine [])
(integer-vector-blas* ByteVectorEngine "s" float-ptr blas_new 4)
(integer-vector-blas-plus* ByteVectorEngine "s" float-ptr byte-float blas_new openblas_full 4)

;; ================= GE Engine ========================================

(defmacro accelerate-ge-axpby [blas geadd ptr alpha a beta b]
  `(do
     (when (< 0 (dim ~a))
       (let [nav-b# (navigator ~b)
             trans-a# (if (= (navigator ~a) nav-b#)
                        ~(:no-trans blas-transpose)
                        ~(:trans blas-transpose))]
         (. ~blas ~geadd (.layout (navigator ~b))
            trans-a# ~(:no-trans blas-transpose) (mrows ~b) (ncols ~b)
            ~alpha (~ptr ~a) (stride ~a) ~beta (~ptr ~b) (stride ~b)
            (~ptr ~b) (stride ~b))))
     ~b))

(defmacro accelerate-real-ge-blas* [name t ptr cast blas openblas]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (matrix-map ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (when (< 0 (dim a#))
         (let [stor-b# (full-storage b#)
               no-trans# (= (navigator a#) (navigator b#))]
           (. ~openblas ~(cblas t 'omatcopy) ~(int (blas-layout :column))
              (int (if no-trans# ~(blas-transpose :no-trans) ~(blas-transpose :trans)))
              (if no-trans# (.sd stor-b#) (.fd stor-b#)) (if no-trans# (.fd stor-b#) (.sd stor-b#))
              1.0 (~ptr a#) (stride a#) (~ptr b#) (.ld stor-b#))))
       b#)
     (dot [_# a# b#]
       (ge-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (ge-lan ~openblas ~(lapacke t 'lange) ~ptr \O a#))
     (nrm2 [_# a#]
       (ge-lan ~openblas ~(lapacke t 'lange) ~ptr \F a#))
     (nrmi [_# a#]
       (ge-lan ~openblas ~(lapacke t 'lange) ~ptr \I a#))
     (asum [_# a#]
       (ge-sum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (when (< 0 (dim a#))
         (. ~blas ~(cblas "appleblas_" t 'geadd) (.layout (navigator a#))
            ~(:no-trans blas-transpose) ~(:no-trans blas-transpose) (mrows a#) (ncols a#)
            (~cast 0.0) (~ptr a#) (stride a#) (~cast alpha#) (~ptr a#) (stride a#)
            (~ptr a#) (stride a#)))
       a#)
     (axpy [_# alpha# a# b#]
       (accelerate-ge-axpby ~blas ~(cblas "appleblas_" t 'geadd) ~ptr (~cast alpha#) a# (~cast 1.0) b#))
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

(defmacro accelerate-real-ge-blas-plus* [name t ptr cast blas openblas ones]
  `(extend-type ~name
     BlasPlus
     (amax [_# a#]
       (ge-lan ~openblas ~(lapacke t 'lange) ~ptr \M a#))
     (sum [_# a#]
       (ge-sum ~blas ~(cblas t 'dot) ~ptr a# ~ones))
     (set-all [_# alpha# a#]
       (with-lapack-check "laset"
         (. ~openblas ~(lapacke t 'laset) (.layout (navigator a#)) ~(byte (int \g))
            (mrows a#) (ncols a#) (~cast alpha#) (~cast alpha#) (~ptr a#) (stride a#)))
       a#)
     (axpby [_# alpha# a# beta# b#]
       (accelerate-ge-axpby ~blas ~(cblas "appleblas_" t 'geadd) ~ptr (~cast alpha#) a# (~cast beta#) b#)
       b#)
     (trans [_# a#]
       (when (< 0 (dim a#))
         (let [stor# (full-storage a#)]
           (if (.isGapless stor#)
             (. ~openblas ~(cblas t 'imatcopy) ~(int (blas-layout :column)) ~(int (blas-transpose :trans))
                (.sd stor#) (.fd stor#) (~cast 1.0) (~ptr a#) (.ld stor#) (.fd stor#));;TODO
             (dragan-says-ex "You can not hard-transpose the content of a matrix with a gap in memory. Sorry."
                             {:a (info a#)}))))
       a#)))

(defmacro matrix-vforce
  ([method ptr a y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-y# (~ptr ~y 0)
              surface# (cpp/int-ptr (pointer (int (.surface (region ~y)))))]
          (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                            (. vforce ~method buff-y# buff-a# surface#)
                            (. vforce ~method buff-y# buff-a# (cpp/int-ptr (pointer (int len#)))))))
      ~y))
  ([method ptr a b y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-b# (~ptr ~b 0)
              buff-y# (~ptr ~y 0)
              surface# (cpp/int-ptr (pointer (int (.surface (region ~y)))))]
          (full-storage-map ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
                            (. vforce ~method buff-y# buff-a# buff-b# surface#)
                            (. vforce ~method buff-y# buff-a# buff-b# (cpp/int-ptr (pointer (int len#)))))))
      ~y)))

(defmacro matrix-vsdsp [method ptr cpp-ptr a s y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             s# (~cpp-ptr (pointer ~s))
             surface# (cpp/int-ptr (pointer (int (.surface (region ~y)))))]
         (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                           (. vdsp ~method buff-a# 1 s# buff-y# 1 surface#)
                           (. vdsp ~method buff-a# ld-a# s# buff-y# 1 (cpp/int-ptr (pointer (int len#)))))))
     ~y))

(defmacro matrix-vdsp
  ([method ptr a y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~y))]
          (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                            (. vdsp ~method buff-a# 1 buff-y# 1 surface#)
                            (. vdsp ~method buff-a# ld-a# buff-y# 1 len#))))
      ~y))
  ([method ptr a b y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-b# (~ptr ~b 0)
              buff-y# (~ptr ~y 0)
              surface# (.surface (region ~y))]
          (full-storage-map ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
                            (. vdsp ~method buff-a# 1 buff-b# 1 buff-y# 1 surface#)
                            (. vdsp ~method buff-a# ld-a# buff-b# ld-b# buff-y# 1 len#))))
      ~y))
  ([method ptr a b zero y]
   `(do
      (when (< 0 (dim ~a))
        (let [buff-a# (~ptr ~a 0)
              buff-b# (~ptr ~b 0)
              buff-y# (~ptr ~y 0)
              zero# (~ptr ~zero)
              surface# (.surface (region ~y))]
          (full-storage-map ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
                            (. vdsp ~method buff-a# 1 buff-b# 1 zero# 0 zero# 0 buff-y# 1 surface#)
                            (. vdsp ~method buff-a# ld-a# buff-b# ld-b# zero# 0 zero# 0 buff-y# 1 len#))))
      ~y)))

(defmacro matrix-linear-frac [dsuffix ptr cpp-ptr cast a b scalea shifta scaleb shiftb y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)
             buff-y# (~ptr ~y 0)
             n# (dim ~a)
             scalea# (~cpp-ptr (pointer (~cast ~scalea)))
             shifta# (~cpp-ptr (pointer (~cast ~shifta)))
             scaleb# (~cpp-ptr (pointer (~cast ~scaleb)))
             shiftb# (~cpp-ptr (pointer (~cast ~shiftb)))
             surface# (.surface (region ~y))]
         (cond (and (f= 0.0 ~scalea) (f= 0.0 ~scaleb)) (entry! ~y (/ (~cast ~shifta) (~cast ~shiftb)))
               (and (f= 0.0 ~scaleb) (f= 0.0 ~shiftb)) (dragan-says-ex "Division by zero is not allowed.")
               (f= 0.0 ~scaleb)
               (full-storage-map
                ~a ~y len# buff-a# buff-y# ld-a#
                (do
                  (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# 1 scalea# buff-y# 1 surface#)
                  (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 shifta# buff-y# 1 surface#)
                  (. vdsp ~(vvdsp dsuffix 'vsdiv) buff-y# 1 shiftb# buff-y# 1 surface#))
                (do
                  (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# ld-a# scalea# buff-y# 1 len#)
                  (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 shifta# buff-y# 1 len#)
                  (. vdsp ~(vvdsp dsuffix 'vsdiv) buff-y# 1 shiftb# buff-y# 1 len#)))
               :default
               (with-release [temp-stripe# (delay (when-not (contiguous? ~y)
                                                    (create-vector (factory ~y) (.ld (full-storage ~y)) false)))]
                 (full-storage-map
                  ~a ~b ~y len# buff-a# buff-b# buff-y# ld-a# ld-b#
                  (with-release [temp# (raw ~y)]
                    (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# 1 scalea# (~ptr temp#) 1 surface#)
                    (. vdsp ~(vvdsp dsuffix 'vsadd) (~ptr temp#) 1 shifta# (~ptr temp#) 1 surface#)
                    (. vdsp ~(vvdsp dsuffix 'vsmul) buff-b# 1 scaleb# buff-y# 1 surface#)
                    (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 shiftb# buff-y# 1 surface#)
                    (. vdsp ~(vvdsp dsuffix 'vdiv) buff-y# 1 (~ptr temp#) 1 buff-y# 1 surface#))
                  (let [temp# (~ptr (deref temp-stripe#))]
                    (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# ld-a# scalea# temp# 1 len#)
                    (. vdsp ~(vvdsp dsuffix 'vsadd) temp# 1 shifta# temp# 1 len#)
                    (. vdsp ~(vvdsp dsuffix 'vsmul) buff-b# ld-b# scaleb# buff-y# 1 len#)
                    (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 shiftb# buff-y# 1 len#)
                    (. vdsp ~(vvdsp dsuffix 'vdiv) buff-y# 1 temp# 1 buff-y# 1 len#)
                    ))))))
     ~y))

(defmacro matrix-sigmoid-over-tanh [vsuffix dsuffix ptr cpp-ptr cast a y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             p05# (~cpp-ptr (pointer (~cast 0.5)))
             p10# (~cpp-ptr (pointer (~cast 1.0)))
             surface# (.surface (region ~y))
             surface-ptr# (cpp/int-ptr (pointer (int surface#)))]
         (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                           (do
                             (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# 1 p05# buff-y# 1 surface#)
                             (. vforce ~(vvforce vsuffix 'tanh) buff-y# buff-y# surface-ptr#)
                             (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 p10# buff-y# 1 surface#)
                             (. vdsp ~(vvdsp dsuffix 'vsmul) buff-y# 1 p05# buff-y# 1 surface#))
                           (do
                             (. vdsp ~(vvdsp dsuffix 'vsmul) buff-a# ld-a# p05# buff-y# 1 len#)
                             (. vforce ~(vvforce vsuffix 'tanh) buff-y# buff-y# (cpp/int-ptr (pointer (int len#))))
                             (. vdsp ~(vvdsp dsuffix 'vsadd) buff-y# 1 p10# buff-y# 1 len#)
                             (. vdsp ~(vvdsp dsuffix 'vsmul) buff-y# 1 p05# buff-y# 1 len#)))))
     ~y))

(defmacro matrix-ramp [dsuffix ptr a zero y]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-y# (~ptr ~y 0)
             zero# (~ptr ~zero)
             surface# (.surface (region ~y))]
         (full-storage-map ~a ~y len# buff-a# buff-y# ld-a#
                           (. vdsp ~(vvdsp dsuffix 'vmax) buff-a# 1 zero# 0 buff-y# 1 surface#)
                           (. vdsp ~(vvdsp dsuffix 'vmax) buff-a# ld-a# zero# 0 buff-y# 1 len#))))
     ~y))

(defmacro matrix-relu [t dsuffix ptr alpha a zero y]
  `(if (identical? ~a ~y)
     (dragan-says-ex "Accelerate ReLU requires distinct arguments a and y.")
     (do
       (when (< 0 (dim ~a))
         (with-release [temp-stripe# (delay (when-not (contiguous? ~y)
                                              (create-vector (factory ~y) (.ld (full-storage ~y)) false)))]
           (let [buff-a# (~ptr ~a 0)
                 buff-y# (~ptr ~y 0)
                 buff-alpha# (~ptr ~alpha ~0)
                 zero# (~ptr ~zero)
                 surface# (.surface (region ~y))]
             (full-storage-map ~alpha ~a ~y len# buff-alpha# buff-a# buff-y# ld-alpha# ld-a#
                               (with-release [temp# (raw ~y)]
                                 (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# 1 zero# 0 buff-y# 1 surface#)
                                 (. vdsp ~(vvdsp dsuffix 'vmul) buff-y# 1 buff-alpha# 1 buff-y# 1 surface#)
                                 (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# 1 (~ptr temp#) 1 surface#)
                                 (. blas_new ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                               (let [temp# (~ptr (deref temp-stripe#))]
                                 (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# ld-a# zero# 0 buff-y# 1 len#)
                                 (. vdsp ~(vvdsp dsuffix 'vmul) buff-y# 1 buff-alpha# ld-alpha# buff-y# 1 len#)
                                 (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# ld-a# temp# 1 len#)
                                 (. blas_new ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1)))
             ~y))))))

(defmacro matrix-elu
  ([t vsuffix dsuffix ptr alpha a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "Accelerate ELU requires distinct arguments a and y.")
      (do
        (when (< 0 (dim ~a))
          (with-release [temp-stripe# (delay (when-not (contiguous? ~y)
                                               (create-vector (factory ~y) (.ld (full-storage ~y)) false)))]
            (let [buff-a# (~ptr ~a 0)
                  buff-y# (~ptr ~y 0)
                  buff-alpha# (~ptr ~alpha ~0)
                  zero# (~ptr ~zero)
                  surface# (.surface (region ~y))]
              (full-storage-map ~alpha ~a ~y len# buff-alpha# buff-a# buff-y# ld-alpha# ld-a#
                                (with-release [temp# (raw ~y)]
                                  (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# 1 zero# 0 buff-y# 1 surface#)
                                  (. vforce ~(vvforce vsuffix 'expm1) buff-y# buff-y# (cpp/int-ptr (pointer (int surface#))))
                                  (. vdsp ~(vvdsp dsuffix 'vmul) buff-y# 1 buff-alpha# 1 buff-y# 1 surface#)
                                  (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# 1 (~ptr temp#) 1 surface#)
                                  (. blas_new ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                                (let [temp# (~ptr (deref temp-stripe#))]
                                  (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# ld-a# zero# 0 buff-y# 1 len#)
                                  (. vforce ~(vvforce vsuffix 'expm1) buff-y# buff-y# (cpp/int-ptr (pointer (int len#))))
                                  (. vdsp ~(vvdsp dsuffix 'vmul) buff-y# 1 buff-alpha# ld-alpha# buff-y# 1 len#)
                                  (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# ld-a# temp# 1 len#)
                                  (. blas_new ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1))))
            ~y)))))
  ([t vsuffix dsuffix ptr a zero y]
   `(if (identical? ~a ~y)
      (dragan-says-ex "Accelerate ELU requires distinct arguments a and y.")
      (do
        (when (< 0 (dim ~a))
          (with-release [temp-stripe# (delay (when-not (contiguous? ~y)
                                               (create-vector (factory ~y) (.ld (full-storage ~y)) false)))]
            (let [buff-a# (~ptr ~a 0)
                  buff-y# (~ptr ~y 0)
                  zero# (~ptr ~zero)
                  surface# (.surface (region ~y))]
              (full-storage-map ~a ~y len#  buff-a# buff-y# ld-a#
                                (with-release [temp# (raw ~y)]
                                  (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# 1 zero# 0 buff-y# 1 surface#)
                                  (. vforce ~(vvforce vsuffix 'expm1) buff-y# buff-y# (cpp/int-ptr (pointer (int surface#))))
                                  (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# 1 (~ptr temp#) 1 surface#)
                                  (. blas_new ~(cblas t 'axpy) surface# 1.0 (~ptr temp#) 1 buff-y# 1))
                                (let [temp# (~ptr (deref temp-stripe#))]
                                  (. vdsp ~(vvdsp dsuffix 'vmin) buff-a# ld-a# zero# 0 buff-y# 1 len#)
                                  (. vforce ~(vvforce vsuffix 'expm1) buff-y# buff-y# (cpp/int-ptr (pointer (int len#))))
                                  (. vdsp ~(vvdsp dsuffix 'vmax) zero# 0 buff-a# ld-a# temp# 1 len#)
                                  (. blas_new ~(cblas t 'axpy) len# 1.0 temp# 1 buff-y# 1)))
              ~y)))))))

(defmacro real-matrix-math* [name t vsuffix dsuffix ptr cpp-ptr cast zero]
  `(real-math* ~name ~t ~vsuffix ~dsuffix ~ptr ~cpp-ptr ~cast
               matrix-vforce matrix-vdsp
               matrix-linear-frac matrix-sigmoid-over-tanh
               matrix-ramp matrix-relu matrix-elu
               ~zero))

(deftype FloatGEEngine [])
(accelerate-real-ge-blas* FloatGEEngine "s" float-ptr float blas_new openblas_full)
(accelerate-real-ge-blas-plus* FloatGEEngine "s" float-ptr float blas_new openblas_full ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full ge-zero-float)
(real-matrix-math* FloatGEEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)
(real-ge-rng* FloatGEEngine "s" float-ptr float blas_new openblas_full "catlas_saxpby" ones-float)

(deftype DoubleGEEngine [])
(accelerate-real-ge-blas* DoubleGEEngine "d" double-ptr double blas_new openblas_full)
(accelerate-real-ge-blas-plus* DoubleGEEngine "d" double-ptr double blas_new openblas_full ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full ge-zero-double)
(real-matrix-math* DoubleGEEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)
(real-ge-rng* DoubleGEEngine "d" double-ptr double blas_new openblas_full "catlas_daxpby" ones-double)

;;TODO
(deftype LongGEEngine [])
(integer-ge-blas* LongGEEngine "d" double-ptr blas_new openblas_full 1)

(deftype IntGEEngine [])
(integer-ge-blas* IntGEEngine "s" float-ptr blas_new openblas_full 1)

(deftype ShortGEEngine []) ;; TODO

(deftype ByteGEEngine []) ;; TODO

;; ========================= TR matrix engines ===============================================

(deftype FloatTREngine [])
(real-tr-blas* FloatTREngine "s" float-ptr float blas_new openblas_full)
(real-tr-blas-plus* FloatTREngine "s" float-ptr float blas_new openblas_full "catlas_saxpby" ones-float)
(real-tr-lapack* FloatTREngine "s" float-ptr cpp/float-pointer float blas_new openblas_full)
(real-matrix-math* FloatTREngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleTREngine [])
(real-tr-blas* DoubleTREngine "d" double-ptr double blas_new openblas_full)
(real-tr-blas-plus* DoubleTREngine "d" double-ptr double blas_new openblas_full "catlas_daxpby" ones-double)
(real-tr-lapack* DoubleTREngine "d" double-ptr cpp/double-pointer double blas_new openblas_full)
(real-matrix-math* DoubleTREngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongTREngine [])
;;(integer-tr-blas* LongTREngine "d" double-ptr long-double openblas_full openblas_full 1)

(deftype IntTREngine [])
;;(integer-tr-blas* IntTREngine "s" float-ptr int-float openblas_full openblas_full 1)

(deftype ShortTREngine []) ;; TODO

(deftype ByteTREngine []) ;; TODO

;; ========================= SY matrix engines ===============================================

(deftype FloatSYEngine [])
(real-sy-blas* FloatSYEngine "s" float-ptr float blas_new openblas_full)
(real-sy-blas-plus* FloatSYEngine "s" float-ptr float blas_new openblas_full "catlas_saxpby" ones-float)
(real-sy-lapack* FloatSYEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full ge-zero-float)
(real-matrix-math* FloatSYEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleSYEngine [])
(real-sy-blas* DoubleSYEngine "d" double-ptr double blas_new openblas_full)
(real-sy-blas-plus* DoubleSYEngine "d" double-ptr double blas_new openblas_full "catlas_daxpby" ones-double)
(real-sy-lapack* DoubleSYEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full ge-zero-double)
(real-matrix-math* DoubleSYEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

;;TODO
(deftype LongSYEngine [])
;;(integer-tr-blas* LongSYEngine "d" double-ptr long-double openblas_full openblas_full 1)

(deftype IntSYEngine [])
;;(integer-tr-blas* IntSYEngine "s" float-ptr int-float openblas_full openblas_full 1)

(deftype ShortSYEngine []);; TODO

(deftype ByteSYEngine []);; TODO

;; ============================ GB matrix engines ==================================================

(deftype FloatGBEngine [])
(real-gb-blas* FloatGBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full ones-float)
(real-gb-blas-plus* FloatGBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-gb-lapack* FloatGBEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)
(real-matrix-math* FloatGBEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleGBEngine [])
(real-gb-blas* DoubleGBEngine "d" double-ptr cpp/double-ptr double openblas_full blas_new ones-double)
(real-gb-blas-plus* DoubleGBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-gb-lapack* DoubleGBEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)
(real-matrix-math* DoubleGBEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongGBEngine [])
(deftype IntGBEngine [])
(deftype ShortGBEngine [])
(deftype ByteGBEngine [])

;; ============================ SB matrix engines ==================================================

(deftype FloatSBEngine [])
(real-sb-blas* FloatSBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-sb-blas-plus* FloatSBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-sb-lapack* FloatSBEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-matrix-math* FloatSBEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleSBEngine [])
(real-sb-blas* DoubleSBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-sb-blas-plus* DoubleSBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-sb-lapack* DoubleSBEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-matrix-math* DoubleSBEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongSBEngine [])
(deftype IntSBEngine [])
(deftype ShortSBEngine [])
(deftype ByteSBEngine [])

;; ============================ TB matrix engines ==================================================

(deftype FloatTBEngine [])
(real-tb-blas* FloatTBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-tb-blas-plus* FloatTBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-tb-lapack* FloatTBEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-matrix-math* FloatTBEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleTBEngine [])
(real-tb-blas* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-tb-blas-plus* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-tb-lapack* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-matrix-math* DoubleTBEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongTBEngine [])
(deftype IntTBEngine [])
(deftype ShortTBEngine [])
(deftype ByteTBEngine [])

;; ============================ TP matrix engines ====================================================

(deftype FloatTPEngine [])
(real-tp-blas* FloatTPEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-tp-blas-plus* FloatTPEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-tp-lapack* FloatTPEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-matrix-math* FloatTPEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleTPEngine [])
(real-tp-blas* DoubleTPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-tp-blas-plus* DoubleTPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-tp-lapack* DoubleTPEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-matrix-math* DoubleTPEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongTPEngine [])
(deftype IntTPEngine [])
(deftype ShortTPEngine [])
(deftype ByteTPEngine [])

;; ============================ SP matrix engines ====================================================

(deftype FloatSPEngine [])
(real-sp-blas* FloatSPEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-sp-blas-plus* FloatSPEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-sp-lapack* FloatSPEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)
(real-matrix-math* FloatSPEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleSPEngine [])
(real-sp-blas* DoubleSPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-sp-blas-plus* DoubleSPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-sp-lapack* DoubleSPEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)
(real-matrix-math* DoubleSPEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongSPEngine [])
(deftype IntSPEngine [])
(deftype ShortSPEngine [])
(deftype ByteSPEngine [])

;; ============================ GD matrix engines ==================================================

(deftype FloatGDEngine [])
(real-gd-blas* FloatGDEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-diagonal-blas-plus* FloatGDEngine "s" float-ptr float blas_new openblas_full
                          "catlas_saxpby" ones-float)
(real-gd-lapack* FloatGDEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-matrix-math* FloatGDEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleGDEngine [])
(real-gd-blas* DoubleGDEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-diagonal-blas-plus* DoubleGDEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-gd-lapack* DoubleGDEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-matrix-math* DoubleGDEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongGDEngine [])
(deftype IntGDEngine [])
(deftype ShortGDEngine [])
(deftype ByteGDEngine [])

(defmacro accelerate-real-tridiagonal-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (diagonal-method ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (diagonal-method ~blas ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (diagonal-method ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (tridiagonal-lan ~lapack ~(lapacke "" t 'langt_) ~ptr \O a# clong-pointer))
     (nrm2 [_# a#]
       (diagonal-method ~blas ~(cblas t 'nrm2) ~ptr a#))
     (nrmi [_# a#]
       (tridiagonal-lan ~lapack ~(lapacke "" t 'langt_) ~ptr \I a# clong-pointer))
     (asum [_# a#]
       (diagonal-method ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~lapack ~(lapacke "" t 'lagtm_) ~ptr ~cpp-ptr clong-pointer
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
        (tridiagonal-mm ~lapack ~(lapacke "" t 'lagtm_) ~ptr ~cpp-ptr clong-pointer
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(deftype FloatGTEngine [])
(accelerate-real-tridiagonal-blas* FloatGTEngine "s" float-ptr cpp/float-ptr float blas_new lapack_new)
(real-diagonal-blas-plus* FloatGTEngine "s" float-ptr float blas_new openblas_full
                          "catlas_saxpby" ones-float)
(real-gt-lapack* FloatGTEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)
(real-matrix-math* FloatGTEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleGTEngine [])
(accelerate-real-tridiagonal-blas* DoubleGTEngine "d" double-ptr cpp/double-ptr double blas_new lapack_new)
(real-diagonal-blas-plus* DoubleGTEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-gt-lapack* DoubleGTEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)
(real-matrix-math* DoubleGTEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongGTEngine [])
(deftype IntGTEngine [])
(deftype ShortGTEngine [])
(deftype ByteGTEngine [])

(deftype FloatDTEngine [])
(accelerate-real-tridiagonal-blas* FloatDTEngine "s" float-ptr cpp/float-ptr float blas_new lapack_new)
(real-diagonal-blas-plus* FloatDTEngine "s" float-ptr float blas_new openblas_full
                          "catlas_saxpby" ones-float)
(real-dt-lapack* FloatDTEngine "s" float-ptr float openblas_full)
(real-matrix-math* FloatDTEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleDTEngine [])
(accelerate-real-tridiagonal-blas* DoubleDTEngine "d" double-ptr cpp/double-ptr double blas_new lapack_new)
(real-diagonal-blas-plus* DoubleDTEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-dt-lapack* DoubleDTEngine "d" double-ptr double openblas_full)
(real-matrix-math* DoubleDTEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongDTEngine [])
(deftype IntDTEngine [])
(deftype ShortDTEngine [])
(deftype ByteDTEngine [])

(defmacro accelerate-real-st-blas* [name t ptr cpp-ptr cast blas lapack]
  `(extend-type ~name
     Blas
     (swap [_# a# b#]
       (diagonal-method ~blas ~(cblas t 'swap) ~ptr a# b#)
       a#)
     (copy [_# a# b#]
       (diagonal-method ~blas ~(cblas t 'copy) ~ptr a# b#)
       b#)
     (dot [_# a# b#]
       (st-dot ~blas ~(cblas t 'dot) ~ptr a# b#))
     (nrm1 [_# a#]
       (tridiagonal-lan ~lapack ~(lapacke "" t 'langt_) ~ptr \O a# clong-pointer))
     (nrm2 [_# a#]
       (tridiagonal-lan ~lapack ~(lapacke "" t 'langt_) ~ptr \F a# clong-pointer))
     (nrmi [_# a#]
       (tridiagonal-lan ~lapack ~(lapacke "" t 'langt_) ~ptr \I a# clong-pointer))
     (asum [_# a#]
       (st-asum ~blas ~(cblas t 'asum) ~ptr a#))
     (scal [_# alpha# a#]
       (diagonal-scal ~blas ~(cblas t 'scal) ~ptr (~cast alpha#) a#))
     (axpy [_# alpha# a# b#]
       (diagonal-axpy ~blas ~(cblas t 'axpy) ~ptr (~cast alpha#) a# b#))
     (mv
       ([_# alpha# a# x# beta# y#]
        (tridiagonal-mv ~lapack ~(lapacke "" t 'lagtm_) ~ptr ~cpp-ptr clong-pointer
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
        (tridiagonal-mm ~lapack ~(lapacke "" t 'lagtm_) ~ptr ~cpp-ptr clong-pointer
                        (~cast alpha#) a# b# (~cast beta#) c# left#)))))

(deftype FloatSTEngine [])
(accelerate-real-st-blas* FloatSTEngine "s" float-ptr cpp/float-ptr float blas_new lapack_new)
(real-st-blas-plus* FloatSTEngine "s" float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-st-lapack* FloatSTEngine "s" float-ptr float openblas_full)
(real-matrix-math* FloatSTEngine "s" "f" "" float-ptr cpp/float-ptr float zero-float)

(deftype DoubleSTEngine [])
(accelerate-real-st-blas* DoubleSTEngine "d" double-ptr cpp/double-ptr double blas_new lapack_new)
(real-st-blas-plus* DoubleSTEngine "d" double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-st-lapack* DoubleSTEngine "d" double-ptr double openblas_full)
(real-matrix-math* DoubleSTEngine "d" "" "D" double-ptr cpp/double-ptr double zero-double)

(deftype LongSTEngine [])
(deftype IntSTEngine [])
(deftype ShortSTEngine [])
(deftype ByteSTEngine [])

;; ================================================================================


(def accelerate-int (->BlasIntegerFactory
                     accelerate-int int-accessor
                     (->IntVectorEngine) (->IntGEEngine)
                     (->IntTREngine) (->IntSYEngine) (->IntGBEngine)
                     (->IntSBEngine) (->IntTBEngine) (->IntSPEngine)
                     (->IntTPEngine) (->IntGDEngine) (->IntGTEngine)
                     (->IntDTEngine) (->IntSTEngine)))

(def accelerate-long (->BlasIntegerFactory
                      accelerate-int long-accessor
                      (->LongVectorEngine) (->LongGEEngine)
                      (->LongTREngine) (->LongSYEngine) (->LongGBEngine)
                      (->LongSBEngine) (->LongTBEngine) (->LongSPEngine)
                      (->LongTPEngine) (->LongGDEngine) (->LongGTEngine)
                      (->LongDTEngine) (->LongSTEngine)))

(def accelerate-short (->BlasIntegerFactory
                       accelerate-int short-accessor
                       (->ShortVectorEngine) (->ShortGEEngine)
                       (->ShortTREngine) (->ShortSYEngine) (->ShortGBEngine)
                       (->ShortSBEngine) (->ShortTBEngine) (->ShortSPEngine)
                       (->ShortTPEngine) (->ShortGDEngine) (->ShortGTEngine)
                       (->ShortDTEngine) (->ShortSTEngine)))

(def accelerate-byte (->BlasIntegerFactory
                      accelerate-int byte-accessor
                      (->ByteVectorEngine) (->ByteGEEngine)
                      (->ByteTREngine) (->ByteSYEngine) (->ByteGBEngine)
                      (->ByteSBEngine) (->ByteTBEngine) (->ByteSPEngine)
                      (->ByteTPEngine) (->ByteGDEngine) (->ByteGTEngine)
                      (->ByteDTEngine) (->ByteSTEngine)))

(def accelerate-float (->BlasRealFactory
                       accelerate-int float-accessor
                       (->FloatVectorEngine) (->FloatGEEngine)
                       (->FloatTREngine) (->FloatSYEngine) (->FloatGBEngine)
                       (->FloatSBEngine) (->FloatTBEngine) (->FloatSPEngine)
                       (->FloatTPEngine) (->FloatGDEngine) (->FloatGTEngine)
                       (->FloatDTEngine) (->FloatSTEngine)
                       nil nil))

(def accelerate-double (->BlasRealFactory
                        accelerate-int double-accessor
                        (->DoubleVectorEngine) (->DoubleGEEngine)
                        (->DoubleTREngine) (->DoubleSYEngine) (->DoubleGBEngine)
                        (->DoubleSBEngine) (->DoubleTBEngine) (->DoubleSPEngine)
                        (->DoubleTPEngine) (->DoubleGDEngine) (->DoubleGTEngine)
                        (->DoubleDTEngine) (->DoubleSTEngine)
                        nil nil))
