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
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release view]]
             [utils :refer [dragan-says-ex with-check generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap! extract]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [pointer long-pointer float-pointer double-pointer clong-pointer get-entry malloc! free!]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols cols rows matrix-type entry] :as core]
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
             [factory :refer :all]])
  (:import java.nio.ByteBuffer
           [uncomplicate.neanderthal.internal.api DataAccessor Vector LayoutNavigator Region
            GEMatrix UploMatrix DenseStorage]

           [uncomplicate.javacpp.accelerate.global thread blas_new lapack_new bnns]))

(let [openblas-load (System/getProperty "org.bytedeco.openblas.load")
      library-path (System/getProperty "java.library.path") ]
  (try
    (System/setProperty "java.library.path" (str library-path ":/usr/lib"))
    (System/setProperty "org.bytedeco.openblas.load" "blas")
    (import 'org.bytedeco.openblas.global.openblas_full)
    (catch Exception e
      (System/setProperty "java.library.path" (str library-path))
      (System/setProperty "org.bytedeco.openblas.load" (str openblas-load)))))

(defn threading? []
  (let [threading (thread/BLASGetThreading)]
    (cond
      (= thread/BLAS_THREADING_MULTI_THREADED threading) true
      (= thread/BLAS_THREADING_SINGLE_THREADED threading) false
      :default (throw (dragan-says-ex "Accelerate returned a threading option unknown to this library."
                                      {:threading threading
                                       :max-threading-options thread/BLAS_THREADING_MAX_OPTIONS})))))

(defn threading! [multi-threading?]
  (let [threading (if multi-threading?
                    thread/BLAS_THREADING_MULTI_THREADED
                    thread/BLAS_THREADING_SINGLE_THREADED)]
    (if (= 0 (thread/BLASSetThreading threading))
      multi-threading?
      (throw (dragan-says-ex "The current platform does not support the requested threading model."
                             {:threading threading
                              :max-threading-options thread/BLAS_THREADING_MAX_OPTIONS})))))

;; ============ Vector Engines ============================================

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float blas_new openblas_full)
(real-vector-blas-plus* FloatVectorEngine "s" float-ptr float blas_new openblas_full
                             "catlas_saxpby" ones-float)
(real-vector-lapack* FloatVectorEngine "s" float-ptr float openblas_full)
(real-vector-rng* FloatVectorEngine "s" float-ptr float blas_new openblas_full
                  "catlas_saxpby" ones-float)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double blas_new openblas_full)
(real-vector-blas-plus* DoubleVectorEngine "d" double-ptr double blas_new openblas_full
                        "catlas_daxpby" ones-double)
(real-vector-lapack* DoubleVectorEngine "d" double-ptr double openblas_full)
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

(deftype FloatGEEngine [])
(accelerate-real-ge-blas* FloatGEEngine "s" float-ptr float blas_new openblas_full)
(accelerate-real-ge-blas-plus* FloatGEEngine "s" float-ptr float blas_new openblas_full ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full zero-float)
(real-ge-rng* FloatGEEngine "s" float-ptr float blas_new openblas_full "catlas_saxpby" ones-float)

(deftype DoubleGEEngine [])
(accelerate-real-ge-blas* DoubleGEEngine "d" double-ptr double blas_new openblas_full)
(accelerate-real-ge-blas-plus* DoubleGEEngine "d" double-ptr double blas_new openblas_full ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full zero-double)
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

(deftype DoubleTREngine [])
(real-tr-blas* DoubleTREngine "d" double-ptr double blas_new openblas_full)
(real-tr-blas-plus* DoubleTREngine "d" double-ptr double blas_new openblas_full "catlas_daxpby" ones-double)
(real-tr-lapack* DoubleTREngine "d" double-ptr cpp/double-pointer double blas_new openblas_full)

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
(real-sy-lapack* FloatSYEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full zero-float)

(deftype DoubleSYEngine [])
(real-sy-blas* DoubleSYEngine "d" double-ptr double blas_new openblas_full)
(real-sy-blas-plus* DoubleSYEngine "d" double-ptr double blas_new openblas_full "catlas_daxpby" ones-double)
(real-sy-lapack* DoubleSYEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full zero-double)

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

(deftype DoubleGBEngine [])
(real-gb-blas* DoubleGBEngine "d" double-ptr cpp/double-ptr double openblas_full blas_new ones-double)
(real-gb-blas-plus* DoubleGBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-gb-lapack* DoubleGBEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

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

(deftype DoubleSBEngine [])
(real-sb-blas* DoubleSBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-sb-blas-plus* DoubleSBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-sb-lapack* DoubleSBEngine "d" double-ptr cpp/double-ptr double openblas_full)

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

(deftype DoubleTBEngine [])
(real-tb-blas* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-tb-blas-plus* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-tb-lapack* DoubleTBEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)

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

(deftype DoubleTPEngine [])
(real-tp-blas* DoubleTPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-tp-blas-plus* DoubleTPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-tp-lapack* DoubleTPEngine "d" double-ptr cpp/double-ptr double openblas_full)

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

(deftype DoubleSPEngine [])
(real-sp-blas* DoubleSPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-sp-blas-plus* DoubleSPEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-sp-lapack* DoubleSPEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

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

(deftype DoubleGDEngine [])
(real-gd-blas* DoubleGDEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-diagonal-blas-plus* DoubleGDEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-gd-lapack* DoubleGDEngine "d" double-ptr cpp/double-ptr double openblas_full)

(deftype LongGDEngine [])
(deftype IntGDEngine [])
(deftype ShortGDEngine [])
(deftype ByteGDEngine [])

(deftype FloatGTEngine [])
(real-tridiagonal-blas* FloatGTEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-diagonal-blas-plus* FloatGTEngine "s" float-ptr float blas_new openblas_full
                          "catlas_saxpby" ones-float)
(real-gt-lapack* FloatGTEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)

(deftype DoubleGTEngine [])
(real-tridiagonal-blas* DoubleGTEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-diagonal-blas-plus* DoubleGTEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-gt-lapack* DoubleGTEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

(deftype LongGTEngine [])
(deftype IntGTEngine [])
(deftype ShortGTEngine [])
(deftype ByteGTEngine [])

(deftype FloatDTEngine [])
(real-tridiagonal-blas* FloatDTEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-diagonal-blas-plus* FloatDTEngine "s" float-ptr float blas_new openblas_full
                          "catlas_saxpby" ones-float)
(real-dt-lapack* FloatDTEngine "s" float-ptr float openblas_full)

(deftype DoubleDTEngine [])
(real-tridiagonal-blas* DoubleDTEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-diagonal-blas-plus* DoubleDTEngine "d" double-ptr double blas_new openblas_full
                          "catlas_daxpby" ones-double)
(real-dt-lapack* DoubleDTEngine "d" double-ptr double openblas_full)

(deftype LongDTEngine [])
(deftype IntDTEngine [])
(deftype ShortDTEngine [])
(deftype ByteDTEngine [])

(deftype FloatSTEngine [])
(real-st-blas* FloatSTEngine "s" float-ptr cpp/float-ptr float blas_new openblas_full)
(real-st-blas-plus* FloatSTEngine "s" float-ptr float blas_new openblas_full
                    "catlas_saxpby" ones-float)
(real-st-lapack* FloatSTEngine "s" float-ptr float openblas_full)

(deftype DoubleSTEngine [])
(real-st-blas* DoubleSTEngine "d" double-ptr cpp/double-ptr double blas_new openblas_full)
(real-st-blas-plus* DoubleSTEngine "d" double-ptr double blas_new openblas_full
                    "catlas_daxpby" ones-double)
(real-st-lapack* DoubleSTEngine "d" double-ptr double openblas_full)

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
