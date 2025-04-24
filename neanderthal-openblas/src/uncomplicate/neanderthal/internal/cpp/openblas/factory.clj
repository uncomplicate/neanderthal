;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.openblas.factory
  (:refer-clojure :exclude [abs])
  (:require [clojure.string :refer [trim split]]
            [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release view]]
             [utils :refer [dragan-says-ex with-check generate-seed]]]
            [uncomplicate.fluokitten.core :refer [fmap! extract]]
            [uncomplicate.clojure-cpp :as cpp :refer [long-pointer float-pointer double-pointer]]
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
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer BytePointer]
           [uncomplicate.neanderthal.internal.api DataAccessor LayoutNavigator Region
            GEMatrix UploMatrix DenseStorage]
           [org.bytedeco.openblas.global openblas_full]))

;; ===================== System ================================================================

(defn version []
  (let [ver-seq (split (second (split (trim openblas_full/OPENBLAS_VERSION) #" ")) #"\.")]
    {:major (Long/parseLong (ver-seq 0))
     :minor (Long/parseLong (ver-seq 1))
     :update (Long/parseLong (ver-seq 2))}))

(defn vendor []
  (case (openblas_full/blas_get_vendor)
    0 :unknown
    1 :cublas
    2 :openblas
    3 :mkl))

;; ===================== Miscellaneous =========================================================

(defn num-threads ^long []
  (openblas_full/blas_get_num_threads))

(defn num-threads!
  ([]
   (openblas_full/blas_set_num_threads -1))
  ([^long n]
   (openblas_full/blas_set_num_threads n)))

;; ============ Vector Engines ============================================

(deftype FloatVectorEngine [])
(real-vector-blas* FloatVectorEngine "s" float-ptr float openblas_full openblas_full)
(real-vector-blas-plus* FloatVectorEngine "s" float-ptr float openblas_full openblas_full
                        "cblas_saxpby" ones-float)
(real-vector-lapack* FloatVectorEngine "s" float-ptr float openblas_full)
(real-vector-rng* FloatVectorEngine "s" float-ptr float openblas_full openblas_full ones-float)

(deftype DoubleVectorEngine [])
(real-vector-blas* DoubleVectorEngine "d" double-ptr double openblas_full openblas_full)
(real-vector-blas-plus* DoubleVectorEngine "d" double-ptr double openblas_full openblas_full
                        "cblas_daxpby" ones-double)
(real-vector-lapack* DoubleVectorEngine "d" double-ptr double openblas_full)
(real-vector-rng* DoubleVectorEngine "d" double-ptr double openblas_full openblas_full ones-double)

(deftype LongVectorEngine [])
(integer-vector-blas* LongVectorEngine "d" double-ptr openblas_full 1)
(integer-vector-blas-plus* LongVectorEngine "d" double-ptr long-double openblas_full openblas_full 1)

(deftype IntVectorEngine [])
(integer-vector-blas* IntVectorEngine "s" float-ptr openblas_full 1)
(integer-vector-blas-plus* IntVectorEngine "s" float-ptr int-float openblas_full openblas_full 1)

(deftype ShortVectorEngine [])
(integer-vector-blas* ShortVectorEngine "s" float-ptr openblas_full 2)
(integer-vector-blas-plus* ShortVectorEngine "s" float-ptr short-float openblas_full openblas_full 2)

(deftype ByteVectorEngine [])
(integer-vector-blas* ByteVectorEngine "s" float-ptr openblas_full 4)
(integer-vector-blas-plus* ByteVectorEngine "s" float-ptr byte-float openblas_full openblas_full 4)

;; ================= Real GE Engine ========================================

(deftype FloatGEEngine [])
(real-ge-blas* FloatGEEngine "s" float-ptr float openblas_full openblas_full)
(real-ge-blas-plus* FloatGEEngine "s" float-ptr float openblas_full openblas_full ones-float)
(real-ge-lapack* FloatGEEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full zero-float)
(real-ge-rng* FloatGEEngine "s" float-ptr float openblas_full openblas_full ones-float)

(deftype DoubleGEEngine [])
(real-ge-blas* DoubleGEEngine "d" double-ptr double openblas_full openblas_full)
(real-ge-blas-plus* DoubleGEEngine "d" double-ptr double openblas_full openblas_full ones-double)
(real-ge-lapack* DoubleGEEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full zero-double)
(real-ge-rng* DoubleGEEngine "d" double-ptr double openblas_full openblas_full ones-double)

;;TODO
(deftype LongGEEngine [])
(integer-ge-blas* LongGEEngine "d" double-ptr openblas_full openblas_full 1)

(deftype IntGEEngine [])


(integer-ge-blas* IntGEEngine "s" float-ptr openblas_full openblas_full 1)

(deftype ShortGEEngine []) ;; TODO

(deftype ByteGEEngine []) ;; TODO

;; ========================= TR matrix engines ===============================================

(deftype FloatTREngine [])
(real-tr-blas* FloatTREngine "s" float-ptr float openblas_full openblas_full)
(real-tr-blas-plus* FloatTREngine "s" float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-tr-lapack* FloatTREngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)

(deftype DoubleTREngine [])
(real-tr-blas* DoubleTREngine "d" double-ptr double openblas_full openblas_full)
(real-tr-blas-plus* DoubleTREngine "d" double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-tr-lapack* DoubleTREngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)

(deftype LongTREngine [])
;;(integer-tr-blas* LongTREngine "d" double-ptr long-double openblas_full openblas_full 1)

(deftype IntTREngine [])
;;(integer-tr-blas* IntTREngine "s" float-ptr int-float openblas_full openblas_full 1)

(deftype ShortTREngine []) ;; TODO

(deftype ByteTREngine []) ;; TODO

;; ========================= SY matrix engines ===============================================

(deftype FloatSYEngine [])
(real-sy-blas* FloatSYEngine "s" float-ptr float openblas_full openblas_full)
(real-sy-blas-plus* FloatSYEngine "s" float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-sy-lapack* FloatSYEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full zero-float)

(deftype DoubleSYEngine [])
(real-sy-blas* DoubleSYEngine "d" double-ptr double openblas_full openblas_full)
(real-sy-blas-plus* DoubleSYEngine "d" double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
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
(real-gb-blas* FloatGBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full ones-float)
(real-gb-blas-plus* FloatGBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-gb-lapack* FloatGBEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)

(deftype DoubleGBEngine [])
(real-gb-blas* DoubleGBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full ones-double)
(real-gb-blas-plus* DoubleGBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-gb-lapack* DoubleGBEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

(deftype LongGBEngine [])
(deftype IntGBEngine [])
(deftype ShortGBEngine [])
(deftype ByteGBEngine [])

;; ============================ SB matrix engines ==================================================

(deftype FloatSBEngine [])
(real-sb-blas* FloatSBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-sb-blas-plus* FloatSBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-sb-lapack* FloatSBEngine "s" float-ptr cpp/float-ptr float openblas_full)

(deftype DoubleSBEngine [])
(real-sb-blas* DoubleSBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-sb-blas-plus* DoubleSBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-sb-lapack* DoubleSBEngine "d" double-ptr cpp/double-ptr double openblas_full)

(deftype LongSBEngine [])
(deftype IntSBEngine [])
(deftype ShortSBEngine [])
(deftype ByteSBEngine [])

;; ============================ TB matrix engines ==================================================

(deftype FloatTBEngine [])
(real-tb-blas* FloatTBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-tb-blas-plus* FloatTBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-tb-lapack* FloatTBEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)

(deftype DoubleTBEngine [])
(real-tb-blas* DoubleTBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-tb-blas-plus* DoubleTBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-tb-lapack* DoubleTBEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)

(deftype LongTBEngine [])
(deftype IntTBEngine [])
(deftype ShortTBEngine [])
(deftype ByteTBEngine [])

;; ============================ TP matrix engines ====================================================

(deftype FloatTPEngine [])
(real-tp-blas* FloatTPEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-tp-blas-plus* FloatTPEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-tp-lapack* FloatTPEngine "s" float-ptr cpp/float-ptr float openblas_full)

(deftype DoubleTPEngine [])
(real-tp-blas* DoubleTPEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-tp-blas-plus* DoubleTPEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-tp-lapack* DoubleTPEngine "d" double-ptr cpp/double-ptr double openblas_full)

(deftype LongTPEngine [])
(deftype IntTPEngine [])
(deftype ShortTPEngine [])
(deftype ByteTPEngine [])

;; ============================ SP matrix engines ====================================================

(deftype FloatSPEngine [])
(real-sp-blas* FloatSPEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-sp-blas-plus* FloatSPEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-sp-lapack* FloatSPEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)

(deftype DoubleSPEngine [])
(real-sp-blas* DoubleSPEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-sp-blas-plus* DoubleSPEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-sp-lapack* DoubleSPEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

(deftype LongSPEngine [])
(deftype IntSPEngine [])
(deftype ShortSPEngine [])
(deftype ByteSPEngine [])

;; ============================ GD matrix engines ==================================================

(deftype FloatGDEngine [])
(real-gd-blas* FloatGDEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-diagonal-blas-plus* FloatGDEngine "s" float-ptr float openblas_full openblas_full
                          "cblas_saxpby" ones-float)
(real-gd-lapack* FloatGDEngine "s" float-ptr cpp/float-ptr float openblas_full)

(deftype DoubleGDEngine [])
(real-gd-blas* DoubleGDEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-diagonal-blas-plus* DoubleGDEngine "d" double-ptr double openblas_full openblas_full
                          "cblas_daxpby" ones-double)
(real-gd-lapack* DoubleGDEngine "d" double-ptr cpp/double-ptr double openblas_full)

(deftype LongGDEngine [])
(deftype IntGDEngine [])
(deftype ShortGDEngine [])
(deftype ByteGDEngine [])

;; ============================ Tridiagonal matrix engines =====================

(deftype FloatGTEngine [])
(real-tridiagonal-blas* FloatGTEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-diagonal-blas-plus* FloatGTEngine "s" float-ptr float openblas_full openblas_full
                          "cblas_saxpby" ones-float)
(real-gt-lapack* FloatGTEngine "s" float-ptr cpp/float-ptr int-ptr float openblas_full)

(deftype DoubleGTEngine [])
(real-tridiagonal-blas* DoubleGTEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-diagonal-blas-plus* DoubleGTEngine "d" double-ptr double openblas_full openblas_full
                          "cblas_daxpby" ones-double)
(real-gt-lapack* DoubleGTEngine "d" double-ptr cpp/double-ptr int-ptr double openblas_full)

(deftype LongGTEngine [])
(deftype IntGTEngine [])
(deftype ShortGTEngine [])
(deftype ByteGTEngine [])

(deftype FloatDTEngine [])
(real-tridiagonal-blas* FloatDTEngine "s" float-ptr cpp/float-ptr float openblas_full)
(real-diagonal-blas-plus* FloatDTEngine "s" float-ptr float openblas_full openblas_full
                          "cblas_saxpby" ones-float)
(real-dt-lapack* FloatDTEngine "s" float-ptr float openblas_full)

(deftype DoubleDTEngine [])
(real-tridiagonal-blas* DoubleDTEngine "d" double-ptr cpp/double-ptr double openblas_full)
(real-diagonal-blas-plus* DoubleDTEngine "d" double-ptr double openblas_full openblas_full
                          "cblas_daxpby" ones-double)
(real-dt-lapack* DoubleDTEngine "d" double-ptr double openblas_full)

(deftype LongDTEngine [])
(deftype IntDTEngine [])
(deftype ShortDTEngine [])
(deftype ByteDTEngine [])

(deftype FloatSTEngine [])
(real-st-blas* FloatSTEngine "s" float-ptr cpp/float-ptr float openblas_full openblas_full)
(real-st-blas-plus* FloatSTEngine "s" float-ptr float openblas_full openblas_full
                    "cblas_saxpby" ones-float)
(real-st-lapack* FloatSTEngine "s" float-ptr float openblas_full)

(deftype DoubleSTEngine [])
(real-st-blas* DoubleSTEngine "d" double-ptr cpp/double-ptr double openblas_full openblas_full)
(real-st-blas-plus* DoubleSTEngine "d" double-ptr double openblas_full openblas_full
                    "cblas_daxpby" ones-double)
(real-st-lapack* DoubleSTEngine "d" double-ptr double openblas_full)

(deftype LongSTEngine [])
(deftype IntSTEngine [])
(deftype ShortSTEngine [])
(deftype ByteSTEngine [])

;; ================================================================================

(def openblas-int (->BlasIntegerFactory openblas-int int-accessor (->IntVectorEngine) (->IntGEEngine)
                                        (->IntTREngine) (->IntSYEngine)
                                        (->IntGBEngine) (->IntSBEngine) (->IntTBEngine)
                                        (->IntSPEngine) (->IntTPEngine) (->IntGDEngine)
                                        (->IntGTEngine) (->IntDTEngine) (->IntSTEngine)))

(def openblas-long (->BlasIntegerFactory openblas-int long-accessor (->LongVectorEngine) (->LongGEEngine)
                                         (->LongTREngine) (->LongSYEngine)
                                         (->LongGBEngine) (->LongSBEngine) (->LongTBEngine)
                                         (->LongSPEngine) (->LongTPEngine) (->LongGDEngine)
                                         (->LongGTEngine) (->LongDTEngine) (->LongSTEngine)))

(def openblas-short (->BlasIntegerFactory openblas-int short-accessor (->ShortVectorEngine) (->ShortGEEngine)
                                          (->ShortTREngine) (->ShortSYEngine)
                                          (->ShortGBEngine) (->ShortSBEngine) (->ShortTBEngine)
                                          (->ShortSPEngine) (->ShortTPEngine) (->ShortGDEngine)
                                          (->ShortGTEngine) (->ShortDTEngine) (->ShortSTEngine)))

(def openblas-byte (->BlasIntegerFactory openblas-int byte-accessor (->ByteVectorEngine) (->ByteGEEngine)
                                         (->ByteTREngine) (->ByteSYEngine)
                                         (->ByteGBEngine) (->ByteSBEngine) (->ByteTBEngine)
                                         (->ByteSPEngine) (->ByteTPEngine) (->ByteGDEngine)
                                         (->ByteGTEngine) (->ByteDTEngine) (->ByteSTEngine)))

(def openblas-float (->BlasRealFactory openblas-int float-accessor (->FloatVectorEngine) (->FloatGEEngine)
                                       (->FloatTREngine) (->FloatSYEngine)
                                       (->FloatGBEngine) (->FloatSBEngine) (->FloatTBEngine)
                                       (->FloatSPEngine) (->FloatTPEngine) (->FloatGDEngine)
                                       (->FloatGTEngine) (->FloatDTEngine) (->FloatSTEngine)
                                       nil nil))

(def openblas-double (->BlasRealFactory openblas-int double-accessor (->DoubleVectorEngine) (->DoubleGEEngine)
                                        (->DoubleTREngine) (->DoubleSYEngine)
                                        (->DoubleGBEngine) (->DoubleSBEngine) (->DoubleTBEngine)
                                        (->DoubleSPEngine) (->DoubleTPEngine) (->DoubleGDEngine)
                                        (->DoubleGTEngine) (->DoubleDTEngine) (->DoubleSTEngine)
                                        nil nil))
