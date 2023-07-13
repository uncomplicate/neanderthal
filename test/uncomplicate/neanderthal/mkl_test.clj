;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.mkl-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [ge tr sy]]
             [native :refer [factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [math-test :as math-test]
             [random-test :as random-test]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory
             :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]))

(facts "factory-by-type test"
       (factory-by-type :float) => mkl-float
       (factory-by-type :double) => mkl-double
       (factory-by-type :int) => mkl-int
       (factory-by-type :long) => mkl-long
       (factory-by-type :short) => mkl-short
       (factory-by-type :byte) => mkl-byte)

(defn test-host [factory]
  (block-test/test-uplo-contiguous factory sy)
  (block-test/test-vctr-ifn factory)
  (block-test/test-vctr-functor factory)
  (block-test/test-vctr-fold factory)
  (block-test/test-vctr-reducible factory)
  (block-test/test-vctr-seq factory)
  (block-test/test-ge-functor-laws factory ge)
  (block-test/test-ge-ifn factory)
  (block-test/test-ge-functor factory)
  (block-test/test-ge-fold factory)
  (block-test/test-ge-reducible factory)
  (block-test/test-ge-seq factory)
  (block-test/test-tr-functor factory)
  (block-test/test-tr-fold factory)
  (block-test/test-tr-reducible factory)
  (block-test/test-tr-seq factory)
  (block-test/test-vctr-functor-laws factory)
  (block-test/test-ge-functor-laws factory ge)
  (block-test/test-sspar-mat-functor-laws factory tr)
  (block-test/test-sspar-mat-functor-laws factory sy)
  ;;TODO
  ;; (test-sspar-mat-functor-laws factory sb)
  ;; (test-sspar-mat-functor-laws factory sp)
  ;; (test-sspar-mat-functor-laws factory tp)
  ;; (test-sspar-mat-functor-laws factory gd)
  ;; (test-sspar-mat-functor-laws factory gt)
  ;; (test-sspar-mat-functor-laws factory st)
  )

(defn test-blas-host [factory]
  (real-test/test-iamin factory)
  (real-test/test-imax factory)
  (real-test/test-imin factory)
  ;;TODO
  ;; (real-test/test-rot factory)
  ;; (real-test/test-rotg factory)
  ;; (real-test/test-rotm factory)
  ;; (real-test/test-rotmg factory)
  (real-test/test-sum-asum factory)
  (real-test/test-vctr-entry factory)
  (real-test/test-vctr-entry! factory)
  (real-test/test-vctr-alter! factory)
  (real-test/test-vctr-amax factory)
  (real-test/test-ge-entry factory)
  (real-test/test-ge-entry! factory)
  (real-test/test-ge-alter! factory)
  (real-test/test-ge-amax factory)
  (real-test/test-ge-sum factory)
  (real-test/test-ge-sum-asum factory)
  (real-test/test-ge-dot-strided factory)
  (real-test/test-ge-trans! factory)
  (real-test/test-tr-entry factory tr)
  (real-test/test-tr-entry! factory tr)
  (real-test/test-tr-bulk-entry! factory tr)
  (real-test/test-tr-alter! factory tr)
  (real-test/test-tr-dot factory tr)
  (real-test/test-tr-nrm2 factory tr)
  (real-test/test-tr-asum factory tr)
  (real-test/test-tr-sum factory tr)
  (real-test/test-tr-amax factory tr)
  ;;TODO
  #_(real-test/test-sy-rk factory sp))

(defn test-lapack [factory]
  (real-test/test-vctr-srt factory)
  (real-test/test-ge-srt factory)
  (real-test/test-ge-swap factory)
  (real-test/test-uplo-srt factory tr)
  (real-test/test-uplo-srt factory sy)
  ;; (real-test/test-uplo-srt factory sp)
  ;; (real-test/test-uplo-srt factory tp)
  ;; (test-gb-srt factory)
  ;; (test-banded-uplo-srt factory tb)
  ;; (test-banded-uplo-srt factory sb)
  ;; (test-gt-srt factory gt)
  ;; (test-gt-srt factory dt)
  ;; (test-gd-srt factory)
  ;; (test-ge-trf factory)
  ;; (test-ge-trs factory)
  ;; (test-tr-trs factory tr)
  ;; (test-tr-trs factory tp)
  ;; (test-ge-tri factory)
  ;; (test-tr-tri factory tr)
  ;; (test-tr-tri factory tp)
  ;; (test-ge-con factory)
  ;; (test-tr-con factory tr)
  ;; (test-tr-con factory tp)
  ;; (test-ge-det factory)
  ;; (test-ge-det2x2 factory)
  ;; (test-ge-sv factory)
  ;; (test-tr-sv factory tr)
  ;; (test-tr-sv factory tp)
  ;; (test-sy-sv factory)
  ;; (test-sy-trx factory sy)
  ;; (test-sy-potrx factory)
  ;; (test-gb-trx factory)
  ;; (test-sb-trx factory)
  ;; (test-tb-trx factory)
  ;; (test-sy-trx factory sp)
  ;; (test-gt-trx factory)
  ;; (test-sp-potrx factory)
  ;; (test-gd-trx factory)
  ;; (test-gt-trx factory)
  ;; (test-dt-trx factory)
  ;; (test-st-trx factory)
  ;; (test-ge-qr factory)
  ;; (test-ge-qp factory)
  ;; (test-ge-rq factory)
  ;; (test-ge-lq factory)
  ;; (test-ge-ql factory)
  ;; (test-ge-ls factory)
  ;; (test-ge-lse factory)
  ;; (test-ge-gls factory)
  ;; (test-ge-ev factory)
  ;; (test-sy-evd factory)
  ;; (test-sy-evr factory)
  ;; (test-ge-es factory)
  ;; (test-ge-svd factory)
  )

(defn test-all-host [factory]
  (math-test/test-math factory math-test/diff-vctr-1 math-test/diff-vctr-2)
  (math-test/test-math factory math-test/diff-ge-1 math-test/diff-ge-2)
  (math-test/test-math factory (partial math-test/diff-square-1 tr) (partial math-test/diff-square-2 tr))
  (math-test/test-math factory (partial math-test/diff-square-1 sy) (partial math-test/diff-square-2 sy))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 tp) (partial math-test/diff-square-2 tp))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 sp) (partial math-test/diff-square-2 sp))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 gd) (partial math-test/diff-square-2 gd))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 gt) (partial math-test/diff-square-2 gt))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 dt) (partial math-test/diff-square-2 dt))
  ;; (math-test/test-math factory (partial math-test/diff-square-1 st) (partial math-test/diff-square-2 st))
  (math-test/test-math-host factory math-test/diff-vctr-1 math-test/diff-vctr-2)
  (math-test/test-math-host factory math-test/diff-ge-1 math-test/diff-ge-2)
  ;;(math-test/test-math-host factory (partial math-test/diff-square-1 tr) (partial math-test/diff-square-2 tr))
  ;;   (math-test/test-math-host factory (partial math-test/diff-square-1 sy) (partial math-test/diff-square-2 sy))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 tp) (partial math-test/diff-square-2 tp))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 sp) (partial math-test/diff-square-2 sp))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 gd) (partial math-test/diff-square-2 gd))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 gt) (partial math-test/diff-square-2 gt))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 dt) (partial math-test/diff-square-2 dt))
  ;; (math-test/test-math-host factory (partial math-test/diff-square-1 st) (partial math-test/diff-square-2 st))
  (math-test/test-math-inv factory math-test/diff-vctr-1 math-test/diff-vctr-2)
  (math-test/test-math-inv factory math-test/diff-ge-1 math-test/diff-ge-2)
  (math-test/test-math-inv factory (partial math-test/diff-square-1 tr) (partial math-test/diff-square-2 tr))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 sy) (partial math-test/diff-square-2 sy))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 tp) (partial math-test/diff-square-2 tp))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 sp) (partial math-test/diff-square-2 sp))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 gd) (partial math-test/diff-square-2 gd))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 gt) (partial math-test/diff-square-2 gt))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 dt) (partial math-test/diff-square-2 dt))
  ;; (math-test/test-math-inv factory (partial math-test/diff-square-1 st) (partial math-test/diff-square-2 st))
  (math-test/test-vctr-linear-frac factory)
  (math-test/test-ge-linear-frac factory)
  (math-test/test-tr-linear-frac factory)
  (math-test/test-sy-linear-frac factory))

(test-all-host mkl-double)

(block-test/test-all mkl-double)
(block-test/test-all mkl-float)
(test-host mkl-float)
(test-host mkl-double)


(for [factory0 [mkl-double mkl-float]
      factory1 [mkl-long mkl-int mkl-short mkl-byte]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)]) ;;TODO

(for [factory0 [mkl-double mkl-float]
      factory1 [mkl-double mkl-float]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)])

(block-test/test-extend-buffer mkl-float)

(real-test/test-blas mkl-double)
(real-test/test-blas mkl-float)
(test-blas-host mkl-double)
(test-blas-host mkl-float)
(real-test/test-basic-integer mkl-long)
(real-test/test-basic-integer mkl-int)
(real-test/test-basic-integer mkl-short)
(real-test/test-basic-integer mkl-byte)
(real-test/test-basic-integer-host mkl-long)
(real-test/test-basic-integer-host mkl-int)
(real-test/test-basic-integer-host mkl-short)
(real-test/test-basic-integer-host mkl-byte)
(test-lapack mkl-double)
(test-lapack mkl-float)
(real-test/test-blas-sy-host mkl-double)
(real-test/test-blas-sy-host mkl-float)


(test-all-host mkl-double)
(test-all-host mkl-float)
(random-test/test-all mkl-double)
(random-test/test-all mkl-float)
(random-test/test-all-host mkl-double)
(random-test/test-all-host mkl-float)

;; ================= Sparse tests ===============================
