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
             [core :refer [ge tr sy gb tr tb sb sp tp gt dt st gd]]
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
  (real-test/test-sy-rk factory sp))

(test-blas-host mkl-double)
(test-blas-host mkl-float)

(defn test-lapack [factory]
  (real-test/test-vctr-srt factory)
  (real-test/test-ge-srt factory)
  (real-test/test-ge-swap factory)
  (real-test/test-uplo-srt factory tr)
  (real-test/test-uplo-srt factory sy)
  (real-test/test-gb-srt factory)
  (real-test/test-banded-uplo-srt factory tb)
  (real-test/test-banded-uplo-srt factory sb)
  (real-test/test-uplo-srt factory sp)
  (real-test/test-uplo-srt factory tp)
  (real-test/test-gt-srt factory gt)
  (real-test/test-gt-srt factory dt)
  (real-test/test-gd-srt factory)
  (real-test/test-ge-trf factory)
  (real-test/test-ge-trs factory)
  (real-test/test-tr-trs factory tr)
  (real-test/test-tr-trs factory tp)
  (real-test/test-ge-tri factory)
  (real-test/test-tr-tri factory tr)
  (real-test/test-tr-tri factory tp)
  (real-test/test-ge-con factory)
  (real-test/test-tr-con factory tr)
  (real-test/test-tr-con factory tp)
  (real-test/test-ge-det factory)
  (real-test/test-ge-det2x2 factory)
  (real-test/test-ge-sv factory)
  (real-test/test-tr-sv factory tr)
  (real-test/test-tr-sv factory tp)
  (real-test/test-sy-sv factory)
  (real-test/test-sy-trx factory sy)
 (real-test/test-sy-potrx factory)
  (real-test/test-gb-trx factory)
  (real-test/test-sb-trx factory)
  (real-test/test-tb-trx factory)
  (real-test/test-sy-trx factory sp)
  (real-test/test-gt-trx factory)
  (real-test/test-sp-potrx factory)
  (real-test/test-gd-trx factory)
  (real-test/test-gt-trx factory)
  (real-test/test-dt-trx factory)
  (real-test/test-st-trx factory)
  (real-test/test-ge-qr factory)
  (real-test/test-ge-qp factory)
  (real-test/test-ge-rq factory)
  (real-test/test-ge-lq factory)
  (real-test/test-ge-ql factory)
  (real-test/test-ge-ls factory)
  (real-test/test-ge-lse factory)
  (real-test/test-ge-gls factory)
  (real-test/test-ge-ev factory)
  (real-test/test-sy-evd factory)
  (real-test/test-sy-evr factory)
  (real-test/test-ge-es factory)
  (real-test/test-ge-svd factory)
  )

(test-lapack mkl-double)
(test-lapack mkl-float)


;; TODO math-test for GB, SB, TB
#_(test-all-host mkl-float)

(block-test/test-all mkl-double)
(block-test/test-all mkl-float)
(block-test/test-host mkl-float)
(block-test/test-host mkl-double)


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
(real-test/test-lapack mkl-double)
(real-test/test-lapack mkl-float)
(real-test/test-blas-sy-host mkl-double)
(real-test/test-blas-sy-host mkl-float)

(real-test/test-blas-gb mkl-double)
(real-test/test-blas-gb mkl-float)
(real-test/test-blas-gb-host mkl-double)
(real-test/test-blas-gb-host mkl-float)

(real-test/test-blas-sb mkl-double)
(real-test/test-blas-sb mkl-float)
(real-test/test-blas-sb-host mkl-double)
(real-test/test-blas-sb-host mkl-float)

(real-test/test-blas-tb mkl-double)
(real-test/test-blas-tb mkl-float)
(real-test/test-blas-tb-host mkl-double)
(real-test/test-blas-tb-host mkl-float)


(math-test/test-all-host mkl-double)
(math-test/test-all-host mkl-float)
(random-test/test-all mkl-double)
(random-test/test-all mkl-float)
(random-test/test-all-host mkl-double)
(random-test/test-all-host mkl-float)



;; ================= Sparse tests ===============================
