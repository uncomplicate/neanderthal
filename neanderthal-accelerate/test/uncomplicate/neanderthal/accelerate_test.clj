 ;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.accelerate-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [ge tr sy gb tr tb sb sp tp gt dt st gd]]
             [native :refer [factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [math-test :as math-test]
             [random-test :as random-test]
             #_[sparse-test :as sparse-test]];;TODO dependent on MKL constants!
            [uncomplicate.neanderthal.internal.cpp.accelerate.factory
             :refer [accelerate-float accelerate-double accelerate-int accelerate-long accelerate-short accelerate-byte]]))
(facts "factory-by-type test"
       (factory-by-type :float) => accelerate-float
       (factory-by-type :double) => accelerate-double
       (factory-by-type :int) => accelerate-int
       (factory-by-type :long) => accelerate-long
       (factory-by-type :short) => accelerate-short
       (factory-by-type :byte) => accelerate-byte)

(block-test/test-all accelerate-double)
(block-test/test-all accelerate-float)
(block-test/test-host accelerate-float)
(block-test/test-host accelerate-double)

(for [factory0 [accelerate-double accelerate-float]
      factory1 [accelerate-long accelerate-int accelerate-short accelerate-byte]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)]) ;;TODO

(for [factory0 [accelerate-double accelerate-float]
      factory1 [accelerate-double accelerate-float]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)])

(block-test/test-extend-buffer accelerate-float)
(real-test/test-diag-equal accelerate-double)
(real-test/test-diag-equal accelerate-float)

(real-test/test-blas accelerate-double)
(real-test/test-blas accelerate-float)
(real-test/test-blas-host accelerate-double)
(real-test/test-blas-host accelerate-float)
(real-test/test-basic-integer accelerate-long)
(real-test/test-basic-integer accelerate-int)
(real-test/test-basic-integer accelerate-short)
(real-test/test-basic-integer accelerate-byte)
(real-test/test-basic-integer-host accelerate-long)
(real-test/test-basic-integer-host accelerate-int)
(real-test/test-basic-integer-host accelerate-short)
(real-test/test-basic-integer-host accelerate-byte)

(real-test/test-blas-sy-host accelerate-double)
(real-test/test-blas-sy-host accelerate-float)

(real-test/test-blas-gb accelerate-double)
(real-test/test-blas-gb accelerate-float)
(real-test/test-blas-gb-host accelerate-double)
(real-test/test-blas-gb-host accelerate-float)

(real-test/test-blas-sb accelerate-double)
(real-test/test-blas-sb accelerate-float)
(real-test/test-blas-sb-host accelerate-double)
(real-test/test-blas-sb-host accelerate-float)

(real-test/test-blas-tb accelerate-double)
(real-test/test-blas-tb accelerate-float)
(real-test/test-blas-tb-host accelerate-double)
(real-test/test-blas-tb-host accelerate-float)

(real-test/test-blas-tp accelerate-double)
(real-test/test-blas-tp accelerate-float)
(real-test/test-blas-tp-host accelerate-double)
(real-test/test-blas-tp-host accelerate-float)

(real-test/test-blas-sp accelerate-double)
(real-test/test-blas-sp accelerate-float)
(real-test/test-blas-sp-host accelerate-double)
(real-test/test-blas-sp-host accelerate-float)

(real-test/test-blas-gd accelerate-double)
(real-test/test-blas-gd accelerate-float)
(real-test/test-blas-gd-host accelerate-double)
(real-test/test-blas-gd-host accelerate-float)

(real-test/test-blas-gt accelerate-double)
(real-test/test-blas-gt accelerate-float)
(real-test/test-blas-gt-host accelerate-double)
(real-test/test-blas-gt-host accelerate-float)

(real-test/test-blas-dt accelerate-double)
(real-test/test-blas-dt accelerate-float)
(real-test/test-blas-dt-host accelerate-double)
(real-test/test-blas-dt-host accelerate-float)

(real-test/test-blas-st accelerate-double)
(real-test/test-blas-st accelerate-float)
(real-test/test-blas-st-host accelerate-double)
(real-test/test-blas-st-host accelerate-float)

(use 'uncomplicate.neanderthal.math-test)
(use 'uncomplicate.neanderthal.core)
(require  '[uncomplicate.neanderthal.vect-math :as vm])
(use 'uncomplicate.neanderthal.native)
(defn test-all-host [factory]
  (test-math factory diff-vctr-1 diff-vctr-2)
  (test-math factory diff-ge-1 diff-ge-2)
  (test-math factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  (test-math factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  (test-math factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  (test-math factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  (test-math factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  (test-math factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  (test-math factory (partial diff-square-1 st) (partial diff-square-2 st))
  ;;(test-math-host factory diff-vctr-1 diff-vctr-2)
  ;; (test-math-host factory diff-ge-1 diff-ge-2)
  ;; (test-math-host factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  ;; (test-math-host factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  ;; (test-math-host factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  ;; (test-math-host factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  ;; (test-math-host factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  ;; (test-math-host factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  ;; (test-math-host factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  ;; (test-math-host factory (partial diff-square-1 st) (partial diff-square-2 st))
  (test-math-inv factory diff-vctr-1 diff-vctr-2)
  (test-math-inv factory diff-ge-1 diff-ge-2)
  (test-math-inv factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math-inv factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  (test-math-inv factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  (test-math-inv factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  (test-math-inv factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  (test-math-inv factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  (test-math-inv factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  (test-math-inv factory (partial diff-square-1 st) (partial diff-square-2 st))
  (math-test/test-vctr-linear-frac factory)
  (math-test/test-ge-linear-frac factory)
  ;; (test-tr-linear-frac factory)
  ;; (test-sy-linear-frac factory)
  )
;; TODO not available in accelerate
(defn test-tr-linear-frac [factory]
  (facts "tr-linear-frac"
         (let [a (tr factory 3 [1 2 3 1 2 3])
                        b (tr factory 3 [2 3 4 2 3 4])
                        host-expected (tr factory 3 [5/13 7/17 9/21 5/13 7/17 9/21])
                        result (vm/linear-frac 2 a 3 4 b 5)
                                        ;
               host-result (transfer result)
               ]
           (nrm2 (axpy! -1 host-result host-expected)) => 0.0
           )))
;;(test-tr-linear-frac accelerate-double)
(test-all-host accelerate-float)

(random-test/test-all accelerate-double)
(random-test/test-all accelerate-float)
(random-test/test-all-host accelerate-double)
(random-test/test-all-host accelerate-float)

;; (sparse-test/test-all accelerate-float accelerate-double)
;; (sparse-test/test-all accelerate-double accelerate-float)

(real-test/test-lapack accelerate-double)
(real-test/test-lapack accelerate-float)
