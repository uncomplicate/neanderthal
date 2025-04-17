;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.openblas-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [ge tr sy gb tr tb sb sp tp gt dt st gd]]
             [native :refer [factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [math-test :as math-test]
             [random-test :as random-test]]
            [uncomplicate.neanderthal.internal.cpp.openblas.factory
             :refer [openblas-float openblas-double openblas-int openblas-long
                     openblas-short openblas-byte]]))

(facts "factory-by-type test"
       (factory-by-type :float) => openblas-float
       (factory-by-type :double) => openblas-double
       (factory-by-type :int) => openblas-int
       (factory-by-type :long) => openblas-long
       (factory-by-type :short) => openblas-short
       (factory-by-type :byte) => openblas-byte)

(block-test/test-all openblas-double)
(block-test/test-all openblas-float)
(block-test/test-host openblas-float)
(block-test/test-host openblas-double)

(for [factory0 [openblas-double openblas-float]
      factory1 [openblas-long openblas-int openblas-short openblas-byte]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)]) ;;TODO

(for [factory0 [openblas-double openblas-float]
      factory1 [openblas-double openblas-float]]
  [(block-test/test-two-factories factory0 factory1)
   (block-test/test-two-factories factory1 factory0)])

(block-test/test-extend-buffer openblas-float)
(real-test/test-diag-equal openblas-double)
(real-test/test-diag-equal openblas-float)

(real-test/test-blas openblas-double)
(real-test/test-blas openblas-float)
(real-test/test-blas-host openblas-double)
(real-test/test-blas-host openblas-float)
(real-test/test-basic-integer openblas-long)
(real-test/test-basic-integer openblas-int)
(real-test/test-basic-integer openblas-short)
(real-test/test-basic-integer openblas-byte)
(real-test/test-basic-integer-host openblas-long)
(real-test/test-basic-integer-host openblas-int)
(real-test/test-basic-integer-host openblas-short)
(real-test/test-basic-integer-host openblas-byte)
(real-test/test-blas-sy-host openblas-double)
(real-test/test-blas-sy-host openblas-float)

(real-test/test-blas-gb openblas-double)
(real-test/test-blas-gb openblas-float)
(real-test/test-blas-gb-host openblas-double)
(real-test/test-blas-gb-host openblas-float)

(real-test/test-blas-sb openblas-double)
(real-test/test-blas-sb openblas-float)
(real-test/test-blas-sb-host openblas-double)
(real-test/test-blas-sb-host openblas-float)

(real-test/test-blas-tb openblas-double)
(real-test/test-blas-tb openblas-float)
(real-test/test-blas-tb-host openblas-double)
(real-test/test-blas-tb-host openblas-float)

(real-test/test-blas-tp openblas-double)
(real-test/test-blas-tp openblas-float)
(real-test/test-blas-tp-host openblas-double)
(real-test/test-blas-tp-host openblas-float)

(real-test/test-blas-sp openblas-double)
(real-test/test-blas-sp openblas-float)
(real-test/test-blas-sp-host openblas-double)
(real-test/test-blas-sp-host openblas-float)

(real-test/test-blas-gd openblas-double)
(real-test/test-blas-gd openblas-float)
(real-test/test-blas-gd-host openblas-double)
(real-test/test-blas-gd-host openblas-float)

(defn test-blas-gt [factory]
  (real-test/test-gt-constructor factory gt)
  (real-test/test-gt factory gt)
  (real-test/test-gt-copy factory gt)
  (real-test/test-gt-swap factory gt)
  (real-test/test-gt-scal factory gt)
  (real-test/test-gt-axpy factory gt)
  ;;(real-test/test-gt-mv factory gt) TODO not supported by openblas
  ;;(real-test/test-gt-mv factory gt) TODO not supported by openblas
  )

(test-blas-gt openblas-double)
(test-blas-gt openblas-float)
(real-test/test-blas-gt-host openblas-double)
(real-test/test-blas-gt-host openblas-float)

(defn test-blas-dt [factory]
  (real-test/test-gt-constructor factory dt)
  (real-test/test-gt factory dt)
  (real-test/test-gt-copy factory dt)
  (real-test/test-gt-swap factory dt)
  (real-test/test-gt-scal factory dt)
  (real-test/test-gt-axpy factory dt)
  ;; (real-test/test-gt-mv factory dt) ;;TODO not supported by openblas
  ;; (real-test/test-gt-mm factory dt) ;;TODO not supported by openblas
  )
(test-blas-dt openblas-double)
(test-blas-dt openblas-float)
(real-test/test-blas-dt-host openblas-double)
(real-test/test-blas-dt-host openblas-float)

(defn test-blas-st [factory]
  (real-test/test-st-constructor factory)
  (real-test/test-st factory)
  (real-test/test-gt-copy factory st)
  (real-test/test-gt-swap factory st)
  (real-test/test-gt-scal factory st)
  (real-test/test-gt-axpy factory st)
  ;;(real-test/test-st-mv factory) TODO not supported by openblas
  ;;(real-test/test-st-mm factory) TODO not supported by openblas
  )

(test-blas-st openblas-double)
(test-blas-st openblas-float)
(real-test/test-blas-st-host openblas-double)
(real-test/test-blas-st-host openblas-float)

;; TODO not available in openblas
;; (math-test/test-all-host openblas-double)
;; (math-test/test-all-host openblas-float)

(random-test/test-all openblas-double)
(random-test/test-all openblas-float)
(random-test/test-all-host openblas-double)
(random-test/test-all-host openblas-float)

;; (sparse-test/test-all openblas-float openblas-double)
;; (sparse-test/test-all openblas-double openblas-float)

(real-test/test-lapack openblas-double)
(real-test/test-lapack openblas-float)
