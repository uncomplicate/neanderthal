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

;; TODO
#_(facts "factory-by-type test"
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

(defn test-blas-gt [factory]
  (real-test/test-gt-constructor factory gt)
  (real-test/test-gt factory gt)
  (real-test/test-gt-copy factory gt)
  (real-test/test-gt-swap factory gt)
  (real-test/test-gt-scal factory gt)
  (real-test/test-gt-axpy factory gt)
  ;;(real-test/test-gt-mv factory gt) TODO not supported by accelerate
  ;;(real-test/test-gt-mv factory gt) TODO not supported by accelerate
  )

(test-blas-gt accelerate-double)
(test-blas-gt accelerate-float)
(real-test/test-blas-gt-host accelerate-double)
(real-test/test-blas-gt-host accelerate-float)

(defn test-blas-dt [factory]
  (real-test/test-gt-constructor factory dt)
  (real-test/test-gt factory dt)
  (real-test/test-gt-copy factory dt)
  (real-test/test-gt-swap factory dt)
  (real-test/test-gt-scal factory dt)
  (real-test/test-gt-axpy factory dt)
  ;; (real-test/test-gt-mv factory dt) ;;TODO not supported by accelerate
  ;; (real-test/test-gt-mm factory dt) ;;TODO not supported by accelerate
  )
(test-blas-dt accelerate-double)
(test-blas-dt accelerate-float)
(real-test/test-blas-dt-host accelerate-double)
(real-test/test-blas-dt-host accelerate-float)

(defn test-blas-st [factory]
  (real-test/test-st-constructor factory)
  (real-test/test-st factory)
  (real-test/test-gt-copy factory st)
  (real-test/test-gt-swap factory st)
  (real-test/test-gt-scal factory st)
  (real-test/test-gt-axpy factory st)
  ;;(real-test/test-st-mv factory) TODO not supported by accelerate
  ;;(real-test/test-st-mm factory) TODO not supported by accelerate
  )

(test-blas-st accelerate-double)
(test-blas-st accelerate-float)
(real-test/test-blas-st-host accelerate-double)
(real-test/test-blas-st-host accelerate-float)

;; TODO not available in accelerate
;; (math-test/test-all-host accelerate-double)
;; (math-test/test-all-host accelerate-float)

(random-test/test-all accelerate-double)
(random-test/test-all accelerate-float)
(random-test/test-all-host accelerate-double)
(random-test/test-all-host accelerate-float)

;; (sparse-test/test-all accelerate-float accelerate-double)
;; (sparse-test/test-all accelerate-double accelerate-float)

(real-test/test-lapack accelerate-double)
(real-test/test-lapack accelerate-float)
