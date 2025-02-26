;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.mkl-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer [ge tr sy gb tr tb sb sp tp gt dt st gd]]
             [native :refer [factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [math-test :as math-test]
             [random-test :as random-test]
             [sparse-test :as sparse-test]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory
             :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]))

(facts "factory-by-type test"
       (factory-by-type :float) => mkl-float
       (factory-by-type :double) => mkl-double
       (factory-by-type :int) => mkl-int
       (factory-by-type :long) => mkl-long
       (factory-by-type :short) => mkl-short
       (factory-by-type :byte) => mkl-byte)

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
(real-test/test-diag-equal mkl-double)
(real-test/test-diag-equal mkl-float)

(real-test/test-blas mkl-double)
(real-test/test-blas mkl-float)
(real-test/test-blas-host mkl-double)
(real-test/test-blas-host mkl-float)
(real-test/test-basic-integer mkl-long)
(real-test/test-basic-integer mkl-int)
(real-test/test-basic-integer mkl-short)
(real-test/test-basic-integer mkl-byte)
(real-test/test-basic-integer-host mkl-long)
(real-test/test-basic-integer-host mkl-int)
(real-test/test-basic-integer-host mkl-short)
(real-test/test-basic-integer-host mkl-byte)

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

(real-test/test-blas-tp mkl-double)
(real-test/test-blas-tp mkl-float)
(real-test/test-blas-tp-host mkl-double)
(real-test/test-blas-tp-host mkl-float)

(real-test/test-blas-sp mkl-double)
(real-test/test-blas-sp mkl-float)
(real-test/test-blas-sp-host mkl-double)
(real-test/test-blas-sp-host mkl-float)

(real-test/test-blas-gd mkl-double)
(real-test/test-blas-gd mkl-float)
(real-test/test-blas-gd-host mkl-double)
(real-test/test-blas-gd-host mkl-float)

(real-test/test-blas-gt mkl-double)
(real-test/test-blas-gt mkl-float)
(real-test/test-blas-gt-host mkl-double)
(real-test/test-blas-gt-host mkl-float)

(real-test/test-blas-dt mkl-double)
(real-test/test-blas-dt mkl-float)
(real-test/test-blas-dt-host mkl-double)
(real-test/test-blas-dt-host mkl-float)

(real-test/test-blas-st mkl-double)
(real-test/test-blas-st mkl-float)
(real-test/test-blas-st-host mkl-double)
(real-test/test-blas-st-host mkl-float)

(math-test/test-all-host mkl-double)
(math-test/test-all-host mkl-float)

(random-test/test-all mkl-double)
(random-test/test-all mkl-float)
(random-test/test-all-host mkl-double)
(random-test/test-all-host mkl-float)

(sparse-test/test-all mkl-float mkl-double)
(sparse-test/test-all mkl-double mkl-float)

(real-test/test-lapack mkl-double)
(real-test/test-lapack mkl-float)

(real-test/test-dt-trx mkl-double)
(real-test/test-dt-trx mkl-float)
