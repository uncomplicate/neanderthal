;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.mkl-test
  (:require [uncomplicate.neanderthal.block-test :as block-test]
            [uncomplicate.neanderthal.real-test :as real-test]
            [uncomplicate.neanderthal.internal.host.mkl :refer [mkl-float mkl-double]]))

(block-test/test-all mkl-double)
(block-test/test-all mkl-float)
(block-test/test-host mkl-double)
(block-test/test-host mkl-float)
(block-test/test-both-factories mkl-float mkl-double)
(block-test/test-both-factories mkl-float mkl-double)
(real-test/test-blas mkl-double)
(real-test/test-blas mkl-float)
(real-test/test-blas-host mkl-double)
(real-test/test-blas-host mkl-float)
(real-test/test-lapack mkl-double)
(real-test/test-lapack mkl-float)

(real-test/test-blas-sy mkl-double)
(real-test/test-blas-sy mkl-float)
(real-test/test-blas-sy-host mkl-double)
(real-test/test-blas-sy-host mkl-float)

(real-test/test-blas-gb mkl-double)
(real-test/test-blas-gb-host mkl-double)
(real-test/test-blas-gb mkl-float)
(real-test/test-blas-gb-host mkl-float)

(real-test/test-blas-tp mkl-double)
(real-test/test-blas-tp-host mkl-double)
(real-test/test-lapack-tp mkl-double)

(real-test/test-blas-sp mkl-double)
(real-test/test-blas-sp-host mkl-double)
(real-test/test-lapack-sp mkl-double)
