;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.mkl.constants-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.neanderthal.internal.constants :refer :all])
  (:import org.bytedeco.mkl.global.mkl_rt))

(facts "BLAS constants should match mkl_rt enums."
       (:row blas-layout) => mkl_rt/CblasRowMajor
       (:col blas-layout) => mkl_rt/CblasColMajor
       (:column blas-layout) => mkl_rt/CblasColMajor

       (:upper blas-uplo) => mkl_rt/CblasUpper
       (:low blas-uplo) => mkl_rt/CblasLower
       (:lower blas-uplo) => mkl_rt/CblasLower

       (:trans blas-transpose) => mkl_rt/CblasTrans
       (:no-trans blas-transpose) => mkl_rt/CblasNoTrans
       (:no blas-transpose) => mkl_rt/CblasNoTrans

       (:left blas-side) => mkl_rt/CblasLeft
       (:right blas-side) => mkl_rt/CblasRight

       (:unit blas-diag) => mkl_rt/CblasUnit
       (:non-unit blas-diag) => mkl_rt/CblasNonUnit
       )
