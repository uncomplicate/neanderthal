;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import org.bytedeco.mkl.global.mkl_rt))

(def ^:const mkl-enable-instructions
  {:avx mkl_rt/MKL_ENABLE_AVX
   :sse42 mkl_rt/MKL_ENABLE_SSE4_2
   :avx512 mkl_rt/MKL_ENABLE_AVX512
   :avx512-e1 mkl_rt/MKL_ENABLE_AVX512_E1
   :avx512-e2 mkl_rt/MKL_ENABLE_AVX512_E2
   :avx512-e3 mkl_rt/MKL_ENABLE_AVX512_E3
   :avx512-e4 mkl_rt/MKL_ENABLE_AVX512_E4
   :avx2 mkl_rt/MKL_ENABLE_AVX2
   :avx2-e1 mkl_rt/MKL_ENABLE_AVX2_E1})

(def ^:const mkl-verbose-mode
  {:timing 2
   :log 1
   :none 0})

(defn dec-verbose-mode [^long mode]
  (case mode
    2 :timing
    1 :log
    0 :none
    (dragan-says-ex "Unknown verbose mode." {:mode mode})))

(defn dec-mkl-result [^long result]
  (case result
    1 true
    0 false
    (dragan-says-ex "Unknown MKL result type." {:result result})))

(def ^:const mkl-peak-mem
  {:report mkl_rt/MKL_PEAK_MEM
   :enable mkl_rt/MKL_PEAK_MEM_ENABLE
   :disable mkl_rt/MKL_PEAK_MEM_DISABLE
   :reset mkl_rt/MKL_PEAK_MEM_RESET})

(defn dec-sparse-status [^long status]
  (case status
    0 :success
    1 :not-initialized
    2 :alloc-failed
    3 :invalid-value
    4 :execution-failed
    5 :internal-error
    6 :not-supported
    :unknown))

(def ^:const mkl-sparse-operation
  {:no-trans mkl_rt/SPARSE_OPERATION_NON_TRANSPOSE
   111 mkl_rt/SPARSE_OPERATION_NON_TRANSPOSE
   :trans mkl_rt/SPARSE_OPERATION_TRANSPOSE
   112 mkl_rt/SPARSE_OPERATION_TRANSPOSE
   :conj-trans mkl_rt/SPARSE_OPERATION_CONJUGATE_TRANSPOSE
   113 mkl_rt/SPARSE_OPERATION_CONJUGATE_TRANSPOSE})

(def ^:const mkl-sparse-matrix-type
  {:ge mkl_rt/SPARSE_MATRIX_TYPE_GENERAL
   :sy mkl_rt/SPARSE_MATRIX_TYPE_SYMMETRIC
   :he mkl_rt/SPARSE_MATRIX_TYPE_HERMITIAN
   :tr mkl_rt/SPARSE_MATRIX_TYPE_TRIANGULAR
   :gd mkl_rt/SPARSE_MATRIX_TYPE_DIAGONAL
   :btr mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR
   :bgd mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL})

(def ^:const mkl-sparse-fill-mode
  {:lower mkl_rt/SPARSE_FILL_MODE_LOWER
   mkl_rt/SPARSE_FILL_MODE_LOWER mkl_rt/SPARSE_FILL_MODE_LOWER
   122 mkl_rt/SPARSE_FILL_MODE_LOWER
   :upper mkl_rt/SPARSE_FILL_MODE_UPPER
   mkl_rt/SPARSE_FILL_MODE_UPPER mkl_rt/SPARSE_FILL_MODE_UPPER
   121 mkl_rt/SPARSE_FILL_MODE_UPPER
   :full mkl_rt/SPARSE_FILL_MODE_FULL
   mkl_rt/SPARSE_FILL_MODE_FULL mkl_rt/SPARSE_FILL_MODE_FULL
   :ge mkl_rt/SPARSE_FILL_MODE_FULL
   mkl_rt/SPARSE_MATRIX_TYPE_GENERAL mkl_rt/SPARSE_FILL_MODE_FULL
   :sy mkl_rt/SPARSE_FILL_MODE_LOWER
   mkl_rt/SPARSE_MATRIX_TYPE_SYMMETRIC mkl_rt/SPARSE_FILL_MODE_LOWER
   :he mkl_rt/SPARSE_FILL_MODE_LOWER
   mkl_rt/SPARSE_MATRIX_TYPE_HERMITIAN mkl_rt/SPARSE_FILL_MODE_LOWER
   :tr mkl_rt/SPARSE_FILL_MODE_LOWER
   mkl_rt/SPARSE_MATRIX_TYPE_TRIANGULAR mkl_rt/SPARSE_FILL_MODE_LOWER
   :gd mkl_rt/SPARSE_FILL_MODE_FULL
   mkl_rt/SPARSE_MATRIX_TYPE_DIAGONAL mkl_rt/SPARSE_FILL_MODE_FULL
   :btr mkl_rt/SPARSE_FILL_MODE_LOWER
   mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR mkl_rt/SPARSE_FILL_MODE_LOWER
   :bgd mkl_rt/SPARSE_FILL_MODE_FULL
   mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL mkl_rt/SPARSE_FILL_MODE_FULL})

(def ^:const mkl-sparse-diag-mode
  {:non-unit mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_DIAG_NON_UNIT mkl_rt/SPARSE_DIAG_NON_UNIT
   131 mkl_rt/SPARSE_DIAG_NON_UNIT
   :unit mkl_rt/SPARSE_DIAG_UNIT
   mkl_rt/SPARSE_DIAG_UNIT mkl_rt/SPARSE_DIAG_UNIT
   132 mkl_rt/SPARSE_DIAG_UNIT
   :ge mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_GENERAL mkl_rt/SPARSE_DIAG_NON_UNIT
   :sy mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_SYMMETRIC mkl_rt/SPARSE_DIAG_NON_UNIT
   :he mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_HERMITIAN mkl_rt/SPARSE_DIAG_NON_UNIT
   :tr mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_TRIANGULAR mkl_rt/SPARSE_DIAG_NON_UNIT
   :gd mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_DIAGONAL mkl_rt/SPARSE_DIAG_NON_UNIT
   :btr mkl_rt/SPARSE_DIAG_NON_UNIT
   mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR mkl_rt/SPARSE_DIAG_NON_UNIT
   :bgd mkl_rt/SPARSE_DIAG_NON_UNIT

   mkl_rt/SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL mkl_rt/SPARSE_DIAG_NON_UNIT})

(def ^:const mkl-sparse-layout
  {:row mkl_rt/SPARSE_LAYOUT_ROW_MAJOR
   101 mkl_rt/SPARSE_LAYOUT_ROW_MAJOR
   :column mkl_rt/SPARSE_LAYOUT_COLUMN_MAJOR
   102 mkl_rt/SPARSE_LAYOUT_COLUMN_MAJOR})

(def ^:const mkl-sparse-request
  {:count mkl_rt/SPARSE_STAGE_NNZ_COUNT
   :finalize-no-val mkl_rt/SPARSE_STAGE_FINALIZE_MULT_NO_VAL
   :finalize mkl_rt/SPARSE_STAGE_FINALIZE_MULT
   :full-no-val mkl_rt/SPARSE_STAGE_FULL_MULT_NO_VAL
   :full mkl_rt/SPARSE_STAGE_FULL_MULT})
