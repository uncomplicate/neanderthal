;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.cuda.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import org.bytedeco.cuda.global.cublas))

(def ^{:const true
       :doc "CUDA Error messages as defined in nvrtc."}
  cublas-status-codes
  {cublas/CUBLAS_STATUS_ALLOC_FAILED :alloc-failed
   cublas/CUBLAS_STATUS_ARCH_MISMATCH :arch-mismatch
   cublas/CUBLAS_STATUS_EXECUTION_FAILED :executin-failed
   cublas/CUBLAS_STATUS_INTERNAL_ERROR :internal-error
   cublas/CUBLAS_STATUS_INVALID_VALUE :invalid-value
   cublas/CUBLAS_STATUS_LICENSE_ERROR :license-error
   cublas/CUBLAS_STATUS_MAPPING_ERROR :mapping-error
   cublas/CUBLAS_STATUS_NOT_INITIALIZED :not-initialized
   cublas/CUBLAS_STATUS_NOT_SUPPORTED :not-supported
   cublas/CUBLAS_STATUS_SUCCESS :success})

(def ^:const cublas-trans
  {:no-trans cublas/CUBLAS_OP_N
   :trans cublas/CUBLAS_OP_T})

(def ^:const cublas-uplo
  {:lower cublas/CUBLAS_FILL_MODE_LOWER
   :upper cublas/CUBLAS_FILL_MODE_UPPER})

(def ^:const cublas-diag-unit
  {:unit cublas/CUBLAS_DIAG_UNIT
   :diag-unit cublas/CUBLAS_DIAG_UNIT
   :non-unit cublas/CUBLAS_DIAG_NON_UNIT})

(def ^:const cublas-side-mode
  {:left cublas/CUBLAS_SIDE_LEFT
   :right cublas/CUBLAS_SIDE_RIGHT})
