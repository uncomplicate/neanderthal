;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.openblas.core
  (:require [clojure.string :refer [trim split]])
  (:import org.bytedeco.openblas.global.openblas))

;; ===================== System ================================================================

(defn version []
  (let [ver-seq (split (second (split (trim openblas/OPENBLAS_VERSION) #" ")) #"\.")]
    {:major (Long/parseLong (ver-seq 0))
     :minor (Long/parseLong (ver-seq 1))
     :update (Long/parseLong (ver-seq 2))}))

(defn vendor []
  (case (openblas/blas_get_vendor)
    0 :unknown
    1 :cublas
    2 :openblas
    3 :mkl))

;; ===================== Miscellaneous =========================================================

(defn num-threads ^long []
  (openblas/blas_get_num_threads))

(defn num-threads!
  ([]
   (openblas/blas_set_num_threads -1))
  ([^long n]
   (openblas/blas_set_num_threads n)))
