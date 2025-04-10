;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.accelerate.core
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import uncomplicate.javacpp.accelerate.global.thread))

(defn threading []
  (let [threading (thread/BLASGetThreading)]
    (case threading
      thread/BLAS_THREADING_MULTI_THREADED true
      thread/BLAS_THREADING_SINGLE_THREADED false
      (throw (dragan-says-ex "Accelerate returned a threading option unknown to this library."
                             {:threading thriading
                              :max-threading-options thread/BLAS_THREADING_MAX_OPTIONS})))))

(defn threading! [multi-threading?]
  (thread/BLASSetThreading (if multi-threading
                             thread/BLAS_THREADING_MULTI_THREADED
                             thread/BLAS_THREADING_SINGLE_THREADED)))
