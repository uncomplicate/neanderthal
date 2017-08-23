;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.aux
  "Contains type-agnostic auxiliary functions roughly corresponding to the functionality
  usually defined in auxiliary LAPACK (sorting etc.), or useful functions that may not commonly be
  implemented by BLAS engines, but are helpful vectorized ruoutines. This namespace works similarly
  to the [[uncomplicate.neanderthal.core]] namespace; see there for more details about the intended use.

  ### Cheat Sheet

  - Sorting:  [[sort!]], [[sort+!]], [[sort-!]].
  "
  (:require [uncomplicate.neanderthal.internal
             [api :as api]
             [common :refer [dragan-says-ex]]])
  (:import uncomplicate.neanderthal.internal.api.IntegerVector))

(defn sort!
  "Sorts input vector or all matrix slices (columns or rows, according to layout).

  If `x` is a vector with stride different than 1, throws ExceptionInfo."
  [x increasing]
  (api/srt (api/engine x) x increasing))

(defn sort+!
  "Sorts input vector or all matrix slices (columns or rows, according to layout) in ascending order.

  If `x` is a vector with stride different than 1, throws ExceptionInfo."
  [x]
  (sort! x true))

(defn sort-!
  "Sorts input vector or all matrix slices (columns or rows, according to layout) in descending order.

  If `x` is a vector with stride different than 1, throws ExceptionInfo."
  [x]
  (sort! x false))

(defn laswp!
  "Performs a series of row interchanges on a general rectangular matrix.

  If `(dim x)` is smaller than k2 - k1, throws ExceptionInfo."
  ([a ^IntegerVector x ^long k1 ^long k2]
   (if (and (<= 1 k1 k2 (.dim x)) (<= (- k2 k1) (.dim x)))
     (api/laswp (api/engine a) a x k1 k2)
     (dragan-says-ex "There is not enough indices in ipiv. Check the ipiv and k1 and k2."
                     {:a (api/info a) :x (api/info x)})))
  ([a ^IntegerVector x]
   (laswp! a x 1 (.dim x))))
