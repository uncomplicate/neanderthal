;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.auxil
  "Contains type-agnostic auxiliary functions roughly corresponding to the functionality
  usually defined in auxiliary LAPACK (sorting etc.), or useful functions that may not commonly be
  implemented by BLAS engines, but are helpful vectorized ruoutines. This namespace works similarly
  to the [[uncomplicate.neanderthal.core]] namespace; see there for more details about the intended use.

  ### Cheat Sheet

  - Sorting:  [[sort!]], [[sort+!]], [[sort-!]].
  - Interchanges: [[swap-rows!]], [[swap-rows]], [[swap-cols!]], [[swap-cols]].
  - Permutations: [[permute-rows!]], [[permute-rows]], [[permute-cols!]], [[permute-cols]].
  "
  (:require [uncomplicate.commons
             [core :refer [let-release info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.core :refer [copy]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api IntegerVector Matrix]))

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

(defn swap-rows!
  "Performs a series of destructive row interchanges on a general rectangular matrix.

  If `(dim x)` is smaller than k2 - k1, throws ExceptionInfo."
  ([a ^IntegerVector ipiv ^long k1 ^long k2]
   (if (and (<= 1 k1 k2 (.dim ipiv)) (<= (- k2 k1) (.dim ipiv)))
     (api/laswp (api/engine a) a ipiv k1 k2)
     (dragan-says-ex "There is not enough indices in ipiv. Check the ipiv and k1 and k2."
                     {:a (info a) :ipiv (info ipiv)})))
  ([a ^IntegerVector ipiv]
   (swap-rows! a ipiv 1 (.dim ipiv))))

(defn swap-rows
  "Performs a series of destructive row interchanges on a general rectangular matrix.

  If `(dim x)` is smaller than k2 - k1, throws ExceptionInfo."
  ([a ipiv ^long k1 ^long k2]
   (let-release [a-copy (copy a)]
     (swap-rows! a-copy ipiv k1 k2)))
  ([a ^IntegerVector ipiv]
   (swap-rows a ipiv 1 (.dim ipiv))))

(defn swap-cols!
  "Performs a series of destructive column interchanges on a general rectangular matrix.

  If `(dim x)` is smaller than k2 - k1, throws ExceptionInfo."
  ([^Matrix a ^IntegerVector jpiv ^long k1 ^long k2]
   (.transpose ^Matrix (swap-rows! (.transpose a) jpiv k1 k2)))
  ([a ^IntegerVector jpiv]
   (swap-cols! a jpiv 1 (.dim jpiv))))

(defn swap-cols
  "Performs a series of column interchanges on a general rectangular matrix.

  If `(dim x)` is smaller than k2 - k1, throws ExceptionInfo."
  ([a ipiv ^long k1 ^long k2]
   (let-release [a-copy (copy a)]
     (swap-cols! a-copy ipiv k1 k2)))
  ([a ^IntegerVector ipiv]
   (swap-cols a ipiv 1 (.dim ipiv))))

(defn permute-rows!
  "Destructively rearranges rows of a given matrix as specified by a permutation vector `ipiv`
  forward or backward."
  ([^Matrix a ^IntegerVector ipiv ^Boolean forward]
   (if (and (= (.mrows a) (.dim ipiv)))
     (api/lapmr (api/engine a) a ipiv forward)
     (dragan-says-ex "Ipiv dimension must be mrows of a."
                     {:a (info a) :ipiv (info ipiv)})))
  ([a ipiv]
   (permute-rows! a ipiv true)))

(defn permute-rows
  "Rearranges rows of a given matrix as specified by a permutation vector `ipiv`
  forward or backward."
  ([a ipiv forward]
   (let-release [a-copy (copy a)]
     (permute-rows! a-copy ipiv forward)))
  ([a ^IntegerVector ipiv]
   (permute-rows a ipiv true)))

(defn permute-cols!
  "Destructively rearranges columns of a given matrix as specified by a permutation vector `jpiv`
  forward or backward."
  ([^Matrix a ^IntegerVector jpiv ^Boolean forward]
   (if (and (= (.ncols a) (.dim jpiv)))
     (api/lapmt (api/engine a) a jpiv forward)
     (dragan-says-ex "Jpiv dimension must be ncols of a."
                     {:a (info a) :jpiv (info jpiv)})))
  ([a jpiv]
   (permute-cols! a jpiv true)))

(defn permute-cols
  "Rearranges columns of a given matrix as specified by a permutation vector `jpiv`
  forward or backward."
  ([a jpiv forward]
   (let-release [a-copy (copy a)]
     (permute-cols! a-copy jpiv forward)))
  ([a ^IntegerVector jpiv]
   (permute-cols a jpiv true)))
