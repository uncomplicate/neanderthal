;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.real
  "Contains type-specific primitive floating point functions, equivalents of functions from the
  [[uncomplicate.neanderthal.core]] namespace. Typically, you would want to require this namespace
  if you need to compute real matrices containing doubles and/or floats.

  ### Example

      (ns test
        (:require [uncomplicate.neanderthal
                  [core :refer :all :exclude [entry entry! dot nrm2 asum sum]]
                  [real :refer :all]]))
  "
  (:require [uncomplicate.commons.core :refer [with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [foldmap]]
            [uncomplicate.neanderthal
             [math :refer [sqr]]
             [core :as core]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api RealVector RealMatrix RealChangeable Vector]))

;; ============ Vector and Matrix access methods ===

(defn entry
  "The primitive, much faster, version of [[uncomplicate.neanderthal.core/entry]]."
  (^double [^RealVector x ^long i]
   (try
     (.entry x i)
     (catch IndexOutOfBoundsException e
       (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim (.dim x)})))))
  (^double [^RealMatrix a ^long i ^long j]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)))
     (.entry a i j)
     (throw (ex-info "Requested element is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn entry!
  "The primitive, much faster, version of [[uncomplicate.neanderthal.core/entry!]]."
  ([^RealChangeable x ^double val]
   (.set x val))
  ([^RealChangeable x ^long i ^double val]
   (try
     (.set x i val)
     (catch IndexOutOfBoundsException e
       (throw (ex-info "The element you're trying to set is out of bounds of the vector."
                       {:i i :dim (core/dim x)})))))
  ([^RealMatrix a ^long i ^long j ^double val]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)) (.isAllowed ^RealChangeable a i j))
     (.set ^RealChangeable a i j val)
     (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn dot
  "Primitive wrapper of [[uncomplicate.neanderthal.core/dot]]."
  ^double [x y]
  (core/dot x y))

(defn nrm2
  "Primitive wrapper of [[uncomplicate.neanderthal.core/nrm2]]."
  ^double [x]
  (core/nrm2 x))

(defn asum
  "Primitive wrapper of [[uncomplicate.neanderthal.core/asum]]."
  ^double [x]
  (core/asum x))

(defn sum
  "Primitive wrapper of [[uncomplicate.neanderthal.core/sum]]."
  ^double [x]
  (core/sum x))

(defn amax
  "Primitive wrapper of [[uncomplicate.neanderthal.core/amax]]."
  ^double [x]
  (core/amax x))

(defn ls-residual
  "Computes the residual sum of squares for the solution of a linear system returned by
  [[uncomplicate.neanderthal.linalg/ls!]] (Linear Least Squares (LLS) problem)."
  [^RealMatrix a ^RealMatrix b]
  (if (<= (.ncols a) (.mrows a))
    (let [res (api/raw (.row b 0))
          residuals (.submatrix b (.ncols a) 0 (- (.mrows b) (.ncols a)) (.ncols b))]
      (dotimes [j (.ncols residuals)]
        (entry! res j (foldmap sqr (.col residuals j))))
      res)
    (api/zero (.row b 0))))
