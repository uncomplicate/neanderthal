;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.real
  "Contains type-specific primitive floating point functions. Typically,
  you would want to require this namespace if you need to compute
  real matrices containing doubles and/or floats.
  Aditionally, you need to require core namespace to use
  type-agnostic functions.
  You need to take care to only use vectors and matrices
  of the same type in the same function call. These functions do not support
  arguments of mixed real types. For example, you can not call the
  dot function with one double vector (dv) and one float vector (fv).

  ## Example
  (ns test
    (:require [uncomplicate.neanderthal
               [core :refer :all :exclude [entry entry! dot nrm2 asum sum]]
               [real :refer :all]]))
  "
  (:require [uncomplicate.commons.core :refer [with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [foldmap fold]]
            [uncomplicate.neanderthal
             [math :refer [sqr]]
             [core :as core]
             [linalg :as linalg]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api RealVector RealMatrix RealChangeable Vector]))

;; ============ Vector and Matrix access methods ===

(defn entry
  "Returns a primitive ^double i-th entry of vector x, or ij-th entry of matrix m.

  (entry (dv 1 2 3) 1) => 2.0
  "
  (^double [^RealVector x ^long i]
   (.entry x i))
  (^double [^RealMatrix m ^long i ^long j]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.entry m i j)
     (throw (IndexOutOfBoundsException. (format api/MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

(defn entry!
  "Sets the i-th entry of vector x, or ij-th entry of matrix m,
  or all entries if no index is provided, to the primitive ^double value val and
  returns the updated container.

  (entry! (dv 1 2 3) 2 -5)
  => #<RealBlockVector| double, n:3, stride:1>(1.0 2.0 -5.0)<>
  "
  ([^RealChangeable c ^double val]
   (.set c val))
  ([^RealChangeable v ^long i ^double val]
   (.set v i val))
  ([^RealMatrix m ^long i ^long j ^double val]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)) (.isAllowed ^RealChangeable m i j))
     (.set ^RealChangeable m i j val)
     (throw (IndexOutOfBoundsException. (format api/MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

(defn dot
  "Primitive wrapper for core dot function."
  ^double [x y]
  (double (core/dot x y)))

(defn nrm2
  "Primitive wrapper for core nrm2 function."
  ^double [x]
  (double (core/nrm2 x)))

(defn asum
  "Primitive wrapper for core asum function."
  ^double [x]
  (double (core/asum x)))

(defn sum
  "Primitive wrapper for core sum function."
  ^double [x]
  (double (core/sum x)))

(defn amax
  "TODO "
  ^double [x]
  (double (core/amax x)))

(defn ls-residual
  "TODO"
  [^RealMatrix a ^RealMatrix b]
  (let [res (api/raw (.row b 0))]
    (if (<= (.ncols a) (.mrows a))
      (let [residuals (.submatrix b (.ncols a) 0 (- (.mrows b) (.ncols a)) (.ncols b))]
        (dotimes [j (.ncols residuals)]
          (entry! res j (foldmap sqr (.col residuals j))))
        res)
      res)))

(def ^:private f* (double-fn *))

(defn det! ^double [^RealMatrix lu ^Vector ipiv]
  (with-release [res (double (fold f* 1.0 (.dia lu)))]
    (if (even? (.dim ipiv))
      res
      (- res))))
