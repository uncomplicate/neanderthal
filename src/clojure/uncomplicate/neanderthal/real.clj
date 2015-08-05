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
               [core :refer :all :exclude [entry entry!]]
               [real :refer :all]]))
  "
  (:require [uncomplicate.neanderthal
             [core :refer [mrows ncols MAT_BOUNDS_MSG]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix RealChangeable]))

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
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

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
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.set ^RealChangeable m i j val)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))
