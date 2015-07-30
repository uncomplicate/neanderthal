(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.real
  "Contains type-specific linear algebraic functions. Typically,
  you would want to require this namespace if you need to compute
  real matrices containing doubles and/or floats.
  Aditionally, you need to require core namespace to use
  type-agnostic functions.
  You need to take care to only use vectors and matrices
  of the same type in the same function call. These functions do not support
  arguments of mixed real types. For example, you can not call the
  dot function with one double vector (dv) and one float vector (fv).

  ----- Example
  (ns test
    (:require [uncomplicate.neanderthal core real]))
  "
  (:require [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [mrows ncols rank! mm! INCOMPATIBLE_BLOCKS_MSG]]
             [block :refer [MAT_BOUNDS_MSG DEFAULT_ORDER
                            real-vector real-matrix]]
             [cblas :refer [dv-engine sv-engine dge-engine sge-engine]]])
  (:import [uncomplicate.neanderthal.protocols
            Block RealVector RealMatrix RealChangeable]
           [java.nio ByteBuffer]))

;; ============ Creating real constructs  ==============



(defn sv
  "Creates a native-backed float vector from source.

  Accepts following source:
  - java.nio.ByteBuffer with a capacity divisible by 4,
  . which will be used as-is for backing the vector.
  - a positive integer, which will be used as a dimension
  . of new vector with zeroes as entries.
  - a floating point number, which will be the only entry
  . in a one-dimensional vector.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.
  - varargs will be treated as a clojure sequence.

  (sv (java.nio.ByteBuffer/allocateDirect 16))
  => #<FloatBlockVector| n:4, stride:1 (0.0 0.0 0.0 0.0)>

  (sv 4)
  => #<FloatBlockVector| n:4, stride:1 (0.0 0.0 0.0 0.0)>

  (sv 12.4)
  => #<FloatBlockVector| n:1, stride:1 (12.4)>

  (sv (range 4))
  => #<FloatBlockVector| n:4, stride:1 (0.0 1.0 2.0 3.0)>

  (sv 1 2 3)
  => #<FloatBlockVector| n:3, stride:1, (1.0 2.0 3.0)>
  "
  ([source]
   (real-vector Float/BYTES source))
  ([x & xs]
   (sv (cons x xs))))

(defn dv
  "Creates a native-backed double vector from source.

  Accepts following source:
  - java.nio.ByteBuffer with a capacity divisible by 8,
  . which will be used as-is for backing the vector.
  - a positive integer, which will be used as a dimension
  . of new vector with zeroes as entries.
  - a floating point number, which will be the only entry
  . in a one-dimensional vector.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.
  - varargs will be treated as a clojure sequence.

  (dv (java.nio.ByteBuffer/allocateDirect 16))
  => #<DoubleBlockVector| n:2, stride:1 (0.0 0.0)>

  (dv 4)
  => #<DoubleBlockVector| n:4, stride:1 (0.0 0.0 0.0 0.0)>

  (dv 12.4)
  => #<DoubleBlockVector| n:1, stride:1 (12.4)>

  (dv (range 4))
  => #<DoubleBlockVector| n:4, stride:1 (0.0 1.0 2.0 3.0)>

  (dv 1 2 3)
  => #<DoubleBlockVector| n:3, stride:1, (1.0 2.0 3.0)>
  "
  ([source]
   (real-vector Double/BYTES source dv-engine dge-engine))
  ([x & xs]
   (dv (cons x xs))))

(defn dge
  "Creates a native-backed, dense, column-oriented
  double mxn matrix from source.

  If called with two arguments, creates a zero matrix
  with dimensions mxn.

  Accepts following sources:
  - java.nio.ByteBuffer with a capacity = Double/BYTES * m * n,
  . which will be used as-is for backing the matrix.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.

  (dge 2 3)
  => #<DoubleGeneralMatrix| COL, mxn: 2x3, ld:2, ((0.0 0.0) (0.0 0.0) (0.0 0.0))>

  (dge 3 2 (range 6))
  => #<DoubleGeneralMatrix| COL, mxn: 3x2, ld:3 ((0.0 1.0 2.0) (3.0 4.0 5.0))>

  (dge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<DoubleGeneralMatrix| COL, mxn: 3x2, ld:3 ((0.0 0.0 0.0) (0.0 0.0 0.0))>
  "
  ([^long m ^long n source]
   (real-matrix Double/BYTES m n source dv-engine dge-engine))
  ([^long m ^long n]
   (real-matrix Double/BYTES m n dv-engine dge-engine)))

(defn sge
  "Creates a native-backed, dense, column-oriented
  float mxn matrix from source.

  If called with two arguments, creates a zero matrix
  with dimensions mxn.

  Accepts following sources:
  - java.nio.ByteBuffer with a capacity = Float/BYTES * m * n,
  . which will be used as-is for backing the matrix.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.

  (sge 2 3)
  => #<DoubleGeneralMatrix| COL, mxn: 2x3, ld:2, ((0.0 0.0) (0.0 0.0) (0.0 0.0))>

  (sge 3 2 (range 6))
  => #<DoubleGeneralMatrix| COL, mxn: 3x2, ld:3 ((0.0 1.0 2.0) (3.0 4.0 5.0))>

  (sge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<DoubleGeneralMatrix| COL, mxn: 3x2, ld:3 ((0.0 0.0 0.0) (0.0 0.0 0.0))>
  "
  ([^long m ^long n source]
   (real-matrix Float/BYTES m n source sv-engine sge-engine))
  ([^long m ^long n]
   (real-matrix Float/BYTES m n sv-engine sge-engine)))

;; ============ Vector and Matrix access methods ===

(defn entry
  {:doc "Returns the i-th entry of vector x, or ij-th entry of matrix m."}
  (^double [^RealVector x ^long i]
   (.entry x i))
  (^double [^RealMatrix m ^long i ^long j]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.entry m i j)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

(defn entry!
  "Sets the i-th entry of vector x, or ij-th entry of matrix m,
  or all entries if no index is provided,
  to value val and returns the vector or matrix."
  ([^RealChangeable c ^double val]
   (.set c val))
  ([^RealChangeable v ^long i ^double val]
   (.set v i val))
  ([^RealChangeable m ^long i ^long j ^double val]
   (if (and (< -1 i (mrows m)) (< -1 j (ncols m)))
     (.set m i j val)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (mrows m) (ncols m)))))))

(defn alter! []
  (throw (UnsupportedOperationException.))) ;;TODO
