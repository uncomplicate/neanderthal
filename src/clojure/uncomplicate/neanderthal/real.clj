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
  (:require  [vertigo
              [bytes :refer [direct-buffer]]
              [core :refer [wrap marshal-seq]]
              [structs :refer [float64 float32 unwrap-byte-seq]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [mrows ncols rank! mm! INCOMPATIBLE_BLOCKS_MSG]]
             [block :refer [MAT_BOUNDS_MSG DEFAULT_ORDER ->RealBlockVector ->RealGeneralMatrix]]])
  (:import [uncomplicate.neanderthal.protocols
            Block RealVector RealMatrix BLAS RealChangeable]
           [java.nio ByteBuffer]
           [vertigo.bytes ByteSeq]))

;; ============ Creating double constructs  ==============

(defn to-buffer
  ([^long bytesize s]
   (.buf ^ByteSeq (unwrap-byte-seq
                   (marshal-seq (case bytesize
                                  8 float64
                                  4 float32)
                                s))))
  ([s]
   (to-buffer Double/BYTES s)))

(defn real-vector
  ([^long bytesize source]
   (cond
     (and (instance? ByteBuffer source)
          (zero? (long (mod (.capacity ^ByteBuffer source) bytesize))))
     (->RealBlockVector source (case bytesize 8 float64 4 float32) bytesize
                        (/ (.capacity ^ByteBuffer source) bytesize) 1)
     (and (integer? source) (<= 0 (long source)))
     (real-vector bytesize (direct-buffer (* bytesize (long source))))
     (float? source) (real-vector bytesize [source])
     (sequential? source) (real-vector bytesize (to-buffer bytesize source))
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a vector from %s."
                              (type source)))))))

(defn real-matrix
  ([^long bytesize ^long m ^long n source]
   (cond
     (and (instance? ByteBuffer source)
          (zero? (long (mod (.capacity ^ByteBuffer source) bytesize)))
          (= (* m n) (quot (.capacity ^ByteBuffer source) bytesize)))
     (if (= (* bytesize m n) (.capacity ^ByteBuffer source))
       (->RealGeneralMatrix source (case bytesize 8 float64 4 float32) bytesize
                            m n (max m 1) DEFAULT_ORDER)
       (throw (IllegalArgumentException.
               (format "Matrix dimensions (%dx%d) are not compatible with the buffer capacity."
                       m n))))
     (sequential? source) (real-matrix bytesize m n (to-buffer bytesize source))
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a double matrix from %s ."
                              (type source))))))
  ([^long bytesize ^long m ^long n]
   (real-matrix bytesize m n (direct-buffer (* bytesize m n)))))

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
   (real-vector Double/BYTES source))
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
   (real-matrix Double/BYTES m n source))
  ([^long m ^long n]
   (real-matrix Double/BYTES m n)))

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
   (real-matrix Float/BYTES m n source))
  ([^long m ^long n]
   (real-matrix Float/BYTES m n)))

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

;; ================== BLAS 1 =======================

(defn rank
  "A pure version of rank! that returns the result
  in a new matrix instance.
  "
  ([^double alpha ^RealVector x ^RealVector y]
   ;;   (rank! (real-matrix (p/byte-size x) (.dim x) (.dim y)) alpha x y)
   );;TODO
  ([x y]
   (rank 1.0 x y)))

(defn mm
  "A pure version of mm!, that returns the result
  in a new matrix instance.
  Computes alpha a * b"
  ([alpha ^RealMatrix a ^RealMatrix b]
   ;;(mm! (real-matrix (p/byte-size a) (.mrows a) (.ncols b)) alpha a b 0.0)
   );;TODO
  ([a b]
   (mm 1.0 a b)))
