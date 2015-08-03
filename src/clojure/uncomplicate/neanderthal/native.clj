(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.native
  (:require [uncomplicate.neanderthal.impl
             [buffer-block :refer [create-vector create-matrix
                                   double-accessor float-accessor]]
             [cblas :refer [dv-engine sv-engine dge-engine sge-engine]]]))

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
   (create-vector float-accessor sv-engine sge-engine source))
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
   (create-vector double-accessor dv-engine dge-engine source))
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
   (create-matrix double-accessor dv-engine dge-engine m n source))
  ([^long m ^long n]
   (create-matrix double-accessor dv-engine dge-engine m n)))

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
   (create-matrix float-accessor sv-engine sge-engine m n source))
  ([^long m ^long n]
   (create-matrix float-accessor sv-engine sge-engine m n)))
