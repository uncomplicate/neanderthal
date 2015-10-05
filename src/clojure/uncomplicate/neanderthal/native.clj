(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.native
  (:require [uncomplicate.neanderthal.core
             :refer [create create-vector create-ge-matrix]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]))

;; ============ Creating real constructs  ==============

(defn sv
  "Creates a native-backed float vector from source.

  Accepts the following source:
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
  => #<RealBlockVector| float, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (sv 4)
  => #<RealBlockVector| float, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (sv 12.4)
  => #<RealBlockVector| float, n:1, stride:1>(12.399999618530273)<>

  (sv (range 4))
  => #<RealBlockVector| float, n:4, stride:1>(0.0 1.0 2.0 3.0)<>

  (sv 1 2 3)
  => #<RealBlockVector| float, n:3, stride:1>(1.0 2.0 3.0)<>
  "
  ([source]
   (create-vector cblas-single source))
  ([x & xs]
   (sv (cons x xs))))

(defn dv
  "Creates a native-backed double vector from source.

  Accepts the following source:
  - java.nio.ByteBuffer with a capacity divisible by 4,
  . which will be used as-is for backing the vector.
  - a positive integer, which will be used as a dimension
  . of new vector with zeroes as entries.
  - a floating point number, which will be the only entry
  . in a one-dimensional vector.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.
  - varargs will be treated as a clojure sequence.

  (dv (java.nio.ByteBuffer/allocateDirect 16))
  => #<RealBlockVector| double, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (dv 4)
  => #<RealBlockVector| double, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (dv 12.4)
  => #<RealBlockVector| double, n:1, stride:1>(12.399999618530273)<>

  (dv (range 4))
  => #<RealBlockVector| double, n:4, stride:1>(0.0 1.0 2.0 3.0)<>

  (dv 1 2 3)
  => #<RealBlockVector| double, n:3, stride:1>(1.0 2.0 3.0)<>
  "
  ([source]
   (create-vector cblas-double source))
  ([x & xs]
   (dv (cons x xs))))

(defn dge
  "Creates a native-backed, dense, column-oriented double mxn matrix from source.

  If called with two arguments, creates a zero matrix with dimensions mxn.

  Accepts the following sources:
  - java.nio.ByteBuffer with a capacity = Double/BYTES * m * n,
  . which will be used as-is for backing the matrix.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.

  (dge 2 3)
  => #<GeneralMatrix| double, COL, mxn: 2x3, ld:2>((0.0 0.0) (0.0 0.0) (0.0 0.0))<>

  (dge 3 2 (range 6))
  => #<GeneralMatrix| double, COL, mxn: 3x2, ld:3>((0.0 1.0 2.0) (3.0 4.0 5.0))<>

  (dge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<GeneralMatrix| double, COL, mxn: 3x2, ld:3>((0.0 0.0 0.0) (0.0 0.0 0.0))<>
  "
  ([^long m ^long n source]
   (create-ge-matrix cblas-double m n source))
  ([^long m ^long n]
   (create cblas-double m n)))

(defn sge
  "Creates a native-backed, dense, column-oriented float mxn matrix from source.

  If called with two arguments, creates a zero matrix with dimensions mxn.

  Accepts the following sources:
  - java.nio.ByteBuffer with a capacity = Float/BYTES * m * n,
  . which will be used as-is for backing the matrix.
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.

  (sge 2 3)
  => #<GeneralMatrix| float, COL, mxn: 2x3, ld:2>((0.0 0.0) (0.0 0.0) (0.0 0.0))<>

  (sge 3 2 (range 6))
  => #<GeneralMatrix| float, COL, mxn: 3x2, ld:3>((0.0 1.0 2.0) (3.0 4.0 5.0))<>

  (sge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<GeneralMatrix| float, COL, mxn: 3x2, ld:3>((0.0 0.0 0.0) (0.0 0.0 0.0))<>
  "
  ([^long m ^long n source]
   (create-ge-matrix cblas-single m n source))
  ([^long m ^long n]
   (create cblas-single m n)))
