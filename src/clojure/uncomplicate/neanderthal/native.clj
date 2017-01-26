;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.native
  (:require [uncomplicate.neanderthal.core
             :refer [vctr ge tr]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-float cblas-double]]))

;; ============ Creating real constructs  ==============

(defn fv
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

  (fv (java.nio.ByteBuffer/allocateDirect 16))
  => #<RealBlockVector| float, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (fv 4)
  => #<RealBlockVector| float, n:4, stride:1>(0.0 0.0 0.0 0.0)<>

  (fv 12.4)
  => #<RealBlockVector| float, n:1, stride:1>(12.399999618530273)<>

  (fv (range 4))
  => #<RealBlockVector| float, n:4, stride:1>(0.0 1.0 2.0 3.0)<>

  (fv 1 2 3)
  => #<RealBlockVector| float, n:3, stride:1>(1.0 2.0 3.0)<>
  "
  ([source]
   (vctr cblas-float source))
  ([x & xs]
   (fv (cons x xs))))

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
   (vctr cblas-double source))
  ([x & xs]
   (dv (cons x xs))))

(defn dge
  "Creates a native-backed, dense, double mxn matrix from source and/or options.

  If called with two arguments, creates a zero matrix with dimensions mxn.

  Accepts the following sources:
  - a clojure sequence, which will be copied into a
  . direct ByteBuffer that backs the new vector.
  -

  (dge 2 3)
  => #<GeneralMatrix| double, COL, mxn: 2x3, ld:2>((0.0 0.0) (0.0 0.0) (0.0 0.0))<>

  (dge 3 2 (range 6))
  => #<GeneralMatrix| double, COL, mxn: 3x2, ld:3>((0.0 1.0 2.0) (3.0 4.0 5.0))<>

  (dge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<GeneralMatrix| double, COL, mxn: 3x2, ld:3>((0.0 0.0 0.0) (0.0 0.0 0.0))<>
  "
  ([^long m ^long n source options]
   (ge cblas-double m n source options))
  ([^long m ^long n arg]
   (ge cblas-double m n arg))
  ([^long m ^long n]
   (ge cblas-double m n))
  ([a]
   (ge cblas-double a)))

(defn fge
  "Creates a native-backed, dense, column-oriented float mxn matrix from source.

  If called with two arguments, creates a zero matrix with dimensions mxn.

  Accepts the following sources:
  - java.nio.ByteBuffer with a capacity = Float/BYTES * m * n,
  . which will be used as-is for backing the matrix.

  (fge 2 3)
  => #<GeneralMatrix| float, COL, mxn: 2x3, ld:2>((0.0 0.0) (0.0 0.0) (0.0 0.0))<>

  (fge 3 2 (range 6))
  => #<GeneralMatrix| float, COL, mxn: 3x2, ld:3>((0.0 1.0 2.0) (3.0 4.0 5.0))<>

  (fge 3 2 (java.nio.ByteBuffer/allocateDirect 48))
  => #<GeneralMatrix| float, COL, mxn: 3x2, ld:3>((0.0 0.0 0.0) (0.0 0.0 0.0))<>
  "
  ([^long m ^long n source options]
   (ge cblas-float m n source options))
  ([^long m ^long n arg]
   (ge cblas-float m n arg))
  ([^long m ^long n]
   (ge cblas-float m n))
  ([a]
   (ge cblas-float a)))

(defn dtr
  "TODO"
  ([^long n source options]
   (tr cblas-double n source options))
  ([^long n arg]
   (tr cblas-double n arg))
  ([arg]
   (tr cblas-double arg)))

(defn ftr
  "TODO"
  ([^long n source options]
   (tr cblas-float n source options))
  ([^long n arg]
   (tr cblas-float n arg))
  ([arg]
   (tr cblas-float arg)))
