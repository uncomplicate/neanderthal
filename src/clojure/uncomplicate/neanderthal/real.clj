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
  (:require [primitive-math]
            [vertigo
             [bytes :refer [direct-buffer]]
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [copy]]
             [math :refer [f= pow sqrt]]
             [cblas :refer [MAT_BOUNDS_MSG DEFAULT_ORDER]]])
  (:import [uncomplicate.neanderthal.cblas
            DoubleBlockVector DoubleGeneralMatrix]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Carrier
            RealVectorEditor RealMatrixEditor]
           [java.nio ByteBuffer]))

(primitive-math/use-primitive-operators)

;; ============ Creating double constructs  ==============

(defn seq-to-buffer
  "Copies the entries from sequence s to new direct
  java.nio.ByteBuffer.
  "
  [s]
  (.position ^ByteBuffer
             (reduce (fn [^ByteBuffer bb ^double e]
                       (.putDouble bb e))
                     (direct-buffer (* 8 (count s))) s)
             0))

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
     (cond
      (and (instance? ByteBuffer source)
           (zero? (long (mod (.capacity ^ByteBuffer source) 8))))
      (DoubleBlockVector. source
                          (/ (.capacity ^ByteBuffer source) 8)
                          1)
      (and (integer? source) (pos? source)) (dv (direct-buffer (* 8 (long source))))
      (float? source) (dv [source])
      (sequential? source) (dv (seq-to-buffer source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a double vector from %s ."
                               (type source))))))
  ([x & xs]
     (dv (cons x xs))))

(defn dge
  "Creates a native-backed, dense, column-oriented
  double mxn matrix from source.

  If called with two arguments, creates a zero matrix
  with dimensions mxn.

  Accepts following sources:
  - java.nio.ByteBuffer with a capacity = 8 * m * n,
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
     (cond
      (and (instance? ByteBuffer source)
           (zero? (long (mod (.capacity ^ByteBuffer source) 8))))
      (if (= (* 8 m n) (.capacity ^ByteBuffer source))
        (DoubleGeneralMatrix.
         source m n (max m 1) DEFAULT_ORDER)
        (throw (IllegalArgumentException.
                (format "Matrix dimensions (%dx%d) are not compatible with the buffer capacity."
                        m n))))
      (sequential? source) (dge m n (seq-to-buffer source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a double vector from %s ."
                               (type source))))))
  ([^long m ^long n]
     (dge m n (direct-buffer (* 8 m n)))))

;; ============ Vector and Matrix access methods ===
(defn entry ^double
  {:doc "Returns the i-th entry of vector x, or ij-th entry of matrix m."}
  ([^RealVector x ^long i]
   (.entry x i))
  ([^RealMatrix m ^long i ^long j]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.entry m i j)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

(defn set-entry!
  "Sets the i-th entry of vector x, or ij-th entry of matrix m,
  to value val and returns the vector or matrix."
  ([^RealVectorEditor v ^long i ^double val]
   (.setEntry v i val))
  ([^RealMatrixEditor m ^long i ^long j ^double val]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.setEntry m i j val)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

;; ================== Category functions  ==============
(defn fmap!
  ([f x]
   (p/fmap! x f))
  ([f x y]
   (p/fmap! x f y))
  ([f x y z]
   (p/fmap! x f y z))
  ([f x y z w]
   (p/fmap! x f y z w))
  ([f x y z w & ws]
   (apply p/fmap! x f y z w ws)))

(defn fold
  ([x]
   (p/fold x))
  ([f x]
   (p/fold x f))
  ([f id x]
   (p/fold x f id)))

(defn freduce
  ([f x]
   (p/freduce x f))
  ([f acc x]
   (p/freduce x acc f))
  ([f acc x y]
   (p/freduce x acc f y))
  ([f acc x y z]
   (p/freduce x acc f y z))
  ([f acc x y z & ws]
   (apply p/freduce x acc f y z ws)))

;; ================== BLAS 1 =======================

(defn dot ^double
  {:doc "BLAS 1: Dot product.
        Computes the dot product of vectors x and y.
        (dot (dv 1 2 3) (dv 1 2 3)) => 14.0"}
  [^RealVector x ^RealVector y]
  (if (= (.dim x) (.dim y))
    (.dot x y)
    (throw (IllegalArgumentException.
            (format "Incompatible dimensions - x:%d and y:%d."
                    (.dim x) (.dim y))))))

(defn nrm2 ^double
  {:doc "BLAS 1: Euclidean norm.
        Computes the Euclidan (L2) norm of vector x.
        (nrm2 (dv 1 2 3)) => 3.7416573867739413"}
  [^RealVector x]
  (.nrm2 x))

(defn asum ^double
  {:doc "BLAS 1: Sum absolute values.
         Sums absolute values of entries of vector x.
         (asum (dv -1 2 -3)) => 6.0"}
  [^RealVector x]
  (.asum x))

(defn rot!
  "BLAS 1: Apply plane rotation.
  "
  ([^RealVector x ^RealVector y
    ^double c ^double s]
   (cond
    (not= (.dim x) (.dim y))
    (throw (IllegalArgumentException.
            (format "Incompatible dimensions - x:%d and y:%d."
                    (.dim x) (.dim y))))
    (not (and (<= -1.0 c 1.0) (<= -1.0 c 1.0)
              (f= 1.0 (+ (pow c 2) (pow s 2)))))
    (throw (IllegalArgumentException.
            (format "s:%f and d:%f must be between -1 and 1, and sin2 + cos2 = 1."
                    s c)))
    :default (.rot x y c s)))
  ([x y c]
   (rot! x y c (sqrt (- 1.0 (pow c 2))))))

(defn scal!
  "BLAS 1: Scale vector.

  Computes x = ax
  where:
  - x is a vector,
  - a is a scalar.

  Multiplies vector x by scalar alpha, i.e scales the vector.
  After scal!, x will be changed.

  (scal! 1.5  (dv 1 2 3))
  => #<DoubleBlockVector| n:3, stride:1, (1.5 3.0 4.5)>
  "
  [^double alpha ^RealVector x]
  (.scal x alpha))

(defn axpy!
  "BLAS 1: Vector scale and add.

  Computes y = ax + y.
  where:
  x and y are vectors,
  a is a scalar.

  Multiplies vector x by scalar alpha and adds it to
  vector y. After axpy!, y will be changed.

  If called with 2 arguments, x and y, adds vector x
  to vector y.

  If called with more than 2 arguments, at least every
  other have to be a vector. A scalar multiplier may be
  included before each vector.

  If the dimensions of x and y are not compatible,
  throws IllegalArgumentException.

  (def x (dv 1 2 3))
  (def y (dv 2 3 4))
  (axpy! y 1.5 x)
  => #<DoubleBlockVector| n:3, stride:1, (3.5 6.0 8.5)>

  y => #<DoubleBlockVector| n:3, stride:1, (3.5 6.0 8.5)>

  (axpy! y x)
  => #<DoubleBlockVector| n:3, stride:1, (4.5 8.0 11.5)>

  (axpy! y x (dv 3 4 5) 2 (dv 1 2 3))
  => #<DoubleBlockVector| n:3, stride:1, (10.5 18.0 25.5)>
  "
  ([^RealVector y ^double alpha ^RealVector x]
     (if (= (.dim x) (.dim y))
       (.axpy x alpha y)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - x:%d and y:%d."
                       (.dim x) (.dim y))))))
  ([y x]
   (axpy! y 1.0 x))
  ([y x z & zs]
   (if (number? x)
     (loop [res (axpy! y x z) s zs]
       (if-let [f (first s)]
         (let [r (rest s)]
           (if (number? f)
             (recur (axpy! res f (first r)) (rest r))
             (recur (axpy! res 1.0 f) r)))
         res))
     (apply axpy! y 1.0 x z zs))))

(defn axpy
  "A pure variant of axpy! that does not change
  any of the arguments. The result is a new vector instance.
  "
  ([x y]
   (if (number? x)
     (axpy! (p/zero y) x y)
     (axpy! (copy y) 1.0 x)))
  ([x y & zs]
   (apply axpy! (if (number? x)
                  (p/zero y)
                  (p/zero x))
          x y zs)))

(defn ax
  "Multiplies vector x by a scalar a.
  Similar to scal!, but does not change x. The result
  is a new vector instance."
  [^double alpha x]
  (axpy! (p/zero x) alpha x))

(defn xpy
  "Sums vectors x, y, & zs. The result is a new vector instance."
  ([x y]
   (axpy! (copy y) 1.0 x))
  ([x y & zs]
   (loop [res (axpy! (p/zero y) 1.0 x) s zs]
     (if s
       (recur (axpy! res 1.0 (first s)) (next s))
       res))))

;;================= BLAS 2 ========================
(defn mv!
  "BLAS 2: Matrix-vector multiplication.

  Computes y = alpha a * x + beta * y
  where:
  a is a matrix,
  x and y are vectors,
  alpha and beta are scalars.

  Multiplies matrix a by scalar alpha and then multiplies
  the the resulting matrix by vector x. Adds the resulting
  vector to vector y previously scaled by scalar beta.
  Returns vector y, which contains the result and is changed by
  the operation.

  If alpha and beta are not provided, uses 1.0 as their values.

  If the dimensions of a, x and y are not compatible,
  throws IllegalArgumentException.

  (def a (dge 3 2 (range 6)))
  (def x (dv 1 2))
  (def y (dv 2 3 4))

  (mv! y 2.0 a x 1.5)
  => #<DoubleBlockVector| n:3, stride:1, (15.0 22.5 30.0)>
  "
  ([^RealVector y alpha ^RealMatrix a ^RealVector x beta]
     (if (and (= (.ncols a) (.dim x))
              (= (.mrows a) (.dim y)))
       (.mv a alpha x beta y)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - a:%dx%d, x:%d, y:%d."
                       (.mrows a) (.ncols a) (.dim x) (.dim y))))))
  ([y alpha a x]
     (mv! y alpha a x 0.0))
  ([y a x]
     (mv! y 1.0 a x 0.0)))

(defn mv
  "A pure version of mv! that returns the result
  in a new vector instance. Computes alpha a * x."
  ([^double alpha ^RealMatrix a ^Carrier x]
     (mv! (dv (.mrows a)) alpha a x 0.0))
  ([a x]
     (mv 1.0 a x)))

(defn rank!
  "BLAS 2: General rank-1 update.

  Computes a = a + alpha * x * y'
  where:
  a is a mxn matrix,
  x and y are vectors, y' is a transposed vector,
  a is a scalar.

  If called with 3 arguments, a, x, y, alpha is 1.0.

  (def a (dge 3 2 [1 1 1 1 1 1]))
  (rank! a 1.5 (dv 1 2 3) (dv 4 5))
  => #<DoubleGeneralMatrix| COL, mxn: 3x2, ld:3, ((7.0 13.0 19.0) (8.5 16.0 23.5))>
  "
  ([^RealMatrix a ^double alpha ^RealVector x ^RealVector y]
   (if (and (= (.mrows a) (.dim x))
            (= (.ncols a) (.dim y)))
     (.rank a alpha x y)
     (throw (IllegalArgumentException.
             (format "Incompatible dimensions - a:%dx%d, x:%d, y:%d."
                     (.mrows a) (.ncols a) (.dim x) (.dim y))))))
  ([a x y]
   (rank! a 1.0 x y)))

(defn rank
  "A pure version of rank! that returns the result
  in a new matrix instance.
  "
  ([^double alpha ^RealVector x ^RealVector y]
   (rank! (dge (.dim x) (.dim y)) alpha x y))
  ([x y]
   (rank 1.0 x y)))

;; ================ BLAS 3 ========================
(defn mm!
  "BLAS 3: Matrix-matrix multiplication.

  Computes c = alpha a * b + beta * c
  where:
  a, b and c are matrices,
  alpha and beta are scalars.

  Multiplies matrix a by scalar alpha and then multiplies
  the resulting matrix by matrix b. Adds the resulting
  matrix to matrix c previously scaled by scalar beta.
  Returns matrix c, which contains the result and is changed by
  the operation.

  Can be called without scalars, with three matrix arguments.

  If the dimensions of a, b and c are not compatible,
  throws IllegalArgumentException.

  (def a (dge 2 3 (range 6)))
  (def a (dge 3 2 (range 2 8)))
  (def c (dge 2 2 [1 1 1 1]))

  (mm! c 1.5 a b 2.5)
  => #<DoubleGeneralMatrix| COL, mxn: 2x2, ld:2, ((35.5 49.0) (62.5 89.5))>

  (def c (dge 2 2))
  (mm! c a b)
  => #<DoubleGeneralMatrix| COL, mxn: 2x2, ld:2, ((22.0 31.0) (40.0 58.0))>
  "
  ([^RealMatrix c alpha ^RealMatrix a ^RealMatrix b beta]
     (if (and (= (.ncols a) (.mrows b))
              (= (.mrows a) (.mrows c))
              (= (.ncols b) (.ncols c)))
       (.mm a alpha b beta c)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - a:%dx%d, b:%dx%d, c:%dx%d."
                       (.mrows c) (.ncols c)
                       (.mrows a) (.ncols a)
                       (.mrows b) (.ncols b))))))
  ([c a b]
   (mm! c 1.0 a b 1.0)))

(defn mm
  "A pure version of mm!, that returns the result
  in a new matrix instance.
  Computes alpha a * b"
  ([alpha ^RealMatrix a ^RealMatrix b]
   (mm! (dge (.mrows a) (.ncols b)) alpha a b 0.0))
  ([a b]
   (mm 1.0 a b)))

(primitive-math/unuse-primitive-operators)
