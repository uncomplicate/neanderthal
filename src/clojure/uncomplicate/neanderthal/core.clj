(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.core
  "Contains type-agnostic linear algebraic functions. Typically,
  you would want to require this namespace regardless of the actual type
  (real, complex etc.) of the vectors and matrices that you use.
  Aditionally, you need to require a namespace specific to the
  type of primitive data you use (real, complex, etc.)

  You need to take care to only use vectors and matrices
  of the same type in the same function call. These functions do not support
  arguments of mixed real types. For example, you can not call the
  dot function with one double vector (dv) and one float vector (fv).
  ----- Example
  (ns test
    (:require [uncomplicate.neanderthal core real]))
  "
  (:require  [uncomplicate.neanderthal.protocols :as p])
  (:import [uncomplicate.neanderthal.protocols
            Vector Matrix Carrier]))

(def ^:private SEGMENT_MSG
  "Required segment %d-%d does not fit in a %d-dim vector.")

(def ^:private ROW_COL_MSG
  "Required %s %d is higher than the row count %d.")

(def ^:private DIMENSION_MSG
  "Different dimensions - required:%d, is:%d.")

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
  (^double [x] ;;TODO all categorical functions than can be primitive should be checked
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

;; ================= Carrier  ===================================
(defn zero
  "Returns an empty instance of the same type and dimension(s)
  as x, which can be vector, matrix or any other neanderthal object."
  [x]
  (p/zero x))

;; ================= Vector =======================
(defn vect?
  "Returns true if x implements uncomplicate.neanderthal.protocols.Vector.
  (vect? (dv 1 2 3)) => true
  (vect? [1 2 3]) => false
  "
  [x]
  (instance? Vector x))

(defn matrix?
  "Returns true if x implements uncomplicate.neanderthal.protocols.Matrix.
  (matrix? (dge 3 2 [1 2 3 4 5 6])) => true
  (matrix? [[1 2] [3 4] [5 6]]) => false
  "
  [x]
  (instance? Matrix x))

(defn dim
  "Returns the dimension of a neanderthal vector x.
  (dim (dv 1 2 3) => 3)
  "
  ^long [^Vector x]
  (.dim x))

(defn subvector
  "Returns a subvector starting witk k, l entries long,
  which is a part of a neanderthal vector x.

  The resulting subvector has a live connection to the
  vector data. Any change to the subvector data will affect
  the vector data.
  If you wish to disconnect the subvector from the parent
  vector, make a copy of the subvector (copy subx) prior
  to any changing operation.

  (subvector (dv 1 2 3 4 5 6) 2 3)
  => #<DoubleBlockVector| n:3, stride:1 (3.0 4.0 5.0)>
  "
  [^Vector x ^long k ^long l]
  (.subvector x k l))

(defn rotg!
  "BLAS 1: Generate plane rotation.

  # Description:

  Computes  the  elements  of  a Givens plane rotation matrix:
  .           __    __     __ __    __ __
  .           | c  s |     | a |    | r |
  .           |-s  c | *   | b | =  | 0 |
  .           --    --     -- --    -- --

  where  r = +- sqrt ( a**2 + b**2 )
  and    c**2 + s**2 = 1.

  a, b, c, s are the four entries in the vector x.

  This rotation can be used to selectively introduce zero elements
  into a matrix.

  # Arguments:

  x: contains a, b, c, and s arguments (see the description).
  .  Has to have at least these four entries. Other entries are ignored.

  # Result:

  x with the first four elements changed to be r, z, c, s.
  For a detailed description see
  http://www.mathkeisan.com/usersguide/man/drotg.html
  "
  [^Vector x];;TODO ensure that the stride is 1
  (if (< 3 (.dim x))
    (.rotg x)
    (throw (IllegalArgumentException.
            (format DIMENSION_MSG 4 (.dim x))))))

(defn rotmg!
  "BLAS 1: Generate modified plane rotation.
  "
  [^Vector p ^Vector args] ;;TODO ensue that the stride is 1
  (cond
   (< (.dim p) 5)
   (throw (IllegalArgumentException.
           (format DIMENSION_MSG 5 (.dim p))))
   (< (.dim args) 4)
   (throw (IllegalArgumentException.
           (format DIMENSION_MSG 4 (.dim args))))
   :default (.rotmg p args)))

(defn rotm!
  "BLAS 1: Apply modified plane rotation.
  "
  [^Vector x ^Vector y ^Vector p] ;;TODO ensue that the stride is 1
  (cond
   (< (.dim p) 5)
   (throw (IllegalArgumentException.
           (format DIMENSION_MSG 5 (.dim p))))
   (not= (.dim x) (.dim y))
   (throw (IllegalArgumentException.
           (format DIMENSION_MSG (.dim x) (.dim y))))
   :default (.rotm x y p)))

;; ================= Matrix =======================
(defn mrows
  "Returns the number of rows of the matrix m.
  (mrows (dge 3 2 [1 2 3 4 5 6])) => 3
  "
  ^long [^Matrix m]
  (.mrows m))

(defn ncols
  "Returns the number of columns of the matrix m.
  (mrows (dge 3 2 [1 2 3 4 5 6])) => 2
  "
  ^long [^Matrix m]
  (.ncols m))

(defn row
  "Returns the i-th row of the matrix m as a vector.

  The resulting vector has a live connection to the
  matrix data. Any change to the vector data will affect
  the matrix data.
  If you wish to disconnect the vector from the parent
  matrix, make a copy of the vector (copy x) prior
  to any changing operation.

  (row (dge 3 2 [1 2 3 4 5 6]) 1)
  => #<DoubleBlockVector| n:2, stride:3 (2.0 5.0)>
  "
  [^Matrix m ^long i]
  (if (< -1 i (.mrows m))
    (.row m i)
    (throw (IndexOutOfBoundsException.
            (format ROW_COL_MSG "row" i (.mrows m))))))

(defn col
  "Returns the i-th column of the matrix m as a vector.

  The resulting vector has a live connection to the
  matrix data. Any change to the vector data will affect
  the matrix data.
  If you wish to disconnect the vector from the parent
  matrix, make a copy of the vector (copy x) prior
  to any changing operation.

  (col (dge 3 2 [1 2 3 4 5 6]) 0)
  #<DoubleBlockVector| n:3, stride:1 (1.0 2.0 3.0)>
  "
  [^Matrix m ^long j]
  (if (< -1 j (.ncols m))
    (.col m j)
    (throw (IndexOutOfBoundsException.
            (format ROW_COL_MSG "col" j (.ncols m))))))

(defn cols
  "Returns a lazy sequence of vectors that represent
  columns of the matrix m.
  The vectors have a live connection to the matrix data.
  "
  [^Matrix m]
  (map #(.col m %) (range (.ncols m))))

(defn rows
  "Returns a lazy sequence of vectors that represent
  rows of the matrix m.
  The vectors have a live connection to the matrix data.
  "
  [^Matrix m]
  (map #(.row m %) (range (.mrows m))))

(defn submatrix
  "Returns a submatrix of m starting with row i, column j,
  that has k columns and l rows.

  The resulting submatrix has a live connection to the
  matrix m's data. Any change to the submatrix data will affect
  m's data.
  If you wish to disconnect the submatrix from the parent
  matrix, make a copy of the submatrix (copy subm) prior
  to any changing operation.

  (submatrix (dge 4 3 (range 12)) 1 1 2 1)
  => #<DoubleGeneralMatrix| COL, mxn: 2x1, ld:4 ((5.0 6.0))>
  "
  ([^Matrix a i j k l]
   (if (and (<= 0 (long i) (+ (long i) (long k)) (.mrows a))
            (<= 0 (long j) (+ (long j) (long l)) (.ncols a)))
     (.submatrix a i j k l)
     (throw (IndexOutOfBoundsException.
             (format "Submatrix %d,%d %d,%d is out of bounds of %dx%d."
                     i j k l (.mrows a) (.ncols a))))))
  ([^Matrix a k l]
   (submatrix a 0 0 k l)))

(defn trans
  "Transposes matrix m, i.e returns a matrix that has
  m's columns as rows.
  The resulting matrix has a live connection to m's data.

  (trans (dge 3 2 [1 2 3 4 5 6]))
  => #<DoubleGeneralMatrix| ROW, mxn: 2x3, ld:3 ((1.0 2.0 3.0) (4.0 5.0 6.0))>
  "
  [^Matrix m]
  (.transpose m))

;;================== BLAS 1 =======================
(defn iamax
  "BLAS 1: Index of maximum absolute value.
  The index of a first entry that has the maximum value.
  (iamax (dv 1 3 2)) => 1
  "
  ^long [^Vector x]
  (.iamax x))

(defn swp!
  "BLAS 1: Swap vectors.
  Swaps the entries of vectors x and y. x and y must have
  equal dimensions. Both x and y will be changed.
  Also works on matrices.

  (def x (dv 1 2 3))
  (def y (dv 3 4 5))
  (swp! x y)
  => #<DoubleBlockVector| n:3, stride:1 (3.0 4.0 5.0)>
  y
  => #<DoubleBlockVector| n:3, stride:1 (1.0 2.0 3.0)>
  "
  [^VectorBlock x ^VectorBlock y]
  (if (.compatible x y)
    (.swap x y)
    (throw (IllegalArgumentException.
            (format INCOMPATIBLE_VECTOR_BLOCKS_MSG
                    (.dim x) (.stride x) (.dim y) (.stride y))))))

(defn copy!
  "BLAS 1: Copy vector.
  Copies the entries of x into y. x and y must have
  equal dimensions. y will be changed. Also works on
  matrices.

  (def x (dv 1 2 3))
  (def y (dv 3 4 5))
  (copy! x y)
  => #<DoubleBlockVector| n:3, stride:1 (1.0 2.0 3.0)>
  y
  => #<DoubleBlockVector| n:3, stride:1 (1.0 2.0 3.0)>
  "
  [x y]
  (p/copy x y))

(defn copy
  "Returns a new vector and copies the entries from x.
  Changes to the resulting vectors do not affect x.

  (copy (dv 1 2 3))
  => #<DoubleBlockVector| n:3, stride:1 (1.0 2.0 3.0)>
  "
  [^Carrier x]
  (copy! x (p/zero x)))
