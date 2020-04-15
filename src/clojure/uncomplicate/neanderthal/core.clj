;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.core
  "Contains core type-agnostic linear algebraic functions roughly corresponding to functionality
  defined in BLAS 123, and functions to create and work with various kinds of vectors and matrices.
  Typically, you would want to require this namespace regardless of the actual type
  (real, complex, CPU, GPU, pure Java etc.) of the vectors and matrices that you use.

  In cases when you need to repeatedly call a function from this namespace that accesses
  individual entries, and the entries are primitive, it is better to use a primitive version
  of the function from [[uncomplicate.neanderthal.real]] namespace. Constructor functions
  for different specialized types (native, GPU, pure java) are in respective specialized namespaces
  ([[uncomplicate.neanderthal.native]], [[uncomplicate.neanderthal.opencl]], etc).

  Please take care to only use vectors and matrices of the same type in one call of a
  linear algebra operation. Compute operations typically (and on purpose!) do not support arguments
  of mixed types. For example, you can not call the [[dot]] function with one double vector (dv) and
  one float vector (fv), or for one vector in the CPU memory and one in the GPU memory.
  If you try to do bthat, an `ex-info` is thrown. You can use those different types side-by-side
  and transfer data between them though.

  ### How to use

      (ns test
        (:require [uncomplicate.neanderthal core native]))

      (ns test
        (:require [uncomplicate.neanderthal core native opencl]))

  ### Examples

  The best and most accurate examples can be found in the
  [comprehensive test suite](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal):
  see [here](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/real_test.clj),
  [here](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/block_test.clj),
  and [here](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/opencl_test.clj).
  Also, there are tutorial test examples [here](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples),
  the tutorials at [the Neanderthal web site](http://neanderthal.uncomplicate.org),
  and [my blog dragan.rocks](http://dragan.rocks).

  ### Cheat Sheet

  [Naming conventions for BLAS routines](https://software.intel.com/en-us/node/468382).

  * Create: [[vctr]], [[view]], [[view-vctr]], [[ge]], [[view-ge]], [[tr]], [[view-tr]], [[sy]], [[view-sy]],
  [[gb]], [[tb]], [[sb]], [[tp]], [[sp]], [[gd]], [[gt]], [[dt]], [[st]], [[raw]], [[zero]].

  * Move data around: [[transfer!]], [[transfer]], [[native]], [[copy!]], [[copy]], [[swp!]].

  * Clean up!: `with-release`, `let-release`, and `release` from the `uncomplicate.commons.core` namespace.

  * Vector: [[vctr?]], [[dim]], [[subvector]], [[entry]], [[entry!]], [[alter!]].

  * Meta:  [[vspace?]], [[matrix?]], [[symmetric?]], [[triangular?]], [[matrix-type?]], [[compatible?]]

  * Matrix: [[matrix?]], [[ge]], [[view-ge]], [[tr]], [[view-tr]], [[sy]], [[view-sy]],
  [[gb]], [[tb]], [[sb]], [[tp]], [[sp]], [[gd]], [[gt]], [[dt]], [[st]], [[mrows]], [[ncols]],
  [[row]], [[col]], [[dia]], [[cols]], [[rows]], [[submatrix]], [[trans]], [[trans!]], [[entry]],
  [[entry!]], [[alter!]], [[dim]], [[dia]], [[dias]], [[subband]].

  * Change: [[trans!]], [[entry!]], [[alter!]].

  * Help: [[info]]

  * [Monadic functions](http://fluokitten.uncomplicate.org): `fmap!`, `fmap`, `fold`, `foldmap`,
  `pure`, `op`, `id`, from the `uncomplicate.fluokitten.core` namespace.

  * [Compute level 1](https://software.intel.com/en-us/node/468390): [[dot]], [[nrm1]], [[nrm2]], [[nrmi]], [[asum]],
  [[iamax]], [[iamin]], [[amax]], [[iamin]], [[imax]], [[imin]], [[swp!]], [[copy!]], [[copy]], [[scal!]],
  [[scal]], [[rot!]], [[rotg!]], [[rotm!]], [[rotmg!]], [[axpy!]], [[axpy]], [[ax]], [[xpy]],
  [[axpby!]], [[sum]].

  * [Compute level 2](https://software.intel.com/en-us/node/468426): [[mv!]], [[mv]], [[rk!]], [[rk]].

  * [Compute level 3](https://software.intel.com/en-us/node/468478): [[mm!]], [[mm]], [[mmt]].

  "
  (:require [uncomplicate.commons
             [core :refer [release let-release with-release info]]
             [utils :refer [cond-into dragan-says-ex]]]
            [uncomplicate.neanderthal.math :refer [f= pow sqrt]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api VectorSpace Vector Matrix Changeable GEMatrix
            MatrixImplementation Region]))

(defmulti transfer!
  "Transfers the data from source to destination regardless of the structure type or memory context.

  Typically you would use it when you want to move data between the host memory and external device
  memory, or between incompatible data types. If you want to simply move data from one object
  to another in the same memory context, you should prefer [[copy!]].
  If possible, the data will simply be copied, but with multimethod call overhead.

      (transfer! (fv 1 2 3) device-vctr)
      (transfer! device-vctr (fv 3))
  "
  (fn ([source destination] [(class source) (class destination)])))

(defmethod transfer! [nil Object]
  [source destination]
  (dragan-says-ex "You cannot transfer data from nil." {:source source :destination (info destination)}))

(defmethod transfer! [Object nil]
  [source destination]
  (dragan-says-ex "You cannot transfer data to nil." {:source (info source) :destination destination}))

(defmethod transfer! [Object Object]
  [source destination]
  (dragan-says-ex "You cannot transfer data between these types of objects."
                  {:source (info source) :destination (info destination)}))

(defn transfer
  "Transfers the data to the memory context defined by `factory` (native, OpenCL, CUDA, etc.).

  If `factory` is not provided, moves the data to the main host memory. If `x` is already in the
  main memory, makes a fresh copy.

      (transfer (fv [1 2 3]) opencl-factory)
      (transfer (sge 2 3 (range 6)) opencl-factory)

      (transfer (fv [1 2 3]))
  "
  ([factory x]
   (let-release [res (api/raw x (api/factory factory))]
     (transfer! x res)
     res))
  ([x]
   (api/host x)))

(defn native
  "Ensures that `x` is in the native main memory, and if not, transfers it there.

      (let [v (fv [1 2 3])]
        (identical? (native v) v)) => true
  "
  [x]
  (api/native x))

(defn native!
  "Ensures that `x` is in the native main memory, and if not, transfers it there and releases `x`.
  "
  [x]
  (let-release [native-x (api/native x)]
    (when-not (identical? native-x x)
      (release x))
    native-x))

(defn view
  "Attach a default dense structure to the raw data of `x`. `x` can be anything that implements
  Viewable, such as DirectByteBuffer.

  Changes to the resulting object affect the source `x`, even the parts of data that might not
  be accessible by `x`. Use with caution!

  view always creates a new instance that reuses the master's data,
  but releasing a view never releases the master data.

      (view (buffer (vctr float-factory 1 2 3)))
  "
  ([x]
   (api/view x)))

(defn vctr
  "Creates a dense vector in the context of `factory`, from the provided `source`.

  If `source` is an integer, creates a vector of zeroes. If `source` is a number, puts it
  in the resulting vector. Otherwise, transfers the data from `source` (a sequence, vector, etc.)
  to the resulting vector.

    If the provided source do not make sense, throws ExceptionInfo.

      (vctr double-factory 3)
      (vctr float-factory 1 2 3)
      (vctr opencl-factory [1 2 3])
      (vctr opencl-factory (vctr float-factory [1 2 3]))
  "
  ([factory source]
   (cond
     (integer? source) (if (<= 0 (long source))
                         (api/create-vector (api/factory factory) source true)
                         (dragan-says-ex  "Vector cannot have a negative dimension." {:source (info source)}))
     (number? source) (.setBoxed ^Changeable (vctr factory 1) 0 source)
     :default (transfer factory source)))
  ([factory x & xs]
   (vctr factory (cons x xs))))

(defn view-vctr
  "Attach a dense vector to the raw data of `x`, with optional stride multiplier `stride-mult`.

  Changes to the resulting object affect the source `x`, even the parts of data that might not
  be accessible by `x`. Use with caution!

  view-vctr might return the master object itself, or create a new instance that reuses the master's data.

      (view-vctr (ge float-factory 2 3 (range 6)))
  "
  ([a]
   (api/view-vctr a))
  ([a ^long stride-mult]
   (if (< 0 stride-mult)
     (api/view-vctr a stride-mult)
     (dragan-says-ex "Cannot use a negative stride multiplier." {:stirde-mult stride-mult}))))

(defn vctr?
  "Tests if x is a (neanderthal) vector."
  [x]
  (instance? Vector x))

(defn vspace?
  "Tests if x is a vector space."
  [x]
  (instance? VectorSpace x))

(defn matrix?
  "Tests if x is a matrix of any kind."
  [x]
  (instance? Matrix x))

(defn symmetric?
  "Tests if x is a symmetric matrix of any kind."
  [^MatrixImplementation a]
  (.isSymmetric a))

(defn triangular?
  "Tests if x is a triangular matrix of any kind."
  [^MatrixImplementation a]
  (.isTriangular a))

(defn matrix-type
  "Returns the type of the matrix implementation represented by keyword (:ge, :tr, :sy, etc.)."
  [^MatrixImplementation a]
  (.matrixType a))

(defn compatible?
  "Checks whether x and y are compatible in the sense that they use compatible
  data formats (float, double, half) and computation contexts (native host memory, CUDA contexts etc.)."
  [x y]
  (api/compatible? x y))

(defn ge
  "Creates a dense matrix (GE) in the context of `factory`, with `m` rows and `n` columns.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimensions `m` and `n` are not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (ge float-factory 2 3)
      (ge opencl-factory 2 3 (range 6))
      (ge opencl-factory (ge double-factory 2 3 (range 6)))
      (ge float-factory [[1 2 3] [4 5] [] [6 7 8 9]])
  "
  ([factory m n source options]
   (if (and (<= 0 (long m)) (<= 0 (long n)))
     (let-release [res (api/create-ge (api/factory factory) m n (api/options-column? options)
                                      (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "GE matrix cannot have a negative dimension." {:m m :n n})))
  ([factory ^long m ^long n arg]
   (if (or (not arg) (map? arg))
     (ge factory m n nil arg)
     (ge factory m n arg nil)))
  ([factory ^long m ^long n]
   (ge factory m n nil nil))
  ([factory a]
   (transfer (api/factory factory) a)))

(defn view-ge
  "Attach a GE matrix to the raw data of `a`, with optional dimensions and/or stride multiplier.

  Changes to the resulting object affect the source `a`, even the parts of data that might not
  be accessible by a. Use with caution!

  view-ge might return the master object itself, or create a new instance that reuses the master's data.

      (view-ge (tr float-factory 3 (range 6)))
  "
  ([a]
   (api/view-ge a))
  ([a ^long stride-mult]
   (if (< 0 stride-mult)
     (api/view-ge a stride-mult)
     (dragan-says-ex "You cannot use a negative stride multiplier." {:stirde-mult stride-mult})))
  ([^VectorSpace a ^long m ^long n]
   (if (and (<= 0 m) (<= 0 n) (<= (* m n) (.dim a)))
     (api/view-ge a m n)
     (dragan-says-ex "You cannot generate a GE view with incompatible or ill-fitting dimension."
                     {:m m :n n :dim (.dim a) :errors
                      (cond-into []
                                 (not (< (* m n) (.dim a))) "source is smaller than destination"
                                 (not (< 0 m)) "m is negative"
                                 (not (< 0 n)) "n is negative")}))))

(defn tr
  "Creates a dense triangular matrix (TR) in the context of `factory`, with `n` rows and `n` columns.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimension `n` is not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  `:uplo` (`:upper` or `:lower`), and `:diag` (`:unit` or `:non-unit`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (tr float-factory 2)
      (tr opencl-factory 3 (range 6))
      (tr opencl-factory (ge double-factory 2 3 (range 6)))
  "
  ([factory ^long n source options]
   (if (<= 0 n)
     (let-release [res (api/create-tr (api/factory factory) n (api/options-column? options)
                                      (api/options-lower? options) (api/options-diag-unit? options)
                                      (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "TR matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (tr factory n nil arg)
     (tr factory n arg nil)))
  ([factory source]
   (if (number? source)
     (tr factory source nil nil)
     (tr factory (min (.mrows ^Matrix source) (.ncols ^Matrix source)) source nil))))

(defn view-tr
  "Attach a triangular matrix to the raw data of `a`.

  Changes to the resulting object affect the source `a`, even the parts of data that might not be
  accessible by a. Use with caution!

  Options: `:uplo` (`upper` or `:lower`), and `:diag` (`:unit` or `:non-unit`).

  view-tr creates a new instance that reuses the master's data.

      (view-tr (ge float-factory 3 2 (range 6)))
  "
  ([a]
   (api/view-tr a true false))
  ([^Matrix a options]
   (api/view-tr a (api/options-lower? options) (api/options-diag-unit? options))))

(defn sy
  "Creates a dense symmetric matrix (SY) in the context of `factory`, with `n` rows and `n` columns.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimension `n` is not mandatory

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  and `:uplo` (`:upper` or `:lower`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (sy float-factory 2)
      (sy opencl-factory 3 (range 6))
      (sy opencl-factory (ge double-factory 2 3 (range 6)))
  "
  ([factory ^long n source options]
   (if (<= 0 n)
     (let-release [res (api/create-sy (api/factory factory) n (api/options-column? options)
                                      (api/options-lower? options) (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "SY matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (sy factory n nil arg)
     (sy factory n arg nil)))
  ([factory source]
   (if (number? source)
     (sy factory source nil nil)
     (sy factory (min (.mrows ^Matrix source) (.ncols ^Matrix source)) source nil))))

(defn view-sy
  "Attach a symmetric matrix to the raw data of `a`.

  Changes to the resulting object affect the source `a`, even the parts of data that might not be
  accessible by a. Use with caution!

  Options: `:uplo` (`upper` or `:lower`)..

  view-sy creates a new instance that reuses the master's data.

      (view-sy (tr float-factory 3 (range 6)))
  "
  ([a]
   (api/view-sy a true))
  ([^Matrix a options]
   (api/view-sy a (api/options-lower? options))))

(defn gb
  "Creates a general banded matrix (GB) in the context of `factory`, with `m` rows, `n` columns,
  `kl` sub (lower) diagonals and `ku` super (upper) diagonals.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimensions `m`, `n`, `kl`, and `ku` are not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (gb float-factory 4 3 1 2 (range 20) {:layout :row})
  "
  ([factory m n kl ku source options]
   (if (and (or (= 0 kl m) (< -1 (long kl) (long m))) (or (= 0 ku n) (< -1 (long ku) (long n))))
     (let-release [res (api/create-gb (api/factory factory) m n kl ku
                                      (api/options-column? options) (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "GB matrix cannot have a negative dimension nor overflow diagonals."
                     {:m m :n n :kl kl :ku ku})))
  ([factory m n kl ku arg]
   (if (or (not arg) (map? arg))
     (gb factory m n kl ku nil arg)
     (gb factory m n kl ku arg nil)))
  ([factory m n arg]
   (gb factory m n 0 0 arg))
  ([factory m n kl ku]
   (gb factory m n kl ku nil nil))
  ([factory m n]
   (gb factory m n 0 0 nil nil))
  ([factory ^Matrix a]
   (let [reg (api/region a)]
     (gb factory (.mrows a) (.ncols a) (.kl reg) (.ku reg) nil))))

(defn sb
  "Creates a symmetric banded matrix (SB) in the context of `factory`, with `n` rows and columns,
  `k` sub (lower) and 'k' super (upper) diagonals.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimensions `n` and `k` are not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  and `:uplo` (`:upper` or `:lower`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (sb float-factory 4 2 (range 20) {:layout :row})
  "
  ([factory n k source options]
   (if (or (< -1 (long k) (long n)) (= 0 k n))
     (let-release [res (api/create-sb (api/factory factory) n k (api/options-column? options)
                                      (api/options-lower? options) (not (:raw options)))]
       (if source
         (transfer! source res)
         res))
     (dragan-says-ex "SB matrix cannot have a negative dimension nor overflow diagonals." {:n n :k k})))
  ([factory n source arg]
   (let [[k src] (if (number? source) [source nil] [(max 0 (dec (long n))) source])]
     (if (or (not arg) (map? arg))
       (sb factory n k src arg)
       (sb factory n k arg nil))))
  ([factory ^long n arg]
   (cond
     (number? arg) (sb factory n arg nil nil)
     (or (not arg) (map? arg)) (sb factory n (max 0 (dec n)) nil arg)
     :default (sb factory n (max 0 (dec n)) arg nil)))
  ([factory source]
   (if (number? source)
     (sb factory source (max 0 (dec (long source))) nil nil)
     (let [n (min (.mrows ^Matrix source) (.ncols ^Matrix source))
           reg (api/region source)]
       (sb factory n (max 0 (min (dec n) (max (.kl reg) (.ku reg))))  source nil)))))

(defn tb
  "Creates a triangular banded matrix (TB) in the context of `factory`, with `n` rows and columns,
  `k` sub (lower)  or super (upper) diagonals.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimensions `n` and `k` are not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  `:uplo` (`:upper` or `:lower`), and `:diag` (`:unit` or `:non-unit`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (tb float-factory 4 2 (range 20) {:layout :row})
  "
  ([factory n k source options]
   (if (or (< -1 (long k) (long n)) (= 0 k n))
     (let-release [res (api/create-tb (api/factory factory) n k (api/options-column? options)
                                      (api/options-lower? options) (api/options-diag-unit? options)
                                      (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "TB matrix cannot have a negative dimension nor overflow diagonals." {:n n :k k})))
  ([factory n source arg]
   (let [[k src] (if (number? source) [source nil] [(max 0 (dec (long n))) source])]
     (if (or (not arg) (map? arg))
       (tb factory n k src arg)
       (tb factory n k arg nil))))
  ([factory ^long n arg]
   (cond
     (number? arg) (tb factory n arg nil nil)
     (or (not arg) (map? arg)) (tb factory n (max 0 (dec n)) nil arg)
     :default (tb factory n (max 0 (dec n)) arg nil)))
  ([factory source]
   (if (number? source)
     (tb factory source (max 0 (dec (long source))) nil nil)
     (let [n (min (.mrows ^Matrix source) (.ncols ^Matrix source))
           reg (api/region source)]
       (tb factory n (min (max 0 (dec n)) (max (.kl reg) (.ku reg))) source nil)))))

(defn tp
  "Creates a triangular packed matrix (TP) in the context of `factory`, with `n` rows and columns.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimension `n` is not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  `:uplo` (`:upper` or `:lower`), and `:diag` (`:unit` or `:non-unit`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (tp float-factory 4 2 (range 20) {:uplo :upper :diag :unit})
  "
  ([factory n source options]
   (if  (< -1 (long n))
     (let-release [res (api/create-tp (api/factory factory) n (api/options-column? options)
                                      (api/options-lower? options) (api/options-diag-unit? options)
                                      (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "TP matrix cannot have a negative dimension." {:n n})))
  ([factory n arg]
   (if (or (not arg) (map? arg))
     (tp factory n nil arg)
     (tp factory n arg nil)))
  ([factory source]
   (if (number? source)
     (tp factory source nil nil)
     (tp factory (min (.mrows ^Matrix source) (.ncols ^Matrix source)) source nil))))

(defn sp
  "Creates a symmetric packed matrix (SP) in the context of `factory`, with `n` rows and columns.

  If `source` is provided, transfers the data to the result. `source` is typically a sequence,
  a matrix, or another structure that can be transferred to the context of `factory`.
  If `source` is a matrix, dimension `n` is not mandatory.

  The internal structure can be specified with a map of options: `:layout` (`:column` or `:row`),
  and `:uplo` (`:upper` or `:lower`).

  If the provided indices or source do not make sense, throws ExceptionInfo.

      (sp float-factory 4 2 (range 20) {:uplo :upper})
  "
  ([factory ^long n source options]
   (if  (< -1 n)
     (let-release [res (api/create-sp (api/factory factory) n (api/options-column? options)
                                      (api/options-lower? options) (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "SP matrix cannot have a negative dimension." {:n n})))
  ([factory n arg]
   (if (or (not arg) (map? arg))
     (sp factory n nil arg)
     (sp factory n arg nil)))
  ([factory arg]
   (if (number? arg)
     (sp factory arg nil nil)
     (sp factory (min (.mrows ^Matrix arg) (.ncols ^Matrix arg)) arg nil))))

(defn gd
  "Creates a diagonal matrix (GD) in the context of `factory`, with `n` rows and columns.

  If the provided side is negative, throws ExceptionInfo.
  "
  ([factory ^long n source options]
   (if (< -1 n)
     (let-release [res (api/create-diagonal (api/factory factory) n :gd (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "GT matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (gd factory n nil arg)
     (gd factory n arg nil)))
  ([factory arg]
   (if (number? arg)
     (gd factory arg nil nil)
     (gd factory (min (.mrows ^Matrix arg) (.ncols ^Matrix arg)) arg nil))))

(defn gt
  "Creates a tridiagonal matrix (GT) in the context of `factory`, with `n` rows and columns.

  If the provided side is negative, throws ExceptionInfo.
  "
  ([factory ^long n source options]
   (if (< -1 n)
     (let-release [res (api/create-diagonal (api/factory factory) n :gt (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "GT matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (gt factory n nil arg)
     (gt factory n arg nil)))
  ([factory arg]
   (if (number? arg)
     (gt factory arg nil nil)
     (gt factory (min (.mrows ^Matrix arg) (.ncols ^Matrix arg)) arg nil))))

(defn dt
  "Creates a diagonally dominant tridiagonal matrix (DT) in the context of `factory`,
  with `n` rows and columns.

  If the provided side is negative, throws ExceptionInfo.
  "
  ([factory ^long n source options]
   (if (< -1 n)
     (let-release [res (api/create-diagonal (api/factory factory) n :dt (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "DT matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (dt factory n nil arg)
     (dt factory n arg nil)))
  ([factory arg]
   (if (number? arg)
     (dt factory arg nil nil)
     (dt factory (min (.mrows ^Matrix arg) (.ncols ^Matrix arg)) arg nil))))

(defn st
  "Creates a symmetric tridiagonal matrix (ST) in the context of `factory`,
  with `n` rows and columns.

  If the provided side is negative, throws ExceptionInfo.
  "
  ([factory ^long n source options]
   (if (< -1 n)
     (let-release [res (api/create-diagonal (api/factory factory) n :st (not (:raw options)))]
       (if source (transfer! source res) res))
     (dragan-says-ex "ST matrix cannot have a negative dimension." {:n n})))
  ([factory ^long n arg]
   (if (or (not arg) (map? arg))
     (st factory n nil arg)
     (st factory n arg nil)))
  ([factory arg]
   (if (number? arg)
     (st factory arg nil nil)
     (st factory (min (.mrows ^Matrix arg) (.ncols ^Matrix arg)) arg nil))))

;; ================= Container  ================================================

(defn raw
  "Returns an uninitialized instance of the same type and dimension(s) as `x`."
  ([x]
   (api/raw x))
  ([factory x]
   (api/raw x factory)))

(defn zero
  "Returns an instance of the same type and dimension(s) as the `x`, initialized with `0`."
  ([x]
   (api/zero x))
  ([factory x]
   (api/zero x factory)))

;; ================= Vector ====================================================

(defn dim
  "Returns the dimension of the vector `x`."
  ^long [^Vector x]
  (.dim x))

(defn subvector
  "Returns a subvector starting witk `k`, `l` entries long, which is a part of the neanderthal vector `x`.

  The resulting subvector has a live connection to `x`'s data. Any change to the subvector data
  will affect the vector data. If you wish to disconnect the subvector from the parent vector,
  make a copy prior to any destructive operation.

  If the requested region is not within the dimensions of `x`, throws ExceptionInfo.
  "
  [^Vector x ^long k ^long l]
  (if (<= (+ k l) (.dim x))
    (.subvector x k l)
    (throw (ex-info "Requested subvector is out of bounds." {:k k :l l :k+l (+ k l) :dim (.dim x)}))))

;; ================= Matrix =======================

(defn mrows
  "Returns the number of rows of the matrix `a`."
  ^long [^Matrix a]
  (.mrows a))

(defn ncols
  "Returns the number of columns of the matrix `a`."
  ^long [^Matrix a]
  (.ncols a))

(defn row
  "Returns the `i`-th row of the matrix `a` as a vector.

  The vector has access to and can change the same data as the original matrix.

  If the requested column is not within the dimensions of `a`, throws ExceptionInfo.
  "
  [^Matrix a ^long i]
  (if (< -1 i (.mrows a))
    (.row a i)
    (throw (ex-info "Requested row is out of bounds." {:i i :mrows (.mrows a)}))))

(defn col
  "Returns the `j`-th column of the matrix `a` as a vector.

  The vector has access to and can change the same data as the original matrix.

  If the requested column is not within the dimensions of `a`, throws ExceptionInfo.
  "
  [^Matrix a ^long j]
  (if (< -1 j (.ncols a))
    (.col a j)
    (throw (ex-info "Requested column is out of bounds." {:j j :ncols (.ncols a)}))))

(defn dia
  "Returns the diagonal elements of the matrix `a` in a vector. If called with an
  index `k`, returns a lower `(< k 0)` or upper `(< 0 k)` diagonal.

  The vector has access to and can change the same data as the original matrix."
  ([^Matrix a]
   (.dia a))
  ([^Matrix a ^long k]
   (if (< (- (.mrows a)) k (.ncols a))
     (.dia a k)
     (throw (ex-info "Requested diagonal is out of bounds." {:k k :mrows (.mrows a) :ncols (.ncols a)})))))

(defn cols
  "Returns a lazy sequence of column vectors of the matrix `a`.

  The vectors has access to and can change the same data as the original matrix."
  [^Matrix a]
  (.cols a))

(defn rows
  "Returns a lazy sequence of row vectors of the matrix `a`.

  The vectors has access to and can change the same data as the original matrix."
  [^Matrix a]
  (.rows a))

(defn dias
  "Returns a lazy sequence of diagonal vectors of the matrix `a`.

  The vectors has access to and can change the same data as the original matrix."
  [^Matrix a]
  (.dias a))

(defn submatrix
  "Returns a submatrix of the matrix `a` starting from row `i`, column `j`,
  that has `k` rows and `l` columns.

  The resulting submatrix has a live connection to `a`'s data. Any change to the subvector data
  will affect the original data. If you wish to disconnect the submatrix from the parent matrix,
  make a copy prior to any destructive operation.

  If the requested region is not within the dimensions of `a`, throws ExceptionInfo.

      (submatrix (ge double-factroy 4 3 (range 12)) 1 1 2 1)
  "
  ([^Matrix a i j k l]
   (if (and (<= 0 (long i) (+ (long i) (long k)) (.mrows a))
            (<= 0 (long j) (+ (long j) (long l)) (.ncols a)))
     (.submatrix a i j k l)
     (throw (ex-info "Requested submatrix is out of bounds."
                     {:i i :j j :k k :l l :mrows (.mrows a) :ncols (.ncols a)
                      :i+k (+ (long i) (long k)) :j+l (+ (long j) (long l))}))))
  ([^Matrix a k l]
   (submatrix a 0 0 k l)))

(defn subband
  "Returns a part of the banded matrix `a` starting from row `0`, column `0`,
  that has `kl` subdiagonals and `ku` superdiagonals.

  The resulting submatrix has a live connection to `a`'s data. Any change to the subband data
  will affect the original data.

  If the requested region is not within the dimensions of `a`, throws ExceptionInfo.
  "
  ([a kl ku]
   (api/subband a kl ku)))

(defn trans!
  "Transposes matrix `a`'s data **in-place**. For the 'real' transpose, use the `trans` function.

  The transpose affects the internal structure of `a`.

      (trans! (dge 2 3 [1 2, 3 4, 5 6])) => (dge 2 3 [1 3, 5 2, 4 ,6])
  "
  [^Matrix a]
  (api/trans (api/engine a) a))

(defn trans
  "Transposes matrix `a`, i.e returns a matrix that has m's columns as rows.

  The transpose does not affect the internal structure of `a`. The resulting matrix has
  a live connection to `a`'s data."
  [^Matrix a]
  (.transpose a))

;; ============ Vector and Matrix access methods ===============================

(defn entry
  "Returns the **boxed** `i`-th entry of vector `x`, or `(i, j)`-th entry of matrix `a`.

  For optimally accessing the elements of native objects use the equivalent `entry`
  function from `uncomplicate.neanderthal.real` namespace.

  If `i` or `j` is not within the dimensions of the object, throws ExceptionInfo.
  "
  ([^Vector x ^long i]
   (try
     (.boxedEntry x i)
     (catch IndexOutOfBoundsException e
       (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim (.dim x)})))))
  ([^Matrix a ^long i ^long j]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)))
     (.boxedEntry a i j)
     (throw (ex-info "Requested element is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn entry!
  "Sets the `i`-th entry of vector `x`, or `(i, j)`-th entry of matrix `a` using a **boxed** method.

  For optimally accessing the elements of native objects use the equivalent `entry!`
  function from `uncomplicate.neanderthal.real` namespace.

  If `i` or `j` is not within the dimensions of the object, throws ExceptionInfo.
  "
  ([^Changeable x val]
   (.setBoxed x val))
  ([^Changeable x ^long i val]
   (try
     (.setBoxed x i val)
     (catch IndexOutOfBoundsException e
       (throw (ex-info "The element you're trying to set is out of bounds of the vector."
                       {:i i :dim (dim x)})))))
  ([^Matrix a ^long i ^long j val]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)) (.isAllowed ^Changeable a i j))
     (.setBoxed ^Changeable a i j val)
     (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn alter!
  "Alters the `i`-th entry of vector `x`, or `(i, j)`-th entry of matrix `a` by
  evaluating function f on one or all of its elements.

  If no index is passed, the function f will alter all elements and feeds the indices to f.
  If the structure holds primitive elements, the function f must accept appropriate primitive
  unboxed arguments, and it will work faster if it also returns unboxed result.

  If `i` or `j` is not within the dimensions of the object, throws ExceptionInfo.

      (alter! (dv 1 2 3) 2 (fn ^double [^double x] (inc x)))
      (alter! (dge 2 2) (fn ^double [^long i ^long j ^double x] (double (+ i j))))
  "
  ([^Changeable x f]
   (.alter x f))
  ([^Changeable x ^long i f]
   (try
     (.alter x i f)
     (catch IndexOutOfBoundsException e
       (throw (ex-info "The element you're trying to alter is out of bounds of the vector."
                       {:i i :dim (dim x)})))))
  ([^Matrix a ^long i ^long j f]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)) (.isAllowed ^Changeable a i j))
     (.alter ^Changeable a i j f)
     (throw (ex-info "The element you're trying to alter is out of editable bounds of the matrix."
                     {:a (info a)})))))

;;================== BLAS 1 =======================

(defn dot
  "Computes the dot product of vectors or matrices `x` and `y`.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?dot](https://software.intel.com/en-us/node/520734).

      (dot (dv 1 2 3) (dv 1 2 3)) => 14.0
  "
  [^Vector x ^Vector y]
  (if (and (api/compatible? x y) (api/fits? x y))
    (api/dot (api/engine x) x y)
    (dragan-says-ex "You cannot compute dot of incompatible or ill-fitting vectors."
                    {:x (info x) :y (info y)})))

(defn nrm1
  "Computes the 1-norm of vector or matrix `x`."
  [x]
  (api/nrm1 (api/engine x) x))

(defn nrm2
  "Computes the Euclidan norm of vector `x`, or Frobenius norm of matrix `x`.

  See related info about [cblas_?nrm2](https://software.intel.com/en-us/node/520738).

      (nrm2 (dv 1 2 3)) => 3.7416573867739413
  "
  [x]
  (api/nrm2 (api/engine x) x))

(defn nrmi
  "Computes the infinity-norm of vector or matrix `x`."
  [x]
  (api/nrmi (api/engine x) x))

(defn asum
  "Sums absolute values of entries of vector or matrix `x`''.

  See related info about [cblas_?asum](https://software.intel.com/en-us/node/520731).

      (asum (dv -1 2 -3)) => 6.0
  "
  [x]
  (api/asum (api/engine x) x))

(defn iamax
  "The index of the first entry of vector `x` that has the largest absolute value.

    See related info about [cblas_i?amax](https://software.intel.com/en-us/node/520745).

      (iamax (dv 1 -3 2)) => 1
  "
  ^long [^VectorSpace x]
  (if (< 0 (.dim x))
    (api/iamax (api/engine x) x)
    0))

(defn iamin
  "The index of the first entry of vector `x` that has the smallest absolute value.

    See related info about [cblas_i?amin](https://software.intel.com/en-us/node/520746).

      (iamin (dv 1 -3 2)) => 0
  "
  ^long [^VectorSpace x]
  (if (< 0 (.dim x))
    (api/iamin (api/engine x) x)
    0))

(defn amax
  "Absolute value of the first entry of vector `x` that has the largest absolute value.

  See related info about [cblas_i?amax](https://software.intel.com/en-us/node/520745).

      (amax (dv 1 -3 2)) => 3
  "
  [x]
  (api/amax (api/engine x) x))

(defn imax
  "The index of the first entry of vector `x` that has the largest value.

  See related info about [cblas_i?amax](https://software.intel.com/en-us/node/520745).

      (imax (dv 1 -3 2)) => 2
  "
  ^long [^VectorSpace x]
  (if (< 0 (.dim x))
    (api/imax (api/engine x) x)
    0))

(defn imin
  "The index of the first entry of vector `x` that has the smallest value.

  See related info about [cblas_i?amin](https://software.intel.com/en-us/node/520746).

      (imin (dv 1 -3 2)) => 2
  "
  ^long [^VectorSpace x]
  (if (< 0 (.dim x))
    (api/imin (api/engine x) x)
    0))

(defn swp!
  "Swaps all entries of vectors or matrices `x` and `y`.

  Both `x` and `y` will be changed.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?swap](https://software.intel.com/en-us/node/520744).
  "
  [^VectorSpace x y]
  (if-not (identical? x y)
    (if (and (api/compatible? x y) (api/fits? x y))
      (if (< 0 (.dim x))
        (api/swap (api/engine x) x y)
        x)
      (dragan-says-ex "You cannot swap data of incompatible or ill-fitting structures."
                      {:x (info x) :y (info y)}))
    x))

(defn copy!
  "Copies all entries of a vector or a matrix `x` to the compatible structure `y`.

  Only `y` will be changed.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?copy](https://software.intel.com/en-us/node/520733).
  "
  ([^VectorSpace x y]
   (if-not (identical? x y)
     (if (and (api/compatible? x y) (api/fits? x y))
       (if (< 0 (.dim x))
         (api/copy (api/engine x) x y)
         y)
       (dragan-says-ex "You cannot copy data of incompatible or ill-fitting structures."
                       {:x (info x) :y (info y)}))
     y))
  ([^Vector x ^Vector y offset-x length offset-y]
   (if-not (identical? x y)
     (if (and (api/compatible? x y)
              (<= (+ (long offset-x) (long length)) (.dim x))
              (<= (+ (long offset-y) (long length)) (.dim y)))
       (if (< 0 (.dim x))
         (api/subcopy (api/engine x) x y (long offset-x) (long length) (long offset-y))
         y)
       (dragan-says-ex "You cannot copy data of incompatible vectors"
                       {:x (info x) :y (info y) :length length}))
     y)))

(defn copy
  "Copies all entries of a vector or a matrix `x` to the newly created structure (see [[copy!]])."
  ([x]
   (let-release [res (api/raw x)]
     (copy! x res)))
  ([x offset length]
   (let-release [res (vctr (api/factory x) length)]
     (copy! x res offset length 0))))

(defn scal!
  "Multiplies all entries of a vector or matrix `x` by scalar `alpha`.

  After `scal!`, `x` will be changed.

  See related info about [cblas_?scal](https://software.intel.com/en-us/node/520743).
  "
  [alpha ^VectorSpace x]
  (when (< 0 (.dim x))
    (api/scal (api/engine x) alpha x))
  x)

(defn scal
  "Multiplies all entries of a copy a vector or matrix `x` by scalar `alpha`.

  See related info about [cblas_?scal](https://software.intel.com/en-us/node/520743).
  "
  [alpha x]
  (let-release [res (copy x)]
    (scal! alpha res)))

(defn rot!
  "Rotates points of vectors `x` and `y` in the plane by cos `c` and sin `s`.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?rot](https://software.intel.com/en-us/node/520739).
  "
  ([^Vector x ^Vector y ^double c ^double s]
   (if (and (api/compatible? x y))
     (if (and (<= -1.0 c 1.0) (<= -1.0 s 1.0) (f= 1.0 (+ (pow c 2.0) (pow s 2.0))))
       (api/rot (api/engine x) x y c s)
       (dragan-says-ex "You cannot rotate vectors with c and s that are not sin and cos."
                       {:c c :s s :errors
                        (cond-into []
                                   (not (<= -1.0 c 1.0)) "c is not between -1.0 and 1.0"
                                   (not (<= -1.0 s 1.0)) "s is not between -1.0 and 1.0"
                                   (not (f= 1.0 (+ (pow c 2.0) (pow s 2.0)))) "s^2 + c^2 is not 1.0")}))
     (dragan-says-ex "You cannot rotate incompatible vectors." {:x (info x) :y (info y)})))
  ([x y ^double c]
   (rot! x y c (sqrt (- 1.0 (pow c 2.0))))))

(defn rotg!
  "Computes the parameters for a givens rotation and puts them as entries of a 4-element vector `abcs`,
  that can be used as parameters in `rot!`.

  If  `abcs` does not have at least 4 entries, throws ExceptionInfo.

  See related info about [cblas_?rotg](https://software.intel.com/en-us/node/520740).
  "
  [^Vector abcs]
  (if (< 3 (.dim abcs))
    (api/rotg (api/engine abcs) abcs)
    (throw (ex-info "Vector abcs must have at least 4 elements." {:dim (.dim abcs)}))))

(defn rotm!
  "Rotates points of vectors `x` and `y` in the plane using Givens rotation.

  `param` must be a vector of at least 4 entries.

  See related info about [cblas_?rotm](https://software.intel.com/en-us/node/520741).
  "
  [^Vector x ^Vector y ^Vector param]
  (if (and (< 0 (.dim x)) (< 0 (.dim y)))
    (if (and (api/compatible? x y) (api/compatible? x param) (api/fits? x y) (< 4 (.dim param)))
      (api/rotm (api/engine x) x y param)
      (dragan-says-ex "You cannot apply modified plane rotation with incompatible or ill-fitting vectors."
                      {:x (info x) :y (info y) :param (info param) :errors
                       (cond-into []
                                  (not (api/compatible? x y)) "incompatible x and y"
                                  (not (api/compatible? x param)) "incompatible x and param"
                                  (not (api/fits? x y)) "ill-fitting x and y"
                                  (not (< 4 (.dim param))) "param is shorter than 4")}))
    x))

(defn rotmg!
  "BLAS 1: Generate modified plane rotation.

  If the context or dimensions of  `d1d2xy` and `param` are not compatible, or `d1d2xy` does not have
  at least at least 4 entries, or `param` does not have at least 5 entries, throws ExceptionInfo.

  See related info about [cblas_?rotmg](https://software.intel.com/en-us/node/520742).
  "
  [^Vector d1d2xy ^Vector param]
  (if (and (api/compatible? d1d2xy param) (< 3 (.dim d1d2xy)) (< 4 (.dim param)))
    (api/rotmg (api/engine param) d1d2xy param)
    (dragan-says-ex "You cannot generate modified plane rotation with incompatible or ill-fitting vectors."
                    {:d1d2xy (info d1d2xy) :param (info param) :errors
                     (cond-into []
                                (not (api/compatible? d1d2xy param)) "incompatible d2d2xy and param"
                                (not (< 3 (.dim param))) "d1d2xy is shorter than 3"
                                (not (< 4 (.dim param))) "param is shorter than 4")})))

(defn axpy!
  "Multiplies elements of a vector or matrix `x` by scalar `alpha` and adds it to a vector or matrix `y`.

  The contents of `y` will be changed.


  If called with 2 arguments, `x` and `y`, adds vector or matrix `x` to a vector or matrix `y`.
  If called with more than 3 arguments, at least every other have to be a vector or matrix.
  A scalar multiplier may be included before each object.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?axpy](https://software.intel.com/en-us/node/520732).

      (axpy! 3 x y)
      (axpy! x y)
      (axpy! 3 x y 4 z 0.4 v)
  "
  ([alpha ^VectorSpace x y]
   (if (and (api/compatible? x y) (api/fits? x y))
     (if (< 0 (.dim x))
       (api/axpy (api/engine x) alpha x y)
       y)
     (dragan-says-ex "You cannot add incompatible or ill-fitting structures."
                     {:x (info x) :y (info y)})))
  ([x y]
   (axpy! 1.0 x y))
  ([alpha x y & zs]
   (if (number? alpha)
     (do (axpy! alpha x y)
         (loop [z (first zs) zs (rest zs)]
           (if z
             (if (number? z)
               (do (axpy! z (first zs) y) (recur (second zs) (rest (rest zs))))
               (do (axpy! 1.0 z y) (recur (first zs) (rest zs))))
             y)))
     (apply axpy! 1.0 alpha x y zs))))

(declare axpby!)

(defn axpy
  "A pure variant of [[axpy!]] that does not change any of the arguments. The result is a new instance."
  ([x y]
   (let-release [res (copy y)]
     (axpy! 1.0 x res)))
  ([x y z]
   (if (number? x)
     (let-release [res (copy z)]
       (axpy! x y res))
     (let-release [res (copy y)]
       (axpy! 1.0 x res z))))
  ([x y z w & ws]
   (if (number? x)
     (if (number? z)
       (let-release [res (copy w)]
         (apply axpby! x y z res ws))
       (let-release [res (copy z)]
         (apply axpy! x y res w ws)))
     (apply axpy 1.0 x y z w ws))))

(defn ax
  "Multiplies vector or matrix `x` by a scalar `alpha`. Similar to [[scal!]], but does not change x.
  The result is a new instance. See [[axpy!]] and [[axpy]]."
  [alpha x]
  (let-release [res (zero x)]
    (axpy! alpha x res)))

(defn xpy
  "Sums vectors or matrices `x`, `y`, and `zs`. The result is a new instance. See [[axpy!]] and [[axpy]]."
  ([x y]
   (let-release [res (copy y)]
     (axpy! 1.0 x res)))
  ([x y & zs]
   (let-release [cy (copy y)]
     (loop [res (axpy! 1.0 x cy) s zs]
       (if s
         (recur (axpy! 1.0 (first s) res) (next s))
         res)))))

(defn axpby!
  "Multiplies elements of a vector or matrix `x` by scalar `alpha` and adds it to
   vector or matrix `y`, that was previously scaled by scalar `beta`.

  The contents of `y` will be changed.


  If called with 2 arguments, `x` and `y`, adds vector or matrix `x` to a vector or matrix `y`.
  If called with more than 4 arguments, at least every other have to be a vector or matrix.
  A scalar multiplier may be included before each object.

  If the context or dimensions of  `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?axpby](https://software.intel.com/en-us/node/520858).

      (axpy! 3 x 0.9 y)
      (axpy! x y)
      (axpy! 3 x 0.2 y 4 z 0.4 v)
  "
  ([alpha ^VectorSpace x beta y]
   (if (and (api/compatible? x y) (api/fits? x y))
     (if (< 0 (.dim x))
       (api/axpby (api/engine x) alpha x beta y)
       y)
     (dragan-says-ex "You cannot add incompatible or ill-fitting structures."
                     {:x (info x) :y (info y)})))
  ([x beta y]
   (axpby! 1.0 x beta y))
  ([x y]
   (axpy! 1.0 x y))
  ([alpha x beta y & zs]
   (if (number? alpha)
     (do (axpby! alpha x beta y)
         (loop [z (first zs) zs (rest zs)]
           (if z
             (if (number? z)
               (do (axpy! z (first zs) y) (recur (second zs) (rest (rest zs))))
               (do (axpy! 1.0 z y) (recur (first zs) (rest zs))))
             y)))
     (apply axpby! 1.0 alpha x beta y zs))))

;;============================== BLAS 2 ========================================

(defn mv!
  "Matrix-vector multiplication. Multiplies matrix `a`, scaled by scalar `alpha`, by vector `x`,
  and ads it to vector `y` previously scaled by scalar `beta`.

  If called with 2 arguments, `a` and `x`, multiplies matrix `a` by a vector `x`, and puts the
  result in the vector `x`. Scaling factors `alpha` and/or `beta` may be left out.

  The contents of the destination vector will be changed.

  If the context or dimensions of `a`, `x` and `y` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?gemv](https://software.intel.com/en-us/node/520750), and
  [cblas_?trmv](https://software.intel.com/en-us/node/520772).

      (mv! 3 a x y)
      (mv! a x y)
      (mv! a x)
  "
  ([alpha ^Matrix a ^Vector x beta ^Vector y]
   (if (and (api/compatible? a x) (api/compatible? a y)
            (= (.ncols a) (.dim x)) (= (.mrows a) (.dim y)))
     (api/mv (api/engine a) alpha a x beta y)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting structures."
                     {:a (info a) :x (info x) :y (info y) :errors
                      (cond-into []
                                 (not (api/compatible? a x)) "incompatible a and x"
                                 (not (api/compatible? a y)) "incompatible a and y"
                                 (not (= (.ncols a) (.dim x))) "(dim x) is not equals to (ncols a)"
                                 (not (= (.mrows a) (.dim y))) "(dim y) is not equals to (mrows a)")})))
  ([alpha a x y]
   (mv! alpha a x 1.0 y))
  ([a x y]
   (mv! 1.0 a x 0.0 y))
  ([^Matrix a ^Vector x]
   (if (and (api/compatible? a x) (= (.ncols a) (.dim x)))
     (api/mv (api/engine a) a x)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting structures."
                     {:a (info a) :x (info x) :errors
                      (cond-into []
                                 (not (api/compatible? a x)) "incompatible a and x"
                                 (not (= (.ncols a) (.dim x))) "(dim x) is not equals to (ncols a)")}))))

(defn mv
  "Matrix-vector multiplication. A pure version of mv! that returns the result in a new vector
  instance. Computes alpha * a * x."
  ([alpha a x beta y]
   (let-release [res (copy y)]
     (mv! alpha a x beta res)))
  ([alpha a x y]
   (mv alpha a x 1.0 y))
  ([alpha ^Matrix a x]
   (let-release [res (vctr (api/factory a) (.mrows a))]
     (mv! alpha a x 0.0 res)))
  ([a x]
   (if-not (triangular? a)
     (mv 1.0 a x)
     (let-release [res (copy x)]
       (mv! a res)))))

(defn rk!
  "The outer product of two vectors. Multiplies vector `x` with transposed vector `y`,
  scales the resulting matrix  by scalar `alpha`, and adds it to the matrix `a`.

  The contents of `a` will be changed.

  If the context or dimensions of `a`, `x` and `y` are not compatible, throws ExceptionInfo.

  If `y` is not provided, uses `x` in its place, and computes a symmetric outer product.

  See related info about [cblas_?ger](https://software.intel.com/en-us/node/520751).


      (rk! 1.5 (dv 1 2 3) (dv 4 5) a)
  "
  ([alpha ^Vector x ^Vector y ^Matrix a]
   (if (and (api/compatible? a x) (api/compatible? a y) (= (.mrows a) (.dim x)) (= (.ncols a) (.dim y)))
     (api/rk (api/engine a) alpha x y a)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting structures."
                     {:a (info a) :x (info x) :y (info y) :errors
                      (cond-into []
                                 (not (api/compatible? a x)) "incompatible a and x"
                                 (not (api/compatible? a y)) "incompatible a and y"
                                 (not (= (.ncols a) (.dim x))) "(dim x) is not equals to (ncols a)"
                                 (not (= (.mrows a) (.dim y))) "(dim y) is not equals to (mrows a)")})))
  ([alpha-or-x ^Vector y ^Matrix a]
   (if (number? alpha-or-x)
     (if (and (api/compatible? a y) (= (.mrows a) (.dim y)))
       (api/rk (api/engine a) alpha-or-x y a)
       (dragan-says-ex "You cannot multiply incompatible or ill-fitting structures."
                       {:a (info a) :y (info y) :errors
                        (cond-into []
                                   (not (api/compatible? a y)) "incompatible a and y"
                                   (not (= (.ncols a) (.dim y))) "(dim y) is not equals to (ncols a)")}))
     (rk! 1.0 alpha-or-x y a)))
  ([^Vector y ^Matrix a]
   (rk! 1.0 y a)))

(defn rk
  "Pure outer product of two vectors. A pure version of [[rk!]] that returns
  the result in a new matrix instance."
  ([alpha x y a]
   (let-release [res (copy a)]
     (rk! alpha x y res)))
  ([alpha ^Vector x ^Vector y]
   (let-release [res (ge (api/factory x) (.dim x) (.dim y))]
     (rk! alpha x y res)))
  ([alpha-or-x ^Vector y]
   (if (number? alpha-or-x)
     (let-release [res (sy (api/factory y) (.dim y))]
       (rk! alpha-or-x y res))
     (rk 1.0 alpha-or-x y)))
  ([x]
   (rk 1.0 x)))

(defn mmt!
  "Multiplies matrix `a` by its transpose, scales the result by `alpha`,
  and adds it to a symmetric matrix `c` scaled by `beta`.

  If `alpha` and/or `beta` are not provided, scales by `1.0`."
  ([alpha ^Matrix a beta ^Matrix c]
   (if (and (api/compatible? a c) (= (.mrows a) (.mrows c)))
     (api/srk (api/engine c) alpha a beta c)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting matrices."
                     {:a (info a) :c (info c) :errors
                      (cond-into []
                                 (not (api/compatible? a c)) "incompatible a and c"
                                 (not (= (.mrows a) (.mrows c))) "(mrows a) is not equals to (mrows c)")})))
  ([alpha a c]
   (mmt! alpha a 1.0 c))
  ([a c]
   (mmt! 1.0 a 1.0 c)))

(defn mmt
  "Pure variant of [[mmt]], multiplies matrix `a` by its transpose and return
  the scaled result in a new instance of SY matrix `c`.

  If `alpha` is not provided, the scaling factor is `1.0`. "
  ([alpha ^Matrix a]
   (let-release [res (sy (api/factory a) (.mrows a))]
     (mmt! alpha a 0.0 res)))
  ([a]
   (mmt 1.0 a)))

;; =========================== BLAS 3 ==========================================

(defn mm!
  "Matrix-matrix multiplication. Multiplies matrix `a`, scaled by scalar `alpha`, by matrix `b`,
  and ads it to matrix `c` previously scaled by scalar `beta`.

  If called with only 2 matrices, `a` and `b`, multiplies matrix `a` by a matrix `b`, and puts the
  result in the one that is a GE matrix. In this case, exactly one of `a` or `b` has to be,
  and one **not** be a GE matrix, but TR, or a matrix type that supports in-place multiplication.
  Scaling factors `alpha` and/or `beta` may be left out.

  The contents of the destination matrix will be changed.

  If the context or dimensions of `a`, `b` and `c` are not compatible, throws ExceptionInfo.

  See related info about [cblas_?gemm](https://software.intel.com/en-us/node/520775), and
  [cblas_?trmm](https://software.intel.com/en-us/node/520782).

      (def a (dge 2 3 (range 6)))
      (def b (dge 3 2 (range 2 8)))
      (def c (dge 2 2 [1 1 1 1]))
      (def e (dtr 3 (range 6)))

      (mm! 1.5 a b 2.5 c)
      (mm! 2.3 a e)
      (mm! e a)
  "
  ([alpha ^Matrix a ^Matrix b beta ^Matrix c]
   (if (and (api/compatible? a b) (api/compatible? a c)
            (= (.ncols a) (.mrows b)) (= (.mrows a) (.mrows c)) (= (.ncols b) (.ncols c)))
     (api/mm (api/engine a) alpha a b beta c true)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting matrices."
                     {:a (info a) :b (info b) :c (info c) :errors
                      (cond-into []
                                 (not (api/compatible? a b)) "incompatible a and b"
                                 (not (api/compatible? a c)) "incompatible a and c"
                                 (not (= (.ncols a) (.mrows b))) "(ncols a) is not equals to (mrows b)"
                                 (not (= (.mrows a) (.mrows c))) "(mrows a) is not equals to (mrows c)"
                                 (not (= (.ncols b) (.ncols c))) "(ncols b) is not equals to (ncols c)")})))
  ([alpha a b c]
   (mm! alpha a b 1.0 c))
  ([alpha ^Matrix a ^Matrix b]
   (if (and (api/compatible? a b) (= (.ncols a) (.mrows b)))
     (api/mm (api/engine a) alpha a b true)
     (dragan-says-ex "You cannot multiply incompatible or ill-fitting matrices."
                     {:a (info a) :b (info b) :errors
                      (cond-into []
                                 (not (api/compatible? a b)) "incompatible a and b"
                                 (not (= (.ncols a) (.mrows b))) "(ncols a) is not equals to (mrows b)")})))
  ([a b]
   (mm! 1.0 a b)))

(defn ^:private elem* ^long [^longs a ^long n ^long i ^long j]
  (aget a (+ (* n j) i)))

(defn ^:private matrix-chain [as]
  (let [cnt (count as)
        p (long-array (inc cnt))
        m (mrows (as 0))]
    (aset p 0 m)
    (if (loop [i 0 k m quad true]
          (if (< i cnt)
            (let [a (as i)
                  n (ncols a)]
              (if (= k (mrows a))
                (recur (inc i) (aset p (inc i) n) (and quad (= m n) (instance? GEMatrix a)))
                (dragan-says-ex "You cannot chain-multiply ill-fitting matrices." {:ill-fitting-idx i})))
            quad))
      nil
      p)))

(defn ^:private optimize-chain [^longs p]
  (let [n (dec (alength p))
        m (long-array (* n n))]
    (loop [l 2]
      (when (<= l n)
        (dotimes [i (inc (- n l))]
          (let [j (dec (+ i l))]
            (aset m (+ (* n j) i) Long/MAX_VALUE)
            (loop [k i]
              (when (< k j)
                (let [q (+ (elem* m n i k) (elem* m n (inc k) j)
                           (* (aget p i) (aget p (inc k)) (aget p (inc j))))]
                  (when (< q (elem* m n i j))
                    (aset m (+ (* n j) i) q)
                    (aset m (+ (* n i) j) k)))
                (recur (inc k))))))
        (recur (inc l))))
    m))

(defn ^:private mm*
  ([alpha ^Matrix a ^Matrix b]
   (if-not (triangular? b)
     (if-not (triangular? a)
       (let-release [res (ge (api/factory b) (.mrows a) (.ncols b))]
         (mm! alpha a b 1.0 res))
       (let-release [res (copy b)]
         (mm! alpha a res)))
     (let-release [res (copy a)]
       (mm! alpha res b))))
  ([ms ^longs s ^long i ^long j]
   (if (= i j)
     (ms i)
     (let [k (elem* s (count ms) j i)]
       (let-release [ml (mm* ms s i k)
                     mr (mm* ms s (inc k) j)
                     res (mm* 1.0 ml mr)]
         (when (< i k) (release ml))
         (when (< (inc k) j) (release mr))
         res)))))

(defn mm
  "Pure matrix multiplication that returns the resulting matrix in a new instance.
  Computes `alpha a * b`, if alpha is scalar, or `alpha * a * b * c * ... * ds` if `alpha` is a vector.
  Does matrix composition by optimally multiplying the chain of matrices.

  If any consecutive pair's dimensions do not fit for matrix multiplication, throws `ExceptionInfo`.
  "
  ([a b c & ds]
   (if (or ds (matrix? a))
     (let [ms (into [a b c] ds)]
       (if-let [p (matrix-chain ms)]
         (mm* ms (optimize-chain p) 0 (dec (count ms)))
         (let-release [temp-0 (raw a)
                       temp-1 (raw a)]
           (loop [a a qs (next ms) temp temp-0 res temp-1]
             (if-let [b (first qs)]
               (recur (mm! 1.0 a b 0.0 temp) (next qs) res temp)
               (with-release [t temp]
                 res))))))
     (mm* a b c)))
  ([a b]
   (mm* 1.0 a b)))

;; ============================== BLAS Plus ====================================

(defn sum
  "Sums values of entries of a vector or matrix `x`.

      (sum (dv -1 2 -3)) => -2.0
  "
  [x]
  (api/sum (api/engine x) x))
