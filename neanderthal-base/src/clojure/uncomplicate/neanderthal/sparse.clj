;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.sparse
  "Functions for creating sparse vectors and matrices. Sparse vectors or matrices
  are structures in which most elements are zeroes. Therefore, it makes sense to
  store only the few non-zero entries. There is a performance penalty to pay for these
  entries, in terms of both storage and computation, as they are stored using one of many possible
  compression schemes, but if there is only a small fraction of non-zero elements compared to
  zero elements, that penalty is offset by the fact that only a fraction of computations
  need to be done.

  Compressed sparse storage schemes that we use here use dense vectors to store non-zero entries
  and appropriate indices. Therefore, most operations can be offloaded to these objects if needed.
  Neanderthal core functions are supported where it makes sense and where it is technically
  possible.

  Please see examples in [[uncomplicate.neanderthal.sparse-test]] and [Intel documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/sparse-blas-level-2-and-level-3-routines-001.html).

  ### Cheatsheet

  [[csv]], [[csv?]]
  [[csr]], [[csr?]]
  "
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [dim transfer transfer! subvector matrix?]]
             [integer :refer [amax]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all :exclude [amax]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [cs-vector seq-to-csr]])
  (:import [uncomplicate.neanderthal.internal.cpp.structures CSVector]))

(defn csv?
  "Checks whether `x` is a compressed sparse vector (CSV)."
  [x]
  (instance? CSVector x))

(defn csv
  "Creates a compressed sparse vector (CSV) using the supplied arguments.

  Arguments:
  - `fact`: the factory that should create the vector.
  - `n`: vector dimension; number of both zero and non-zero entries.
  - `nz`: a dense vector of values of non-zero entries, or varargs of those entries.
  - `idx`: integer dense vector of indices of the matching entries from `nz`.
  - `source`: can be another CSV, in which case its constituent indices vector is shared,
  or a Clojure sequence that contains data. Please see examples in [[uncomplicate.neanderthal.sparse-test]].
  The same formats are accepted by [[uncomplicate.neanderthal.core/transfer!]].

  The indices are zero-based by default.

  See more about sparse vectors in [Sparse BLAS Routines](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/sparse-blas-level-1-routines.html)."
  ([fact n idx nz & nzs]
   (let [fact (factory fact)
         idx-fact (index-factory fact)
         idx (or idx [])
         nz (or nz [])]
     (let-release [idx (if (compatible? idx-fact idx)
                         (view idx)
                         (transfer idx-fact idx))
                   res (cs-vector fact n (view idx) true)]
       (if-not nzs
         (transfer! nz (entries res))
         (transfer! (cons nz nzs) (entries res)))
       res)))
  ([fact ^long n source]
   (if (csv? source)
     (csv fact n (indices source) (entries source))
     (if (number? (first source))
       (csv fact n source nil)
       (csv fact n (first source) (second source)))))
  ([fact cs]
   (if (number? cs)
     (csv fact cs nil)
     (csv fact (dim cs) (indices cs) (entries cs)))))

(defn csr?
  "Checks whether `x` is a compressed sparse (row) matrix (CSR). CSR is the
  default compression scheme, supported by all routines in major libraries.
  Compressed columns usually do not have such a good coverage, and they are
  redundant anyway.
  "
  [x]
  (satisfies? CSR x))

(defn csr
  "Creates a compressed sparse (row) matrix using the supplied arguments and options.

  Arguments:
  - `fact`: the factory that should create the matrix.
  - `m`: number of rows
  - `n`: number of columns
  - `nz`: a dense vector of values of non-zero entries.
  - `idx`: integer dense vector of indices that represent a column of the matching entry from `nz`.
  - `idx-b`: integer dense vector of indices that represent the index of the first non-zero element in the `nz` vector.
  - `idx-e`: and integer dense vector of indices that represent the index of the last non-zero element in the `nz` vector.
  - `options`: supported option is `:layout` (`:column` or `:row`).

  - `idx-be`: `idx-b` and `idx-e` can be provided in one shared vector since each ending index is the
  beginning index of the next row.
  - `source`: can be another CSR, in which case its constituent index vectors are shared,
  or a Clojure sequence that contains data. This function will analyze the sequence and will try to
  guess its format. Please see examples in [[uncomplicate.neanderthal.sparse-test]] for the examples.
  The same formats are accepted by [[uncomplicate.neanderthal.core/transfer!]].

  The indices are zero-based by default.

  Please see [Sparse BLAS CSR Matrix Storage Format](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/sparse-blas-csr-matrix-storage-format.html) for a more detailed technical explanation of the CSR format, and the illustrative example.
  "
  ([fact m n idx idx-b idx-e nz options] ;; TODO error messages
   (if (and (<= 0 (long m)) (<= 0 (long n)))
     (let [idx-fact (index-factory fact)
           column? (= :column (:layout options))]
       (let-release [idx (if (compatible? idx-fact idx)
                           (view idx)
                           (transfer idx-fact idx))
                     idx-b (if (compatible? idx-fact idx-b)
                             (view idx-b)
                             (transfer idx-fact idx-b))
                     idx-e (if (compatible? idx-fact idx-e)
                             (view idx-e)
                             (transfer idx-fact idx-e))
                     res (create-ge-csr (factory fact) m n idx idx-b idx-e
                                        column? true)]
         (when-not (and (< (amax idx) (max 1 (long (if column? m n))))
                        (or (zero? (dim idx)) (< (amax idx-b) (dim idx)))
                        (<= (amax idx-e) (dim idx)))
           (dragan-says-ex "Sparse index outside of bounds."
                           {:requested (amax idx) :available (dim (entries res))}))
         (when nz (transfer! nz (entries res)))
         res))
     (dragan-says-ex "Compressed sparse matrix cannot have a negative dimension." {:m m :n n})))
  ([fact m n idx idx-be nz options]
   (csr fact m n idx (pop idx-be) (rest idx-be) nz options))
  ([fact m n idx idx-be options]
   (csr fact m n idx (pop idx-be) (rest idx-be) nil options))
  ([fact m n source options]
   (if (csr? source)
     (csr fact m n (columns source) (indexb source) (indexe source) (entries source) options)
     (if (nil? source)
       (let-release [idx-be (repeat (if (= :column (:layout options)) n m) 0)]
         (csr fact m n [] idx-be idx-be [] options))
       (let [[_ ptrs idx vals] (seq-to-csr source)]
         (csr fact m n idx (pop ptrs) (rest ptrs) vals options)))))
  ([fact m n arg]
   (if (map? arg)
     (csr fact m n nil arg)
     (csr fact m n arg nil)))
  ([arg0 arg1 arg2]
   (if (csr? arg0)
     (create-ge-csr (factory arg0) arg0 arg1 (= true (:index arg2)))
     (csr arg0 arg1 arg2 nil nil)))
  ([fact a]
   (let-release [res (transfer (factory fact) a)]
     (if (csr? res)
       res
       (dragan-says-ex "This is not a valid source for CSR matrices.")))))
