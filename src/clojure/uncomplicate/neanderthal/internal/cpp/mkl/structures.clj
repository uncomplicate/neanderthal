;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.structures
  (:require
   [uncomplicate.commons
    [core :refer [Releaseable release let-release with-release Info info Viewable view]]
    [utils :refer [dragan-says-ex with-check]]]
   [uncomplicate.fluokitten.protocols
    :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative fold]]
   [uncomplicate.clojure-cpp :refer [get-entry fill! int-pointer pointer null? capacity]]
   [uncomplicate.neanderthal
    [core :refer [dim transfer! mrows ncols subvector]]
    [block :refer [entry-type offset stride buffer column?]]
    [integer :refer [entry]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [navigation :refer :all]
    [common :refer [dense-rows dense-cols]]
    [printing :refer [print-vector]]]
   [uncomplicate.neanderthal.internal.host.fluokitten :refer :all]
   [uncomplicate.neanderthal.internal.cpp.structures
    :refer [CompressedSparse entries indices vector-seq csr-engine CSR indexb indexe columns
            real-block-vector integer-block-vector cs-vector]]
   [uncomplicate.neanderthal.internal.cpp.mkl
    [constants :refer [mkl-sparse-request]]
    [core :refer [create-csr matrix-descr export-csr sparse-error sparse-matrix]]])
  (:import [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL IFn$LLLL]
           [org.bytedeco.javacpp IntPointer]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$sparse_matrix mkl_rt$matrix_descr]
           [uncomplicate.neanderthal.internal.api Block Matrix DataAccessor RealNativeMatrix
            IntegerVector LayoutNavigator MatrixImplementation RealAccessor IntegerAccessor]
           uncomplicate.neanderthal.internal.cpp.structures.CSVector))

(declare csr-matrix)

;; ======================= Compressed Sparse Matrix ======================================

(deftype CSRMatrix [^LayoutNavigator nav fact eng spm desc
                    ^Block nzx ^IntegerVector indx ^IntegerVector pb ^IntegerVector pe
                    ^long m ^long n]
  Object
  (hashCode [_]
    (-> (hash :CSRMatrix) (hash-combine nzx) (hash-combine indx) (hash-combine pb) (hash-combine pe)))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (instance? CSRMatrix b)
      (and (= m (mrows b)) (= n (ncols b))
           (= nzx (entries b)) (= indx (indices b)) (= pb (.pb ^CSRMatrix b)) (= pe (.pe ^CSRMatrix b)))
      :default false))
  (toString [_]
    (format "#CSRMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type (data-accessor nzx)) m n (dec-property (.layout nav))))
  MatrixImplementation
  (matrixType [_]
    :cs)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Info
  (info [x]
    {:entry-type (.entryType (data-accessor nzx))
     :class (class x)
     :device (info nzx :device)
     :dim n
     :engine (info eng)})
  (info [x info-type]
    (case info-type
      :entry-type (.entryType (data-accessor nzx))
      :class (class x)
      :device (info nzx :device)
      :dim n
      :engine (info eng)
      nil))
  Releaseable
  (release [_]
    (release spm)
    (release desc)
    (release nzx)
    (release indx)
    (release pb)
    (release pe)
    true)
  Seqable
  (seq [x]
    (vector-seq nzx))
  MemoryContext
  (compatible? [_ b]
    (and (compatible? nzx (entries b))))
  (fits? [_ b]
    (and (instance? CSRMatrix b) (fits? nzx (entries b)) (fits? indx (indices b))
         (fits? pe (.pe ^CSRMatrix b)) (fits? pb (.pb ^CSRMatrix b)))) ;; TODO region?
  (device [_]
    (device nzx))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor fact))
  Navigable
  (navigator [_]
    nav)
  Container
  (raw [a]
    (raw a fact))
  (raw [_ fact]
    (csr-matrix (factory fact) m n (view indx) (view pb) (view pe) (view desc) false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (csr-matrix (factory fact) m n (view indx) (view pb) (view pe) (view desc) true)) ;;TODO indices on the gpu etc.
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)
      res))
  (native [a]
    a)
  Viewable
  (view [a]
    (csr-matrix m n (view indx) (view pb) (view pe) (view nzx) nav (view desc)))
  DenseContainer
  (view-vctr [_]
    nzx)
  (view-vctr [_ stride-mult]
    (view-vctr nzx stride-mult))
  (view-ge [_]
    (view-ge nzx))
  (view-ge [_ stride-mult]
    (view-ge nzx stride-mult))
  (view-ge [_ m n]
    (view-ge nzx m n))
  Block
  (buffer [_]
    (.buffer nzx))
  (offset [_]
    (.offset nzx))
  (stride [_]
    (.stride nzx))
  (isContiguous [_]
    (.isContiguous ^Block nzx))
  Matrix
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (row [a i]
    (if (.isRowMajor nav)
      (let [j (entry pb i)
            k (- (entry pe i) j)]
        (cs-vector n (subvector indx j k) (subvector nzx j k)))
      (dragan-says-ex "Sparse rows are available only in row-major CSR matrices." {:a (info a)})))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (if (.isColumnMajor nav)
      (let [i (entry pb j)
            k (- (entry pe j) i)]
        (cs-vector n (subvector indx i k) (subvector nzx i k)))
      (dragan-says-ex "Sparse columns are available only in column-major CSR matrices." {:a (info a)})))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (dragan-says-ex "Diagonals of a GE sparse matrix is not available."))
  (dia [a k]
    (dragan-says-ex "Diagonals of a GE sparse matrix is not available."))
  (dias [a]
    (dragan-says-ex "Diagonals of a GE sparse matrix is not available."))
  (submatrix [a i j k l]
    (let [i (long i)
          j (long j)
          k (long k)
          l (long l)
          [ok? b cnt] (if (.isRowMajor nav)
                        [(and (= 0 j) (= n l)) i k]
                        [(and (= 0 i) (= m k)) j l])
          sub-pb (subvector pb b cnt)
          sub-pe (subvector pe b cnt)
          sub-b (entry sub-pb 0)
          sub-cnt (- (entry sub-pe (dec (long cnt))) sub-b)]
      (csr-matrix k l (subvector indx sub-b sub-cnt) sub-pb sub-pe (subvector nzx sub-b sub-cnt) nav desc)))
  (transpose [a]
    (csr-matrix n m (view indx) (view pb) (view pe) (view nzx) (flip nav) (flip desc)))
  CompressedSparse
  (entries [_]
    nzx)
  (indices [_]
    indx)
  CSR
  (columns [_]
    indx)
  (indexb [_]
    pb)
  (indexe [_]
    pe))

(defn ^mkl_rt$sparse_matrix spmat [^CSRMatrix a]
  (.spm a))

(defn ^mkl_rt$matrix_descr descr [^CSRMatrix a]
  (.desc a))

(defn csr-matrix
  ([m n indx pb pe nzx ^LayoutNavigator nav desc]
   (let [fact (factory nzx)
         [sp-m sp-n] (if (.isColumnMajor nav) [n m] [m n])]
     (if (compatible? (index-factory nzx) indx)
       (if (and (<= 0 (dim indx) (dim nzx)) (<= (long sp-m) (dim pe) (dim pb))
                (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx)) (fits? indx nzx)) ;;TODO improve error message. Perhaps automatize (.isColumnmajor part)
         (->CSRMatrix nav fact (csr-engine fact)
                      (create-csr (buffer nzx) 0 (max 1 (long sp-m)) (max 1 (long sp-n))
                                  (buffer pb) (buffer pe) (buffer indx))
                      desc nzx indx pb pe m n)
         (dragan-says-ex "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx})) ;;TODO error message
       (dragan-says-ex "Incompatible index vector." {:required (index-factory nzx) :provided (factory indx)}))))
  ([fact m n indx pb pe nav desc init]
   (let-release [nzx (create-vector fact (dim indx) init)]
     (csr-matrix m n indx pb pe nzx nav desc)))
  ([fact sparse-matrix ^LayoutNavigator nav desc]
   (let-release [indexing (int-pointer 1)
                 m (int-pointer 1)
                 n (int-pointer 1)
                 rows-start (int-pointer nil)
                 rows-end (int-pointer nil)
                 col-idx (int-pointer nil)
                 nz (pointer (data-accessor fact))]
     (export-csr nz sparse-matrix indexing m n rows-start rows-end col-idx)
     (let [[m n] (if (.isColumnMajor nav)
                   [(get-entry n 0) (get-entry m 0)]
                   [(get-entry m 0) (get-entry n 0)])
           idx-fact (index-factory fact)
           nzx (create-vector fact false nz (capacity nz) 0 1)
           idx (create-vector idx-fact false col-idx (capacity col-idx) 0 1)
           pb (create-vector idx-fact false rows-start (capacity rows-start) 0 1)
           pe (create-vector idx-fact false rows-end (capacity rows-end) 0 1)]
       (->CSRMatrix nav fact (csr-engine fact) sparse-matrix desc nzx idx pb pe m n)))))

(defn ge-csr-matrix
  ([fact m n indx pb pe column? init]
   (csr-matrix fact m n indx pb pe (layout-navigator column?) (matrix-descr :ge) init))
  ([fact m n indx pb pe column?]
   (csr-matrix fact m n indx pb pe column? true))
  ([fact sparse-matrix column?]
   (csr-matrix fact sparse-matrix (layout-navigator column?) (matrix-descr :ge))))

(defmethod print-method CSRMatrix [^CSRMatrix x ^java.io.Writer w] ;; TODO transform to nested vectors
  (.write w (format "%s\n%s" (str x) (pr-str (seq (indices x)))))
  (when-not (null? (buffer (entries x)))
    (print-vector w (entries x))))

(defmethod transfer! [CSRMatrix CSRMatrix]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

(defn seq-to-csr [source]
  (if (number? (get-in source [0 0]))
    (seq-to-csr (partition 2 source))
    (reduce (fn [[^long row ptrs idx vals] [ridx rvals]]
              (let [fill (- (count ridx) (count rvals))]
                (if (<= 0 fill)
                  [(inc row)
                   (conj ptrs (+ (long (peek ptrs)) (long (count ridx))))
                   (into idx ridx)
                   (-> vals (into rvals) (into (repeat fill 0.0)))]
                  (dragan-says-ex "Each value of a sparse matrix need its row/col position."
                                  {:row row :idx-count (count ridx) :val-count (count rvals)}))))
            [0 [0] [] []]
            source)))

(defmethod transfer! [clojure.lang.Sequential CSRMatrix]
  [source destination]
  (transfer! (seq-to-csr 3) (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[D") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[F") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[J") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[I") CSRMatrix]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [CSRMatrix (Class/forName "[D")]
  [source destination]
  (transfer! (entries source) destination))

(defmethod transfer! [CSRMatrix (Class/forName "[F")]
  [source destination]
  (transfer! (entries source) destination))

(defmethod transfer! [CSVector CSRMatrix]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

(defmethod transfer! [CSRMatrix CSVector]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

;;TODO handle heterogenous types (float/double...)
#_(defmethod transfer! [RealBlockVector CSMatrix]
  [^RealBlockVector source ^CSMatrix destination]
  (gthr (engine destination) source destination)
  destination)

(defn sparse-transpose ^long [a]
  (if (.isColumnMajor (navigator a))
    mkl_rt/SPARSE_OPERATION_TRANSPOSE
    mkl_rt/SPARSE_OPERATION_NON_TRANSPOSE))

(defn sparse-layout ^long [a]
  (if (.isColumnMajor (navigator a))
    mkl_rt/SPARSE_LAYOUT_COLUMN_MAJOR
    mkl_rt/SPARSE_LAYOUT_ROW_MAJOR))

(defn csr-ge-sp2m
  ([^CSRMatrix a ^CSRMatrix b request]
   (let-release [c (sparse-matrix)]
     (with-check sparse-error
       (mkl_rt/mkl_sparse_sp2m (sparse-transpose a) (descr a) (spmat a)
                               (sparse-transpose b) (descr b) (spmat b)
                               (get mkl-sparse-request request request)
                               c)
       (csr-matrix (factory a) c (layout-navigator true) (matrix-descr :ge)))))
  ([^CSRMatrix a ^CSRMatrix b ^CSRMatrix c request]
   (with-release [indexing (int-pointer 1)
                  m (int-pointer 1)
                  n (int-pointer 1)]
     (with-check sparse-error
       (mkl_rt/mkl_sparse_sp2m (sparse-transpose a) (descr a) (spmat a)
                               (sparse-transpose b) (descr b) (spmat b)
                               (get mkl-sparse-request request request)
                               (spmat c))
       (export-csr (buffer (entries c)) (spmat c) indexing m n
                   (buffer (indexb c)) (buffer (indexe c)) (buffer (columns c))))
     c)))
