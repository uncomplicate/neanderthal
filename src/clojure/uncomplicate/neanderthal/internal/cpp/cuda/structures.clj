;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.cuda.structures
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info Viewable view size]]
             [utils :refer [with-check dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Foldable Applicative extract]]
            [uncomplicate.clojure-cpp :refer [pointer fill! float-pointer double-pointer long-pointer
                                              int-pointer short-pointer byte-pointer null?
                                              PointerCreator capacity! byte-pointer]]
            [uncomplicate.clojurecuda.core :refer :all :exclude [device]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! vctr ge dim]]
             [block :refer [stride column?]]
             [real :refer [entry]]
             [math :refer [ceil]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dense-rows dense-cols dense-dias region-dias require-trf]]
             [printing :refer [print-vector print-ge print-uplo]]
             [navigation :refer :all]
             [fluokitten :refer [vector-op matrix-op]]]
            [uncomplicate.neanderthal.internal.cpp.structures
             :refer [extend-base extend-vector extend-matrix extend-ge-matrix extend-uplo-matrix
                     extend-trf block-vector ge-matrix uplo-matrix]]
            [uncomplicate.neanderthal.internal.cpp.cuda.constants :refer :all]
            [uncomplicate.neanderthal.internal.device.common :refer [device-matrix-equals]]
            [uncomplicate.neanderthal.internal.device.clblock :as clblock])
  (:import [clojure.lang IFn IFn$L IFn$LD IFn$LDD IFn$LLD]
           org.bytedeco.cuda.global.cublas
           [uncomplicate.neanderthal.internal.api Block Matrix DataAccessor RealNativeVector
            IntegerNativeVector RealNativeMatrix FullStorage Region Default GEMatrix CUMatrix
            UploMatrix
            IntegerVector LayoutNavigator MatrixImplementation RealAccessor IntegerAccessor
            CUVector CUMatrix CLVector CLMatrix Changeable]))

(declare cu-block-vector cu-ge-matrix cu-uplo-matrix)

(def ^{:private true :const true} INEFFICIENT_STRIDE_MSG
  "This operation would be inefficient when stride is not 1.")

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! to be reminded of that.")

(def ^{:private true :const true} UNAVAILABLE_CUDA_MSG
  "This operation is not available in CUDA (yet).")

(defprotocol ContextProvider
  (get-context [this]))

(defmacro def-accessor-type [name accessor-interface pointer-class entry-class pointer cast cast-get]
  `(deftype ~name [ctx# hstream# construct# destruct#]
     DataAccessor
     (entryType [_#]
       (. ~entry-class TYPE))
     (entryWidth [_#]
       (. ~entry-class BYTES))
     (count [_# p#]
       (size p#))
     (createDataSource [_# n#]
       (let [n# (max 1 n#)]
         (capacity! (~pointer (construct# (* (. ~entry-class BYTES) n#))) n#)))
     (initialize [_# p#]
       (memset! p# (byte 0) hstream#))
     (wrapPrim [_# v#]
       (pointer (~cast v#)))
     (castPrim [_# v#]
       (~cast v#))
     FlowProvider
     (flow [_]
       hstream#)
     ContextProvider
     (get-context [_]
       (extract ctx#))
     DataAccessorProvider
     (data-accessor [this#]
       this#)
     Destructor
     (destruct [_# p#]
       (if-not (null? p#)
         (destruct# p#)
         p#))
     PointerCreator
     (pointer* [_#]
       (~pointer nil))
     MemoryContext
     (compatible? [this# o#]
       (let [da# (data-accessor o#)]
         (or (identical? this# da#)
             (and (instance? ~name da#) (= ctx# (get-context da#))))))
     (device [_#]
       :cuda)))

(def-accessor-type DoublePointerAccessor RealAccessor DoublePointer Double double-pointer double double)
(def-accessor-type FloatPointerAccessor RealAccessor FloatPointer Float float-pointer float float)
(def-accessor-type LongPointerAccessor IntegerAccessor LongPointer Long long-pointer long long)
(def-accessor-type IntPointerAccessor IntegerAccessor IntPointer Integer int-pointer int int)
(def-accessor-type ShortPointerAccessor IntegerAccessor ShortPointer Short short-pointer short long)
(def-accessor-type BytePointerAccessor IntegerAccessor BytePointer Byte byte-pointer byte long)

;; ================ CUDA memory transfer ======================================

(defn cublas-error [^long err-code details]
  (let [err (cublas-status-codes err-code)]
    (ex-info (format "cuBLAS error: %s." err)
             {:name err :code err-code :type :cublas-error :details details})))

(defn get-vector! [cu host]
  (if (< 0 (dim host))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (cublas/cublasGetVector_64 (dim cu) width
                                     (byte-pointer (extract cu)) (stride cu)
                                     (byte-pointer (extract host)) (stride host))
          host)
        (throw (ex-info "You cannot get incompatible or ill-fitting vector."
                        {:cu (info cu) :host (info host)}))))
    host))

(defn set-vector! [host cu]
  (if (< 0 (dim cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (cublas/cublasSetVector_64 (dim cu) width
                                     (byte-pointer (extract host)) (stride host)
                                     (byte-pointer (extract cu)) (stride cu))
          cu)
        (throw (ex-info "You cannot set incompatible or ill-fitting vector."
                        {:cu (info cu) :host (info host)}))))
    cu))

(defn get-matrix! [cu host]
  (if (< 0 (dim host))
    (let [da (data-accessor cu)
          width (.entryWidth da)
          stor (full-storage cu)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (navigator cu) (navigator host)))
        (with-check cublas-error
          (cublas/cublasGetMatrix_64 (.sd stor) (.fd stor) width
                                     (byte-pointer (extract cu)) (.ld stor)
                                     (byte-pointer (extract host)) (stride host))
          host)
        (throw (ex-info "You cannot get an incompatible or ill-fitting matrix."
                        {:cu (info cu) :host (info host)}))))
    host))

(defn set-matrix! [host cu]
  (if (< 0 (dim cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)
          stor (full-storage cu)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (navigator cu) (navigator host)))
        (with-check cublas-error
          (cublas/cublasSetMatrix_64 (.sd stor) (.fd stor) width
                                     (byte-pointer (extract host)) (stride host)
                                     (byte-pointer (extract cu)) (.ld stor))
          cu)
        (throw (ex-info "You cannot set an incompatible or ill-fitting matrix."
                        {:cu (info cu) :host (info host)}))))
    cu))

;; =========================== CUDA Vector ==================================================

(defmacro extend-cu-block-vector [name block-vector]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~block-vector (.-fact this#) (.-n this#)))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) false)))
     (zero
       ([this#]
        (create-vector (.-fact this#) (.-n this#) true))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) true)))
     (host [this#]
       (let-release [res# (raw this# (native-factory this#))]
         (get-vector! this# res#)))
     (native [this#]
       (host this#))))

(deftype CUBlockVector [fact ^DataAccessor da eng master buf-ptr ^long n ^long strd]
  Object
  (hashCode [x]
    (-> (hash :CUBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? CUBlockVector y) (compatible? x y) (fits? x y))
      (equals-block eng x y)
      :default false))
  (toString [this]
    (format "#CUBlockVector[%s, n:%d, stride:%d]" (.entryType da) n strd))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (invoke [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (invoke [x]
    n)
  Changeable
  (setBoxed [x val]
    (if-not (Double/isNaN val)
      (set-all eng val x))
    x)
  (setBoxed [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUVector
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
  (dim [_]
    n)
  (boxedEntry [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (subvector [_ k l]
    (cu-block-vector fact false buf-ptr l (* k strd) strd))
  Foldable
  (fold [x]
    (sum eng x))
  Magma
  (op [x]
    vector-op))

(extend-base CUBlockVector)
(extend-vector CUBlockVector cu-block-vector cu-ge-matrix)
(extend-cu-block-vector CUBlockVector cu-block-vector)

(defmethod print-method CUBlockVector
  [^CUBlockVector x ^java.io.Writer w]
  (.write w (str x))
  (when (and (< 0 (.dim x)) (extract x) (get-context (.da x)))
    (with-release [host-x (host x)]
      (print-vector w host-x))))

(def cu-block-vector (partial block-vector ->CUBlockVector))

(defmethod transfer! [CUBlockVector CUBlockVector]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUBlockVector RealNativeVector]
  [source destination]
  (get-vector! source destination))

(defmethod transfer! [RealNativeVector CUBlockVector]
  [source destination]
  (set-vector! source destination))

(defmethod transfer! [CUBlockVector IntegerNativeVector]
  [source destination]
  (get-vector! source destination))

(defmethod transfer! [IntegerNativeVector CUBlockVector]
  [source destination]
  (set-vector! source destination))

(defmethod transfer! [CUVector Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUVector]
  [source destination]
  (with-release [h (raw destination (native-factory destination))]
    (transfer! (transfer! source h) destination)))

;; ================== CUDA Matrix ============================================

(defmacro extend-cu-ge-matrix [name ge-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~ge-matrix (.-fact this#) (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? this#) false)))
     (zero
       ([this#]
        (create-ge (.-fact this#) (.-m this#) (.-n this#) (column? this#) true))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? this#) true)))
     (host [this#]
       (let-release [res# (raw this# (native-factory this#))]
         (get-matrix! this# res#)))
     (native [this#]
       (host this#))))

(deftype CUGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg fact ^DataAccessor da eng
                     master buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :CUGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [_]
    (format "#CUGEMatrix[%s, mxn:%dx%d, layout%s]" (.entryType da) m n (dec-property (.layout nav))))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  IFn$L
  (invokePrim [a]
    (.fd stor))
  IFn
  (invoke [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (invoke [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (invoke [a]
    (.fd stor))
  Changeable
  (isAllowed [a i j]
    true)
  (setBoxed [a val]
    (set-all eng val a))
  (setBoxed [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    (.isGapless stor))
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (boxedEntry [a i j]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (row [a i]
    (cu-block-vector fact false buf-ptr n (.index nav stor i 0) (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (cu-block-vector fact false buf-ptr m (.index nav stor 0 j) (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cu-block-vector fact false buf-ptr (min m n) (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (cu-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (inc (.ld stor)))
      (cu-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (cu-ge-matrix fact false buf-ptr k l (.index nav stor i j)
                  nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (cu-ge-matrix fact false buf-ptr n m (flip nav) stor (flip reg)))
  Magma
  (op [_]
    matrix-op))

(defmethod print-method CUGEMatrix [^CUGEMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a))  (extract a) (get-context (.da a)))
    (with-release [host-a (host a)]
      (print-ge w host-a))))

(extend-base CUGEMatrix)
(extend-matrix CUGEMatrix)
(extend-ge-matrix CUGEMatrix cu-block-vector cu-ge-matrix cu-uplo-matrix)
(extend-cu-ge-matrix CUGEMatrix cu-ge-matrix)

(def cu-ge-matrix (partial ge-matrix ->CUGEMatrix))

;; ============ CUDA Uplo Matrix =======================================

(defmacro extend-cu-uplo-matrix [name uplo-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~uplo-matrix (.-fact this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)
         (.-matrix-type this#) (.-default this#) (.-eng this#)))
       ([this# fact#]
        (create-uplo (factory fact#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) false)))
     (zero
       ([this#]
        (create-uplo (.-fact this#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true))
       ([this# fact#]
        (create-uplo (factory fact#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true)))
     (host [this#]
       (let-release [res# (raw this# (native-factory this#))]
         (get-matrix! this# res#)
         res#))
     (native [this#]
       (host this#))))

(deftype CUUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^Default default
                       fact ^DataAccessor da eng matrix-type master buf-ptr ^long n]
  Object
  (hashCode [this]
    (-> (hash :CUUploMatrix) (hash-combine n) (hash-combine (nrm2 eng this))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [a]
    (format "#CUUploMatrix[%s, type%s, mxn:%dx%d, layout%s]"
            (.entryType da) matrix-type n n (dec-property (.layout nav))))
  UploMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tr matrix-type))
  (isSymmetric [_]
    (= :sy matrix-type))
  IFn
  (invoke [a]
    (.fd stor))
  Changeable
  (isAllowed [a i j]
    (.accessible reg i j))
  (setBoxed [x val]
    (set-all eng val x))
  (setBoxed [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    false)
  (dim [_]
    (* n n))
  (mrows [_]
    n)
  (ncols [_]
    n)
  (boxedEntry [this i j]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (row [a i]
    (let [start (.rowStart reg i)]
      (cu-block-vector fact false buf-ptr (- (.rowEnd reg i) start) (.index nav stor i start)
                       (if (.isRowMajor nav) 1 (.ld ^FullStorage stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (cu-block-vector fact false buf-ptr (- (.colEnd reg j) start) (.index nav stor start j)
                       (if (.isColumnMajor nav) 1 (.ld ^FullStorage stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cu-block-vector fact false buf-ptr n (inc (.ld ^FullStorage stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (cu-block-vector fact false buf-ptr (- n k) (.index nav stor 0 k) (inc (.ld ^FullStorage stor)))
        (cu-block-vector fact false buf-ptr (+ n k) (.index nav stor (- k) 0) (inc (.ld ^FullStorage stor))))
      (cu-block-vector fact false buf-ptr 0 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (cu-uplo-matrix fact false buf-ptr k (.index nav stor i j) nav
                      (full-storage (.isColumnMajor nav) k k (.ld ^FullStorage stor))
                      (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (cu-uplo-matrix fact false buf-ptr n (flip nav) stor (flip reg) matrix-type default eng))
  Triangularizable
  (create-trf [a pure]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_CUDA_MSG)
      a))
  (create-ptrf [a]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_CUDA_MSG)
      a))
  Magma
  (op [_]
    matrix-op))

(defmethod print-method CUUploMatrix [^CUUploMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a)) (extract a) (get-context (.da a)))
    (with-release [host-a (host a)]
      (print-uplo w host-a "*"))))

(extend-base CUUploMatrix)
(extend-matrix CUUploMatrix)
(extend-uplo-matrix CUUploMatrix cu-block-vector cu-ge-matrix cu-uplo-matrix)
(extend-cu-uplo-matrix CUUploMatrix cu-uplo-matrix)
(extend-trf CUUploMatrix)

(def cu-uplo-matrix (partial uplo-matrix ->CUUploMatrix))

(defmethod transfer! [CUGEMatrix CUGEMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUUploMatrix CUUploMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUUploMatrix CUGEMatrix]
  [source destination]
  (let [reg (region source)]
    (copy! source (view-tr destination (.isLower reg) (.isDiagUnit reg)))))

(defmethod transfer! [CUGEMatrix CUUploMatrix]
  [source destination]
  (let [reg (region destination)]
    (copy! (view-tr source (.isLower reg) (.isDiagUnit reg)) destination)))

(defmethod transfer! [CUMatrix RealNativeMatrix]
  [source destination]
  (if (= (navigator source) (navigator destination))
    (get-matrix! source destination)
    (with-release [h (host source)]
      (copy! (get-matrix! source h) destination))))

(defmethod transfer! [RealNativeMatrix CUMatrix]
  [source destination]
  (if (= (navigator source) (navigator destination))
    (set-matrix! source destination)
    (with-release [h (raw destination (factory host))]
      (set-matrix! (copy! source h) destination))))

(defmethod transfer! [CUMatrix Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUMatrix]
  [source destination]
  (with-release [h (raw destination (native-factory destination))]
    (transfer! (transfer! source h) destination)))

;; =============== Transfer preferences ========================================

(prefer-method transfer! [CUVector Object] [Object CUVector])
(prefer-method transfer! [CUMatrix Object] [Object CUMatrix])
(prefer-method transfer! [CUVector Object] [Object CLVector])
(prefer-method transfer! [CUMatrix Object] [Object CLMatrix])
(prefer-method transfer! [CLVector Object] [Object CUVector])
(prefer-method transfer! [CLMatrix Object] [Object CUMatrix])
