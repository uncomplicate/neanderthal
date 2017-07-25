;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.cublock
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Mappable mmap unmap
                           wrap-float wrap-double wrap-int wrap-long]]
             [utils :refer [with-check]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Foldable Applicative]]
            [uncomplicate.clojurecuda
             [protocols :refer :all]
             [core :refer :all]]
            [uncomplicate.neanderthal
             [math :refer [ceil]]
             [core :refer [transfer! copy! vctr ge]]
             [real :refer [entry]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dense-rows dense-cols dense-dias]]
             [printing :refer [print-vector print-ge print-uplo]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.host
             [fluokitten :refer [vector-op matrix-op vector-pure matrix-pure]]
             [buffer-block :refer [real-block-vector real-ge-matrix real-tr-matrix]]]
            [uncomplicate.neanderthal.internal.device.clblock :as clblock])
  (:import [clojure.lang IFn IFn$L IFn$LD IFn$LLD ExceptionInfo]
           [jcuda.jcublas JCublas2 cublasStatus]
           [uncomplicate.neanderthal.internal.api DataAccessor DenseMatrix Vector DenseVector
            RealVector Matrix RealMatrix GEMatrix TRMatrix RealChangeable RealOrderNavigator
            UploNavigator StripeNavigator DenseMatrix]
           [uncomplicate.neanderthal.internal.host.buffer_block RealBlockVector IntegerBlockVector
            RealGEMatrix RealTRMatrix]
           [uncomplicate.neanderthal.internal.device.clblock CLBlockVector CLGEMatrix CLTRMatrix]))

(def ^{:private true :const true} INEFFICIENT_STRIDE_MSG
  "This operation would be inefficient when stride is not 1.")

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer. Please use transfer! of map-memory to be reminded of that.")

;; ================== Declarations ============================================

(declare cu-block-vector)
(declare cu-ge-matrix)
(declare cu-tr-matrix)

;; ================== Accessors ================================================

(defprotocol CUAccessor
  ;;TODO (get-stream [this])
  (offset [this buf ofst]))

(deftype TypedCUAccessor [ctx et ^long w wrap-fn]
  DataAccessor
  (entryType [_]
    et)
  (entryWidth [_]
    w)
  (count [_ b]
    (quot (long (size b)) w))
  (createDataSource [_ n]
    (mem-alloc (* w (max 1 (long n)))))
  (initialize [_ buf]
    (memset! buf 0)
    buf)
  (wrapPrim [_ s]
    (wrap-fn s))
  CUAccessor
  (offset [_ buf-ptr ofst]
    (if (= 0 (long ofst))
      buf-ptr
      (with-offset buf-ptr (* (long ofst) w))))
  DataAccessorProvider
  (data-accessor [this]
    this)
  Contextual
  (cu-context [_]
    ctx)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or
       (identical? this o)
       (and (instance? TypedCUAccessor da)
            (= et (.et ^TypedCUAccessor da)) (= ctx (.ctx ^TypedCUAccessor da)))
       (= ctx o)
       (= et o)))))

(defn cu-float-accessor [ctx]
  (->TypedCUAccessor ctx Float/TYPE Float/BYTES wrap-float))

(defn cu-double-accessor [ctx]
  (->TypedCUAccessor ctx Double/TYPE Double/BYTES wrap-double))

(defn cu-int-accessor [ctx]
  (->TypedCUAccessor ctx Integer/TYPE Integer/BYTES wrap-int))

(defn cu-long-accessor [ctx]
  (->TypedCUAccessor ctx Long/TYPE Long/BYTES wrap-long))

;; ================ CUDA memory transfer ======================================

(defn cublas-error [^long err-code details]
  (let [err (cublasStatus/stringFor err-code)]
    (ex-info (format "cuBLAS error: %s." err)
             {:name err :code err-code :type :cublas-error :details details})))

(defn get-vector! [^DenseVector cu ^DenseVector host]
  (if (< 0 (.count host))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (JCublas2/cublasGetVector (.dim ^Vector cu) width
                                    (offset da (cu-ptr (.buffer cu)) (.offset cu)) (.stride cu)
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host))
          host)
        (throw (ex-info "You cannot get  incompatible or ill-fitting vector."
                        {:cu (str cu) :host (str host)}))))
    host))

(defn set-vector! [^DenseVector host ^DenseVector cu]
  (if (< 0 (.count cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (JCublas2/cublasSetVector (.dim ^Vector cu) width
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host)
                                    (offset da (cu-ptr (.buffer cu)) (.offset cu)) (.stride cu))
          cu)
        (throw (ex-info "You cannot set incompatible or ill-fitting vector."
                        {:cu (str cu) :host (str host)}))))
    cu))

(defn get-matrix! [^DenseMatrix cu ^DenseMatrix host]
  (if (< 0 (.count host))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (.order cu) (.order host)))
        (with-check cublas-error
          (JCublas2/cublasGetMatrix (.sd cu) (.fd cu) width
                                    (offset da (cu-ptr (.buffer cu)) (.offset cu)) (.stride cu)
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host))
          host)
        (throw (ex-info "You cannot get an incompatible or ill-fitting matrix."
                        {:cu (str cu) :host (str host)}))))
    host))

(defn set-matrix! [^DenseMatrix host ^DenseMatrix cu]
  (if (< 0 (.count cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (.order cu) (.order host)))
        (with-check cublas-error
          (JCublas2/cublasSetMatrix (.sd cu) (.fd cu) width
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host)
                                    (offset da (cu-ptr (.buffer cu)) (.offset cu)) (.stride cu))
          cu)
        (throw (ex-info "You cannot set an incompatible or ill-fitting matrix."
                        {:cu (str cu) :host (str host)}))))
    cu))

;; =============================================================================

(deftype CUBlockVector [fact ^DataAccessor da eng master buf ^long n ^long ofst ^long strd]
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
    (format "#CUBlockVector[%s, n:%d, offset:%d stride:%d]" (.entryType da) n ofst strd))
  Releaseable
  (release [_]
    (when (compare-and-set! master true false)
      (release @buf)
      (reset! buf nil))
    true)
  Container
  (raw [_]
    (cu-block-vector fact n))
  (raw [_ fact]
    (create-vector fact n false))
  (zero [x]
    (zero x fact))
  (zero [_ fact]
    (create-vector fact n true))
  (host [x]
    (let-release [res (raw x (native-factory fact))]
      (get-vector! x res)))
  (native [x]
    (host x))
  DenseContainer
  (view-vctr [_]
    (cu-block-vector fact false buf n ofst strd))
  (view-vctr [_ stride-mult]
    (cu-block-vector fact false buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (cu-ge-matrix fact false buf n 1 ofst n COLUMN_MAJOR))
  (view-ge [x stride-mult]
    (view-ge (view-ge x) stride-mult))
  (view-tr [x uplo diag]
    (view-tr (view-ge x) uplo diag))
  MemoryContext
  (fully-packed? [_]
    (= 1 strd))
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^Vector y)))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  DenseVector
  (buffer [_]
    @buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (count [_]
    n)
  IFn
  (invoke [x i]
    (.entry x i))
  (invoke [x]
    n)
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn$L
  (invokePrim [x]
    n)
  RealChangeable
  (set [x val]
    (set-all eng val x)
    x)
  (set [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [_ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (alter [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  RealVector
  (dim [_]
    n)
  (entry [_ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (cu-block-vector fact (atom false) buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (cu-block-vector fact 0))
  Foldable
  (fold [x]
    (sum eng x)))

(defn cu-block-vector
  ([fact ^Boolean master buf-atom n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da @buf-atom)))
       (->CUBlockVector fact da (vector-engine fact) (atom master) buf-atom n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da @buf-atom)})))))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (cu-block-vector fact true (atom buf) n 0 1))))

(extend CUBlockVector
  Applicative
  {:pure vector-pure}
  Magma
  {:op (constantly vector-op)})

(defmethod print-method CUBlockVector
  [^CUBlockVector x ^java.io.Writer w]
  (.write w (str x))
  (when (and (< 0 (.dim x)) (.buffer x))
    (with-release [host-x (host x)]
      (print-vector w host-x))))

(defmethod transfer! [CUBlockVector CUBlockVector]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUBlockVector RealBlockVector]
  [source destination]
  (get-vector! source destination))

(defmethod transfer! [RealBlockVector CUBlockVector]
  [source destination]
  (set-vector! source destination))

(defmethod transfer! [CUBlockVector Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUBlockVector]
  [source destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

;; ================== CUDA Matrix ============================================

(deftype CUGEMatrix [^RealOrderNavigator navigator fact ^DataAccessor da eng master buf
                     ^long m ^long n ^long ofst ^long ld ^long sd ^long fd ^long ord]
  Object
  (hashCode [a]
    (-> (hash :CUGEMatrix) (hash-combine m) (hash-combine n)
        (hash-combine (nrm2 eng (.stripe navigator a 0)))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? CUGEMatrix b) (compatible? a b) (fits? a b))
      (equals-block eng a b)
      :default false))
  (toString [_]
    (format "#CUGEMatrix[%s, mxn:%dx%d, order%s, offset:%d, ld:%d]"
            (.entryType da) m n (dec-property ord) ofst ld))
  Releaseable
  (release [_]
    (when (compare-and-set! master true false)
      (release @buf)
      (reset! buf nil))
    true)
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (cu-ge-matrix fact m n ord))
  (raw [_ fact]
    (create-ge fact m n ord false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (create-ge fact m n ord true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (get-matrix! a res)))
  (native [a]
    (host a))
  DenseContainer
  (view-vctr [_]
    (if (= ld sd)
      (cu-block-vector fact false buf (* m n) ofst 1)
      (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:ld ld :sd sd}))))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [_]
    (cu-ge-matrix fact false buf m n ofst ld ord))
  (view-ge [_ stride-mult]
    (let [shrinked (ceil (/ fd (long stride-mult)))]
      (cu-ge-matrix fact false buf (.sd navigator sd shrinked) (.fd navigator sd shrinked)
                    ofst (* ld (long stride-mult)) ord)))
  (view-tr [_ uplo diag]
    (cu-tr-matrix fact false buf (min m n) ofst ld ord uplo diag))
  Navigable
  (order-navigator [_]
    navigator)
  MemoryContext
  (fully-packed? [_]
    (= sd ld))
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= m (.mrows ^GEMatrix b)) (= n (.ncols ^GEMatrix b))))
  (fits-navigation? [_ b]
    (= ord (.order ^DenseMatrix b)))
  GEMatrix
  (buffer [_]
    @buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (order [_]
    ord)
  (count [_]
    (* m n))
  (sd [_]
    sd)
  (fd [_]
    fd)
  IFn$LLD
  (invokePrim [a i j]
    (.entry a i j))
  IFn
  (invoke [a i j]
    (.entry a i j))
  (invoke [a]
    sd)
  IFn$L
  (invokePrim [a]
    sd)
  RealChangeable
  (isAllowed [a i j]
    true)
  (set [a val]
    (set-all eng val a)
    a)
  (set [_ _ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (alter [a _ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (cu-block-vector fact false buf n (.index navigator ofst ld i 0) (if (= ROW_MAJOR ord) 1 ld)))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (cu-block-vector fact false buf m (.index navigator ofst ld 0 j) (if (= COLUMN_MAJOR ord) 1 ld)))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cu-block-vector fact false buf (min m n) ofst (inc ld)))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (cu-ge-matrix fact false buf k l (.index navigator ofst ld i j) ld ord))
  (transpose [a]
    (cu-ge-matrix fact false buf n m ofst ld (flip-layout ord)))
  Monoid
  (id [a]
    (cu-ge-matrix fact 0 0)))

(defn cu-ge-matrix
  ([fact master buf-atom m n ofst ld ord]
   (let [^RealOrderNavigator navigator (if (= COLUMN_MAJOR ord) col-navigator row-navigator)]
     (->CUGEMatrix (if (= COLUMN_MAJOR ord) col-navigator row-navigator) fact (data-accessor fact)
                   (ge-engine fact) (atom master) buf-atom m n ofst (max (long ld) (.sd navigator m n))
                   (.sd navigator m n) (.fd navigator m n) ord)))
  ([fact ^long m ^long n ord]
   (let-release [buf (.createDataSource (data-accessor fact) (* m n))]
     (cu-ge-matrix fact true (atom buf) m n 0 0 ord)))
  ([fact ^long m ^long n]
   (cu-ge-matrix fact m n DEFAULT_ORDER)))

(extend CUGEMatrix
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defmethod print-method CUGEMatrix [^CUGEMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.count a)) (.buffer a))
    (with-release [host-a (host a)]
      (print-ge w host-a))))

(defmethod transfer! [CUGEMatrix CUGEMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUGEMatrix RealGEMatrix]
  [source destination]
  (if (= (.order ^CUGEMatrix source) (.order ^RealGEMatrix destination))
    (get-matrix! source destination)
    (with-release [h (host source)]
      (copy! (get-matrix! source h) destination))))

(defmethod transfer! [RealGEMatrix CUGEMatrix]
  [source destination]
  (if (= (.order ^RealGEMatrix source) (.order ^CUGEMatrix destination))
    (set-matrix! source destination)
    (with-release [h (raw destination (factory host))]
      (set-matrix! (copy! source h) destination))))

(defmethod transfer! [CUGEMatrix Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUGEMatrix]
  [source destination]
  (with-release [h (raw destination (native-factory destination))]
    (transfer! (transfer! source h) destination)))

;; ============ CUDA Triangular Matrix =======================================

(deftype CUTRMatrix [^RealOrderNavigator navigator ^UploNavigator uplo-nav ^StripeNavigator stripe-nav
                     fact ^DataAccessor da eng master buf ^long n ^long ofst ^long ld ^long ord
                     ^long fuplo ^long fdiag]
  Object
  (hashCode [this]
    (-> (hash :CUTRMatrix) (hash-combine n) (hash-combine (nrm2 eng (.stripe navigator this 0)))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? CUTRMatrix b) (compatible? da b) (fits? a b))
      (equals-block eng a b)
      :default false))
  (toString [a]
    (format "#CUTRMatrix[%s, mxn:%dx%d, order%s, uplo%s, diag%s, offset:%d, ld:%d]"
            (.entryType da) n n (dec-property ord) (dec-property fuplo) (dec-property fdiag) ofst ld ))
  Releaseable
  (release [_]
    (when (compare-and-set! master true false)
      (release @buf)
      (reset! buf nil))
    true)
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (cu-tr-matrix fact n ord fuplo fdiag))
  (raw [_ fact]
    (create-tr fact n ord fuplo fdiag false))
  (zero [_]
    (zero _ fact))
  (zero [_ fact]
    (create-tr fact n ord fuplo fdiag true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (get-matrix! a res)))
  (native [a]
    (host a))
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (cu-ge-matrix fact false buf n n ofst ld ord))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-tr [_ uplo diag]
    (cu-tr-matrix fact false buf n ofst ld ord uplo diag))
  Navigable
  (order-navigator [_]
    navigator)
  (stripe-navigator [_]
    stripe-nav)
  (uplo-navigator [_]
    uplo-nav)
  MemoryContext
  (fully-packed? [_]
    false)
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= n (.mrows ^TRMatrix b)) (= fuplo (.uplo ^TRMatrix b)) (= fdiag (.diag ^TRMatrix b))))
  (fits-navigation? [_ b]
    (and (= ord (.order ^DenseMatrix b))
         (or (not (instance? TRMatrix b)) (= fuplo (.uplo ^TRMatrix b))) (= fdiag (.diag ^TRMatrix b))))
  Monoid
  (id [a]
    (cu-tr-matrix fact 0))
  TRMatrix
  (buffer [_]
    @buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (count [_]
    (* n n))
  (uplo [_]
    fuplo)
  (diag [_]
    fdiag)
  (order [_]
    ord)
  (sd [_]
    n)
  (fd [_]
    n)
  IFn$LLD
  (invokePrim [a i j]
    (.entry a i j))
  IFn
  (invoke [a i j]
    (.entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (= 2 (.defaultEntry uplo-nav i j)))
  (set [a val]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (set [a i j val]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (alter [a _ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (let [start (.rowStart uplo-nav n i)]
      (cu-block-vector fact false buf (- (.rowEnd uplo-nav n i) start)
                       (.index navigator ofst ld i start) (if (= ROW_MAJOR ord) 1 ld))))
  (col [a j]
    (let [start (.colStart uplo-nav n j)]
      (cu-block-vector fact false buf (- (.colEnd uplo-nav n j) start)
                       (.index navigator ofst ld start j) (if (= COLUMN_MAJOR ord) 1 ld))))
  (dia [a]
    (cu-block-vector fact false buf n ofst (inc ld)))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (cu-tr-matrix fact false buf k (.index navigator ofst ld i j) ld ord fuplo fdiag)
      (throw (ex-info "You cannot use regions outside the triangle in TR submatrix"
                      {:a (str a) :i i :j j :k k :l l}))))
  (transpose [a]
    (cu-tr-matrix fact false buf n ofst ld (flip-layout ord) (flip-uplo fuplo) fdiag)))

(extend CUTRMatrix
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defn cu-tr-matrix
  ([fact master buf-atom n ofst ld ord uplo diag]
   (let [unit (= DIAG_UNIT diag)
         lower (= LOWER uplo)
         column (= COLUMN_MAJOR ord)
         bottom (if lower column (not column))
         order-nav (if column col-navigator row-navigator)
         uplo-nav (if lower
                    (if unit unit-lower-nav non-unit-lower-nav)
                    (if unit unit-upper-nav non-unit-upper-nav))
         stripe-nav (if bottom
                      (if unit unit-bottom-navigator non-unit-bottom-navigator)
                      (if unit unit-top-navigator non-unit-top-navigator))]
     (->CUTRMatrix order-nav uplo-nav stripe-nav fact (data-accessor fact) (tr-engine fact)
                   (atom master) buf-atom n ofst (max (long ld) (long n)) ord uplo diag)))
  ([fact n ord uplo diag]
   (let-release [buf (.createDataSource (data-accessor fact) (* (long n) (long n)))]
     (cu-tr-matrix fact true (atom buf) n 0 n ord uplo diag)))
  ([fact n]
   (cu-tr-matrix fact n DEFAULT_ORDER DEFAULT_UPLO DEFAULT_DIAG)))

(defmethod print-method CUTRMatrix [^CUTRMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.count a)) (.buffer a))
    (with-release [host-a (host a)]
      (print-uplo w host-a))))

(defmethod transfer! [CUTRMatrix CUTRMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUTRMatrix RealTRMatrix]
  [^CUTRMatrix source ^RealTRMatrix destination]
  (if (= (.order source) (.order destination))
    (get-matrix! source destination)
    (with-release [h (host source)]
      (copy! (get-matrix! source h) destination))))

(defmethod transfer! [RealTRMatrix CUTRMatrix]
  [source destination]
  (if (= (.order ^RealTRMatrix source) (.order ^CUTRMatrix destination))
    (set-matrix! source destination)
    (with-release [h (raw destination (factory host))]
      (set-matrix! (copy! source h) destination))))

(defmethod transfer! [CUTRMatrix Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUTRMatrix]
  [source destination]
  (with-release [h (raw destination (native-factory destination))]
    (transfer! (transfer! source h) destination)))

;; =============== Transfer preferences ========================================

(prefer-method transfer! [CUBlockVector Object] [Object CLBlockVector])
(prefer-method transfer! [CUGEMatrix Object] [Object CLGEMatrix])
(prefer-method transfer! [CUTRMatrix Object] [Object CLGEMatrix])
(prefer-method transfer! [CLBlockVector Object] [Object CUBlockVector])
(prefer-method transfer! [CLGEMatrix Object] [Object CUGEMatrix])
(prefer-method transfer! [CLTRMatrix Object] [Object CUTRMatrix])
