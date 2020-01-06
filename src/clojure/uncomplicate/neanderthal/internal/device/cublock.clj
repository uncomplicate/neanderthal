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
             [core :refer [Releaseable release let-release with-release Info info
                           Wrapper extract wrap-float wrap-double wrap-int wrap-long]]
             [utils :refer [with-check dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Foldable Applicative]]
            [uncomplicate.clojurecuda.core :refer :all :exclude [device]]
            [uncomplicate.clojurecuda.internal.protocols :refer [size ptr with-offset]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! vctr ge]]
             [real :refer [entry]]
             [math :refer [ceil]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dense-rows dense-cols dense-dias region-dias require-trf]]
             [printing :refer [print-vector print-ge print-uplo]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.host
             [fluokitten :refer [vector-op matrix-op]]
             [buffer-block :refer [real-block-vector real-ge-matrix real-uplo-matrix]]]
            [uncomplicate.neanderthal.internal.device.common :refer [device-matrix-equals]]
            [uncomplicate.neanderthal.internal.device.clblock :as clblock])
  (:import [clojure.lang IFn IFn$L IFn$LD IFn$LDD IFn$LLD]
           [jcuda.jcublas JCublas2 cublasStatus]
           [uncomplicate.neanderthal.internal.api DataAccessor VectorSpace Vector CLVector CUVector
            Matrix CLMatrix CUMatrix GEMatrix RealChangeable LayoutNavigator RealLayoutNavigator
            Region MatrixImplementation NativeBlock FullStorage RealDefault UploMatrix
            RealNativeVector RealNativeMatrix]
           [uncomplicate.neanderthal.internal.host.buffer_block RealBlockVector RealGEMatrix
            RealUploMatrix]))

(def ^{:private true :const true} INEFFICIENT_STRIDE_MSG
  "This operation would be inefficient when stride is not 1.")

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! to be reminded of that.")

(def ^{:private true :const true} UNAVAILABLE_CUDA_MSG
  "This operation is not available in CUDA (yet).")

;; ================== Declarations ============================================

(declare cu-block-vector cu-ge-matrix cu-uplo-matrix ->CUUploMatrix)

;; ================== Accessors ================================================

(extend-type Object
  Wrapper
  (extract [_]
    nil))

(defprotocol CUAccessor
  (offset [this buf ofst])
  (active? [this]))

(deftype TypedCUAccessor [active ctx hstream et ^long w wrap-fn cast-fn]
  Releaseable
  (release [_]
    (vreset! active false)
    true)
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
    (memset! buf 0 hstream)
    buf)
  (initialize [_ buf v]
    (memset! buf v hstream)
    buf)
  (wrapPrim [_ s]
    (wrap-fn s))
  (castPrim [_ v]
    (cast-fn v))
  FlowProvider
  (flow [_]
    hstream)
  CUAccessor
  (offset [_ buf-ptr ofst]
    (if (= 0 (long ofst))
      buf-ptr
      (with-offset buf-ptr (* (long ofst) w))))
  (active? [_]
    @active)
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or
       (identical? this o) (identical? this da)
       (and (instance? TypedCUAccessor da)
            (= et (.et ^TypedCUAccessor da)) (= ctx (.ctx ^TypedCUAccessor da)))
       (= ctx o)
       (= et o)))))

(defn cu-float-accessor [ctx hstream]
  (->TypedCUAccessor (volatile! true) ctx hstream Float/TYPE Float/BYTES wrap-float float))

(defn cu-double-accessor [ctx hstream]
  (->TypedCUAccessor (volatile! true) ctx hstream Double/TYPE Double/BYTES wrap-double double))

(defn cu-int-accessor [ctx hstream]
  (->TypedCUAccessor (volatile! true) ctx hstream Integer/TYPE Integer/BYTES wrap-int int))

(defn cu-long-accessor [ctx hstream]
  (->TypedCUAccessor (volatile! true) ctx hstream Long/TYPE Long/BYTES wrap-long long))

;; ================ CUDA memory transfer ======================================

(defn cublas-error [^long err-code details]
  (let [err (cublasStatus/stringFor err-code)]
    (ex-info (format "cuBLAS error: %s." err)
             {:name err :code err-code :type :cublas-error :details details})))

(defn get-vector! [^CUVector cu ^RealNativeVector host]
  (if (< 0 (.dim host))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (JCublas2/cublasGetVector (.dim cu) width
                                    (offset da (extract (.buffer cu)) (.offset cu)) (.stride cu)
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host))
          host)
        (throw (ex-info "You cannot get incompatible or ill-fitting vector."
                        {:cu (info cu) :host (info host)}))))
    host))

(defn set-vector! [^RealNativeVector host ^CUVector cu]
  (if (< 0 (.dim cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
        (with-check cublas-error
          (JCublas2/cublasSetVector (.dim ^Vector cu) width
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host)
                                    (offset da (extract (.buffer cu)) (.offset cu)) (.stride cu))
          cu)
        (throw (ex-info "You cannot set incompatible or ill-fitting vector."
                        {:cu (info cu) :host (info host)}))))
    cu))

(defn get-matrix! [^CUMatrix cu ^RealNativeMatrix host]
  (if (< 0 (.dim host))
    (let [da (data-accessor cu)
          width (.entryWidth da)
          stor (full-storage cu)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (navigator cu) (navigator host)))
        (with-check cublas-error
          (JCublas2/cublasGetMatrix (.sd stor) (.fd stor) width
                                    (offset da (extract (.buffer cu)) (.offset cu)) (.ld stor)
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host))
          host)
        (throw (ex-info "You cannot get an incompatible or ill-fitting matrix."
                        {:cu (info cu) :host (info host)}))))
    host))

(defn set-matrix! [^RealNativeMatrix host ^CUMatrix cu]
  (if (< 0 (.dim cu))
    (let [da (data-accessor cu)
          width (.entryWidth da)
          stor (full-storage cu)]
      (if (and (fits? cu host) (= width (.entryWidth (data-accessor host)))
               (= (navigator cu) (navigator host)))
        (with-check cublas-error
          (JCublas2/cublasSetMatrix (.sd stor) (.fd stor) width
                                    (offset da (ptr (.buffer host)) (.offset host)) (.stride host)
                                    (offset da (extract (.buffer cu)) (.offset cu)) (.ld stor))
          cu)
        (throw (ex-info "You cannot set an incompatible or ill-fitting matrix."
                        {:cu (info cu) :host (info host)}))))
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
    (format "#CUBlockVector[%s, n:%d, offset:%d, stride:%d]" (.entryType da) n ofst strd))
  Info
  (info [x]
    {:entry-type (.entryType da)
     :class (class x)
     :device :cuda
     :dim n
     :offset ofst
     :stride strd
     :master master
     :engine eng});;TODO (info [x info-type])
  Releaseable
  (release [_]
    (if master (release buf) true))
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
  Viewable
  (view [x]
    (cu-block-vector fact false buf n ofst strd))
  DenseContainer
  (view-vctr [x]
    x)
  (view-vctr [_ stride-mult]
    (cu-block-vector fact false buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (cu-ge-matrix fact false buf n 1 ofst (layout-navigator true) (full-storage true n 1) (ge-region n 1)))
  (view-ge [x stride-mult]
    (view-ge (view-ge x) stride-mult))
  (view-ge [x m n]
    (view-ge (view-ge x) m n))
  (view-tr [x uplo diag]
    (view-tr (view-ge x) uplo diag))
  (view-sy [x uplo]
    (view-sy (view-ge x) uplo))
  MemoryContext
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^Vector y)))
  (device [_]
    :cuda)
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
  IFn$LDD
  (invokePrim [x i v]
    (.set x i v))
  IFn$LD
  (invokePrim [x i]
    (.boxedEntry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i]
    (.boxedEntry x i))
  (invoke [x]
    n)
  RealChangeable
  (set [x val]
    (set-all eng val x)
    x)
  (set [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUVector
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
  (dim [_]
    n)
  (boxedEntry [x i]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (subvector [_ k l]
    (cu-block-vector fact false buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (cu-block-vector fact 0))
  Applicative
  (pure [_ v]
    (let-release [res (cu-block-vector fact 1)]
      (.set ^CUBlockVector res 0 v)))
  (pure [_ v vs]
    (vctr fact (cons v vs)))
  Foldable
  (fold [x]
    (sum eng x)))

(defn cu-block-vector
  ([fact master buf n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da buf)))
       (->CUBlockVector fact da (vector-engine fact) master buf n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf)})))))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (cu-block-vector fact true buf n 0 1))))

(extend CUBlockVector
  Magma
  {:op (constantly vector-op)})

(defmethod print-method CUBlockVector
  [^CUBlockVector x ^java.io.Writer w]
  (.write w (str x))
  (when (and (< 0 (.dim x)) (extract (.buffer x)) (active? (.da x)))
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

(defmethod transfer! [CUVector Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! h destination)))

(defmethod transfer! [Object CUVector]
  [source destination]
  (with-release [h (raw destination (native-factory destination))]
    (transfer! (transfer! source h) destination)))

;; ================== CUDA Matrix ============================================

(deftype CUGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg fact ^DataAccessor da eng
                     master buf ^long m ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :CUGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [_]
    (format "#CUGEMatrix[%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) m n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cuda
     :matrix-type :ge
     :dim (.dim ^Matrix a)
     :m m
     :n n
     :offset ofst
     :stride (.ld stor)
     :master master
     :layout (:layout (info nav))
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (cu-ge-matrix fact m n nav stor reg))
  (raw [_ fact]
    (create-ge fact m n (.isColumnMajor nav) false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (create-ge fact m n (.isColumnMajor nav) true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (get-matrix! a res)))
  (native [a]
    (host a))
  Viewable
  (view [a]
    (cu-ge-matrix fact false buf m n ofst nav stor reg))
  DenseContainer
  (view-vctr [a]
    (if (.isGapless stor)
      (cu-block-vector fact false buf (.dim a) ofst 1)
      (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:a (info a)}))))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [a]
    a)
  (view-ge [_ stride-mult]
    (let [shrinked (ceil (/ (.fd stor) (long stride-mult)))
          column-major (.isColumnMajor nav)
          [m n] (if column-major [m shrinked] [shrinked n])]
      (cu-ge-matrix fact false buf m n ofst nav
                    (full-storage column-major m n (* (long stride-mult) (.ld stor)))
                    (ge-region m n))))
  (view-ge [a m n]
    (if (.isGapless stor)
      (cu-ge-matrix fact false buf m n ofst nav (full-storage (.isColumnMajor nav) m n) (ge-region m n))
      (throw (ex-info "Strided GE matrix cannot be viewed through different dimensions." {:a (info a)}))))
  (view-tr [_ lower? diag-unit?]
    (let [n (min m n)]
      (cu-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld  stor))
                      (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?)
                      (tr-engine fact))))
  (view-sy [_ lower?]
    (let [n (min m n)]
      (cu-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld stor))
                      (band-region n lower?) :sy sy-default (sy-engine fact))))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? GEMatrix b)) (= reg (region b)))
  (fits-navigation? [_ b]
    (= nav (navigator b)))
  (device [_]
    :cuda)
  Monoid
  (id [a]
    (cu-ge-matrix fact 0 0))
  Applicative
  (pure [_ v]
    (let-release [res (cu-ge-matrix fact 1 1 (.isColumnMajor nav))]
      (.set ^CUGEMatrix res 0 0 v)))
  (pure [_ v vs]
    (ge fact (cons v vs)))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  IFn
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    true)
  (set [a val]
    (set-all eng val a)
    a)
  (set [_ _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [a _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
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
    (cu-block-vector fact false buf n (+ ofst (.index nav stor i 0))
                     (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (cu-block-vector fact false buf m (+ ofst (.index nav stor 0 j))
                     (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cu-block-vector fact false buf (min m n) ofst (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (cu-block-vector fact false buf (min m (- n k)) (+ ofst (.index nav stor 0 k)) (inc (.ld stor)))
      (cu-block-vector fact false buf (min (+ m k) n) (+ ofst (.index nav stor (- k) 0)) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (cu-ge-matrix fact false buf k l (+ ofst (.index nav stor i j))
                  nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (cu-ge-matrix fact false buf n m ofst (flip nav) stor (flip reg))))

(defn cu-ge-matrix
  ([fact master buf m n ofst nav stor reg]
   (->CUGEMatrix nav stor reg fact (data-accessor fact) (ge-engine fact) master buf m n ofst))
  ([fact m n nav ^FullStorage stor reg]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (cu-ge-matrix fact true buf m n 0 nav stor reg)))
  ([fact ^long m ^long n column?]
   (cu-ge-matrix fact m n (layout-navigator column?) (full-storage column? m n) (ge-region m n)))
  ([fact ^long m ^long n]
   (cu-ge-matrix fact m n true)))

(extend CUGEMatrix
  Magma
  {:op (constantly matrix-op)})

(defmethod print-method CUGEMatrix [^CUGEMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a)) (extract (.buffer a)) (active? (.da a)))
    (with-release [host-a (host a)]
      (print-ge w host-a))))

;; ============ CUDA Triangular Matrix =======================================

(deftype CUUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^RealDefault default
                       fact ^DataAccessor da eng matrix-type master buf ^long n ^long ofst]
  Object
  (hashCode [this]
    (-> (hash :CUUploMatrix) (hash-combine n) (hash-combine (nrm2 eng this))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [a]
    (format "#CUUploMatrix[%s, type%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) matrix-type n n (dec-property (.layout nav)) ofst))
  Releaseable
  (release [_]
    (if master (release buf) true))
  UploMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tr matrix-type))
  (isSymmetric [_]
    (= :sy matrix-type))
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (cu-uplo-matrix fact n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (get-matrix! a res)))
  (native [a]
    (host a))
  Viewable
  (view [a]
    (->CUUploMatrix nav stor reg default fact da eng matrix-type false buf n ofst))
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (cu-ge-matrix fact false buf n n ofst nav stor (ge-region n n)))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-ge [a m n]
    (view-ge (view-ge a) m n))
  (view-tr [_ lower? diag-unit?]
    (cu-uplo-matrix fact false buf n ofst nav stor (band-region n lower? diag-unit?)
                    :tr (real-default :tr diag-unit?) (tr-engine fact)))
  (view-sy [_ lower?]
    (cu-uplo-matrix fact false buf n ofst nav stor (band-region n lower?)
                    :sy sy-default (sy-engine fact)))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? UploMatrix b)
         (let [reg-b (region b)]
           (or (= reg reg-b)
               (and (= :sy matrix-type) (.isSymmetric ^UploMatrix b)
                    (not= nav (navigator b)) (not= (.uplo reg) (.uplo reg-b))
                    (= n (.ncols ^Matrix b)))))))
  (fits-navigation? [_ b]
    (and (= nav (navigator b))
         (or (instance? GEMatrix b) (= reg (region b)))))
  (device [_]
    :cuda)
  Monoid
  (id [a]
    (cu-uplo-matrix fact 0 (.isColumnMajor nav) matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (cu-uplo-matrix fact 1 (.isColumnMajor nav) matrix-type)]
      (.set ^CUUploMatrix res 0 0 v)))
  (pure [_ v vs]
    (let [source (cons v vs)]
      (let-release [res (cu-uplo-matrix fact (long (Math/sqrt (count source)))
                                        (.isColumnMajor nav) matrix-type)]
        (transfer! source res))))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  IFn
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    (.accessible reg i j))
  (set [a val]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (set [a i j val]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [a _ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  CUMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
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
      (cu-block-vector fact false buf (- (.rowEnd reg i) start) (+ ofst (.index nav stor i start))
                       (if (.isRowMajor nav) 1 (.ld ^FullStorage stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (cu-block-vector fact false buf (- (.colEnd reg j) start) (+ ofst (.index nav stor start j))
                       (if (.isColumnMajor nav) 1 (.ld ^FullStorage stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cu-block-vector fact false buf n ofst (inc (.ld ^FullStorage stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (cu-block-vector fact false buf (- n k) (+ ofst (.index nav stor 0 k))
                         (inc (.ld ^FullStorage stor)))
        (cu-block-vector fact false buf (+ n k) (+ ofst (.index nav stor (- k) 0))
                         (inc (.ld ^FullStorage stor))))
      (cu-block-vector fact false buf 0 ofst 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (cu-uplo-matrix fact false buf k (+ ofst (.index nav stor i j)) nav
                      (full-storage (.isColumnMajor nav) k k (.ld ^FullStorage stor))
                      (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (cu-uplo-matrix fact false buf n ofst (flip nav) stor (flip reg) matrix-type default eng))
  Triangularizable
  (create-trf [a pure]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_CUDA_MSG)
      a))
  (create-ptrf [a]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_CUDA_MSG)
      a))
  TRF
  (trtrs [a b]
    (if (= :tr matrix-type)
      (let-release [res (raw b)]
        (copy (engine b) b res)
        (trs eng a res))
      (require-trf)))
  (trtrs! [a b]
    (if (= :tr matrix-type)
      (trs eng a b)
      (require-trf)))
  (trtri! [a]
    (if (= :tr matrix-type)
      (tri eng a)
      (require-trf)))
  (trtri [a]
    (if (= :tr matrix-type)
      (let-release [res (raw a)]
        (tri eng (copy eng a res)))
      (require-trf)))
  (trcon [a _ nrm1?]
    (if (= :tr matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trcon [a nrm1?]
    (if (= :tr matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trdet [a]
    (dragan-says-ex UNAVAILABLE_CUDA_MSG)))

(extend CUUploMatrix
  Magma
  {:op (constantly matrix-op)})

(defn cu-uplo-matrix
  ([fact master buf n ofst nav stor reg matrix-type default engine]
   (->CUUploMatrix nav stor reg default fact (data-accessor fact) engine matrix-type
                   master buf n ofst))
  ([fact n nav ^FullStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (cu-uplo-matrix fact true buf n 0 nav stor reg matrix-type default engine)))
  ([fact n column? lower? diag-unit? matrix-type]
   (cu-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower? diag-unit?) matrix-type (real-default matrix-type diag-unit?)
                   (case matrix-type
                     :tr (tr-engine fact)
                     :sy (sy-engine fact)
                     (dragan-says-ex (format "%s is not a valid UPLO matrix type. Please send me a bug report."
                                             matrix-type)
                                     {:type matrix-type}))))
  ([fact n column? lower? diag-unit?]
   (cu-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?) (tr-engine fact)))
  ([fact n column? lower?]
   (cu-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower?) :sy (real-default :sy) (sy-engine fact))))

(defmethod print-method CUUploMatrix [^CUUploMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a)) (extract (.buffer a)) (active? (.da a)))
    (with-release [host-a (host a)]
      (print-uplo w host-a "*"))))

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
