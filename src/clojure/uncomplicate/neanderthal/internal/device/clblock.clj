;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.clblock
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Mappable mmap unmap
                           Info info Wrapper extract wrap-float wrap-double wrap-int wrap-long]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Foldable Applicative]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.internal.protocols :refer [size]]
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
            [uncomplicate.neanderthal.internal.device.common :refer [device-vector-equals device-matrix-equals]])
  (:import [clojure.lang IFn IFn$L IFn$LD IFn$LDD IFn$LLD]
           [uncomplicate.neanderthal.internal.api DataAccessor VectorSpace Vector CLVector Matrix
            CLMatrix GEMatrix RealChangeable LayoutNavigator RealLayoutNavigator Region MatrixImplementation
            NativeBlock FullStorage RealDefault UploMatrix RealNativeMatrix]
           [uncomplicate.neanderthal.internal.host.buffer_block RealBlockVector RealGEMatrix
            RealUploMatrix]))

(def ^{:private true :const true} INEFFICIENT_STRIDE_MSG
  "This operation would be inefficient when stride is not 1.")

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! of map-memory to be reminded of that.")

(def ^{:private true :const true} UNAVAILABLE_OPENCL_MSG
  "This operation is not available in OpenCL (yet).")

(defn cl-to-host [cl host]
  (let [mapped-host (mmap cl :read)]
    (try
      (copy! mapped-host host)
      (finally (unmap cl mapped-host)))))

(defn host-to-cl [host cl]
  (let [mapped-host (mmap cl :write-invalidate-region)]
    (try
      (copy! host mapped-host)
      cl
      (finally (unmap cl mapped-host)))))

(defn cl-to-obj [cl obj]
  (let [mapped-host (mmap cl :read)]
    (try
      (transfer! mapped-host obj)
      (finally (unmap cl mapped-host)))))

(defn obj-to-cl [obj cl]
  (let [mapped-host (mmap cl :write-invalidate-region)]
    (try
      (transfer! obj mapped-host)
      cl
      (finally (unmap cl mapped-host)))))

(defprotocol CLAccessor
  (active? [this]))

;; ================== Declarations ============================================

(declare cl-block-vector cl-ge-matrix cl-uplo-matrix ->CLUploMatrix)

;; ================== Accessors ================================================

(extend-type Object
  Wrapper
  (extract [_]
    nil))

(deftype TypedCLAccessor [active ctx queue et ^long w array-fn wrap-fn cast-fn]
  Releaseable
  (release [this]
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
    (cl-buffer ctx (* w (max 1 (long n))) :read-write))
  (initialize [_ buf]
    (enq-fill! queue buf (array-fn 1))
    buf)
  (initialize [_ buf v]
    (enq-fill! queue buf (wrap-fn v))
    buf)
  (wrapPrim [_ s]
    (wrap-fn s))
  (castPrim [_ v]
    (cast-fn v))
  CLAccessor
  (active? [_]
    @active)
  FlowProvider
  (flow [_]
    queue)
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or
       (identical? this o) (identical? this da)
       (and (instance? TypedCLAccessor da)
            (= et (.et ^TypedCLAccessor da)) (= ctx (.ctx ^TypedCLAccessor da)))
       (= ctx o)
       (= et o)))))

(defn cl-float-accessor [ctx queue]
  (->TypedCLAccessor (volatile! true) ctx queue Float/TYPE Float/BYTES float-array wrap-float float))

(defn cl-double-accessor [ctx queue]
  (->TypedCLAccessor (volatile! true) ctx queue Double/TYPE Double/BYTES double-array wrap-double double))

(defn cl-int-accessor [ctx queue]
  (->TypedCLAccessor (volatile! true) ctx queue Integer/TYPE Integer/BYTES int-array wrap-int int))

(defn cl-long-accessor [ctx queue]
  (->TypedCLAccessor (volatile! true) ctx queue Long/TYPE Long/BYTES long-array wrap-long long))

;; =============================================================================

(deftype CLBlockVector [fact ^DataAccessor da eng master buf ^long n ^long ofst ^long strd]
  Object
  (hashCode [x]
    (-> (hash :CLBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (device-vector-equals eng x y))
  (toString [this]
    (format "#CLBlockVector[%s, n:%d, offset:%d stride:%d]" (.entryType da) n ofst strd))
  Info
  (info [x]
    {:entry-type (.entryType da)
     :class (class x)
     :device :opencl
     :dim n
     :offset ofst
     :stride strd
     :master master
     :engine eng})
  Releaseable
  (release [_]
    (if master (release buf) true))
  Container
  (raw [_]
    (cl-block-vector fact n))
  (raw [_ fact]
    (create-vector fact n false))
  (zero [x]
    (zero x fact))
  (zero [_ fact]
    (create-vector fact n true))
  (host [x]
    (let-release [res (raw x (native-factory fact))]
      (cl-to-host x res)))
  (native [x]
    (host x))
  Viewable
  (view [_]
    (cl-block-vector fact false buf n ofst strd))
  DenseContainer
  (view-vctr [x]
    x)
  (view-vctr [_ stride-mult]
    (cl-block-vector fact false buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (cl-ge-matrix fact false buf n 1 ofst (layout-navigator true) (full-storage true n 1) (ge-region n 1)))
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
    :opencl)
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
  CLVector
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
    (cl-block-vector fact false buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (cl-block-vector fact 0))
  Applicative
  (pure [_ v]
    (let-release [res (cl-block-vector fact 1)]
      (.set ^CLBlockVector res 0 v)))
  (pure [_ v vs]
    (vctr fact (cons v vs)))
  Foldable
  (fold [x]
    (sum eng x))
  Mappable
  (mmap [_ flags]
    (let [host-fact (native-factory fact)
          queue (flow da)
          mapped-buf (enq-map-buffer! queue buf true (* ofst (.entryWidth da))
                                      (* strd n (.entryWidth da)) flags nil nil)]
      (try
        (real-block-vector host-fact true mapped-buf n 0 strd)
        (catch Exception e
          (enq-unmap! queue buf mapped-buf)
          (throw e)))))
  (unmap [x mapped]
    (enq-unmap! (flow da) buf (.buffer ^NativeBlock mapped))
    x))

(defn cl-block-vector
  ([fact master buf n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da buf)))
       (->CLBlockVector fact da (vector-engine fact) master buf n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf)})))))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (cl-block-vector fact true buf n 0 1))))

(extend CLBlockVector
  Magma
  {:op (constantly vector-op)})

(defmethod print-method CLBlockVector
  [^CLBlockVector x ^java.io.Writer w]
  (.write w (str x))
  (when (and (< 0 (.dim x)) (extract (.buffer x)) (active? (.da x)))
    (let [mapped-x (mmap x :read)]
      (try
        (print-vector w mapped-x)
        (finally (unmap x mapped-x))))))

(defmethod transfer! [CLBlockVector CLBlockVector]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLBlockVector RealBlockVector]
  [source destination]
  (cl-to-host source destination))

(defmethod transfer! [RealBlockVector CLBlockVector]
  [source destination]
  (host-to-cl source destination))

(defmethod transfer! [CLVector Object]
  [source destination]
  (cl-to-obj source destination))

(defmethod transfer! [Object CLVector]
  [source destination]
  (obj-to-cl source destination))

;; ================== CL Matrix ============================================

(deftype CLGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg fact ^DataAccessor da eng
                     master buf ^long m ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :CLGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [_]
    (format "#CLGEMatrix[%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) m n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :opencl
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
    (cl-ge-matrix fact m n nav stor reg))
  (raw [_ fact]
    (create-ge fact m n (.isColumnMajor nav) false))
  (zero [a]
    (zero a fact))
  (zero [_ fact]
    (create-ge fact m n (.isColumnMajor nav) true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (cl-to-host a res)))
  (native [a]
    (host a))
  Viewable
  (view [a]
    (cl-ge-matrix fact false buf m n ofst nav stor reg))
  DenseContainer
  (view-vctr [a]
    (if (.isGapless stor)
      (cl-block-vector fact false buf (.dim a) ofst 1)
      (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:a (info a)}))))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [a]
    a)
  (view-ge [_ stride-mult]
    (let [shrinked (ceil (/ (.fd stor) (long stride-mult)))
          column-major (.isColumnMajor nav)
          [m n] (if column-major [m shrinked] [shrinked n])]
      (cl-ge-matrix fact false buf m n ofst nav
                    (full-storage column-major m n (* (long stride-mult) (.ld stor)))
                    (ge-region m n))))
  (view-ge [a m n]
    (if (.isGapless stor)
      (cl-ge-matrix fact false buf m n ofst nav (full-storage (.isColumnMajor nav) m n) (ge-region m n))
      (throw (ex-info "Strided GE matrix cannot be viewed through different dimensions." {:a (info a)}))))
  (view-tr [_ lower? diag-unit?]
    (let [n (min m n)]
      (cl-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld stor))
                      (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?)
                      (tr-engine fact))))
  (view-sy [_ lower?]
    (let [n (min m n)]
      (cl-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld stor))
                      (band-region n lower?) :sy sy-default (sy-engine fact))))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? GEMatrix b)) (= reg (region b)))
  (fits-navigation? [_ b]
    (= nav (navigator b)))
  (device [_]
    :opencl)
  Monoid
  (id [a]
    (cl-ge-matrix fact 0 0 (.isColumnMajor nav)))
  Applicative
  (pure [_ v]
    (let-release [res (cl-ge-matrix fact 1 1 (.isColumnMajor nav))]
      (.set ^CLGEMatrix res 0 0 v)))
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
  CLMatrix
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
    (cl-block-vector fact false buf n (+ ofst (.index nav stor i 0))
                     (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (cl-block-vector fact false buf m (+ ofst (.index nav stor 0 j))
                     (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cl-block-vector fact false buf (min m n) ofst (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (cl-block-vector fact false buf (min m (- n k)) (+ ofst (.index nav stor 0 k)) (inc (.ld stor)))
      (cl-block-vector fact false buf (min (+ m k) n) (+ ofst (.index nav stor (- k) 0)) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (cl-ge-matrix fact false buf k l (+ ofst (.index nav stor i j))
                  nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (cl-ge-matrix fact false buf n m ofst (flip nav) stor (flip reg)))
  Mappable
  (mmap [a flags]
    (let [host-fact (native-factory fact)
          queue (flow da)
          mapped-buf (enq-map-buffer! queue buf true (* ofst (.entryWidth da))
                                      (* (.capacity stor) (.entryWidth da)) flags nil nil)]
      (try
        (real-ge-matrix host-fact true mapped-buf m n 0 nav stor reg)
        (catch Exception e
          (enq-unmap! queue buf mapped-buf)
          (throw e)))))
  (unmap [this mapped]
    (enq-unmap! (flow da) buf (.buffer ^NativeBlock mapped))
    this))

(defn cl-ge-matrix
  ([fact master buf m n ofst nav stor reg]
   (->CLGEMatrix nav stor reg fact (data-accessor fact) (ge-engine fact) master buf m n ofst))
  ([fact m n nav ^FullStorage stor reg]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (cl-ge-matrix fact true buf m n 0 nav stor reg)))
  ([fact ^long m ^long n column?]
   (cl-ge-matrix fact m n (layout-navigator column?) (full-storage column? m n) (ge-region m n)))
  ([fact ^long m ^long n]
   (cl-ge-matrix fact m n true)))

(extend CLGEMatrix
  Magma
  {:op (constantly matrix-op)})

(defmethod print-method CLGEMatrix [^CLGEMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a)) (extract (.buffer a)) (active? (.da a)))
    (let [mapped-a (mmap a :read)]
      (try
        (print-ge w mapped-a)
        (finally (unmap a mapped-a))))))

;; ============ OpenCL Uplo Matrix =======================================

(deftype CLUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^RealDefault default
                       fact ^DataAccessor da eng matrix-type master buf ^long n ^long ofst]
  Object
  (hashCode [this]
    (-> (hash :CLLUploMatrix) (hash-combine n) (hash-combine (nrm2 eng this))))
  (equals [a b]
    (device-matrix-equals eng a b))
  (toString [a]
    (format "#CLUploMatrix[%s, type%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) matrix-type n n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :opencl
     :matrix-type matrix-type
     :dim (.dim ^Matrix a)
     :m n
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
    (cl-uplo-matrix fact n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) false))
  (zero [_]
    (zero _ fact))
  (zero [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) true))
  (host [a]
    (let-release [res (raw a (native-factory fact))]
      (cl-to-host a res)))
  (native [a]
    (host a))
  Viewable
  (view [a]
    (->CLUploMatrix nav stor reg default fact da eng matrix-type false buf n ofst))
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (cl-ge-matrix fact false buf n n ofst nav stor (ge-region n n)))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-ge [a m n]
    (view-ge (view-ge a) m n))
  (view-tr [_ lower? diag-unit?]
    (cl-uplo-matrix fact false buf n ofst nav stor (band-region n lower? diag-unit?)
                    :tr (real-default :tr diag-unit?) (tr-engine fact)))
  (view-sy [_ lower?]
    (cl-uplo-matrix fact false buf n ofst nav stor (band-region n lower?)
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
    :opencl)
  Monoid
  (id [a]
    (cl-uplo-matrix fact 0 (.isColumnMajor nav) matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (cl-uplo-matrix fact 1 (.isColumnMajor nav) matrix-type)]
      (.set ^CLUploMatrix res 0 0 v)))
  (pure [_ v vs]
    (let [source (cons v vs)]
      (let-release [res (cl-uplo-matrix fact (long (Math/sqrt (count source)))
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
  CLMatrix
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
      (cl-block-vector fact false buf (- (.rowEnd reg i) start) (+ ofst (.index nav stor i start))
                       (if (.isRowMajor nav) 1 (.ld ^FullStorage stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (cl-block-vector fact false buf (- (.colEnd reg j) start) (+ ofst (.index nav stor start j))
                       (if (.isColumnMajor nav) 1 (.ld ^FullStorage stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (cl-block-vector fact false buf n ofst (inc (.ld ^FullStorage stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (cl-block-vector fact false buf (- n k) (+ ofst (.index nav stor 0 k))
                         (inc (.ld ^FullStorage stor)))
        (cl-block-vector fact false buf (+ n k) (+ ofst (.index nav stor (- k) 0))
                         (inc (.ld ^FullStorage stor))))
      (cl-block-vector fact false buf 0 ofst 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (cl-uplo-matrix fact false buf k (+ ofst (.index nav stor i j)) nav
                      (full-storage (.isColumnMajor nav) k k (.ld ^FullStorage stor))
                      (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (cl-uplo-matrix fact false buf n ofst (flip nav) stor (flip reg) matrix-type default eng))
  Mappable
  (mmap [a flags]
    (let [host-fact (native-factory fact)
          queue (flow da)
          mapped-buf (enq-map-buffer! queue buf true (* ofst (.entryWidth da))
                                      (* (.capacity stor) (.entryWidth da)) flags nil nil)]
      (try
        (real-uplo-matrix host-fact true mapped-buf n 0 nav stor reg matrix-type default (tr-engine host-fact))
        (catch Exception e
          (enq-unmap! queue buf mapped-buf)
          (throw e)))))
  (unmap [this mapped]
    (enq-unmap! (flow da) buf (.buffer ^NativeBlock mapped))
    this)
  Triangularizable
  (create-trf [a pure]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_OPENCL_MSG)
      a))
  (create-ptrf [a]
    (if (= :sy matrix-type)
      (dragan-says-ex UNAVAILABLE_OPENCL_MSG)
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
    (dragan-says-ex UNAVAILABLE_OPENCL_MSG)))

(extend CLUploMatrix
  Magma
  {:op (constantly matrix-op)})

(defn cl-uplo-matrix
  ([fact master buf n ofst nav stor reg matrix-type default engine]
   (->CLUploMatrix nav stor reg default fact (data-accessor fact) engine matrix-type
                   master buf n ofst))
  ([fact n nav ^FullStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (cl-uplo-matrix fact true buf n 0 nav stor reg matrix-type default engine)))
  ([fact n column? lower? diag-unit? matrix-type]
   (cl-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower? diag-unit?) matrix-type (real-default matrix-type diag-unit?)
                   (case matrix-type
                     :tr (tr-engine fact)
                     :sy (sy-engine fact)
                     (dragan-says-ex (format "%s is not a valid UPLO matrix type. Please send me a bug report."
                                             matrix-type)
                                     {:type matrix-type}))))
  ([fact n column? lower? diag-unit?]
   (cl-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?) (tr-engine fact)))
  ([fact n column? lower?]
   (cl-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                   (band-region n lower?) :sy (real-default :sy) (sy-engine fact))))

(defmethod print-method CLUploMatrix [^CLUploMatrix a ^java.io.Writer w]
  (.write w (str a))
  (when (and (< 0 (.dim a)) (extract (.buffer a)) (active? (.da a)))
    (let [mapped-a (mmap a :read)]
      (try
        (print-uplo w mapped-a "*")
        (finally (unmap a mapped-a))))))

(defmethod transfer! [CLGEMatrix CLGEMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLUploMatrix CLUploMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLUploMatrix CLGEMatrix]
  [source destination]
  (let [reg (region source)]
    (copy! source (view-tr destination (.isLower reg) (.isDiagUnit reg)))))

(defmethod transfer! [CLGEMatrix CLUploMatrix]
  [source destination]
  (let [reg (region destination)]
    (copy! (view-tr source (.isLower reg) (.isDiagUnit reg)) destination)))

(defmethod transfer! [CLMatrix RealNativeMatrix]
  [source destination]
  (cl-to-host source destination))

(defmethod transfer! [RealNativeMatrix CLMatrix]
  [source destination]
  (host-to-cl source destination))

(defmethod transfer! [CLMatrix Object]
  [source destination]
  (cl-to-obj source destination))

(defmethod transfer! [Object CLMatrix]
  [source destination]
  (obj-to-cl source destination))

(prefer-method transfer! [CLVector Object] [Object CLVector])
(prefer-method transfer! [CLMatrix Object] [Object CLMatrix])
