;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cuda.cublock
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
             [core :refer [transfer! copy! vctr]]
             [real :refer [entry]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer :all]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.host
             [fluokitten :refer [vector-op matrix-op vector-pure matrix-pure]]
             [buffer-block :refer [real-block-vector real-ge-matrix real-tr-matrix]]])
  (:import [clojure.lang IFn IFn$L IFn$LD IFn$LLD]
           [jcuda.jcublas JCublas2 cublasStatus]
           [uncomplicate.neanderthal.internal.api DataAccessor Block Vector
            RealVector Matrix RealMatrix GEMatrix TRMatrix RealChangeable
            RealOrderNavigator UploNavigator StripeNavigator]
           [uncomplicate.neanderthal.internal.host.buffer_block RealBlockVector IntegerBlockVector
            RealGEMatrix RealTRMatrix]))

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
  ;;(get-stream [this])
  (offset-cu-ptr [this buf ofst]))

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
  (offset-cu-ptr [_ buf ofst]
    (if (= 0 (long ofst))
      (cu-ptr buf)
      (with-offset (cu-ptr buf) (* (long ofst) w))))
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

(defn get-vector! [^Block cu ^Block host]
  (let [width (.entryWidth (data-accessor cu))]
    (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
      (with-check cublas-error
        (JCublas2/cublasGetVector (.dim ^Vector cu) width
                                  (cu-ptr (.buffer cu)) (.stride cu) (ptr (.buffer host)) (.stride host))
        host)
      (throw (ex-info "You cannot get  incompatible or ill-fitting vector."
                      {:cu (str cu) :host (str host)})))))

(defn set-vector! [^Block host ^Block cu]
  (let [width (.entryWidth (data-accessor cu))]
    (if (and (fits? cu host) (= width (.entryWidth (data-accessor host))))
      (with-check cublas-error
        (JCublas2/cublasSetVector (.dim ^Vector cu) (.entryWidth (data-accessor cu))
                                  (ptr (.buffer host)) (.stride host) (cu-ptr (.buffer cu)) (.stride cu))
        cu)
      (throw (ex-info "You cannot set incompatible or ill-fitting vector."
                      {:cu (str cu) :host (str host)})))))

(defn cl-to-obj [cl obj]
  (let [mapped-host (mmap cl :read)]
    (try
      (transfer! mapped-host obj)
      (finally (unmap cl mapped-host)))))

#_(defn obj-to-cl [obj cl]
  (let [mapped-host (mmap cl :write-invalidate-region)]
    (try
      (transfer! obj mapped-host)
      cl
      (finally (unmap cl mapped-host)))))


(defprotocol BlockEngine
  (equals-block [_ cu-x cu-y]))

;; =============================================================================

(deftype CUBlockVector [^uncomplicate.neanderthal.internal.api.Factory fact
                        ^DataAccessor da eng master buf ^long n ^long ofst ^long strd]
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
      (release @buf))
    (reset! buf nil)
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
    (cu-block-vector fact false @buf n ofst strd))
  (view-vctr [_ stride-mult]
    (cu-block-vector fact false @buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (cu-ge-matrix fact false @buf n 1 ofst n COLUMN_MAJOR))
  (view-ge [x stride-mult]
    (view-ge (view-ge x) stride-mult))
  (view-tr [x uplo diag]
    (view-tr (view-ge x) uplo diag))
  MemoryContext
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
  Block
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
    (cu-block-vector fact (atom false) @buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (cu-block-vector fact 0))
  Foldable
  (fold [x]
    (sum eng x)))

(defn cu-block-vector
  ([fact ^Boolean master buf n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da buf)))
       (->CUBlockVector fact da (vector-engine fact) (atom master) (atom buf) n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf)})))))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (cu-block-vector fact true buf n 0 1))))

(extend CUBlockVector
  Applicative
  {:pure vector-pure}
  Magma
  {:op (constantly vector-op)})

(defmethod print-method CUBlockVector
  [^CUBlockVector x ^java.io.Writer w]
  (if (and (< 0 (.dim x)) (.buffer x))
    (let-release [host-x (host x)]
      (.write w (str x "\n["))
      (let [max-value (double (amax (engine host-x) host-x))
            min-value (entry host-x (iamin (engine host-x) host-x))
            formatter (if (and (not (< 0.0 min-value 0.01)) (< max-value 10000.0)) format-f format-g)]
        (format-vector w formatter host-x))
      (.write w "]"))
    (.write w (str x))))

(defmethod transfer! [CUBlockVector CUBlockVector]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CUBlockVector RealBlockVector]
  [source destination]
  (get-vector! source destination))

(defmethod transfer! [RealBlockVector CUBlockVector]
  [source destination]
  (set-vector! source destination))

(defmethod transfer! [CUBlockVector IntegerBlockVector]
  [source destination]
  (get-vector! source destination))

(defmethod transfer! [IntegerBlockVector CUBlockVector]
  [source destination]
  (set-vector! source destination))

(defmethod transfer! [clojure.lang.Sequential CUBlockVector]
  [source ^CUBlockVector destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

(defmethod transfer! [(Class/forName "[D") CUBlockVector]
  [source ^CUBlockVector destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

(defmethod transfer! [(Class/forName "[F") CUBlockVector]
  [source ^CUBlockVector destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

(defmethod transfer! [(Class/forName "[J") CUBlockVector]
  [source ^CUBlockVector destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

(defmethod transfer! [(Class/forName "[I") CUBlockVector]
  [source ^CUBlockVector destination]
  (with-release [h (vctr (native-factory destination) source)]
    (set-vector! h destination)))

(defmethod transfer! [CUBlockVector Object]
  [source destination]
  (with-release [h (host source)]
    (transfer! (get-vector! source h) destination)))
