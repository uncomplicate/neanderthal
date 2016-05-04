(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.clblock
  (:require [uncomplicate.commons.core
             :refer [Releaseable release wrap-float wrap-double]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.core :refer [transfer! copy! create sum]]
            [uncomplicate.neanderthal.impl.fluokitten
             :refer [vector-op matrix-op]]
            [uncomplicate.neanderthal.impl.buffer-block
             :refer [->RealBlockVector ->RealGeneralMatrix]]
            [uncomplicate.neanderthal.impl.cblas :refer
             [cblas-single cblas-double]])
  (:import [uncomplicate.neanderthal CBLAS]
           [uncomplicate.neanderthal.protocols
            BLAS Vector Matrix Changeable Block DataAccessor]
           [uncomplicate.neanderthal.impl.buffer_block
            RealBlockVector RealGeneralMatrix]))

(defprotocol CLAccessor
  (get-queue [this])
  (wrap-seq [this s])
  (wrap-prim [this p]))

;; ================== Accessors ================================================
(deftype TypedCLAccessor [ctx queue et ^long w array-fn wrap-fn host-fact]
  DataAccessor
  (entryType [_]
    et)
  (entryWidth [_]
    w)
  (count [_ b]
    (quot (long (size b)) w))
  (createDataSource [_ n]
    (cl-buffer ctx (* w (max 1 (long n))) :read-write))
  (initialize [_ cl-buf]
    (do
      (enq-fill! queue cl-buf (array-fn 1))
      cl-buf))
  CLAccessor
  (get-queue [_]
    queue)
  (wrap-seq [_ s]
    (array-fn s))
  (wrap-prim [_ s]
    (wrap-fn s))
  FactoryProvider
  (factory [_]
    host-fact))

(defn cl-float-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES
                     float-array wrap-float cblas-single))

(defn cl-double-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES
                     double-array wrap-double cblas-double))

;; ================== Non-blas kernels =========================================
(defprotocol BlockEngine
  (equals-block [_ cl-x cl-y]))

;; =============================================================================

(deftype CLBlockVector [^uncomplicate.neanderthal.protocols.Factory fact
                        ^DataAccessor claccessor ^BLAS eng
                        ^Class entry-type ^Boolean master
                        ^uncomplicate.clojurecl.core.CLBuffer cl-buf
                        ^long n ^long ofst ^long strd]
  Object
  (hashCode [this]
    (-> (hash :CLBlockVector) (hash-combine n)
        (hash-combine (.nrm2 eng this))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (compatible x y) (= n (.dim ^Vector y)))
      (equals-block eng x y)
      :default false))
  (toString [this]
    (format "#CLBlockVector[%s, n:%d, offset:%d stride:%d]"
            entry-type n ofst strd))
  Releaseable
  (release [_]
    (if master (release cl-buf) true))
  Container
  (raw [_]
    (create-vector fact n (.createDataSource claccessor n) nil))
  (raw [_ fact]
    (create-vector fact n (.createDataSource (data-accessor fact) n) nil))
  (zero [this]
    (zero this fact))
  (zero [this fact]
    (let [r (raw this fact)
          clacc (data-accessor fact)]
      (.initialize clacc (.buffer ^Block r))
      r))
  Monoid
  (id [x]
    (create-vector fact 0 (.createDataSource claccessor 0) nil))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  Memory
  (compatible [_ y]
    (and (instance? CLBlockVector y) (= entry-type (.entryType ^Block y))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    cl-buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (count [_]
    n)
  Vector
  (dim [_]
    n)
  (subvector [this k l]
    (CLBlockVector. fact claccessor eng entry-type false cl-buf l (+ ofst k) strd))
  Mappable
  (map-memory [this flags]
    (let [host-fact (factory fact)
          acc (data-accessor host-fact)
          queue (get-queue claccessor)
          mapped-buf (enq-map-buffer! queue cl-buf true
                                      (* ofst (.entryWidth claccessor))
                                      (* strd n (.entryWidth claccessor))
                                      flags nil nil)]
      (try
        (->RealBlockVector host-fact acc (vector-engine host-fact)
                           (.entryType acc) true mapped-buf n strd)
        (catch Exception e (enq-unmap! queue cl-buf mapped-buf)))))
  (unmap [this mapped]
    (do
      (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
      this)))

(extend CLBlockVector
  Magma
  {:op vector-op})

(defmethod print-method CLBlockVector
  [x ^java.io.Writer w]
  (.write w (str x)))

(defmethod transfer! [CLBlockVector CLBlockVector]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLBlockVector RealBlockVector]
  [source destination]
  (let [mapped-host (map-memory source :read)]
    (try
      (copy! mapped-host destination)
      (finally (unmap source mapped-host)))))

(defmethod transfer! [RealBlockVector CLBlockVector]
  [source destination]
  (let [mapped-host (map-memory destination :write-invalidate-region)]
    (try
      (do
        (copy! source mapped-host)
        destination)
      (finally (unmap destination mapped-host)))))

(defmethod transfer! [clojure.lang.Sequential CLBlockVector]
  [source ^CLBlockVector destination]
  (transfer!
   (transfer! source (create (factory (factory destination))
                             (.dim destination)))
   destination))

;; ================== CL Matrix ============================================

(deftype CLGeneralMatrix [^uncomplicate.neanderthal.protocols.Factory fact
                          ^DataAccessor claccessor ^BLAS eng
                          ^Class entry-type ^Boolean master
                          ^uncomplicate.clojurecl.core.CLBuffer cl-buf
                          ^long m ^long n ^long ofst ^long ld ^long ord]
  Object
  (hashCode [this]
    (-> (hash :CLGeneralMatrix) (hash-combine m) (hash-combine n)))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= m (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (equals-block eng a b)
      :default false))
  (toString [this]
    (format "#CLGeneralMatrix[%s, %s, mxn: %dx%d, offset:%d, ld:%d>]"
            entry-type "COL" m n ofst ld))
  Releaseable
  (release [_]
    (if master (release cl-buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  Memory
  (compatible [_ y]
    (and (or (instance? CLGeneralMatrix y) (instance? CLBlockVector y))
         (= entry-type (.entryType ^Block y))))
  Container
  (raw [_]
    (create-matrix fact m n (.createDataSource claccessor (* m n)) ord))
  (raw [_ fact]
    (create-matrix fact m n (.createDataSource (data-accessor fact) (* m n)) ord))
  (zero [this]
    (zero this fact))
  (zero [this fact]
    (let [r (raw this fact)
          clacc (data-accessor fact)]
      (.initialize clacc (.buffer ^Block r))
      r))
  Monoid
  (id [a]
    (create-matrix fact 0 0 (.createDataSource claccessor 0) nil))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    cl-buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (order [_]
    ord)
  (count [_]
    (* m n))
  Matrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (row [a i]
    (if (column-major? a)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type false cl-buf n (+ ofst i) ld)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type false cl-buf n (+ ofst (* ld i)) 1)))
  (col [a j]
    (if (column-major? a)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type false cl-buf m (+ ofst (* ld j)) 1)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type false cl-buf m (+ ofst j) ld)))
  (submatrix [a i j k l]
    (let [o (if (column-major? a) (+ ofst (* j ld) i) (+ ofst (* i ld) j))]
      (CLGeneralMatrix. fact claccessor eng
                        entry-type false cl-buf k l o ld ord)))
  Mappable
  (map-memory [a flags]
    (let [host-fact (factory fact)
          acc (data-accessor host-fact)
          queue (get-queue claccessor)
          mapped-buf (enq-map-buffer! queue cl-buf true
                                      (* ofst (.entryWidth claccessor))
                                      (* (if (column-major? a) n m)
                                         ld (.entryWidth claccessor))
                                      flags nil nil)]
      (try
        (->RealGeneralMatrix host-fact acc (matrix-engine host-fact)
                             (.entryType acc) true mapped-buf m n ld ord)
        (catch Exception e (enq-unmap! queue cl-buf mapped-buf)))))
  (unmap [this mapped]
    (do
      (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
      this)))

(extend CLGeneralMatrix
  Magma
  {:op matrix-op})

(defmethod transfer! [CLGeneralMatrix CLGeneralMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLGeneralMatrix RealGeneralMatrix]
  [source destination]
  (let [mapped-host (map-memory source :read)]
    (try
      (copy! mapped-host destination)
      (finally
        (unmap source mapped-host)))))

(defmethod transfer! [RealGeneralMatrix CLGeneralMatrix]
  [source destination]
  (let [mapped-host (map-memory destination :write-invalidate-region)]
    (try
      (do
        (copy! source mapped-host)
        destination)
      (finally (unmap destination mapped-host)))))

(defmethod transfer! [clojure.lang.Sequential CLGeneralMatrix]
  [source destination]
  (transfer!
   (transfer! source (create (factory (factory destination))
                             (.mrows ^Matrix destination)
                             (.ncols ^Matrix destination)))
   destination))

(defmethod print-method CLGeneralMatrix
  [x ^java.io.Writer w]
  (.write w (str x)))
