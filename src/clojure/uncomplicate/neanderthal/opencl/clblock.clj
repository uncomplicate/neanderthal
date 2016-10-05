;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.opencl.clblock
  (:require [uncomplicate.commons.core
             :refer [Releaseable release let-release wrap-float wrap-double]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid]]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [transfer! copy!]]]
            [uncomplicate.neanderthal.impl.fluokitten
             :refer [vector-op matrix-op]]
            [uncomplicate.neanderthal.impl.buffer-block
             :refer [->RealBlockVector ->RealGeneralMatrix]]
            [uncomplicate.neanderthal.impl.cblas :refer
             [cblas-single cblas-double]])
  (:import [clojure.lang IFn IFn$L]
           [uncomplicate.clojurecl.core CLBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS Vector RealVector Matrix RealMatrix RealChangeable
            Block DataAccessor]
           [uncomplicate.neanderthal.impl.buffer_block
            RealBlockVector RealGeneralMatrix]))

(def ^{:private true :const true} INEFFICIENT_STRIDE_MSG
  "This operation would be inefficient when stride is not 1.")

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer. Please use transfer! of map-memory to be reminded of that.")

(defn cl-to-host [cl host]
  (let [mapped-host (map-memory cl :read)]
    (try
      (copy! mapped-host host)
      (finally (unmap cl mapped-host)))))

(defn host-to-cl [host cl]
  (let [mapped-host (map-memory cl :write-invalidate-region)]
    (try
      (copy! host mapped-host)
      cl
      (finally (unmap cl mapped-host)))))

(defprotocol CLAccessor
  (get-queue [this]))

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
    (enq-fill! queue cl-buf (array-fn 1))
    cl-buf)
  (initialize [_ cl-buf v]
    (enq-fill! queue cl-buf (wrap-fn v))
    cl-buf)
  (wrapPrim [_ s]
    (wrap-fn s))
  CLAccessor
  (get-queue [_]
    queue)
  Contextual
  (cl-context [_]
    ctx)
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible [_ o]
    (let [da (data-accessor o)]
      (or
       (and (instance? TypedCLAccessor da)
            (= et (.et ^TypedCLAccessor da)) (= ctx (.ctx ^TypedCLAccessor da)))
       (= ctx da)
       (= et da))))
  FactoryProvider
  (native-factory [_]
    host-fact)
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
                        ^Class entry-type master ^CLBuffer cl-buf
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
    (if (compare-and-set! master true false)
      (release cl-buf)
      true))
  Container
  (raw [_]
    (create-vector fact n (.createDataSource claccessor n) nil))
  (raw [_ fact]
    (create-vector fact n (.createDataSource (data-accessor fact) n) nil))
  (zero [this]
    (zero this fact))
  (zero [this fact]
    (let-release [r (raw this fact)]
      (.initialize (data-accessor fact) (.buffer ^Block r))
      r))
  (host [this]
    (cl-to-host this (raw this (native-factory this))))
  (native [this]
    (host this))
  Monoid
  (id [x]
    (create-vector fact 0 (.createDataSource claccessor 0) nil))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory claccessor))
  DataAccessorProvider
  (data-accessor [_]
    claccessor)
  MemoryContext
  (compatible [_ y]
    (and (instance? CLBlockVector y)
         (compatible claccessor (data-accessor y))))
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
  IFn
  (invoke [x]
    n)
  IFn$L
  (invokePrim [x]
    n)
  RealChangeable
  (set [x val]
    (if (and (= 0 ofst) (= 1 strd))
      (.initialize claccessor cl-buf val)
      (throw (IllegalArgumentException. INEFFICIENT_STRIDE_MSG)))
    x)
  (set [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (setBoxed [x val]
    (.set x val))
  (setBoxed [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (alter [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  RealVector
  (dim [_]
    n)
  (entry [_ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (boxedEntry [_ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (subvector [this k l]
    (CLBlockVector. fact claccessor eng entry-type (atom false) cl-buf l (+ ofst k) strd))
  Mappable
  (map-memory [this flags]
    (let [host-fact (native-factory this)
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
    (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
    this))

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
  (cl-to-host source destination))

(defmethod transfer! [RealBlockVector CLBlockVector]
  [source destination]
  (host-to-cl source destination))

(defmethod transfer! [clojure.lang.Sequential CLBlockVector]
  [source ^CLBlockVector destination]
  (let-release [host (raw destination (native-factory destination))]
    (host-to-cl (transfer! source host) destination)))

;; ================== CL Matrix ============================================

(deftype CLGeneralMatrix [^uncomplicate.neanderthal.protocols.Factory fact
                          ^DataAccessor claccessor ^BLAS eng
                          ^Class entry-type master ^CLBuffer cl-buf
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
    (if (compare-and-set! master true false)
      (release cl-buf)
      true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory claccessor))
  DataAccessorProvider
  (data-accessor [_]
    claccessor)
  MemoryContext
  (compatible [_ b]
    (and (or (instance? CLGeneralMatrix b) (instance? CLBlockVector b))
         (compatible claccessor (data-accessor b))))
  Container
  (raw [_]
    (create-matrix fact m n (.createDataSource claccessor (* m n)) ord))
  (raw [_ fact]
    (create-matrix fact m n (.createDataSource (data-accessor fact) (* m n)) ord))
  (zero [this]
    (zero this fact))
  (zero [this fact]
    (let-release [r (raw this fact)]
      (.initialize (data-accessor fact) (.buffer ^Block r))
      r))
  (host [this]
    (cl-to-host this (raw this (native-factory this))))
  (native [this]
    (host this))
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
  IFn
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (set [a val]
    (if (and (= ld (if (column-major? a) m n))
             (= 0 ofst)
             (= (* m n) (.count claccessor cl-buf)))
      (.initialize claccessor cl-buf val)
      (throw (IllegalArgumentException. INEFFICIENT_STRIDE_MSG)))
    a)
  (set [_ _ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (alter [a i j f]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (boxedEntry [_ _ _]
    (throw (UnsupportedOperationException. INEFFICIENT_OPERATION_MSG)))
  (row [a i]
    (if (column-major? a)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type (atom false) cl-buf n (+ ofst i) ld)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type (atom false) cl-buf n (+ ofst (* ld i)) 1)))
  (col [a j]
    (if (column-major? a)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type (atom false) cl-buf m (+ ofst (* ld j)) 1)
      (CLBlockVector. fact claccessor (vector-engine fact)
                      entry-type (atom false) cl-buf m (+ ofst j) ld)))
  (submatrix [a i j k l]
    (let [o (if (column-major? a) (+ ofst (* j ld) i) (+ ofst (* i ld) j))]
      (CLGeneralMatrix. fact claccessor eng
                        entry-type (atom false) cl-buf k l o ld ord)))
  (transpose [a]
    (CLGeneralMatrix. fact claccessor eng
                      entry-type (atom false) cl-buf n m ofst ld
                      (if (column-major? a) ROW_MAJOR COLUMN_MAJOR)))
  Mappable
  (map-memory [a flags]
    (let [host-fact (native-factory a)
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
    (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
    this))

(extend CLGeneralMatrix
  Magma
  {:op matrix-op})

(defmethod transfer! [CLGeneralMatrix CLGeneralMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLGeneralMatrix RealGeneralMatrix]
  [source destination]
  (cl-to-host source destination))

(defmethod transfer! [RealGeneralMatrix CLGeneralMatrix]
  [source destination]
  (host-to-cl source destination))

(defmethod transfer! [clojure.lang.Sequential CLGeneralMatrix]
  [source destination]
  (let-release [host (raw destination (native-factory destination))]
    (host-to-cl (transfer! source host) destination)))

(defmethod print-method CLGeneralMatrix
  [x ^java.io.Writer w]
  (.write w (str x)))

(defn create-cl-vector [factory vector-eng ^long n buf]
  (let [claccessor (data-accessor factory)]
    (if (and (<= 0 n (.count claccessor buf)) (instance? CLBuffer buf))
      (->CLBlockVector factory claccessor vector-eng (.entryType claccessor)
                       (atom true) buf n 0 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count claccessor buf) (class buf)))))))

(defn create-cl-ge-matrix [factory matrix-eng m n buf ord]
  (let [claccessor (data-accessor factory)]
    (if (and (<= 0 (* (long m) (long n)) (.count claccessor buf))
             (instance? CLBuffer buf))
      (let [ord (or ord DEFAULT_ORDER)
            ld (max (long (if (= COLUMN_MAJOR ord) m n)) 1)]
        (->CLGeneralMatrix factory claccessor matrix-eng (.entryType claccessor)
                           (atom true) buf m n 0 ld ord))
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d matrix from %s."
                      m n (type buf)))))))
