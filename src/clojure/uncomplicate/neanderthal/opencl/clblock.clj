(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.clblock
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.core :refer [transfer! copy!]]
            [uncomplicate.neanderthal.impl.buffer-block :refer
             [COLUMN_MAJOR float-accessor double-accessor
              ->RealBlockVector ->RealGeneralMatrix column-major?]]
            [uncomplicate.neanderthal.impl.cblas :refer
             [cblas-single cblas-double]])
  (:import [uncomplicate.neanderthal CBLAS])
  (:import [uncomplicate.neanderthal.protocols
            BLAS Vector Matrix Changeable Block DataAccessor])
  (:import [uncomplicate.neanderthal.impl.buffer_block
            RealBlockVector RealGeneralMatrix]))

(def ^:private INCOMPATIBLE_BLOCKS_MSG
  "Operation is not permited on vectors with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s")

;; ================== Accessors ================================================
(deftype TypedCLAccessor [ctx queue et ^long w array-fn]
  DataAccessor
  (entryType [_]
    et)
  (entryWidth [_]
    w)
  CLAccessor
  (get-queue [_]
    queue)
  (create-buffer [_ n]
    (cl-buffer ctx (* w (long n)) :read-write))
  (fill-buffer [_ cl-buf v]
    (do
      (enq-fill! queue cl-buf (array-fn v))
      cl-buf))
  (array [_ s]
    (array-fn s))
  (slice [_ cl-buf k l]
    (cl-sub-buffer cl-buf (* w (long k)) (* w (long l)))))

;; ================== Non-blas kernels =========================================
(defprotocol BlockEngine
  (equals-vector [_ ^Block cl-x ^Block cl-y]))

;; =============================================================================

(declare create-vector)
(declare create-ge-matrix)

(deftype CLBlockVector [engine-factory ^DataAccessor claccessor ^BLAS eng
                        entry-type master cl-buf
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
      (equals-vector eng x y)
      :default false))
  (toString [_]
    (format "#<CLBlockVector| %s, n:%d, offset:%d stride:%d>"
            entry-type n ofst strd))
  Releaseable
  (release [_]
    (and
     (if master (release cl-buf) true)
     (release eng)))
  Group
  (zero [_]
    (create-vector engine-factory n))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ y]
    (and (instance? CLBlockVector y)
         (= entry-type (.entryType ^Block y))))
  BlockCreator
  (create-block [_ m n]
    (create-ge-matrix engine-factory m n))
  (create-block [_ n]
    (create-vector engine-factory n))
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
  (subvector [_ k l]
    (CLBlockVector. engine-factory claccessor
                    (vector-engine engine-factory cl-buf l (+ ofst k) strd)
                    entry-type false cl-buf l (+ ofst k) strd))
  Mappable
  (map-memory [_ flags]
    (let [host-engine-factory (cond (= Float/TYPE entry-type) cblas-single
                                    (= Double/TYPE entry-type) cblas-double)
          acc ^RealBufferAccessor (data-accessor host-engine-factory)
          queue (get-queue claccessor)
          mapped-buf (enq-map-buffer! queue cl-buf true
                                      (* ofst (.entryWidth claccessor))
                                      (* strd n (.entryWidth claccessor))
                                      flags nil nil)]
      (try
        (->RealBlockVector host-engine-factory acc
                           (vector-engine host-engine-factory mapped-buf n 0 strd)
                           entry-type true mapped-buf n strd)
        (catch Exception e (enq-unmap! queue cl-buf mapped-buf)))))
  (unmap [this mapped]
    (do
      (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
      this)))

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

(deftype CLGeneralMatrix [engine-factory ^DataAccessor claccessor
                          eng entry-type master cl-buf
                          ^long m ^long n ^long ld]
  Object
  (toString [_]
    (format "#<CLGeneralMatrix| %s, %s, mxn: %dx%d, ld:%d>"
            entry-type "COL" m n ld))
  Releaseable
  (release [_]
    (and
     (if master (release cl-buf) true)
     (release eng)))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ y]
    (and (or (instance? CLGeneralMatrix y) (instance? CLBlockVector y))
         (= entry-type (.entryType ^Block y))))
  Group
  (zero [_]
    (create-ge-matrix engine-factory m n))
  BlockCreator
  (create-block [_ m1 n1]
    (create-ge-matrix engine-factory m1 n1))
  (create-block [_ n]
    (create-vector engine-factory n))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    cl-buf)
  (stride [_]
    ld)
  (order [_]
    COLUMN_MAJOR)
  (count [_]
    (* m n ))
  Matrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (row [a i]
    (if (column-major? a)
      (CLBlockVector. engine-factory claccessor
                      (vector-engine engine-factory cl-buf n i ld)
                      entry-type false cl-buf n i ld)
      (CLBlockVector. engine-factory claccessor
                      (vector-engine engine-factory cl-buf n (* ld i) 1)
                      entry-type false cl-buf n (* ld i) 1)))
  (col [a j]
    (if (column-major? a)
      (CLBlockVector. engine-factory claccessor
                      (vector-engine engine-factory cl-buf m (* ld j) 1)
                      entry-type false cl-buf m (* ld j) 1)
      (CLBlockVector. engine-factory claccessor
                      (vector-engine engine-factory cl-buf m j ld)
                      entry-type false cl-buf m j ld)))
  Mappable
  (map-memory [this flags]
    (let [host-engine-factory (cond (= Float/TYPE entry-type) cblas-single
                                    (= Double/TYPE entry-type) cblas-double)
          acc ^RealBufferAccessor (data-accessor host-engine-factory)
          queue (get-queue claccessor)
          mapped-buf (enq-map-buffer! queue cl-buf true
                                      0;;TODO offset (* ofst (.entryWidth claccessor))
                                      (* (.count ^Block this) (.entryWidth claccessor))
                                      flags nil nil)]
      (try
        (->RealGeneralMatrix host-engine-factory acc
                             (matrix-engine host-engine-factory mapped-buf m n ld)
                             entry-type true mapped-buf m n ld COLUMN_MAJOR);;TODO add order
        (catch Exception e (enq-unmap! queue cl-buf mapped-buf)))))
  (unmap [this mapped]
    (do
      (enq-unmap! (get-queue claccessor) cl-buf (.buffer ^Block mapped))
      this)))

(defmethod transfer! [CLGeneralMatrix CLGeneralMatrix]
  [source destination]
  (copy! source destination))

(defmethod transfer! [CLGeneralMatrix RealGeneralMatrix]
  [source destination]
  (let [mapped-host (map-memory source :read)]
    (try
      (copy! mapped-host destination)
      (finally (unmap source mapped-host)))))

(defmethod transfer! [RealGeneralMatrix CLGeneralMatrix]
  [source destination]
  (let [mapped-host (map-memory destination :write-invalidate-region)]
    (try
      (do
        (copy! source mapped-host)
        destination)
      (finally (unmap destination mapped-host)))))

(defmethod print-method CLGeneralMatrix
  [x ^java.io.Writer w]
  (.write w (str x)))

(defn create-vector
  ([engine-factory ^long n cl-buf]
   (let [claccessor (data-accessor engine-factory)]
     (->CLBlockVector engine-factory claccessor
                      (vector-engine engine-factory cl-buf n 0 1)
                      (.entryType ^DataAccessor claccessor) true cl-buf n 0 1)))
  ([engine-factory ^long n]
   (let [claccessor (data-accessor engine-factory)]
     (create-vector engine-factory n
                    (fill-buffer claccessor (create-buffer claccessor n) 1)))))

(defn create-ge-matrix
  ([engine-factory ^long m ^long n cl-buf]
   (let [claccessor (data-accessor engine-factory)]
     (->CLGeneralMatrix engine-factory claccessor
                        (matrix-engine engine-factory cl-buf m n 1)
                        (.entryType ^DataAccessor claccessor) true cl-buf m n m)))

  ([engine-factory ^long m ^long n]
   (let [claccessor (data-accessor engine-factory)]
     (create-ge-matrix engine-factory m n
                       (fill-buffer claccessor (create-buffer claccessor (* m n)) 1)))))
