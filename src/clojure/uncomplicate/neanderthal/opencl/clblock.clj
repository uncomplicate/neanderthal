(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.clblock
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.core
             :refer [INCOMPATIBLE_BLOCKS_MSG COLUMN_MAJOR]])
  (:import [uncomplicate.neanderthal.protocols BLAS
            Vector Matrix Changeable Block
            Group Memory BlockCreator EngineProvider]))

(defprotocol Mappable
  (read! [this host])
  (write! [this host])
  (map-host [this])
  (unmap [this]))

(defprotocol EngineFactory
  (cl-accessor [this])
  (vector-engine [this buf n])
  (matrix-engine [this cl-buf m n]))

;; ================== Accessors ================================================

(defprotocol CLAccessor
  (get-queue [this])
  (entryType [this])
  (width [this])
  (create-buffer [this n])
  (fill-buffer [this cl-buf val])
  (array [this s])
  (slice [this cl-buf k l]))

(set! *unchecked-math* false)

(deftype TypedAccessor [ctx queue et ^long w array-fn]
  CLAccessor
  (get-queue [_]
    queue)
  (entryType [_]
    et)
  (width [_]
    w)
  (create-buffer [_ n]
    (let [res (cl-buffer ctx (* w n) :read-write)]
      (enq-fill! queue res (array-fn 1))
      res))
  (fill-buffer [_ cl-buf v]
    (do
      (enq-fill! queue cl-buf (array-fn v))
      cl-buf))
  (array [_ s]
    (array-fn s))
  (slice [_ cl-buf k l]
    (cl-sub-buffer cl-buf (* w k) (* w l))))

(set! *unchecked-math* :warn-on-boxed)

(defn float-accessor [ctx queue]
  (->TypedAccessor ctx queue Float/TYPE 4 float-array))

(defn double-accessor [ctx queue]
  (->TypedAccessor ctx queue Double/TYPE 8 double-array))

;; =============================================================================

(declare clv)
(declare clge)

(deftype CLBlockVector [engine-factory claccessor eng entry-type
                        cl-buf ^long n ^long strd]
  Object
  (toString [_]
    (format "#<CLBlockVector| %s, n:%d, stride:%d>" entry-type n strd))
  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release eng)))
  Group
  (zero [_]
    (clv engine-factory n))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ y]
    (and (instance? CLBlockVector y)
         (= entry-type (.entryType ^Block y))))
  BlockCreator
  (create-block [_ m n]
    (clge engine-factory m n))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    cl-buf)
  (stride [_]
    strd)
  (count [_]
    n)
  Changeable
  (setBoxed [x val]
    (do
      (fill-buffer claccessor cl-buf [val])
      x))
  Vector
  (dim [_]
    n)
  (subvector [_ k l]
    (let [buf-slice (slice claccessor cl-buf (* k strd) (* l strd))]
      (CLBlockVector. engine-factory claccessor
                      (vector-engine engine-factory buf-slice l)
                      entry-type buf-slice l strd)))
  Mappable
  (read! [this host]
    (if (and (instance? Vector host) (= entry-type (.entryType ^Block host)))
      (do
        (enq-read! (get-queue claccessor) cl-buf (.buffer ^Block host))
        host)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host)))))
  (write! [this host]
    (if (and (instance? Vector host) (= entry-type (.entryType ^Block host)))
      (do
        (enq-write! (get-queue claccessor) cl-buf (.buffer ^Block host))
        this)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host))))))

(defmethod print-method CLBlockVector
  [x ^java.io.Writer w]
  (.write w (str x)))

(deftype CLGeneralMatrix [engine-factory claccessor eng entry-type
                          cl-buf ^long m ^long n ^long ld]
  Object
  (toString [_]
    (format "#<GeneralMatrixCL| %s, %s, mxn: %dx%d, ld:%d>"
            entry-type "COL" m n ld))
  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release eng)))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ y]
    (and (or (instance? CLGeneralMatrix y) (instance? CLBlockVector y))
         (= entry-type (.entryType ^Block y))))
  BlockCreator
  (create-block [_ m n]
    (clge engine-factory m n))
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
  Changeable
  (setBoxed [x val]
    (do
      (fill-buffer claccessor cl-buf [val])
      x))
  Mappable
  (read! [this host]
    (if (and (instance? Matrix host) (= entry-type (.entryType ^Block host)))
      (do
        (enq-read! (get-queue claccessor) cl-buf (.buffer ^Block host))
        host)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host)))))
  (write! [this host]
    (if (and (instance? Matrix host) (= entry-type (.entryType ^Block host)))
      (do
        (enq-write! (get-queue claccessor) cl-buf (.buffer ^Block host))
        this)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host)))))
  Matrix
  (mrows [_]
    m)
  (ncols [_]
    n))

(defmethod print-method CLGeneralMatrix
  [x ^java.io.Writer w]
  (.write w (str x)))

(defn clv
  ([engine-factory ^long n cl-buf]
   (let [claccessor (cl-accessor engine-factory)]
     (->CLBlockVector engine-factory claccessor (vector-engine engine-factory cl-buf n)
                      (entryType claccessor) cl-buf n 1)))
  ([engine-factory ^long n]
   (clv engine-factory n (create-buffer (cl-accessor engine-factory) n))))

(defn clge
  ([engine-factory ^long m ^long n cl-buf]
   (let [claccessor (cl-accessor engine-factory)]
     (->CLGeneralMatrix engine-factory claccessor (matrix-engine engine-factory cl-buf m n)
                        (entryType claccessor) cl-buf m n m)))

  ([engine-factory ^long m ^long n]
   (clge engine-factory m n (create-buffer (cl-accessor engine-factory) (* m n)))))
