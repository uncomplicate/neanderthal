(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.clblock
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.impl.buffer-block :refer [COLUMN_MAJOR]])
  (:import [uncomplicate.neanderthal.protocols
            BLAS Vector Matrix Changeable Block DataAccessor]))

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

;; =============================================================================

(declare create-vector)
(declare create-ge-matrix)

(deftype CLBlockVector [engine-factory ^DataAccessor claccessor eng entry-type
                        cl-buf ^long n ^long ofst ^long strd]
  Object
  (toString [_]
    (format "#<CLBlockVector| %s, n:%d, offset:%d stride:%d>"
            entry-type n ofst strd))
  Releaseable
  (release [_]
    (and
     (release cl-buf)
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
  Changeable
  (setBoxed [x val]
    (do;;TODO now when there are offset and stride, this must be a kernel
      (fill-buffer claccessor cl-buf [val])
      x))
  Vector
  (dim [_]
    n)
  (subvector [_ k l]
    (CLBlockVector. engine-factory claccessor
                    (vector-engine engine-factory cl-buf l (+ ofst k) strd)
                    entry-type cl-buf l (+ ofst k) strd))
  Mappable
  (read! [this host]
    (if (and (instance? Vector host)
             (= entry-type (.entryType ^Block host))
             (= 1 strd) (= 0 ofst));;TODO use enq-map for this
      (do
        (enq-read! (get-queue claccessor) cl-buf (.buffer ^Block host)
                   true (* ofst (.entryWidth claccessor)) nil nil)
        host)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host)))))
  (write! [this host]
    (if (and (instance? Vector host) (= entry-type (.entryType ^Block host)) (= 1 strd) (= 0 ofst));;TODO
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
    (format "#<CLGeneralMatrix| %s, %s, mxn: %dx%d, ld:%d>"
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
  Changeable
  (setBoxed [x val]
    (do
      (fill-buffer claccessor cl-buf [val]);;TODO offset and stride
      x))
  Mappable
  (read! [this host]
    (if (and (instance? Matrix host) (= entry-type (.entryType ^Block host)));;TODO
      (do
        (enq-read! (get-queue claccessor) cl-buf (.buffer ^Block host))
        host)
      (throw (IllegalArgumentException.
              (format INCOMPATIBLE_BLOCKS_MSG this host)))))
  (write! [this host]
    (if (and (instance? Matrix host) (= entry-type (.entryType ^Block host)));;TODO
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

(defn create-vector
  ([engine-factory ^long n cl-buf]
   (let [claccessor (data-accessor engine-factory)]
     (->CLBlockVector engine-factory claccessor
                      (vector-engine engine-factory cl-buf n 0 1)
                      (.entryType ^DataAccessor claccessor) cl-buf n 0 1)))
  ([engine-factory ^long n]
   (let [claccessor (data-accessor engine-factory)]
     (create-vector engine-factory n
                    (fill-buffer claccessor (create-buffer claccessor n) 1)))))

(defn create-ge-matrix
  ([engine-factory ^long m ^long n cl-buf]
   (let [claccessor (data-accessor engine-factory)]
     (->CLGeneralMatrix engine-factory claccessor
                        (matrix-engine engine-factory cl-buf m n 1)
                        (.entryType ^DataAccessor claccessor) cl-buf m n m)))

  ([engine-factory ^long m ^long n]
   (let [claccessor (data-accessor engine-factory)]
     (create-ge-matrix engine-factory m n
                       (fill-buffer claccessor (create-buffer claccessor (* m n)) 1)))))
