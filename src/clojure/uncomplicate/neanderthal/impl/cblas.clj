(ns uncomplicate.neanderthal.impl.cblas
  (:require [vertigo.bytes :refer [direct-buffer]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [dim ecount]]
             [block :refer [order buffer offset stride]]]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Vector Matrix RealVector DataAccessor BufferAccessor]))

;; =============== Common vector engine  macros and functions ==================

(defmacro ^:private vector-swap-copy [method x y]
  `(~method (dim ~x) (buffer ~x) 0 (stride ~x) (buffer ~y) 0 (stride ~y)))

(defmacro ^:private vector-axpy [method alpha x y]
  `(~method (dim ~x) ~alpha (buffer ~x) (stride ~x) (buffer ~y) (stride ~y)))

(defmacro ^:private vector-scal [method alpha x]
  `(~method (dim ~x) ~alpha (buffer ~x) (stride ~x)))

(defmacro ^:private vector-rotg [method x]
  `(if (= 1 (stride ~x))
     (~method (buffer ~x))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (stride ~x))))))

(defmacro ^:private vector-rotm [method x y p]
  `(if (= 1 (stride ~p))
     (~method (dim ~x) (buffer ~x) (stride ~x)
      (buffer ~y) (stride ~y) (buffer ~p))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (stride ~p))))))

(defmacro ^:private vector-rotmg [method p args]
  `(if (= 1 (stride ~p) (stride ~args))
     (~method (buffer ~args) (buffer ~p))
     (throw (IllegalArgumentException.
             (format STRIDE_MSG 1 (str (stride ~p) " or " (stride ~args)))))))

(defn ^:private vector-imax [^RealVector x]
  (let [cnt (.dim x)]
    (loop [i 1 max-idx 0 max-val (.entry x 0)]
      (if (< i cnt)
        (let [v (.entry x i)]
          (if (< max-val v)
            (recur (inc i) i v)
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn ^:private vector-imin [^RealVector x]
  (let [cnt (.dim x)]
    (loop [i 1 min-idx 0 min-val (.entry x 0)]
      (if (< i cnt)
        (let [v (.entry x i)]
          (if (< v min-val)
            (recur (inc i) i v)
            (recur (inc i) min-idx min-val)))
        min-idx))))

;; =============== Common matrix macros and functions ==========================

(defmacro ^:private matrix-swap-copy [method a b]
  `(when (< 0 (* (.mrows ~a) (.ncols ~a)))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (.mrows ~a) (.ncols ~a))
                 (stride ~a) (stride ~b)))
       (~method (* (.mrows ~a) (.ncols ~a)) (buffer ~a) 0 1 (buffer ~b) 0 1)
       (if (column-major? ~a)
         (dotimes [i# (.ncols ~a)]
           (vector-swap-copy ~method (.col ~a i#) (.col ~b i#)))
         (dotimes [i# (.mrows ~a)]
           (vector-swap-copy ~method (.row ~a i#) (.row ~b i#)))))))

(defmacro ^:private matrix-scal [method alpha a]
  `(when (< 0 (* (.mrows ~a) (.ncols ~a)))
     (if (= (if (column-major? ~a) (.mrows ~a) (.ncols ~a)) (stride ~a))
       (~method (* (.mrows ~a) (.ncols ~a)) ~alpha (buffer ~a) 1)
       (if (column-major? ~a)
         (dotimes [i# (.ncols ~a)]
           (vector-scal ~method ~alpha (.col ~a i#)))
         (dotimes [i# (.mrows ~a)]
           (vector-scal ~method ~alpha (.row ~a i#)))))))

(defmacro ^:private matrix-axpy [method alpha a b]
  `(when (< 0 (* (.mrows ~a) (.ncols ~a)))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (.mrows ~a) (.ncols ~a))
                 (stride ~a) (stride ~b)))
       (~method (* (.mrows ~a) (.ncols ~a)) ~alpha (buffer ~a) 1 (buffer ~b) 1)
       (if (column-major? ~a)
         (dotimes [i# (.ncols ~a)]
           (vector-axpy ~method ~alpha (.col ~a i#) (.col ~b i#)))
         (dotimes [i# (.mrows ~a)]
           (vector-axpy ~method ~alpha (.row ~a i#) (.row ~b i#)))))))

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/dswap (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (copy [_ x y]
    (CBLAS/dcopy (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (dot [_ x y]
    (CBLAS/ddot (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (nrm2 [_ x]
    (CBLAS/dnrm2 (.dim ^Vector x) (.buffer x) (.stride x)))
  (asum [_ x]
    (CBLAS/dasum (.dim ^Vector x) (.buffer x) (.stride x)))
  (iamax [_ x]
    (CBLAS/idamax (.dim ^Vector x) (.buffer x) (.stride x)))
  (rot [_ x y c s]
    (CBLAS/drot (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y) c s))
  (rotg [_ x]
    (vector-rotg CBLAS/drotg x))
  (rotm [_ x y p]
    (vector-rotm CBLAS/drotm x y p))
  (rotmg [_ p args]
    (vector-rotmg CBLAS/drotmg p args))
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^Vector x)
                 (double alpha) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (CBLAS/dsum (.dim ^Vector x) (.buffer x) (.stride x)))
  (imax [_ x]
    (vector-imax x))
  (imin [_ x]
    (vector-imin x)))

(deftype SingleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/sswap (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (copy [_ x y]
    (CBLAS/scopy (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (dot [_ x y]
    (CBLAS/dsdot (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (nrm2 [_ x]
    (CBLAS/snrm2 (.dim ^Vector x) (.buffer x) (.stride x)))
  (asum [_ x]
    (CBLAS/sasum (.dim ^Vector x) (.buffer x) (.stride x)))
  (iamax [_ x]
    (CBLAS/isamax (.dim ^Vector x) (.buffer x) (.stride x)))
  (rot [_ x y c s]
    (CBLAS/srot (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y) c s))
  (rotg [_ x]
    (vector-rotg CBLAS/srotg x))
  (rotm [_ x y p]
    (vector-rotm CBLAS/srotm x y p))
  (rotmg [_ p args]
    (vector-rotmg CBLAS/srotmg p args))
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^Vector x)
                 (float alpha) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (CBLAS/ssum (.dim ^Vector x) (.buffer x) (.stride x)))
  (imax [_ x]
    (vector-imax x))
  (imin [_ x]
    (vector-imin x)))

;; ================= General Matrix Engines ====================================

(deftype DoubleGeneralMatrixEngine []
  BLAS
  (swap [_ a b]
    (matrix-swap-copy CBLAS/dswap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy CBLAS/dcopy ^Matrix a ^Matrix b))
  (scal [_ alpha a]
    (matrix-scal CBLAS/dscal alpha ^Matrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/daxpy alpha ^Matrix a ^Matrix b))
  (mv [_ alpha a x beta y]
    (CBLAS/dgemv (.order a) CBLAS/TRANSPOSE_NO_TRANS
                 (.mrows ^Matrix a) (.ncols ^Matrix a)
                 alpha (.buffer a) (.stride a) (.buffer x) (.stride x)
                 beta (.buffer y) (.stride y)))
  (rank [_ alpha x y a]
    (CBLAS/dger (.order a) (.mrows ^Matrix a) (.ncols ^Matrix a)
                alpha (.buffer x) (.stride x) (.buffer y) (.stride y)
                (.buffer a) (.stride a)))
  (mm [_ alpha a b beta c]
    (CBLAS/dgemm (.order c)
                 (if (= (.order a) (.order c))
                   CBLAS/TRANSPOSE_NO_TRANS
                   CBLAS/TRANSPOSE_TRANS)
                 (if (= (.order b) (.order c))
                   CBLAS/TRANSPOSE_NO_TRANS
                   CBLAS/TRANSPOSE_TRANS)
                 (.mrows ^Matrix a) (.ncols ^Matrix b) (.ncols ^Matrix a)
                 alpha (.buffer a) (.stride a) (.buffer b) (.stride b)
                 beta (.buffer c) (.stride c))))

(deftype SingleGeneralMatrixEngine []
  BLAS
  (swap [_ a b]
    (matrix-swap-copy CBLAS/sswap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy CBLAS/scopy ^Matrix a ^Matrix b))
  (scal [_ alpha a]
    (matrix-scal CBLAS/sscal alpha ^Matrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/saxpy alpha ^Matrix a ^Matrix b))
  (mv [_ alpha a x beta y]
    (CBLAS/sgemv (.order a) CBLAS/TRANSPOSE_NO_TRANS
                 (.mrows ^Matrix a) (.ncols ^Matrix a)
                 alpha (.buffer a) (.stride a) (.buffer x) (.stride x)
                 beta (.buffer y) (.stride y)))
  (rank [_ alpha x y a]
    (CBLAS/sger (.order a) (.mrows ^Matrix a) (.ncols ^Matrix a)
                alpha (.buffer x) (.stride x) (.buffer y) (.stride y)
                (.buffer a) (.stride a)))
  (mm [_ alpha a b beta c]
    (CBLAS/sgemm (.order c)
                 (if (= (.order a) (.order c))
                   CBLAS/TRANSPOSE_NO_TRANS
                   CBLAS/TRANSPOSE_TRANS)
                 (if (= (.order b) (.order c))
                   CBLAS/TRANSPOSE_NO_TRANS
                   CBLAS/TRANSPOSE_TRANS)
                 (.mrows ^Matrix a) (.ncols ^Matrix b) (.ncols ^Matrix a)
                 alpha (.buffer a) (.stride a) (.buffer b) (.stride b)
                 beta (.buffer c) (.stride c))))

(deftype CblasFactory [^DataAccessor acc ^BLAS vector-eng ^BLAS matrix-eng]
  DataAccessorProvider
  (data-accessor [_]
    acc)
  MemoryContext
  (compatible [_ o]
    (compatible acc o))
  Factory
  (create-vector [this n buf _]
    (if (and (<= 0 (long n) (.count acc buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (->RealBlockVector this acc vector-eng (.entryType acc) true buf n 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count acc buf) (class buf))))))
  (create-matrix [this m n buf order]
    (if (and (<= 0 (* (long m) (long n)) (.count acc buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (let [order (or order DEFAULT_ORDER)
            ld (max (long (if (= COLUMN_MAJOR order) m n)) 1)]
        (->RealGeneralMatrix this acc matrix-eng (.entryType acc) true
                             buf m n ld order))
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d matrix from %s."
                      m n (type buf))))))
  (vector-engine [_]
    vector-eng)
  (matrix-engine [_]
    matrix-eng))

(def cblas-single
  (->CblasFactory (->FloatBufferAccessor)
                  (->SingleVectorEngine)
                  (->SingleGeneralMatrixEngine)))

(def cblas-double
  (->CblasFactory (->DoubleBufferAccessor)
                  (->DoubleVectorEngine)
                  (->DoubleGeneralMatrixEngine)))
