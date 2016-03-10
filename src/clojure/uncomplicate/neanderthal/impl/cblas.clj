(ns uncomplicate.neanderthal.impl.cblas
  (:require [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Vector Matrix RealVector DataAccessor BufferAccessor]))

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
    (if (= 1 (.stride x))
      (CBLAS/drotg (.buffer x))
      (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride x))))))
  (rotm [_ x y p]
    (if (= 1 (.stride p))
      (CBLAS/drotm (.dim ^Vector x) (.buffer x) (.stride x)
                   (.buffer y) (.stride y) (.buffer p))
      (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride p))))))
  (rotmg [_ p args]
    (if (= 1 (.stride p) (.stride args))
      (CBLAS/drotmg (.buffer args) (.buffer p))
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (str (.stride p) " or " (.stride args)))))))
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^Vector x) alpha (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (let [cnt (.dim ^Vector x)]
      (loop [i 0 res 0.0]
        (if (< i cnt)
          (recur (inc i)
                 (+ res (.entry ^RealVector x i)))
          res))))
  (imax [_ x]
    (let [cnt (.dim ^Vector x)]
      (loop [i 1 max-idx 0 max-val (.entry ^RealVector x 0)]
        (if (< i cnt)
          (let [v (.entry ^RealVector x i)]
            (if (< max-val v)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx))))
  (imin [_ x]
    (let [cnt (.dim ^Vector x)]
      (loop [i 1 min-idx 0 min-val (.entry ^RealVector x 0)]
        (if (< i cnt)
          (let [v (.entry ^RealVector x i)]
            (if (< v min-val)
              (recur (inc i) i v)
              (recur (inc i) min-idx min-val)))
          min-idx)))))

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
    (if (= 1 (.stride x))
      (CBLAS/srotg (.buffer x))
      (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride x))))))
  (rotm [_ x y p]
    (if (= 1 (.stride p))
      (CBLAS/srotm (.dim ^Vector x) (.buffer x) (.stride x)
                   (.buffer y) (.stride y) (.buffer p))
      (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride p))))))
  (rotmg [_ p args]
    (if (= 1 (.stride p) (.stride args))
      (CBLAS/srotmg (.buffer args) (.buffer p))
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (str (.stride p) " or " (.stride args)))))))
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^Vector x) alpha (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (loop [i 0 res 0.0]
      (if (< i (.dim ^Vector x))
        (recur (inc i)
               (+ res (.entry ^RealVector x i)))
        res)))
  (imax [_ x]
    (let [cnt (.dim ^Vector x)]
      (loop [i 1 max-idx 0 max-val (.entry ^RealVector x 0)]
        (if (< i cnt)
          (let [v (.entry ^RealVector x i)]
            (if (< max-val v)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx))))
  (imin [_ x]
    (let [cnt (.dim ^Vector x)]
      (loop [i 1 min-idx 0 min-val (.entry ^RealVector x 0)]
        (if (< i cnt)
          (let [v (.entry ^RealVector x i)]
            (if (< v min-val)
              (recur (inc i) i v)
              (recur (inc i) min-idx min-val)))
          min-idx)))))

;; ================= General Matrix Engines ====================================

(deftype DoubleGeneralMatrixEngine [^BLAS dv-engine]
  BLAS
  (swap [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/dswap (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 0 1 (.buffer b) 0 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.swap dv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.swap dv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (copy [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/dcopy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 0 1 (.buffer b) 0 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.copy dv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.copy dv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (axpy [_ alpha a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/daxpy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   alpha (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.axpy dv-engine alpha (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.axpy dv-engine alpha (.row ^Matrix a i) (.row ^Matrix b i))))))
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

(deftype SingleGeneralMatrixEngine [^BLAS sv-engine]
  BLAS
  (swap [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/sswap (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 0 1 (.buffer b) 0 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.swap sv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.swap sv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (copy [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/scopy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 0 1 (.buffer b) 0 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.copy sv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.copy sv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (axpy [_ alpha a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.stride a) (.stride b)))
      (CBLAS/saxpy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   alpha (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.axpy sv-engine alpha (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.axpy sv-engine alpha (.row ^Matrix a i) (.row ^Matrix b i))))))
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
  (data-accessor [_]
    acc)
  (vector-engine [_ _]
    vector-eng)
  (matrix-engine [_ _]
    matrix-eng))

(def cblas-single
  (let [vect-eng (->SingleVectorEngine)]
    (->CblasFactory (->FloatBufferAccessor) vect-eng
                    (->SingleGeneralMatrixEngine vect-eng))))

(def cblas-double
  (let [vect-eng (->DoubleVectorEngine)]
    (->CblasFactory (->DoubleBufferAccessor) vect-eng
                    (->DoubleGeneralMatrixEngine vect-eng))))
