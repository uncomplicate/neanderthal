(ns uncomplicate.neanderthal.impl.cblas
  (:require [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Vector Matrix RealVector]))

(def ^:private STRIDE_MSG
  "I cannot use vectors with stride other than %d: stride: %d.")

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/dswap (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (copy [_ x y]
    (CBLAS/dcopy (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
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
  (sum [_ x] ;;TODO implement in C
    (loop [i 0 res 0.0]
      (if (< i (.dim ^Vector x))
        (recur (inc i)
               (+ res (.entry ^RealVector x i)))
        res))))

(deftype SingleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/sswap (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (copy [_ x y]
    (CBLAS/scopy (.dim ^Vector x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
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
  (sum [_ x] ;;TODO implement in C
    (loop [i 0 res 0.0]
      (if (< i (.dim ^Vector x))
        (recur (inc i)
               (+ res (.entry ^RealVector x i)))
        res))))

;; ================= General Matrix Engines ====================================

(deftype DoubleGeneralMatrixEngine [^BLAS dv-engine]
  BLAS
  (swap [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.order a) (.order b)))
      (CBLAS/dswap (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.swap dv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.swap dv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (copy [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.order a) (.order b)))
      (CBLAS/dcopy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.copy dv-engine (.col ^Matrix a i) (.col ^Matrix b i)));;TODO enable raw in C
        (dotimes [i (.mrows ^Matrix a)]
          (.copy dv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (axpy [_ alpha a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.order a) (.order b)))
      (CBLAS/daxpy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   alpha (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.axpy dv-engine alpha (.col ^Matrix a i) (.col ^Matrix b i)));;TODO enable raw in C
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
                (.order a) (.order b)))
      (CBLAS/sswap (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.swap sv-engine (.col ^Matrix a i) (.col ^Matrix b i)))
        (dotimes [i (.mrows ^Matrix a)]
          (.swap sv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (copy [_ a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.order a) (.order b)))
      (CBLAS/scopy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.copy sv-engine (.col ^Matrix a i) (.col ^Matrix b i)));;TODO enable raw in C
        (dotimes [i (.mrows ^Matrix a)]
          (.copy sv-engine (.row ^Matrix a i) (.row ^Matrix b i))))))
  (axpy [_ alpha a b]
    (if (and (= (.order a) (.order b))
             (= (if (column-major? a) (.mrows ^Matrix a) (.ncols ^Matrix a))
                (.order a) (.order b)))
      (CBLAS/saxpy (* (.mrows ^Matrix a) (.ncols ^Matrix a))
                   alpha (.buffer a) 1 (.buffer b) 1)
      (if (column-major? a)
        (dotimes [i (.ncols ^Matrix a)]
          (.axpy sv-engine alpha (.col ^Matrix a i) (.col ^Matrix b i)));;TODO enable raw in C
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


(deftype CblasEngineFactory [acc ^BLAS vector-eng ^BLAS matrix-eng]
  EngineFactory
  (data-accessor [_]
    acc)
  (vector-engine [_ buf n ofst strd]
    vector-eng)
  (matrix-engine [_ buf m n ofst ld]
    matrix-eng))

(def cblas-single
  (let [vect-eng (->SingleVectorEngine)]
    (->CblasEngineFactory (->FloatBufferAccessor)
                          vect-eng
                          (->SingleGeneralMatrixEngine vect-eng))))

(def cblas-double
  (let [vect-eng (->DoubleVectorEngine)]
    (->CblasEngineFactory (->DoubleBufferAccessor)
                          vect-eng
                          (->DoubleGeneralMatrixEngine vect-eng))))
