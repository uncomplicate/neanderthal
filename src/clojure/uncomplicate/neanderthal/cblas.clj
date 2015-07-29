(ns uncomplicate.neanderthal.cblas
  (:require [uncomplicate.neanderthal.block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [uncomplicate.neanderthal.protocols BLAS]))

(def ^:private STRIDE_MSG
  "I cannot use vectors with stride other than %d: stride: %d.")

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  BLAS
  (swap [_ x y];;TODO move the do/return parts to core and real namespace
    (do (CBLAS/dswap (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y))
        x))
  (copy [_ x y]
    (do (CBLAS/dcopy (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y))
        y))
  (dot [_ x y]
    (CBLAS/ddot (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (nrm2 [_ x]
    (CBLAS/dnrm2 (.dim x) (.buffer x) (.stride x)))
  (asum [_ x]
    (CBLAS/dasum (.dim x) (.buffer x) (.stride x)))
  (iamax [_ x]
    (CBLAS/idamax (.dim x) (.buffer x) (.stride x)))
  (rot [_ x y c s]
    (do (CBLAS/drot (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y) c s)
        x))
  (rotg [_ x]
    (do (CBLAS/drotg (.buffer x))
        x))
  (rotm [_ x y p]
    (do (CBLAS/drotm (.dim x) (.buffer x) (.stride x)
                     (.buffer y) (.stride y) (.bufer p))
        x))
  (rotmg [_ p args]
    (do (CBLAS/drotmg (.buffer args) (.buffer p))
        p))
  (scal [_ alpha x]
    (do (CBLAS/dscal (.dim x) alpha (.buffer x) (.stride x))
        x))
  (axpy [_ alpha x y]
    (do (CBLAS/daxpy (.dim x) alpha (.buffer x) (.stride x) (.buffer y) (.stride y))
        y)))

(def dv-engine (DoubleVectorEngine.))

(deftype SingleVectorEngine []
  BLAS
  (swap [_ x y];;TODO move the do/return parts to core and real namespace
    (do (CBLAS/sswap (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y))
        x))
  (copy [_ x y]
    (do (CBLAS/scopy (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y))
        y))
  (dot [_ x y]
    (CBLAS/dsdot (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  (nrm2 [_ x]
    (CBLAS/snrm2 (.dim x) (.buffer x) (.stride x)))
  (asum [_ x]
    (CBLAS/sasum (.dim x) (.buffer x) (.stride x)))
  (iamax [_ x]
    (CBLAS/isamax (.dim x) (.buffer x) (.stride x)))
  (rot [_ x y c s]
    (do (CBLAS/srot (.dim x) (.buffer x) (.stride x) (.buffer y) (.stride y) c s)
        x))
  (rotg [_ x]
    (do (CBLAS/srotg (.buffer x))
        x))
  (rotm [_ x y p]
    (do
      (CBLAS/srotm (.dim x) (.buffer x) (.stride x)
                   (.buffer y) (.stride y) (.bufer p))
        x))
  (rotmg [_ p args]
    (do (CBLAS/srotmg (.buffer args) (.buffer p))
        p))
  (scal [_ alpha x]
    (do (CBLAS/sscal (.dim x) alpha (.buffer x) (.stride x))
        x))
  (axpy [_ alpha x y]
    (do
      (CBLAS/saxpy (.dim x) alpha (.buffer x) (.stride x) (.buffer y) (.stride y))
      y)))

(def sv-engine (SingleVectorEngine.))

;; ================= General Matrix Engines =====================================

(deftype DoubleGeneralMatrixEngine []
  BLAS
  (swap [_ a b] ;;TODO move the do/return parts to core and real namespace
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/dswap (* (.mrows a) (.ncols a)) (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.swap dv-engine (.col a i) (.col b i)))
          (dotimes [i (.mrows a)]
            (.swap dv-engine (.row a i) (.row b i)))))
      a))
  (copy [_ a b]
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/dcopy (* (.mrows a) (.ncols a)) (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.copy dv-engine (.col a i) (.col b i)));;TODO enable raw in C
          (dotimes [i (.mrows a)]
            (.copy dv-engine (.row a i) (.row b i)))))
      b))
  (axpy [_ alpha a b]
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/daxpy (* (.mrows a) (.ncols a)) alpha (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.axpy dv-engine (.col a i) (.col b i)));;TODO enable raw in C
          (dotimes [i (.mrows a)]
            (.axpy dv-engine (.row a i) (.row b i)))))
      b))
  (mv [_ alpha a x beta y]
    (do
      (CBLAS/dgemv (.order a) CBLAS/TRANSPOSE_NO_TRANS (.mrows a) (.ncols a)
                   alpha (.buffer a) (.stride a) (.buffer x) (.stride x)
                   beta (.buffer y) (.stride y))
      y))
  (rank [_ alpha x y a]
    (do
      (CBLAS/dger (.order a) (.mrows a) (.ncols a)
                  alpha (.buffer x) (.stride x) (.buffer y) (.stride y)
                  (.buffer a) (.stride a))
      a))
  (mm [_ alpha a b beta c]
    (do
      (CBLAS/dgemm (.order c)
                   (if (= (.order a) (.order c))
                     CBLAS/TRANSPOSE_NO_TRANS
                     CBLAS/TRANSPOSE_TRANS)
                   (if (= (.order b) (.order c))
                     CBLAS/TRANSPOSE_NO_TRANS
                     CBLAS/TRANSPOSE_TRANS)
                   (.mrows a) (.ncols b) (.ncols a)
                   alpha (.buffer a) (.stride a) (.buffer b) (.stride b)
                   beta (.buffer c) (.stride c))
      c)))

(def dge-engine (DoubleGeneralMatrixEngine.))

(deftype SingleGeneralMatrixEngine []
  BLAS
  (swap [_ a b] ;;TODO move the do/return parts to core and real namespace
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/sswap (* (.mrows a) (.ncols a)) (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.swap sv-engine (.col a i) (.col b i)))
          (dotimes [i (.mrows a)]
            (.swap sv-engine (.row a i) (.row b i)))))
      a))
  (copy [_ a b]
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/scopy (* (.mrows a) (.ncols a)) (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.copy sv-engine (.col a i) (.col b i)));;TODO enable raw in C
          (dotimes [i (.mrows a)]
            (.copy sv-engine (.row a i) (.row b i)))))
      b))
  (axpy [_ alpha a b]
    (do
      (if (and (= (.order a) (.order b))
               (= (if (column-major? a) (.mrows a) (.ncols a))
                  (.order a) (.order b)))
        (CBLAS/saxpy (* (.mrows a) (.ncols a)) alpha (.buffer a) 1 (.buffer b) 1)
        (if (column-major? a)
          (dotimes [i (.ncols a)]
            (.axpy sv-engine (.col a i) (.col b i)));;TODO enable raw in C
          (dotimes [i (.mrows a)]
            (.axpy sv-engine (.row a i) (.row b i)))))
      b))
  (mv [_ alpha a x beta y]
    (do
      (CBLAS/sgemv (.order a) CBLAS/TRANSPOSE_NO_TRANS (.mrows a) (.ncols a)
                   alpha (.buffer a) (.stride a) (.buffer x) (.stride x)
                   beta (.buffer y) (.stride y))
      y))
  (rank [_ a alpha x y]
    (do
      (CBLAS/sger (.order a) (.mrows a) (.ncols a)
                  alpha (.buffer x) (.stride x) (.buffer y) (.stride y)
                  (.buffer a)(.stride a))
      a))
  (mm [_ alpha a b beta c]
    (do
      (CBLAS/sgemm (.order c)
                   (if (= (.order a) (.order c))
                     CBLAS/TRANSPOSE_NO_TRANS
                     CBLAS/TRANSPOSE_TRANS)
                   (if (= (.order b) (.order c))
                     CBLAS/TRANSPOSE_NO_TRANS
                     CBLAS/TRANSPOSE_TRANS)
                   (.mrows a) (.ncols b) (.ncols a)
                   alpha (.buffer a) (.stride a) (.buffer b) (.stride b)
                   beta (.buffer c) (.stride c))
      c)))

(def sge-engine (SingleGeneralMatrixEngine.))
