(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.opencl.clblast
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release
                     wrap-int wrap-double wrap-float]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [enq-read-int enq-read-double enq-read-float]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [block :refer [buffer offset stride order]]
             [core :refer [copy dim ecount mrows ncols]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [org.jocl.blast CLBlast Transpose]
           [uncomplicate.clojurecl.core CLBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Vector Matrix Block DataAccessor]))

(defn ^:private equals-block-vector [ctx queue prog x y]
  (if (< 0 (dim x))
    (with-release [equals-vector-kernel (kernel prog "equals_vector")
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-vector-kernel eq-flag-buf
                   (buffer x) (wrap-int (offset x)) (wrap-int (stride x))
                   (buffer y) (wrap-int (offset y)) (wrap-int (stride y)))
        (enq-nd! queue equals-vector-kernel (work-size-1d (dim x)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    true))

(defn ^:private equals-block-matrix [ctx queue prog a b]
  (if (and (= (order a) (order b))
           (= (if (column-major? a) (mrows a) (ncols a))
              (stride a) (stride b)))
    (with-release [equals-matrix-kernel (kernel prog "equals_matrix")
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-matrix-kernel eq-flag-buf
                   (buffer a) (wrap-int (offset a)) (wrap-int (stride a))
                   (buffer b) (wrap-int (offset b)) (wrap-int (stride b)))
        (enq-nd! queue equals-matrix-kernel
                 (work-size-2d (mrows a) (ncols a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (if (column-major? a)
      (let [ncols-a (ncols a)]
        (loop [i 0]
          (if (= i ncols-a)
            true
            (if (with-release [x (.col ^Matrix a i)
                               y (.col ^Matrix b i)]
                  (= x y))
              (recur (inc i))
              false))))
      (let [mrows-a (mrows a)]
        (loop [i 0]
          (if (= i mrows-a)
            true
            (if (with-release [x (.row ^Matrix a i)
                               y (.row ^Matrix b i)]
                  (= x y))
              (recur (inc i))
              false)))))))

(defmacro ^:private vector-swap-copy [queue method x y]
  `(if (< 0 (dim ~x))
     (~method
      (dim ~x)
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
      ~queue nil)
     ~queue))

(defmacro ^:private vector-dot [ctx queue res-bytes read-method method x y]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (~method
        (dim ~x) (cl-mem res-buffer#) 0
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        ~queue nil)
       (~read-method ~queue res-buffer#))
     0.0))

(defmacro ^:private vector-sum-nrm2 [ctx queue res-bytes read-method method x]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (~method
        (dim ~x) (cl-mem res-buffer#) 0
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        ~queue nil)
       (~read-method ~queue res-buffer#))
     0.0))

(defmacro ^:private vector-ipeak [ctx queue method x]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx Integer/BYTES :read-write)]
       (~method
        (dim ~x) (cl-mem res-buffer#) 0
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        ~queue nil)
       (enq-read-int ~queue res-buffer#))
     0))

(defmacro ^:private vector-scal [queue method alpha x]
  `(if (< 0 (dim ~x))
     (~method
      (dim ~x) ~alpha
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      ~queue nil)
     ~queue))

(defmacro ^:private vector-axpy [queue method alpha x y]
  `(if (< 0 (dim ~x))
     (~method
      (dim ~x) ~alpha
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
      ~queue nil)
     ~queue))

(defmacro ^:private vector-subcopy [queue method x y kx lx ky]
  `(if (< 0 ~lx)
     (~method
      ~lx
      (cl-mem (buffer ~x)) ~kx (stride ~x)
      (cl-mem (buffer ~y)) ~ky (stride ~y)
      ~queue nil)
     ~queue))

(defmacro ^:private matrix-swap-copy [queue vector-eng method vector-method a b]
  `(if (< 0 (ecount ~a))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (mrows ~a) (ncols ~a))
                 (stride ~a) (stride ~b)))
       (~method
        (ecount ~a)
        (cl-mem (buffer ~a)) (offset ~a) 1 (cl-mem (buffer ~b)) (offset ~b) 1
        ~queue nil)
       (if (column-major? ~a)
         (dotimes [i# (ncols ~a)]
           (with-release [x# (.col ~a i#)
                          y# (.col ~b i#)]
             (. ~vector-eng ~vector-method x# y#)))
         (dotimes [i# (mrows ~a)]
           (with-release [x# (.row ~a i#)
                          y# (.row ~b i#)]
             (. ~vector-eng ~vector-method x# y#)))))
     ~queue))

(defmacro ^:private matrix-axpy [queue vector-eng method alpha a b]
  `(if (< 0 (ecount ~a))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (mrows ~a) (ncols ~a))
                 (stride ~a) (stride ~b)))
       (~method
        (ecount ~a)
        ~alpha (cl-mem (buffer ~a)) (offset ~a) 1
        (cl-mem (buffer ~b)) (offset ~b) 1
        ~queue nil)
       (if (column-major? ~a)
         (dotimes [i# (ncols ~a)]
           (with-release [x# (.col ~a i#)
                          y# (.col ~b i#)]
             (. ~vector-eng axpy ~alpha x# y#)))
         (dotimes [i# (mrows ~a)]
           (with-release [x# (.row ~a i#)
                          y# (.row ~b i#)]
             (. ~vector-eng axpy ~alpha x# y#)))))
     ~queue))

(defmacro ^:private matrix-gemv [queue method alpha a x beta y]
  `(if (< 0 (ecount ~a))
     (~method
      (order ~a) Transpose/kNo (mrows ~a) (ncols ~a)
      ~alpha (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      ~beta (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
      ~queue nil)
     ~queue))
(defmacro ^:private matrix-ger [queue method alpha x y a]
  `(if (< 0 (ecount ~a))
     (~method
      (order ~a) (mrows ~a) (ncols ~a)
      ~alpha
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
      (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
      ~queue nil)
     ~queue))

(defmacro ^:private matrix-gemm [queue method alpha a b beta c]
  `(if (< 0 (ecount ~a))
     (~method
      (order ~a)
      (if (= (order ~a) (order ~c)) Transpose/kNo Transpose/kYes)
      (if (= (order ~b) (order ~c)) Transpose/kNo Transpose/kYes)
      (mrows ~a) (ncols ~b) (ncols ~a)
      ~alpha (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
      (cl-mem (buffer ~b)) (offset ~b) (stride ~b)
      ~beta (cl-mem (buffer ~c)) (offset ~c) (stride ~c)
      ~queue nil)
     ~queue))

(deftype CLBlastDoubleVectorEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ x y]
    (equals-block-vector ctx queue prog x y))
  BLAS
  (swap [_ x y]
    (vector-swap-copy queue CLBlast/CLBlastDswap x y))
  (copy [_ x y]
    (vector-swap-copy queue CLBlast/CLBlastDcopy x y))
  (dot [_ x y]
    (vector-dot ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDdot x y))
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDnrm2 x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDasum x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDamax x))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException. "TODO.")))
  (scal [_ alpha x]
    (vector-scal queue CLBlast/CLBlastDscal alpha x))
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastDaxpy alpha x y))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastDcopy x y kx lx ky))
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDsum x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmax x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmin x)))

(deftype CLBlastSingleVectorEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ x y]
    (equals-block-vector ctx queue prog x y))
  BLAS
  (swap [_ x y]
    (vector-swap-copy queue CLBlast/CLBlastSswap x y))
  (copy [_ x y]
    (vector-swap-copy queue CLBlast/CLBlastScopy x y))
  (dot [_ x y]
    (vector-dot ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSdot x y))
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSnrm2 x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSasum x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSamax x))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException. "TODO.")))
  (scal [_ alpha x]
    (vector-scal queue CLBlast/CLBlastSscal alpha x))
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastSaxpy alpha x y))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastScopy x y kx lx ky))
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSsum x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmax x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmin x)))

(deftype CLBlastDoubleGeneralMatrixEngine [ctx queue prog ^BLAS vector-eng]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-block-matrix ctx queue prog a b))
  BLAS
  (swap [_ a b]
    (matrix-swap-copy queue vector-eng CLBlast/CLBlastDswap swap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy queue vector-eng CLBlast/CLBlastDcopy copy ^Matrix a ^Matrix b))
  (axpy [_ alpha a b]
    (matrix-axpy queue vector-eng CLBlast/CLBlastDaxpy alpha ^Matrix a ^Matrix b))
  (mv [_ alpha a x beta y]
    (matrix-gemv queue CLBlast/CLBlastDgemv alpha a x beta y))
  (rank [_ alpha x y a]
    (matrix-ger queue CLBlast/CLBlastDger alpha x y a))
  (mm [_ alpha a b beta c]
    (matrix-gemm queue CLBlast/CLBlastDgemm alpha a b beta c)))

(deftype CLBlastSingleGeneralMatrixEngine [ctx queue prog ^BLAS vector-eng]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-block-matrix ctx queue prog a b))
  BLAS
  (swap [_ a b]
    (matrix-swap-copy queue vector-eng CLBlast/CLBlastSswap swap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy queue vector-eng CLBlast/CLBlastScopy copy ^Matrix a ^Matrix b))
  (axpy [_ alpha a b]
    (matrix-axpy queue vector-eng CLBlast/CLBlastSaxpy alpha ^Matrix a ^Matrix b ))
  (mv [_ alpha a x beta y]
    (matrix-gemv queue CLBlast/CLBlastSgemv alpha a x beta y))
  (rank [_ alpha x y a]
    (matrix-ger queue CLBlast/CLBlastSger alpha x y a))
  (mm [_ alpha a b beta c]
    (matrix-gemm queue CLBlast/CLBlastSgemm alpha a b beta c)))

(deftype CLFactory [ctx queue prog ^DataAccessor claccessor vector-eng matrix-eng]
  Releaseable
  (release [_]
    (and
     (= 0 (CLBlast/CLBlastClearCache))
     (release prog)
     (release vector-eng)
     (release matrix-eng)))
  FactoryProvider
  (factory [_]
    (factory claccessor))
  Factory
  (create-vector [this n buf _]
    (if (and (<= 0 (long n) (.count claccessor buf)) (instance? CLBuffer buf))
      (->CLBlockVector this claccessor vector-eng (.entryType claccessor) true
                       buf n 0 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (ecount claccessor buf) (class buf))))))
  (create-matrix [this m n buf order]
    (if (and (<= 0 (* (long m) (long n)) (.count claccessor buf))
             (instance? CLBuffer buf))
      (let [order (or order DEFAULT_ORDER)
            ld (max (long (if (= COLUMN_MAJOR order) m n)) 1)]
        (->CLGeneralMatrix this claccessor matrix-eng (.entryType claccessor)
                           true buf m n 0 ld order))
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d matrix from %s."
                      m n (type buf))))))
  (data-accessor [_]
    claccessor)
  (vector-engine [_]
    vector-eng)
  (matrix-engine [_]
    matrix-eng))

(let [src [(slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/equals.cl"))]]

  (org.jocl.blast.CLBlast/setExceptionsEnabled true)

  (defn clblast-double [ctx queue]
    (let [accessor (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES
                                      double-array wrap-double cblas-double)
          prog (build-program! (program-with-source ctx src) "-DREAL=double" nil)
          vector-eng (->CLBlastDoubleVectorEngine ctx queue prog)]
      (->CLFactory
       ctx queue prog accessor
       vector-eng
       (->CLBlastDoubleGeneralMatrixEngine ctx queue prog vector-eng))))

  (defn clblast-single [ctx queue]
    (let [accessor (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES
                                      float-array wrap-float cblas-single)
          prog (build-program! (program-with-source ctx src) "-DREAL=float" nil)
          vector-eng (->CLBlastSingleVectorEngine ctx queue prog)]
      (->CLFactory
       ctx queue prog accessor
       vector-eng
       (->CLBlastSingleGeneralMatrixEngine ctx queue prog vector-eng)))))
