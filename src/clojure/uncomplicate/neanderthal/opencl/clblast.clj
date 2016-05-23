(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.opencl.clblast
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release
                     wrap-int wrap-double wrap-float]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [constants :refer [dec-error]]
             [utils :refer [with-check]]
             [toolbox :refer [enq-read-int enq-read-double enq-read-float]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [block :refer [buffer offset stride order]]
             [core :refer [copy dim ecount mrows ncols]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [org.jocl.blast CLBlast Transpose]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Vector Matrix Block DataAccessor]))

;; =============== OpenCL and CLBlast error handling functions =================

(defn ^:private dec-clblast-error
  "Decodes CLBlast error code to a meaningful string.
  If called with a number that is not recognized as an existing OpenCL error,
  returns nil."
  [^long code]
  (case code
    -1024 "kNotImplemented"
    -1022 "kInvalidMatrixA"
    -1021 "kInvalidMatrixB"
    -1020 "kInvalidMatrixC"
    -1019 "kInvalidVectorX"
    -1018 "kInvalidVectorY"
    -1017 "kInvalidDimension"
    -1016 "kInvalidLeadDimA"
    -1015 "kInvalidLeadDimB"
    -1014 "kInvalidLeadDimC"
    -1013 "kInvalidIncrementX"
    -1012 "kInvalidIncrementY"
    -1011 "kInsufficientMemoryA"
    -1010 "kInsufficientMemoryB"
    -1009 "kInsufficientMemoryC"
    -1008 "kInsufficientMemoryX"
    -1007 "kInsufficientMemoryY"
    -2048 "kKernellaunchError"
    -2047 "kKernelRunError"
    -2046 "kInvalidLocalMemUsage"
    -2045 "kNoHalfPrecision"
    -2044 "kNoDoublePrecision"
    -2043 "kInvalidVectorDot"
    -2042 "kInsufficientMemoryDot"
    nil))

(defn ^:private error [^long err-code details]
  (if-let [err (dec-clblast-error err-code)]
    (ex-info (format "CLBlast error: %s." err)
             {:name err :code err-code :type :clblast-error :details details})
    (let [err (dec-error err-code)]
      (ex-info (format "OpenCL error: %s." err)
               {:name err :code err-code :type :opencl-error :details details}))))

;; =============== Common vector engine  macros and functions ==================

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
            (if (= (.col ^Matrix a i) (.col ^Matrix b i))
              (recur (inc i))
              false))))
      (let [mrows-a (mrows a)]
        (loop [i 0]
          (if (= i mrows-a)
            true
            (if (= (.row ^Matrix a i) (.row ^Matrix b i))
              (recur (inc i))
              false)))))))

(defmacro ^:private vector-swap-copy [queue method x y]
  `(when (< 0 (dim ~x))
     (with-check error
       (~method
        (dim ~x)
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        ~queue nil)
       nil)))

(defmacro ^:private vector-dot [ctx queue res-bytes read-method method x y]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (with-check error
         (~method
          (dim ~x) (cl-mem res-buffer#) 0
          (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
          (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
          ~queue nil)
         (~read-method ~queue res-buffer#)))
     0.0))

(defmacro ^:private vector-sum-nrm2 [ctx queue res-bytes read-method method x]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (with-check error
         (~method
          (dim ~x) (cl-mem res-buffer#) 0
          (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
          ~queue nil)
         (~read-method ~queue res-buffer#)))
     0.0))

(defmacro ^:private vector-ipeak [ctx queue method x]
  `(if (< 0 (dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx Integer/BYTES :read-write)]
       (with-check error
         (~method
          (dim ~x) (cl-mem res-buffer#) 0
          (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
          ~queue nil)
         (enq-read-int ~queue res-buffer#)))
     0))

(defmacro ^:private vector-scal [queue method alpha x]
  `(when (< 0 (dim ~x))
     (with-check error
       (~method
        (dim ~x) ~alpha
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        ~queue nil)
       nil)))

(defmacro ^:private vector-axpy [queue method alpha x y]
  `(when (< 0 (dim ~x))
     (with-check error
       (~method
        (dim ~x) ~alpha
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        ~queue nil)
       nil)))

(defmacro ^:private vector-subcopy [queue method x y kx lx ky]
  `(when (< 0 ~lx)
     (with-check error
       (~method
        ~lx
        (cl-mem (buffer ~x)) ~kx (stride ~x)
        (cl-mem (buffer ~y)) ~ky (stride ~y)
        ~queue nil)
       nil)))

;; NOTE: rotXX methods are not supported by CLBlast yet
;; These signatures might change a bit once they are supported
(defmacro ^:private vector-rotg [queue method x]
  `(let [mem# (cl-mem (buffer ~x))
         ofst# (offset ~x)
         strd# (stride ~x)]
     (with-check error
       (~method
        mem# ofst
        mem# (+ ofst strd)
        mem# (+ ofst (* 2 strd))
        mem# (+ ofst (* 3 strd))
        ~queue nil)
       nil)))

(defmacro ^:private vector-rotm [queue method x y p]
  `(when (< 0 (dim ~x))
     (with-check error
       (~method (dim ~x)
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        (cl-mem (buffer ~p)) (offset ~p) (stride ~p)
        ~queue nil)
       nil)))

(defmacro ^:private vector-rotmg [queue method p args]
  `(let [mem# (cl-mem (buffer ~args))
         ofst# (offset ~args)
         strd# (stride ~args)]
     (with-check error
       (~method
        mem# ofst
        mem# (+ ofst strd)
        mem# (+ ofst (* 2 strd))
        mem# (+ ofst (* 3 strd))
        (cl-mem (buffer ~p)) (offset ~p)
        ~queue nil)
       nil)))

(defmacro ^:private vector-rot [queue method x y c s]
  `(with-check error
     (~method (dim ~x)
      (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
      (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
      c s
      ~queue nil)
     nil))

;; =============== Common matrix engine  macros and functions ==================

(defmacro ^:private matrix-swap-copy [queue method a b]
  `(when (< 0 (ecount ~a))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (mrows ~a) (ncols ~a))
                 (stride ~a) (stride ~b)))
       (with-check error
         (~method
          (ecount ~a)
          (cl-mem (buffer ~a)) (offset ~a) 1 (cl-mem (buffer ~b)) (offset ~b) 1
          ~queue nil)
         nil)
       (if (column-major? ~a)
         (dotimes [i# (ncols ~a)]
           (vector-swap-copy ~queue ~method (.col ~a i#) (.col ~b i#)))
         (dotimes [i# (mrows ~a)]
           (vector-swap-copy ~queue ~method (.row ~a i#) (.row ~b i#)))))))

(defmacro ^:private matrix-axpy [queue method alpha a b]
  `(when (< 0 (ecount ~a))
     (if (and (= (order ~a) (order ~b))
              (= (if (column-major? ~a) (mrows ~a) (ncols ~a))
                 (stride ~a) (stride ~b)))
       (with-check error
         (~method
          (ecount ~a)
          ~alpha (cl-mem (buffer ~a)) (offset ~a) 1
          (cl-mem (buffer ~b)) (offset ~b) 1
          ~queue nil)
         nil)
       (if (column-major? ~a)
         (dotimes [i# (ncols ~a)]
           (vector-axpy ~queue ~method ~alpha (.col ~a i#) (.col ~b i#)))
         (dotimes [i# (mrows ~a)]
           (vector-axpy ~queue ~method ~alpha (.row ~a i#) (.row ~b i#)))))))

(defmacro ^:private matrix-gemv [queue method alpha a x beta y]
  `(when (< 0 (ecount ~a))
     (with-check error
       (~method
        (order ~a) Transpose/kNo (mrows ~a) (ncols ~a)
        ~alpha (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        ~beta (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        ~queue nil)
       nil)))

(defmacro ^:private matrix-ger [queue method alpha x y a]
  `(when (< 0 (ecount ~a))
     (with-check error
       (~method
        (order ~a) (mrows ~a) (ncols ~a)
        ~alpha
        (cl-mem (buffer ~x)) (offset ~x) (stride ~x)
        (cl-mem (buffer ~y)) (offset ~y) (stride ~y)
        (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
        ~queue nil)
       nil)))

(defmacro ^:private matrix-gemm [queue method alpha a b beta c]
  `(when (< 0 (ecount ~a))
     (with-check error
       (~method
        (order ~c)
        (if (= (order ~a) (order ~c)) Transpose/kNo Transpose/kYes)
        (if (= (order ~b) (order ~c)) Transpose/kNo Transpose/kYes)
        (mrows ~a) (ncols ~b) (ncols ~a)
        ~alpha (cl-mem (buffer ~a)) (offset ~a) (stride ~a)
        (cl-mem (buffer ~b)) (offset ~b) (stride ~b)
        ~beta (cl-mem (buffer ~c)) (offset ~c) (stride ~c)
        ~queue nil)
       nil)))

;; =============== CLBlast based engines =======================================

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

(deftype CLBlastDoubleGeneralMatrixEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-block-matrix ctx queue prog a b))
  BLAS
  (swap [_ a b]
    (matrix-swap-copy queue CLBlast/CLBlastDswap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy queue CLBlast/CLBlastDcopy ^Matrix a ^Matrix b))
  (axpy [_ alpha a b]
    (matrix-axpy queue CLBlast/CLBlastDaxpy alpha ^Matrix a ^Matrix b))
  (mv [_ alpha a x beta y]
    (matrix-gemv queue CLBlast/CLBlastDgemv alpha a x beta y))
  (rank [_ alpha x y a]
    (matrix-ger queue CLBlast/CLBlastDger alpha x y a))
  (mm [_ alpha a b beta c]
    (matrix-gemm queue CLBlast/CLBlastDgemm alpha a b beta c)))

(deftype CLBlastSingleGeneralMatrixEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-block-matrix ctx queue prog a b))
  BLAS
  (swap [_ a b]
    (matrix-swap-copy queue CLBlast/CLBlastSswap ^Matrix a ^Matrix b))
  (copy [_ a b]
    (matrix-swap-copy queue CLBlast/CLBlastScopy ^Matrix a ^Matrix b))
  (axpy [_ alpha a b]
    (matrix-axpy queue CLBlast/CLBlastSaxpy alpha ^Matrix a ^Matrix b ))
  (mv [_ alpha a x beta y]
    (matrix-gemv queue CLBlast/CLBlastSgemv alpha a x beta y))
  (rank [_ alpha x y a]
    (matrix-ger queue CLBlast/CLBlastSger alpha x y a))
  (mm [_ alpha a b beta c]
    (matrix-gemm queue CLBlast/CLBlastSgemm alpha a b beta c)))

(deftype CLFactory [ctx queue prog ^DataAccessor claccessor vector-eng matrix-eng]
  Releaseable
  (release [_]
    (try
      (and
       (release prog)
       (release vector-eng)
       (release matrix-eng))
      (finally
        (CLBlast/CLBlastClearCache))))
  DataAccessorProvider
  (data-accessor [_]
    claccessor)
  Factory
  (create-vector [this n buf _]
    (create-cl-vector this vector-eng n buf))
  (create-matrix [this m n buf ord]
    (create-cl-ge-matrix this matrix-eng m n buf ord))
  (vector-engine [_]
    vector-eng)
  (matrix-engine [_]
    matrix-eng))

(let [src [(slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/equals.cl"))]]

  (org.jocl.blast.CLBlast/setExceptionsEnabled false)

  (defn clblast-double [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=double" nil)]
      (->CLFactory
       ctx queue prog
       (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES
                          double-array wrap-double cblas-double)
       (->CLBlastDoubleVectorEngine ctx queue prog)
       (->CLBlastDoubleGeneralMatrixEngine ctx queue prog))))

  (defn clblast-single [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=float" nil)]
      (->CLFactory
       ctx queue prog
       (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES
                          float-array wrap-float cblas-single)
       (->CLBlastSingleVectorEngine ctx queue prog)
       (->CLBlastSingleGeneralMatrixEngine ctx queue prog)))))
