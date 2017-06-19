;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.cublas
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release wrap-int wrap-double wrap-float]]
             [utils :refer [with-check]]]
            [uncomplicate.clojurecuda
             [protocols :refer [cu-ptr ptr]]
             [core :refer :all]
             [toolbox :refer [launch-reduce! count-blocks]]
             [nvrtc :refer [program compile!]]
             [utils :refer [error]]]
            [uncomplicate.neanderthal
             [core :refer [transfer!]]
             [native :refer [native-float native-double]]
             [block :as block]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.neanderthal.internal.device
             [common :refer :all]
             [cublock :refer :all]
             [clblock :as clblock]])
  (:import [jcuda.runtime JCuda cudaStream_t]
           jcuda.driver.CUstream
           [jcuda.jcublas JCublas2 cublasHandle cublasOperation cublasSideMode cublasDiagType
            cublasFillMode]
           [uncomplicate.neanderthal.internal.api Vector Matrix TRMatrix Block DataAccessor
            StripeNavigator RealBufferAccessor]
           [uncomplicate.neanderthal.internal.device.cublock CUBlockVector CUGEMatrix CUTRMatrix]
           [uncomplicate.neanderthal.internal.device.clblock CLBlockVector CLGEMatrix CLTRMatrix]))

(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in CUDA. Please use a host instance.")))

;; =============== Transfer preferences ========================================

(prefer-method transfer! [CUBlockVector Object] [Object CLBlockVector])
(prefer-method transfer! [CUGEMatrix Object] [Object CLGEMatrix])
(prefer-method transfer! [CUTRMatrix Object] [Object CLTRMatrix])

;; =============== Common vector macros and functions =======================

(defn ^:private vector-equals [modl hstream ^CUBlockVector x ^CUBlockVector y]
  (let [cnt (.dim x)]
    (if (< 0 cnt)
      (with-release [equals-kernel (function modl "vector_equals")
                     eq-flag-buf (mem-alloc Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-1d cnt) hstream
                 (parameters cnt
                             (.buffer x) (.offset x) (.stride x)
                             (.buffer y) (.offset y) (.stride y)
                             eq-flag-buf))
        (= 0 (aget ^ints (memcpy-host! eq-flag-buf (int-array 1)) 0)))
      (= 0 (.dim y)))))

(defn ^:private vector-subcopy [modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky]
  (when (< 0 (long lx))
    (with-release [copy-kernel (function modl "vector_copy")]
      (launch! copy-kernel (grid-1d lx) hstream
               (parameters (long lx) (.buffer x) (+ (.offset x) (* (long kx) (.stride x))) (.stride x)
                           (.buffer y) (+ (.offset y) (* (long ky) (.stride y))) (.stride y))))))

(defn ^:private vector-sum [modl hstream ^CUBlockVector x]
  (let [cnt (.dim x)
        block-dim 1024]
    (if (< 0 cnt)
      (with-release [sum-kernel (function modl "vector_sum")
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc (* Double/BYTES (count-blocks block-dim cnt)))]
        (launch-reduce! hstream sum-kernel sum-reduction-kernel
                        [(.buffer x) (.offset x) (.stride x) cu-acc] [cu-acc] cnt block-dim)
        (get ^doubles (memcpy-host! cu-acc (double-array 1)) 0))
      0.0)))

(defn ^:private vector-set [modl hstream alpha ^CUBlockVector x]
  (with-release [set-kernel (function modl "vector_set")]
    (launch! set-kernel (grid-1d (.dim x)) hstream
             (parameters (.dim x) alpha (.buffer x) (.offset x) (.stride x)))))

(defn ^:private vector-axpby [modl hstream alpha ^CUBlockVector x beta ^CUBlockVector y]
  (with-release [axpby-kernel (function modl "vector_axpby")]
    (launch! axpby-kernel (grid-1d (.dim x)) hstream
             (parameters (.dim x)
                         alpha (.buffer x) (.offset x) (.stride x)
                         beta (.buffer y) (.offset y) (.stride y)))))

(defmacro ^:private vector-method
  ([cublas-handle method x]
   `(when (< 0 (.dim ~x))
      (with-check cublas-error
        (~method ~cublas-handle (.dim ~x)
         (offset (data-accessor ~x) (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x))
        nil)))
  ([cublas-handle method x y]
   `(when (< 0 (.dim ~x))
      (let [da# (data-accessor ~x)]
        (with-check cublas-error
          (~method ~cublas-handle (.dim ~x)
           (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y))
          nil))))
  ([cublas-handle method x y z]
   `(when (< 0 (.dim x))
      (let [da# (data-accessor ~x)]
        (with-check cublas-error
          (~method ~cublas-handle (.dim ~x)
           (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y)
           (offset da# (cu-ptr (.buffer ~z)) (.offset ~z)) (.stride ~z)))
        nil))))

(defmacro ^:private vector-dot [cublas-handle array-fn method x y]
  `(if (< 0 (.dim ~x))
     (let [da# (data-accessor ~x)
           res# (~array-fn 1)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y)
          (ptr res#))
         (first res#)))
     0.0))

(defmacro ^:private vector-reducer [cublas-handle array-fn method x]
  `(if (< 0 (.dim ~x))
     (let [res# (~array-fn 1)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset (data-accessor ~x) (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x) (ptr res#))
         (first res#)))
     0.0))

(defmacro ^:private vector-scal [cublas-handle method alpha x]
  `(when (< 0 (.dim ~x))
     (with-check cublas-error
       (~method ~cublas-handle (.dim ~x) (ptr ~alpha)
        (offset (data-accessor ~x) (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x))
       nil)))

(defmacro ^:private vector-axpy [cublas-handle method alpha x y]
  `(when (< 0 (.dim ~x))
     (let [da# (data-accessor ~x)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x) (ptr ~alpha)
          (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y))
         nil))))

(defmacro ^:private vector-rot [cublas-handle method x y c s]
  `(let [da# (data-accessor ~x)]
     (with-check cublas-error
       (~method ~cublas-handle (.dim ~x)
        (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
        (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y)
        (ptr ~c) (ptr ~s))
       nil)))

(defmacro ^:private vector-rotm [cublas-handle method x y param]
  `(if (= 1 (.stride ~param))
     (let [da# (data-accessor ~x)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y)
          (ptr (.buffer ~param)))
         nil))
     (throw (ex-info "You cannot use strided vector as param." {:param (str ~param)}))))

;; =============== Common GE matrix macros and functions =======================

(defn ^:private ge-equals [modl hstream ^CUGEMatrix a ^CUGEMatrix b]
  (if (< 0 (.count a))
    (with-release [equals-kernel (function modl (name-transp "ge_equals" a b))
                   eq-flag-buf (mem-alloc Integer/BYTES)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d (.sd a) (.fd a)) hstream
               (parameters (.sd a) (.fd a) (.buffer a) (.offset a) (.stride a)
                           (.buffer b) (.offset b) (.stride b) eq-flag-buf))
      (= 0 (aget ^ints (memcpy-host! eq-flag-buf (int-array 1)) 0)))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private ge-set [modl hstream alpha ^CUGEMatrix a]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [ge-set-kernel (function modl "ge_set")]
        (launch! ge-set-kernel (grid-2d (.sd a) (.fd a)) hstream
                 (parameters (.sd a) (.fd a) alpha (.buffer a) (.offset a) (.stride a)))))))

(defmacro ^:private ge-swap [cublas-handle method modl hstream a b]
  `(when (< 0 (.count ~a))
     (if (fits-buffer? ~a ~b)
       (let [da# (data-accessor ~a)]
         (with-check cublas-error
           (~method ~cublas-handle (.count ~a)
            (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) 1
            (offset da# (cu-ptr (.buffer ~b)) (.offset ~b)) 1)
           nil))
       (with-release [ge-swap-kernel# (function ~modl (name-transp "ge_swap" ~a ~b))]
         (launch! ge-swap-kernel# (grid-2d (.sd ~a) (.fd ~a)) ~hstream
                  (parameters (.sd ~a) (.fd ~a) (.buffer ~a) (.offset ~a) (.stride ~a)
                              (.buffer ~b) (.offset ~b) (.stride ~b)))))))

(defmacro ^:private ge-dot [cublas-handle array-fn method a b]
  `(if (< 0 (.count ~a))
     (if (and (fully-packed? ~a) (fully-packed? ~b) (= (.order ~a) (.order ~b)))
       (let [da# (data-accessor ~a)
             res# (~array-fn 1)]
         (with-check cublas-error
           (~method ~cublas-handle (.count ~a)
            (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) 1
            (offset da# (cu-ptr (.buffer ~b)) (.offset ~b)) 1
            (ptr res#))
           (first res#)))
       (not-available))
     0.0))

(defmacro ^:private ge-asum-nrm2 [cublas-handle array-fn method modl hstream op-name a]
  `(if (< 0 (.count ~a))
     (let [res# (~array-fn 1)]
       (if (fully-packed? ~a)
         (with-check cublas-error
           (~method ~cublas-handle (.count ~a)
            (offset (data-accessor ~a) (cu-ptr (.buffer ~a)) (.offset ~a)) 1 (ptr res#))
           (first res#))
         (not-available)))
     0.0))

(defmacro ^:private ge-am
  ([cublas-handle method alpha a beta b]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)
            b# (offset da# (cu-ptr (.buffer ~b)) (.offset ~b))]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (= (.order ~a) (.order ~b)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           cublasOperation/CUBLAS_OP_N (.sd ~b) (.fd ~b)
           (ptr ~alpha) (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (ptr ~beta) b# (.stride ~b) b# (.stride ~b))
          nil))))
  ([cublas-handle method alpha a]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)
            a# (offset da# (cu-ptr (.buffer ~a)) (.offset ~a))
            ld-a# (.stride ~a)]
        (with-check cublas-error
          (~method ~cublas-handle cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_N
           (.sd ~a) (.fd ~a) (ptr ~alpha) a# ld-a# (ptr 0.0) a# ld-a# a# ld-a#)
          nil)))))

(defmacro ^:private ge-mv
  ([cublas-handle method alpha a x beta y]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (= COLUMN_MAJOR (.order ~a)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           (.sd ~a) (.fd ~a)
           (ptr ~alpha) (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (ptr ~beta) (offset da# (cu-ptr (.buffer ~y)) (.offset ~y)) (.stride ~y))
          nil))))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (str ~a)}))))

(defmacro ^:private ge-rk [cublas-handle method alpha x y a]
  `(when (< 0 (.count ~a))
     (let [da# (data-accessor ~a)]
       (with-check cublas-error
         (let [[v# w#] (if (= COLUMN_MAJOR (.order ~a)) [~x ~y] [~y ~x])]
           (~method ~cublas-handle (.sd ~a) (.fd ~a)
            (ptr ~alpha) (offset da# (cu-ptr (block/buffer v#)) (block/offset v#)) (block/stride v#)
            (offset da# (cu-ptr (block/buffer w#)) (block/offset w#)) (block/stride w#)
            (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) (.stride ~a)))
         nil))))

(defmacro ^:private ge-mm
  ([alpha a b left]
   `(if ~left
      (mm (engine ~b) ~alpha ~b ~a false)
      (throw (ex-info "In-place mm! is not supported for GE matrices." {:a (str ~a)}))))
  ([cublas-handle method alpha a b beta c]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (let [[x# y# trans-x# trans-y#]
                (if (= COLUMN_MAJOR (.order ~c))
                  [~a ~b (= (.order ~c) (.order ~a)) (= (.order ~c) (.order ~b))]
                  [~b ~a (= (.order ~c) (.order ~b)) (= (.order ~c) (.order ~a))])]
            (~method ~cublas-handle
             (if trans-x# cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
             (if trans-y# cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
             (.sd ~c) (.fd ~c) (.ncols ~a)
             (ptr ~alpha) (offset da# (cu-ptr (block/buffer x#)) (block/offset x#)) (block/stride x#)
             (offset da# (cu-ptr (block/buffer y#)) (block/offset y#)) (block/stride y#)
             (ptr ~beta) (offset da# (cu-ptr (.buffer ~c)) (.offset ~c)) (.stride ~c)))
          nil)))))

;; =============== Common TR matrix macros and functions =======================

(defn ^:private tr-equals [modl hstream ^CUTRMatrix a ^CUTRMatrix b]
  (if (< 0 (.count a))
    (with-release [tr-equals-kernel (function modl (name-transp "tr_equals" a b))
                   eq-flag-buf (mem-alloc Integer/BYTES)]
      (memset! eq-flag-buf 0)
      (launch! tr-equals-kernel (grid-2d (.sd a) (.fd a)) hstream
               (parameters (.sd a) (.diag a) (if (tr-bottom a) 1 -1)
                           (.buffer a) (.offset a) (.stride a) (.buffer b) (.offset b) (.stride b)
                           eq-flag-buf))
      (= 0 (aget ^ints (memcpy-host! eq-flag-buf (int-array 1)) 0)))
    (= 0 (.count b))))

(defn ^:private tr-map [modl hstream op-name ^CUTRMatrix a ^CUTRMatrix b]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [tr-map-kernel (function modl (name-transp op-name a b))]
        (launch! tr-map-kernel (grid-2d (.sd a) (.fd a)) hstream
                 (parameters (.sd a) (.diag a) (if (tr-bottom a) 1 -1)
                             (.buffer a) (.offset a) (.stride a) (.buffer b) (.offset b) (.stride b)))))))

(defn ^:private tr-axpby [modl hstream alpha ^CUTRMatrix a beta ^CUTRMatrix b]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [tr-axpby-kernel (function modl (name-transp "tr_axpby" a b))]
        (launch! tr-axpby-kernel (grid-2d (.sd a) (.fd a)) hstream
                 (parameters (.sd a) (.diag a) (if (tr-bottom a) 1 -1)
                             alpha (.buffer a) (.offset a) (.stride a)
                             beta (.buffer b) (.offset b) (.stride b)))))))

(defn ^:private tr-set-scal [modl hstream op-name alpha ^CUTRMatrix a]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [tr-op-kernel (function modl op-name)]
        (launch! tr-op-kernel (grid-2d (.sd a) (.fd a)) hstream
                 (parameters (.sd a) (.diag a) (if (tr-bottom a) 1 -1)
                             alpha (.buffer a) (.offset a) (.stride a)))))))

(defn ^:private blas-to-cublas ^long [^long blas]
  (case blas
    101 cublasOperation/CUBLAS_OP_T
    102 cublasOperation/CUBLAS_OP_N
    121 cublasFillMode/CUBLAS_FILL_MODE_UPPER
    122 cublasFillMode/CUBLAS_FILL_MODE_LOWER
    131 cublasDiagType/CUBLAS_DIAG_NON_UNIT
    132 cublasDiagType/CUBLAS_DIAG_UNIT
    -1000))

(defmacro ^:private tr-mv
  ([cublas-handle method a x]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (blas-to-cublas (if (tr-bottom ~a) LOWER UPPER)) (blas-to-cublas (.order ~a))
           (blas-to-cublas (.diag ~a)) (.sd ~a)
           (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (cu-ptr (.buffer ~x)) (.offset ~x)) (.stride ~x))
          nil))))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

(defmacro ^:private tr-mm
  ([cublas-handle method alpha a b left]
   `(when (< 0 (.count ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if ~left cublasSideMode/CUBLAS_SIDE_LEFT cublasSideMode/CUBLAS_SIDE_RIGHT)
           (blas-to-cublas (.uplo ~a))
           (if (= (.order ~a) (.order ~b)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           (blas-to-cublas (.diag ~a)) (.sd ~b) (.fd ~b)
           (ptr ~alpha) (offset da# (cu-ptr (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (cu-ptr (.buffer ~b)) (.offset ~b)) (.stride ~b)
           (offset da# (cu-ptr (.buffer ~b)) (.offset ~b)) (.stride ~b))
          nil))))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

;; ======================== Engines ===========================================

(deftype DoubleVectorEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals modl hstream ^CUBlockVector x ^CUBlockVector y))
  Blas
  (swap [_ x y]
    (vector-method cublas-handle JCublas2/cublasDswap ^CUBlockVector x ^CUBlockVector y)
    x)
  (copy [_ x y]
    (vector-method cublas-handle JCublas2/cublasDcopy ^CUBlockVector x ^CUBlockVector y)
    y)
  (dot [_ x y]
    (vector-dot cublas-handle double-array JCublas2/cublasDdot ^CUBlockVector x ^CUBlockVector y))
  (nrm2 [_ x]
    (vector-reducer cublas-handle double-array JCublas2/cublasDnrm2 ^CUBlockVector x))
  (asum [_ x]
    (vector-reducer cublas-handle double-array JCublas2/cublasDasum ^CUBlockVector x))
  (iamax [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIdamax ^CUBlockVector x)))))
  (iamin [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIdamin ^CUBlockVector x)))))
  (rot [_ x y c s]
    (vector-rot cublas-handle JCublas2/cublasDrot ^CUBlockVector x ^CUBlockVector y (double c) (double s))
    x)
  (rotg [_ _]
    (not-available))
  (rotm [_ x y param]
    (vector-rotm cublas-handle JCublas2/cublasDrotm  ^CUBlockVector x ^CUBlockVector y ^CUBlockVector param)
    x)
  (rotmg [_ _ _]
    (not-available))
  (scal [_ alpha x]
    (vector-scal cublas-handle JCublas2/cublasDscal (double alpha) ^CUBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy cublas-handle JCublas2/cublasDaxpy (double alpha) ^CUBlockVector x ^CUBlockVector y)
    y)
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky)
    y)
  (sum [_ x]
    (vector-sum modl hstream ^CUBlockVector x))
  (imax [_ x]
    (not-available))
  (imin [this x]
    (not-available))
  (set-all [_ alpha x]
    (vector-set modl hstream (double alpha) ^CUBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby modl hstream (double alpha) x (double beta) y)
    y))

(deftype FloatVectorEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals modl hstream ^CUBlockVector x ^CUBlockVector y))
  Blas
  (swap [_ x y]
    (vector-method cublas-handle JCublas2/cublasSswap ^CUBlockVector x ^CUBlockVector y)
    x)
  (copy [_ x y]
    (vector-method cublas-handle JCublas2/cublasScopy ^CUBlockVector x ^CUBlockVector y)
    y)
  (dot [_ x y]
    (vector-dot cublas-handle float-array JCublas2/cublasSdot ^CUBlockVector x ^CUBlockVector y))
  (nrm2 [_ x]
    (vector-reducer cublas-handle float-array JCublas2/cublasSnrm2 ^CUBlockVector x))
  (asum [_ x]
    (vector-reducer cublas-handle float-array JCublas2/cublasSasum ^CUBlockVector x))
  (iamax [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIsamax ^CUBlockVector x)))))
  (iamin [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIsamin ^CUBlockVector x)))))
  (rot [_ x y c s]
    (vector-rot cublas-handle JCublas2/cublasSrot ^CUBlockVector x ^CUBlockVector y (float c) (float s))
    x)
  (rotg [_ _]
    (not-available))
  (rotm [_ x y param]
    (vector-rotm cublas-handle JCublas2/cublasSrotm  ^CUBlockVector x ^CUBlockVector y ^CUBlockVector param)
    x)
  (rotmg [_ _ _]
    (not-available))
  (scal [_ alpha x]
    (vector-scal cublas-handle JCublas2/cublasSscal (float alpha) ^CUBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy cublas-handle JCublas2/cublasSaxpy (float alpha) ^CUBlockVector x ^CUBlockVector y)
    y)
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky)
    y)
  (sum [_ x]
    (vector-sum modl hstream ^CUBlockVector x))
  (imax [_ x]
    (not-available))
  (imin [this x]
    (not-available))
  (set-all [_ alpha x]
    (vector-set modl hstream (float alpha) ^CUBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby modl hstream (float alpha) x (float beta) y)
    y))

(deftype DoubleGEEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (ge-swap cublas-handle JCublas2/cublasDswap modl hstream ^CUGEMatrix a ^CUGEMatrix b)
    a)
  (copy [_ a b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double 1) ^CUGEMatrix a (double 0) ^CUGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a)
    a)
  (dot [_ a b]
    (ge-dot cublas-handle double-array JCublas2/cublasDdot ^CUGEMatrix a ^CUGEMatrix b))
  (nrm2 [this a]
    (ge-asum-nrm2 cublas-handle double-array JCublas2/cublasDnrm2 modl hstream "ge_nrm2" ^CUGEMatrix a))
  (asum [this a]
    (ge-asum-nrm2 cublas-handle double-array JCublas2/cublasDasum modl hstream "ge_asum" ^CUGEMatrix a))
  (axpy [_ alpha a b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a (double 1.0) ^CUGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (ge-mv cublas-handle JCublas2/cublasDgemv
           (double alpha) ^CUGEMatrix a ^CUBlockVector x (double beta) ^CUBlockVector y)
    y)
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk cublas-handle JCublas2/cublasDger (double alpha) ^CUBlockVector x ^CUBlockVector y ^CUGEMatrix a)
    a)
  (mm [_ alpha a b left]
    (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
    (ge-mm cublas-handle JCublas2/cublasDgemm
           (double alpha) ^CUGEMatrix a ^CUGEMatrix b (double beta) ^CUGEMatrix c)
    c)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (ge-set modl hstream (double alpha) a)
    a)
  (axpby [_ alpha a beta b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a (double beta) ^CUGEMatrix b)
    b)
  (trans [_ a]
    (not-available)
    a))

(deftype FloatGEEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (ge-swap cublas-handle JCublas2/cublasSswap modl hstream ^CUGEMatrix a ^CUGEMatrix b)
    a)
  (copy [_ a b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float 1) ^CUGEMatrix a (float 0) ^CUGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a)
    a)
  (dot [_ a b]
    (ge-dot cublas-handle float-array JCublas2/cublasSdot ^CUGEMatrix a ^CUGEMatrix b))
  (nrm2 [this a]
    (ge-asum-nrm2 cublas-handle float-array JCublas2/cublasSnrm2 modl hstream "ge_nrm2" ^CUGEMatrix a))
  (asum [this a]
    (ge-asum-nrm2 cublas-handle float-array JCublas2/cublasSasum modl hstream "ge_asum" ^CUGEMatrix a))
  (axpy [_ alpha a b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a (float 1.0) ^CUGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (ge-mv cublas-handle JCublas2/cublasSgemv
           (float alpha) ^CUGEMatrix a ^CUBlockVector x (float beta) ^CUBlockVector y)
    y)
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk cublas-handle JCublas2/cublasSger (float alpha) ^CUBlockVector x ^CUBlockVector y ^CUGEMatrix a)
    a)
  (mm [_ alpha a b left]
    (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
    (ge-mm cublas-handle JCublas2/cublasSgemm
           (float alpha) ^CUGEMatrix a ^CUGEMatrix b (float beta) ^CUGEMatrix c)
    c)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (ge-set modl hstream (float alpha) a)
    a)
  (axpby [_ alpha a beta b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a (float beta) ^CUGEMatrix b)
    b)
  (trans [_ a]
    (not-available)
    a))

(deftype DoubleTREngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (tr-map modl hstream "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map modl hstream "tr_copy" a b)
    b)
  (scal [_ alpha a]
    (tr-set-scal modl hstream "tr_scal" (double alpha) a)
    a)
  (axpy [_ alpha a b]
    (tr-axpby modl hstream (double alpha) ^CUTRMatrix a (double 1.0) b)
    b)
  (dot [_ _ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv cublas-handle JCublas2/cublasDtrmv ^CUTRMatrix a ^CUBlockVector x)
    x)
  (mm [this alpha a b beta c]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm cublas-handle JCublas2/cublasDtrmm (double alpha) ^CUTRMatrix a ^CUGEMatrix b left)
    b)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal modl hstream "tr_set" (double alpha) a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby modl hstream (double alpha) ^CUTRMatrix a (double beta) b)
    b))

(deftype FloatTREngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (tr-map modl hstream "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map modl hstream "tr_copy" a b)
    b)
  (scal [_ alpha a]
    (tr-set-scal modl hstream "tr_scal" (float alpha) a)
    a)
  (axpy [_ alpha a b]
    (tr-axpby modl hstream (float alpha) a (float 1.0) b)
    b)
  (dot [_ _ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv cublas-handle JCublas2/cublasStrmv ^CUTRMatrix a ^CUBlockVector x)
    x)
  (mm [this alpha a b beta c]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm cublas-handle JCublas2/cublasStrmm (float alpha) ^CUTRMatrix a ^CUGEMatrix b left)
    b)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal modl hstream "tr_set" (float alpha) a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby modl hstream (float alpha) a (float beta) b)
    b))

(deftype CUFactory [modl hstream ^DataAccessor da native-fact vector-eng ge-eng tr-eng]
  Releaseable
  (release [_]
    (release vector-eng)
    (release ge-eng)
    (release tr-eng)
    (release modl)
    true)
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    native-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  Factory
  (create-vector [this n init]
    (let-release [res (cu-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-ge [this m n ord init]
    (let-release [res (cu-ge-matrix this m n ord)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n ord uplo diag init]
    (let-release [res (cu-tr-matrix this n ord uplo diag)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng))

(extend-type cublasHandle
  Releaseable
  (release [this]
    (with-check cublas-error (JCublas2/cublasDestroy this) true)))

(defn ^:private get-stream [handle]
  (let [res (cudaStream_t.)]
    (with-check cublas-error (JCublas2/cublasGetStream handle res) (CUstream. res))))

(defn cublas-handle
  "Creates a cuBLAS context handler on the specific `device-id` (default `0`) and `stream`
  (default is a per-thread cuda stream)"
  ([^long device-id ^CUstream stream]
   (with-check error (JCuda/cudaSetDevice device-id)
     (let [handle (cublasHandle.)
           cuda-stream (cudaStream_t. ^CUStream stream)]
       (with-check cublas-error (JCublas2/cublasCreate handle)
         (with-check cublas-error (JCublas2/cublasSetStream handle cuda-stream)
           handle)))))
  ([^long device-id]
   (cublas-handle device-id default-stream))
  ([]
   (cublas-handle 0)))

(let [src (str (slurp (io/resource "uncomplicate/clojurecuda/kernels/reduction.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/blas-plus.cu")))]

  (JCublas2/setExceptionsEnabled false)

  (defn cublas-double [handle]
    (with-release [prog (compile! (program src) ["-DREAL=double" "-DACCUMULATOR=double" "-arch=compute_30"])]
      (let-release [modl (module prog)
                    hstream (get-stream handle)]
        (->CUFactory modl hstream (cu-double-accessor (current-context)) native-double
                     (->DoubleVectorEngine handle modl hstream) (->DoubleGEEngine handle modl hstream)
                     (->DoubleTREngine handle modl hstream)))))

  (defn cublas-float [handle]
    (with-release [prog (compile! (program src) ["-DREAL=float" "-DACCUMULATOR=double" "-arch=compute_30"])]
      (let-release [modl (module prog)
                    hstream (get-stream handle)]
        (->CUFactory modl hstream (cu-float-accessor (current-context)) native-float
                     (->FloatVectorEngine handle modl hstream) (->FloatGEEngine handle modl hstream)
                     (->FloatTREngine handle modl hstream))))))
