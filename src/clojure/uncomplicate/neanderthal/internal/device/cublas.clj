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
             [core :refer [Releaseable release let-release with-release info
                           extract wrap wrap-int wrap-double wrap-float]]
             [utils :refer [with-check dragan-says-ex count-groups generate-seed count-groups]]]
            [uncomplicate.clojurecuda
             [core :refer :all :as cuda :exclude [device]]
             [info :refer [driver-version]]
             [toolbox :refer [launch-reduce! read-int]]]
            [uncomplicate.clojurecuda.internal
             [protocols :refer [ptr]]
             [utils :refer [error]]]
            [uncomplicate.neanderthal
             [core :refer [transfer!]]
             [native :refer [native-float native-double]]
             [block :as block]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [check-eq-navigators]]]
            [uncomplicate.neanderthal.internal.device
             [common :refer [name-transp uplo-bottom? layout-match? symmetric-match?]]
             [cublock :refer :all]])
  (:import jcuda.runtime.cudaStream_t
           jcuda.driver.CUstream
           [jcuda.jcublas JCublas2 cublasHandle cublasOperation cublasSideMode cublasDiagType
            cublasFillMode]
           [uncomplicate.neanderthal.internal.api Vector Matrix GEMatrix Block DataAccessor Region
            DenseStorage FullStorage LayoutNavigator]
           [uncomplicate.neanderthal.internal.device.cublock CUBlockVector CUGEMatrix CUUploMatrix]))

(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in CUDA. Please use a host instance.")))

;; =============== Random Number Generators =================================

(defn ^:private vector-random [modl hstream kernel-name rng-state a b ^CUBlockVector x]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [random-kernel (function modl kernel-name)]
        (launch! random-kernel (grid-1d (count-groups 4 (.dim x))) hstream
                 (parameters (.dim x) (long (swap! rng-state inc))
                             (.wrapPrim da a) (.wrapPrim da b)
                             (.buffer x) (.offset x) (.stride x))))))
  x)

(defn ^:private ge-random [modl hstream kernel-name rng-state a b ^CUGEMatrix x]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)
          stor (full-storage x)]
      (with-release [random-kernel (function modl kernel-name)]
        (launch! random-kernel (grid-2d (count-groups 4 (.sd stor)) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (long (swap! rng-state inc))
                             (.wrapPrim da a) (.wrapPrim da b)
                             (.buffer x) (.offset x) (.ld stor))))))
  x)

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
        (= 0 (read-int hstream eq-flag-buf)))
      (= 0 (.dim y)))))

(defn ^:private vector-subcopy [modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky]
  (when (< 0 (long lx))
    (with-release [copy-kernel (function modl "vector_copy")]
      (launch! copy-kernel (grid-1d lx) hstream
               (parameters (long lx) (.buffer x) (+ (.offset x) (* (long kx) (.stride x))) (.stride x)
                           (.buffer y) (+ (.offset y) (* (long ky) (.stride y))) (.stride y)))))
  y)

(defn ^:private vector-sum [modl hstream ^CUBlockVector x]
  (let [da (data-accessor x)
        cnt (.dim x)
        block-dim 1024]
    (if (< 0 cnt)
      (with-release [sum-kernel (function modl "vector_sum")
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc (* (.entryWidth da) (count-groups block-dim cnt)))]
        (launch-reduce! hstream sum-kernel sum-reduction-kernel
                        [(.buffer x) (.offset x) (.stride x) cu-acc] [cu-acc] cnt block-dim)
        (first (memcpy-host! cu-acc (.wrapPrim da 0.0) hstream)))
      0.0)))

(defn ^:private vector-set [modl hstream alpha ^CUBlockVector x]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [set-kernel (function modl "vector_set")]
        (launch! set-kernel (grid-1d (.dim x)) hstream
                 (parameters (.dim x) (.wrapPrim da alpha) (.buffer x) (.offset x) (.stride x))))))
  x)

(defn ^:private vector-axpby [modl hstream alpha ^CUBlockVector x beta ^CUBlockVector y]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [axpby-kernel (function modl "vector_axpby")]
        (launch! axpby-kernel (grid-1d (.dim x)) hstream
                 (parameters (.dim x)
                             (.wrapPrim da alpha) (.buffer x) (.offset x) (.stride x)
                             (.wrapPrim da beta) (.buffer y) (.offset y) (.stride y))))))
  y)

(defmacro ^:private vector-method
  ([cublas-handle method x]
   `(if (< 0 (.dim ~x))
      (with-check cublas-error
        (~method ~cublas-handle (.dim ~x)
         (offset (data-accessor ~x) (extract (.buffer ~x)) (.offset ~x)) (.stride ~x))
        ~x)
      ~x))
  ([cublas-handle method x y]
   `(if (< 0 (.dim ~x))
      (let [da# (data-accessor ~x)]
        (with-check cublas-error
          (~method ~cublas-handle (.dim ~x)
           (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y))
          ~y))
      ~y))
  ([cublas-handle method x y z]
   `(if (< 0 (.dim x))
      (let [da# (data-accessor ~x)]
        (with-check cublas-error
          (~method ~cublas-handle (.dim ~x)
           (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y)
           (offset da# (extract (.buffer ~z)) (.offset ~z)) (.stride ~z))
        ~z))
      ~z)))

(defmacro ^:private vector-dot [cublas-handle array-fn method x y]
  `(if (< 0 (.dim ~x))
     (let [da# (data-accessor ~x)
           res# (~array-fn 1)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y)
          (ptr res#))
         (first res#)))
     0.0))

(defmacro ^:private vector-reducer [cublas-handle array-fn method x]
  `(if (< 0 (.dim ~x))
     (let [res# (~array-fn 1)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset (data-accessor ~x) (extract (.buffer ~x)) (.offset ~x)) (.stride ~x) (ptr res#))
         (first res#)))
     0.0))

(defmacro ^:private vector-scal [cublas-handle method alpha x]
  `(if (< 0 (.dim ~x))
     (with-check cublas-error
       (~method ~cublas-handle (.dim ~x) (ptr ~alpha)
        (offset (data-accessor ~x) (extract (.buffer ~x)) (.offset ~x)) (.stride ~x))
       ~x)
     ~x))

(defmacro ^:private vector-axpy [cublas-handle method alpha x y]
  `(if (< 0 (.dim ~x))
     (let [da# (data-accessor ~x)]
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x) (ptr ~alpha)
          (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y))
         ~y))
     ~y))

(defmacro ^:private vector-rot [cublas-handle method x y c s]
  `(let [da# (data-accessor ~x)]
     (if (and (< 0 (.dim ~x)) (< 0 (.dim ~y)))
       (with-check cublas-error
         (~method ~cublas-handle (.dim ~x)
          (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
          (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y)
          (ptr ~c) (ptr ~s))
         ~x)
       ~x)))

(defmacro ^:private vector-rotm [cublas-handle method x y param]
  `(if (= 1 (.stride ~param))
     (let [da# (data-accessor ~x)]
       (if (and (< 0 (.dim ~x)) (< 0 (.dim ~y)))
         (with-check cublas-error
           (~method ~cublas-handle (.dim ~x)
            (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
            (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y)
            (ptr (.buffer ~param)))
           ~param)
         ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (info ~param)}))))

;; =============== Common GE matrix macros and functions =======================

(defn ^:private ge-equals [modl hstream ^CUGEMatrix a ^CUGEMatrix b]
  (if (< 0 (.dim a))
    (let [stor (full-storage a)]
      (with-release [equals-kernel (function modl (name-transp "ge_equals" a b))
                     eq-flag-buf (mem-alloc Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (.buffer a) (.offset a) (.ld stor)
                             (.buffer b) (.offset b) (.stride b) eq-flag-buf))
        (= 0 (read-int hstream eq-flag-buf))))
    (= 0 (.dim b))))

(defn ^:private ge-set [modl hstream alpha ^CUGEMatrix a]
  (if (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [ge-set-kernel (function modl "ge_set")]
        (launch! ge-set-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (.wrapPrim da alpha) (.buffer a) (.offset a) (.ld stor))))
      a)))

(defmacro ^:private ge-swap [cublas-handle method modl hstream a b]
  `(if (< 0 (.dim ~a))
     (let [da# (data-accessor ~a)
           stor# (full-storage ~a)]
       (if (and (= (navigator ~a) (navigator ~b)) (.isGapless stor#) (.isGapless (storage ~b)))
         (with-check cublas-error
           (~method ~cublas-handle (.dim ~a)
            (offset da# (extract (.buffer ~a)) (.offset ~a)) 1
            (offset da# (extract (.buffer ~b)) (.offset ~b)) 1)
           ~a)
         (with-release [ge-swap-kernel# (function ~modl (name-transp "ge_swap" ~a ~b))]
           (launch! ge-swap-kernel# (grid-2d (.sd stor#) (.fd stor#)) ~hstream
                    (parameters (.sd stor#) (.fd stor#) (.buffer ~a) (.offset ~a) (.ld stor#)
                                (.buffer ~b) (.offset ~b) (.stride ~b)))
           ~a)))
     ~a))

(defmacro ^:private ge-dot [cublas-handle array-fn method a b]
  `(if (< 0 (.dim ~a))
     (if (and (= (navigator ~a) (navigator ~b)) (.isGapless (storage ~a)) (.isGapless (storage ~b)))
       (let [da# (data-accessor ~a)
             res# (~array-fn 1)]
         (with-check cublas-error
           (~method ~cublas-handle (.dim ~a)
            (offset da# (extract (.buffer ~a)) (.offset ~a)) 1
            (offset da# (extract (.buffer ~b)) (.offset ~b)) 1
            (ptr res#))
           (first res#)))
       (not-available))
     0.0))

(defmacro ^:private ge-asum-nrm2 [cublas-handle array-fn method modl hstream op-name a]
  `(if (< 0 (.dim ~a))
     (let [res# (~array-fn 1)]
       (if (.isGapless (storage ~a))
         (with-check cublas-error
           (~method ~cublas-handle (.dim ~a)
            (offset (data-accessor ~a) (extract (.buffer ~a)) (.offset ~a)) 1 (ptr res#))
           (first res#))
         (not-available)))
     0.0))

(defn ^:private ge-sum [modl hstream ^CUGEMatrix a]
  (let [da (data-accessor a)
        stor (full-storage a)
        cnt-sd (.sd stor)
        cnt-fd (.fd stor)
        wgs-sd 32
        wgs-fd 32
        wgs 1024
        grid-dim-x (count-groups wgs-sd cnt-sd)
        grid-dim-y (count-groups wgs-fd cnt-fd)
        acc-count (* grid-dim-x grid-dim-y)]
    (if (< 0 (.dim a))
      (with-release [sum-kernel (function modl "ge_sum")
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc (* Double/BYTES acc-count))
                     params (parameters acc-count cu-acc)]
        (launch! sum-kernel (grid-2d cnt-sd cnt-fd wgs-sd wgs-fd) hstream
                 (parameters (int cnt-sd) (int cnt-fd) cu-acc (.buffer a) (.offset a) (.stride a)))
        (if (< 1 acc-count)
          (launch-reduce! hstream sum-reduction-kernel sum-reduction-kernel
                          params params acc-count wgs))
        (first (memcpy-host! cu-acc (.wrapPrim da 0.0) hstream)))
      0.0)))

(defmacro ^:private ge-am
  ([cublas-handle method alpha a beta b]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)
            b# (offset da# (extract (.buffer ~b)) (.offset ~b))
            stor-b# (full-storage ~b)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (= (navigator ~a) (navigator ~b)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           cublasOperation/CUBLAS_OP_N (.sd stor-b#) (.fd stor-b#)
           (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (ptr ~beta) b# (.stride ~b) b# (.ld stor-b#))
          ~b))
      ~b))
  ([cublas-handle method alpha a]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)
            a# (offset da# (extract (.buffer ~a)) (.offset ~a))
            stor# (full-storage ~a)
            ld-a# (.ld stor#)]
        (with-check cublas-error
          (~method ~cublas-handle cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_N
           (.sd stor#) (.fd stor#) (ptr ~alpha) a# ld-a# (ptr 0.0) a# ld-a# a# ld-a#)
          ~a))
      ~a)))

(defmacro ^:private ge-mv
  ([cublas-handle method alpha a x beta y]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)
            stor# (full-storage ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (.isColumnMajor (navigator ~a)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           (.sd stor#) (.fd stor#)
           (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.ld stor#)
           (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (ptr ~beta) (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y))
          ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (info ~a)}))))

(defmacro ^:private ge-rk [cublas-handle method alpha x y a]
  `(if (< 0 (.dim ~a))
     (let [da# (data-accessor ~a)
           stor# (full-storage ~a)]
       (with-check cublas-error
         (let [[v# w#] (if (.isColumnMajor (navigator ~a)) [~x ~y] [~y ~x])]
           (~method ~cublas-handle (.sd stor#) (.fd stor#)
            (ptr ~alpha) (offset da# (extract (block/buffer v#)) (block/offset v#)) (block/stride v#)
            (offset da# (extract (block/buffer w#)) (block/offset w#)) (block/stride w#)
            (offset da# (extract (.buffer ~a)) (.offset ~a)) (.ld stor#)))
         ~a))
     ~a))

(defmacro ^:private ge-mm
  ([alpha a b]
   `(if-not (instance? GEMatrix ~b)
      (mm (engine ~b) ~alpha ~b ~a false)
      (dragan-says-ex "In-place mm! is not supported for GE matrices. Use QR factorization."
                      {:a (info ~a) :b (info ~b)} )))
  ([cublas-handle method alpha a b beta c]
   `(if (< 0 (.dim ~a))
      (if (instance? GEMatrix ~b)
        (let [da# (data-accessor ~a)
              nav-c# (navigator ~c)
              stor-c# (full-storage ~c)]
          (with-check cublas-error
            (let [[x# y# trans-x# trans-y#]
                  (if (.isColumnMajor nav-c#)
                    [~a ~b (= nav-c# (navigator ~a)) (= nav-c# (navigator ~b))]
                    [~b ~a (= nav-c# (navigator ~b)) (= nav-c# (navigator ~a))])]
              (~method ~cublas-handle
               (if trans-x# cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
               (if trans-y# cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
               (.sd stor-c#) (.fd stor-c#) (.ncols ~a)
               (ptr ~alpha) (offset da# (extract (block/buffer x#)) (block/offset x#)) (block/stride x#)
               (offset da# (extract (block/buffer y#)) (block/offset y#)) (block/stride y#)
               (ptr ~beta) (offset da# (extract (.buffer ~c)) (.offset ~c)) (.ld stor-c#)))
            ~c))
        (mm (engine ~b) ~alpha ~b ~a ~beta ~c false))
      ~c)))

;; =============== Common UPLO matrix macros and functions =======================

(defn ^:private uplo-equals [modl hstream transpf ^CUUploMatrix a ^CUUploMatrix b]
  (if (< 0 (.dim a))
    (let [stor (full-storage a)]
      (with-release [equals-kernel (function modl (name-transp (transpf a b) "uplo_equals" a b))
                     eq-flag-buf (mem-alloc Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.buffer a) (.offset a) (.ld stor) (.buffer b) (.offset b) (.stride b)
                             eq-flag-buf))
        (= 0 (read-int hstream eq-flag-buf))))
    (= 0 (.dim b))))

(defn ^:private uplo-map [modl hstream transpf op-name ^CUUploMatrix a ^CUUploMatrix b]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [map-kernel (function modl (name-transp (transpf a b) op-name a b))]
        (launch! map-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.buffer a) (.offset a) (.ld stor) (.buffer b) (.offset b) (.stride b))))))
  b)

(defn ^:private uplo-axpby [modl hstream transpf alpha ^CUUploMatrix a beta ^CUUploMatrix b]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [axpby-kernel (function modl (name-transp (transpf a b) "uplo_axpby" a b))]
        (launch! axpby-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.wrapPrim da alpha) (.buffer a) (.offset a) (.ld stor)
                             (.wrapPrim da beta) (.buffer b) (.offset b) (.stride b))))))
  b)

(defn ^:private uplo-set-scal [modl hstream op-name alpha ^CUUploMatrix a]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [op-kernel (function modl op-name)]
        (launch! op-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.wrapPrim da alpha) (.buffer a) (.offset a) (.ld stor))))))
  a)

(defn ^:private uplo-sum ^double [modl hstream sum-kernel-name ^CUUploMatrix a]
  (let [da (data-accessor a)
        stor (full-storage a)
        cnt-sd (.sd stor)
        cnt-fd (.fd stor)
        wgs-sd 32
        wgs-fd 32
        wgs 1024
        grid-dim-x (count-groups wgs-sd cnt-sd)
        grid-dim-y (count-groups wgs-fd cnt-fd)
        acc-count (* grid-dim-x grid-dim-y)]
    (if (< 0 (.dim a))
      (with-release [sum-kernel (function modl sum-kernel-name)
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc (* Double/BYTES acc-count))
                     params (parameters acc-count cu-acc)]
        (launch! sum-kernel (grid-2d cnt-sd cnt-fd wgs-sd wgs-fd) hstream
                 (parameters cnt-sd cnt-fd (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             cu-acc (.buffer a) (.offset a) (.stride a)))
        (if (< 1 acc-count)
          (launch-reduce! hstream sum-reduction-kernel sum-reduction-kernel
                          params params acc-count wgs))
        (first (memcpy-host! cu-acc (.wrapPrim da 0.0) hstream)))
      0.0)))

(defmacro ^:private tr-mv
  ([cublas-handle method a x]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
           (if (.isColumnMajor (navigator ~a)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           (if (.isDiagUnit (region ~a)) cublasDiagType/CUBLAS_DIAG_UNIT cublasDiagType/CUBLAS_DIAG_NON_UNIT)
           (.ncols ~a)
           (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x))
          ~x))
      ~x))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private tr-mm
  ([cublas-handle method alpha a b left]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)
            stor-b# (full-storage ~b)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if ~left cublasSideMode/CUBLAS_SIDE_LEFT cublasSideMode/CUBLAS_SIDE_RIGHT)
           (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
           (if (= (navigator ~a) (navigator ~b)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
           (if (.isDiagUnit (region ~a)) cublasDiagType/CUBLAS_DIAG_UNIT cublasDiagType/CUBLAS_DIAG_NON_UNIT)
           (.sd stor-b#) (.fd stor-b#)
           (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (extract (.buffer ~b)) (.offset ~b)) (.ld stor-b#)
           (offset da# (extract (.buffer ~b)) (.offset ~b)) (.ld stor-b#))
          ~b))
      ~b))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private sy-mv
  ([cublas-handle method alpha a x beta y]
   `(if (< 0 (.dim ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (~method ~cublas-handle
           (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
           (.ncols ~a)
           (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
           (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
           (ptr ~beta) (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y))
          ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for SY matrices." {:a (info ~a)}))))

(defmacro ^:private sy-r
  ([cublas-handle method alpha x y a]
   `(let [da# (data-accessor ~a)]
      (~method ~cublas-handle
       (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
       (.mrows ~a)
       (ptr ~alpha) (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
       (offset da# (extract (.buffer ~y)) (.offset ~y)) (.stride ~y)
       (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a))
      ~a))
  ([cublas-handle method alpha x a]
   `(let [da# (data-accessor ~a)]
      (~method ~cublas-handle
       (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
       (.mrows ~a)
       (ptr ~alpha) (offset da# (extract (.buffer ~x)) (.offset ~x)) (.stride ~x)
       (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a))
      ~a)))

(defmacro ^:private sy-rk [cublas-handle method alpha a beta c]
  `(if (instance? CUUploMatrix ~c)
     (let [da# (data-accessor ~a)]
       (~method ~cublas-handle
        (if (uplo-bottom? ~c) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
        (if (.isColumnMajor (navigator ~a)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
        (.mrows ~c) (.ncols ~a)
        (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
        (ptr ~beta) (offset da# (extract (.buffer ~c)) (.offset ~c)) (.stride ~c))
       ~c)
     (throw (ex-info "sy-rk is only available for symmetric matrices." {:c (info ~c)}))))

(defmacro ^:private sy-mm
  ([cublas-handle method alpha a b beta c left]
   `(if (< 0 (.dim ~a))
      (let [nav-c# (navigator ~c)
            da# (data-accessor ~a)
            stor-c# (full-storage ~c)]
        (if (= nav-c# (navigator ~b))
          (with-check cublas-error
            (~method ~cublas-handle
             (if ~left cublasSideMode/CUBLAS_SIDE_LEFT cublasSideMode/CUBLAS_SIDE_RIGHT)
             (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
             (.sd stor-c#) (.fd stor-c#)
             (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
             (offset da# (extract (.buffer ~b)) (.offset ~b)) (.stride ~b)
             (ptr ~beta) (offset da# (extract (.buffer ~c)) (.offset ~c)) (.stride ~c))
            ~c)
          (dragan-says-ex "Both GE matrices in symmetric multiplication must have the same orientation."
                          {:b (info ~b) :c (info ~c)})))
      ~c))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private tr-sv
  [cublas-handle method alpha a b]
  `(if (< 0 (.dim ~a))
     (let [da# (data-accessor ~a)
           stor-b# (full-storage ~b)]
       (with-check cublas-error
         (~method ~cublas-handle cublasSideMode/CUBLAS_SIDE_LEFT
          (if (uplo-bottom? ~a) cublasFillMode/CUBLAS_FILL_MODE_LOWER cublasFillMode/CUBLAS_FILL_MODE_UPPER)
          (if (= (navigator ~a) (navigator ~b)) cublasOperation/CUBLAS_OP_N cublasOperation/CUBLAS_OP_T)
          (if (.isDiagUnit (region ~a)) cublasDiagType/CUBLAS_DIAG_UNIT cublasDiagType/CUBLAS_DIAG_NON_UNIT)
          (.sd stor-b#) (.fd stor-b#)
          (ptr ~alpha) (offset da# (extract (.buffer ~a)) (.offset ~a)) (.stride ~a)
          (offset da# (extract (.buffer ~b)) (.offset ~b)) (.ld stor-b#))
         ~b))
     ~b))

;; =============== Common vectorized math functions ============================

(defn ^:private vector-math
  ([modl hstream kernel-name ^CUBlockVector x ^CUBlockVector y]
   (when (< 0 (.dim x))
     (with-release [math-kernel (function modl kernel-name)]
       (launch! math-kernel (grid-1d (.dim x)) hstream
                (parameters (.dim x)
                            (.buffer x) (.offset x) (.stride x)
                            (.buffer y) (.offset y) (.stride y)))))
   y)
  ([modl hstream kernel-name ^CUBlockVector x ^CUBlockVector y ^CUBlockVector z]
   (when (< 0 (.dim x))
     (with-release [math-kernel (function modl kernel-name)]
       (launch! math-kernel (grid-1d (.dim x)) hstream
                (parameters (.dim x)
                            (.buffer x) (.offset x) (.stride x)
                            (.buffer y) (.offset y) (.stride y)
                            (.buffer z) (.offset z) (.stride z)))))
   z))

(defn ^:private vector-linear-frac [modl hstream ^CUBlockVector x ^CUBlockVector y
                                    scalea shifta scaleb shiftb ^CUBlockVector z]
 (when (< 0 (.dim x))
   (let [da (data-accessor x)]
     (if (and (= 0.0 scaleb) (= 1.0 shiftb))
       (with-release [math-kernel (function modl "vector_scale_shift")]
         (launch! math-kernel (grid-1d (.dim x)) hstream
                  (parameters (.dim x)
                              (.buffer x) (.offset x) (.stride x)
                              (.wrapPrim da scalea) (.wrapPrim da shifta)
                              (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                              (.buffer z) (.offset z) (.stride z))))
       (with-release [math-kernel (function modl "vector_linear_frac")]
         (launch! math-kernel (grid-1d (.dim x)) hstream
                  (parameters (.dim x)
                              (.buffer x) (.offset x) (.stride x)
                              (.buffer y) (.offset y) (.stride y)
                              (.wrapPrim da scalea) (.wrapPrim da shifta)
                              (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                              (.buffer z) (.offset z) (.stride z)))))))
  z)

(defn ^:private vector-powx [modl hstream ^CUBlockVector x b ^CUBlockVector y]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [math-kernel (function modl "vector_powx")]
        (launch! math-kernel (grid-1d (.dim x)) hstream
                 (parameters (.dim x)
                             (.buffer x) (.offset x) (.stride x)
                             (.wrapPrim da b)
                             (.buffer y) (.offset y) (.stride y))))))
  y)

(defn ^:private vector-relu [modl hstream kernel-name alpha ^CUBlockVector x ^CUBlockVector y]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [math-kernel (function modl kernel-name)]
        (launch! math-kernel (grid-1d (.dim x)) hstream
                 (parameters (.dim x)
                             (.wrapPrim da alpha)
                             (.buffer x) (.offset x) (.stride x)
                             (.buffer y) (.offset y) (.stride y))))))
  y)

(defn ^:private ge-math
  ([modl hstream kernel-name ^CUGEMatrix a ^CUGEMatrix b]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl kernel-name)]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.fd stor)
                              (.buffer a) (.offset a) (.stride a)
                              (.buffer b) (.offset b) (.stride b))))))
   b)
  ([modl hstream kernel-name ^CUGEMatrix a ^CUGEMatrix b ^CUGEMatrix c]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl kernel-name)]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.fd stor)
                              (.buffer a) (.offset a) (.stride a)
                              (.buffer b) (.offset b) (.stride b)
                              (.buffer c) (.offset c) (.stride c))))))
   c))

(defn ^:private ge-linear-frac [modl hstream ^CUGEMatrix a ^CUGEMatrix b
                                scalea shifta scaleb shiftb ^CUGEMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (function modl "ge_scale_shift")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.fd stor)
                               (.buffer a) (.offset a) (.stride a)
                               (.wrapPrim da scalea) (.wrapPrim da shifta)
                               (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                               (.buffer c) (.offset c) (.stride c))))
        (with-release [math-kernel (function modl "ge_linear_frac")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.fd stor)
                               (.buffer a) (.offset a) (.stride a)
                               (.buffer b) (.offset b) (.stride b)
                               (.wrapPrim da scalea) (.wrapPrim da shifta)
                               (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                               (.buffer c) (.offset c) (.stride c)))))))
  c)

(defn ^:private ge-powx [modl hstream ^CUGEMatrix a b ^CUGEMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl "ge_powx")]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor)
                             (.buffer a) (.offset a) (.stride a)
                             (.wrapPrim da b)
                             (.buffer c) (.offset c) (.stride c))))))
  c)

(defn ^:private ge-relu [modl hstream kernel-name alpha ^CUGEMatrix a ^CUGEMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl kernel-name)]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor)
                             (.wrapPrim da alpha)
                             (.buffer a) (.offset a) (.stride a)
                             (.buffer c) (.offset c) (.stride c))))))
  c)

(defn ^:private uplo-math
  ([modl hstream kernel-name ^CUUploMatrix a ^CUUploMatrix b]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl kernel-name)]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                              (.buffer a) (.offset a) (.stride a)
                              (.buffer b) (.offset b) (.stride b))))))
   b)
  ([modl hstream kernel-name ^CUUploMatrix a ^CUUploMatrix b ^CUUploMatrix c]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl kernel-name)]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                              (.buffer a) (.offset a) (.stride a)
                              (.buffer b) (.offset b) (.stride b)
                              (.buffer c) (.offset c) (.stride c))))))
   c))

(defn ^:private uplo-linear-frac [modl hstream ^CUUploMatrix a ^CUUploMatrix b
                                  scalea shifta scaleb shiftb ^CUUploMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (function modl "uplo_scale_shift")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                               (.buffer a) (.offset a) (.stride a)
                               (.wrapPrim da scalea) (.wrapPrim da shifta)
                               (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                               (.buffer c) (.offset c) (.stride c))))
        (with-release [math-kernel (function modl "uplo_linear_frac")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                               (.buffer a) (.offset a) (.stride a)
                               (.buffer b) (.offset b) (.stride b)
                               scalea shifta scaleb shiftb
                               (.buffer c) (.offset c) (.stride c)))))))
  c)

(defn ^:private uplo-powx [modl hstream ^CUUploMatrix a b ^CUUploMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl "uplo_powx")]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.buffer a) (.offset a) (.stride a)
                             (.wrapPrim da b)
                             (.buffer c) (.offset c) (.stride c))))))
  c)

(defn ^:private uplo-relu [modl hstream kernel-name alpha ^CUUploMatrix a ^CUUploMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl kernel-name)]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.wrapPrim da alpha)
                             (.buffer a) (.offset a) (.stride a)
                             (.buffer c) (.offset c) (.stride c))))))
  c)

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
    (vector-method cublas-handle JCublas2/cublasDcopy ^CUBlockVector x ^CUBlockVector y))
  (dot [_ x y]
    (vector-dot cublas-handle double-array JCublas2/cublasDdot ^CUBlockVector x ^CUBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-reducer cublas-handle double-array JCublas2/cublasDnrm2 ^CUBlockVector x))
  (nrmi [_ _]
    (not-available))
  (asum [_ x]
    (vector-reducer cublas-handle double-array JCublas2/cublasDasum ^CUBlockVector x))
  (iamax [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIdamax ^CUBlockVector x)))))
  (iamin [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIdamin ^CUBlockVector x)))))
  (rot [_ x y c s]
    (vector-rot cublas-handle JCublas2/cublasDrot ^CUBlockVector x ^CUBlockVector y (double c) (double s)))
  (rotg [_ _]
    (not-available))
  (rotm [_ x y param]
    (vector-rotm cublas-handle JCublas2/cublasDrotm  ^CUBlockVector x ^CUBlockVector y ^CUBlockVector param))
  (rotmg [_ _ _]
    (not-available))
  (scal [_ alpha x]
    (vector-scal cublas-handle JCublas2/cublasDscal (double alpha) ^CUBlockVector x))
  (axpy [_ alpha x y]
    (vector-axpy cublas-handle JCublas2/cublasDaxpy (double alpha) ^CUBlockVector x ^CUBlockVector y))
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky))
  (sum [_ x]
    (vector-sum modl hstream ^CUBlockVector x))
  (imax [_ x]
    (not-available))
  (imin [this x]
    (not-available))
  (set-all [_ alpha x]
    (vector-set modl hstream alpha ^CUBlockVector x))
  (axpby [_ alpha x beta y]
    (vector-axpby modl hstream alpha x beta y))
  VectorMath
  (sqr [_ a y]
    (vector-math modl hstream "vector_sqr" a y))
  (mul [_ a b y]
    (vector-math modl hstream "vector_mul" a b y))
  (div [_ a b y]
    (vector-math modl hstream "vector_div" a b y))
  (inv [_ a y]
    (vector-math modl hstream "vector_inv" a y))
  (abs [_ a y]
    (vector-math modl hstream "vector_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (vector-math modl hstream "vector_fmod" a b y))
  (frem [_ a b y]
    (vector-math modl hstream "vector_frem" a b y))
  (sqrt [_ a y]
    (vector-math modl hstream "vector_sqrt" a y))
  (inv-sqrt [_ a y]
    (vector-math modl hstream "vector_inv_sqrt" a y))
  (cbrt [_ a y]
    (vector-math modl hstream "vector_cbrt" a y))
  (inv-cbrt [_ a y]
    (vector-math modl hstream "vector_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (vector-math modl hstream "vector_pow2o3" a y))
  (pow3o2 [_ a y]
    (vector-math modl hstream "vector_pow3o2" a y))
  (pow [_ a b y]
    (vector-math modl hstream "vector_pow" a b y))
  (powx [_ a b y]
    (vector-powx modl hstream a b y))
  (hypot [_ a b y]
    (vector-math modl hstream "vector_hypot" a b y))
  (exp [_ a y]
    (vector-math modl hstream "vector_exp" a y))
  (expm1 [_ a y]
    (vector-math modl hstream "vector_expm1" a y))
  (log [_ a y]
    (vector-math modl hstream "vector_log" a y))
  (log10 [_ a y]
    (vector-math modl hstream "vector_log10" a y))
  (sin [_ a y]
    (vector-math modl hstream "vector_sin" a y))
  (cos [_ a y]
    (vector-math modl hstream "vector_cos" a y))
  (tan [_ a y]
    (vector-math modl hstream "vector_tan" a y))
  (sincos [_ a y z]
    (vector-math modl hstream "vector_sincos" a y z))
  (asin [_ a y]
    (vector-math modl hstream "vector_asin" a y))
  (acos [_ a y]
    (vector-math modl hstream "vector_acos" a y))
  (atan [_ a y]
    (vector-math modl hstream "vector_atan" a y))
  (atan2 [_ a b y]
    (vector-math modl hstream "vector_atan2" a b y))
  (sinh [_ a y]
    (vector-math modl hstream "vector_sinh" a y))
  (cosh [_ a y]
    (vector-math modl hstream "vector_cosh" a y))
  (tanh [_ a y]
    (vector-math modl hstream "vector_tanh" a y))
  (asinh [_ a y]
    (vector-math modl hstream "vector_asinh" a y))
  (acosh [_ a y]
    (vector-math modl hstream "vector_acosh" a y))
  (atanh [_ a y]
    (vector-math modl hstream "vector_atanh" a y))
  (erf [_ a y]
    (vector-math modl hstream "vector_erf" a y))
  (erfc [_ a y]
    (vector-math modl hstream "vector_erfc" a y))
  (erf-inv [_ a y]
    (vector-math modl hstream "vector_erf_inv" a y))
  (erfc-inv [_ a y]
    (vector-math modl hstream "vector_erfc_inv" a y))
  (cdf-norm [_ a y]
    (vector-math modl hstream "vector_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (vector-math modl hstream "vector_cdf_norm_inv" a y))
  (gamma [_ a y]
    (vector-math modl hstream "vector_gamma" a y))
  (lgamma [_ a y]
    (vector-math modl hstream "vector_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (vector-math modl hstream "vector_floor" a y))
  (fceil [_ a y]
    (vector-math modl hstream "vector_ceil" a y))
  (trunc [_ a y]
    (vector-math modl hstream "vector_trunc" a y))
  (round [_ a y]
    (vector-math modl hstream "vector_round" a y))
  (modf [_ a y z]
    (vector-math modl hstream "vector_modf" a y z))
  (frac [_ a y]
    (vector-math modl hstream "vector_frac" a y))
  (fmin [_ a b y]
    (vector-math modl hstream "vector_fmin" a b y))
  (fmax [_ a b y]
    (vector-math modl hstream "vector_fmax" a b y))
  (copy-sign [_ a b y]
    (vector-math modl hstream "vector_copysign" a b y))
  (sigmoid [this a y]
    (vector-math modl hstream "vector_sigmoid" a y))
  (ramp [this a y]
    (vector-math modl hstream "vector_ramp" a y))
  (relu [this alpha a y]
    (vector-relu modl hstream "vector_relu" alpha a y))
  (elu [this alpha a y]
    (vector-relu modl hstream "vector_elu" alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (vector-random modl hstream "vector_uniform_double"
                   (or rng-stream (atom (generate-seed))) lower upper x))
  (rand-normal [_ rng-stream mu sigma x]
    (vector-random modl hstream "vector_normal_double"
                   (or rng-stream (atom (generate-seed))) mu sigma x)))

(deftype FloatVectorEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals modl hstream ^CUBlockVector x ^CUBlockVector y))
  Blas
  (swap [_ x y]
    (vector-method cublas-handle JCublas2/cublasSswap ^CUBlockVector x ^CUBlockVector y)
    x)
  (copy [_ x y]
    (vector-method cublas-handle JCublas2/cublasScopy ^CUBlockVector x ^CUBlockVector y))
  (dot [_ x y]
    (vector-dot cublas-handle float-array JCublas2/cublasSdot ^CUBlockVector x ^CUBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-reducer cublas-handle float-array JCublas2/cublasSnrm2 ^CUBlockVector x))
  (nrmi [_ _]
    (not-available))
  (asum [_ x]
    (vector-reducer cublas-handle float-array JCublas2/cublasSasum ^CUBlockVector x))
  (iamax [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIsamax ^CUBlockVector x)))))
  (iamin [_ x]
    (max 0 (dec (long (vector-reducer cublas-handle int-array JCublas2/cublasIsamin ^CUBlockVector x)))))
  (rot [_ x y c s]
    (vector-rot cublas-handle JCublas2/cublasSrot ^CUBlockVector x ^CUBlockVector y (float c) (float s)))
  (rotg [_ _]
    (not-available))
  (rotm [_ x y param]
    (vector-rotm cublas-handle JCublas2/cublasSrotm  ^CUBlockVector x ^CUBlockVector y ^CUBlockVector param))
  (rotmg [_ _ _]
    (not-available))
  (scal [_ alpha x]
    (vector-scal cublas-handle JCublas2/cublasSscal (float alpha) ^CUBlockVector x))
  (axpy [_ alpha x y]
    (vector-axpy cublas-handle JCublas2/cublasSaxpy (float alpha) ^CUBlockVector x ^CUBlockVector y))
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy modl hstream ^CUBlockVector x ^CUBlockVector y kx lx ky))
  (sum [_ x]
    (vector-sum modl hstream ^CUBlockVector x))
  (imax [_ x]
    (not-available))
  (imin [this x]
    (not-available))
  (set-all [_ alpha x]
    (vector-set modl hstream alpha ^CUBlockVector x))
  (axpby [_ alpha x beta y]
    (vector-axpby modl hstream alpha x beta y))
  VectorMath
  (sqr [_ a y]
    (vector-math modl hstream "vector_sqr" a y))
  (mul [_ a b y]
    (vector-math modl hstream "vector_mul" a b y))
  (div [_ a b y]
    (vector-math modl hstream "vector_div" a b y))
  (inv [_ a y]
    (vector-math modl hstream "vector_inv" a y))
  (abs [_ a y]
    (vector-math modl hstream "vector_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (vector-math modl hstream "vector_fmod" a b y))
  (frem [_ a b y]
    (vector-math modl hstream "vector_frem" a b y))
  (sqrt [_ a y]
    (vector-math modl hstream "vector_sqrt" a y))
  (inv-sqrt [_ a y]
    (vector-math modl hstream "vector_inv_sqrt" a y))
  (cbrt [_ a y]
    (vector-math modl hstream "vector_cbrt" a y))
  (inv-cbrt [_ a y]
    (vector-math modl hstream "vector_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (vector-math modl hstream "vector_pow2o3" a y))
  (pow3o2 [_ a y]
    (vector-math modl hstream "vector_pow3o2" a y))
  (pow [_ a b y]
    (vector-math modl hstream "vector_pow" a b y))
  (powx [_ a b y]
    (vector-powx modl hstream a b y))
  (hypot [_ a b y]
    (vector-math modl hstream "vector_hypot" a b y))
  (exp [_ a y]
    (vector-math modl hstream "vector_exp" a y))
  (expm1 [_ a y]
    (vector-math modl hstream "vector_expm1" a y))
  (log [_ a y]
    (vector-math modl hstream "vector_log" a y))
  (log10 [_ a y]
    (vector-math modl hstream "vector_log10" a y))
  (sin [_ a y]
    (vector-math modl hstream "vector_sin" a y))
  (cos [_ a y]
    (vector-math modl hstream "vector_cos" a y))
  (tan [_ a y]
    (vector-math modl hstream "vector_tan" a y))
  (sincos [_ a y z]
    (vector-math modl hstream "vector_sincos" a y z))
  (asin [_ a y]
    (vector-math modl hstream "vector_asin" a y))
  (acos [_ a y]
    (vector-math modl hstream "vector_acos" a y))
  (atan [_ a y]
    (vector-math modl hstream "vector_atan" a y))
  (atan2 [_ a b y]
    (vector-math modl hstream "vector_atan2"  a b y))
  (sinh [_ a y]
    (vector-math modl hstream "vector_sinh" a y))
  (cosh [_ a y]
    (vector-math modl hstream "vector_cosh" a y))
  (tanh [_ a y]
    (vector-math modl hstream "vector_tanh"  a y))
  (asinh [_ a y]
    (vector-math modl hstream "vector_asinh" a y))
  (acosh [_ a y]
    (vector-math modl hstream "vector_acosh" a y))
  (atanh [_ a y]
    (vector-math modl hstream "vector_atanh" a y))
  (erf [_ a y]
    (vector-math modl hstream "vector_erf" a y))
  (erfc [_ a y]
    (vector-math modl hstream "vector_erfc" a y))
  (erf-inv [_ a y]
    (vector-math modl hstream "vector_erf_inv" a y))
  (erfc-inv [_ a y]
    (vector-math modl hstream "vector_erfc_inv" a y))
  (cdf-norm [_ a y]
    (vector-math modl hstream "vector_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (vector-math modl hstream "vector_cdf_norm_inv" a y))
  (gamma [_ a y]
    (vector-math modl hstream "vector_gamma" a y))
  (lgamma [_ a y]
    (vector-math modl hstream "vector_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (vector-math modl hstream "vector_floor" a y))
  (fceil [_ a y]
    (vector-math modl hstream "vector_ceil" a y))
  (trunc [_ a y]
    (vector-math modl hstream "vector_trunc" a y))
  (round [_ a y]
    (vector-math modl hstream "vector_round" a y))
  (modf [_ a y z]
    (vector-math modl hstream "vector_modf" a y z))
  (frac [_ a y]
    (vector-math modl hstream "vector_frac" a y))
  (fmin [_ a b y]
    (vector-math modl hstream "vector_fmin" a b y))
  (fmax [_ a b y]
    (vector-math modl hstream "vector_fmax" a b y))
  (copy-sign [_ a b y]
    (vector-math modl hstream "vector_copysign" a b y))
  (sigmoid [this a y]
    (vector-math modl hstream "vector_sigmoid" a y))
  (ramp [this a y]
    (vector-math modl hstream "vector_ramp" a y))
  (relu [this alpha a y]
    (vector-relu modl hstream "vector_relu" alpha a y))
  (elu [this alpha a y]
    (vector-relu modl hstream "vector_elu" alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (vector-random modl hstream "vector_uniform_float"
                   (or rng-stream (atom (generate-seed))) lower upper x))
  (rand-normal [_ rng-stream mu sigma x]
    (vector-random modl hstream "vector_normal_float"
                   (or rng-stream (atom (generate-seed))) mu sigma x)))

(deftype DoubleGEEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (ge-swap cublas-handle JCublas2/cublasDswap modl hstream ^CUGEMatrix a ^CUGEMatrix b)
    a)
  (copy [_ a b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double 1.0) ^CUGEMatrix a (double 0.0) ^CUGEMatrix b))
  (scal [_ alpha a]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a))
  (dot [_ a b]
    (ge-dot cublas-handle double-array JCublas2/cublasDdot ^CUGEMatrix a ^CUGEMatrix b))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [this a]
    (ge-asum-nrm2 cublas-handle double-array JCublas2/cublasDnrm2 modl hstream "ge_nrm2" ^CUGEMatrix a))
  (nrmi [_ _]
    (not-available))
  (asum [this a]
    (ge-asum-nrm2 cublas-handle double-array JCublas2/cublasDasum modl hstream "ge_asum" ^CUGEMatrix a))
  (axpy [_ alpha a b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a (double 1.0) ^CUGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv cublas-handle JCublas2/cublasDgemv
           (double alpha) ^CUGEMatrix a ^CUBlockVector x (double beta) ^CUBlockVector y))
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk cublas-handle JCublas2/cublasDger (double alpha) ^CUBlockVector x ^CUBlockVector y ^CUGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm cublas-handle JCublas2/cublasDgemm
           (double alpha) ^CUGEMatrix a ^CUGEMatrix b (double beta) ^CUGEMatrix c))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (ge-sum modl hstream a))
  (set-all [_ alpha a]
    (ge-set modl hstream alpha a))
  (axpby [_ alpha a beta b]
    (ge-am cublas-handle JCublas2/cublasDgeam (double alpha) ^CUGEMatrix a (double beta) ^CUGEMatrix b))
  (trans [_ a]
    (not-available))
  VectorMath
  (sqr [_ a y]
    (ge-math modl hstream "ge_sqr" a y))
  (mul [_ a b y]
    (ge-math modl hstream "ge_mul" a b y))
  (div [_ a b y]
    (ge-math modl hstream "ge_div" a b y))
  (inv [_ a y]
    (ge-math modl hstream "ge_inv" a y))
  (abs [_ a y]
    (ge-math modl hstream "ge_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (ge-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (ge-math modl hstream "ge_fmod" a b y))
  (frem [_ a b y]
    (ge-math modl hstream "ge_frem" a b y))
  (sqrt [_ a y]
    (ge-math modl hstream "ge_sqrt" a y))
  (inv-sqrt [_ a y]
    (ge-math modl hstream "ge_inv_sqrt" a y))
  (cbrt [_ a y]
    (ge-math modl hstream "ge_cbrt" a y))
  (inv-cbrt [_ a y]
    (ge-math modl hstream "ge_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (ge-math modl hstream "ge_pow2o3" a y))
  (pow3o2 [_ a y]
    (ge-math modl hstream "ge_pow3o2" a y))
  (pow [_ a b y]
    (ge-math modl hstream "ge_pow" a b y))
  (powx [_ a b y]
    (ge-powx modl hstream a b y))
  (hypot [_ a b y]
    (ge-math modl hstream "ge_hypot" a b y))
  (exp [_ a y]
    (ge-math modl hstream "ge_exp" a y))
  (expm1 [_ a y]
    (ge-math modl hstream "ge_expm1" a y))
  (log [_ a y]
    (ge-math modl hstream "ge_log" a y))
  (log10 [_ a y]
    (ge-math modl hstream "ge_log10" a y))
  (sin [_ a y]
    (ge-math modl hstream "ge_sin" a y))
  (cos [_ a y]
    (ge-math modl hstream "ge_cos" a y))
  (tan [_ a y]
    (ge-math modl hstream "ge_tan" a y))
  (sincos [_ a y z]
    (ge-math modl hstream "ge_sincos" a y z))
  (asin [_ a y]
    (ge-math modl hstream "ge_asin" a y))
  (acos [_ a y]
    (ge-math modl hstream "ge_acos" a y))
  (atan [_ a y]
    (ge-math modl hstream "ge_atan" a y))
  (atan2 [_ a b y]
    (ge-math modl hstream "ge_atan2"  a b y))
  (sinh [_ a y]
    (ge-math modl hstream "ge_sinh" a y))
  (cosh [_ a y]
    (ge-math modl hstream "ge_cosh" a y))
  (tanh [_ a y]
    (ge-math modl hstream "ge_tanh"  a y))
  (asinh [_ a y]
    (ge-math modl hstream "ge_asinh" a y))
  (acosh [_ a y]
    (ge-math modl hstream "ge_acosh" a y))
  (atanh [_ a y]
    (ge-math modl hstream "ge_atanh" a y))
  (erf [_ a y]
    (ge-math modl hstream "ge_erf" a y))
  (erfc [_ a y]
    (ge-math modl hstream "ge_erfc" a y))
  (erf-inv [_ a y]
    (ge-math modl hstream "ge_erf_inv" a y))
  (erfc-inv [_ a y]
    (ge-math modl hstream "ge_erfc_inv"a y))
  (cdf-norm [_ a y]
    (ge-math modl hstream "ge_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (ge-math modl hstream "ge_cdf_norm_inv" a y))
  (gamma [_ a y]
    (ge-math modl hstream "ge_gamma" a y))
  (lgamma [_ a y]
    (ge-math modl hstream "ge_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (ge-math modl hstream "ge_floor" a y))
  (fceil [_ a y]
    (ge-math modl hstream "ge_ceil" a y))
  (trunc [_ a y]
    (ge-math modl hstream "ge_trunc" a y))
  (round [_ a y]
    (ge-math modl hstream "ge_round" a y))
  (modf [_ a y z]
    (ge-math modl hstream "ge_modf" a y z))
  (frac [_ a y]
    (ge-math modl hstream "ge_frac" a y))
  (fmin [_ a b y]
    (ge-math modl hstream "ge_fmin" a b y))
  (fmax [_ a b y]
    (ge-math modl hstream "ge_fmax" a b y))
  (copy-sign [_ a b y]
    (ge-math modl hstream "ge_copysign" a b y))
  (sigmoid [this a y]
    (ge-math modl hstream "ge_sigmoid" a y))
  (ramp [this a y]
    (ge-math modl hstream "ge_ramp" a y))
  (relu [this alpha a y]
    (ge-relu modl hstream "ge_relu" alpha a y))
  (elu [this alpha a y]
    (ge-relu modl hstream "ge_elu" alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (ge-random modl hstream "ge_uniform_double"
               (or rng-stream (atom (generate-seed))) lower upper x))
  (rand-normal [_ rng-stream mu sigma x]
    (ge-random modl hstream "ge_normal_double"
               (or rng-stream (atom (generate-seed))) mu sigma x)))

(deftype FloatGEEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals modl hstream a b))
  Blas
  (swap [_ a b]
    (ge-swap cublas-handle JCublas2/cublasSswap modl hstream ^CUGEMatrix a ^CUGEMatrix b)
    a)
  (copy [_ a b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float 1) ^CUGEMatrix a (float 0) ^CUGEMatrix b))
  (scal [_ alpha a]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a))
  (dot [_ a b]
    (ge-dot cublas-handle float-array JCublas2/cublasSdot ^CUGEMatrix a ^CUGEMatrix b))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [this a]
    (ge-asum-nrm2 cublas-handle float-array JCublas2/cublasSnrm2 modl hstream "ge_nrm2" ^CUGEMatrix a))
  (nrmi [_ _]
    (not-available))
  (asum [this a]
    (ge-asum-nrm2 cublas-handle float-array JCublas2/cublasSasum modl hstream "ge_asum" ^CUGEMatrix a))
  (axpy [_ alpha a b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a (float 1.0) ^CUGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv cublas-handle JCublas2/cublasSgemv
           (float alpha) ^CUGEMatrix a ^CUBlockVector x (float beta) ^CUBlockVector y))
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk cublas-handle JCublas2/cublasSger (float alpha) ^CUBlockVector x ^CUBlockVector y ^CUGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm cublas-handle JCublas2/cublasSgemm
           (float alpha) ^CUGEMatrix a ^CUGEMatrix b (float beta) ^CUGEMatrix c))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (ge-sum modl hstream a))
  (set-all [_ alpha a]
    (ge-set modl hstream alpha a))
  (axpby [_ alpha a beta b]
    (ge-am cublas-handle JCublas2/cublasSgeam (float alpha) ^CUGEMatrix a (float beta) ^CUGEMatrix b))
  (trans [_ a]
    (not-available))
  VectorMath
  (sqr [_ a y]
    (ge-math modl hstream "ge_sqr" a y))
  (mul [_ a b y]
    (ge-math modl hstream "ge_mul" a b y))
  (div [_ a b y]
    (ge-math modl hstream "ge_div" a b y))
  (inv [_ a y]
    (ge-math modl hstream "ge_inv" a y))
  (abs [_ a y]
    (ge-math modl hstream "ge_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (ge-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (ge-math modl hstream "ge_fmod" a b y))
  (frem [_ a b y]
    (ge-math modl hstream "ge_frem" a b y))
  (sqrt [_ a y]
    (ge-math modl hstream "ge_sqrt" a y))
  (inv-sqrt [_ a y]
    (ge-math modl hstream "ge_inv_sqrt" a y))
  (cbrt [_ a y]
    (ge-math modl hstream "ge_cbrt" a y))
  (inv-cbrt [_ a y]
    (ge-math modl hstream "ge_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (ge-math modl hstream "ge_pow2o3" a y))
  (pow3o2 [_ a y]
    (ge-math modl hstream "ge_pow3o2" a y))
  (pow [_ a b y]
    (ge-math modl hstream "ge_pow" a b y))
  (powx [_ a b y]
    (ge-powx modl hstream a b y))
  (hypot [_ a b y]
    (ge-math modl hstream "ge_hypot" a b y))
  (exp [_ a y]
    (ge-math modl hstream "ge_exp" a y))
  (expm1 [_ a y]
    (ge-math modl hstream "ge_expm1" a y))
  (log [_ a y]
    (ge-math modl hstream "ge_log" a y))
  (log10 [_ a y]
    (ge-math modl hstream "ge_log10" a y))
  (sin [_ a y]
    (ge-math modl hstream "ge_sin" a y))
  (cos [_ a y]
    (ge-math modl hstream "ge_cos" a y))
  (tan [_ a y]
    (ge-math modl hstream "ge_tan" a y))
  (sincos [_ a y z]
    (ge-math modl hstream "ge_sincos" a y z))
  (asin [_ a y]
    (ge-math modl hstream "ge_asin" a y))
  (acos [_ a y]
    (ge-math modl hstream "ge_acos" a y))
  (atan [_ a y]
    (ge-math modl hstream "ge_atan" a y))
  (atan2 [_ a b y]
    (ge-math modl hstream "ge_atan2"  a b y))
  (sinh [_ a y]
    (ge-math modl hstream "ge_sinh" a y))
  (cosh [_ a y]
    (ge-math modl hstream "ge_cosh" a y))
  (tanh [_ a y]
    (ge-math modl hstream "ge_tanh"  a y))
  (asinh [_ a y]
    (ge-math modl hstream "ge_asinh" a y))
  (acosh [_ a y]
    (ge-math modl hstream "ge_acosh" a y))
  (atanh [_ a y]
    (ge-math modl hstream "ge_atanh" a y))
  (erf [_ a y]
    (ge-math modl hstream "ge_erf" a y))
  (erfc [_ a y]
    (ge-math modl hstream "ge_erfc" a y))
  (erf-inv [_ a y]
    (ge-math modl hstream "ge_erf_inv" a y))
  (erfc-inv [_ a y]
    (ge-math modl hstream "ge_erfc_inv"a y))
  (cdf-norm [_ a y]
    (ge-math modl hstream "ge_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (ge-math modl hstream "ge_cdf_norm_inv" a y))
  (gamma [_ a y]
    (ge-math modl hstream "ge_gamma" a y))
  (lgamma [_ a y]
    (ge-math modl hstream "ge_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (ge-math modl hstream "ge_floor" a y))
  (fceil [_ a y]
    (ge-math modl hstream "ge_ceil" a y))
  (trunc [_ a y]
    (ge-math modl hstream "ge_trunc" a y))
  (round [_ a y]
    (ge-math modl hstream "ge_round" a y))
  (modf [_ a y z]
    (ge-math modl hstream "ge_modf" a y z))
  (frac [_ a y]
    (ge-math modl hstream "ge_frac" a y))
  (fmin [_ a b y]
    (ge-math modl hstream "ge_fmin" a b y))
  (fmax [_ a b y]
    (ge-math modl hstream "ge_fmax" a b y))
  (copy-sign [_ a b y]
    (ge-math modl hstream "ge_copysign" a b y))
  (sigmoid [this a y]
    (ge-math modl hstream "ge_sigmoid" a y))
  (ramp [this a y]
    (ge-math modl hstream "ge_ramp" a y))
  (relu [this alpha a y]
    (ge-relu modl hstream "ge_relu" alpha a y))
  (elu [this alpha a y]
    (ge-relu modl hstream "ge_elu" alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (ge-random modl hstream "ge_uniform_float"
               (or rng-stream (atom (generate-seed))) lower upper x))
  (rand-normal [_ rng-stream mu sigma x]
    (ge-random modl hstream "ge_normal_float"
               (or rng-stream (atom (generate-seed))) mu sigma x)))

(deftype DoubleTREngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (uplo-equals modl hstream layout-match? a b))
  Blas
  (swap [_ a b]
    (uplo-map modl hstream layout-match? "uplo_swap" a b)
    a)
  (copy [_ a b]
    (uplo-map modl hstream layout-match? "uplo_copy" a b))
  (scal [_ alpha a]
    (uplo-set-scal modl hstream "uplo_scal" alpha a))
  (axpy [_ alpha a b]
    (uplo-axpby modl hstream layout-match? alpha ^CUUploMatrix a 1.0 b))
  (dot [_ _ _]
    (not-available))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (nrmi [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv cublas-handle JCublas2/cublasDtrmv ^CUUploMatrix a ^CUBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm cublas-handle JCublas2/cublasDtrmm (double alpha) ^CUUploMatrix a ^CUGEMatrix b left))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (uplo-sum modl hstream "tr_sum" a))
  (set-all [_ alpha a]
    (uplo-set-scal modl hstream "uplo_set" alpha a))
  (axpby [_ alpha a beta b]
    (uplo-axpby modl hstream layout-match? alpha ^CUUploMatrix a beta b))
  Lapack
  (srt [_ a increasing]
    (not-available))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (not-available))
  (trs [_ a b]
    (tr-sv cublas-handle JCublas2/cublasDtrsm (double 1.0) ^CUUploMatrix a ^CUGEMatrix b))
  (sv [_ a b _]
    (tr-sv cublas-handle JCublas2/cublasDtrsm (double 1.0) ^CUUploMatrix a ^CUGEMatrix b))
  (con [_ a nrm1?]
    (not-available))
  VectorMath
  (sqr [_ a y]
    (uplo-math modl hstream "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math modl hstream "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math modl hstream "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math modl hstream "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math modl hstream "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math modl hstream "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math modl hstream "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math modl hstream "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math modl hstream "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math modl hstream "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math modl hstream "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math modl hstream "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math modl hstream "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math modl hstream "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx modl hstream a b y))
  (hypot [_ a b y]
    (uplo-math modl hstream "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math modl hstream "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math modl hstream "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math modl hstream "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math modl hstream "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math modl hstream "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math modl hstream "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math modl hstream "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math modl hstream "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math modl hstream "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math modl hstream "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math modl hstream "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math modl hstream "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math modl hstream "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math modl hstream "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math modl hstream "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math modl hstream "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math modl hstream "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math modl hstream "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math modl hstream "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math modl hstream "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math modl hstream "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (uplo-math modl hstream "uplo_erfc_inv"a y))
  (cdf-norm [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm_inv" a y))
  (gamma [_ a y]
    (uplo-math modl hstream "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math modl hstream "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math modl hstream "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math modl hstream "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math modl hstream "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math modl hstream "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math modl hstream "uplo_modf" a y z))
  (frac [_ a y]
    (uplo-math modl hstream "uplo_frac" a y))
  (fmin [_ a b y]
    (uplo-math modl hstream "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math modl hstream "uplo_fmax" a b y))
  (copy-sign [_ a b y]
    (uplo-math modl hstream "uplo_copysign" a b y))
  (sigmoid [this a y]
    (uplo-math modl hstream "uplo_sigmoid" a y))
  (ramp [this a y]
    (uplo-math modl hstream "uplo_ramp" a y))
  (relu [this alpha a y]
    (uplo-relu modl hstream "uplo_relu" alpha a y))
  (elu [this alpha a y]
    (uplo-relu modl hstream "uplo_elu" alpha a y)))

(deftype FloatTREngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (uplo-equals modl hstream layout-match? a b))
  Blas
  (swap [_ a b]
    (uplo-map modl hstream layout-match? "uplo_swap" a b)
    a)
  (copy [_ a b]
    (uplo-map modl hstream layout-match? "uplo_copy" a b))
  (scal [_ alpha a]
    (uplo-set-scal modl hstream "uplo_scal" alpha a))
  (axpy [_ alpha a b]
    (uplo-axpby modl hstream layout-match? alpha a 1.0 b))
  (dot [_ _ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv cublas-handle JCublas2/cublasStrmv ^CUUploMatrix a ^CUBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm cublas-handle JCublas2/cublasStrmm (float alpha) ^CUUploMatrix a ^CUGEMatrix b left))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (uplo-sum modl hstream "tr_sum" a))
  (set-all [_ alpha a]
    (uplo-set-scal modl hstream "uplo_set" alpha a))
  (axpby [_ alpha a beta b]
    (uplo-axpby modl hstream layout-match? alpha a beta b))
  Lapack
  (srt [_ a increasing]
    (not-available))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (not-available))
  (trs [_ a b]
    (tr-sv cublas-handle JCublas2/cublasStrsm (float 1.0) ^CUUploMatrix a ^CUGEMatrix b))
  (sv [_ a b _]
    (tr-sv cublas-handle JCublas2/cublasStrsm (float 1.0) ^CUUploMatrix a ^CUGEMatrix b))
  (con [_ a nrm1?]
    (not-available))
  VectorMath
  (sqr [_ a y]
    (uplo-math modl hstream "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math modl hstream "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math modl hstream "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math modl hstream "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math modl hstream "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math modl hstream "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math modl hstream "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math modl hstream "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math modl hstream "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math modl hstream "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math modl hstream "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math modl hstream "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math modl hstream "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math modl hstream "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx modl hstream a b y))
  (hypot [_ a b y]
    (uplo-math modl hstream "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math modl hstream "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math modl hstream "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math modl hstream "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math modl hstream "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math modl hstream "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math modl hstream "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math modl hstream "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math modl hstream "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math modl hstream "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math modl hstream "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math modl hstream "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math modl hstream "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math modl hstream "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math modl hstream "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math modl hstream "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math modl hstream "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math modl hstream "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math modl hstream "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math modl hstream "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math modl hstream "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math modl hstream "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (uplo-math modl hstream "uplo_erfc_inv"a y))
  (cdf-norm [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm_inv" a y))
  (gamma [_ a y]
    (uplo-math modl hstream "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math modl hstream "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math modl hstream "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math modl hstream "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math modl hstream "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math modl hstream "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math modl hstream "uplo_modf" a y z))
  (frac [_ a y]
    (uplo-math modl hstream "uplo_frac" a y))
  (fmin [_ a b y]
    (uplo-math modl hstream "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math modl hstream "uplo_fmax" a b y))
  (copy-sign [_ a b y]
    (uplo-math modl hstream "uplo_copysign" a b y))
  (sigmoid [this a y]
    (uplo-math modl hstream "uplo_sigmoid" a y))
  (ramp [this a y]
    (uplo-math modl hstream "uplo_ramp" a y))
  (relu [this alpha a y]
    (uplo-relu modl hstream "uplo_relu" alpha a y))
  (elu [this alpha a y]
    (uplo-relu modl hstream "uplo_elu" alpha a y)))

(deftype DoubleSYEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (uplo-equals modl hstream symmetric-match? a b))
  Blas
  (swap [_ a b]
    (uplo-map modl hstream symmetric-match? "uplo_swap" a b)
    a)
  (copy [_ a b]
    (uplo-map modl hstream symmetric-match? "uplo_copy" a b))
  (scal [_ alpha a]
    (uplo-set-scal modl hstream "uplo_scal" alpha a))
  (axpy [_ alpha a b]
    (uplo-axpby modl hstream symmetric-match? alpha ^CUUploMatrix a 1.0 b))
  (dot [_ _ _]
    (not-available))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (nrmi [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (sy-mv cublas-handle JCublas2/cublasDsymv
           (double alpha) ^CUUploMatrix a ^CUBlockVector x (double beta) ^CUBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (rk [_ alpha x y a]
    (sy-r cublas-handle JCublas2/cublasDsyr2
          (double alpha) ^CUBlockVector x ^CUBlockVector y ^CUUploMatrix a))
  (rk [_ alpha x a]
    (sy-r cublas-handle JCublas2/cublasDsyr
          (double alpha) ^CUBlockVector x ^CUUploMatrix a))
  (srk [_ alpha a beta c]
    (sy-rk cublas-handle JCublas2/cublasDsyrk
           (double alpha) ^CUGEMatrix a (double beta) ^CUUploMatrix c))
  (mm [this alpha a b beta c left]
    (sy-mm cublas-handle JCublas2/cublasDsymm
           (double alpha) ^CUUploMatrix a ^CUGEMatrix b (double beta) ^CUGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (uplo-sum modl hstream "sy_sum" a))
  (set-all [_ alpha a]
    (uplo-set-scal modl hstream "uplo_set" alpha a))
  (axpby [_ alpha a beta b]
    (uplo-axpby modl hstream symmetric-match? alpha ^CUUploMatrix a beta b))
  VectorMath
  (sqr [_ a y]
    (uplo-math modl hstream "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math modl hstream "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math modl hstream "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math modl hstream "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math modl hstream "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math modl hstream "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math modl hstream "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math modl hstream "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math modl hstream "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math modl hstream "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math modl hstream "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math modl hstream "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math modl hstream "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math modl hstream "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx modl hstream a b y))
  (hypot [_ a b y]
    (uplo-math modl hstream "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math modl hstream "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math modl hstream "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math modl hstream "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math modl hstream "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math modl hstream "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math modl hstream "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math modl hstream "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math modl hstream "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math modl hstream "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math modl hstream "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math modl hstream "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math modl hstream "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math modl hstream "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math modl hstream "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math modl hstream "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math modl hstream "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math modl hstream "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math modl hstream "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math modl hstream "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math modl hstream "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math modl hstream "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (uplo-math modl hstream "uplo_erfc_inv"a y))
  (cdf-norm [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm_inv" a y))
  (gamma [_ a y]
    (uplo-math modl hstream "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math modl hstream "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math modl hstream "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math modl hstream "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math modl hstream "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math modl hstream "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math modl hstream "uplo_modf" a y z))
  (frac [_ a y]
    (uplo-math modl hstream "uplo_frac" a y))
  (fmin [_ a b y]
    (uplo-math modl hstream "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math modl hstream "uplo_fmax" a b y))
  (copy-sign [_ a b y]
    (uplo-math modl hstream "uplo_copysign" a b y))
  (sigmoid [this a y]
    (uplo-math modl hstream "uplo_sigmoid" a y))
  (ramp [this a y]
    (uplo-math modl hstream "uplo_ramp" a y))
  (relu [this alpha a y]
    (uplo-relu modl hstream "uplo_relu" alpha a y))
  (elu [this alpha a y]
    (uplo-relu modl hstream "uplo_elu" alpha a y)))

(deftype FloatSYEngine [cublas-handle modl hstream]
  BlockEngine
  (equals-block [_ a b]
    (uplo-equals modl hstream symmetric-match? a b))
  Blas
  (swap [_ a b]
    (uplo-map modl hstream symmetric-match? "uplo_swap" a b)
    a)
  (copy [_ a b]
    (uplo-map modl hstream symmetric-match? "uplo_copy" a b))
  (scal [_ alpha a]
    (uplo-set-scal modl hstream "uplo_scal" alpha a))
  (axpy [_ alpha a b]
    (uplo-axpby modl hstream symmetric-match? alpha a 1.0 b))
  (dot [_ _ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (mv [this alpha a x beta y]
    (sy-mv cublas-handle JCublas2/cublasSsymv
           (float alpha) ^CUUploMatrix a ^CUBlockVector x (float beta)  ^CUBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (rk [_ alpha x y a]
    (sy-r cublas-handle JCublas2/cublasSsyr2
          (float alpha) ^CUBlockVector x ^CUBlockVector y ^CUUploMatrix a))
  (rk [_ alpha x a]
    (sy-r cublas-handle JCublas2/cublasSsyr
          (float alpha) ^CUBlockVector x ^CUUploMatrix a))
  (srk [_ alpha a beta c]
    (sy-rk cublas-handle JCublas2/cublasSsyrk
           (float alpha) ^CUGEMatrix a (float beta) ^CUUploMatrix c))
  (mm [this alpha a b beta c left]
    (sy-mm cublas-handle JCublas2/cublasSsymm
           (float alpha) ^CUUploMatrix a ^CUGEMatrix b (float beta) ^CUGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ a]
    (uplo-sum modl hstream "sy_sum" a))
  (set-all [_ alpha a]
    (uplo-set-scal modl hstream "uplo_set" alpha a))
  (axpby [_ alpha a beta b]
    (uplo-axpby modl hstream symmetric-match? alpha a beta b))
  VectorMath
  (sqr [_ a y]
    (uplo-math modl hstream "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math modl hstream "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math modl hstream "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math modl hstream "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math modl hstream "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac modl hstream a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math modl hstream "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math modl hstream "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math modl hstream "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math modl hstream "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math modl hstream "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math modl hstream "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math modl hstream "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math modl hstream "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math modl hstream "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx modl hstream a b y))
  (hypot [_ a b y]
    (uplo-math modl hstream "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math modl hstream "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math modl hstream "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math modl hstream "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math modl hstream "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math modl hstream "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math modl hstream "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math modl hstream "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math modl hstream "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math modl hstream "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math modl hstream "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math modl hstream "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math modl hstream "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math modl hstream "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math modl hstream "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math modl hstream "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math modl hstream "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math modl hstream "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math modl hstream "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math modl hstream "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math modl hstream "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math modl hstream "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (uplo-math modl hstream "uplo_erfc_inv"a y))
  (cdf-norm [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (uplo-math modl hstream "uplo_cdf_norm_inv" a y))
  (gamma [_ a y]
    (uplo-math modl hstream "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math modl hstream "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math modl hstream "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math modl hstream "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math modl hstream "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math modl hstream "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math modl hstream "uplo_modf" a y z))
  (frac [_ a y]
    (uplo-math modl hstream "uplo_frac" a y))
  (fmin [_ a b y]
    (uplo-math modl hstream "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math modl hstream "uplo_fmax" a b y))
  (copy-sign [_ a b y]
    (uplo-math modl hstream "uplo_copysign" a b y))
  (sigmoid [this a y]
    (uplo-math modl hstream "uplo_sigmoid" a y))
  (ramp [this a y]
    (uplo-math modl hstream "uplo_ramp" a y))
  (relu [this alpha a y]
    (uplo-relu modl hstream "uplo_relu" alpha a y))
  (elu [this alpha a y]
    (uplo-relu modl hstream "uplo_elu" alpha a y)))

(deftype CUFactory [modl hstream ^DataAccessor da native-fact vector-eng ge-eng tr-eng sy-eng]
  Releaseable
  (release [_]
    (release modl)
    (release da)
    true)
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [_]
    native-fact)
  FlowProvider
  (flow [_]
    hstream)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  RngStreamFactory
  (create-rng-state [_ seed]
    (atom seed))
  Factory
  (create-vector [this n init]
    (let-release [res (cu-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-ge [this m n column? init]
    (let-release [res (cu-ge-matrix this m n column?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (cu-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (cu-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (cu-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng))

(extend-type cublasHandle
  Releaseable
  (release [this]
    (with-check cublas-error (JCublas2/cublasDestroy this) true)))

(defn ^:private get-cublas-stream [handle]
  (let [res (cudaStream_t.)]
    (with-check cublas-error (JCublas2/cublasGetStream handle res) (wrap (CUstream. res)))))

(defn ^:private cublas-handle
  "Creates a cuBLAS context handler on the specific `stream`."
  [hstream]
  (let [handle (cublasHandle.)
        cuda-stream (cudaStream_t. ^CUStream (extract hstream))]
    (with-check cublas-error (JCublas2/cublasCreate handle)
      (with-check cublas-error (JCublas2/cublasSetStream handle cuda-stream) handle))))

(let [src (str (slurp (io/resource "uncomplicate/clojurecuda/kernels/reduction.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/blas-plus.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/vect-math.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/random.cu")))
      standard-headers {"stdint.h" (slurp (io/resource "uncomplicate/clojurecuda/include/jitify/stdint.h"))
                        "float.h" (slurp (io/resource "uncomplicate/clojurecuda/include/jitify/float.h"))}
      philox-headers
      (merge standard-headers
             {"Random123/philox.h"
              (slurp (io/resource "uncomplicate/neanderthal/internal/device/include/Random123/philox.h"))
              "features/compilerfeatures.h"
              (slurp (io/resource "uncomplicate/neanderthal/internal/device/include/Random123/features/compilerfeatures.h"))
              "nvccfeatures.h"
              (slurp (io/resource "uncomplicate/neanderthal/internal/device/include/Random123/features/nvccfeatures.h"))
              "array.h" (slurp (io/resource "uncomplicate/neanderthal/internal/device/include/Random123/array.h"))})]

  (JCublas2/setExceptionsEnabled false)

  (defn cublas-double [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program src philox-headers)
                                   ["-DREAL=double" "-DACCUMULATOR=double"
                                    "-DCAST(fun)=fun" "-arch=compute_30"
                                    #_"-use_fast_math" "-default-device"
                                    (format "-DCUDART_VERSION=%s" (driver-version))])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)]
         (->CUFactory modl hstream (cu-double-accessor (current-context) hstream) native-double
                      (->DoubleVectorEngine handle modl hstream) (->DoubleGEEngine handle modl hstream)
                      (->DoubleTREngine handle modl hstream) (->DoubleSYEngine handle modl hstream))))))

  (defn cublas-float [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program src philox-headers)
                                   ["-DREAL=float" "-DACCUMULATOR=float"
                                    "-DCAST(fun)=fun##f" "-arch=compute_30"
                                    #_"-use_fast_math" "-default-device"
                                    (format "-DCUDART_VERSION=%s" (driver-version))])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)]
         (->CUFactory modl hstream (cu-float-accessor (current-context) hstream) native-float
                      (->FloatVectorEngine handle modl hstream) (->FloatGEEngine handle modl hstream)
                      (->FloatTREngine handle modl hstream) (->FloatSYEngine handle modl hstream)))))))
