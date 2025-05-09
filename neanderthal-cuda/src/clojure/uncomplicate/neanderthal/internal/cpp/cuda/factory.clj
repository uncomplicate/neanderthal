;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.cuda.factory
  (:refer-clojure :exclude [abs])
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release info]]
             [utils :refer [with-check dragan-says-ex count-groups generate-seed count-groups]]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.clojure-cpp :as cpp :refer [get-entry]]
            [uncomplicate.clojurecuda
             [core :refer :all :exclude [device]]
             [info :refer [driver-version]]
             [toolbox :refer [launch-reduce! read-int]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim ncols mrows entry matrix-type]]
             [native :refer [native-float native-double native-long native-int
                             native-short native-byte]]
             [block :refer [stride offset contiguous? column?]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage diag-unit?]]
             [common :refer [check-eq-navigators]]]
            [uncomplicate.neanderthal.internal.cpp.common :refer :all]
            [uncomplicate.neanderthal.internal.cpp.cuda
             [constants :refer :all]
             [structures :refer :all]]
            [uncomplicate.neanderthal.internal.device.common
             :refer [name-transp uplo-bottom? layout-match? symmetric-match?]])
  (:import org.bytedeco.cuda.global.cublas
           org.bytedeco.cuda.cublas.cublasContext
           org.bytedeco.cuda.cudart.CUstream_st
           [uncomplicate.neanderthal.internal.api Vector Matrix GEMatrix DataAccessor Region
            DenseStorage FullStorage LayoutNavigator]
           [uncomplicate.neanderthal.internal.cpp.cuda.structures
            CUBlockVector CUGEMatrix CUUploMatrix]))

(defprotocol HandleProvider
  (handle ^cublasContext [this]))

(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in CUDA. Please use a host instance.")))

(defn cu-blas
  ([prefix type name]
   (symbol (format "%s%s%s" prefix type name)))
  ([type name]
   (cu-blas "cublas" type name)))

;; =============== Random Number Generators =================================

(defn ^:private vector-random [modl hstream kernel-name rng-state a b x]
  (when (< 0 (dim x))
    (let [da (data-accessor x)]
      (with-release [random-kernel (function modl (str "vector_" kernel-name))]
        (launch! random-kernel (grid-1d (count-groups 4 (dim x))) hstream
                 (parameters (dim x) (long (swap! rng-state inc))
                             (.castPrim da a) (.castPrim da b) (extract x) (offset x) (stride x))))))
  x)

(defn ^:private ge-random [modl hstream kernel-name rng-state a b x]
  (when (< 0 (dim x))
    (let [da (data-accessor x)
          stor (full-storage x)]
      (with-release [random-kernel (function modl (str "ge_" kernel-name))]
        (launch! random-kernel (grid-2d (count-groups 4 (.sd stor)) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (long (swap! rng-state inc))
                             (.castPrim da a) (.castPrim da b)
                             (extract x) (offset x) (.ld stor))))))
  x)

;; =============== Common vector macros and functions =======================

(defn ^:private vector-equals [modl hstream x y]
  (let [cnt (dim x)]
    (if (< 0 cnt)
      (with-release [equals-kernel (function modl "vector_equals")
                     eq-flag-buf (mem-alloc-driver Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-1d cnt) hstream
                 (parameters cnt (extract x) (offset x) (stride x) (extract y) (offset y) (stride y)
                             eq-flag-buf))
        (= 0 (read-int hstream eq-flag-buf)))
      (= 0 (dim y)))))

(defn ^:private vector-copy
  ([modl hstream x y kx lx ky]
   (when (< 0 (int lx))
     (with-release [copy-kernel (function modl "vector_copy")]
       (launch! copy-kernel (grid-1d lx) hstream
                (parameters lx (extract x) (* (offset x) (long kx) (stride x)) (stride x)
                            (extract y) (+ (offset y) (* (long ky) (stride y))) (stride y)))))
   y)
  ([modl hstream x y]
   (with-release [copy-kernel (function modl "vector_copy")]
     (launch! copy-kernel (grid-1d (dim x)) hstream
              (parameters (dim x) (extract x) (offset x) (stride x) (extract y) (offset y) (stride y))))
   y))

(defn ^:private vector-swap [modl hstream x y]
  (with-release [copy-kernel (function modl "vector_swap")]
    (launch! copy-kernel (grid-1d (dim x)) hstream
             (parameters (dim x) (extract x) (offset x) (stride x) (extract y) (offset y) (stride y))))
  x)

(defn ^:private vector-sum [modl hstream x]
  (let [da (data-accessor x)
        cnt (dim x)
        block-dim 1024]
    (if (< 0 cnt)
      (with-release [sum-kernel (function modl "vector_sum")
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc-driver (* (.entryWidth da) (count-groups block-dim cnt)))
                     res (.wrapPrim da 0.0)]
        (launch-reduce! hstream sum-kernel sum-reduction-kernel
                        [(extract x) (offset x) (stride x) cu-acc] [cu-acc] cnt block-dim)
        (get-entry (memcpy-host! cu-acc res hstream) 0))
      0.0)))

(defn ^:private vector-set [modl hstream alpha x]
  (when (< 0 (dim x))
    (with-release [set-kernel (function modl "vector_set")]
      (launch! set-kernel (grid-1d (dim x)) hstream
               (parameters (dim x) (.castPrim (data-accessor x) alpha) (extract x) (offset x) (stride x)))))
  x)

(defn ^:private vector-axpby [modl hstream alpha x beta y]
  (when (< 0 (dim x))
    (let [da (data-accessor x)]
      (with-release [axpby-kernel (function modl "vector_axpby")]
        (launch! axpby-kernel (grid-1d (dim x)) hstream
                 (parameters (dim x)
                             (.castPrim da alpha) (extract x) (offset x) (stride x)
                             (.castPrim da beta) (extract y) (offset y) (stride y))))))
  y)

(defmacro ^:private vector-method
  ([cublas-handle cublas method ptr x]
   `(if (< 0 (dim ~x))
      (with-check cublas-error
        (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x))
        ~x)
      ~x))
  ([cublas-handle cublas method ptr x y]
   `(if (< 0 (dim ~x))
      (with-check cublas-error
        (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y))
        ~y)
      ~y))
  ([cublas-handle cublas method ptr x y z]
   `(if (< 0 (dim ~x))
      (with-check cublas-error
        (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y)
           (~ptr ~z) (stride ~z))
        ~z)
      ~z)))

(defmacro ^:private vector-dot [cublas-handle cublas method ptr x y]
  `(if (< 0 (dim ~x))
     (with-release [res# (create-vector (native-factory ~x) 1 false)]
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y) (~ptr res#))
         (entry res# 0)))
     0.0))

(defmacro ^:private vector-reducer [cublas-handle cublas method ptr x]
  `(if (< 0 (dim ~x))
     (with-release [res# (create-vector (native-factory ~x) 1 false)]
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr res#))
         (entry res# 0)))
     0.0))

(defmacro ^:private vector-imaxmin [cublas-handle cublas method ptr x]
  `(if (< 0 (dim ~x))
     (with-release [res# (create-vector (index-factory (native-factory ~x)) 1 false)]
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (int-ptr res#))
         (entry res# 0)))
     0.0))

(defmacro ^:private vector-scal [cublas-handle cublas method ptr cpp-ptr alpha x]
  `(if (< 0 (dim ~x))
     (with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~x) ~alpha))]
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) alpha# (~ptr ~x) (stride ~x))
         ~x))
     ~x))

(defmacro ^:private vector-axpy [cublas-handle cublas method ptr cpp-ptr alpha x y]
  `(if (< 0 (dim ~x))
     (with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~x) ~alpha))]
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) alpha# (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y))
         ~y))
     ~y))

(defmacro ^:private vector-rot [cublas-handle cublas method ptr cpp-ptr x y c s]
  `(if (and (< 0 (dim ~x)) (< 0 (dim ~y)))
     (let [da# (data-accessor ~x)]
       (with-release [c# (~cpp-ptr (.wrapPrim da# ~c))
                      s# (~cpp-ptr (.wrapPrim da# ~s))]
         (with-check cublas-error
           (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y) c# s#)
           ~x)))
     ~x))

(defmacro ^:private vector-rotm [cublas-handle cublas method ptr x y param]
  `(if (= 1 (stride ~param))
     (if (and (< 0 (dim ~x)) (< 0 (dim ~y)))
       (with-check cublas-error
         (. ~cublas ~method ~cublas-handle (dim ~x) (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y) (~ptr ~param))
         ~param)
       ~param)
     (throw (ex-info "You cannot use strided vector as param." {:param (info ~param)}))))

;; =============== Common GE matrix macros and functions =======================

(defn ^:private ge-equals [modl hstream a b]
  (if (< 0 (dim a))
    (let [stor (full-storage a)]
      (with-release [equals-kernel (function modl (name-transp "ge_equals" a b))
                     eq-flag-buf (mem-alloc-driver Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (extract a) (offset a) (.ld stor)
                             (extract b) (offset b) (stride b) eq-flag-buf))
        (= 0 (read-int hstream eq-flag-buf))))
    (= 0 (dim b))))

(defn ^:private ge-set [modl hstream alpha a]
  (if (< 0 (dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [ge-set-kernel (function modl "ge_set")]
        (launch! ge-set-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (.castPrim da alpha) (extract a) (offset a) (.ld stor))))
      a)))

(defmacro ^:private ge-swap [cublas-handle cublas method modl hstream ptr a b]
  `(if (< 0 (dim ~a))
     (let [da# (data-accessor ~a)
           stor# (full-storage ~a)]
       (if (and (= (navigator ~a) (navigator ~b)) (.isGapless stor#) (.isGapless (storage ~b)))
         (with-check cublas-error
           (. ~cublas ~method ~cublas-handle (dim ~a) (~ptr ~a) 1 (~ptr ~b) 1)
           ~a)
         (with-release [ge-swap-kernel# (function ~modl (name-transp "ge_swap" ~a ~b))]
           (launch! ge-swap-kernel# (grid-2d (.sd stor#) (.fd stor#)) ~hstream
                    (parameters (.sd stor#) (.fd stor#) (extract ~a) (offset ~a) (.ld stor#)
                                (extract ~b) (offset ~b) (stride ~b)))
           ~a)))
     ~a))

(defmacro ^:private ge-dot [cublas-handle cublas method ptr a b]
  `(if (< 0 (dim ~a))
     (if (and (= (navigator ~a) (navigator ~b)) (contiguous? ~a) (contiguous? ~b))
       (with-release [res# (create-vector (native-factory ~a) 1 false)]
         (with-check cublas-error
           (. ~cublas ~method ~cublas-handle (dim ~a) (~ptr ~a) 1 (~ptr ~b) 1 (~ptr res#))
           (entry res# 0)))
       (not-available))
     0.0))

(defmacro ^:private ge-asum-nrm2 [cublas-handle cublas method ptr a]
  `(if (< 0 (dim ~a))
     (if (contiguous? ~a)
       (with-release [res# (create-vector (native-factory ~a) 1 false)]
         (with-check cublas-error
           (. ~cublas ~method ~cublas-handle (dim ~a) (~ptr ~a) 1 (~ptr res#))
           (entry res# 0)))
       (not-available))
     0.0))

(defn ^:private ge-sum [modl hstream a]
  (let [da (data-accessor a)
        stor (full-storage a)
        cnt-sd (.sd stor)
        cnt-fd (.fd stor)
        wgs-sd 32
        wgs-fd 32
        wgs 1024
        grid-dim-x (count-groups wgs-sd cnt-sd)
        grid-dim-y (count-groups wgs-fd cnt-fd)
        acc-count (int (* grid-dim-x grid-dim-y))]
    (if (< 0 (dim a))
      (with-release [sum-kernel (function modl "ge_sum")
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc-driver (* Double/BYTES acc-count))
                     res (.wrapPrim da 0.0)
                     params (parameters acc-count cu-acc)]
        (launch! sum-kernel (grid-2d cnt-sd cnt-fd wgs-sd wgs-fd) hstream
                 (parameters cnt-sd cnt-fd cu-acc (extract a) (offset a) (stride a)))
        (if (< 1 acc-count)
          (launch-reduce! hstream sum-reduction-kernel sum-reduction-kernel
                          params params acc-count wgs))
        (get-entry (memcpy-host! cu-acc res hstream) 0))
      0.0)))

(defmacro ^:private ge-am
  ([cublas-handle cublas method ptr cpp-ptr alpha a beta b]
   `(if (< 0 (dim ~a))
      (let [b# (~ptr ~b)
            stor-b# (full-storage ~b)
            da# (data-accessor ~a)]
        (with-check cublas-error
          (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                         beta# (~cpp-ptr (.wrapPrim da# ~beta))]
            (. ~cublas ~method ~cublas-handle
               (if (= (navigator ~a) (navigator ~b)) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
               ~(:no-trans cublas-trans) (.sd stor-b#) (.fd stor-b#)
               alpha# (~ptr ~a) (stride ~a) beta# b# (stride ~b) b# (.ld stor-b#)))
          ~b))
      ~b))
  ([cublas-handle cublas method ptr cpp-ptr alpha a]
   `(if (< 0 (dim ~a))
      (let [a# (~ptr ~a)
            stor# (full-storage ~a)
            ld-a# (.ld stor#)
            da# (data-accessor ~a)]
        (with-check cublas-error
          (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                         beta# (~cpp-ptr (.wrapPrim da# 0))]
            (. ~cublas ~method ~cublas-handle ~(:no-trans cublas-trans) ~(:no-trans cublas-trans)
               (.sd stor#) (.fd stor#) alpha# a# ld-a# beta# a# ld-a# a# ld-a#))
          ~a))
      ~a)))

(defmacro ^:private ge-mv
  ([cublas-handle cublas method ptr cpp-ptr alpha a x beta y]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            da# (data-accessor ~a)]
        (with-check cublas-error
          (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                         beta# (~cpp-ptr (.wrapPrim da# ~beta))]
            (. ~cublas ~method ~cublas-handle
               (if (.isColumnMajor (navigator ~a)) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
               (.sd stor#) (.fd stor#) alpha# (~ptr ~a) (.ld stor#) (~ptr ~x) (stride ~x)
               beta# (~ptr ~y) (stride ~y)))
          ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (info ~a)}))))

(defmacro ^:private ge-rk [cublas-handle cublas method ptr cpp-ptr alpha x y a]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)]
       (with-check cublas-error
         (let [[v# w#] (if (column? ~a) [~x ~y] [~y ~x])]
           (with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~a) ~alpha))]
             (. ~cublas ~method ~cublas-handle (.sd stor#) (.fd stor#) alpha#
                (~ptr v#) (stride v#) (~ptr w#) (stride w#) (~ptr ~a) (.ld stor#))))
         ~a))
     ~a))

(defmacro ^:private ge-mm
  ([alpha a b]
   `(if-not (instance? GEMatrix ~b)
      (mm (engine ~b) ~alpha ~b ~a false)
      (dragan-says-ex "In-place mm! is not supported for GE matrices. Use QR factorization."
                      {:a (info ~a) :b (info ~b)} )))
  ([cublas-handle cublas method ptr cpp-ptr alpha a b beta c]
   `(if (< 0 (dim ~a))
      (if (instance? GEMatrix ~b)
        (let [da# (data-accessor ~a)
              nav-c# (navigator ~c)
              stor-c# (full-storage ~c)]
          (with-check cublas-error
            (let [[x# y# trans-x# trans-y#]
                  (if (.isColumnMajor nav-c#)
                    [~a ~b (= nav-c# (navigator ~a)) (= nav-c# (navigator ~b))]
                    [~b ~a (= nav-c# (navigator ~b)) (= nav-c# (navigator ~a))])]
              (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                             beta# (~cpp-ptr (.wrapPrim da# ~beta))]
                (. ~cublas ~method ~cublas-handle
                   (if trans-x# ~(:no-trans cublas-trans) ~(:trans cublas-trans))
                   (if trans-y# ~(:no-trans cublas-trans) ~(:trans cublas-trans))
                   (.sd stor-c#) (.fd stor-c#) (ncols ~a) alpha# (~ptr x#) (stride x#) (~ptr y#) (stride y#)
                   beta# (~ptr ~c) (.ld stor-c#))))
            ~c))
        (mm (engine ~b) ~alpha ~b ~a ~beta ~c false))
      ~c)))

;; =============== Common UPLO matrix macros and functions =======================

(defn ^:private uplo-equals [modl hstream transpf a b]
  (if (< 0 (dim a))
    (let [stor (full-storage a)]
      (with-release [equals-kernel (function modl (name-transp (transpf a b) "uplo_equals" a b))
                     eq-flag-buf (mem-alloc-driver Integer/BYTES)]
        (memset! eq-flag-buf 0)
        (launch! equals-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (extract a) (offset a) (.ld stor) (extract b) (offset b) (stride b)
                             eq-flag-buf))
        (= 0 (read-int hstream eq-flag-buf))))
    (= 0 (dim b))))

(defn ^:private uplo-map [modl hstream transpf op-name a b]
  (when (< 0 (dim a))
    (let [stor (full-storage a)]
      (with-release [map-kernel (function modl (name-transp (transpf a b) op-name a b))]
        (launch! map-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (extract a) (offset a) (.ld stor) (extract b) (offset b) (stride b))))))
  b)

(defn ^:private uplo-axpby [modl hstream transpf alpha a beta b]
  (when (< 0 (dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [axpby-kernel (function modl (name-transp (transpf a b) "uplo_axpby" a b))]
        (launch! axpby-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.castPrim da alpha) (extract a) (offset a) (.ld stor)
                             (.castPrim da beta) (extract b) (offset b) (stride b))))))
  b)

(defn ^:private uplo-set-scal [modl hstream op-name alpha a]
  (when (< 0 (dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [op-kernel (function modl op-name)]
        (launch! op-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (.castPrim da alpha) (extract a) (offset a) (.ld stor))))))
  a)

(defn ^:private uplo-sum ^double [modl hstream sum-kernel-name a]
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
    (if (< 0 (dim a))
      (with-release [sum-kernel (function modl sum-kernel-name)
                     sum-reduction-kernel (function modl "sum_reduction")
                     cu-acc (mem-alloc-driver (* Double/BYTES acc-count))
                     res (.wrapPrim da 0.0)
                     params (parameters acc-count cu-acc)]
        (launch! sum-kernel (grid-2d cnt-sd cnt-fd wgs-sd wgs-fd) hstream
                 (parameters cnt-sd cnt-fd (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             cu-acc (extract a) (offset a) (stride a)))
        (if (< 1 acc-count)
          (launch-reduce! hstream sum-reduction-kernel sum-reduction-kernel
                          params params acc-count wgs))
        (get-entry (memcpy-host! cu-acc res hstream) 0))
      0.0)))

(defmacro ^:private tr-mv
  ([cublas-handle cublas method ptr a x]
   `(if (< 0 (dim ~a))
      (with-check cublas-error
        (. ~cublas ~method ~cublas-handle
           (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
           (if (.isColumnMajor (navigator ~a)) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
           (if (.isDiagUnit (region ~a)) ~(:unit cublas-diag-unit) ~(:non-unit cublas-diag-unit))
           (ncols ~a) (~ptr ~a) (stride ~a) (~ptr ~x) (stride ~x))
        ~x)
      ~x))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private tr-mm
  ([cublas-handle cublas method ptr cpp-ptr alpha a b left]
   `(if (< 0 (dim ~a))
      (let [stor-b# (full-storage ~b)]
        (with-check cublas-error
          (with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~a) ~alpha))]
            (. ~cublas ~method ~cublas-handle
               (if ~left ~(:left cublas-side-mode) ~(:right cublas-side-mode))
               (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
               (if (= (navigator ~a) (navigator ~b)) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
               (if (.isDiagUnit (region ~a)) ~(:unit cublas-diag-unit) ~(:non-unit cublas-diag-unit))
               (.sd stor-b#) (.fd stor-b#)
               alpha# (~ptr ~a) (stride ~a) (~ptr ~b) (.ld stor-b#) (~ptr ~b) (.ld stor-b#)))
          ~b))
      ~b))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private sy-mv
  ([cublas-handle cublas method ptr cpp-ptr alpha a x beta y]
   `(if (< 0 (dim ~a))
      (let [da# (data-accessor ~a)]
        (with-check cublas-error
          (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                         beta# (~cpp-ptr (.wrapPrim da# ~beta))]
            (. ~cublas ~method ~cublas-handle
               (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
               (ncols ~a) alpha# (~ptr ~a) (stride ~a) (~ptr ~x) (stride ~x)
               beta# (~ptr ~y) (stride ~y)))
          ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for SY matrices." {:a (info ~a)}))))

(defmacro ^:private sy-r
  ([cublas-handle cublas method ptr cpp-ptr alpha x y a]
   `(with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~a) ~alpha))]
      (. ~cublas ~method ~cublas-handle
       (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
       (mrows ~a) alpha# (~ptr ~x) (stride ~x) (~ptr ~y) (stride ~y) (~ptr ~a) (stride ~a))
      ~a))
  ([cublas-handle cublas method ptr cpp-ptr alpha x a]
   `(with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~a) ~alpha))]
      (. ~cublas ~method ~cublas-handle
       (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
       (mrows ~a) alpha# (~ptr ~x) (stride ~x) (~ptr ~a) (stride ~a))
      ~a)))

(defmacro ^:private sy-rk [cublas-handle cublas method ptr cpp-ptr alpha a beta c]
  `(if (and (= :ge (matrix-type ~a)) (= :sy (matrix-type ~c)))
     (let [da# (data-accessor (native-factory ~a))]
       (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                      beta# (~cpp-ptr (.wrapPrim da# ~beta))]
         (. ~cublas ~method ~cublas-handle
            (if (uplo-bottom? ~c) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
            (if (column? ~a) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
            (mrows ~c) (ncols ~a) alpha# (~ptr ~a) (stride ~a) beta# (~ptr ~c) (stride ~c)))
       ~c)
     (throw (ex-info "sy-rk is only available for symmetric matrices." {:a (info ~a) :c (info ~c)}))))

(defmacro ^:private sy-mm
  ([cublas-handle cublas method ptr cpp-ptr alpha a b beta c left]
   `(if (< 0 (dim ~a))
      (let [da# (data-accessor ~a)
            stor-c# (full-storage ~c)]
        (if (= (navigator ~c) (navigator ~b))
          (with-check cublas-error
            (with-release [alpha# (~cpp-ptr (.wrapPrim da# ~alpha))
                           beta# (~cpp-ptr (.wrapPrim da# ~beta))]
              (. ~cublas ~method ~cublas-handle
                 (if ~left ~(:left cublas-side-mode) ~(:right cublas-side-mode))
                 (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
                 (.sd stor-c#) (.fd stor-c#) alpha# (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)
                 beta# (~ptr ~c) (stride ~c)))
            ~c)
          (dragan-says-ex "Both GE matrices in symmetric multiplication must have the same orientation."
                          {:b (info ~b) :c (info ~c)})))
      ~c))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro ^:private tr-sv
  [cublas-handle cublas method ptr cpp-ptr alpha a b]
  `(if (< 0 (dim ~a))
     (let [stor-b# (full-storage ~b)]
       (with-check cublas-error
         (with-release [alpha# (~cpp-ptr (.wrapPrim (data-accessor ~a) ~alpha))]
           (. ~cublas ~method ~cublas-handle ~(:left cublas-side-mode)
              (if (uplo-bottom? ~a) ~(:lower cublas-uplo) ~(:upper cublas-uplo))
              (if (= (navigator ~a) (navigator ~b)) ~(:no-trans cublas-trans) ~(:trans cublas-trans))
              (if (diag-unit? (region ~a)) ~(:unit cublas-diag-unit) ~(:non-unit cublas-diag-unit))
              (.sd stor-b#) (.fd stor-b#) alpha# (~ptr ~a) (stride ~a) (~ptr ~b) (.ld stor-b#)))
         ~b))
     ~b))

;; =============== Common vectorized math functions ============================

(defn ^:private vector-math
  ([modl hstream kernel-name x y]
   (when (< 0 (dim x))
     (with-release [math-kernel (function modl (str "vector_" kernel-name))]
       (launch! math-kernel (grid-1d (dim x)) hstream
                (parameters (dim x) (extract x) (offset x) (stride x) (extract y) (offset y) (stride y)))))
   y)
  ([modl hstream kernel-name x y z]
   (when (< 0 (dim x))
     (with-release [math-kernel (function modl (str "vector_" kernel-name))]
       (launch! math-kernel (grid-1d (dim x)) hstream
                (parameters (dim x) (extract x) (offset x) (stride x) (extract y) (offset y) (stride y)
                            (extract z) (offset z) (stride z)))))
   z))

(defn ^:private vector-linear-frac [modl hstream x y scalea shifta scaleb shiftb z]
  (when (< 0 (dim x))
    (let [da (data-accessor x)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (function modl "vector_scale_shift")]
          (launch! math-kernel (grid-1d (dim x)) hstream
                   (parameters (dim x) (extract x) (offset x) (stride x)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract z) (offset z) (stride z))))
        (with-release [math-kernel (function modl "vector_linear_frac")]
          (launch! math-kernel (grid-1d (dim x)) hstream
                   (parameters (dim x) (extract x) (offset x) (stride x)
                               (extract y) (offset y) (stride y)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract z) (offset z) (stride z)))))))
  z)

(defn ^:private vector-powx [modl hstream x b y]
  (when (< 0 (dim x))
    (let [da (data-accessor x)]
      (with-release [math-kernel (function modl "vector_powx")]
        (launch! math-kernel (grid-1d (dim x)) hstream
                 (parameters (dim x) (extract x) (offset x) (stride x) (.castPrim da b)
                             (extract y) (offset y) (stride y))))))
  y)

(defn ^:private ge-math
  ([modl hstream kernel-name a b]
   (when (< 0 (dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl (str "ge_" kernel-name))]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.fd stor) (extract a) (offset a) (stride a)
                              (extract b) (offset b) (stride b))))))
   b)
  ([modl hstream kernel-name a b c]
   (when (< 0 (dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl (str "ge_" kernel-name))]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.fd stor) (extract a) (offset a) (stride a)
                              (extract b) (offset b) (stride b)
                              (extract c) (offset c) (stride c))))))
   c))

(defn ^:private ge-linear-frac [modl hstream a b scalea shifta scaleb shiftb c]
  (when (< 0 (dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (function modl "ge_scale_shift")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.fd stor) (extract a) (offset a) (stride a)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract c) (offset c) (stride c))))
        (with-release [math-kernel (function modl "ge_linear_frac")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.fd stor)
                               (extract a) (offset a) (stride a) (extract b) (offset b) (stride b)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract c) (offset c) (stride c)))))))
  c)

(defn ^:private ge-powx [modl hstream a b c]
  (when (< 0 (dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl "ge_powx")]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.fd stor) (extract a) (offset a) (stride a)
                             (.castPrim da b) (extract c) (offset c) (stride c))))))
  c)

(defn ^:private uplo-math
  ([modl hstream kernel-name a b]
   (when (< 0 (dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl (str "uplo_" kernel-name))]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                              (extract a) (offset a) (stride a) (extract b) (offset b) (stride b))))))
   b)
  ([modl hstream kernel-name a b c]
   (when (< 0 (dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (function modl (str "uplo_" kernel-name))]
         (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                  (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                              (extract a) (offset a) (stride a) (extract b) (offset b) (stride b)
                              (extract c) (offset c) (stride c))))))
   c))

(defn ^:private uplo-linear-frac [modl hstream a b scalea shifta scaleb shiftb c]
  (when (< 0 (dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (function modl "uplo_scale_shift")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                               (extract a) (offset a) (stride a)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract c) (offset c) (stride c))))
        (with-release [math-kernel (function modl "uplo_linear_frac")]
          (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                   (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                               (extract a) (offset a) (stride a) (extract b) (offset b) (stride b)
                               (.castPrim da scalea) (.castPrim da shifta)
                               (.castPrim da scaleb) (.castPrim da shiftb)
                               (extract c) (offset c) (stride c)))))))
  c)

(defn ^:private uplo-powx [modl hstream a b c]
  (when (< 0 (dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)
          da (data-accessor a)]
      (with-release [math-kernel (function modl "uplo_powx")]
        (launch! math-kernel (grid-2d (.sd stor) (.fd stor)) hstream
                 (parameters (.sd stor) (.diag (region a)) (if (uplo-bottom? a) 1 -1)
                             (extract a) (offset a) (stride a) (.castPrim da b)
                             (extract c) (offset c) (stride c))))))
  c)

;; ======================== Integer Vector Engines ===========================================

(deftype IntegerVectorEngine [modl hstream]
  BlockEngine
  (equals-block [this x y]
    (vector-equals modl hstream x y))
  Blas
  (swap [this x y]
    (vector-swap modl hstream x y)
    x)
  (copy [this x y]
    (vector-copy modl hstream x y)
    y)
  (dot [_ _  _]
    (not-available))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [_ _]
    (not-available))
  (nrmi [_ _]
    (not-available))
  (asum [_ _]
    (not-available))
  (iamax [_ _]
    (not-available))
  (iamin [_ _]
    (not-available))
  (rot [_ _ _ _ _]
    (not-available))
  (rotg [_ _]
    (not-available))
  (rotm [_ _ _ _]
    (not-available))
  (rotmg [_ _ _]
    (not-available))
  (scal [_ _ _]
    (not-available))
  (axpy [_ _ _ _]
    (not-available))
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [this x y kx lx ky]
    (vector-copy modl hstream x y kx lx ky))
  (sum [_ _]
    (not-available))
  (imax [_ _]
    (not-available))
  (imin [_ _]
    (not-available))
  (set-all [this alpha x]
    (vector-set modl hstream alpha x))
  (axpby [_ alpha x beta y]
    (not-available)))

;; ================= Real Vector Engines ========================================

(defmacro real-vector-blas* [name t T ptr cpp-ptr cublas]
  `(extend-type ~name
     BlockEngine
     (equals-block [this# x# y#]
       (vector-equals (.-modl this#) (.-hstream this#) x# y#))
     Blas
     (swap [this# x# y#]
       (vector-method (handle this#) ~cublas ~(cu-blas T 'swap_v2) ~ptr x# y#)
       x#)
     (copy [this# x# y#]
       (vector-method (handle this#) ~cublas ~(cu-blas T 'copy_v2) ~ptr x# y#)
       y#)
     (dot [this# x# y#]
       (vector-dot (handle this#) ~cublas ~(cu-blas T 'dot_v2) ~ptr x# y#))
     (nrm1 [this# x#]
       (asum this# x#))
     (nrm2 [this# x#]
       (vector-reducer (handle this#) ~cublas ~(cu-blas T 'nrm2_v2) ~ptr x#))
     (nrmi [this# x#]
       (not-available))
     (asum [this# x#]
       (vector-reducer (handle this#) ~cublas ~(cu-blas T 'asum_v2) ~ptr x#))
     (iamax [this# x#]
       (max 0 (dec (long (vector-imaxmin (handle this#) ~cublas ~(cu-blas 'cublasI t 'amax_v2) ~ptr x#)))))
     (iamin [this# x#]
       (max 0 (dec (long (vector-imaxmin (handle this#) ~cublas ~(cu-blas 'cublasI t 'amin_v2) ~ptr x#)))))
     (rot [this# x# y# c# s#]
       (vector-rot (handle this#) ~cublas ~(cu-blas T 'rot_v2) ~ptr ~cpp-ptr x# y# c# s#))
     (rotg [this# abcs#]
       (not-available))
     (rotm [this# x# y# param#]
       (vector-rotm (handle this#) ~cublas ~(cu-blas T 'rotm_v2) ~ptr x# y# param#))
     (rotmg [this# d1d2xy# param#]
       (not-available))
     (scal [this# alpha# x#]
       (vector-scal (handle this#) ~cublas ~(cu-blas T 'scal_v2) ~ptr ~cpp-ptr alpha# x#))
     (axpy [this# alpha# x# y#]
       (vector-axpy (handle this#) ~cublas ~(cu-blas T 'axpy_v2) ~ptr ~cpp-ptr alpha# x# y#))))

(defmacro real-vector-blas-plus* [name]
  `(extend-type ~name
     BlasPlus
     (amax [_# _#]
       (not-available))
     (subcopy [this# x# y# kx# lx# ky#]
       (vector-copy (.-modl this#) (.-hstream this#) x# y# kx# lx# ky#))
     (sum [this# x#]
       (vector-sum (.-modl this#) (.-hstream this#) x#))
     (imax [this# x#]
       (not-available))
     (imin [this# x#]
       (not-available))
     (set-all [this# alpha# x#]
       (vector-set (.-modl this#) (.-hstream this#) alpha# x#)
       x#)
     (axpby [this# alpha# x# beta# y#]
       (vector-axpby (.-modl this#) (.-hstream this#) alpha# x# beta# y#)
       y#)))

(defmacro real-math* [name math-method linear-frac powx relu]
  `(extend-type ~name
     VectorMath
     (sqr [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "sqr" a# y#))
     (mul [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "mul" a# b# y#))
     (div [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "div" a# b# y#))
     (inv [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "inv" a# y#))
     (abs [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "abs" a# y#))
     (linear-frac [this# a# b# scalea# shifta# scaleb# shiftb# y#]
       (~linear-frac (.-modl this#) (.-hstream this#) a# b# scalea# shifta# scaleb# shiftb# y#))
     (fmod [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "fmod" a# b# y#))
     (frem [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "frem" a# b# y#))
     (sqrt [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "sqrt" a# y#))
     (inv-sqrt [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "inv_sqrt" a# y#))
     (cbrt [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "cbrt" a# y#))
     (inv-cbrt [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "inv_cbrt" a# y#))
     (pow2o3 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "pow2o3" a# y#))
     (pow3o2 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "pow3o2" a# y#))
     (pow [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "pow" a# b# y#))
     (powx [this# a# b# y#]
       (~powx (.-modl this#) (.-hstream this#) a# b# y#))
     (hypot [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "hypot" a# b# y#))
     (exp [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "exp" a# y#))
     (exp2 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "exp2" a# y#))
     (exp10 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "exp10" a# y#))
     (expm1 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "expm1" a# y#))
     (log [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "log" a# y#))
     (log2 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "log2" a# y#))
     (log10 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "log10" a# y#))
     (log1p [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "log1p" a# y#))
     (sin [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "sin" a# y#))
     (cos [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "cos" a# y#))
     (tan [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "tan" a# y#))
     (sincos [this# a# y# z#]
       (~math-method (.-modl this#) (.-hstream this#) "sincos" a# y# z#))
     (asin [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "asin" a# y#))
     (acos [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "acos" a# y#))
     (atan [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "atan" a# y#))
     (atan2 [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "atan2" a# b# y#))
     (sinh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "sinh" a# y#))
     (cosh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "cosh"a# y#))
     (tanh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "tanh" a# y#))
     (asinh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "asinh" a# y#))
     (acosh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "acosh" a# y#))
     (atanh [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "atanh" a# y#))
     (erf [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "erf" a# y#))
     (erfc [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "erfc" a# y#))
     (erf-inv [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "erf_inv" a# y#))
     (erfc-inv [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "erfc_inv" a# y#))
     (cdf-norm [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "cdf_norm" a# y#))
     (cdf-norm-inv [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "cdf_norm_inv" a# y#))
     (gamma [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "gamma" a# y#))
     (lgamma [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "lgamma" a# y#))
     (expint1 [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "expint1" a# y#))
     (floor [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "floor"  a# y#))
     (fceil [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "ceil" a# y#))
     (trunc [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "trunc" a# y#))
     (round [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "round" a# y#))
     (modf [this# a# y# z#]
       (~math-method (.-modl this#) (.-hstream this#) "modf" a# y# z#))
     (frac [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "frac" a# y#))
     (fmin [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "fmin" a# b# y#))
     (fmax [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "fmax" a# b# y#))
     (copy-sign [this# a# b# y#]
       (~math-method (.-modl this#) (.-hstream this#) "copysign" a# b# y#))
     (sigmoid [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "sigmoid" a# y#))
     (ramp [this# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "ramp" a# y#))
     (relu [this# alpha# a# y#]
       (~math-method (.-modl this#) (.-hstream this#) "relu" alpha# a# y#))
     (elu
       ([this# alpha# a# y#]
        (~math-method (.-modl this#) (.-hstream this#) "elu" alpha# a# y#))
       ([this# a# y#]
        (~math-method (.-modl this#) (.-hstream this#) "elu_1" a# y#)))))

(defmacro real-rng* [name random type]
  `(extend-type ~name
     RandomNumberGenerator
     (rand-uniform [this# rng-stream# lower# upper# x#]
       (~random (.-modl this#) (.-hstream this#) ~(str "uniform_" type)
        (or rng-stream# (atom (generate-seed))) lower# upper# x#)
       x#)
     (rand-normal [this# rng-stream# mu# sigma# x#]
       (~random (.-modl this#) (.-hstream this#) ~(str "normal_" type)
        (or rng-stream# (atom (generate-seed))) mu# sigma# x#)
       x#)))

(deftype FloatVectorEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-vector-blas* FloatVectorEngine "s" "S" float-ptr cpp/float-ptr cublas)
(real-vector-blas-plus* FloatVectorEngine)
(real-math* FloatVectorEngine vector-math vector-linear-frac vector-powx vector-relu)
(real-rng* FloatVectorEngine vector-random "float")

(deftype DoubleVectorEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-vector-blas* DoubleVectorEngine "d" "D" double-ptr cpp/double-ptr cublas)
(real-vector-blas-plus* DoubleVectorEngine)
(real-math* DoubleVectorEngine vector-math vector-linear-frac vector-powx vector-relu)
(real-rng* DoubleVectorEngine vector-random "double")

;; ================= Real GE Engines ========================================

(defmacro real-ge-blas* [name t ptr cpp-ptr cublas]
  `(extend-type ~name
     BlockEngine
     (equals-block [this# a# b#]
       (ge-equals (.-modl this#) (.-hstream this#) a# b#))
     Blas
     (swap [this# a# b#]
       (ge-swap (handle this#) ~cublas ~(cu-blas t 'swap_v2) (.-modl this#) (.-hstream this#) ~ptr a# b#)
       a#)
     (copy [this# a# b#]
       (ge-am (handle this#) ~cublas ~(cu-blas t 'geam) ~ptr ~cpp-ptr 1.0 a# 0.0 b#)
       b#)
     (dot [this# a# b#]
       (ge-dot (handle this#) ~cublas ~(cu-blas t 'dot_v2) ~ptr a# b#))
     (nrm1 [_# _#]
       (not-available))
     (nrm2 [this# a#]
       (ge-asum-nrm2 (handle this#) ~cublas ~(cu-blas t 'nrm2_v2) ~ptr a#))
     (nrmi [_# _#]
       (not-available))
     (asum [this# a#]
       (ge-asum-nrm2 (handle this#) ~cublas ~(cu-blas t 'asum_v2) ~ptr a#))
     (scal [this# alpha# a#]
       (ge-am (handle this#) ~cublas ~(cu-blas t 'geam) ~ptr ~cpp-ptr alpha# a#))
     (axpy [this# alpha# a# b#]
       (ge-am (handle this#) ~cublas ~(cu-blas t 'geam) ~ptr ~cpp-ptr alpha# a# 1.0 b#))
     (mv
       ([this# alpha# a# x# beta# y#]
        (ge-mv (handle this#) ~cublas ~(cu-blas t 'gemv_v2) ~ptr ~cpp-ptr alpha# a# x# beta# y#))
       ([_# a# _#]
        (ge-mv a#)))
     (rk [this# alpha# x# y# a#]
       (ge-rk (handle this#) ~cublas ~(cu-blas t 'ger_v2) ~ptr ~cpp-ptr alpha# x# y# a#))
     (mm
       ([this# alpha# a# b# _#]
        (ge-mm alpha# a# b#))
       ([this# alpha# a# b# beta# c# _#]
        (ge-mm (handle this#) ~cublas ~(cu-blas t 'gemm_v2) ~ptr ~cpp-ptr alpha# a# b# beta# c#)))))

(defmacro real-ge-blas-plus* [name t ptr cpp-ptr]
  `(extend-type ~name
     BlasPlus
     (amax [_# _#]
       (not-available))
     (sum [this# a#]
       (ge-sum (.-modl this#) (.-hstream this#) a#))
     (set-all [this# alpha# a#]
       (ge-set (.-modl this#) (.-hstream this#) alpha# a#))
     (axpby [this# alpha# a# beta# b#]
       (ge-am (handle this#) ~cublas ~(cu-blas t 'geam) ~ptr ~cpp-ptr alpha# a# beta# b#))
     (trans [_# _#]
       (not-available))))

(deftype FloatGEEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-ge-blas* FloatGEEngine "S" float-ptr cpp/float-ptr cublas)
(real-ge-blas-plus* FloatGEEngine "S" float-ptr cpp/float-ptr)
(real-math* FloatGEEngine ge-math ge-linear-frac ge-powx ge-relu)
(real-rng* FloatGEEngine ge-random "float")

(deftype DoubleGEEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-ge-blas* DoubleGEEngine "D" double-ptr cpp/double-ptr cublas)
(real-ge-blas-plus* DoubleGEEngine "D" double-ptr cpp/double-ptr)
(real-math* DoubleGEEngine ge-math ge-linear-frac ge-powx ge-relu)
(real-rng* DoubleGEEngine ge-random "double")

;; ================= Real TR Engines ========================================

(defmacro real-tr-blas* [name t ptr cpp-ptr cublas]
  `(extend-type ~name
     BlockEngine
     (equals-block [this# a# b#]
       (uplo-equals (.-modl this#) (.-hstream this#) layout-match? a# b#))
     Blas
     (swap [this# a# b#]
       (uplo-map (.-modl this#) (.-hstream this#) layout-match? "uplo_swap" a# b#)
       a#)
     (copy [this# a# b#]
       (uplo-map (.-modl this#) (.-hstream this#) layout-match? "uplo_copy" a# b#)
       b#)
     (dot [_# _# _#]
       (not-available))
     (nrm1 [_# _#]
       (not-available))
     (nrm2 [_# _#]
       (not-available))
     (nrmi [_# _#]
       (not-available))
     (asum [_# _#]
       (not-available))
     (scal [this# alpha# a#]
       (uplo-set-scal (.-modl this#) (.-hstream this#) "uplo_scal" alpha# a#))
     (axpy [this# alpha# a# b#]
       (uplo-axpby (.-modl this#) (.-hstream this#) layout-match? alpha# a# 1.0 b#))
     (mv
       ([_# _# a# _# _# _#]
        (tr-mv a#))
       ([this# a# x#]
        (tr-mv (handle this#) ~cublas ~(cu-blas t 'trmv_v2) ~ptr a# x#)))
     (rk [_# _# _# _# _#]
       (not-available))
     (mm
       ([this# alpha# a# b# left#]
        (tr-mm (handle this#) ~cublas ~(cu-blas t 'trmm_v2) ~ptr ~cpp-ptr alpha# a# b# left#))
       ([_# _# a# _# _# _# _#]
        (tr-mm a#)))))

(defmacro real-tr-blas-plus* [name]
  `(extend-type ~name
     BlasPlus
     (amax [_# _#]
       (not-available))
     (sum [this# a#]
       (uplo-sum (.-modl this#) (.-hstream this#) "tr_sum" a#))
     (set-all [this# alpha# a#]
       (uplo-set-scal (.-modl this#) (.-hstream this#) "uplo_set" alpha# a#))
     (axpby [this# alpha# a# beta# b#]
       (uplo-axpby (.-modl this#) (.-hstream this#) layout-match? alpha# a# 1.0 b#))
     (trans [_# _#]
       (not-available))))

(defmacro real-tr-lapack* [name t ptr cpp-ptr cublas]
  `(extend-type ~name
     Lapack
     (srt [_# _# _#]
       (not-available))
     (laswp [_# _# _# _# _#]
       (dragan-says-ex "There is no use for pivots when working with TR matrices."))
     (lapmr [_# _# _# _#]
       (dragan-says-ex "Not implemented for TR matrices."))
     (lapmt [_# _# _# _#]
       (dragan-says-ex "Not implemented for TR matrices."))
     (trf
       ([_# _# _#]
        (dragan-says-ex "Not implemented for TR matrices."))
       ([_# _#]
        (dragan-says-ex "Not implemented for TR matrices.")))
     (tri [_# _#]
       (not-available))
     (trs [this# a# b#]
       (tr-sv (handle this#) ~cublas ~(cu-blas t 'trsm_v2) ~ptr ~cpp-ptr 1.0 a# b#))
     (sv [this# a# b# _#]
       (tr-sv (handle this#) ~cublas ~(cu-blas t 'trsm_v2) ~ptr ~cpp-ptr 1.0 a# b#))
     (con [_# a# nrm1?#]
       (not-available))))

(deftype FloatTREngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-tr-blas* FloatTREngine "S" float-ptr cpp/float-ptr cublas)
(real-tr-blas-plus* FloatTREngine )
(real-tr-lapack* FloatTREngine "S" float-ptr cpp/float-ptr cublas)
(real-math* FloatTREngine ge-math ge-linear-frac ge-powx ge-relu)

(deftype DoubleTREngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-tr-blas* DoubleTREngine "D" double-ptr cpp/double-ptr cublas)
(real-tr-blas-plus* DoubleTREngine)
(real-tr-lapack* DoubleTREngine "D" double-ptr cpp/double-ptr cublas)
(real-math* DoubleTREngine ge-math ge-linear-frac ge-powx ge-relu)

;; ================= Real SY Engines ========================================

(defmacro real-sy-blas* [name t ptr cpp-ptr cublas]
  `(extend-type ~name
     BlockEngine
     (equals-block [this# a# b#]
       (uplo-equals (.-modl this#) (.-hstream this#) symmetric-match? a# b#))
     Blas
     (swap [this# a# b#]
       (uplo-map (.-modl this#) (.-hstream this#) symmetric-match? "uplo_swap" a# b#)
       a#)
     (copy [this# a# b#]
       (uplo-map (.-modl this#) (.-hstream this#) symmetric-match? "uplo_copy" a# b#)
       b#)
     (dot [_# _# _#]
       (not-available))
     (nrm1 [_# _#]
       (not-available))
     (nrm2 [_# _#]
       (not-available))
     (nrmi [_# _#]
       (not-available))
     (asum [_# _#]
       (not-available))
     (scal [this# alpha# a#]
       (uplo-set-scal (.-modl this#) (.-hstream this#) "uplo_scal" alpha# a#))
     (axpy [this# alpha# a# b#]
       (uplo-axpby (.-modl this#) (.-hstream this#) symmetric-match? alpha# a# 1.0 b#))
     (mv
       ([this# alpha# a# x# beta# y#]
        (sy-mv (handle this#) ~cublas ~(cu-blas t 'symv_v2) ~ptr ~cpp-ptr alpha# a# x# beta# y#))
       ([_# a# _#]
        (sy-mv a#)))
     (rk
       ([this# alpha# x# y# a#]
        (sy-r (handle this#) ~cublas ~(cu-blas t 'syr2_v2) ~ptr ~cpp-ptr alpha# x# y# a#))
       ([this# alpha# x# a#]
        (sy-r (handle this#) ~cublas ~(cu-blas t 'syr_v2) ~ptr ~cpp-ptr alpha# x# a#)))
     (srk [this# alpha# a# beta# c#]
       (sy-rk (handle this#) ~cublas ~(cu-blas t 'syrk_v2) ~ptr ~cpp-ptr alpha# a# beta# c#))
     (mm
       ([this# alpha# a# b# beta# c# left#]
        (sy-mm (handle this#) ~cublas ~(cu-blas t 'symm_v2) ~ptr ~cpp-ptr alpha# a# b# beta# c# left#))
       ([_# _# a# _# _##]
        (sy-mm a#)))))

(defmacro real-sy-blas-plus* [name]
  `(extend-type ~name
     BlasPlus
     (amax [_# _#]
       (not-available))
     (sum [this# a#]
       (uplo-sum (.-modl this#) (.-hstream this#) "sy_sum" a#))
     (set-all [this# alpha# a#]
       (uplo-set-scal (.-modl this#) (.-hstream this#) "uplo_set" alpha# a#))
     (axpby [this# alpha# a# beta# b#]
       (uplo-axpby (.-modl this#) (.-hstream this#) layout-match? alpha# a# 1.0 b#))
     (trans [_# _#]
       (not-available))))

(deftype FloatSYEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-sy-blas* FloatSYEngine "S" float-ptr cpp/float-ptr cublas)
(real-sy-blas-plus* FloatSYEngine )
(real-math* FloatSYEngine ge-math ge-linear-frac ge-powx ge-relu)

(deftype DoubleSYEngine [cublas-handle modl hstream]
  HandleProvider
  (handle [_]
    cublas-handle))
(real-sy-blas* DoubleSYEngine "D" double-ptr cpp/double-ptr cublas)
(real-sy-blas-plus* DoubleSYEngine)
(real-math* DoubleSYEngine ge-math ge-linear-frac ge-powx ge-relu)

;; ========================== cuBLAS Factory ==================================================

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
  (device [_]
    :cuda)
  RngStreamFactory
  (create-rng-state [_ seed]
    (atom seed))
  Factory
  (create-vector [this n init]
    (let-release [res (cu-block-vector this n)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-ge [this m n column? init]
    (let-release [res (cu-ge-matrix this m n column?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (cu-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (cu-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (extract res)))
      res))
  (create-sy [this n column? lower? init]
    (let-release [res (cu-uplo-matrix this n column? lower?)]
      (when init
        (.initialize da (extract res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng))

(extend-type cublasContext
  Releaseable
  (release [this]
    (with-check cublas-error (cublas/cublasDestroy_v2 this) true)))

(defn ^:private get-cublas-stream [^cublasContext handle]
  (let [res (CUstream_st.)]
    (with-check cublas-error (cublas/cublasGetStream_v2 handle res) res)))

(defn ^:private cublas-handle
  "Creates a cuBLAS context handler on the specific `stream`."
  [^CUstream_st stream]
  (let [handle (cublasContext.)]
    (with-check cublas-error (cublas/cublasCreate_v2 handle)
      (with-check cublas-error (cublas/cublasSetStream_v2 handle stream) handle))))

(let [src (str (slurp (io/resource "uncomplicate/clojurecuda/kernels/reduction.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/number.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/blas-plus.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/vect-math.cu"))
               (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/random.cu")))

      integer-src (slurp (io/resource "uncomplicate/neanderthal/internal/device/cuda/number.cu"))

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

  (defn cublas-double [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program src philox-headers)
                                   ["-DNUMBER=double" "-DREAL=double" "-DACCUMULATOR=double"
                                    "-DCAST(fun)=fun" #_"-use_fast_math" "-default-device"
                                    (format "-DCUDART_VERSION=%s" (driver-version))])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)
                     double-accessor (->DoublePointerAccessor ctx hstream
                                                              (fn [^long size]
                                                                (cuda-malloc size :double))
                                                              cuda-free!)]
         (->CUFactory modl hstream double-accessor native-double
                      (->DoubleVectorEngine handle modl hstream) (->DoubleGEEngine handle modl hstream)
                      (->DoubleTREngine handle modl hstream) (->DoubleSYEngine handle modl hstream))))))

  (defn cublas-float [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program src philox-headers)
                                   ["-DNUMBER=float" "-DREAL=float" "-DACCUMULATOR=float"
                                    "-DCAST(fun)=fun##f" #_"-use_fast_math" "-default-device"
                                    (format "-DCUDART_VERSION=%s" (driver-version))])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)
                     float-accessor (->FloatPointerAccessor ctx hstream
                                                            (fn [^long size]
                                                              (cuda-malloc size :float))
                                                            cuda-free!)]
         (->CUFactory modl hstream float-accessor native-float
                      (->FloatVectorEngine handle modl hstream) (->FloatGEEngine handle modl hstream)
                      (->FloatTREngine handle modl hstream) (->FloatSYEngine handle modl hstream))))))

  (defn cublas-long [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program integer-src)
                                   ["-DNUMBER=long" #_"-use_fast_math" "-default-device"])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)
                     long-accessor (->LongPointerAccessor ctx hstream
                                                          (fn [^long size]
                                                            (cuda-malloc size :long))
                                                          cuda-free!)]
         (->CUFactory modl hstream long-accessor native-long
                      (->IntegerVectorEngine modl hstream) nil nil nil)))))

  (defn cublas-int [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program integer-src)
                                   ["-DNUMBER=int" "-use_fast_math" "-default-device"])]
       (let-release [modl (module prog)
                     handle (cublas-handle hstream)
                     hstream (get-cublas-stream handle)
                     int-accessor (->IntPointerAccessor ctx hstream
                                                        (fn [^long size]
                                                          (cuda-malloc size :int))
                                                        cuda-free!)]
         (->CUFactory modl hstream int-accessor native-int
                      (->IntegerVectorEngine modl hstream) nil nil nil)))))

  (defn cublas-short [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program integer-src)
                                   ["-DNUMBER=short" #_"-use_fast_math" "-default-device"])]
       (let-release [modl (module prog)
                     short-accessor (->ShortPointerAccessor ctx hstream
                                                            (fn [^long size]
                                                              (cuda-malloc size :short))
                                                            cuda-free!)]
         (->CUFactory modl hstream short-accessor native-short
                      (->IntegerVectorEngine modl hstream) nil nil nil)))))

  (defn cublas-byte [ctx hstream]
    (in-context
     ctx
     (with-release [prog (compile! (program integer-src)
                                   ["-DNUMBER=char" #_"-use_fast_math" "-default-device"])]
       (let-release [modl (module prog)
                     byte-accessor (->BytePointerAccessor ctx hstream
                                                          (fn [^long size]
                                                            (cuda-malloc size :byte))
                                                          cuda-free!)]
         (->CUFactory modl hstream byte-accessor native-byte
                      (->IntegerVectorEngine modl hstream) nil nil nil))))))
