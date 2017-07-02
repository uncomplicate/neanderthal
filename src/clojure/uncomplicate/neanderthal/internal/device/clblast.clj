;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.clblast
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release wrap-int wrap-double wrap-float]]
             [utils :refer [with-check]]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [constants :refer [dec-error]]
             [toolbox :refer [enq-read-int enq-read-double enq-read-float]]]
            [uncomplicate.neanderthal
             [core :refer [transfer!]]
             [native :refer [native-float native-double]]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.neanderthal.internal.device
             [common :refer :all]
             [clblock :refer :all]
             [cublock :refer :all]])
  (:import [org.jocl.blast CLBlast CLBlastStatusCode CLBlastTranspose CLBlastSide]
           [uncomplicate.neanderthal.internal.api Vector Matrix Block DataAccessor]
           [uncomplicate.neanderthal.internal.device.clblock CLBlockVector CLGEMatrix CLTRMatrix]
           [uncomplicate.neanderthal.internal.device.cublock CUBlockVector CUGEMatrix CUTRMatrix]))

;; =============== Transfer preferences ========================================

(prefer-method transfer! [CLBlockVector Object] [Object CUBlockVector])
(prefer-method transfer! [CLGEMatrix Object] [Object CUGEMatrix])
(prefer-method transfer! [CLTRMatrix Object] [Object CUTRMatrix])

;; =============== OpenCL and CLBlast error handling functions =================

(defn ^:private error [^long err-code details]
  (if (< -10000 err-code -1003)
    (let [err (CLBlastStatusCode/stringFor err-code)]
      (ex-info (format "CLBlast error: %s." err) {:name err :code err-code :type :clblast-error :details details}))
    (let [err (dec-error err-code)]
      (ex-info (format "OpenCL error: %s." err) {:name err :code err-code :type :opencl-error :details details}))))

(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in OpenCL. Please use a host instance.")))

;; =============== Common vector macros and functions =======================

(defn ^:private vector-equals [ctx queue prog ^CLBlockVector x ^CLBlockVector y]
  (if (< 0 (.dim x))
    (with-release [vector-equals-kernel (kernel prog "vector_equals")
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! vector-equals-kernel
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y))
                   eq-flag-buf)
        (enq-nd! queue vector-equals-kernel (work-size-1d (.dim x)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.dim y))))

(defn ^:private vector-set [ctx queue prog alpha ^CLBlockVector x]
  (if (< 0 (.dim x))
    (let [da (data-accessor x)]
      (if (= (.dim x) (.count da (.buffer x)))
        (.initialize da (.buffer x) alpha)
        (with-release [vector-set-kernel (kernel prog "vector_set")]
          (set-args! vector-set-kernel (.wrapPrim da alpha)
                     (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
          (enq-nd! queue vector-set-kernel (work-size-1d (.dim x))))))))

;; NOTE: rotXX methods are not supported by CLBlast yet
;; These signatures might change a bit once they are supported
(defmacro ^:private vector-rot
  ([queue method x y c s]
   `(with-check error
      (when (and (< 0 (.dim ~x)) (< 0 (.dim ~y)))
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~c ~s
         ~queue nil))
      nil)))

(defmacro ^:private vector-rotg [queue method x]
  `(let [mem# (cl-mem (.buffer ~x))
         ofst# (.offset ~x)
         strd# (.stride ~x)]
     (with-check error
       (~method mem# ofst mem# (+ ofst# strd#) mem# (+ ofst# (* 2 strd#)) mem# (+ ofst# (* 3 strd#))
        ~queue nil)
       nil)))

(defmacro ^:private vector-rotm [queue method x y param]
  `(when (and (< 0 (.dim ~x)) (< 0 (.dim ~y)) (< 4 (.dim param)))
     (with-check error
       (~method (.dim ~x)
        (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-men (.buffer ~y)) (.offset ~y) (.stride ~y)
        (cl-mem (.buffer ~param) (.offset ~param) (.stride ~param))
        ~queue nil)
       nil)))

(defmacro ^:private vector-rotmg [queue method ^CLBlockVector d1d2xy ^CLBlockVector param]
  `(if (= 1 (.stride ~param))
     (let [mem# (cl-mem (.buffer ~d1d2xy))
           ofst# (.offset ~d1d2xy)
           strd# (.stride ~d1d2xy)]
       (with-check error
         (~method mem# ofst mem# (+ ofst# strd#) mem# (+ ofst# (* 2 strd#)) mem# (+ ofst# (* 3 strd#))
          (cl-mem (.buffer ~param)) (.offset ~param) ~queue nil)
         nil))
     (throw (ex-info "You cannot use strided vector as param." {:param (str ~param)}))))

(defmacro ^:private vector-method
  ([queue method x]
   `(when (< 0 (.dim ~x))
      (with-check error
        (~method (.dim ~x) (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
        nil)))
  ([queue method x y]
   `(when (< 0 (.dim ~x))
      (with-check error
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~queue nil)
        nil)))
  ([queue method x y z]
   `(when (< 0 (.dim x))
      (with-check error
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         (cl-mem (.buffer ~z)) (.offset ~z) (.stride ~z)
         ~queue nil)
        nil))))

(defmacro ^:private vector-dot [ctx queue res-bytes read-method method x y]
  `(if (< 0 (.dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (with-check error
         (~method (.dim ~x) (cl-mem res-buffer#) 0
          (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
          (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
          ~queue nil)
         (~read-method ~queue res-buffer#)))
     0.0))

(defmacro ^:private vector-sum-nrm2 [ctx queue res-bytes read-method method x]
  `(if (< 0 (.dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
       (with-check error
         (~method (.dim ~x) (cl-mem res-buffer#) 0 (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
         (~read-method ~queue res-buffer#)))
     0.0))

(defmacro ^:private vector-ipeak [ctx queue method x]
  `(if (< 0 (.dim ~x))
     (with-release [res-buffer# (cl-buffer ~ctx Integer/BYTES :read-write)]
       (with-check error
         (~method (.dim ~x) (cl-mem res-buffer#) 0 (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
         (enq-read-int ~queue res-buffer#)))
     0))

(defmacro ^:private vector-scal-set [queue method alpha x]
  `(when (< 0 (.dim ~x))
     (with-check error
       (~method (.dim ~x) ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
       nil)))

(defmacro ^:private vector-axpy [queue method alpha x y]
  `(when (< 0 (.dim ~x))
     (with-check error
       (~method (.dim ~x) ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y) ~queue nil)
       nil)))

(defn ^:private vector-axpby [queue prog alpha ^CLBlockVector x beta ^CLBlockVector y]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [vector-axpby-kernel (kernel prog "vector_axpby")]
        (set-args! vector-axpby-kernel 0
                   (.wrapPrim da alpha) (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.wrapPrim da beta) (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue vector-axpby-kernel (work-size-1d (.dim x)))))))

(defmacro ^:private vector-subcopy [queue method x y kx lx ky]
  `(when (< 0 (long ~lx))
     (with-check error
       (~method ~lx (cl-mem (.buffer ~x)) ~kx (.stride ~x) (cl-mem (.buffer ~y)) ~ky (.stride ~y)
        ~queue nil)
       nil)))

;; =============== Common GE matrix macros and functions =======================

(defn ^:private ge-equals [ctx queue prog ^CLGEMatrix a ^CLGEMatrix b]
  (if (< 0 (.count a))
    (with-release [ge-equals-kernel (kernel prog (name-transp "ge_equals" a b))
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! ge-equals-kernel 0
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                   eq-flag-buf)
        (enq-nd! queue ge-equals-kernel (work-size-2d (.sd a) (.fd a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private ge-set [queue prog alpha ^CLGEMatrix a]
  (when (< 0 (.count a))
    (let [da (data-accessor a)]
      (if (and (= (.stride a) (.sd a)) (= 0 (.offset a))
               (= (* (.mrows a) (.ncols a)) (.count da (.buffer a))))
        (.initialize da (.buffer a) alpha)
        (with-release [ge-set-kernel (kernel prog "ge_set")]
          (set-args! ge-set-kernel 0 (.wrapPrim da alpha)
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a)))
          (enq-nd! queue ge-set-kernel (work-size-2d (.sd a) (.fd a))))))))

(defmacro ^:private ge-swap [queue prog method a b]
  `(when (< 0 (.count ~a))
     (if (fits-buffer? ~a ~b)
       (with-check error
         (~method (.count ~a)
          (cl-mem (.buffer ~a)) (.offset ~a) 1 (cl-mem (.buffer ~b)) (.offset ~b) 1
          ~queue nil)
         nil)
       (with-release [ge-swap-kernel# (kernel ~prog (name-transp "ge_swap" ~a ~b))]
         (set-args! ge-swap-kernel# (.buffer ~a) (wrap-int (.offset ~a)) (wrap-int (.stride ~a))
                    (.buffer ~b) (wrap-int (.offset ~b)) (wrap-int (.stride ~b)))
         (enq-nd! ~queue ge-swap-kernel# (work-size-2d (.sd ~a) (.fd ~a)))))))

(defmacro ^:private ge-sum-nrm2 [ctx queue prog res-bytes read-method method op-name a]
  `(if (< 0 (.count ~a))
     (if (fully-packed? ~a)
       (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
         (with-check error
           (~method (.count ~a) (cl-mem res-buffer#) 0 (cl-mem (.buffer ~a)) (.offset ~a) 1 ~queue nil)
           (~read-method ~queue res-buffer#)))
       (not-available))
     0.0))

(defmacro ^:private ge-dot [ctx queue res-bytes read-method method a b]
  `(if (< 0 (.count ~a))
     (if (and (fully-packed? ~a) (fully-packed? ~b) (= (.order ~a) (.order ~b)))
       (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
         (with-check error
           (~method (.count ~a) (cl-mem res-buffer#) 0
            (cl-mem (.buffer ~a)) (.offset ~a) 1
            (cl-mem (.buffer ~b)) (.offset ~b) 1
            ~queue nil)
           (~read-method ~queue res-buffer#)))
       (not-available))
     0.0))

(defmacro ^:private ge-omatcopy
  ([queue method alpha a b]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~a)
         (if (= (.order ~a) (.order ~b)) CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
         (.mrows ~a) (.ncols ~a) ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
         ~queue nil)
        nil)))
  ([queue method alpha a]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~a) CLBlastTranspose/CLBlastTransposeNo
         (.mrows ~a) (.ncols ~a) ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         ~queue nil)
        nil)))
  ([queue method a]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method COLUMN_MAJOR CLBlastTranspose/CLBlastTransposeYes
         (.sd ~a) (.fd ~a) 1.0 (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~a)) (.offset ~a) (.fd ~a)
         ~queue nil)
        nil))))

(defn ^:private ge-axpby [queue prog alpha ^CLGEMatrix a beta ^CLGEMatrix b]
  (when (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [ge-axpby-kernel (kernel prog (name-transp "ge_axpby" a b))]
        (set-args! ge-axpby-kernel 0
                   (.wrapPrim da alpha) (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.wrapPrim da beta) (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue ge-axpby-kernel (work-size-2d (.sd a) (.fd a)))))))

(defmacro ^:private ge-axpy [queue prog method alpha a b]
  `(when (< 0 (.count ~a))
     (if (fits-buffer? ~a ~b)
       (with-check error
         (~method (.count ~a) ~alpha
          (cl-mem (.buffer ~a)) (.offset ~a) 1 (cl-mem (.buffer ~b)) (.offset ~b) 1
          ~queue nil)
         nil)
       (ge-axpby ~queue ~prog ~alpha ~a 1.0 ~b))))

(defmacro ^:private ge-mv
  ([queue method alpha a x beta y]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~a) CLBlastTranspose/CLBlastTransposeNo (.mrows ~a) (.ncols ~a)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         ~beta (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~queue nil)
        nil)))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (str ~a)}))))

(defmacro ^:private ge-rk [queue method alpha x y a]
  `(when (< 0 (.count ~a))
     (with-check error
       (~method (.order ~a) (.mrows ~a) (.ncols ~a)
        ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
        (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
        ~queue nil)
       nil)))

(defmacro ^:private ge-mm
  ([alpha a b]
   `(mm (engine ~b) ~alpha ~b ~a false))
  ([queue method alpha a b beta c]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~c)
         (if (= (.order ~a) (.order ~c)) CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
         (if (= (.order ~b) (.order ~c)) CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
         (.mrows ~a) (.ncols ~b) (.ncols ~a)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
         ~beta (cl-mem (.buffer ~c)) (.offset ~c) (.stride ~c)
         ~queue nil)
        nil))))

;; =============== Common TR matrix macros and functions ==========================

(defn ^:private tr-equals [ctx queue prog ^CLTRMatrix a ^CLTRMatrix b]
  (if (< 0 (.count a))
    (let [res (wrap-int 0)]
      (with-release [tr-equals-kernel (kernel prog (name-transp "tr_equals" a b))
                     eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! tr-equals-kernel 0 (wrap-int (.diag a)) (wrap-int (if (tr-bottom a) 1 -1))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                   eq-flag-buf)
        (enq-nd! queue tr-equals-kernel (work-size-2d (.sd a) (.fd a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private tr-map [queue prog op-name ^CLTRMatrix a ^CLTRMatrix b]
  (if (< 0 (.count a))
    (with-release [tr-map-kernel (kernel prog (name-transp op-name a b))]
      (set-args! tr-map-kernel 0 (wrap-int (.diag a)) (wrap-int (if (tr-bottom a) 1 -1))
                 (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                 (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
      (enq-nd! queue tr-map-kernel (work-size-2d (.sd a) (.fd a))))))

(defn ^:private tr-axpby [queue prog alpha ^CLTRMatrix a beta ^CLTRMatrix b]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (with-release [tr-axpby-kernel (kernel prog (name-transp "tr_axpby" a b))]
        (set-args! tr-axpby-kernel 0 (wrap-int (.diag a)) (wrap-int (if (tr-bottom a) 1 -1))
                   (.wrapPrim da alpha) (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.wrapPrim da beta) (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue tr-axpby-kernel (work-size-2d (.sd a) (.fd a)))))))

(defn ^:private tr-set-scal [queue prog op-name alpha ^CLTRMatrix a]
  (if (< 0 (.count a))
    (with-release [tr-op-kernel (kernel prog op-name)]
      (set-args! tr-op-kernel 0 (wrap-int (.diag a)) (wrap-int (if (tr-bottom a) 1 -1))
                 (.wrapPrim (data-accessor a) alpha)
                 (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a)))
      (enq-nd! queue tr-op-kernel (work-size-2d (.sd a) (.fd a))))))

(defmacro ^:private tr-mv
  ([queue method a x]
   `(with-check error
      (~method (.order ~a) (.uplo ~a) CLBlastTranspose/CLBlastTransposeNo (.diag ~a) (.ncols ~a)
       (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
       (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
       ~queue nil)
      nil))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

(defmacro ^:private tr-mm
  ([queue method alpha a b left]
   `(with-check error
      (~method (.order ~b)
       (if ~left CLBlastSide/CLBlastSideLeft CLBlastSide/CLBlastSideRight)
       (.uplo ~a)
       (if (= (.order ~a) (.order ~b)) CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
       (.diag ~a) (.mrows ~b) (.ncols ~b)
       ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
       (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
       ~queue nil)
      nil))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

;; =============== CLBlast based engines =======================================

(deftype DoubleVectorEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals ctx queue prog x y))
  Blas
  (swap [_ x y]
    (vector-method queue CLBlast/CLBlastDswap ^CLBlockVector x ^CLBlockVector y)
    x)
  (copy [_ x y]
    (vector-method queue CLBlast/CLBlastDcopy ^CLBlockVector x ^CLBlockVector y)
    y)
  (dot [_ x y]
    (vector-dot ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDdot
                ^CLBlockVector x ^CLBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDnrm2 ^CLBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDasum ^CLBlockVector x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDamax ^CLBlockVector x))
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
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastDscal alpha ^CLBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastDaxpy alpha ^CLBlockVector x ^CLBlockVector y)
    y)
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastDcopy ^CLBlockVector x ^CLBlockVector y kx lx ky)
    y)
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDsum ^CLBlockVector x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmax ^CLBlockVector x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmin ^CLBlockVector x))
  (set-all [_ alpha x]
    (vector-set ctx queue prog alpha x)
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby queue prog alpha x beta y)
    y))

(deftype FloatVectorEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals ctx queue prog x y))
  Blas
  (swap [_ x y]
    (vector-method queue CLBlast/CLBlastSswap ^CLBlockVector x ^CLBlockVector y)
    x)
  (copy [_ x y]
    (vector-method queue CLBlast/CLBlastScopy ^CLBlockVector x ^CLBlockVector y)
    y)
  (dot [_ x y]
    (vector-dot ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSdot ^CLBlockVector x ^CLBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSnrm2 ^CLBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSasum ^CLBlockVector x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSamax ^CLBlockVector x))
  (iamin [_ x]
    (not-available))
  (rot [_ _ y c s]
    (not-available))
  (rotg [_ _]
    (not-available))
  (rotm [_ _ y p]
    (not-available))
  (rotmg [_ _ args]
    (not-available))
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastSscal alpha ^CLBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastSaxpy alpha ^CLBlockVector x ^CLBlockVector y)
    y)
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastScopy ^CLBlockVector x ^CLBlockVector y kx lx ky)
    y)
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSsum ^CLBlockVector x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmax ^CLBlockVector x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmin ^CLBlockVector x))
  (set-all [_ alpha x]
    (vector-set ctx queue prog alpha x)
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby queue prog alpha x beta y)
    y))

(deftype DoubleGEEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue prog CLBlast/CLBlastDswap ^CLGEMatrix a ^CLGEMatrix b)
    a)
  (copy [_ a b]
    (ge-omatcopy queue CLBlast/CLBlastDomatcopy 1.0 ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-omatcopy queue CLBlast/CLBlastDomatcopy alpha ^CLGEMatrix a))
  (dot [_ a b]
    (ge-dot ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDdot ^CLGEMatrix a ^CLGEMatrix b))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [this a]
    (ge-sum-nrm2 ctx queue prog Double/BYTES enq-read-double CLBlast/CLBlastDnrm2 "ge_nrm2" ^CLGEMatrix a))
  (nrmi [_ _]
    (not-available))
  (asum [this a]
    (ge-sum-nrm2 ctx queue prog Double/BYTES enq-read-double CLBlast/CLBlastDasum "ge_asum" ^CLGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpy queue prog CLBlast/CLBlastDaxpy alpha ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (ge-mv queue CLBlast/CLBlastDgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y)
    y)
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastDger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a)
    a)
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm queue CLBlast/CLBlastDgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c)
    c)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [this a]
    (ge-sum-nrm2 ctx queue prog Double/BYTES enq-read-double CLBlast/CLBlastDsum "ge_sum" ^CLGEMatrix a))
  (set-all [_ alpha a]
    (ge-set queue prog alpha a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby queue prog alpha a beta b)
    b)
  (trans [_ a]
    (ge-omatcopy queue CLBlast/CLBlastDomatcopy ^CLGEMatrix a)
    a))

(deftype FloatGEEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue prog CLBlast/CLBlastSswap ^CLGEMatrix a ^CLGEMatrix b)
    a)
  (copy [_ a b]
    (ge-omatcopy queue CLBlast/CLBlastSomatcopy 1.0 ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-omatcopy queue  CLBlast/CLBlastSomatcopy alpha ^CLGEMatrix a))
  (dot [_ a b]
    (ge-dot ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSdot ^CLGEMatrix a ^CLGEMatrix b))
  (nrm1 [_ _]
    (not-available))
  (nrm2 [this a]
    (ge-sum-nrm2 ctx queue prog Float/BYTES enq-read-float CLBlast/CLBlastSnrm2 "ge_nrm2" ^CLGEMatrix a))
  (nrmi [_ _]
    (not-available))
  (asum [this a]
    (ge-sum-nrm2 ctx queue prog Float/BYTES enq-read-float CLBlast/CLBlastSasum "ge_asum" ^CLGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpy queue prog CLBlast/CLBlastSaxpy alpha ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (ge-mv queue CLBlast/CLBlastSgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y)
    y)
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastSger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a)
    a)
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm queue CLBlast/CLBlastSgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c)
    c)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [this a]
    (ge-sum-nrm2 ctx queue prog Float/BYTES enq-read-float CLBlast/CLBlastSsum "ge_sum" ^CLGEMatrix a))
  (set-all [_ alpha a]
    (ge-set queue prog alpha a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby queue prog alpha a beta b)
    b)
  (trans [_ a]
    (ge-omatcopy queue CLBlast/CLBlastSomatcopy ^CLGEMatrix a)
    a))

(deftype DoubleTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-map queue prog "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map queue prog "tr_copy" a b)
    b)
  (scal [_ alpha a]
    (tr-set-scal queue prog "tr_scal" alpha a)
    a)
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
  (axpy [_ alpha a b]
    (tr-axpby queue prog alpha a 1.0 b)
    b)
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv queue CLBlast/CLBlastDtrmv ^CLTRMatrix a ^CLBlockVector x)
    x)
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm queue CLBlast/CLBlastDtrmm alpha ^CLTRMatrix a ^CLGEMatrix b left)
    b)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal queue prog "tr_set" alpha a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby queue prog alpha a beta b)
    b))

(deftype FloatTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-map queue prog "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map queue prog "tr_copy" a b)
    b)
  (scal [_ alpha a]
    (tr-set-scal queue prog "tr_scal" alpha a)
    a)
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
  (axpy [_ alpha a b]
    (tr-axpby queue prog alpha a 1.0 b)
    b)
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv queue CLBlast/CLBlastStrmv ^CLTRMatrix a ^CLBlockVector x)
    x)
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm queue CLBlast/CLBlastStrmm alpha ^CLTRMatrix a ^CLGEMatrix b left)
    b)
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal queue prog "tr_set" alpha a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby queue prog alpha a beta b)
    b))

(deftype CLFactory [ctx queue prog ^DataAccessor da native-fact vector-eng ge-eng tr-eng]
  Releaseable
  (release [_]
    (try
      (release prog)
      (release vector-eng)
      (release ge-eng)
      true
      (finally (CLBlast/CLBlastClearCache))))
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
    (let-release [res (cl-block-vector this n)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-ge [this m n ord init]
    (let-release [res (cl-ge-matrix this m n ord)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n ord uplo diag init]
    (let-release [res (cl-tr-matrix this n ord uplo diag)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng))

(let [src [(slurp (io/resource "uncomplicate/neanderthal/internal/device/blas-plus.cl"))]]

  (org.jocl.blast.CLBlast/setExceptionsEnabled false)

  (defn clblast-double [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=double" nil)]
      (->CLFactory ctx queue prog
                   (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES double-array wrap-double)
                   native-double
                   (->DoubleVectorEngine ctx queue prog) (->DoubleGEEngine ctx queue prog)
                   (->DoubleTREngine ctx queue prog))))

  (defn clblast-float [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=float" nil)]
      (->CLFactory ctx queue prog
                   (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES float-array wrap-float)
                   native-float
                   (->FloatVectorEngine ctx queue prog) (->FloatGEEngine ctx queue prog)
                   (->FloatTREngine ctx queue prog)))))
