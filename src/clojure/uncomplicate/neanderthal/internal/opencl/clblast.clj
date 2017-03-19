;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.opencl.clblast
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release wrap-int wrap-double wrap-float]]
             [utils :refer [with-check]]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [constants :refer [dec-error]]
             [toolbox :refer [enq-read-int enq-read-double enq-read-float]]]
            [uncomplicate.neanderthal.native :refer [native-float native-double]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.neanderthal.internal.opencl.clblock :refer :all])
  (:import [org.jocl.blast CLBlast CLBlastStatusCode]
           [uncomplicate.neanderthal.internal.api Vector Matrix Block DataAccessor StripeNavigator]
           [uncomplicate.neanderthal.internal.opencl.clblock CLBlockVector CLGEMatrix CLTRMatrix]))

;; =============== OpenCL and CLBlast error handling functions =================

(defn ^:private error [^long err-code details]
  (if (< -10000 err-code -1003)
     (let [err (CLBlastStatusCode/stringFor err-code)]
       (ex-info (format "CLBlast error: %s." err) {:name err :code err-code :type :clblast-error :details details}))
     (let [err (dec-error err-code)]
       (ex-info (format "OpenCL error: %s." err) {:name err :code err-code :type :opencl-error :details details}))))

(defn ^:private equals-vector [ctx queue prog ^CLBlockVector x ^CLBlockVector y]
  (if (< 0 (.dim x))
    (with-release [equals-vector-kernel (kernel prog "equals_vector")
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-vector-kernel eq-flag-buf
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue equals-vector-kernel (work-size-1d (.dim x)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.dim y))))

(defn ^:private equals-ge [ctx queue prog ^CLGEMatrix a ^CLGEMatrix b]
  (if (< 0 (.count a))
    (with-release [equals-matrix-kernel (kernel prog (if (= (.order a) (.order b))
                                                       "equals_ge_no_transp"
                                                       "equals_ge_transp"))
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-matrix-kernel eq-flag-buf
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.ld b)))
        (enq-nd! queue equals-matrix-kernel (work-size-2d (.sd a) (.fd a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private equals-tr [ctx queue prog ^CLTRMatrix a ^CLTRMatrix b]
  (if (< 0 (.count a))
    (let [res (wrap-int 0)
          bottom (if (= (.order a) COLUMN_MAJOR) (= (.uplo a) LOWER) (= (.uplo a) UPPER))
          kernel-name (if (= (.order a) (.order b))
                        (if bottom "equals_tr_bottom" "equals_tr_top")
                        (if bottom "equals_tr_bottom_transp" "equals_tr_top_transp"))]
      (with-release [equals-matrix-kernel (kernel prog kernel-name)
                     eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-matrix-kernel eq-flag-buf (wrap-int (.diag a))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.ld b)))
        (enq-nd! queue equals-matrix-kernel (work-size-2d (.sd a) (.fd a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private ge-set [ctx queue prog alpha ^CLGEMatrix a]
  (if (< 0 (.count a))
    (let [da (data-accessor a)]
      (if (and (= (.ld a) (.sd a)) (= 0 (.offset a))
               (= (* (.mrows a) (.ncols a)) (.count da (.buffer a))))
        (.initialize da (.buffer a) alpha)
        (with-release [ge-set-kernel (kernel prog "ge_set")]
          (set-args! ge-set-kernel (.wrapPrim da alpha)
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld a)))
          (enq-nd! queue ge-set-kernel (work-size-2d (.sd a) (.fd a))))))))

;; =============== Common vector engine  macros and functions ==================

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
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride ~param))))))

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
         (~method
          (.dim ~x) (cl-mem res-buffer#) 0 (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
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

(defmacro ^:private vector-axpby [queue method scal-method alpha x beta y]
  `(when (< 0 (.dim ~x))
     (with-check error
       (~scal-method (.dim ~x) ~beta (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y) ~queue nil)
       nil)
     (with-check error
       (~method (.dim ~x) ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y) ~queue nil)
       nil)))

(defmacro ^:private vector-subcopy [queue method x y kx lx ky]
  `(when (< 0 (long ~lx))
     (with-check error
       (~method ~lx (cl-mem (.buffer ~x)) ~kx (.stride ~x) (cl-mem (.buffer ~y)) ~ky (.stride ~y)
        ~queue nil)
       nil)))

;; =============== Common GE matrix macros and functions =======================

(defmacro ^:private ge-swap [queue method a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# (.sd ~b) ld-a# ld-b#)
           (with-check error
             (~method (.count ~a) buff-a# offset-a# 1 buff-b# offset-b# 1
              ~queue nil)
             nil)
           (dotimes [j# (.fd ~a)]
             (with-check error
               (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1
                ~queue nil)
               nil)))
         (dotimes [j# (.fd ~a)]
           (with-check error
             (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# j#) ld-b#
              ~queue nil)
             nil))))))

(defmacro ^:private ge-copy [queue method a b]
  `(when (< 0 (.count ~a))
     (with-check error
       (~method (.order ~a) (if (= (.order ~a) (.order ~b)) NO_TRANS TRANS) (.mrows ~a) (.ncols ~a)
        1.0 (cl-mem (.buffer ~a)) (.offset ~a) (.ld ~a) (cl-mem (.buffer ~b)) (.offset ~b) (.ld ~b)
        ~queue nil)
       nil)))

(defmacro ^:private ge-scal-set [queue method alpha a]
  `(when (< 0 (.count ~a))
     (let [ld# (.stride ~a)
           sd# (.sd ~a)
           offset# (.offset ~a)
           buff# (cl-mem (.buffer ~a))]
       (if (= sd# ld#)
         (with-check error
           (~method (.count ~a) ~alpha buff# offset# 1 ~queue nil)
           nil)
         (dotimes [j# (.fd ~a)]
           (with-check error
             (~method sd# ~alpha buff# (+ offset# (* ld# j#)) 1 ~queue nil)
             nil))))))

(defmacro ^:private ge-axpy [queue method alpha a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (with-check error
             (~method (.count ~a) ~alpha buff-a# offset-a# 1 buff-b# offset-b# 1
              ~queue nil)
             nil)
           (dotimes [j# fd-a#]
             (with-check error
               (~method sd-a# ~alpha buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1
                ~queue nil)
               nil)))
         (dotimes [j# fd-b#]
           (with-check error
             (~method sd-b# ~alpha buff-a# (+ offset-a# j#) ld-a# buff-b# (+ offset-b# (* ld-b# j#)) 1
              ~queue nil)
             nil))))))

(defmacro ^:private ge-axpby [queue method scal-method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (do
             (with-check error
               (~scal-method (.count ~a) ~beta buff-b# offset-b# 1 ~queue nil)
               nil)
             (with-check error
               (~method (.count ~a) ~alpha buff-a# offset-a# 1 buff-b# offset-b# 1
                ~queue nil)
               nil))
           (dotimes [j# fd-a#]
             (with-check error
               (~scal-method sd-a# ~beta buff-b# (+ offset-b# (* ld-b# j#)) 1 ~queue nil)
               nil)
             (with-check error
               (~method sd-a# ~alpha buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1
                ~queue nil)
               nil)))
         (dotimes [j# fd-b#]
           (with-check error
             (~scal-method sd-b# ~beta buff-b# (+ offset-b# (* ld-b# j#)) 1 ~queue nil)
             nil)
           (with-check error
             (~method sd-b# ~alpha buff-a# (+ offset-a# j#) ld-a# buff-b# (+ offset-b# (* ld-b# j#)) 1
              ~queue nil)
             nil))))))

(defmacro ^:private ge-mv
  ([queue method alpha a x beta y]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~a) NO_TRANS (.mrows ~a) (.ncols ~a)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         ~beta (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~queue nil)
        nil)))
  ([]
   `(throw (IllegalArgumentException. "In-place mv! is not supported for GE matrices."))))


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
  ([alpha a b left]
   `(if ~left
      (mm (engine ~b) ~alpha ~b ~a false)
      (throw (IllegalArgumentException. "In-place mm! is not supported for GE matrices."))))
  ([queue method alpha a b beta c]
   `(when (< 0 (.count ~a))
      (with-check error
        (~method (.order ~c)
         (if (= (.order ~a) (.order ~c)) NO_TRANS TRANS)
         (if (= (.order ~b) (.order ~c)) NO_TRANS TRANS)
         (.mrows ~a) (.ncols ~b) (.ncols ~a)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
         ~beta (cl-mem (.buffer ~c)) (.offset ~c) (.stride ~c)
         ~queue nil)
        nil))))

;; =============== Common TR matrix macros and functions ==========================

(defmacro ^:private tr-swap-copy [queue stripe-nav method a b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                buff-b# (+ offset-b# j# (* ld-b# start#)) n# ~queue nil)
               nil)))))))

(defmacro ^:private tr-scal-set [queue stripe-nav method alpha a]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld# (.stride ~a)
           offset# (.offset ~a)
           buff# (cl-mem (.buffer ~a))]
       (dotimes [j# n#]
         (let [start# (.start ~stripe-nav n# j#)
               n-j# (- (.end ~stripe-nav n# j#) start#)]
           (with-check error
             (~method n-j# ~alpha buff# (+ offset# (* ld# j#) start#) 1 ~queue nil)
             nil))))))

(defmacro ^:private tr-axpy [queue stripe-nav method alpha a b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~method n-j# ~alpha buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~method n-j# ~alpha buff-a# (+ offset-a# j# (* ld-a# start#)) n#
                buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)))))))

(defmacro ^:private tr-axpby [queue stripe-nav method scal-method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (cl-mem (.buffer ~a))
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (cl-mem (.buffer ~b))]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~scal-method n-j# ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)
             (with-check error
               (~method n-j# ~alpha buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (with-check error
               (~scal-method n-j# ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)
             (with-check error
               (~method n-j# ~alpha buff-a# (+ offset-a# j# (* ld-a# start#)) n#
                buff-b# (+ offset-b# (* ld-b# j#) start#) 1 ~queue nil)
               nil)))))))

(defmacro ^:private tr-mv
  ([queue method a x]
   `(with-check error
      (~method (.order ~a) (.uplo ~a) NO_TRANS (.diag ~a) (.ncols ~a)
       (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
       (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
       ~queue nil)
      nil))
  ([]
   `(throw (IllegalArgumentException. "Only in-place mv! is supported for TR matrices."))))

(defmacro ^:private tr-mm
  ([queue method alpha a b left]
   `(with-check error
      (~method (.order ~b) (if ~left LEFT RIGHT) (.uplo ~a)
       (if (= (.order ~a) (.order ~b)) NO_TRANS TRANS) (.diag ~a) (.mrows ~b) (.ncols ~b)
       ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
       (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
       ~queue nil)
      nil))
  ([]
   `(throw (IllegalArgumentException. "Only in-place mm! is supported for TR matrices."))))

;; =============== CLBlast based engines =======================================

(deftype DoubleVectorEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ x y]
    (equals-vector ctx queue prog ^CLBlockVector x ^CLBlockVector y))
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
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDnrm2 ^CLBlockVector x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDasum ^CLBlockVector x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDamax ^CLBlockVector x))
  (rot [_ _ _ _ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotm [_ _ _ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotmg [_ _ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastDscal alpha ^CLBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastDaxpy alpha ^CLBlockVector x ^CLBlockVector y)
    y)
  BlasPlus
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
    #_(vector-scal-set queue CLBlast/CLBlastDset alpha ^CLBlockVector x);;TODO CLBlast
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby queue CLBlast/CLBlastDaxpy CLBlast/CLBlastDscal
                  alpha ^CLBlockVector x beta ^CLBlockVector y)
    y))

(deftype FloatVectorEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ x y]
    (equals-vector ctx queue prog ^CLBlockVector x ^CLBlockVector y))
  Blas
  (swap [_ x y]
    (vector-method queue CLBlast/CLBlastSswap ^CLBlockVector x ^CLBlockVector y)
    x)
  (copy [_ x y]
    (vector-method queue CLBlast/CLBlastScopy ^CLBlockVector x ^CLBlockVector y)
    y)
  (dot [_ x y]
    (vector-dot ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSdot ^CLBlockVector x ^CLBlockVector y))
  (nrm2 [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSnrm2 ^CLBlockVector x))
  (asum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSasum ^CLBlockVector x))
  (iamax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSamax ^CLBlockVector x))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException. "TODO.")))
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastSscal alpha ^CLBlockVector x)
    x)
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastSaxpy alpha ^CLBlockVector x ^CLBlockVector y)
    y)
  BlasPlus
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
    #_(vector-scal-set queue CLBlast/CLBlastSset alpha ^CLBlockVector x);;TODO CLBlast
    x)
  (axpby [_ alpha x beta y]
    (vector-axpby queue CLBlast/CLBlastSaxpy CLBlast/CLBlastSscal
                  alpha ^CLBlockVector x beta ^CLBlockVector y);;TODO CLBlast
    y))

(deftype DoubleGEEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-ge ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue CLBlast/CLBlastDswap ^CLGEMatrix a ^CLGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy queue CLBlast/CLBlastDomatcopy ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal-set queue CLBlast/CLBlastDscal alpha ^CLGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy queue CLBlast/CLBlastDaxpy alpha ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv queue CLBlast/CLBlastDgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y)
   y)
  (mv [this a x]
   (ge-mv))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastDger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a)
    a)
  (mm [_ alpha a b left]
   (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
   (ge-mm queue CLBlast/CLBlastDgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c)
   c)
  BlasPlus
  (set-all [_ alpha a]
    (ge-set ctx queue prog alpha a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby queue CLBlast/CLBlastDaxpy CLBlast/CLBlastDscal alpha ^CLGEMatrix a beta ^CLGEMatrix b)
    b))

(deftype FloatGEEngine [ctx queue prog]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (equals-ge ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue CLBlast/CLBlastSswap ^CLGEMatrix a ^CLGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy queue CLBlast/CLBlastSomatcopy ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal-set queue CLBlast/CLBlastSscal alpha ^CLGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy queue CLBlast/CLBlastSaxpy alpha ^CLGEMatrix a ^CLGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv queue CLBlast/CLBlastSgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y)
   y)
  (mv [this a x]
   (ge-mv))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastSger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a)
    a)
  (mm [_ alpha a b left]
   (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
   (ge-mm queue CLBlast/CLBlastSgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c)
   c)
  BlasPlus
  (set-all [_ alpha a]
    (ge-set ctx queue prog alpha a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby queue CLBlast/CLBlastSaxpy CLBlast/CLBlastSscal alpha ^CLGEMatrix a beta ^CLGEMatrix b)
    b))

(deftype DoubleTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (equals-tr ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-swap-copy queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                  CLBlast/CLBlastDswap ^CLTRMatrix a ^CLTRMatrix b)
    a)
  (copy [_ a b]
    (tr-swap-copy queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                  CLBlast/CLBlastDcopy ^CLTRMatrix a ^CLTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal-set queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                 CLBlast/CLBlastDscal alpha ^CLTRMatrix a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
             CLBlast/CLBlastDaxpy alpha ^CLTRMatrix a ^CLTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv))
  (mv [_ a x]
   (tr-mv queue CLBlast/CLBlastDtrmv ^CLTRMatrix a ^CLBlockVector x)
   x)
  (mm [this alpha a b beta c]
   (tr-mm))
  (mm [_ alpha a b left]
   (tr-mm queue CLBlast/CLBlastDtrmm alpha ^CLTRMatrix a ^CLGEMatrix b left)
   b)
  BlasPlus
  (set-all [_ alpha a]
    #_(tr-scal-set queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                   CLBlast/CLBlastDset alpha ^CLTRMatrix a);;TODO CLBlast
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
              CLBlast/CLBlastDaxpy CLBlast/CLBlastDscal
              alpha ^CLTRMatrix a beta ^CLTRMatrix b);;TODO CLBlast
    b))

(deftype FloatTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (equals-tr ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-swap-copy queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                  CLBlast/CLBlastSswap ^CLTRMatrix a ^CLTRMatrix b)
    a)
  (copy [_ a b]
    (tr-swap-copy queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                  CLBlast/CLBlastScopy ^CLTRMatrix a ^CLTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal-set queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                 CLBlast/CLBlastSscal alpha ^CLTRMatrix a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy queue ^StripeNavigator  (.stripe-nav ^CLTRMatrix a)
             CLBlast/CLBlastSaxpy alpha ^CLTRMatrix a ^CLTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv))
  (mv [_ a x]
   (tr-mv queue CLBlast/CLBlastStrmv ^CLTRMatrix a ^CLBlockVector x)
   x)
  (mm [this alpha a b beta c]
   (tr-mm))
  (mm [_ alpha a b left]
   (tr-mm queue CLBlast/CLBlastStrmm alpha ^CLTRMatrix a ^CLGEMatrix b left)
   b)
  BlasPlus
  (set-all [_ alpha a]
    #_(tr-scal-set queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
                   CLBlast/CLBlastSset alpha ^CLTRMatrix a);;TODO CLBlast
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby queue ^StripeNavigator (.stripe-nav ^CLTRMatrix a)
              CLBlast/CLBlastSaxpy CLBlast/CLBlastSscal
              alpha ^CLTRMatrix a beta ^CLTRMatrix b);;TODO CLBlast
    b))

(deftype CLFactory [ctx queue prog ^DataAccessor da vector-eng ge-eng tr-eng]
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

(let [src [(slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/blas-plus.cl"))]]

  (org.jocl.blast.CLBlast/setExceptionsEnabled false)

  (defn clblast-double [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=double" nil)]
      (->CLFactory ctx queue prog
                   (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES
                                      double-array wrap-double native-double)
                   (->DoubleVectorEngine ctx queue prog) (->DoubleGEEngine ctx queue prog)
                   (->DoubleTREngine ctx queue prog))))

  (defn clblast-float [ctx queue]
    (let [prog (build-program! (program-with-source ctx src) "-DREAL=float" nil)]
      (->CLFactory ctx queue prog
                   (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES
                                      float-array wrap-float native-float)
                   (->FloatVectorEngine ctx queue prog) (->FloatGEEngine ctx queue prog)
                   (->FloatTREngine ctx queue prog)))))
