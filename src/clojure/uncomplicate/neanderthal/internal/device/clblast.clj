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
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [dragan-says-ex check-eq-navigators]]]
            [uncomplicate.neanderthal.internal.device
             [common :refer [name-transp tr-bottom]]
             [clblock :refer :all]])
  (:import uncomplicate.neanderthal.internal.host.CBLAS
           [org.jocl.blast CLBlast CLBlastStatusCode CLBlastTranspose CLBlastSide]
           [uncomplicate.neanderthal.internal.api Vector Matrix Block DataAccessor Region
            DenseStorage FullStorage LayoutNavigator]
           [uncomplicate.neanderthal.internal.device.clblock CLBlockVector CLGEMatrix CLUploMatrix]))

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
          (enq-nd! queue vector-set-kernel (work-size-1d (.dim x)))))))
  x)

;; NOTE: rotXX methods are not supported by CLBlast yet
;; These signatures might change a bit once they are supported
(defmacro ^:private vector-rot
  ([queue method x y c s]
   `(with-check error
      (when (and (< 0 (.dim ~x)) (< 0 (.dim ~y)))
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stri)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~c ~s
         ~queue nil))
      ~x)))

(defmacro ^:private vector-rotg [queue method x]
  `(let [mem# (cl-mem (.buffer ~x))
         ofst# (.offset ~x)
         strd# (.stride ~x)]
     (with-check error
       (~method mem# ofst mem# (+ ofst# strd#) mem# (+ ofst# (* 2 strd#)) mem# (+ ofst# (* 3 strd#))
        ~queue nil)
       ~x)))

(defmacro ^:private vector-rotm [queue method x y param]
  `(if (and (< 0 (.dim ~x)) (< 0 (.dim ~y)) (< 4 (.dim param)))
     (with-check error
       (~method (.dim ~x)
        (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-men (.buffer ~y)) (.offset ~y) (.stride ~y)
        (cl-mem (.buffer ~param) (.offset ~param) (.stride ~param))
        ~queue nil)
       ~param)
     ~param))

(defmacro ^:private vector-rotmg [queue method ^CLBlockVector d1d2xy ^CLBlockVector param]
  `(if (= 1 (.stride ~param))
     (let [mem# (cl-mem (.buffer ~d1d2xy))
           ofst# (.offset ~d1d2xy)
           strd# (.stride ~d1d2xy)]
       (with-check error
         (~method mem# ofst mem# (+ ofst# strd#) mem# (+ ofst# (* 2 strd#)) mem# (+ ofst# (* 3 strd#))
          (cl-mem (.buffer ~param)) (.offset ~param) ~queue nil)
         ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (str ~param)}))))

(defmacro ^:private vector-method
  ([queue method x]
   `(if (< 0 (.dim ~x))
      (with-check error
        (~method (.dim ~x) (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
        ~x)
      ~x))
  ([queue method x y]
   `(if (< 0 (.dim ~x))
      (with-check error
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~queue nil)
        ~y)
      ~y))
  ([queue method x y z]
   `(if (< 0 (.dim x))
      (with-check error
        (~method (.dim ~x)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         (cl-mem (.buffer ~z)) (.offset ~z) (.stride ~z)
         ~queue nil)
        ~z)
      ~z)))

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
  `(if (< 0 (.dim ~x))
     (with-check error
       (~method (.dim ~x) ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x) ~queue nil)
       ~x)
     ~x))

(defmacro ^:private vector-axpy [queue method alpha x y]
  `(if (< 0 (.dim ~x))
     (with-check error
       (~method (.dim ~x) ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y) ~queue nil)
       ~y)
     ~y))

(defn ^:private vector-axpby [queue prog alpha ^CLBlockVector x beta ^CLBlockVector y]
  (when (< 0 (.dim x))
    (let [da (data-accessor x)]
      (with-release [vector-axpby-kernel (kernel prog "vector_axpby")]
        (set-args! vector-axpby-kernel 0
                   (.wrapPrim da alpha) (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.wrapPrim da beta) (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue vector-axpby-kernel (work-size-1d (.dim x))))))
  y)

(defmacro ^:private vector-subcopy [queue method x y kx lx ky]
  `(if (< 0 (long ~lx))
     (with-check error
       (~method ~lx (cl-mem (.buffer ~x)) ~kx (.stride ~x) (cl-mem (.buffer ~y)) ~ky (.stride ~y)
        ~queue nil)
       ~y)
     ~y))

;; =============== Common GE matrix macros and functions =======================

(defn ^:private ge-equals [ctx queue prog ^CLGEMatrix a ^CLGEMatrix b]
  (if (< 0 (.dim a))
    (with-release [ge-equals-kernel (kernel prog (name-transp "ge_equals" a b))
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)
            stor (full-storage a)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! ge-equals-kernel 0
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                   eq-flag-buf)
        (enq-nd! queue ge-equals-kernel (work-size-2d (.sd stor) (.fd stor)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.dim b))))

(defn ^:private ge-set [queue prog alpha ^CLGEMatrix a]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= (.isGapless stor)) (= 0 (.offset a))
               (= (.dim a) (.count da (.buffer a))))
        (.initialize da (.buffer a) alpha)
        (with-release [ge-set-kernel (kernel prog "ge_set")]
          (set-args! ge-set-kernel 0 (.wrapPrim da alpha)
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor)))
          (enq-nd! queue ge-set-kernel (work-size-2d (.sd stor) (.fd stor)))))))
  a)

(defmacro ^:private ge-swap [queue prog method a b]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)]
       (if (and (= (navigator ~a) (navigator ~b)) (.isGapless stor#) (.isGapless (storage ~b)))
         (with-check error
           (~method (.dim ~a)
            (cl-mem (.buffer ~a)) (.offset ~a) 1 (cl-mem (.buffer ~b)) (.offset ~b) 1
            ~queue nil)
           ~a)
         (with-release [ge-swap-kernel# (kernel ~prog (name-transp "ge_swap" ~a ~b))]
           (set-args! ge-swap-kernel# (.buffer ~a) (wrap-int (.offset ~a)) (wrap-int (.ld stor#))
                      (.buffer ~b) (wrap-int (.offset ~b)) (wrap-int (.stride ~b)))
           (enq-nd! ~queue ge-swap-kernel# (work-size-2d (.sd stor#) (.fd stor#)))
           ~a)))
     ~a))

(defmacro ^:private ge-sum-nrm2 [ctx queue prog res-bytes read-method method op-name a]
  `(if (< 0 (.dim ~a))
     (if (.isGapless (storage ~a))
       (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
         (with-check error
           (~method (.dim ~a) (cl-mem res-buffer#) 0 (cl-mem (.buffer ~a)) (.offset ~a) 1 ~queue nil)
           (~read-method ~queue res-buffer#)))
       (not-available))
     0.0))

(defmacro ^:private ge-dot [ctx queue res-bytes read-method method a b]
  `(if (< 0 (.dim ~a))
     (if (and (.isGapless (storage ~a)) (.isGapless (storage ~b)) (= (navigator ~a) (navigator ~b)))
       (with-release [res-buffer# (cl-buffer ~ctx ~res-bytes :read-write)]
         (with-check error
           (~method (.dim ~a) (cl-mem res-buffer#) 0
            (cl-mem (.buffer ~a)) (.offset ~a) 1
            (cl-mem (.buffer ~b)) (.offset ~b) 1
            ~queue nil)
           (~read-method ~queue res-buffer#)))
       (not-available))
     0.0))

(defmacro ^:private ge-omatcopy
  ([queue method alpha a b]
   `(let [nav-a# (navigator ~a)]
      (if (< 0 (.dim ~a))
        (with-check error
          (~method (.layout nav-a#)
           (if (= nav-a# (navigator ~b))
             CLBlastTranspose/CLBlastTransposeNo
             CLBlastTranspose/CLBlastTransposeYes)
           (.mrows ~a) (.ncols ~a) ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
           (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
           ~queue nil)
          ~b)
        ~b)))
  ([queue method alpha a]
   `(let [stor# (full-storage ~a)]
      (if (< 0 (.dim ~a))
        (with-check error
          (~method (.layout (navigator ~a)) CLBlastTranspose/CLBlastTransposeNo
           (.mrows ~a) (.ncols ~a) ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.ld stor#)
           (cl-mem (.buffer ~a)) (.offset ~a) (.ld stor#)
           ~queue nil)
          ~a)
        ~a)))
  ([queue method a]
   `(let [stor# (full-storage ~a)]
      (if (< 0 (.dim ~a))
        (if (.isGapless stor#)
            (with-check error
              (~method CBLAS/ORDER_COLUMN_MAJOR CLBlastTranspose/CLBlastTransposeYes
               (.sd stor#) (.fd stor#) 1.0 (cl-mem (.buffer ~a)) (.offset ~a) (.ld stor#)
               (cl-mem (.buffer ~a)) (.offset ~a) (.fd stor#)
               ~queue nil)
              ~a)
          (dragan-says-ex "You can not hard-transpose the content of a matrix with a gap in memory. Sorry."
                          {:a (info ~a)}))
        ~a))))

(defn ^:private ge-axpby [queue prog alpha ^CLGEMatrix a beta ^CLGEMatrix b]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [ge-axpby-kernel (kernel prog (name-transp "ge_axpby" a b))]
        (set-args! ge-axpby-kernel 0
                   (.wrapPrim da alpha) (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor))
                   (.wrapPrim da beta) (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue ge-axpby-kernel (work-size-2d (.sd stor) (.fd stor))))))
  b)

(defmacro ^:private ge-axpy [queue prog method alpha a b]
  `(if (< 0 (.dim ~a))
     (if (and (= (navigator ~a) (navigator ~b)) (.isGapless (storage ~a)) (.isGapless (storage ~b)))
       (with-check error
         (~method (.dim ~a) ~alpha
          (cl-mem (.buffer ~a)) (.offset ~a) 1 (cl-mem (.buffer ~b)) (.offset ~b) 1
          ~queue nil)
         ~b)
       (ge-axpby ~queue ~prog ~alpha ~a 1.0 ~b))
     ~b))

(defmacro ^:private ge-mv
  ([queue method alpha a x beta y]
   `(if (< 0 (.dim ~a))
      (with-check error
        (~method (.layout (navigator ~a)) CLBlastTranspose/CLBlastTransposeNo (.mrows ~a) (.ncols ~a)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
         ~beta (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
         ~queue nil)
        ~y)
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (str ~a)}))))

(defmacro ^:private ge-rk [queue method alpha x y a]
  `(if (< 0 (.dim ~a))
     (with-check error
       (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a)
        ~alpha (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
        (cl-mem (.buffer ~y)) (.offset ~y) (.stride ~y)
        (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
        ~queue nil)
       ~a)
     ~a))

(defmacro ^:private ge-mm
  ([alpha a b]
   `(mm (engine ~b) ~alpha ~b ~a false))
  ([queue method alpha a b beta c]
   `(if (< 0 (.dim ~a))
      (with-check error
        (let [nav-c# (navigator ~c)]
          (~method (.layout nav-c#)
           (if (= (navigator ~a) nav-c#)
             CLBlastTranspose/CLBlastTransposeNo
             CLBlastTranspose/CLBlastTransposeYes)
           (if (= (navigator ~b) nav-c#)
             CLBlastTranspose/CLBlastTransposeNo
             CLBlastTranspose/CLBlastTransposeYes)
           (.mrows ~a) (.ncols ~b) (.ncols ~a)
           ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
           (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
           ~beta (cl-mem (.buffer ~c)) (.offset ~c) (.stride ~c)
           ~queue nil))
        ~c)
      ~c)))

;; =============== Common TR matrix macros and functions ==========================

(defn ^:private tr-equals [ctx queue prog ^CLUploMatrix a ^CLUploMatrix b]
  (if (< 0 (.dim a))
    (let [res (wrap-int 0)
          stor (full-storage a)]
      (with-release [tr-equals-kernel (kernel prog (name-transp "tr_equals" a b))
                     eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! tr-equals-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                   eq-flag-buf)
        (enq-nd! queue tr-equals-kernel (work-size-2d (.sd stor) (.fd stor)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0))))
    (= 0 (.mrows b) (.ncols b))))

(defn ^:private tr-map [queue prog op-name ^CLUploMatrix a ^CLUploMatrix b]
  (when (< 0 (.dim a))
    (let [stor (full-storage a)]
      (with-release [tr-map-kernel (kernel prog (name-transp op-name a b))]
        (set-args! tr-map-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue tr-map-kernel (work-size-2d (.sd stor) (.fd stor))))))
  b)

(defn ^:private tr-axpby [queue prog alpha ^CLUploMatrix a beta ^CLUploMatrix b]
  (when (< 0 (.dim a))
    (let [da (data-accessor a)
          stor (full-storage a)]
      (with-release [tr-axpby-kernel (kernel prog (name-transp "tr_axpby" a b))]
        (set-args! tr-axpby-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                   (.wrapPrim da alpha) (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor))
                   (.wrapPrim da beta) (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue tr-axpby-kernel (work-size-2d (.sd stor) (.fd stor))))))
  b)

(defn ^:private tr-set-scal [queue prog op-name alpha ^CLUploMatrix a]
  (when (< 0 (.dim a))
    (let [stor (full-storage a)]
      (with-release [tr-op-kernel (kernel prog op-name)]
        (set-args! tr-op-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                   (.wrapPrim (data-accessor a) alpha)
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.ld stor)))
        (enq-nd! queue tr-op-kernel (work-size-2d (.sd stor) (.fd stor))))))
  a)

(defmacro ^:private tr-mv
  ([queue method a x]
   `(with-check error
      (~method (.layout (navigator ~a)) (.uplo (region ~a)) CLBlastTranspose/CLBlastTransposeNo
       (.diag (region ~a)) (.ncols ~a)
       (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
       (cl-mem (.buffer ~x)) (.offset ~x) (.stride ~x)
       ~queue nil)
      ~x))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

(defmacro ^:private tr-mm
  ([queue method alpha a b left]
   `(with-check error
      (let [reg# (region ~a)]
        (~method (.layout (navigator ~b))
         (if ~left CLBlastSide/CLBlastSideLeft CLBlastSide/CLBlastSideRight)
         (.uplo reg#)
         (if (= (navigator ~a) (navigator ~b))
           CLBlastTranspose/CLBlastTransposeNo
           CLBlastTranspose/CLBlastTransposeYes)
         (.diag reg#) (.mrows ~b) (.ncols ~b)
         ~alpha (cl-mem (.buffer ~a)) (.offset ~a) (.stride ~a)
         (cl-mem (.buffer ~b)) (.offset ~b) (.stride ~b)
         ~queue nil))
      ~b))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

;; =============== Common vectorized math functions ============================

(defn ^:private vector-math
  ([queue prog kernel-name ^CLBlockVector x ^CLBlockVector y]
   (when (< 0 (.dim x))
     (with-release [math-kernel (kernel prog kernel-name)]
       (set-args! math-kernel 0
                  (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                  (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
       (enq-nd! queue math-kernel (work-size-1d (.dim x)))))
   y)
  ([queue prog kernel-name ^CLBlockVector x ^CLBlockVector y ^CLBlockVector z]
   (when (< 0 (.dim x))
     (with-release [math-kernel (kernel prog kernel-name)]
       (set-args! math-kernel 0
                  (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                  (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y))
                  (.buffer z) (wrap-int (.offset z)) (wrap-int (.stride z)))
       (enq-nd! queue math-kernel (work-size-1d (.dim x)))))
   z))

(defn ^:private vector-linear-frac [queue prog ^CLBlockVector x ^CLBlockVector y
                                    scalea shifta scaleb shiftb ^CLBlockVector z]
 (when (< 0 (.dim x))
   (let [da (data-accessor x)]
     (if (and (= 0.0 scaleb) (= 1.0 shiftb))
       (with-release [math-kernel (kernel prog "vector_scale_shift")]
         (set-args! math-kernel 0
                    (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                    (.wrapPrim da scalea) (.wrapPrim da shifta)
                    (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
         (enq-nd! queue math-kernel (work-size-1d (.dim x))))
       (with-release [math-kernel (kernel prog "vector_linear_frac")]
         (set-args! math-kernel 0
                    (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                    (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y))
                    (.wrapPrim da scalea) (.wrapPrim da shifta) (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                    (.buffer z) (wrap-int (.offset z)) (wrap-int (.stride z)))
         (enq-nd! queue math-kernel (work-size-1d (.dim x)))))))
  z)

(defn ^:private vector-powx [queue prog ^CLBlockVector x b ^CLBlockVector y]
  (when (< 0 (.dim x))
    (with-release [math-kernel (kernel prog "vector_powx")]
      (set-args! math-kernel 0
                 (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                 (.wrapPrim (data-accessor x) b)
                 (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
      (enq-nd! queue math-kernel (work-size-1d (.dim x)))))
 y)

(defn ^:private ge-math
  ([queue prog kernel-name ^CLGEMatrix a ^CLGEMatrix b]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (kernel prog kernel-name)]
         (set-args! math-kernel 0
                    (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                    (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
         (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
   b)
  ([queue prog kernel-name ^CLGEMatrix a ^CLGEMatrix b ^CLGEMatrix c]
   (when (< 0 (.dim a))
     (check-eq-navigators a b c)
     (let [stor (full-storage a)]
       (with-release [math-kernel (kernel prog kernel-name)]
         (set-args! math-kernel 0
                    (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                    (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                    (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
         (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
   c))

(defn ^:private ge-linear-frac [queue prog ^CLGEMatrix a ^CLGEMatrix b
                                scalea shifta scaleb shiftb ^CLGEMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (kernel prog "ge_scale_shift")]
          (set-args! math-kernel 0
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                     (.wrapPrim da scalea) (.wrapPrim da shifta)
                     (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
          (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))
        (with-release [math-kernel (kernel prog "ge_linear_frac")]
          (set-args! math-kernel 0
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                     (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                     (.wrapPrim da scalea) (.wrapPrim da shifta) (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                     (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
          (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor)))))))
  c)

(defn ^:private ge-powx [queue prog ^CLGEMatrix a b ^CLGEMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)]
      (with-release [math-kernel (kernel prog "ge_powx")]
        (set-args! math-kernel 0
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.wrapPrim (data-accessor a) b)
                   (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
        (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
  c)

(defn ^:private uplo-math
  ([queue prog kernel-name ^CLUploMatrix a ^CLUploMatrix b]
   (when (< 0 (.dim a))
     (check-eq-navigators a b)
     (let [stor (full-storage a)]
       (with-release [math-kernel (kernel prog kernel-name)]
         (set-args! math-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                    (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                    (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
         (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
   b)
  ([queue prog kernel-name ^CLUploMatrix a ^CLUploMatrix b ^CLUploMatrix c]
   (when (< 0 (.dim a))
     (check-eq-navigators a b c)
     (let [stor (full-storage a)]
       (with-release [math-kernel (kernel prog kernel-name)]
         (set-args! math-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                    (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                    (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                    (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
         (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
   c))

(defn ^:private uplo-linear-frac [queue prog ^CLUploMatrix a ^CLUploMatrix b
                                scalea shifta scaleb shiftb ^CLUploMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a b c)
    (let [da (data-accessor a)
          stor (full-storage a)]
      (if (and (= 0.0 scaleb) (= 1.0 shiftb))
        (with-release [math-kernel (kernel prog "uplo_scale_shift")]
          (set-args! math-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                     (.wrapPrim da scalea) (.wrapPrim da shifta)
                     (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
          (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))
        (with-release [math-kernel (kernel prog "uplo_linear_frac")]
          (set-args! math-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                     (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                     (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                     (.wrapPrim da scalea) (.wrapPrim da shifta) (.wrapPrim da scaleb) (.wrapPrim da shiftb)
                     (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
          (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor)))))))
  c)

(defn ^:private uplo-powx [queue prog ^CLUploMatrix a b ^CLUploMatrix c]
  (when (< 0 (.dim a))
    (check-eq-navigators a c)
    (let [stor (full-storage a)]
      (with-release [math-kernel (kernel prog "uplo_powx")]
        (set-args! math-kernel 0 (wrap-int (.diag (region a))) (wrap-int (if (tr-bottom a) 1 -1))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.wrapPrim (data-accessor a) b)
                   (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c)))
        (enq-nd! queue math-kernel (work-size-2d (.sd stor) (.fd stor))))))
  c)
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
    (vector-method queue CLBlast/CLBlastDcopy ^CLBlockVector x ^CLBlockVector y))
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
  (iamin [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDamin ^CLBlockVector x))
  (rot [_ _ _ _ _]
    (not-available))
  (rotg [_ _]
    (not-available))
  (rotm [_ _ _ _]
    (not-available))
  (rotmg [_ _ _]
    (not-available))
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastDscal alpha ^CLBlockVector x))
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastDaxpy alpha ^CLBlockVector x ^CLBlockVector y))
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastDcopy ^CLBlockVector x ^CLBlockVector y kx lx ky))
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Double/BYTES enq-read-double CLBlast/CLBlastDsum ^CLBlockVector x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmax ^CLBlockVector x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiDmin ^CLBlockVector x))
  (set-all [_ alpha x]
    (vector-set ctx queue prog alpha x))
  (axpby [_ alpha x beta y]
    (vector-axpby queue prog alpha x beta y))
  VectorMath
  (sqr [_ a y]
    (vector-math queue prog "vector_sqr" a y))
  (mul [_ a b y]
    (vector-math queue prog "vector_mul" a b y))
  (div [_ a b y]
    (vector-math queue prog "vector_div" a b y))
  (inv [_ a y]
    (vector-math queue prog "vector_inv" a y))
  (abs [_ a y]
    (vector-math queue prog "vector_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (vector-math queue prog "vector_fmod" a b y))
  (frem [_ a b y]
    (vector-math queue prog "vector_frem" a b y))
  (sqrt [_ a y]
    (vector-math queue prog "vector_sqrt" a y))
  (inv-sqrt [_ a y]
    (vector-math queue prog "vector_inv_sqrt" a y))
  (cbrt [_ a y]
    (vector-math queue prog "vector_cbrt" a y))
  (inv-cbrt [_ a y]
    (vector-math queue prog "vector_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (vector-math queue prog "vector_pow2o3" a y))
  (pow3o2 [_ a y]
    (vector-math queue prog "vector_pow3o2" a y))
  (pow [_ a b y]
    (vector-math queue prog "vector_pow" a b y))
  (powx [_ a b y]
    (vector-powx queue prog a b y))
  (hypot [_ a b y]
    (vector-math queue prog "vector_hypot" a b y))
  (exp [_ a y]
    (vector-math queue prog "vector_exp" a y))
  (expm1 [_ a y]
    (vector-math queue prog "vector_expm1" a y))
  (log [_ a y]
    (vector-math queue prog "vector_log" a y))
  (log10 [_ a y]
    (vector-math queue prog "vector_log10" a y))
  (sin [_ a y]
    (vector-math queue prog "vector_sin" a y))
  (cos [_ a y]
    (vector-math queue prog "vector_cos" a y))
  (tan [_ a y]
    (vector-math queue prog "vector_tan" a y))
  (sincos [_ a y z]
    (vector-math queue prog "vector_sincos" a y z))
  (asin [_ a y]
    (vector-math queue prog "vector_asin" a y))
  (acos [_ a y]
    (vector-math queue prog "vector_acos" a y))
  (atan [_ a y]
    (vector-math queue prog "vector_atan" a y))
  (atan2 [_ a b y]
    (vector-math queue prog "vector_atan2"  a b y))
  (sinh [_ a y]
    (vector-math queue prog "vector_sinh" a y))
  (cosh [_ a y]
    (vector-math queue prog "vector_cosh" a y))
  (tanh [_ a y]
    (vector-math queue prog "vector_tanh"  a y))
  (asinh [_ a y]
    (vector-math queue prog "vector_asinh" a y))
  (acosh [_ a y]
    (vector-math queue prog "vector_acosh" a y))
  (atanh [_ a y]
    (vector-math queue prog "vector_atanh" a y))
  (erf [_ a y]
    (vector-math queue prog "vector_erf" a y))
  (erfc [_ a y]
    (vector-math queue prog "vector_erfc" a y))
  (erf-inv [_ a y]
    (vector-math queue prog "vector_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (vector-math queue prog "vector_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (vector-math queue prog "vector_gamma" a y))
  (lgamma [_ a y]
    (vector-math queue prog "vector_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (vector-math queue prog "vector_floor" a y))
  (fceil [_ a y]
    (vector-math queue prog "vector_ceil" a y))
  (trunc [_ a y]
    (vector-math queue prog "vector_trunc" a y))
  (round [_ a y]
    (vector-math queue prog "vector_round" a y))
  (modf [_ a y z]
    (vector-math queue prog "vector_modf" a y z))
  (frac [_ a y]
    (not-available))
  (fmin [_ a b y]
    (vector-math queue prog "vector_fmin" a b y))
  (fmax [_ a b y]
    (vector-math queue prog "vector_fmax" a b y)))

(deftype FloatVectorEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ x y]
    (vector-equals ctx queue prog x y))
  Blas
  (swap [_ x y]
    (vector-method queue CLBlast/CLBlastSswap ^CLBlockVector x ^CLBlockVector y)
    x)
  (copy [_ x y]
    (vector-method queue CLBlast/CLBlastScopy ^CLBlockVector x ^CLBlockVector y))
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
    (vector-ipeak ctx queue CLBlast/CLBlastiSamin ^CLBlockVector x))
  (rot [_ _ y c s]
    (not-available))
  (rotg [_ _]
    (not-available))
  (rotm [_ _ y p]
    (not-available))
  (rotmg [_ _ args]
    (not-available))
  (scal [_ alpha x]
    (vector-scal-set queue CLBlast/CLBlastSscal alpha ^CLBlockVector x))
  (axpy [_ alpha x y]
    (vector-axpy queue CLBlast/CLBlastSaxpy alpha ^CLBlockVector x ^CLBlockVector y))
  BlasPlus
  (amax [_ _]
    (not-available))
  (subcopy [_ x y kx lx ky]
    (vector-subcopy queue CLBlast/CLBlastScopy ^CLBlockVector x ^CLBlockVector y kx lx ky))
  (sum [_ x]
    (vector-sum-nrm2 ctx queue Float/BYTES enq-read-float CLBlast/CLBlastSsum ^CLBlockVector x))
  (imax [_ x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmax ^CLBlockVector x))
  (imin [this x]
    (vector-ipeak ctx queue CLBlast/CLBlastiSmin ^CLBlockVector x))
  (set-all [_ alpha x]
    (vector-set ctx queue prog alpha x))
  (axpby [_ alpha x beta y]
    (vector-axpby queue prog alpha x beta y))
  VectorMath
  (sqr [_ a y]
    (vector-math queue prog "vector_sqr" a y))
  (mul [_ a b y]
    (vector-math queue prog "vector_mul" a b y))
  (div [_ a b y]
    (vector-math queue prog "vector_div" a b y))
  (inv [_ a y]
    (vector-math queue prog "vector_inv" a y))
  (abs [_ a y]
    (vector-math queue prog "vector_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (vector-math queue prog "vector_fmod" a b y))
  (frem [_ a b y]
    (vector-math queue prog "vector_frem" a b y))
  (sqrt [_ a y]
    (vector-math queue prog "vector_sqrt" a y))
  (inv-sqrt [_ a y]
    (vector-math queue prog "vector_inv_sqrt" a y))
  (cbrt [_ a y]
    (vector-math queue prog "vector_cbrt" a y))
  (inv-cbrt [_ a y]
    (vector-math queue prog "vector_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (vector-math queue prog "vector_pow2o3" a y))
  (pow3o2 [_ a y]
    (vector-math queue prog "vector_pow3o2" a y))
  (pow [_ a b y]
    (vector-math queue prog "vector_pow" a b y))
  (powx [_ a b y]
    (vector-powx queue prog a b y))
  (hypot [_ a b y]
    (vector-math queue prog "vector_hypot" a b y))
  (exp [_ a y]
    (vector-math queue prog "vector_exp" a y))
  (expm1 [_ a y]
    (vector-math queue prog "vector_expm1" a y))
  (log [_ a y]
    (vector-math queue prog "vector_log" a y))
  (log10 [_ a y]
    (vector-math queue prog "vector_log10" a y))
  (sin [_ a y]
    (vector-math queue prog "vector_sin" a y))
  (cos [_ a y]
    (vector-math queue prog "vector_cos" a y))
  (tan [_ a y]
    (vector-math queue prog "vector_tan" a y))
  (sincos [_ a y z]
    (vector-math queue prog "vector_sincos" a y z))
  (asin [_ a y]
    (vector-math queue prog "vector_asin" a y))
  (acos [_ a y]
    (vector-math queue prog "vector_acos" a y))
  (atan [_ a y]
    (vector-math queue prog "vector_atan" a y))
  (atan2 [_ a b y]
    (vector-math queue prog "vector_atan2"  a b y))
  (sinh [_ a y]
    (vector-math queue prog "vector_sinh" a y))
  (cosh [_ a y]
    (vector-math queue prog "vector_cosh" a y))
  (tanh [_ a y]
    (vector-math queue prog "vector_tanh"  a y))
  (asinh [_ a y]
    (vector-math queue prog "vector_asinh" a y))
  (acosh [_ a y]
    (vector-math queue prog "vector_acosh" a y))
  (atanh [_ a y]
    (vector-math queue prog "vector_atanh" a y))
  (erf [_ a y]
    (vector-math queue prog "vector_erf" a y))
  (erfc [_ a y]
    (vector-math queue prog "vector_erfc" a y))
  (erf-inv [_ a y]
    (vector-math queue prog "vector_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (vector-math queue prog "vector_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (vector-math queue prog "vector_gamma" a y))
  (lgamma [_ a y]
    (vector-math queue prog "vector_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (vector-math queue prog "vector_floor" a y))
  (fceil [_ a y]
    (vector-math queue prog "vector_ceil" a y))
  (trunc [_ a y]
    (vector-math queue prog "vector_trunc" a y))
  (round [_ a y]
    (vector-math queue prog "vector_round" a y))
  (modf [_ a y z]
    (vector-math queue prog "vector_modf" a y z))
  (frac [_ a y]
    (vector-math queue prog "vector_frac" a y))
  (fmin [_ a b y]
    (vector-math queue prog "vector_fmin" a b y))
  (fmax [_ a b y]
    (vector-math queue prog "vector_fmax" a b y)))

(deftype DoubleGEEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue prog CLBlast/CLBlastDswap ^CLGEMatrix a ^CLGEMatrix b))
  (copy [_ a b]
    (ge-omatcopy queue CLBlast/CLBlastDomatcopy 1.0 ^CLGEMatrix a ^CLGEMatrix b))
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
    (ge-axpy queue prog CLBlast/CLBlastDaxpy alpha ^CLGEMatrix a ^CLGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv queue CLBlast/CLBlastDgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y))
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastDger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm queue CLBlast/CLBlastDgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [this a]
    (ge-sum-nrm2 ctx queue prog Double/BYTES enq-read-double CLBlast/CLBlastDsum "ge_sum" ^CLGEMatrix a))
  (set-all [_ alpha a]
    (ge-set queue prog alpha a))
  (axpby [_ alpha a beta b]
    (ge-axpby queue prog alpha a beta b))
  (trans [_ a]
    (ge-omatcopy queue CLBlast/CLBlastDomatcopy ^CLGEMatrix a))
  VectorMath
  (sqr [_ a y]
    (ge-math queue prog "ge_sqr" a y))
  (mul [_ a b y]
    (ge-math queue prog "ge_mul" a b y))
  (div [_ a b y]
    (ge-math queue prog "ge_div" a b y))
  (inv [_ a y]
    (ge-math queue prog "ge_inv" a y))
  (abs [_ a y]
    (ge-math queue prog "ge_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (ge-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (ge-math queue prog "ge_fmod" a b y))
  (frem [_ a b y]
    (ge-math queue prog "ge_frem" a b y))
  (sqrt [_ a y]
    (ge-math queue prog "ge_sqrt" a y))
  (inv-sqrt [_ a y]
    (ge-math queue prog "ge_inv_sqrt" a y))
  (cbrt [_ a y]
    (ge-math queue prog "ge_cbrt" a y))
  (inv-cbrt [_ a y]
    (ge-math queue prog "ge_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (ge-math queue prog "ge_pow2o3" a y))
  (pow3o2 [_ a y]
    (ge-math queue prog "ge_pow3o2" a y))
  (pow [_ a b y]
    (ge-math queue prog "ge_pow" a b y))
  (powx [_ a b y]
    (ge-powx queue prog a b y))
  (hypot [_ a b y]
    (ge-math queue prog "ge_hypot" a b y))
  (exp [_ a y]
    (ge-math queue prog "ge_exp" a y))
  (expm1 [_ a y]
    (ge-math queue prog "ge_expm1" a y))
  (log [_ a y]
    (ge-math queue prog "ge_log" a y))
  (log10 [_ a y]
    (ge-math queue prog "ge_log10" a y))
  (sin [_ a y]
    (ge-math queue prog "ge_sin" a y))
  (cos [_ a y]
    (ge-math queue prog "ge_cos" a y))
  (tan [_ a y]
    (ge-math queue prog "ge_tan" a y))
  (sincos [_ a y z]
    (ge-math queue prog "ge_sincos" a y z))
  (asin [_ a y]
    (ge-math queue prog "ge_asin" a y))
  (acos [_ a y]
    (ge-math queue prog "ge_acos" a y))
  (atan [_ a y]
    (ge-math queue prog "ge_atan" a y))
  (atan2 [_ a b y]
    (ge-math queue prog "ge_atan2"  a b y))
  (sinh [_ a y]
    (ge-math queue prog "ge_sinh" a y))
  (cosh [_ a y]
    (ge-math queue prog "ge_cosh" a y))
  (tanh [_ a y]
    (ge-math queue prog "ge_tanh"  a y))
  (asinh [_ a y]
    (ge-math queue prog "ge_asinh" a y))
  (acosh [_ a y]
    (ge-math queue prog "ge_acosh" a y))
  (atanh [_ a y]
    (ge-math queue prog "ge_atanh" a y))
  (erf [_ a y]
    (ge-math queue prog "ge_erf" a y))
  (erfc [_ a y]
    (ge-math queue prog "ge_erfc" a y))
  (erf-inv [_ a y]
    (ge-math queue prog "ge_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (ge-math queue prog "ge_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (ge-math queue prog "ge_gamma" a y))
  (lgamma [_ a y]
    (ge-math queue prog "ge_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (ge-math queue prog "ge_floor" a y))
  (fceil [_ a y]
    (ge-math queue prog "ge_ceil" a y))
  (trunc [_ a y]
    (ge-math queue prog "ge_trunc" a y))
  (round [_ a y]
    (ge-math queue prog "ge_round" a y))
  (modf [_ a y z]
    (ge-math queue prog "ge_modf" a y z))
  (frac [_ a y]
    (not-available))
  (fmin [_ a b y]
    (ge-math queue prog "ge_fmin" a b y))
  (fmax [_ a b y]
    (ge-math queue prog "ge_fmax" a b y)))

(deftype FloatGEEngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (ge-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (ge-swap queue prog CLBlast/CLBlastSswap ^CLGEMatrix a ^CLGEMatrix b))
  (copy [_ a b]
    (ge-omatcopy queue CLBlast/CLBlastSomatcopy 1.0 ^CLGEMatrix a ^CLGEMatrix b) b)
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
    (ge-axpy queue prog CLBlast/CLBlastSaxpy alpha ^CLGEMatrix a ^CLGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv queue CLBlast/CLBlastSgemv alpha ^CLGEMatrix a ^CLBlockVector x beta ^CLBlockVector y))
  (mv [this a x]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk queue CLBlast/CLBlastSger alpha ^CLBlockVector x ^CLBlockVector y ^CLGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm queue CLBlast/CLBlastSgemm alpha ^CLGEMatrix a ^CLGEMatrix b beta ^CLGEMatrix c))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [this a]
    (ge-sum-nrm2 ctx queue prog Float/BYTES enq-read-float CLBlast/CLBlastSsum "ge_sum" ^CLGEMatrix a))
  (set-all [_ alpha a]
    (ge-set queue prog alpha a))
  (axpby [_ alpha a beta b]
    (ge-axpby queue prog alpha a beta b))
  (trans [_ a]
    (ge-omatcopy queue CLBlast/CLBlastSomatcopy ^CLGEMatrix a))
  VectorMath
  (sqr [_ a y]
    (ge-math queue prog "ge_sqr" a y))
  (mul [_ a b y]
    (ge-math queue prog "ge_mul" a b y))
  (div [_ a b y]
    (ge-math queue prog "ge_div" a b y))
  (inv [_ a y]
    (ge-math queue prog "ge_inv" a y))
  (abs [_ a y]
    (ge-math queue prog "ge_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (ge-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (ge-math queue prog "ge_fmod" a b y))
  (frem [_ a b y]
    (ge-math queue prog "ge_frem" a b y))
  (sqrt [_ a y]
    (ge-math queue prog "ge_sqrt" a y))
  (inv-sqrt [_ a y]
    (ge-math queue prog "ge_inv_sqrt" a y))
  (cbrt [_ a y]
    (ge-math queue prog "ge_cbrt" a y))
  (inv-cbrt [_ a y]
    (ge-math queue prog "ge_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (ge-math queue prog "ge_pow2o3" a y))
  (pow3o2 [_ a y]
    (ge-math queue prog "ge_pow3o2" a y))
  (pow [_ a b y]
    (ge-math queue prog "ge_pow" a b y))
  (powx [_ a b y]
    (ge-powx queue prog a b y))
  (hypot [_ a b y]
    (ge-math queue prog "ge_hypot" a b y))
  (exp [_ a y]
    (ge-math queue prog "ge_exp" a y))
  (expm1 [_ a y]
    (ge-math queue prog "ge_expm1" a y))
  (log [_ a y]
    (ge-math queue prog "ge_log" a y))
  (log10 [_ a y]
    (ge-math queue prog "ge_log10" a y))
  (sin [_ a y]
    (ge-math queue prog "ge_sin" a y))
  (cos [_ a y]
    (ge-math queue prog "ge_cos" a y))
  (tan [_ a y]
    (ge-math queue prog "ge_tan" a y))
  (sincos [_ a y z]
    (ge-math queue prog "ge_sincos" a y z))
  (asin [_ a y]
    (ge-math queue prog "ge_asin" a y))
  (acos [_ a y]
    (ge-math queue prog "ge_acos" a y))
  (atan [_ a y]
    (ge-math queue prog "ge_atan" a y))
  (atan2 [_ a b y]
    (ge-math queue prog "ge_atan2"  a b y))
  (sinh [_ a y]
    (ge-math queue prog "ge_sinh" a y))
  (cosh [_ a y]
    (ge-math queue prog "ge_cosh" a y))
  (tanh [_ a y]
    (ge-math queue prog "ge_tanh"  a y))
  (asinh [_ a y]
    (ge-math queue prog "ge_asinh" a y))
  (acosh [_ a y]
    (ge-math queue prog "ge_acosh" a y))
  (atanh [_ a y]
    (ge-math queue prog "ge_atanh" a y))
  (erf [_ a y]
    (ge-math queue prog "ge_erf" a y))
  (erfc [_ a y]
    (ge-math queue prog "ge_erfc" a y))
  (erf-inv [_ a y]
    (ge-math queue prog "ge_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (ge-math queue prog "ge_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (ge-math queue prog "ge_gamma" a y))
  (lgamma [_ a y]
    (ge-math queue prog "ge_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (ge-math queue prog "ge_floor" a y))
  (fceil [_ a y]
    (ge-math queue prog "ge_ceil" a y))
  (trunc [_ a y]
    (ge-math queue prog "ge_trunc" a y))
  (round [_ a y]
    (ge-math queue prog "ge_round" a y))
  (modf [_ a y z]
    (ge-math queue prog "ge_modf" a y z))
  (frac [_ a y]
    (not-available))
  (fmin [_ a b y]
    (ge-math queue prog "ge_fmin" a b y))
  (fmax [_ a b y]
    (ge-math queue prog "ge_fmax" a b y)))

(deftype DoubleTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-map queue prog "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map queue prog "tr_copy" a b) b)
  (scal [_ alpha a]
    (tr-set-scal queue prog "tr_scal" alpha a))
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
    (tr-axpby queue prog alpha a 1.0 b))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv queue CLBlast/CLBlastDtrmv ^CLUploMatrix a ^CLBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm queue CLBlast/CLBlastDtrmm alpha ^CLUploMatrix a ^CLGEMatrix b left))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal queue prog "tr_set" alpha a))
  (axpby [_ alpha a beta b]
    (tr-axpby queue prog alpha a beta b))
  VectorMath
  (sqr [_ a y]
    (uplo-math queue prog "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math queue prog "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math queue prog "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math queue prog "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math queue prog "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math queue prog "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math queue prog "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math queue prog "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math queue prog "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math queue prog "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math queue prog "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math queue prog "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math queue prog "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math queue prog "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx queue prog a b y))
  (hypot [_ a b y]
    (uplo-math queue prog "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math queue prog "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math queue prog "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math queue prog "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math queue prog "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math queue prog "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math queue prog "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math queue prog "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math queue prog "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math queue prog "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math queue prog "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math queue prog "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math queue prog "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math queue prog "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math queue prog "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math queue prog "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math queue prog "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math queue prog "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math queue prog "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math queue prog "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math queue prog "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math queue prog "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (uplo-math queue prog "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (uplo-math queue prog "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math queue prog "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math queue prog "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math queue prog "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math queue prog "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math queue prog "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math queue prog "uplo_modf" a y z))
  (frac [_ a y]
    (not-available))
  (fmin [_ a b y]
    (uplo-math queue prog "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math queue prog "uplo_fmax" a b y)))

(deftype FloatTREngine [ctx queue prog]
  BlockEngine
  (equals-block [_ a b]
    (tr-equals ctx queue prog a b))
  Blas
  (swap [_ a b]
    (tr-map queue prog "tr_swap" a b)
    a)
  (copy [_ a b]
    (tr-map queue prog "tr_copy" a b))
  (scal [_ alpha a]
    (tr-set-scal queue prog "tr_scal" alpha a))
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
    (tr-axpby queue prog alpha a 1.0 b))
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv queue CLBlast/CLBlastStrmv ^CLUploMatrix a ^CLBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm queue CLBlast/CLBlastStrmm alpha ^CLUploMatrix a ^CLGEMatrix b left))
  BlasPlus
  (amax [_ _]
    (not-available))
  (sum [_ _]
    (not-available))
  (set-all [_ alpha a]
    (tr-set-scal queue prog "tr_set" alpha a))
  (axpby [_ alpha a beta b]
    (tr-axpby queue prog alpha a beta b))
  VectorMath
  (sqr [_ a y]
    (uplo-math queue prog "uplo_sqr" a y))
  (mul [_ a b y]
    (uplo-math queue prog "uplo_mul" a b y))
  (div [_ a b y]
    (uplo-math queue prog "uplo_div" a b y))
  (inv [_ a y]
    (uplo-math queue prog "uplo_inv" a y))
  (abs [_ a y]
    (uplo-math queue prog "uplo_abs" a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (uplo-linear-frac queue prog a b scalea shifta scaleb shiftb y))
  (fmod [_ a b y]
    (uplo-math queue prog "uplo_fmod" a b y))
  (frem [_ a b y]
    (uplo-math queue prog "uplo_frem" a b y))
  (sqrt [_ a y]
    (uplo-math queue prog "uplo_sqrt" a y))
  (inv-sqrt [_ a y]
    (uplo-math queue prog "uplo_inv_sqrt" a y))
  (cbrt [_ a y]
    (uplo-math queue prog "uplo_cbrt" a y))
  (inv-cbrt [_ a y]
    (uplo-math queue prog "uplo_inv_cbrt" a y))
  (pow2o3 [_ a y]
    (uplo-math queue prog "uplo_pow2o3" a y))
  (pow3o2 [_ a y]
    (uplo-math queue prog "uplo_pow3o2" a y))
  (pow [_ a b y]
    (uplo-math queue prog "uplo_pow" a b y))
  (powx [_ a b y]
    (uplo-powx queue prog a b y))
  (hypot [_ a b y]
    (uplo-math queue prog "uplo_hypot" a b y))
  (exp [_ a y]
    (uplo-math queue prog "uplo_exp" a y))
  (expm1 [_ a y]
    (uplo-math queue prog "uplo_expm1" a y))
  (log [_ a y]
    (uplo-math queue prog "uplo_log" a y))
  (log10 [_ a y]
    (uplo-math queue prog "uplo_log10" a y))
  (sin [_ a y]
    (uplo-math queue prog "uplo_sin" a y))
  (cos [_ a y]
    (uplo-math queue prog "uplo_cos" a y))
  (tan [_ a y]
    (uplo-math queue prog "uplo_tan" a y))
  (sincos [_ a y z]
    (uplo-math queue prog "uplo_sincos" a y z))
  (asin [_ a y]
    (uplo-math queue prog "uplo_asin" a y))
  (acos [_ a y]
    (uplo-math queue prog "uplo_acos" a y))
  (atan [_ a y]
    (uplo-math queue prog "uplo_atan" a y))
  (atan2 [_ a b y]
    (uplo-math queue prog "uplo_atan2"  a b y))
  (sinh [_ a y]
    (uplo-math queue prog "uplo_sinh" a y))
  (cosh [_ a y]
    (uplo-math queue prog "uplo_cosh" a y))
  (tanh [_ a y]
    (uplo-math queue prog "uplo_tanh"  a y))
  (asinh [_ a y]
    (uplo-math queue prog "uplo_asinh" a y))
  (acosh [_ a y]
    (uplo-math queue prog "uplo_acosh" a y))
  (atanh [_ a y]
    (uplo-math queue prog "uplo_atanh" a y))
  (erf [_ a y]
    (uplo-math queue prog "uplo_erf" a y))
  (erfc [_ a y]
    (uplo-math queue prog "uplo_erfc" a y))
  (erf-inv [_ a y]
    (uplo-math queue prog "uplo_erf_inv" a y))
  (erfc-inv [_ a y]
    (not-available))
  (cdf-norm [_ a y]
    (uplo-math queue prog "uplo_cdf_norm" a y))
  (cdf-norm-inv [_ a y]
    (not-available))
  (gamma [_ a y]
    (uplo-math queue prog "uplo_gamma" a y))
  (lgamma [_ a y]
    (uplo-math queue prog "uplo_lgamma" a y))
  (expint1 [_ a y]
    (not-available))
  (floor [_ a y]
    (uplo-math queue prog "uplo_floor" a y))
  (fceil [_ a y]
    (uplo-math queue prog "uplo_ceil" a y))
  (trunc [_ a y]
    (uplo-math queue prog "uplo_trunc" a y))
  (round [_ a y]
    (uplo-math queue prog "uplo_round" a y))
  (modf [_ a y z]
    (uplo-math queue prog "uplo_modf" a y z))
  (frac [_ a y]
    (not-available))
  (fmin [_ a b y]
    (uplo-math queue prog "uplo_fmin" a b y))
  (fmax [_ a b y]
    (uplo-math queue prog "uplo_fmax" a b y)))

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
  (create-ge [this m n column? init]
    (let-release [res (cl-ge-matrix this m n column?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-uplo [this n mat-type column? lower? diag-unit? init]
    (let-release [res (cl-uplo-matrix this n column? lower? diag-unit? mat-type)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (create-tr [this n column? lower? diag-unit? init]
    (let-release [res (cl-uplo-matrix this n column? lower? diag-unit?)]
      (when init
        (.initialize da (.buffer ^Block res)))
      res))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng))

(let [src [(slurp (io/resource "uncomplicate/neanderthal/internal/device/blas-plus.cl"))
           (slurp (io/resource "uncomplicate/neanderthal/internal/device/vect-math.cl"))]]

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
