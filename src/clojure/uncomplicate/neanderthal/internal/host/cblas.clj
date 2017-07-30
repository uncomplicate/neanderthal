;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.cblas
  (:require [uncomplicate.neanderthal.block :refer [buffer offset stride]]
            [uncomplicate.neanderthal.internal
             [api :refer [engine mm iamax stripe-navigator swap copy scal axpy axpby fits-navigation? region]]
             [common :refer [dragan-says-ex]]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS]
           [uncomplicate.neanderthal.internal.api RealVector GEMatrix BandedMatrix Matrix Region]))

;; =============== Common vector engine  macros and functions ==================

(defmacro vector-rot [method x y c s]
  `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y) ~c ~s))

(defmacro vector-rotm [method x y param]
  `(if (= 1 (.stride ~param))
     (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
      (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (str ~param)}))))

(defmacro vector-rotmg [method d1d2xy param]
  `(if (= 1 (.stride ~param))
     (~method (.buffer ~d1d2xy) (.stride ~d1d2xy) (.offset ~d1d2xy) (.buffer ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (str ~param)}))))

(defmacro vector-amax [x]
  `(if (< 0 (.dim ~x))
     (Math/abs (.entry ~x (iamax (engine ~x) ~x)))
     0.0))

(defmacro vector-method
  ([method x]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)))
  ([method x y]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([method x y z]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y)
     (.buffer ~z) (.offset ~z) (.stride ~z))))

(defn vector-imax [^RealVector x]
  (let [cnt (.dim x)]
    (if (< 0 cnt)
      (loop [i 1 max-idx 0 max-val (.entry x 0)]
        (if (< i cnt)
          (let [v (.entry x i)]
            (if (< max-val v)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx))
      0)))

(defn vector-imin [^RealVector x]
  (let [cnt (.dim x)]
    (if (< 0 cnt)
      (loop [i 1 min-idx 0 min-val (.entry x 0)]
        (if (< i cnt)
          (let [v (.entry x i)]
            (if (< v min-val)
              (recur (inc i) i v)
              (recur (inc i) min-idx min-val)))
          min-idx))
      0)))

;; =============== Common GE matrix macros and functions =======================

(defmacro ge-sum [method a]
  `(if (< 0 (.count ~a))
     (let [ld# (.stride ~a)
           sd# (.sd ~a)
           fd# (.fd ~a)
           offset# (.offset ~a)
           buff# (.buffer ~a)]
       (if (= sd# ld#)
         (~method (.count ~a) buff# offset# 1)
         (loop [j# 0 acc# 0.0]
           (if (< j# fd#)
             (recur (inc j#) (+ acc# (~method sd# buff# (+ offset# (* ld# j#)) 1)))
             acc#))))
     0.0))

(defmacro ge-dot [method a b]
  `(if (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# (.sd ~b) ld-a# ld-b#)
           (~method (.count ~a) buff-a# offset-a# 1 buff-b# offset-b# 1)
           (loop [j# 0 acc# 0.0]
             (if (< j# fd-a#)
               (recur (inc j#) (+ acc# (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1
                                        buff-b# (+ offset-b# (* ld-b# j#)) 1)))
               acc#)))
         (loop [j# 0 acc# 0.0]
           (if (< j# fd-a#)
             (recur (inc j#) (+ acc# (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1
                                      buff-b# (+ offset-b# j#) ld-b#)))
             acc#))))
     0.0))

(defmacro ge-swap [method a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# (.sd ~b) ld-a# ld-b#)
           (~method (.count ~a) buff-a# offset-a# 1 buff-b# offset-b# 1)
           (dotimes [j# fd-a#]
             (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1)))
         (dotimes [j# fd-a#]
           (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# j#) ld-b#))))))

(defmacro ge-mv
  ([method alpha a x beta y]
   `(~method (.order ~a) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a)
     ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~x) (.offset ~x) (.stride ~x) ~beta (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (str ~a)}))))

(defmacro ge-rk [method alpha x y a]
  `(~method (.order ~a) (.mrows ~a) (.ncols ~a) ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
    (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a) (.stride ~a)))

(defmacro ge-mm
  ([alpha a b]
   `(mm (engine ~b) ~alpha ~b ~a false))
  ([method alpha a b beta c]
   `(if (instance? GEMatrix ~b)
      (~method (.order ~c)
       (if (= (.order ~a) (.order ~c)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
       (if (= (.order ~b) (.order ~c)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
       (.mrows ~a) (.ncols ~b) (.ncols ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
       (.buffer ~b) (.offset ~b) (.stride ~b) ~beta (.buffer ~c) (.offset ~c) (.stride ~c))
      (mm (engine ~b) ~alpha ~b ~a ~beta ~c false))))

;; =============== Common UPLO matrix macros and functions ==========================

(defmacro uplo-swap [method a b]
  `(when (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# j# (* ld-b# start#)) n#)))))))

(defmacro uplo-axpy [method alpha a b]
  `(when (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j# ~alpha
              buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j# ~alpha
              buff-a# (+ offset-a# j# (* ld-a# start#)) n#
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))))))

(defmacro uplo-axpby [method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j#
              ~alpha buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (~method n-j#
              ~alpha buff-a# (+ offset-a# j# (* ld-a# start#)) n#
              ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))))))

;; =============== Common TR matrix macros and functions ============================

(defmacro tr-dot [method a b]
  `(if (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (loop [j# 0 acc# 0.0]
           (if (< j# n#)
             (let [start# (.start stripe-nav# n# j#)
                   n-j# (- (.end stripe-nav# n# j#) start#)]
               (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                                        buff-b# (+ offset-b# (* ld-b# j#) start#) 1))))
             acc#))
         (loop [j# 0 acc# 0.0]
           (if (< j# n#)
             (let [start# (.start stripe-nav# n# j#)
                   n-j# (- (.end stripe-nav# n# j#) start#)]
               (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                                        buff-b# (+ offset-b# j# (* ld-b# start#)) n#))))
             acc#))))
     0.0))

(defmacro tr-sum [method a]
  `(if (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)]
       (loop [j# 0 acc# 0.0]
         (if (< j# n#)
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1))))
           acc#)))
     0.0))

(defmacro tr-mv
  ([method a x]
   `(~method (.order ~a) (.uplo ~a) CBLAS/TRANSPOSE_NO_TRANS (.diag ~a) (.ncols ~a)
     (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x)))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

(defmacro tr-mm
  ([method alpha a b left]
   `(~method (.order ~b) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT) (.uplo ~a)
     (if (= (.order ~a) (.order ~b)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
     (.diag ~a) (.mrows ~b) (.ncols ~b)
     ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (str ~a)}))))

;; -------------------- SY

(defmacro sy-dot [method a b]
  `(if (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (- (* 2.0 (double
                  (if (= (.order ~a) (.order ~b))
                    (loop [j# 0 acc# 0.0]
                      (if (< j# n#)
                        (let [start# (.start stripe-nav# n# j#)
                              n-j# (- (.end stripe-nav# n# j#) start#)]
                          (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                                                   buff-b# (+ offset-b# (* ld-b# j#) start#) 1))))
                        acc#))
                    (loop [j# 0 acc# 0.0]
                      (if (< j# n#)
                        (let [start# (.start stripe-nav# n# j#)
                              n-j# (- (.end stripe-nav# n# j#) start#)]
                          (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
                                                   buff-b# (+ offset-b# j# (* ld-b# start#)) n#))))
                        acc#)))))
          (~method n# buff-a# offset-a# (inc ld-a#) buff-b# offset-b# (inc ld-b#))))
     0.0))

(defmacro sy-sum [method a]
  `(if (< 0 (.count ~a))
     (let [stripe-nav# (stripe-navigator ~a)
           n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)]
       (loop [j# 0 acc# 0.0]
         (if (< j# n#)
           (let [start# (.start stripe-nav# n# j#)
                 n-j# (- (.end stripe-nav# n# j#) start#)]
             (recur (inc j#) (+ acc# (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1))))
           (- (* 2.0 acc#) (~method n# buff-a# offset-a# (inc ld-a#))))))
     0.0))

(defmacro sy-mv
  ([method alpha a x beta y]
   `(~method (.order ~a) (.uplo ~a) (.ncols ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~x) (.offset ~x) (.stride ~x) ~beta (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for SY matrices." {:a (str ~a)}))))

(defmacro sy-mm
  ([method alpha a b beta c left]
   `(~method (.order ~c) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT) (.uplo ~a)
     (.mrows ~c) (.ncols ~c) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~b) (.offset ~b) (.stride ~b) ~beta (.buffer ~c) (.offset ~c) (.stride ~c)))
  ([a]
   `(throw (ex-info "In-place mm! is not supported for SY matrices." {:a (str ~a)}))))

;; ====================== Banded Matrix ===========================================

(defn gb-map [fun ^BandedMatrix a ^BandedMatrix b]
  (let [eng (engine (.dia a))]
    (loop [i (- (.kl a))]
      (when (< i (inc (.ku a)))
        (fun eng (.dia a i) (.dia b i))
        (recur (inc i))))
    b))

(defn gb-scalset [fun alpha ^BandedMatrix a]
  (let [eng (engine (.dia a))]
    (loop [i (- (.kl a))]
      (when (< i (inc (.ku a)))
        (fun eng alpha (.dia a i))
        (recur (inc i)))))
  a)

(defn gb-reduce
  ([fun ^BandedMatrix a]
   (let [eng (engine (.dia a))]
     (loop [i (- (.kl a)) acc 0.0]
       (if (< i (inc (.ku a)))
         (recur (inc i) (+ acc (double (fun eng (.dia a i)))))
         acc))))
  ([fun ^BandedMatrix a ^BandedMatrix b]
   (let [eng (engine (.dia a))]
     (loop [i (- (.kl a)) acc 0.0]
       (if (< i (inc (.ku a)))
         (recur (inc i) (+ acc (double (fun eng (.dia a i) (.dia b i)))))
         acc)))))

(defn gb-axpy [alpha ^BandedMatrix a ^BandedMatrix b]
  (let [eng (engine (.dia a))]
    (loop [i (- (.kl a))]
      (when (< i (inc (.ku a)))
        (axpy eng alpha (.dia a i) (.dia b i))
        (recur (inc i))))
    b))

(defn gb-axpby [alpha ^BandedMatrix a beta ^BandedMatrix b]
  (let [eng (engine (.dia a))]
    (loop [i (- (.kl a))]
      (when (< i (inc (.ku a)))
        (axpby eng alpha (.dia a i) beta (.dia b i))
        (recur (inc i))))
    b))

(defmacro gb-mv
  ([method alpha a x beta y]
   `(do
      (~method (.order ~a) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a) (.kl ~a) (.ku ~a) ~alpha
       (.buffer ~a) (.offset ~a) (.stride ~a) (buffer ~x) (offset ~x) (stride ~x)
       ~beta (buffer ~y) (offset ~y) (stride ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GB matrices." {:a (str ~a)}))))

(defmacro gb-mm
  ([method alpha a b beta c _]
   `(do
      (dotimes [j# (.ncols ~b)]
        (gb-mv ~method ~alpha ~a (.col ~b j#) ~beta (.col ~c j#)))
      ~c))
  ([a]
   `(throw (ex-info "In-place mm! is not supported for GB matrices." {:a (str ~a)}))))

;; ===================== Packed Matrix ==============================

(defn packed-len ^long [^Matrix a]
  (let [n (.ncols a)]
    (unchecked-divide-int (* n (inc n)) 2)))

(defmacro with-layout [a b & expr]
  `(if (= (.order ~a) (.order ~b))
     (do ~@expr)
     (dragan-says-ex "This operation on packed matrices is not efficient if the layouts do not match. Copy to a matcking layout."
                     {:a (str ~a) :b (str ~b)})))

(defmacro packed-amax [method da a]
  `(Math/abs (.get ~da (.buffer ~a) (+ (.offset ~a) (~method (packed-len ~a) (.buffer ~a) (.offset ~a) 1)))))

(defmacro packed-map
  ([method a]
   `(do
      (~method (packed-len ~a) (.buffer ~a) (.offset ~a) 1)
      ~a))
  ([method a b]
   `(with-layout ~a ~b
      (~method (packed-len ~a) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
      ~b)))

(defmacro packed-scal [method alpha a]
  `(do
     (~method (packed-len ~a) ~alpha (.buffer ~a) (.offset ~a) 1)
     ~a))

(defmacro packed-axpy [method alpha a b]
  `(with-layout ~a ~b
     (~method (packed-len ~a) ~alpha (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro packed-axpby [method alpha a beta b]
  `(with-layout ~a ~b
     (~method (packed-len ~a) ~alpha (.buffer ~a) (.offset ~a) 1 ~beta (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro packed-set-all [method alpha a]
  `(do
     (~method (packed-len ~a) ~alpha (.buffer ~a) (.offset ~a) 1)
     ~a))

(defn blas-uplo ^long [^Region region]
  (if (.isLower region) CBLAS/UPLO_LOWER CBLAS/UPLO_UPPER))

(defn blas-diag ^long [^Region region]
  (if (.isDiagUnit region) CBLAS/DIAG_UNIT CBLAS/DIAG_NON_UNIT))

(defmacro tp-reduce [method a]
  `(~method (packed-len ~a) (.buffer ~a) (.offset ~a) 1))

(defmacro tp-dot [method a b]
  `(with-layout ~a ~b
     (~method (packed-len ~a) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)))

(defmacro tp-mv
  ([method a x]
   `(let [region# (region ~a)]
      (~method (.order ~a) (blas-uplo region#) CBLAS/TRANSPOSE_NO_TRANS (blas-diag region#)
       (.ncols ~a) (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported by TP matrices." {:a (str ~a)})))

(defmacro tp-mm
  ([method alpha a b left]
   `(if ~left
      (let [region# (region ~a)
            layout# (.order ~a)
            n-a# (.ncols ~a)
            buff-a# (.buffer ~a)
            ofst-a# (.offset ~a)
            buff-b# (.buffer ~b)
            ofst-b# (.offset ~b)
            ld-b# (.stride ~b)
            uplo# (blas-uplo region#)
            diag# (blas-diag region#)]
        (if (= CBLAS/ORDER_COLUMN_MAJOR (.order ~b))
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
             buff-a# ofst-a# buff-b# (+ ofst-b# (* j# ld-b#)) 1))
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
             buff-a# ofst-a# buff-b# (+ ofst-b# j#) ld-b#)))
        ~b)
      (dragan-says-ex "Transforming a TP matrix by another matrix type is not supported. Copy TP to TR."
                       {:a (str ~a) :b (str ~b)})))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TP matrices. Copy to GE." {:a (str ~a)})))

(defmacro sp-mv
  ([method alpha a x beta y]
   `(let [region# (region ~a)]
      (~method (.order ~a) (blas-uplo region#) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported by SY matrices." {:a (str ~a)})))

(defmacro sp-mm
  ([method alpha a b beta c left]
   `(if ~left
      (with-layout ~a ~b
        (dotimes [j# (.ncols ~c)]
          (sp-mv ~method ~alpha ~a (.col ~b j#) ~beta (.col ~c j#)))
        ~b)
      (dragan-says-ex "Transforming a SP matrix by another matrix type is not supported. Copy SP to SY."
                       {:a (str ~a) :b (str ~b)})))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SP matrices. Copy to GE." {:a (str ~a)})))
