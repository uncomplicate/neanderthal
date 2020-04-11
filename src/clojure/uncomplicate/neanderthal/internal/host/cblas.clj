;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.cblas
  (:require [uncomplicate.neanderthal.math :refer [f=]]
            [uncomplicate.commons
             [core :refer [info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.internal
             [api :refer [engine mm mv iamax swap copy scal axpy axpby region navigator storage]]
             [common :refer [real-accessor check-eq-navigators flip-uplo]]
             [navigation :refer [accu-layout full-storage]]])
  (:import uncomplicate.neanderthal.internal.host.CBLAS
           [uncomplicate.neanderthal.internal.api RealVector Matrix Region GEMatrix UploMatrix]))

;; =============== Common vector engine  macros and functions ==================

(defmacro vector-rot [method x y c s]
  `(do
     (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y) ~c ~s)
     ~x))

(defmacro vector-rotm [method x y param]
  `(if (= 1 (.stride ~param))
     (do
       (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
        (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~param) (.offset ~param))
       ~x)
     (dragan-says-ex "You cannot use strided vector as param." {:param (info ~param)})))

(defmacro vector-rotmg [method d1d2xy param]
  `(if (= 1 (.stride ~param))
     (do
       (~method (.buffer ~d1d2xy) (.stride ~d1d2xy) (.offset ~d1d2xy) (.buffer ~param) (.offset ~param))
       ~param)
     (dragan-says-ex "You cannot use strided vector as param." {:param (info ~param)})))

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
   `(do
      (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y)
       (.buffer ~z) (.offset ~z) (.stride ~z))
      ~z)))

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

(defmacro dot-ones [dot n buff ofst strd n1 ones]
  `(let [remain# (rem ~n ~n1)
         ni# (- ~n remain#)]
     (loop [i# 0 res# (~dot remain# ~buff (+ ni# ~ofst) ~strd ~ones 0 1)]
       (if (< i# ni#)
         (recur (+ i# ~n1)
                (+ res# (~dot ~n1 ~buff (+ i# ~ofst) ~strd ~ones 0 1)))
         res#))))

(defmacro vector-sum [dot x ones]
  `(let [n# (.dim ~x)
         n1# (.dim ~ones)
         buff# (.buffer ~x)
         ofst# (.offset ~x)
         strd# (.stride ~x)
         ones# (.buffer ~ones)]
     (dot-ones ~dot n# buff# ofst# strd# n1# ones#)))

;; =============== Common GE matrix macros and functions =======================

(defmacro full-storage-reduce
  ([a b len offset-a offset-b ld-b acc init expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~a)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          n# (.fd stor-a#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (if (= nav-a# nav-b#)
        (if (and (.isGapless stor-a#) (.isGapless stor-b#))
          ~expr-direct
          (let [~ld-b 1]
            (loop [j# 0 ~acc ~init]
              (if (< j# n#)
                (recur (inc j#)
                       (let [start# (.start nav-a# reg# j#)
                             ~len (- (.end nav-a# reg# j#) start#)
                             ~offset-a (+ offset-a# (.index stor-a# start# j#))
                             ~offset-b (+ offset-b# (.index stor-b# start# j#))]
                         ~expr))
                ~acc))))
        (let [~ld-b (.ld stor-b#)]
          (loop [j# 0 ~acc ~init]
            (if (< j# n#)
              (recur (inc j#)
                     (let [start# (.start nav-a# reg# j#)
                           ~len (- (.end nav-a# reg# j#) start#)
                           ~offset-a (+ offset-a# (.index stor-a# start# j#))
                           ~offset-b (+ offset-b# (.index stor-b# j# start#))]
                       ~expr))
              ~acc)))))))

(defmacro full-storage-map
  ([a b len offset-a offset-b ld-a expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-b# (.fd stor-b#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (if (= nav-a# nav-b#)
        (if (and (.isGapless stor-a#) (.isGapless stor-b#))
          ~expr-direct
          (let [~ld-a 1]
            (dotimes [j# fd-b#]
              (let [start# (.start nav-b# reg# j#)
                    ~len (- (.end nav-b# reg# j#) start#)
                    ~offset-a (+ offset-a# (.index stor-a# start# j#))
                    ~offset-b (+ offset-b# (.index stor-b# start# j#))]
                ~expr))))
        (let [~ld-a (.ld stor-a#)]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg# j#)
                  ~len (- (.end nav-b# reg# j#) start#)
                  ~offset-a (+ offset-a# (.index stor-a# j# start#))
                  ~offset-b (+ offset-b# (.index stor-b# start# j#))]
              ~expr)))))))

(defmacro symmetric-full-storage-reduce
  ([a b len offset-a offset-b ld-b acc init expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~a)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          n# (.fd stor-a#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (if (or (= nav-a# nav-b#) (and (not= nav-a# nav-b#) (not= (.uplo (region ~a)) (.uplo (region ~b)))))
        (let [~ld-b 1]
          (loop [j# 0 ~acc ~init]
            (if (< j# n#)
              (recur (inc j#)
                     (let [start# (.start nav-a# reg# j#)
                           ~len (- (.end nav-a# reg# j#) start#)
                           ~offset-a (+ offset-a# (.index stor-a# start# j#))
                           ~offset-b (+ offset-b# (.index stor-b# start# j#))]
                       ~expr))
              ~acc)))
        (let [~ld-b (.ld stor-b#)]
          (loop [j# 0 ~acc ~init]
            (if (< j# n#)
              (recur (inc j#)
                     (let [start# (.start nav-a# reg# j#)
                           ~len (- (.end nav-a# reg# j#) start#)
                           ~offset-a (+ offset-a# (.index stor-a# start# j#))
                           ~offset-b (+ offset-b# (.index stor-b# j# start#))]
                       ~expr))
              ~acc)))))))

(defmacro symmetric-full-storage-map
  ([a b len offset-a offset-b ld-a expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-b# (.fd stor-b#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (if (or (= nav-a# nav-b#) (and (not= nav-a# nav-b#) (not= (.uplo (region ~a)) (.uplo (region ~b)))))
        (let [~ld-a 1]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg# j#)
                  ~len (- (.end nav-b# reg# j#) start#)
                  ~offset-a (+ offset-a# (.index stor-a# start# j#))
                  ~offset-b (+ offset-b# (.index stor-b# start# j#))]
              ~expr)))
        (let [~ld-a (.ld stor-a#)]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg# j#)
                  ~len (- (.end nav-b# reg# j#) start#)
                  ~offset-a (+ offset-a# (.index stor-a# j# start#))
                  ~offset-b (+ offset-b# (.index stor-b# start# j#))]
              ~expr)))))))

(defmacro matrix-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)]
       (full-storage-reduce ~a ~b len# offset-a# offset-b# ld-b# acc# 0.0
                            (~method (.dim ~a) buff-a# (.offset ~a) 1 buff-b# (.offset ~b) 1)
                            (+ acc# (~method len# buff-a# offset-a# 1 buff-b# offset-b# ld-b#))))
     0.0))

(defmacro matrix-swap [method a b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                           (~method (.dim ~a) buff-a# (.offset ~a) 1 buff-b# (.offset ~b) 1)
                           (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# 1))))
     ~a))

(defmacro matrix-axpy [method alpha a b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                           (~method (.dim ~a) ~alpha buff-a# (.offset ~a) 1 buff-b# (.offset ~b) 1)
                           (~method len# ~alpha buff-a# offset-a# ld-a# buff-b# offset-b# 1))))
     ~b))

(defmacro matrix-axpby [method alpha a beta b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                           (~method (.dim ~a) ~alpha buff-a# (.offset ~a) 1 ~beta buff-b# (.offset ~b) 1)
                           (~method len# ~alpha buff-a# offset-a# ld-a# ~beta buff-b# offset-b# 1))))
     ~b))

(defmacro matrix-sum
  ([method a]
    `(if (< 0 (.dim ~a))
       (let [buff# (.buffer ~a)
             offset# (.offset ~a)]
         (if (.isGapless (storage ~a))
           (~method (.dim ~a) buff# offset# 1)
           (accu-layout ~a len# idx# acc# 0.0 (+ acc# (~method len# buff# (+ offset# idx#) 1)))))
       0.0))
  ([dot a ones]
   `(if (< 0 (.dim ~a))
      (let [buff# (.buffer ~a)
            ofst# (.offset ~a)
            ones# (.buffer ~ones)
            n1# (.dim ~ones)]
        (if (.isGapless (storage ~a))
          (let [n# (.dim ~a)]
            (dot-ones ~dot n# buff# ofst# 1 n1# ones#))
          (accu-layout ~a len# idx# acc# 0.0 (+ acc# (double (dot-ones ~dot len# buff# (+ ofst# idx#) 1 n1# ones#))))))
      0.0)))

(defmacro ge-mv
  ([method alpha a x beta y]
   `(do
      (~method (.layout (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for GE matrices." {:a (info ~a)})))

(defmacro ge-rk [method alpha x y a]
  `(do
     (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a)
      ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
      (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a) (.stride ~a))
     ~a))

(defmacro ge-mm
  ([alpha a b]
   `(if-not (instance? GEMatrix ~b)
      (mm (engine ~b) ~alpha ~b ~a false)
      (dragan-says-ex "In-place mm! is not supported for GE matrices. Use QR factorization."
                      {:a (info ~a) :b (info ~b)} )))
  ([method alpha a b beta c]
   `(if (instance? GEMatrix ~b)
      (let [nav# (navigator ~c)]
        (~method (.layout nav#)
         (if (= nav# (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
         (if (= nav# (navigator ~b)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
         (.mrows ~a) (.ncols ~b) (.ncols ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
         (.buffer ~b) (.offset ~b) (.stride ~b) ~beta (.buffer ~c) (.offset ~c) (.stride ~c))
        ~c)
      (mm (engine ~b) ~alpha ~b ~a ~beta ~c false))))

;; =============== Common TR matrix macros and functions ============================

(defmacro tr-sum [method a]
  `(if (.isDiagUnit (region ~a))
     (+ (.ncols ~a) (double (matrix-sum ~method ~a)))
     (matrix-sum ~method ~a)))

(defmacro tr-dot [method a b]
  `(if (.isDiagUnit (region ~a))
     (+ (.ncols ~a) (double (matrix-dot ~method ~a ~b)))
     (matrix-dot ~method ~a ~b)))

(defmacro tr-mv
  ([method a x]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo reg#) CBLAS/TRANSPOSE_NO_TRANS (.diag reg#)
       (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)})))

(defmacro tr-mm
  ([method alpha a b left]
   `(let [nav# (navigator ~b)
          reg# (region ~a)
          nav-eq# (= nav# (navigator ~a))]
      (~method (.layout nav#) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT)
       (if nav-eq# (.uplo reg#) (flip-uplo (.uplo reg#)))
       (if nav-eq# CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
       (.diag reg#) (.mrows ~b) (.ncols ~b)
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b))
      ~b))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported for TR matrices." {:a (info ~a)})))

;; =============== Common SY matrix macros and functions ============================

(defmacro sy-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)
           n# (.fd stor#)
           buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           uplo-dot# (symmetric-full-storage-reduce
                      ~a ~b len# offset-a# offset-b# ld-b# acc# 0.0
                      (+ acc# (~method len# buff-a# offset-a# 1 buff-b# offset-b# ld-b#)))
           uplo-diag# (~method n# (.buffer ~a) (.offset ~a) (inc (.ld stor#))
                       (.buffer ~b) (.offset ~b) (inc (.ld (full-storage ~b))))]
       (- (* 2.0 (double uplo-dot#)) uplo-diag#))
     0.0))

(defmacro sy-swap [method a b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (symmetric-full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                                     (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# 1))))
     ~a))

(defmacro sy-axpy [method alpha a b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (symmetric-full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                                     (~method len# ~alpha buff-a# offset-a# ld-a# buff-b# offset-b# 1))))
     ~b))

(defmacro sy-axpby [method alpha a beta b]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)]
         (symmetric-full-storage-map ~a ~b len# offset-a# offset-b# ld-a#
                                     (~method len# ~alpha buff-a# offset-a# ld-a# ~beta buff-b# offset-b# 1))))
     ~b))

(defmacro sy-sum [method a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)
           n# (.fd stor#)]
       (- (* 2.0 (double (matrix-sum ~method ~a)))
          (~method n# (.buffer ~a) (.offset ~a) (inc (.ld stor#)))))
     0.0))

(defmacro sy-mv
  ([method alpha a x beta y]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo reg#) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for SY matrices." {:a (info ~a)}))))

(defmacro sy-r
  ([method alpha x y a]
   `(do
      (~method (.layout (navigator ~a)) (.uplo (region ~a)) (.mrows ~a)
       ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
       (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a) (.stride ~a))
      ~a))
  ([method alpha x a]
   `(do
      (~method (.layout (navigator ~a)) (.uplo (region ~a)) (.mrows ~a)
       ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
       (.buffer ~a) (.offset ~a) (.stride ~a))
      ~a)))

(defmacro sy-rk [method alpha a beta c]
  `(if (instance? UploMatrix ~c)
     (let [nav# (navigator ~c)]
       (~method (.layout nav#) (.uplo (region ~c))
        (if (= nav# (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
        (.mrows ~c) (.ncols ~a)
        ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
        ~beta (.buffer ~c) (.offset ~c) (.stride ~c))
       ~c)
     (throw (ex-info "sy-rk is only available for symmetric matrices." {:c (info ~c)}))))

(defmacro sy-mm
  ([method alpha a b beta c left]
   `(let [nav-c# (navigator ~c)
          uplo# (if (= nav-c# (navigator ~a)) (.uplo (region ~a)) (flip-uplo (.uplo (region ~a))))]
      (if (= nav-c# (navigator ~b))
        (~method (.layout (navigator ~c)) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT)
         uplo# (.mrows ~c) (.ncols ~c)
         ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)
         ~beta (.buffer ~c) (.offset ~c) (.stride ~c))
        (dragan-says-ex "Both GE matrices in symmetric multiplication must have the same orientation."
                        {:b (info ~b) :c (info ~c)}))
      ~c))
  ([a]
   `(throw (ex-info "In-place mm! is not supported for SY matrices." {:a (info ~a)}))))

;; ====================== Banded Matrix ===========================================

(defmacro band-storage-reduce
  ([a len offset acc init expr]
   `(let [reg# (region ~a)
          nav# (navigator ~a)
          stor# (full-storage ~a)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset# (.offset ~a)]
      (let [acc# (loop [k# 0 ~acc ~init]
                   (if (< k# (inc kl#))
                     (recur (inc k#)
                            (let [~len (min (- m# k#) n#)
                                  ~offset (+ offset# (.index nav# stor# k# 0))]
                              ~expr))
                     ~acc))]
        (loop [k# 1 ~acc (double acc#)]
          (if (< k# (inc ku#))
            (recur (inc k#)
                   (let [~len (min m# (- n# k#))
                         ~offset (+ offset# (.index nav# stor# 0 k#))]
                     ~expr))
            ~acc)))))
  ([a b len offset-a offset-b acc init expr]
   `(let [reg# (region ~a)
          nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (let [acc# (loop [k# 0 ~acc ~init]
                   (if (< k# (inc kl#))
                     (recur (inc k#)
                            (let [~len (min (- m# k#) n#)
                                  ~offset-a (+ offset-a# (.index nav-a# stor-a# k# 0))
                                  ~offset-b (+ offset-b# (.index nav-b# stor-b# k# 0))]
                              ~expr))
                     ~acc))]
        (loop [k# 1 ~acc (double acc#)]
          (if (< k# (inc ku#))
            (recur (inc k#)
                   (let [~len (min m# (- n# k#))
                         ~offset-a (+ offset-a# (.index nav-a# stor-a# 0 k#))
                         ~offset-b (+ offset-b# (.index nav-b# stor-b# 0 k#))]
                     ~expr))
            ~acc))))))

(defmacro band-storage-map
  ([a len offset expr]
   `(let [reg# (region ~a)
          nav# (navigator ~a)
          stor# (full-storage ~a)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset# (.offset ~a)]
      (dotimes [k# (inc kl#)]
        (let [~len (min (- m# k#) n#)
              ~offset (+ offset# (.index nav# stor# k# 0))]
          ~expr))
      (dotimes [k# ku#]
        (let [~len (min m# (- n# (inc k#)))
              ~offset (+ offset# (.index nav# stor# 0 (inc k#)))]
          ~expr))
      ~a))
  ([a b len offset-a offset-b expr]
   `(let [reg# (region ~a)
          nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (dotimes [k# (inc kl#)]
        (let [~len (min (- m# k#) n#)
              ~offset-a (+ offset-a# (.index nav-a# stor-a# k# 0))
              ~offset-b (+ offset-b# (.index nav-b# stor-b# k# 0))]
          ~expr))
      (dotimes [k# ku#]
        (let [~len (min m# (- n# (inc k#)))
              ~offset-a (+ offset-a# (.index nav-a# stor-a# 0 (inc k#)))
              ~offset-b (+ offset-b# (.index nav-b# stor-b# 0 (inc k#)))]
          ~expr))
      ~b)))

(defmacro symmetric-band-storage-map [a b len offset-a offset-b expr]
  `(let [reg# (region ~a)
         nav# (navigator ~a)
         stor-a# (full-storage ~a)
         stor-b# (full-storage ~b)
         m# (.mrows ~a)
         n# (.ncols ~a)
         kl# (.kl reg#)
         ku# (.ku reg#)
         offset-a# (.offset ~a)
         offset-b# (.offset ~b)]
     (dotimes [k# (inc kl#)]
       (let [~len (min (- m# k#) n#)
             ~offset-a (+ offset-a# (.index nav# stor-a# k# 0))
             ~offset-b (+ offset-b# (.index nav# stor-b# k# 0))]
         ~expr))
     (dotimes [k# ku#]
       (let [~len (min m# (- n# (inc k#)))
             ~offset-a (+ offset-a# (.index nav# stor-a# 0 (inc k#)))
             ~offset-b (+ offset-b# (.index nav# stor-b# 0 (inc k#)))]
         ~expr))
     ~b))

(defmacro symmetric-band-storage-reduce
  ([a len offset acc init expr]
   `(let [reg# (region ~a)
          nav# (navigator ~a)
          stor# (full-storage ~a)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset# (.offset ~a)]
      (let [acc# (loop [k# 1 ~acc ~init]
                   (if (< k# (inc kl#))
                     (recur (inc k#)
                            (let [~len (min (- m# k#) n#)
                                  ~offset (+ offset# (.index nav# stor# k# 0))]
                              ~expr))
                     ~acc))]
        (let [~acc (* 2.0 (double
                           (loop [k# 1 ~acc (double acc#)]
                             (if (< k# (inc ku#))
                               (recur (inc k#)
                                      (let [~len (min m# (- n# k#))
                                            ~offset (+ offset# (.index nav# stor# 0 k#))]
                                        ~expr))
                               ~acc))))
              ~len (min m# n#)
              ~offset (+ offset# (.index nav# stor# 0 0))]
          ~expr))))
  ([a b len offset-a offset-b acc init expr]
   `(let [reg# (region ~a)
          nav# (navigator ~a)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (let [acc# (loop [k# 1 ~acc ~init]
                   (if (< k# (inc kl#))
                     (recur (inc k#)
                            (let [~len (min (- m# k#) n#)
                                  ~offset-a (+ offset-a# (.index nav# stor-a# k# 0))
                                  ~offset-b (+ offset-b# (.index nav# stor-b# k# 0))]
                              ~expr))
                     ~acc))]
        (let [~acc (* 2.0 (double
                           (loop [k# 1 ~acc (double acc#)]
                             (if (< k# (inc ku#))
                               (recur (inc k#)
                                      (let [~len (min m# (- n# k#))
                                            ~offset-a (+ offset-a# (.index nav# stor-a# 0 k#))
                                            ~offset-b (+ offset-b# (.index nav# stor-b# 0 k#))]
                                        ~expr))
                               ~acc))))
              ~len (min m# n#)
              ~offset-a (+ offset-a# (.index nav# stor-a# 0 0))
              ~offset-b (+ offset-b# (.index nav# stor-b# 0 0))]
          ~expr)))))

(defmacro gb-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-reduce ~a ~b len# offset-a# offset-b# acc# 0.0
                            (+ acc# (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#))))
     0.0))

(defmacro sb-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (symmetric-band-storage-reduce ~a ~b len# offset-a# offset-b# acc# 0.0
                                      (+ acc# (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#))))
     0.0))

(defmacro tb-dot [method a b]
  `(if (.isDiagUnit (region ~a))
     (+ (.ncols ~a) (double (gb-dot ~method ~a ~b)))
     (gb-dot ~method ~a ~b)))

(defmacro gb-map [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro sb-map [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (symmetric-band-storage-map ~a ~b len# offset-a# offset-b#
                                   (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro gb-scal [method alpha a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ld-a# (.stride ~a)]
       (band-storage-map ~a len# offset# (~method len# ~alpha buff# offset# ld-a#)))
     ~a))

(defmacro gb-sum [method a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ld-a# (.stride ~a)]
       (band-storage-reduce ~a len# offset# acc# 0.0 (+ acc# (~method len# buff# offset# ld-a#))))
     0.0))

(defmacro sb-sum [method a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ld-a# (.stride ~a)]
       (symmetric-band-storage-reduce ~a len# offset# acc# 0.0
                                      (+ acc# (~method len# buff# offset# ld-a#))))
     0.0))

(defmacro tb-sum [method a]
  `(if (.isDiagUnit (region ~a))
     (+ (.ncols ~a) (double (gb-sum ~method ~a)))
     (gb-sum ~method ~a)))

(defmacro gb-axpy [method alpha a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# ~alpha buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro sb-axpy [method alpha a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (symmetric-band-storage-map ~a ~b len# offset-a# offset-b#
                                   (~method len# ~alpha buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro gb-axpby [method alpha a beta b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# ~alpha buff-a# offset-a# ld-a# ~beta buff-b# offset-b# ld-b#)))
     ~b))

(defmacro sb-axpby [method alpha a beta b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (symmetric-band-storage-map ~a ~b len# offset-a# offset-b#
                                   (~method len# ~alpha buff-a# offset-a# ld-a# ~beta buff-b# offset-b# ld-b#)))
     ~b))

(defmacro gb-mv
  ([method alpha a x beta y]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a)
       (.kl reg#) (.ku reg#) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
       (.buffer ~x) (.offset ~x) (.stride ~x) ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for GB matrices." {:a (info ~a)})))

(defmacro sb-mv
  ([method alpha a x beta y]
   `(let [reg# (region ~a)]
      (~method CBLAS/ORDER_COLUMN_MAJOR CBLAS/UPLO_LOWER (.ncols ~a) (max (.kl reg#) (.ku reg#))
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
       (.buffer ~x) (.offset ~x) (.stride ~x) ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for SB matrices." {:a (info ~a)})))

(defmacro tb-mv
  ([method a x]
   `(let [reg# (region ~a)]
      (~method CBLAS/ORDER_COLUMN_MAJOR CBLAS/UPLO_LOWER
       (if (.isLower reg#) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS) (.diag reg#) (.ncols ~a)
       (max (.kl reg#) (.ku reg#))
       (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported for TB matrices." {:a (info ~a)})))

(defmacro gb-mm
  ([method alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (full-storage ~b)
              stor-c# (full-storage ~c)
              m-a# (.mrows ~a)
              n-a# (.ncols ~a)
              kl-a# (.kl reg#)
              ku-a# (.ku reg#)
              buff-a# (.buffer ~a)
              ofst-a# (.offset ~a)
              ld-a# (.stride ~a)
              buff-b# (.buffer ~b)
              ofst-b# (.offset ~b)
              buff-c# (.buffer ~c)
              ofst-c# (.offset ~c)
              stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.ld stor-b#))
              stride-col-c# (if (.isColumnMajor (navigator ~c)) 1 (.ld stor-c#))]
          (dotimes [j# (.ncols ~b)]
            (~method layout# CBLAS/TRANSPOSE_NO_TRANS m-a# n-a# kl-a# ku-a#
             ~alpha buff-a# ofst-a# ld-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#
             ~beta buff-c# (+ ofst-c# (.index nav-c# stor-c# 0 j#)) stride-col-c#)))
        (mm (engine ~a) ~alpha (.transpose ~a) (.transpose ~b) ~beta (.transpose ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by GB matrices. Copy GB to GE." {:a (info ~a)})))

(defmacro sb-mm
  ([method alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (storage ~b)
              stor-c# (storage ~c)
              uplo# (.uplo reg#)
              n-a# (.ncols ~a)
              k-a# (max (.kl reg#) (.ku reg#))
              m-b# (.mrows ~b)
              n-b# (.ncols ~b)
              buff-a# (.buffer ~a)
              ofst-a# (.offset ~a)
              ld-a# (.stride ~a)
              buff-b# (.buffer ~b)
              ofst-b# (.offset ~b)
              buff-c# (.buffer ~c)
              ofst-c# (.offset ~c)
              stride-x# (if (.isColumnMajor (navigator ~b)) (if ~left 1 (.stride ~b)) (if ~left (.stride ~b) 1))
              stride-col-c# (if (.isColumnMajor nav-c#) 1 (.stride ~c))]
          (dotimes [j# n-b#]
            (~method CBLAS/ORDER_COLUMN_MAJOR CBLAS/UPLO_LOWER n-a# k-a#
             ~alpha buff-a# ofst-a# ld-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-x#
             ~beta buff-c# (+ ofst-c# (.index nav-c# stor-c# 0 j#)) stride-col-c#)))
        (mm (engine ~a) ~alpha (.transpose ~a) (.transpose ~b) ~beta (.transpose ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SB matrices. Copy SB to GE." {:a (info ~a)})))

(defmacro tb-mm
  ([method alpha a b left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              stor-b# (storage ~b)
              uplo# (.uplo reg#)
              diag# (.diag reg#)
              n-a# (.ncols ~a)
              k-a# (max (.kl reg#) (.ku reg#))
              m-b# (.mrows ~b)
              n-b# (.ncols ~b)
              buff-a# (.buffer ~a)
              ofst-a# (.offset ~a)
              ld-a# (.stride ~a)
              buff-b# (.buffer ~b)
              ofst-b# (.offset ~b)
              stride-x# (if (.isColumnMajor (navigator ~b)) (if ~left 1 (.stride ~b)) (if ~left (.stride ~b) 1))
              transpose# (if (.isLower reg#) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)]
          (dotimes [j# n-b#]
            (~method CBLAS/ORDER_COLUMN_MAJOR CBLAS/UPLO_LOWER transpose# diag# n-a# k-a#
             buff-a# ofst-a# ld-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-x#))
          (when-not (f= 1.0 ~alpha)
            (scal (engine ~b) ~alpha ~b)))
        (mm (engine ~a) ~alpha (.transpose ~a) (.transpose ~b) true))
      ~b))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TB matrices. Copy TB to GE." {:a (info ~a)})))

(defmacro tb-sv
  ([method a b]
   `(let [reg# (region ~a)
          layout# (.layout (navigator ~a))
          nav-b# (navigator ~b)
          stor-b# (storage ~b)
          uplo# (.uplo reg#)
          diag# (.diag reg#)
          m-b# (.mrows ~b)
          n-a# (.ncols ~a)
          ku-a# (.ku reg#)
          buff-a# (.buffer ~a)
          ofst-a# (.offset ~a)
          ld-a# (.stride ~a)
          buff-b# (.buffer ~b)
          ofst-b# (.offset ~b)
          stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.stride ~b))]
      (dotimes [j# (.ncols ~b)]
        (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a# ku-a#
         buff-a# ofst-a# ld-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
      ~b)))

(defmacro tr-sv [method a b]
  `(let [nav-a# (navigator ~a)
         nav-b# (navigator ~b)
         reg# (region ~a)
         uplo# (if (= nav-a# nav-b#)
                 (if (.isLower reg#) CBLAS/UPLO_LOWER CBLAS/UPLO_UPPER)
                 (if (.isLower reg#) CBLAS/UPLO_UPPER CBLAS/UPLO_LOWER))]
     (~method (.layout nav-b#) CBLAS/SIDE_LEFT uplo#
      (if (= nav-a# nav-b#) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
      (.diag reg#) (.mrows ~b) (.ncols ~b) 1.0
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b))
     ~b))

;; ===================== Packed Matrix ==============================

(defmacro packed-map
  ([method a]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1)
      ~a))
  ([method a b]
   `(do
      (check-eq-navigators ~a ~b)
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
      ~b)))

(defmacro packed-scal [method alpha a]
  `(do
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1)
     ~a))

(defmacro packed-axpy [method alpha a b]
  `(do
     (check-eq-navigators ~a ~b)
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro packed-axpby [method alpha a beta b]
  `(do
     (check-eq-navigators ~a ~b)
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 ~beta (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro tp-sum
  ([method etype a]
   `(if (< 0 (.dim ~a))
      (let [stor# (storage ~a)
            da# (real-accessor ~a)
            n# (.ncols ~a)
            buff# (.buffer ~a)
            ofst# (.offset ~a)]
        (if-not (.isDiagUnit (region ~a))
          (~method (.surface (region ~a)) buff# ofst# 1)
          (loop [i# 0 acc# (+ n# (~method (+ n# (.surface (region ~a))) buff# ofst# 1))]
            (if (< i# n#)
              (recur (inc i#) (- acc# (~etype (.get da# buff# (+ ofst# (.index stor# i# i#))))))
              acc#))))
      0.0))
  ([dot etype a ones]
   `(if (< 0 (.dim ~a))
      (let [stor# (storage ~a)
            da# (real-accessor ~a)
            n# (.ncols ~a)
            buff# (.buffer ~a)
            ofst# (.offset ~a)
            n1# (.dim ~ones)
            ones# (.buffer ~ones)]
        (if-not (.isDiagUnit (region ~a))
          (dot-ones ~dot (.surface (region ~a)) buff# ofst# 1 n1# ones#)
          (loop [i# 0 acc# (+ n# (~etype (dot-ones ~dot (+ n# (.surface (region ~a))) buff# ofst# 1 n1# ones#)))]
            (if (< i# n#)
              (recur (inc i#) (- acc# (~etype (.get da# buff# (+ ofst# (.index stor# i# i#))))))
              acc#))))
      0.0)))

(defmacro tp-dot [method a b]
  `(do
     (check-eq-navigators ~a ~b)
     (if (< 0 (.dim ~a))
       (let [da# (real-accessor ~a)
             n# (.ncols ~a)
             buff-a# (.buffer ~a)
             ofst-a# (.offset ~a)
             buff-b# (.buffer ~b)
             ofst-b# (.offset ~b)
             stor# (storage ~a)
             dot# (~method (.surface (region ~a)) buff-a# ofst-a# 1 buff-b# ofst-b# 1)]
         (if-not (.isDiagUnit (region ~a))
           dot#
           (loop [i# 0 acc# (+ n# dot#)]
             (if (< i# n#)
               (recur (inc i#) (- acc# (* (.get da# buff-a# (+ ofst-a# (.index stor# i# i#)))
                                          (.get da# buff-b# (+ ofst-b# (.index stor# i# i#))))))
               acc#))))
       0.0)))

(defmacro tp-mv
  ([method a x]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo reg#) CBLAS/TRANSPOSE_NO_TRANS (.diag reg#)
       (.ncols ~a) (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported by TP matrices." {:a (info ~a)})))

(defmacro tp-mm
  ([method alpha a b left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              stor-b# (storage ~b)
              uplo# (.uplo reg#)
              diag# (.diag reg#)
              n-a# (.ncols ~a)
              buff-a# (.buffer ~a)
              ofst-a# (.offset ~a)
              buff-b# (.buffer ~b)
              ofst-b# (.offset ~b)
              stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.stride ~b))]
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
             buff-a# ofst-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
          (when-not (f= 1.0 ~alpha)
            (scal (engine ~b) ~alpha ~b)))
        (mm (engine ~a) ~alpha (.transpose ~a) (.transpose ~b) true))
      ~b))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TP matrices. Copy to GE." {:a (info ~a)})))

(defmacro tp-sv [method a b]
  `(let [reg# (region ~a)
         layout# (.layout (navigator ~a))
         nav-b# (navigator ~b)
         stor-b# (storage ~b)
         uplo# (.uplo reg#)
         diag# (.diag reg#)
         n-a# (.ncols ~a)
         buff-a# (.buffer ~a)
         ofst-a# (.offset ~a)
         buff-b# (.buffer ~b)
         ofst-b# (.offset ~b)
         stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.stride ~b))]
     (dotimes [j# (.ncols ~b)]
       (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
        buff-a# ofst-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
     ~b))

;; ============================ Symmetric Packed Matrix ================================

(defmacro sp-sum
  ([method etype a]
   `(if (< 0 (.dim ~a))
      (let [da# (real-accessor ~a)
            n# (.ncols ~a)
            buff# (.buffer ~a)
            ofst# (.offset ~a)
            stor# (storage ~a)]
        (loop [i# 0 acc# (* 2.0 (~method (.surface (region ~a)) buff# ofst# 1))]
          (if (< i# n#)
            (recur (inc i#) (- acc# (~etype (.get da# buff# (+ ofst# (.index stor# i# i#))))))
            acc#)))
      0.0))
  ([dot etype a ones]
   `(if (< 0 (.dim ~a))
      (let [da# (real-accessor ~a)
            n# (.ncols ~a)
            buff# (.buffer ~a)
            ofst# (.offset ~a)
            stor# (storage ~a)
            n1# (.dim ~ones)
            ones# (.buffer ~ones)]
        (loop [i# 0 acc# (* 2.0 (~etype (dot-ones ~dot (.surface (region ~a)) buff# ofst# 1 n1# ones#)))]
          (if (< i# n#)
            (recur (inc i#) (- acc# (~etype (.get da# buff# (+ ofst# (.index stor# i# i#))))))
            acc#)))
      0.0)))

(defmacro sp-dot [method a b]
  `(do
     (check-eq-navigators ~a ~b)
     (if (< 0 (.dim ~a))
       (let [da# (real-accessor ~a)
             n# (.ncols ~a)
             buff-a# (.buffer ~a)
             ofst-a# (.offset ~a)
             buff-b# (.buffer ~b)
             ofst-b# (.offset ~b)
             stor# (storage ~a)
             dot# (~method (.surface (region ~a)) buff-a# ofst-a# 1 buff-b# ofst-b# 1)]
         (loop [i# 0 acc# (* 2.0 dot#)]
           (if (< i# n#)
             (recur (inc i#)
                    (- acc#
                       (* (.get da# buff-a# (+ ofst-a# (.index stor# i# i#)))
                          (.get da# buff-b# (+ ofst-b# (.index stor# i# i#))))))
             acc#)))
       0.0)))

(defmacro sp-mv
  ([method alpha a x beta y]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo reg#) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported by SY matrices. Now way around that." {:a (info ~a)})))

(defmacro sp-r
  ([method alpha x y a]
   `(do
      (~method (.layout (navigator ~a)) (.uplo (region ~a)) (.mrows ~a)
       ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
       (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a))
      ~a))
  ([method alpha x a]
   `(do
      (~method (.layout (navigator ~a)) (.uplo (region ~a)) (.mrows ~a)
       ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
       (.buffer ~a) (.offset ~a))
      ~a)))

(defmacro sp-mm
  ([method alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (storage ~b)
              stor-c# (storage ~c)
              uplo# (.uplo reg#)
              n-a# (.ncols ~a)
              buff-a# (.buffer ~a)
              ofst-a# (.offset ~a)
              buff-b# (.buffer ~b)
              ofst-b# (.offset ~b)
              buff-c# (.buffer ~c)
              ofst-c# (.offset ~c)
              stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.stride ~b))
              stride-col-c# (if (.isColumnMajor (navigator ~c)) 1 (.stride ~c))]
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# n-a#
             ~alpha buff-a# ofst-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#
             ~beta buff-c# (+ ofst-c# (.index nav-c# stor-c# 0 j#)) stride-col-c#))
          ~c)
        (mm (engine ~a) ~alpha (.transpose ~a) (.transpose ~b) ~beta (.transpose ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SP matrices. No way around that." {:a (info ~a)})))

;; ===================== Tridiagonal matrix ==================================================

(defmacro diagonal-method
  ([method a]
   `(~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1))
  ([method a b]
   `(~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1))
  ([method a b c]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1
       (.buffer ~c) (.offset ~c) 1)
      ~c)))

(defmacro diagonal-scal [method alpha a]
  `(do
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1)
     ~a))

(defmacro diagonal-axpy [method alpha a b]
  `(do
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro diagonal-axpby [method alpha a beta b]
  `(do
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 ~beta (.buffer ~b) (.offset ~b) 1)
     ~b))

(defmacro diagonal-amax [method a]
  `(if (< 0 (.dim ~a))
     (let [da# (real-accessor ~a)
           buff# (.buffer ~a)
           ofst# (.offset ~a)]
       (Math/abs (.get da# buff# (+ ofst# (~method (.surface (region ~a)) buff# ofst# 1)))))
     0.0))

(defmacro st-sum
  ([method a]
   `(let [n# (.ncols ~a)
          buff# (.buffer ~a)
          ofst# (.offset ~a)]
      (+ (~method n# buff# ofst# 1) (* 2.0 (~method (dec n#) buff# (+ ofst# n#) 1)))))
  ([method a b]
   `(let [n# (.ncols ~a)
          buff-a# (.buffer ~a)
          ofst-a# (.offset ~a)
          buff-b# (.buffer ~b)
          ofst-b# (.offset ~b)]
      (+ (~method n# buff-a# ofst-a# 1 buff-b# ofst-b# 1)
         (* 2.0 (~method (dec n#) buff-a# (+ ofst-a# n#) 1  buff-b# (+ ofst-b# n#) 1))))))

(defmacro st-nrm2 [method a]
  `(let [n# (.ncols ~a)
         buff# (.buffer ~a)
         ofst# (.offset ~a)]
     (Math/sqrt (+ (Math/pow (~method n# buff# ofst# 1) 2)
                   (* 2.0 (Math/pow (~method (dec n#) buff# (+ ofst# n#) 1) 2))))))
