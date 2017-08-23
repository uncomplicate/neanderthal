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
             [api :refer [engine mm iamax swap copy scal axpy axpby region navigator storage info]]
             [common :refer [dragan-says-ex]]
             [navigation :refer [accu-layout full-storage]]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS]
           [uncomplicate.neanderthal.internal.api RealVector Matrix Region GEMatrix]))

(defn flip-uplo ^long [^long uplo]
  (case uplo
    121 122
    122 121
    (throw (ex-info "Invalid CBLAS uplo"))))

;; =============== Common vector engine  macros and functions ==================

(defmacro vector-rot [method x y c s]
  `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y) ~c ~s))

(defmacro vector-rotm [method x y param]
  `(if (= 1 (.stride ~param))
     (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
      (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (info ~param)}))))

(defmacro vector-rotmg [method d1d2xy param]
  `(if (= 1 (.stride ~param))
     (~method (.buffer ~d1d2xy) (.stride ~d1d2xy) (.offset ~d1d2xy) (.buffer ~param))
     (throw (ex-info "You cannot use strided vector as param." {:param (info ~param)}))))

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

(defmacro matrix-sum [method a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           offset# (.offset ~a)]
       (if (.isGapless (storage ~a))
         (~method (.dim ~a) buff# offset# 1)
         (accu-layout ~a len# idx# acc# 0.0 (+ acc# (~method len# buff# (+ offset# idx#) 1)))))
     0.0))

(defmacro ge-mv
  ([method alpha a x beta y]
   `(do
      (~method (.layout (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GE matrices." {:a (info ~a)}))))

(defmacro ge-rk [method alpha x y a]
  `(do
     (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a)
      ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
      (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a) (.stride ~a))
     ~a))

(defmacro ge-mm
  ([alpha a b]
   `(mm (engine ~b) ~alpha ~b ~a false))
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
  `(if-not (.isDiagUnit (region ~a))
     (matrix-sum ~method ~a)
     (+ (.ncols ~a) (double (matrix-sum ~method ~a)))))

(defmacro tr-dot [method a b]
  `(if-not (.isDiagUnit (region ~a))
     (matrix-dot ~method ~a ~b)
     (+ (.ncols ~a) (double (matrix-dot ~method ~a ~b)))))

(defmacro tr-mv
  ([method a x]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo reg#) CBLAS/TRANSPOSE_NO_TRANS (.diag reg#)
       (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

(defmacro tr-mm
  ([method alpha a b left]
   `(let [nav# (navigator ~b)
          reg# (region ~a)]
      (~method (.layout nav#) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT) (.uplo reg#)
       (if (= nav# (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS CBLAS/TRANSPOSE_TRANS)
       (.diag reg#) (.mrows ~b) (.ncols ~b)
       ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b))
      ~b))
  ([a]
   `(throw (ex-info "Out-of-place mv! is not supported for TR matrices." {:a (info ~a)}))))

;; =============== Common SY matrix macros and functions ============================

(defmacro sy-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)
           n# (.fd stor#)]
       (- (* 2.0 (double (matrix-dot ~method ~a ~b)))
          (~method n# (.buffer ~a) (.offset ~a) (inc (.ld stor#))
           (.buffer ~b) (.offset ~b) (inc (.ld (full-storage ~b))))))
     0.0))

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

(defmacro banded-dot [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-reduce ~a ~b len# offset-a# offset-b# acc# 0.0
                            (+ acc# (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#))))
     0.0))

(defmacro banded-map [method a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro banded-scal [method alpha a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ld-a# (.stride ~a)]
       (band-storage-map ~a len# offset# (~method len# ~alpha buff# offset# ld-a#)))
     ~a))

(defmacro banded-sum [method a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ld-a# (.stride ~a)]
       (band-storage-reduce ~a len# offset# acc# 0.0 (+ acc# (~method len# buff# offset# ld-a#))))
     0.0))

(defmacro banded-axpy [method alpha a b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# ~alpha buff-a# offset-a# ld-a# buff-b# offset-b# ld-b#)))
     ~b))

(defmacro banded-axpby [method alpha a beta b]
  `(if (< 0 (.dim ~a))
     (let [buff-a# (.buffer ~a)
           buff-b# (.buffer ~b)
           ld-a# (.stride ~a)
           ld-b# (.stride ~b)]
       (band-storage-map ~a ~b len# offset-a# offset-b#
                         (~method len# ~alpha buff-a# offset-a# ld-a# ~beta buff-b# offset-b# ld-b#)))
     ~b))

(defmacro gb-mv
  ([method alpha a x beta y]
   `(let [reg# (region ~a)]
      (~method (.layout (navigator ~a)) CBLAS/TRANSPOSE_NO_TRANS (.mrows ~a) (.ncols ~a)
       (.kl reg#) (.ku reg#) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
       (buffer ~x) (offset ~x) (stride ~x) ~beta (buffer ~y) (offset ~y) (stride ~y))
      ~y))
  ([a]
   `(throw (ex-info "In-place mv! is not supported for GB matrices." {:a (info ~a)}))))

(defmacro gb-mm
  ([method alpha a b beta c _]
   `(do
      (dotimes [j# (.ncols ~b)]
        (gb-mv ~method ~alpha ~a (.col ~b j#) ~beta (.col ~c j#)))
      ~c))
  ([a]
   `(throw (ex-info "In-place mm! is not supported for GB matrices." {:a (info ~a)}))))

;; ===================== Packed Matrix ==============================

(defmacro check-layout [a b & expr]
  `(if (= (navigator ~a) (navigator ~b))
     (do ~@expr)
     (dragan-says-ex "This operation on packed matrices is not efficient if the layouts do not match. Copy to a matcking layout."
                     {:a (info ~a) :b (info ~b)})))

(defmacro packed-amax [method da a]
  `(if (< 0 (.dim ~a))
     (let [buff# (.buffer ~a)
           ofst# (.offset ~a)]
       (if-not (.isDiagUnit (region ~a))
         (Math/abs (.get ~da buff# (+ ofst# (~method (.surface (region ~a)) buff# ofst# 1))))
         (accu-layout ~a len# idx# max# 1.0
                      (max max# (Math/abs (.get ~da buff# (+ ofst# idx# (~method len# buff# (+ ofst# idx#) 1))))))))
     0.0))

(defmacro packed-map
  ([method a]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1)
      ~a))
  ([method a b]
   `(check-layout ~a ~b
                  (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
                  ~b)))

(defmacro packed-scal [method alpha a]
  `(do
     (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1)
     ~a))

(defmacro packed-axpy [method alpha a b]
  `(check-layout
    ~a ~b
    (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 (.buffer ~b) (.offset ~b) 1)
    ~b))

(defmacro packed-axpby [method alpha a beta b]
  `(check-layout
    ~a ~b
    (~method (.surface (region ~a)) ~alpha (.buffer ~a) (.offset ~a) 1 ~beta (.buffer ~b) (.offset ~b) 1)
    ~b))

(defmacro tp-sum [method transform da a]
  `(if (< 0 (.dim ~a))
     (let [n# (.ncols ~a)
           stor# (storage ~a)
           buff# (.buffer ~a)
           ofst# (.offset ~a)]
       (if-not (.isDiagUnit (region ~a))
         (~method (.surface (region ~a)) buff# ofst# 1)
         (loop [i# 0 acc# (+ n# (~method (+ n# (.surface (region ~a))) buff# ofst# 1))]
           (if (< i# n#)
             (recur (inc i#) (- acc# (~transform (.get ~da buff# (+ ofst# (.index stor# i# i#))))))
             acc#))))
     0.0))

(defmacro tp-dot [method da a b]
  `(check-layout
    ~a ~b
    (if (< 0 (.dim ~a))
      (let [n# (.ncols ~a)
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
                 (recur (inc i#) (- acc# (* (.get ~da buff-a# (+ ofst-a# (.index stor# i# i#)))
                                            (.get ~da buff-b# (+ ofst-b# (.index stor# i# i#))))))
                 acc#))))
      0.0)))

(defmacro tp-nrm2 [method da a]
  `(if (< 0 (.dim ~a))
     (let [n# (.ncols ~a)
           buff# (.buffer ~a)
           ofst# (.offset ~a)
           stor# (storage ~a)
           nrm# (~method (.surface (region ~a)) buff# ofst# 1)]
       (if-not (.isDiagUnit (region ~a))
         nrm#
         (Math/sqrt (loop [i# 0 acc# (+ n# (* nrm# nrm#))]
                      (if (< i# n#)
                        (recur (inc i#) (- acc# (Math/pow (.get ~da buff# (+ ofst# (.index stor# i# i#))) 2)))
                        acc#)))))
     0.0))

(defmacro tp-mv
  ([method a x]
   `(let [region# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo region#) CBLAS/TRANSPOSE_NO_TRANS (.diag region#)
       (.ncols ~a) (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported by TP matrices." {:a (info ~a)})))

(defmacro tp-mm
  ([method alpha a b left]
   `(if ~left
      (let [region# (region ~a)
            layout# (.layout (navigator ~a))
            n-a# (.ncols ~a)
            buff-a# (.buffer ~a)
            ofst-a# (.offset ~a)
            buff-b# (.buffer ~b)
            ofst-b# (.offset ~b)
            ld-b# (.stride ~b)
            uplo# (.uplo region#)
            diag# (.diag region#)]
        (if (.isColumnMajor (navigator ~b))
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
             buff-a# ofst-a# buff-b# (+ ofst-b# (* j# ld-b#)) 1))
          (dotimes [j# (.ncols ~b)]
            (~method layout# uplo# CBLAS/TRANSPOSE_NO_TRANS diag# n-a#
             buff-a# ofst-a# buff-b# (+ ofst-b# j#) ld-b#)))
        ~b)
      (dragan-says-ex "Transforming a TP matrix by another matrix type is not supported. Copy TP to TR."
                      {:a (info ~a) :b (info ~b)})))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TP matrices. Copy to GE." {:a (info ~a)})))

;; ============================ Symmetric Packed Matrix ================================

(defmacro sp-sum [method transform da a]
  `(if (< 0 (.dim ~a))
     (let [n# (.ncols ~a)
           buff# (.buffer ~a)
           ofst# (.offset ~a)
           stor# (storage ~a)]
       (loop [i# 0 acc# (* 2.0 (~method (.surface (region ~a)) buff# ofst# 1))]
         (if (< i# n#)
           (recur (inc i#) (- acc# (~transform (.get ~da buff# (+ ofst# (.index stor# i# i#))))))
           acc#)))
     0.0))

(defmacro sp-nrm2 [method da a]
  `(if (< 0 (.dim ~a))
     (let [n# (.ncols ~a)
           buff# (.buffer ~a)
           ofst# (.offset ~a)
           stor# (storage ~a)]
       (Math/sqrt (loop [i# 0 acc# (* 2.0 (Math/pow (~method (.surface (region ~a)) buff# ofst# 1) 2))]
                    (if (< i# n#)
                      (recur (inc i#) (- acc# (Math/pow (.get ~da buff# (+ ofst# (.index stor# i# i#))) 2)))
                      acc#))))
     0.0))

(defmacro sp-dot [method da a b]
  `(check-layout
    ~a ~b
    (if (< 0 (.dim ~a))
      (let [n# (.ncols ~a)
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
                      (* (.get ~da buff-a# (+ ofst-a# (.index stor# i# i#)))
                         (.get ~da buff-b# (+ ofst-b# (.index stor# i# i#))))))
            acc#)))
      0.0)))

(defmacro sp-mv
  ([method alpha a x beta y]
   `(let [region# (region ~a)]
      (~method (.layout (navigator ~a)) (.uplo region#) (.ncols ~a)
       ~alpha (.buffer ~a) (.offset ~a) (.buffer ~x) (.offset ~x) (.stride ~x)
       ~beta (.buffer ~y) (.offset ~y) (.stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported by SY matrices. Now way around that." {:a (info ~a)})))

(defmacro sp-mm
  ([method alpha a b beta c left]
   `(if ~left
      (let [region# (region ~a)
            layout# (.layout (navigator ~a))
            n-a# (.ncols ~a)
            buff-a# (.buffer ~a)
            ofst-a# (.offset ~a)
            buff-b# (.buffer ~b)
            ofst-b# (.offset ~b)
            ld-b# (.stride ~b)
            buff-c# (.buffer ~c)
            ofst-c# (.offset ~c)
            ld-c# (.stride ~c)
            uplo# (.uplo region#)]
        (if (.isColumnMajor (navigator ~b))
          (if (.isColumnMajor (navigator ~c))
            (dotimes [j# (.ncols ~b)]
              (~method layout# uplo# n-a# ~alpha buff-a# ofst-a# buff-b# (+ ofst-b# (* j# ld-b#)) 1
               ~beta buff-c# (+ ofst-c# (* j# ld-c#)) 1))
            (dotimes [j# (.ncols ~b)]
              (~method layout# uplo# n-a# ~alpha buff-a# ofst-a# buff-b# (+ ofst-b# (* j# ld-b#)) 1
               ~beta buff-c# (+ ofst-c# j#) ld-c#)))
          (if (.isColumnMajor (navigator ~c))
            (dotimes [j# (.ncols ~b)]
              (~method layout# uplo# n-a# ~alpha buff-a# ofst-a# buff-b# (+ ofst-b# j#) ld-b#
               ~beta buff-c# (+ ofst-c# (* j# ld-c#)) 1))
            (dotimes [j# (.ncols ~b)]
              (~method layout# uplo# n-a# ~alpha buff-a# ofst-a# buff-b# (+ ofst-b# j#) ld-b#
               ~beta buff-c# (+ ofst-c# j#) ld-c#))))
        ~c)
      (dragan-says-ex "Transforming a SP matrix by another matrix type is not supported. Copy SP to SY."
                      {:a (info ~a) :b (info ~b)})))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SP matrices. No way around that." {:a (info ~a)})))
