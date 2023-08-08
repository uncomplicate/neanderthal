;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://openpsource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.blas
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [block :refer [buffer offset stride contiguous?]]
             [core :refer [dim entry mrows ncols symmetric? trans]]
             [math :refer [f= sqr sqrt]]]
            [uncomplicate.neanderthal.internal
             [constants :refer :all]
             [api :refer [iamax engine navigator storage region mm scal]]
             [common :refer [check-eq-navigators skip real-accessor]]
             [navigation :refer [full-storage accu-layout diag-unit?]]]
            [uncomplicate.neanderthal.internal.cpp.common :refer :all])
  (:import org.bytedeco.mkl.global.mkl_rt))

;; TODO Copied from host
(defmacro vector-iopt [opt x entry]
  `(let [cnt# (dim ~x)]
     (if (< 0 cnt#)
       (loop [i# 1 max-idx# 0 opt-val# (~entry ~x 0)]
         (if (< i# cnt#)
           (let [v# (~entry ~x i#)]
             (if (~opt opt-val# v#)
               (recur (inc i#) i# v#)
               (recur (inc i#) max-idx# opt-val#)))
           max-idx#))
       0)))

(defmacro vector-iaopt [opt x entry]
  `(let [cnt# (dim ~x)]
     (if (< 0 cnt#)
       (loop [i# 1 max-idx# 0 opt-val# (Math/abs (~entry ~x 0))]
         (if (< i# cnt#)
           (let [v# (Math/abs (~entry ~x i#))]
             (if (~opt opt-val# v#)
               (recur (inc i#) i# v#)
               (recur (inc i#) max-idx# opt-val#)))
           max-idx#))
       0)))

(defmacro full-storage-reduce [a b len buff-a buff-b ld-b acc init expr-direct expr]
  `(let [nav-a# (navigator ~a)
         reg-a# (region ~a)
         stor-a# (full-storage ~a)
         stor-b# (full-storage ~b)
         n# (.fd stor-a#)]
     (if (or (= nav-a# (navigator ~b)) (and (symmetric? ~a) (not= (.uplo reg-a#) (.uplo (region ~b)))))
       (if (and (contiguous? ~a) (contiguous? ~b))
         ~expr-direct
         (let [~ld-b 1]
           (loop [j# 0 ~acc ~init]
             (if (< j# n#)
               (recur (inc j#)
                      (let [start# (.start nav-a# reg-a# j#)
                            ~len (- (.end nav-a# reg-a# j#) start#)]
                        (.position ~buff-a (.index stor-a# start# j#))
                        (.position ~buff-b (.index stor-b# start# j#))
                        ~expr))
               ~acc))))
       (let [~ld-b (.ld stor-b#)]
         (loop [j# 0 ~acc ~init]
           (if (< j# n#)
             (recur (inc j#)
                    (let [start# (.start nav-a# reg-a# j#)
                          ~len (- (.end nav-a# reg-a# j#) start#)]
                      (.position ~buff-a (.index stor-a# start# j#))
                      (.position ~buff-b (.index stor-b# j# start#))
                      ~expr))
             ~acc))))))

(defmacro full-storage-map
  ([a b len buff-a buff-b ld-a expr-direct expr]
   `(let [nav-b# (navigator ~b)
          reg-b# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-b# (.fd stor-b#)]
      (if (or (= (navigator ~a) nav-b#) (and (symmetric? ~a) (not= (.uplo (region ~a)) (.uplo reg-b#))))
        (if (and (contiguous? ~a) (contiguous? ~b))
          ~expr-direct
          (let [~ld-a 1]
            (dotimes [j# fd-b#]
              (let [start# (.start nav-b# reg-b# j#)
                    ~len (- (.end nav-b# reg-b# j#) start#)]
                (.position ~buff-a (.index stor-a# start# j#))
                (.position ~buff-b (.index stor-b# start# j#))
                ~expr))))
        (let [~ld-a (.ld stor-a#)]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg-b# j#)
                  ~len (- (.end nav-b# reg-b# j#) start#)]
              (.position ~buff-a (.index stor-a# j# start#))
              (.position ~buff-b (.index stor-b# start# j#))
              ~expr)))))))

(defmacro full-matching-map
  ([a b len buff-a buff-b expr-direct expr]
   `(let [nav-a# (navigator ~a)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-a# (.fd stor-a#)]
      (check-eq-navigators ~a ~b)
      (if (and (contiguous? ~a) (contiguous? ~b))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)]
            (.position ~buff-a (.index stor-a# start# j#))
            (.position ~buff-b (.index stor-b# start# j#))
            ~expr)))))
  ([a b c len buff-a buff-b buff-c expr-direct expr]
   `(let [nav-a# (navigator ~a)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          stor-c# (full-storage ~c)
          fd-a# (.fd stor-a#)]
      (check-eq-navigators ~a ~b ~c)
      (if (and (contiguous? ~a) (contiguous? ~b) (contiguous? ~c))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)]
            (.position ~buff-a (.index stor-a# start# j#))
            (.position ~buff-b (.index stor-b# start# j#))
            (.position ~buff-c (.index stor-c# start# j#))
            ~expr))))))

(defmacro matrix-map [blas method ptr a b]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)]
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a#
                           (. ~blas ~method (dim ~a) buff-a# 1 buff-b# 1)
                           (. ~blas ~method len# buff-a# ld-a# buff-b# 1))))
     ~a))

(defmacro ge-dot [blas method ptr a b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)]
       (full-storage-reduce ~a ~b len# buff-a# buff-b# ld-b# acc# 0.0
                            (. ~blas ~method (dim ~a) buff-a# 1 buff-b# 1)
                            (+ acc# (double (. ~blas ~method len# buff-a# 1 buff-b# ld-b#)))))
     0.0))

(defmacro ge-sum
  ([blas method ptr a]
   `(if (< 0 (dim ~a))
      (let [buff-a# (~ptr ~a 0)]
        (if (contiguous?  ~a)
          (. ~blas ~method (.surface (region ~a)) buff-a# 1)
          (accu-layout ~a len# idx# acc# 0.0
                       (+ acc# (double (. ~blas ~method len# (.position buff-a# idx#) 1))))))
      0.0))
  ([blas method ptr a ones]
   `(if (< 0 (dim ~a))
      (if (contiguous? ~a)
        (. ~blas ~method (.surface (region ~a)) (~ptr ~a) 1 (~ptr ~ones) 0)
        (let [buff-a# (~ptr ~a 0)
              ones# (~ptr ~ones)]
          (accu-layout ~a len# idx# acc# 0.0
                       (+ acc# (double (. ~blas ~method len# (.position buff-a# idx#) 1 ones# 0))))))
      0.0)))

;;TODO there's saxpy_batch_strided in new MKL. Explore whether that can simplify our code.
(defmacro matrix-axpy [blas method ptr alpha a b]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)]
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a#
                           (. ~blas ~method (dim ~a) ~alpha buff-a# 1 buff-b# 1)
                           (. ~blas ~method len# ~alpha buff-a# ld-a# buff-b# 1))))
     ~b))

(defmacro matrix-axpby [blas method ptr alpha a beta b]
  `(do
     (when (< 0 (dim ~a))
       (let [buff-a# (~ptr ~a 0)
             buff-b# (~ptr ~b 0)]
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a#
                           (. ~blas ~method (dim ~a) ~alpha buff-a# 1 ~beta buff-b# 1)
                           (. ~blas ~method len# ~alpha buff-a# ld-a# ~beta buff-b# 1))))
     ~b))

;; ========================= TR matrix macros ===============================================

(defmacro tr-dot [blas method ptr a b]
  `(if (diag-unit? (region ~a))
     (+ (ncols ~a) (double (ge-dot ~blas ~method ~ptr ~a ~b)))
     (ge-dot ~blas ~method ~ptr ~a ~b)))

(defmacro tr-sum
  ([blas method ptr a]
   `(if (diag-unit? (region ~a))
      (+ (ncols ~a) (double (ge-sum ~blas ~method ~ptr ~a)))
      (ge-sum ~blas ~method ~ptr ~a)))
  ([blas method ptr a ones]
   `(if (diag-unit? (region ~a))
      (+ (ncols ~a) (double (ge-sum ~blas ~method ~ptr ~a ~ones)))
      (ge-sum ~blas ~method ~ptr ~a ~ones))))

(defmacro tr-sv [blas method ptr a b]
  `(let [nav-a# (navigator ~a)
         nav-b# (navigator ~b)
         reg# (region ~a)
         uplo# (if (= nav-a# nav-b#)
                 (if (.isLower reg#) ~(:lower blas-uplo) ~(:upper blas-uplo))
                 (if (.isLower reg#) ~(:upper blas-uplo) ~(:lower blas-uplo)))]
     (. ~blas ~method (.layout nav-b#) ~(:left blas-side) uplo#
        (if (= nav-a# nav-b#) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
        (.diag reg#) (mrows ~b) (ncols ~b) 1.0 (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b))
     ~b))

(defmacro tb-sv
  ([blas method ptr a b]
   `(let [reg# (region ~a)
          layout# (.layout (navigator ~a))
          nav-b# (navigator ~b)
          stor-b# (storage ~b)
          uplo# (.uplo reg#)
          diag# (.diag reg#)
          m-b# (mrows ~b)
          n-a# (ncols ~a)
          ku-a# (.ku reg#)
          buff-a# (~ptr ~a)
          ld-a# (stride ~a)
          buff-b# (~ptr ~b 0)
          stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (stride ~b))]
      (dotimes [j# (ncols ~b)]
        (. ~blas ~method layout# uplo# ~(:no-trans blas-transpose) diag# n-a# ku-a#
           buff-a# ld-a# (.position buff-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
      ~b)))

(defmacro tp-sv [blas method ptr a b]
  `(let [reg# (region ~a)
         layout# (.layout (navigator ~a))
         nav-b# (navigator ~b)
         stor-b# (storage ~b)
         uplo# (.uplo reg#)
         diag# (.diag reg#)
         n-a# (ncols ~a)
         buff-a# (~ptr ~a)
         buff-b# (~ptr ~b 0)
         stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (stride ~b))]
     (dotimes [j# (ncols ~b)]
       (. ~blas ~method layout# uplo# ~(:no-trans blas-transpose) diag# n-a#
          buff-a# (.position buff-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
     ~b))

;; ========================= SY matrix macros ===============================================

(defmacro sy-dot [blas method ptr a b]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)
           n# (.fd stor#)
           buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)
           uplo-dot# (double (ge-dot ~blas ~method ~ptr ~a ~b))
           uplo-diag# (. ~blas ~method n# buff-a# (inc (.ld stor#)) buff-b# (inc (.ld (full-storage ~b))))]
       (- (* 2.0 (double uplo-dot#)) uplo-diag#))
     0.0))

(defmacro sy-sum
  ([blas method ptr a]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            n# (.fd stor#)]
        (- (* 2.0 (double (ge-sum ~blas ~method ~ptr ~a)))
           (. ~blas ~method n# (~ptr ~a) (inc (.ld stor#)))))
      0.0))
  ([blas method ptr a ones]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            n# (.fd stor#)]
        (- (* 2.0 (double (ge-sum ~blas ~method ~ptr ~a ~ones)))
           (. ~blas ~method n# (~ptr ~a) (inc (.ld stor#)) (~ptr ~ones) 0)))
      0.0)))

;; ========================== Banded matrix macros =========================================

(defmacro band-storage-map
  ([a len buff-a expr-direct expr]
   `(if (contiguous? ~a)
      ~expr-direct
      (let [reg# (region ~a)
            nav# (navigator ~a)
            stor# (full-storage ~a)
            m# (mrows ~a)
            n# (ncols ~a)
            kl# (.kl reg#)
            ku# (.ku reg#)]
        (dotimes [k# (inc kl#)]
          (let [~len (min (- m# k#) n#)]
            (.position ~buff-a (.index nav# stor# k# 0))
            ~expr))
        (dotimes [k# ku#]
          (let [~len (min m# (- n# (inc k#)))]
            (.position ~buff-a (.index nav# stor# 0 (inc k#)))
            ~expr))
        ~a)))
  ([a b len buff-a buff-b expr-direct expr]
   `(let [reg# (region ~a)
          nav-a# (navigator ~a)
          nav-b# (if (symmetric? ~a) nav-a# (navigator ~b))
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          m# (mrows ~a)
          n# (ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)]
      (if (and (contiguous? ~a) (contiguous? ~b)
               (or (= nav-a# nav-b#) (and (symmetric? ~a) (not= (.uplo (region ~a)) (.uplo (region ~b))))))
        ~expr-direct
        (do
          (dotimes [k# (inc kl#)]
            (let [~len (min (- m# k#) n#)]
              (.position ~buff-a (.index nav-a# stor-a# k# 0))
              (.position ~buff-b (.index nav-b# stor-b# k# 0))
              ~expr))
          (dotimes [k# ku#]
            (let [~len (min m# (- n# (inc k#)))]
              (.position ~buff-a (.index nav-a# stor-a# 0 (inc k#)))
              (.position ~buff-b (.index nav-b# stor-b# 0 (inc k#)))
              ~expr))))
      ~b)))

(defmacro band-storage-reduce
  ([a len buff-a acc init expr-direct expr]
   `(if (and (contiguous? ~a) (not (symmetric? ~a)))
      ~expr-direct
      (let [reg# (region ~a)
            nav# (navigator ~a)
            stor# (full-storage ~a)
            m# (mrows ~a)
            n# (ncols ~a)
            kl# (.kl reg#)
            ku# (.ku reg#)
            coeff# (double (if (symmetric? ~a) 2.0 1.0))]
        (let [acc# (loop [k# 1 ~acc ~init]
                     (if (< k# (inc ku#))
                       (recur (inc k#)
                              (let [~len (min m# (- n# k#))]
                                (.position ~buff-a (.index nav# stor# 0 k#))
                                ~expr))
                       ~acc))]
          (let [~acc (* coeff# (double
                                (loop [k# 1 ~acc (double acc#)]
                                  (if (< k# (inc kl#))
                                    (recur (inc k#)
                                           (let [~len (min (- m# k#) n#)]
                                             (.position ~buff-a (.index nav# stor# k# 0))
                                             ~expr))
                                    ~acc))))
                ~len (min m# n#)]
            (.position ~buff-a (.index nav# stor# 0 0))
            ~expr)))))
  ([a b len buff-a buff-b acc init expr-direct expr]
   `(let [reg# (region ~a)
          nav-a# (navigator ~a)
          nav-b# (if (symmetric? ~a) nav-a# (navigator ~b))
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          m# (mrows ~a)
          n# (ncols ~a)
          kl# (.kl reg#)
          ku# (.ku reg#)
          coeff# (double (if (symmetric? ~a) 2.0 1.0))]
      (if (and (contiguous? ~a) (contiguous? ~b) (not (symmetric? ~a)) (not (symmetric? ~b)))
        ~expr-direct
        (let [acc# (loop [k# 1 ~acc ~init]
                     (if (< k# (inc ku#))
                       (recur (inc k#)
                              (let [~len (min m# (- n# k#))]
                                (.position ~buff-a (.index nav-a# stor-a# 0 k#))
                                (.position ~buff-b (.index nav-b# stor-b# 0 k#))
                                ~expr))
                       ~acc))]
          (let [~acc (* coeff# (double
                                (loop [k# 1 ~acc (double acc#)]
                                  (if (< k# (inc kl#))
                                    (recur (inc k#)
                                           (let [~len (min (- m# k#) n#)]
                                             (.position ~buff-a (.index nav-a# stor-a# k# 0))
                                             (.position ~buff-b (.index nav-b# stor-b# k# 0))
                                             ~expr))
                                    ~acc))))
                ~len (min m# n#)]
            (.position ~buff-a (.index nav-a# stor-a# 0 0))
            (.position ~buff-b (.index nav-b# stor-b# 0 0))
            ~expr))))))

(defmacro gb-map [blas method ptr a b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)
           ld-a# (stride ~a)
           ld-b# (stride ~b)]
       (band-storage-map ~a ~b len# buff-a# buff-b#
                         (. ~blas ~method (.surface (region ~a)) buff-a# ld-a# buff-b# ld-b#)
                         (. ~blas ~method len# buff-a# ld-a# buff-b# ld-b#)))
     ~b))

(defmacro gb-dot [blas method ptr a b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)
           ld-a# (stride ~a)
           ld-b# (stride ~b)]
       (band-storage-reduce ~a ~b len# buff-a# buff-b# acc# 0.0
                            (. ~blas ~method (.surface (region ~a)) buff-a# ld-a# buff-b# ld-b#)
                            (+ acc# (. ~blas ~method len# buff-a# ld-a# buff-b# ld-b#))))
     0.0))

(defmacro gb-scal [blas method ptr alpha a]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           ld-a# (stride ~a)]
       (band-storage-map ~a len# buff-a#
                         (. ~blas ~method (.surface (region ~a)) ~alpha buff-a# ld-a#)
                         (. ~blas ~method len# ~alpha buff-a# ld-a#)))
     ~a))

(defmacro gb-sum
  ([blas method ptr a]
   `(if (< 0 (dim ~a))
      (let [buff-a# (~ptr ~a 0)
            ld-a# (stride ~a)]
        (band-storage-reduce ~a len# buff-a# acc# 0.0
                             (. ~blas ~method (.surface (region ~a)) buff-a# ld-a#)
                             (+ acc# (. ~blas ~method len# buff-a# ld-a#))))
      0.0))
  ([blas method ptr a ones]
   `(if (< 0 (dim ~a))
      (let [buff-a# (~ptr ~a 0)
            ld-a# (stride ~a)
            ones# (~ptr ~ones)]
        (band-storage-reduce ~a len# buff-a# acc# 0.0
                             (. ~blas ~method (.surface (region ~a)) buff-a# 1 ones# 0)
                             (double (+ acc# (. ~blas ~method len# buff-a# ld-a# ones# 0)))))
      0.0)))

(defmacro gb-axpy [blas method ptr alpha a b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)
           ld-a# (stride ~a)
           ld-b# (stride ~b)]
       (band-storage-map ~a ~b len# buff-a# buff-b#
                         (. ~blas ~method (.surface (region ~a)) ~alpha buff-a# ld-a# buff-b# ld-b#)
                         (. ~blas ~method len# ~alpha buff-a# ld-a# buff-b# ld-b#)))
     ~b))

(defmacro gb-mv
  ([blas method ptr alpha a x beta y]
   `(let [reg# (region ~a)]
      (. ~blas ~method (.layout (navigator ~a)) ~(:no-trans blas-transpose) (mrows ~a) (ncols ~a)
         (.kl reg#) (.ku reg#) ~alpha (~ptr ~a) (stride ~a) (~ptr ~x) (stride ~x)
         ~beta (~ptr ~y) (stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for GB matrices." {:a (info ~a)})))

(defmacro sb-mv
  ([blas method ptr alpha a x beta y]
   `(let [reg# (region ~a)]
      (. ~blas ~method ~(:column blas-layout) ~(:lower blas-uplo) (ncols ~a) (max (.kl reg#) (.ku reg#))
         ~alpha (~ptr ~a) (stride ~a) (~ptr ~x) (stride ~x) ~beta (~ptr ~y) (stride ~y))
      ~y))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for SB matrices." {:a (info ~a)})))

(defmacro tb-mv
  ([blas method ptr a x]
   `(let [reg# (region ~a)]
      (. ~blas ~method ~(:column blas-layout) ~(:lower blas-uplo)
         (if (.isLower reg#) ~(:no-trans blas-transpose) ~(:trans blas-transpose)) (.diag reg#)
         (ncols ~a) (max (.kl reg#) (.ku reg#)) (~ptr ~a) (stride ~a) (~ptr ~x) (stride ~x))
      ~x))
  ([a]
   `(dragan-says-ex "Out-of-place mv! is not supported for TB matrices." {:a (info ~a)})))

(defmacro gb-mm
  ([blas method ptr alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (full-storage ~b)
              stor-c# (full-storage ~c)
              m-a# (mrows ~a)
              n-a# (ncols ~a)
              kl-a# (.kl reg#)
              ku-a# (.ku reg#)
              buff-a# (~ptr ~a)
              ld-a# (stride ~a)
              buff-b# (~ptr ~b 0)
              buff-c# (~ptr ~c 0)
              stride-col-b# (if (.isColumnMajor (navigator ~b)) 1 (.ld stor-b#))
              stride-col-c# (if (.isColumnMajor (navigator ~c)) 1 (.ld stor-c#))]
          (dotimes [j# (ncols ~b)]
            (.position buff-b# (.index nav-b# stor-b# 0 j#))
            (.position buff-c# (.index nav-c# stor-c# 0 j#))
            (. ~blas ~method layout# ~(:no-trans blas-transpose) m-a# n-a# kl-a# ku-a#
               ~alpha buff-a# ld-a# buff-b# stride-col-b# ~beta buff-c# stride-col-c#)))
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) ~beta (trans ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by GB matrices. Copy GB to GE." {:a (info ~a)})))

(defmacro sb-mm
  ([blas method ptr alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (storage ~b)
              stor-c# (storage ~c)
              uplo# (.uplo reg#)
              n-a# (ncols ~a)
              k-a# (max (.kl reg#) (.ku reg#))
              m-b# (mrows ~b)
              n-b# (ncols ~b)
              buff-a# (~ptr ~a)
              ld-a# (stride ~a)
              buff-b# (~ptr ~b 0)
              buff-c# (~ptr ~c 0)
              stride-x# (if (.isColumnMajor (navigator ~b))
                          (if ~left 1 (stride ~b))
                          (if ~left (stride ~b) 1))
              stride-col-c# (if (.isColumnMajor nav-c#) 1 (stride ~c))]
          (dotimes [j# n-b#]
            (.position buff-b# (.index nav-b# stor-b# 0 j#))
            (.position buff-c# (.index nav-c# stor-c# 0 j#))
            (. ~blas ~method ~(:column blas-layout) ~(:lower blas-uplo) n-a# k-a#
               ~alpha buff-a# ld-a# buff-b# stride-x# ~beta buff-c# stride-col-c#)))
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) ~beta (trans ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SB matrices. Copy SB to GE." {:a (info ~a)})))

(defmacro tb-mm
  ([blas method ptr alpha a b left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              stor-b# (storage ~b)
              uplo# (.uplo reg#)
              diag# (.diag reg#)
              n-a# (ncols ~a)
              k-a# (max (.kl reg#) (.ku reg#))
              m-b# (mrows ~b)
              n-b# (ncols ~b)
              buff-a# (~ptr ~a)
              ld-a# (stride ~a)
              buff-b# (~ptr ~b 0)
              stride-x# (if (.isColumnMajor (navigator ~b)) (if ~left 1 (stride ~b)) (if ~left (stride ~b) 1))
              transpose# (if (.isLower reg#) ~(:no-trans blas-transpose) ~(:trans blas-transpose))]
          (dotimes [j# n-b#]
            (.position buff-b# (.index nav-b# stor-b# 0 j#))
            (. ~blas ~method ~(:column blas-layout) ~(:lower blas-uplo) transpose# diag# n-a# k-a#
               buff-a# ld-a# buff-b# stride-x#))
          (when-not (f= 1.0 ~alpha)
            (scal (engine ~b) ~alpha ~b)))
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) true))
      ~b))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TB matrices. Copy TB to GE." {:a (info ~a)})))

(defmacro gb-axpby [blas method ptr alpha a beta b]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)
           ld-a# (stride ~a)
           ld-b# (stride ~b)]
       (band-storage-map ~a ~b len# buff-a# buff-b#
                         (. ~blas ~method (.surface (region ~a)) ~alpha buff-a# ld-a# ~beta buff-b# ld-b#)
                         (.~blas ~method len# ~alpha buff-a# ld-a# ~beta buff-b# ld-b#)))
     ~b))

;; ===================== Packed Matrix ==============================

(defmacro packed-map
  ([blas method ptr a]
   `(do
      (. ~blas ~method (.surface (region ~a)) (~ptr ~a) 1)
      ~a))
  ([blas method ptr a b]
   `(do
      (check-eq-navigators ~a ~b)
      (. ~blas ~method (.surface (region ~a)) (~ptr ~a) 1 (~ptr ~b) 1)
      ~b)))

(defmacro packed-scal [blas method ptr alpha a]
  `(do
     (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) 1)
     ~a))

(defmacro packed-axpy [blas method ptr alpha a b]
  `(do
     (check-eq-navigators ~a ~b)
     (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) 1 (~ptr ~b) 1)
     ~b))

(defmacro packed-axpby [blas method ptr alpha a beta b]
  `(do
     (check-eq-navigators ~a ~b)
     (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) 1 ~beta (~ptr ~b) 1)
     ~b))

(defmacro tp-dot [blas method ptr a b]
  `(do
     (check-eq-navigators ~a ~b)
     (if (< 0 (dim ~a))
       (let [da# (real-accessor ~a)
             n# (ncols ~a)
             buff-a# (~ptr ~a)
             buff-b# (~ptr ~b)
             stor-a# (storage ~a)
             stor-b# (storage ~b)
             dot# (. ~blas ~method (.surface (region ~a)) buff-a# 1 buff-b# 1)]
         (if-not (.isDiagUnit (region ~a))
           dot#
           (loop [i# 0 acc# (double (+ n# dot#))]
             (if (< i# n#)
               (recur (inc i#) (- acc# (* (.get da# buff-a# (.index stor-a# i# i#))
                                          (.get da# buff-b# (.index stor-b# i# i#)))))
               acc#))))
       0.0)))

(defmacro sp-dot [blas method ptr a b]
  `(do
     (check-eq-navigators ~a ~b)
     (if (< 0 (dim ~a))
       (let [da# (real-accessor ~a)
             n# (ncols ~a)
             buff-a# (~ptr ~a)
             buff-b# (~ptr ~b)
             stor-a# (storage ~a)
             stor-b# (storage ~b)
             dot# (. ~blas ~method (.surface (region ~a)) buff-a# 1 buff-b# 1)]
         (loop [i# 0 acc# (double (* 2.0 dot#))]
           (if (< i# n#)
             (recur (inc i#) (- acc# (* (.get da# buff-a# (.index stor-a# i# i#))
                                        (.get da# buff-b# (.index stor-b# i# i#)))))
             acc#)))
       0.0)))

(defmacro tp-mm
  ([blas method ptr alpha a b left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              stor-b# (storage ~b)
              uplo# (.uplo reg#)
              diag# (.diag reg#)
              n-a# (ncols ~a)
              buff-a# (~ptr ~a)
              buff-b# (~ptr ~b 0)
              stride-col-b# (if (.isColumnMajor nav-b#) 1 (stride ~b))]
          (dotimes [j# (ncols ~b)]
            (. ~blas ~method layout# uplo# ~(:no-trans blas-transpose) diag# n-a#
               buff-a# (.position buff-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#))
          (when-not (f= 1.0 ~alpha)
            (scal (engine ~b) ~alpha ~b)))
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) true))
      ~b))
  ([a]
   `(dragan-says-ex "Out-of-place mm! is not supported by TP matrices. Copy to GE." {:a (info ~a)})))

(defmacro sp-mm
  ([blas method ptr alpha a b beta c left]
   `(do
      (if ~left
        (let [reg# (region ~a)
              layout# (.layout (navigator ~a))
              nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (storage ~b)
              stor-c# (storage ~c)
              uplo# (.uplo reg#)
              n-a# (ncols ~a)
              buff-a# (~ptr ~a)
              buff-b# (~ptr ~b 0)
              buff-c# (~ptr ~c 0)
              stride-col-b# (if (.isColumnMajor nav-b#) 1 (stride ~b))
              stride-col-c# (if (.isColumnMajor nav-c#) 1 (stride ~b))]
          (dotimes [j# (ncols ~b)]
            (. ~blas ~method layout# uplo# n-a#
               ~alpha buff-a# (.position buff-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#
               ~beta (.position buff-c# (.index nav-c# stor-c# 0 j#)) stride-col-c#)))
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) ~beta (trans ~c) true))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported by SP matrices. Copy to GE." {:a (info ~a)})))

(defmacro tp-sum
  ([blas method ptr etype a]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            da# (real-accessor ~a)
            n# (ncols ~a)
            buff-a# (~ptr ~a)]
        (if-not (.isDiagUnit (region ~a))
          (. ~blas ~method (.surface (region ~a)) buff-a# 1)
          (loop [i# 0 acc# (+ n# (. ~blas ~method (+ n# (.surface (region ~a))) buff-a# 1))]
            (if (< i# n#)
              (recur (inc i#) (- acc# (~etype (.get da# buff-a# (.index stor# i# i#)))))
              acc#))))
      0.0))
  ([blas method ptr etype a ones]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            da# (real-accessor ~a)
            n# (ncols ~a)
            buff-a# (~ptr ~a)
            ones# (~ptr ~ones)]
        (if-not (.isDiagUnit (region ~a))
          (. ~blas ~method (.surface (region ~a)) buff-a# 1 ones# 0)
          (loop [i# 0 acc# (+ n# (~etype (. ~blas ~method (+ n# (.surface (region ~a)))
                                            buff-a# 1 ones# 0)))]
            (if (< i# n#)
              (recur (inc i#) (- acc# (~etype (.get da# buff-a# (.index stor# i# i#)))))
              acc#))))
      0.0)))

(defmacro sp-sum
  ([blas method ptr etype a]
   `(if (< 0 (dim ~a))
      (let [da# (real-accessor ~a)
            n# (ncols ~a)
            buff-a# (~ptr ~a)
            stor# (storage ~a)]
        (loop [i# 0 acc# (* 2.0 (. ~blas ~method (.surface (region ~a)) buff-a# 1))]
          (if (< i# n#)
            (recur (inc i#) (- acc# (~etype (.get da# buff-a# (.index stor# i# i#)))))
            acc#)))
      0.0))
  ([blas method ptr etype a ones]
   `(if (< 0 (dim ~a))
      (let [da# (real-accessor ~a)
            n# (ncols ~a)
            buff-a# (~ptr ~a)
            stor# (storage ~a)]
        (loop [i# 0 acc# (* 2.0 (~etype (. ~blas ~method (.surface (region ~a)) buff-a# 1 (~ptr ~ones) 0)))]
          (if (< i# n#)
            (recur (inc i#) (- acc# (~etype (.get da# buff-a# (.index stor# i# i#)))))
            acc#)))
      0.0)))

;; ===================== Tridiagonal matrix ==================================================

(defmacro diagonal-method
  ([blas method ptr a]
   `(. ~blas ~method (.surface (region ~a)) (~ptr ~a) (stride ~a)))
  ([blas method ptr a b]
   `(. ~blas ~method (.surface (region ~a)) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)))
  ([blas method ptr a b c]
   `(do
      (. ~blas ~method (.surface (region ~a))
         (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) (~ptr ~c) (stride ~c))
      ~c)))

(defmacro diagonal-scal [blas method ptr alpha a]
  `(do
     (when (< 0 (dim ~a))
       (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) (stride ~a)))
     ~a))

(defmacro diagonal-axpy [blas method ptr alpha a b]
  `(do
     (when (< 0 (dim ~a))
       (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)))
     ~b))

(defmacro diagonal-axpby [blas method ptr alpha a beta b]
  `(do
     (when (< 0 (dim ~a))
       (. ~blas ~method (.surface (region ~a)) ~alpha (~ptr ~a) (stride ~a) ~beta (~ptr ~b) (stride ~b)))
     ~b))

(defmacro diagonal-amax [blas method ptr a]
  `(if (< 0 (dim ~a))
     (let [da# (real-accessor ~a)
           buff-a# (~ptr ~a)]
       (Math/abs (.get da# buff-a# (. ~blas ~method (.surface (region ~a)) buff-a# (stride ~a)))))
     0.0))

(defmacro st-dot [blas method ptr a b]
  `(let [n# (ncols ~a)
         buff-a# (~ptr ~a 0)
         buff-b# (~ptr ~b 0)]
     (+ (. ~blas ~method n# buff-a# (stride ~a) buff-b# (stride ~b))
        (* 2.0 (. ~blas ~method (dec n#) (.position buff-a# n#) (stride ~a)
                  (.position buff-b# n#) (stride ~b))))))

(defmacro st-asum [blas method ptr a]
  `(let [n# (ncols ~a)
         buff-a# (~ptr ~a 0)]
     (+ (. ~blas ~method n# buff-a# (stride ~a))
        (* 2.0 (. ~blas ~method (dec n#) (.position buff-a# n#) (stride ~a))))))

(defmacro st-sum [blas method ptr a ones]
  `(let [n# (ncols ~a)
         buff-a# (~ptr ~a 0)
         ones# (~ptr ~ones)]
     (+ (. ~blas ~method n# buff-a# (stride ~a) ones# 0)
        (* 2.0 (. ~blas ~method (dec n#) (.position buff-a# n#) (stride ~a) ones# 0)))))

(defmacro st-nrm2 [blas method ptr a]
  `(let [n# (ncols ~a)
         buff-a# (~ptr ~a 0)]
     (sqrt (+ (sqr (. ~blas ~method n# buff-a# (stride ~a)))
              (* 2.0 (sqr (. ~blas ~method (dec n#) (.position buff-a# n#) (stride ~a)) ))))))
