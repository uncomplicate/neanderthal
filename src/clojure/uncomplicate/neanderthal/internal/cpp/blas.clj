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
            [uncomplicate.clojure-cpp :refer [position! pointer] :as cpp]
            [uncomplicate.neanderthal
             [block :refer [buffer offset stride contiguous?]]
             [core :refer [dim entry ncols symmetric?]]]
            [uncomplicate.neanderthal.internal
             [api :refer [iamax engine navigator storage region]]
             [common :refer [check-eq-navigators skip]]
             [navigation :refer [full-storage accu-layout diag-unit?]]])
  (:import [org.bytedeco.mkl.global mkl_rt]
           [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [uncomplicate.neanderthal.internal.api VectorSpace Block RealVector]))

(def ^:const blas-layout
  {:row 101
   :column 102
   :col 102})

(def ^:const blas-uplo
  {:upper 121
   :lower 122
   :up 121
   :low 122})

(def ^:const blas-transpose
  {:no-trans 111
   :trans 112})

(def ^:const blas-side
  {:left 141
   :right 142})

;; TODO move to structures

(defn float-ptr
  (^FloatPointer [^Block x]
   (cpp/float-ptr (pointer (.buffer x))))
  (^FloatPointer [^Block x ^long i]
   (cpp/float-ptr (pointer (.buffer x)) i)))

(defn double-ptr
  (^DoublePointer [^Block x]
   (cpp/double-ptr (pointer (.buffer x))))
  (^DoublePointer [^Block x ^long i]
   (cpp/double-ptr (pointer (.buffer x)) i)))

(defn int-ptr ^IntPointer [^Block x]
  (cpp/int-ptr (pointer (.buffer x))))

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
                        (position! ~buff-a (.index stor-a# start# j#))
                        (position! ~buff-b (.index stor-b# start# j#))
                        ~expr))
               ~acc))))
       (let [~ld-b (.ld stor-b#)]
         (loop [j# 0 ~acc ~init]
           (if (< j# n#)
             (recur (inc j#)
                    (let [start# (.start nav-a# reg-a# j#)
                          ~len (- (.end nav-a# reg-a# j#) start#)]
                      (position! ~buff-a (.index stor-a# start# j#))
                      (position! ~buff-b (.index stor-b# j# start#))
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
                (position! ~buff-a (.index stor-a# start# j#))
                (position! ~buff-b (.index stor-b# start# j#))
                ~expr))))
        (let [~ld-a (.ld stor-a#)]
          (dotimes [j# fd-b#]
            (let [start# (.start nav-b# reg-b# j#)
                  ~len (- (.end nav-b# reg-b# j#) start#)]
              (position! ~buff-a (.index stor-a# j# start#))
              (position! ~buff-b (.index stor-b# start# j#))
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
            (position! ~buff-a (.index stor-a# start# j#))
            (position! ~buff-b (.index stor-b# start# j#))
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
            (position! ~buff-a (.index stor-a# start# j#))
            (position! ~buff-b (.index stor-b# start# j#))
            (position! ~buff-c (.index stor-c# start# j#))
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
      (let [buff# (~ptr ~a 0)]
        (if (contiguous?  ~a)
          (. ~blas ~method (dim ~a) (~ptr ~a) 1)
          (accu-layout ~a len# idx# acc# 0.0
                       (+ acc# (double (. ~blas ~method len# (.position buff# idx#) 1))))))
      0.0))
  ([blas method ptr a ones]
   `(if (< 0 (dim ~a))
      (if (contiguous? ~a)
        (. ~blas ~method (dim ~a) (~ptr ~a) 1 (~ptr ~ones) 0)
        (let [buff# (~ptr ~a 0)
              ones# (~ptr ~ones)]
          (accu-layout ~a len# idx# acc# 0.0
                       (+ acc# (double (. ~blas ~method len# (.position buff# idx#) 1 ones# 0))))))
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
