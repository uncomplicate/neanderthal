;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.lapack
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [long-ptr byte-ptr pointer]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols]]
             [block :refer [offset stride row?]]]
            [uncomplicate.neanderthal.math :refer [sqrt pow abs]]
            [uncomplicate.neanderthal.internal
             [common :refer [skip]]
             [api :refer [navigator region storage data-accessor]]
             [navigation :refer [diag-unit? lower? upper? dostripe-layout full-storage]]]
            [uncomplicate.neanderthal.internal.cpp.blas
             :refer [full-storage-map blas-layout band-storage-map band-storage-reduce]])
  (:import uncomplicate.neanderthal.internal.navigation.BandStorage))

(defmacro with-lapack-check [rout expr]
  `(let [err# ~expr]
     (if (zero? err#)
       err#
       (throw (ex-info "LAPACK error." {:routine (str ~rout) :error-code err# :bad-argument (- err#)})))))

;; TODO remove
(defmacro vctr-laset [method ptr alpha x]
  `(do
     (with-lapack-check
       (~method ~(:row blas-layout) (byte (int \g)) (dim ~x) 1 ~alpha ~alpha (~ptr ~x) (stride ~x)))
     ~x))

(defmacro vctr-lasrt [method ptr x increasing]
  `(if (= 1 (stride ~x))
     (do
       (with-lapack-check "lasrt"
         (~method (byte (int (if ~increasing \I \D))) (dim ~x) (~ptr ~x)))
       ~x)
     (dragan-says-ex "You cannot sort a vector with stride different than 1." {:stride (stride ~x)})))

(defmacro matrix-lasrt [lapack method ptr a increasing]
  `(let [incr# (byte (int (if ~increasing \I \D)))
         buff-a# (~ptr ~a 0)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check "lasrt" (. ~lapack ~method incr# len# (.position buff-a# idx#))))
     ~a))

(defmacro ge-lan [blas method ptr norm a]
  `(. ~blas ~method (.layout (navigator ~a)) ~(byte (int norm)) (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a)))

;; ========================= TR matrix macros ===============================================

(defmacro tr-lacpy [lapack method copy ptr a b]
  `(if (and (< 0 (dim ~a)) (< 0 (dim ~b)))
     (let [nav# (navigator ~a)]
       (if (= nav# (navigator ~b))
         (let [reg# (region ~a)
               diag-pad# (long (if (diag-unit? reg#) 1 0))
               buff-a# (~ptr ~a (.index (storage ~a) diag-pad# 0))
               buff-b# (~ptr ~b (.index (storage ~b) diag-pad# 0))]
           (with-lapack-check "lacpy"
             (. ~lapack ~method (.layout nav#) (byte (int (if (lower? reg#) \L \U)))
                (- (mrows ~a) diag-pad#) (- (ncols ~a) diag-pad#)
                buff-a# (stride ~a) buff-b# (stride ~b))))
         (let [buff-a# (~ptr ~a 0)
               buff-b# (~ptr ~b 0)]
           (full-storage-map ~a ~b len# buff-a# buff-b# ld-a# (skip "tr-lacpy")
                             (. ~lapack ~copy len# buff-a# ld-a# buff-b# 1))))
       ~b)
     ~b))

;; TODO this was the comment from 2020 or so. Check whether Intel fixed this and update accordingly.
;; There seems to be a bug in MKL's LAPACK_?lantr. If the order is column major,
;; it returns 0.0 as a result. To fix this, I had to do the uplo# trick.
(defmacro tr-lan [lapack method ptr norm a]
  `(if (< 0 (dim ~a))
     (let [reg# (region ~a)
           row-layout# (row? ~a)
           lower# (lower? reg#)
           uplo# (byte (int (if row-layout# (if lower# \L \U) (if lower# \U \L))))
           norm# (byte (int (if row-layout# ~norm (cond (= \I ~norm) \O
                                                        (= \O ~norm) \I
                                                        :default ~norm))))]
       (. ~lapack ~method ~(:row blas-layout) norm# uplo# (byte (int (if (diag-unit? reg#) \U \N)))
          (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a)))
     0.0))

(defmacro uplo-lascl [lapack method ptr alpha a]
  `(if (< 0 (dim ~a))
     (let [reg# (region ~a)
           diag-pad# (long (if (diag-unit? reg#) 1 0))]
       (with-lapack-check "lascl"
         (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (lower? reg#) \L \U)))
            0 0 1.0 ~alpha (- (mrows ~a) diag-pad#) (- (ncols ~a) diag-pad#)
            (~ptr ~a diag-pad#) (stride ~a)))
       ~a)
     ~a))

(defmacro uplo-laset [blas method ptr alpha beta a]
  `(if (< 0 (dim ~a))
     (let [reg# (region ~a)
           nav# (navigator ~a)
           diag-pad# (long (if (diag-unit? reg#) 1 0))
           idx# (if (lower? reg#)
                  (.index nav# (storage ~a) diag-pad# 0)
                  (.index nav# (storage ~a) 0 diag-pad#))]
       (with-lapack-check "laset"
         (. ~blas ~method (.layout nav#) (byte (int (if (.isLower reg#) \L \U)))
            (- (mrows ~a) diag-pad#) (- (ncols ~a) diag-pad#) ~alpha ~beta (~ptr ~a idx#) (stride ~a)))
       ~a)
     ~a))

;; ========================= SY matrix macros ===============================================

(defmacro sy-lacpy [lapack method copy ptr a b]
  `(if (and (< 0 (dim ~a)) (< 0 (dim ~b)))
     (let [nav# (navigator ~a)
           reg-a# (region ~a)
           buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)]
       (if (or (= nav# (navigator ~b)) (not= (.uplo reg-a#) (.uplo (region ~b))))
         (with-lapack-check "lacpy"
           (. ~lapack ~method (.layout nav#) (byte (int (if (lower? reg-a#) \L \U)))
              (mrows ~a) (ncols ~a) buff-a# (stride ~a) buff-b# (stride ~b)))
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a# (skip "tr-lacpy")
                           (. ~lapack ~copy len# buff-a# ld-a# buff-b# 1)))
       ~b)
     ~b))

(defmacro sy-lan [lapack method ptr norm a]
  `(if (< 0 (dim ~a))
     (. ~lapack ~method (.layout (navigator ~a)) ~(byte (int norm))
        (byte (int (if (.isLower (region ~a)) \L \U))) (mrows ~a) (~ptr ~a) (stride ~a))
     0.0))

;; ========================= GB matrix macros ================================================

(defn band-storage-kl ^long [^BandStorage s]
  (.kl s))

(defn band-storage-ku ^long [^BandStorage s]
  (.ku s))

(defmacro gb-lan [typed-accessor lapack langb nrm ptr cpp-ptr norm a]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)
           fd# (.fd stor#)
           ld# (.ld stor#)
           kl# (band-storage-kl stor#)
           ku# (band-storage-ku stor#)
           buff-a# (~ptr ~a 0)
           da# (~typed-accessor ~a)]
       (cond
         (= (mrows ~a) (ncols ~a)) (with-release [work# (~cpp-ptr (.createDataSource da# fd#))
                                                  fd# (long-ptr (pointer fd#))
                                                  ld# (long-ptr (pointer ld#))
                                                  kl# (long-ptr (pointer kl#))
                                                  ku# (long-ptr (pointer ku#))
                                                  norm# (byte-pointer (pointer ~norm))]
                                     (. ~lapack ~langb norm# fd# kl# ku# buff-a# ld# work#))
         (= \F ~norm) (sqrt (band-storage-reduce ~a len# buff-a# acc# 0.0
                                                 (+ acc# (pow (. ~lapack ~nrm len# buff-a# ld#) 2))))
         (= \M ~norm) (band-storage-reduce ~a len# buff-a# amax# 0.0
                                           (let [iamax# (. ~lapack ~nrm len# buff-a# ld#)]
                                             (max amax# (abs (.get da# buff-a# (* ld# iamax#))))))
         :default (dragan-says-ex "This operation has not been implemented for non-square banded matrix.")))
     0.0))

(defmacro sb-lan [typed-accessor lapack lansb ptr cpp-ptr norm a]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)
           reg# (region ~a)]
       (with-release [work# (~cpp-ptr (.createDataSource (~typed-accessor ~a) (.fd stor#)))
                      norm# (byte-pointer (pointer ~norm))
                      uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a))
                                                     (if (.isLower reg#) \L \U)
                                                     (if (.isLower reg#) \U \L))))
                      fd# (long-ptr (pointer (.fd stor#)))
                      ld# (long-ptr (pointer (.ld stor#)))
                      k# (long-ptr (pointer (max (.kl reg#) (.ku reg#))))]
         (. ~lapack ~lansb norm# uplo# fd# k# (~ptr ~a) ld# work#)))
     0.0))

(defmacro tb-lan [typed-accessor lapack lantb ptr cpp-ptr norm a]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)
           reg# (region ~a)]
       (with-release [work# (~cpp-ptr (.createDataSource (~typed-accessor ~a) (.fd stor#)))
                      norm# (byte-pointer (pointer ~norm))
                      uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a))
                                                     (if (.isLower reg#) \L \U)
                                                     (if (.isLower reg#) \U \L))))
                      diag# (byte-pointer (pointer (if (.isDiagUnit reg#) \U \N)))
                      fd# (long-ptr (pointer (.fd stor#)))
                      ld# (long-ptr (pointer (.ld stor#)))
                      k# (long-ptr (pointer (max (.kl reg#) (.ku reg#))))]
         (. ~lapack ~lantb norm# uplo# diag# fd# k# (~ptr ~a) ld# work#)))
     0.0))

(defmacro gb-laset [blas method ptr alpha a]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           ld# (stride ~a)]
       (band-storage-map ~a len# buff-a#
                         (with-lapack-check "laset"
                           (. ~blas ~method ~(:row blas-layout) ~(byte (int \g))
                              len# 1 ~alpha ~alpha buff-a# ld#)))
       ~a)
     ~a))
