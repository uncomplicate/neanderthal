;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.lapack
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols]]
             [block :refer [offset stride row?]]]
            [uncomplicate.neanderthal.internal
             [common :refer [skip]]
             [api :refer [navigator region storage]]
             [navigation :refer [diag-unit? lower? upper? dostripe-layout]]]
            [uncomplicate.neanderthal.internal.cpp.blas :refer [full-storage-map blas-layout]]))

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
       (with-lapack-check "method"
         (~method (byte (int (if ~increasing \I \D))) (dim ~x) (~ptr ~x)))
       ~x)
     (dragan-says-ex "You cannot sort a vector with stride different than 1." {:stride (stride ~x)})))

(defmacro matrix-lasrt [lapack method ptr a increasing]
  `(let [incr# (byte (int (if ~increasing \I \D)))
         buff# (~ptr ~a 0)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check "method" (. ~lapack ~method incr# len# (.position buff# idx#))))
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
     (. ~lapack ~method (.layout (navigator ~a)) ~(byte (int norm)) (byte (int (if (.isLower (region ~a)) \L \U)))
        (mrows ~a) (~ptr ~a) (stride ~a))
     0.0))
