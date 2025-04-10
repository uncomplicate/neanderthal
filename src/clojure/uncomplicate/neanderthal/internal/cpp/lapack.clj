;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.lapack
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [with-release info]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.clojure-cpp
             :refer [long-ptr int-ptr byte-pointer int-pointer long-pointer pointer get-entry]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols col trans symmetric?]]
             [block :refer [stride row?]]
             [math :refer [f=]]]
            [uncomplicate.neanderthal.math :refer [sqrt pow abs]]
            [uncomplicate.neanderthal.internal
             [common :refer [skip real-accessor check-stride]]
             [api :refer [factory index-factory engine data-accessor raw mm scal flip
                          create-vector create-ge create-gb create-sb create-sy fits-navigation?
                          nrm1 nrmi copy nrm2 amax trf tri trs con sv navigator region storage
                          view-ge]]
             [navigation :refer [diag-unit? lower? upper? dostripe-layout full-storage]]]
            [uncomplicate.neanderthal.internal.constants
             :refer [blas-layout blas-uplo blas-transpose blas-diag]]
            [uncomplicate.neanderthal.internal.cpp.blas
             :refer [full-storage-map band-storage-map band-storage-reduce]])
  (:import [uncomplicate.neanderthal.internal.api DiagonalMatrix LayoutNavigator Region]
           uncomplicate.neanderthal.internal.navigation.BandStorage))

;; =========================== Auxiliary LAPACK Routines =========================

(defmacro with-lapack-check [rout expr]
  `(let [res# ~expr]
     (if (zero? res#)
       res#
       (throw (ex-info "LAPACK error." {:routine (str ~rout) :error-code res# :bad-argument (- res#)})))))

(defmacro matrix-lasrt [lapack method ptr a increasing]
  `(let [incr# (byte (int (if ~increasing \I \D)))
         buff-a# (~ptr ~a 0)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check "lasrt" (. ~lapack ~method incr# len# (.position buff-a# idx#))))
     ~a))

(defmacro ge-lan [lapack method ptr norm a]
  `(. ~lapack ~method (.layout (navigator ~a)) ~(byte (int norm))
      (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a)))

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

(defmacro sy-lan [lapack method ptr norm a]
  `(if (< 0 (dim ~a))
     (. ~lapack ~method (.layout (navigator ~a)) ~(byte (int norm))
        (byte (int (if (.isLower (region ~a)) \L \U))) (mrows ~a) (~ptr ~a) (stride ~a))
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

(defmacro uplo-laset [lapack method ptr alpha beta a]
  `(if (< 0 (dim ~a))
     (let [reg# (region ~a)
           nav# (navigator ~a)
           diag-pad# (long (if (diag-unit? reg#) 1 0))
           idx# (if (lower? reg#)
                  (.index nav# (storage ~a) diag-pad# 0)
                  (.index nav# (storage ~a) 0 diag-pad#))]
       (with-lapack-check "laset"
         (. ~lapack ~method (.layout nav#) (byte (int (if (.isLower reg#) \L \U)))
            (- (mrows ~a) diag-pad#) (- (ncols ~a) diag-pad#) ~alpha ~beta (~ptr ~a idx#) (stride ~a)))
       ~a)
     ~a))

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
                buff-a# (max 1 (stride ~a)) buff-b# (max 1 (stride ~b)))))
         (let [buff-a# (~ptr ~a 0)
               buff-b# (~ptr ~b 0)]
           (full-storage-map ~a ~b len# buff-a# buff-b# ld-a# (skip "tr-lacpy")
                             (. ~lapack ~copy len# buff-a# ld-a# buff-b# 1))))
       ~b)
     ~b))

(defmacro sy-lacpy [lapack method copy ptr a b]
  `(if (and (< 0 (dim ~a)) (< 0 (dim ~b)))
     (let [nav# (navigator ~a)
           reg-a# (region ~a)
           buff-a# (~ptr ~a 0)
           buff-b# (~ptr ~b 0)]
       (if (or (= nav# (navigator ~b)) (not= (.uplo reg-a#) (.uplo (region ~b))))
         (with-lapack-check "lacpy"
           (. ~lapack ~method (.layout nav#) (byte (int (if (lower? reg-a#) \L \U)))
              (mrows ~a) (ncols ~a) buff-a# (max 1 (stride ~a)) buff-b# (max 1 (stride ~b))))
         (full-storage-map ~a ~b len# buff-a# buff-b# ld-a# (skip "tr-lacpy")
                           (. ~lapack ~copy len# buff-a# (max 1 ld-a#) buff-b# 1)))
       ~b)
     ~b))

;; ========================= Banded matrix macros ================================================

(defn band-storage-kl ^long [^BandStorage s]
  (.kl s))

(defn band-storage-ku ^long [^BandStorage s]
  (.ku s))

(defmacro gb-lan [lapack langb nrm ptr cpp-ptr norm a]
  `(if (< 0 (dim ~a))
     (let [stor# (full-storage ~a)
           fd# (.fd stor#)
           ld# (.ld stor#)
           kl# (band-storage-kl stor#)
           ku# (band-storage-ku stor#)
           buff-a# (~ptr ~a 0)
           da# (real-accessor ~a)]
       (with-release [norm# (byte-pointer (pointer ~norm))
                      work# (~cpp-ptr (.createDataSource da# fd#))
                      fd-ptr# (long-ptr (pointer fd#))
                      ld-ptr# (long-ptr (pointer ld#))
                      kl-ptr# (long-ptr (pointer kl#))
                      ku-ptr# (long-ptr (pointer ku#))]
         (cond
           (= (mrows ~a) (ncols ~a)) (. ~lapack ~langb norm# fd-ptr# kl-ptr# ku-ptr# buff-a# ld-ptr# work#)
           (= \F ~norm) (sqrt (band-storage-reduce ~a len# buff-a# acc# 0.0
                                                   (double (. ~lapack ~langb norm# fd-ptr# kl-ptr#
                                                              ku-ptr# buff-a# ld-ptr# work#))
                                                   (+ acc# (pow (. ~lapack ~nrm len# buff-a# ld#) 2))))
           (= \M ~norm) (band-storage-reduce ~a len# buff-a# amax# 0.0
                                             (let [iamax# (. ~lapack ~nrm (.surface (region ~a)) buff-a# 1)]
                                               (abs (.get da# buff-a# iamax#)))
                                             (let [iamax# (. ~lapack ~nrm len# buff-a# ld#)]
                                               (max amax# (abs (.get da# buff-a# (* ld# iamax#))))))
           :default (dragan-says-ex "This operation has not been implemented for non-square banded matrix."))))
     0.0))

(defmacro sb-lan
  ([lapack lansb ptr cpp-ptr norm a]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a))
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))
                       fd# (long-ptr (pointer (.fd stor#)))
                       ld# (long-ptr (pointer (.ld stor#)))
                       k# (long-ptr (pointer (max (.kl reg#) (.ku reg#))))]
          (. ~lapack ~lansb norm# uplo# fd# k# (~ptr ~a) ld# work#)))
      0.0))
  ([lapack lansb ptr cpp-ptr norm a fortran-strlen]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a))
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))
                       fd# (int-ptr (pointer (int (.fd stor#))))
                       ld# (int-ptr (pointer (int (.ld stor#))))
                       k# (int-ptr (pointer (int (max (.kl reg#) (.ku reg#)))))]
          (. ~lapack ~lansb norm# uplo# fd# k# (~ptr ~a) ld# work# ~fortran-strlen ~fortran-strlen)))
      0.0)))

(defmacro tb-lan
  ([lapack lantb ptr cpp-ptr norm a]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
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
  ([lapack lantb ptr cpp-ptr norm a fortran-strlen]
   `(if (< 0 (dim ~a))
      (let [stor# (full-storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a))
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))
                       diag# (byte-pointer (pointer (if (.isDiagUnit reg#) \U \N)))
                       fd# (int-ptr (pointer (int (.fd stor#))))
                       ld# (int-ptr (pointer (int (.ld stor#))))
                       k# (int-ptr (pointer (max (.kl reg#) (.ku reg#))))]
          (. ~lapack ~lantb norm# uplo# diag# fd# k# (~ptr ~a) ld# work#
             ~fortran-strlen ~fortran-strlen ~fortran-strlen)))
      0.0)))

(defmacro gb-laset [blas method ptr alpha a]
  `(if (< 0 (dim ~a))
     (let [buff-a# (~ptr ~a 0)
           ld# (stride ~a)]
       (band-storage-map ~a len# buff-a#
                         (with-lapack-check "laset"
                           (. ~blas ~method ~(:row blas-layout) ~(byte (int \g))
                              (.surface (region ~a)) 1 ~alpha ~alpha buff-a# 1))
                         (with-lapack-check "laset"
                           (. ~blas ~method ~(:row blas-layout) ~(byte (int \g))
                              len# 1 ~alpha ~alpha buff-a# ld#)))
       ~a)
     ~a))

;; ===================== Packed Matrix ==============================

(defmacro packed-laset [lapack method ptr alpha a]
  `(do
     (when (< 0 (dim ~a))
       (if-not (.isDiagUnit (region ~a))
         (with-lapack-check ""
           (. ~lapack ~method ~(:row blas-layout) ~(byte (int \g))
              (.capacity (storage ~a)) 1 ~alpha ~alpha (~ptr ~a) 1))
         (let [buff-a# (~ptr ~a 0)]
           (dostripe-layout ~a len# idx#
                            (with-lapack-check "laset"
                              (. ~lapack ~method ~(:row blas-layout) ~(byte (int \g))
                                 len# 1 ~alpha ~alpha (.position buff-a# idx#) 1))))))
     ~a))

(defmacro packed-lasrt [lapack method ptr a increasing]
  `(let [increasing# (byte (int (if ~increasing \I \D)))
         n# (ncols ~a)
         buff-a# (~ptr ~a 0)]
     (when (< 0 (dim ~a))
       (dostripe-layout ~a len# idx#
                        (with-lapack-check "lasrt"
                          (. ~lapack ~method increasing# len# (.position buff-a# idx#)))))
     ~a))

(defmacro sp-lan
  ([lapack lansp ptr cpp-ptr norm a]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       fd# (long-ptr (pointer (.fd stor#)))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a)) ;;TODO extract
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))]
          (. ~lapack ~lansp norm# uplo# fd# (~ptr ~a) work#)))
      0.0))
  ([lapack lansp ptr cpp-ptr norm a fortran-strlen]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       fd# (int-ptr (pointer (int (.fd stor#))))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a)) ;;TODO extract
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))]
          (. ~lapack ~lansp norm# uplo# fd# (~ptr ~a) work# ~fortran-strlen ~fortran-strlen)))
      0.0)))

(defmacro tp-lan
  ([lapack lantp ptr cpp-ptr norm a]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       fd# (long-ptr (pointer (.fd stor#)))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a)) ;;TODO extract
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))
                       diag-unit# (byte-pointer (pointer (if (.isDiagUnit reg#) \U \N)))]
          (. ~lapack ~lantp norm# uplo# diag-unit# fd# (~ptr ~a) work#)))
      0.0))
  ([lapack lantp ptr cpp-ptr norm a fortran-strlen]
   `(if (< 0 (dim ~a))
      (let [stor# (storage ~a)
            reg# (region ~a)]
        (with-release [work# (~cpp-ptr (.createDataSource (real-accessor ~a) (.fd stor#)))
                       fd# (int-ptr (pointer (int (.fd stor#))))
                       norm# (byte-pointer (pointer ~norm))
                       uplo# (byte-pointer (pointer (if (.isColumnMajor (navigator ~a)) ;;TODO extract
                                                      (if (.isLower reg#) \L \U)
                                                      (if (.isLower reg#) \U \L))))
                       diag-unit# (byte-pointer (pointer (if (.isDiagUnit reg#) \U \N)))]
          (. ~lapack ~lantp norm# uplo# diag-unit# fd# (~ptr ~a) work#
             ~fortran-strlen ~fortran-strlen ~fortran-strlen)))
      0.0)))

;; ----------------- Diagonal matrix -----------------------------------------------

(defmacro diagonal-laset [lapack method ptr alpha a]
  `(do
     (with-lapack-check "laset"
       (. ~lapack ~method ~(:row blas-layout) ~(byte (int \g)) (.surface (region ~a)) 1
          ~alpha ~alpha (~ptr ~a) 1))
     ~a))

(defmacro diagonal-lasrt [lapack method ptr a increasing]
  `(do
     (with-lapack-check "lasrt"
       (. ~lapack ~method (byte (int (if ~increasing \I \D))) (.surface (region ~a)) (~ptr ~a)))
     ~a))

(defmacro tridiagonal-lan
  ([lapack method ptr norm a]
   `(if (< 0 (dim ~a))
      (let [n# (mrows ~a)
            n1# (if (< 0 n#) (dec n#) 0)
            du# (~ptr ~a n#)
            dl# (if (symmetric? ~a) du# (~ptr ~a (+ n# n1#)))]
        (with-release [norm# (byte-pointer (pointer ~norm))
                       n# (long-ptr (pointer (mrows ~a)))]
          (. ~lapack ~method norm# n# dl# (~ptr ~a) du#)))
      0.0))
  ([lapack method ptr norm a fortran-strlen]
   `(if (< 0 (dim ~a))
      (let [n# (mrows ~a)
            n1# (if (< 0 n#) (dec n#) 0)
            du# (~ptr ~a n#)
            dl# (if (symmetric? ~a) du# (~ptr ~a (+ n# n1#)))]
        (with-release [norm# (byte-pointer (pointer ~norm))
                       n# (int-ptr (pointer (int (mrows ~a))))]
          (. ~lapack ~method norm# n# dl# (~ptr ~a) du# ~fortran-strlen)))
      0.0)))

(defmacro tridiagonal-mv
  ([lapack method ptr cpp-ptr alpha a x beta y]
   `(if (or (= 0.0 ~alpha) (= 1.0 ~alpha) (= -1.0 ~alpha))
      (let [n# (ncols ~a)
            n1# (if (< 0 n#) (dec n#) 0)
            du# (~ptr ~a n#)
            dl# (if (symmetric? ~a) du# (~ptr ~a (+ n# n1#)))]
        (with-release [beta# (~cpp-ptr (.wrapPrim (real-accessor ~a) (if (f= 0.0 ~beta) 0.0 1.0)))
                       trans# (byte-pointer (pointer \N))
                       n# (long-ptr (pointer n#))
                       nrhs# (long-ptr (pointer 1))
                       alpha# (~cpp-ptr (pointer ~alpha))
                       ldx# (long-ptr (pointer (stride ~x)))
                       ldy# (long-ptr (pointer (stride ~y)))]
          (when-not (f= 1.0 ~beta)
            (scal (engine ~y) ~beta ~y))
          (. ~lapack ~method trans# n# nrhs# alpha# dl# (~ptr ~a) du#
             (~ptr ~x) ldx# beta# (~ptr ~y) ldy#)
          ~y))
      (dragan-says-ex "Tridiagonal mv! supports only 0.0, 1.0, or -1.0 for alpha." {:alpha (info ~alpha)})))
  ([a]
   `(dragan-says-ex "In-place mv! is not supported for trigiagonal matrices." {:a (info ~a)})))

(defmacro tridiagonal-mm
  ([lapack method ptr cpp-ptr alpha a b beta c left]
   `(let [nav-b# (navigator ~b)]
      (if (or (= 0.0 ~alpha) (= 1.0 ~alpha) (= -1.0 ~alpha))
        (if (= nav-b# (navigator ~c))
          (let [n# (ncols ~a)
                n1# (if (< 0 n#) (dec n#) 0)
                du# (~ptr ~a n#)
                dl# (if (symmetric? ~a) du# (~ptr ~a (+ n# n1#)))]
            (with-release [beta# (~cpp-ptr (.wrapPrim (real-accessor ~a)
                                                                (if (f= 0.0 ~beta) 0.0 1.0)))
                           n# (long-ptr (pointer n#))
                           alpha# (~cpp-ptr (pointer ~alpha))
                           ldb# (long-ptr (pointer (stride ~b)))
                           ldc# (long-ptr (pointer (stride ~c)))]
              (when-not (f= 1.0 ~beta)
                (scal (engine ~c) ~beta ~c))
              (if ~left
                (with-release [trans# (byte-pointer (pointer \N))
                               nrhs# (long-ptr (pointer (ncols ~b)))]
                  (. ~lapack ~method trans# n# nrhs# alpha# dl# (~ptr ~a) du#
                     (~ptr ~b) ldb# beta# (~ptr ~c) ldc#))
                (let [b-t# (trans ~b)
                      c-t# (trans ~c)]
                  (with-release [trans# (byte-pointer (pointer \T))
                                 nrhs# (long-ptr (pointer (ncols b-t#)))]
                    (. ~lapack ~method trans# n# nrhs# alpha# dl# (~ptr ~a) du#
                       (~ptr ~b) ldb# beta# (~ptr ~c) ldc#))))))
          (dragan-says-ex "Tridiagonal mm! supports only b and c with the same layout." {:b (info ~b) :c (info ~c)}))
        (dragan-says-ex "Tridiagonal mm! supports only 0.0, 1.0, or -1.0 for alpha." {:alpha (info ~alpha)}))
      ~c))
  ([a]
   `(dragan-says-ex "In-place mm! is not supported for GT matrices." {:a (info ~a)})))

(defmacro gd-mm
  ([lapack gd-method scal-method ptr alpha a b left]
   `(let [nav-b# (navigator ~b)
          reg-b# (region ~b)
          stor-b# (full-storage ~b)
          buff-a# (~ptr ~a)
          buff-b# (~ptr ~b 0)
          ld-b# (stride ~b)
          m-a# (mrows ~a)
          m-b# (mrows ~a)
          n-b# (ncols ~b)]
      (if (instance? DiagonalMatrix ~b)
        (with-release [m# (long-ptr (pointer m-a#))
                       n# (long-ptr [1])
                       ldx# (long-ptr (pointer (inc ld-b#)))]
          (if (= 0 (+ (.kl reg-b#) (.ku reg-b#)))
            (. ~lapack ~gd-method m# n# buff-a# buff-b# ldx#)
            (dragan-says-ex "In-place mm! is not supported for GD with GT or PT matrices." {:b (info ~b)}))
          (when-not (f= 1.0 ~alpha)
            (scal (engine ~b) ~alpha ~b)))
        (if ~left
          (with-release [m# (long-ptr (pointer m-a#))
                         n# (long-ptr (pointer n-b#))
                         ldx# (long-ptr (pointer ld-b#))]
            (if (.isColumnMajor nav-b#)
              (do (. ~lapack ~gd-method m# n# buff-a# buff-b# ldx#)
                  (when-not (f= 1.0 ~alpha)
                    (scal (engine ~b) ~alpha ~b)))
              (dotimes [i# m-b#]
                (. ~lapack ~scal-method n-b# (* ~alpha (double (get-entry buff-a# i#)))
                   (.position buff-b# (.index nav-b# stor-b# i# 0)) 1))))
          (mm (engine ~a) ~alpha (trans ~a) (trans ~b) true)))
      ~b))
  ([lapack method ptr alpha a b beta c left]
   `(do
      (if ~left
        (let [nav-b# (navigator ~b)
              nav-c# (navigator ~c)
              stor-b# (storage ~b)
              stor-c# (storage ~c)
              n-a# (ncols ~a)
              buff-a# (~ptr ~a)
              buff-b# (~ptr ~b 0)
              buff-c# (~ptr ~c 0)
              stride-col-b# (if (.isColumnMajor nav-b#) 1 (stride ~b))
              stride-col-c# (if (.isColumnMajor nav-c#) 1 (stride ~c))]
          (dotimes [j# (ncols ~b)]
            (. ~lapack ~method ~(:column blas-layout) ~(:lower blas-uplo) n-a# 0
               ~alpha buff-a# 1 (.position buff-b# (.index nav-b# stor-b# 0 j#)) stride-col-b#
               ~beta (.position buff-c# (.index nav-c# stor-c# 0 j#)) stride-col-c#))
          ~c)
        (mm (engine ~a) ~alpha (trans ~a) (trans ~b) ~beta (trans ~c) true))
      ~c)))

(defmacro gd-mv
  ([lapack method ptr a x]
   `(with-release [m# (long-ptr (pointer (mrows ~a)))
                   n# (long-ptr (pointer 1))
                   ldx# (long-ptr (pointer (stride ~x)))]
      (. ~lapack ~method m# n# (~ptr ~a) (~ptr ~x) ldx#)
      ~x))
  ([lapack method ptr alpha a x beta y]
   `(do
      (. ~lapack ~method ~(:column blas-layout) ~(:lower blas-uplo) (ncols ~a) 0
         ~alpha (~ptr ~a) 1 (~ptr ~x) (stride ~x) ~beta (~ptr ~y) (stride ~y))
      ~y)))

;; =========== Drivers and Computational LAPACK Routines ===========================

;; ------------- Singular Value Decomposition LAPACK -------------------------------

(defmacro with-sv-check [res expr]
  `(let [info# (long ~expr)]
     (cond
       (= 0 info#) ~res
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (dragan-says-ex "The factorization is singular. The solution could not be computed using this method."
                             {:info info#}))))

;; -------------- Solving Systems of Linear Equations ------------------------------

(defmacro ge-laswp [lapack method ptr idx-ptr a ipiv k1 k2]
  `(do
     (check-stride ~ipiv)
     (with-sv-check ~a
       (. ~lapack ~method (.layout (navigator ~a)) (ncols ~a) (~ptr ~a) (stride ~a)
          (int ~k1) (int ~k2) (~idx-ptr ~ipiv) (stride ~ipiv)))))

(defmacro ge-lapm [lapack method ptr idx-ptr a k forward]
  `(do
     (check-stride ~k)
     (with-sv-check ~a
       (. ~lapack ~method (.layout (navigator ~a)) (int (if ~forward 1 0))
          (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a) (~idx-ptr ~k)))))

(defmacro ge-trf [lapack method ptr idx-ptr a ipiv]
  `(with-sv-check (check-stride ~ipiv)
     (. ~lapack ~method (.layout (navigator ~a))
        (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a) (~idx-ptr ~ipiv))))

(defmacro sy-trx
  ([lapack method ptr idx-ptr a ipiv]
   `(let [stor# (full-storage ~a)]
      (check-stride ~ipiv)
      (with-sv-check ~a
        (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (.isLower (region ~a)) \L \U)))
           (ncols ~a) (~ptr ~a) (.ld stor#) (~idx-ptr ~ipiv)))))
  ([lapack method ptr a]
   `(let [stor# (full-storage ~a)]
      (with-sv-check ~a
        (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (.isLower (region ~a)) \L \U)))
           (ncols ~a) (~ptr ~a) (.ld stor#))))))

(defmacro sy-trfx [lapack method ptr a]
  `(let [info# (. ~lapack ~method (.layout (navigator ~a))
                  (byte (int (if (.isLower (region ~a)) \L \U))) (ncols ~a) (~ptr ~a) (stride ~a))]
     (if (<= 0 info#)
       info#
       (throw (ex-info "There has been an illegal argument in the native function call."
                       {:arg-index (- info#)})))))

(defmacro sp-trfx [lapack method ptr a]
  `(let [info# (. ~lapack ~method (.layout (navigator ~a))
                  (byte (int (if (.isLower (region ~a)) \L \U))) (ncols ~a) (~ptr ~a))]
     (if (<= 0 info#)
       info#
       (throw (ex-info "There has been an illegal argument in the native function call."
                       {:arg-index (- info#)})))))

(defn check-gb-submatrix [a]
  (let [reg (region a)
        stor (storage a)]
    (if (.isColumnMajor (navigator a))
      (if (<= (inc (+ (* 2 (.kl reg)) (.ku reg))) (.ld ^BandStorage stor))
        a
        (dragan-says-ex "Banded factorizations require additional kl superdiagonals. Supply a subband."))
      (dragan-says-ex "Banded factorizations are available only for column-major matrices due to a bug in MKL."))))

(defmacro gb-trf [lapack method ptr idx-ptr a ipiv]
  `(let [reg# (region ~a)]
     (check-gb-submatrix ~a)
     (with-sv-check (check-stride ~ipiv)
       (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a) (.kl reg#) (.ku reg#)
          (~ptr ~a (- (.kl reg#))) (stride ~a) (~idx-ptr ~ipiv)))))

(defmacro sb-trf
  ([lapack method ptr a]
   `(let [reg# (region ~a)]
      (with-sv-check ~a
        (. ~lapack ~method ~(:column blas-layout) ~(byte (int \L))
           (ncols ~a) (max (.kl reg#) (.ku reg#)) (~ptr ~a) (stride ~a))))))

(defmacro dt-trf [lapack method ptr a]
  `(with-sv-check ~a
     (let [n# (ncols ~a)
           n1# (if (< 0 n#) (dec n#) 0)
           dl# (~ptr ~a (+ n# n1#))
           du# (~ptr ~a n#)]
       (with-release [n# (long-ptr (pointer (ncols ~a)))
                      info# (long-pointer 1)]
         (. ~lapack ~method n# dl# (~ptr ~a) du# info#)
         (get-entry info# 0)))))

(defmacro gt-trf [lapack method ptr idx-ptr a ipiv]
  `(with-sv-check (check-stride ~ipiv)
     (let [n# (ncols ~a)
           n1# (if (< 0 n#) (dec n#) 0)
           dl# (~ptr ~a (+ n# n1#))
           du# (~ptr ~a n#)
           du2# (~ptr ~a (+ n# n1# n1#))]
       (. ~lapack ~method n# dl# (~ptr ~a) du# du2# (~idx-ptr ~ipiv)))))

(defmacro st-trf [lapack method ptr a]
 `(with-sv-check ~a
    (. ~lapack ~method (ncols ~a) (~ptr ~a) (~ptr ~a (ncols ~a)))))

(defmacro ge-tri [lapack method ptr idx-ptr a ipiv]
  `(let [stor# (full-storage ~a)]
     (check-stride ~ipiv)
     (with-sv-check ~a
       (. ~lapack ~method (.layout (navigator ~a))
          (.sd stor#) (~ptr ~a) (.ld stor#) (~idx-ptr ~ipiv)))))

(defmacro tr-tri [lapack method ptr a]
  `(let [stor# (full-storage ~a)
         reg# (region ~a)]
     (. ~lapack ~method (.layout (navigator ~a))
        (byte (int (if (.isLower reg#) \L \U))) (byte (int (if (.isDiagUnit reg#) \U \N)))
        (.sd stor#) (~ptr ~a) (.ld stor#))
     ~a))

(defmacro tp-tri [lapack method ptr a]
  `(let [stor# (full-storage ~a)
         reg# (region ~a)]
     (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (.isLower reg#) \L \U)))
      (byte (int (if (.isDiagUnit reg#) \U \N))) (ncols ~a) (~ptr ~a))
     ~a))

(defmacro ge-trs [lapack method ptr idx-ptr lu b ipiv]
  `(let [nav-b# (navigator ~b)]
     (check-stride ~ipiv)
     (with-sv-check ~b
       (. ~lapack ~method (.layout nav-b#) (byte (int (if (= nav-b# (navigator ~lu)) \N \T)))
          (mrows ~b) (ncols ~b) (~ptr ~lu) (stride ~lu) (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b)))))

(defmacro tr-trs [lapack method ptr a b]
  `(let [reg-a# (region ~a)
         nav-b# (navigator ~b)]
     (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isLower reg-a#) \L \U)))
        (byte (int (if (= nav-b# (navigator ~a)) \N \T))) (byte (int (if (.isDiagUnit reg-a#) \U \N)))
        (mrows ~b) (ncols ~b) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b))
     ~b))

(defmacro sy-trs
  ([lapack method ptr idx-ptr ldl b ipiv]
   `(let [nav-b# (navigator ~b)]
      (check-stride ~ipiv)
      (if (= nav-b# (navigator ~ldl))
        (with-sv-check ~b
          (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isLower (region ~ldl)) \L \U)))
             (mrows ~b) (ncols ~b) (~ptr ~ldl) (stride ~ldl) (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b)))
        (dragan-says-ex "SY pivoted solver (trs only) requires that both matrices have the same layout."
                        {:ldl (info ~ldl) :b (info ~b)}))))
  ([lapack method ptr gg b]
   `(let [nav-b# (navigator ~b)
          uplo# (byte (int (if (= nav-b# (navigator ~gg))
                             (if (.isLower (region ~gg)) \L \U)
                             (if (.isLower (region ~gg)) \U \L))))]
      (with-sv-check ~b
        (. ~lapack ~method (.layout (navigator ~b)) uplo#
           (mrows ~b) (ncols ~b) (~ptr ~gg) (stride ~gg) (~ptr ~b) (stride ~b))))))

(defmacro gb-trs [lapack method ptr idx-ptr lu b ipiv]
  `(let [nav-b# (navigator ~b)
         reg# (region ~lu)]
     (check-gb-submatrix ~lu)
     (check-stride ~ipiv)
     (with-sv-check ~b
       (. ~lapack ~method (.layout nav-b#) (byte (int (if (= nav-b# (navigator ~lu)) \N \T)))
          (mrows ~b) (.kl reg#) (.ku reg#) (ncols ~b)
          (~ptr ~lu (- (.kl reg#))) (stride ~lu) (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b)))))

(defmacro sb-trs [lapack method ptr a b]
  `(let [nav-b# (navigator ~b)
         reg# (region ~a)]
     (if (.isColumnMajor nav-b#)
       (with-sv-check ~b
         (. ~lapack ~method ~(:column blas-layout) ~(byte (int \L)) (mrows ~b)
            (max (.kl reg#) (.ku reg#)) (ncols ~b) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)))
       (dragan-says-ex "SB solver requires that the right hand matrix have column layout."
                       {:a (info ~a) :b (info ~b)}))))

(defmacro tb-trs [lapack method ptr a b]
  `(let [nav-b# (navigator ~b)
         reg# (region ~a)]
     (with-sv-check ~b
       (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isLower reg#) \L \U)))
        (byte (int (if (= nav-b# (navigator ~a)) \N \T))) (byte (int (if (.isDiagUnit reg#) \U \N)))
        (mrows ~b) (max (.kl reg#) (.ku reg#)) (ncols ~b) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)))))

(defmacro gd-trs [lapack method ptr a b]
  `(let [nav-b# (navigator ~b)
         reg# (region ~a)]
     (with-sv-check ~b
       (. ~lapack ~method ~(:column blas-layout) ~(byte (int \L)) ~(byte (int \N)) ~(byte (int \N))
          (mrows ~b) 0 (ncols ~b) (~ptr ~a) 1 (~ptr ~b) (stride ~b)))))

(defmacro gt-trs [lapack method ptr idx-ptr lu b ipiv]
  `(let [nav-b# (navigator ~b)]
     (check-stride ~ipiv)
     (with-sv-check ~b
       (let [n# (ncols ~lu)
             n1# (if (< 0 n#) (dec n#) 0)
             dl# (~ptr ~lu (+ n# n1#))
             du# (~ptr ~lu n#)
             du2# (~ptr ~lu (+ n# n1# n1#))]
         (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isColumnMajor nav-b#) \N \T)))
            (ncols ~lu) (ncols ~b) dl# (~ptr ~lu) du# du2# (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b))))))

(defmacro dt-trs [lapack method ptr lu b]
  `(with-sv-check ~b
     (let [n# (ncols ~lu)
           n1# (if (< 0 n#) (dec n#) 0)
           dl# (~ptr ~lu (+ n# n1#))
           du# (~ptr ~lu n#)]
       (with-release [trans# (byte-pointer (pointer (if (.isColumnMajor (navigator ~b)) \N \T)))
                      n# (long-ptr (pointer (ncols ~lu)))
                      rhs# (long-ptr (pointer (ncols ~b)))
                      ldb# (long-ptr (pointer (stride ~b)))
                      info# (long-pointer 1)]
         (. ~lapack ~method trans# n# rhs# dl# (~ptr ~lu) du# (~ptr ~b) ldb# info#)
         (get-entry info# 0)))))

(defmacro st-trs [lapack method ptr lu b]
  `(with-sv-check ~b
     (. ~lapack ~method (.layout (navigator ~b)) (ncols ~lu) (ncols ~b)
        (~ptr ~lu) (~ptr ~lu (ncols ~lu)) (~ptr ~b) (stride ~b))))

(defmacro tp-trs [lapack method ptr a b]
  `(let [reg-a# (region ~a)
         nav-b# (navigator ~b)]
     (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isLower reg-a#) \L \U)))
        (byte (int (if (= nav-b# (navigator ~a)) \N \T))) (byte (int (if (.isDiagUnit reg-a#) \U \N)))
        (mrows ~b) (ncols ~b) (~ptr ~a) (~ptr ~b) (stride ~b))
     ~b))

(defmacro sp-trs
  ([lapack method ptr idx-ptr ldl b ipiv]
   `(let [nav-b# (navigator ~b)]
      (check-stride ~ipiv)
      (if (= nav-b# (navigator ~ldl))
        (with-sv-check ~b
          (. ~lapack ~method (.layout nav-b#) (byte (int (if (.isLower (region ~ldl)) \L \U)))
             (mrows ~b) (ncols ~b) (~ptr ~ldl) (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b)))
        (dragan-says-ex "SP pivoted solver (trs only) requires that both matrices have the same layout."
                        {:ldl (info ~ldl) :b (info ~b)}))))
  ([lapack method ptr a b]
   `(let [reg-a# (region ~a)
          nav-b# (navigator ~b)
          nav# (byte (int (if (= nav-b# (navigator ~a))
                            (if (.isLower reg-a#) \L \U)
                            (if (.isLower reg-a#) \U \L))))]
      (. ~lapack ~method (.layout nav-b#) nav# (mrows ~b) (ncols ~b) (~ptr ~a) (~ptr ~b) (stride ~b))
      ~b)))

(defmacro sp-trx
  ([lapack method ptr idx-ptr a ipiv]
   `(do
      (check-stride ~ipiv)
      (with-sv-check ~a
        (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (.isLower (region ~a)) \L \U)))
           (ncols ~a) (~ptr ~a) (~idx-ptr ~ipiv)))))
  ([lapack method ptr a]
   `(do
      (with-sv-check ~a
        (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if (.isLower (region ~a)) \L \U)))
           (ncols ~a) (~ptr ~a))))))

(defmacro ge-sv
  ([lapack method ptr idx-ptr a b pure]
   `(if ~pure
      (with-release [a# (create-ge (factory ~a) (mrows ~a) (ncols ~a) (.isColumnMajor (navigator ~b)) false)]
        (copy (engine ~a) ~a a#)
        (sv (engine ~a) a# ~b false))
      (let [nav-b# (navigator ~b)]
        (if (= (navigator ~a) nav-b#)
          (with-release [ipiv# (create-vector (index-factory ~a) (ncols ~a) false)]
            (check-stride ipiv#)
            (with-sv-check ~b
              (. ~lapack ~method (.layout nav-b#) (mrows ~b) (ncols ~b) (~ptr ~a) (stride ~a)
                 (~idx-ptr ipiv#) (~ptr ~b) (stride ~b))))
          (dragan-says-ex "GE solver requires that both matrices have the same layout."
                          {:a (info ~a) :b (info ~b)}))))))

(defmacro sy-sv
  ([lapack po-method sy-method ptr idx-ptr a b pure]
   `(let [nav-b# (navigator ~b)
          uplo# (byte (int (if (= nav-b# (navigator ~a))
                             (if (.isLower (region ~a)) \L \U)
                             (if (.isLower (region ~a)) \U \L))))]
      (if ~pure
        (with-release [a# (raw ~a)]
          (copy (engine ~a) ~a a#)
          (let [info# (. ~lapack ~po-method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b)
                         (~ptr a#) (stride a#) (~ptr ~b) (stride ~b))]
            (cond
              (= 0 info#) ~b
              (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                          {:arg-index (- info#)}))
              :else (do
                      (copy (engine ~a) ~a a#)
                      (sv (engine ~a) a# ~b false)))))
        (with-release [ipiv# (create-vector (index-factory ~a) (ncols ~a) false)]
          (check-stride ipiv#)
          (with-sv-check ~b
            (. ~lapack ~sy-method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b)
               (~ptr ~a) (stride ~a) (~idx-ptr ipiv#) (~ptr ~b) (stride ~b)))))))
  ([lapack method ptr a b]
   `(let [nav-b# (navigator ~b)
          uplo# (byte (int(if (= nav-b# (navigator ~a))
                            (if (.isLower (region ~a)) \L \U)
                            (if (.isLower (region ~a)) \U \L))))]
      (with-sv-check ~b
        (. ~lapack ~method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b)
           (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b))))))

(defmacro sp-sv
  ([lapack po-method sp-method ptr idx-ptr a b pure]
   `(let [nav-b# (navigator ~b)
          uplo# (byte (int (if (= nav-b# (navigator ~a))
                             (if (.isLower (region ~a)) \L \U)
                             (if (.isLower (region ~a)) \U \L))))]
      (if ~pure
        (with-release [a# (raw ~a)]
          (copy (engine ~a) ~a a#)
          (let [info# (. ~lapack ~po-method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b)
                         (~ptr a#) (~ptr ~b) (stride ~b))]
            (cond
              (= 0 info#) ~b
              (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                          {:arg-index (- info#)}))
              :else (do
                      (copy (engine ~a) ~a a#)
                      (sv (engine ~a) a# ~b false)))))
        (with-release [ipiv# (create-vector (index-factory ~a) (ncols ~a) false)]
          (check-stride ipiv#)
          (with-sv-check ~b
            (. ~lapack ~sp-method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b) (~ptr ~a)
               (~idx-ptr ipiv#) (~ptr ~b) (stride ~b)))))))
  ([lapack method ptr a b]
   `(let [nav-b# (navigator ~b)
          uplo# (byte (int (if (= nav-b# (navigator ~a))
                             (if (.isLower (region ~a)) \L \U)
                             (if (.isLower (region ~a)) \U \L))))]
      (with-sv-check ~b
        (. ~lapack ~method (.layout nav-b#) uplo# (mrows ~b) (ncols ~b) (~ptr ~a) (~ptr ~b) (stride ~b))))))

(defmacro gb-sv
  ([lapack method ptr idx-ptr a b pure]
   `(let [reg# (region ~a)
          nav-a# (navigator ~a)
          nav-b# (navigator ~b)]
      (if ~pure
        (with-release [a# (create-gb (factory ~a) (mrows ~a) (ncols ~a) (.kl reg#) (.ku reg#)
                                     (.isColumnMajor nav-b#) false)]
          (copy (engine ~a) ~a a#)
          (sv (engine ~a) a# ~b false))
        (if (= nav-b# nav-a#)
          (with-release [ipiv# (create-vector (index-factory ~a) (ncols ~a) false)]
            (check-gb-submatrix ~a)
            (check-stride ipiv#)
            (with-sv-check ~b
              (. ~lapack ~method (.layout nav-b#) (mrows ~b) (.kl reg#) (.ku reg#) (ncols ~b)
                 (~ptr ~a (- (.kl reg#))) (stride ~a) (~idx-ptr ipiv#) (~ptr ~b) (stride ~b))))
          (dragan-says-ex "GB solver requires that both matrices have the same layout."
                          {:a (info ~a) :b (info ~b)})))
      ~b)))

(defmacro sb-sv [lapack method ptr a b pure]
  `(let [reg# (region ~a)
         nav-b# (navigator ~b)]
     (if ~pure
       (with-release [a# (raw ~a)]
         (copy (engine ~a) ~a a#)
         (sv (engine ~a) a# ~b false))
       (if (.isColumnMajor nav-b#)
         (with-sv-check ~b
           (. ~lapack ~method ~(:column blas-layout) ~(byte (int \L)) (mrows ~b)
              (max (.kl reg#) (.ku reg#)) (ncols ~b) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b)))
         (dragan-says-ex "SB solver requires that the right hand matrix have column layout."
                         {:a (info ~a) :b (info ~b)})))
     ~b))

(defmacro gt-sv [lapack method ptr a b pure]
  `(if ~pure
     (with-release [a# (raw ~a)]
       (copy (engine ~a) ~a a#)
       (sv (engine a#) a# ~b false))
     (with-sv-check ~b
       (let [n# (ncols ~a)
             n1# (if (< 0 n#) (dec n#) 0)
             dl# (~ptr ~a (+ n# n1#))
             du# (~ptr ~a n#)]
         (. ~lapack ~method (.layout (navigator ~b))
            n# (ncols ~b) dl# (~ptr ~a) du# (~ptr ~b) (stride ~b))))))

(defmacro dt-sv [lapack method ptr a b pure]
  `(if ~pure
     (with-release [a# (raw ~a)]
       (copy (engine ~a) ~a a#)
       (sv (engine a#) a# ~b false))
     (with-sv-check ~b
       (let [n# (ncols ~a)
             n1# (if (< 0 n#) (dec n#) 0)
             dl# (~ptr ~a (+ n# n1#))
             du# (~ptr ~a n#)]
         (with-release [n# (long-ptr (pointer (ncols ~a)))
                        rhs# (long-ptr (pointer (ncols ~b)))
                        ldb# (long-ptr (pointer (stride ~b)))
                        info# (long-pointer 1)]
           (. ~lapack ~method n# rhs# dl# (~ptr ~a) du# (~ptr ~b) ldb# info#)
           (get-entry info# 0))))))

(defmacro st-sv [lapack method ptr a b pure]
  `(if ~pure
     (with-release [a# (raw ~a)]
       (copy (engine ~a) ~a a#)
       (sv (engine a#) a# ~b false))
     (with-sv-check ~b
       (. ~lapack ~method (int (if (.isColumnMajor (navigator ~b)) \N \T))
          (ncols ~a) (ncols ~b) (~ptr ~a) (~ptr ~a (ncols ~a)) (~ptr ~b) (stride ~b)))))

(defmacro ge-con [lapack method ptr cpp-ptr lu nrm nrm1?]
  `(with-release [da# (real-accessor ~lu)
                  res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (get-entry res# 0)
       (. ~lapack ~method (.layout (navigator ~lu)) (byte (int (if ~nrm1? \O \I)))
          (min (mrows ~lu) (ncols ~lu)) (~ptr ~lu) (stride ~lu) ~nrm res#))))

(defmacro tr-con [lapack method ptr cpp-ptr a nrm1?]
  `(with-release [da# (real-accessor ~a)
                  res# (~cpp-ptr (.createDataSource da# 1))
                  reg# (region ~a)]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if ~nrm1? \O \I)))
          (byte (int (if (.isLower reg#) \L \U))) (byte (int (if (.isDiagUnit reg#) \U \N)))
          (ncols ~a) (~ptr ~a) (stride ~a) res#))))

(defmacro sy-con
  ([lapack method ptr cpp-ptr idx-ptr ldl ipiv nrm]
   `(with-release [da# (real-accessor ~ldl)
                   res# (~cpp-ptr (.createDataSource da# 1))]
      (with-sv-check (.get da# res# 0)
        (. ~lapack ~method (.layout (navigator ~ldl)) (byte (int (if (.isLower (region ~ldl)) \L \U)))
           (ncols ~ldl) (~ptr ~ldl) (stride ~ldl) (~idx-ptr ~ipiv) ~nrm res#))))
  ([lapack method ptr cpp-ptr gg nrm]
   `(with-release [da# (real-accessor ~gg)
                   res# (~cpp-ptr (.createDataSource da# 1))]
      (with-sv-check (.get da# res# 0)
        (. ~lapack ~method (.layout (navigator ~gg)) (byte (int (if (.isLower (region ~gg)) \L \U)))
           (ncols ~gg) (~ptr ~gg) (stride ~gg) ~nrm res#)))))

(defmacro gb-con [lapack method ptr cpp-ptr idx-ptr lu ipiv nrm nrm1?]
  `(with-release [da# (real-accessor ~lu)
                  reg# (region ~lu)
                  res# (~cpp-ptr (.createDataSource da# 1))]
     (check-gb-submatrix ~lu)
     (check-stride ~ipiv)
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method (.layout (navigator ~lu)) (byte (int (if ~nrm1? \O \I)))
        (min (mrows ~lu) (ncols ~lu)) (.kl reg#) (.ku reg#)
        (~ptr ~lu (- (.kl reg#))) (stride ~lu) (~idx-ptr ~ipiv) ~nrm res#))))

(defmacro sb-con [lapack method ptr cpp-ptr gg nrm]
  `(let [da# (real-accessor ~gg)
         reg# (region ~gg)
         res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method ~(:column blas-layout) ~(byte (int \L))
          (ncols ~gg) (max (.kl reg#) (.ku reg#)) (~ptr ~gg) (stride ~gg) ~nrm res#))))

(defmacro tb-con [lapack method ptr cpp-ptr a nrm1?]
  `(let [da# (real-accessor ~a)
         reg# (region ~a)
         res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if ~nrm1? \O \I)))
          (byte (int (if (.isLower reg#) \L \U))) (byte (int (if (.isDiagUnit reg#) \U \N)))
          (ncols ~a) (max (.kl reg#) (.ku reg#)) (~ptr ~a) (stride ~a) res#))))

(defmacro tp-con [lapack method ptr cpp-ptr a nrm1?]
  `(with-release [da# (real-accessor ~a)
                  res# (~cpp-ptr (.createDataSource da# 1))
                  reg# (region ~a)]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if ~nrm1? \O \I)))
          (byte (int (if (.isLower reg#) \L \U))) (byte (int (if (.isDiagUnit reg#) \U \N)))
          (ncols ~a) (~ptr ~a) res#))))

(defmacro sp-con
  ([lapack method ptr cpp-ptr idx-ptr ldl ipiv nrm]
   `(with-release [da# (real-accessor ~ldl)
                   res# (~cpp-ptr (.createDataSource da# 1))]
      (with-sv-check (.get da# res# 0)
        (. ~lapack ~method (.layout (navigator ~ldl)) (byte (int (if (.isLower (region ~ldl)) \L \U)))
           (ncols ~ldl) (~ptr ~ldl) (~idx-ptr ~ipiv) ~nrm res#))))
  ([lapack method ptr cpp-ptr gg nrm]
   `(with-release [da# (real-accessor ~gg)
                   res# (~cpp-ptr (.createDataSource da# 1))]
      (with-sv-check (.get da# res# 0)
        (. ~lapack ~method (.layout (navigator ~gg)) (byte (int (if (.isLower (region ~gg)) \L \U)))
           (ncols ~gg) (~ptr ~gg) ~nrm res#)))))

(defmacro gd-con [lapack method ptr cpp-ptr a nrm1?]
  `(let [da# (real-accessor ~a)
         reg# (region ~a)
         res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method  ~(:column blas-layout) (byte (int (if ~nrm1? \O \I)))
          ~(byte (int \L)) ~(byte (int \N)) (ncols ~a) 0 (~ptr ~a) (stride ~a) res#))))

(defmacro gt-con [lapack method ptr cpp-ptr idx-ptr lu ipiv nrm nrm1?]
  `(with-release [da# (real-accessor ~lu)
                  res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (.get da# res# 0)
       (let [n# (ncols ~lu)
             n1# (if (< 0 n#) (dec n#) 0)
             dl# (~ptr ~lu (+ n# n1#))
             du# (~ptr ~lu n#)
             du2# (~ptr ~lu (+ n# n1# n1#))]
         (. ~lapack ~method (byte (int (if ~nrm1? \O \I)))
            n# dl# (~ptr ~lu) du# du2# (~idx-ptr ~ipiv) ~nrm res#)))))

(defmacro st-con [lapack method ptr lu nrm]
  `(with-release [da# (real-accessor ~lu)
                  res# (.createDataSource da# 1)]
     (with-sv-check (.get da# res# 0)
       (. ~lapack ~method  (ncols ~lu) (~ptr ~lu) ~nrm res#))))

;; ------------- Orthogonal Factorization (L, Q, R) LAPACK -------------------------------

(defmacro with-lqr-check [res expr]
  `(let [info# ~expr]
     (if (= 0 info#)
       ~res
       (throw (ex-info "There has been an illegal argument in the native function call."
                       {:arg-index (- info#)})))))

(defmacro ge-lqrf [lapack method ptr a tau]
  `(if (< 0 (dim ~a))
     (do
       (check-stride ~tau)
       (with-lqr-check ~a
         (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a)
            (~ptr ~a) (stride ~a) (~ptr ~tau))))
     ~a))

(defmacro ge-qp3 [lapack method ptr idx-ptr a jpiv tau]
  `(if (< 0 (dim ~a))
     (do
       (check-stride ~jpiv)
       (check-stride ~tau)
       (with-lqr-check ~a
         (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a)
            (~ptr ~a) (stride ~a) ( ~idx-ptr ~jpiv) (~ptr ~tau))))
     ~a))

(defmacro or-glqr [lapack method ptr a tau]
  `(if (< 0 (dim ~a))
     (do
       (check-stride ~tau)
       (with-lqr-check ~a
         (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a) (dim ~tau)
            (~ptr ~a) (stride ~a) (~ptr ~tau))))
     ~a))

(defmacro or-mlqr [lapack method ptr a tau c left]
  `(if (< 0 (dim ~a))
     (do
       (check-stride ~tau)
       (with-lqr-check ~c
         (. ~lapack ~method (.layout (navigator ~c)) (byte (int (if ~left \L \R)))
            (byte (int (if (= (navigator ~a) (navigator ~c)) \N \T))) (mrows ~c) (ncols ~c) (dim ~tau)
            (~ptr ~a) (stride ~a) (~ptr ~tau) (~ptr ~c) (stride ~c))))
     ~a))

;; ------------- Linear Least Squares Routines LAPACK -------------------------------

(defmacro ge-ls [lapack method ptr a b]
  `(let [nav# (navigator ~a)
         info# (. ~lapack ~method (.layout nav#) (byte (int (if (= nav# (navigator ~b)) \N \T)))
                  (mrows ~a) (ncols ~a) (ncols ~b) (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b))]
     (cond
       (= 0 info#) ~b
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (throw (ex-info "The i-th diagonal element of a is zero, so the matrix does not have full rank."
                             {:arg-index info#})))))

(defmacro ge-lse [lapack method ptr a b c d x]
  `(do
     (check-stride ~c)
     (check-stride ~d)
     (check-stride ~x)
     (let [nav# (navigator ~a)
           info# (. ~lapack ~method (.layout nav#) (mrows ~a) (ncols ~a) (mrows ~b)
                    (~ptr ~a) (stride ~a) (~ptr ~b) (stride ~b) (~ptr ~c) (~ptr ~d) (~ptr ~x))]
       (cond
         (= 0 info#) ~x
         (= 1 info#) (throw (ex-info "rank(b) < p; the least squares solution could not be computed."
                                     {:arg-index info#}))
         (= 2 info#) (throw (ex-info "rank(a|b) < n; the least squares solution could not be computed."
                                     {:arg-index info#}))
         (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                     {:arg-index (- info#)}))
         :else (throw (ex-info "This info# should not happen. Please report the issue."
                               {:arg-index info#}))))))

(defmacro ge-gls [lapack method ptr a b d x y]
  `(do
     (check-stride ~d)
     (check-stride ~x)
     (check-stride ~y)
     (let [nav# (navigator ~a)
           info# (. ~lapack ~method (.layout nav#) (mrows ~a) (ncols ~a) (ncols ~b)
                    (~ptr ~a) (stride ~a) (~ptr ~b)  (stride ~b) (~ptr ~d) (~ptr ~x) (~ptr ~y))]
       (cond
         (= 0 info#) ~y
         (= 1 info#) (throw (ex-info "rank(a) < n; the least squares solution could not be computed."
                                     {:arg-index info#}))
         (= 2 info#) (throw (ex-info "rank(ab) < m; the least squares solution could not be computed."
                                     {:arg-index info#}))
         (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                     {:arg-index (- info#)}))
         :else (throw (ex-info "This info# should not happen. Please report the issue."
                               {:arg-index info#}))))))

;; ------------- Non-Symmetric Eigenvalue Problem Routines LAPACK -------------------------------

(defmacro with-eigen-check [res expr]
  `(let [info# ~expr]
     (cond
       (= 0 info#) ~res
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (throw (ex-info "The algorithm failed to compute all the eigenvalues."
                             {:first-converged (inc info#)})))))

(defmacro ge-ev [lapack method ptr a w vl vr]
  `(if (and (.isColumnMajor (navigator ~w)) (= 2 (ncols ~w)) (= (mrows ~a) (mrows ~w)))
     (let [wr# (col ~w 0)
           wi# (col ~w 1)]
       (with-eigen-check ~w
         (. ~lapack ~method (.layout (navigator ~a))
            (byte (int (if (< 0 (dim ~vl)) \V \N))) (byte (int (if (< 0 (dim ~vr)) \V \N)))
            (ncols ~a) (~ptr ~a) (stride ~a) (~ptr wr#) (~ptr wi#) (~ptr ~vl) (stride ~vl)
            (~ptr ~vr) (stride ~vr))))
     (dragan-says-ex "Matrix w is not properly formatted to hold eigenvalues."
                     {:w (info ~w) :errors
                      (cond-into []
                                 (not (.isColumnMajor (navigator  ~w))) "w is not column-major"
                                 (not (= 2 (ncols ~w))) "w does not have 2 columns"
                                 (not (= (mrows ~a) (mrows ~w))) "w does not have equal number of rows with a")})))

(defmacro ge-es [lapack method ptr a w vs]
  `(if (and (.isColumnMajor (navigator ~w)) (= 2 (ncols ~w)) (= (mrows ~a) (mrows ~w)))
     (let [wr# (col ~w 0)
           wi# (col ~w 1)]
       (with-release [sdim# (int-pointer [0])]
         (with-eigen-check ~w
           (. ~lapack ~method (.layout (navigator  ~a)) (byte (int (if (< 0 (dim ~vs)) \V \N)))
              (byte (int \N)) nil  (ncols ~a) (~ptr ~a) (stride ~a)
              sdim# (~ptr wr#) (~ptr wi#) (~ptr ~vs) (stride ~vs)))))
     (dragan-says-ex "Matrix w is not properly formatted to hold eigenvalues."
                     {:w (info ~w) :errors
                      (cond-into []
                                 (not (.isColumnMajor (navigator  ~w))) "w is not column-major"
                                 (not (= 2 (ncols ~w))) "w does not have 2 columns"
                                 (not (= (mrows ~a) (mrows ~w))) "w does not have equal number of rows with a")})))

;; ------------- Symmetric Eigenvalue Problem Routines LAPACK -------------------------------

(defmacro sy-evr [lapack method ptr a w z]
  `(if (or (= 0 (ncols ~z)) (= (mrows ~w) (ncols ~z)))
     (let [jobz# (< 0 (ncols ~z))
           abstol# -1.0
           vl# -1.0
           vu# 0.0
           il# 1
           iu# (mrows ~w)]
       (with-release [dummy# (int-pointer [(max 1 (mrows ~a))])]
         (with-eigen-check ~w
           (. ~lapack ~method (.layout (navigator ~a)) (byte (int (if jobz# \V \N)))
              (byte (int (if (< 0 iu# (mrows ~a)) \I \A))) (byte (int (if (.isLower (region ~a)) \L \U)))
              (ncols ~a) (~ptr ~a) (stride ~a) vl# vu# il# iu# abstol# dummy#
              (~ptr ~w) (~ptr ~z) (stride ~z) dummy#))))
     (dragan-says-ex "Eigenvalue matrix w and eigenvector matrix dimensions do not match."
                     {:mrows-w (mrows ~w) :ncols-z (ncols ~z)})))

(defmacro sy-evd [lapack method ptr a w v]
  `(if (= (mrows ~a) (mrows ~w))
     (let [jobz# (= (ncols ~a) (ncols ~v))
           a# (if (and jobz# (not (identical? (~ptr ~a) (~ptr ~v))))
                (let [ga# (view-ge ~a)] (copy (engine ga#) ga# ~v) ~v)
                ~a)]
       (with-eigen-check ~w
         (. ~lapack ~method (.layout (navigator ~a)) (byte(int (if jobz# \V \N)))
            (byte (int (if (.isLower (region ~a)) \L \U)))
            (ncols ~a) (~ptr a#) (stride a#) (~ptr ~w))))
     (dragan-says-ex "Matrix w of eigenvalues must have the same number of rows as a."
                     {:mrows-w (mrows ~w) :mrows-a (mrows ~a)})))

(defmacro sy-ev [lapack evd-method evr-method ptr a w v]
  `(if (and (= 1 (ncols ~w)) (<= 0 (mrows ~w) (mrows ~a)))
     (if (= (mrows ~w) (mrows ~a))
       (sy-evd ~lapack ~evd-method ~ptr ~a ~w ~v)
       (sy-evr ~lapack ~evr-method ~ptr ~a ~w ~v))
     (dragan-says-ex "Eigenvalue matrix for symmetric matrices must have proper dimensions."
                     {:w (info ~w) :a (info ~a) :errors
                      (cond-into []
                                 (not (= 1 (ncols ~w))) "w does not have 1 column"
                                 (< (mrows ~a) (mrows ~w)) "w has too many rows")})))

;; ------------- Singular Value Decomposition Routines LAPACK -------------------------------

(defmacro with-svd-check [s expr]
  `(let [info# ~expr]
     (cond
       (= 0 info#) ~s
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (throw (ex-info "The reduction to bidiagonal form did not converge"
                             {:non-converged-superdiagonals info#})))))

(defmacro ge-svd
  ([lapack method ptr a sigma u vt superb]
   `(let [m# (mrows ~a)
          n# (ncols ~a)
          mu# (mrows ~u)
          nu# (ncols ~u)
          mvt# (mrows ~vt)
          nvt# (ncols ~vt)]
      (with-svd-check ~sigma
        (. ~lapack ~method (.layout (navigator  ~a))
           (byte (int (cond (= m# mu# nu#) \A
                            (and (= m# mu#) (= (min m# n#) nu#)) \S
                            (and (= 0 mu#) (< 0 nvt#)) \O
                            :default \N)))
           (byte (int (cond (and (= n# mvt# nvt#)) \A
                            (and (= n# nvt#) (= (min m# n#) mvt#)) \S
                            (and (= 0 nvt#) (< 0 mu#)) \O
                            :default \N)))
           m# n# (~ptr ~a) (stride ~a) (~ptr ~sigma)
           (~ptr ~u) (stride ~u) (~ptr ~vt) (stride ~vt) (~ptr ~superb)))))
  ([lapack method ptr a sigma zero-uvt superb]
   `(with-svd-check ~sigma
      (. ~lapack ~method (.layout (navigator  ~a)) ~(byte (int \N)) ~(byte (int \N))
         (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a) (~ptr ~sigma) (~ptr ~zero-uvt) (stride ~zero-uvt)
         (~ptr ~zero-uvt) (stride ~zero-uvt) (~ptr ~superb)))))

(defmacro ge-sdd
  ([lapack method ptr a sigma u vt]
   `(let [m# (mrows ~a)
          n# (ncols ~a)
          mu# (mrows ~u)
          nu# (ncols ~u)
          mvt# (mrows ~vt)
          nvt# (ncols ~vt)]
      (with-svd-check ~sigma
        (. ~lapack ~method (.layout (navigator  ~a))
           (byte (int (cond (and (= m# mu# nu#) (= n# mvt# nvt#)) \A
                            (and (= m# mu#) (= n# nvt#) (= (min m# n#) nu# mvt#)) \S
                            (or (and (= 0 mu#) (<= n# m#) (= n# mvt# nvt#))
                                (and (= 0 nvt#) (< m# n#) (= m# mu# nu#))) \O
                            :default \N)))
           m# n# (~ptr ~a) (stride ~a) (~ptr ~sigma) (~ptr ~u) (stride ~u) (~ptr ~vt) (stride ~vt)))))
  ([lapack method ptr a sigma zero-uvt]
   `(with-svd-check ~sigma
      (. ~lapack ~method (.layout (navigator  ~a)) ~(byte (int \N)) (mrows ~a) (ncols ~a)
         (~ptr ~a) (stride ~a) (~ptr ~sigma) (~ptr ~zero-uvt) (stride ~zero-uvt)
         (~ptr ~zero-uvt) (stride ~zero-uvt)))))
