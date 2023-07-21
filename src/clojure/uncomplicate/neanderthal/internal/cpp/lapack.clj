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
             [core :refer [with-release info]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.clojure-cpp :refer [long-ptr byte-pointer int-pointer pointer get-entry]]
            [uncomplicate.neanderthal
             [core :refer [dim mrows ncols col]]
             [block :refer [offset stride row?]]]
            [uncomplicate.neanderthal.math :refer [sqrt pow abs]]
            [uncomplicate.neanderthal.internal
             [common :refer [skip real-accessor check-stride]]
             [api :refer [factory index-factory engine data-accessor raw mm scal flip
                          create-vector create-ge create-gb create-sb create-sy fits-navigation?
                          nrm1 nrmi copy nrm2 amax trf tri trs con sv navigator region storage
                          view-ge]]
             [navigation :refer [diag-unit? lower? upper? dostripe-layout full-storage]]]
            [uncomplicate.neanderthal.internal.cpp.blas
             :refer [full-storage-map blas-layout band-storage-map band-storage-reduce]])
  (:import uncomplicate.neanderthal.internal.navigation.BandStorage))

(defmacro with-lapack-check [rout expr]
  `(let [err# ~expr]
     (if (zero? err#)
       err#
       (throw (ex-info "LAPACK error." {:routine (str ~rout) :error-code err# :bad-argument (- err#)})))))

(defmacro with-sv-check [res expr]
  `(let [info# ~expr]
     (cond
       (= 0 info#) ~res
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (dragan-says-ex "The factorization is singular. The solution could not be computed using this method."
                             {:info info#}))))

(defmacro matrix-lasrt [lapack method ptr a increasing]
  `(let [incr# (byte (int (if ~increasing \I \D)))
         buff-a# (~ptr ~a 0)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check "lasrt" (. ~lapack ~method incr# len# (.position buff-a# idx#))))
     ~a))

(defmacro ge-lan [lapack method ptr norm a]
  `(. ~lapack ~method (.layout (navigator ~a)) ~(byte (int norm))
      (mrows ~a) (ncols ~a) (~ptr ~a) (stride ~a)))

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

(defmacro ge-tri [lapack method ptr idx-ptr a ipiv]
  `(let [stor# (full-storage ~a)]
     (check-stride ~ipiv)
     (with-sv-check ~a
       (. ~lapack ~method (.layout (navigator ~a))
          (.sd stor#) (~ptr ~a) (.ld stor#) (~idx-ptr ~ipiv)))))

(defmacro ge-trs [lapack method ptr idx-ptr lu b ipiv]
  `(let [nav-b# (navigator ~b)]
     (check-stride ~ipiv)
     (with-sv-check ~b
       (. ~lapack ~method (.layout nav-b#) (byte (int (if (= nav-b# (navigator ~lu)) \N \T)))
          (mrows ~b) (ncols ~b) (~ptr ~lu) (stride ~lu) (~idx-ptr ~ipiv) (~ptr ~b) (stride ~b)))))

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

(defmacro ge-con [lapack method ptr cpp-ptr lu nrm nrm1?]
  `(with-release [da# (real-accessor ~lu)
                  res# (~cpp-ptr (.createDataSource da# 1))]
     (with-sv-check (get-entry res# 0)
       (. ~lapack ~method (.layout (navigator ~lu)) (byte (int (if ~nrm1? \O \I)))
          (min (mrows ~lu) (ncols ~lu)) (~ptr ~lu) (stride ~lu) ~nrm res#))))

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

(defmacro sb-lan [lapack lansb ptr cpp-ptr norm a]
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

(defmacro tb-lan [lapack lantb ptr cpp-ptr norm a]
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

(defmacro sp-lan [lapack lansp ptr cpp-ptr norm a]
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

(defmacro tp-lan [lapack lantp ptr cpp-ptr norm a]
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

;; ------------- Orthogonal Factorization (L, Q, R) LAPACK -------------------------------

(defmacro with-lqr-check [res expr]
  `(let [info# ~expr]
     (if (= 0 info#)
       ~res
       (throw (ex-info "There has been an illegal argument in the native function call."
                       {:arg-index (- info#)})))))

(defmacro ge-lqrf [lapack method ptr a tau]
  `(do
     (check-stride ~tau)
     (with-lqr-check ~a
       (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a)
          (~ptr ~a) (stride ~a) (~ptr ~tau)))))

(defmacro ge-qp3 [lapack method ptr idx-ptr a jpiv tau]
  `(do
     (check-stride ~jpiv)
     (check-stride ~tau)
     (with-lqr-check ~a
       (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a)
          (~ptr ~a) (stride ~a) ( ~idx-ptr ~jpiv) (~ptr ~tau)))))

(defmacro or-glqr [lapack method ptr a tau]
  `(do
     (check-stride ~tau)
     (with-lqr-check ~a
       (. ~lapack ~method (.layout (navigator ~a)) (mrows ~a) (ncols ~a) (dim ~tau)
        (~ptr ~a) (stride ~a) (~ptr ~tau)))))

(defmacro or-mlqr [lapack method ptr a tau c left]
  `(do
     (check-stride ~tau)
     (with-lqr-check ~c
       (. ~lapack ~method (.layout (navigator ~c)) (byte (int (if ~left \L \R)))
          (byte (int (if (= (navigator ~a) (navigator ~c)) \N \T))) (mrows ~c) (ncols ~c) (dim ~tau)
          (~ptr ~a) (stride ~a) (~ptr ~tau) (~ptr ~c) (stride ~c)))))

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
      (. ~lapack ~method (.layout (navigator  ~a)) (int \N) (int \N) (mrows ~a) (ncols ~a)
       (~ptr ~a) (stride ~a) (~ptr ~sigma) (~ptr ~zero-uvt) (stride ~zero-uvt)
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
      (. ~lapack ~method (.layout (navigator  ~a)) (byte (int \N)) (mrows ~a) (ncols ~a)
       (~ptr ~a) (stride ~a) (~ptr ~sigma) (~ptr ~zero-uvt) (stride ~zero-uvt)
       (~ptr ~zero-uvt) (stride ~zero-uvt)))))
