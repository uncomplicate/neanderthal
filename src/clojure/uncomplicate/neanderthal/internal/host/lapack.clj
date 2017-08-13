;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.lapack
  (:require [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.neanderthal
             [block :refer [buffer offset stride]]
             [math :refer [pow sqrt]]]
            [uncomplicate.neanderthal.internal
             [api :refer [factory index-factory engine data-accessor
                          create-vector create-ge create-sy fits-navigation? nrm1 nrmi trf tri trs con sv
                          copy nrm2 amax navigator region storage]]
             [common :refer [real-accessor]]
             [navigation :refer [dostripe-layout full-storage]]]
            [uncomplicate.neanderthal.internal.host.cblas
             :refer [band-storage-reduce band-storage-map full-storage-map]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS]
           [uncomplicate.neanderthal.internal.api Region LayoutNavigator])) ;;TODO clean up

(defmacro with-lapack-check [expr]
  ` (let [err# ~expr]
      (if (zero? err#)
        err#
        (throw (ex-info "LAPACK error." {:error-code err# :bad-argument (- err#)})))))

;; =========================== Auxiliary LAPACK Routines =========================

;; ----------------- Common vector macros and functions -----------------------

(defmacro vctr-laset [method alpha x]
  `(do
     (with-lapack-check
       (~method CBLAS/ORDER_ROW_MAJOR (int \g) (.dim ~x) 1 ~alpha ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)))
     ~x))

(defmacro vctr-lasrt [method x increasing]
  `(if (= 1 (.stride ~x))
     (do
       (with-lapack-check
         (~method (int (if ~increasing \I \D)) (.dim ~x) (.buffer ~x) (.offset ~x)))
       ~x)
     (throw (ex-info "You cannot sort a vector with stride different than 1." {:stride (.stride ~x)}))))

;; ----------------- Common GE matrix macros and functions -----------------------

(defmacro matrix-lasrt [method a increasing]
  `(let [incr# (int (if ~increasing \I \D))
         buff# (.buffer ~a)
         ofst# (.offset ~a)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check (~method incr# len# buff# (+ ofst# idx#))))
     ~a))

(defmacro ge-lan [method norm a]
  `(~method (.layout (navigator ~a)) ~norm (.mrows ~a) (.ncols ~a)
    (.buffer ~a) (.offset ~a) (.stride ~a)))

(defmacro ge-laset [method alpha beta a]
  `(do
     (with-lapack-check
       (~method (.layout (navigator ~a)) (int \g) (.mrows ~a) (.ncols ~a)
        ~alpha ~beta (.buffer ~a) (.offset ~a) (.stride ~a)))
     ~a))

;; ----------------- Common TR matrix macros and functions -----------------------

;; There seems to be a bug in MKL's LAPACK_?lantr. If the order is column major,
;; it returns 0.0 as a result. To fix this, I had to do the uplo# trick.
(defmacro tr-lan [method norm a]
  `(if (< 0 (.dim ~a))
     (let [reg# (region ~a)
           row-order# (.isRowMajor (navigator ~a))
           lower# (.isLower reg#)
           uplo# (if row-order# (if lower# \L \U) (if lower# \U \L))
           norm# (if row-order# ~norm (cond (= ~norm (int \I)) (int \O)
                                            (= ~norm (int \O)) (int \I)
                                            :default ~norm))]
       (~method CBLAS/ORDER_ROW_MAJOR norm#
        (int uplo#) (int (if (.isDiagUnit reg#) \U \N))
        (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a)))
     0.0))

(defmacro tr-lacpy [lacpy copy a b]
  `(let [nav# (navigator ~a)
         buff-a# (.buffer ~a)
         buff-b# (.buffer ~b)]
     (if (= nav# (navigator ~b))
       (with-lapack-check
         (let [reg# (region ~a)
               diag-pad# (long (if (.isDiagUnit reg#) 1 0))]
           (~lacpy (.layout nav#) (int (if (.isLower reg#) \L \U))
            (- (.mrows ~a) diag-pad#) (- (.ncols ~a) diag-pad#)
            buff-a# (+ (.offset ~a) (.index (storage ~a) diag-pad# 0)) (.stride ~a)
            buff-b# (+ (.offset ~b) (.index (storage ~b) diag-pad# 0)) (.stride ~b))))
       (full-storage-map ~a ~b len# offset-a# offset-b# ld-a# :skip
                         (~copy len# buff-a# offset-a# ld-a# buff-b# offset-b# 1)))
     ~b))

(defmacro tr-lascl [method alpha a]
  `(do
     (with-lapack-check
       (let [reg# (region ~a)
             diag-pad# (long (if (.isDiagUnit reg#) 1 0))]
         (~method (.layout (navigator ~a)) (int (if (.isLower reg#) \L \U))
          0 0 1.0 ~alpha (- (.mrows ~a) diag-pad#) (- (.ncols ~a) diag-pad#)
          (.buffer ~a) (+ (.offset ~a) diag-pad#) (.stride ~a))))
     ~a))

(defmacro tr-laset [method alpha beta a]
  `(do
     (with-lapack-check
       (let [reg# (region ~a)
             nav# (navigator ~a)
             diag-pad# (long (if (.isDiagUnit reg#) 1 0))
             idx# (if (.isLower reg#) (.index nav# (storage ~a) diag-pad# 0) (.index nav# (storage ~a) 0 diag-pad#))]
         (~method (.layout nav#) (int (if (.isLower reg#) \L \U))
          (- (.mrows ~a) diag-pad#) (- (.ncols ~a) diag-pad#)
          ~alpha ~beta (.buffer ~a) (+ (.offset ~a) idx#) (.stride ~a))))
     ~a))

;; ----------- Symmetric Matrix -----------------------------------------------------

(defmacro sy-lan [method norm a]
  `(if (< 0 (.dim ~a))
     (~method (.layout (navigator ~a)) ~norm (int (if (.isLower (region ~a)) \L \U))
      (.mrows ~a) (.buffer ~a) (.offset ~a) (.stride ~a))
     0.0))

(defmacro sy-lacpy [lacpy copy a b]
  `(let [nav# (navigator ~a)
         buff-a# (.buffer ~a)
         buff-b# (.buffer ~b)]
     (if (= nav# (navigator ~b))
       (with-lapack-check
         (~lacpy (.layout nav#) (int (if (.isLower (region ~a)) \L \U)) (.mrows ~a) (.ncols ~a)
          (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)))
       (full-storage-map ~a ~b len# offset-a# offset-b# ld-a# :skip
                         (~copy len# buff-a# offset-a# ld-a# buff-b# offset-b# 1)))
     ~b))

(defmacro sy-lascl [method alpha a]
  `(do
     (with-lapack-check
       (~method (.layout (navigator ~a)) (int (if (.isLower (region ~a)) \L \U))
        0 0 1.0 ~alpha (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a)))
     ~a))

(defmacro sy-laset [method alpha beta a]
  `(do
     (with-lapack-check
       (~method (.layout (navigator ~a)) (int (if (.isLower (region ~a)) \L \U))
        (.mrows ~a) (.ncols ~a) ~alpha ~beta (.buffer ~a) (.offset ~a) (.stride ~a)))
     ~a))

;; ----------- Banded Matrix --------------------------------------------------------

(defmacro banded-lan [langb nrm norm a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)
           reg# (region ~a)
           sd# (.sd stor#)
           fd# (.fd stor#)
           ld# (.ld stor#)
           kl# (.kl reg#)
           ku# (.ku reg#)
           buff# (.buffer ~a)]
       (cond
         (= sd# fd#) (with-release [work# (.createDataSource (data-accessor ~a) (min sd# fd#))]
                       (~langb ~norm fd# kl# ku# buff# (.offset ~a) ld# work#))
         (= ~norm (long \F)) (Math/sqrt (band-storage-reduce ~a len# offset# acc# 0.0
                                                             (+ acc# (pow (~nrm len# buff# offset# ld#) 2))))
         (= ~norm (long \M)) (let [da# (real-accessor ~a)]
                               (band-storage-reduce ~a len# offset# amax# 0.0
                                                    (let [iamax# (~nrm len# buff# offset# ld#)]
                                                      (max amax# (Math/abs (.get da# buff# (+ offset# (* ld# iamax#))))))))
         :default (throw (ex-info "Dragan says: This operation has not been implemented for non-square banded matrix." {}))))
     0.0))

(defmacro banded-laset [method alpha a]
  `(let [buff# (.buffer ~a)
         ofst# (.offset ~a)
         ld# (.stride ~a)]
     (band-storage-map ~a len# offset#
                       (with-lapack-check
                         (~method CBLAS/ORDER_ROW_MAJOR (int \g) len# 1 ~alpha ~alpha buff# offset# ld#)))
     ~a))

;;------------------------ Packed Matrix -------------------------------------------

(defmacro packed-laset [method alpha a]
  `(let [buff# (.buffer ~a)
         ofst# (.offset ~a)]
     (if-not (.isDiagUnit (region ~a))
       (with-lapack-check
         (~method CBLAS/ORDER_ROW_MAJOR (int \g) (.capacity (storage ~a)) 1 ~alpha ~alpha buff# ofst# 1))
       (dostripe-layout ~a len# idx#
                        (with-lapack-check
                          (~method CBLAS/ORDER_ROW_MAJOR (int \g) len# 1 ~alpha ~alpha
                           buff# (+ ofst# idx#) 1))))
     ~a))

(defmacro packed-lasrt [method a increasing]
  `(let [increasing# (int (if ~increasing \I \D))
         n# (.ncols ~a)
         buff# (.buffer ~a)
         ofst# (.offset ~a)]
     (dostripe-layout ~a len# idx#
                      (with-lapack-check (~method increasing# len# buff# (+ ofst# idx#))))
     ~a))

;; =========== Drivers and Computational LAPACK Routines ===========================

;; ------------- Singular Value Decomposition LAPACK -------------------------------

(defmacro with-sv-check [ipiv expr]
  `(if (= 1 (stride ~ipiv))
     (let [info# ~expr]
       (cond
         (= 0 info#) ~ipiv
         (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                     {:arg-index (- info#)}))
         :else (throw (ex-info "The factor U is singular, the solution could not be computed."
                               {:info info#}))))
     (throw (ex-info "You cannot use ipiv with stride different than 1." {:stride (stride ~ipiv)}))))

(defmacro ge-trf [method a ipiv]
  `(with-sv-check ~ipiv
     (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a)
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~ipiv) (.offset ~ipiv))))

(defmacro sy-trf [method a ipiv]
  `(with-sv-check ~ipiv
     (~method (.layout (navigator ~a)) (int (if (.isLower (region ~a)) \L \U)) (.sd (full-storage ~a))
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~ipiv) (.offset ~ipiv))))

(defmacro ge-tri [method a ipiv]
  `(let [stor# (full-storage ~a)]
     (with-sv-check ~ipiv
       (~method (.layout (navigator ~a)) (.sd stor#)
        (.buffer ~a) (.offset ~a) (.ld stor#) (.buffer ~ipiv) (.offset ~ipiv)))
     ~a))

(defmacro tr-tri [method a]
  `(let [stor# (full-storage ~a)
         reg# (region ~a)]
     (~method (.layout (navigator ~a))
      (int (if (.isLower reg#) \L \U)) (int (if (.isDiagUnit reg#) \U \N))
      (.sd stor#) (.buffer ~a) (.offset ~a) (.ld stor#))
     ~a))

(defmacro sy-tri [method a ipiv]
  `(let [stor# (full-storage ~a)]
     (with-sv-check ~ipiv
       (~method (.layout (navigator ~a)) (int (if (.isLower (region ~a)) \L \U)) (.sd stor#)
        (.buffer ~a) (.offset ~a) (.ld stor#) (.buffer ~ipiv) (.offset ~ipiv)))
     ~a))

(defmacro ge-trs [method a b ipiv]
  `(let [nav-b# (navigator ~b)]
     (with-sv-check ~ipiv
       (~method (.layout nav-b#) (int (if (= nav-b# (navigator ~a)) \N \T))
        (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
        (.buffer ~ipiv) (.offset ~ipiv) (.buffer ~b) (.offset ~b) (.stride ~b)))
     ~b))

(defmacro tr-trs [method a b]
  `(let [reg-a# (region ~a)
         nav-b#(navigator ~b)]
     (~method (.layout nav-b#)
      (int (if (.isLower reg-a#) \L \U)) (int (if (= nav-b# (navigator ~a)) \N \T))
      (int (if (.isDiagUnit reg-a#) \U \N)) (.mrows ~b) (.ncols ~b)
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b))
     ~b))

(defmacro sy-trs [method a b ipiv]
  `(do
     (with-sv-check ~ipiv
       (~method (.layout (navigator ~b)) (int (if (.isLower (region ~a)) \L \U))
        (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
        (.buffer ~ipiv) (.offset ~ipiv) (.buffer ~b) (.offset ~b) (.stride ~b)))
     ~b))

(defmacro ge-sv
  ([method a b pure]
   `(if ~pure
      (with-release [a# (create-ge (factory ~a) (.mrows ~a) (.ncols ~a) (.isColumnMajor (navigator ~b)) false)]
        (copy (engine ~a) ~a a#)
        (sv (engine ~a) a# ~b false))
      (let [nav-b# (navigator ~b)]
        (if (= (navigator ~a) nav-b#)
          (with-release [ipiv# (create-vector (index-factory ~a) (.ncols ~a) false)]
            (with-sv-check ipiv#
              (~method (.layout nav-b#) (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
               (buffer ipiv#) (offset ipiv#) (.buffer ~b) (.offset ~b) (.stride ~b)))
            ~b)
          (throw (ex-info "Orientation of a and b do not fit." {:a (str ~a) :b (str ~b)})))))))

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

(defmacro sy-sv
  ([method a b pure]
   `(if ~pure
      (with-release [a# (create-sy (factory ~a) (.ncols ~a) (.isColumnMajor (navigator ~b))
                                   (.isLower (region ~a)) false)]
        (copy (engine ~a) ~a a#)
        (sv (engine ~a) a# ~b false))
      (if (fits-navigation? ~a ~b)
        (with-release [ipiv# (create-vector (index-factory ~a) (.ncols ~a) nil)]
          (with-sv-check ipiv#
            (~method (.layout (navigator ~b)) (int (if (.isLower (region ~a)) \L \U))
             (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
             (buffer ipiv#) (offset ipiv#) (.buffer ~b) (.offset ~b) (.stride ~b)))
          ~b)
        (throw (ex-info "Orientation of a and b do not fit." {:a (str ~a) :b (str ~b)}))))))

;; ------------------ Condition Number ----------------------------------------------

(defmacro ge-con [da method lu nrm nrm1?]
  `(with-release [res# (.createDataSource ~da 1)]
     (let [info# (~method (.layout (navigator ~lu)) (int (if ~nrm1? \O \I))
                  (min (.mrows ~lu) (.ncols ~lu))
                  (.buffer ~lu) (.offset ~lu) (.stride ~lu) ~nrm res#)]
       (if (= 0 info#)
         (.get ~da res# 0)
         (throw (ex-info "There has been an illegal argument in the native function call."
                         {:arg-index (- info#)}))))))

(defmacro tr-con [da method a nrm1?]
  `(with-release [res# (.createDataSource ~da 1)
                  reg# (region ~a)
                  info# (~method (.layout (navigator ~a)) (int (if ~nrm1? \O \I))
                         (int (if (.isLower reg#) \L \U)) (int (if (.isDiagUnit reg#) \U \N))
                         (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a) res#)]
     (if (= 0 info#)
       (.get ~da res# 0)
       (throw (ex-info "There has been an illegal argument in the native function call."
                       {:arg-index (- info#)})))))

(defmacro sy-con [da method ldl ipiv nrm]
  `(with-release [res# (.createDataSource ~da 1)]
     (let [info# (~method (.layout (navigator ~ldl)) (int (if (.isLower (region ~ldl)) \L \U)) (.ncols ~ldl)
                  (.buffer ~ldl) (.offset ~ldl) (.stride ~ldl) (.buffer ~ipiv) (.offset ~ipiv)
                  ~nrm res#)]
       (if (= 0 info#)
         (.get ~da res# 0)
         (throw (ex-info "There has been an illegal argument in the native function call."
                         {:arg-index (- info#)}))))))

;; ------------- Orthogonal Factorization (L, Q, R) LAPACK -------------------------------

(defmacro with-lqr-check [tau res expr]
  `(if (= 1 (.stride ~tau))
     (let [info# ~expr]
       (if (= 0 info#)
         ~res
         (throw (ex-info "There has been an illegal argument in the native function call."
                         {:arg-index (- info#)}))))
     (throw (ex-info "You cannot use tau with stride different than 1." {:stride (.stride ~tau)}))))

(defmacro ge-lqrf [method a tau]
  `(with-lqr-check ~tau ~tau
     (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a)
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~tau) (.offset ~tau))))

(defmacro or-glqr [method a tau]
  `(with-lqr-check ~tau ~a
     (~method (.layout (navigator ~a)) (.mrows ~a) (.ncols ~a) (.dim ~tau)
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~tau) (.offset ~tau))))

(defmacro or-mlqr [method a tau c left]
  `(with-lqr-check ~tau ~c
     (~method (.layout (navigator ~c)) (int (if ~left \L \R))
      (int (if (= (navigator ~a) (navigator ~c)) \N \T)) (.mrows ~c) (.ncols ~c) (.dim ~tau)
      (.buffer ~a) (.offset ~a) (.stride ~a)
      (.buffer ~tau) (.offset ~tau) (.buffer ~c) (.offset ~c) (.stride ~c))))

;; ------------- Linear Least Squares Routines LAPACK -------------------------------

(defmacro ge-ls [method a b]
  `(let [nav# (navigator ~a)
         info# (~method (.layout nav#) (int (if (= nav# (navigator ~b)) \N \T))
                (.mrows ~a) (.ncols ~a) (.ncols ~b)
                (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b))]
     (cond
       (= 0 info#) ~b
       (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                   {:arg-index (- info#)}))
       :else (throw (ex-info "The i-th diagonal element of a is zero, so the matrix does not have full rank."
                             {:arg-index info#})))))

;; ------------- Non-Symmetric Eigenvalue Problem Routines LAPACK -------------------------------

(defmacro ge-ev [method a w vl vr]
  `(if (.isColumnMajor (navigator ~w))
     (let [wr# (.col ~w 0)
           wi# (.col ~w 1)
           info# (~method (.layout (navigator  ~a))
                  (int (if (< 0 (.mrows ~vl)) \V \N)) (int (if (< 0 (.mrows ~vr)) \V \N))
                  (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a)
                  (buffer wr#) (offset wr#) (buffer wi#) (offset wi#)
                  (.buffer ~vl) (.offset ~vl) (.stride ~vl) (.buffer ~vr) (.offset ~vr) (.stride ~vr))]
       (cond
         (= 0 info#) ~w
         (< info# 0) (throw (ex-info "There has been an illegal argument in the native function call."
                                     {:arg-index (- info#)}))
         :else (throw (ex-info "The QR algorithm failed to compute all the eigenvalues."
                               {:first-converged (inc info#)}))))
     (throw (ex-info "You cannot use w that is not column-oriented and has less than 2 columns."
                     {:column-major? (.isColumnMajor (navigator  ~w)) :ncols (.ncols ~w)}))))

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
  ([method a s u vt superb]
   `(let [m# (.mrows ~a)
          n# (.ncols ~a)]
      (with-svd-check ~s
        (~method (.layout (navigator  ~a))
         (int (cond (= m# (.mrows ~u) (.ncols ~u)) \A
                    (and (= m# (.mrows ~u)) (= (min m# n#) (.ncols ~u))) \S
                    (nil? ~u) \O
                    :else \N))
         (int (cond (= n# (.mrows ~vt) (.ncols ~vt)) \A
                    (and (= (min m# n#) (.mrows ~vt)) (= n# (.ncols ~vt))) \S
                    (and ~u (nil? ~vt)) \O
                    :else \N))
         m# n# (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~s) (.offset ~s)
         (.buffer ~u) (.offset ~u) (.stride ~u) (.buffer ~vt) (.offset ~vt) (.stride ~vt)
         (.buffer ~superb) (.offset ~superb)))))
  ([method a s zero-uvt superb]
   `(with-svd-check ~s
      (~method (.layout (navigator  ~a)) (int \N) (int \N) (.mrows ~a) (.ncols ~a)
       (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~s) (.offset ~s)
       (.buffer ~zero-uvt) (.offset ~zero-uvt) (.stride ~zero-uvt)
       (.buffer ~zero-uvt) (.offset ~zero-uvt) (.stride ~zero-uvt)
       (.buffer ~superb) (.offset ~superb)))))
