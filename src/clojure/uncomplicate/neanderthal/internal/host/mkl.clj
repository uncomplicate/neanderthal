;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.mkl
  (:require [vertigo.bytes :refer [direct-buffer]]
            [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.neanderthal.math :refer [f=]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [dragan-says-ex check-stride real-accessor]]]
            [uncomplicate.neanderthal.internal.host
             [buffer-block :refer :all]
             [cblas :refer :all]
             [lapack :refer :all]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS MKL LAPACK]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.internal.api DataAccessor RealBufferAccessor
            Block RealVector Region LayoutNavigator DenseStorage RealNativeMatrix]
           [uncomplicate.neanderthal.internal.host.buffer_block IntegerBlockVector RealBlockVector
            RealGEMatrix RealUploMatrix RealBandedMatrix RealPackedMatrix RealDiagonalMatrix]))

(defn ^:private not-available [kind]
  (throw (UnsupportedOperationException. "Operation not available for %s matrix")))

(def ^{:no-doc true :const true} INTEGER_UNSUPPORTED_MSG
  "\nInteger BLAS operations are not supported. Please transform data to float or double.\n")

;; =========== MKL-spicific routines ====================================================

(defmacro ge-copy [method a b]
  ` (if (< 0 (.dim ~a))
      (let [stor-b# (full-storage ~b)
            no-trans# (= (navigator ~a) (navigator ~b))
            rows# (if no-trans# (.sd stor-b#) (.fd stor-b#))
            cols# (if no-trans# (.fd stor-b#) (.sd stor-b#))]
        (~method (int \C) (int (if no-trans# \N \T)) rows# cols#
         1.0 (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.ld stor-b#))
        ~b)
      ~b))

(defmacro ge-scal [method alpha a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)]
       (~method (int \c) (int \n) (.sd stor#) (.fd stor#)
        ~alpha (.buffer ~a) (.offset ~a) (.ld stor#) (.ld stor#))
       ~a)
     ~a))

(defmacro ge-trans [method a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)]
       (if (.isGapless stor#)
         (~method (int \c) (int \t) (.sd stor#) (.fd stor#)
          1.0 (.buffer ~a) (.offset ~a) (.ld stor#) (.fd stor#))
         (dragan-says-ex "You can not hard-transpose the content of a matrix with a gap in memory. Sorry."
                         {:a (info ~a)}))
       ~a)
     ~a))

(defmacro ge-axpby [method alpha a beta b]
  `(if (< 0 (.dim ~a))
     (let [nav-b# (navigator ~b)]
       (~method (int (if (.isColumnMajor nav-b#) \C \R))
        (int (if (= (navigator ~a) nav-b#) \n \t)) (int \n) (.mrows ~b) (.ncols ~b)
        ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) ~beta (.buffer ~b) (.offset ~b) (.stride ~b)
        (.buffer ~b) (.offset ~b) (.stride ~b))
       ~b)
     ~b))

(defmacro gd-sv [vdiv-method sv-method a b]
  `(let [n-a# (.ncols ~a)
         n-b# (.ncols ~b)
         nav-b# (navigator ~b)
         stor-b# (storage ~b)
         buff-a# (.buffer ~a)
         ofst-a# (.offset ~a)
         buff-b# (.buffer ~b)
         ofst-b# (.offset ~b)
         strd-b# (.stride ~b)]
     (if (.isColumnMajor nav-b#)
       (dotimes [j# n-b#]
         (~vdiv-method n-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) buff-a# ofst-a#
          buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#))))
       (dotimes [j# (.ncols ~b)]
         (~sv-method CBLAS/ORDER_ROW_MAJOR CBLAS/UPLO_LOWER CBLAS/TRANSPOSE_NO_TRANS CBLAS/DIAG_NON_UNIT n-a# 0
          buff-a# ofst-a# 1 buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) strd-b#)))
     ~b))

(defmacro gd-tri [method a]
  `(do
     (~method (.ncols ~a) (.buffer ~a) (.offset ~a) (.buffer ~a) (.offset ~a))
     ~a))

;; ============ Integer Vector Engines ============================================

(deftype LongVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/dswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/dlaset alpha ^RealBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

(deftype IntVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/sswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/slaset alpha ^RealBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/dswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/ddot ^RealBlockVector x ^RealBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-method CBLAS/dnrm2 ^RealBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-method CBLAS/dasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/idamax ^RealBlockVector x))
  (iamin [_ x]
    (vector-method CBLAS/idamin ^RealBlockVector x))
  (rot [_ x y c s]
    (vector-rot CBLAS/drot ^RealBlockVector x ^RealBlockVector y c s)
    x)
  (rotg [_ abcs]
    (CBLAS/drotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
    abcs)
  (rotm [_ x y param]
    (vector-rotm CBLAS/drotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param))
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/drotmg ^RealBlockVector d1d2xy ^RealBlockVector param))
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^RealBlockVector x)
                 alpha  (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  BlasPlus
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (vector-method CBLAS/dsum ^RealBlockVector x))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/dlaset alpha ^RealBlockVector x))
  (axpby [_ alpha x beta y]
    (MKL/daxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/dlasrt ^RealBlockVector x increasing)))

(deftype FloatVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/sswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/sdot ^RealBlockVector x ^RealBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-method CBLAS/snrm2 ^RealBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-method CBLAS/sasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/isamax ^RealBlockVector x))
  (iamin [_ x]
    (vector-method CBLAS/isamin ^RealBlockVector x))
  (rot [_ x y c s]
    (vector-rot CBLAS/srot ^RealBlockVector x ^RealBlockVector y c s))
  (rotg [_ abcs]
    (CBLAS/srotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
    abcs)
  (rotm [_ x y param]
    (vector-rotm CBLAS/srotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param))
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/srotmg ^RealBlockVector d1d2xy ^RealBlockVector param))
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  BlasPlus
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (vector-method CBLAS/ssum ^RealBlockVector x))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/slaset alpha ^RealBlockVector x))
  (axpby [_ alpha x beta y]
    (MKL/saxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/slasrt ^RealBlockVector x increasing)))

;; ================= General Matrix Engines ====================================
(let [zero-storage (full-storage true 0 0 1)]
  (def ^:private zero-matrix ^RealGEMatrix
    (->RealGEMatrix nil zero-storage nil nil nil nil true (ByteBuffer/allocateDirect 0) 0 0 0)))

(deftype DoubleGEEngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/dswap ^RealGEMatrix a ^RealGEMatrix b))
  (copy [_ a b]
    (ge-copy MKL/domatcopy ^RealGEMatrix a ^RealGEMatrix b))
  (scal [_ alpha a]
    (ge-scal MKL/dimatcopy alpha ^RealGEMatrix a))
  (dot [_ a b]
    (matrix-dot CBLAS/ddot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/dlange (int \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/dlange (int \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/dlange (int \I) ^RealGEMatrix a))
  (asum [_ a]
    (matrix-sum CBLAS/dasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv CBLAS/dgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/dger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm CBLAS/dgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c))
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/dlange (int \M) ^RealGEMatrix a))
  (sum [_ a]
    (matrix-sum CBLAS/dsum ^RealGEMatrix a))
  (set-all [_ alpha a]
    (ge-laset LAPACK/dlaset alpha alpha ^RealGEMatrix a))
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b))
  (trans [_ a]
    (ge-trans MKL/dimatcopy ^RealGEMatrix a))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealGEMatrix a increasing))
  (laswp [_ a ipiv k1 k2]
    (ge-laswp LAPACK/dlaswp ^RealGEMatrix a ^IntegerBlockVector ipiv k1 k2))
  (lapmr [_ a k forward]
    (ge-lapm LAPACK/dlapmr ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (lapmt [_ a k forward]
    (ge-lapm LAPACK/dlapmt ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (trf [_ a ipiv]
    (ge-trf LAPACK/dgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for general matrices."))
  (tri [_ lu ipiv]
    (ge-tri LAPACK/dgetri ^RealGEMatrix lu ^IntegerBlockVector ipiv))
  (trs [_ lu b ipiv]
    (ge-trs LAPACK/dgetrs ^RealGEMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (ge-sv LAPACK/dgesv ^RealGEMatrix a ^RealGEMatrix b pure))
  (con [_ lu _ nrm nrm1?]
    (ge-con LAPACK/dgecon ^RealGEMatrix lu nrm nrm1?))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/dgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/dgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (qp3 [_ a jpiv tau]
    (ge-qp3 LAPACK/dgeqp3 ^RealGEMatrix a ^IntegerBlockVector jpiv ^RealBlockVector tau))
  (gqr [_ a tau]
    (or-glqr LAPACK/dorgqr ^RealGEMatrix a ^RealBlockVector tau))
  (mqr [_ a tau c left]
    (or-mlqr LAPACK/dormqr ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/dgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (grq [_ a tau]
    (or-glqr LAPACK/dorgrq ^RealGEMatrix a ^RealBlockVector tau))
  (mrq [_ a tau c left]
    (or-mlqr LAPACK/dormrq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/dgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (glq [_ a tau]
    (or-glqr LAPACK/dorglq ^RealGEMatrix a ^RealBlockVector tau))
  (mlq [_ a tau c left]
    (or-mlqr LAPACK/dormlq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/dgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (gql [_ a tau]
    (or-glqr LAPACK/dorgql ^RealGEMatrix a ^RealBlockVector tau))
  (mql [_ a tau c left]
    (or-mlqr LAPACK/dormql ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (ls [_ a b]
    (ge-ls LAPACK/dgels ^RealGEMatrix a ^RealGEMatrix b))
  (lse [_ a b c d x]
    (ge-lse LAPACK/dgglse ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector c ^RealBlockVector d ^RealBlockVector x))
  (gls [_ a b d x y]
    (ge-gls LAPACK/dggglm ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector d ^RealBlockVector x ^RealBlockVector y))
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/dgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
              ^RealGEMatrix u ^RealGEMatrix vt ^RealDiagonalMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealDiagonalMatrix superb)))

(deftype FloatGEEngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b))
  (copy [_ a b]
    (ge-copy MKL/somatcopy ^RealGEMatrix a ^RealGEMatrix b))
  (scal [_ alpha a]
    (ge-scal MKL/simatcopy alpha ^RealGEMatrix a))
  (dot [_ a b]
    (matrix-dot CBLAS/sdot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/slange (int \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/slange (int \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/slange (int \I) ^RealGEMatrix a))
  (asum [_ a]
    (matrix-sum CBLAS/sasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv CBLAS/sgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/sger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm CBLAS/sgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c))
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/slange (int \M) ^RealGEMatrix a))
  (sum [_ a]
    (matrix-sum CBLAS/ssum ^RealGEMatrix a))
  (set-all [_ alpha a]
    (ge-laset LAPACK/slaset alpha alpha ^RealGEMatrix a))
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b))
  (trans [_ a]
    (ge-trans MKL/simatcopy ^RealGEMatrix a))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealGEMatrix a increasing))
  (laswp [_ a ipiv k1 k2]
    (ge-laswp LAPACK/slaswp ^RealGEMatrix a ^IntegerBlockVector ipiv k1 k2))
  (lapmr [_ a k forward]
    (ge-lapm LAPACK/slapmr ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (lapmt [_ a k forward]
    (ge-lapm LAPACK/slapmt ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (trf [_ a ipiv]
    (ge-trf LAPACK/sgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for general matrices."))
  (tri [_ lu ipiv]
    (ge-tri LAPACK/sgetri ^RealGEMatrix lu ^IntegerBlockVector ipiv))
  (trs [_ lu b ipiv]
    (ge-trs LAPACK/sgetrs ^RealGEMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (ge-sv LAPACK/sgesv ^RealGEMatrix a ^RealGEMatrix b pure))
  (con [_ lu _ nrm nrm1?]
    (ge-con LAPACK/sgecon ^RealGEMatrix lu nrm nrm1?))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/sgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/sgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (qp3 [_ a jpiv tau]
    (ge-qp3 LAPACK/sgeqp3 ^RealGEMatrix a ^IntegerBlockVector jpiv ^RealBlockVector tau))
  (gqr [_ a tau]
    (or-glqr LAPACK/sorgqr ^RealGEMatrix a ^RealBlockVector tau))
  (mqr [_ a tau c left]
    (or-mlqr LAPACK/sormqr ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/sgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (grq [_ a tau]
    (or-glqr LAPACK/sorgrq ^RealGEMatrix a ^RealBlockVector tau))
  (mrq [_ a tau c left]
    (or-mlqr LAPACK/sormrq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/sgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (glq [_ a tau]
    (or-glqr LAPACK/sorglq ^RealGEMatrix a ^RealBlockVector tau))
  (mlq [_ a tau c left]
    (or-mlqr LAPACK/sormlq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/sgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (gql [_ a tau]
    (or-glqr LAPACK/sorgql ^RealGEMatrix a ^RealBlockVector tau))
  (mql [_ a tau c left]
    (or-mlqr LAPACK/sormql ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (ls [_ a b]
    (ge-ls LAPACK/sgels ^RealGEMatrix a ^RealGEMatrix b))
  (lse [_ a b c d x]
    (ge-lse LAPACK/sgglse ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector c ^RealBlockVector d ^RealBlockVector x))
  (gls [_ a b d x y]
    (ge-gls LAPACK/sggglm ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector d ^RealBlockVector x ^RealBlockVector y))
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/sgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
              ^RealGEMatrix u ^RealGEMatrix vt ^RealDiagonalMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealDiagonalMatrix superb)))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/dswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (tr-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (tr-lascl LAPACK/dlascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (tr-dot CBLAS/ddot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/dlantr (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/dlantr (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/dlantr (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/dasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/daxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ _ a _ _ _]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/dtrmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/dtrmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tr-lan LAPACK/dlantr (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (tr-sum CBLAS/dsum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/dlaset alpha alpha ^RealUploMatrix a) a)
  (axpby [_ alpha a beta b]
    (matrix-axpby MKL/daxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for TR matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealUploMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (tr-tri LAPACK/dtrtri ^RealUploMatrix a))
  (trs [_ a b]
    (tr-trs LAPACK/dtrtrs ^RealUploMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tr-sv CBLAS/dtrsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/dtrcon ^RealUploMatrix a nrm1?)))

(deftype FloatTREngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/sswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (tr-lacpy LAPACK/slacpy CBLAS/scopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (tr-lascl LAPACK/slascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (tr-dot CBLAS/sdot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/slantr (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/slantr (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/slantr (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/sasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/saxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ _ a _ _ _]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/strmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/strmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tr-lan LAPACK/slantr (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (tr-sum  CBLAS/ssum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/slaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (matrix-axpby MKL/saxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for TR matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealUploMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (tr-tri LAPACK/strtri ^RealUploMatrix a))
  (trs [_ a b]
    (tr-trs LAPACK/strtrs ^RealUploMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tr-sv CBLAS/strsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/strcon ^RealUploMatrix a nrm1?)))

;; =============== Symmetric Matrix Engines ===================================

(deftype DoubleSYEngine []
  Blas
  (swap [_ a b]
    (sy-swap CBLAS/dswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (sy-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (sy-lascl LAPACK/dlascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (sy-dot CBLAS/ddot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/dlansy (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/dlansy (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/dlansy (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/dasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (sy-axpy CBLAS/daxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/dsymv alpha ^RealUploMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/dsymm alpha ^RealUploMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/dlansy (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/dsum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/dlaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (sy-axpby MKL/daxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for SY matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealUploMatrix a increasing))
  (trf [_ a ipiv]
    (sy-trx LAPACK/dsytrf ^RealUploMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sy-trx LAPACK/dpotrf ^RealUploMatrix a))
  (trfx [_ a]
    (sy-trfx LAPACK/dpotrf ^RealUploMatrix a))
  (tri [_ ldl ipiv]
    (sy-trx LAPACK/dsytri ^RealUploMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sy-trx LAPACK/dpotri ^RealUploMatrix gg))
  (trs [_ ldl b ipiv]
    (sy-trs LAPACK/dsytrs ^RealUploMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sy-trs LAPACK/dpotrs ^RealUploMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sy-sv LAPACK/dposv LAPACK/dsysv ^RealUploMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sy-sv LAPACK/dposv ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sy-con LAPACK/dsycon ^RealUploMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sy-con LAPACK/dpocon ^RealUploMatrix gg nrm)))

(deftype FloatSYEngine []
  Blas
  (swap [_ a b]
    (sy-swap CBLAS/sswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (sy-lacpy LAPACK/slacpy CBLAS/scopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (sy-lascl LAPACK/slascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (sy-dot CBLAS/sdot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/slansy (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/slansy (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/slansy (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/sasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (sy-axpy CBLAS/saxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/ssymv alpha ^RealUploMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/ssymm alpha ^RealUploMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/slansy (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/ssum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/slaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (sy-axpby MKL/saxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for SY matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealUploMatrix a increasing))
  (trf [_ a ipiv]
    (sy-trx LAPACK/ssytrf ^RealUploMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sy-trx LAPACK/spotrf ^RealUploMatrix a))
  (trfx [_ a]
    (sy-trfx LAPACK/spotrf ^RealUploMatrix a))
  (tri [_ ldl ipiv]
    (sy-trx LAPACK/ssytri ^RealUploMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sy-trx LAPACK/spotri ^RealUploMatrix gg))
  (trs [_ ldl b ipiv]
    (sy-trs LAPACK/ssytrs ^RealUploMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sy-trs LAPACK/spotrs ^RealUploMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sy-sv LAPACK/sposv LAPACK/ssysv ^RealUploMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sy-sv LAPACK/sposv ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sy-con LAPACK/ssycon ^RealUploMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sy-con LAPACK/spocon ^RealUploMatrix gg nrm)))

;; =============== Banded Matrix Engines ===================================

(deftype DoubleGBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (gb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/dlangb CBLAS/dnrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (gb-mv CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [_ alpha a b beta c left]
    (gb-mm CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (gb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (trf [_ a ipiv]
    (gb-trf LAPACK/dgbtrf ^RealBandedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (dragan-says-ex "Pivotless factorization is not available for GB matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ lu b ipiv]
    (gb-trs LAPACK/dgbtrs ^RealBandedMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gb-sv LAPACK/dgbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (con [_ ldl ipiv nrm nrm1?]
    (gb-con LAPACK/dgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype FloatGBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (gb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/slangb CBLAS/snrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (gb-mv CBLAS/sgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [_ alpha a b beta c left]
    (gb-mm CBLAS/sgbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (gb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (trf [_ a ipiv]
    (gb-trf LAPACK/sgbtrf ^RealBandedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (dragan-says-ex "Pivotless factorization is not available for GB matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ lu b ipiv]
    (gb-trs LAPACK/sgbtrs ^RealBandedMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gb-sv LAPACK/sgbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (con [_ ldl ipiv nrm nrm1?]
    (gb-con LAPACK/sgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype DoubleSBEngine []
  Blas
  (swap [_ a b]
    (sb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (sb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (sb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (sb-lan LAPACK/dlansb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (sb-lan LAPACK/dlansb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (sb-lan LAPACK/dlansb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (sb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (sb-mv CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [_ alpha a b beta c left]
    (sb-mm CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (sb-lan LAPACK/dlansb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (sb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (sb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (trf [_ a]
    (sb-trf LAPACK/dpbtrf ^RealBandedMatrix a))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ gg b]
    (sb-trs LAPACK/dpbtrs ^RealBandedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sb-sv LAPACK/dpbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sb-sv LAPACK/dpbsv ^RealBandedMatrix a ^RealGEMatrix b false))
  (con [_ gg nrm _]
    (sb-con LAPACK/dpbcon ^RealBandedMatrix gg nrm)))

(deftype FloatSBEngine []
  Blas
  (swap [_ a b]
    (sb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (sb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (sb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (sb-lan LAPACK/slansb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (sb-lan LAPACK/slansb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (sb-lan LAPACK/slansb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (sb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (sb-mv CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [_ alpha a b beta c left]
    (sb-mm CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (sb-lan LAPACK/slansb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (sb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (sb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (trf [_ a]
    (sb-trf LAPACK/spbtrf ^RealBandedMatrix a))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ gg b]
    (sb-trs LAPACK/spbtrs ^RealBandedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sb-sv LAPACK/spbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sb-sv LAPACK/spbsv ^RealBandedMatrix a ^RealGEMatrix b false))
  (con [_ gg nrm _]
    (sb-con LAPACK/spbcon ^RealBandedMatrix gg nrm)))

(deftype DoubleTBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (tb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (tb-lan LAPACK/dlantb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (tb-lan LAPACK/dlantb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (tb-lan LAPACK/dlantb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (tb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ _ a _ _ _]
    (tb-mv a))
  (mv [_ a x]
    (tb-mv CBLAS/dtbmv ^RealBandedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tb-mm a))
  (mm [_ alpha a b left]
    (tb-mm CBLAS/dtbmv alpha ^RealBandedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tb-lan LAPACK/dlantb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (tb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ a b]
    (tb-trs LAPACK/dtbtrs ^RealBandedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tb-sv CBLAS/dtbsv ^RealBandedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tb-con LAPACK/dtbcon ^RealBandedMatrix a nrm1?)))

(deftype FloatTBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (tb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (tb-lan LAPACK/slantb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (tb-lan LAPACK/slantb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (tb-lan LAPACK/slantb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (tb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ _ a _ _ _]
    (tb-mv a))
  (mv [_ a x]
    (tb-mv CBLAS/stbmv ^RealBandedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tb-mm a))
  (mm [_ alpha a b left]
    (tb-mm CBLAS/stbmv alpha ^RealBandedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tb-lan LAPACK/slantb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (tb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ a b]
    (tb-trs LAPACK/stbtrs ^RealBandedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tb-sv CBLAS/stbsv ^RealBandedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tb-con LAPACK/stbcon ^RealBandedMatrix a nrm1?)))

;; =============== Packed Matrix Engines ===================================

(deftype DoubleTPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/dswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/dcopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/dscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (tp-dot CBLAS/ddot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (tp-lan LAPACK/dlantp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (tp-lan LAPACK/dlantp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (tp-lan LAPACK/dlantp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (tp-sum CBLAS/dasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ _ a _ _ _]
    (tp-mv a))
  (mv [_ a x]
    (tp-mv CBLAS/dtpmv ^RealPackedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tp-mm a))
  (mm [_ alpha a b left]
    (tp-mm CBLAS/dtpmv alpha ^RealPackedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tp-lan LAPACK/dlantp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (tp-sum CBLAS/dsum double ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TP matrices."))
  (tri [_ a]
    (tp-tri LAPACK/dtptri ^RealPackedMatrix a))
  (trs [_ a b]
    (tp-trs LAPACK/dtptrs ^RealPackedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tp-sv CBLAS/dtpsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tp-con LAPACK/dtpcon ^RealPackedMatrix a nrm1?)))

(deftype FloatTPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/sswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/scopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/sscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (tp-dot CBLAS/sdot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (tp-lan LAPACK/slantp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (tp-lan LAPACK/slantp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (tp-lan LAPACK/slantp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (tp-sum CBLAS/sasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/saxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ _ a _ _ _]
    (tp-mv a))
  (mv [_ a x]
    (tp-mv CBLAS/stpmv ^RealPackedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tp-mm a))
  (mm [_ alpha a b left]
    (tp-mm CBLAS/stpmv alpha ^RealPackedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tp-lan LAPACK/slantp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (tp-sum CBLAS/ssum double ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/slaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/saxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/slasrt ^RealPackedMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TP matrices."))
  (tri [_ a]
    (tp-tri LAPACK/stptri ^RealPackedMatrix a))
  (trs [_ a b]
    (tp-trs LAPACK/stptrs ^RealPackedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tp-sv CBLAS/stpsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tp-con LAPACK/stpcon ^RealPackedMatrix a nrm1?)))

(deftype DoubleSPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/dswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/dcopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/dscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (sp-dot CBLAS/ddot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (sp-lan LAPACK/dlansp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (sp-lan LAPACK/dlansp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (sp-lan LAPACK/dlansp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (sp-sum CBLAS/dasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ alpha a x beta y]
    (sp-mv CBLAS/dspmv alpha ^RealPackedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sp-mv a))
  (mm [_ alpha a b beta c left]
    (sp-mm CBLAS/dspmv alpha ^RealPackedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sp-mm a))
  BlasPlus
  (amax [_ a]
    (sp-lan LAPACK/dlansp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (sp-sum CBLAS/dsum double ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  (trf [_ a ipiv]
    (sp-trx LAPACK/dsptrf ^RealPackedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sp-trx LAPACK/dpptrf ^RealPackedMatrix a))
  (trfx [_ a]
    (sp-trfx LAPACK/dpptrf ^RealPackedMatrix a))
  (tri [_ ldl ipiv]
    (sp-trx LAPACK/dsptri ^RealPackedMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sp-trx LAPACK/dpptri ^RealPackedMatrix gg))
  (trs [_ ldl b ipiv]
    (sp-trs LAPACK/dsptrs ^RealPackedMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sp-trs LAPACK/dpptrs ^RealPackedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sp-sv LAPACK/dppsv LAPACK/dspsv ^RealPackedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sp-sv LAPACK/dppsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sp-con LAPACK/dspcon ^RealPackedMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sp-con LAPACK/dppcon ^RealPackedMatrix gg nrm)))

(deftype FloatSPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/sswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/scopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/sscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (sp-dot CBLAS/sdot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (sp-lan LAPACK/slansp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (sp-lan LAPACK/slansp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (sp-lan LAPACK/slansp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (sp-sum CBLAS/sasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/saxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ alpha a x beta y]
    (sp-mv CBLAS/sspmv alpha ^RealPackedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sp-mv a))
  (mm [_ alpha a b beta c left]
    (sp-mm CBLAS/sspmv alpha ^RealPackedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sp-mm a))
  BlasPlus
  (amax [_ a]
    (sp-lan LAPACK/slansp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (sp-sum CBLAS/ssum double ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/slaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/saxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/slasrt ^RealPackedMatrix a increasing))
  (trf [_ a ipiv]
    (sp-trx LAPACK/ssptrf ^RealPackedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sp-trx LAPACK/spptrf ^RealPackedMatrix a))
  (trfx [_ a]
    (sp-trfx LAPACK/spptrf ^RealPackedMatrix a))
  (tri [_ ldl ipiv]
    (sp-trx LAPACK/ssptri ^RealPackedMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sp-trx LAPACK/spptri ^RealPackedMatrix gg))
  (trs [_ ldl b ipiv]
    (sp-trs LAPACK/ssptrs ^RealPackedMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sp-trs LAPACK/spptrs ^RealPackedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sp-sv LAPACK/sppsv LAPACK/sspsv ^RealPackedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sp-sv LAPACK/sppsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sp-con LAPACK/sspcon ^RealPackedMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sp-con LAPACK/sppcon ^RealPackedMatrix gg nrm)))

;; =============== Tridiagonal Matrix Engines =================================

(deftype DoubleGTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a ipiv]
    (diagonal-trf LAPACK/dgttrf ^RealDiagonalMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for GT matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for GT matrices."))
  (trs [_ lu b ipiv]
    (gt-trs LAPACK/dgttrs ^RealDiagonalMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gt-sv LAPACK/dgtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (gt-con LAPACK/dgtcon ^RealDiagonalMatrix lu ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype FloatGTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a ipiv]
    (diagonal-trf LAPACK/sgttrf ^RealDiagonalMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for GT matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for GT matrices."))
  (trs [_ lu b ipiv]
    (gt-trs LAPACK/sgttrs ^RealDiagonalMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gt-sv LAPACK/sgtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (gt-con LAPACK/sgtcon ^RealDiagonalMatrix lu ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype DoubleGDEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (gd-mv CBLAS/dsbmv alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (gd-mv LAPACK/dlascl2 ^RealDiagonalMatrix a ^RealBlockVector x))
  (mm [_ alpha a b beta c left]
    (gd-mm CBLAS/dsbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b left]
    (gd-mm LAPACK/dlascl2 CBLAS/dtbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (tri [_ a]
    (gd-tri MKL/vdinv ^RealDiagonalMatrix a))
  (trs [_ a b]
    (gd-trs LAPACK/dtbtrs ^RealDiagonalMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (gd-sv MKL/vddiv CBLAS/dtbsv ^RealDiagonalMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (gd-con LAPACK/dtbcon ^RealDiagonalMatrix a nrm1?)))

(deftype FloatGDEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (gd-mv CBLAS/ssbmv alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (gd-mv LAPACK/slascl2 ^RealDiagonalMatrix a ^RealBlockVector x))
  (mm [_ alpha a b beta c left]
    (gd-mm CBLAS/ssbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b left]
    (gd-mm LAPACK/slascl2 CBLAS/stbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (tri [_ a]
    (gd-tri MKL/vsinv ^RealDiagonalMatrix a))
  (trs [_ a b]
    (gd-trs LAPACK/stbtrs ^RealDiagonalMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (gd-sv MKL/vsdiv CBLAS/stbsv ^RealDiagonalMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (gd-con LAPACK/stbcon ^RealDiagonalMatrix a nrm1?)))

(deftype DoubleDTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/ddttrfb ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for DT matrices."))
  (trs [_ lu b]
    (dt-trs LAPACK/ddttrsb ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (dt-sv LAPACK/ddtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for DT matrices.")))

(deftype FloatDTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/slangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/slangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/sdttrfb ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for DT matrices."))
  (trs [_ lu b]
    (dt-trs LAPACK/sdttrsb ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (dt-sv LAPACK/sdtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for DT matrices.")))

(deftype DoubleSTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (st-sum CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \F) ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (st-sum CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlastm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlastm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (st-sum CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/dpttrf ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for ST matrices."))
  (trs [_ lu b]
    (st-trs LAPACK/dpttrs ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (st-sv LAPACK/dptsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for ST matrices.")))

(deftype FloatSTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (st-sum CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/slanst (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (tridiagonal-lan LAPACK/slanst (int \F) ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/slanst (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (st-sum CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slastm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slastm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (st-sum CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/spttrf ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for ST matrices."))
  (trs [_ lu b]
    (st-trs LAPACK/spttrs ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (st-sv LAPACK/sptsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for ST matrices.")))

;; =============== Factories ==================================================

(deftype MKLRealFactory [index-fact ^DataAccessor da
                         vector-eng ge-eng tr-eng sy-eng gb-eng sb-eng tb-eng sp-eng tp-eng
                         gd-eng gt-eng dt-eng st-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    @index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  Factory
  (create-vector [this n _]
    (real-block-vector this n))
  (create-ge [this m n column? _]
    (real-ge-matrix this m n column?))
  (create-uplo [this n mat-type column? lower? diag-unit? _]
    (real-uplo-matrix this n column? lower? diag-unit? mat-type))
  (create-tr [this n column? lower? diag-unit? _]
    (real-uplo-matrix this n column? lower? diag-unit?))
  (create-sy [this n column? lower? _]
    (real-uplo-matrix this n column? lower?))
  (create-banded [this m n kl ku matrix-type column? _]
    (real-banded-matrix this m n kl ku column? matrix-type))
  (create-gb [this m n kl ku lower? _]
    (real-banded-matrix this m n kl ku lower?))
  (create-tb [this n k column? lower? diag-unit? _]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (real-tb-matrix this n k column? lower? diag-unit?)
      (dragan-says-ex "TB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-sb [this n k column? lower? _]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (real-sb-matrix this n k column? lower?)
      (dragan-says-ex "SB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-packed [this n matrix-type column? lower? diag-unit? _]
    (case matrix-type
      :tp (real-packed-matrix this n column? lower? diag-unit?)
      :sy (real-packed-matrix this n column? lower?)
      (dragan-says-ex "Packed matrices have to be either triangular or symmetric."
                      {:matrix-type matrix-type})))
  (create-tp [this n column? lower? diag-unit? _]
    (real-packed-matrix this n column? lower? diag-unit?))
  (create-sp [this n column? lower? _]
    (real-packed-matrix this n column? lower?))
  (create-diagonal [this n matrix-type _]
    (real-diagonal-matrix this n matrix-type))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng)
  (gb-engine [_]
    gb-eng)
  (sb-engine [_]
    sb-eng)
  (tb-engine [_]
    tb-eng)
  (tp-engine [_]
    tp-eng)
  (sp-engine [_]
    sp-eng)
  (gd-engine [_]
    gd-eng)
  (gt-engine [_]
    gt-eng)
  (dt-engine [_]
    dt-eng)
  (st-engine [_]
    st-eng))

(deftype MKLIntegerFactory [index-fact ^DataAccessor da vector-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    @index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  Factory
  (create-vector [this n _]
    (integer-block-vector this n))
  (vector-engine [_]
    vector-eng))

(let [index-fact (volatile! nil)]

  (def mkl-int
    (->MKLIntegerFactory index-fact int-accessor (->IntVectorEngine)))

  (def mkl-long
    (->MKLIntegerFactory index-fact long-accessor (->LongVectorEngine)))

  (def mkl-float
    (->MKLRealFactory index-fact float-accessor
                      (->FloatVectorEngine) (->FloatGEEngine) (->FloatTREngine) (->FloatSYEngine)
                      (->FloatGBEngine) (->FloatSBEngine) (->FloatTBEngine)
                      (->FloatSPEngine) (->FloatTPEngine)
                      (->FloatGDEngine) (->FloatGTEngine) (->FloatDTEngine) (->FloatSTEngine)))

  (def mkl-double
    (->MKLRealFactory index-fact double-accessor
                      (->DoubleVectorEngine) (->DoubleGEEngine) (->DoubleTREngine) (->DoubleSYEngine)
                      (->DoubleGBEngine) (->DoubleSBEngine) (->DoubleTBEngine)
                      (->DoubleSPEngine) (->DoubleTPEngine)
                      (->DoubleGDEngine) (->DoubleGTEngine) (->DoubleDTEngine) (->DoubleSTEngine)))

  (vreset! index-fact mkl-int))
