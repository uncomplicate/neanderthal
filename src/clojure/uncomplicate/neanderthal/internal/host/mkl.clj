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
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.internal.host
             [buffer-block :refer :all]
             [cblas :refer :all]
             [lapack :refer :all]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS MKL LAPACK]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.internal.api DataAccessor RealBufferAccessor
            Block RealVector Region LayoutNavigator DenseStorage]
           [uncomplicate.neanderthal.internal.host.buffer_block IntegerBlockVector RealBlockVector
            RealGEMatrix RealUploMatrix RealBandedMatrix RealPackedMatrix]))

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
    (vector-rotm CBLAS/drotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param)
    x)
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/drotmg ^RealBlockVector d1d2xy ^RealBlockVector param)
    param)
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
    (vctr-laset LAPACK/dlaset alpha ^RealBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (MKL/daxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/dlasrt ^RealBlockVector x increasing)
    x))

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
    (vector-rot CBLAS/srot ^RealBlockVector x ^RealBlockVector y c s)
    x)
  (rotg [_ abcs]
    (CBLAS/srotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
    abcs)
  (rotm [_ x y param]
    (vector-rotm CBLAS/srotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param)
    x)
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/srotmg ^RealBlockVector d1d2xy ^RealBlockVector param)
    param)
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
    (vctr-laset LAPACK/slaset alpha ^RealBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (MKL/saxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/slasrt ^RealBlockVector x increasing)
    x))

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
  (mv [this a x]
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
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/dgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealBandedMatrix sigma ^RealGEMatrix u ^RealGEMatrix vt
              ^RealBandedMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealBandedMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealBandedMatrix superb))
  TRF
  (create-trf [_ a pure]
    (matrix-create-trf matrix-lu a pure)))

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
  (mv [this a x]
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
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/sgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealBandedMatrix sigma ^RealGEMatrix u ^RealGEMatrix vt
              ^RealBandedMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealBandedMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealBandedMatrix superb))
  TRF
  (create-trf [_ a master]
    (matrix-create-trf matrix-lu a master)))

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
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/dtrmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/dtrmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [this a]
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
  (sv [this a b _]
    (tr-sv CBLAS/dtrsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/dtrcon ^RealUploMatrix a nrm1?))
  TRF
  (create-trf [_ a _]
    a))

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
  (mv [this alpha a x beta y]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/strmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [this alpha a b beta c _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/strmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [this a]
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
  (sv [this a b _]
    (tr-sv CBLAS/strsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/strcon ^RealUploMatrix a nrm1?))
  TRF
  (create-trf [_ a _]
    a))

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
    (sy-con LAPACK/dpocon ^RealUploMatrix gg nrm))
  TRF
  (create-trf [_ a pure]
    (sy-cholesky-lu LAPACK/dpotrf ^RealUploMatrix a pure))
  (create-ptrf [_ a]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a false)))

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
    (sy-con LAPACK/spocon ^RealUploMatrix gg nrm))
  TRF
  (create-trf [_ a pure]
    (sy-cholesky-lu LAPACK/spotrf ^RealUploMatrix a pure))
  (create-ptrf [_ a]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a false)))

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
  (mv [this alpha a x beta y]
    (gb-mv CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [this alpha a b beta c left]
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
    (gb-con LAPACK/dgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?))
  TRF
  (create-trf [_ a pure]
    (matrix-create-trf matrix-lu ^RealBandedMatrix a pure))
  (create-ptrf [_ _]
    (dragan-says-ex "Cholesky factorization is not available for GB matrices.")))

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
  (mv [this alpha a x beta y]
    (gb-mv CBLAS/sgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [this alpha a b beta c left]
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
    (gb-con LAPACK/sgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?))
  TRF
  (create-trf [this a pure]
    (matrix-create-trf matrix-lu ^RealBandedMatrix a pure))
  (create-ptrf [_ _]
    (dragan-says-ex "Cholesky factorization is not available for GB matrices.")))

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
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/dlangb CBLAS/dnrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [this alpha a x beta y]
    (sb-mv CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [this alpha a b beta c left]
    (sb-mm CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/dsum ^RealBandedMatrix a));;TODO 2*
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
    (sb-con LAPACK/dpbcon ^RealBandedMatrix gg nrm))
  TRF
  (create-trf [_ a pure]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a pure))
  (create-ptrf [_ a]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a false)))

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
    (gb-lan LAPACK/slangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/slangb CBLAS/snrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [this alpha a x beta y]
    (sb-mv CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [this alpha a b beta c left]
    (sb-mm CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/ssum ^RealBandedMatrix a));;TODO 2*
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
    (sb-con LAPACK/spbcon ^RealBandedMatrix gg nrm))
  TRF
  (create-trf [_ a pure]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a pure))
  (create-ptrf [_ a]
    (matrix-create-trf matrix-cholesky ^RealBandedMatrix a false)))

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
    (let [da ^RealBufferAccessor (data-accessor a)]
      (tp-dot CBLAS/ddot da ^RealPackedMatrix a ^RealPackedMatrix b)))
  (nrm1 [this a]
    (ex-info "TODO" nil))
  (nrm2 [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (tp-nrm2 CBLAS/dnrm2 da ^RealPackedMatrix a)))
  (nrmi [_ a]
    (ex-info "TODO" nil))
  (asum [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (tp-sum CBLAS/dasum Math/abs da ^RealPackedMatrix a)))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [this alpha a x beta y]
    (tp-mv a))
  (mv [_ a x]
    (tp-mv CBLAS/dtpmv ^RealPackedMatrix a ^RealBlockVector x))
  (mm [this _ a _ _ _ _]
    (tp-mm a))
  (mm [_ alpha a b left]
    (tp-mm CBLAS/dtpmv alpha ^RealPackedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (sum [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (tp-sum CBLAS/dsum double da ^RealPackedMatrix a)))
  (amax [_ a]
    (packed-amax CBLAS/idamax ^RealBufferAccessor (data-accessor a) ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  TRF
  (create-trf [_ a _]
    a))

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
    (let [da ^RealBufferAccessor (data-accessor a)]
      (sp-dot CBLAS/ddot da ^RealPackedMatrix a ^RealPackedMatrix b)))
  (nrm1 [this a]
    (ex-info "TODO" nil))
  (nrm2 [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (sp-nrm2 CBLAS/dnrm2 da ^RealPackedMatrix a)))
  (nrmi [_ a]
    (ex-info "TODO" nil))
  (asum [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (sp-sum CBLAS/dasum Math/abs da ^RealPackedMatrix a)))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [this alpha a x beta y]
    (sp-mv CBLAS/dspmv alpha ^RealPackedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sp-mv a))
  (mm [this alpha a b beta c left]
    (sp-mm CBLAS/dspmv alpha ^RealPackedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sp-mm a))
  BlasPlus
  (sum [_ a]
    (let [da ^RealBufferAccessor (data-accessor a)]
      (sp-sum CBLAS/dsum double da ^RealPackedMatrix a)))
  (amax [_ a]
    (packed-amax CBLAS/idamax ^RealBufferAccessor (data-accessor a) ^RealPackedMatrix a))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  TRF
  (create-trf [this a master]
    (matrix-create-trf matrix-lu this ^RealPackedMatrix a master)))

;; =============== Factories ==================================================

(deftype MKLRealFactory [index-fact ^DataAccessor da vector-eng
                         ge-eng tr-eng sy-eng gb-eng sb-eng tb-eng tp-eng sp-eng]
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
  (create-banded [this m n kl ku mat-type column? _]
    (real-banded-matrix this m n kl ku column? mat-type))
  (create-gb [this m n kl ku lower? _]
    (real-banded-matrix this m n kl ku lower?))
  (create-tb [this n k column? lower? diag-unit? _]
    (real-tb-matrix this n k column? lower? diag-unit?))
  (create-sb [this n k column? lower? _]
    (real-sb-matrix this n k column? lower?))
  (create-packed [this n mat-type column? lower? diag-unit? _]
    (case mat-type
      :tp (real-packed-matrix this n column? lower? diag-unit?)
      :sy (real-packed-matrix this n column? lower?)
      (throw (ex-info "TODO" {}))))
  (create-tp [this n column? lower? diag-unit? _]
    (real-packed-matrix this n column? lower? diag-unit?))
  (create-sp [this n column? lower? _]
    (real-packed-matrix this n column? lower?))
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
    sp-eng))

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
                      (->FloatGBEngine) (->FloatSBEngine) nil nil nil))

  (def mkl-double
    (->MKLRealFactory index-fact double-accessor
                      (->DoubleVectorEngine) (->DoubleGEEngine) (->DoubleTREngine) (->DoubleSYEngine)
                      (->DoubleGBEngine) (->DoubleSBEngine) nil (->DoubleTPEngine) (->DoubleSPEngine)))

  (vreset! index-fact mkl-int))
