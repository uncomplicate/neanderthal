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
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.neanderthal.internal.host
             [buffer-block :refer :all]
             [cblas :refer :all]
             [lapack :refer :all]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS MKL LAPACK]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.internal.api DataAccessor RealBufferAccessor
            Block RealVector StripeNavigator RealOrderNavigator BandNavigator]
           [uncomplicate.neanderthal.internal.host.buffer_block IntegerBlockVector RealBlockVector
            RealTRMatrix RealGEMatrix RealSYMatrix RealBandedMatrix]))

(defn ^:private not-available [kind]
  (throw (UnsupportedOperationException. "Operation not available for %s matrix")))

(def ^{:no-doc true :const true} INTEGER_UNSUPPORTED_MSG
  "\nInteger BLAS operations are not supported. Please transform data to float or double.\n")

;; =========== MKL-spicific routines ====================================================

(defmacro ge-copy [method a b]
  `(when (< 0 (.count ~a))
     (let [no-trans# (= (.order ~a) (.order ~b))
           rows# (if no-trans# (.sd ~b) (.fd ~b))
           cols# (if no-trans# (.fd ~b) (.sd ~b))]
       (~method (int \c)
        (int (if no-trans# \n \t)) rows# cols#
        1.0 (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)))))

(defmacro ge-scal [method alpha a]
  `(when (< 0 (.count ~a))
     (~method (int \c) (int \n) (.sd ~a) (.fd ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.stride ~a))))

(defmacro ge-trans [method a]
  `(when (< 0 (.count ~a))
     (~method (int \c) (int \t) (.sd ~a) (.fd ~a) 1.0 (.buffer ~a) (.offset ~a) (.stride ~a) (.fd ~a))))

(defmacro ge-axpby [method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [no-trans# (= (.order ~a) (.order ~b))
           rows# (if no-trans# (.sd ~a) (.fd ~a))
           cols# (if no-trans# (.fd ~a) (.sd ~a))]
       (~method (int \c)
        (int (if no-trans# \n \t)) (int \n) rows# cols#
        ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) ~beta (.buffer ~b) (.offset ~b) (.stride ~b)
        (.buffer ~b) (.offset ~b) (.stride ~b)))))

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

(def ^:private zero-matrix ^RealGEMatrix
  (->RealGEMatrix nil nil nil nil true (ByteBuffer/allocateDirect 0) 0 0 0 1 0 0 CBLAS/ORDER_COLUMN_MAJOR))

(deftype DoubleGEEngine []
  Blas
  (swap [_ a b]
    (ge-swap CBLAS/dswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy MKL/domatcopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal MKL/dimatcopy alpha ^RealGEMatrix a)
    a)
  (dot [_ a b]
    (ge-dot CBLAS/ddot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/dlange (long \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/dlange (long \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/dlange (long \I) ^RealGEMatrix a))
  (asum [_ a]
    (ge-sum CBLAS/dasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv CBLAS/dgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y)
   y)
  (mv [this a x]
   (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/dger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a)
    a)
  (mm [_ alpha a b _]
   (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
   (ge-mm CBLAS/dgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c)
   c)
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/dlange (long \M) ^RealGEMatrix a))
  (sum [_ a]
    (ge-sum CBLAS/dsum ^RealGEMatrix a))
  (set-all [_ alpha a]
    (ge-laset LAPACK/dlaset alpha alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b)
    b)
  (trans [_ a]
    (ge-trans MKL/dimatcopy ^RealGEMatrix a)
    a)
  Lapack
  (srt [_ a increasing]
    (ge-lasrt LAPACK/dlasrt ^RealGEMatrix a increasing)
    a)
  (trf [_ a ipiv]
    (ge-trf LAPACK/dgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (tri [_ a ipiv]
    (ge-tri LAPACK/dgetri ^RealGEMatrix a ^IntegerBlockVector ipiv)
    a)
  (trs [_ a b ipiv]
    (ge-trs LAPACK/dgetrs ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv)
    b)
  (sv [_ a b pure]
    (ge-sv LAPACK/dgesv ^RealGEMatrix a ^RealGEMatrix b pure)
    b)
  (con [_ lu nrm nrm1?]
    (let [da (data-accessor lu)]
      (ge-con ^RealBufferAccessor da LAPACK/dgecon ^RealGEMatrix lu nrm nrm1?)))
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
  (svd [_ a s u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealBlockVector s ^RealGEMatrix u ^RealGEMatrix vt
              ^RealBlockVector superb)))
  (svd [_ a s superb]
    (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealBlockVector s
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealBlockVector superb)))

(deftype FloatGEEngine []
  Blas
  (swap [_ a b]
    (ge-swap CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy MKL/somatcopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal MKL/simatcopy alpha ^RealGEMatrix a)
    a)
  (dot [_ a b]
    (ge-dot CBLAS/sdot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/slange (long \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/slange (long \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/slange (long \I) ^RealGEMatrix a))
  (asum [_ a]
    (ge-sum CBLAS/sasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv CBLAS/sgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y)
   y)
  (mv [this a x]
   (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/sger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a)
    a)
  (mm [_ alpha a b _]
   (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
   (ge-mm CBLAS/sgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c)
   c)
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/slange (long \M) ^RealGEMatrix a))
  (sum [_ a]
    (ge-sum CBLAS/ssum ^RealGEMatrix a))
  (set-all [_ alpha a]
    (ge-laset LAPACK/slaset alpha alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b)
    b)
  (trans [_ a]
    (ge-trans MKL/simatcopy ^RealGEMatrix a)
    a)
  Lapack
  (srt [_ a increasing]
    (ge-lasrt LAPACK/slasrt ^RealGEMatrix a increasing)
    a)
  (trf [_ a ipiv]
    (ge-trf LAPACK/sgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (tri [_ a ipiv]
    (ge-tri LAPACK/sgetri ^RealGEMatrix a ^IntegerBlockVector ipiv)
    a)
  (trs [_ a b ipiv]
    (ge-trs LAPACK/sgetrs ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv)
    b)
  (sv [_ a b pure]
    (ge-sv LAPACK/sgesv ^RealGEMatrix a ^RealGEMatrix b pure)
    b)
  (con [_ lu nrm nrm1?]
    (let [da (data-accessor lu)]
      (ge-con ^RealBufferAccessor da LAPACK/sgecon ^RealGEMatrix lu nrm nrm1?)))
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
  (svd [_ a s u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealBlockVector s ^RealGEMatrix u ^RealGEMatrix vt
              ^RealBlockVector superb)))
  (svd [_ a s superb]
    (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealBlockVector s
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealBlockVector superb)))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine []
  Blas
  (swap [_ a b]
    (uplo-swap CBLAS/dswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-lascl LAPACK/dlascl alpha ^RealTRMatrix a)
    a)
  (dot [_ a b]
    (tr-dot CBLAS/ddot ^RealTRMatrix a ^RealTRMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/dlantr (long \O) ^RealTRMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/dlantr (long \F) ^RealTRMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/dlantr (long \I) ^RealTRMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/dasum ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (uplo-axpy CBLAS/daxpy alpha ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv a))
  (mv [_ a x]
   (tr-mv CBLAS/dtrmv ^RealTRMatrix a ^RealBlockVector x)
   x)
  (mm [this alpha a b beta c _]
   (tr-mm a))
  (mm [_ alpha a b left]
   (tr-mm CBLAS/dtrmm alpha ^RealTRMatrix a ^RealGEMatrix b left)
   b)
  BlasPlus
  (amax [this a]
    (tr-lan LAPACK/dlantr (long \M) ^RealTRMatrix a))
  (sum [_ a]
    (tr-sum CBLAS/dsum ^RealTRMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/dlaset alpha alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (uplo-axpby MKL/daxpby alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b)
  Lapack
  (srt [_ a increasing]
    (uplo-lasrt LAPACK/dlasrt ^RealTRMatrix a increasing)
    a)
  (tri [_ a]
    (tr-tri LAPACK/dtrtri ^RealTRMatrix a)
    a)
  (trs [_ a b]
    (tr-trs LAPACK/dtrtrs ^RealTRMatrix a ^RealGEMatrix b)
    b)
  (sv [this a b _]
    (tr-sv CBLAS/dtrsm ^RealTRMatrix a ^RealGEMatrix b)
    b)
  (con [_ a nrm1?]
    (let [da (data-accessor a)]
      (tr-con ^RealBufferAccessor da LAPACK/dtrcon ^RealTRMatrix a nrm1?))))

(deftype FloatTREngine []
  Blas
  (swap [_ a b]
    (uplo-swap CBLAS/sswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-lacpy LAPACK/slacpy CBLAS/scopy ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-lascl LAPACK/slascl alpha ^RealTRMatrix a)
    a)
  (dot [_ a b]
    (tr-dot CBLAS/sdot ^RealTRMatrix a ^RealTRMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/slantr (long \O) ^RealTRMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/slantr (long \F) ^RealTRMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/slantr (long \I) ^RealTRMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/sasum ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (uplo-axpy CBLAS/saxpy alpha ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv a))
  (mv [_ a x]
   (tr-mv CBLAS/strmv ^RealTRMatrix a ^RealBlockVector x)
   x)
  (mm [this alpha a b beta c _]
   (tr-mm a))
  (mm [_ alpha a b left]
   (tr-mm CBLAS/strmm alpha ^RealTRMatrix a ^RealGEMatrix b left)
   b)
  BlasPlus
  (amax [this a]
    (tr-lan LAPACK/slantr (long \M) ^RealTRMatrix a))
  (sum [_ a]
    (tr-sum  CBLAS/ssum ^RealTRMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/slaset alpha alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (uplo-axpby MKL/saxpby alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b)
  Lapack
  (srt [_ a increasing]
    (uplo-lasrt LAPACK/slasrt ^RealTRMatrix a increasing)
    a)
  (tri [_ a]
    (tr-tri LAPACK/strtri ^RealTRMatrix a)
    a)
  (trs [_ a b]
    (tr-trs LAPACK/strtrs ^RealTRMatrix a ^RealGEMatrix b)
    b)
  (sv [this a b _]
    (tr-sv CBLAS/strsm ^RealTRMatrix a ^RealGEMatrix b)
    b)
  (con [_ a nrm1?]
    (let [da (data-accessor a)]
      (tr-con ^RealBufferAccessor da LAPACK/strcon ^RealTRMatrix a nrm1?))))

;; =============== Symmetric Matrix Engines ===================================

(deftype DoubleSYEngine []
  Blas
  (swap [_ a b]
    (uplo-swap CBLAS/dswap ^RealSYMatrix a ^RealSYMatrix b)
    a)
  (copy [_ a b]
    (sy-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealSYMatrix a ^RealSYMatrix b)
    b)
  (scal [_ alpha a]
    (sy-lascl LAPACK/dlascl alpha ^RealSYMatrix a)
    a)
  (dot [_ a b]
    (sy-dot CBLAS/ddot ^RealSYMatrix a ^RealSYMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/dlansy (long \O) ^RealSYMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/dlansy (long \F) ^RealSYMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/dlansy (long \I) ^RealSYMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/dasum ^RealSYMatrix a))
  (axpy [_ alpha a b]
    (uplo-axpy CBLAS/daxpy alpha ^RealSYMatrix a ^RealSYMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/dsymv alpha ^RealSYMatrix a ^RealBlockVector x beta ^RealBlockVector y)
    y)
  (mv [_ a x]
    (sy-mv a))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/dsymm alpha ^RealSYMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left)
    c)
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/dlansy (long \M) ^RealSYMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/dsum ^RealSYMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/dlaset alpha alpha ^RealSYMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (uplo-axpby MKL/daxpby alpha ^RealSYMatrix a beta ^RealSYMatrix b)
    b)
  Lapack
  (srt [_ a increasing]
    (uplo-lasrt LAPACK/dlasrt ^RealSYMatrix a increasing)
    a)
  (trf [_ a ipiv]
    (sy-trf LAPACK/dsytrf ^RealSYMatrix a ^IntegerBlockVector ipiv))
  (tri [_ a ipiv]
    (sy-tri LAPACK/dsytri ^RealSYMatrix a ^IntegerBlockVector ipiv)
    a)
  (trs [_ a b ipiv]
    (sy-trs LAPACK/dsytrs ^RealSYMatrix a ^RealSYMatrix b ^IntegerBlockVector ipiv)
    b)
  (sv [_ a b pure]
    (sy-sv LAPACK/dsysv ^RealSYMatrix a ^RealSYMatrix b pure)
    b)
  (con [_ ldl ipiv nrm _]
    (let [da (data-accessor ldl)]
      (sy-con ^RealBufferAccessor da LAPACK/dsycon ^RealSYMatrix ldl ^IntegerBlockVector ipiv nrm))))

(deftype FloatSYEngine []
  Blas
  (swap [_ a b]
    (uplo-swap CBLAS/sswap ^RealSYMatrix a ^RealSYMatrix b)
    a)
  (copy [_ a b]
    (sy-lacpy LAPACK/slacpy CBLAS/scopy ^RealSYMatrix a ^RealSYMatrix b)
    b)
  (scal [_ alpha a]
    (sy-lascl LAPACK/slascl alpha ^RealSYMatrix a)
    a)
  (dot [_ a b]
    (sy-dot CBLAS/sdot ^RealSYMatrix a ^RealSYMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/slansy (long \O) ^RealSYMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/slansy (long \F) ^RealSYMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/slansy (long \I) ^RealSYMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/sasum ^RealSYMatrix a))
  (axpy [_ alpha a b]
    (uplo-axpy CBLAS/saxpy alpha ^RealSYMatrix a ^RealSYMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/ssymv alpha ^RealSYMatrix a ^RealBlockVector x beta ^RealBlockVector y)
    y)
  (mv [_ a x]
    (sy-mv a))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/ssymm alpha ^RealSYMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left)
    c)
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/slansy (long \M) ^RealSYMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/ssum ^RealSYMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/slaset alpha alpha ^RealSYMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (uplo-axpby MKL/saxpby alpha ^RealSYMatrix a beta ^RealSYMatrix b)
    b)
  Lapack
  (srt [_ a increasing]
    (uplo-lasrt LAPACK/slasrt ^RealSYMatrix a increasing)
    a)
  (trf [_ a ipiv]
    (sy-trf LAPACK/ssytrf ^RealSYMatrix a ^IntegerBlockVector ipiv))
  (tri [_ a ipiv]
    (sy-tri LAPACK/ssytri ^RealSYMatrix a ^IntegerBlockVector ipiv)
    a)
  (trs [_ a b ipiv]
    (sy-trs LAPACK/ssytrs ^RealSYMatrix a ^RealSYMatrix b ^IntegerBlockVector ipiv)
    b)
  (sv [_ a b pure]
    (sy-sv LAPACK/ssysv ^RealSYMatrix a ^RealSYMatrix b pure)
    b)
  (con [_ ldl ipiv nrm _]
    (let [da (data-accessor ldl)]
      (sy-con ^RealBufferAccessor da LAPACK/ssycon ^RealSYMatrix ldl ^IntegerBlockVector ipiv nrm))))

;; =============== Banded Matrix Engines ===================================

(deftype DoubleGBEngine []
  Blas
  (swap [_ a b]
    (gb-map swap a b)
    a)
  (copy [_ a b]
    (gb-map copy a b))
  (scal [_ alpha a]
    (gb-scalset scal alpha a))
  (dot [_ a b]
    (gb-reduce dot a b))
  (nrm1 [_ a]
    (gb-lan LAPACK/dlangb (long \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/dlangb (long \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/dlangb (long \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-reduce asum a))
  (axpy [_ alpha a b]
    (gb-axpy alpha a b))
  (mv [this alpha a x beta y]
    (gb-mv CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (gb-mv a))
  (mm [this alpha a b beta c _]
    (gb-mm CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c true))
  (mm [_ alpha a b left]
    (gb-mm a))
  BlasPlus
  (sum [_ a]
    (gb-reduce sum a))
  (amax [_ a]
    (gb-lan LAPACK/dlangb (long \M) ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-scalset set-all alpha a))
  (axpby [_ alpha a beta b]
    (gb-axpby alpha a beta b)))

;; =============== Factories ==================================================

(deftype MKLRealFactory [index-fact ^DataAccessor da vector-eng ge-eng tr-eng sy-eng gb-eng tb-eng sb-eng]
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
  (create-ge [this m n ord _]
    (real-ge-matrix this m n ord))
  (create-tr [this n ord uplo diag _]
    (real-tr-matrix this n ord uplo diag))
  (create-sy [this n ord uplo _]
    (real-sy-matrix this n ord uplo))
  (create-gb [this m n kl ku ord _]
    (real-banded-matrix this m n kl ku ord gb-eng))
  (create-tb [this n k ord uplo _]
    (let [[kl ku] (if (= UPPER uplo) [0 k] [k 0])]
      (real-banded-matrix this n n kl ku ord tb-eng)))
  (create-sb [this n k ord uplo _]
    (let [[kl ku] (if (= UPPER uplo) [0 k] [k 0])]
      (real-banded-matrix this n n kl ku ord sb-eng)))
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
  (tb-engine [_]
    tb-eng)
  (sb-engine [_]
    tb-eng))

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
                      nil nil nil))

  (def mkl-double
    (->MKLRealFactory index-fact double-accessor
                      (->DoubleVectorEngine) (->DoubleGEEngine) (->DoubleTREngine) (->DoubleSYEngine)
                      (->DoubleGBEngine) nil nil))

  (vreset! index-fact mkl-int))
