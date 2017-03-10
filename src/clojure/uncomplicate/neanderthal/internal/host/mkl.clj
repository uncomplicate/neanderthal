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
           [uncomplicate.neanderthal.internal.api DataAccessor BufferAccessor StripeNavigator Block RealVector]
           [uncomplicate.neanderthal.internal.host.buffer_block IntegerBlockVector RealBlockVector
            RealTRMatrix RealGEMatrix]))

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
  BLAS
  (swap [_ x y]
    (vector-method CBLAS/dswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
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
  BLASPlus
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
  BLAS
  (swap [_ x y]
    (vector-method CBLAS/sswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
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
  BLASPlus
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
  BLAS
  (swap [_ x y]
    (vector-method CBLAS/dswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/ddot ^RealBlockVector x ^RealBlockVector y))
  (nrm2 [_ x]
    (vector-method CBLAS/dnrm2 ^RealBlockVector x))
  (asum [_ x]
    (vector-method CBLAS/dasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/idamax ^RealBlockVector x))
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
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
  BLASPlus
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
    y))

(deftype FloatVectorEngine []
  BLAS
  (swap [_ x y]
    (vector-method CBLAS/sswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/sdot ^RealBlockVector x ^RealBlockVector y))
  (nrm2 [_ x]
    (vector-method CBLAS/snrm2 ^RealBlockVector x))
  (asum [_ x]
    (vector-method CBLAS/sasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/isamax ^RealBlockVector x))
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
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
  BLASPlus
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
    y))

;; ================= General Matrix Engines ====================================

(deftype DoubleGEEngine []
  BLAS
  (amax [this a]
    (ge-lan LAPACK/dlange (long \m) ^RealGEMatrix a))
  (swap [_ a b]
    (ge-swap CBLAS/dswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy MKL/domatcopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal MKL/dimatcopy alpha ^RealGEMatrix a)
    a)
  (asum [_ a]
    (ge-asum CBLAS/dasum ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/dlange (long \f) ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv CBLAS/dgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y)
   y)
  (mv [this a x]
   (ge-mv))
  (rank [_ alpha x y a]
    (ge-rank CBLAS/dger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a)
    a)
  (mm [_ alpha a b left]
   (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
   (ge-mm CBLAS/dgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c)
   c)
  BLASPlus
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
  (trf [_ a ipiv]
    (ge-trf LAPACK/dgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trs [_ a b ipiv]
    (ge-trs LAPACK/dgetrs ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b ipiv]
    (ge-sv LAPACK/dgesv ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/dgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/dgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/dgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/dgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/dgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (ls [_ a b]
    (ge-ls LAPACK/dgels ^RealGEMatrix a ^RealGEMatrix b)))

(deftype FloatGEEngine []
  BLAS
  (amax [this a]
    (ge-lan LAPACK/slange (long \m) ^RealGEMatrix a))
  (swap [_ a b]
    (ge-swap CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-copy MKL/somatcopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal MKL/simatcopy alpha ^RealGEMatrix a)
    a)
  (asum [_ a]
    (ge-asum CBLAS/sasum ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/slange (long \f) ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
   (ge-mv CBLAS/sgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y)
   y)
  (mv [this a x]
   (ge-mv))
  (rank [_ alpha x y a]
    (ge-rank CBLAS/sger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a)
    a)
  (mm [_ alpha a b left]
   (ge-mm alpha a b left))
  (mm [_ alpha a b beta c]
   (ge-mm CBLAS/sgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c)
   c)
  BLASPlus
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
  (trf [_ a ipiv]
    (ge-trf LAPACK/sgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trs [_ a b ipiv]
    (ge-trs LAPACK/sgetrs ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b ipiv]
    (ge-sv LAPACK/sgesv ^RealGEMatrix a ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/sgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/sgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/sgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/sgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/sgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (ls [_ a b]
    (ge-ls LAPACK/sgels ^RealGEMatrix a ^RealGEMatrix b)))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine []
  BLAS
  (amax [this a]
    (tr-lan LAPACK/dlantr (long \m) ^RealTRMatrix a))
  (swap [_ a b]
    (tr-swap ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/dswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-lacpy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) LAPACK/dlacpy CBLAS/dcopy
              ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-lascl LAPACK/dlascl alpha ^RealTRMatrix a)
    a)
  (nrm2 [_ a]
    (tr-lan LAPACK/dlantr (long \f) ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (tr-axpy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/daxpy
             alpha ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv))
  (mv [_ a x]
   (tr-mv CBLAS/dtrmv ^RealTRMatrix a ^RealBlockVector x)
   x)
  (mm [this alpha a b beta c]
   (tr-mm))
  (mm [_ alpha a b left]
   (tr-mm CBLAS/dtrmm alpha ^RealTRMatrix a ^RealGEMatrix b left)
   b)
  BLASPlus
  (set-all [_ alpha a]
    (tr-laset LAPACK/dlaset alpha alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/daxpby
              alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b))

(deftype FloatTREngine []
  BLAS
  (amax [this a]
    (tr-lan LAPACK/slantr (long \m) ^RealTRMatrix a))
  (swap [_ a b]
    (tr-swap ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/sswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-lacpy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) LAPACK/slacpy CBLAS/scopy
              ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-lascl LAPACK/slascl alpha ^RealTRMatrix a)
    a)
  (nrm2 [_ a]
    (tr-lan LAPACK/slantr (long \f) ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (tr-axpy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/saxpy alpha
             ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (mv [this alpha a x beta y]
   (tr-mv))
  (mv [_ a x]
   (tr-mv CBLAS/strmv ^RealTRMatrix a ^RealBlockVector x)
   x)
  (mm [this alpha a b beta c]
   (tr-mm))
  (mm [_ alpha a b left]
   (tr-mm CBLAS/strmm alpha ^RealTRMatrix a ^RealGEMatrix b left)
   b)
  BLASPlus
  (set-all [_ alpha a]
    (tr-laset LAPACK/slaset alpha alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/saxpby
              alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b))

;; =============== Factories ==================================================

(deftype CblasRealFactory [index-fact ^DataAccessor da vector-eng ge-eng tr-eng]
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
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng))

(deftype CblasIntegerFactory [index-fact ^DataAccessor da vector-eng]
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
    (->CblasIntegerFactory index-fact int-accessor (->IntVectorEngine)))

  (def mkl-long
    (->CblasIntegerFactory index-fact long-accessor (->LongVectorEngine)))

  (def mkl-float
    (->CblasRealFactory index-fact float-accessor
                        (->FloatVectorEngine) (->FloatGEEngine) (->FloatTREngine)))

  (def mkl-double
    (->CblasRealFactory index-fact double-accessor
                        (->DoubleVectorEngine) (->DoubleGEEngine) (->DoubleTREngine)))

  (vreset! index-fact mkl-int))
