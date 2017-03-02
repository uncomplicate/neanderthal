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
             [cblas :refer :all]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS MKL]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.internal.api DataAccessor BufferAccessor StripeNavigator Block RealVector]
           [uncomplicate.neanderthal.internal.host.buffer_block RealBlockVector RealTRMatrix RealGEMatrix]))

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
    (MKL/dset (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
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
    (MKL/sset (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
    x)
  (axpby [_ alpha x beta y]
    (MKL/saxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y))

;; ================= General Matrix Engines ====================================

(deftype DoubleGEEngine []
  BLAS
  (swap [_ a b]
    (ge-swap-copy CBLAS/dswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-swap-copy CBLAS/dcopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal-set CBLAS/dscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy CBLAS/daxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
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
    (ge-scal-set MKL/dset alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/daxpby alpha ^RealGEMatrix a beta ^RealGEMatrix b)
    b))

(deftype FloatGEEngine []
  BLAS
  (swap [_ a b]
    (ge-swap-copy CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-swap-copy CBLAS/scopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal-set CBLAS/sscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy CBLAS/saxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
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
    (ge-scal-set MKL/sset alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/saxpby alpha ^RealGEMatrix a beta ^RealGEMatrix b)
    b))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine []
  BLAS
  (swap [_ a b]
    (tr-swap-copy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/dswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-swap-copy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/dcopy ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/dscal alpha ^RealTRMatrix a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/daxpy alpha ^RealTRMatrix a ^RealTRMatrix b)
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
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/dset alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/daxpby
              alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b))

(deftype FloatTREngine []
  BLAS
  (swap [_ a b]
    (tr-swap-copy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/sswap ^RealTRMatrix a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-swap-copy ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/scopy ^RealTRMatrix a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/sscal alpha ^RealTRMatrix a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy ^StripeNavigator  (.stripe-nav ^RealTRMatrix a) CBLAS/saxpy alpha ^RealTRMatrix a ^RealTRMatrix b)
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
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/sset alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) MKL/saxpby
              alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b))

;; =============== Factories ==================================================

(deftype CblasFactory [^DataAccessor da vector-eng ge-eng tr-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
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

(def mkl-float
  (->CblasFactory (->FloatBufferAccessor) (->FloatVectorEngine)
                  (->FloatGEEngine) (->FloatTREngine)))

(def mkl-double
  (->CblasFactory (->DoubleBufferAccessor) (->DoubleVectorEngine)
                  (->DoubleGEEngine) (->DoubleTREngine)))
