;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.impl.cblas
  (:require [vertigo.bytes :refer [direct-buffer]]
            [uncomplicate.commons.core :refer [with-release let-release]]
            [uncomplicate.neanderthal
             [protocols :refer :all]]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.protocols BLAS BLASPlus DataAccessor BufferAccessor
            StripeNavigator Block RealVector]
           [uncomplicate.neanderthal.impl.buffer_block RealBlockVector RealTRMatrix RealGEMatrix]))

;; =============== Common vector engine  macros and functions ==================

(defmacro ^:private vector-rot
  ([method x y c s]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y) ~c ~s)))

(defmacro ^:private vector-rotm [method x y param]
  `(when (and (< 0 (.dim ~x)) (< 0 (.dim ~y)))
     (if (= 1 (.stride ~param))
       (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
        (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~param))
       (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride ~param)))))))

(defmacro ^:private vector-rotmg [method d1d2xy param]
  `(if (= 1 (.stride ~param))
     (~method (.buffer ~d1d2xy) (.stride ~d1d2xy) (.offset ~d1d2xy) (.buffer ~param))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride ~param))))))

(defmacro ^:private vector-method
  ([method x]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)))
  ([method x y]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([method x y z]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x) (.buffer ~y) (.offset ~y) (.stride ~y)
     (.buffer ~z) (.offset ~z) (.stride ~z))))

(defn ^:private vector-imax [^RealVector x]
  (let [cnt (.dim x)]
    (loop [i 1 max-idx 0 max-val (.entry x 0)]
      (if (< i cnt)
        (let [v (.entry x i)]
          (if (< max-val v)
            (recur (inc i) i v)
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn ^:private vector-imin [^RealVector x]
  (let [cnt (.dim x)]
    (loop [i 1 min-idx 0 min-val (.entry x 0)]
      (if (< i cnt)
        (let [v (.entry x i)]
          (if (< v min-val)
            (recur (inc i) i v)
            (recur (inc i) min-idx min-val)))
        min-idx))))

;; =============== Common GE matrix macros and functions =======================

(defmacro ^:private ge-swap-copy [method a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# (.sd ~b) ld-a# ld-b#)
           (~method (.count ~a) buff-a# offset-a# 1 buff-b# offset-b# 1)
           (dotimes [j# (.fd ~a)]
             (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1)))
         (dotimes [j# (.fd ~a)]
           (~method sd-a# buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# j#) fd-b#))))))

(defmacro ^:private ge-scal-set [method alpha a]
  `(when (< 0 (.count ~a))
     (let [ld# (.stride ~a)
           sd# (.sd ~a)
           offset# (.offset ~a)
           buff# (.buffer ~a)]
       (if (= sd# ld#)
         (~method (.count ~a) ~alpha buff# offset# 1)
         (dotimes [j# (.fd ~a)]
           (~method sd# ~alpha buff# (+ offset# (* ld# j#)) 1))))))

(defmacro ^:private ge-axpy [method alpha a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (~method (.count ~a) ~alpha buff-a# offset-a# 1 buff-b# offset-b# 1)
           (dotimes [j# fd-a#]
             (~method sd-a# ~alpha buff-a# (+ offset-a# (* ld-a# j#)) 1 buff-b# (+ offset-b# (* ld-b# j#)) 1)))
         (dotimes [j# fd-b#]
           (~method sd-a# ~alpha buff-a# (+ offset-a# j#) fd-a# buff-b# (+ offset-b# (* ld-b# j#)) 1))))))

(defmacro ^:private ge-axpby [method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (~method (.count ~a) ~alpha buff-a# offset-a# 1 ~beta buff-b# offset-b# 1)
           (dotimes [j# fd-a#]
             (~method sd-a# ~alpha buff-a# (+ offset-a# (* ld-a# j#)) 1 ~beta buff-b# (+ offset-b# (* ld-b# j#)) 1)))
         (dotimes [j# fd-b#]
           (~method sd-a# ~alpha buff-a# (+ offset-a# j#) fd-a# ~beta buff-b# (+ offset-b# (* ld-b# j#)) 1))))))

(defmacro ^:private ge-mv
  ([method alpha a x beta y]
   `(~method (.order ~a) NO_TRANS (.mrows ~a) (.ncols ~a)
     ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~x) (.offset ~x) (.stride ~x) ~beta (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([]
   `(throw (IllegalArgumentException. "In-place mv! is not supported for GE matrices."))))

(defmacro ^:private ge-rank [method alpha x y a]
  `(~method (.order ~a) (.mrows ~a) (.ncols ~a) ~alpha (.buffer ~x) (.offset ~x) (.stride ~x)
    (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~a) (.offset ~a) (.stride ~a)))

(defmacro ^:private ge-mm
  ([alpha a b left]
   `(if ~left
      (.mm (engine ~b) ~alpha ~b ~a false)
      (throw (IllegalArgumentException. "In-place mm! is not supported for GE matrices."))))
  ([method alpha a b beta c]
   `(~method (.order ~c)
     (if (= (.order ~a) (.order ~c)) NO_TRANS TRANS) (if (= (.order ~b) (.order ~c)) NO_TRANS TRANS)
     (.mrows ~a) (.ncols ~b) (.ncols ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~b) (.offset ~b) (.stride ~b) ~beta (.buffer ~c) (.offset ~c) (.stride ~c))))

;; =============== Common TR matrix macros and functions ==========================

(defmacro ^:private tr-swap-copy [stripe-nav method a b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# j# (* ld-b# start#)) n#)))))))

(defmacro ^:private tr-scal-set [stripe-nav method alpha a]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld# (.stride ~a)
           offset# (.offset ~a)
           buff# (.buffer ~a)]
       (dotimes [j# n#]
         (let [start# (.start ~stripe-nav n# j#)
               n-j# (- (.end ~stripe-nav n# j#) start#)]
           (~method n-j# ~alpha buff# (+ offset# (* ld# j#) start#) 1))))))

(defmacro ^:private tr-axpy [stripe-nav method alpha a b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j# ~alpha
              buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j# ~alpha
              buff-a# (+ offset-a# j# (* ld-a# start#)) n#
              buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))))))

(defmacro ^:private tr-axpby [stripe-nav method alpha a beta b]
  `(when (< 0 (.count ~a))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j#
              ~alpha buff-a# (+ offset-a# (* ld-a# j#) start#) 1
              ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))
         (dotimes [j# n#]
           (let [start# (.start ~stripe-nav n# j#)
                 n-j# (- (.end ~stripe-nav n# j#) start#)]
             (~method n-j#
              ~alpha buff-a# (+ offset-a# j# (* ld-a# start#)) n#
              ~beta buff-b# (+ offset-b# (* ld-b# j#) start#) 1)))))))

(defmacro ^:private tr-mv
  ([method a x]
   `(~method (.order ~a) (.uplo ~a) NO_TRANS (.diag ~a) (.ncols ~a)
     (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~x) (.offset ~x) (.stride ~x)))
  ([]
   `(throw (IllegalArgumentException. "Only in-place mv! is supported for TR matrices."))))

(defmacro ^:private tr-mm
  ([method alpha a b left]
   `(~method (.order ~b) (if ~left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT) (.uplo ~a)
     (if (= (.order ~a) (.order ~b)) NO_TRANS TRANS) (.diag ~a) (.mrows ~b) (.ncols ~b)
     ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)))
  ([]
   `(throw (IllegalArgumentException. "Only in-place mm! is supported for TR matrices."))))

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
    (CBLAS/dscal (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x)
                 (.buffer ^RealBlockVector y) (.offset y) (.stride y))
    y)
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^RealBlockVector x) (+ kx (.offset x)) (.stride x)
                 (.buffer ^RealBlockVector y) (+ ky (.offset y)) (.stride y))
    y)
  (sum [_ x]
    (vector-method CBLAS/dsum ^RealBlockVector x))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set [_ alpha x]
    (CBLAS/dset (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x))
    x)
  (axpby [_ alpha x beta y]
    (CBLAS/daxpby (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x)
                  beta (.buffer ^RealBlockVector y) (.offset y) (.stride y))
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
    (CBLAS/sscal (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x)
                 (.buffer ^RealBlockVector y) (.offset y) (.stride y))
    y)
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^RealBlockVector x) (+ kx (.offset x)) (.stride x)
                 (.buffer ^RealBlockVector y) (+ ky (.offset y)) (.stride y))
    y)
  (sum [_ x]
    (vector-method CBLAS/ssum ^RealBlockVector x))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set [_ alpha x]
    (CBLAS/sset (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x))
    x)
  (axpby [_ alpha x beta y]
    (CBLAS/saxpby (.dim ^RealBlockVector x) alpha (.buffer x) (.offset x) (.stride x)
                  beta (.buffer ^RealBlockVector y) (.offset y) (.stride y))
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
  (set [_ alpha a]
    (ge-scal-set CBLAS/dset alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby CBLAS/daxpby alpha ^RealGEMatrix a beta ^RealGEMatrix b)
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
  (set [_ alpha a]
    (ge-scal-set CBLAS/sset alpha ^RealGEMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (ge-axpby CBLAS/saxpby alpha ^RealGEMatrix a beta ^RealGEMatrix b)
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
  (set [_ alpha a]
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/dset alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/daxpby
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
  (set [_ alpha a]
    (tr-scal-set ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/sset alpha ^RealTRMatrix a)
    a)
  (axpby [_ alpha a beta b]
    (tr-axpby ^StripeNavigator (.stripe-nav ^RealTRMatrix a) CBLAS/saxpby
              alpha ^RealTRMatrix a beta ^RealTRMatrix b)
    b))

;; =============== Factories ==================================================

(deftype CblasFactory [^DataAccessor da ^BLASPlus vector-eng ^BLAS ge-eng ^BLAS tr-eng]
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

(def cblas-float
  (->CblasFactory (->FloatBufferAccessor) (->FloatVectorEngine)
                  (->FloatGEEngine) (->FloatTREngine)))

(def cblas-double
  (->CblasFactory (->DoubleBufferAccessor) (->DoubleVectorEngine)
                  (->DoubleGEEngine) (->DoubleTREngine)))
