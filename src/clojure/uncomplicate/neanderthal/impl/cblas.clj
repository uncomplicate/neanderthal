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
             [protocols :refer :all]
             [core :refer [dim copy copy!]]
             [block :refer [order buffer offset stride]]]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [clojure.lang IFn$OLO]
           [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus DataAccessor BufferAccessor Block
            Vector RealVector ContiguousBlock Matrix GEMatrix TRMatrix]
           [uncomplicate.neanderthal.impl.buffer_block
            RealBlockVector RealTRMatrix RealGEMatrix]))

;; =============== Common vector engine  macros and functions ==================

(defmacro vector-rotm [method x y param]
  `(if (= 1 (.stride ~param))
     (~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
      (.buffer ~y) (.offset ~y) (.stride ~y) (.buffer ~param))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride ~param))))))

(defmacro vector-rotmg [method d1d2xy param]
  `(if (= 1 (.stride ~param))
     (~method (.buffer ~d1d2xy) (.stride ~d1d2xy) (.offset ~d1d2xy) (.buffer ~param))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (.stride ~param))))))

(defmacro vector-method
  ([method x]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)))
  ([method x y]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
     (.buffer ~y) (.offset ~y) (.stride ~y)))
  ([method x y z]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
     (.buffer ~y) (.offset ~y) (.stride ~y)
     (.buffer ~z) (.offset ~z) (.stride ~z))))

(defmacro vector-rot
  ([method x y c s]
   `(~method (.dim ~x) (.buffer ~x) (.offset ~x) (.stride ~x)
     (.buffer ~y) (.offset ~y) (.stride ~y) ~c ~s)))

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
           (~method (.count ~a) buff-a# offset-a#  1 buff-b# offset-b# 1)
           (dotimes [i# fd-a#]
             (~method sd-a# buff-a# (+ offset-a# (* ld-a# i#)) 1
              buff-b# (+ offset-b# (* ld-b# i#)) 1)))
         (dotimes [i# fd-a#]
           (~method sd-a# buff-a# (+ offset-a# (* ld-a# i#)) 1
            buff-b# (+ offset-b# i#) fd-b#))))))

(defmacro ^:private ge-scal [method alpha a]
  `(when (< 0 (.count ~a))
     (let [ld# (.stride ~a)
           sd# (.sd ~a)
           offset# (.offset ~a)
           buff# (.buffer ~a)]
       (if (= sd# ld#)
         (~method (.count ~a) ~alpha buff# offset# 1)
         (dotimes [i# (.fd ~a)]
           (~method sd# ~alpha buff# (+ offset# (* ld# i#)) 1))))))

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
           (dotimes [i# fd-a#]
             (~method sd-a# ~alpha
              buff-a# (+ offset-a# (* ld-a# i#)) 1
              buff-b# (+ offset-b# (* ld-b# i#)) 1)))
         (dotimes [i# fd-b#]
           (~method sd-a# ~alpha
            buff-a# (+ offset-a# i#) fd-a#
            buff-b# (+ offset-b# (* ld-b# i#)) 1))))))

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
     (if (= (.order ~a) (.order ~c)) NO_TRANS TRANS)
     (if (= (.order ~b) (.order ~c)) NO_TRANS TRANS)
     (.mrows ~a) (.ncols ~b) (.ncols ~a) ~alpha (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~b) (.offset ~b) (.stride ~b) ~beta (.buffer ~c) (.offset ~c) (.stride ~c))))

;; =============== Common TR matrix macros and functions ==========================

(defn tr-swap [^BLAS vector-eng ^IFn$OLO col-row ^TRMatrix a ^TRMatrix b]
  (when (< 0 (.count a))
    (dotimes [i (.sd a)]
      (.swap vector-eng (.invokePrim col-row a i) (.invokePrim col-row b i)))))

(defn tr-copy [^BLAS vector-eng ^IFn$OLO col-row ^TRMatrix a ^TRMatrix b]
  (when (< 0 (.count a))
    (dotimes [i (.sd a)]
      (.copy vector-eng (.invokePrim col-row a i) (.invokePrim col-row b i)))))

(defn tr-scal [^BLAS vector-eng ^IFn$OLO col-row alpha ^TRMatrix a]
  (when (< 0 (.count a))
    (dotimes [i (.sd a)]
      (.scal vector-eng alpha (.invokePrim col-row a i)))))

(defn tr-axpy [^BLAS vector-eng ^IFn$OLO col-row alpha ^TRMatrix a ^TRMatrix b]
  (when (< 0 (.count a))
    (dotimes [i (.sd a)]
      (.axpy vector-eng alpha (.invokePrim col-row a i) (.invokePrim col-row b i)))))

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
    (CBLAS/daxpy (.dim ^RealBlockVector x)
                 (double alpha) (.buffer x) (.offset x) (.stride x)
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
    (vector-imin ^RealBlockVector x)))

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
    (CBLAS/saxpy (.dim ^RealBlockVector x)
                 (double alpha) (.buffer x) (.offset x) (.stride x)
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
    (vector-imin ^RealBlockVector x)))

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
    (ge-scal ^BufferAccessor CBLAS/dscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor CBLAS/daxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
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
    c))

(deftype FloatGEEngine []
    BLAS
  (swap [_ a b]
    (ge-swap-copy CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b)
    a)
  (copy [_ a b]
    (ge-swap-copy CBLAS/scopy ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (scal [_ alpha a]
    (ge-scal ^BufferAccessor CBLAS/sscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor CBLAS/saxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
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
    c))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine [^DoubleVectorEngine vector-eng]
  BLAS
  (swap [_ a b]
    (tr-swap vector-eng (.col_row_STAR_ ^RealTRMatrix a) a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-copy vector-eng (.col_row_STAR_ ^RealTRMatrix a) a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal vector-eng (.col_row_STAR_ ^RealTRMatrix a) alpha a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy vector-eng (.col_row_STAR_ ^RealTRMatrix a) alpha a ^RealTRMatrix b)
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
    b))

(deftype FloatTREngine [^FloatVectorEngine vector-eng]
  BLAS
  (swap [_ a b]
    (tr-swap vector-eng (.col_row_STAR_ ^RealTRMatrix a) a ^RealTRMatrix b)
    a)
  (copy [_ a b]
    (tr-copy vector-eng (.col_row_STAR_ ^RealTRMatrix a) a ^RealTRMatrix b)
    b)
  (scal [_ alpha a]
    (tr-scal vector-eng (.col_row_STAR_ ^RealTRMatrix a) alpha a)
    a)
  (axpy [_ alpha a b]
    (tr-axpy vector-eng (.col_row_STAR_ ^RealTRMatrix a) alpha a ^RealTRMatrix b)
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
    b))

;; =============== Factories ==================================================

(deftype CblasFactory [^DataAccessor da ^BLAS vector-eng ^BLAS ge-eng ^BLAS tr-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  MemoryContext
  (compatible [_ o]
    (compatible da o))
  Factory
   (create-vector [this buf n]
    (if (and (<= 0 (long n) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (real-block-vector this true buf n 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count da buf) (class buf))))))
  (create-ge [this buf m n ord]
    (if (and (<= 0 (* (long m) (long n)) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (real-ge-matrix this true buf m n 0 ord)
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d general matrix from %s."
                      m n (type buf))))))
  (create-tr [this buf n ord uplo diag]
    (if (and (<= 0 (* (long n) (long n)) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (real-tr-matrix this true buf n (max (long n) 1) ord uplo diag)
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d triangular matrix from %s."
                      n n (type buf))))))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng))

(def cblas-float
  (let [float-vector-engine (->FloatVectorEngine)]
    (->CblasFactory (->FloatBufferAccessor)
                    float-vector-engine
                    (->FloatGEEngine)
                    (->FloatTREngine float-vector-engine))))

(def cblas-double
  (let [double-vector-engine (->DoubleVectorEngine)]
    (->CblasFactory (->DoubleBufferAccessor)
                    double-vector-engine
                    (->DoubleGEEngine)
                    (->DoubleTREngine double-vector-engine))))
