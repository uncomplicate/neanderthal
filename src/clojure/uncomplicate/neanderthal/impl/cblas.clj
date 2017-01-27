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

(defmacro ^:private vector-rotg [method x]
  `(if (= 1 (stride ~x))
     (~method (buffer ~x))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (stride ~x))))))

(defmacro ^:private vector-rotm [method x y p]
  `(if (= 1 (stride ~p))
     (~method (dim ~x) (buffer ~x) (stride ~x)
      (buffer ~y) (stride ~y) (buffer ~p))
     (throw (IllegalArgumentException. (format STRIDE_MSG 1 (stride ~p))))))

(defmacro ^:private vector-rotmg [method p args]
  `(if (= 1 (stride ~p) (stride ~args))
     (~method (buffer ~args) (buffer ~p))
     (throw (IllegalArgumentException.
             (format STRIDE_MSG 1 (str (stride ~p) " or " (stride ~args)))))))

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

(defn ^:private slice [^BufferAccessor da buff ^long k ^long l]
  (.slice da buff k l))

(defmacro ^:private ge-swap-copy [method a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           buff-b# (.buffer ~b)]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (~method (.count ~a) buff-a# 0 1 buff-b# 0 1)
           (dotimes [i# fd-a#]
             (~method sd-a# buff-a# (* ld-a# i#) 1 buff-b# (* ld-b# i#) 1)))
         (dotimes [i# fd-a#]
           (~method sd-a# buff-a# (* ld-a# i#) 1 buff-b# i# fd-b#))))))

(defmacro ^:private ge-scal [da method alpha a]
  `(when (< 0 (.count ~a))
     (let [ld# (.stride ~a)
           sd# (.sd ~a)
           fd# (.fd ~a)
           buff# (.buffer ~a)
           da# ~da]
       (if (= sd# ld#)
         (~method (.count ~a) ~alpha buff# 1)
         (dotimes [i# fd#]
           (~method sd# ~alpha (.slice da# buff# (* ld# i#) sd#) 1))))))

(defmacro ^:private ge-axpy [da method alpha a b]
  `(when (< 0 (.count ~a))
     (let [ld-a# (.stride ~a)
           sd-a# (.sd ~a)
           fd-a# (.fd ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           sd-b# (.sd ~b)
           fd-b# (.fd ~b)
           buff-b# (.buffer ~b)
           da# ~da]
       (if (= (.order ~a) (.order ~b))
         (if (= sd-a# sd-b# ld-a# ld-b#)
           (~method (.count ~a) ~alpha buff-a# 1 buff-b# 1)
           (dotimes [i# fd-a#]
             (~method sd-a# ~alpha
              (.slice da# buff-a# (* ld-a# i#) sd-a#) 1
              (.slice da# buff-b# (* ld-b# i#) sd-b#) 1)))
         (dotimes [i# fd-b#]
           (~method sd-a# ~alpha
            (.slice da# buff-a# i# (inc (* (dec fd-a#) ld-a#))) fd-a#
            (.slice da# buff-b# (* ld-b# i#) sd-b#) 1))))))


(defn ge-mm [^BLAS eng alpha ^GEMatrix a ^Matrix b left]
  (if left
    (.mm (engine b) alpha b a false)
    (throw (IllegalArgumentException.
            (format "In-place mm! is not supported for GE matrices.")))))

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

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/dswap (.dim ^RealBlockVector x)
                 (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y))
    x)
  (copy [_ x y]
    (CBLAS/dcopy (.dim ^RealBlockVector x)
                 (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y))
    y)
  (dot [_ x y]
    (CBLAS/ddot (.dim ^RealBlockVector x)
                (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y)))
  (nrm2 [_ x]
    (CBLAS/dnrm2 (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (asum [_ x]
    (CBLAS/dasum (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (iamax [_ x]
    (CBLAS/idamax (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (rot [_ x y c s]
    (CBLAS/drot (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)
                (.buffer ^Block y) (.stride ^Block y) c s)
    x)
  (rotg [_ x]
    (vector-rotg CBLAS/drotg ^RealBlockVector x)
    x)
  (rotm [_ x y p]
    (vector-rotm CBLAS/drotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector p)
    x)
  (rotmg [_ p args]
    (vector-rotmg CBLAS/drotmg ^RealBlockVector p ^RealBlockVector args)
    p)
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^RealBlockVector x) alpha (.buffer x) (.stride x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^RealBlockVector x)
                 (double alpha) (.buffer x) (.stride x)
                 (.buffer ^RealBlockVector y) (.stride y))
    y)
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^RealBlockVector x) kx (.stride x)
                 (.buffer ^RealBlockVector y) ky (.stride y))
    y)
  (sum [_ x]
    (CBLAS/dsum (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x)))

(deftype FloatVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/sswap (.dim ^RealBlockVector x)
                 (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y))
    x)
  (copy [_ x y]
    (CBLAS/scopy (.dim ^RealBlockVector x)
                 (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y))
    y)
  (dot [_ x y]
    (CBLAS/sdot (.dim ^RealBlockVector x)
                (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y)))
  (nrm2 [_ x]
    (CBLAS/snrm2 (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (asum [_ x]
    (CBLAS/sasum (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (iamax [_ x]
    (CBLAS/isamax (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
  (rot [_ x y c s]
    (CBLAS/srot (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)
                (.buffer ^Block y) (.stride ^Block y) c s)
    x)
  (rotg [_ x]
    (vector-rotg CBLAS/srotg ^RealBlockVector x)
    x)
  (rotm [_ x y p]
    (vector-rotm CBLAS/srotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector p)
    x)
  (rotmg [_ p args]
    (vector-rotmg CBLAS/srotmg ^RealBlockVector p ^RealBlockVector args)
    p)
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^RealBlockVector x) alpha (.buffer x) (.stride x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^RealBlockVector x)
                 (double alpha) (.buffer x) (.stride x)
                 (.buffer ^RealBlockVector y) (.stride y))
    y)
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^RealBlockVector x) kx (.stride x)
                 (.buffer ^RealBlockVector y) ky (.stride y))
    y)
  (sum [_ x]
    (CBLAS/ssum (.dim ^RealBlockVector x) (.buffer ^Block x) (.stride ^Block x)))
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
    (ge-scal ^BufferAccessor (data-accessor a) CBLAS/dscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor (data-accessor a) CBLAS/daxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (CBLAS/dgemv (.order ^RealGEMatrix a) NO_TRANS
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^RealBlockVector x) (.stride ^Block x)
                 beta (.buffer ^RealBlockVector y) (.stride ^Block y))
    y)
  (mv [this a x]
    (throw (IllegalArgumentException.
            (format "In-place mv! is not supported for GE matrices."))))
  (rank [_ alpha x y a]
    (CBLAS/dger (.order ^RealGEMatrix a) (.mrows a) (.ncols a)
                alpha (.buffer ^RealBlockVector x) (.stride ^Block x)
                (.buffer ^RealBlockVector y) (.stride ^Block y)
                (.buffer ^GEMatrix a) (.stride ^GEMatrix a))
    a)
  (mm [this alpha a b left]
    (ge-mm this alpha a b left))
  (mm [_ alpha a b beta c]
    (CBLAS/dgemm (.order ^RealGEMatrix c)
                 (if (= (.order ^RealGEMatrix a) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (if (= (.order ^RealGEMatrix b) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix b) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b)
                 beta (.buffer ^GEMatrix c) (.stride ^GEMatrix c))
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
    (ge-scal ^BufferAccessor (data-accessor a) CBLAS/sscal alpha ^RealGEMatrix a)
    a)
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor (data-accessor a) CBLAS/saxpy alpha ^RealGEMatrix a ^RealGEMatrix b)
    b)
  (mv [_ alpha a x beta y]
    (CBLAS/sgemv (.order ^RealGEMatrix a) NO_TRANS
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^RealBlockVector x) (.stride ^Block x)
                 beta (.buffer ^RealBlockVector y) (.stride ^Block y))
    y)
  (mv [this a x]
    (throw (IllegalArgumentException.
            (format "In-place mv! is not supported for GE matrices."))))
  (rank [_ alpha x y a]
    (CBLAS/sger (.order ^RealGEMatrix a) (.mrows a) (.ncols a)
                alpha (.buffer ^RealBlockVector x) (.stride ^Block x)
                (.buffer ^RealBlockVector y) (.stride ^Block y)
                (.buffer ^GEMatrix a) (.stride ^GEMatrix a))
    a)
  (mm [this alpha a b left]
    (ge-mm this alpha a b left))
  (mm [_ alpha a b beta c]
    (CBLAS/sgemm (.order ^RealGEMatrix c)
                 (if (= (.order ^RealGEMatrix a) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (if (= (.order ^RealGEMatrix b) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix b) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b)
                 beta (.buffer ^GEMatrix c) (.stride ^GEMatrix c))
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
    (throw (IllegalArgumentException.
            (format "Only in-place mv! is supported for TR matrices."))))
  (mv [_ a x]
    (CBLAS/dtrmv (.order ^RealTRMatrix a) (.uplo ^TRMatrix a) NO_TRANS
                 (.diag ^TRMatrix a) (.ncols a)
                 (.buffer ^Block a) (.stride ^Block a)
                 (.buffer ^RealBlockVector x) (.stride ^Block x))
    x)
  (mm [this alpha a b beta c]
    (throw (IllegalArgumentException.
            (format "Only in-place mm! is supported for TR matrices."))))
  (mm [_ alpha a b left]
    (CBLAS/dtrmm (.order ^RealGEMatrix b) (if left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT)
                 (.uplo ^RealTRMatrix a)
                 (if (= (.order ^TRMatrix a) (.order ^GEMatrix b)) NO_TRANS TRANS)
                 (.diag ^TRMatrix a)
                 (.mrows b) (.ncols b)
                 alpha (.buffer ^TRMatrix a) (.stride ^TRMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b))
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
    (throw (IllegalArgumentException.
            (format "Only in-place mv! is supported for TR matrices."))))
  (mv [_ a x]
    (CBLAS/strmv (.order ^RealTRMatrix a) (.uplo ^TRMatrix a) NO_TRANS
                 (.diag ^TRMatrix a) (.ncols a)
                 (.buffer ^Block a) (.stride ^Block a)
                 (.buffer ^RealBlockVector x) (.stride ^Block x))
    x)
  (mm [this alpha a b beta c]
    (throw (IllegalArgumentException.
            (format "Only in-place mm! is supported for TR matrices."))))
  (mm [_ alpha a b left]
    (CBLAS/strmm (.order ^RealGEMatrix b) (if left CBLAS/SIDE_LEFT CBLAS/SIDE_RIGHT)
                 (.uplo ^RealTRMatrix a)
                 (if (= (.order ^TRMatrix a) (.order ^GEMatrix b)) NO_TRANS TRANS)
                 (.diag ^TRMatrix a)
                 (.mrows b) (.ncols b)
                 alpha (.buffer ^TRMatrix a) (.stride ^TRMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b))
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
   (create-vector [this buf n _]
    (if (and (<= 0 (long n) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (real-block-vector this true buf n 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count da buf) (class buf))))))
  (create-ge [this buf m n options]
    (if (and (<= 0 (* (long m) (long n)) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (real-ge-matrix this true buf m n 0 (enc-order (:order options)))
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d general matrix from %s."
                      m n (type buf))))))
  (create-tr [this buf n options]
    (if (and (<= 0 (* (long n) (long n)) (.count da buf))
             (instance? ByteBuffer buf) (.isDirect ^ByteBuffer buf))
      (let [ord (enc-order (:order options))
            uplo (enc-uplo (:uplo options))
            diag (enc-diag (:diag options))]
        (real-tr-matrix this true buf n (max (long n) 1) ord uplo diag))
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
