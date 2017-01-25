;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.impl.cblas
  (:require [vertigo.bytes :refer [direct-buffer]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [dim]]
             [block :refer [order buffer offset stride]]]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer DirectByteBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus DataAccessor BufferAccessor Block
            Vector RealVector ContiguousBlock GEMatrix TRMatrix]
           [uncomplicate.neanderthal.impl.buffer_block RealTRMatrix]))

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

;; =============== Common GE matrix macros and functions ==========================

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

;; =============== Common TR matrix macros and functions ==========================

(defmacro ^:private tr-swap [vector-engine a b]
  `(when (< 0 (.count ~a))
     (let [col-row# (.col_row_STAR_ ~a)]
       (dotimes [i# (.sd ~a)]
         (.swap ~vector-engine (col-row# ~a i#) (col-row# ~b i#))))))

(defmacro ^:private tr-copy [vector-engine a b]
  `(when (< 0 (.count ~a))
     (let [col-row# (.col_row_STAR_ ~a)]
       (dotimes [i# (.sd ~a)]
         (.copy ~vector-engine (col-row# ~a i#) (col-row# ~b i#))))))

(defmacro ^:private tr-scal [vector-engine alpha a]
  `(when (< 0 (.count ~a))
     (let [col-row# (.col_row_STAR_ ~a)]
       (dotimes [i# (.sd ~a)]
         (.scal ~vector-engine ~alpha (col-row# ~a i#))))))

(defmacro ^:private tr-axpy [vector-engine alpha a b]
  `(when (< 0 (.count ~a))
     (let [col-row# (.col_row_STAR_ ~a)]
       (dotimes [i# (.sd ~a)]
         (.axpy ~vector-engine ~alpha (col-row# ~a i#) (col-row# ~b i#))))))

;; ============ Real Vector Engines ============================================

(deftype DoubleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/dswap (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (copy [_ x y]
    (CBLAS/dcopy (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (dot [_ x y]
    (CBLAS/ddot (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y)))
  (nrm2 [_ x]
    (CBLAS/dnrm2 (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x)))
  (asum [_ x]
    (CBLAS/dasum (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x)))
  (iamax [_ x]
    (CBLAS/idamax (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x)))
  (rot [_ x y c s]
    (CBLAS/drot (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y) c s))
  (rotg [_ x]
    (vector-rotg CBLAS/drotg x))
  (rotm [_ x y p]
    (vector-rotm CBLAS/drotm x y p))
  (rotmg [_ p args]
    (vector-rotmg CBLAS/drotmg p args))
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^Vector x)
                 (double alpha) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (CBLAS/dsum (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x)))
  (imax [_ x]
    (vector-imax x))
  (imin [_ x]
    (vector-imin x)))

(deftype SingleVectorEngine []
  BLAS
  (swap [_ x y]
    (CBLAS/sswap (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (copy [_ x y]
    (CBLAS/scopy (.dim ^Vector x) (.buffer x) 0 (.stride x) (.buffer y) 0 (.stride y)))
  (dot [_ x y]
    (CBLAS/dsdot (.dim x) (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y)))
  (nrm2 [_ x]
    (CBLAS/snrm2 (.dim x) (.buffer ^Block x) (.stride ^Block x)))
  (asum [_ x]
    (CBLAS/sasum (.dim x) (.buffer ^Block x) (.stride ^Block x)))
  (iamax [_ x]
    (CBLAS/isamax (.dim x) (.buffer ^Block x) (.stride ^Block x)))
  (rot [_ x y c s]
    (CBLAS/srot (.dim x) (.buffer ^Block x) (.stride ^Block x) (.buffer ^Block y) (.stride ^Block y) c s))
  (rotg [_ x]
    (vector-rotg CBLAS/srotg x))
  (rotm [_ x y p]
    (vector-rotm CBLAS/srotm x y p))
  (rotmg [_ p args]
    (vector-rotmg CBLAS/srotmg p args))
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^Vector x) alpha (.buffer x) (.stride x)))
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^Vector x)
                 (float alpha) (.buffer x) (.stride x) (.buffer y) (.stride y)))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer x) kx (.stride x) (.buffer y) ky (.stride y)))
  (sum [_ x]
    (CBLAS/ssum (.dim ^Vector x) (.buffer ^Block x) (.stride ^Block x)))
  (imax [_ x]
    (vector-imax x))
  (imin [_ x]
    (vector-imin x)))

;; ================= General Matrix Engines ====================================

(deftype DoubleGEEngine []
  BLAS
  (swap [_ a b]
    (ge-swap-copy CBLAS/dswap ^GEMatrix a ^GEMatrix b))
  (copy [_ a b]
    (ge-swap-copy CBLAS/dcopy ^GEMatrix a ^GEMatrix b))
  (scal [_ alpha a]
    (ge-scal ^BufferAccessor (data-accessor a) CBLAS/dscal alpha ^GEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor (data-accessor a) CBLAS/daxpy alpha ^GEMatrix a ^GEMatrix b))
  (mv [_ alpha a x beta y]
    (CBLAS/dgemv (.order ^GEMatrix a) NO_TRANS
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^Block x) (.stride ^Block x)
                 beta (.buffer ^Block y) (.stride ^Block y)))
  (mv [this a x]
    (.mv this 1.0 a x 0.0 x))
  (rank [_ alpha x y a]
    (CBLAS/dger (.order ^GEMatrix a) (.mrows a) (.ncols a)
                alpha (.buffer ^Block x) (.stride ^Block x)
                (.buffer ^Block y) (.stride ^Block y)
                (.buffer ^GEMatrix a) (.stride ^GEMatrix a)))
  (mm [_ alpha a b beta c]
    (CBLAS/dgemm (.order ^GEMatrix c)
                 (if (= (.order ^GEMatrix a) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (if (= (.order ^GEMatrix b) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (.mrows ^GEMatrix a) (.ncols ^GEMatrix b) (.ncols ^GEMatrix a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b)
                 beta (.buffer ^GEMatrix c) (.stride ^GEMatrix c))))

(deftype SingleGEEngine []
  BLAS
  (swap [_ a b]
    (ge-swap-copy CBLAS/sswap ^GEMatrix a ^GEMatrix b))
  (copy [_ a b]
    (ge-swap-copy CBLAS/scopy ^GEMatrix a ^GEMatrix b))
  (scal [_ alpha a]
    (ge-scal ^BufferAccessor (data-accessor a) CBLAS/sscal alpha ^GEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpy ^BufferAccessor (data-accessor a) CBLAS/saxpy alpha ^GEMatrix a ^GEMatrix b))
  (mv [_ alpha a x beta y]
    (CBLAS/sgemv (.order ^GEMatrix a) NO_TRANS (.mrows a) (.ncols a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^Block x) (.stride ^Block x)
                 beta (.buffer ^Block y) (.stride ^Block y)))
  (mv [this a x]
    (.mv this 1.0 a x 0.0 x))
  (rank [_ alpha x y a]
    (CBLAS/sger (.order ^GEMatrix a) (.mrows a) (.ncols a)
                alpha (.buffer ^Block x) (.stride ^Block x)
                (.buffer ^Block y) (.stride ^Block y)
                (.buffer ^GEMatrix a) (.stride ^GEMatrix a)))
  (mm [_ alpha a b beta c]
    (CBLAS/sgemm (.order ^GEMatrix c)
                 (if (= (.order ^GEMatrix a) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (if (= (.order ^GEMatrix b) (.order ^GEMatrix c)) NO_TRANS TRANS)
                 (.mrows a) (.ncols b) (.ncols a)
                 alpha (.buffer ^GEMatrix a) (.stride ^GEMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b)
                 beta (.buffer ^GEMatrix c) (.stride ^GEMatrix c))))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine [^DoubleVectorEngine vector-eng]
  BLAS
  (swap [_ a b]
    (tr-swap vector-eng ^RealTRMatrix a ^RealTRMatrix b))
  (copy [_ a b]
    (tr-swap vector-eng ^RealTRMatrix a ^RealTRMatrix b))
  (scal [_ alpha a]
    (tr-scal vector-eng alpha ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (tr-axpy vector-eng alpha ^RealTRMatrix a ^RealTRMatrix b))
  (mv [_ a x]
    (CBLAS/dtrmv (.order ^TRMatrix a) (.uplo ^TRMatrix a) NO_TRANS (.diag ^TRMatrix a)
                 (.ncols a)
                 (.buffer ^TRMatrix a) (.stride ^TRMatrix a) (.buffer ^Block x) (.stride ^Block x)))
  (mm [_ alpha a b right]
    (CBLAS/dtrmm (.order ^GEMatrix b) (if right CBLAS/SIDE_RIGHT CBLAS/SIDE_LEFT)
                 (.uplo ^TRMatrix a)
                 (if (= (.order ^TRMatrix a) (.order ^GEMatrix b)) NO_TRANS TRANS)
                 (.diag ^TRMatrix a)
                 (.mrows b) (.ncols b)
                 alpha (.buffer ^TRMatrix a) (.stride ^TRMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b))))

(deftype SingleTREngine [^SingleVectorEngine vector-eng]
  BLAS
  (swap [_ a b]
    (tr-swap vector-eng ^RealTRMatrix a ^RealTRMatrix b))
  (copy [_ a b]
    (tr-swap vector-eng ^RealTRMatrix a ^RealTRMatrix b))
  (scal [_ alpha a]
    (tr-scal vector-eng alpha ^RealTRMatrix a))
  (axpy [_ alpha a b]
    (tr-axpy vector-eng alpha ^RealTRMatrix a ^RealTRMatrix b))
  (mv [_ a x]
    (CBLAS/strmv (.order ^TRMatrix a) (.uplo ^TRMatrix a) NO_TRANS (.diag ^TRMatrix a)
                 (.ncols a)
                 (.buffer ^TRMatrix a) (.stride ^TRMatrix a) (.buffer ^Block x) (.stride ^Block x)))
  (mm [_ alpha a b right]
    (CBLAS/strmm (.order ^GEMatrix b) (if right CBLAS/SIDE_RIGHT CBLAS/SIDE_LEFT)
                 (.uplo ^TRMatrix a)
                 (if (= (.order ^TRMatrix a) (.order ^GEMatrix b)) NO_TRANS TRANS)
                 (.diag ^TRMatrix a)
                 (.mrows b) (.ncols b)
                 alpha (.buffer ^TRMatrix a) (.stride ^TRMatrix a)
                 (.buffer ^GEMatrix b) (.stride ^GEMatrix b))))

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

(def cblas-single
  (let [float-vector-engine (->SingleVectorEngine)]
    (->CblasFactory (->FloatBufferAccessor)
                    float-vector-engine
                    (->SingleGEEngine)
                    (->SingleTREngine float-vector-engine))))

(def cblas-double
  (let [double-vector-engine (->DoubleVectorEngine)]
    (->CblasFactory (->DoubleBufferAccessor)
                    double-vector-engine
                    (->DoubleGEEngine)
                    (->DoubleTREngine double-vector-engine))))
