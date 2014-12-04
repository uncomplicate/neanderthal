(ns uncomplicate.neanderthal.real
  (:require [primitive-math]
            [vertigo
             [bytes :refer [direct-buffer]]
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [copy]]
             [cblas :refer [MAT_BOUNDS_MSG]]])
  (:import [uncomplicate.neanderthal.cblas
            DoubleBlockVector DoubleGeneralMatrix]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Carrier
            RealVectorEditor RealMatrixEditor]
           [java.nio ByteBuffer]))

(primitive-math/use-primitive-operators)

;; ============ Creating double constructs  ==============

(defn seq-to-buffer [s]
  (.position ^ByteBuffer
             (reduce (fn [^ByteBuffer bb ^double e]
                       (.putDouble bb e))
                     (direct-buffer (* 8 (count s))) s)
             0))

(defn dv
  ([source]
     (cond
      (and (instance? ByteBuffer source)
           (zero? (long (mod (.capacity ^ByteBuffer source) 8))))
      (DoubleBlockVector. source
                          (/ (.capacity ^ByteBuffer source) 8)
                          1)
      (and (integer? source) (pos? source)) (dv (direct-buffer (* 8 (long source))))
      (float? source) (dv [source])
      (sequential? source) (dv (seq-to-buffer source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a double vector from %s ."
                               (type source))))))
  ([x & xs]
     (dv (cons x xs))))

(defn dge
  ([^long m ^long n source]
     (cond
      (and (instance? ByteBuffer source)
           (zero? (long (mod (.capacity ^ByteBuffer source) 8))))
      (if (= (* 8 m n) (.capacity ^ByteBuffer source))
        (DoubleGeneralMatrix.
         source m n (max m 1) p/COLUMN_MAJOR)
        (throw (IllegalArgumentException.
                (format "Matrix dimensions (%dx%d) are not compatible with the buffer capacity."
                        m n))))
      (sequential? source) (dge m n (seq-to-buffer source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a double vector from %s ."
                               (type source))))))
  ([^long m ^long n]
     (dge m n (direct-buffer (* 8 m n)))))

;; ============ Vector and Matrix access methods ===
(defn entry ^double
  ([^RealVector x ^long i]
   (.entry x i))
  ([^RealMatrix m ^long i ^long j]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.entry m i j)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

(defn set-entry!
  ([^RealVectorEditor v ^long i ^double val]
   (.setEntry v i val))
  ([^RealMatrixEditor m ^long i ^long j ^double val]
   (if (and (< -1 i (.mrows m)) (< -1 j (.ncols m)))
     (.setEntry m i j val)
     (throw (IndexOutOfBoundsException.
             (format MAT_BOUNDS_MSG i j (.mrows m) (.ncols m)))))))

;; ================== Category functions  ==============
(defn fmap!
  ([f x]
   (p/fmap! x f))
  ([f x y]
   (p/fmap! x f y))
  ([f x y z]
   (p/fmap! x f y z))
  ([f x y z w]
   (p/fmap! x f y z w))
  ([f x y z w & ws]
   (apply p/fmap! x f y z w ws)))

(defn fold
  ([x]
   (p/fold x))
  ([f x]
   (p/fold x f))
  ([f id x]
   (p/fold x f id)))

(defn freduce
  ([f x]
   (p/freduce x f))
  ([f acc x]
   (p/freduce x acc f))
  ([f acc x y]
   (p/freduce x acc f y))
  ([f acc x y z]
   (p/freduce x acc f y z))
  ([f acc x y z & ws]
   (apply p/freduce x acc f y z ws)))

;; ================== BLAS 1 =======================

(defn dot ^double [^RealVector x ^RealVector y]
  (if (= (.dim x) (.dim y))
    (.dot x y)
    (throw (IllegalArgumentException.
            (format "Incompatible dimensions - x:%d and y:%d."
                    (.dim x) (.dim y))))))

(defn nrm2 ^double [^RealVector x]
  (.nrm2 x))

(defn asum ^double [^RealVector x]
  (.asum x))

#_(defn rot! [x y c s]
    (.rot x y c s))

#_(defn rotg! [a b c s]
    (.rotg a b c s))

#_(defn rotmg! [d1 d2 b1 b2 p]
    (.rotmg d1 d2 b2 b2 p))

#_(defn rotm! [x y p]
    (.rotm x y p))

(defn scal! [^double alpha ^RealVector x]
  (.scal x alpha))

(defn axpy!
  ([^RealVector y ^double alpha ^RealVector x]
     (if (= (.dim x) (.dim y))
       (.axpy x alpha y)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - x:%d and y:%d."
                       (.dim x) (.dim y))))))
  ([y x]
   (axpy! y 1.0 x))
  ([y x z & zs]
   (if (number? x)
     (loop [res (axpy! y x z) s zs]
       (if-let [f (first s)]
         (let [r (rest s)]
           (if (number? f)
             (recur (axpy! res f (first r)) (rest r))
             (recur (axpy! res 1.0 f) r)))
         res))
     (apply axpy! y 1.0 x z zs))))

(defn axpy
  ([x y]
   (if (number? x)
     (axpy! (p/zero y) x y)
     (axpy! (copy y) 1.0 x)))
  ([x y & zs]
   (apply axpy! (if (number? x)
                  (p/zero y)
                  (p/zero x))
          x y zs)))

(defn ax [^double alpha x]
  (axpy! (p/zero x) alpha x))

(defn xpy
  ([x y]
   (axpy! (copy y) 1.0 x))
  ([x y & zs]
   (loop [res (axpy! (p/zero y) 1.0 x) s zs]
     (if s
       (recur (axpy! res 1.0 (first s)) (next s))
       res))))

;;================= BLAS 2 ========================
(defn mv!
  ([^RealVector y alpha ^RealMatrix a ^RealVector x beta]
     (if (and (= (.ncols a) (.dim x))
              (= (.mrows a) (.dim y)))
       (.mv a alpha x beta y p/NO_TRANS)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - a:%dx%d and x:%d."
                       (.mrows a) (.ncols a) (.dim x))))))
  ([y alpha a x]
     (mv! y alpha a x 0.0))
  ([y a x]
     (mv! y 1.0 a x 0.0)))

(defn mv
  ([^double alpha ^RealMatrix a ^Carrier x]
     (mv! (dv (.mrows a)) alpha a x 0.0))
  ([a x]
     (mv 1.0 a x)))

;; ================ BLAS 3 ========================
(defn mm!
  ([^RealMatrix c alpha ^RealMatrix a ^RealMatrix b beta]
     (if (and (= (.ncols a) (.mrows b))
              (= (.mrows a) (.mrows c))
              (= (.ncols b) (.ncols c)))
       (.mm a alpha b beta c p/NO_TRANS p/NO_TRANS)
       (throw (IllegalArgumentException.
               (format "Incompatible dimensions - a:%dx%d, b:%dx%d, c:%dx%d."
                       (.mrows c) (.ncols c)
                       (.mrows a) (.ncols a)
                       (.mrows b) (.ncols b))))))
  ([c a b]
     (mm! c 1.0 a b 1.0)))

(defn mm
  ([alpha ^RealMatrix a ^RealMatrix b]
       (mm! (dge (.mrows a) (.ncols b)) a alpha b))
    ([a b]
       (mm a 1.0 b)))

(primitive-math/unuse-primitive-operators)
