(ns uncomplicate.neanderthal.double
  (:require [primitive-math]
            [vertigo
             [bytes :refer [direct-buffer]]
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [cblas]])
  (:import [uncomplicate.neanderthal.cblas
            DoubleBlockVector DoubleGeneralMatrix]
           [uncomplicate.neanderthal.protocols
            DoubleVector DoubleMatrix Carrier]
           [java.nio ByteBuffer]))

;; ============ Creating double array ==============
(defn double-array? [a]
  (let [c (class a)]
    (and (.isArray c)
         (= Double/TYPE (.getComponentType c)))))

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
           (zero? (mod (.capacity ^ByteBuffer source) 8))) 
      (DoubleBlockVector. source
                          (/ (.capacity ^ByteBuffer source) 8)
                          1)
      (and (integer? source) (pos? source)) (dv (direct-buffer (* 8 source)))
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
           (zero? (mod (.capacity ^ByteBuffer source) 8)))
      (if (= (* 8 m n) (.capacity ^ByteBuffer source))
        (DoubleGeneralMatrix.
         source m n (max m 1) p/COLUMN_MAJOR)
        (throw (IllegalArgumentException.
                (format "Matrix dimensions (%d x %d) are not compatible with the buffer capacity."
                        m n))))
      (sequential? source) (dge m n (seq-to-buffer source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a double vector from %s ."
                               (type source))))))
  ([^long m ^long n]
     (dge m n (direct-buffer (* 8 m n)))))

;; ============ Vector and Matrix access methods ===
(defn entry ^double 
  ([^DoubleVector x ^long i]
     (.entry x i))
  ([^DoubleMatrix m ^long i ^long j]
     (.entry m i j)))

;; ================== BLAS 1 =======================

(defn dot ^double [^DoubleVector x ^DoubleVector y]
  (if (= (.dim x) (.dim y))
    (.dot x y)
    (throw (IllegalArgumentException.
            "Arguments should have the same number of elements."))))

(defn nrm2 ^double [^DoubleVector x]
  (.nrm2 x))

(defn asum ^double [^DoubleVector x]
  (.asum x))

#_(defn rot! [x y c s]
    (.rot x y c s))

#_(defn rotg! [a b c s] 
    (.rotg a b c s))

#_(defn rotmg! [d1 d2 b1 b2 p]
    (.rotmg d1 d2 b2 b2 p))

#_(defn rotm! [x y p]
    (.rotm x y p))

(defn scal! [^double alpha ^DoubleVector x]
  (.scal x alpha))

(defn axpy!
  ([^DoubleVector y ^double alpha ^DoubleVector x]
     (if (= (.dim x) (.dim y))
       (.axpy x alpha y)
       (throw (IllegalArgumentException.
               "Arguments should have the same number of elements."))))
  ([^DoubleVector y ^DoubleVector x]
     (axpy! y 1.0 x)))

;;================= BLAS 2 ========================
(defn mv!
  ([^DoubleVector y alpha ^DoubleMatrix a ^DoubleVector x beta]
     (if (and (= (.ncols a) (.dim x))
              (= (.mrows a) (.dim y)))
       (.mv a alpha x beta y p/NO_TRANS)
       (throw (IllegalArgumentException.
               "Matrix columns must be equals to vector dimensions."))))
  ([y alpha a x]
     (mv! y alpha a x 0.0))
  ([y a x]
     (mv! y 1.0 a x 0.0)))

(defn mv
  ([^double alpha ^DoubleMatrix a ^Carrier x]
     (mv! (dv (.mrows a)) alpha a x 0.0))
  ([a x]
     (mv 1.0 a x)))

;; ================ BLAS 3 ========================
(defn mm!
  ([^DoubleMatrix c alpha ^DoubleMatrix a ^DoubleMatrix b beta]
     (if (and (= (.ncols a) (.mrows b))
              (= (.mrows a) (.mrows c))
              (= (.ncols b) (.ncols c)))
       (.mm a alpha b beta c p/NO_TRANS p/NO_TRANS)
       (throw (IllegalArgumentException.
               "Matrix dimensions have to be compatible."))))
  ([c a b]
     (mm! c 1.0 a b 1.0)))

(defn mm
  ([alpha ^DoubleMatrix a ^DoubleMatrix b]
       (mm! (dge (.mrows a) (.ncols b)) a alpha b))
    ([a b]
       (mm a 1.0 b)))
