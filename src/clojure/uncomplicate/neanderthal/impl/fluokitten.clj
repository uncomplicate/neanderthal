;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.impl.fluokitten
  (:refer-clojure :exclude [accessor])
  (:require [uncomplicate.commons.core :refer [let-release]]
            [uncomplicate.fluokitten.protocols :refer [fmap!]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [copy create-raw trans dim ncols mrows col row]]
             [real :refer [sum]]])
  (:import [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [uncomplicate.neanderthal.protocols
            BLASPlus RealVector RealMatrix Vector Matrix RealChangeable Block]))

(def ^{:no-doc true :const true} FITTING_DIMENSIONS_MATRIX_MSG
  "Matrices should have fitting dimensions.")

;; ================== Fluokitten implementation  ===============================
;; ---------------------- Vector Fluokitten funcitions -------------------------
(defn entry ^double [^RealVector x ^long i]
  (.entry x i))

(defmacro  invoke-entry [f i & xs]
  `(.invokePrim ~f ~@(map #(list `entry % i) xs)))

(defmacro ^:private fn-entry [f i & xs]
  `(~f ~@(map #(list `entry % i) xs)))

(defmacro ^:private vector-reduce* [f init invoker g & xs]
  `(let [dim-x# (dim ~(first xs))]
     (loop [i# 0 acc# ~init]
       (if (< i# dim-x#)
         (recur (inc i#) (.invokePrim ~f acc# (~invoker ~g i# ~@xs)))
         acc#))))

(defmacro ^:private vector-reduce-indexed* [f init invoker g & xs]
  `(let [dim-x# (dim ~(first xs))]
     (loop [i# 0 acc# ~init]
       (if (< i# dim-x#)
         (recur (inc i#) (.invokePrim ~f acc# i# (~invoker ~g i# ~@xs)))
         acc#))))

(defn check-vector-dimensions
  ([^Vector x]
   true)
  ([^Vector x ^Vector y]
   (<= (.dim x) (.dim y)))
  ([^Vector x ^Vector y ^Vector z]
   (<= (.dim x) (min (.dim y) (.dim z))))
  ([^Vector x ^Vector y ^Vector z ^Vector v]
   (<= (.dim x) (min (.dim y) (.dim z) (.dim v)))))

(defmacro vector-fmap* [f & xs]
  (if (< (count xs) 5)
    `(do
       (if (check-vector-dimensions ~@xs)
         (dotimes [i# (dim ~(first xs))]
           (.set ~(first xs) i# (invoke-entry ~f i# ~@xs)))
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (dim ~(first xs))))))
       ~(first xs))
    `(throw (UnsupportedOperationException. "Vector fmap supports up to 4 vectors."))))

(defn copy-fmap
  ([x f]
   (let-release [res (copy x)]
     (fmap! res f)))
  ([x f xs]
   (let-release [res (copy x)]
     (apply fmap! res f xs))))

(defmacro vector-fold* [f init & xs]
  (if (< (count xs) 5)
    `(do
       (if (check-vector-dimensions ~@xs)
         (vector-reduce ~f ~init ~@xs)
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (dim ~(first xs)))))))
    `(throw (UnsupportedOperationException. "Vector fold supports up to 4 vectors."))))

(defmacro vector-foldmap* [f init g & xs]
  (if (< (count xs) 5)
    `(do
       (if (check-vector-dimensions ~@xs)
         (vector-reduce-map ~f ~init ~g ~@xs)
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (dim ~(first xs)))))))
    `(throw (UnsupportedOperationException. "Vector foldmap supports up to 4 vectors."))))

(defn ^:private vector-op* [^Vector x & ws]
  (let-release [res ^Vector (create-raw (factory x) (transduce (map dim) + (.dim x) ws))]
    (loop [pos 0 w x ws ws]
      (when w
        (if (compatible res w)
          (.subcopy ^BLASPlus (engine w) w res 0 (dim w) pos)
          (throw (UnsupportedOperationException. (format INCOMPATIBLE_BLOCKS_MSG res w))))
        (recur (+ pos (dim w)) (first ws) (next ws))))
    res))

(defn vector-op
  ([^Vector x ^Vector y]
   (vector-op* x y))
  ([^Vector x ^Vector y ^Vector z]
   (vector-op* x y z))
  ([^Vector x ^Vector y ^Vector z ^Vector v]
   (vector-op* x y z v))
  ([^Vector x ^Vector y ^Vector z ^Vector v ws]
   (apply vector-op* x y z v ws)))

(defn vector-pure
  ([x ^double v]
   (.set ^RealChangeable (raw x) 0 v))
  ([x ^double v ws]
   (throw (UnsupportedOperationException. "This operation would be slow on primitive vectors."))))

;; ---------------------- Matrix Fluokitten funcitions -------------------------

(defn check-matrix-dimensions
  ([^Matrix a]
   true)
  ([^Matrix a ^Matrix b]
   (and (<= (.ncols a) (.ncols b))
        (<= (.mrows a) (.mrows b))))
  ([^Matrix a ^Matrix b ^Matrix c]
   (and (<= (.ncols a) (min (.ncols b) (.ncols c)))
        (<= (.mrows a) (min (.mrows b) (.ncols c)))))
  ([^Matrix a ^Matrix b ^Matrix c ^Matrix d]
   (and (<= (.ncols a) (min (.ncols b) (.ncols c) (.ncols d)))
        (<= (.mrows a) (min (.mrows b) (.ncols c) (.ncols d))))))

(defmacro invoke-matrix-entry [get* f i j & as]
  `(.invokePrim ~f ~@(map #(list get* % i j) as)))

(defmacro matrix-fmap* [set* get* fd sd f & as]
  (if (< (count as) 5)
    `(do
       (if (check-matrix-dimensions ~@as)
         (dotimes [j# ~fd]
           (dotimes [i# ~sd]
             (~set* ~(first as) i# j# (invoke-matrix-entry ~get* ~f i# j# ~@as))))
         (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG)))
       ~(first as))
    `(throw (UnsupportedOperationException. "Matrix fmap support up to 4 matrices."))))

(defmacro matrix-reduce* [col-row j f init & as]
  `(vector-reduce ~f ~init ~@(map #(list col-row % j) as)))

(defmacro matrix-reduce-map* [col-row j f init g & as]
  `(vector-reduce-map ~f ~init ~g ~@(map #(list col-row % j) as)))

(defmacro matrix-fold* [fd col-row f init & as]
  (if (< (count as) 5)
    `(do
       (if (check-matrix-dimensions ~@as)
         (loop [j# 0 acc# ~init]
           (if (< j# ~fd)
             (recur (inc j#) (matrix-reduce* ~col-row j# ~f acc# ~@as))
             acc#))
         (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
    `(throw (UnsupportedOperationException. "Matrix fold supports up to 4 vectors."))))

(defmacro matrix-foldmap* [fd col-row f init g & as]
  (if (< (count as) 5)
    `(do
       (if (check-matrix-dimensions ~@as)
         (loop [j# 0 acc# ~init]
           (if (< j# ~fd)
             (recur (inc j#) (matrix-reduce-map* ~col-row j# ~f acc# ~g ~@as))
             acc#))
         (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
    `(throw (UnsupportedOperationException. "Matrix fold supports up to 4 vectors."))))


(defn matrix-op* [^Matrix a & bs]
  (let-release [res ^Matrix (if (= COLUMN_MAJOR (.order ^Block a))
                              (create-raw (factory a)
                                          (.mrows a)
                                          (transduce (map ncols) + (.ncols a) bs))
                              (trans (create-raw (factory a)
                                                 (.ncols a)
                                                 (transduce (map mrows) + (.mrows a) bs))))]
    (let [column-reducer
          (fn ^long [^long pos ^Matrix w]
            (if (compatible res w)
              (.copy ^BLASPlus (engine w) w
                     (.submatrix ^Matrix res 0 pos (.mrows w) (.ncols w)))
              (throw (IllegalArgumentException.
                      (format INCOMPATIBLE_BLOCKS_MSG res w))))
            (+ pos (.ncols w)))
          row-reducer
          (fn ^long [^long pos ^Matrix w]
            (if (compatible res w)
              (.copy ^BLASPlus (engine w) w
                     (.submatrix ^Matrix res pos 0 (.mrows w) (.ncols w)))
              (throw (IllegalArgumentException.
                      (format INCOMPATIBLE_BLOCKS_MSG res w))))
            (+ pos (.mrows w)))]
      (.copy ^BLASPlus (engine a) a
             (.submatrix ^Matrix res 0 0 (.mrows a) (.ncols a)))
      (if (= COLUMN_MAJOR (.order ^Block a))
        (reduce column-reducer (.ncols a) bs)
        (reduce row-reducer (.mrows a) bs))
      res)))

(defn matrix-op
  ([^Matrix a ^Matrix b]
   (matrix-op* a b))
  ([^Matrix a ^Matrix b ^Matrix c]
   (matrix-op* a b c))
  ([^Matrix a ^Matrix b ^Matrix c ^Matrix d]
   (matrix-op* a b c d))
  ([^Matrix a ^Matrix b ^Matrix c ^Vector d es]
   (apply matrix-op* a b c d es)))

(defn matrix-pure
  ([a ^double v]
   (.set ^RealChangeable (raw a) 0 0 v))
  ([a ^double v cs]
   (throw (UnsupportedOperationException.
           "This operation would be slow on primitive matrices."))))

(extend-type IFn$DDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* this (double init) fn-entry + x))
    ([this init x y]
     (vector-reduce* this (double init) fn-entry + x y))
    ([this init x y z]
     (vector-reduce* this (double init) fn-entry + x y z))
    ([this init x y z v]
     (vector-reduce* this (double init) fn-entry + x y z v)))
  (vector-reduce-map
    ([this init g x]
     (vector-reduce* this (double init) invoke-entry ^IFn$DD g x))
    ([this init g x y]
     (vector-reduce* this (double init) invoke-entry ^IFn$DDD g x y))
    ([this init g x y z]
     (vector-reduce* this (double init) invoke-entry ^IFn$DDDD g  x y z))
    ([this init g x y z v]
     (vector-reduce* this (double init) invoke-entry ^IFn$DDDDD g x y z v))))

(extend-type IFn$ODO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* this init fn-entry + x))
    ([this init x y]
     (vector-reduce* this init fn-entry + x y))
    ([this init x y z]
     (vector-reduce* this init fn-entry + x y z))
    ([this init x y z v]
     (vector-reduce* this init fn-entry + x y z v)))
  (vector-reduce-map
    ([this init g x]
     (vector-reduce* this init invoke-entry ^IFn$DD g x))
    ([this init g x y]
     (vector-reduce* this init invoke-entry ^IFn$DDD g x y))
    ([this init g x y z]
     (vector-reduce* this init invoke-entry ^IFn$DDDD g x y z))
    ([this init g x y z v]
     (vector-reduce* this init invoke-entry ^IFn$DDDDD g x y z v))))

(extend-type IFn$DLDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* this (double init) fn-entry + x))
    ([this init x y]
     (vector-reduce-indexed* this (double init) fn-entry + x y))
    ([this init x y z]
     (vector-reduce-indexed* this (double init) fn-entry + x y z))
    ([this init x y z v]
     (vector-reduce-indexed* this (double init) fn-entry + x y z v)))
  (vector-reduce-map
    ([this init g x]
     (vector-reduce-indexed* this (double init) invoke-entry ^IFn$DD g x))
    ([this init g x y]
     (vector-reduce-indexed* this (double init) invoke-entry ^IFn$DDD g x y))
    ([this init g x y z]
     (vector-reduce-indexed* this (double init) invoke-entry ^IFn$DDDD g x y z))
    ([this init g x y z v]
     (vector-reduce-indexed* this (double init) invoke-entry ^IFn$DDDDD g x y z v))))

(extend-type IFn$OLDO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* this init fn-entry + x))
    ([this init x y]
     (vector-reduce-indexed* this init fn-entry + x y))
    ([this init x y z]
     (vector-reduce-indexed* this init fn-entry + x y z))
    ([this init x y z v]
     (vector-reduce-indexed* this init fn-entry + x y z v)))
  (vector-reduce-map
    ([this init g x]
     (vector-reduce-indexed* this init invoke-entry ^IFn$DD g x))
    ([this init g x y]
     (vector-reduce-indexed* this init invoke-entry ^IFn$DDD g x y))
    ([this init g x y z]
     (vector-reduce-indexed* this init invoke-entry ^IFn$DDDD g x y z))
    ([this init g x y z v]
     (vector-reduce-indexed* this init invoke-entry ^IFn$DDDDD g x y z v))))
