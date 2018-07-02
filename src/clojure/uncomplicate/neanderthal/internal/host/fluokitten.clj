;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.fluokitten
  (:refer-clojure :exclude [accessor])
  (:require [uncomplicate.commons.core :refer [let-release double-fn]]
            [uncomplicate.fluokitten.protocols :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [vctr copy dim ncols transfer!]]
             [block :refer [buffer offset]]]
            [uncomplicate.neanderthal.internal
             [api :refer [factory compatible? engine raw subcopy sum ReductionFunction
                          vector-reduce vector-map-reduce matrix-reduce matrix-map-reduce
                          data-accessor create-ge navigator storage region view-vctr]]
             [common :refer [real-accessor]]]
            [uncomplicate.neanderthal.internal.navigation :refer [doall-layout real-navigator]])
  (:import [clojure.lang IFn IFn$D IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD
            IFn$DLDD IFn$ODO IFn$OLDO]
           [uncomplicate.neanderthal.internal.api RealBufferAccessor RealVector RealMatrix
            Vector Matrix DiagonalMatrix RealChangeable LayoutNavigator RealLayoutNavigator
            DenseStorage Region]))

(def ^{:no-doc true :const true} FITTING_DIMENSIONS_MATRIX_MSG
  "Matrices should have fitting dimensions.")

(def ^{:no-doc true :const true} DIMENSIONS_MSG
  "Vectors should have fitting dimensions.")

(defn copy-fmap
  ([x f]
   (let-release [res (copy x)]
     (fmap! res f)))
  ([x f xs]
   (let-release [res (copy x)]
     (apply fmap! res f xs))))

;; ==================== Vector Fluokitten funcitions =======================

(defn check-vector-dimensions
  ([^Vector x]
   true)
  ([^Vector x ^Vector y]
   (<= (.dim x) (.dim y)))
  ([^Vector x ^Vector y ^Vector z]
   (<= (.dim x) (min (.dim y) (.dim z))))
  ([^Vector x ^Vector y ^Vector z ^Vector v]
   (<= (.dim x) (min (.dim y) (.dim z) (.dim v)))))

(defmacro map-entries-i [f i & xs]
  `(.invokePrim ~f ~@(map #(list `.entry % i) xs)))

(defmacro add-entries-i [i & xs]
  `(+ ~@(map #(list `.entry % i) xs)))

(defmacro vector-fmap* [f & xs]
  (if (< (count xs) 5)
    `(do
       (if (check-vector-dimensions ~@xs)
         (dotimes [i# (dim ~(first xs))]
           (.set ~(first xs) i# (map-entries-i ~f i# ~@xs)))
         (throw (ex-info DIMENSIONS_MSG {:xs (map str ~xs)})))
       ~(first xs))
    `(throw (UnsupportedOperationException. "Vector fmap supports up to 4 vectors."))))

(defmacro vector-reduce* [f init & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# ~init]
           (if (< i# dim-x#)
             (recur (inc i#) (.invokePrim ~f acc# (add-entries-i i# ~@xs)))
             acc#)))
       (throw (ex-info DIMENSIONS_MSG {:xs (map str ~xs)})))
    `(throw (UnsupportedOperationException. "Vector fold supports up to 4 vectors."))))

(defmacro vector-map-reduce* [f init g & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# ~init]
           (if (< i# dim-x#)
             (recur (inc i#) (.invokePrim ~f acc# (map-entries-i ~g i# ~@xs)))
             acc#)))
       (throw (ex-info DIMENSIONS_MSG {:xs (map str ~xs)})))
    `(throw (UnsupportedOperationException. "Vector foldmap supports up to 4 vectors."))))

(defmacro ^:private vector-reduce-indexed* [f init & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# ~init]
           (if (< i# dim-x#)
             (recur (inc i#) (.invokePrim ~f acc# i# (add-entries-i i# ~@xs)))
             acc#)))
       (throw (ex-info DIMENSIONS_MSG {:xs (map str ~xs)})))
    `(throw (UnsupportedOperationException. "Vector fold supports up to 4 vectors."))))

(defmacro ^:private vector-map-reduce-indexed* [f init g & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# ~init]
           (if (< i# dim-x#)
             (recur (inc i#) (.invokePrim ~f acc# i# (map-entries-i ~g i# ~@xs)))
             acc#)))
       (throw (ex-info DIMENSIONS_MSG {:xs (map str ~xs)})))
    `(throw (UnsupportedOperationException. "Vector foldmap supports up to 4 vectors."))))

(defn vector-op [^Vector x & ws]
  (let-release [res (vctr (factory x) (transduce (map dim) + (.dim x) ws))]
    (loop [pos 0 w x ws ws]
      (when w
        (if (compatible? res w)
          (subcopy (engine w) w res 0 (dim w) pos)
          (throw (ex-info "You can not apply op on incompatiple vectors." {:res res :w w})))
        (recur (+ pos (dim w)) (first ws) (next ws))))
    res))

(defn vector-pure
  ([x ^double v]
   (.set ^RealChangeable (raw x) v))
  ([x ^double v ws]
   (throw (UnsupportedOperationException. "This operation would be slow on primitive vectors."))))

;; ==================== Matrix Fluokitten funcitions ========================

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

(defn matrix-op [^Matrix a & bs]
  (let-release [res (create-ge (factory a) (.mrows a) (transduce (map ncols) + (.ncols a) bs)
                               (.isColumnMajor (navigator a)) true)]
    (loop [pos 0 ^Matrix w a ws bs]
      (when w
        (transfer! w (.submatrix ^Matrix res 0 pos (.mrows w) (.ncols w)))
        (recur (+ pos (.ncols w)) (first ws) (next ws))))
    res))

(defn matrix-pure
  ([a ^double v]
   (.set ^RealChangeable (raw a) v))
  ([a ^double v cs]
   (throw (UnsupportedOperationException. "This operation would be slow on primitive matrices."))))

(defmacro map-entries-ij [nav f i j & xs]
  `(.invokePrim ~f ~@(map #(list `.get nav % i j) xs)))

(defmacro add-entries-ij [nav i j & xs]
  `(+ ~@(map #(list `.get nav % i j) xs)))

(defmacro ^:private matrix-fmap* [f & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~(first as))
             stor# (storage ~(first as))
             reg# (region ~(first as))
             da# (real-accessor ~(first as))
             buff# (buffer ~(first as))
             ofst# (offset ~(first as))]
         (doall-layout nav# stor# reg# i# j#
                       (.set da# buff# (+ ofst# (.index stor# i# j#))
                             (map-entries-ij nav# ~f i# j# ~@as)))
         ~(first as))
       (throw (ex-info FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)})))
    `(throw (UnsupportedOperationException. "Matrix fmap supports up to 4 matrices."))))

(defmacro ^:private matrix-reduce* [loop-unboxer f init & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~(first as))
             stor# (storage ~(first as))
             reg# (region ~(first as))
             fd# (.fd stor#)]
         (loop [j# 0 acc# ~init]
           (if (< j# fd#)
             (recur (inc j#)
                    (~loop-unboxer
                     (let [end# (.end nav# reg# j#)]
                       (loop [i# (.start nav# reg# j#) acc# acc#]
                         (if (< i# end#)
                           (recur (inc i#) (.invokePrim ~f acc# (add-entries-ij nav# i# j# ~@as)))
                           acc#)))))
             acc#)))
       (throw (ex-info FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)})))
    `(throw (UnsupportedOperationException. "Matrix fold supports up to 4 matrices."))))

(defmacro ^:private  matrix-map-reduce* [loop-unboxer f init g & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~(first as))
             stor# (storage ~(first as))
             reg# (region ~(first as))
             fd# (.fd stor#)]
         (loop [j# 0 acc# ~init]
           (if (< j# fd#)
             (recur
              (inc j#)
              (~loop-unboxer
               (let [end# (.end nav# reg# j#)]
                 (loop [i# (.start nav# reg# j#) acc# acc#]
                   (if (< i# end#)
                     (recur (inc i#) (.invokePrim ~f acc# (map-entries-ij nav# ~g i# j# ~@as)))
                     acc#)))))
             acc#)))
       (throw (ex-info FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)})))
    `(throw (UnsupportedOperationException. "Matrix foldmap supports up to 4 matrices."))))

(defn matrix-fmap!
  ([^RealMatrix a ^IFn$DD f]
   (matrix-fmap* f a))
  ([^RealMatrix a ^IFn$DDD f ^RealMatrix b]
   (matrix-fmap* f a b))
  ([^RealMatrix a ^IFn$DDDD f ^RealMatrix b ^RealMatrix c]
   (matrix-fmap* f a b c))
  ([^RealMatrix a ^IFn$DDDDD f ^RealMatrix b ^RealMatrix c  ^RealMatrix d]
   (matrix-fmap* f a b c d))
  ([a f b c d es]
   (throw (UnsupportedOperationException. "Matrix fmap! supports up to 4 matrices."))))

(defn diagonal-fmap!
  ([a ^IFn$DD f]
   (let [va ^RealVector (view-vctr a)]
     (fmap! va f)
     a))
  ([^Matrix a ^IFn$DDD f ^Matrix b]
   (if (instance? DiagonalMatrix b)
     (let [va ^RealVector (view-vctr a)
           vb ^RealVector (view-vctr b)]
       (fmap! va f vb))
     (dotimes [i 3]
       (let [va ^RealVector (.dia a (dec i))
             vb ^RealVector (.dia b (dec i))]
         (fmap! va f vb))))
   a)
  ([^Matrix a ^IFn$DDDD f ^Matrix b ^Matrix c]
   (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c))
     (let [va ^RealVector (view-vctr a)
           vb ^RealVector (view-vctr b)
           vc ^RealVector (view-vctr c)]
       (fmap! va f vb vc))
     (dotimes [i 3]
       (let [va ^RealVector (.dia a (dec i))
             vb ^RealVector (.dia b (dec i))
             vc ^RealVector (.dia c (dec i))]
         (fmap! va f vb vc))))
   a)
  ([^Matrix a ^IFn$DDDDD f ^Matrix b ^Matrix c ^Matrix d]
   (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c) (instance? DiagonalMatrix d))
     (let [va ^RealVector (view-vctr a)
           vb ^RealVector (view-vctr b)
           vc ^RealVector (view-vctr c)
           vd ^RealVector (view-vctr d)]
       (fmap! va f vb vc vd))
     (dotimes [i 3]
       (let [va ^RealVector (.dia a (dec i))
             vb ^RealVector (.dia b (dec i))
             vc ^RealVector (.dia c (dec i))
             vd ^RealVector (.dia d (dec i))]
         (fmap! va f vb vc vd))))
   a)
  ([a f b c d es]
   (throw (UnsupportedOperationException. "Matrix fmap! supports up to 4 matrices."))))

(defn matrix-fold
  ([a]
   (sum (engine a) a))
  ([a f init]
   (matrix-reduce f init a))
  ([a f init b]
   (matrix-reduce f init a b))
  ([a f init b c]
   (matrix-reduce f init a b c))
  ([a f init b c d]
   (matrix-reduce f init a b c d))
  ([a f init b c d es]
   (throw (UnsupportedOperationException. "Matrix fold supports up to 4 matrices."))))

(defn diagonal-fold
  ([a]
   (sum (engine a) a))
  ([a f init]
   (vector-reduce f init (view-vctr a)))
  ([^Matrix a f init ^Matrix b]
   (if (instance? DiagonalMatrix b)
     (vector-reduce f init (view-vctr a) (view-vctr b))
     (vector-reduce f
                    (vector-reduce f
                                   (vector-reduce f init (.dia a) (.dia b))
                                   (.dia a 1) (.dia b 1))
                    (.dia a -1) (.dia b -1))))
  ([^Matrix a f init ^Matrix b ^Matrix c]
   (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c))
     (vector-reduce f init (view-vctr a) (view-vctr b) (view-vctr c))
     (vector-reduce f
                    (vector-reduce f
                                   (vector-reduce f init (.dia a) (.dia b) (.dia c))
                                   (.dia a 1) (.dia b 1) (.dia c 1))
                    (.dia a -1) (.dia b -1) (.dia c -1))))
  ([^Matrix a f init ^Matrix b ^Matrix c ^Matrix d]
   ([^Matrix a f init ^Matrix b ^Matrix c]
    (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c) (instance? DiagonalMatrix d))
      (vector-reduce f init (view-vctr a) (view-vctr b) (view-vctr c) (view-vctr d))
      (vector-reduce f
                     (vector-reduce f
                                    (vector-reduce f init (.dia a) (.dia b) (.dia c) (.dia d))
                                    (.dia a 1) (.dia b 1) (.dia c 1) (.dia d 1))
                     (.dia a -1) (.dia b -1) (.dia c -1) (.dia d -1)))))
  ([a f init b c d es]
   (throw (UnsupportedOperationException. "Matrix fold supports up to 4 matrices."))))

(let [p+ (double-fn +)]

  (defn matrix-foldmap
    ([a g]
     (matrix-map-reduce* double ^IFn$DDD p+ 0.0 ^IFn$DD g a))
    ([a g f init]
     (matrix-map-reduce f init g a))
    ([a g f init b]
     (matrix-map-reduce f init g a b))
    ([a g f init b c]
     (matrix-map-reduce f init g a b c))
    ([a g f init b c d]
     (matrix-map-reduce f init g a b c d))
    ([a g f init b c d es]
     (throw (UnsupportedOperationException. "Matrix foldmap supports up to 4 matrices.")))))

(defn diagonal-foldmap
  ([a g]
   (sum (engine a) a))
  ([a g f init]
   (vector-map-reduce f init g (view-vctr a)))
  ([^Matrix a g f init ^Matrix b]
   (if (instance? DiagonalMatrix b)
     (vector-map-reduce f init g (view-vctr a) (view-vctr b))
     (vector-map-reduce f
                        (vector-map-reduce f
                                           (vector-map-reduce f init g (.dia a) (.dia b))
                                           g (.dia a 1) (.dia b 1))
                        g (.dia a -1) (.dia b -1))))
  ([^Matrix a g f init ^Matrix b ^Matrix c]
   (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c))
     (vector-map-reduce f init g (view-vctr a) (view-vctr b) (view-vctr c))
     (vector-map-reduce f
                        (vector-map-reduce f
                                           (vector-map-reduce f init g (.dia a) (.dia b) (.dia c))
                                           g (.dia a 1) (.dia b 1) (.dia c 1))
                        g (.dia a -1) (.dia b -1) (.dia c -1))))
  ([^Matrix a g f init ^Matrix b ^Matrix c ^Matrix d]
   ([^Matrix a f init ^Matrix b ^Matrix c]
    (if (and (instance? DiagonalMatrix b) (instance? DiagonalMatrix c) (instance? DiagonalMatrix d))
      (vector-map-reduce f init g (view-vctr a) (view-vctr b) (view-vctr c) (view-vctr d))
      (vector-map-reduce f
                         (vector-map-reduce f
                                            (vector-map-reduce f init g (.dia a) (.dia b) (.dia c) (.dia d))
                                            g (.dia a 1) (.dia b 1) (.dia c 1) (.dia d 1))
                         g (.dia a -1) (.dia b -1) (.dia c -1) (.dia d -1)))))
  ([a g f init b c d es]
   (throw (UnsupportedOperationException. "Matrix foldmap supports up to 4 matrices."))))

;; ============================ Primitive function extensions ==================

(extend-type IFn$DDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* this (double init) ^RealVector x))
    ([this init x y]
     (vector-reduce* this (double init) ^RealVector x ^RealVector y))
    ([this init x y z]
     (vector-reduce* this (double init) ^RealVector x ^RealVector y ^RealVector z))
    ([this init x y z v]
     (vector-reduce* this (double init) ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce* this (double init) ^IFn$DD g ^RealVector x))
    ([this init g x y]
     (vector-map-reduce* this (double init) ^IFn$DDD g ^RealVector x ^RealVector y))
    ([this init g x y z]
     (vector-map-reduce* this (double init) ^IFn$DDDD g ^RealVector x ^RealVector y ^RealVector z))
    ([this init g x y z v]
     (vector-map-reduce* this (double init) ^IFn$DDDDD g ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (matrix-reduce
    ([this init a]
     (matrix-reduce* double this (double init) a))
    ([this init a b]
     (matrix-reduce* double this (double init) a b))
    ([this init a b c]
     (matrix-reduce* double this (double init) a b c))
    ([this init a b c d]
     (matrix-reduce* double this (double init) a b c d)))
  (matrix-map-reduce
    ([this init g a]
     (matrix-map-reduce* double this (double init) ^IFn$DD g a))
    ([this init g a b]
     (matrix-map-reduce* double this (double init) ^IFn$DDD g a b))
    ([this init g a b c]
     (matrix-map-reduce* double this (double init) ^IFn$DDDD g a b c))
    ([this init g a b c d]
     (matrix-map-reduce* double this (double init) ^IFn$DDDDD g a b c d))))

(extend-type IFn$ODO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* this init ^RealVector x))
    ([this init x y]
     (vector-reduce* this init ^RealVector x ^RealVector y))
    ([this init x y z]
     (vector-reduce* this init ^RealVector x ^RealVector y ^RealVector z))
    ([this init x y z v]
     (vector-reduce* this init ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce* this init ^IFn$DD g ^RealVector x))
    ([this init g x y]
     (vector-map-reduce* this init ^IFn$DDD g ^RealVector x ^RealVector y))
    ([this init g x y z]
     (vector-map-reduce* this init ^IFn$DDDD g ^RealVector x ^RealVector y ^RealVector z))
    ([this init g x y z v]
     (vector-map-reduce* this init ^IFn$DDDDD g ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (matrix-reduce
    ([this init a]
     (matrix-reduce* identity this init a))
    ([this init a b]
     (matrix-reduce* identity this init a b))
    ([this init a b c]
     (matrix-reduce* identity this init a b c))
    ([this init a b c d]
     (matrix-reduce* identity this init a b c d)))
  (matrix-map-reduce
    ([this init g a]
     (matrix-map-reduce* identity this init ^IFn$DD g a))
    ([this init g a b]
     (matrix-map-reduce* identity this init ^IFn$DDD g a b))
    ([this init g a b c]
     (matrix-map-reduce* identity this init ^IFn$DDDD g a b c))
    ([this init g a b c d]
     (matrix-map-reduce* identity this init ^IFn$DDDDD g a b c d))))

(extend-type IFn$DLDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* this (double init) ^RealVector x))
    ([this init x y]
     (vector-reduce-indexed* this (double init) ^RealVector x ^RealVector y))
    ([this init x y z]
     (vector-reduce-indexed* this (double init) ^RealVector x ^RealVector y ^RealVector z))
    ([this init x y z v]
     (vector-reduce-indexed* this (double init) ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce-indexed* this (double init) ^IFn$DD g ^RealVector x))
    ([this init g x y]
     (vector-map-reduce-indexed* this (double init) ^IFn$DDD g ^RealVector x ^RealVector y))
    ([this init g x y z]
     (vector-map-reduce-indexed* this (double init) ^IFn$DDDD g ^RealVector x ^RealVector y ^RealVector z))
    ([this init g x y z v]
     (vector-map-reduce-indexed* this (double init) ^IFn$DDDDD g ^RealVector x ^RealVector y ^RealVector z ^RealVector v))))

(extend-type IFn$OLDO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* this init ^RealVector x))
    ([this init x y]
     (vector-reduce-indexed* this init ^RealVector x ^RealVector y))
    ([this init x y z]
     (vector-reduce-indexed* this init ^RealVector x ^RealVector y ^RealVector z))
    ([this init x y z v]
     (vector-reduce-indexed* this init ^RealVector x ^RealVector y ^RealVector z ^RealVector v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce-indexed* this init ^IFn$DD g ^RealVector x))
    ([this init g x y]
     (vector-map-reduce-indexed* this init ^IFn$DDD g ^RealVector x ^RealVector y))
    ([this init g x y z]
     (vector-map-reduce-indexed* this init ^IFn$DDDD g ^RealVector x ^RealVector y ^RealVector z))
    ([this init g x y z v]
     (vector-map-reduce-indexed* this init ^IFn$DDDDD g ^RealVector x ^RealVector y ^RealVector z ^RealVector v))))
