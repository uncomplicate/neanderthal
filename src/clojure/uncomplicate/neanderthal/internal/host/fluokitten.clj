;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.fluokitten
  (:refer-clojure :exclude [accessor])
  (:require [uncomplicate.commons
             [core :refer [let-release double-fn]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [copy dim ncols transfer! vctr ge dia]]
             [block :refer [buffer offset]]]
            [uncomplicate.neanderthal.internal
             [api :refer [factory compatible? engine raw subcopy sum ReductionFunction
                          vector-reduce vector-map-reduce matrix-reduce matrix-map-reduce
                          data-accessor create-ge navigator storage region view-vctr create-vector]]
             [common :refer [real-accessor]]
             [navigation :refer [doall-layout real-navigator]]])
  (:import [clojure.lang IFn IFn$DLDD IFn$ODO IFn$OLDO]
           [uncomplicate.neanderthal.internal.api RealBufferAccessor RealVector RealMatrix
            Vector Matrix DiagonalMatrix Changeable LayoutNavigator RealLayoutNavigator
            DenseStorage Region]))

(def ^{:no-doc true :const true} FITTING_DIMENSIONS_MATRIX_MSG
  "Matrices should have fitting dimensions.")

(def ^{:no-doc true :const true} DIMENSIONS_MSG
  "Vectors should have fitting dimensions.")

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

(defmacro entry*
  ([vtype v i]
   `(.entry ~(with-meta v {:tag vtype}) ~i))
  ([vtype v i etype e]
   `(.set ~(with-meta v {:tag vtype}) ~i (~etype ~e))))

(defmacro map-entries-i [vtype f i & xs]
  `(~f ~@(map #(list `entry* vtype % i) xs)))

(defmacro vector-fmap* [vtype etype res f & xs]
  (if (< (count xs) 5)
    `(do
       (if (check-vector-dimensions ~@xs)
         (dotimes [i# (dim ~res)]
           (entry* ~vtype ~res i# ~etype (map-entries-i ~vtype ~f i# ~@xs)))
         (dragan-says-ex DIMENSIONS_MSG {:xs (map str ~xs)}))
       ~res)
    `(dragan-says-ex "Vector fmap supports up to 4 vectors.")))

(defmacro vector-reduce* [vtype etype acctype f init & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# (~acctype ~init)]
           (if (< i# dim-x#)
             (recur (inc i#) (~acctype (~f acc# (~etype (map-entries-i ~vtype + i# ~@xs)))))
             acc#)))
       (dragan-says-ex DIMENSIONS_MSG {:xs (map str ~xs)}))
    `(dragan-says-ex "Vector fold supports up to 4 vectors.")))

(defmacro vector-map-reduce* [vtype etype acctype f init g & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# (~acctype ~init)]
           (if (< i# dim-x#)
             (recur (inc i#) (~acctype (~f acc# (~etype (map-entries-i ~vtype ~g i# ~@xs)))))
             acc#)))
       (dragan-says-ex DIMENSIONS_MSG {:xs (map str ~xs)}))
    `(dragan-says-ex "Vector foldmap supports up to 4 vectors.")))

(defmacro ^:private vector-reduce-indexed* [vtype etype acctype f init & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# (~acctype ~init)]
           (if (< i# dim-x#)
             (recur (inc i#) (~acctype (~f acc# i# (~etype (map-entries-i ~vtype + i# ~@xs)))))
             acc#)))
       (dragan-says-ex DIMENSIONS_MSG {:xs (map str ~xs)}))
    `(dragan-says-ex "Vector fold supports up to 4 vectors.")))

(defmacro ^:private vector-map-reduce-indexed* [vtype etype acctype f init g & xs]
  (if (< (count xs) 5)
    `(if (check-vector-dimensions ~@xs)
       (let [dim-x# (dim ~(first xs))]
         (loop [i# 0 acc# (~acctype ~init)]
           (if (< i# dim-x#)
             (recur (inc i#) (~acctype (~f acc# i# (~etype (map-entries-i ~vtype ~g i# ~@xs)))))
             acc#)))
       (dragan-says-ex DIMENSIONS_MSG {:xs (map str ~xs)}))
    `(dragan-says-ex "Vector foldmap supports up to 4 vectors.")))

(defn vector-op [^Vector x & ws]
  (let-release [res (create-vector (factory x) (transduce (map dim) + (.dim x) ws) false)]
    (loop [pos 0 w x ws ws]
      (when w
        (if (compatible? res w)
          (subcopy (engine w) w res 0 (dim w) pos)
          (dragan-says-ex "You can not apply op on incompatiple vectors." {:res res :w w}))
        (recur (+ pos (dim w)) (first ws) (next ws))))
    res))

(defmacro vector-fmap
  ([creator vtype etype]
   `(fn
      ([x# f#]
       (let-release [res# (~creator x#)]
         (vector-fmap* ~vtype ~etype res# f# x#)))
      ([x# f# y#]
       (let-release [res# (~creator x#)]
         (vector-fmap* ~vtype ~etype res# f# x# y#)))
      ([x# f# y# z#]
       (let-release [res# (~creator x#)]
         (vector-fmap* ~vtype ~etype res# f# x# y# z#)))
      ([x# f# y# z# v#]
       (let-release [res# (~creator x#)]
         (vector-fmap* ~vtype ~etype res# f# x# y# z# v#)))
      ([x# f# y# z# v# es#]
       (dragan-says-ex "Vector fmap supports up to 4 arrays."))))
  ([vtype etype]
   `(let [fmap-fn# (vector-fmap raw ~vtype ~etype)]
      (fn
        ([x# f#]
         (fmap-fn# x# f#))
        ([x# f# xs#]
         (apply fmap-fn# x# f# xs#))))))

(defn vector-fold
  ([x]
   (sum (engine x) x))
  ([x f init]
   (vector-reduce f init x))
  ([x f init y]
   (vector-reduce f init x y))
  ([x f init y z]
   (vector-reduce f init x y z))
  ([x f init y z v]
   (vector-reduce f init x y z v))
  ([_ _ _ _ _ _ _]
   (dragan-says-ex "fold with more than four arguments is not available for vectors.")))

(defmacro vector-foldmap [vtype etype]
  `(fn
     ([x# g#]
      (vector-map-reduce* ~vtype ~etype ~etype + 0 g# x#))
     ([x# g# f# init#]
      (vector-map-reduce f# init# g# x#))
     ([x# g# f# init# y#]
      (vector-map-reduce f# init# g# x# y#))
     ([x# g# f# init# y# z#]
      (vector-map-reduce f# init# g# x# y# z#))
     ([x# g# f# init# y# z# v#]
      (vector-map-reduce f# init# g# x# y# z# v#))
     ([_# _# _# _# _# _# _# _#]
      (dragan-says-ex "foldmap with more than four arguments is not available for vectors."))))

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

(defmacro map-entries-ij [nav f i j & xs]
  `(~f ~@(map #(list `.get nav % i j) xs)))

(defmacro matrix-fmap* [etype res f & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~res)
             stor# (storage ~res)
             reg# (region ~res)
             da# (real-accessor ~res)
             buff# (buffer ~res)
             ofst# (offset ~res)]
         (doall-layout nav# stor# reg# i# j#
                       (.set da# buff# (+ ofst# (.index stor# i# j#))
                             (~etype (map-entries-ij nav# ~f i# j# ~@as))))
         ~res)
       (dragan-says-ex FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)}))
    `(dragan-says-ex "Matrix fmap supports up to 4 matrices.")))

(defmacro ^:private matrix-reduce* [acctype f init & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~(first as))
             stor# (storage ~(first as))
             reg# (region ~(first as))
             fd# (.fd stor#)]
         (loop [j# 0 acc# (~acctype ~init)]
           (if (< j# fd#)
             (recur (inc j#)
                    (~acctype
                     (let [end# (.end nav# reg# j#)]
                       (loop [i# (.start nav# reg# j#) acc# acc#]
                         (if (< i# end#)
                           (recur (inc i#) (~acctype (~f acc# (map-entries-ij nav# + i# j# ~@as))))
                           acc#)))))
             acc#)))
       (dragan-says-ex FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)}))
    `(dragan-says-ex "Matrix fold supports up to 4 matrices.")))

(defmacro ^:private  matrix-map-reduce* [acctype f init g & as]
  (if (< (count as) 5)
    `(if (check-matrix-dimensions ~@as)
       (let [nav# (real-navigator ~(first as))
             stor# (storage ~(first as))
             reg# (region ~(first as))
             fd# (.fd stor#)]
         (loop [j# 0 acc# (~acctype ~init)]
           (if (< j# fd#)
             (recur
              (inc j#)
              (~acctype
               (let [end# (.end nav# reg# j#)]
                 (loop [i# (.start nav# reg# j#) acc# acc#]
                   (if (< i# end#)
                     (recur (inc i#) (~acctype (~f acc# (map-entries-ij nav# ~g i# j# ~@as))))
                     acc#)))))
             acc#)))
       (dragan-says-ex FITTING_DIMENSIONS_MATRIX_MSG {:as (map str ~as)}))
    `(dragan-says-ex "Matrix foldmap supports up to 4 matrices.")))

(defmacro matrix-fmap
  ([creator etype]
   `(fn
      ([a# f#]
       (let-release [res# (~creator a#)]
         (matrix-fmap* ~etype res# f# a#)))
      ([a# f# b#]
       (let-release [res# (~creator a#)]
         (matrix-fmap* ~etype res# f# a# b#)))
      ([a# f# b# c#]
       (let-release [res# (~creator a#)]
         (matrix-fmap* ~etype res# f# a# b# c#)))
      ([a# f# b# c# d#]
       (let-release [res# (~creator a#)]
         (matrix-fmap* ~etype res# f# a# b# c# d#)))
      ([a# f# b# c# d# es#]
       (dragan-says-ex "Matrix fmap! supports up to 4 matrices."))))
  ([etype]
   `(let [fmap-fn# (matrix-fmap raw ~etype)]
      (fn
        ([a# f#]
         (fmap-fn# a# f#))
        ([a# f# as#]
         (apply fmap-fn# a# f# as#))))))

(defmacro diagonal-fmap
  ([creator vtype etype]
   `(fn
      ([a# f#]
       (let-release [res# (~creator a#)]
         (let [vr# (view-vctr res#)
               va# (view-vctr a#)]
           (vector-fmap* ~vtype ~etype vr# f# va#))
         res#))
      ([a# f# b#]
       (let-release [res# (~creator a#)]
         (if (instance? DiagonalMatrix b#)
           (let [vr# (view-vctr res#)
                 va# (view-vctr a#)
                 vb# (view-vctr b#)]
             (vector-fmap* ~vtype ~etype vr# f# va# vb#))
           (dotimes [i# 3]
             (let [vr# (dia res# (dec i#))
                   va# (dia a# (dec i#))
                   vb# (dia b# (dec i#))]
               (vector-fmap* ~vtype ~etype vr# f# va# vb#))))
         res#))
      ([a# f# b# c#]
       (let-release [res# (~creator (view-vctr a#))]
         (if (instance? DiagonalMatrix b#)
           (let [vr# (view-vctr res#)
                 va# (view-vctr a#)
                 vb# (view-vctr b#)
                 vc# (view-vctr c#)]
             (vector-fmap* ~vtype ~etype vr# f# va# vb# vc#))
           (dotimes [i# 3]
             (let [vr# (dia res# (dec i#))
                   va# (dia a# (dec i#))
                   vb# (dia b# (dec i#))
                   vc# (dia c# (dec i#))]
               (vector-fmap* ~vtype ~etype vr# f# va# vb# vc#))))
         res#))
      ([a# f# b# c# d#]
       (let-release [res# (~creator (view-vctr a#))]
         (if (instance? DiagonalMatrix b#)
           (let [vr# (view-vctr res#)
                 va# (view-vctr a#)
                 vb# (view-vctr b#)
                 vc# (view-vctr c#)
                 vd# (view-vctr d#)]
             (vector-fmap* ~vtype ~etype vr# f# va# vb# vc# vd#))
           (dotimes [i# 3]
             (let [vr# (dia res# (dec i#))
                   va# (dia a# (dec i#))
                   vb# (dia b# (dec i#))
                   vc# (dia c# (dec i#))
                   vd# (dia d# (dec i#))]
               (vector-fmap* ~vtype ~etype vr# f# va# vb# vc# vd#))))
         res#))
      ([a# f# b# c# d# es#]
       (dragan-says-ex "Matrix fmap! supports up to 4 matrices."))))
  ([vtype etype]
   `(let [fmap-fn# (diagonal-fmap raw ~vtype ~etype)]
      (fn
        ([a# f#]
         (fmap-fn# a# f#))
        ([a# f# as#]
         (apply fmap-fn# a# f# as#))))))

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
   (dragan-says-ex "Matrix fold supports up to 4 matrices.")))

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
   (dragan-says-ex "Matrix fold supports up to 4 matrices.")))

(let [p+ (double-fn +)]

  (defn matrix-foldmap
    ([a g]
     (matrix-map-reduce* double p+ 0.0 g a))
    ([a g f init]
     (matrix-map-reduce f init g a))
    ([a g f init b]
     (matrix-map-reduce f init g a b))
    ([a g f init b c]
     (matrix-map-reduce f init g a b c))
    ([a g f init b c d]
     (matrix-map-reduce f init g a b c d))
    ([a g f init b c d es]
     (dragan-says-ex "Matrix foldmap supports up to 4 matrices."))))

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
   (dragan-says-ex "Matrix foldmap supports up to 4 matrices.")))

;; ============================ Primitive function extensions ==================

(extend-type Object
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* RealVector double double this init x))
    ([this init x y]
     (vector-reduce* RealVector double double this init x y))
    ([this init x y z]
     (vector-reduce* RealVector double double this init x y z))
    ([this init x y z v]
     (vector-reduce* RealVector double double this init x y z v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce* RealVector double double this init g x))
    ([this init g x y]
     (vector-map-reduce* RealVector double double this init g x y))
    ([this init g x y z]
     (vector-map-reduce* RealVector double double this init g x y z))
    ([this init g x y z v]
     (vector-map-reduce* RealVector double double this init g x y z v)))
  (matrix-reduce
    ([this init a]
     (matrix-reduce* double this init a))
    ([this init a b]
     (matrix-reduce* double this init a b))
    ([this init a b c]
     (matrix-reduce* double this init a b c))
    ([this init a b c d]
     (matrix-reduce* double this init a b c d)))
  (matrix-map-reduce
    ([this init g a]
     (matrix-map-reduce* double this init g a))
    ([this init g a b]
     (matrix-map-reduce* double this init g a b))
    ([this init g a b c]
     (matrix-map-reduce* double this init g a b c))
    ([this init g a b c d]
     (matrix-map-reduce* double this init g a b c d))))

(extend-type IFn$ODO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce* RealVector double identity this init x))
    ([this init x y]
     (vector-reduce* RealVector double identity this init x y))
    ([this init x y z]
     (vector-reduce* RealVector double identity this init x y z))
    ([this init x y z v]
     (vector-reduce* RealVector double identity this init x y z v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce* RealVector double identity this init g x))
    ([this init g x y]
     (vector-map-reduce* RealVector double identity this init g x y))
    ([this init g x y z]
     (vector-map-reduce* RealVector double identity this init g x y z))
    ([this init g x y z v]
     (vector-map-reduce* RealVector double identity this init g x y z v)))
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
     (matrix-map-reduce* identity this init g a))
    ([this init g a b]
     (matrix-map-reduce* identity this init g a b))
    ([this init g a b c]
     (matrix-map-reduce* identity this init g a b c))
    ([this init g a b c d]
     (matrix-map-reduce* identity this init g a b c d))))

(extend-type IFn$DLDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* RealVector double double this init x))
    ([this init x y]
     (vector-reduce-indexed* RealVector double double this init x y))
    ([this init x y z]
     (vector-reduce-indexed* RealVector double double this init x y z))
    ([this init x y z v]
     (vector-reduce-indexed* RealVector double double this init x y z v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce-indexed* RealVector double double this init g x))
    ([this init g x y]
     (vector-map-reduce-indexed* RealVector double double this init g x y))
    ([this init g x y z]
     (vector-map-reduce-indexed* RealVector double double this init g x y z))
    ([this init g x y z v]
     (vector-map-reduce-indexed* RealVector double double this init g x y z v))))

(extend-type IFn$OLDO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (vector-reduce-indexed* RealVector double identity this init x))
    ([this init x y]
     (vector-reduce-indexed* RealVector double identity this init x y))
    ([this init x y z]
     (vector-reduce-indexed* RealVector double identity this init x y z))
    ([this init x y z v]
     (vector-reduce-indexed* RealVector double identity this init x y z v)))
  (vector-map-reduce
    ([this init g x]
     (vector-map-reduce-indexed* RealVector double identity this init g x))
    ([this init g x y]
     (vector-map-reduce-indexed* RealVector double identity this init g x y))
    ([this init g x y z]
     (vector-map-reduce-indexed* RealVector double identity this init g x y z))
    ([this init g x y z v]
     (vector-map-reduce-indexed* RealVector double identity this init g x y z v))))
