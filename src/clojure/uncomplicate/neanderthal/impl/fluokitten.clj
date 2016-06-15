(ns uncomplicate.neanderthal.impl.fluokitten
  (:refer-clojure :exclude [accessor])
  (:require [uncomplicate.commons.core :refer [let-release]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [copy create-raw trans dim ncols mrows sum col row]]])
  (:import [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [uncomplicate.neanderthal.protocols
            BLASPlus RealVector RealMatrix Vector Matrix RealChangeable Block]))

(def ^{:no-doc true :const true} FITTING_DIMENSIONS_MATRIX_MSG
  "Matrices should have fitting dimensions.")

;; ================== Fluokitten implementation  ===============================
;; ---------------------- Vector Fluokitten funcitions -------------------------

(extend-type IFn$DDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (.entry ^RealVector x i)))
           acc))))
    ([this init x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i))))
           acc))))
    ([this init x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i)
                                                   (.entry ^RealVector z i))))
           acc))))
    ([this init x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i)
                                                   (.entry ^RealVector z i)
                                                   (.entry ^RealVector v i))))
           acc)))))
  (vector-reduce-map
    ([this init g x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DD g (.entry ^RealVector x i))))
           acc))))
    ([this init g x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i))))
           acc))))
    ([this init g x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i)
                                               (.entry ^RealVector z i))))
           acc))))
    ([this init g x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDDDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i)
                                               (.entry ^RealVector z i)
                                               (.entry ^RealVector v i))))
           acc))))))

(extend-type IFn$ODO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (.entry ^RealVector x i)))
           acc))))
    ([this init x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i))))
           acc))))
    ([this init x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i)
                                                   (.entry ^RealVector z i))))
           acc))))
    ([this init x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc (+ (.entry ^RealVector x i)
                                                   (.entry ^RealVector y i)
                                                   (.entry ^RealVector z i)
                                                   (.entry ^RealVector v i))))
           acc)))))
  (vector-reduce-map
    ([this init g x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DD g (.entry ^RealVector x i))))
           acc))))
    ([this init g x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i))))
           acc))))
    ([this init g x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i)
                                               (.entry ^RealVector z i))))
           acc))))
    ([this init g x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc (.invokePrim ^IFn$DDDDD g
                                               (.entry ^RealVector x i)
                                               (.entry ^RealVector y i)
                                               (.entry ^RealVector z i)
                                               (.entry ^RealVector v i))))
           acc))))))


(extend-type IFn$DLDD
  ReductionFunction
  (vector-reduce
    ([this init x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (.entry ^RealVector x i)))
           acc))))
    ([this init x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i))))
           acc))))
    ([this init x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i)
                                                     (.entry ^RealVector z i))))
           acc))))
    ([this init x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i)
                                                     (.entry ^RealVector z i)
                                                     (.entry ^RealVector v i))))
           acc)))))
  (vector-reduce-map
    ([this init g x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DD g (.entry ^RealVector x i))))
           acc))))
    ([this init g x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i))))
           acc))))
    ([this init g x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i)
                                                 (.entry ^RealVector z i))))
           acc))))
    ([this init g x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc (double init)]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDDDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i)
                                                 (.entry ^RealVector z i)
                                                 (.entry ^RealVector v i))))
           acc))))))

(extend-type IFn$OLDO
  ReductionFunction
  (vector-reduce
    ([this init x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (.entry ^RealVector x i)))
           acc))))
    ([this init x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i))))
           acc))))
    ([this init x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i)
                                                     (.entry ^RealVector z i))))
           acc))))
    ([this init x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur (inc i) (.invokePrim this acc i (+ (.entry ^RealVector x i)
                                                     (.entry ^RealVector y i)
                                                     (.entry ^RealVector z i)
                                                     (.entry ^RealVector v i))))
           acc)))))
  (vector-reduce-map
    ([this init g x]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DD g
                                                 (.entry ^RealVector x i))))
           acc))))
    ([this init g x y]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i))))
           acc))))
    ([this init g x y z]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i)
                                                 (.entry ^RealVector z i))))
           acc))))
    ([this init g x y z v]
     (let [dim-x (.dim ^Vector x)]
       (loop [i 0 acc init]
         (if (< i dim-x)
           (recur
            (inc i)
            (.invokePrim this acc i (.invokePrim ^IFn$DDDDD g
                                                 (.entry ^RealVector x i)
                                                 (.entry ^RealVector y i)
                                                 (.entry ^RealVector z i)
                                                 (.entry ^RealVector v i))))
           acc))))))

(defn ^:private vector-fmap*
  ([f ^Vector x ]
   (dotimes [i (.dim x)]
     (.alter ^RealChangeable x i f))
   x)
  ([^IFn$DDD f ^RealVector x^RealVector y]
   (dotimes [i (.dim ^Vector x)]
     (.set ^RealChangeable x i
           (.invokePrim f (.entry x i) (.entry y i)))))
  ([^IFn$DDDD f ^RealVector x ^RealVector y ^RealVector z]
   (dotimes [i (.dim ^Vector x)]
     (.set ^RealChangeable x i
           (.invokePrim f (.entry x i) (.entry y i) (.entry z i)))))
  ([^IFn$DDDDD f ^RealVector x ^RealVector y ^RealVector z ^RealVector v]
   (dotimes [i (.dim ^Vector x)]
     (.set ^RealChangeable x i
           (.invokePrim f (.entry x i) (.entry y i) (.entry z i) (.entry v i))))))

(defn vector-fmap!
  ([x f]
   (vector-fmap* f x)
   x)
  ([^Vector x f ^Vector y]
   (do
     (if (<= (.dim x) (.dim y))
       (vector-fmap* f x y)
       (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
     x))
  ([^Vector x f ^Vector y ^Vector z]
   (do
     (if (<= (.dim x) (min (.dim y) (.dim z)))
       (vector-fmap* f x y z)
       (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
     x))
  ([^Vector x f ^Vector y ^Vector z ^Vector v]
   (do
     (if (<= (.dim x) (min (.dim y) (.dim z) (.dim v)))
       (vector-fmap* f x y z v)
       (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
     x))
  ([x f y z w ws]
   (throw (UnsupportedOperationException. "Vector fmap support up to 4 vectors."))))

(defn vector-fmap
  ([x f]
   (let-release [res (copy x)]
     (vector-fmap! res f)))
  ([x f xs]
   (let-release [res (copy x)]
     (apply vector-fmap! res f xs))))

(defn vector-fold
  ([^RealVector x]
   (sum x))
  ([^Vector x f init]
   (vector-reduce f init x))
  ([^Vector x f init ^Vector y]
   (if (<= (.dim x) (.dim y))
     (vector-reduce f init x y)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([^Vector x f init ^Vector y ^Vector z]
   (if (<= (.dim x) (min (.dim y) (.dim z)))
     (vector-reduce f init x y z)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([^Vector x f init ^Vector y ^Vector z ^Vector v]
   (if (<= (.dim x) (min (.dim y) (.dim z) (.dim v)))
     (vector-reduce f init x y z v)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([x f init y z v ws]
   (throw (UnsupportedOperationException. "Vector fold support up to 4 vectors."))))

(defn vector-foldmap
  ([^RealVector x ^IFn$DD g]
   (let [dim-x (.dim x)]
     (loop [i 0 acc 0.0]
       (if (< i dim-x)
         (recur (inc i)
                (+ acc (.invokePrim g (.entry x i))))
         acc))))
  ([^Vector x g f init]
   (vector-reduce-map f init g x))
  ([^Vector x g f init ^Vector y]
   (if (<= (.dim x) (.dim y))
     (vector-reduce-map f init g x y)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([^Vector x g f init ^Vector y ^Vector z]
   (if (<= (.dim x) (min (.dim y) (.dim z)))
     (vector-reduce-map f init g x y z)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([^Vector x g f init ^Vector y ^Vector z ^Vector v]
   (if (<= (.dim x) (min (.dim y) (.dim z) (.dim v)))
     (vector-reduce-map f init g x y z v)
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([x g f init y z v ws]
   (throw (UnsupportedOperationException. "Vector foldmap support up to 4 vectors."))))

(defn ^:private vector-op* [^Vector x & ws]
  (let-release [res ^Vector (create-raw (factory x) (transduce (map dim) + (.dim x) ws))]
    (.subcopy ^BLASPlus (engine x) x res 0 (.dim x) 0)
    (reduce (fn ^long [^long pos ^Vector w]
              (if (compatible res w)
                (do
                  (.subcopy ^BLASPlus (engine w) w res 0 (.dim w) pos)
                  (+ pos (.dim w)))
                (throw (UnsupportedOperationException.
                        (format INCOMPATIBLE_BLOCKS_MSG res w)))))
            (.dim x)
            ws)
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
   (throw (UnsupportedOperationException.
           "This operation would be slow on primitive vectors."))))

;; ---------------------- Matrix Fluokitten funcitions -------------------------

(defn ^:private check-matrix-dimensions
  ([^Matrix a ^Matrix b]
   (and (<= (.ncols a) (.ncols b))
        (<= (.mrows a) (.mrows b))))
  ([^Matrix a ^Matrix b ^Matrix c]
   (and (<= (.ncols a) (min (.ncols b) (.ncols c)))
        (<= (.mrows a) (min (.mrows b) (.ncols c)))))
  ([^Matrix a ^Matrix b ^Matrix c ^Matrix d]
   (and (<= (.ncols a) (min (.ncols b) (.ncols c) (.ncols d)))
        (<= (.mrows a) (min (.mrows b) (.ncols c) (.ncols d))))))

(defn matrix-fmap!
  ([^Matrix a f]
   (if (column-major? a)
     (dotimes [i (.ncols a)]
       (vector-fmap* f (.col a i)))
     (dotimes [i (.mrows a)]
       (vector-fmap* f (.row a i))))
   a)
  ([^Matrix a f ^Matrix b]
   (if (check-matrix-dimensions a b)
     (if (column-major? a)
       (dotimes [i (.ncols a)]
         (vector-fmap* f (.col a i) (.col b i)))
       (dotimes [i (.mrows a)]
         (vector-fmap* f (.row a i) (.row b i))))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG)))
   a)
  ([^Matrix a f ^Matrix b ^Matrix c]
   (if (check-matrix-dimensions a b c)
     (if (column-major? a)
       (dotimes [i (.ncols a)]
         (vector-fmap* f (.col a i) (.col b i) (.col c i)))
       (dotimes [i (.mrows a)]
         (vector-fmap* f (.row a i) (.row b i) (.row c i))))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG)))
   a)
  ([^Matrix a f ^Matrix b ^Matrix c ^Matrix d]
   (if (check-matrix-dimensions a b c d)
     (if (column-major? a)
            (dotimes [i (.ncols a)]
              (vector-fmap* f (.col a i) (.col b i) (.col c i) (.col d i)))
            (dotimes [i (.mrows a)]
              (vector-fmap* f (.row a i) (.row b i) (.row c i) (.row d i))))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG)))
   a)
  ([a f b c d es]
   (throw (UnsupportedOperationException. "Matrix fmap support up to 4 matrices."))))

(defn matrix-fmap
  ([a f]
   (let-release [res (copy a)]
     (matrix-fmap! res f)))
  ([a f as]
   (let-release [res (copy a)]
     (apply matrix-fmap! res f as))))

(defn matrix-fold
  ([^RealMatrix a]
   (let [vf (if (column-major? a) col row)
         n (if (column-major? a) (.ncols a) (.mrows a))]
     (loop [i 0 acc 0.0]
       (if (< i n)
         (recur (inc i) (+ acc (sum (vf a i))))
         acc))))
  ([^Matrix a f init]
   (loop [i 0 acc init]
     (if (< i (.ncols a))
       (recur (inc i) (vector-reduce f acc (.col a i)))
       acc)))
  ([^Matrix a f init ^Matrix b]
   (if (check-matrix-dimensions a b)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i) (vector-reduce f acc (.col a i) (.col b i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix a f init ^Matrix b ^Matrix c]
   (if (check-matrix-dimensions a b c)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i) (vector-reduce f acc (.col a i) (.col b i) (.col c i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix a f init ^Matrix b ^Matrix c ^Matrix d]
   (if (check-matrix-dimensions a b c d)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i)
                (vector-reduce f acc (.col a i) (.col b i) (.col c i) (.col d i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([a f init b c d es]
   (throw (UnsupportedOperationException. "Matrix fold support up to 4 matrices."))))

(defn matrix-foldmap
  ([^RealMatrix a ^IFn$DD g]
   (loop [j 0 acc 0.0]
     (if (< j (.ncols a))
       (recur (inc j)
              (double
               (loop [i 0 acc acc]
                 (if (< i (.mrows a))
                   (recur (inc i)
                          (+ acc (.invokePrim g (.entry a i j))))
                   acc))))
       acc)))
  ([^Matrix a g f init]
   (loop [i 0 acc init]
     (if (< i (.ncols a))
       (recur (inc i) (vector-reduce-map f acc g (.col a i)))
       acc)))
  ([^Matrix a g f init ^Matrix b]
   (if (check-matrix-dimensions a b)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i) (vector-reduce-map f acc g (.col a i) (.col b i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix a g f init ^Matrix b ^Matrix c]
   (if (check-matrix-dimensions a b c)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i) (vector-reduce-map f acc g (.col a i) (.col b i) (.col c i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix a g f init ^Matrix b ^Matrix c ^Matrix d]
   (if (check-matrix-dimensions a b c d)
     (loop [i 0 acc init]
       (if (< i (.ncols a))
         (recur (inc i)
                (vector-reduce-map f acc g (.col a i) (.col b i) (.col c i) (.col d i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([x g f init y z v ws]
   (throw (UnsupportedOperationException. "Matrix fold supports up to 4 matrices."))))

(defn matrix-op* [^Matrix a & bs]
  (let-release [res ^Matrix (if (column-major? a)
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
      (if (column-major? a)
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
