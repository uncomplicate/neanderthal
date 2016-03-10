(ns uncomplicate.neanderthal.impl.buffer-block
  (:refer-clojure :exclude [accessor])
  (:require [vertigo
             [core :refer [wrap]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 wrap-byte-seq]]]
            [uncomplicate.fluokitten.protocols
             :refer [PseudoFunctor Foldable Magma Monoid]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [transfer! copy! dim mrows ncols create-raw]]]
            [uncomplicate.clojurecl.core :refer [Releaseable release]])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus RealBufferAccessor BufferAccessor DataAccessor
            RealVector RealMatrix Vector Matrix RealChangeable Block]))

(def ^{:no-doc true :const true} FITTING_DIMENSIONS_MATRIX_MSG
  "Matrices should have fitting dimensions.")

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

(defn ^:private p- ^double [^double x ^double y]
  (- x y))

(defn ^:private p+ ^double [^double x ^double y]
  (+ x y))

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

(defn ^:private vector-fmap!
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

(defn ^:private vector-fold
  ([^RealVector x]
   (let [dim-x (.dim x)]
     (loop [i 0 acc 0.0]
       (if (< i dim-x)
         (recur (inc i)
                (+ acc (.entry x i)))
         acc))))
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

(defn ^:private vector-foldmap
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
  (let [res ^Vector (create-raw (factory x) (transduce (map dim) + (.dim x) ws))
        eng ^BLASPlus (engine res)]
   (try
     (.subcopy eng x res 0 (.dim x) 0)
     (reduce (fn ^long [^long pos ^Vector w]
               (if (compatible res w)
                 (do
                   (.subcopy eng w res 0 (.dim w) pos )
                   (+ pos (.dim w)))
                 (throw (UnsupportedOperationException.
                         (format INCOMPATIBLE_BLOCKS_MSG res w)))))
             (.dim x)
             ws)
     res
     (catch Exception e
       (do
         (release res)
         (throw e))))))

(defn ^:private vector-op
  ([^Vector x ^Vector y]
   (vector-op* x y))
  ([^Vector x ^Vector y ^Vector z]
   (vector-op* x y z))
  ([^Vector x ^Vector y ^Vector z ^Vector v]
   (vector-op* x y z v))
  ([^Vector x ^Vector y ^Vector z ^Vector v ws]
   (apply vector-op* x y z v ws)))

;; ---------------------- Matrix Fluokitten funcitions -------------------------

(defn ^:private check-matrix-dimensions
  ([^Matrix x ^Matrix y]
   (and (<= (.ncols x) (.ncols y))
        (<= (.mrows x) (.mrows y))))
  ([^Matrix x ^Matrix y ^Matrix z]
   (and (<= (.ncols x) (min (.ncols y) (.ncols z)))
        (<= (.mrows x) (min (.mrows y) (.ncols z)))))
  ([^Matrix x ^Matrix y ^Matrix z ^Matrix v]
   (and (<= (.ncols x) (min (.ncols y) (.ncols z) (.ncols v)))
        (<= (.mrows x) (min (.mrows y) (.ncols z) (.ncols v))))))

(defn ^:private matrix-fmap!
  ([^Matrix x f]
   (do (if (column-major? x)
         (dotimes [i (.ncols x)]
           (vector-fmap* f (.col x i)))
         (dotimes [i (.mrows x)]
           (vector-fmap* f (.row x i))))
       x))
  ([^Matrix x f ^Matrix y]
   (if (check-matrix-dimensions x y)
     (do (if (column-major? x)
           (dotimes [i (.ncols x)]
             (vector-fmap* f (.col x i) (.col y i)))
           (dotimes [i (.mrows x)]
             (vector-fmap* f (.row x i) (.row y i))))
         x)
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix x f ^Matrix y ^Matrix z]
   (if (check-matrix-dimensions x y z)
     (do (if (column-major? x)
           (dotimes [i (.ncols x)]
             (vector-fmap* f (.col x i) (.col y i) (.col z i)))
           (dotimes [i (.mrows x)]
             (vector-fmap* f (.row x i) (.row y i) (.row z i))))
         x)
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix x f ^Matrix y ^Matrix z ^Matrix v]
   (if (check-matrix-dimensions x y z v)
     (do (if (column-major? x)
           (dotimes [i (.ncols x)]
             (vector-fmap* f (.col x i) (.col y i) (.col z i) (.col v i)))
           (dotimes [i (.mrows x)]
             (vector-fmap* f (.row x i) (.row y i) (.row z i) (.col v i))))
         x)
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([x f y z w ws]
   (throw (UnsupportedOperationException. "Matrix fmap support up to 4 matrices."))))

(defn ^:private matrix-fold
  ([^RealMatrix x]
   (loop [j 0 acc 0.0]
     (if (< j (.ncols x))
       (recur (inc j)
              (double
               (loop [i 0 acc acc]
                 (if (< i (.mrows x))
                   (recur (inc i)
                          (+ acc (.entry x i j)))
                   acc))))
       acc)))
  ([^Matrix x f init]
   (loop [i 0 acc init]
     (if (< i (.ncols x))
       (recur (inc i) (vector-reduce f acc (.col x i)))
       acc)))
  ([^Matrix x f init ^Matrix y]
   (if (check-matrix-dimensions x y)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce f acc (.col x i) (.col y i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix x f init ^Matrix y ^Matrix z]
   (if (check-matrix-dimensions x y z)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce f acc (.col x i) (.col y i) (.col z i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix x f init ^Matrix y ^Matrix z ^Matrix v]
   (if (check-matrix-dimensions x y z v)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce f acc (.col x i) (.col y i) (.col z i) (.col v i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([x f init y z v ws]
   (throw (UnsupportedOperationException. "Matrix fold support up to 4 matrices."))))

(defn ^:private matrix-foldmap
  ([^RealMatrix x ^IFn$DD g]
   (loop [j 0 acc 0.0]
     (if (< j (.ncols x))
       (recur (inc j)
              (double
               (loop [i 0 acc acc]
                 (if (< i (.mrows x))
                   (recur (inc i)
                          (+ acc (.invokePrim g (.entry x i j))))
                   acc))))
       acc)))
  ([^Matrix x g f init]
   (loop [i 0 acc init]
     (if (< i (.ncols x))
       (recur (inc i) (vector-reduce-map f acc g (.col x i)))
       acc)))
  ([^Matrix x g f init ^Matrix y]
   (if (check-matrix-dimensions x y)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce-map f acc g (.col x i) (.col y i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))     ))
  ([^Matrix x g f init ^Matrix y ^Matrix z]
   (if (check-matrix-dimensions x y z)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce-map f acc g (.col x i) (.col y i) (.col z i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([^Matrix x g f init ^Matrix y ^Matrix z ^Matrix v]
   (if (check-matrix-dimensions x y z v)
     (loop [i 0 acc init]
       (if (< i (.ncols x))
         (recur (inc i) (vector-reduce-map f acc g (.col x i) (.col y i) (.col z i) (.col v i)))
         acc))
     (throw (IllegalArgumentException. FITTING_DIMENSIONS_MATRIX_MSG))))
  ([x g f init y z v ws]
   (throw (UnsupportedOperationException. "Matrix fold support up to 4 matrices."))))

;; ============ Realeaseable ===================================================

(defn ^:private clean-buffer [^ByteBuffer buffer]
  (do
    (if (.isDirect buffer)
      (.clean (.cleaner ^DirectByteBuffer buffer)))
    true))

;; ============ Real Buffer ====================================================

(deftype FloatBufferAccessor []
  DataAccessor
  (entryType [_]
    Float/TYPE)
  (entryWidth [_]
    Float/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Float/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Float/BYTES n)))
  (initialize [_ b]
    b)
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float32 (* Float/BYTES stride) 0 (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Float/BYTES k) (* Float/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getFloat buf (* Float/BYTES i)))
  (set [_ buf i val]
    (.putFloat buf (* Float/BYTES i) val)))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  DataAccessor
  (entryType [_]
    Double/TYPE)
  (entryWidth [_]
    Float/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Double/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Double/BYTES n)))
  (initialize [_ b]
    b)
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^uncomplicate.neanderthal.protocols.Factory fact
                          ^RealBufferAccessor accessor ^BLAS eng
                          ^Class entry-type ^Boolean master
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [this]
    (vector-fold this hash* (-> (hash :RealBlockVector) (hash-combine n))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (compatible x y) (= n (.dim ^Vector y)))
      (= 0.0 (vector-reduce-map p+ 0.0 p- x y))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, stride:%d]" entry-type n strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  clojure.lang.Seqable
  (seq [_]
    (.toSeq accessor buf strd))
  Container
  (raw [_]
    (create-vector fact n (.createDataSource accessor n) nil))
  (zero [this]
    (raw this))
  Monoid
  (id [x]
    (create-vector fact 0 (.createDataSource accessor 0) nil))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  Memory
  (compatible [_ y]
    (and (instance? RealBlockVector y) (= entry-type (.entryType ^Block y))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    buf)
  (offset [_]
    0)
  (stride [_]
    strd)
  (count [_]
    n)
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn
  (invoke [x i]
    (.entry x i))
  RealChangeable
  (set [x val]
    (do
      (dotimes [i n]
        (.set accessor buf (* strd i) val))
      x))
  (set [x i val]
    (do
      (.set accessor buf (* strd i) val)
      x))
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.get accessor buf (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (let [b (.slice accessor buf (* k strd) (* l strd))]
      (RealBlockVector. fact accessor eng entry-type false b l strd))))

(extend RealBlockVector
  PseudoFunctor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Magma
  {:op vector-op})

(defmethod transfer! [RealBlockVector RealBlockVector]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealBlockVector]
  [source ^RealBlockVector destination]
  (let [n (.dim destination)]
    (do
      (loop [i 0 src source]
        (if (and src (< i n))
          (do
            (.set destination i (first src))
            (recur (inc i) (next src)))
          destination)))))

(defmethod print-method RealBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(deftype RealGeneralMatrix [^uncomplicate.neanderthal.protocols.Factory fact
                            ^RealBufferAccessor accessor ^BLAS eng
                            ^Class entry-type ^Boolean master
                            ^ByteBuffer buf ^long m ^long n ^long ld ^long ord]
  Object
  (hashCode [this]
    (matrix-fold this hash* (-> (hash :RealGeneralMatrix) (hash-combine m) (hash-combine n))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= m (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (= 0.0 (matrix-foldmap a p- p+ 0.0 b))
      :default false))
  (toString [_]
    (format "#RealGeneralMatrix[%s, ord:%s, mxn:%dx%d, ld:%d]"
            entry-type (if (= COLUMN_MAJOR ord) "COL" "ROW")
            m n ld))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Container
  (raw [_]
    (create-matrix fact m n (.createDataSource accessor (* m n)) ord))
  (zero [this]
    (raw this))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  Memory
  (compatible [_ b]
    (and (or (instance? RealGeneralMatrix b) (instance? RealBlockVector b))
         (= entry-type (.entryType ^Block b))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    buf)
  (offset [_]
    0)
  (stride [_]
    ld)
  (order [_]
    ord)
  (count [_]
    (* m n))
  clojure.lang.Seqable
  (seq [a]
    (if (column-major? a)
      (map #(seq (.col a %)) (range 0 n))
      (map #(seq (.row a %)) (range 0 m))))
  clojure.lang.IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i m) (< -1 j n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j m n)))))
  clojure.lang.IFn
  (invoke [a i j]
    (if (and (< -1 (long i) m) (< -1 (long j) n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j m n)))))
  RealChangeable
  (set [a val]
    (do
      (if (= ld (if (column-major? a) m n))
        (dotimes [i (* m n)]
          (.set accessor buf i val))
        (if (column-major? a)
          (dotimes [i n]
            (.set ^RealChangeable (.col a i) val))
          (dotimes [i (.mrows a)]
            (.set ^RealChangeable (.row a i) val))))
      a))
  (set [a i j val]
    (do
      (if (= COLUMN_MAJOR ord)
        (.set accessor buf (+ (* ld j) i) val)
        (.set accessor buf (+ (* ld i) j) val))
      a))
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (.set a i j (.invokePrim ^IFn$DD f (.entry a i j))))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= COLUMN_MAJOR ord)
      (.get accessor buf (+ (* ld j) i))
      (.get accessor buf (+ (* ld i) j))))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (if (column-major? a)
      (let [b (.slice accessor buf i (inc (* (dec n) ld)))]
        (RealBlockVector. fact accessor
                          (vector-engine fact nil)
                          entry-type false b n ld))
      (let [b (.slice accessor buf (* ld i) n)]
        (RealBlockVector. fact accessor
                          (vector-engine fact nil)
                          entry-type false b n 1))))
  (col [a j]
    (if (column-major? a)
      (let [b (.slice accessor buf (* ld j) m)]
        (RealBlockVector. fact accessor
                          (vector-engine fact nil)
                          entry-type false b m 1))
      (let [b (.slice accessor buf j (inc (* (dec m) ld)))]
        (RealBlockVector. fact accessor
                          (vector-engine fact nil)
                          entry-type false b m ld))))
  (submatrix [a i j k l]
    (let [b (if (column-major? a)
              (.slice accessor buf (+ (* ld j) i) (* ld l))
              (.slice accessor buf (+ (* ld i) j) (* ld k)))]
      (RealGeneralMatrix. fact accessor eng entry-type false
                          b k l ld ord)))
  (transpose [a]
    (RealGeneralMatrix. fact accessor eng entry-type false
                        buf n m ld
                        (if (column-major? a) ROW_MAJOR COLUMN_MAJOR))))

(extend RealGeneralMatrix
  PseudoFunctor
  {:fmap! matrix-fmap!}
  Foldable
  {:fold matrix-fold})

(defmethod transfer! [RealGeneralMatrix RealGeneralMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealGeneralMatrix]
  [source ^RealGeneralMatrix destination]
  (let [m (.mrows destination)
        n (.ncols destination)
        d (* m n)]
    (loop [i 0 src source]
      (if (and src (< i d))
        (do
          (if (column-major? destination)
            (.set destination (rem i m) (quot i m) (first src))
            (.set destination (rem i n) (quot i n) (first src)))
          (recur (inc i) (next src)))
        destination))))

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))
