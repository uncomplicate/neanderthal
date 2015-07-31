(ns uncomplicate.neanderthal.block
  (:require [vertigo
             [core :refer [wrap marshal-seq]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 wrap-byte-seq unwrap-byte-seq]]]
            [uncomplicate.neanderthal.protocols :refer :all])
  (:import [java.nio ByteBuffer]
           [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS RealBufferAccessor BufferAccessor
            RealVector RealMatrix Vector Matrix RealChangeable Block]))

(def ^:private DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^:private PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(def ^:const ROW_MAJOR 101)

(def ^:const COLUMN_MAJOR 102)

(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(defn column-major? [^Block a]
  (= 102 (.order a)))

(defn ^:private hash* [^long h ^double x]
  (clojure.lang.Util/hashCombine h (Double/hashCode x)))

(defn entry-eq [res ^double x ^double y]
  (= x y))

;; ================== map/reduce functions =====================================

(defn ^:private vector-fmap!
  ([^RealVector x f]
   (do (cond
         (instance? IFn$DD f)
         (dotimes [i (.dim x)]
           (.alter ^RealChangeable x i f))
         (instance? IFn$LDD f)
         (dotimes [i (.dim x)]
           (.set ^RealChangeable x i (.invokePrim ^IFn$LDD f i (.entry x i))))
         (instance? IFn$LD f)
         (dotimes [i (.dim x)]
           (.set ^RealChangeable x i (.invokePrim ^IFn$LD f i)))
         :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
       x))
  ([^RealVector x f ^RealVector y]
   (do (if (<= (.dim x) (.dim y))
         (cond
           (instance? IFn$DDD f)
           (dotimes [i (.dim x)]
             (.set ^RealChangeable x i
                   (.invokePrim ^IFn$DDD f (.entry x i) (.entry y i))))
           (instance? IFn$LDDD f)
           (dotimes [i (.dim x)]
             (.set ^RealChangeable x i
                   (.invokePrim ^IFn$LDDD f i (.entry x i) (.entry y i))))
           :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
       x))
  ([^RealVector x f ^RealVector y ^RealVector z]
   (do (if (<= (.dim x) (min (.dim y) (.dim z)))
         (cond
           (instance? IFn$DDDD f)
           (dotimes [i (.dim x)]
             (.set ^RealChangeable x i
                   (.invokePrim ^IFn$DDDD f (.entry x i) (.entry y i) (.entry z i))))
           (instance? IFn$LDDDD f)
           (dotimes [i (.dim x)]
             (.set ^RealChangeable x i
                   (.invokePrim ^IFn$LDDDD f i (.entry x i) (.entry y i) (.entry z i))))
           :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
       x))
  ([^RealVector x f ^RealVector y ^RealVector z ^RealVector w]
   (do (if (<= (.dim x) (min (.dim y) (.dim z) (.dim w)))
         (dotimes [i (.dim x)]
           (.set ^RealChangeable x i
                 (.invokePrim ^IFn$DDDDD f
                              (.entry x i) (.entry y i) (.entry z i) (.entry w i))))
         (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x)))))
       x))
  ([x f y z w ws]
   (throw (UnsupportedOperationException.
           "Primitive functions support max 4 args."))))

(defn ^:private vector-fold
  ([^RealVector x]
   (loop [i 0 res 0.0]
     (if (< i (.dim x))
       (recur (inc i)
              (+ res (.entry x i)))
       res)))
  ([x f]
   (freduce x f))
  ([x f id]
   (freduce x id f)))

(defn ^:private vector-freduce
  ([^RealVector x f]
   (loop [i 2 res (.invokePrim ^IFn$DDD f (.entry x 0) (.entry x 1))]
     (if (< i (.dim x))
       (recur (inc i) (.invokePrim ^IFn$DDD f res (.entry x i)))
       res)))
  ([^RealVector x acc f]
   (cond
     (instance? IFn$DDD f)
     (loop [i 0 res (double acc)]
       (if (< i (.dim x))
         (recur (inc i) (.invokePrim ^IFn$DDD f res (.entry x i)))
         res))
     (instance? IFn$DLDD f)
     (loop [i 0 res (double acc)]
       (if (< i (.dim x))
         (recur (inc i) (.invokePrim ^IFn$DLDD f res i (.entry x i)))
         res))
     (instance? IFn$ODO f)
     (loop [i 0 res acc]
       (if (and (< i (.dim x)) res)
         (recur (inc i) (.invokePrim ^IFn$ODO f res (.entry x i)))
         res))
     (instance? IFn$OLDO f)
     (loop [i 0 res acc]
       (if (and (< i (.dim x)) res)
         (recur (inc i) (.invokePrim ^IFn$OLDO f res i (.entry x i)))
         res))
     :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG))))
  ([^RealVector x acc f ^RealVector y]
   (if (<= (.dim x) (.dim y))
     (cond
       (instance? IFn$DDDD f)
       (loop [i 0 res (double acc)]
         (if (and (< i (.dim x)) (Double/isFinite res))
           (recur (inc i) (.invokePrim ^IFn$DDDD f res (.entry x i) (.entry y i)))
           res))
       (instance? IFn$DLDDD f)
       (loop [i 0 res (double acc)]
         (if (and (< i (.dim x)) (Double/isFinite res))
           (recur (inc i) (.invokePrim ^IFn$DLDDD f res i
                                       (.entry x i) (.entry y i)))
           res))
       (instance? IFn$ODDO f)
       (loop [i 0 res acc]
         (if (and (< i (.dim x)) res)
           (recur (inc i) (.invokePrim ^IFn$ODDO f res (.entry x i) (.entry y i)))
           res))
       (instance? IFn$OLDDO f)
       (loop [i 0 res acc]
         (if (and (< i (.dim x)) res)
           (recur (inc i) (.invokePrim ^IFn$OLDDO f res i
                                       (.entry x i) (.entry y i)))
           res))
       :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([^RealVector x acc f ^RealVector y ^RealVector z]
   (if (<= (.dim x) (min (.dim y) (.dim z)))
     (if (number? acc)
       (loop [i 0 res (double acc)]
         (if (and (< i (.dim x)) (Double/isFinite res))
           (recur (inc i)
                  (.invokePrim ^IFn$DDDDD f res
                               (.entry x i) (.entry y i) (.entry z i)))
           res))
       (loop [i 0 res acc]
         (if (and (< i (.dim x)) res)
           (recur (inc i)
                  (.invokePrim ^IFn$ODDDO f res
                               (.entry x i) (.entry y i) (.entry z i)))
           res)))
     (throw (IllegalArgumentException. (format DIMENSIONS_MSG (.dim x))))))
  ([x acc f y z ws]
   (throw (UnsupportedOperationException.
           "Primitive functions support max 4 args."))))

(defn ^:private matrix-fmap!
  ([^Matrix x f]
   (do (if (column-major? x)
         (dotimes [i (.ncols x)]
           (fmap! (.col x i) f))
         (dotimes [i (.mrows x)]
           (fmap! (.row x i) f)))
       x))
  ([^Matrix x f ^Matrix y]
   (do (if (column-major? x)
         (dotimes [i (.ncols x)]
           (fmap! (.col x i) f (.col y i)))
         (dotimes [i (.mrows x)]
           (fmap! (.row x i) f (.row y i))))
       x))
  ([^Matrix x f ^Matrix y ^Matrix z]
   (do (if (column-major? x)
         (dotimes [i (.ncols x)]
           (fmap! (.col x i) f (.col y i) (.col z i)))
         (dotimes [i (.mrows x)]
           (fmap! (.row x i) f (.row y i) (.row z i))))
       x))
  ([^Matrix x f ^Matrix y ^Matrix z ^Matrix w]
   (do (if (column-major? x)
         (dotimes [i (.ncols x)]
           (fmap! (.col x i) f (.col y i) (.col z i) (.col w i)))
         (dotimes [i (.mrows x)]
           (fmap! (.row x i) f (.row y i) (.row z i) (.col w i))))
       x))
  ([x f y z w ws]
   (throw (UnsupportedOperationException.
           "Primitive functions support max 4 args."))))

(defn ^:private matrix-fold
 ([^RealMatrix x]
  (loop [j 0 res 0.0]
    (if (< j (.ncols x))
      (recur (inc j)
             (double
              (loop [i 0 res res]
                (if (< i (.mrows x))
                  (recur (inc i)
                         (+ res (.entry x i j)))
                  res))))
      res)))
 ([x f]
  (fold f 0.0))
 ([x f id]
  (freduce x id f)))

(defn ^:private matrix-freduce
  ([x f]
   (fold x f))
  ([^Matrix x acc f]
   (loop [i 0 res acc]
     (if (< i (.ncols x))
       (recur (inc i) (freduce (.col x i) res f))
       res)))
  ([^Matrix x acc f ^Matrix y]
   (loop [i 0 res acc]
     (if (< i (.ncols x))
       (recur (inc i) (freduce (.col x i) res f (.col y i)))
       res)))
  ([^Matrix x acc f ^Matrix y ^Matrix z]
   (loop [i 0 res acc]
     (if (< i (.ncols x))
       (recur (inc i) (freduce (.col x i) res f (.col y i) (.col z i)))
       res)))
  ([x acc f y z ws]
   (throw (UnsupportedOperationException.
           "Primitive functions support max 4 args."))))

;; ============ Real Buffer ====================================================

(deftype FloatBufferAccessor []
  RealBufferAccessor
  (entryType [_]
    Float/TYPE)
  (toBuffer [_ s]
    (.buf ^ByteSeq (unwrap-byte-seq (marshal-seq float32 s))))
  (toSeq [_ buf stride]
    (wrap-byte-seq float32 (* 4 stride) 0 (byte-seq buf)))
  (directBuffer [_ n]
    (direct-buffer (* 4 n)))
  (slice [_ buf k l]
    (slice-buffer buf (* 4 k) (* 4 l)))
  (count [_ b]
    (quot (.capacity b) 4))
  (get [_ buf i]
    (.getFloat buf (* 4 i)))
  (set [_ buf i val]
    (.putFloat buf (* 4 i) val)))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  RealBufferAccessor
  (entryType [_]
    Double/TYPE)
  (toBuffer [_ s]
    (.buf ^ByteSeq (unwrap-byte-seq (marshal-seq float64 s))))
  (toSeq [_ buf stride]
    (wrap-byte-seq float64 (* 8 stride) 0 (byte-seq buf)))
  (directBuffer [_ n]
    (direct-buffer (* 8 n)))
  (slice [_ buf k l]
    (slice-buffer buf (* 8 k) (* 8 l)))
  (count [_ b]
    (quot (.capacity b) 8))
  (get [_ buf i]
    (.getDouble buf (* 8 i)))
  (set [_ buf i val]
    (.putDouble buf (* 8 i) val)))

(def double-accessor (->DoubleBufferAccessor))

(declare ->RealGeneralMatrix)

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^RealBufferAccessor accessor
                          ^BLAS vector-engine ^BLAS matrix-engine
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealBlockVector) (hash-combine n))
             hash*) )
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (and (compatible x y) (= n (.dim ^Vector y)))
     (freduce x true entry-eq y)
     :default false))
  (toString [_]
    (format "#<RealBlockVector| %s, n:%d, stride:%d>"
            (.entryType accessor) n strd))
  clojure.lang.Seqable
  (seq [_]
    (.toSeq accessor buf strd))
  Group
  (zero [_]
    (RealBlockVector. accessor vector-engine matrix-engine
                      (.directBuffer accessor n) n 1))
  EngineProvider
  (engine [_]
    vector-engine)
  Memory
  (compatible [_ y]
    (and (instance? RealBlockVector y)
         (= accessor (.accessor ^RealBlockVector y))))
  BlockCreator
  (create-block [_ m n]
    (->RealGeneralMatrix accessor vector-engine matrix-engine
                         (.directBuffer accessor (* (long m) (long n)))
                         m n m DEFAULT_ORDER))
  Block
  (buffer [_]
    buf)
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
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.get accessor buf (* strd i)))
  (subvector [_ k l]
    (RealBlockVector. accessor vector-engine matrix-engine
                      (.slice accessor buf (* k strd) (* l strd)) l strd)))

(extend RealBlockVector
  Functor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Reducible
  {:freduce vector-freduce})

(defmethod print-method RealBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s<>" (str x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(deftype RealGeneralMatrix [^RealBufferAccessor accessor
                            ^BLAS vector-engine ^BLAS matrix-engine
                            ^ByteBuffer buf ^long m ^long n ^long ld ^long ord]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealGeneralMatrix)
                 (hash-combine m) (hash-combine n))
             hash*))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (and (compatible x y) (= m (.mrows ^Matrix y)) (= n (.ncols ^Matrix y)))
     (freduce x true entry-eq y)
     :default false))
  (toString [_]
    (format "#<GeneralMatrix| %s, %s, mxn: %dx%d, ld:%d>"
            (.entryType accessor) (if (= COLUMN_MAJOR ord) "COL" "ROW")
            m n ld))
  Group
  (zero [_]
    (RealGeneralMatrix. accessor vector-engine matrix-engine
                        (.directBuffer accessor (* m n)) m n m ord))
  EngineProvider
  (engine [_]
    matrix-engine)
  Memory
  (compatible [_ b]
    (or (and (instance? RealGeneralMatrix b)
             (= accessor (.accessor ^RealGeneralMatrix b)))
        (and (instance? RealBlockVector b)
             (= (.entryType accessor)
                (.entryType ^BufferAccessor (.accessor ^RealBlockVector b))))))
  BlockCreator
  (create-block [_ m1 n1]
    (->RealGeneralMatrix accessor vector-engine matrix-engine
                         (.directBuffer accessor (* (long m1) (long n1)))
                         m1 n1 (max (long m1) 1) DEFAULT_ORDER))
  Block
  (buffer [_]
    buf)
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
  (invokePrim [x i j]
    (if (and (< -1 i m) (< -1 j n))
      (.entry x i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j m n)))))
  clojure.lang.IFn
  (invoke [x i j]
    (if (and (< -1 (long i) m) (< -1 (long j) n))
      (.entry x i j)
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
  (row [a i]
    (if (column-major? a)
      (RealBlockVector. accessor vector-engine matrix-engine
                        (.slice accessor buf i (inc (* (dec n) ld))) n ld)
      (RealBlockVector. accessor vector-engine matrix-engine
                        (.slice accessor buf (* ld i) n) n 1)))
  (col [a j]
    (if (column-major? a)
      (RealBlockVector. accessor vector-engine matrix-engine
                        (.slice accessor buf (* ld j) m) m 1)
      (RealBlockVector. accessor vector-engine matrix-engine
                        (.slice accessor buf j (inc (* (dec m) ld))) m ld)))
  (submatrix [a i j k l]
    (RealGeneralMatrix. accessor vector-engine matrix-engine
                        (if (column-major? a)
                          (.slice accessor buf (+ (* ld j) i) (* ld l))
                          (.slice accessor buf (+ (* ld i) j) (* ld k)))
                        k l ld ord))
  (transpose [a]
    (RealGeneralMatrix. accessor vector-engine matrix-engine buf n m ld
                        (if (column-major? a) ROW_MAJOR COLUMN_MAJOR))))

(extend RealGeneralMatrix
  Functor
  {:fmap! matrix-fmap!}
  Foldable
  {:fold matrix-fold}
  Reducible
  {:freduce matrix-freduce})

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "%s%s<>" (str a) (pr-str (seq a)))))

;; ========================== Creators =========================================

(defn create-vector
  ([^BufferAccessor accessor vector-engine matrix-engine source]
   (cond
     (and (instance? ByteBuffer source))
     (->RealBlockVector accessor vector-engine matrix-engine
                        source (.count accessor source) 1)
     (and (integer? source) (<= 0 (long source)))
     (create-vector accessor vector-engine matrix-engine
                  (.directBuffer accessor source))
     (float? source) (create-vector accessor vector-engine matrix-engine [source])
     (sequential? source) (create-vector accessor vector-engine matrix-engine
                                         (.toBuffer accessor source))
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a vector from %s."
                              (type source)))))))

(defn create-matrix
  ([^BufferAccessor accessor vector-engine matrix-engine m n source]
   (cond
     (and (instance? ByteBuffer source)
          (= (* (long m) (long n)) (.count accessor source)))
     (->RealGeneralMatrix accessor vector-engine matrix-engine
                          source m n (max (long m) 1) DEFAULT_ORDER)
     (sequential? source) (create-matrix accessor vector-engine matrix-engine
                                       m n (.toBuffer accessor source))
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a %dx%d matrix from %s."
                              m n (type source))))))
  ([^BufferAccessor accessor vector-engine matrix-engine m n]
   (create-matrix accessor vector-engine matrix-engine m n
                  (.directBuffer accessor (* (long m) (long n))))))
