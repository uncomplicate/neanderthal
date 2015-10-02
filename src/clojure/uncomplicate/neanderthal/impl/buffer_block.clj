(ns uncomplicate.neanderthal.impl.buffer-block
  (:refer-clojure :exclude [accessor])
  (:require [vertigo
             [core :refer [wrap marshal-seq]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 wrap-byte-seq unwrap-byte-seq]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [transfer!]]]
            [uncomplicate.clojurecl.core :refer [Releaseable]])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS RealBufferAccessor BufferAccessor
            RealVector RealMatrix Vector Matrix RealChangeable Block]))

(def ^:const ROW_MAJOR 101)

(def ^:const COLUMN_MAJOR 102)

(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def INCOMPATIBLE_BLOCKS_MSG
  "Operation is not permited on vectors with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s")

(def ^:private DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def ^:private PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(defn column-major? [^Block a]
  (= COLUMN_MAJOR (.order a)))

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

;; ============ Realeaseable ===================================================
(defn clean-buffer [^ByteBuffer buffer]
  (do
    (if (.isDirect buffer)
      (.clean (.cleaner ^DirectByteBuffer buffer)))
    true))

;; ============ Real Buffer ====================================================

(deftype FloatBufferAccessor []
  RealBufferAccessor
  (entryType [_]
    Float/TYPE)
  (entryWidth [_]
    Float/BYTES)
  (toBuffer [_ s]
    (.buf ^ByteSeq (unwrap-byte-seq (marshal-seq float32 s))))
  (toSeq [_ buf stride]
    (wrap-byte-seq float32 (* Float/BYTES stride) 0 (byte-seq buf)))
  (directBuffer [_ n]
    (direct-buffer (* Float/BYTES n)))
  (slice [_ buf k l]
    (slice-buffer buf (* Float/BYTES k) (* Float/BYTES l)))
  (count [_ b]
    (quot (.capacity b) Float/BYTES))
  (get [_ buf i]
    (.getFloat buf (* Float/BYTES i)))
  (set [_ buf i val]
    (.putFloat buf (* Float/BYTES i) val)))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  RealBufferAccessor
  (entryType [_]
    Double/TYPE)
  (entryWidth [_]
    Float/BYTES)
  (toBuffer [_ s]
    (.buf ^ByteSeq (unwrap-byte-seq (marshal-seq float64 s))))
  (toSeq [_ buf stride]
    (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf)))
  (directBuffer [_ n]
    (direct-buffer (* Double/BYTES n)))
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  (count [_ b]
    (quot (.capacity b) Double/BYTES))
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

(declare create-vector)
(declare create-ge-matrix)

;; ============ Real Vector ====================================================

(deftype RealBlockVector [engine-factory ^RealBufferAccessor accessor
                          ^BLAS eng ^Class entry-type ^Boolean master
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealBlockVector) (hash-combine n))
             hash*))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (compatible x y) (= n (.dim ^Vector y)))
      (freduce x true entry-eq y)
      :default false))
  (toString [_]
    (format "#<RealBlockVector| %s, n:%d, stride:%d>" entry-type n strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  clojure.lang.Seqable
  (seq [_]
    (.toSeq accessor buf strd))
  Group
  (zero [_]
    (create-vector engine-factory (.directBuffer accessor n)))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ y]
    (and (instance? RealBlockVector y)
         (= entry-type (.entryType ^Block y))))
  BlockCreator
  (create-block [_ m n]
    (create-ge-matrix engine-factory m n))
  (create-block [_ n]
    (create-vector engine-factory n))
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
      (RealBlockVector. engine-factory accessor (vector-engine engine-factory b l 0 strd)
                        entry-type false b l strd))))

(extend RealBlockVector
  Functor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Reducible
  {:freduce vector-freduce})

(defmethod transfer! [RealBlockVector RealBlockVector]
  [^RealBlockVector source ^RealBlockVector destination]
  (do
    (.copy (engine source) source destination)
    destination))

(defmethod print-method RealBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s<>" (str x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(deftype RealGeneralMatrix [engine-factory ^RealBufferAccessor accessor
                            ^BLAS eng ^Class entry-type ^Boolean master
                            ^ByteBuffer buf ^long m ^long n ^long ld ^long ord]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealGeneralMatrix) (hash-combine m) (hash-combine n))
             hash*))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= m (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (freduce a true entry-eq b)
      :default false))
  (toString [_]
    (format "#<GeneralMatrix| %s, %s, mxn: %dx%d, ld:%d>"
            entry-type (if (= COLUMN_MAJOR ord) "COL" "ROW")
            m n ld))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Group
  (zero [_]
    (create-ge-matrix engine-factory m n (.directBuffer accessor (* m n)) ord))
  EngineProvider
  (engine [_]
    eng)
  Memory
  (compatible [_ b]
    (and (or (instance? RealGeneralMatrix b) (instance? RealBlockVector b))
         (= entry-type (.entryType ^Block b))))
  BlockCreator
  (create-block [_ m1 n1]
    (create-ge-matrix engine-factory m1 n1))
  (create-block [_ n1]
    (create-vector engine-factory n1))
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
        (RealBlockVector. engine-factory accessor
                          (vector-engine engine-factory b n 0 ld)
                          entry-type false b n ld))
      (let [b (.slice accessor buf (* ld i) n)]
        (RealBlockVector. engine-factory accessor
                          (vector-engine engine-factory b n 0 1)
                          entry-type false b n 1))))
  (col [a j]
    (if (column-major? a)
      (let [b (.slice accessor buf (* ld j) m)]
        (RealBlockVector. engine-factory accessor
                          (vector-engine engine-factory b m 0 1)
                          entry-type false b m 1))
      (let [b (.slice accessor buf j (inc (* (dec m) ld)))]
        (RealBlockVector. engine-factory accessor
                          (vector-engine engine-factory b m 0 ld)
                          entry-type false b m ld))))
  (submatrix [a i j k l]
    (let [b (if (column-major? a)
              (.slice accessor buf (+ (* ld j) i) (* ld l))
              (.slice accessor buf (+ (* ld i) j) (* ld k)))]
      (RealGeneralMatrix. engine-factory accessor
                          (matrix-engine engine-factory b k l 0 ld)
                          entry-type false b k l ld ord)))
  (transpose [a]
    (RealGeneralMatrix. engine-factory accessor
                        (matrix-engine engine-factory buf n m 0 ld)
                        entry-type false buf n m ld
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
  [engine-factory source]
  (let [acc ^RealBufferAccessor (data-accessor engine-factory)]
    (cond
      (and (instance? ByteBuffer source))
      (->RealBlockVector engine-factory acc
                         (vector-engine engine-factory source (.count acc source) 0 1)
                         (.entryType acc) true source (.count acc source) 1)
      (and (integer? source) (<= 0 (long source)))
      (create-vector engine-factory (.directBuffer acc source))
      (float? source) (create-vector engine-factory [source])
      (sequential? source) (create-vector engine-factory (.toBuffer acc source))
      :default (throw (IllegalArgumentException.
                       (format "I do not know how to create a vector from %s."
                               (type source)))))))

(defn create-ge-matrix
  ([engine-factory m n source order]
   (let [acc ^RealBufferAccessor (data-accessor engine-factory)
         ld (max (long (if (= COLUMN_MAJOR order) m n)) 1)]
     (cond
       (and (instance? ByteBuffer source)
            (= (* (long m) (long n)) (.count acc source)))
       (->RealGeneralMatrix engine-factory acc
                            (matrix-engine engine-factory source m n 0 ld)
                            (.entryType acc) true source m n ld order)
       (sequential? source) (create-ge-matrix engine-factory m n
                                              (.toBuffer acc source))
       :default
       (throw (IllegalArgumentException.
               (format "I do not know how to create a %dx%d matrix from %s."
                                m n (type source)))))))
  ([engine-factory m n source]
   (create-ge-matrix engine-factory m n source DEFAULT_ORDER))
  ([engine-factory m n]
   (create-ge-matrix engine-factory m n
                     (.directBuffer ^BufferAccessor (data-accessor engine-factory)
                                    (* (long m) (long n))))))
