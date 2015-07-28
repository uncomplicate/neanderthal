(ns uncomplicate.neanderthal.block
  (:require [vertigo
             [bytes :refer [direct-buffer byte-seq
                            slice-buffer cross-section]]
             [structs :refer [float64 float32 wrap-byte-seq]]]
            [uncomplicate.neanderthal.protocols :refer :all])
  (:import [java.nio ByteBuffer]
           [clojure.lang IFn IFn$D IFn$DD IFn$LD IFn$DDD IFn$LDD IFn$DDDD
            IFn$LDDD IFn$DDDDD IFn$DLDD IFn$DLDDD IFn$LDDDD IFn$DO IFn$ODO
            IFn$OLDO IFn$ODDO IFn$OLDDO IFn$ODDDO]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Vector Matrix RealChangeable GeneralBlock]))

;;TODO clean up
(def ^:const DEFAULT_ORDER CBLAS/ORDER_COLUMN_MAJOR)

(def ^:private INCOMPATIBLE_VECTOR_BLOCKS_MSG
  "Operation is not permited on vectors with incompatible buffers.
  x: dim:%d, stride:%d
  y: dim:%d, stride:%d.")

(def ^:private DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^:private PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(defn ^:private hash* [^long h ^double x]
  (clojure.lang.Util/hashCombine h (Double/hashCode x)))

(defn entry-eq [res ^double x ^double y]
  (= x y))

;; ================== map/reduce functions ================================

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
  (get [_ buf i]
    (.getFloat buf (* 4 i)))
  (set [_ buf i val]
    (.putFloat buf (* 4 i) val))
  (elementBytes [_]
    4))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* 4 i)))
  (set [_ buf i val]
    (.putDouble buf (* 4 i) val))
  (elementBytes [_]
    8))

(def double-accessor (->DoubleBufferAccessor))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^ByteBuffer buf ^RealBufferAccessor accessor
                          ^long n ^long strd]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealBlockVector)
                 (hash-combine (.elementBytes accessor)) (hash-combine n))
             hash*) )
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? RealBlockVector y)
     (and (.compatible x y) (freduce x true entry-eq y))
     :default false))
  clojure.lang.Seqable
  (seq [_]
    (wrap-byte-seq (case (.elementBytes accessor)
                     8 float64
                     4 float32)
                   (* (.elementBytes accessor) strd) 0 (byte-seq buf)))
  Group
  (zero [_]
    (RealBlockVector. (direct-buffer (* (.elementBytes accessor) n))
                      accessor n 1))
  GeneralBlock
  (compatible [_ y]
    (and (= (.elementBytes accessor) (.elementBytes y))
         (= n (.dim y))))
  (elementBytes [_]
    (.elementBytes accessor))
  (buffer [_]
    buf)
  (stride strd)
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
        (.put accessor buf (* strd i) val))
      x))
  (set [x i val]
    (do
      (.put accessor buf (* strd i) val)
      x))
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.get accessor buf (* accessor strd i)))
  (subvector [_ k l]
    (RealBlockVector.
     (slice-buffer buf (* (.elementBytes accessor) k strd)
                   (* (.elemenBytes accessor) l strd))
     accessor l strd)))

(extend RealBlockVector
  Functor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Reducible
  {:freduce vector-freduce})

(defmethod print-method RealBlockVector
  [^VectorBlock x ^java.io.Writer w]
  (.write w (format "#<%sBlockVector| n:%d, stride:%d, %s>"
                    (case (.element-bytes x) 4 "Single" 8 "Double")
                    (.dim x) (.stride x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(defn column-major? [^GeneralBlock a]
  (= CBLAS/ORDER_COLUMN_MAJOR (.order a)))

(deftype RealGeneralMatrix [^ByteBuffer buf ^RealBufferAccessor accessor
                            ^long m ^long n ^long ld ^long ord]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealGeneralMatrix) (hash-combine (.elementBytes accessor))
                 (hash-combine m) (hash-combine n))
             hash*))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? RealGeneralMatrix y)
     (and (= m (.mrows ^Matrix y))
          (= n (.ncols ^Matrix y))
          (freduce x true entry-eq y))
     :default false))
  Group
  (zero [_]
    (RealGeneralMatrix. (direct-buffer (* (.elementBytes accessor) m n))
                        accessor m n m ord))
  GeneralBlock
  (compatible [_ b]
    (and (= (.elementBytes accessor) (.elementBytes b))
         (= m (.mrows ^GeneralBlock b)) (= n (.ncols ^GeneralBlock b))))
  (dim [_]
    (cond
      (= 1 m) n
      (= 1 n) m
      :default 0))
  (elementBytes [_]
    (.elementBytes accessor))
  (buffer [_]
    buf)
  (stride
    ld)
  (order [_]
    ord)
  clojure.lang.Seqable
  (seq [a]
    (if (column-major? a)
      (map #(seq (.col a %)) (range 0 n))
      (map #(seq (.row a %)) (range 0 m))))
  clojure.lang.IFn$LLD
  (invokePrim [x i j]
    (if (and (< -1 i m) (< -1 j n))
      (.entry x i j)
      (throw (IndexOutOfBoundsException.
              (format MAT_BOUNDS_MSG i j m n)))))
  clojure.lang.IFn
  (invoke [x i j]
    (if (and (< -1 (long i) m) (< -1 (long j) n))
      (.entry x i j)
      (throw (IndexOutOfBoundsException.
              (format MAT_BOUNDS_MSG i j m n)))))
  RealChangeable
  (set [a val]
    (do (if (= ld (if (column-major? a) m n))
          (dotimes [i (* m n)]
            (.put accessor buf i val))
          (if (column-major? a)
            (dotimes [i n]
              (.set ^RealChangeable (.col a i) val))
            (dotimes [i (.mrows a)]
              (.set ^RealChangeable (.row a i) val))))
        a))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR ord)
        (.put accessor buf (+ (* ld j) i) val)
        (.put accessor buf (+ (* ld i) j) val))
      a))
  (alter [a i j f]
    (.set a i j (.invokePrim ^IFn$DD f (.entry a i j))))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR ord)
      (.get accessor buf (+ (* ld j) i))
      (.get accessor buf (+ (* ld i) j))))
  (row [a i]
    (let [elem-width (.elementBytes accessor)]
      (if (column-major? a)
        (RealBlockVector.
         (slice-buffer buf (* elem-width i) (- (.capacity buf) (* elem-width i)))
         accessor n ld)
        (RealBlockVector.
         (slice-buffer buf (* elem-width ld i) (* elem-width n))
         accessor n 1))))
  (col [a j]
    (let [elem-width (.elementBytes accessor)]
      (if (column-major? a)
        (RealBlockVector.
         (slice-buffer buf (* elem-width ld j) (* elem-width m))
         accessor m 1)
        (RealBlockVector.
         (slice-buffer buf (* elem-width j) (- (.capacity buf) (* elem-width j)))
         accessor m ld))))
  (submatrix [a i j k l]
    (let [elem-width (.elementBytes accessor)]
      (RealGeneralMatrix.
       (if (column-major? a)
         (slice-buffer buf (+ (* elem-width ld j) (* elem-width i))
                       (* elem-width ld l))
         (slice-buffer buf (+ (* elem-width ld i) (* elem-width j))
                       (* elem-width ld k)))
       accessor k l ld ord)))
  (transpose [a]
    (RealGeneralMatrix. buf accessor n m ld
                        (if (column-major? a)
                          CBLAS/ORDER_ROW_MAJOR
                          CBLAS/ORDER_COLUMN_MAJOR))))

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "#<%SGeneralMatrix| %s, mxn: %dx%d, ld:%d, %s>"
                    (case (.elementBytes a) 4 "Single" 8 "Double")
                    (if (= CBLAS/ORDER_COLUMN_MAJOR (.order a)) "COL" "ROW")
                    (.mrows a) (.ncols a) (.ld a) (pr-str (seq a)))))
