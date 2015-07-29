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
           [uncomplicate.neanderthal CBLAS]
           [uncomplicate.neanderthal.protocols RealBufferAccessor
            RealVector RealMatrix Vector Matrix RealChangeable Block]))

;;TODO clean up

(def ^:private DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^:private PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(defn ^:private hash* [^long h ^double x]
  (clojure.lang.Util/hashCombine h (Double/hashCode x)))

(def ^:const DEFAULT_ORDER CBLAS/ORDER_COLUMN_MAJOR)

(defn column-major? [^Block a]
  (= CBLAS/ORDER_COLUMN_MAJOR (.order a)))

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
    (.putFloat buf (* 4 i) val)))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* 8 i)))
  (set [_ buf i val]
    (.putDouble buf (* 8 i) val)))

(def double-accessor (->DoubleBufferAccessor))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^ByteBuffer buf ^RealBufferAccessor accessor
                          elem-type ^long elem-width ^long n ^long strd]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealBlockVector)
                 (hash-combine elem-width) (hash-combine n))
             hash*) )
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (and (.compatible x y) (= n (.dim ^Vector y)))
     (freduce x true entry-eq y)
     :default false))
  (toString [_]
    (format "#<RealBlockVector| %s, n:%d, stride:%d>"
            elem-type n strd))
  clojure.lang.Seqable
  (seq [_]
    (wrap-byte-seq elem-type (* elem-width strd) 0 (byte-seq buf)))
  Group
  (zero [_]
    (RealBlockVector. (direct-buffer (* elem-width n)) accessor
                      elem-type elem-width n 1))
  Block
  (buffer [_]
    buf)
  (compatible [_ y]
    (and (instance? RealBlockVector y)
         (= elem-type (.elementType ^Block y))))
  (stride [_]
    strd)
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
    (.get accessor buf (* elem-width strd i)))
  (subvector [_ k l]
    (RealBlockVector.
     (slice-buffer buf (* elem-width k strd) (* elem-width l strd))
     accessor elem-type elem-width l strd)))

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

(deftype RealGeneralMatrix [^ByteBuffer buf ^RealBufferAccessor accessor
                            elem-type ^long elem-width
                            ^long m ^long n ^long ld ^long ord]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :RealGeneralMatrix) (hash-combine elem-width)
                 (hash-combine m) (hash-combine n))
             hash*))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (and (.compatible x y) (= m (.mrows ^Matrix y)) (= n (.ncols ^Matrix y)))
     (freduce x true entry-eq y)
     :default false))
  (toString [_]
    (format "#<GeneralMatrix| %s, %s, mxn: %dx%d, ld:%d>"
            elem-type (if (= CBLAS/ORDER_COLUMN_MAJOR ord) "COL" "ROW")
            m n ld))
  Group
  (zero [_]
    (RealGeneralMatrix. (direct-buffer (* elem-width m n))
                        accessor elem-type elem-width
                        m n m ord))
  Block
  (compatible [_ b]
    (and (or (instance? RealGeneralMatrix b) (instance? RealBlockVector b))
         (= elem-type (.elementType ^Block b))))
  (buffer [_]
    buf)
  (stride [_]
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
            (.set accessor buf i val))
          (if (column-major? a)
            (dotimes [i n]
              (.set ^RealChangeable (.col a i) val))
            (dotimes [i (.mrows a)]
              (.set ^RealChangeable (.row a i) val))))
        a))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR ord)
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
    (if (= CBLAS/ORDER_COLUMN_MAJOR ord)
      (.get accessor buf (+ (* ld j) i))
      (.get accessor buf (+ (* ld i) j))))
  (row [a i]
    (if (column-major? a)
      (RealBlockVector.
       (slice-buffer buf (* elem-width i) (- (.capacity buf) (* elem-width i)))
       accessor elem-type elem-width n ld)
      (RealBlockVector.
       (slice-buffer buf (* elem-width ld i) (* elem-width n))
       accessor elem-type elem-width n 1)))
  (col [a j]
    (if (column-major? a)
      (RealBlockVector.
       (slice-buffer buf (* elem-width ld j) (* elem-width m))
       accessor elem-type elem-width m 1)
      (RealBlockVector.
       (slice-buffer buf (* elem-width j) (- (.capacity buf) (* elem-width j)))
       accessor elem-type elem-width m ld)))
  (submatrix [a i j k l]
    (RealGeneralMatrix.
     (if (column-major? a)
       (slice-buffer buf (+ (* elem-width ld j) (* elem-width i))
                     (* elem-width ld l))
       (slice-buffer buf (+ (* elem-width ld i) (* elem-width j))
                     (* elem-width ld k)))
     accessor elem-type elem-width k l ld ord))
  (transpose [a]
    (RealGeneralMatrix. buf accessor elem-type elem-width n m ld
                        (if (column-major? a)
                          CBLAS/ORDER_ROW_MAJOR
                          CBLAS/ORDER_COLUMN_MAJOR))))

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "%s%s<>" (str a) (pr-str (seq a)))))
