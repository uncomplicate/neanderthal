(ns uncomplicate.neanderthal.cblas
  (:require [vertigo
             [bytes :refer [direct-buffer byte-seq
                            slice-buffer cross-section]]
             [structs :refer [float64 float32 wrap-byte-seq]]]
            [uncomplicate.neanderthal.protocols :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [clojure.lang IFn IFn$D IFn$DD IFn$LD
            IFn$DDD IFn$LDD IFn$DDDD IFn$LDDD IFn$DDDDD
            IFn$DLDD IFn$DLDDD
            IFn$LDDDD IFn$DO IFn$ODO IFn$OLDO IFn$ODDO
            IFn$OLDDO IFn$ODDDO]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Vector Matrix
            RealChangeable]))

(def ^:const DEFAULT_ORDER CBLAS/ORDER_COLUMN_MAJOR)

(def ^:private DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def ^:private STRIDE_MSG
  "I cannot use vectors with stride other than %d: stride: %d.")

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^:private DIFF_DIM_MSG
  "Different dimensions - required:%d, is:%d.")

(def ^:private PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(defn ^:private hash-comb [h ^double x]
  (hash-combine h x))

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

;; ============ Real Vectors ===========================================
;; -------------- Double Vector -----------------------------

(deftype DoubleBlockVector [^ByteBuffer buf ^long n ^long stride]
  Object
  (hashCode [this]
    (loop [i 0 res (hash-combine (hash :DoubleBlockVector) n)]
      (if (< i n)
        (recur (inc i) (hash-combine res (.entry this i)))
        res)))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? DoubleBlockVector y)
     (and (= n (.dim ^Vector y))
          (freduce x true entry-eq y))
     :default false))
  clojure.lang.Seqable
  (seq [_]
    (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf)))
  Carrier
  (zero [_]
    (DoubleBlockVector. (direct-buffer (* Double/BYTES n)) n 1))
  (byte-size [_]
    Double/BYTES)
  (swp [x y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/dswap n buf stride
                       (.buf ^DoubleBlockVector y)
                       (.stride ^DoubleBlockVector y))
          x)
      (throw (IllegalArgumentException.
              (format DIFF_DIM_MSG \x n \y (.dim ^Vector y))))))
  (copy [_ y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/dcopy n buf stride
                       (.buf ^DoubleBlockVector y)
                       (.stride ^DoubleBlockVector y))
          y)
      (throw (IllegalArgumentException.
              (format DIFF_DIM_MSG \x n \y (.dim ^Vector y))))))
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
        (.putDouble buf (* Double/BYTES stride i) val))
      x))
  (set [x i val]
    (do
      (.putDouble buf (* Double/BYTES stride i) val)
      x))
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.getDouble buf (* Double/BYTES stride i)))
  (subvector [_ k l]
    (DoubleBlockVector.
     (slice-buffer buf (* Double/BYTES k stride) (* Double/BYTES l stride))
     l stride))
  (dot [_ y]
    (CBLAS/ddot n buf stride
                (.buf ^DoubleBlockVector y)
                (.stride ^DoubleBlockVector y)))
  (nrm2 [_]
    (CBLAS/dnrm2 n buf stride))
  (asum [_]
    (CBLAS/dasum n buf stride))
  (iamax [_]
    (CBLAS/idamax n buf stride))
  (rot [x y c s]
    (do (CBLAS/drot n buf stride
                    (.buf ^DoubleBlockVector y)
                    (.stride ^DoubleBlockVector y)
                    c s)
        x))
  (rotg [x]
    (if (= 1 stride)
      (do (CBLAS/drotg buf)
          x)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 stride)))))
  (rotm [x y p]
    (if (= 1 (.stride ^DoubleBlockVector p))
      (do (CBLAS/drotm n buf stride
                       (.buf ^DoubleBlockVector y)
                       (.stride ^DoubleBlockVector y)
                       (.buf ^DoubleBlockVector p))
          x)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (.stride ^DoubleBlockVector p))))))
  (rotmg [p args]
    (if (= 1 stride (.stride ^DoubleBlockVector p))
      (do (CBLAS/drotmg (.buf ^DoubleBlockVector args) buf)
          p)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (.stride ^DoubleBlockVector p))))))
  (scal [x alpha]
    (do (CBLAS/dscal n alpha buf stride)
        x))
  (axpy [_ alpha y]
    (do (CBLAS/daxpy n alpha buf stride
                     (.buf ^DoubleBlockVector y)
                     (.stride ^DoubleBlockVector y))
        y)))

(extend DoubleBlockVector
  Functor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Reducible
  {:freduce vector-freduce})

(defmethod print-method DoubleBlockVector
  [^DoubleBlockVector dv ^java.io.Writer w]
  (.write w (format "#<DoubleBlockVector| n:%d, stride:%d, %s>"
                    (.dim dv) (.stride dv) (pr-str (take 100 (seq dv))))))

;;-------------- Float Vector -----------------------------

(deftype FloatBlockVector [^ByteBuffer buf ^long n ^long stride]
  Object
  (hashCode [this]
    (loop [i 0 res (hash-combine (hash :FloatBlockVector) n)]
      (if (< i n)
        (recur (inc i) (hash-combine res (.entry this i)))
        res)))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (instance? FloatBlockVector y)
      (and (= n (.dim ^Vector y))
           (freduce x true entry-eq y))
      :default false))
  clojure.lang.Seqable
  (seq [_]
    (wrap-byte-seq float32 (* Float/BYTES stride) 0 (byte-seq buf)))
  Carrier
  (zero [_]
    (FloatBlockVector. (direct-buffer (* Float/BYTES n)) n 1))
  (byte-size [_]
    Float/BYTES)
  (swp [x y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/sswap n buf stride
                       (.buf ^FloatBlockVector y)
                       (.stride ^FloatBlockVector y))
          x)
      (throw (IllegalArgumentException.
              (format DIFF_DIM_MSG \x n \y (.dim ^Vector y))))))
  (copy [_ y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/scopy n buf stride
                       (.buf ^FloatBlockVector y)
                       (.stride ^FloatBlockVector y))
          y)
      (throw (IllegalArgumentException.
              (format DIFF_DIM_MSG \x n \y (.dim ^Vector y))))))

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
        (.putFloat buf (* Float/BYTES stride i) val))
      x))
  (set [x i val]
    (do
      (.putFloat buf (* Float/BYTES stride i) val)
      x))
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.getFloat buf (* Float/BYTES stride i)))
  (subvector [_ k l]
    (FloatBlockVector.
     (slice-buffer buf (* Float/BYTES k stride) (* Float/BYTES l stride))
     l stride))
  (dot [_ y]
    (CBLAS/sdot n buf stride
                (.buf ^FloatBlockVector y)
                (.stride ^FloatBlockVector y)))
  (nrm2 [_]
    (CBLAS/snrm2 n buf stride))
  (asum [_]
    (CBLAS/sasum n buf stride))
  (iamax [_]
    (CBLAS/isamax n buf stride))
  (rot [x y c s]
    (do (CBLAS/srot n buf stride
                    (.buf ^FloatBlockVector y)
                    (.stride ^FloatBlockVector y)
                    c s)
        x))
  (rotg [x]
    (if (= 1 stride)
      (do (CBLAS/srotg buf)
          x)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 stride)))))
  (rotm [x y p]
    (if (= 1 (.stride ^FloatBlockVector p))
      (do (CBLAS/srotm n buf stride
                       (.buf ^FloatBlockVector y)
                       (.stride ^FloatBlockVector y)
                       (.buf ^FloatBlockVector p))
          x)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (.stride ^FloatBlockVector p))))))
  (rotmg [p args]
    (if (= 1 stride (.stride ^FloatBlockVector p))
      (do (CBLAS/srotmg (.buf ^FloatBlockVector args) buf)
          p)
      (throw (IllegalArgumentException.
              (format STRIDE_MSG 1 (.stride ^FloatBlockVector p))))))
  (scal [x alpha]
    (do (CBLAS/sscal n alpha buf stride)
        x))
  (axpy [_ alpha y]
    (do (CBLAS/saxpy n alpha buf stride
                     (.buf ^FloatBlockVector y)
                     (.stride ^FloatBlockVector y))
        y)))

(extend FloatBlockVector
  Functor
  {:fmap! vector-fmap!}
  Foldable
  {:fold vector-fold}
  Reducible
  {:freduce vector-freduce})

(defmethod print-method FloatBlockVector
  [^FloatBlockVector dv ^java.io.Writer w]
  (.write w (format "#<FloatBlockVector| n:%d, stride:%d, %s>"
                    (.dim dv) (.stride dv) (pr-str (take 100 (seq dv))))))

;; ================= GE General Matrix =====================

;;-------------- Double General Matrix  -----------------------------

(deftype DoubleGeneralMatrix [^ByteBuffer buf ^long m
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :DoubleGeneralMatrix)
                 (hash-combine m) (hash-combine n))
             hash-comb))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? DoubleGeneralMatrix y)
     (and (= m (.mrows ^Matrix y))
          (= n (.ncols ^Matrix y))
          (freduce x true entry-eq y))
     :default false))
  Carrier
  (zero [_]
    (DoubleGeneralMatrix. (direct-buffer (* Double/BYTES m n)) m n m order))
  (byte-size [_]
    Double/BYTES)
  (copy [a b]
    (let [a ^DoubleGeneralMatrix a
          b ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (do
          (if (and (= order (.order b))
                   (= (if (column-major? a) m n) ld (.ld b)))
            (CBLAS/dcopy (* m n) buf 1 (.buf b) 1)
            (if (column-major? a)
              (dotimes [i n]
                (copy (.col a i) (.col b i)))
              (dotimes [i m]
                (copy (.row a i) (.row b i)))))
          b)
        (throw (IllegalArgumentException.
                "I can not copy incompatible matrices.")))))
  (swp [a b]
    (let [a ^DoubleGeneralMatrix a
          b ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (do
          (if (and (= order (.order b))
                   (= (if (column-major? a) m n) ld (.ld b)))
            (CBLAS/dswap (* m n) buf 1 (.buf b) 1)
            (if (column-major? a)
              (dotimes [i n]
                (swp (.col a i) (.col b i)))
              (dotimes [i m]
                (swp (.row a i) (.row b i)))))
          a)
        (throw (IllegalArgumentException.
                "I can not copy incompatible matrices.")))))
  (column-major? [a]
    (= CBLAS/ORDER_COLUMN_MAJOR order))
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
            (.putDouble buf (* Double/BYTES i) val))
          (if (column-major? a)
            (dotimes [i n]
              (.set ^RealChangeable (.col a i) val))
            (dotimes [i (.mrows a)]
              (.set ^RealChangeable (.row a i) val))))
        a))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR order)
        (.putDouble buf (+ (* Double/BYTES ld j) (* Double/BYTES i)) val)
        (.putDouble buf (+ (* Double/BYTES ld i) (* Double/BYTES j)) val))
      a))
  (alter [a i j f]
    (do
      (let [ind (if (column-major? a)
                  (+ (* Double/BYTES ld j) (* Double/BYTES i))
                  (+ (* Double/BYTES ld i) (* Double/BYTES j)))]
        (.putDouble buf ind
                    (.invokePrim ^IFn$DD f
                                 (.getDouble buf ind))))
      a))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (.getDouble buf (+ (* Double/BYTES ld j) (* Double/BYTES i)))
      (.getDouble buf (+ (* Double/BYTES ld i) (* Double/BYTES j)))))
  (row [a i]
    (if (column-major? a)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES i) (- (.capacity buf) (* Double/BYTES i)))
       n ld)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES ld i) (* Double/BYTES n))
       n 1)))
  (col [a j]
    (if (column-major? a)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES ld j) (* Double/BYTES m))
       m 1)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES j) (- (.capacity buf) (* Double/BYTES j)))
       m ld)))
  (submatrix [a i j k l]
    (DoubleGeneralMatrix.
     (if (column-major? a)
       (slice-buffer buf (+ (* Double/BYTES ld j) (* Double/BYTES i))
                     (* Double/BYTES ld l))
       (slice-buffer buf (+ (* Double/BYTES ld i) (* Double/BYTES j))
                     (* Double/BYTES ld k)))
     k l ld order))
  (transpose [a]
    (DoubleGeneralMatrix. buf n m ld
                          (if (column-major? a)
                            CBLAS/ORDER_ROW_MAJOR
                            CBLAS/ORDER_COLUMN_MAJOR)))
  (mv [_ alpha x beta y]
    (do (CBLAS/dgemv order
                     CBLAS/TRANSPOSE_NO_TRANS
                     m n
                     alpha
                     buf ld
                     (.buf ^DoubleBlockVector x)
                     (.stride ^DoubleBlockVector x)
                     beta
                     (.buf ^DoubleBlockVector y)
                     (.stride ^DoubleBlockVector y))
        y))
  (rank [a alpha x y]
      (do (CBLAS/dger order
                      m n
                      alpha
                      (.buf ^DoubleBlockVector x)
                      (.stride ^DoubleBlockVector x)
                      (.buf ^DoubleBlockVector y)
                      (.stride ^DoubleBlockVector y)
                      buf ld)
          a))
  (mm [_ alpha b beta c]
    (let [co (.order ^DoubleGeneralMatrix c)]
      (do (CBLAS/dgemm co
                       (if (= co order)
                         CBLAS/TRANSPOSE_NO_TRANS
                         CBLAS/TRANSPOSE_TRANS)
                       (if (= co (.order ^DoubleGeneralMatrix b))
                         CBLAS/TRANSPOSE_NO_TRANS
                         CBLAS/TRANSPOSE_TRANS)
                       m (.ncols b) n
                       alpha
                       buf ld
                       (.buf ^DoubleGeneralMatrix b)
                       (.ld ^DoubleGeneralMatrix b)
                       beta
                       (.buf ^DoubleGeneralMatrix c)
                       (.ld ^DoubleGeneralMatrix c))
          c))))

(extend DoubleGeneralMatrix
  Functor
  {:fmap! matrix-fmap!}
  Foldable
  {:fold matrix-fold}
  Reducible
  {:freduce matrix-freduce})

(defmethod print-method DoubleGeneralMatrix
  [^DoubleGeneralMatrix m ^java.io.Writer w]
  (.write w (format "#<DoubleGeneralMatrix| %s, mxn: %dx%d, ld:%d, %s>"
                    (if (= CBLAS/ORDER_COLUMN_MAJOR (.order m)) "COL" "ROW")
                    (.mrows m) (.ncols m) (.ld m) (pr-str (seq m)))))

;;-------------- Float General Matrix  -----------------------------

(deftype FloatGeneralMatrix [^ByteBuffer buf ^long m
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :FloatGeneralMatrix)
                 (hash-combine m) (hash-combine n))
             hash-comb))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? FloatGeneralMatrix y)
     (and (= m (.mrows ^Matrix y))
          (= n (.ncols ^Matrix y))
          (freduce x true entry-eq y))
     :default false))
  Carrier
  (zero [_]
    (FloatGeneralMatrix. (direct-buffer (* Float/BYTES m n)) m n m order))
  (byte-size [_]
    Float/BYTES)
  (copy [a b]
    (let [a ^FloatGeneralMatrix a
          b ^FloatGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (do
          (if (and (= order (.order b))
                   (= (if (column-major? a) m n) ld (.ld b)))
            (CBLAS/scopy (* m n) buf 1 (.buf b) 1)
            (if (column-major? a)
              (dotimes [i n]
                (copy (.col a i) (.col b i)))
              (dotimes [i m]
                (copy (.row a i) (.row b i)))))
          b)
        (throw (IllegalArgumentException.
                "I can not copy incompatible matrices.")))))
  (swp [a b]
    (let [a ^FloatGeneralMatrix a
          b ^FloatGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (do
          (if (and (= order (.order b))
                   (= (if (column-major? a) m n) ld (.ld b)))
            (CBLAS/sswap (* m n) buf 1 (.buf b) 1)
            (if (column-major? a)
              (dotimes [i n]
                (swp (.col a i) (.col b i)))
              (dotimes [i m]
                (swp (.row a i) (.row b i)))))
          a)
        (throw (IllegalArgumentException.
                "I can not copy incompatible matrices.")))))
  (column-major? [a]
    (= CBLAS/ORDER_COLUMN_MAJOR order))
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
            (.putFloat buf (* Float/BYTES i) val))
          (if (column-major? a)
            (dotimes [i n]
              (.set ^RealChangeable (.col a i) val))
            (dotimes [i (.mrows a)]
              (.set ^RealChangeable (.row a i) val))))
        a))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR order)
        (.putFloat buf (+ (* Float/BYTES ld j) (* Float/BYTES i)) val)
        (.putFloat buf (+ (* Float/BYTES ld i) (* Float/BYTES j)) val))
      a))
  (alter [a i j f]
    (do
      (let [ind (if (column-major? a)
                  (+ (* Float/BYTES ld j) (* Float/BYTES i))
                  (+ (* Float/BYTES ld i) (* Float/BYTES j)))]
        (.putFloat buf ind
                    (.invokePrim ^IFn$DD f
                                 (.getFloat buf ind))))
      a))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (.getFloat buf (+ (* Float/BYTES ld j) (* Float/BYTES i)))
      (.getFloat buf (+ (* Float/BYTES ld i) (* Float/BYTES j)))))
  (row [a i]
    (if (column-major? a)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES i) (- (.capacity buf) (* Float/BYTES i)))
       n ld)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES ld i) (* Float/BYTES n))
       n 1)))
  (col [a j]
    (if (column-major? a)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES ld j) (* Float/BYTES m))
       m 1)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES j) (- (.capacity buf) (* Float/BYTES j)))
       m ld)))
  (submatrix [a i j k l]
    (FloatGeneralMatrix.
     (if (column-major? a)
       (slice-buffer buf (+ (* Float/BYTES ld j) (* Float/BYTES i))
                     (* Float/BYTES ld l))
       (slice-buffer buf (+ (* Float/BYTES ld i) (* Float/BYTES j))
                     (* Float/BYTES ld k)))
     k l ld order))
  (transpose [a]
    (FloatGeneralMatrix. buf n m ld
                         (if (column-major? a)
                           CBLAS/ORDER_ROW_MAJOR
                           CBLAS/ORDER_COLUMN_MAJOR)))
  (mv [_ alpha x beta y]
    (do (CBLAS/sgemv order
                     CBLAS/TRANSPOSE_NO_TRANS
                     m n
                     alpha
                     buf ld
                     (.buf ^FloatBlockVector x)
                     (.stride ^FloatBlockVector x)
                     beta
                     (.buf ^FloatBlockVector y)
                     (.stride ^FloatBlockVector y))
        y))
  (rank [a alpha x y]
      (do (CBLAS/sger order
                      m n
                      alpha
                      (.buf ^FloatBlockVector x)
                      (.stride ^FloatBlockVector x)
                      (.buf ^FloatBlockVector y)
                      (.stride ^FloatBlockVector y)
                      buf ld)
          a))
  (mm [_ alpha b beta c]
    (let [co (.order ^FloatGeneralMatrix c)]
      (do (CBLAS/sgemm co
                       (if (= co order)
                         CBLAS/TRANSPOSE_NO_TRANS
                         CBLAS/TRANSPOSE_TRANS)
                       (if (= co (.order ^FloatGeneralMatrix b))
                         CBLAS/TRANSPOSE_NO_TRANS
                         CBLAS/TRANSPOSE_TRANS)
                       m (.ncols b) n
                       alpha
                       buf ld
                       (.buf ^FloatGeneralMatrix b)
                       (.ld ^FloatGeneralMatrix b)
                       beta
                       (.buf ^FloatGeneralMatrix c)
                       (.ld ^FloatGeneralMatrix c))
          c))))

(extend FloatGeneralMatrix
  Functor
  {:fmap! matrix-fmap!}
  Foldable
  {:fold matrix-fold}
  Reducible
  {:freduce matrix-freduce})

(defmethod print-method FloatGeneralMatrix
  [^FloatGeneralMatrix m ^java.io.Writer w]
  (.write w (format "#<FloatGeneralMatrix| %s, mxn: %dx%d, ld:%d, %s>"
                    (if (= CBLAS/ORDER_COLUMN_MAJOR (.order m)) "COL" "ROW")
                    (.mrows m) (.ncols m) (.ld m) (pr-str (seq m)))))
