(ns uncomplicate.neanderthal.cblas
  (:require [primitive-math]
            [vertigo
             [bytes :refer [direct-buffer byte-seq
                            slice-buffer cross-section]]
             [structs :refer [float64 wrap-byte-seq]]]
            [uncomplicate.neanderthal.protocols :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [clojure.lang IFn
            IFn$LD IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD
            IFn$DO IFn$ODO IFn$ODDO IFn$ODDDO]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Vector Matrix
            RealVectorEditor RealMatrixEditor]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

(def ^:const DEFAULT_ORDER CBLAS/ORDER_COLUMN_MAJOR)

(def ^:private DIFF_DIM_MSG
  "Vector dimensions should be %d.")

(def ^:private STRIDE_MSG
  "I cannot use vectors with stride other than %d: stride: %d.")

(def  MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^:private DIMENSION_MSG
  "Different dimensions - required:%d, is:%d.")

(defn ^:private hash-comb [h ^double x]
  (hash-combine h x))

(defn entry-eq [res ^double x ^double y]
  (= x y))
;;-------------- Double Vector -----------------------------

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
    (wrap-byte-seq float64 (* 8 stride) 0 (byte-seq buf)))
  Carrier
  (zero [_]
    (DoubleBlockVector. (direct-buffer (* 8 n)) n 1))
  (swp [x y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/dswap n buf stride
                       (.buf ^DoubleBlockVector y)
                       (.stride ^DoubleBlockVector y))
          x)
      (throw (IllegalArgumentException.
              (format DIMENSION_MSG \x n \y (.dim ^Vector y))))))
  (copy [_ y]
    (if (= n (.dim ^Vector y))
      (do (CBLAS/dcopy n buf stride
                       (.buf ^DoubleBlockVector y)
                       (.stride ^DoubleBlockVector y))
          y)
      (throw (IllegalArgumentException.
              (format DIMENSION_MSG \x n \y (.dim ^Vector y))))))
  Functor
  (fmap! [x f]
    (loop [i 0 res buf]
      (if (< i n)
        (recur (inc i)
               (.putDouble
                res (* 8 stride i)
                (.invokePrim ^IFn$DD f
                             (.getDouble buf (* 8 stride i)))))
        x)))
  (fmap! [x f y]
    (if (<= n (.dim ^Vector y))
      (loop [i 0 res buf]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDD f
                               (.getDouble buf (* 8 stride i))
                               (.entry ^RealVector y i))))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (loop [i 0 res buf]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDDD f
                               (.getDouble buf (* 8 stride i))
                               (.entry ^RealVector y i)
                               (.entry ^RealVector z i))))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z w]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z) (.dim ^Vector w)))
      (loop [i 0 res buf]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDDDD f
                               (.getDouble buf (* 8 stride i))
                               (.entry ^RealVector y i)
                               (.entry ^RealVector z i)
                               (.entry ^RealVector w i))))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z w ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  Foldable
  (fold [x]
    (loop [i 0 res 0.0]
      (if (< i n)
        (recur (inc i)
               (+ res (.getDouble buf (* 8 stride i))))
        res)))
  (fold [x f]
    (freduce x f))
  (fold [x f id]
    (freduce x id f))
  Reducible
  (freduce [x f]
    (loop [i 0 res 0.0]
      (if (< i n)
        (recur (inc i)
               (.invokePrim ^IFn$DDD f res
                            (.getDouble buf (* 8 stride i))))
        res)))
  (freduce [x acc f]
    (if (number? acc)
      (loop [i 0 res (double acc)]
        (if (< i n)
          (recur (inc i)
                 (.invokePrim ^IFn$DDD f res
                              (.getDouble buf (* 8 stride i))))
          res))
      (loop [i 0 res acc]
        (if (and (< i n) res)
          (recur (inc i)
                 (.invokePrim ^IFn$ODO f res
                              (.getDouble buf (* 8 stride i))))
          res))))
  (freduce [x acc f y]
    (if (<= n (.dim ^Vector y))
      (if (number? acc)
        (loop [i 0 res (double acc)]
          (if (and (< i n) (Double/isFinite res))
            (recur (inc i)
                   (.invokePrim ^IFn$DDDD f res
                                (.getDouble buf (* 8 stride i))
                                (.entry ^RealVector y i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDO f res
                                (.getDouble buf (* 8 stride i))
                                (.entry ^RealVector y i)))
            res)))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (freduce [x acc f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (if (number? acc)
        (loop [i 0 res (double acc)]
          (if (and (< i n) (Double/isFinite res))
            (recur (inc i)
                   (.invokePrim ^IFn$DDDDD f res
                                (.getDouble buf (* 8 stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDDO f res
                                (.getDouble buf (* 8 stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res)))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn
  (invoke [x i]
    (.entry x i))
  RealVectorEditor
  (dim [_]
    n)
  (entry [_ i]
    (.getDouble buf (* 8 stride i)))
  (setEntry [x i val]
    (do
      (.putDouble buf (* 8 stride i) val)
      x))
  (update [x i f]
    (do
      (.putDouble buf (* 8 stride i)
                  (.invokePrim ^IFn$DD f
                               (.getDouble buf (* 8 stride i))))))
  (subvector [_ k l]
    (DoubleBlockVector.
     (slice-buffer buf (* 8 k stride) (* 8 l stride))
     l stride))
  (dot [_ y]
    (CBLAS/ddot n
                buf stride
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

(defmethod print-method DoubleBlockVector
  [^DoubleBlockVector dv ^java.io.Writer w]
  (.write w (format "#<DoubleBlockVector| n:%d, stride:%d, %s>"
                    (.dim dv) (.stride dv) (pr-str (seq dv)))))

(declare update-segment)

;; ================= GE General Matrix =====================
(deftype DoubleGeneralMatrix [^ByteBuffer buf ^long m
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :DoubleBandMatrix)
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
    (DoubleGeneralMatrix. (direct-buffer (* 8 m n)) m n m order))
  (copy [a b]
    (let [dgeb ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order dgeb))
                     (= m ld (.ld dgeb)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order dgeb))
                     (= n ld (.ld dgeb))))
          (do
            (CBLAS/dcopy (* m n) buf 1 (.buf b) 1)
            b)
          (if (= CBLAS/ORDER_COLUMN_MAJOR order)
            (loop [i 0]
              (if (< i n)
                (do
                  (copy (.col a i) (.col b i))
                  (recur (inc i)))
                b))
            (loop [i 0]
              (if (< i m)
                (do
                  (copy (.row a i) (.row b i))
                  (recur (inc i)))
                b))))
        (throw (IllegalArgumentException.
                "I can not copy incompatible matrices.")))))
  (swp [a b]
    (let [dgeb ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order dgeb))
                     (= m ld (.ld dgeb)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order dgeb))
                     (= n ld (.ld dgeb))))
          (do
            (CBLAS/dswap (* m n) buf 1 (.buf b) 1)
            a)
          (if (= CBLAS/ORDER_COLUMN_MAJOR order)
            (loop [i 0]
              (if (< i n)
                (do
                  (swp (.col a i) (.col b i))
                  (recur (inc i)))
                a))
            (loop [i 0]
              (if (< i m)
                (do
                  (swp (.row a i) (.row b i))
                  (recur (inc i)))
                a))))
        (throw (IllegalArgumentException.
                "I can not swap incompatible matrices.")))))
  clojure.lang.Seqable
  (seq [a]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
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
  Functor
  (fmap! [x f]
    (let [end (if (= CBLAS/ORDER_COLUMN_MAJOR order) n m)]
      (loop [j 0]
        (if (< j end)
          (do (update-segment x j f)
              (recur (inc j)))
          x))))
  (fmap! [x f y]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y))
             (<= m (.mrows ^Matrix z)) (<= n (.ncols ^Matrix z)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j) (.col ^Matrix z j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z w]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y))
             (<= m (.mrows ^Matrix z)) (<= n (.ncols ^Matrix z))
             (<= m (.mrows ^Matrix w)) (<= n (.ncols ^Matrix w)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j)
                   (.col ^Matrix z j) (.col ^Matrix w j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z w ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  Foldable
  (fold [x]
    (loop [j 0 res 0.0]
      (if (< j n)
        (recur (inc j)
               (double
                (loop [i 0 res res]
                  (if (< i m)
                    (recur (inc i)
                           (+ res (.entry x i j)))
                    res))))
        res)))
  (fold [x f]
    (fold f 0.0))
  (fold [x f id]
    (freduce x id f))
  Reducible
  (freduce [x f]
    (fold x f))
  (freduce [x acc f]
    (if (number? acc)
      (loop [j 0 res (double acc)]
        (if (< j n)
          (recur (inc j)
                 (double
                  (loop [i 0 res res]
                    (if (< i m)
                      (recur (inc i)
                             (.invokePrim ^IFn$DDD f res
                                          (.entry x i j)))
                      res))))
          res))
      (loop [j 0 res acc]
        (if (< j n)
          (recur (inc j)
                 (loop [i 0 res res]
                   (if (< i m)
                     (recur (inc i)
                            (.invokePrim ^IFn$ODO f res
                                         (.entry x i j)))
                     res)))
          res))))
  (freduce [x acc f y];;TODO benchmark this against fmap! approach
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y)))
      (if (number? acc)
        (loop [j 0 res (double acc)]
          (if (< j n)
            (recur (inc j)
                   (double (loop [i 0 res res]
                             (if (< i m)
                               (recur (inc i)
                                      (.invokePrim ^IFn$DDDD f res
                                                   (.entry x i j)
                                                   (.entry ^RealMatrix y i j)))
                               res))))
            res))
        (loop [j 0 res acc]
          (if (and (< j n) res)
            (recur (inc j)
                   (loop [i 0 res res]
                     (if (and (< i m) res)
                       (recur (inc i)
                              (.invokePrim ^IFn$ODDO f res
                                           (.entry x i j)
                                           (.entry ^RealMatrix y i j)))
                       res)))
            res)))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (freduce [x acc f y z]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y))
             (<= m (.mrows ^Matrix z)) (<= n (.ncols ^Matrix z)))
      (if (number? acc)
        (loop [j 0 res (double acc)]
          (if (< j n)
            (recur (inc j)
                   (double (loop [i 0 res res]
                             (if (< i m)
                               (recur (inc i)
                                      (.invokePrim ^IFn$DDDDD f res
                                                   (.entry x i j)
                                                   (.entry ^RealMatrix y i j)
                                                   (.entry ^RealMatrix z i j)))
                               res))))
            res))
        (loop [j 0 res acc]
          (if (and (< j n) res)
            (recur (inc j)
                   (loop [i 0 res res]
                     (if (and (< i m) res)
                       (recur (inc i)
                              (.invokePrim ^IFn$ODDDO f res
                                           (.entry x i j)
                                           (.entry ^RealMatrix y i j)
                                           (.entry ^RealMatrix z i j)))
                       res)))
            res)))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  RealMatrixEditor
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (.getDouble buf (+ (* 8 ld j) (* 8 i)))
      (.getDouble buf (+ (* 8 ld i) (* 8 j)))))
  (setEntry [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR order)
        (.putDouble buf (+ (* 8 ld j) (* 8 i)) val)
        (.putDouble buf (+ (* 8 ld i) (* 8 j)) val))
      a))
  (update [a i j f]
    (do
      (let [ind (if (= CBLAS/ORDER_COLUMN_MAJOR order)
                  (+ (* 8 ld j) (* 8 i))
                  (+ (* 8 ld i) (* 8 j)))]
        (.putDouble buf ind
                    (.invokePrim ^IFn$DD f
                                 (.getDouble buf ind))))
      a))
  (row [_ i]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer buf (* 8 i) (- (.capacity buf) (* 8 i)))
       n ld)
      (DoubleBlockVector.
       (slice-buffer buf (* 8 ld i) (* 8 n))
       n 1)))
  (col [_ j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer buf (* 8 ld j) (* 8 m))
       m 1)
      (DoubleBlockVector.
       (slice-buffer buf (* 8 j) (- (.capacity buf) (* 8 j)))
       m ld)))
  (submatrix [_ i j k l]
    (DoubleGeneralMatrix. (if (= CBLAS/ORDER_COLUMN_MAJOR order)
                            (slice-buffer buf (+ (* 8 ld j) (* 8 i))
                                          (* 8 ld l))
                            (slice-buffer buf (+ (* 8 ld i) (* 8 j))
                                          (* 8 ld k)))
                          k l ld order))
  (transpose [_]
    (DoubleGeneralMatrix. buf n m ld
                          (if (= CBLAS/ORDER_COLUMN_MAJOR order)
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

(defn ^:private update-segment [^DoubleGeneralMatrix a ^long k f]
  (let [start (* 8 (long (.ld a)) k)
        end (+ start (* 8 (if (= CBLAS/ORDER_COLUMN_MAJOR (.order a))
                            (.m a)
                            (.n a))))
        buf ^ByteBuffer (.buf a)]
    (loop [ind start]
      (if (< ind end)
        (do (.putDouble buf ind
                        (.invokePrim ^IFn$DD f
                                     (.getDouble buf ind)))
            (recur (+ 8 ind)))
        a))))

(defmethod print-method DoubleGeneralMatrix
  [^DoubleGeneralMatrix m ^java.io.Writer w]
  (.write w (format "#<DoubleGeneralMatrix| %s, mxn: %dx%d, ld:%d, %s>"
                    (if (= CBLAS/ORDER_COLUMN_MAJOR (.order m)) "COL" "ROW")
                    (.mrows m) (.ncols m) (.ld m) (pr-str (seq m)))))

(primitive-math/unuse-primitive-operators)
