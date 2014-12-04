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
            Block RealVector RealMatrix Vector Matrix
            RealVectorEditor RealMatrixEditor]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

(def ^:private DIFF_DIM_MSG
  "Vector dimensions should be %d.")

(def  MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(defmacro ^:private vector-fmap
  [x put f & zs]
  `(let [n# (.dim ~(with-meta x {:tag (class x)}))
         stride#  (.stride ~(with-meta x {:tag (class x)}))]
     (loop [i# 0 acc# (.buf ~(with-meta x {:tag (class x)}))]
       (if (< i# n#)
         (recur (inc i#)
                (. (with-meta acc# {:tag java.nio.ByteBuffer})
                   (~put (* 8 stride# i#)
                         (.invokePrim ~(with-meta f {:tag (class f)})
                                      (.entry ~(with-meta x {:tag (class x)}) i#)
                                      ~@(map (fn [s#]
                                               `(.entry ~(with-meta s# {:tag (class x)}) i#))
                                             zs)
                                      ))))
         ~x))))

(defn entry-eq [res ^double x ^double y]
  (= x y))
;;-------------- Double Vector -----------------------------

(deftype DoubleBlockVector [^ByteBuffer arr ^long n ^long stride]
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
    (wrap-byte-seq float64 (* 8 stride) 0 (byte-seq arr)))
  Carrier
  (zero [_]
    (DoubleBlockVector. (direct-buffer (* 8 n)) n 1))
  Functor
  (fmap! [x f]
    (loop [i 0 res arr]
      (if (< i n)
        (recur (inc i)
               (.putDouble
                res (* 8 stride i)
                (.invokePrim ^IFn$DD f
                             (.getDouble arr (* 8 stride i)))))
        x)))
  (fmap! [x f y]
    (if (<= n (.dim ^Vector y))
      (loop [i 0 res arr]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDD f
                               (.getDouble arr (* 8 stride i))
                               (.entry ^RealVector y i))))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (loop [i 0 res arr]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDDD f
                               (.getDouble arr (* 8 stride i))
                               (.entry ^RealVector y i)
                               (.entry ^RealVector z i))))
          x))
      (throw (IllegalArgumentException. (format DIFF_DIM_MSG n)))))
  (fmap! [x f y z w]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z) (.dim ^Vector w)))
      (loop [i 0 res arr]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* 8 stride i)
                  (.invokePrim ^IFn$DDDDD f
                               (.getDouble arr (* 8 stride i))
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
               (+ res (.getDouble arr (* 8 stride i))))
        res)))
  (fold [x f]
    (loop [i 0 res 0.0]
      (if (< i n)
        (recur (inc i)
               (.invokePrim ^IFn$DDD f res
                            (.getDouble arr (* 8 stride i))))
        res)))
  (fold [x f id]
    (loop [i 0 res (double id)]
      (if (< i n)
        (recur (inc i)
               (.invokePrim ^IFn$DDD f res
                            (.getDouble arr (* 8 stride i))))
        res)))
  Reducible
  (freduce [x f]
    (fold x f))
  (freduce [x acc f]
    (if (number? acc)
      (fold x f acc)
      (loop [i 0 res acc]
        (if (and (< i n) res)
          (recur (inc i)
                 (.invokePrim ^IFn$ODO f res
                              (.getDouble arr (* 8 stride i))))
          res))))
  (freduce [x acc f y]
    (if (<= n (.dim ^Vector y))
      (if (number? acc)
        (loop [i 0 res (double acc)]
          (if (and (< i n) (Double/isFinite res))
            (recur (inc i)
                   (.invokePrim ^IFn$DDDD f res
                                (.getDouble arr (* 8 stride i))
                                (.entry ^RealVector y i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDO f res
                                (.getDouble arr (* 8 stride i))
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
                                (.getDouble arr (* 8 stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDDO f res
                                (.getDouble arr (* 8 stride i))
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
  Block
  (buf [_]
    arr)
  (stride [_]
    stride)
  (length [_]
    n)
  RealVectorEditor
  (dim [_]
    n)
  (entry [_ i]
    (.getDouble arr (* 8 stride i)))
  (setEntry [a i val]
    (do
      (.putDouble arr (* 8 stride i) val)
      a))
  (segment [_ k l]
    (DoubleBlockVector.
     (slice-buffer arr (* 8 k stride) (* 8 l stride))
     l stride))
  (dot [_ y]
    (let [by ^Block y]
      (CBLAS/ddot n
                  arr stride
                  (.buf by) (.stride by))))
  (nrm2 [_]
    (CBLAS/dnrm2 n arr stride))
  (asum [_]
    (CBLAS/dasum n arr stride))
  (iamax [_]
    (CBLAS/idamax n arr stride))
  (rot [x y c s]
    (let [by ^Block y]
      (do (CBLAS/drot n
                      arr stride
                      (.buf by) (.stride by)
                      c s)
          x)))
  (swap [x y]
    (let [by ^Block y]
      (do (CBLAS/dswap n
                       arr stride
                       (.buf by) (.stride by))
          x)))
  (scal [x alpha]
    (do (CBLAS/dscal n alpha arr stride)
        x))
  (copy [_ y]
    (let [by ^Block y]
      (do (CBLAS/dcopy n
                       arr stride
                       (.buf by) (.stride by))
          y)))
  (axpy [_ alpha y]
    (let [by ^Block y]
      (do (CBLAS/daxpy n alpha arr stride
                       (.buf by)
                       (.stride by))
          y))))

(defmethod print-method DoubleBlockVector
  [^DoubleBlockVector dv ^java.io.Writer w]
  (.write w (format "#<DoubleBlockVector| n:%d, stride:%d %s>"
                    (.dim dv) (.stride dv) (pr-str (seq dv)))))

;; ================= GE General Matrix =====================
;; TODO all algorithms are for order=COLUMN_MAJOR and ld=m
(deftype DoubleGeneralMatrix [^ByteBuffer arr ^long m
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (let [dim (* m n)]
      (loop [i 0 res (hash-combine
                      (hash :DoubleGeneralMatrix) dim)]
        (if (< i dim)
          (recur (inc i)
                 (hash-combine res (.entry this (rem i m) (rem i n))))
          res))))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? DoubleGeneralMatrix y)
     (and (= (.mrows x) (.mrows ^Matrix y))
          (= (.ncols x) (.ncols ^Matrix y))
          (freduce x true entry-eq y))

     :default false))
  Carrier
  (zero [_]
    (DoubleGeneralMatrix. (direct-buffer (* 8 m n)) m n m order))
  clojure.lang.Seqable
  (seq [_]
    (map (partial wrap-byte-seq float64)
         (cross-section (byte-seq arr) 0 (* 8 m) (* 8 m))))
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
    (let [lnt (* m n)]
      (loop [i 0 res arr]
        (if (< i lnt)
          (recur (inc i)
                 (.putDouble
                  res (* 8 i)
                  (.invokePrim ^IFn$DD f
                               (.getDouble arr (* 8 i)))))
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
    (let [lnt (* m n)]
      (loop [i 0 res 0.0]
        (if (< i lnt)
          (recur (inc i)
                 (+ res (.getDouble arr (* 8 i))))
          res))))
  (fold [x f]
    (let [lnt (* m n)]
      (loop [i 0 res 0.0]
        (if (< i lnt)
          (recur (inc i)
                 (.invokePrim ^IFn$DDD f res
                              (.getDouble arr (* 8 i))))
          res))))
  (fold [x f id]
    (let [lnt (* m n)]
      (loop [i 0 res (double id)]
        (if (< i lnt)
          (recur (inc i)
                 (.invokePrim ^IFn$DDD f res
                              (.getDouble arr (* 8 i))))
          res))))
  Reducible
  (freduce [x f]
    (fold x f))
  (freduce [x acc f]
    (let [lnt (* m n)]
      (if (number? acc)
        (fold x f acc)
        (loop [i 0 res acc]
          (if (and (< i lnt) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODO f res
                                (.getDouble arr (* 8 i))))
            res)))))
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
  Block
  (buf [_]
    arr)
  (stride [_]
    ld)
  (length [_]
    (/ (.capacity ^ByteBuffer arr) 8))
  RealMatrixEditor
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (if (= COLUMN_MAJOR order)
      (.getDouble arr (+ (* 8 m j) (* 8 i)))
      (.getDouble arr (+ (* 8 n i) (* 8 j)))))
  (setEntry [a i j val]
    (do
      (if (= COLUMN_MAJOR order)
        (.putDouble arr (+ (* 8 m j) (* 8 i)) val)
        (.putDouble arr (+ (* 8 n i) (* 8 j)) val))
      a))
  (row [a i]
    (if (= COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer arr (* 8 i) (* 8 (- (.length a) i)))
       n m)
      (DoubleBlockVector.
       (slice-buffer arr (* 8 n i) (* 8 n))
       n 1)))
  (col [a j]
    (if (= COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer arr (* 8 m j) (* 8 m))
       m 1)
      (DoubleBlockVector.
       (slice-buffer arr (* 8 j) (* 8 (- (.length a) j)))
       m n)))
  (transpose [_]
    (DoubleGeneralMatrix. arr n m ld (if (= order COLUMN_MAJOR)
                                      ROW_MAJOR
                                      COLUMN_MAJOR)))
  (mv [_ alpha x beta y transa]
    (let [bx ^Block x
          by ^Block y]
      (do (CBLAS/dgemv order transa
                       m n
                       alpha
                       arr ld
                       (.buf bx) (.stride bx)
                       beta
                       (.buf by) (.stride by))
          y)))
  #_(rank [a alpha x y] ;;TODO
    (let [bx ^Block x
          by ^Block y]
      (do (CBLAS/dger order
                      m n
                      alpha
                      (.buf bx) (.stride bx)
                      (.buf by) (.stride by)
                      arr ld)
          a)))
  (mm [_ alpha b beta c transa transb]
    (let [bb ^Block b
          bc ^Block c]
      (do (CBLAS/dgemm order
                       transa transb
                       m (.ncols b) n
                       alpha
                       arr ld
                       (.buf bb) (.stride bb)
                       beta
                       (.buf bc) (.stride bc))
          c))))

(defmethod print-method DoubleGeneralMatrix
  [^DoubleGeneralMatrix m ^java.io.Writer w]
  (.write w (format "#<DoubleGeneralMatrix| mxn: %dx%d, ld:%d %s>"
                    (.mrows m) (.ncols m) (.stride m) (pr-str (seq m)))))

(primitive-math/unuse-primitive-operators)
