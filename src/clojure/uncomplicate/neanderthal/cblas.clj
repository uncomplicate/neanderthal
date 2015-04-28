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
  Functor
  (fmap! [x f]
    (cond (instance? IFn$DD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putFloat
                      res (* Float/BYTES stride i)
                      (.invokePrim ^IFn$DD f
                                   (.getFloat buf (* Float/BYTES stride i)))))
              x))
          (instance? IFn$D f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putFloat
                      res (* Float/BYTES stride i)
                      (.invokePrim ^IFn$D f)))
              x))
          (instance? IFn$LDD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putFloat
                      res (* Float/BYTES stride i)
                      (.invokePrim ^IFn$LDD f i
                                   (.getFloat buf (* Float/BYTES stride i)))))
              x))
          (instance? IFn$LD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putFloat
                      res (* Float/BYTES stride i)
                      (.invokePrim ^IFn$LD f i)))
              x))
          :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG))))
  (fmap! [x f y]
    (if (<= n (.dim ^Vector y))
      (cond (instance? IFn$DDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putFloat
                        res (* Float/BYTES stride i)
                        (.invokePrim ^IFn$DDD f
                                     (.getFloat buf (* Float/BYTES stride i))
                                     (.entry ^RealVector y i))))
                x))
            (instance? IFn$LDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putFloat
                        res (* Float/BYTES stride i)
                        (.invokePrim ^IFn$LDDD f i
                                     (.getFloat buf (* Float/BYTES stride i))
                                     (.entry ^RealVector y i))))
                x))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (cond (instance? IFn$DDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putFloat
                        res (* Float/BYTES stride i)
                        (.invokePrim ^IFn$DDDD f
                                     (.getFloat buf (* Float/BYTES stride i))
                                     (.entry ^RealVector y i)
                                     (.entry ^RealVector z i))))
                x))
            (instance? IFn$LDDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putFloat
                        res (* Float/BYTES stride i)
                        (.invokePrim ^IFn$LDDDD f i
                                     (.getFloat buf (* Float/BYTES stride i))
                                     (.entry ^RealVector y i)
                                     (.entry ^RealVector z i))))
                x))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z w]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z) (.dim ^Vector w)))
      (loop [i 0 res buf]
        (if (< i n)
          (recur (inc i)
                 (.putFloat
                  res (* Float/BYTES stride i)
                  (.invokePrim ^IFn$DDDDD f
                               (.getFloat buf (* Float/BYTES stride i))
                               (.entry ^RealVector y i)
                               (.entry ^RealVector z i)
                               (.entry ^RealVector w i))))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z w ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  Foldable
  (fold [x]
    (loop [i 0 res 0.0]
      (if (< i n)
        (recur (inc i)
               (+ res (.getFloat buf (* Float/BYTES stride i))))
        res)))
  (fold [x f]
    (freduce x f))
  (fold [x f id]
    (freduce x id f))
  Reducible
  (freduce [x f]
    (loop [i 2 res (.invokePrim ^IFn$DDD f (.entry x 0) (.entry x 1))]
      (if (< i n)
        (recur (inc i)
               (.invokePrim ^IFn$DDD f res
                            (.getFloat buf (* Float/BYTES stride i))))
        res)))
  (freduce [x acc f]
    (cond (instance? IFn$DDD f)
          (loop [i 0 res (double acc)]
            (if (< i n)
              (recur (inc i)
                     (.invokePrim ^IFn$DDD f res
                                  (.getFloat buf (* Float/BYTES stride i))))
              res))
          (instance? IFn$DLDD f)
          (loop [i 0 res (double acc)]
            (if (< i n)
              (recur (inc i)
                     (.invokePrim ^IFn$DLDD f res i
                                  (.getFloat buf (* Float/BYTES stride i))))
              res))
          (instance? IFn$ODO f)
          (loop [i 0 res acc]
            (if (and (< i n) res)
              (recur (inc i)
                     (.invokePrim ^IFn$ODO f res
                                  (.getFloat buf (* Float/BYTES stride i))))
              res))
          (instance? IFn$OLDO f)
          (loop [i 0 res acc]
            (if (and (< i n) res)
              (recur (inc i)
                     (.invokePrim ^IFn$OLDO f res i
                                  (.getFloat buf (* Float/BYTES stride i))))
              res))
          :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG))))
  (freduce [x acc f y]
    (if (<= n (.dim ^Vector y))
      (cond (instance? IFn$DDDD f)
            (loop [i 0 res (double acc)]
              (if (and (< i n) (Double/isFinite res))
                (recur (inc i)
                       (.invokePrim ^IFn$DDDD f res
                                    (.getFloat buf (* Float/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$DLDDD f)
            (loop [i 0 res (double acc)]
              (if (and (< i n) (Double/isFinite res))
                (recur (inc i)
                       (.invokePrim ^IFn$DLDDD f res i
                                    (.getFloat buf (* Float/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$ODDO f)
            (loop [i 0 res acc]
              (if (and (< i n) res)
                (recur (inc i)
                       (.invokePrim ^IFn$ODDO f res
                                    (.getFloat buf (* Float/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$OLDDO f)
            (loop [i 0 res acc]
              (if (and (< i n) res)
                (recur (inc i)
                       (.invokePrim ^IFn$OLDDO f res i
                                    (.getFloat buf (* Float/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (if (number? acc)
        (loop [i 0 res (double acc)]
          (if (and (< i n) (Double/isFinite res))
            (recur (inc i)
                   (.invokePrim ^IFn$DDDDD f res
                                (.getFloat buf (* Float/BYTES stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDDO f res
                                (.getFloat buf (* Float/BYTES stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn
  (invoke [x i]
    (.entry x i))
  RealChangeable
  (set [x val]
    (loop [i 0]
      (if (< i n)
        (do
          (.putFloat buf (* Float/BYTES stride i) val)
          (recur (inc i)))
        x)))
  (set [x i val]
    (do
      (.putFloat buf (* Float/BYTES stride i) val)
      x))
  (alter [x i f]
    (do
      (.putFloat buf (* Float/BYTES stride i)
                  (.invokePrim ^IFn$DD f
                               (.getFloat buf (* Float/BYTES stride i))))))
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
    (CBLAS/sdot n
                buf stride
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

(defmethod print-method FloatBlockVector
  [^FloatBlockVector dv ^java.io.Writer w]
  (.write w (format "#<FloatBlockVector| n:%d, stride:%d, %s>"
                    (.dim dv) (.stride dv) (pr-str (take 100 (seq dv))))))
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
    (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf)))
  Carrier
  (zero [_]
    (DoubleBlockVector. (direct-buffer (* Double/BYTES n)) n 1))
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
  Functor
  (fmap! [x f]
    (cond (instance? IFn$DD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putDouble
                      res (* Double/BYTES stride i)
                      (.invokePrim ^IFn$DD f
                                   (.getDouble buf (* Double/BYTES stride i)))))
              x))
          (instance? IFn$D f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putDouble
                      res (* Double/BYTES stride i)
                      (.invokePrim ^IFn$D f)))
              x))
          (instance? IFn$LDD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putDouble
                      res (* Double/BYTES stride i)
                      (.invokePrim ^IFn$LDD f i
                                   (.getDouble buf (* Double/BYTES stride i)))))
              x))
          (instance? IFn$LD f)
          (loop [i 0 res buf]
            (if (< i n)
              (recur (inc i)
                     (.putDouble
                      res (* Double/BYTES stride i)
                      (.invokePrim ^IFn$LD f i)))
              x))
          :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG))))
  (fmap! [x f y]
    (if (<= n (.dim ^Vector y))
      (cond (instance? IFn$DDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putDouble
                        res (* Double/BYTES stride i)
                        (.invokePrim ^IFn$DDD f
                                     (.getDouble buf (* Double/BYTES stride i))
                                     (.entry ^RealVector y i))))
                x))
            (instance? IFn$LDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putDouble
                        res (* Double/BYTES stride i)
                        (.invokePrim ^IFn$LDDD f i
                                     (.getDouble buf (* Double/BYTES stride i))
                                     (.entry ^RealVector y i))))
                x))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (cond (instance? IFn$DDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putDouble
                        res (* Double/BYTES stride i)
                        (.invokePrim ^IFn$DDDD f
                                     (.getDouble buf (* Double/BYTES stride i))
                                     (.entry ^RealVector y i)
                                     (.entry ^RealVector z i))))
                x))
            (instance? IFn$LDDDD f)
            (loop [i 0 res buf]
              (if (< i n)
                (recur (inc i)
                       (.putDouble
                        res (* Double/BYTES stride i)
                        (.invokePrim ^IFn$LDDDD f i
                                     (.getDouble buf (* Double/BYTES stride i))
                                     (.entry ^RealVector y i)
                                     (.entry ^RealVector z i))))
                x))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z w]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z) (.dim ^Vector w)))
      (loop [i 0 res buf]
        (if (< i n)
          (recur (inc i)
                 (.putDouble
                  res (* Double/BYTES stride i)
                  (.invokePrim ^IFn$DDDDD f
                               (.getDouble buf (* Double/BYTES stride i))
                               (.entry ^RealVector y i)
                               (.entry ^RealVector z i)
                               (.entry ^RealVector w i))))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z w ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  Foldable
  (fold [x]
    (loop [i 0 res 0.0]
      (if (< i n)
        (recur (inc i)
               (+ res (.getDouble buf (* Double/BYTES stride i))))
        res)))
  (fold [x f]
    (freduce x f))
  (fold [x f id]
    (freduce x id f))
  Reducible
  (freduce [x f]
    (loop [i 2 res (.invokePrim ^IFn$DDD f
                                (.entry x 0)
                                (.entry x 1))]
      (if (< i n)
        (recur (inc i)
               (.invokePrim ^IFn$DDD f res
                            (.getDouble buf (* Double/BYTES stride i))))
        res)))
  (freduce [x acc f]
    (cond (instance? IFn$DDD f)
          (loop [i 0 res (double acc)]
            (if (< i n)
              (recur (inc i)
                     (.invokePrim ^IFn$DDD f res
                                  (.getDouble buf (* Double/BYTES stride i))))
              res))
          (instance? IFn$DLDD f)
          (loop [i 0 res (double acc)]
            (if (< i n)
              (recur (inc i)
                     (.invokePrim ^IFn$DLDD f res i
                                  (.getDouble buf (* Double/BYTES stride i))))
              res))
          (instance? IFn$ODO f)
          (loop [i 0 res acc]
            (if (and (< i n) res)
              (recur (inc i)
                     (.invokePrim ^IFn$ODO f res
                                  (.getDouble buf (* Double/BYTES stride i))))
              res))
          (instance? IFn$OLDO f)
          (loop [i 0 res acc]
            (if (and (< i n) res)
              (recur (inc i)
                     (.invokePrim ^IFn$OLDO f res i
                                  (.getDouble buf (* Double/BYTES stride i))))
              res))
          :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG))))
  (freduce [x acc f y]
    (if (<= n (.dim ^Vector y))
      (cond (instance? IFn$DDDD f)
            (loop [i 0 res (double acc)]
              (if (and (< i n) (Double/isFinite res))
                (recur (inc i)
                       (.invokePrim ^IFn$DDDD f res
                                    (.getDouble buf (* Double/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$DLDDD f)
            (loop [i 0 res (double acc)]
              (if (and (< i n) (Double/isFinite res))
                (recur (inc i)
                       (.invokePrim ^IFn$DLDDD f res i
                                    (.getDouble buf (* Double/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$ODDO f)
            (loop [i 0 res acc]
              (if (and (< i n) res)
                (recur (inc i)
                       (.invokePrim ^IFn$ODDO f res
                                    (.getDouble buf (* Double/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            (instance? IFn$OLDDO f)
            (loop [i 0 res acc]
              (if (and (< i n) res)
                (recur (inc i)
                       (.invokePrim ^IFn$OLDDO f res i
                                    (.getDouble buf (* Double/BYTES stride i))
                                    (.entry ^RealVector y i)))
                res))
            :default (throw (IllegalArgumentException. ^String PRIMITIVE_FN_MSG)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z]
    (if (<= n (min (.dim ^Vector y) (.dim ^Vector z)))
      (if (number? acc)
        (loop [i 0 res (double acc)]
          (if (and (< i n) (Double/isFinite res))
            (recur (inc i)
                   (.invokePrim ^IFn$DDDDD f res
                                (.getDouble buf (* Double/BYTES stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res))
        (loop [i 0 res acc]
          (if (and (< i n) res)
            (recur (inc i)
                   (.invokePrim ^IFn$ODDDO f res
                                (.getDouble buf (* Double/BYTES stride i))
                                (.entry ^RealVector y i)
                                (.entry ^RealVector z i)))
            res)))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn
  (invoke [x i]
    (.entry x i))
  RealChangeable
  (set [x val]
    (loop [i 0]
      (if (< i n)
        (do
          (.putDouble buf (* Double/BYTES stride i) val)
          (recur (inc i)))
        x)))
  (set [x i val]
    (do
      (.putDouble buf (* Double/BYTES stride i) val)
      x))
  (alter [x i f]
    (do
      (.putDouble buf (* Double/BYTES stride i)
                  (.invokePrim ^IFn$DD f
                               (.getDouble buf (* Double/BYTES stride i))))))
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
                    (.dim dv) (.stride dv) (pr-str (take 100 (seq dv))))))

;; ================= GE General Matrix =====================
(declare update-segment)

(declare set-segment)

(declare compact?)

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
    (DoubleGeneralMatrix. (direct-buffer (* Double/BYTES m n)) m n m order))
  (copy [a b]
    (let [a ^DoubleGeneralMatrix a
          b ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order b))
                     (= m ld (.ld b)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order b))
                     (= n ld (.ld b))))
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
    (let [a ^DoubleGeneralMatrix a
          b ^DoubleGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order b))
                     (= m ld (.ld b)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order b))
                     (= n ld (.ld b))))
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
  (fmap! [x f y] ;; TODO support IFn$LDxxx
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y))
             (<= m (.mrows ^Matrix z)) (<= n (.ncols ^Matrix z)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j) (.col ^Matrix z j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  RealChangeable
  (set [a val]
    (if (compact? a)
      (loop [i 0]
        (if (< i (* m n))
          (do (.putDouble buf (* Double/BYTES i) val)
              (recur (inc i)))
          a))
      (let [end (if (= CBLAS/ORDER_COLUMN_MAJOR order) n m)]
        (loop [j 0]
          (if (< j end)
            (do (set-segment a j val)
                (recur (inc j)))
            a)))))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR order)
        (.putDouble buf (+ (* Double/BYTES ld j) (* Double/BYTES i)) val)
        (.putDouble buf (+ (* Double/BYTES ld i) (* Double/BYTES j)) val))
      a))
  (alter [a i j f]
    (do
      (let [ind (if (= CBLAS/ORDER_COLUMN_MAJOR order)
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
  (row [_ i]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES i) (- (.capacity buf) (* Double/BYTES i)))
       n ld)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES ld i) (* Double/BYTES n))
       n 1)))
  (col [_ j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES ld j) (* Double/BYTES m))
       m 1)
      (DoubleBlockVector.
       (slice-buffer buf (* Double/BYTES j) (- (.capacity buf) (* Double/BYTES j)))
       m ld)))
  (submatrix [_ i j k l]
    (DoubleGeneralMatrix. (if (= CBLAS/ORDER_COLUMN_MAJOR order)
                            (slice-buffer buf (+ (* Double/BYTES ld j) (* Double/BYTES i))
                                          (* Double/BYTES ld l))
                            (slice-buffer buf (+ (* Double/BYTES ld i) (* Double/BYTES j))
                                          (* Double/BYTES ld k)))
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

(deftype FloatGeneralMatrix [^ByteBuffer buf ^long m
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (freduce this
             (-> (hash :FloatBandMatrix)
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
  (copy [a b]
    (let [a ^FloatGeneralMatrix a
          b ^FloatGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order b))
                     (= m ld (.ld b)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order b))
                     (= n ld (.ld b))))
          (do
            (CBLAS/scopy (* m n) buf 1 (.buf b) 1)
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
    (let [a ^FloatGeneralMatrix a
          b ^FloatGeneralMatrix b]
      (if (and (= m (.m b)) (= n (.n b)))
        (if (or (and (= CBLAS/ORDER_COLUMN_MAJOR order (.order b))
                     (= m ld (.ld b)))
                (and (= CBLAS/ORDER_ROW_MAJOR order (.order b))
                     (= n ld (.ld b))))
          (do
            (CBLAS/sswap (* m n) buf 1 (.buf b) 1)
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
  (fmap! [x f y] ;; TODO support IFn$LDxxx
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (fmap! [x f y z]
    (if (and (<= m (.mrows ^Matrix y)) (<= n (.ncols ^Matrix y))
             (<= m (.mrows ^Matrix z)) (<= n (.ncols ^Matrix z)))
      (loop [j 0]
        (if (< j n)
          (do
            (fmap! (.col x j) f (.col ^Matrix y j) (.col ^Matrix z j))
            (recur (inc j)))
          x))
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
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
      (throw (IllegalArgumentException. (format DIMENSIONS_MSG n)))))
  (freduce [x acc f y z ws]
    (throw (UnsupportedOperationException.
            "Primitive functions support max 4 args.")))
  RealChangeable
  (set [a val]
    (if (compact? a)
      (loop [i 0]
        (if (< i (* m n))
          (do (.putFloat buf (* Float/BYTES i) val)
              (recur (inc i)))
          a))
      (let [end (if (= CBLAS/ORDER_COLUMN_MAJOR order) n m)]
        (loop [j 0]
          (if (< j end)
            (do (set-segment a j val)
                (recur (inc j)))
            a)))))
  (set [a i j val]
    (do
      (if (= CBLAS/ORDER_COLUMN_MAJOR order)
        (.putFloat buf (+ (* Float/BYTES ld j) (* Float/BYTES i)) val)
        (.putFloat buf (+ (* Float/BYTES ld i) (* Float/BYTES j)) val))
      a))
  (alter [a i j f]
    (do
      (let [ind (if (= CBLAS/ORDER_COLUMN_MAJOR order)
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
  (row [_ i]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES i) (- (.capacity buf) (* Float/BYTES i)))
       n ld)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES ld i) (* Float/BYTES n))
       n 1)))
  (col [_ j]
    (if (= CBLAS/ORDER_COLUMN_MAJOR order)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES ld j) (* Float/BYTES m))
       m 1)
      (FloatBlockVector.
       (slice-buffer buf (* Float/BYTES j) (- (.capacity buf) (* Float/BYTES j)))
       m ld)))
  (submatrix [_ i j k l]
    (FloatGeneralMatrix. (if (= CBLAS/ORDER_COLUMN_MAJOR order)
                            (slice-buffer buf (+ (* Float/BYTES ld j) (* Float/BYTES i))
                                          (* Float/BYTES ld l))
                            (slice-buffer buf (+ (* Float/BYTES ld i) (* Float/BYTES j))
                                          (* Float/BYTES ld k)))
                          k l ld order))
  (transpose [_]
    (FloatGeneralMatrix. buf n m ld
                          (if (= CBLAS/ORDER_COLUMN_MAJOR order)
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

(defn compact? [^DoubleGeneralMatrix a]
  (if (= CBLAS/ORDER_COLUMN_MAJOR (.order a))
    (= (.m a) (.ld a))
    (= (.n a) (.ld a))))

(defn ^:private update-segment [^DoubleGeneralMatrix a ^long k f]
  (let [start (* Double/BYTES (long (.ld a)) k)
        end (+ start (* Double/BYTES (if (= CBLAS/ORDER_COLUMN_MAJOR (.order a))
                            (.m a)
                            (.n a))))
        buf ^ByteBuffer (.buf a)]
    (loop [ind start]
      (if (< ind end)
        (do (.putDouble buf ind
                        (.invokePrim ^IFn$DD f
                                     (.getDouble buf ind)))
            (recur (+ Double/BYTES ind)))
        a))))

(defn ^:private set-segment [^DoubleGeneralMatrix a ^long k ^long end ^double val]
  (let [start (* Double/BYTES (.ld a) k)
        end (+ start (* Double/BYTES (if (= CBLAS/ORDER_COLUMN_MAJOR (.order a))
                            (.m a)
                            (.n a))))
        buf ^ByteBuffer (.buf a)]
    (loop [ind start]
      (if (< ind end)
        (do (.putDouble buf ind val)
            (recur (+ Double/BYTES ind)))
        a))))

(defmethod print-method DoubleGeneralMatrix
  [^DoubleGeneralMatrix m ^java.io.Writer w]
  (.write w (format "#<DoubleGeneralMatrix| %s, mxn: %dx%d, ld:%d, %s>"
                    (if (= CBLAS/ORDER_COLUMN_MAJOR (.order m)) "COL" "ROW")
                    (.mrows m) (.ncols m) (.ld m) (pr-str (seq m)))))

(primitive-math/unuse-primitive-operators)
