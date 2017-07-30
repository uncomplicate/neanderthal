;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.buffer-block
  (:require [vertigo
             [core :refer [wrap]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 int32 int64 wrap-byte-seq]]]
            [uncomplicate.commons.core
             :refer [Releaseable release let-release clean-buffer double-fn
                     wrap-float wrap-double wrap-int wrap-long]]
            [uncomplicate.fluokitten.protocols
             :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative fold]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! subvector]]
             [real :refer [entry entry!]]
             [math :refer [ceil abs]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dense-rows dense-cols dense-dias banded-rows banded-cols banded-dias
                             dragan-says-ex]]
             [printing :refer [print-vector print-ge print-uplo print-banded print-packed]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.host.fluokitten :refer :all])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.internal.api BufferAccessor RealBufferAccessor IntegerBufferAccessor
            Vector Matrix RealVector IntegerVector RealMatrix GEMatrix TRMatrix SYMatrix
            DataAccessor RealChangeable IntegerChangeable RealOrderNavigator UploNavigator StripeNavigator
            DenseMatrix DenseVector Block BandedMatrix BandNavigator DenseStorage RealDefault LayoutNavigator
            Region RealLayoutNavigator BlockMatrix]));;TODO clean up

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

(def ^:private f* (double-fn *))

(defn ^:private require-trf []
  (throw (ex-info "Please do the triangular factorization of this matrix first." {})))

(extend-type DirectByteBuffer
  Releaseable
  (release [this]
    (clean-buffer this)))

;; ================== Declarations ============================================

(declare integer-block-vector)
(declare real-block-vector)
(declare real-ge-matrix)
(declare real-tr-matrix)
(declare real-sy-matrix)
(declare real-banded-matrix)
(declare real-packed-matrix)

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
  (initialize [this b v]
    (let [v (double v)
          strd Float/BYTES]
      (dotimes [i (.count this b)]
        (.putFloat ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-float s))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? FloatBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf offset stride]
    (if (< offset (.count this buf))
      (wrap-byte-seq float32 (* Float/BYTES stride) (* Float/BYTES offset) (byte-seq buf))
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
    Double/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Double/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Double/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v (double v)
          strd Double/BYTES]
      (dotimes [i (.count this b)]
        (.putDouble ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-double s))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? DoubleBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf offset stride]
    (if (< offset (.count this buf))
      (wrap-byte-seq float64 (* Double/BYTES stride) (* Double/BYTES offset) (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

(deftype IntBufferAccessor []
  DataAccessor
  (entryType [_]
    Integer/TYPE)
  (entryWidth [_]
    Integer/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Integer/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Integer/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v (double v)
          strd Integer/BYTES]
      (dotimes [i (.count this b)]
        (.putInt ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-int s))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? IntBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf offset stride]
    (if (< offset (.count this buf))
      (wrap-byte-seq int32 (* Integer/BYTES stride) (* Integer/BYTES offset) (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Integer/BYTES k) (* Integer/BYTES l)))
  IntegerBufferAccessor
  (get [_ buf i]
    (.getInt buf (* Integer/BYTES i)))
  (set [_ buf i val]
    (.putInt buf (* Integer/BYTES i) val)))

(def int-accessor (->IntBufferAccessor))

(deftype LongBufferAccessor []
  DataAccessor
  (entryType [_]
    Long/TYPE)
  (entryWidth [_]
    Long/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Long/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Long/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v (double v)
          strd Long/BYTES]
      (dotimes [i (.count this b)]
        (.putInt ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-long s))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? IntBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf offset stride]
    (if (< offset (.count this buf))
      (wrap-byte-seq int64 (* Long/BYTES stride) (* Long/BYTES offset) (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Long/BYTES k) (* Long/BYTES l)))
  IntegerBufferAccessor
  (get [_ buf i]
    (.getLong buf (* Long/BYTES i)))
  (set [_ buf i val]
    (.putLong buf (* Long/BYTES i) val)))

(def long-accessor (->LongBufferAccessor))

;; ==================== Transfer macros and functions  =============================================

(defmacro ^:private transfer-vector-vector [source destination]
  `(do
     (if (and (compatible? ~source ~destination) (fits? ~source ~destination))
       (copy! ~source ~destination)
       (dotimes [i# (min (.dim ~source) (.dim ~destination))]
         (.set ~destination i# (.entry ~source i#))))
     ~destination))

(defmacro ^:private transfer-vector-array [source destination]
  `(let [n# (min (.dim ~source) (alength ~destination))]
     (dotimes [i# n#]
       (aset ~destination i# (.entry ~source i#)))
     ~destination))

(defmacro ^:private transfer-array-vector [source destination]
  `(let [n# (min (alength ~source) (.dim ~destination))]
     (dotimes [i# n#]
       (.set ~destination i# (aget ~source i#)))
     ~destination))

(defmacro ^:private transfer-seq-vector [source destination]
  `(let [n# (.dim ~destination)]
     (loop [i# 0 src# (seq ~source)]
       (when (and src# (< i# n#))
         (.set ~destination i# (first src#))
         (recur (inc i#) (next src#))))
     ~destination))

(defmacro ^:private transfer-vector-ge [source destination]
  `(let [dim# (.dim ~source)
         sd# (.sd ~destination)
         n# (min (.fd ~destination) (ceil (/ dim# sd#)))
         navigator# (order-navigator ~destination)]
     (dotimes [j# n#]
       (dotimes [i# sd#]
         (let [idx# (+ (* sd# j#) i#)]
           (when (< idx# dim#)
             (.set navigator# ~destination i# j# (.entry ~source idx#))))))
     ~destination))

(defmacro ^:private transfer-ge-vector [source destination]
  `(let [dim# (.dim ~destination)
         sd# (.sd ~source)
         n# (min (.fd ~source) (ceil (/ dim# sd#)))
         navigator# (order-navigator ~source)]
     (dotimes [j# n#]
       (dotimes [i# sd#]
         (let [idx# (+ (* sd# j#) i#)]
           (when (< idx# dim#)
             (.set ~destination idx# (.get navigator# ~source i# j#))))))
     ~destination))

(defmacro ^:private transfer-array-ge [source destination]
  `(let [len# (alength ~source)
         sd# (.sd ~destination)
         n# (min (.fd ~destination) (ceil (/ len# sd#)))
         navigator# (order-navigator ~destination)]
     (dotimes [j# n#]
       (dotimes [i# sd#]
         (let [idx# (+ (* sd# j#) i#)]
           (when (< idx# len#)
             (.set navigator# ~destination i# j# (aget ~source idx#))))))
     ~destination))

(defmacro ^:private transfer-ge-array [source destination]
  `(let [len# (alength ~destination)
         sd# (.sd ~source)
         n# (min (.fd ~source) (ceil (/ len# sd#)))
         navigator# (order-navigator ~source)]
     (dotimes [j# n#]
       (dotimes [i# sd#]
         (let [idx# (+ (* sd# j#) i#)]
           (when (< idx# len#)
             (aset ~destination idx# (.get navigator# ~source i# j#))))))
     ~destination))

(defn ^:private transfer-uplo-uplo [^Matrix source ^Matrix destination]
  (if (and (compatible? source destination) (fits? source destination))
    (copy! source destination)
    (let [navigator (order-navigator source)
          stripe-nav (stripe-navigator source)
          n (min (.ncols source) (.ncols destination))]
      (dotimes [j n]
        (let [start (.start stripe-nav n j)]
          (dotimes [i (- (.end stripe-nav n j) start)]
            (.set navigator destination (+ start i) j (.get navigator source (+ start i) j)))))
      destination)))

(defn ^:private transfer-seq-uplo [source ^Matrix destination]
  (let [n (.ncols destination)
        navigator (order-navigator destination)
        stripe-nav (stripe-navigator destination)]
    (loop [j 0 src (seq source)]
      (if (and src (< j n))
        (recur (inc j)
               (let [end (.end stripe-nav n j)]
                 (loop [i (.start stripe-nav n j) src src]
                   (if (and src (< i end))
                     (do (.set navigator destination i j (first src))
                         (recur (inc i) (next src)))
                     src))))
        destination))))

(defmacro ^:private transfer-array-uplo [source destination]
  `(let [len# (alength ~source)
         n# (.ncols ~destination)
         navigator# (order-navigator ~destination)
         stripe-nav# (stripe-navigator ~destination)]
     (loop [j# 0 cnt# 0]
       (if (and (< cnt# len#) (< j# n#))
         (recur (inc j#)
                (long (let [end# (.end stripe-nav# n# j#)]
                        (loop [i# (.start stripe-nav# n# j#) idx# cnt#]
                          (if (and (< idx# len#) (< i# end#))
                            (do (.set navigator# ~destination i# j# (aget ~source idx#))
                                (recur (inc i#) (inc idx#)))
                            idx#)))))
         ~destination))))

(defmacro ^:private transfer-uplo-array [source destination]
  `(let [len# (alength ~destination)
         n# (.ncols ~source)
         navigator# (order-navigator ~source)
         stripe-nav# (stripe-navigator ~source)]
     (loop [j# 0 cnt# 0]
       (if (and (< cnt# len#) (< j# n#))
         (recur (inc j#)
                (long (let [end# (.end stripe-nav# n# j#)]
                        (loop [i# (.start stripe-nav# n# j#) idx# cnt#]
                          (if (and (< idx# len#) (< i# end#))
                            (do (aset ~destination idx# (.get navigator# ~source i# j#))
                                (recur (inc i#) (inc idx#)))
                            idx#)))))
         ~destination))))

;; ============ Integer Vector =================================================

(deftype IntegerBlockVector [fact ^IntegerBufferAccessor da eng ^Boolean master ^ByteBuffer buf
                             ^long n ^long ofst ^long strd]
  Object
  (hashCode [x]
    (-> (hash :IntegerBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? IntegerBlockVector y) (compatible? da y) (fits? x y))
      (loop [i 0]
        (if (< i n)
          (and (= (.entry x i) (.entry ^IntegerBlockVector y i)) (recur (inc i)))
          true))
      :default false))
  (toString [_]
    (format "#IntegerBlockVector[%s, n:%d, offset: %d, stride:%d]" (.entryType da) n ofst strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Seqable
  (seq [_]
    (take n (.toSeq da buf ofst strd)))
  Container
  (raw [_]
    (integer-block-vector fact n))
  (raw [_ fact]
    (create-vector fact n false))
  (zero [_]
    (integer-block-vector fact n))
  (zero [_ fact]
    (create-vector fact n true))
  (host [x]
    (let-release [res (raw x)]
      (copy eng x res)))
  (native [x]
    x)
  MemoryContext
  (fully-packed? [_]
    (= 1 strd))
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^DenseVector y)))
  EngineProvider
  (engine [_]
    nil)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  DenseVector
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (count [_]
    n)
  IFn$LLL
  (invokePrim [x i v]
    (.set x i v))
  IFn$LL
  (invokePrim [x i]
    (.entry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.set x i v))
  (invoke [x i]
    (.entry x i))
  (invoke [x]
    n)
  IntegerChangeable
  (set [x val]
    (set-all eng val x)
    x)
  (set [x i val]
    (.set da buf (+ ofst (* strd i)) val)
    x)
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x f]
    (if (instance? IFn$LL f)
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LL f (.entry x i))))
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LLL f i (.entry x i)))))
    x)
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$LL f (.entry x i))))
  IntegerVector
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf (+ ofst (* strd i))))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (integer-block-vector fact false buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (integer-block-vector fact 0)))

(defn integer-block-vector
  ([fact master ^ByteBuffer buf n ofst strd]
   (->IntegerBlockVector fact (data-accessor fact) (vector-engine fact) master buf n ofst strd))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (integer-block-vector fact true buf n 0 1))))

(defmethod print-method IntegerBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take 100 (seq x))))))

(defmethod transfer! [IntegerBlockVector IntegerBlockVector]
  [^IntegerBlockVector source ^IntegerBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [clojure.lang.Sequential IntegerBlockVector]
  [source ^IntegerBlockVector destination]
  (transfer-seq-vector source destination))

(defmethod transfer! [(Class/forName "[D") IntegerBlockVector]
  [^doubles source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[F") IntegerBlockVector]
  [^floats source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[J") IntegerBlockVector]
  [^longs source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[I") IntegerBlockVector]
  [^ints source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[J")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[I")]
  [^IntegerBlockVector source ^ints destination]
  (transfer-vector-array source destination))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [fact ^RealBufferAccessor da eng ^Boolean master ^ByteBuffer buf
                          ^long n ^long ofst ^long strd]
  Object
  (hashCode [x]
    (-> (hash :RealBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? RealBlockVector y) (compatible? da y) (fits? x y))
      (loop [i 0]
        (if (< i n)
          (and (= (.entry x i) (.entry ^RealBlockVector y i)) (recur (inc i)))
          true))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, offset: %d, stride:%d]" (.entryType da) n ofst strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Seqable
  (seq [_]
    (take n (.toSeq da buf ofst strd)))
  Container
  (raw [_]
    (real-block-vector fact n))
  (raw [_ fact]
    (create-vector fact n false))
  (zero [_]
    (real-block-vector fact n))
  (zero [_ fact]
    (create-vector fact n true))
  (host [x]
    (let-release [res (raw x)]
      (copy eng x res)))
  (native [x]
    x)
  DenseContainer
  (view-vctr [_]
    (real-block-vector fact false buf n ofst strd))
  (view-vctr [_ stride-mult]
    (real-block-vector fact false buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (real-ge-matrix fact false buf n 1 ofst n COLUMN_MAJOR))
  (view-ge [x stride-mult]
    (view-ge (view-ge x) stride-mult))
  (view-tr [x uplo diag]
    (view-tr (view-ge x) uplo diag))
  (view-sy [x uplo]
    (view-sy (view-ge x) uplo))
  MemoryContext
  (fully-packed? [_]
    (= 1 strd))
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^DenseVector y)))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  DenseVector
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (count [_]
    n)
  IFn$LDD
  (invokePrim [x i v]
    (.set x i v))
  IFn$LD
  (invokePrim [x i]
    (entry x i))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.set x i v))
  (invoke [x i]
    (entry x i))
  (invoke [x]
    n)
  RealChangeable
  (set [x val]
    (if (not (Double/isNaN val))
      (set-all eng val x)
      (dotimes [i n]
        (.set x i val)))
    x)
  (set [x i val]
    (.set da buf (+ ofst (* strd i)) val)
    x)
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x f]
    (if (instance? IFn$DD f)
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
      (dotimes [i n]
        (.set x i (.invokePrim ^IFn$LDD f i (.entry x i)))))
    x)
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf (+ ofst (* strd i))))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (real-block-vector fact false buf l (+ ofst (* k strd)) strd))
  Monoid
  (id [x]
    (real-block-vector fact 0))
  PseudoFunctor
  (fmap! [x f]
    (vector-fmap ^IFn$DD f x))
  (fmap! [x f y]
    (vector-fmap ^IFn$DDD f x ^RealVector y))
  (fmap! [x f y z]
    (vector-fmap ^IFn$DDDD f x ^RealVector y ^RealVector z))
  (fmap! [x f y z v]
    (vector-fmap ^IFn$DDDDD f x ^RealVector y ^RealVector z ^RealVector v))
  (fmap! [x f y z v ws]
    (vector-fmap f x y z v ws))
  Foldable
  (fold [x]
    (sum eng x))
  (fold [x f init]
    (vector-fold f init x))
  (fold [x f init y]
    (vector-fold f init x ^RealVector y))
  (fold [x f init y z]
    (vector-fold f init x ^RealVector y ^RealVector z))
  (fold [x f init y z v]
    (vector-fold f init x ^RealVector y ^RealVector z ^RealVector v))
  (fold [x f init y z v ws]
    (vector-fold f init x y z v ws))
  (foldmap [x g]
    (loop [i 0 acc 0.0]
      (if (< i n)
        (recur (inc i) (+ acc (.invokePrim ^IFn$DD g (.entry x i))))
        acc)))
  (foldmap [x g f init]
    (vector-foldmap f init ^IFn$DD g x))
  (foldmap [x g f init y]
    (vector-foldmap f init ^IFn$DDD g x ^RealVector y))
  (foldmap [x g f init y z]
    (vector-foldmap f int ^IFn$DDDD g x ^RealVector y ^RealVector z))
  (foldmap [x g f init y z v]
    (vector-foldmap f init ^IFn$DDDDD g x ^RealVector y ^RealVector z ^RealVector v))
  (foldmap [x g f init y z v ws]
    (vector-foldmap f init g x y z v ws)))()

(defn real-block-vector
  ([fact master ^ByteBuffer buf n ofst strd]
   (let [da (data-accessor fact)]
     (if (and (<= 0 n (.count da buf)))
       (->RealBlockVector fact da (vector-engine fact) master buf n ofst strd)
       (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf)})))))
  ([fact n]
   (let-release [buf (.createDataSource (data-accessor fact) n)]
     (real-block-vector fact true buf n 0 1))))

(extend RealBlockVector
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure vector-pure}
  Magma
  {:op (constantly vector-op)})

(defmethod print-method RealBlockVector [^Vector x ^java.io.Writer w]
  (.write w (str x))
  (print-vector w x))

(defmethod transfer! [RealBlockVector RealBlockVector]
  [^RealBlockVector source ^RealBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [IntegerBlockVector RealBlockVector]
  [^IntegerBlockVector source ^RealBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [RealBlockVector IntegerBlockVector]
  [^RealBlockVector source ^IntegerBlockVector destination]
  (transfer-vector-vector source destination))

(defmethod transfer! [clojure.lang.Sequential RealBlockVector]
  [source ^RealBlockVector destination]
  (transfer-seq-vector source destination))

(defmethod transfer! [(Class/forName "[D") RealBlockVector]
  [^doubles source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[F") RealBlockVector]
  [^floats source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[J") RealBlockVector]
  [^longs source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[I") RealBlockVector]
  [^ints source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[D")]
  [^RealBlockVector source ^doubles destination]
  (transfer-vector-array source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[F")]
  [^RealBlockVector source ^floats destination]
  (transfer-vector-array source destination))

;; =================== Real Matrix =============================================

(deftype RealGEMatrix [^RealOrderNavigator navigator fact ^RealBufferAccessor da eng ^Boolean master
                       ^ByteBuffer buf ^long m ^long n ^long ofst ^long ld ^long sd ^long fd ^long ord]
  Object
  (hashCode [a]
    (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n)
        (hash-combine (nrm2 eng (.stripe navigator a 0)))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealGEMatrix b) (compatible? da b) (fits? a b))
      (loop [j 0]
        (if (< j fd)
          (and (loop [i 0]
                 (if (< i sd)
                   (and (= (.get da buf (+ ofst (* ld j) i)) (.get navigator b i j))
                        (recur (inc i)))
                   true))
               (recur (inc j)))
          true))
      :default false))
  (toString [a]
    (format "#RealGEMatrix[%s, mxn:%dx%d, order%s, offset:%d, ld:%d]"
            (.entryType da) m n (dec-property ord) ofst ld))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Seqable
  (seq [a]
    (map #(seq (.stripe navigator a %)) (range 0 fd)))
  Container
  (raw [_]
    (real-ge-matrix fact m n ord))
  (raw [_ fact]
    (create-ge fact m n ord false))
  (zero [_]
    (real-ge-matrix fact m n ord))
  (zero [_ fact]
    (create-ge fact m n ord true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  DenseContainer
  (view-vctr [_]
    (if (= ld sd)
      (real-block-vector fact false buf (* m n) ofst 1)
      (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:ld ld :sd sd}))))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [_]
    (real-ge-matrix fact false buf m n ofst ld ord))
  (view-ge [_ stride-mult]
    (let [shrinked (ceil (/ fd (long stride-mult)))]
      (real-ge-matrix fact false buf (.sd navigator sd shrinked) (.fd navigator sd shrinked)
                      ofst (* ld (long stride-mult)) ord)))
  (view-tr [_ uplo diag]
    (real-tr-matrix fact false buf (min m n) ofst ld ord uplo diag))
  (view-sy [_ uplo]
    (real-sy-matrix fact false buf (min m n) ofst ld ord uplo))
  Navigable
  (order-navigator [_]
    navigator)
  MemoryContext
  (fully-packed? [_]
    (= sd ld))
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= m (.mrows ^GEMatrix b)) (= n (.ncols ^GEMatrix b))))
  (fits-navigation? [_ b]
    (= ord (.order ^DenseMatrix b)))
  GEMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (order [_]
    ord)
  (count [_]
    (* m n))
  (sd [_]
    sd)
  (fd [_]
    fd)
  IFn$LLDD
  (invokePrim [a i j v]
    (entry! a i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [a i j v]
    (entry! a i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    true)
  (set [a val]
    (if (not (Double/isNaN val))
      (set-all eng val a)
      (dotimes [j fd]
        (dotimes [i sd]
          (.set da buf (+ ofst (* ld j) i) val))))
    a)
  (set [a i j val]
    (.set da buf (.index navigator ofst ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (dotimes [j fd]
        (dotimes [i sd]
          (let [idx (+ ofst (* ld j) i)]
            (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))))
      (dotimes [j fd]
        (dotimes [i sd]
          (let [idx (+ ofst (* ld j) i)]
            (.set da buf idx (.invokePrimitive navigator f i j (.get da buf idx)))))))
    a)
  (alter [a i j f]
    (let [idx (.index navigator ofst ld i j)]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))
    a)
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get da buf (.index navigator ofst ld i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (real-block-vector fact false buf n (.index navigator ofst ld i 0) (if (= ROW_MAJOR ord) 1 ld)))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (real-block-vector fact false buf m (.index navigator ofst ld 0 j) (if (= COLUMN_MAJOR ord) 1 ld)))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf (min m n) ofst (inc ld)))
  (dia [a k]
    (let [dia-dim (if (< 0 k) (min m (- n k)) (min (+ m k) n))]
      (real-block-vector fact false buf dia-dim (.index navigator ofst ld k) (inc ld))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (real-ge-matrix fact false buf k l (.index navigator ofst ld i j) ld ord))
  (transpose [a]
    (real-ge-matrix fact false buf n m ofst ld (flip-layout ord)))
  Monoid
  (id [a]
    (real-ge-matrix fact 0 0))
  TRF
  (trtrs [_ _]
    (require-trf))
  (trtri! [_]
    (require-trf))
  (trtri [_]
    (require-trf))
  (trcon [_ _ _]
    (require-trf))
  (trcon [_ _]
    (require-trf))
  (trdet [_]
    (require-trf))
  PseudoFunctor
  (fmap! [a f]
    (ge-fmap navigator fd sd ^IFn$DD f a))
  (fmap! [a f b]
    (ge-fmap navigator fd sd ^IFn$DDD f a b))
  (fmap! [a f b c]
    (ge-fmap navigator fd sd ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (ge-fmap navigator fd sd ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (ge-fmap navigator fd sd f a b c d es))
  Foldable
  (fold [_]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i) (+ acc (.get da buf (+ ofst (* ld j) i))))
                    acc))))
        acc)))
  (fold [a f init]
    (matrix-fold navigator fd f init a))
  (fold [a f init b]
    (matrix-fold navigator fd f init a b))
  (fold [a f init b c]
    (matrix-fold navigator fd f init a b c))
  (fold [a f init b c d]
    (matrix-fold navigator fd f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold navigator fd f init a b c d es))
  (foldmap [_ g]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i)
                           (+ acc (.invokePrim ^IFn$DD g (.get da buf (+ ofst (* ld j) i)))))
                    acc))))
        acc)))
  (foldmap [a g f init]
    (matrix-foldmap navigator fd f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap navigator fd f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap navigator fd f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap navigator fd f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap navigator fd f init g a b c d es)))

(defn real-ge-matrix
  ([fact master buf m n ofst ld ord]
   (let [^RealOrderNavigator navigator (if (= COLUMN_MAJOR ord) col-navigator row-navigator)]
     (->RealGEMatrix navigator fact (data-accessor fact) (ge-engine fact)
                     master buf m n ofst (max (long ld) (.sd navigator m n))
                     (.sd navigator m n) (.fd navigator m n) ord)))
  ([fact ^long m ^long n ord]
   (let-release [buf (.createDataSource (data-accessor fact) (* m n))]
     (real-ge-matrix fact true buf m n 0 0 ord)))
  ([fact ^long m ^long n]
   (real-ge-matrix fact m n DEFAULT_ORDER)))

(extend RealGEMatrix
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defmethod print-method RealGEMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-ge w a))

(defmethod transfer! [RealGEMatrix RealGEMatrix]
  [^RealGEMatrix source ^RealGEMatrix destination]
  (if (and (compatible? source destination) (fits? source destination))
    (copy! source destination)
    (let [navigator (order-navigator source)
          sd (min (.sd source) (.sd navigator (.mrows destination) (.ncols destination)))
          fd (min (.fd source) (.fd navigator (.mrows destination) (.ncols destination)))]
      (dotimes [j fd]
        (dotimes [i sd]
          (.set navigator destination i j (.get navigator source i j))))
      destination)))

(defmethod transfer! [clojure.lang.Sequential RealGEMatrix]
  [source ^RealGEMatrix destination]
  (let [sd (.sd destination)
        fd (.fd destination)
        navigator (order-navigator destination)]
    (loop [j 0 src source]
      (if (and src (< j fd))
        (recur (inc j)
               (loop [i 0 src src]
                 (if (and src (< i sd))
                   (do (.set navigator destination i j (first src))
                       (recur (inc i) (next src)))
                   src)))
        destination))))

(defmethod transfer! [RealBlockVector RealGEMatrix]
  [^RealBlockVector source ^RealGEMatrix destination]
  (let [m (.mrows destination)
        n (.ncols destination)]
    (if (and (compatible? (data-accessor source) destination) (<= (* m n) (.dim source)))
      (copy! (real-ge-matrix (factory source) false (.buffer source)
                             m n (.offset source) 0 (.order destination))
             destination)
      (transfer-vector-ge source destination))))

(defmethod transfer! [RealGEMatrix RealBlockVector]
  [^RealGEMatrix source ^RealBlockVector destination]
  (let [m (.mrows source)
        n (.ncols source)]
    (if (and (compatible? (data-accessor destination) source) (<= (* m n) (.dim destination)))
      (copy! source (real-ge-matrix (factory destination) false (.buffer destination)
                                    m n (.offset destination) 0 (.order source)))
      (transfer-ge-vector source destination))
    destination))

(defmethod transfer! [IntegerBlockVector RealGEMatrix]
  [^IntegerBlockVector source ^RealGEMatrix destination]
  (transfer-vector-ge source destination))

(defmethod transfer! [RealGEMatrix IntegerBlockVector]
  [^RealGEMatrix source ^IntegerBlockVector destination]
  (transfer-ge-vector source destination))

(defmethod transfer! [(Class/forName "[D") RealGEMatrix]
  [^doubles source ^RealGEMatrix destination]
  (transfer-array-ge source destination))

(defmethod transfer! [(Class/forName "[F") RealGEMatrix]
  [^floats source ^RealGEMatrix destination]
  (transfer-array-ge source destination))

(defmethod transfer! [(Class/forName "[J") RealGEMatrix]
  [^longs source ^RealGEMatrix destination]
  (transfer-array-ge source destination))

(defmethod transfer! [(Class/forName "[I") RealGEMatrix]
  [^ints source ^RealGEMatrix destination]
  (transfer-array-ge source destination))

(defmethod transfer! [RealGEMatrix (Class/forName "[D")]
  [^RealGEMatrix source ^doubles destination]
  (transfer-ge-array source destination))

(defmethod transfer! [RealGEMatrix (Class/forName "[F")]
  [^RealGEMatrix source ^floats destination]
  (transfer-ge-array source destination))

;; ====================== Generic triangular functions =============================

(defn ^:private uplo-equals [^Block a b type]
  (let [n (.ncols ^Matrix a)
        navigator (order-navigator a)
        stripe-nav (stripe-navigator a)
        da (data-accessor a)
        buf (.buffer a)
        ofst (.offset a)
        ld (.stride a)]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? type b) (compatible? da b) (fits? a b))
      (loop [j 0]
        (if (< j n)
          (let [end (.end stripe-nav n j)]
            (and (loop [i (.start stripe-nav n j)]
                   (if (< i end)
                     (and (= (.get ^RealBufferAccessor da buf (+ ofst (* ld j) i)) (.get navigator b i j))
                          (recur (inc i)))
                     true))
                 (recur (inc j))))
          true))
      :default false)))

(defn ^:private uplo-set [^Block a ^double val]
  (let [n (.ncols ^Matrix a)
        eng (engine a)
        stripe-nav (stripe-navigator a)
        da ^RealBufferAccessor (data-accessor a)
        buf (.buffer a)
        ofst (.offset a)
        ld (.stride a)]
    (if (not (Double/isNaN val))
      (set-all eng val a)
      (dotimes [j n]
        (let [start (.start stripe-nav n j)
              end (.end stripe-nav n j)]
          (dotimes [i (- end start)]
            (.set da buf (+ ofst (* ld j) (+ start i)) val)))))
    a))

(defn ^:private uplo-alter [^Block a f]
  (let [n (.ncols ^Matrix a)
        navigator (order-navigator a)
        stripe-nav (stripe-navigator a)
        da ^RealBufferAccessor (data-accessor a)
        buf (.buffer a)
        ofst (.offset a)
        ld (.stride a)]
    (if (instance? IFn$DD f)
      (dotimes [j n]
        (let [start (.start stripe-nav n j)
              end (.end stripe-nav n j)]
          (dotimes [i (- end start)]
            (let [idx (+ ofst (* ld j) (+ start i))]
              (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))))))
      (dotimes [j n]
        (let [start (.start stripe-nav n j)
              end (.end stripe-nav n j)]
          (dotimes [i (- end start)]
            (let [idx (+ ofst (* ld j) (+ start i))]
              (.set da buf idx (.invokePrimitive navigator f (+ start i) j (.get da buf idx))))))))
    a))

;; =================== Real Triangular Matrix ==================================

(deftype RealTRMatrix [^RealOrderNavigator navigator ^UploNavigator uplo-nav ^StripeNavigator stripe-nav
                       fact ^RealBufferAccessor da eng ^Boolean master ^ByteBuffer buf
                       ^long n ^long ofst ^long ld ^long ord ^long fuplo ^long fdiag]
  Object
  (hashCode [a]
    (-> (hash :RealTRMatrix) (hash-combine n) (hash-combine (nrm2 eng (.stripe navigator a 0)))))
  (equals [a b]
    (uplo-equals a b TRMatrix))
  (toString [a]
    (format "#RealTRMatrix[%s, mxn:%dx%d, order%s, uplo%s, diag%s, offset:%d, ld:%d]"
            (.entryType da) n n (dec-property ord) (dec-property fuplo) (dec-property fdiag) ofst ld ))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (real-tr-matrix fact n ord fuplo fdiag))
  (raw [_ fact]
    (create-tr fact n ord fuplo fdiag false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-tr fact n ord fuplo fdiag true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (real-ge-matrix fact false buf n n ofst ld ord))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-tr [_ uplo diag]
    (real-tr-matrix fact false buf n ofst ld ord uplo diag))
  (view-sy [_ uplo]
    (real-sy-matrix fact false buf n ofst ld ord uplo))
  Navigable
  (order-navigator [_]
    navigator)
  (stripe-navigator [_]
    stripe-nav)
  (uplo-navigator [_]
    uplo-nav)
  MemoryContext
  (fully-packed? [_]
    false)
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= n (.mrows ^TRMatrix b)) (= fuplo (.uplo ^TRMatrix b)) (= fdiag (.diag ^TRMatrix b))))
  (fits-navigation? [_ b]
    (and (= ord (.order ^DenseMatrix b))
         (if (instance? TRMatrix b)
           (and (= fuplo (.uplo ^TRMatrix b)) (= fdiag (.diag ^TRMatrix b)))
           true)))
  Monoid
  (id [a]
    (real-tr-matrix fact 0))
  TRMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (count [_]
    (* n n))
  (uplo [_]
    fuplo)
  (diag [_]
    fdiag)
  (order [_]
    ord)
  (sd [_]
    n)
  (fd [_]
    n)
  Seqable
  (seq [a]
    (map #(seq (.stripe navigator a %)) (range 0 n)))
  IFn$LLDD
  (invokePrim [x i j v]
    (entry! x i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [x i j v]
    (entry! x i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (= 2 (.defaultEntry uplo-nav i j)))
  (set [a val]
    (uplo-set a val))
  (set [a i j val]
    (.set da buf (.index navigator ofst ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (uplo-alter a f))
  (alter [a i j f]
    (let [idx (.index navigator ofst ld i j)]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))
    a)
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (let [res (.defaultEntry uplo-nav i j)]
      (if (= 2 res)
        (.get da buf (.index navigator ofst ld i j))
        res)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart uplo-nav n i)]
      (real-block-vector fact false buf (- (.rowEnd uplo-nav n i) start)
                         (.index navigator ofst ld i start) (if (= ROW_MAJOR ord) 1 ld))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart uplo-nav n j)]
      (real-block-vector fact false buf (- (.colEnd uplo-nav n j) start)
                         (.index navigator ofst ld start j) (if (= COLUMN_MAJOR ord) 1 ld))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf (.diaDim uplo-nav n 0) ofst (inc ld)))
  (dia [a k]
    (real-block-vector fact false buf (.diaDim uplo-nav n k) (.index navigator ofst ld k) (inc ld)))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (real-tr-matrix fact false buf k (.index navigator ofst ld i j) ld ord fuplo fdiag)
      (throw (ex-info "You cannot use regions outside the triangle in TR submatrix"
                      {:a (str a) :i i :j j :k k :l l}))))
  (transpose [a]
    (real-tr-matrix fact false buf n ofst ld (flip-layout ord) (flip-uplo fuplo) fdiag))
  TRF
  (trtrs [a b]
    (trs eng a b))
  (trtri! [a]
    (tri eng a))
  (trtri [a]
    (let-release [res (raw a)]
      (tri eng (copy eng a res))))
  (trcon [a _ nrm1?]
    (con eng a nrm1?))
  (trcon [a nrm1?]
    (con eng a nrm1?))
  (trdet [a]
    (if (= DIAG_NON_UNIT fdiag)
      (fold (.diag a) f* 1.0)
      1.0))
  PseudoFunctor
  (fmap! [a f]
    (uplo-fmap navigator stripe-nav n ^IFn$DD f a))
  (fmap! [a f b]
    (uplo-fmap navigator stripe-nav n ^IFn$DDD f a b))
  (fmap! [a f b c]
    (uplo-fmap navigator stripe-nav n ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (uplo-fmap navigator stripe-nav n ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (uplo-fmap navigator stripe-nav n f a b c d nil))
  Foldable
  (fold [a]
    (loop [j 0 acc 0.0]
      (if (< j n)
        (recur (inc j)
               (double
                (let [end (.end stripe-nav n j)]
                  (loop [i (.start stripe-nav n j) acc acc]
                    (if (< i end)
                      (recur (inc i) (+ acc (.get da buf (+ ofst (* ld j) i))))
                      acc)))))
        acc)))
  (fold [a f init]
    (matrix-fold navigator n f init a))
  (fold [a f init b]
    (matrix-fold navigator n f init a b))
  (fold [a f init b c]
    (matrix-fold navigator n f init a b c))
  (fold [a f init b c d]
    (matrix-fold navigator n f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold navigator n f init a b c d es))
  (foldmap [a g]
    (uplo-foldmap a g))
  (foldmap [a g f init]
    (matrix-foldmap navigator n f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap navigator n f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap navigator n f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap navigator n f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap navigator n f init g a b c d es)))

(extend RealTRMatrix
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defn real-tr-matrix
  ([fact master ^ByteBuffer buf n ofst ld ord uplo diag]
   (let [unit (= DIAG_UNIT diag)
         lower (= LOWER uplo)
         column (= COLUMN_MAJOR ord)
         bottom (if lower column (not column))
         order-nav (if column col-navigator row-navigator)
         uplo-nav (if lower
                    (if unit unit-lower-nav non-unit-lower-nav)
                    (if unit unit-upper-nav non-unit-upper-nav))
         stripe-nav (if bottom
                      (if unit unit-bottom-navigator non-unit-bottom-navigator)
                      (if unit unit-top-navigator non-unit-top-navigator))]
     (->RealTRMatrix order-nav uplo-nav stripe-nav fact (data-accessor fact) (tr-engine fact)
                     master buf n ofst (max (long ld) (long n)) ord uplo diag)))
  ([fact n ord uplo diag]
   (let-release [buf (.createDataSource (data-accessor fact) (* (long n) (long n)))]
     (real-tr-matrix fact true buf n 0 n ord uplo diag)))
  ([fact n]
   (real-tr-matrix fact n DEFAULT_ORDER DEFAULT_UPLO DEFAULT_DIAG)))

(defmethod print-method RealTRMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-uplo w a))

(defmethod transfer! [RealTRMatrix RealTRMatrix]
  [source destination]
  (transfer-uplo-uplo source destination))

(defmethod transfer! [clojure.lang.Sequential RealTRMatrix]
  [source destination]
  (transfer-seq-uplo source destination))

(defmethod transfer! [(Class/forName "[D") RealTRMatrix]
  [^doubles source ^RealTRMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[F") RealTRMatrix]
  [^floats source ^RealTRMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[J") RealTRMatrix]
  [^longs source ^RealTRMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[I") RealTRMatrix]
  [^ints source ^RealTRMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [RealTRMatrix (Class/forName "[D")]
  [^RealTRMatrix source ^doubles destination]
  (transfer-uplo-array source destination))

(defmethod transfer! [RealTRMatrix (Class/forName "[F")]
  [^RealTRMatrix source ^floats destination]
  (transfer-uplo-array source destination))

;; =================== Real Symmetric Matrix ==================================

(deftype RealSYMatrix [^RealOrderNavigator navigator ^UploNavigator uplo-nav ^StripeNavigator stripe-nav
                       fact ^RealBufferAccessor da eng ^Boolean master ^ByteBuffer buf
                       ^long n ^long ofst ^long ld ^long ord ^long fuplo]
  Object
  (hashCode [a]
    (-> (hash :RealSYMatrix) (hash-combine n) (hash-combine (nrm2 eng (.stripe navigator a 0)))))
  (equals [a b]
    (uplo-equals a b SYMatrix))
  (toString [a]
    (format "#RealSYMatrix[%s, mxn:%dx%d, order%s, uplo%s, offset:%d, ld:%d]"
            (.entryType da) n n (dec-property ord) (dec-property fuplo) ofst ld ))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (real-sy-matrix fact n ord fuplo))
  (raw [_ fact]
    (create-sy fact n ord fuplo false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-sy fact n ord fuplo true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (real-ge-matrix fact false buf n n ofst ld ord))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-tr [_ uplo diag]
    (real-tr-matrix fact false buf n ofst ld ord uplo diag))
  (view-sy [_ uplo]
    (real-sy-matrix fact false buf n ofst ld ord uplo))
  Navigable
  (order-navigator [_]
    navigator)
  (stripe-navigator [_]
    stripe-nav)
  (uplo-navigator [_]
    uplo-nav)
  MemoryContext
  (fully-packed? [_]
    false)
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= n (.mrows ^SYMatrix b)) (= fuplo (.uplo ^SYMatrix b))))
  (fits-navigation? [_ b]
    (and (= ord (.order ^DenseMatrix b))
         (if (instance? SYMatrix b)
           (= fuplo (.uplo ^SYMatrix b))
           true)))
  Monoid
  (id [a]
    (real-sy-matrix fact 0))
  SYMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (count [_]
    (* n n))
  (uplo [_]
    fuplo)
  (order [_]
    ord)
  (sd [_]
    n)
  (fd [_]
    n)
  Seqable
  (seq [a]
    (map #(seq (.stripe navigator a %)) (range 0 n)))
  IFn$LLDD
  (invokePrim [x i j v]
    (entry! x i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [x i j v]
    (entry! x i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (= 2 (.defaultEntry uplo-nav i j)))
  (set [a val]
    (uplo-set a val))
  (set [a i j val]
    (.set da buf (.index navigator ofst ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (uplo-alter a f))
  (alter [a i j f]
    (let [idx (if (= 2 (.defaultEntry uplo-nav i j))
                (.index navigator ofst ld i j)
                (.index navigator ofst ld j i))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))
    a)
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (let [res (.defaultEntry uplo-nav i j)]
      (if (= 2 res)
        (.get da buf (.index navigator ofst ld i j))
        (.get da buf (.index navigator ofst ld j i)))))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart uplo-nav n i)]
      (real-block-vector fact false buf (- (.rowEnd uplo-nav n i) start)
                         (.index navigator ofst ld i start) (if (= ROW_MAJOR ord) 1 ld))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart uplo-nav n j)]
      (real-block-vector fact false buf (- (.colEnd uplo-nav n j) start)
                         (.index navigator ofst ld start j) (if (= COLUMN_MAJOR ord) 1 ld))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf n ofst (inc ld)))
  (dia [a k]
    (real-block-vector fact false buf (- n (long (abs k))) (.index navigator ofst ld k) (inc ld)))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (real-sy-matrix fact false buf k (.index navigator ofst ld i j) ld ord fuplo)
      (throw (ex-info "You cannot use regions that are not diagonal-centered in SY submatrix."
                      {:a (str a) :i i :j j :k k :l l}))))
  (transpose [a]
    (real-sy-matrix fact false buf n ofst ld (flip-layout ord) (flip-uplo fuplo)))
  ;; TODO LU is different a bit. It's probably LDL/UDU
  PseudoFunctor
  (fmap! [a f]
    (uplo-fmap navigator stripe-nav n ^IFn$DD f a))
  (fmap! [a f b]
    (uplo-fmap navigator stripe-nav n ^IFn$DDD f a b))
  (fmap! [a f b c]
    (uplo-fmap navigator stripe-nav n ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (uplo-fmap navigator stripe-nav n ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (uplo-fmap navigator stripe-nav n f a b c d nil))
  Foldable
  (fold [a]
    (uplo-fold a))
  (fold [a f init]
    (matrix-fold navigator n f init a))
  (fold [a f init b]
    (matrix-fold navigator n f init a b))
  (fold [a f init b c]
    (matrix-fold navigator n f init a b c))
  (fold [a f init b c d]
    (matrix-fold navigator n f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold navigator n f init a b c d es))
  (foldmap [a g]
    (uplo-foldmap a g))
  (foldmap [a g f init]
    (matrix-foldmap navigator n f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap navigator n f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap navigator n f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap navigator n f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap navigator n f init g a b c d es)))

(extend RealSYMatrix
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defn real-sy-matrix
  ([fact master ^ByteBuffer buf n ofst ld ord uplo]
   (let [lower (= LOWER uplo)
         column (= COLUMN_MAJOR ord)
         bottom (if lower column (not column))
         order-nav (if column col-navigator row-navigator)
         uplo-nav (if lower non-unit-lower-nav non-unit-upper-nav)
         stripe-nav (if bottom non-unit-bottom-navigator non-unit-top-navigator)]
     (->RealSYMatrix order-nav uplo-nav stripe-nav fact (data-accessor fact) (sy-engine fact)
                     master buf n ofst (max (long ld) (long n)) ord uplo)))
  ([fact n ord uplo]
   (let-release [buf (.createDataSource (data-accessor fact) (* (long n) (long n)))]
     (real-sy-matrix fact true buf n 0 n ord uplo)))
  ([fact n]
   (real-sy-matrix fact n DEFAULT_ORDER DEFAULT_UPLO)))

(defmethod print-method RealSYMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-uplo w a))

(defmethod transfer! [RealSYMatrix RealSYMatrix]
  [source destination]
  (transfer-uplo-uplo source destination))

(defmethod transfer! [clojure.lang.Sequential RealSYMatrix]
  [source destination]
  (transfer-seq-uplo source destination))

(defmethod transfer! [(Class/forName "[D") RealSYMatrix]
  [^doubles source ^RealSYMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[F") RealSYMatrix]
  [^floats source ^RealSYMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[J") RealSYMatrix]
  [^longs source ^RealSYMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [(Class/forName "[I") RealSYMatrix]
  [^ints source ^RealSYMatrix destination]
  (transfer-array-uplo source destination))

(defmethod transfer! [RealSYMatrix (Class/forName "[D")]
  [^RealSYMatrix source ^doubles destination]
  (transfer-uplo-array source destination))

(defmethod transfer! [RealSYMatrix (Class/forName "[F")]
  [^RealSYMatrix source ^floats destination]
  (transfer-uplo-array source destination))

;; ================= Banded Matrix ==============================================================

(defmacro ^:private banded-doall
  ([band-nav da m n kl ku ofst ld i j idx expr];;TODO da is not used!
   `(let [kd# (.kd ~band-nav ~kl ~ku)]
      (dotimes [~j (.fd ~band-nav ~m ~n ~kl ~ku)]
        (let [start# (.start ~band-nav ~kl ~ku ~j)
              end# (.end ~band-nav ~m ~n ~kl ~ku ~j)]
          (dotimes [i# (- end# start#)]
            (let [~i (+ start# i#)
                  ~idx (+ ~ofst (* ~j ~ld) (- kd# ~j) ~i)]
              ~expr))))))
  ([navigator a i j expr]
   `(let [band-nav# (band-navigator ~a)
          kl# (.kl ~a)
          ku# (.ku ~a)
          m# (.mrows ~a)
          n# (.ncols ~a)
          kd# (.kd band-nav# kl# ku#)]
      (dotimes [~j (.fd band-nav# m# n# kl# ku#)]
        (let [start# (.start band-nav# kl# ku# ~j)
              end# (.end band-nav# m# n# kl# ku# ~j)]
          (dotimes [i# (- end# start#)]
            (let [~i (+ start# i#)]
              ~expr)))))))

(deftype RealBandedMatrix [^RealOrderNavigator navigator ^BandNavigator band-nav fact ^RealBufferAccessor da
                           eng ^Boolean master ^ByteBuffer buf ^long m ^long n ^long fkl ^long fku
                           ^long ofst ^long ld ^long fd ^long ord]
  Object
  (hashCode [a]
    (-> (hash :RealBandedMatrix) (hash-combine m) (hash-combine n) (hash-combine fkl) (hash-combine fku)
        (hash-combine (nrm2 eng (.dia a)))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealBandedMatrix b) (compatible? da b) (fits? a b))
      (let [kd (.kd band-nav fkl fku)]
        (loop [j 0]
          (if (< j fd)
            (let [end (.end band-nav m n fkl fku j)]
              (and (loop [i (.start band-nav fkl fku j)]
                     (if (< i end)
                       (and (= (.get da buf (+ ofst (* j ld) (- kd j) i)) (.get navigator b i j))
                            (recur (inc i)))
                       true))
                   (recur (inc j))))
            true)))
      :default false))
  (toString [a]
    (format "#RealBandedMatrix[%s, mxn:%dx%d (%dx%d), order%s, kl:%d, ku:%d, offset:%d, ld:%d]"
            (.entryType da) m n (.height band-nav m n fkl fku) (.width band-nav m n fkl fku)
            (dec-property ord) fkl fku ofst ld fd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (real-banded-matrix fact m n fkl fku ord eng))
  (raw [_ fact]
    "TODO create-banded in factory")
  (zero [_]
    (real-banded-matrix fact m n fkl fku ord eng))
  (zero [_ fact]
    "TODO create-banded in factory")
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  DenseContainer
  (view-ge [_]
    (real-ge-matrix fact false buf (.height band-nav m n fkl fku) (.width band-nav m n fkl fku) ofst ld ord))
  Navigable
  (order-navigator [_]
    navigator)
  (band-navigator [_]
    band-nav)
  MemoryContext
  (fully-packed? [_]
    (= 0 fkl fku))
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (= m (.mrows ^BandedMatrix b)) (= n (.ncols ^BandedMatrix b))
         (= fkl (.kl ^BandedMatrix b)) (= fku (.ku ^BandedMatrix b))))
  (fits-navigation? [_ b]
    (and (= ord (.order ^BandedMatrix b))))
  Monoid
  (id [a]
    (real-banded-matrix fact 0))
  BandedMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    ld)
  (count [_]
    (* m n))
  (kl [_]
    fkl)
  (ku [_]
    fku)
  (order [_]
    ord)
  Seqable
  (seq [a]
    (map #(seq (.stripe navigator a %)) (range 0 fd)))
  IFn$LLDD
  (invokePrim [x i j v]
    (entry! x i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [x i j v]
    (entry! x i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (<= (- j fku) i (+ j fkl)))
  (set [a val]
    (banded-doall band-nav da m n fkl fku ofst ld i j idx (.set da buf idx val))
    a)
  (set [a i j val]
    (.set da buf (.index band-nav ofst ld fkl fku i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (banded-doall band-nav da m n fkl fku ofst ld i j idx
                    (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))
      (banded-doall band-nav da m n fkl fku ofst ld i j idx
                    (.set da buf idx (.invokePrimitive navigator f i j (.get da buf idx)))))
    a)
  (alter [a i j f]
    (let [idx (.index band-nav ofst ld fkl fku i j)]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
      a))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (if (<= (- j fku) i (+ j fkl))
      (.get da buf (.index band-nav ofst ld fkl fku i j))
      0.0))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (real-block-vector fact false buf (min (- n (max 0 (- i fkl))) (+ fku 1 (min i fkl)))
                       (.index band-nav ofst ld fkl fku i (max 0 (- i fkl)))
                       (if (= ROW_MAJOR ord) 1 (dec ld))))
  (rows [a]
    (banded-rows a))
  (col [a j]
    (real-block-vector fact false buf (min (- m (max 0 (- j fku))) (+ fkl 1 (min j fku)))
                       (.index band-nav ofst ld fkl fku (max 0 (- j fku)) j)
                       (if (= COLUMN_MAJOR ord) 1 (dec ld))))
  (cols [a]
    (banded-cols a))
  (dia [a]
    (real-block-vector fact false buf (min m n) (.index band-nav ofst ld fkl fku 0) ld))
  (dia [a k]
    (if (<= (- fkl) k fku)
      (let [dia-dim (if (< 0 k) (min m (- n k)) (min (+ m k) n))]
        (real-block-vector fact false buf dia-dim (.index band-nav ofst ld fkl fku k) ld))
      (real-block-vector fact false buf 0 ofst ld)))
  (dias [a]
    (banded-dias a))
  (submatrix [a i j k l]
    (if (= i j)
      (let [kl (min fkl (dec k))
            ku (min fku (dec l))]
        (real-banded-matrix fact false buf k l kl ku
                            (- (.index band-nav ofst ld fkl fku i j) (.kd band-nav kl ku))
                            ld ord eng))
      (throw (ex-info "You cannot use regions outside the band in banded submatrix."
                      {:a (str a) :i i :j j :k k :l l}))))
  (transpose [a]
    (real-banded-matrix fact false buf n m fku fkl ofst ld (flip-layout ord) eng)))

(defn real-banded-matrix
  ([fact master buf m n kl ku ofst ld ord engine]
   (let [navigator (if (= COLUMN_MAJOR ord) col-navigator row-navigator)
         band-nav (if (= COLUMN_MAJOR ord) col-band-navigator row-band-navigator)]
     (->RealBandedMatrix navigator band-nav fact (data-accessor fact) engine true buf m n kl ku ofst
                         (max (long ld) (inc (+ (long kl) (long ku)))) (.fd ^BandNavigator band-nav m n kl ku) ord)))
  ([fact m n kl ku ord engine]
   (let [m (long m)
         n (long n)
         kl (long kl)
         ku (long ku)
         dia-dim (min m n)
         fd (long (if (= COLUMN_MAJOR ord) (min n (+ dia-dim ku)) (min m (+ dia-dim kl))))]
     (let-release [buf (.createDataSource (data-accessor fact) (* (inc (+ kl ku)) fd))]
       (real-banded-matrix fact true buf m n kl ku 0 0 ord engine))))
  ([fact m n kl ku engine]
   (real-banded-matrix fact m n kl ku DEFAULT_ORDER engine)))

(defmethod print-method RealBandedMatrix [a ^java.io.Writer w]
  (.write w (str a "\n"))
  (print-banded w a))

(defmethod transfer! [RealBandedMatrix RealBandedMatrix]
  [^RealBandedMatrix source destination]
  (if (and (compatible? source destination) (fits? source destination))
    (copy! source destination)
    (let [navigator (order-navigator source)]
      (banded-doall navigator source i j (.set navigator destination i j (.get navigator source i j)))
      destination)))

(defmethod transfer! [clojure.lang.Sequential RealBandedMatrix]
  [source ^RealBandedMatrix destination]
  (let [navigator (order-navigator destination)
        band-nav (band-navigator destination)
        m (.mrows destination)
        n (.ncols destination)
        kl (.kl destination)
        ku (.ku destination)
        fd (.fd band-nav m n kl ku)]
    (let [kd (.kd band-nav kl ku)]
      (loop [j 0 src (seq source)]
        (if (and src (< j fd))
          (recur (inc j)
                 (let [end (.end band-nav m n kl ku j)]
                   (loop [i (.start band-nav kl ku j) src src]
                     (if (and src (< i end))
                       (do (.set navigator destination i j (first src))
                           (recur (inc i) (next src)))
                       src))))
          destination)))))

;; =================== Real Packed Matrix ==================================

(deftype RealPackedMatrix [^LayoutNavigator nav ^DenseStorage stor ^Region region ^RealDefault default
                           fact ^RealBufferAccessor da eng matrix-type
                           ^Boolean master ^ByteBuffer buf ^long n ^long ofst ^long layout]
  Object
  (hashCode [a]
    (-> (hash :RealPackedMatrix) (hash-combine matrix-type) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (or (identical? a b)
        (and (instance? RealBandedMatrix b) (= matrix-type (.matrix-type ^RealPackedMatrix b))
             (compatible? da b) (= region (region b))
             (and-layout nav stor region i j idx
                         (= (.get da buf (+ ofst idx)) (.get ^RealLayoutNavigator nav b i j))))))
  (toString [a]
    (format "#RealPackedMatrix[%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) n n (str (if (.isColumnMajor stor) :column :row)) ofst))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    (native-factory fact))
  (index-factory [_]
    (index-factory fact))
  DataAccessorProvider
  (data-accessor [_]
    da)
  StorageProvider
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    region)
  Container
  (raw [_]
    (real-packed-matrix fact n nav stor region matrix-type default eng))
  (raw [_ fact]
    (create-packed fact n matrix-type (.isColumnMajor stor) (.isLower region) false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-packed fact n matrix-type (.isColumnMajor stor) (.isLower region) true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (= stor (storage b)))
  Monoid
  (id [a]
    (real-packed-matrix fact 0 nav stor region matrix-type default eng))
  BlockMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (order [_]
    layout)
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 n)))
  IFn$LLDD
  (invokePrim [x i j v]
    (entry! x i j v))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [x i j v]
    (entry! x i j v))
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (.accessible region i j))
  (set [a val]
    (if (not (Double/isNaN val))
      (set-all eng val a)
      (doall-layout nav stor region i j idx (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (doall-layout nav stor region i j idx
                    (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (doall-layout nav stor region i j idx
                    (.set da buf (+ ofst idx) (.invokePrimitive ^RealLayoutNavigator nav f i j
                                                                (.get da buf (+ ofst idx)))))))
  (alter [a i j f]
    (let [idx (+ ofst (.index stor i j))]
      (if (instance? IFn$DD f)
        (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
        (.invokePrimitive ^RealLayoutNavigator nav f i j (.get da buf idx))))
    a)
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (if (.accessible region i j)
      (.get da buf (+ ofst (.index stor i j)))
      (.entry default stor da buf ofst i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (if (.isRowMajor nav)
      (let [j (.rowStart region i)]
        (real-block-vector fact false buf (- (.rowEnd region i) j) (+ ofst (.index stor i j)) 1))
      (dragan-says-ex "You have to unpack column-major packed matrix to access its rows."
                      {:a a :layout :column})))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (if (.isColumnMajor nav)
      (let [i (.colStart region j)]
        (real-block-vector fact false buf (- (.colEnd region j) i) (+ ofst (.index stor i j)) 1))
      (dragan-says-ex "You have to unpack row-major packed matrix to access its columns."
                      {:a a :layout :row})))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (dragan-says-ex "You have to unpack a packed matrix to access its diagonals." {:a a}))
  (dia [a k]
    (dragan-says-ex "You have to unpack a packed matrix to access its diagonals." {:a a}))
  (dias [a]
    (dragan-says-ex "You have to unpack a packed matrix to access its diagonals." {:a a}))
  (submatrix [a i j k l]
    (dragan-says-ex "You have to unpack a packed matrix to access its submatrices." {:a a}))
  (transpose [a]
    (real-packed-matrix fact false buf n ofst (.isRowMajor nav) (.isUpper region) matrix-type default eng))
  ;; TODO LU is different a bit. It's probably LDL/UDU
  ;; TODO fluokitten should be easier, since the packing is full
  )

(defn real-packed-matrix
  ([fact master buf n ofst nav stor region matrix-type default engine]
   (->RealPackedMatrix nav stor region default fact (data-accessor fact) engine
                       matrix-type true buf n ofst
                       (if (.isColumnMajor ^LayoutNavigator nav) COLUMN_MAJOR ROW_MAJOR)))
  ([fact master buf n ofst column? lower? matrix-type]
   (let [nav (if column? real-column-navigator real-row-navigator)]
     (real-packed-matrix fact master buf n ofst nav (packed-storage column? lower? n)
                         (band-region n lower?) matrix-type (real-default matrix-type false)
                         (if (= :tr matrix-type) (tp-engine fact) (sp-engine fact)))))
  ([fact n column? lower? matrix-type]
   (let-release [buf (.createDataSource (data-accessor fact) (/ (* (long n) (inc (long n))) 2))]
     (real-packed-matrix fact true buf n 0 column? lower? matrix-type))))

(defmethod print-method RealPackedMatrix [a ^java.io.Writer w]
  (.write w (str a "\n"))
  (print-packed w a))

(defmethod transfer! [RealPackedMatrix RealPackedMatrix]
  [^RealPackedMatrix source destination]
  (if (and (compatible? source destination) (fits? source destination)
           (fits-navigation? source destination))
    (copy (engine source) source destination)
    (let [nav ^RealLayoutNavigator (navigator source)
          da ^RealBufferAccessor (data-accessor source)
          buf (.buffer source)
          ofst (.offset source)]
      (doall-layout source i j idx (.set nav destination i j (.get da buf (+ ofst idx))))
      destination)))

(defmethod transfer! [clojure.lang.Sequential RealPackedMatrix]
  [source ^RealPackedMatrix destination]
  (let [da ^RealBufferAccessor (data-accessor destination)
        buf (.buffer destination)
        ofst (.offset destination)
        len (unchecked-divide-int (* (.ncols destination) (inc (.ncols destination))) 2)]
    (loop [i 0 src source]
      (when (and src (< i len))
        (.set da buf (+ ofst i) (first src))
        (recur (inc i) (next src))))
    destination))
