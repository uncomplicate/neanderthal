;;   Copyright (c) Dragan Djuric. All rights reserved.;
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.buffer-block
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release Info info double-fn
                           wrap-float wrap-double wrap-int wrap-long wrap-short wrap-byte]]
             [utils :refer [direct-buffer slice-buffer dragan-says-ex mapped-buffer]]]
            [uncomplicate.fluokitten.protocols
             :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative fold]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! subvector vctr ge]]
             [real :refer [entry entry!]]
             [math :refer [ceil]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dense-rows dense-cols dense-dias region-dias region-cols region-rows
                             lu-factorization pivotless-lu-factorization dual-lu-factorization
                             real-accessor require-trf]]
             [printing :refer [print-vector print-ge print-uplo print-banded print-diagonal]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.host.fluokitten :refer :all])
  (:import [java.nio Buffer ByteBuffer DirectByteBuffer]
           java.nio.channels.FileChannel
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LLL]
           [uncomplicate.neanderthal.internal.api BufferAccessor RealBufferAccessor IntegerBufferAccessor
            VectorSpace Vector RealVector Matrix IntegerVector DataAccessor RealChangeable IntegerChangeable
            RealNativeMatrix RealNativeVector IntegerNativeVector DenseStorage FullStorage RealDefault
            LayoutNavigator RealLayoutNavigator Region MatrixImplementation GEMatrix UploMatrix
            BandedMatrix PackedMatrix DiagonalMatrix]
           uncomplicate.neanderthal.internal.navigation.BandStorage))

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

(def ^:private f* (double-fn *))

;; ================== Declarations ============================================

(declare integer-block-vector real-block-vector real-ge-matrix real-uplo-matrix
         real-banded-matrix real-packed-matrix real-diagonal-matrix
         ->RealUploMatrix ->RealBandedMatrix ->RealPackedMatrix ->RealDiagonalMatrix)

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
    (let [v (float v)
          strd Float/BYTES]
      (dotimes [i (.count this b)]
        (.putFloat ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ v]
    (wrap-float v))
  (castPrim [_ v]
    (float v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? FloatBufferAccessor da))))
  BufferAccessor
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
  (castPrim [_ v]
    (double v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? DoubleBufferAccessor da))))
  (device [_]
    :cpu)
  BufferAccessor
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

(deftype ByteBufferAccessor []
  DataAccessor
  (entryType [_]
    Byte/TYPE)
  (entryWidth [_]
    Byte/BYTES)
  (count [_ b]
    (.capacity ^ByteBuffer b))
  (createDataSource [_ n]
    (direct-buffer n))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v ^double v]
      (dotimes [i (.count this b)]
        (.put ^ByteBuffer b i v))
      b))
  (wrapPrim [_ s]
    (wrap-byte s))
  (castPrim [_ v]
    (byte v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? ByteBufferAccessor da))))
  (device [_]
    :cpu)
  BufferAccessor
  (slice [_ buf k l]
    (slice-buffer buf k l))
  IntegerBufferAccessor
  (get [_ buf i]
    (long (.get buf i)))
  (set [_ buf i val]
    (.put buf i val)))

(def byte-accessor (->ByteBufferAccessor))

(deftype ShortBufferAccessor []
  DataAccessor
  (entryType [_]
    Short/TYPE)
  (entryWidth [_]
    Short/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Short/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Short/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v ^double v
          strd Short/BYTES]
      (dotimes [i (.count this b)]
        (.putInt ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-long s))
  (castPrim [_ v]
    (long v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? ShortBufferAccessor da))))
  (device [_]
    :cpu)
  BufferAccessor
  (slice [_ buf k l]
    (slice-buffer buf (* Short/BYTES k) (* Short/BYTES l)))
  IntegerBufferAccessor
  (get [_ buf i]
    (long (.getShort buf (* Short/BYTES i))))
  (set [_ buf i val]
    (.putShort buf (* Short/BYTES i) val)))

(def short-accessor (->ShortBufferAccessor))

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
    (let [v ^double v
          strd Integer/BYTES]
      (dotimes [i (.count this b)]
        (.putInt ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-int s))
  (castPrim [_ v]
    (int v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? IntBufferAccessor da))))
  (device [_]
    :cpu)
  BufferAccessor
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
    (let [v ^double v
          strd Long/BYTES]
      (dotimes [i (.count this b)]
        (.putInt ^ByteBuffer b (* strd i) v))
      b))
  (wrapPrim [_ s]
    (wrap-long s))
  (castPrim [_ v]
    (long v))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible? [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? IntBufferAccessor da))))
  (device [_]
    :cpu)
  BufferAccessor
  (slice [_ buf k l]
    (slice-buffer buf (* Long/BYTES k) (* Long/BYTES l)))
  IntegerBufferAccessor
  (get [_ buf i]
    (.getLong buf (* Long/BYTES i)))
  (set [_ buf i val]
    (.putLong buf (* Long/BYTES i) val)))

(def long-accessor (->LongBufferAccessor))

(defn ^:private vector-seq [^Vector vector ^long i]
  (lazy-seq
   (if (< -1 i (.dim vector))
     (cons (.boxedEntry vector i) (vector-seq vector (inc i)))
     '())))

;; ==================== Transfer macros and functions  =============================================

(defn matrix-equals [^RealNativeMatrix a ^RealNativeMatrix b]
  (or (identical? a b)
      (and (instance? (class a) b)
           (= (.matrixType ^MatrixImplementation a) (.matrixType ^MatrixImplementation b))
           (compatible? a b) (= (.mrows a) (.mrows b)) (= (.ncols a) (.ncols b))
           (let [nav (real-navigator a)
                 da (real-accessor a)
                 buf (.buffer a)
                 ofst (.offset a)]
             (and-layout a i j idx (= (.get da buf (+ ofst idx)) (.get nav b i j)))))))

(defmacro ^:private transfer-matrix-matrix
  ([condition source destination]
   `(do
      (if (and (<= (.mrows ~destination) (.mrows ~source)) (<= (.ncols ~destination) (.ncols ~source)))
        (if (and (compatible? ~source ~destination) (fits? ~source ~destination) ~condition)
          (copy (engine ~source) ~source ~destination)
          (let [nav# (real-navigator ~destination)
                da# (real-accessor ~destination)
                buf# (.buffer ~destination)
                ofst# (.offset ~destination)]
            (doall-layout ~destination i# j# idx# (.set da# buf# (+ ofst# idx#) (.get nav# ~source i# j#)))))
        (dragan-says-ex "There is not enough entries in the source matrix. Take appropriate submatrix of the destination.."
                        {:source (info ~source) :destination (info ~destination)}))
      ~destination))
  ([source destination]
   `(transfer-matrix-matrix true ~source ~destination)))

(defmacro ^:private transfer-seq-matrix [source destination]
  `(if (sequential? (first ~source))
     (let [n# (.fd (storage ~destination))
           nav# (navigator ~destination)
           transfer-method# (get-method transfer!
                                        [(type (first ~source))
                                         (type (.stripe nav# ~destination 0))])]
       (loop [i# 0 s# ~source]
         (if (and s# (< i# n#))
           (do
             (transfer-method# (first s#) (.stripe nav# ~destination i#))
             (recur (inc i#) (next s#)))
           ~destination)))
     (let [da# (real-accessor ~destination)
           buf# (.buffer ~destination)
           ofst# (.offset ~destination)]
       (doseq-layout ~destination i# j# idx# ~source e# (.set da# buf# (+ ofst# idx#) e#))
       ~destination)))

(defmacro ^:private transfer-vector-matrix [source destination]
  `(let [stor# (storage ~destination)]
     (if (and (compatible? ~source ~destination) (.isGapless stor#))
       (let [dst-view# ^VectorSpace (view-vctr ~destination)
             n# (min (.dim ~source) (.dim dst-view#))]
         (when (pos? n#)
           (subcopy (engine ~source) ~source dst-view# 0 n# 0)))
       (let [da# (real-accessor ~destination)
             nav# (navigator ~destination)
             reg# (region ~destination)
             buf# (.buffer ~destination)
             ofst# (.offset ~destination)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set da# buf# (+ ofst# idx#) (.entry ~source cnt#))))))
     ~destination))

(defmacro ^:private transfer-matrix-vector [source destination]
  `(let [stor# (storage ~source)]
     (if (and (compatible? ~destination ~source) (.isGapless stor#))
       (let [src-view# ^VectorSpace (view-vctr ~source)
             n# (min (.dim src-view#) (.dim ~destination))]
         (when (pos? n#)
           (subcopy (engine src-view#) src-view# ~destination 0 n# 0)))
       (let [da# (real-accessor ~source)
             nav# (navigator ~source)
             reg# (region ~source)
             buf# (.buffer ~source)
             ofst# (.offset ~source)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set ~destination cnt# (.get da# buf# (+ ofst# idx#)))))))
     ~destination))

(defmacro ^:private transfer-array-matrix [source destination]
  ` (let [da# (real-accessor ~destination)
          nav# (navigator ~destination)
          stor# (storage ~destination)
          reg# (region ~destination)
          buf# (.buffer ~destination)
          ofst# (.offset ~destination)
          len# (alength ~source)]
      (doall-layout nav# stor# reg# i# j# idx# cnt#
                    (when (< cnt# len#)
                      (.set da# buf# (+ ofst# idx#) (aget ~source cnt#))))
      ~destination))

(defmacro ^:private transfer-matrix-array [source destination]
  `(let [da# (real-accessor ~source)
         nav# (navigator ~source)
         stor# (storage ~source)
         reg# (region ~source)
         buf# (.buffer ~source)
         ofst# (.offset ~source)
         len# (alength ~destination)]
     (doall-layout nav# stor# reg# i# j# idx# cnt#
                   (when (< cnt# len#)
                     (aset ~destination cnt# (.get da# buf# (+ ofst# idx#)))))
     ~destination))

(defmacro ^:private transfer-vector-vector [source destination]
  `(do
     (if (compatible? ~source ~destination)
       (when-not (identical? ~source ~destination)
         (let [n# (min (.dim ~source) (.dim ~destination))]
           (subcopy (engine ~source) ~source ~destination 0 n# 0)))
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

;; ============ Integer Vector =================================================

(deftype IntegerBlockVector [fact ^IntegerBufferAccessor da eng master ^ByteBuffer buf
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
  Info
  (info [x]
    {:entry-type (.entryType da)
     :class (class x)
     :device :cpu
     :dim n
     :offset ofst
     :stride strd
     :master master
     :info (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  Seqable
  (seq [x]
    (vector-seq x 0))
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
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^VectorSpace y)))
  (device [_]
    :cpu)
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
  IntegerNativeVector
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
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
    (integer-block-vector fact 0))
  Applicative
  (pure [_ v]
    (let-release [res (integer-block-vector fact 1)]
      (.set ^IntegerBlockVector res 0 v)))
  (pure [_ v vs]
    (vctr fact (cons v vs))))

(extend IntegerBlockVector
  Functor
  {:fmap (vector-fmap IntegerBlockVector long)}
  PseudoFunctor
  {:fmap! (vector-fmap identity IntegerBlockVector long)}
  Foldable
  {:fold vector-fold
   :foldmap (vector-foldmap IntegerBlockVector long)}
  Magma
  {:op (constantly vector-op)})

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

(deftype RealBlockVector [fact ^RealBufferAccessor da eng master ^ByteBuffer buf
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
  Info
  (info [x]
    {:entry-type (.entryType da)
     :class (class x)
     :device :cpu
     :dim n
     :offset ofst
     :stride strd
     :master master
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  Seqable
  (seq [x]
    (vector-seq x 0))
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
  Viewable
  (view [x]
    (real-block-vector fact false buf n ofst strd))
  DenseContainer
  (view-vctr [x]
    x)
  (view-vctr [_ stride-mult]
    (real-block-vector fact false buf (ceil (/ n (long stride-mult))) ofst (* (long stride-mult) strd)))
  (view-ge [_]
    (real-ge-matrix fact false buf n 1 ofst (layout-navigator true) (full-storage true n 1) (ge-region n 1)))
  (view-ge [x stride-mult]
    (view-ge (view-ge x) stride-mult))
  (view-ge [x m n]
    (view-ge (view-ge x) m n))
  (view-tr [x uplo diag]
    (view-tr (view-ge x) uplo diag))
  (view-sy [x uplo]
    (view-sy (view-ge x) uplo))
  MemoryContext
  (compatible? [_ y]
    (compatible? da y))
  (fits? [_ y]
    (= n (.dim ^VectorSpace y)))
  (device [_]
    :cpu)
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
    (if-not (Double/isNaN val)
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
  RealNativeVector
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
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
  Applicative
  (pure [_ v]
    (let-release [res (real-block-vector fact 1)]
      (.set ^RealBlockVector res 0 v)))
  (pure [_ v vs]
    (vctr fact (cons v vs))))

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
  {:fmap (vector-fmap RealBlockVector double)}
  PseudoFunctor
  {:fmap! (vector-fmap identity RealBlockVector double)}
  Foldable
  {:fold vector-fold
   :foldmap (vector-foldmap RealBlockVector double)}
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

(deftype RealGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg
                       fact ^RealBufferAccessor da eng master
                       ^ByteBuffer buf ^long m ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (matrix-equals a b))
  (toString [a]
    (format "#RealGEMatrix[%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) m n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cpu
     :matrix-type :ge
     :dim (.dim ^Matrix a)
     :m m
     :n n
     :offset ofst
     :stride (.ld stor)
     :master master
     :layout (:layout (info nav))
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (real-ge-matrix fact m n nav stor reg))
  (raw [_ fact]
    (create-ge fact m n (.isColumnMajor nav) false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-ge fact m n (.isColumnMajor nav) true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  Viewable
  (view [a]
    (real-ge-matrix fact false buf m n ofst nav stor reg))
  DenseContainer
  (view-vctr [a]
    (if (.isGapless stor)
      (real-block-vector fact false buf (.dim a) ofst 1)
      (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:a (info a)}))))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [a]
    a)
  (view-ge [_ stride-mult]
    (let [shrinked (ceil (/ (.fd stor) (long stride-mult)))
          column-major (.isColumnMajor nav)
          [m n] (if column-major [m shrinked] [shrinked n])]
      (real-ge-matrix fact false buf m n ofst nav
                      (full-storage column-major m n (* (long stride-mult) (.ld stor)))
                      (ge-region m n))))
  (view-ge [a m n]
    (if (.isGapless stor)
      (real-ge-matrix fact false buf m n ofst nav (full-storage (.isColumnMajor nav) m n) (ge-region m n))
      (throw (ex-info "Strided GE matrix cannot be viewed through different dimensions." {:a (info a)}))))
  (view-tr [_ lower? diag-unit?]
    (let [n (min m n)]
      (real-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld stor))
                        (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?)
                        (tr-engine fact))))
  (view-sy [_ lower?]
    (let [n (min m n)]
      (real-uplo-matrix fact false buf n ofst nav (full-storage (.isColumnMajor nav) n n (.ld stor))
                        (band-region n lower?) :sy sy-default (sy-engine fact))))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? GEMatrix b)) (= reg (region b)))
  (fits-navigation? [_ b]
    (= nav (navigator b)))
  (device [_]
    :cpu)
  Monoid
  (id [_]
    (real-ge-matrix fact 0 0 (.isColumnMajor nav)))
  Applicative
  (pure [_ v]
    (let-release [res (real-ge-matrix fact 1 1 (.isColumnMajor nav))]
      (.set ^RealGEMatrix res 0 0 v)))
  (pure [_ v vs]
    (ge fact (cons v vs)))
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 (.fd stor))))
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
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    true)
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index nav stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrimitive ^RealLayoutNavigator nav f i j
                                                                (.get da buf (+ ofst idx))))))
    a)
  (alter [a i j f]
    (let [idx (+ ofst (.index nav stor i j))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    (.isGapless stor))
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get da buf (+ ofst (.index nav stor i j))))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (real-block-vector fact false buf n (+ ofst (.index nav stor i 0))
                       (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (real-block-vector fact false buf m (+ ofst (.index nav stor 0 j))
                       (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf (min m n) ofst (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (real-block-vector fact false buf (min m (- n k)) (+ ofst (.index nav stor 0 k)) (inc (.ld stor)))
      (real-block-vector fact false buf (min (+ m k) n) (+ ofst (.index nav stor (- k) 0)) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (real-ge-matrix fact false buf k l (+ ofst (.index nav stor i j))
                    nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (real-ge-matrix fact false buf n m ofst (flip nav) stor (flip reg)))
  Triangularizable
  (create-trf [a pure]
    (lu-factorization a pure))
  (create-ptrf [a]
    (dragan-says-ex "Pivotless factorization is not available for GE matrices."))
  TRF
  (trtrs! [_ _]
    (require-trf))
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
    (require-trf)))

(defn real-ge-matrix
  ([fact master buf m n ofst nav stor reg]
   (->RealGEMatrix nav stor reg fact (data-accessor fact) (ge-engine fact) master buf m n ofst))
  ([fact m n nav ^FullStorage stor reg]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (real-ge-matrix fact true buf m n 0 nav stor reg)))
  ([fact ^long m ^long n column?]
   (real-ge-matrix fact m n (layout-navigator column?) (full-storage column? m n) (ge-region m n)))
  ([fact ^long m ^long n]
   (real-ge-matrix fact m n true)))

(extend RealGEMatrix
  Functor
  {:fmap (matrix-fmap double)}
  PseudoFunctor
  {:fmap! (matrix-fmap identity double)}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Magma
  {:op (constantly matrix-op)})

(defmethod print-method RealGEMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-ge w a))

;; =================== Real Uplo Matrix ==================================

(deftype RealUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^RealDefault default
                         fact ^RealBufferAccessor da eng matrix-type master
                         ^ByteBuffer buf ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :RealUploMatrix) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (matrix-equals a b))
  (toString [a]
    (format "#RealUploMatrix[%s, type%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) matrix-type n n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cpu
     :matrix-type matrix-type
     :dim (.dim ^Matrix a)
     :m n
     :n n
     :offset ofst
     :stride (.ld stor)
     :master master
     :layout (:layout (info nav))
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  UploMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tr matrix-type))
  (isSymmetric [_]
    (= :sy matrix-type))
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (real-uplo-matrix fact n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-uplo fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  Viewable
  (view [a]
    (->RealUploMatrix nav stor reg default fact da eng matrix-type false buf n ofst))
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (real-ge-matrix fact false buf n n ofst nav stor (ge-region n n) ))
  (view-ge [a m n]
    (view-ge (view-ge a) m n))
  (view-ge [a stride-mult]
    (view-ge (view-ge a) stride-mult))
  (view-tr [_ lower? diag-unit?]
    (real-uplo-matrix fact false buf n ofst nav stor (band-region n lower? diag-unit?)
                      :tr (real-default :tr diag-unit?) (tr-engine fact)))
  (view-sy [_ lower?]
    (real-uplo-matrix fact false buf n ofst nav stor (band-region n lower?)
                      :sy sy-default (sy-engine fact)))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? UploMatrix b)
         (let [reg-b (region b)]
           (or (= reg reg-b)
               (and (= :sy matrix-type) (.isSymmetric ^UploMatrix b)
                    (not= nav (navigator b)) (not= (.uplo reg) (.uplo reg-b))
                    (= n (.ncols ^Matrix b)))))))
  (fits-navigation? [_ b]
    (and (= nav (navigator b))
         (or (instance? GEMatrix b) (= reg (region b)))))
  (device [_]
    :cpu)
  Monoid
  (id [a]
    (real-uplo-matrix fact 0 (.isColumnMajor nav) matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (real-uplo-matrix fact 1 (.isColumnMajor nav) matrix-type)]
      (.set ^RealUploMatrix res 0 0 v)))
  (pure [_ v vs]
    (let [source (cons v vs)]
      (let-release [res (real-uplo-matrix fact (long (Math/sqrt (count source)))
                                          (.isColumnMajor nav) matrix-type)]
        (transfer! source res))))
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
    (.accessible reg i j))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index nav stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrimitive ^RealLayoutNavigator nav f i j
                                                                (.get da buf (+ ofst idx))))))
    a)
  (alter [a i j f]
    (let [idx (+ ofst (.index nav stor i j))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    false)
  (dim [_]
    (* n n))
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (if (.accessible reg i j)
      (.get da buf (+ ofst (.index nav stor i j)))
      (.entry default nav stor da buf ofst i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart reg i)]
      (real-block-vector fact false buf (- (.rowEnd reg i) start) (+ ofst (.index nav stor i start))
                         (if (.isRowMajor nav) 1 (.ld stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (real-block-vector fact false buf (- (.colEnd reg j) start) (+ ofst (.index nav stor start j))
                         (if (.isColumnMajor nav) 1 (.ld stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf n ofst (inc (.ld stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (real-block-vector fact false buf (- n k) (+ ofst (.index nav stor 0 k)) (inc (.ld  stor)))
        (real-block-vector fact false buf (+ n k) (+ ofst (.index nav stor (- k) 0)) (inc (.ld stor))))
      (real-block-vector fact false buf 0 ofst 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (real-uplo-matrix fact false buf k (+ ofst (.index nav stor i j)) nav
                        (full-storage (.isColumnMajor nav) k k (.ld stor))
                        (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (real-uplo-matrix fact false buf n ofst (flip nav) stor (flip reg) matrix-type default eng))
  Triangularizable
  (create-trf [a pure]
    (if (= :sy matrix-type)
      (dual-lu-factorization a pure)
      a))
  (create-ptrf [a]
    (if (= :sy matrix-type)
      (pivotless-lu-factorization a false)
      a))
  TRF
  (trtrs [a b]
    (if (= :tr matrix-type)
      (let-release [res (raw b)]
        (copy (engine b) b res)
        (trs eng a res))
      (require-trf)))
  (trtrs! [a b]
    (if (= :tr matrix-type)
      (trs eng a b)
      (require-trf)))
  (trtri! [a]
    (if (= :tr matrix-type)
      (tri eng a)
      (require-trf)))
  (trtri [a]
    (if (= :tr matrix-type)
      (let-release [res (raw a)]
        (tri eng (copy eng a res)))
      (require-trf)))
  (trcon [a _ nrm1?]
    (if (= :tr matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trcon [a nrm1?]
    (if (= :tr matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trdet [a]
    (if (= :tr matrix-type)
      (if (.isDiagUnit reg) 1.0 (fold (.dia a) f* 1.0))
      (require-trf))))

(extend RealUploMatrix
  Functor
  {:fmap (matrix-fmap double)}
  PseudoFunctor
  {:fmap! (matrix-fmap identity double)}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Magma
  {:op (constantly matrix-op)})

(defn real-uplo-matrix
  ([fact master buf n ofst nav stor reg matrix-type default engine]
   (->RealUploMatrix nav stor reg default fact (data-accessor fact) engine matrix-type
                     master buf n ofst))
  ([fact n nav ^FullStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (real-uplo-matrix fact true buf n 0 nav stor reg matrix-type default engine)))
  ([fact n column? lower? diag-unit? matrix-type]
   (real-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                     (band-region n lower? diag-unit?) matrix-type (real-default matrix-type diag-unit?)
                     (case matrix-type
                       :tr (tr-engine fact)
                       :sy (sy-engine fact)
                       (dragan-says-ex (format "%s is not a valid UPLO matrix type. Please send me a bug report."
                                               matrix-type)
                                       {:type matrix-type}))))
  ([fact n column? lower? diag-unit?]
   (real-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                     (band-region n lower? diag-unit?) :tr (real-default :tr diag-unit?) (tr-engine fact)))
  ([fact n column? lower?]
   (real-uplo-matrix fact n (layout-navigator column?) (full-storage column? n n)
                     (band-region n lower?) :sy (real-default :sy) (sy-engine fact))))

(defmethod print-method RealUploMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-uplo w a "*"))

;; ================= Banded Matrix ==============================================================

(deftype RealBandedMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^RealDefault default
                           fact ^RealBufferAccessor da eng matrix-type
                           master ^ByteBuffer buf ^long m ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :RealBandedMatrix) (hash-combine matrix-type) (hash-combine m) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (matrix-equals a b))
  (toString [a]
    (format "#RealBandedMatrix[%s, type%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) matrix-type m n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cpu
     :matrix-type matrix-type
     :dim (.dim ^Matrix a)
     :m m
     :n n
     :offset ofst
     :stride (.ld stor)
     :master master
     :layout (:layout (info nav))
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  BandedMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tb matrix-type))
  (isSymmetric [_]
    (= :sb matrix-type))
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (real-banded-matrix fact m n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-banded fact m n (.kl reg) (.ku reg) matrix-type (.isColumnMajor nav) false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-banded fact m n (.kl reg) (.ku reg) matrix-type (.isColumnMajor nav) true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  Viewable
  (view [_]
    (->RealBandedMatrix nav stor reg default fact da eng matrix-type false buf m n ofst))
  DenseContainer
  (view-vctr [a]
    (view-vctr (view-ge a)))
  (view-vctr [a stride-mult]
    (view-vctr (view-ge a) stride-mult))
  (view-ge [_]
    (let [m (if (.isColumnMajor nav) (.sd stor) (.fd stor))
          n (if (.isColumnMajor nav) (.fd stor) (.sd stor))]
      (real-ge-matrix fact false buf m n ofst nav
                      (full-storage (.isColumnMajor nav) m n (.ld stor))
                      (ge-region m n))))
  (view-ge [a m n]
    (if (<= (* (long m) (long n)) (.capacity stor))
      (view-ge (view-ge a) m n)
      (dragan-says-ex "This GB does not have sufficient storage space for required m and n dimensions.")))
  (view-tr [_ _ diag-unit?]
    (if-not (= :gb matrix-type)
      (let [k (max (.kl reg) (.ku reg))]
        (real-banded-matrix fact false buf n n ofst nav
                            (uplo-storage (.isColumnMajor nav) n k (.isLower reg))
                            (tb-region n k (.isLower reg) diag-unit?) :tb (real-default :tb diag-unit?)
                            (tb-engine fact)))
      (dragan-says-ex "GB cannot be viewed as a TB due to specific factorization requirements.")))
  (view-sy [_ _]
    (if-not (= :gb matrix-type)
      (let [k (max (.kl reg) (.ku reg))]
        (real-banded-matrix fact false buf n n ofst nav
                            (uplo-storage (.isColumnMajor nav) n k (.isLower reg))
                            (sb-region n k (.isLower reg)) :sb sb-default (sb-engine fact)))
      (dragan-says-ex "GB cannot be viewed as a SB due to specific factorization requirements.")))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [a b]
    (and (instance? BandedMatrix b)
         (let [reg-b (region b)]
           (or (= reg reg-b)
               (and (= :sb matrix-type) (.isSymmetric ^BandedMatrix b) (not= nav (navigator b))
                    (= (+ (.kl reg) (.ku reg)) (+ (.kl reg-b) (.ku reg-b)))
                    (= n (.ncols ^Matrix b)))))))
  (fits-navigation? [_ b]
    (= nav (navigator b)))
  (device [_]
    :cpu)
  Monoid
  (id [a]
    (real-banded-matrix fact 0 0 0 0 (.isColumnMajor nav) matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (real-banded-matrix fact 1 1 nav stor reg matrix-type default eng)]
      (.set ^RealBandedMatrix res 0 0 v)))
  (pure [_ v vs]
    (dragan-says-ex "Vararg pure is not available for banded matrices."))
  Seqable
  (seq [a]
    (map seq (.dias a)))
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
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    (.accessible reg i j))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index nav stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrimitive ^RealLayoutNavigator nav f i j
                                                                (.get da buf (+ ofst idx))))))
    a)
  (alter [a i j f]
    (let [idx (+ ofst (.index nav stor i j))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    (.ld stor))
  (isContiguous [_]
    (.isGapless stor))
  (dim [_]
    (* m n))
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (if (.accessible reg i j)
      (.get da buf (+ ofst (.index nav stor i j)))
      (.entry default nav stor da buf ofst i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart reg i)]
      (real-block-vector fact false buf (- (.rowEnd reg i) start) (+ ofst (.index nav stor i start))
                         (if (.isRowMajor nav) 1 (dec (.ld stor))))))
  (rows [a]
    (region-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (real-block-vector fact false buf (- (.colEnd reg j) start) (+ ofst (.index nav stor start j))
                         (if (.isColumnMajor nav) 1 (dec (.ld stor))))))
  (cols [a]
    (region-cols a))
  (dia [a]
    (.dia a 0))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (real-block-vector fact false buf (min m (- n k)) (+ ofst (.index nav stor 0 k)) (.ld stor))
        (real-block-vector fact false buf (min (+ m k) n) (+ ofst (.index nav stor (- k) 0)) (.ld stor)))
      (real-block-vector fact false buf 0 ofst 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (= i j)
      (let [kl (min (.kl reg) (dec k))
            ku (min (.ku reg) (dec l))]
        (real-banded-matrix fact false buf k l (- (+ ofst (.index nav stor i j)) (inc kl))
                            nav (band-storage (.isColumnMajor nav) k l (.ld stor) kl ku)
                            (band-region k l kl ku) matrix-type default eng))
      (dragan-says-ex "You cannot create a submatrix of a banded (GB, TB, or SB) matrix outside its region. No way around that."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (real-banded-matrix fact false buf n m ofst (flip nav) stor (flip reg) matrix-type default eng))
  Subband
  (subband [a kl ku]
    (if (and (<= 0 (long kl) (.kl reg)) (<= 0 (long ku) (.ku reg)))
      (let [sub-stor (band-storage (.isColumnMajor nav) m n (.ld stor) kl ku)]
        (real-banded-matrix fact false buf m n
                            (+ ofst (- (.index stor 0 0) (.index ^DenseStorage sub-stor 0 0)))
                            nav sub-stor (band-region m n kl ku) matrix-type default eng))
      (dragan-says-ex "You cannot create a subband outside available region. No way around that."
                      {:a (info a) :kl kl :ku ku})))
  Triangularizable
  (create-trf [a pure]
    (case matrix-type
      :tb a
      :sb (pivotless-lu-factorization a pure)
      :gb (lu-factorization a pure)
      (dragan-says-ex "Triangular factorization is not available for this matrix type"
                      {:matrix-type matrix-type})))
  (create-ptrf [a]
    (case matrix-type
      :tb a
      :sb (pivotless-lu-factorization a false)
      (dragan-says-ex "Pivotless factorization is not available for this matrix type"
                      {:matrix-type matrix-type})))
  TRF
  (trtrs [a b]
    (if (= :tb matrix-type)
      (let-release [res (raw b)]
        (copy (engine b) b res)
        (trs eng a res))
      (require-trf)))
  (trtrs! [a b]
    (if (= :tb matrix-type)
      (trs eng a b)
      (require-trf)))
  (trtri! [a]
    (if (= :tb matrix-type)
      (tri eng a)
      (require-trf)))
  (trtri [a]
    (if (= :tb matrix-type)
      (let-release [res (raw a)]
        (tri eng (copy eng a res)))
      (require-trf)))
  (trcon [a _ nrm1?]
    (if (= :tb matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trcon [a nrm1?]
    (if (= :tb matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trdet [a]
    (if (= :tb matrix-type)
      (if (.isDiagUnit reg) 1.0 (fold (.dia a) f* 1.0))
      (require-trf))))

(extend RealBandedMatrix
  Functor
  {:fmap (matrix-fmap double)}
  PseudoFunctor
  {:fmap! (matrix-fmap identity double)}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Magma
  {:op (constantly matrix-op)})

(defn real-banded-matrix
  ([fact master buf m n ofst nav stor reg matrix-type default engine]
   (->RealBandedMatrix nav stor reg default fact (data-accessor fact) engine matrix-type
                       master buf m n ofst))
  ([fact m n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (real-banded-matrix fact true buf m n 0 nav stor reg matrix-type default engine)))
  ([fact m n kl ku column? matrix-type]
   (real-banded-matrix fact m n (layout-navigator column?) (band-storage column? m n kl ku)
                       (band-region m n kl ku) matrix-type (real-default matrix-type)
                       (case matrix-type
                         :gb (gb-engine fact)
                         :sb (sb-engine fact)
                         :tb (tb-engine fact)
                         (dragan-says-ex (format "%s is not a valid banded matrix type. Please send me a bug report."
                                                 matrix-type)))))
  ([fact m n kl ku column?]
   (real-banded-matrix fact m n (layout-navigator column?) (band-storage column? m n kl ku)
                       (band-region m n kl ku) :gb zero-default (gb-engine fact))))

(defn real-tb-matrix [fact n k column? lower? diag-unit?]
  (real-banded-matrix fact n n (layout-navigator column?) (uplo-storage column? n k lower?)
                      (tb-region n k lower? diag-unit?) :tb (real-default :tb diag-unit?) (tb-engine fact)))

(defn real-sb-matrix [fact n k column? lower?]
  (real-banded-matrix fact n n (layout-navigator column?) (uplo-storage column? n k lower?)
                      (sb-region n k lower?) :sb sb-default (sb-engine fact)))

(defmethod print-method RealBandedMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-banded w a))

;; =================== Real Packed Matrix ==================================

(deftype RealPackedMatrix [^LayoutNavigator nav ^DenseStorage stor ^Region reg ^RealDefault default
                           fact ^RealBufferAccessor da eng matrix-type
                           master ^ByteBuffer buf ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :RealPackedMatrix) (hash-combine matrix-type) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (matrix-equals a b))
  (toString [a]
    (format "#RealPackedMatrix[%s, type%s, mxn:%dx%d, layout%s, offset:%d]"
            (.entryType da) matrix-type n n (dec-property (.layout nav)) ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cpu
     :matrix-type matrix-type
     :dim (.dim ^Matrix a)
     :m n
     :n n
     :offset ofst
     :master master
     :layout (:layout (info nav))
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  PackedMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tp matrix-type))
  (isSymmetric [_]
    (= :sp matrix-type))
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (real-packed-matrix fact n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-packed fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-packed fact n matrix-type (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  Viewable
  (view [_]
    (->RealPackedMatrix nav stor reg default fact da eng matrix-type false buf n ofst))
  DenseContainer
  (view-vctr [a]
    (real-block-vector fact false buf (.surface reg) ofst 1))
  (view-vctr [a stride-mult]
    (view-vctr (view-vctr a) stride-mult))
  (view-ge [_]
    (dragan-says-ex "Packed matrices cannot be viewed as a GE matrix."))
  (view-ge [_ m n]
    (dragan-says-ex "Packed matrices cannot be viewed as a GE matrix."))
  (view-tr [_ _ diag-unit?]
    (real-packed-matrix fact false buf n ofst nav stor (band-region n (.isLower reg) diag-unit?)
                        :tp (real-default :tp diag-unit?) (tp-engine fact)))
  (view-sy [_ _]
    (real-packed-matrix fact false buf n ofst nav stor (band-region n (.isLower reg))
                        :sp sy-default (sp-engine fact)))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [_ b]
    (and (instance? PackedMatrix b) (= reg (region b))))
  (device [_]
    :cpu)
  Monoid
  (id [a]
    (real-packed-matrix fact 0 (.isColumnMajor nav) (.isLower reg) (.isDiagUnit reg) matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (real-packed-matrix fact 1 (.isColumnMajor nav) (.isLower reg)
                                          (.isDiagUnit reg) matrix-type)]
      (.set ^RealPackedMatrix res 0 0 v)))
  (pure [_ v vs]
    (let [source (cons v vs)]
      (let-release [res (real-packed-matrix fact (long (Math/sqrt (count source)))
                                            (.isColumnMajor nav) (.isLower reg)
                                            (.isDiagUnit reg) matrix-type)]
        (transfer! source res))))
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
    (.accessible reg i j))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index nav stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (doall-layout nav stor reg i j idx
                    (.set da buf (+ ofst idx) (.invokePrimitive ^RealLayoutNavigator nav f i j
                                                                (.get da buf (+ ofst idx))))))
    a)
  (alter [a i j f]
    (let [idx (+ ofst (.index nav stor i j))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx))))
    a)
  RealNativeMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (dim [_]
    (* n n))
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (if (.accessible reg i j)
      (.get da buf (+ ofst (.index nav stor i j)))
      (.entry default nav stor da buf ofst i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (if (.isRowMajor nav)
      (let [j (.rowStart reg i)]
        (real-block-vector fact false buf (- (.rowEnd reg i) j) (+ ofst (.index nav stor i j)) 1))
      (dragan-says-ex "You have to unpack column-major packed matrix to access its rows."
                      {:a a :layout :column})))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (if (.isColumnMajor nav)
      (let [i (.colStart reg j)]
        (real-block-vector fact false buf (- (.colEnd reg j) i) (+ ofst (.index nav stor i j)) 1))
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
    (real-packed-matrix fact false buf n ofst (flip nav) stor (flip reg) matrix-type default eng))
  Triangularizable
  (create-trf [a pure]
    (if (= :sp matrix-type)
      (dual-lu-factorization a pure)
      a))
  (create-ptrf [a]
    (if (= :sp matrix-type)
      (pivotless-lu-factorization a false)
      a))
  TRF
  (trtrs [a b]
    (if (= :tp matrix-type)
      (let-release [res (raw b)]
        (copy (engine b) b res)
        (trs eng a b))
      (require-trf)))
  (trtrs! [a b]
    (if (= :tp matrix-type)
      (trs eng a b)
      (require-trf)))
  (trtri! [a]
    (if (= :tp matrix-type)
      (tri eng a)
      (require-trf)))
  (trtri [a]
    (if (= :tp matrix-type)
      (let-release [res (raw a)]
        (tri eng (copy eng a res)))
      (require-trf)))
  (trcon [a _ nrm1?]
    (if (= :tp matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trcon [a nrm1?]
    (if (= :tp matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trdet [a]
    (if (= :tp matrix-type)
      (if (.isDiagUnit reg) 1.0 (fold (.dia a) f* 1.0))
      (require-trf))))

(extend RealPackedMatrix
  Functor
  {:fmap (matrix-fmap double)}
  PseudoFunctor
  {:fmap! (matrix-fmap identity double)}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Magma
  {:op (constantly matrix-op)})

(defn real-packed-matrix
  ([fact master buf n ofst nav stor reg matrix-type default engine]
   (->RealPackedMatrix nav stor reg default fact (data-accessor fact) engine
                       matrix-type master buf n ofst))
  ([fact n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (real-packed-matrix fact true buf n 0 nav stor reg matrix-type default engine)))
  ([fact n column? lower? diag-unit? matrix-type]
   (case matrix-type
     :tp (real-packed-matrix fact n column? lower? diag-unit?)
     :sy (real-packed-matrix fact n column? lower?)
     (dragan-says-ex "Packed matrices have to be either triangular or symmetric."
                     {:matrix-type matrix-type})))
  ([fact n column? lower? diag-unit?]
   (real-packed-matrix fact n (layout-navigator column?) (packed-storage column? lower? n)
                       (band-region n lower? diag-unit?) :tp (real-default :tp diag-unit?) (tp-engine fact)))
  ([fact n column? lower?]
   (real-packed-matrix fact n (layout-navigator column?) (packed-storage column? lower? n)
                       (band-region n lower?) :sp sy-default (sp-engine fact))))

(defmethod print-method RealPackedMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-uplo w a "."))

(deftype RealDiagonalMatrix [^LayoutNavigator nav ^DenseStorage stor ^Region reg ^RealDefault default
                             fact ^RealBufferAccessor da eng matrix-type
                             master ^ByteBuffer buf ^long n ^long ofst]
  Object
  (hashCode [a]
    (-> (hash :RealDiagonalMatrix) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (or (identical? a b)
        (and (instance? RealDiagonalMatrix b) (compatible? a b) (fits? a b)
             (let [n (.surface reg)
                   buf-b (.buffer ^RealDiagonalMatrix b)
                   ofst-b (.offset ^RealDiagonalMatrix b)]
               (loop [i 0]
                 (if (< i n)
                   (and (= (.get da buf (+ i ofst)) (.get da buf-b (+ i ofst-b)))
                        (recur (inc i)))
                   true))))))
  (toString [a]
    (format "#RealDiagonalMatrix[%s, type%s mxn:%dx%d, offset:%d]"
            (.entryType da) matrix-type n n ofst))
  Info
  (info [a]
    {:entry-type (.entryType da)
     :class (class a)
     :device :cpu
     :matrix-type matrix-type
     :dim (.dim ^Matrix a)
     :n n
     :offset ofst
     :master master
     :layout :diagonal
     :storage (info stor)
     :region (info reg)
     :engine (info eng)})
  Releaseable
  (release [_]
    (if master (release buf) true))
  DiagonalMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    (= :st matrix-type))
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
  Navigable
  (navigator [_]
    nav)
  (storage [_]
    stor)
  (region [_]
    reg)
  Container
  (raw [_]
    (real-diagonal-matrix fact n nav stor reg matrix-type default eng))
  (raw [_ fact]
    (create-diagonal fact n matrix-type false))
  (zero [a]
    (raw a))
  (zero [_ fact]
    (create-diagonal fact n matrix-type true))
  (host [a]
    (let-release [res (raw a)]
      (copy eng a res)))
  (native [a]
    a)
  Viewable
  (view [_]
    (->RealDiagonalMatrix nav stor reg default fact da eng matrix-type false buf n ofst))
  DenseContainer
  (view-vctr [a]
    (real-block-vector fact false buf (.surface reg) ofst 1))
  (view-vctr [_ _]
    (dragan-says-ex "TD cannot be viewed as a strided vector."))
  (view-ge [_]
    (dragan-says-ex "TD cannot be viewed as a GE matrix."))
  (view-ge [_ m n]
    (dragan-says-ex "TD cannot be viewed as a GE matrix."))
  (view-tr [_ _ diag-unit?]
    (dragan-says-ex "TD cannot be viewed as a TR matrix."))
  (view-sy [_ _]
    (dragan-says-ex "TD cannot be viewed as a SY matrix."))
  MemoryContext
  (compatible? [_ b]
    (compatible? da b))
  (fits? [a b]
    (and (instance? DiagonalMatrix b) (= reg (region b))))
  (fits-navigation? [_ b]
    true)
  (device [_]
    :cpu)
  Monoid
  (id [a]
    (real-diagonal-matrix fact 0 matrix-type))
  Applicative
  (pure [_ v]
    (let-release [res (real-diagonal-matrix fact 1 matrix-type)]
      (.set ^RealUploMatrix res 0 0 v)))
  (pure [_ v vs]
    (let [source (cons v vs)]
      (let-release [res (real-diagonal-matrix fact (count source) matrix-type)]
        (transfer! source res))))
  Seqable
  (seq [a]
    (map seq (.dias a)))
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
    (.accessible reg i j))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (dotimes [idx (.capacity stor)]
        (.set da buf (+ ofst idx) val)))
    a)
  (set [a i j val]
    (.set da buf (+ ofst (.index nav stor i j)) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (dotimes [idx (.capacity stor)]
        (.set da buf (+ ofst idx) (.invokePrim ^IFn$DD f (.get da buf (+ ofst idx)))))
      (dragan-says-ex "You cannot call indexed alter on diagonal matrices. Use banded matrix."))
    a)
  (alter [a i j f]
    (let [idx (+ ofst (.index nav stor i j))]
      (.set da buf idx (.invokePrim ^IFn$DD f (.get da buf idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    1)
  (isContiguous [_]
    true)
  (dim [_]
    (* n n))
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (if (.accessible reg i j)
      (.get da buf (+ ofst (.index nav stor i j)))
      (.entry default nav stor da buf ofst i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (dragan-says-ex "You cannot access rows of a (tri)diagonal matrix."))
  (rows [a]
    (dragan-says-ex "You cannot access rows of a (tri)diagonal matrix."))
  (col [a j]
    (dragan-says-ex "You cannot access columns of a (tri)diagonal matrix."))
  (cols [a]
    (dragan-says-ex "You cannot access columns of a (tri)diagonal matrix."))
  (dia [a]
    (real-block-vector fact false buf n ofst 1))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (real-block-vector fact false buf (- n (Math/abs k)) (+ ofst (.index stor 0 k)) 1)
      (real-block-vector fact false buf 0 ofst 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (dragan-says-ex "You cannot create a submatrix of a (tri)diagonal matrix."
                    {:a (info a)}))
  (transpose [a]
    (if (or (= :gd matrix-type) (= :st matrix-type))
      a
      (dragan-says-ex "You cannot transpose a (tri)diagonal matrix.")))
  Triangularizable
  (create-trf [a pure]
    (case matrix-type
      :gd a
      :gt (lu-factorization a pure)
      :dt (pivotless-lu-factorization a pure)
      :st (pivotless-lu-factorization a pure)
      (dragan-says-ex "Triangular factorization is not available for this matrix type."
                      {:matrix-type matrix-type})))
  (create-ptrf [a]
    (if (= :sp matrix-type)
      (pivotless-lu-factorization a false)
      (dragan-says-ex "Pivotless factorization is not available for this matrix type."
                      {:matrix-type matrix-type})))
  TRF
  (trtrs [a b]
    (if (= :gd matrix-type)
      (let-release [res (raw b)]
        (copy (engine b) b res)
        (trs eng a res))
      (require-trf)))
  (trtrs! [a b]
    (if (= :gd matrix-type)
      (trs eng a b)
      (require-trf)))
  (trtri! [a]
    (if (= :gd matrix-type)
      (tri eng a)
      (require-trf)))
  (trtri [a]
    (if (= :gd matrix-type)
      (let-release [res (raw a)]
        (tri eng (copy eng a res)))
      (require-trf)))
  (trcon [a _ nrm1?]
    (if (= :gd matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trcon [a nrm1?]
    (if (= :gd matrix-type)
      (con eng a nrm1?)
      (require-trf)))
  (trdet [a]
    (if (= :gd matrix-type)
      (if (.isDiagUnit reg) 1.0 (fold (.dia a) f* 1.0))
      (require-trf))))

(extend RealDiagonalMatrix
  Functor
  {:fmap (diagonal-fmap RealBlockVector double)}
  PseudoFunctor
  {:fmap! (diagonal-fmap identity RealBlockVector double)}
  Foldable
  {:fold diagonal-fold
   :foldmap diagonal-foldmap}
  Magma
  {:op (constantly matrix-op)})

(defn real-diagonal-matrix
  ([fact master buf n ofst nav stor reg matrix-type default engine]
   (->RealDiagonalMatrix nav stor reg default fact (data-accessor fact) engine
                         matrix-type master buf n ofst))
  ([fact n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (real-diagonal-matrix fact true buf n 0 nav stor reg matrix-type default engine)))
  ([fact ^long n matrix-type]
   (let [kl (case matrix-type :st 0 :gd 0 :gt 1 :dt 1 1)
         ku (if (= :gd matrix-type) 0 1)]
     (real-diagonal-matrix fact n diagonal-navigator (diagonal-storage n matrix-type)
                           (band-region n n kl ku) matrix-type (real-default matrix-type)
                           (case matrix-type
                             :gd (gd-engine fact)
                             :gt (gt-engine fact)
                             :dt (dt-engine fact)
                             :st (st-engine fact)
                             (dragan-says-ex (format "%s is not a valid (tri)diagonal matrix type."
                                                     matrix-type)))))))

(defmethod print-method RealDiagonalMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (print-diagonal w a))

(defn ^:private transfer-diagonal [^Matrix source ^Matrix destination]
  (doseq [k (range -1 2)]
    (let [dia-src ^RealBlockVector (.dia source k)
          dia-dest ^RealBlockVector (.dia destination k)]
      (transfer-vector-vector dia-src dia-dest)))
  destination)

(defn ^:private transfer-diagonal-diagonal [source destination]
  (let [diag-src ^RealBlockVector (view-vctr source)
        diag-dest ^RealBlockVector (view-vctr destination)]
    (transfer-vector-vector diag-src diag-dest)
    destination))

(defmethod transfer! [RealNativeMatrix RealNativeMatrix]
  [^RealNativeMatrix source ^RealNativeMatrix destination]
  (transfer-matrix-matrix source destination))

(defmethod transfer! [RealPackedMatrix RealPackedMatrix]
  [^RealPackedMatrix source ^RealPackedMatrix destination]
  (transfer-matrix-matrix (= (navigator source) (navigator destination)) source destination))

(defmethod transfer! [RealDiagonalMatrix RealDiagonalMatrix]
  [source destination]
  (transfer-diagonal-diagonal source destination))

(defmethod transfer! [RealDiagonalMatrix RealNativeMatrix]
  [source destination]
  (transfer-diagonal source destination))

(defmethod transfer! [RealNativeMatrix RealDiagonalMatrix]
  [source destination]
  (transfer-diagonal source destination))

(defmethod transfer! [clojure.lang.Sequential RealNativeMatrix]
  [source ^RealNativeMatrix destination]
  (transfer-seq-matrix source destination))

(defmethod transfer! [RealNativeVector RealNativeMatrix]
  [^RealNativeVector source ^RealNativeMatrix destination]
  (transfer-vector-matrix source destination))

(defmethod transfer! [RealNativeMatrix RealNativeVector]
  [^RealNativeMatrix source ^RealBlockVector destination]
  (transfer-matrix-vector source destination))

(defmethod transfer! [IntegerNativeVector RealNativeMatrix]
  [^IntegerNativeVector source ^RealNativeMatrix destination]
  (transfer-vector-matrix source destination))

(defmethod transfer! [RealNativeMatrix IntegerNativeVector]
  [^RealNativeMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector source destination))

(defmethod transfer! [(Class/forName "[D") RealNativeMatrix]
  [^doubles source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [(Class/forName "[F") RealNativeMatrix]
  [^floats source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [(Class/forName "[J") RealNativeMatrix]
  [^longs source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [(Class/forName "[I") RealNativeMatrix]
  [^ints source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [(Class/forName "[S") RealNativeMatrix]
  [^shorts source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [(Class/forName "[B") RealNativeMatrix]
  [^bytes source ^RealNativeMatrix destination]
  (transfer-array-matrix source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[D")]
  [^RealNativeMatrix source ^doubles destination]
  (transfer-matrix-array source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[F")]
  [^RealNativeMatrix source ^floats destination]
  (transfer-matrix-array source destination))

(defn buf-capacity ^long [^Buffer buf]
  (.capacity buf))

(defn buf-direct? [^Buffer buf]
  (.isDirect buf))

(defmacro buf-attachment [type buf]
  `(.attachment ~(with-meta buf {:tag type})))

(defmacro extend-buffer [t fact buf-type only-direct]
  `(extend ~t
     Viewable
     {:view
      (fn [buf#]
        (if (or (not ~only-direct) (buf-direct? buf#))
          (create-vector ~fact false (buf-attachment ~buf-type buf#) (buf-capacity buf#) 0 1)
          (dragan-says-ex "This engine only supports direct buffers." {:buffer buf#})))}))

(extend-type ByteBuffer
  Viewable
  (view [b]
    (view (.asFloatBuffer b))))

(defn map-channel
  ([fact channel n flag offset-bytes]
   (let [fact (factory fact)
         entry-width (.entryWidth ^DataAccessor (data-accessor fact))]
     (let-release [buf (mapped-buffer channel offset-bytes (* (long n) entry-width) flag)]
       (create-vector fact true buf n 0 1))))
  ([fact channel n flag]
   (map-channel fact channel n flag 0))
  ([fact ^FileChannel channel n-or-flag]
   (if (integer? n-or-flag)
     (map-channel fact channel n-or-flag :read-write)
     (let [n (quot (.size channel) (.entryWidth ^DataAccessor (data-accessor fact)))]
       (map-channel fact channel n n-or-flag))))
  ([fact channel]
   (map-channel fact channel :read-write)))
