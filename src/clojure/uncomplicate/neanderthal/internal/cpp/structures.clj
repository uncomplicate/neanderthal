 ;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.structures
  (:refer-clojure :exclude [abs])
  (:require
   [uncomplicate.commons
    [core :refer [Releaseable release let-release Info info double-fn wrap-float wrap-double
                  Viewable view size]]
    [utils :refer [dragan-says-ex mapped-buffer]]]
   [uncomplicate.fluokitten.protocols :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative
                                              fold foldmap fmap fmap! Comonad extract]]
   [uncomplicate.clojure-cpp :refer [pointer fill! zero! float-pointer double-pointer long-pointer
                                     int-pointer short-pointer byte-pointer null?
                                     PointerCreator capacity! type-pointer]]
   [uncomplicate.neanderthal
    [core :refer [transfer! copy! dim subvector vctr ge matrix-type mrows ncols matrix-type
                  triangular? symmetric? dia]]
    [math :as math :refer [ceil]]
    [block :refer [entry-type offset stride buffer column?]]]
   [uncomplicate.neanderthal.internal
    [api :refer :all]
    [printing :refer [print-sequence print-vector print-ge print-uplo print-banded print-diagonal]]
    [common :refer :all]
    [navigation :refer :all]
    [fluokitten :refer :all]])
  (:import [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$LD IFn$LLD IFn$L IFn$LL
            IFn$LDD IFn$LLDD IFn$LDO IFn$LLDO IFn$LLL IFn$LLLL IFn$LLO IFn$LLLO]
           java.nio.channels.FileChannel
           org.bytedeco.mkl.global.mkl_rt
           [org.bytedeco.javacpp FloatPointer DoublePointer LongPointer IntPointer ShortPointer
            BytePointer]
           [uncomplicate.neanderthal.internal.api Block
            VectorSpace Vector RealVector Matrix IntegerVector DataAccessor RealChangeable
            IntegerChangeable RealNativeMatrix IntegerNativeMatrix RealNativeVector
            IntegerNativeVector DenseStorage FullStorage Default LayoutNavigator Region
            MatrixImplementation GEMatrix UploMatrix BandedMatrix PackedMatrix DiagonalMatrix
            RealAccessor IntegerAccessor RealLayoutFlipper IntegerLayoutFlipper]))

(declare real-block-vector integer-block-vector cs-vector integer-ge-matrix real-ge-matrix
         real-uplo-matrix integer-uplo-matrix real-banded-matrix integer-banded-matrix
         real-packed-matrix integer-packed-matrix real-diagonal-matrix integer-diagonal
         real-block-vector* integer-block-vector*)

(def f* (double-fn *))

;; ================ Pointer data accessors  ====================================

(defmacro put* [pt p i a]
  `(. ~(with-meta p {:tag pt}) put (long ~i) ~a))

(defmacro get* [pt p i]
  `(. ~(with-meta p {:tag pt}) get (long ~i)))

(defmacro def-accessor-type [name accessor-interface pointer-class entry-class pointer cast cast-get]
  `(deftype ~name [construct# destruct#]
     DataAccessor
     (entryType [_#]
       (. ~entry-class TYPE))
     (entryWidth [_#]
       (. ~entry-class BYTES))
     (count [_# p#]
       (size p#))
     (createDataSource [_# n#]
       (let [n# (max 1 n#)]
         (capacity! (~pointer (construct# (* (. ~entry-class BYTES) n#))) n#)))
     (initialize [_# p#]
       (zero! p#))
     (initialize [_# p# v#]
       (fill! p# (~cast v#)))
     (wrapPrim [_# v#]
       (pointer (~cast v#)))
     (castPrim [_# v#]
       (~cast v#))
     DataAccessorProvider
     (data-accessor [this#]
       this#)
     Destructor
     (destruct [_# p#]
       (if-not (null? p#)
         (destruct# p#)
         p#))
     PointerCreator
     (pointer* [_#]
       (~pointer nil))
     MemoryContext
     (compatible? [this# o#]
       (let [da# (data-accessor o#)]
         (or (identical? this# da#) (instance? ~name da#))))
     (device [_#]
       :cpu)
     ~accessor-interface
     (get [_# p# i#]
       (~cast-get (get* ~pointer-class p# i#)))
     (set [_# p# i# val#]
       (put* ~pointer-class p# i# val#)
       p#)))

(def-accessor-type DoublePointerAccessor RealAccessor DoublePointer Double double-pointer double double)
(def-accessor-type FloatPointerAccessor RealAccessor FloatPointer Float float-pointer float float)
(def-accessor-type LongPointerAccessor IntegerAccessor LongPointer Long long-pointer long long)
(def-accessor-type IntPointerAccessor IntegerAccessor IntPointer Integer int-pointer int int)
(def-accessor-type ShortPointerAccessor IntegerAccessor ShortPointer Short short-pointer short long)
(def-accessor-type BytePointerAccessor IntegerAccessor BytePointer Byte byte-pointer byte long)

;; =======================================================================

;; ================ from buffer-block ====================================

(defn vector-seq
  ([^Vector vector ^long i]
   (lazy-seq
    (if (< -1 i (.dim vector))
      (cons (.boxedEntry vector i) (vector-seq vector (inc i)))
      '())))
  ([^Vector vector]
   (if (not (null? (buffer vector)))
     (vector-seq vector 0)
     nil)))

(defmacro ^:private transfer-vector-vector [source destination]
  `(do
     (if (compatible? ~source ~destination)
       (when-not (identical? ~source ~destination)
         (let [n# (min (.dim ~source) (.dim ~destination))]
           (subcopy (engine ~source) ~source ~destination 0 n# 0)))
       (dotimes [i# (min (.dim ~source) (.dim ~destination))]
         (.set ~destination i# (.entry ~source i#))))
     ~destination))

(defmacro ^:private transfer-vector-array [cast source destination]
  `(let [n# (min (.dim ~source) (alength ~destination))]
     (dotimes [i# n#]
       (aset ~destination i# (~cast (.entry ~source i#))))
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

(defmacro ^:private transfer-seq-matrix [typed-accessor source destination]
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
     (let [da# (~typed-accessor ~destination)
           buf# (.buffer ~destination)
           ofst# (.offset ~destination)]
       (doseq-layout ~destination i# j# idx# ~source e# (.set da# buf# (+ ofst# idx#) e#))
       ~destination)))

(defmacro ^:private transfer-matrix-matrix
  ([typed-accessor source-flipper condition source destination]
   `(do
      (if (and (<= (.mrows ~destination) (.mrows ~source)) (<= (.ncols ~destination) (.ncols ~source)))
        (if (and (compatible? ~source ~destination) (fits? ~source ~destination) ~condition)
          (copy (engine ~source) ~source ~destination)
          (let [flipper# (~source-flipper (navigator ~destination))
                da# (~typed-accessor ~destination)
                buf# (.buffer ~destination)
                ofst# (.offset ~destination)]
            (doall-layout ~destination i# j# idx# (.set da# buf# (+ ofst# idx#) (.get flipper# ~source i# j#)))))
        (dragan-says-ex "There is not enough entries in the source matrix. Provide an appropriate submatrix of the destination."
                        {:source (info ~source) :destination (info ~destination)}))
      ~destination))
  ([typed-accessor source-flipper source destination]
   `(transfer-matrix-matrix ~typed-accessor ~source-flipper true ~source ~destination)))

(defmacro ^:private transfer-array-matrix [typed-accessor source destination]
  ` (let [da# (~typed-accessor ~destination)
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

(defmacro ^:private transfer-matrix-array [typed-accessor cast source destination]
  `(let [da# (~typed-accessor ~source)
         nav# (navigator ~source)
         stor# (storage ~source)
         reg# (region ~source)
         buf# (.buffer ~source)
         ofst# (.offset ~source)
         len# (alength ~destination)]
     (doall-layout nav# stor# reg# i# j# idx# cnt#
                   (when (< cnt# len#)
                     (aset ~destination cnt# (~cast (.get da# buf# (+ ofst# idx#))))))
     ~destination))

(defmacro ^:private transfer-vector-matrix [typed-accessor source destination]
  `(let [stor# (storage ~destination)]
     (if (and (compatible? ~source ~destination) (.isGapless stor#))
       (let [dst-view# ^VectorSpace (view-vctr ~destination)
             n# (min (.dim ~source) (.dim dst-view#))]
         (when (pos? n#)
           (subcopy (engine ~source) ~source dst-view# 0 n# 0)))
       (let [da# (~typed-accessor ~destination)
             nav# (navigator ~destination)
             reg# (region ~destination)
             buf# (.buffer ~destination)
             ofst# (.offset ~destination)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set da# buf# (+ ofst# idx#) (.entry ~source cnt#))))))
     ~destination))

(defmacro ^:private transfer-matrix-vector [typed-accessor source destination]
  `(let [stor# (storage ~source)]
     (if (and (compatible? ~destination ~source) (.isGapless stor#))
       (let [src-view# ^VectorSpace (view-vctr ~source)
             n# (min (.dim src-view#) (.dim ~destination))]
         (when (pos? n#)
           (subcopy (engine src-view#) src-view# ~destination 0 n# 0)))
       (let [da# (~typed-accessor ~source)
             nav# (navigator ~source)
             reg# (region ~source)
             buf# (.buffer ~source)
             ofst# (.offset ~source)
             dim# (.dim ~destination)]
         (doall-layout nav# stor# reg# i# j# idx# cnt#
                       (when (< cnt# dim#)
                         (.set ~destination cnt# (.get da# buf# (+ ofst# idx#)))))))
     ~destination))

(defmacro matrix-alter [ifn-oo ifn-lloo f nav stor reg da buf]
  `(if (instance? ~ifn-oo ~f)
     (doall-layout ~nav ~stor ~reg i# j# idx#
                   (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-oo}) (.get ~da ~buf idx#))))
     (if (.isRowMajor ~nav)
       (doall-layout ~nav ~stor ~reg i# j# idx#
                     (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-lloo})
                                                      j# i# (.get ~da ~buf idx#))))
       (doall-layout ~nav ~stor ~reg i# j# idx#
                     (.set ~da ~buf idx# (.invokePrim ~(with-meta f {:tag ifn-lloo})
                                                      i# j# (.get ~da ~buf idx#)))))))

;; =======================================================================

(defn block-vector
  ([constructor fact master buf-ptr n ofst strd]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 n (.count da buf-ptr))
       (constructor fact da (vector-engine fact) master buf-ptr n strd)
       (throw (ex-info "Insufficient buffer size." {:n n :offset ofst :buffer-size (.count da buf-ptr)})))))
  ([constructor fact master buf-ptr n strd]
   (block-vector constructor fact master buf-ptr n 0 strd))
  ([constructor fact n strd]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) n)]
     (block-vector constructor fact true buf-ptr n 0 strd)))
  ([constructor fact n]
   (block-vector constructor fact n 1)))

(defn block-vector* [constructor fact master buf-ptr n strd]
  (let [da (data-accessor fact)]
    (if (<= 0 n (.count da buf-ptr))
      (constructor fact da (vector-engine fact) master buf-ptr n strd)
      (throw (ex-info "Insufficient buffer size." {:n n :buffer-size (.count da buf-ptr)})))))

;; TODO move to general namespace
(defmacro extend-base [name]
  `(extend-type ~name
     Releaseable
     (release [this#]
       (when (.-master this#)
         (destruct (data-accessor this#) (buffer this#)))
       true)
     Comonad
     (extract [this#]
       (extract (.-buf-ptr this#)))
     EngineProvider
     (engine [this#]
       (.-eng this#))
     FactoryProvider
     (factory [this#]
       (.-fact this#))
     (native-factory [this#]
       (native-factory (.-fact this#)))
     (index-factory [this#]
       (index-factory (.-fact this#)))
     DataAccessorProvider
     (data-accessor [this#]
       (.-da this#))
     CompressedSparse
     (entries [this#]
       this#)
     (indices [this#]
       nil)))

;; TODO extract general cpu/gpu parts to a more general macro
(defmacro extend-vector [name block-vector ge-matrix]
  `(extend-type ~name
     Info
     (info
       ([this#]
        {:entry-type (.entryType (data-accessor this#))
         :class ~name
         :device (device this#)
         :dim (dim this#)
         :offset (offset this#)
         :stride (.-strd this#)
         :master (.-master this#)
         :engine (info (.-eng this#))})
       ([this# info-type#]
        (case info-type#
          :entry-type (.entryType (data-accessor this#))
          :class ~name
          :device (device this#)
          :dim (dim this#)
          :offset (offset this#)
          :stride (.-strd this#)
          :master (.-master this#)
          :engine (info (.-eng this#))
          nil)))
     Viewable
     (view [this#]
       (~block-vector (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0 (.-strd this#)))
     DenseContainer
     (view-vctr
       ([this#]
        this#)
       ([this# stride-mult#]
        (~block-vector (.-fact this#) false (.-buf-ptr this#)
         (ceil (/ (.-n this#) (long stride-mult#))) 0 (* (long stride-mult#) (.-strd this#)))))
     (view-ge
       ([this#]
        (~ge-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 1 0
         (layout-navigator true) (full-storage true (.-n this#) 1) (ge-region (.-n this#) 1)))
       ([this# stride-mult#]
        (view-ge (view-ge this#) stride-mult#))
       ([this# m# n#]
        (view-ge (view-ge this#) m# n#)))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# y#]
       (= (.-n this#) (dim y#)))
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~block-vector (.-fact this#) 0))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~block-vector (.-fact this#) 1)]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (vctr (.-fact this#) (cons v# vs#))))))

(defmacro extend-block-vector [name block-vector]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~block-vector (.-fact this#) (.-n this#)))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) false)))
     (zero
       ([this#]
        (create-vector (.-fact this#) (.-n this#) true))
       ([this# fact#]
        (create-vector (factory fact#) (.-n this#) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)))
     (native [this#]
       this#)))

(defmacro extend-vector-fluokitten [t cast indexed-fn]
  `(extend ~t
     Functor
     {:fmap (vector-fmap ~t ~cast)}
     PseudoFunctor
     {:fmap! (vector-fmap identity ~t ~cast)}
     Foldable
     {:fold (vector-fold ~t ~cast ~cast)
      :foldmap (vector-foldmap ~t ~cast ~cast ~indexed-fn)}
     Magma
     {:op (constantly vector-op)}))

;; ============ Integer Vector ====================================================

(deftype IntegerBlockVector [fact ^IntegerAccessor da eng master buf-ptr
                             ^long n ^long strd]
  Object
  (hashCode [x]
    (-> (hash :IntegerBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? IntegerBlockVector y) (compatible? da y) (fits? x y))
      (or (= buf-ptr (.buffer ^Block y))
          (loop [i 0]
            (if (< i n)
              (and (= (.entry x i) (.entry ^IntegerBlockVector y i)) (recur (inc i)))
              true)))
      :default false))
  (toString [_]
    (format "#IntegerBlockVector[%s, n:%d, stride:%d]" (entry-type da) n strd))
  Seqable
  (seq [x]
    (vector-seq x))
  IFn$LLO
  (invokePrim [x i v]
    (if (< -1 i n)
      (.set x i v)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$LL
  (invokePrim [x i]
    (if (< -1 i n)
      (.entry x i)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.invokePrim x i v))
  (invoke [x i]
    (.invokePrim x i))
  (invoke [x]
    n)
  IntegerChangeable
  (isAllowed [x i]
    (< -1 i n))
  (set [x val]
    (set-all eng val x)
    x)
  (set [x i val]
    (.set da buf-ptr (* strd i) val)
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
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    strd)
  (isContiguous [_]
    (or (= 1 strd) (= 0 strd)))
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf-ptr (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (integer-block-vector fact false buf-ptr l (* k strd) strd)))

(extend-base IntegerBlockVector)
(extend-vector IntegerBlockVector integer-block-vector integer-ge-matrix)
(extend-block-vector IntegerBlockVector integer-block-vector)
(extend-vector-fluokitten IntegerBlockVector long IFn$LLL)

(def integer-block-vector (partial block-vector ->IntegerBlockVector))
(def integer-block-vector* (partial block-vector* ->IntegerBlockVector))

(defmethod print-method IntegerBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (str x))
  (when-not (null? (buffer x))
    (print-sequence w x)))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [fact ^RealAccessor da eng master buf-ptr
                          ^long n ^long strd]
  Object
  (hashCode [x]
    (-> (hash :RealBlockVector) (hash-combine n) (hash-combine (nrm2 eng x))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? RealBlockVector y) (compatible? da y) (fits? x y))
      (or (= buf-ptr (.buffer ^Block y))
          (loop [i 0]
            (if (< i n)
              (and (= (.entry x i) (.entry ^RealBlockVector y i)) (recur (inc i)))
              true)))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, stride:%d]" (entry-type da) n strd))
  Seqable
  (seq [x]
    (vector-seq x))
  IFn$LDO
  (invokePrim [x i v]
    (if (< -1 i n)
      (.set x i v)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$LD
  (invokePrim [x i]
    (if (< -1 i n)
      (.entry x i)
      (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim n}))))
  IFn$L
  (invokePrim [x]
    n)
  IFn
  (invoke [x i v]
    (.invokePrim x i v))
  (invoke [x i]
    (.invokePrim x i))
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
    (.set da buf-ptr (* strd i) val)
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
    buf-ptr)
  (offset [_]
    0)
  (stride [_]
    strd)
  (isContiguous [_]
    (= 1 strd))
  (dim [_]
    n)
  (entry [_ i]
    (.get da buf-ptr (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (real-block-vector fact false buf-ptr l (* k strd) strd)))

(extend-base RealBlockVector)
(extend-vector RealBlockVector real-block-vector real-ge-matrix)
(extend-block-vector RealBlockVector real-block-vector)
(extend-vector-fluokitten RealBlockVector double IFn$LDO)

(def real-block-vector (partial block-vector ->RealBlockVector))
(def real-block-vector* (partial block-vector* ->RealBlockVector))

(defmethod print-method RealBlockVector [^Vector x ^java.io.Writer w]
  (.write w (str x))
  (when-not (null? (buffer x))
    (print-vector w x)))

;; ======================= Compressed Sparse Vector ======================================

(deftype CSVector [fact eng ^Block nzx ^IntegerVector indx ^long n]
  Object
  (hashCode [_]
    (-> (hash :CSVector) (hash-combine nzx) (hash-combine indx)))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (instance? CSVector y)
      (and (= n (dim y)) (= nzx (entries y)) (= indx (indices y)))
      :default false))
  (toString [_]
    (format "#CSVector[%s, n:%d, nnz:%d]" (entry-type (data-accessor nzx)) n (dim nzx)))
  Info
  (info [x]
    {:entry-type (.entryType (data-accessor nzx))
     :class (class x)
     :device (info nzx :device)
     :dim n
     :engine (info eng)})
  (info [x info-type]
    (case info-type
      :entry-type (.entryType (data-accessor nzx))
      :class (class x)
      :device (info nzx :device)
      :dim n
      :engine (info eng)
      nil))
  Releaseable
  (release [_]
    (release nzx)
    (release indx)
    true)
  Seqable
  (seq [x]
    (vector-seq nzx))
  MemoryContext
  (compatible? [_ y]
    (compatible? nzx (entries y)))
  (fits? [_ y]
    (and (= n (dim y))
         (or (nil? (indices y)) (= indx (indices y)))))
  (device [_]
    (device nzx))
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
    (data-accessor fact))
  Container
  (raw [x]
    (raw x fact))
  (raw [_ fact]
    (cs-vector (factory fact) n (view indx) false))
  (zero [x]
    (zero x fact))
  (zero [_ fact]
    (cs-vector (factory fact) n (view indx) true))
  (host [x]
    (let-release [host-nzx (host nzx)
                  host-indx (host indx)]
      (cs-vector n host-indx host-nzx)))
  (native [x]
    (let-release [native-nzx (native nzx)]
      (if (= nzx native-nzx)
        x
        (let-release [native-indx (native indx)]
          (cs-vector n native-indx native-nzx)))))
  Viewable
  (view [x]
    (cs-vector n (view indx) (view nzx)))
  DenseContainer
  (view-vctr [_]
    nzx)
  (view-vctr [_ stride-mult]
    (view-vctr nzx stride-mult))
  (view-ge [_]
    (view-ge nzx))
  (view-ge [_ stride-mult]
    (view-ge nzx stride-mult))
  (view-ge [_ m n]
    (view-ge nzx m n))
  Block
  (buffer [_]
    (.buffer nzx))
  (offset [_]
    (.offset nzx))
  (stride [_]
    (.stride nzx))
  (isContiguous [_]
    (.isContiguous ^Block nzx))
  VectorSpace
  (dim [_]
    n)
  CompressedSparse
  (entries [_]
    nzx)
  (indices [_]
    indx)
  Monoid
  (id [_]
    (cs-vector fact 0))
  Applicative
  (pure [_ v]
    (let-release [res (cs-vector fact 1 (vctr (index-factory fact) 1) false)]
      (uncomplicate.neanderthal.core/entry! (indices res) 0 0)
      (uncomplicate.neanderthal.core/entry! (entries res) 0 v)
      res))
  (pure [_ v vs]
    (let [cnt (inc (count vs))]
      (let-release [res (cs-vector fact cnt (vctr (index-factory fact) cnt) false)]
        (transfer! (range) (indices res))
        (transfer! (cons v vs) (entries res))
        res)))
  ;; TODO Magma
  )

(extend-type RealBlockVector
  CompressedSparse
  (entries [this]
    this)
  (indices [this]
    nil))

(defn cs-vector
  ([^long n indx nzx]
   (let [fact (factory nzx)]
     (if (= (factory indx) (index-factory nzx))
       (if (and (<= 0 (dim nzx) n) (<= 0 (dim indx)) (fits? nzx indx)
                (= 1 (stride indx) (stride nzx)) (= 0 (offset indx) (offset nzx)))
         (->CSVector fact (cs-vector-engine fact) nzx indx n)
         (throw (ex-info "Non-zero vector and index vector have to fit each other." {:nzx nzx :indx indx})));;TODO error message
       (throw (ex-info "Incompatible index vector" {:required (index-factory nzx) :provided (factory indx)})))))
  ([fact ^long n indx init]
   (let-release [nzx (create-vector fact (dim indx) init)]
     (cs-vector n indx nzx)))
  ([fact ^long n]
   (let-release [indx (create-vector (index-factory fact) 0 false)
                 nzx (create-vector fact 0 false)]
     (cs-vector n indx nzx))))

(defn check-indices ;;TODO macroize
  ([x y]
   (when-not (= (indices x) (indices y))
     (throw (dragan-says-ex "Compressed sparse operation with incompatible indices is not allowed." {:x x :y y}))))
  ([x y z]
   (when-not (= (indices x) (indices y) (indices z))
     (throw (dragan-says-ex "Compressed sparse operation with incompatible indices is not allowed." {:x x :y y :z z}))))
  ([x y z w]
   (when-not (= (indices x) (indices y) (indices z) (indices w))
     (throw (dragan-says-ex "Compressed sparse operation with incompatible indices is not allowed." {:x x :y y :z z :w w}))))
  ([x y z w ws]
   (when-not (apply = (indices x) (indices y) (indices z) (indices w) (map indices ws))
     (throw (dragan-says-ex "Compressed sparse operation with incompatible indices is not allowed." {:x x :y y :z z :w w :ws ws})))))

(extend-type CSVector
  Functor
  (fmap
    ([x g]
     (cs-vector (dim x) (view (indices x)) (fmap (entries x) g)))
    ([x g ys]
     (if (apply = (indices x) (map indices ys))
       (cs-vector (dim x) (view (indices x)) (fmap (entries x) g (entries ys)))
       (throw (dragan-says-ex "Compressed sparse operation with incompatible indices is not allowed." {:x x :ys ys})))))
  PseudoFunctor
  (fmap!
    ([x f]
     (fmap! (entries x) f)
     x)
    ([x f y]
     (check-indices x y)
     (fmap! (entries x) f (entries y))
     x)
    ([x f y z]
     (check-indices x y z)
     (fmap! (entries x) f (entries y) (entries z))
     x)
    ([x f y z v]
     (check-indices x y z v)
     (fmap! (entries x) f (entries y) (entries z) (entries v))
     x)
    ([x f y z v ws]
     (check-indices x y z v ws)
     (fmap! (entries x) f (entries y) (entries z) (entries v) (entries ws))
     x))
  Foldable
  (fold
    ([x]
     (fold (entries x)))
    ([x f init]
     (fold (entries x) f init))
    ([x f init y]
     (check-indices x y)
     (fold (entries x) f init (entries y)))
    ([x f init y z]
     (check-indices x y z)
     (fold (entries x) f init (entries y) (entries z)))
    ([x f init y z v]
     (check-indices x y z v)
     (fold (entries x) f init (entries y) (entries z) (entries v)))
    ([x f init y z v ws]
     (check-indices x y z v ws)
     (fold (entries x) f init (entries y) (entries z) (entries v) (entries ws))))
  (foldmap
    ([x g]
     (foldmap (entries x) g))
    ([x g f init]
     (foldmap (entries x) g f init))
    ([x g f init y]
     (check-indices x y)
     (foldmap (entries x) g f init (entries y)))
    ([x g f init y z]
     (check-indices x y z)
     (foldmap (entries x) g f init (entries y) (entries z)))
    ([x g f init y z v]
     (check-indices x y z v)
     (foldmap (entries x) g f init (entries y) (entries z) (entries v)))
    ([x g f init y z v ws]
     (check-indices x y z v ws)
     (foldmap (entries x) g f init (entries y) (entries z) (entries v) (entries ws)))))

(defmethod print-method CSVector [^Vector x ^java.io.Writer w]
  (.write w (format "%s\n" (str x)))
  (when-not (null? (buffer (indices x)))
    (print-sequence w (indices x)))
  (when-not (null? (buffer (entries x)))
    (print-vector w (entries x))))

(defmethod transfer! [CSVector CSVector]
  [source destination]
  (transfer! (entries source) (entries destination))
  destination)

(defn transfer-seq-csvector [source destination]
  (let [[s0 s1] source]
    (if (number? s0)
      (transfer! source (entries destination))
      (do
        (transfer! s0 (indices destination))
        (if (sequential? s1)
          (transfer! s1 (entries destination))
          (transfer! (rest source) (entries destination)))))))

(defmethod transfer! [clojure.lang.Sequential CSVector]
  [source destination]
  (transfer-seq-csvector source destination)
  destination)

(defmethod transfer! [(Class/forName "[D") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[F") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[J") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [(Class/forName "[I") CSVector]
  [source destination]
  (transfer! source (entries destination))
  destination)

(defmethod transfer! [CSVector (Class/forName "[D")]
  [source destination]
  (transfer! (entries source) destination))

(defmethod transfer! [CSVector (Class/forName "[F")]
  [source destination]
  (transfer! (entries source) destination))

;;TODO handle heterogenous types (float/double...)
(defmethod transfer! [RealBlockVector CSVector]
  [^RealBlockVector source ^CSVector destination]
  (gthr (engine destination) source destination)
  destination)

;; =================== Matrices ================================================

(defmacro extend-matrix [name]
  `(extend-type ~name
     Info
     (info
       ([this#]
        {:entry-type (.entryType (data-accessor this#))
         :class ~name
         :device :cpu
         :matrix-type (.matrixType this#)
         :dim (dim this#)
         :m (mrows this#)
         :n (ncols this#)
         :offset (offset this#)
         :stride (stride this#)
         :master (.-master this#)
         :layout (:layout (info (.-nav this#)))
         :storage (info (.-nav this#))
         :region (info (.-reg this#))
         :engine (info (.-eng this#))})
       ([this# info-type#]
        (case info-type#
          :entry-type (.entryType (data-accessor this#))
          :class ~name
          :device :cpu
          :matrix-type (.matrixType this#)
          :dim (dim this#)
          :m (mrows this#)
          :n (ncols this#)
          :offset (offset this#)
          :stride (stride this#)
          :master (.-master this#)
          :layout (:layout (info (.-nav this#)))
          :storage (info (.-nav this#))
          :region (info (.-reg this#))
          :engine (info (.-eng this#))
          nil)))
     Navigable
     (navigator [this#]
       (.-nav this#))
     (storage [this#]
       (.-stor this#))
     (region [this#]
       (.-reg this#))))

(defmacro extend-ge-matrix [name block-vector ge-matrix uplo-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~ge-matrix (.-fact this#) (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? this#) false)))
     (zero
       ([this#]
        (create-ge (.-fact this#) (.-m this#) (.-n this#) (column? this#) true))
       ([this# fact#]
        (create-ge (factory fact#) (.-m this#) (.-n this#) (column? this#) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~ge-matrix (.-fact this#) false (.-buf-ptr this#)
        (.-m this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (if (.isContiguous this#)
          (~block-vector (.-fact this#) false (.-buf-ptr this#) (.dim this#) 0 1)
          (throw (ex-info "Strided GE matrix cannot be viewed as a dense vector." {:a (info this#)}))))
       ([this# stride-mult#]
        (view-vctr (view-vctr this#) stride-mult#)))
     (view-ge
       ([this#]
        this#)
       ([this# stride-mult#]
        (let [shrinked# (ceil (/ (.invokePrim this#) (long stride-mult#)))
              column-major# (column? this#)
              [m# n#] (if column-major# [(.-m this#) shrinked#] [shrinked# (.-n this#)])]
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) m# n# (.-nav this#)
           (full-storage column-major# m# n# (* (long stride-mult#) (stride this#)))
           (ge-region m# n#))))
       ([this# m# n#]
        (if (.isContiguous this#)
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) m# n# (.-nav this#)
           (full-storage (column? this#) m# n#) (ge-region m# n#))
          (throw (ex-info "Strided GE matrix cannot be viewed through different dimensions."
                          {:a (info this#)})))))
     (view-tr [this# lower?# diag-unit?#]
       (let [n# (min (.-m this#) (.-n this#))
             fact# (.-fact this#)]
         (~uplo-matrix fact# false (.-buf-ptr this#) n# (.-nav this#)
          (full-storage (column? this#) n# n# (.ld (full-storage this#)))
          (band-region n# lower?# diag-unit?#) :tr (default :tr diag-unit?#)
          (tr-engine fact#))))
     (view-sy [this# lower?#]
       (let [n# (min (.-m this#) (.-n this#))
             fact# (.-fact this#)]
         (~uplo-matrix fact# false (.-buf-ptr this#) n# (.-nav this#)
          (full-storage (column? this#) n# n# (.ld (full-storage this#)))
          (band-region n# lower?#) :sy sy-default (sy-engine fact#))))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# b#]
       (and (instance? GEMatrix b#) (= (.-reg this#) (region b#))))
     (fits-navigation? [this# b#]
       (= (.-nav this#) (navigator b#)))
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~ge-matrix (.-fact this#) 0 0 (column? this#)))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~ge-matrix (.-fact this#) 1 1 (column? this#))]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (ge (.-fact this#) (cons v# vs#))))))

(defmacro extend-ge-trf [name]
  `(extend-type ~name
     Triangularizable
     (create-trf [a# pure#]
       (lu-factorization a# pure#))
     (create-ptrf [a#]
       (dragan-says-ex "Pivotless factorization is not available for GE matrices."))
     TRF
     (trtrs! [a# b#]
       (require-trf))
     (trtrs [a# b#]
       (require-trf))
     (trtri! [a#]
       (require-trf))
     (trtri [a#]
       (require-trf))
     (trcon
       ([a# nrm# nrm1?#]
        (require-trf))
       ([a# nrm1?#]
        (require-trf)))
     (trdet [a#]
       (require-trf))))

(defmacro extend-matrix-fluokitten [t cast typed-flipper typed-accessor]
  `(extend ~t
     Functor
     {:fmap (matrix-fmap ~typed-flipper ~typed-accessor ~cast)}
     PseudoFunctor
     {:fmap! (matrix-fmap ~typed-flipper ~typed-accessor identity ~cast)}
     Foldable
     {:fold (matrix-fold ~typed-flipper ~cast)
      :foldmap (matrix-foldmap ~typed-flipper ~cast)}
     Magma
     {:op (constantly matrix-op)}))

;; =================== Real Matrix =============================================

(defmacro matrix-equals [flipper da a b]
  `(or (identical? ~a ~b)
       (and (instance? (class ~a) ~b)
            (= (.matrixType ~a) (matrix-type ~b))
            (compatible? ~a ~b) (= (.mrows ~a) (mrows ~b)) (= (.ncols ~a) (ncols ~b))
            (let [buff-a# (.buffer ~a)]
              (or (= buff-a# (buffer ~b))
                  (and-layout ~a i# j# idx# (= (.get ~da buff-a# idx#) (.get ~flipper ~b i# j#))))))))

(deftype RealGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg
                       fact ^RealAccessor da eng master buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (real-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#RealGEMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type da) m n (dec-property (.layout nav))))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 (.fd stor))))
  IFn$LLDO
  (invokePrim [a i j v]
    (if (and (< -1 i m) (< -1 j n))
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i m) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
  (invoke [a]
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  RealChangeable
  (isAllowed [a i j]
    (and (< -1 i m) (< -1 j n)))
  (set [a val]
    (if-not (Double/isNaN val)
      (set-all eng val a)
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$DD IFn$LLDD f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
    (.get da buf-ptr (.index nav stor i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (real-block-vector fact false buf-ptr n (.index nav stor i 0)
                       (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (real-block-vector fact false buf-ptr m (.index nav stor 0 j)
                       (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf-ptr (min m n) 0 (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (real-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (inc (.ld stor)))
      (real-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (real-ge-matrix fact false buf-ptr k l (.index nav stor i j) nav
                    (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (real-ge-matrix fact false buf-ptr n m 0 (flip nav) stor (flip reg))))

(extend-base RealGEMatrix)
(extend-matrix RealGEMatrix)
(extend-ge-matrix RealGEMatrix real-block-vector real-ge-matrix real-uplo-matrix)
(extend-ge-trf RealGEMatrix)
(extend-matrix-fluokitten RealGEMatrix double real-flipper real-accessor)

(defmethod print-method RealGEMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-ge w a)))

;; =================== Integer GE Matrix =============================================

(deftype IntegerGEMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg
                          fact ^IntegerAccessor da eng master buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :IntegerGEMatrix) (hash-combine m) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (integer-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#IntegerGEMatrix[%s, mxn:%dx%d, layout%s]"
            (entry-type da) m n (dec-property (.layout nav))))
  GEMatrix
  (matrixType [_]
    :ge)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    false)
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 (.fd stor))))
  IFn$LLLO
  (invokePrim [a i j v]
    (if (and (< -1 i m) (< -1 j n))
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn$LLL
  (invokePrim [a i j]
    (if (and (< -1 i m) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows m :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
  (invoke [a]
    (.fd stor))
  IFn$L
  (invokePrim [a]
    (.fd stor))
  IntegerChangeable
  (isAllowed [a i j]
    (and (< -1 i m) (< -1 j n)))
  (set [a val]
    (set-all eng val a)
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$LL IFn$LLLL f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$LL f (.get da buf-ptr idx)))
      a))
  IntegerNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
    (.get da buf-ptr (.index nav stor i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (integer-block-vector fact false buf-ptr n (.index nav stor i 0)
                          (if (.isRowMajor nav) 1 (.ld stor))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (integer-block-vector fact false buf-ptr m (.index nav stor 0 j)
                          (if (.isColumnMajor nav) 1 (.ld stor))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (integer-block-vector fact false buf-ptr (min m n) 0 (inc (.ld stor))))
  (dia [a k]
    (if (< 0 k)
      (integer-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (inc (.ld stor)))
      (integer-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (inc (.ld stor)))))
  (dias [a]
    (dense-dias a))
  (submatrix [a i j k l]
    (integer-ge-matrix fact false buf-ptr k l (.index nav stor i j)
                       nav (full-storage (.isColumnMajor nav) k l (.ld stor)) (ge-region k l)))
  (transpose [a]
    (integer-ge-matrix fact false buf-ptr n m 0 (flip nav) stor (flip reg))))

(extend-base IntegerGEMatrix)
(extend-matrix IntegerGEMatrix)
(extend-ge-matrix IntegerGEMatrix integer-block-vector integer-ge-matrix integer-uplo-matrix)
(extend-ge-trf IntegerGEMatrix)
(extend-matrix-fluokitten IntegerGEMatrix long integer-flipper integer-accessor)

(defmethod print-method IntegerGEMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-ge w a)))

(defn ge-matrix
  ([constructor fact master buf-ptr m n ofst nav ^FullStorage stor reg]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg fact da (ge-engine fact) master buf-ptr m n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)})))))
  ([constructor fact master buf-ptr m n nav ^FullStorage stor reg]
   (ge-matrix constructor fact master buf-ptr m n 0 nav stor reg))
  ([constructor fact m n nav ^FullStorage stor reg]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) (.capacity stor))]
     (ge-matrix constructor fact true buf-ptr m n 0 nav stor reg)))
  ([constructor fact m n column?]
   (ge-matrix constructor fact m n (layout-navigator column?) (full-storage column? m n) (ge-region m n)))
  ([constructor fact m n]
   (ge-matrix constructor fact m n true)))

(def real-ge-matrix (partial ge-matrix ->RealGEMatrix))
(def integer-ge-matrix (partial ge-matrix ->IntegerGEMatrix))

;; =================== Real Uplo Matrix ==================================

(defmacro extend-uplo-matrix [name block-vector ge-matrix uplo-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~uplo-matrix (.-fact this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)
         (.-matrix-type this#) (.-default this#) (.-eng this#)))
       ([this# fact#]
        (create-uplo (factory fact#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) false)))
     (zero
       ([this#]
        (create-uplo (.-fact this#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true))
       ([this# fact#]
        (create-uplo (factory fact#) (.-n this#) (.-matrix-type this#) (column? this#)
                     (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~uplo-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) (.-nav this#) (.-stor this#)
        (.-reg this#) (.-matrix-type this#) (.-default this#) (.-eng this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (view-vctr (view-ge this#)))
       ([this# stride-mult#]
        (view-vctr (view-ge this#) stride-mult#)))
     (view-ge
       ([this#]
        (let [n# (.-n this#)]
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) n# n#
           (.-nav this#) (.-stor this#) (ge-region n# n#))))
       ([this# stride-mult#]
        (view-ge (view-ge this#) stride-mult#))
       ([this# m# n#]
        (view-ge (view-ge this#) m# n#)))
     (view-tr [this# lower?# diag-unit?#]
       (let [n# (.-n this#)
             fact# (.-fact this#)]
         (~uplo-matrix fact# false (.-buf-ptr this#) n# (.-nav this#) (.-stor this#)
          (band-region n# lower?# diag-unit?#) :tr (default :tr diag-unit?#) (tr-engine fact#))))
     (view-sy [this# lower?#]
       (let [n# (.-n this#)
             fact# (.-fact this#)]
         (~uplo-matrix fact# false (.-buf-ptr this#) n# (.-nav this#) (.-stor this#)
          (band-region n# lower?#) :sy sy-default (sy-engine fact#))))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# b#]
       (and (instance? UploMatrix b#)
            (let [reg# (.-reg this#)
                  reg-b# (region b#)]
              (or (= reg# reg-b#)
                  (and (= :sy (.-matrix-type this#)) (matrix-type b#)
                       (not= (.-nav this#) (navigator b#)) (not (uplo= reg# reg-b#))
                       (= (.-n this#) (ncols b#)))))))
     (fits-navigation? [this# b#]
       (and (= (.-nav this#) (navigator b#))
            (or (instance? GEMatrix b#) (= (.-reg this#) (region b#)))))
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~uplo-matrix (.-fact this#) 0 (column? this#)))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~uplo-matrix (.-fact this#) 1 (column? this#) (.-matrix-type this#))]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (let [source# (cons v# vs#)]
          (let-release [res# (~uplo-matrix (.-fact this#) (math/sqrt (count source#))
                              (column? this#) (.-matrix-type this#))]
            (transfer! source# res#)))))))

(defmacro extend-uplo-triangularizable [name]
  `(extend-type ~name
     Triangularizable
     (create-trf [a# pure#]
       (if (symmetric? a#)
         (dual-lu-factorization a# pure#)
         a#))
     (create-ptrf [a#]
       (if (symmetric? a#)
         (pivotless-lu-factorization a# false)
         a#))))

(defmacro extend-trf [name]
  `(extend-type ~name
     TRF
     (trtrs [a# b#]
       (if (triangular? a#)
         (let-release [res# (raw b#)]
           (copy (engine b#) b# res#)
           (trs (.-eng a#) a# res#))
         (require-trf)))
     (trtrs! [a# b#]
       (if (triangular? a#)
         (trs (.-eng a#) a# b#)
         (require-trf)))
     (trtri! [a#]
       (if (triangular? a#)
         (tri (.-eng a#) a#)
         (require-trf)))
     (trtri [a#]
       (if (triangular? a#)
         (let-release [res# (raw a#)
                       eng# (.-eng a#)]
           (tri eng# (copy eng# a# res#)))
         (require-trf)))
     (trcon
       ([a# _# nrm1?#]
        (if (triangular? a#)
          (con (.-eng a#) a# nrm1?#)
          (require-trf)))
       ([a# nrm1?#]
        (if (triangular? a#)
          (con (.-eng a#) a# nrm1?#)
          (require-trf))))
     (trdet [a#]
       (if (triangular? a#)
         (if (diag-unit? (.-reg a#)) 1.0 (fold (dia a#) f* 1.0))
         (require-trf)))))

(deftype RealUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^Default default
                         fact ^RealAccessor da eng matrix-type master buf-ptr ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealUploMatrix) (hash-combine matrix-type) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (real-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#RealUploMatrix[%s, type%s, mxn:%dx%d, layout%s]"
            (entry-type da) matrix-type n n (dec-property (.layout nav))))
  UploMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tr matrix-type))
  (isSymmetric [_]
    (= :sy matrix-type))
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 n)))
  IFn$LLDO
  (invokePrim [a i j v]
    (if (.accessible reg i j)
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
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
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$DD IFn$LLDD f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
      (.get da buf-ptr (.index nav stor i j))
      (.realEntry default nav stor da buf-ptr 0 i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart reg i)]
      (real-block-vector fact false buf-ptr (- (.rowEnd reg i) start) (.index nav stor i start)
                         (if (.isRowMajor nav) 1 (.ld stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (real-block-vector fact false buf-ptr (- (.colEnd reg j) start) (.index nav stor start j)
                         (if (.isColumnMajor nav) 1 (.ld stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (real-block-vector fact false buf-ptr n 0 (inc (.ld stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (real-block-vector fact false buf-ptr (- n k) (.index nav stor 0 k) (inc (.ld  stor)))
        (real-block-vector fact false buf-ptr (+ n k) (.index nav stor (- k) 0) (inc (.ld stor))))
      (real-block-vector fact false buf-ptr 0 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (real-uplo-matrix fact false buf-ptr k (.index nav stor i j) nav
                        (full-storage (.isColumnMajor nav) k k (.ld stor))
                        (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (real-uplo-matrix fact false buf-ptr n (flip nav) stor (flip reg) matrix-type default eng)))

(extend-base RealUploMatrix)
(extend-matrix RealUploMatrix)
(extend-uplo-matrix RealUploMatrix real-block-vector real-ge-matrix real-uplo-matrix)
(extend-uplo-triangularizable RealUploMatrix)
(extend-trf RealUploMatrix)
(extend-matrix-fluokitten RealUploMatrix double real-flipper real-accessor)

(defmethod print-method RealUploMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-uplo w a "*")))

;; =================== Integer Uplo Matrix ==================================

(deftype IntegerUploMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^Default default
                            fact ^IntegerAccessor da eng matrix-type master buf-ptr ^long n]
  Object
  (hashCode [a]
    (-> (hash :IntegerUploMatrix) (hash-combine n) (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (integer-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#IntegerUploMatrix[%s, type%s, mxn:%dx%d, layout%s]"
            (entry-type da) matrix-type n n (dec-property (.layout nav))))
  UploMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tr matrix-type))
  (isSymmetric [_]
    (= :sy matrix-type))
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 n)))
  IFn$LLLO
  (invokePrim [a i j v]
    (if (and (< -1 i n) (< -1 j n))
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn$LLL
  (invokePrim [a i j]
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  IntegerChangeable
  (isAllowed [a i j]
    (.accessible reg i j))
  (set [a val]
    (set-all eng val a)
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$LL IFn$LLLL f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$LL f (.get da buf-ptr idx)))
      a))
  IntegerNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
      (.get da buf-ptr (.index nav stor i j))
      (.integerEntry default nav stor da buf-ptr 0 i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart reg i)]
      (integer-block-vector fact false buf-ptr (- (.rowEnd reg i) start) (.index nav stor i start)
                            (if (.isRowMajor nav) 1 (.ld stor)))))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (integer-block-vector fact false buf-ptr (- (.colEnd reg j) start) (.index nav stor start j)
                            (if (.isColumnMajor nav) 1 (.ld stor)))))
  (cols [a]
    (dense-cols a))
  (dia [a]
    (integer-block-vector fact false buf-ptr n 0 (inc (.ld stor))))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (integer-block-vector fact false buf-ptr (- n k) (.index nav stor 0 k) (inc (.ld  stor)))
        (integer-block-vector fact false buf-ptr (+ n k) (.index nav stor (- k) 0) (inc (.ld stor))))
      (integer-block-vector fact false buf-ptr 0 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (integer-uplo-matrix fact false buf-ptr k (.index nav stor i j) nav
                           (full-storage (.isColumnMajor nav) k k (.ld stor))
                           (band-region k (.isLower reg) (.isDiagUnit reg)) matrix-type default eng)
      (dragan-says-ex "You cannot create a non-uplo submatrix of a uplo (TR or SY) matrix. Take a view-ge."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (integer-uplo-matrix fact false buf-ptr n (flip nav) stor (flip reg) matrix-type default eng)))

(extend-base IntegerUploMatrix)
(extend-matrix IntegerUploMatrix)
(extend-uplo-matrix IntegerUploMatrix integer-block-vector integer-ge-matrix integer-uplo-matrix)
(extend-uplo-triangularizable IntegerUploMatrix)
(extend-trf IntegerUploMatrix)
(extend-matrix-fluokitten IntegerUploMatrix long integer-flipper integer-accessor)

(defn uplo-matrix
  ([constructor fact master buf-ptr n ofst nav ^FullStorage stor reg matrix-type default engine]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg default fact da engine matrix-type master buf-ptr n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)})))))
  ([constructor fact master buf-ptr n nav stor reg matrix-type default engine]
   (uplo-matrix constructor fact master buf-ptr n 0 nav stor reg matrix-type default engine))
  ([constructor fact n nav ^FullStorage stor reg matrix-type default engine]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) (.capacity stor))]
     (uplo-matrix constructor fact true buf-ptr n 0 nav stor reg matrix-type default engine)))
  ([constructor fact n column? lower? diag-unit? matrix-type]
   (uplo-matrix constructor fact n (layout-navigator column?) (full-storage column? n n)
                (band-region n lower? diag-unit?) matrix-type (default matrix-type diag-unit?)
                (case matrix-type
                  :tr (tr-engine fact)
                  :sy (sy-engine fact)
                  (dragan-says-ex (format "%s is not a valid UPLO matrix type. Please send me a bug report."
                                          matrix-type)
                                  {:type matrix-type}))))
  ([constructor fact n column? lower? diag-unit?]
   (uplo-matrix constructor fact n (layout-navigator column?) (full-storage column? n n)
                (band-region n lower? diag-unit?) :tr (default :tr diag-unit?) (tr-engine fact)))
  ([constructor fact n column? lower?]
   (uplo-matrix constructor fact n (layout-navigator column?) (full-storage column? n n)
                (band-region n lower?) :sy (default :sy) (sy-engine fact))))

(def real-uplo-matrix (partial uplo-matrix ->RealUploMatrix))
(def integer-uplo-matrix (partial uplo-matrix ->IntegerUploMatrix))

;; ================= Banded Matrix ==============================================================

(defn create-banded* [this fact master]
  (let [reg (region this)]
    (create-banded (factory fact) (mrows this) (ncols this) (.kl reg) (.ku reg)
                   (matrix-type this) (column? this) master)))

(defmacro extend-banded-matrix [name block-vector ge-matrix banded-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~banded-matrix (.-fact this#) (.-m this#) (.-n this#) (.-nav this#) (.-stor this#)
         (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#)))
       ([this# fact#]
        (create-banded* this# (.-fact this#) false)))
     (zero
       ([this#]
        (create-banded* this# (.-fact this#) true))
       ([this# fact#]
        (create-banded* this# fact# true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~banded-matrix (.-fact this#) false (.-buf-ptr this#) (.-m this#) (.-n this#) 0 (.-nav this#)
        (.-stor this#) (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (view-vctr (view-ge this#)))
       ([this# stride-mult#]
        (view-vctr (view-ge this#) stride-mult#)))
     (view-ge
       ([this#]
        (let [stor# (full-storage this#)
              m# (if (column? this#) (.sd stor#) (.fd stor#))
              n# (if (column? this#) (.fd stor#) (.sd stor#))]
          (~ge-matrix (.-fact this#) false (.-buf-ptr this#) m# n# (.-nav this#)
           (full-storage (column? this#) m# n# (.ld stor#)) (ge-region m# n#))))
       ([this# stride-mult#]
        (dragan-says-ex "GB matrices do not support stride when viewed as GE."))
       ([this# m# n#]
        (if (<= (* (long m#) (long n#)) (.capacity (storage this#)))
          (view-ge (view-ge this#) m# n#)
          (dragan-says-ex "This GB matrix does not have sufficient storage space for required m and n dimensions."))))
     (view-tr [this# lower?# diag-unit?#]
       (if-not (= :gb matrix-type)
         (let [reg# (region this#)
               n# (.-n this#)
               k# (max (.kl reg#) (.ku reg#))
               fact# (.-fact this#)]
           (~banded-matrix fact# false (.-buf-ptr this#) n# n# 0 (.-nav this#)
            (uplo-storage (column? this#) n# k# (lower? reg#))
            (tb-region n# k# (lower? reg#) diag-unit?#) :tb (default :tb diag-unit?#)
            (tb-engine fact#)))
         (dragan-says-ex "GB cannot be viewed as a TB due to specific factorization requirements.")))
     (view-sy [this# lower?#]
       (if-not (= :gb matrix-type)
         (let [reg# (region this#)
               n# (.-n this#)
               k# (max (.kl reg#) (.ku reg#))
               fact# (.-fact this#)]
           (~banded-matrix fact# false (.-buf-ptr this#) n# n# 0 (.-nav this#)
            (uplo-storage (column? this#) n# k# (lower? reg#))
            (sb-region n# k# (lower? reg#)) :sb sb-default (sb-engine fact#)))
         (dragan-says-ex "GB cannot be viewed as a SB due to specific factorization requirements.")))
     MemoryContext
     (compatible? [this# y#]
       (compatible? (.-da this#) y#))
     (fits? [this# b#]
       (and (instance? BandedMatrix b#)
            (let [reg# (region this#)
                  reg-b# (region b#)]
              (or (= reg# reg-b#)
                  (and (= :sb (.matrixType this#)) (matrix-type b#)
                       (not= (.-nav this#) (navigator b#))
                       (= (+ (.kl reg#) (.ku reg#)) (+ (.kl reg-b#) (.ku reg-b#)))
                       (= (.-n this#) (ncols b#)))))))
     (fits-navigation? [this# b#]
       (= (.-nav this#) (navigator b#)))
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~banded-matrix (.-fact this#) 0 0 0 0 (column? this#) (.matrixType this#)))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~banded-matrix (.-fact this#) 1 1 (.-nav this#) (.-stor this#)
                            (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#))]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (dragan-says-ex "Vararg pure is not available for banded matrices.")))))

(defmacro extend-banded-triangularizable [name]
  `(extend-type ~name
     Triangularizable
     (create-trf [a# pure#]
       (case (.-matrix-type a#)
         :tb a#
         :sb (pivotless-lu-factorization a# pure#)
         :gb (lu-factorization a# pure#)
         (dragan-says-ex "Triangular factorization is not available for this matrix type"
                         {:matrix-type (.-matrix-type a#)})))
     (create-ptrf [a#]
       (case (.-matrix-type a#)
         :tb a#
         :sb (pivotless-lu-factorization a# false)
         (dragan-says-ex "Pivotless factorization is not available for this matrix type"
                         {:matrix-type (.-matrix-type a#)})))))

(deftype RealBandedMatrix [^LayoutNavigator nav ^FullStorage stor ^Region reg ^Default default
                           fact ^RealAccessor da eng matrix-type
                           master buf-ptr ^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealBandedMatrix) (hash-combine matrix-type) (hash-combine m) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (real-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#RealBandedMatrix[%s, type%s, mxn:%dx%d, layout%s]"
            (entry-type da) matrix-type m n (dec-property (.layout nav))))
  BandedMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tb matrix-type))
  (isSymmetric [_]
    (= :sb matrix-type))
  Seqable
  (seq [a]
    (map seq (.dias a)))
  IFn$LLDO
  (invokePrim [a i j v]
    (if (.accessible reg i j)
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
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
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$DD IFn$LLDD f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
      (.get da buf-ptr (.index nav stor i j))
      (.realEntry default nav stor da buf-ptr 0 i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (let [start (.rowStart reg i)]
      (real-block-vector fact false buf-ptr (- (.rowEnd reg i) start) (.index nav stor i start)
                         (if (.isRowMajor nav) 1 (dec (.ld stor))))))
  (rows [a]
    (region-rows a))
  (col [a j]
    (let [start (.colStart reg j)]
      (real-block-vector fact false buf-ptr (- (.colEnd reg j) start) (.index nav stor start j)
                         (if (.isColumnMajor nav) 1 (dec (.ld stor))))))
  (cols [a]
    (region-cols a))
  (dia [a]
    (.dia a 0))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (if (< 0 k)
        (real-block-vector fact false buf-ptr (min m (- n k)) (.index nav stor 0 k) (.ld stor))
        (real-block-vector fact false buf-ptr (min (+ m k) n) (.index nav stor (- k) 0) (.ld stor)))
      (real-block-vector fact false buf-ptr 0 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (= i j)
      (let [kl (min (.kl reg) (dec k))
            ku (min (.ku reg) (dec l))]
        (real-banded-matrix fact false buf-ptr k l (- (.index nav stor i j) (inc kl))
                            nav (band-storage (.isColumnMajor nav) k l (.ld stor) kl ku)
                            (band-region k l kl ku) matrix-type default eng))
      (dragan-says-ex "You cannot create a submatrix of a banded (GB, TB, or SB) matrix outside its region. No way around that."
                      {:a (info a) :i i :j j :k k :l l})))
  (transpose [a]
    (real-banded-matrix fact false buf-ptr n m 0 (flip nav) stor (flip reg) matrix-type default eng))
  Subband
  (subband [a kl ku]
    (if (and (<= 0 (long kl) (.kl reg)) (<= 0 (long ku) (.ku reg)))
      (let [sub-stor (band-storage (.isColumnMajor nav) m n (.ld stor) kl ku)]
        (real-banded-matrix fact false buf-ptr m n
                            (- (.index stor 0 0) (.index ^DenseStorage sub-stor 0 0))
                            nav sub-stor (band-region m n kl ku) matrix-type default eng))
      (dragan-says-ex "You cannot create a subband outside available region. No way around that."
                      {:a (info a) :kl kl :ku ku}))))

(extend-base RealBandedMatrix)
(extend-matrix RealBandedMatrix)
(extend-banded-matrix RealBandedMatrix real-block-vector real-ge-matrix real-banded-matrix)
(extend-banded-triangularizable RealBandedMatrix)
(extend-trf RealBandedMatrix)
(extend-matrix-fluokitten RealBandedMatrix double real-flipper real-accessor)

(defmethod print-method RealBandedMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-banded w a)))

(defn banded-matrix
  ([constructor fact master buf-ptr m n ofst nav ^FullStorage stor reg matrix-type default engine]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg default fact da engine matrix-type master buf-ptr m n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)})))))
  ([constructor fact m n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) (.capacity stor))]
     (banded-matrix constructor fact true buf-ptr m n 0 nav stor reg matrix-type default engine)))
  ([constructor fact m n kl ku column? matrix-type]
   (banded-matrix constructor fact m n (layout-navigator column?) (band-storage column? m n kl ku)
                  (band-region m n kl ku) matrix-type (default matrix-type)
                  (case matrix-type
                    :gb (gb-engine fact)
                    :sb (sb-engine fact)
                    :tb (tb-engine fact)
                    (dragan-says-ex (format "%s is not a valid banded matrix type. Please send me a bug report."
                                            matrix-type)))))
  ([constructor fact m n kl ku column?]
   (banded-matrix constructor fact m n (layout-navigator column?) (band-storage column? m n kl ku)
                  (band-region m n kl ku) :gb zero-default (gb-engine fact))))

(defn tb-matrix [constructor fact n k column? lower? diag-unit?]
  (banded-matrix constructor fact n n (layout-navigator column?) (uplo-storage column? n k lower?)
                 (tb-region n k lower? diag-unit?) :tb (default :tb diag-unit?) (tb-engine fact)))

(defn sb-matrix [constructor fact n k column? lower?]
  (banded-matrix constructor fact n n (layout-navigator column?) (uplo-storage column? n k lower?)
                 (sb-region n k lower?) :sb sb-default (sb-engine fact)))

(def real-banded-matrix (partial banded-matrix ->RealBandedMatrix))
(def real-tb-matrix (partial tb-matrix ->RealBandedMatrix))
(def real-sb-matrix (partial sb-matrix ->RealBandedMatrix))

;;(def integer-banded-matrix (partial banded-matrix ->IntegerBandedMatrix)) TODO

;; =================== Packed Matrix ==================================

(defmacro extend-packed-matrix [name block-vector ge-matrix packed-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~packed-matrix (.-fact this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)
         (.matrixType this#) (.-default this#) (.-eng this#)))
       ([this# fact#]
        (create-packed (.-fact this#) (.-n this#) (.matrixType this#) (column? this#)
                       (lower? (.-reg this#)) (diag-unit? (.-reg this#)) false)))
     (zero
       ([this#]
        (create-packed (.-fact this#) (.-n this#) (.matrixType this#) (column? this#)
                       (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true))
       ([this# fact#]
        (create-packed (factory fact#) (.-n this#) (.matrixType this#) (column? this#)
                       (lower? (.-reg this#)) (diag-unit? (.-reg this#)) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~packed-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0 (.-nav this#)
        (.-stor this#) (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (~block-vector (.-fact this#) false (.-buf-ptr this#) (.surface (region this#)) 0 1))
       ([this# stride-mult#]
        (view-vctr (view-vctr this#) stride-mult#)))
     (view-ge
       ([this#]
        (dragan-says-ex "Packed matrices cannot be viewed as a GE matrix."))
       ([this# stride-mult#]
        (dragan-says-ex "Packed matrices cannot be viewed as a GE matrix."))
       ([this# m# n#]
        (dragan-says-ex "Packed matrices cannot be viewed as a GE matrix.")))
     (view-tr [this# lower?# diag-unit?#]
       (~packed-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0
        (.-nav this#) (.-stor this#) (band-region (.-n this#) lower?# diag-unit?#)
        :tp (default :tp diag-unit?#) (tp-engine (.-fact this#))))
     (view-sy [this# lower?#]
       (~packed-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0
        (.-nav this#) (.-stor this#) (band-region (.-n this#) lower?#)
        :sp sy-default (sp-engine (.-fact this#))))
     MemoryContext
     (compatible? [this# b#]
       (compatible? (.-da this#) b#))
     (fits? [this# b#]
       (and (instance? PackedMatrix b#) (= (.-reg this#) (region b#))))
     (fits-navigation? [this# b#]
       (= (.-nav this#) (navigator b#)))
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~packed-matrix (.-fact this#) 0 (column? this#) (lower? (.-reg this#))
        (diag-unit? (.-reg this#)) (.matrixType this#)))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~packed-matrix (.-fact this#) 1 (.-nav this#) (.-stor this#)
                            (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#))]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (let [source# (cons v# vs#)]
          (let-release [res# (~packed-matrix (.-fact this#) (long (math/sqrt (count source#)))
                              (column? this#) (lower? (.-reg this#)) (diag-unit? (.-reg this#))
                              (.matrixType this#))]
            (transfer! source# res#)))))))

(deftype RealPackedMatrix [^LayoutNavigator nav ^DenseStorage stor ^Region reg ^Default default
                           fact ^RealAccessor da eng matrix-type master buf-ptr ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealPackedMatrix) (hash-combine matrix-type) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (let [fl (real-flipper nav)]
      (matrix-equals fl da a b)))
  (toString [a]
    (format "#RealPackedMatrix[%s, type%s, mxn:%dx%d, layout%s]"
            (.entryType da) matrix-type n n (dec-property (.layout nav))))
  PackedMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    (= :tp matrix-type))
  (isSymmetric [_]
    (= :sp matrix-type))
  Seqable
  (seq [a]
    (map #(seq (.stripe nav a %)) (range 0 n)))
  IFn$LLDO
  (invokePrim [a i j v]
    (if (.accessible reg i j)
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
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
      (doall-layout nav stor reg i j idx (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (matrix-alter IFn$DD IFn$LLDD f nav stor reg da buf-ptr)
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx))))
    a)
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
      (.get da buf-ptr (.index nav stor i j))
      (.realEntry default nav stor da buf-ptr 0 i j)))
  (boxedEntry [a i j]
    (.entry a i j))
  (row [a i]
    (if (.isRowMajor nav)
      (let [j (.rowStart reg i)]
        (real-block-vector fact false buf-ptr (- (.rowEnd reg i) j) (.index nav stor i j) 1))
      (dragan-says-ex "You have to unpack column-major packed matrix to access its rows."
                      {:a a :layout :column})))
  (rows [a]
    (dense-rows a))
  (col [a j]
    (if (.isColumnMajor nav)
      (let [i (.colStart reg j)]
        (real-block-vector fact false buf-ptr (- (.colEnd reg j) i) (.index nav stor i j) 1))
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
    (real-packed-matrix fact false buf-ptr n 0 (flip nav) stor (flip reg) matrix-type default eng)))

(extend-base RealPackedMatrix)
(extend-matrix RealPackedMatrix)
(extend-packed-matrix RealPackedMatrix real-block-vector real-ge-matrix real-packed-matrix)
(extend-uplo-triangularizable RealPackedMatrix)
(extend-trf RealPackedMatrix)
(extend-matrix-fluokitten RealPackedMatrix double real-flipper real-accessor)

(defmethod print-method RealPackedMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-uplo w a "*")))

(defn packed-matrix
  ([constructor fact master buf-ptr n ofst nav ^DenseStorage stor reg matrix-type default engine]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg default fact da engine matrix-type master buf-ptr n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)}))))) ;;TODO extract capacity check function
  ([constructor fact master buf-ptr n nav stor reg matrix-type default engine]
   (packed-matrix constructor fact master buf-ptr n 0 nav stor reg matrix-type default engine))
  ([constructor fact n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf-ptr (.createDataSource (data-accessor fact) (.capacity stor))]
     (packed-matrix constructor fact true buf-ptr n 0 nav stor reg matrix-type default engine)))
  ([constructor fact n column? lower? diag-unit? matrix-type]
   (case matrix-type
     :tp (packed-matrix constructor fact n column? lower? diag-unit?)
     :sp (packed-matrix constructor fact n column? lower?)
     (dragan-says-ex "Packed matrices have to be either triangular or symmetric."
                     {:matrix-type matrix-type})))
  ([constructor fact n column? lower? diag-unit?]
   (packed-matrix constructor fact n (layout-navigator column?) (packed-storage column? lower? n)
                  (band-region n lower? diag-unit?) :tp (default :tp diag-unit?) (tp-engine fact)))
  ([constructor fact n column? lower?]
   (packed-matrix constructor fact n (layout-navigator column?) (packed-storage column? lower? n)
                  (band-region n lower?) :sp sy-default (sp-engine fact))))

(def real-packed-matrix (partial packed-matrix ->RealPackedMatrix))
;;TODO (def integer-packed-matrix (partial packed-matrix ->IntegerPackedMatrix))

;; =================== Diagonal Matrix implementations =========================================

(defmacro extend-diagonal-matrix [name block-vector ge-matrix diagonal-matrix]
  `(extend-type ~name
     Container
     (raw
       ([this#]
        (~diagonal-matrix (.-fact this#) (.-n this#) (.-nav this#) (.-stor this#) (.-reg this#)
         (.matrixType this#) (.-default this#) (.-eng this#)))
       ([this# fact#]
        (create-diagonal (.-fact this#) (.-n this#) (.matrixType this#) false)))
     (zero
       ([this#]
        (create-diagonal (.-fact this#) (.-n this#) (.matrixType this#) true))
       ([this# fact#]
        (create-diagonal (factory fact#) (.-n this#) (.matrixType this#) true)))
     (host [this#]
       (let-release [res# (raw this#)]
         (copy (.-eng this#) this# res#)
         res#))
     (native [this#]
       this#)
     Viewable
     (view [this#]
       (~diagonal-matrix (.-fact this#) false (.-buf-ptr this#) (.-n this#) 0 (.-nav this#)
        (.-stor this#) (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#)))
     DenseContainer
     (view-vctr
       ([this#]
        (~block-vector (.-fact this#) false (.-buf-ptr this#) (.surface (region this#)) 0 1))
       ([this# stride-mult#]
        (dragan-says-ex "TD cannot be viewed as a strided vector.")))
     (view-ge
       ([this#]
        (dragan-says-ex "TD cannot be viewed as a GE matrix."))
       ([this# stride-mult#]
        (dragan-says-ex "TD cannot be viewed as a GE matrix."))
       ([this# m# n#]
        (dragan-says-ex "TD cannot be viewed as a GE matrix.")))
     (view-tr [this# lower?# diag-unit?#]
       (dragan-says-ex "TD cannot be viewed as a TR matrix."))
     (view-sy [this# lower?#]
       (dragan-says-ex "TD cannot be viewed as a TR matrix."))
     MemoryContext
     (compatible? [this# b#]
       (compatible? (.-da this#) b#))
     (fits? [this# b#]
       (and (instance? DiagonalMatrix b#) (= (.-reg this#) (region b#))))
     (fits-navigation? [this# b#]
       true)
     (device [this#]
       (device (.-da this#)))
     Monoid
     (id [this#]
       (~diagonal-matrix (.-fact this#) 0 (.matrixType this#)))
     Applicative
     (pure
       ([this# v#]
        (let-release [res# (~diagonal-matrix (.-fact this#) 1 (.-nav this#) (.-stor this#)
                            (.-reg this#) (.matrixType this#) (.-default this#) (.-eng this#))]
          (uncomplicate.neanderthal.core/entry! res# v#)))
       ([this# v# vs#]
        (let [source# (cons v# vs#)]
          (let-release [res# (~diagonal-matrix (.-fact this#) (count source#) (.matrixType this#))]
            (transfer! source# res#)))))))

(defmacro extend-diagonal-triangularizable [name]
  `(extend-type ~name
     Triangularizable
     (create-trf [this# pure#]
       (case (.matrixType this#)
         :gd this#
         :gt (lu-factorization this# pure#)
         :dt (pivotless-lu-factorization this# pure#)
         :st (pivotless-lu-factorization this# pure#)
         (dragan-says-ex "Triangular factorization is not available for this matrix type."
                         {:matrix-type (.matrixType this#)})))
     (create-ptrf [this#]
       (if (symmetric? this#)
         (pivotless-lu-factorization this# false)
         (dragan-says-ex "Pivotless factorization is not available for this matrix type."
                         {:matrix-type (.matrixType this#)})))))

(defmacro extend-diagonal-trf [name]
  `(extend-type ~name
     TRF
     (trtrs [a# b#]
       (if (= :gd (.matrixType a#))
         (let-release [res# (raw b#)]
           (copy (engine b#) b# res#)
           (trs (.-eng a#) a# res#))
         (require-trf)))
     (trtrs! [a# b#]
       (if (= :gd (.matrixType a#))
         (trs (.-eng a#) a# b#)
         (require-trf)))
     (trtri! [a#]
       (if (= :gd (.matrixType a#))
         (tri (.-eng a#) a#)
         (require-trf)))
     (trtri [a#]
       (if (= :gd (.matrixType a#))
         (let-release [res# (raw a#)
                       eng# (.-eng a#)]
           (tri eng# (copy eng# a# res#)))
         (require-trf)))
     (trcon
       ([a# _# nrm1?#]
        (if (= :gd (.matrixType a#))
          (con (.-eng a#) a# nrm1?#)
          (require-trf)))
       ([a# nrm1?#]
        (if (= :gd (.matrixType a#))
          (con (.-eng a#) a# nrm1?#)
          (require-trf))))
     (trdet [a#]
       (if (= :gd (.matrixType a#))
         (if (diag-unit? (.-reg a#)) 1.0 (fold (dia a#) f* 1.0))
         (require-trf)))))

(defmacro extend-diagonal-fluokitten [t cast typed-flipper vtype]
  `(extend ~t
     Functor
     {:fmap (diagonal-fmap ~vtype ~cast)}
     PseudoFunctor
     {:fmap! (diagonal-fmap identity ~vtype ~cast)}
     Foldable
     {:fold diagonal-fold
      :foldmap diagonal-foldmap}
     Magma
     {:op (constantly matrix-op)}))

(deftype RealDiagonalMatrix [^LayoutNavigator nav ^DenseStorage stor ^Region reg ^Default default
                             fact ^RealAccessor da eng matrix-type master buf-ptr ^long n]
  Object
  (hashCode [a]
    (-> (hash :RealDiagonalMatrix) (hash-combine matrix-type) (hash-combine n)
        (hash-combine (nrm2 eng a))))
  (equals [a b]
    (or (identical? a b)
        (and (instance? RealDiagonalMatrix b) (compatible? a b) (fits? a b)
             (let [n (.surface reg)
                   buf-b (.buffer ^RealDiagonalMatrix b)]
               (loop [i 0]
                 (if (< i n)
                   (and (= (.get da buf-ptr i) (.get da buf-b i))
                        (recur (inc i)))
                   true))))))
  (toString [a]
    (format "#RealDiagonalMatrix[%s, type%s, mxn:%dx%d]"
            (.entryType da) matrix-type n n))
  DiagonalMatrix
  (matrixType [_]
    matrix-type)
  (isTriangular [_]
    false)
  (isSymmetric [_]
    (= :st matrix-type))
  Seqable
  (seq [a]
    (map seq (.dias a)))
  IFn$LLDO
  (invokePrim [a i j v]
    (if (.accessible reg i j)
      (.set a i j v)
      (throw (ex-info "Requested element is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn$LLD
  (invokePrim [a i j]
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                      {:i i :j j :mrows n :ncols n}))))
  IFn
  (invoke [a i j v]
    (.invokePrim a i j v))
  (invoke [a i j]
    (.invokePrim a i j))
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
        (.set da buf-ptr idx val)))
    a)
  (set [a i j val]
    (.set da buf-ptr (.index nav stor i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a f]
    (if (instance? IFn$DD f)
      (dotimes [idx (.capacity stor)]
        (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx))))
      (dragan-says-ex "You cannot call indexed alter on diagonal matrices. Use banded matrix."))
    a)
  (alter [a i j f]
    (let [idx (.index nav stor i j)]
      (.set da buf-ptr idx (.invokePrim ^IFn$DD f (.get da buf-ptr idx)))
      a))
  RealNativeMatrix
  (buffer [_]
    buf-ptr)
  (offset [_]
    0)
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
      (.get da buf-ptr (.index nav stor i j))
      (.realEntry default nav stor da buf-ptr 0 i j)))
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
    (real-block-vector fact false buf-ptr n 1))
  (dia [a k]
    (if (<= (- (.kl reg)) k (.ku reg))
      (real-block-vector fact false buf-ptr (- n (Math/abs k)) (.index stor 0 k) 1)
      (real-block-vector fact false buf-ptr 0 1)))
  (dias [a]
    (region-dias a))
  (submatrix [a i j k l]
    (if (and (= i j) (= k l))
      (real-diagonal-matrix fact false buf-ptr k (.index nav stor i j) nav
                            (diagonal-storage k matrix-type) (band-region k l (.kl reg) (.ku reg))
                            matrix-type default eng)
      (dragan-says-ex "You cannot create such submatrix of a (tri)diagonal matrix."
                      {:a (info a)})))
  (transpose [a]
    (if (or (= :gd matrix-type) (= :st matrix-type))
      a
      (dragan-says-ex "You cannot transpose this (tri)diagonal matrix."))))

(extend-base RealDiagonalMatrix)
(extend-matrix RealDiagonalMatrix)
(extend-diagonal-matrix RealDiagonalMatrix real-block-vector real-ge-matrix real-diagonal-matrix)
(extend-diagonal-triangularizable RealDiagonalMatrix)
(extend-diagonal-trf RealDiagonalMatrix)
(extend-diagonal-fluokitten RealDiagonalMatrix double real-flipper RealBlockVector)

(defmethod print-method RealDiagonalMatrix [a ^java.io.Writer w]
  (.write w (str a))
  (when-not (null? (buffer a))
    (print-diagonal w a)))

(defn diagonal-matrix
  ([constructor fact master buf-ptr n ofst nav ^DenseStorage stor reg matrix-type default engine]
   (let [da (data-accessor fact)
         buf-ptr (pointer buf-ptr ofst)]
     (if (<= 0 (.capacity stor) (.count da buf-ptr))
       (constructor nav stor reg default fact (data-accessor fact) engine
                    matrix-type master buf-ptr n)
       (throw (ex-info "Insufficient buffer size."
                       {:dim (.capacity stor) :buffer-size (.count da buf-ptr)})))))
  ([constructor fact n nav ^DenseStorage stor reg matrix-type default engine]
   (let-release [buf (.createDataSource (data-accessor fact) (.capacity stor))]
     (diagonal-matrix constructor fact true buf n 0 nav stor reg matrix-type default engine)))
  ([constructor fact ^long n matrix-type]
   (let [kl (case matrix-type :st 0 :gd 0 :gt 1 :dt 1 1)
         ku (if (= :gd matrix-type) 0 1)]
     (diagonal-matrix constructor fact n diagonal-navigator (diagonal-storage n matrix-type)
                      (band-region n n kl ku) matrix-type (default matrix-type)
                      (case matrix-type
                        :gd (gd-engine fact)
                        :gt (gt-engine fact)
                        :dt (dt-engine fact)
                        :st (st-engine fact)
                        (dragan-says-ex (format "%s is not a valid (tri)diagonal matrix type."
                                                matrix-type)))))))

(def real-diagonal-matrix (partial diagonal-matrix ->RealDiagonalMatrix))
;;(def integer-diagonal-matrix (partial diagonal-matrix ->IntegerDiagonalMatrix)) TODO

;; =================== transfer method implementations =========================================

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

(defmethod transfer! [(Class/forName "[S") IntegerBlockVector]
  [^longs source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[B") IntegerBlockVector]
  [^ints source ^IntegerBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[D")]
  [^IntegerBlockVector source ^doubles destination]
  (transfer-vector-array double source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[F")]
  [^IntegerBlockVector source ^floats destination]
  (transfer-vector-array float source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[J")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array long source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[I")]
  [^IntegerBlockVector source ^ints destination]
  (transfer-vector-array int source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[S")]
  [^IntegerBlockVector source ^longs destination]
  (transfer-vector-array short source destination))

(defmethod transfer! [IntegerBlockVector (Class/forName "[B")]
  [^IntegerBlockVector source ^ints destination]
  (transfer-vector-array byte source destination))

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

(defmethod transfer! [(Class/forName "[S") RealBlockVector]
  [^shorts source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [(Class/forName "[B") RealBlockVector]
  [^bytes source ^RealBlockVector destination]
  (transfer-array-vector source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[D")]
  [^RealBlockVector source ^doubles destination]
  (transfer-vector-array double  source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[F")]
  [^RealBlockVector source ^floats destination]
  (transfer-vector-array float source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[J")]
  [^RealBlockVector source ^longs destination]
  (transfer-vector-array long source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[I")]
  [^RealBlockVector source ^ints destination]
  (transfer-vector-array int source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[S")]
  [^RealBlockVector source ^shorts destination]
  (transfer-vector-array short source destination))

(defmethod transfer! [RealBlockVector (Class/forName "[B")]
  [^RealBlockVector source ^bytes destination]
  (transfer-vector-array byte source destination))

(defmethod transfer! [clojure.lang.Sequential RealNativeMatrix]
  [source ^RealNativeMatrix destination]
  (transfer-seq-matrix real-accessor source destination))

(defmethod transfer! [RealNativeMatrix RealNativeMatrix]
  [^RealNativeMatrix source ^RealNativeMatrix destination]
  (transfer-matrix-matrix real-accessor real-flipper source destination))

(defmethod transfer! [(Class/forName "[D") RealNativeMatrix]
  [^doubles source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[F") RealNativeMatrix]
  [^floats source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[J") RealNativeMatrix]
  [^longs source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[I") RealNativeMatrix]
  [^ints source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[S") RealNativeMatrix]
  [^shorts source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [(Class/forName "[B") RealNativeMatrix]
  [^bytes source ^RealNativeMatrix destination]
  (transfer-array-matrix real-accessor source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[D")]
  [^RealNativeMatrix source ^doubles destination]
  (transfer-matrix-array real-accessor double source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[F")]
  [^RealNativeMatrix source ^floats destination]
  (transfer-matrix-array real-accessor float source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[J")]
  [^RealNativeMatrix source ^longs destination]
  (transfer-matrix-array real-accessor long source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[I")]
  [^RealNativeMatrix source ^ints destination]
  (transfer-matrix-array real-accessor int source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[S")]
  [^RealNativeMatrix source ^shorts destination]
  (transfer-matrix-array real-accessor short source destination))

(defmethod transfer! [RealNativeMatrix (Class/forName "[B")]
  [^RealNativeMatrix source ^bytes destination]
  (transfer-matrix-array real-accessor byte source destination))

(defmethod transfer! [RealNativeVector RealNativeMatrix]
  [^RealNativeVector source ^RealNativeMatrix destination]
  (transfer-vector-matrix real-accessor source destination))

(defmethod transfer! [RealNativeMatrix RealNativeVector]
  [^RealNativeMatrix source ^RealBlockVector destination]
  (transfer-matrix-vector real-accessor source destination))

(defmethod transfer! [IntegerNativeVector RealNativeMatrix]
  [^IntegerNativeVector source ^RealNativeMatrix destination]
  (transfer-vector-matrix real-accessor source destination))

(defmethod transfer! [RealNativeMatrix IntegerNativeVector]
  [^RealNativeMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector real-accessor source destination))

(defmethod transfer! [clojure.lang.Sequential IntegerNativeMatrix]
  [source ^IntegerNativeMatrix destination]
  (transfer-seq-matrix integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix IntegerNativeMatrix]
  [^IntegerNativeMatrix source ^IntegerNativeMatrix destination]
  (transfer-matrix-matrix integer-accessor integer-flipper source destination))

(defmethod transfer! [(Class/forName "[D") IntegerNativeMatrix]
  [^doubles source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[F") IntegerNativeMatrix]
  [^floats source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[J") IntegerNativeMatrix]
  [^longs source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[I") IntegerNativeMatrix]
  [^ints source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[S") IntegerNativeMatrix]
  [^shorts source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [(Class/forName "[B") IntegerNativeMatrix]
  [^bytes source ^IntegerNativeMatrix destination]
  (transfer-array-matrix integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[D")]
  [^IntegerNativeMatrix source ^doubles destination]
  (transfer-matrix-array integer-accessor double source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[F")]
  [^IntegerNativeMatrix source ^floats destination]
  (transfer-matrix-array integer-accessor float source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[J")]
  [^IntegerNativeMatrix source ^longs destination]
  (transfer-matrix-array integer-accessor long source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[I")]
  [^IntegerNativeMatrix source ^ints destination]
  (transfer-matrix-array integer-accessor int source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[S")]
  [^IntegerNativeMatrix source ^shorts destination]
  (transfer-matrix-array integer-accessor short source destination))

(defmethod transfer! [IntegerNativeMatrix (Class/forName "[B")]
  [^IntegerNativeMatrix source ^bytes destination]
  (transfer-matrix-array integer-accessor byte source destination))

(defmethod transfer! [IntegerNativeVector IntegerNativeMatrix]
  [^IntegerNativeVector source ^IntegerNativeMatrix destination]
  (transfer-vector-matrix integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix IntegerNativeVector]
  [^IntegerNativeMatrix source ^IntegerBlockVector destination]
  (transfer-matrix-vector integer-accessor source destination))

(defmethod transfer! [RealNativeVector IntegerNativeMatrix]
  [^RealNativeVector source ^IntegerNativeMatrix destination]
  (transfer-vector-matrix integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix RealNativeVector]
  [^IntegerNativeMatrix source ^RealBlockVector destination]
  (transfer-matrix-vector integer-accessor source destination))

(defmethod transfer! [IntegerNativeMatrix RealNativeMatrix]
  [^IntegerNativeMatrix source ^RealNativeMatrix destination]
  (transfer-matrix-matrix real-accessor integer-flipper source destination))

(defmethod transfer! [RealNativeMatrix IntegerNativeMatrix]
  [^RealNativeMatrix source ^IntegerNativeMatrix destination]
  (transfer-matrix-matrix integer-accessor real-flipper source destination))

(defmethod transfer! [RealPackedMatrix RealPackedMatrix]
  [^RealPackedMatrix source ^RealPackedMatrix destination]
  (transfer-matrix-matrix real-accessor real-flipper (= (navigator source) (navigator destination))
                          source destination))

(defmethod transfer! [RealDiagonalMatrix RealDiagonalMatrix]
  [source destination]
  (transfer! (view-vctr source) (view-vctr destination))
  destination)

(defmethod transfer! [RealDiagonalMatrix Matrix]
  [source destination]
  (let [reg (region source)]
    (doseq [k (range (.kl reg) (.ku reg))]
      (transfer! (dia source k) (dia destination k))
      destination)))

(defmethod transfer! [Matrix RealDiagonalMatrix]
  [source destination]
  (let [reg (region destination)]
    (doseq [k (range (.kl reg) (.ku reg))]
      (transfer! (dia source k) (dia destination k))
      destination)))

(defmacro extend-pointer [name fact]
  `(extend-type ~name
     DenseContainer
     (view-vctr
       ([this#]
        (create-vector ~fact false this# (size this#) 0 1))
       ([this# stride-mult#]
        (view-vctr (view-vctr this#) stride-mult#)))
     (view-ge
       ([this#]
        (view-ge (view-vctr this#)))
       ([this# stride-mult#]
        (view-ge (view-vctr this#) stride-mult#))
       ([this# m# n#]
        (view-ge (view-vctr this#) m# n#)))
     (view-tr [this# lower?# diag-unit?#]
       (view-tr (view-vctr this#) lower?# diag-unit?#))
     (view-sy [this# lower?#]
       (view-sy (view-vctr this#) lower?#))))

(defn map-channel
  ([fact channel n flag offset-bytes]
   (let [fact (factory fact)
         da ^DataAccessor (data-accessor fact)
         entry-width (.entryWidth da)]
     (let-release [buf ((type-pointer (.entryType da))
                        (mapped-buffer channel offset-bytes (* (long n) entry-width) flag))]
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
