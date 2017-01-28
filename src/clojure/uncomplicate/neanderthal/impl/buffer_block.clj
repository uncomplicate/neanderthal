;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.impl.buffer-block
  (:require [vertigo
             [core :refer [wrap]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 wrap-byte-seq]]]
            [uncomplicate.fluokitten.protocols
             :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [transfer! copy! col row mrows ncols]]
             [real :refer [sum entry]]]
            [uncomplicate.neanderthal.impl.fluokitten :refer :all]
            [uncomplicate.commons.core :refer [Releaseable release let-release]])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$OLLDO
            IFn$LD IFn$LDD IFn$LLD IFn$L IFn$LLL IFn$OLLD IFn$OLLDD IFn$LLLL IFn$OLO]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus RealBufferAccessor BufferAccessor DataAccessor
            Vector Matrix RealVector RealMatrix RealChangeable
            Block GEMatrix TRMatrix]))

(declare real-block-vector)
(declare real-ge-matrix)
(declare real-tr-matrix)

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

;; ============ Realeaseable ===================================================

(defn ^:private clean-buffer [^ByteBuffer buffer]
  (do
    (if (.isDirect buffer)
      (.clean (.cleaner ^DirectByteBuffer buffer)))
    true))

(extend-type DirectByteBuffer
  Releaseable
  (release [this]
    (clean-buffer this)))

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
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? FloatBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float32 (* Float/BYTES stride) 0 (byte-seq buf))
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
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? DoubleBufferAccessor da))))
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^uncomplicate.neanderthal.protocols.Factory fact
                          ^RealBufferAccessor da ^BLASPlus eng ^Boolean master
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [x]
    (vector-fold* hash* (-> (hash :RealBlockVector) (hash-combine n))
                  (.subvector x 0 (min 16 n))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? RealBlockVector y) (compatible da y)
           (= n (.dim ^RealBlockVector y)))
      (loop [i 0]
        (if (< i n)
          (if (= (.entry x i) (.entry ^RealBlockVector y i))
            (recur (inc i))
            false)
          true))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, stride:%d]" (.entryType da) n strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Seqable
  (seq [_]
    (.toSeq da buf strd))
  Container
  (raw [this]
    (raw this fact))
  (raw [_ fact]
    (create-vector fact (.createDataSource (data-accessor fact) n) n nil))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)))
  (native [this]
    this)
  MemoryContext
  (compatible [_ y]
    (compatible da y))
  Monoid
  (id [x]
    (create-vector fact (.createDataSource da 0) 0 nil))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    fact)
  DataAccessorProvider
  (data-accessor [_]
    da)
  Block
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
  (invoke [x]
    n)
  IFn$L
  (invokePrim [x]
    n)
  RealChangeable
  (set [x val]
    (if (= 0 strd)
      (.initialize da buf val)
      (dotimes [i n]
        (.set da buf (* strd i) val)))
    x)
  (set [x i val]
    (.set da buf (* strd i) val)
    x)
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
    (.get da buf (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (let [b (.slice da buf (* k strd) (inc (* (dec l) strd)))]
      (real-block-vector fact false b l strd)))
  PseudoFunctor
  (fmap! [x f]
    (vector-fmap* ^IFn$DD f x))
  (fmap! [x f y]
    (vector-fmap* ^IFn$DDD f x y))
  (fmap! [x f y z]
    (vector-fmap* ^IFn$DDDD f x y z))
  (fmap! [x f y z v]
    (vector-fmap* ^IFn$DDDDD f x y z v))
  (fmap! [x f y z v ws]
    (vector-fmap* f x y z v ws))
  Foldable
  (fold [x]
    (.sum eng x))
  (fold [x f init]
    (vector-fold* f init x))
  (fold [x f init y]
    (vector-fold* f init x y))
  (fold [x f init y z]
    (vector-fold* f init x y z))
  (fold [x f init y z v]
    (vector-fold* f init x y z v))
  (fold [x f init y z v ws]
    (vector-fold* f init x y z v ws))
  (foldmap [x g]
    (loop [i 0 acc 0.0]
      (if (< i n)
        (recur (inc i) (+ acc (.invokePrim ^IFn$DD g (.entry x i))))
        acc)))
  (foldmap [x g f init]
    (vector-foldmap* f init ^IFn$DD g x))
  (foldmap [x g f init y]
    (vector-foldmap* f init ^IFn$DDD g x y))
  (foldmap [x g f init y z]
    (vector-foldmap* f int ^IFn$DDDD g x y z))
  (foldmap [x g f init y z v]
    (vector-foldmap* f init ^IFn$DDDDD g x y z v))
  (foldmap [x g f init y z v ws]
    (vector-foldmap* f init g x y z v ws)))()

(defn real-block-vector [fact master buf n strd]
  (let [da (data-accessor fact)]
    (RealBlockVector. fact da (vector-engine fact) master buf n strd)))

(extend RealBlockVector
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure vector-pure}
  Magma
  {:op (constantly vector-op)})

(extend-type clojure.lang.Sequential
  Container
  (raw [this fact]
    (let [n (count this)]
      (create-vector fact (.createDataSource (data-accessor fact) n) n nil))))

(defmethod transfer! [RealBlockVector RealBlockVector]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealBlockVector]
  [source ^RealBlockVector destination]
  (let [n (.dim destination)]
    (do
      (loop [i 0 src source]
        (if (and src (< i n))
          (do
            (.set destination i (first src))
            (recur (inc i) (next src)))
          destination)))))

(defmethod print-method RealBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(defn ^:private straight-cut [fact ^ByteBuffer buf ld sd fd idx]
  (real-block-vector fact false
                     (.slice ^BufferAccessor (data-accessor fact)
                             buf (* (long ld) (long idx)) sd)
                     sd 1))

(defn ^:private cross-cut [fact ^ByteBuffer buf ld sd fd idx]
  (real-block-vector fact false
                     (.slice ^BufferAccessor (data-accessor fact)
                             buf idx (inc (* (dec (long fd)) (long ld))))
                     fd ld))

(defn ^:private no-trans-get ^double [^RealMatrix a ^long i ^long j]
  (.entry a i j))

(defn ^:private trans-get ^double [^RealMatrix a ^long i ^long j]
  (.entry a j i))

(defn ^:private no-trans-set [^RealChangeable a ^long i ^long j ^double val]
  (.set a i j val))

(defn ^:private trans-set [^RealChangeable a ^long i ^long j ^double val]
  (.set a j i val))

(deftype RealGEMatrix [^IFn$LLLL index* ^IFn$OLLD get* ^IFn$OLLDO set*
                       ^IFn$OLO col-row* col* row*
                       ^uncomplicate.neanderthal.protocols.Factory fact
                       ^RealBufferAccessor da ^BLAS eng ^Boolean master
                       ^ByteBuffer buf ^long m ^long n
                       ^long ld ^long sd ^long fd ^long ord]
  Object
  (hashCode [this]
    (matrix-fold* fd col-row* hash*
                  (-> (hash :RealGEMatrix) (hash-combine m) (hash-combine n))
                  (.submatrix this 0 0 (min 16 m) (min 16 n))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealGEMatrix b) (compatible da b)
           (= m (mrows b)) (= n (ncols b)))
      (loop [j 0]
        (if (< j fd)
          (and (loop [i 0]
                 (if (< i sd)
                   (and (= (.get da buf (+ (* ld j) i))
                           (.invokePrim get* ^RealGEMatrix b i j))
                        (recur (inc i)))
                   true))
               (recur (inc j)))
          true))
      :default false))
  (toString [this]
    (format "#RealGEMatrix[%s, mxn:%dx%d, ld:%d, ord%s]"
            (.entryType da) m n ld (dec-property ord)))
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
    fact)
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (real-ge-matrix fact m n ord))
  (raw [this fact]
    (real-ge-matrix fact m n ord))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)))
  (native [this]
    this)
  MemoryContext
  (compatible [_ b]
    (compatible da b))
  DenseContainer
  (subtriangle [_ uplo diag];;TODO remove and introduce new function similar to copy that reuses memory
    (let [k (min m n)
          b (.slice da buf 0 (+ (* ld (dec k)) k))]
      (real-tr-matrix fact false b k ld ord uplo diag)))
  Monoid
  (id [a]
    (real-ge-matrix fact 0 0))
  GEMatrix
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
  (sd [_]
    sd)
  (fd [_]
    fd)
  Seqable
  (seq [a]
    (map #(seq (col-row* a %)) (range 0 fd)))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
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
    (dotimes [j fd]
      (dotimes [i sd]
        (.set da buf (+ (* ld j) i) val)))
    a)
  (set [a i j val]
    (.set da buf (.invokePrim index* ld i j) val)
    a)
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
  (entry [a i j]
    (.get da buf (.invokePrim index* ld i j)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (row* fact buf ld sd fd i))
  (col [a j]
    (col* fact buf ld sd fd j))
  (submatrix [a i j k l]
    (real-ge-matrix fact false
                    (.slice da buf (.invokePrim index* ld i j)
                            (.invokePrim index* ld k (dec l)))
                    k l ld ord))
  (transpose [a]
    (real-ge-matrix fact false buf n m ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR)))
  PseudoFunctor
  (fmap! [a f]
    (matrix-fmap* set* get* fd sd ^IFn$DD f a))
  (fmap! [a f b]
    (matrix-fmap* set* get* fd sd ^IFn$DDD f a b))
  (fmap! [a f b c]
    (matrix-fmap* set* get* fd sd ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (matrix-fmap* set* get* fd sd ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (matrix-fmap* set* get* fd sd f a b c d nil))
  Foldable
  (fold [_]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i) (+ acc (.get da buf (+ (* ld j) i))))
                    acc))))
        acc)))
  (fold [a f init]
    (matrix-fold* fd col-row* f init a))
  (fold [a f init b]
    (matrix-fold* fd col-row* f init a b))
  (fold [a f init b c]
    (matrix-fold* fd col-row* f init a b c))
  (fold [a f init b c d]
    (matrix-fold* fd col-row* f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold* fd col-row* f init a b c d es))
  (foldmap [_ g]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i)
                           (+ acc (.invokePrim ^IFn$DD g (.get da buf (+ (* ld j) i)))))
                    acc))))
        acc)))
  (foldmap [a g f init]
    (matrix-foldmap* fd col-row* f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap* fd col-row* f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap* fd col-row* f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap* fd col-row* f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap* fd col-row* f init g a b c d es)))

(defn no-trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld j) i))

(defn trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld i) j))

(defn real-ge-matrix
  ([fact master buf m n ld ord]
   (let [da (data-accessor fact)]
     (if (= COLUMN_MAJOR ord)
       (RealGEMatrix. no-trans-index no-trans-get no-trans-set col
                      straight-cut cross-cut fact da (ge-engine fact)
                      master buf m n (max (long ld) (long m)) m n ord)
       (RealGEMatrix. trans-index trans-get trans-set row cross-cut
                      straight-cut fact da (ge-engine fact)
                      master buf m n (max (long ld) (long n)) n m ord))))
  ([fact m n ord]
   (let-release [buf (.createDataSource (data-accessor fact) (* (long m) (long n)))]
     (real-ge-matrix fact true buf m n m ord)))
  ([fact m n]
   (real-ge-matrix fact m n COLUMN_MAJOR)))

(extend RealGEMatrix
  Functor
  {:fmap copy-fmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op (constantly matrix-op)})

(defmethod transfer! [RealGEMatrix RealGEMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealGEMatrix]
  [source ^RealGEMatrix destination]
  (let [ld (.ld destination)
        sd (.sd destination)
        fd (.fd destination)
        da (data-accessor destination)
        buf (.buffer destination)]
    (loop [j 0 src source]
      (if (and src (< j fd))
        (recur (inc j)
               (loop [i 0 src src]
                 (if (and src (< i sd))
                   (do (.set ^RealBufferAccessor da buf (+ (* ld j) i) (first src))
                       (recur (inc i) (next src)))
                   src)))
        destination))))

(defmethod print-method RealGEMatrix
  [^RealGEMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))

;; =================== Real Triangular Matrix ==================================

(deftype RealTRMatrix [^IFn$LLLL index* ^IFn$OLLD get* ^IFn$OLLDO set*
                       ^IFn$LLL start* ^IFn$LLL end*
                       ^IFn$OLO col-row* col* row*
                       ^IFn$LLL col-start* ^IFn$LLL col-len*
                       ^IFn$LLL row-start* ^IFn$LLL row-len*
                       ^long diag-pad ^IFn$LLL default-entry*
                       ^uncomplicate.neanderthal.protocols.Factory fact
                       ^RealBufferAccessor da ^BLAS eng ^Boolean master
                       ^ByteBuffer buf ^long n ^long ld
                       ^long ord ^long fuplo ^long fdiag]
  Object
  (hashCode [this]
    #_(tr-matrix-fold* fd col-row* hash* (-> (hash :RealTRMatrix) (hash-combine n))
                       (.submatrix this 0 0 (min 16 n) (min 16 n))));;TODO check fluokitten and triangle and do with submatrix
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealTRMatrix b) (compatible da b)
           (= n (mrows b))
           (= fuplo (.uplo ^RealTRMatrix b))
           (= fdiag (.diag ^RealTRMatrix b)))
      (loop [j 0]
        (if (< j n)
          (let [end (.invokePrim end* n j)]
            (and (loop [i (.invokePrim start* n j)]
                   (if (< i end)
                     (and (= (.get da buf (+ (* ld j) i))
                             (.invokePrim get* ^RealTRMatrix b i j))
                          (recur (inc i)))
                     true))
                 (recur (inc j))))
          true))
      :default false))
  (toString [a]
    (format "#RealTRMatrix[%s, mxn:%dx%d, ld:%d, ord%s, uplo%s, diag%s]"
            (.entryType da) n n ld (dec-property ord) (dec-property fuplo) (dec-property fdiag)))
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
    fact)
  DataAccessorProvider
  (data-accessor [_]
    da)
  Container
  (raw [_]
    (real-tr-matrix fact n ord fuplo fdiag))
  (raw [_ fact]
    (real-tr-matrix fact n ord fuplo fdiag))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)));;TODO implement TR.copy
  (native [this]
    this)
  MemoryContext
  (compatible [_ b]
    (compatible da b))
  Monoid
  (id [a]
    (real-tr-matrix fact 0))
  TRMatrix
  (buffer [_]
    buf)
  (offset [_]
    0)
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
    (map #(seq (col-row* a %)) (range 0 n)))
  IFn$LLD
  (invokePrim [a i j]
    (entry a i j))
  IFn
  (invoke [a i j]
    (entry a i j))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (isAllowed [a i j]
    (= 2 (.invokePrim default-entry* i j)))
  (set [a val]
    (dotimes [j n]
      (let [end (.invokePrim end* n j)]
        (loop [i (.invokePrim start* n j)]
          (when (< i end)
            (.set da buf (+ (* ld j) i) val)
            (recur (inc i))))))
    a)
  (set [a i j val]
    (.set da buf (.invokePrim index* ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (.set a i j (.invokePrim ^IFn$DD f (.entry a i j))))
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (let [res (.invokePrim default-entry* i j)]
      (if (= 2 res)
        (.get da buf (.invokePrim index* ld i j))
        res)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (.subvector ^Vector (row* fact buf ld n n i)
                (+ (.invokePrim row-start* n i) diag-pad)
                (- (.invokePrim row-len* n i) diag-pad)))
  (col [a j]
    (.subvector ^Vector (col* fact buf ld n n j)
                (+ (.invokePrim col-start* n j) diag-pad)
                (- (.invokePrim col-len* n j) diag-pad)))
  (submatrix [a i j k l]
    (throw (UnsupportedOperationException. "TODO I should return banded matrix or GE, or consider only TR submatrices valid.")))
  (transpose [a]
    (real-tr-matrix fact false buf n ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR) fuplo fdiag))
  PseudoFunctor
  (fmap! [a f]
    (tr-fmap set* get* start* end* n ^IFn$DD f a))
  (fmap! [a f b]
    (tr-fmap set* get* start* end* n ^IFn$DDD f a b))
  (fmap! [a f b c]
    (tr-fmap set* get* start* end* n ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (tr-fmap set* get* start* end* n ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (tr-fmap set* get* start* end* n f a b c d nil))
  Foldable
  (fold [_]
    (loop [j 0 acc 0.0]
      (if (< j n)
        (recur (inc j)
               (double
                (let [end (.invokePrim end* n j)]
                  (loop [i (.invokePrim start* n j) acc acc]
                    (if (< i end)
                      (recur (inc i) (+ acc (.get da buf (+ (* ld j) i))))
                      acc)))))
        acc)))
  (fold [a f init]
    (matrix-fold* n col-row* f init a))
  (fold [a f init b]
    (matrix-fold* n col-row* f init a b))
  (fold [a f init b c]
    (matrix-fold* n col-row* f init a b c))
  (fold [a f init b c d]
    (matrix-fold* n col-row* f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold* n col-row* f init a b c d es))
  (foldmap [_ g]
    (loop [j 0 acc 0.0]
      (if (< j n)
        (recur (inc j)
               (double
                (let [end (.invokePrim end* n j)]
                  (loop [i (.invokePrim start* n j) acc acc]
                    (if (< i end)
                      (recur (inc i)
                             (.invokePrim ^IFn$DD g (.get da buf (+ (* ld j) i))))
                      acc)))))
        acc)))
  (foldmap [a g f init]
    (matrix-foldmap* n col-row* f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap* n col-row* f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap* n col-row* f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap* n col-row* f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap* n col-row* f init g a b c d es)))

(let [zero (fn ^long [^long n ^long i] 0)
      end (fn ^long [^long n ^long i] n)
      id (fn ^long [^long n ^long i] i)
      inc-id (fn ^long [^long n ^long i] (inc i))
      inv (fn ^long [^long n ^long i] (- n i))
      upper-unit-entry (fn ^long [^long i ^long j]
                         (inc (Integer/signum (- j i))))
      lower-unit-entry (fn ^long [^long i ^long j]
                         (inc (Integer/signum (- i j))))
      upper-entry (fn ^long [^long i ^long j]
                    (* 2 (Integer/signum (inc (Integer/signum (- j i))))))
      lower-entry (fn ^long [^long i ^long j]
                    (* 2 (Integer/signum (inc (Integer/signum (- i j))))))]

  (defn real-tr-matrix
    ([fact master buf n ld ord uplo diag]
     (let [no-trans (= COLUMN_MAJOR ord)
           non-unit (= DIAG_NON_UNIT diag)
           left (if no-trans (= LOWER uplo) (= UPPER uplo))
           da (data-accessor fact)]
       (RealTRMatrix. (if no-trans no-trans-index trans-index)
                      (if no-trans no-trans-get trans-get)
                      (if no-trans no-trans-set trans-set)
                      (if left (if non-unit id inc-id) zero)
                      (if left end (if non-unit inc-id id))
                      (if no-trans col row)
                      (if no-trans straight-cut cross-cut)
                      (if no-trans cross-cut straight-cut)
                      (if left id zero)
                      (if left inv inc-id)
                      (if left zero id)
                      (if left inc-id inv)
                      (if non-unit 0 1)
                      (if (= UPPER uplo)
                        (if non-unit upper-entry upper-unit-entry)
                        (if non-unit lower-entry lower-unit-entry))
                      fact da (tr-engine fact) master
                      buf n ld ord uplo diag)))
    ([fact n ord uplo diag]
     (let-release [buf (.createDataSource (data-accessor fact) (* (long n) (long n)))]
       (real-tr-matrix fact true buf n n ord uplo diag)))
    ([fact n]
     (real-tr-matrix fact n DEFAULT_TRANS DEFAULT_UPLO DEFAULT_DIAG))))

(defmethod transfer! [RealTRMatrix RealTRMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealTRMatrix]
  [source ^RealTRMatrix destination]
  (let [ld (.stride destination)
        n (.ncols destination)
        start* (.start_STAR_ destination)
        end* (.end_STAR_ destination)
        da (data-accessor destination)
        buf (.buf destination)]
    (loop [j 0 src source]
      (if (and src (< j n))
        (recur (inc j)
               (let [end (.invokePrim ^IFn$LLL end* n j)]
                 (loop [i (.invokePrim ^IFn$LLL start* n j) src src]
                   (if (and src (< i end))
                     (do (.set ^RealBufferAccessor da buf (+ (* ld j) i) (first src))
                         (recur (inc i) (next src)))
                     src))))
        destination))))

(defmethod print-method RealTRMatrix
  [^RealTRMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))
